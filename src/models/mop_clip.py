# src/models/mop_clip.py

import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import clip


class MixturePromptCLIP(nn.Module):
    """
    Mixture-of-Prompts CLIP with hierarchical text prompts and per-class offsets.

    Each class c has K learnable sub-prompts:
        p[c, k] = normalize(base_text_features[c] + prompt_offsets[c, k])

    Training uses a soft EM-like objective:
        - images are softly assigned to sub-prompts of their own class
        - gradients move sub-prompts toward assigned images
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 32,
        em_tau: float = 1.0,
        cache_dir: str = "text_cache",
    ):
        super().__init__()

        self.clip_model_name = clip_model
        self.metadata = metadata
        self.num_classes = len(metadata)
        self.K = int(K)
        self.em_tau = float(em_tau)

        # -------------------------------------------------------------
        # Load CLIP model (initially on CPU)
        # -------------------------------------------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_cpu, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_cpu.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # text_projection is (D, D_text). We want D.
        self.D = self.clip.text_projection.shape[0]  # 512 for ViT-B/32/B/16

        # -------------------------------------------------------------
        # Hierarchical template strings
        # -------------------------------------------------------------
        templates = [
            "a photo of {species}",
            "a wildlife photo of the species {scientific_name}",
            "an organism belonging to genus {genus}",
            "an organism belonging to family {family}",
            "a species in the order {order}",
        ]

        print(f"Building hierarchical prompts with {len(templates)} templates per class...")

        all_prompts = []
        for md in metadata:
            for tmpl in templates:
                all_prompts.append(
                    tmpl.format(
                        species=md["species"],
                        genus=md["genus"],
                        family=md["family"],
                        order=md["order"],
                        scientific_name=md["scientific_name"],
                    )
                )

        C = self.num_classes
        T = len(templates)
        assert len(all_prompts) == C * T

        # -------------------------------------------------------------
        # Encode prompts ONCE with CLIP text encoder
        # -------------------------------------------------------------
        print("Encoding hierarchical text prompts with CLIP text encoder...")
        device_txt = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip.to(device_txt)

        tokenized = clip.tokenize(all_prompts)  # (C*T, 77)
        batch_size = 256

        text_feats = []
        with torch.no_grad():
            for i in tqdm(
                range(0, tokenized.size(0), batch_size),
                desc="Encoding text prompts",
                ncols=80,
            ):
                batch = tokenized[i: i + batch_size].to(device_txt)
                emb = self.clip.encode_text(batch).float()
                emb = F.normalize(emb, dim=-1)
                text_feats.append(emb.cpu())

        text_feats = torch.cat(text_feats, dim=0)        # (C*T, D)
        text_feats = text_feats.view(C, T, -1).mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1)

        # Keep text features on CPU permanently as buffer
        self.register_buffer("base_text_features", text_feats, persistent=True)
        print(f"Initialized base text features: {self.base_text_features.shape}")

        # Move CLIP back to CPU; train.py will move it to the correct device
        self.clip.to("cpu")

        # -------------------------------------------------------------
        # Per-class learnable prompt offsets
        # Shape: (C, K, D)
        # -------------------------------------------------------------
        self.prompt_offsets = nn.Parameter(torch.randn(C, self.K, self.D) * 0.01)
        print(f"Initialized per-class prompt offsets: {self.prompt_offsets.shape}")

    # ==========================================================
    # Force base_text_features to remain on CPU when model.to() is called
    # ==========================================================
    def _apply(self, fn):
        btf_cpu = self.base_text_features
        super()._apply(fn)
        self.base_text_features = btf_cpu
        return self

    # ==========================================================
    # Build (B, K, D) prompt features for each label
    # ==========================================================
    def _batch_prompt_features(self, labels: torch.Tensor, device: torch.device):
        """
        Build (B, K, D) prompt embeddings for the given labels.
        Ensures final tensor is ALWAYS on `device`.
        """

        # 1. Base text features ALWAYS on CPU → index there → move to GPU
        btf_cpu = self.base_text_features.index_select(0, labels.cpu())  # (B, D)
        btf = btf_cpu.to(device)

        # 2. Per-class prompt offsets (on device after model.to(device))
        offsets = self.prompt_offsets.index_select(0, labels)  # (B, K, D)

        # 3. Combine base + offsets
        final = btf.unsqueeze(1) + offsets  # (B, K, D)

        # 4. Normalize
        final = F.normalize(final, dim=-1, eps=1e-6)

        return final

    # ==========================================================
    # Full-class prompt features (C, K, D)
    # ==========================================================
    def _all_prompt_features(self, device):
        base = self.base_text_features.to(device)     # (C, D)
        offs = self.prompt_offsets.to(device)         # (C, K, D)

        prompts = base.unsqueeze(1) + offs            # (C, K, D)
        return F.normalize(prompts, dim=-1)

    # ==========================================================
    # Training forward
    # ==========================================================
    def forward(self, images, labels):
        """
        images: (B, 3, H, W) on device
        labels: (B,) class indices (0..C-1) on device

        Loss: -E_{k~gamma}[sim(img, prompt_{y,k})]
        where gamma is softmax over K sub-prompts for the correct class.
        """
        device = images.device

        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)

        prompt_feats = self._batch_prompt_features(labels, device)  # (B, K, D)
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)   # (B, K)
        sims = sims.clamp(min=-50, max=50)

        # Soft responsibilities (E-step surrogate)
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        # Expected negative similarity under gamma
        loss = -(gamma * sims).sum(dim=1).mean()
        return loss

    # ==========================================================
    # Prediction over all classes (chunked)
    # ==========================================================
    @torch.no_grad()
    def predict(self, images):
        """
        images: (B, 3, H, W) on device

        Returns:
          preds: (B,) predicted class indices
        """
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)

        all_prompts = self._all_prompt_features(device)  # (C, K, D)
        C = all_prompts.size(0)

        chunk_size = 512
        cpu_accum = []

        for chunk in torch.split(torch.arange(C, device=device), chunk_size):
            pf = all_prompts[chunk]                     # (chunk, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)  # (B, chunk, K)
            scores = sims.max(dim=2).values             # (B, chunk)
            cpu_accum.append(scores.cpu())

        final_scores = torch.cat(cpu_accum, dim=1).to(device)  # (B, C)
        return final_scores.argmax(dim=1)
