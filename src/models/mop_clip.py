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
    Fast Mixture-of-Prompts CLIP with hierarchical text prompts (Option 3).
    - Precomputes text encoder outputs ONCE at initialization.
    - Eliminates CLIP text transformer from training loop.
    - Uses K learnable offsets in embedding space.
    - Hierarchical templates: species / genus / family / order.

    Shapes:
      num_classes = C
      text dim    = D (=512 for ViT-B/32)
      mixture K   = K
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 32,
        ctx_len: int = 8,    # kept for API compatibility
        em_tau: float = 0.5,
        cache_dir: str = "text_cache",
    ):
        super().__init__()

        self.clip_model_name = clip_model
        self.metadata = metadata
        self.num_classes = len(metadata)
        self.K = int(K)
        self.ctx_len = int(ctx_len)
        self.em_tau = float(em_tau)

        # -------------------------------------------------------------
        # Load CLIP model
        # -------------------------------------------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_cpu, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_cpu.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Text embedding dimension
        self.D = self.clip.text_projection.shape[0]

        # -------------------------------------------------------------
        # Build hierarchical prompts
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
        # Encode text prompts ONCE
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

        text_feats = torch.cat(text_feats, dim=0)  # (C*T, D)

        # Collapse templates → (C, D)
        text_feats = text_feats.view(C, T, -1).mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1)

        # Store on CPU permanently
        self.register_buffer(
            "base_text_features",
            text_feats,
            persistent=True
        )

        print(
            f"Initialized base text features: shape={self.base_text_features.shape} "
            f"(C={C}, D={self.D})"
        )

        self.clip.to("cpu")

        # -------------------------------------------------------------
        # Learnable mixture prompt offsets: (K, D)
        # -------------------------------------------------------------
        self.prompt_offsets = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        print(f"Initialized {self.K} learnable prompt offsets: ({self.K}, {self.D})")


    # ==========================================================
    # Correct override: FORCE base_text_features to stay on CPU
    # ==========================================================
    def _apply(self, fn):
        """
        Override internal apply() so base_text_features ALWAYS remains on CPU,
        while everything else moves to the target device (via .to(), cuda(), etc).
        """
        # Save CPU tensor
        btf_cpu = self.base_text_features

        # Apply moves to everything else
        super()._apply(fn)

        # Restore CPU buffer
        self.base_text_features = btf_cpu

        return self

    # ==========================================================
    # Build per-batch prompt features
    # ==========================================================
    def _batch_prompt_features(self, labels: torch.Tensor, device: torch.device):
        """
        Build (B, K, D) mixture prompt embeddings.
        Ensures final tensor is ALWAYS on `device`.
        """

        # 1. Base text features ALWAYS on CPU → index there
        btf_cpu = self.base_text_features.index_select(0, labels.cpu())

        # 2. Clone → move to GPU cleanly
        btf = btf_cpu.clone().to(device)    # (B, D)

        # 3. Learnable offsets on GPU
        offsets = self.prompt_offsets.to(device)  # (K, D)

        # 4. Combine
        final = btf.unsqueeze(1) + offsets.unsqueeze(0)   # (B, K, D)

        # 5. Normalize safely
        final = F.normalize(final, dim=-1, eps=1e-6)

        # 6. Guarantee tensor is on GPU
        final = final.to(device)

        return final



    # ==========================================================
    # Build full prompt features for inference
    # ==========================================================
    def _all_prompt_features(self, device):
        """
        Returns (C, K, D) on GPU efficiently.
        """
        base = self.base_text_features.to(device)        # (C, D)
        offs = self.prompt_offsets.to(device)            # (K, D)

        prompts = base.unsqueeze(1) + offs.unsqueeze(0)  # (C, K, D)
        return F.normalize(prompts, dim=-1)

    # ==========================================================
    # Training forward
    # ==========================================================
    def forward(self, images, labels):
        device = images.device

        # Fast image encoding only
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)

        # Mixture prompt embeddings (B, K, D)
        prompt_feats = self._batch_prompt_features(labels, device)

        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)

        # Numerical stability — clamp similarities
        sims = sims.clamp(min=-50, max=50)

        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        return -(gamma * sims).sum(dim=1).mean()

    # ==========================================================
    # Prediction (full-class, chunked)
    # ==========================================================
    @torch.no_grad()
    def predict(self, images):
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)

        # Build class prompt features
        all_prompts = self._all_prompt_features(device)
        C = all_prompts.size(0)

        chunk_size = 512
        cpu_scores = []

        for chunk in torch.split(torch.arange(C, device=device), chunk_size):
            pf = all_prompts[chunk]                        # (chunk, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)
            scores = sims.max(dim=2).values                # (B, chunk)
            cpu_scores.append(scores.cpu())

        all_scores = torch.cat(cpu_scores, dim=1).to(device)
        preds = all_scores.argmax(dim=1)
        return preds
