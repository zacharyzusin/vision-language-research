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
    Fast Mixture-of-Prompts CLIP with hierarchical text prompts.

    Key ideas:
      • Text encoder is used ONCE at initialization to build base class embeddings.
      • We use hierarchical prompts: species / genus / family / order.
      • We learn K global prompt offsets in embedding space: (K, D).
      • For class c and component k:
            prompt_feat[c, k] = normalize( base_text[c] + offset[k] )
      • Training forward does NOT run the CLIP text transformer.
        It only runs CLIP image encoder + a few matmuls.

    Shapes:
      num_classes = C
      text dim    = D (512 for ViT-B/32)
      K prompts   = K
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 32,
        ctx_len: int = 8,      # kept for config compatibility; not used at token level now
        em_tau: float = 0.5,
        cache_dir: str = "text_cache",  # unused but kept for API compatibility
    ):
        super().__init__()

        self.clip_model_name = clip_model
        self.metadata = metadata
        self.num_classes = len(metadata)
        self.K = int(K)
        self.ctx_len = int(ctx_len)
        self.em_tau = float(em_tau)

        # ---------------------------
        # Load CLIP
        # ---------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_cpu, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_cpu.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Text embed dimension (e.g. 512)
        self.D = self.clip.text_projection.shape[0]

        # ---------------------------
        # Build hierarchical class prompts
        # ---------------------------
        templates = [
            "a photo of {species}",
            "a wildlife photo of the species {scientific_name}",
            "a photograph of an organism in genus {genus}",
            "an organism belonging to family {family}",
            "a photo of the species {scientific_name} in the order {order}",
        ]

        print(f"Building hierarchical prompts with {len(templates)} templates per class...")
        all_prompts = []
        for md in metadata:
            # md has keys: 'species', 'genus', 'family', 'order', 'scientific_name'
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

        assert len(all_prompts) == C * T, "Prompt construction mismatch (C*T)."

        # ---------------------------
        # Encode prompts once with CLIP text encoder
        # ---------------------------
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
                batch = tokenized[i : i + batch_size].to(device_txt)
                emb = self.clip.encode_text(batch).float()      # (B, D)
                emb = F.normalize(emb, dim=-1)
                text_feats.append(emb.cpu())

        text_feats = torch.cat(text_feats, dim=0)  # (C*T, D)

        # Collapse templates: (C, T, D) -> (C, D)
        text_feats = text_feats.view(C, T, -1).mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1)  # (C, D)

        # Register as a buffer so it moves with model.to(device)
        self.register_buffer("base_text_features", text_feats, persistent=True)

        # Move CLIP back to CPU; training will later move it with model.to(device)
        self.clip.to("cpu")

        print(
            f"Initialized base text features: shape={self.base_text_features.shape} "
            f"(C={C}, D={self.D})"
        )

        # ---------------------------
        # Learnable prompt offsets in embedding space
        # ---------------------------
        # These are global across classes: (K, D)
        self.prompt_offsets = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        print(
            f"Initialized {self.K} learnable prompt offsets in embedding space: "
            f"({self.K}, {self.D})"
        )

    # ======================================================================
    # Utility: build per-batch prompt features from base_text + offsets
    # ======================================================================
    def _batch_prompt_features(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build prompt features for a given batch of labels.

        Args:
          labels: (B,) long tensor of class indices.

        Returns:
          prompt_feats: (B, K, D) normalized embeddings.
        """
        device = labels.device
        base = self.base_text_features.to(device)        # (C, D)
        base_batch = base[labels]                       # (B, D)

        offsets = self.prompt_offsets.to(device)        # (K, D)

        # (B, 1, D) + (1, K, D) -> (B, K, D)
        prompts = base_batch.unsqueeze(1) + offsets.unsqueeze(0)
        prompts = F.normalize(prompts, dim=-1)

        return prompts

    # ======================================================================
    # Utility: build full prompt features for all classes (for inference)
    # ======================================================================
    def _all_prompt_features(self, device: torch.device) -> torch.Tensor:
        """
        Build prompt features for ALL classes.

        Returns:
          prompt_feats: (C, K, D) normalized embeddings on `device`.
        """
        base = self.base_text_features.to(device)    # (C, D)
        offsets = self.prompt_offsets.to(device)     # (K, D)

        # (C, 1, D) + (1, K, D) -> (C, K, D)
        prompts = base.unsqueeze(1) + offsets.unsqueeze(0)
        prompts = F.normalize(prompts, dim=-1)       # (C, K, D)

        return prompts

    # ======================================================================
    # Training forward: EM-style loss with precomputed text features
    # ======================================================================
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        images: (B, 3, H, W)
        labels: (B,)
        """
        device = images.device

        # Encode images once with CLIP image encoder
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()   # (B, D)
            img_feat = F.normalize(img_feat, dim=-1)

        # Get batch-specific prompt features: (B, K, D)
        prompt_feats = self._batch_prompt_features(labels)      # (B, K, D)

        # Similarity per mixture component
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)  # (B, K)

        # E-step: responsibilities (soft assignment over K prompts)
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()      # (B, K)

        # M-step: maximize gamma-weighted similarity
        loss = -(gamma * sims).sum(dim=1).mean()
        return loss

    # ======================================================================
    # Inference: predict class indices for a batch
    # ======================================================================
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference over all classes.

        images: (B, 3, H, W)

        Returns:
          preds: (B,) int64 tensor of predicted class indices.
        """
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)    # (B, D)
        B = img_feat.size(0)

        # Build prompt features for ALL classes once per call
        prompt_feats = self._all_prompt_features(device)        # (C, K, D)
        C = prompt_feats.size(0)

        # To keep memory safe, we can optionally chunk over classes.
        # But (8142, 32, 512) ~ 0.5GB in fp32, which is often fine.
        # We'll still chunk for safety on smaller GPUs.
        chunk_size = 512
        class_indices = torch.arange(C, device=device)
        chunks = torch.split(class_indices, chunk_size)

        all_scores = []

        for chunk in chunks:
            pf = prompt_feats[chunk]                            # (chunk_size, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)    # (B, chunk_size, K)
            scores = sims.max(dim=2).values                    # (B, chunk_size)
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=1)              # (B, C)
        preds = all_scores.argmax(dim=1)

        return preds
