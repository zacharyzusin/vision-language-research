"""
Mixture-of-Prompts CLIP Model Implementation.

This module implements a CLIP-based model that uses hierarchical text prompts
and learnable per-class prompt offsets to enable fine-grained classification.
"""

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
        templates: List[str] = None,
        use_semantic_init: bool = True,
        offset_reg_weight: float = 0.001,
        use_hard_assignment: bool = False,
    ):
        """
        Initialize MixturePromptCLIP model.

        Args:
            clip_model: CLIP model name (e.g., "ViT-B/16", "ViT-B/32")
            metadata: List of dicts with keys: species, genus, family, order, scientific_name
                     (or make, model, type, category, full_name for cars)
            K: Number of sub-prompts per class
            em_tau: Temperature parameter for soft EM assignments (higher = softer)
            cache_dir: Directory for caching (currently unused, kept for compatibility)
            templates: Optional list of template strings. If None, auto-detects based on metadata.
        """
        super().__init__()

        self.clip_model_name = clip_model
        self.metadata = metadata
        self.num_classes = len(metadata)
        self.K = int(K)
        self.em_tau = float(em_tau)

        # Validate metadata ordering matches expected class count
        assert len(metadata) == self.num_classes, \
            f"Metadata length ({len(metadata)}) must match num_classes ({self.num_classes})"
        
        # Note: cache_dir parameter is kept for API compatibility but not currently used

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
        # Hierarchical template strings - auto-detect or use provided
        # -------------------------------------------------------------
        if templates is None:
            # Auto-detect template type based on metadata keys
            sample_metadata = metadata[0] if metadata else {}
            if "make" in sample_metadata or "model" in sample_metadata:
                # Car dataset - use car-specific templates
                templates = [
                    "a photo of {full_name}",
                    "a {full_name} car",
                    "a {make} {model}",
                    "a {type} made by {make}",
                    "a {category} vehicle: {full_name}",
                ]
            else:
                # iNaturalist/wildlife dataset - use biological templates
                templates = [
                    "a photo of {species}",
                    "a wildlife photo of the species {scientific_name}",
                    "an organism belonging to genus {genus}",
                    "an organism belonging to family {family}",
                    "a species in the order {order}",
                ]
        else:
            templates = templates

        print(f"Building hierarchical prompts with {len(templates)} templates per class...")

        all_prompts = []
        for md in metadata:
            for tmpl in templates:
                # Try to format with available keys (handles both car and wildlife formats)
                try:
                    all_prompts.append(tmpl.format(**md))
                except KeyError as e:
                    # Fallback: use get() with empty string for missing keys
                    formatted = tmpl
                    for key in md.keys():
                        formatted = formatted.replace(f"{{{key}}}", str(md.get(key, "")))
                    all_prompts.append(formatted)

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
        if use_semantic_init:
            # Semantic initialization: initialize near base text features
            self.prompt_offsets = nn.Parameter(torch.randn(C, self.K, self.D) * 0.01)
        else:
            # Random initialization: start from zero
            self.prompt_offsets = nn.Parameter(torch.zeros(C, self.K, self.D))
        print(f"Initialized per-class prompt offsets: {self.prompt_offsets.shape}")
        if not use_semantic_init:
            print("  (Random initialization, no semantic init)")
        
        # Regularization strength for prompt offsets
        self.offset_reg_weight = offset_reg_weight
        self.use_hard_assignment = use_hard_assignment
        
        # CLIP-style similarity scaling factor
        self.sim_scale = 50.0

    # ==========================================================
    # Force base_text_features to remain on CPU when model.to() is called
    # This prevents moving large text features to GPU unnecessarily
    # ==========================================================
    def _apply(self, fn):
        """Override _apply to keep base_text_features on CPU."""
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
        """
        Compute prompt features for all classes.

        Args:
            device: Target device for computation

        Returns:
            Tensor of shape (C, K, D) with normalized prompt features for all classes
        """
        base = self.base_text_features.to(device)     # (C, D)
        offs = self.prompt_offsets.to(device)         # (C, K, D)

        prompts = base.unsqueeze(1) + offs            # (C, K, D)
        return F.normalize(prompts, dim=-1)

    # ==========================================================
    # Training forward
    # ==========================================================
    def forward(self, images, labels, lambda_mixture=0.5, temp_cls=0.07):
        """
        images: (B, 3, H, W) on device
        labels: (B,) class indices (0..C-1) on device
        lambda_mixture: weight for mixture loss vs classification loss
        temp_cls: temperature for classification logits

        Returns:
            loss: combined classification + mixture loss
            dict with loss components for logging
        """
        device = images.device

        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)

        # ==========================================================
        # Mixture loss (original): intra-class specialization
        # ==========================================================
        prompt_feats = self._batch_prompt_features(labels, device)  # (B, K, D)
        sims_mixture = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)  # (B, K)
        sims_mixture = sims_mixture * self.sim_scale  # CLIP-style scaling

        # Soft or hard responsibilities (E-step surrogate)
        if self.use_hard_assignment:
            # Hard assignment: argmax
            gamma = torch.zeros_like(sims_mixture)
            gamma.scatter_(1, sims_mixture.argmax(dim=1, keepdim=True), 1.0)
            gamma = gamma.detach()
        else:
            # Soft assignment: softmax
            gamma = F.softmax(sims_mixture / self.em_tau, dim=1).detach()

        # Expected negative similarity under gamma
        loss_mixture = -(gamma * sims_mixture).sum(dim=1).mean()

        # ==========================================================
        # Classification loss: inter-class separation
        # ==========================================================
        all_prompts = self._all_prompt_features(device)  # (C, K, D)
        sims_all = torch.einsum("bd,ckd->bck", img_feat, all_prompts)  # (B, C, K)
        sims_all = sims_all * self.sim_scale  # CLIP-style scaling

        # Max pooling over sub-prompts: score(c) = max_k sim(img, p[c,k])
        class_logits = sims_all.max(dim=2).values  # (B, C)

        # Cross-entropy with temperature scaling
        loss_cls = F.cross_entropy(class_logits / temp_cls, labels)

        # ==========================================================
        # Regularization: prevent offsets from drifting too far
        # ==========================================================
        reg_loss = self.offset_reg_weight * (self.prompt_offsets ** 2).mean()

        # ==========================================================
        # Combined loss
        # ==========================================================
        loss = lambda_mixture * loss_mixture + (1.0 - lambda_mixture) * loss_cls + reg_loss

        return loss, {
            "loss_mixture": loss_mixture.item(),
            "loss_cls": loss_cls.item(),
            "loss_reg": reg_loss.item(),
            "loss_total": loss.item(),
        }

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
        scores_accum = []

        for chunk in torch.split(torch.arange(C, device=device), chunk_size):
            pf = all_prompts[chunk]                     # (chunk, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)  # (B, chunk, K)
            scores = sims.max(dim=2).values             # (B, chunk)
            scores_accum.append(scores)

        final_scores = torch.cat(scores_accum, dim=1)  # (B, C) - keep on device
        return final_scores.argmax(dim=1)
