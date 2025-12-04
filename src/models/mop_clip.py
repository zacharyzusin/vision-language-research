import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm


class MixturePromptCLIP(nn.Module):
    """
    EM-style Mixture-of-Prompts CLIP (Option B with aggressive caching).

    - Uses OpenAI CLIP (clip.load).
    - CLIP vision & text encoders are FROZEN.
    - For each class c and each sub-prompt k in {1..K}, we learn ctx_len
      context tokens in CLIP's text embedding space.
    - Context tokens are inserted at positions [1..ctx_len], CoOp-style.
    - E-step: responsibilities gamma(i,k) = softmax(sim / tau), detached.
    - M-step: update context tokens to maximize gamma-weighted similarity.
    - Inference: score for each class is max over its K sub-prompts.

    Caching:
      A) Cache base token embeddings for all classes (token_embedding output).
      B) Reuse positional embeddings, avoid repeated transfers.
      C) Cache full prompt features for ALL classes during inference
         (predict()), computed once per device.
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 8,
        ctx_len: int = 4,
        em_tau: float = 1.0,
        cache_dir: str = "text_cache",
    ):
        super().__init__()

        self.K = int(K)
        self.ctx_len = int(ctx_len)
        self.em_tau = float(em_tau)

        # -------------------------------------------------------
        # Load CLIP and freeze its parameters
        # -------------------------------------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_obj, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_obj
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.metadata = metadata
        self.num_classes = len(metadata)

        # Text width / embedding dimension of text transformer
        self.text_width = self.clip.ln_final.weight.shape[0]

        # -------------------------------------------------------
        # Build per-class base prompt strings and tokenize
        # -------------------------------------------------------
        # We prepend ctx_len placeholder tokens ("X ") which will have their
        # embeddings replaced by learnable context tokens later.
        def make_prompt(md: Dict) -> str:
            return (
                "X " * self.ctx_len
                + f"a photo of {md['species']}, an organism in genus {md['genus']} "
                  f"and family {md['family']} in the order {md['order']}."
            )

        prompts = [make_prompt(md) for md in metadata]

        print(f"Tokenizing {len(prompts)} class prompts...")
        # shape: (C, 77)
        class_tokens = clip.tokenize(prompts)
        self.register_buffer("class_tokens", class_tokens, persistent=True)

        # -------------------------------------------------------
        # CACHE A: precompute base token embeddings for all classes
        #         using CLIP's token_embedding (fixed).
        # -------------------------------------------------------
        with torch.no_grad():
            # (C, L, D)
            base_token_embeds = self.clip.token_embedding(self.class_tokens)
        self.register_buffer("base_token_embeds", base_token_embeds, persistent=True)
        self.seq_len = base_token_embeds.size(1)

        # We will reuse positional embeddings; cache pointer for convenience
        self.register_buffer(
            "positional_embedding",
            self.clip.positional_embedding.clone().detach(),
            persistent=True,
        )

        # -------------------------------------------------------
        # Learnable context tokens: (C, K, ctx_len, D)
        # -------------------------------------------------------
        ctx_shape = (self.num_classes, self.K, self.ctx_len, self.text_width)
        ctx = torch.empty(ctx_shape, dtype=base_token_embeds.dtype)
        nn.init.normal_(ctx, std=0.02)
        self.context_tokens = nn.Parameter(ctx)

        print(
            f"Initialized context tokens with shape {ctx_shape} "
            f"(~{ctx.numel() / 1e6:.1f}M params)"
        )

        # -------------------------------------------------------
        # Inference prompt cache (C)
        #   - Only used in predict(), not during training.
        #   - Stored per device.
        # -------------------------------------------------------
        self._inference_prompt_feats = None  # type: ignore
        self._inference_prompt_device = None  # type: ignore

    # -----------------------------------------------------------
    # Internal helper: encode prompts for specific class indices
    # -----------------------------------------------------------
    def _encode_prompts_for_classes(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode prompts (with learned context tokens) for given class indices.

        Args:
            class_indices: LongTensor of shape (B,) or (N,)

        Returns:
            prompt_features: Tensor of shape (B, K, D), where D is text feature dim.
        """
        device = class_indices.device
        class_indices = class_indices.long()

        # unique classes in this batch to avoid redundant computation
        unique, inv = torch.unique(class_indices, sorted=True, return_inverse=True)
        C_u = unique.shape[0]

        # ------------------------------------------------------------------
        # Pull cached base token embeddings: (C_u, L, D)
        # ------------------------------------------------------------------
        x = self.base_token_embeds[unique].to(device)  # (C_u, L, D)

        # Expand to (C_u, K, L, D) to accommodate K sub-prompts per class
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()

        # Corresponding learnable context tokens: (C_u, K, ctx_len, D)
        ctx = self.context_tokens[unique].to(device)

        # Replace embeddings at positions [1..ctx_len] (position 0 is CLS/EOT start)
        x[:, :, 1 : 1 + self.ctx_len, :] = ctx

        # Add positional embeddings
        # positional_embedding: (L, D)
        pos = self.positional_embedding.to(device)
        x = x + pos  # broadcast over (C_u, K, L, D) via last two dims

        C_uK, L, D = C_u * self.K, self.seq_len, x.size(-1)

        # Flatten sub-prompt dimension into batch for transformer forward:
        x = x.view(C_uK, L, D)

        # Transformer expects (L, N, D)
        x = x.permute(1, 0, 2)  # (L, C_uK, D)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # (C_uK, L, D)

        x = self.clip.ln_final(x)

        # EOT (end-of-text) positions for each sequence
        # We reuse the token ids from base class_tokens
        token_ids = self.class_tokens[unique].to(device)  # (C_u, L)
        token_ids = token_ids.unsqueeze(1).expand(-1, self.K, -1).reshape(C_uK, L)
        eot_indices = token_ids.argmax(dim=-1)

        # Select EOT embeddings and project to text feature space
        text_embeds = x[torch.arange(C_uK, device=device), eot_indices]
        text_embeds = text_embeds @ self.clip.text_projection.to(device)

        # Normalize
        text_embeds = F.normalize(text_embeds, dim=-1)  # (C_uK, D)

        # Reshape back to (C_u, K, D)
        text_embeds = text_embeds.view(C_u, self.K, -1)

        # Map back to the original order via inv
        prompt_features = text_embeds[inv]  # (B, K, D)

        return prompt_features

    # -----------------------------------------------------------
    # Forward: training (EM-style)
    # -----------------------------------------------------------
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass (EM-style).

        Args:
            images: (B, 3, H, W)
            labels: (B,)

        Returns:
            Scalar loss (gamma-weighted negative similarity).
        """
        device = images.device
        labels = labels.to(device)

        # Image embeddings (CLIP vision encoder is frozen)
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)  # (B, D)

        # Get prompt embeddings for each sample's ground-truth class:
        # shape: (B, K, D)
        prompt_feats = self._encode_prompts_for_classes(labels)

        # Similarities for sub-prompts of the ground-truth class: (B, K)
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)

        # E-step: responsibilities gamma (detach for EM)
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        # M-step: maximize gamma-weighted similarity
        loss = -(gamma * sims).sum(dim=1).mean()

        return loss

    # -----------------------------------------------------------
    # Internal: build full prompt cache for ALL classes (inference)
    # -----------------------------------------------------------
    @torch.no_grad()
    def _build_inference_prompt_cache(self, device: torch.device):
        """
        Build and cache prompt features for ALL classes in small chunks.
        Uses a clean tqdm progress bar instead of print spam.
        """
        print("[MixturePromptCLIP] Building inference prompt cache (chunked)...")

        all_classes = torch.arange(self.num_classes, device=device)

        chunk_size = 256   # You used 32 chunks → 8142/256 ≈ 32
        chunks = torch.split(all_classes, chunk_size)

        out_list = []

        from tqdm import tqdm
        pbar = tqdm(
            chunks,
            desc=f"Building cache ({self.num_classes} classes, chunk={chunk_size})",
            unit="chunk"
        )

        for chunk in pbar:
            feats = self._encode_prompts_for_classes(chunk)   # (chunk, K, D)
            out_list.append(feats.cpu())
            torch.cuda.empty_cache()

        # Concatenate all chunks on CPU
        all_feats = torch.cat(out_list, dim=0)   # (C, K, D)

        # Move final cache back to GPU
        self._inference_prompt_feats = all_feats.to(device)
        self._inference_prompt_device = device

        print("[MixturePromptCLIP] Inference prompt cache built successfully.")



    # -----------------------------------------------------------
    # Predict: full classification over all classes
    # -----------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device

        # -------------------------------------------------------
        # Only build the cache ONCE, regardless of epoch.
        # -------------------------------------------------------
        if self._inference_prompt_feats is None:
            self._build_inference_prompt_cache(device)
        else:
            # Cache exists — ensure it resides on current device.
            if self._inference_prompt_device != device:
                self._inference_prompt_feats = self._inference_prompt_feats.to(device)
                self._inference_prompt_device = device

        prompt_feats_all = self._inference_prompt_feats
        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)

        sims = torch.einsum("bd,ckd->bck", img_feat, prompt_feats_all)
        scores = sims.max(dim=2).values
        preds = scores.argmax(dim=1)
        return preds

