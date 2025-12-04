import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm


class MixturePromptCLIP(nn.Module):
    """
    EM-based Mixture-of-Prompts CLIP (CoOp-style), with aggressive caching:
      - Option A: Cache base token embeddings for each class.
      - Option B: Reuse positional embeddings and avoid redundant embedding work.
      - Option C: Chunked inference cache for fast predict().

    Behaviour:
      - During training: no giant cache; text encoding uses unique classes in batch.
      - During inference: builds full (C,K,D) prompt cache ONCE (chunked to avoid OOM).
      - Subsequent predict() calls are very fast.
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

        # ------------------------------------------
        # Load CLIP (OpenAI) and freeze
        # ------------------------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_obj, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_obj
        self.clip.eval()

        for p in self.clip.parameters():
            p.requires_grad = False

        # Metadata
        self.metadata = metadata
        self.num_classes = len(metadata)
        self.text_width = self.clip.ln_final.weight.shape[0]

        # ------------------------------------------
        # Build base prompts and tokenize
        # ------------------------------------------
        def make_prompt(md: Dict) -> str:
            return (
                "X " * self.ctx_len
                + f"a photo of {md['species']}, an organism in genus {md['genus']} "
                  f"and family {md['family']} in the order {md['order']}."
            )

        prompts = [make_prompt(md) for md in metadata]

        print(f"Tokenizing {len(prompts)} class prompts...")
        class_tokens = clip.tokenize(prompts)  # (C,77)
        self.register_buffer("class_tokens", class_tokens, persistent=True)

        # ------------------------------------------
        # CACHE A — Precompute token embeddings
        # ------------------------------------------
        with torch.no_grad():
            base_embeds = self.clip.token_embedding(self.class_tokens)  # (C,L,D)
        self.register_buffer("base_token_embeds", base_embeds, persistent=True)

        self.seq_len = base_embeds.shape[1]

        # Cache positional embeddings
        self.register_buffer(
            "positional_embedding",
            self.clip.positional_embedding.clone().detach(),
            persistent=True
        )

        # ------------------------------------------
        # Learnable context tokens: (C, K, ctx_len, D)
        # ------------------------------------------
        ctx_shape = (self.num_classes, self.K, self.ctx_len, self.text_width)
        ctx = torch.empty(ctx_shape, dtype=base_embeds.dtype)
        nn.init.normal_(ctx, std=0.02)
        self.context_tokens = nn.Parameter(ctx)

        print(
            f"Initialized context tokens with shape {ctx_shape} "
            f"(~{ctx.numel()/1e6:.1f}M params)"
        )

        # ------------------------------------------
        # Inference cache holder
        # ------------------------------------------
        self._inference_prompt_feats = None
        self._inference_prompt_device = None

    # -----------------------------------------------------------
    # Encode prompts for the given class indices
    # -----------------------------------------------------------
    def _encode_prompts_for_classes(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode class prompts with learnable context tokens.

        Args:
            class_indices: (B,)

        Returns:
            (B, K, D) text embeddings for each label.
        """
        device = class_indices.device
        class_indices = class_indices.long()

        # unique classes to reduce computation
        unique, inv = torch.unique(class_indices, sorted=True, return_inverse=True)
        C_u = unique.shape[0]

        # (C_u, L, D)
        x = self.base_token_embeds[unique].to(device)

        # Expand to (C_u, K, L, D)
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()

        # Insert learned context tokens
        ctx = self.context_tokens[unique].to(device)  # (C_u,K,ctx_len,D)
        x[:, :, 1:1+self.ctx_len, :] = ctx

        # Add positional embeddings
        pos = self.positional_embedding.to(device)  # (L,D)
        x = x + pos  # broadcast

        # Flatten K dimension
        C_uK = C_u * self.K
        x = x.view(C_uK, self.seq_len, -1)

        # Transformer expects (L,N,D)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.clip.ln_final(x)

        # EOT extraction
        token_ids = self.class_tokens[unique].to(device)
        token_ids = token_ids.unsqueeze(1).expand(-1, self.K, -1).reshape(C_uK, -1)
        eot_indices = token_ids.argmax(dim=-1)

        text_embeds = x[torch.arange(C_uK, device=device), eot_indices]
        text_embeds = text_embeds @ self.clip.text_projection.to(device)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Reshape to (C_u, K, D) then index back to original order
        text_embeds = text_embeds.view(C_u, self.K, -1)
        return text_embeds[inv]  # (B,K,D)

    # -----------------------------------------------------------
    # TRAINING FORWARD (EM style)
    # -----------------------------------------------------------
    def forward(self, images, labels):
        device = images.device

        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)

        # (B,K,D)
        prompt_feats = self._encode_prompts_for_classes(labels)

        # (B,K)
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)

        # E-step responsibilities (detach)
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        # M-step loss
        loss = -(gamma * sims).sum(dim=1).mean()
        return loss

    # -----------------------------------------------------------
    # BUILD INFERENCE CACHE — chunked, with tqdm
    # -----------------------------------------------------------
    @torch.no_grad()
    def _build_inference_prompt_cache(self, device):
        print("[MixturePromptCLIP] Building inference prompt cache (chunked)...")

        all_classes = torch.arange(self.num_classes, device=device)
        chunk_size = 256  # safe default for 24GB GPU
        chunks = torch.split(all_classes, chunk_size)

        out_list = []

        pbar = tqdm(range(len(chunks)), desc="Encoding prompt chunks", ncols=80)
        for i in pbar:
            chunk = chunks[i]
            feats = self._encode_prompts_for_classes(chunk)
            out_list.append(feats.cpu())
            torch.cuda.empty_cache()

        all_feats = torch.cat(out_list, dim=0)  # (C,K,D)

        self._inference_prompt_feats = all_feats.to(device)
        self._inference_prompt_device = device

        print("[MixturePromptCLIP] Inference prompt cache built successfully.")

    # -----------------------------------------------------------
    # PREDICT (inference)
    # -----------------------------------------------------------
    @torch.no_grad()
    def predict(self, images):
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)

        # Build cache on first call
        if (
            self._inference_prompt_feats is None
            or self._inference_prompt_device != device
        ):
            self._build_inference_prompt_cache(device)

        prompt_feats = self._inference_prompt_feats.to(device)  # (C,K,D)

        # Similarities (B,C,K)
        sims = torch.einsum("bd,ckd->bck", img_feat, prompt_feats)

        # Max-pool across prompts
        scores, _ = sims.max(dim=2)
        preds = scores.argmax(dim=1)
        return preds
