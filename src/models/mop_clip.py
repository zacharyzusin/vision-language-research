import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm
from typing import List, Dict


class MixturePromptCLIP(nn.Module):
    """
    EM-based Mixture-of-Prompts CLIP with SHARED prompts (CoOp-style).

    Key design:
      - CLIP is frozen (vision + text).
      - We learn K shared prompt tokens, each with ctx_len learnable tokens.
      - For each class, we build K sub-prompts by inserting these shared
        context tokens into the class-specific text template.
      - Training uses EM:
          - E-step: responsibilities gamma over K (detached).
          - M-step: maximize gamma-weighted similarity for ground-truth class.
      - Inference:
          - For each class, score = max over K sub-prompts.
      - Caching:
          - A: cache base token embeddings per class.
          - B: reuse positional embeddings.
          - C: build (C, K, D) prompt cache once per device for fast predict().
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 8,
        ctx_len: int = 4,
        em_tau: float = 1.0,
    ):
        super().__init__()

        self.K = int(K)
        self.ctx_len = int(ctx_len)
        self.em_tau = float(em_tau)

        # ------------------------------------------
        # Load CLIP and freeze
        # ------------------------------------------
        print(f"Loading CLIP model: {clip_model}")
        clip_model_obj, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_obj
        self.clip.eval()

        for p in self.clip.parameters():
            p.requires_grad = False

        self.metadata = metadata
        self.num_classes = len(metadata)
        self.text_width = self.clip.ln_final.weight.shape[0]

        # ------------------------------------------
        # Build base prompts and tokenize
        # ------------------------------------------
        def make_prompt(md: Dict) -> str:
            # "X " placeholders for ctx_len positions, which we overwrite
            return (
                "X " * self.ctx_len
                + f"a photo of {md['species']}, an organism in genus {md['genus']} "
                  f"and family {md['family']} in the order {md['order']}."
            )

        prompts = [make_prompt(md) for md in metadata]

        print(f"Tokenizing {len(prompts)} class prompts...")
        class_tokens = clip.tokenize(prompts)  # (C, 77)
        self.register_buffer("class_tokens", class_tokens, persistent=True)

        # ------------------------------------------
        # CACHE A: precompute base token embeddings per class
        # ------------------------------------------
        with torch.no_grad():
            base_embeds = self.clip.token_embedding(self.class_tokens)  # (C, L, D)
        self.register_buffer("base_token_embeds", base_embeds, persistent=True)
        self.seq_len = base_embeds.size(1)

        # CACHE B: positional embeddings
        self.register_buffer(
            "positional_embedding",
            self.clip.positional_embedding.clone().detach(),
            persistent=True,
        )

        # ------------------------------------------
        # SHARED context tokens: (K, ctx_len, D)
        # These are shared across all classes.
        # ------------------------------------------
        ctx_shape = (self.K, self.ctx_len, self.text_width)
        ctx = torch.empty(ctx_shape, dtype=base_embeds.dtype)
        nn.init.normal_(ctx, std=0.02)
        self.context_tokens = nn.Parameter(ctx)

        print(
            f"Initialized SHARED context tokens with shape {ctx_shape} "
            f"(~{ctx.numel() / 1e6:.3f}M params)"
        )

        # ------------------------------------------
        # Inference cache holder
        # ------------------------------------------
        self._inference_prompt_feats = None  # (C, K, D)
        self._inference_prompt_device = None

    # -----------------------------------------------------------
    # Encode prompts for given class indices (uses shared prompts)
    # -----------------------------------------------------------
    def _encode_prompts_for_classes(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode prompts (with K shared context prompts) for given class indices.

        Args:
            class_indices: LongTensor of shape (B,)

        Returns:
            prompt_features: Tensor of shape (B, K, D)
        """
        device = class_indices.device
        class_indices = class_indices.long()

        # unique classes to avoid redundant computation
        unique, inv = torch.unique(class_indices, sorted=True, return_inverse=True)
        C_u = unique.shape[0]

        # (C_u, L, D)
        x = self.base_token_embeds[unique].to(device)

        # Expand to (C_u, K, L, D) for K sub-prompts per class
        # We'll broadcast shared context tokens over classes
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()  # (C_u,K,L,D)

        # Get shared context tokens (K, ctx_len, D) and expand over classes
        shared_ctx = self.context_tokens.to(device)  # (K, ctx_len, D)
        # (1,K,ctx_len,D) -> (C_u,K,ctx_len,D)
        shared_ctx = shared_ctx.unsqueeze(0).expand(C_u, -1, -1, -1)

        # Insert shared context tokens at positions [1..ctx_len]
        x[:, :, 1:1 + self.ctx_len, :] = shared_ctx

        # Add positional embeddings
        pos = self.positional_embedding.to(device)  # (L,D)
        x = x + pos  # broadcast over (C_u, K, L, D)

        C_uK = C_u * self.K
        x = x.view(C_uK, self.seq_len, -1)  # (C_u*K, L, D)

        # Transformer expects (L, N, D)
        x = x.permute(1, 0, 2)  # (L, C_uK, D)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # (C_uK, L, D)

        x = self.clip.ln_final(x)

        # EOT (end-of-text) indices
        token_ids = self.class_tokens[unique].to(device)  # (C_u, L)
        token_ids = token_ids.unsqueeze(1).expand(-1, self.K, -1).reshape(C_uK, -1)
        eot_indices = token_ids.argmax(dim=-1)

        # (C_uK, D)
        text_embeds = x[torch.arange(C_uK, device=device), eot_indices]
        text_embeds = text_embeds @ self.clip.text_projection.to(device)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # (C_u, K, D)
        text_embeds = text_embeds.view(C_u, self.K, -1)

        # Map back to original order
        prompt_features = text_embeds[inv]  # (B, K, D)
        return prompt_features

    # -----------------------------------------------------------
    # TRAINING FORWARD (EM style)
    # -----------------------------------------------------------
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = images.device
        labels = labels.to(device)

        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)  # (B,D)

        # Prompt features for ground-truth class (B,K,D)
        prompt_feats = self._encode_prompts_for_classes(labels)

        # Similarities over K prompts: (B,K)
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)

        # E-step: responsibilities gamma (detach)
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        # M-step: maximize gamma-weighted similarity
        loss = -(gamma * sims).sum(dim=1).mean()

        return loss

    # -----------------------------------------------------------
    # BUILD INFERENCE CACHE (chunked, with tqdm)
    # -----------------------------------------------------------
    @torch.no_grad()
    def _build_inference_prompt_cache(self, device: torch.device):
        """
        Build (C, K, D) prompt cache for ALL classes in chunks
        to avoid OOM. Used only in predict().
        """
        print("[MixturePromptCLIP] Building inference prompt cache (chunked)...")

        all_classes = torch.arange(self.num_classes, device=device)
        chunk_size = 256  # adjust if needed
        chunks = torch.split(all_classes, chunk_size)

        out_list = []

        pbar = tqdm(range(len(chunks)), desc="Encoding prompt chunks", ncols=80)
        for i in pbar:
            chunk = chunks[i]
            feats = self._encode_prompts_for_classes(chunk)  # (chunk,K,D)
            out_list.append(feats.cpu())
            torch.cuda.empty_cache()

        all_feats = torch.cat(out_list, dim=0)  # (C,K,D)

        self._inference_prompt_feats = all_feats.to(device)
        self._inference_prompt_device = device

        print("[MixturePromptCLIP] Inference prompt cache built successfully.")

    # -----------------------------------------------------------
    # PREDICT (inference over all classes)
    # -----------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)  # (B,D)

        # Build cache on first call
        if (
            self._inference_prompt_feats is None
            or self._inference_prompt_device != device
        ):
            self._build_inference_prompt_cache(device)

        prompt_feats = self._inference_prompt_feats.to(device)  # (C,K,D)

        # (B,C,K)
        sims = torch.einsum("bd,ckd->bck", img_feat, prompt_feats)

        # Max over K prompts
        scores, _ = sims.max(dim=2)  # (B,C)

        preds = scores.argmax(dim=1)
        return preds
