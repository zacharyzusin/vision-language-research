import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import clip


class MixturePromptCLIP(nn.Module):
    """
    Mixture-of-Prompts CLIP with:
      • Shared K × ctx_len learnable context tokens
      • EM-style training objective over K prompts per class
      • GPU-accelerated prompt cache building (chunked for safety)
      • CPU-stored prompt cache for inference
      • Disk caching so prompt cache is built only once
    """

    def __init__(
        self,
        clip_model: str,
        metadata: List[Dict],
        K: int = 32,
        ctx_len: int = 8,
        em_tau: float = 0.5,
        cache_dir: str = "text_cache",
    ):
        super().__init__()

        # ---------------------------
        # Basic config
        # ---------------------------
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
        clip_model_gpu, _ = clip.load(clip_model, device="cpu")
        self.clip = clip_model_gpu.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Text embedding dim
        self.D = self.clip.text_projection.shape[0]

        # ---------------------------
        # Build text prompts
        # ---------------------------
        def make_prompt(md: Dict) -> str:
            # Use "X " tokens as placeholders for learnable context
            return (
                "X " * self.ctx_len
                + f"a photo of {md['species']}, an organism in genus {md['genus']} "
                  f"and family {md['family']} belonging to order {md['order']}."
            )

        prompts = [make_prompt(md) for md in metadata]

        print(f"Tokenizing {len(prompts)} class prompts...")
        class_tokens = clip.tokenize(prompts)  # (C, 77)
        self.register_buffer("class_tokens", class_tokens, persistent=True)

        # Base token embeddings & positional embeddings (CPU only)
        with torch.no_grad():
            base_emb = self.clip.token_embedding(self.class_tokens)  # (C, L, D)
        self.base_token_embeds_cpu = base_emb.cpu()  # plain tensor on CPU

        self.positional_embedding_cpu = (
            self.clip.positional_embedding.detach().clone().cpu()
        )  # (L, D)

        self.seq_len = self.base_token_embeds_cpu.size(1)  # e.g. 77

        # ---------------------------
        # Shared K × ctx_len context tokens
        # ---------------------------
        ctx_shape = (self.K, self.ctx_len, self.D)
        ctx = torch.empty(ctx_shape)
        nn.init.normal_(ctx, std=0.02)
        self.context_tokens = nn.Parameter(ctx)

        print(
            f"Initialized SHARED context tokens {ctx_shape} "
            f"(~{ctx.numel() / 1e6:.3f}M params)"
        )

        # ---------------------------
        # Prompt cache (CPU + disk)
        # ---------------------------
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_name = clip_model.replace("/", "_")
        self.prompt_cache_path = os.path.join(
            self.cache_dir,
            f"{safe_name}_K{self.K}_ctx{self.ctx_len}_prompt_cache.pt",
        )

        self._prompt_cache_cpu = None  # (C, K, D) on CPU once built

    # ======================================================================
    # GPU prompt encoder for training (labels in a batch)
    # ======================================================================
    def _encode_prompts_for_classes_gpu(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode prompts for the classes in 'class_indices' on GPU.
        Used during training (batch-sized).
        Returns shape: (B, K, D)
        """
        device = class_indices.device
        class_indices = class_indices.long()

        unique, inv = torch.unique(class_indices, sorted=True, return_inverse=True)
        C_u = unique.numel()

        # Base token embeddings -> GPU
        base_emb = self.base_token_embeds_cpu.to(device)  # (C, L, D)
        x = base_emb[unique]                              # (C_u, L, D)

        # Expand over K prompts
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()  # (C_u,K,L,D)

        # Insert shared context at positions [1..ctx_len]
        ctx = self.context_tokens.to(device)  # (K, ctx_len, D)
        ctx = ctx.unsqueeze(0).expand(C_u, -1, -1, -1)              # (C_u,K,ctx_len,D)
        x[:, :, 1 : 1 + self.ctx_len, :] = ctx

        # Add positional embeddings
        pos = self.positional_embedding_cpu.to(device)  # (L, D)
        x = x + pos                                     # broadcast to (C_u,K,L,D)

        # Flatten for CLIP text transformer
        C_uK = C_u * self.K
        x = x.view(C_uK, self.seq_len, -1)         # (C_uK, L, D)
        x = x.permute(1, 0, 2)                     # (L, C_uK, D)

        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)                     # (C_uK, L, D)
        x = self.clip.ln_final(x)

        # End-of-text indices
        ct = self.class_tokens.to(device)          # (C, L)
        token_ids = ct[unique]                     # (C_u, L)
        token_ids = (
            token_ids.unsqueeze(1)
            .expand(-1, self.K, -1)
            .reshape(C_uK, -1)
        )
        eot = token_ids.argmax(dim=-1)             # (C_uK,)

        # Gather final text embeddings
        text_embeds = x[torch.arange(C_uK, device=device), eot]  # (C_uK, D)
        text_embeds = text_embeds @ self.clip.text_projection    # (C_uK, D)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Reshape back to (C_u, K, D)
        text_embeds = text_embeds.view(C_u, self.K, -1)

        # Reorder to match original class_indices
        return text_embeds[inv]  # (B, K, D)

    # ======================================================================
    # GPU prompt encoder for cache-building (large class chunks)
    # ======================================================================
    @torch.no_grad()
    def _encode_prompts_for_classes_gpu_cache(
        self, class_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Same as _encode_prompts_for_classes_gpu, but:
        - ensures CLIP is on GPU
        - returns CPU tensor for caching

        Used only during prompt cache building.
        """
        device = torch.device("cuda")
        self.clip.to(device)

        class_indices = class_indices.to(device).long()
        feats = self._encode_prompts_for_classes_gpu(class_indices)  # (B,K,D) on GPU
        return feats.cpu()

    # ======================================================================
    # Training forward: EM-style mixture objective over prompts
    # ======================================================================
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = images.device

        # Encode images on GPU
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)  # (B, D)

        # Encode K prompts per label
        prompt_feats = self._encode_prompts_for_classes_gpu(labels)  # (B, K, D)

        # Similarity over prompts
        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)   # (B, K)

        # E-step: responsibilities
        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        # M-step: maximize gamma-weighted similarity
        loss = -(gamma * sims).sum(dim=1).mean()
        return loss

    # ======================================================================
    # Build full (C, K, D) prompt cache using GPU, store on CPU + disk
    # ======================================================================
    @torch.no_grad()
    def _build_prompt_cache_cpu(self):
        """
        Build the full prompt cache:
            prompt_feats[class_id, k, :] = embedding of k-th prompt for class_id

        - Uses GPU in small chunks for speed
        - Stores result on CPU
        - Saves to disk for reuse
        """
        # 1. Try disk cache
        if os.path.exists(self.prompt_cache_path):
            print(f"[MixturePromptCLIP] Loading prompt cache from {self.prompt_cache_path}")
            data = torch.load(self.prompt_cache_path, map_location="cpu")
            self._prompt_cache_cpu = data["prompt_feats"]  # (C, K, D)
            print("[MixturePromptCLIP] Prompt cache loaded from disk.")
            return

        # 2. Build cache on GPU in chunks
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to build prompt cache with Option D.")

        print("[MixturePromptCLIP] Building prompt cache on GPU (chunked)...")

        all_classes = torch.arange(self.num_classes, device=torch.device("cuda"))
        chunk_size = 64  # safe; 64 classes * 32 prompts = 2048 prompts per chunk

        feats_list = []

        pbar = tqdm(
            range(0, self.num_classes, chunk_size),
            desc="GPU prompt chunks",
            ncols=80,
        )

        for start in pbar:
            end = min(start + chunk_size, self.num_classes)
            chunk = all_classes[start:end]  # (chunk_size,)
            f = self._encode_prompts_for_classes_gpu_cache(chunk)  # (chunk_size, K, D) on CPU
            feats_list.append(f)

            # free transient GPU memory from text-forward
            torch.cuda.empty_cache()

        feats = torch.cat(feats_list, dim=0)  # (C, K, D)
        self._prompt_cache_cpu = feats

        # 3. Save to disk
        torch.save({"prompt_feats": feats}, self.prompt_cache_path)
        print(f"[MixturePromptCLIP] Prompt cache saved to {self.prompt_cache_path}")

    # ======================================================================
    # Inference: use cached prompts for all classes
    # ======================================================================
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device

        # Encode images
        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)  # (B, D)
        B = img_feat.size(0)

        # Ensure prompt cache is available
        if self._prompt_cache_cpu is None:
            self._build_prompt_cache_cpu()

        feats_cpu = self._prompt_cache_cpu  # (C, K, D) on CPU
        C = feats_cpu.size(0)

        # Chunk classes to keep GPU memory safe
        chunk_size = 512
        chunks = torch.split(torch.arange(C), chunk_size)

        all_scores = []

        for chunk in chunks:
            pf = feats_cpu[chunk].to(device)           # (chunk_size, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)  # (B, chunk_size, K)
            scores = sims.max(dim=2).values           # (B, chunk_size)
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=1)     # (B, C)
        preds = all_scores.argmax(dim=1)

        return preds
