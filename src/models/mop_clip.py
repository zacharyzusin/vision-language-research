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
    Mixture-of-Prompts CLIP with:
      • Shared K × ctx_len learnable context tokens (global across classes)
      • EM-style objective over K prompts per class
      • GPU-accelerated prompt cache building (chunked)
      • CPU prompt-cache storage (+ on-disk persistence)
      • Auto-load cache on resume
      • Safe torch.load(weights_only=True)
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

        # Text embed dim (512 for ViT-B/32)
        self.D = self.clip.text_projection.shape[0]

        # ---------------------------
        # Build class prompts
        # ---------------------------
        def make_prompt(md: Dict) -> str:
            return (
                "X " * self.ctx_len
                + f"a photo of {md['species']}, an organism in genus {md['genus']} "
                  f"and family {md['family']} belonging to order {md['order']}."
            )

        prompts = [make_prompt(md) for md in metadata]

        print(f"Tokenizing {len(prompts)} class prompts...")
        class_tokens = clip.tokenize(prompts)  # (C, 77)
        self.register_buffer("class_tokens", class_tokens, persistent=True)

        # Store base token embeddings on CPU
        with torch.no_grad():
            base_emb = self.clip.token_embedding(self.class_tokens)  # (C, L, D)
        self.base_token_embeds_cpu = base_emb.cpu()

        self.positional_embedding_cpu = (
            self.clip.positional_embedding.detach().clone().cpu()
        )  # (L, D)

        self.seq_len = self.base_token_embeds_cpu.shape[1]

        # ---------------------------
        # Shared context tokens (K × ctx_len × D)
        # ---------------------------
        ctx = torch.empty(self.K, self.ctx_len, self.D)
        nn.init.normal_(ctx, std=0.02)
        self.context_tokens = nn.Parameter(ctx)

        print(
            f"Initialized SHARED context tokens ({self.K}, {self.ctx_len}, {self.D}) "
            f"(~{ctx.numel()/1e6:.3f}M params)"
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

        self._prompt_cache_cpu = None

        # Try loading disk cache (safe torch.load)
        if os.path.exists(self.prompt_cache_path):
            try:
                data = torch.load(
                    self.prompt_cache_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self._prompt_cache_cpu = data["prompt_feats"]
                print(
                    f"[MixturePromptCLIP] Loaded existing prompt cache from "
                    f"{self.prompt_cache_path}"
                )
            except Exception as e:
                print(
                    f"[MixturePromptCLIP] Warning: cache exists but failed to load: {e}"
                )

    # ======================================================================
    # GPU prompt encoder for training
    # ======================================================================
    def _encode_prompts_for_classes_gpu(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode prompts for the given batch of class_indices.
        Returns (B, K, D).
        """
        device = class_indices.device
        class_indices = class_indices.long()

        # Collapse repeated labels
        unique, inv = torch.unique(class_indices, sorted=True, return_inverse=True)
        C_u = unique.numel()

        base_emb = self.base_token_embeds_cpu.to(device)      # (C, L, D)
        x = base_emb[unique]                                  # (C_u, L, D)

        # Expand for K prompts
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()  # (C_u, K, L, D)

        # Insert shared context
        ctx = self.context_tokens.to(device)  # (K, ctx_len, D)
        ctx_expanded = ctx.unsqueeze(0).expand(C_u, -1, -1, -1)
        x[:, :, 1:1 + self.ctx_len, :] = ctx_expanded

        # Add positional embeddings
        pos = self.positional_embedding_cpu.to(device)  # (L, D)
        x = x + pos

        # Flatten for CLIP text encoder
        C_uK = C_u * self.K
        x = x.view(C_uK, self.seq_len, -1).permute(1, 0, 2)  # (L, C_uK, D)

        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)

        # EOT index
        tok = self.class_tokens.to(device)[unique]
        tok = tok.unsqueeze(1).expand(-1, self.K, -1).reshape(C_uK, -1)
        eot = tok.argmax(dim=-1)

        text_emb = x[torch.arange(C_uK, device=device), eot]
        text_emb = text_emb @ self.clip.text_projection
        text_emb = F.normalize(text_emb, dim=-1)

        text_emb = text_emb.view(C_u, self.K, -1)
        return text_emb[inv]  # reorder to batch order

    # ======================================================================
    # GPU chunked encoder for building full cache
    # ======================================================================
    @torch.no_grad()
    def _encode_prompts_for_classes_gpu_cache(self, class_indices: torch.Tensor):
        """
        Same encoder as above, but returns results on CPU.
        Used during cache-building.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for GPU cache building.")

        device = torch.device("cuda")
        self.clip.to(device)

        feats = self._encode_prompts_for_classes_gpu(class_indices.to(device))
        return feats.cpu()

    # ======================================================================
    # Training forward
    # ======================================================================
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = images.device

        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)

        prompt_feats = self._encode_prompts_for_classes_gpu(labels)  # (B, K, D)

        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)

        gamma = F.softmax(sims / self.em_tau, dim=1).detach()

        loss = -(gamma * sims).sum(dim=1).mean()
        return loss

    # ======================================================================
    # Build full class prompt cache on GPU (chunked)
    # ======================================================================
    @torch.no_grad()
    def _build_prompt_cache_cpu(self):
        if self._prompt_cache_cpu is not None:
            return

        # Safe load from disk
        if os.path.exists(self.prompt_cache_path):
            print(f"[MixturePromptCLIP] Loading prompt cache from {self.prompt_cache_path}")
            data = torch.load(
                self.prompt_cache_path,
                map_location="cpu",
                weights_only=True,
            )
            self._prompt_cache_cpu = data["prompt_feats"]
            print("[MixturePromptCLIP] Prompt cache loaded from disk.")
            return

        # Build cache
        print("[MixturePromptCLIP] Building prompt cache on GPU (chunked)...")

        if not torch.cuda.is_available():
            raise RuntimeError("Cannot cache prompts on CPU when CUDA is unavailable.")

        chunk_size = 64
        all_classes = torch.arange(self.num_classes, device="cuda")

        chunks = list(range(0, self.num_classes, chunk_size))
        feats_list = []

        pbar = tqdm(chunks, desc="GPU prompt chunks", ncols=80)
        for start in pbar:
            end = min(start + chunk_size, self.num_classes)
            chunk = all_classes[start:end]
            f = self._encode_prompts_for_classes_gpu_cache(chunk)
            feats_list.append(f)
            torch.cuda.empty_cache()

        feats = torch.cat(feats_list, dim=0)  # (C, K, D)
        self._prompt_cache_cpu = feats

        # Save to disk
        torch.save({"prompt_feats": feats}, self.prompt_cache_path)
        print(f"[MixturePromptCLIP] Prompt cache saved to {self.prompt_cache_path}")

    # ======================================================================
    # Inference
    # ======================================================================
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device

        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)
        B = img_feat.shape[0]

        # Ensure cache exists
        if self._prompt_cache_cpu is None:
            self._build_prompt_cache_cpu()

        feats_cpu = self._prompt_cache_cpu
        C = feats_cpu.shape[0]

        chunk_size = 512
        class_chunks = torch.split(torch.arange(C), chunk_size)

        all_scores = []

        for chunk in class_chunks:
            pf = feats_cpu[chunk].to(device)          # (chunk_size, K, D)
            sims = torch.einsum("bd,ckd->bck", img_feat, pf)
            scores = sims.max(dim=2).values          # (B, chunk_size)
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=1)    # (B, C)
        preds = all_scores.argmax(dim=1)
        return preds
