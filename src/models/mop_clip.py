import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm


class MixturePromptCLIP(nn.Module):
    """
    Fully upgraded Mixture-of-Prompts CLIP:
        - hierarchical templates
        - learnable template weights
        - K=32 mixture prompts
        - learnable temperature for subprompt pooling
        - caching of text features for instant startup
    """

    def __init__(
        self,
        clip_model: str,
        metadata: dict,
        K: int = 32,
        ctx_len: int = 8,
        cache_dir: str = "text_cache"
    ):
        super().__init__()

        self.clip, self.preprocess = clip.load(clip_model, device="cpu")
        self.clip.eval()

        self.metadata = metadata
        self.num_classes = len(metadata)
        self.K = K
        self.ctx_len = ctx_len
        safe_name = clip_model.replace("/", "_")
        self.cache_path = os.path.join(cache_dir, f"{safe_name}_inat_cache.pt")

        # -------------------------------
        # Build hierarchical prompts
        # -------------------------------
        templates = [
            "a photo of a {species}",
            "a wildlife photo of a {species}, a member of the genus {genus}",
            "a nature photograph of {species}, belonging to the family {family}",
            "a natural habitat photo of the species {species} in the order {order}",
            "a high quality image of the species {scientific_name}",
            "a close-up photo of a {species}",
        ]

        # Build prompt strings
        prompts = []
        for i in range(self.num_classes):
            md = metadata[i]
            species = md["species"]
            genus = md["genus"]
            family = md["family"]
            order = md["order"]
            sci = md["scientific_name"]

            for tmpl in templates:
                p = tmpl.format(
                    species=species,
                    genus=genus,
                    family=family,
                    order=order,
                    scientific_name=sci,
                )
                prompts.append(p)

                
            
        # -------------------------------
        # Load from cache if available
        # -------------------------------
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(self.cache_path):
            print(f"Loading cached text embeddings from {self.cache_path}")
            data = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            text_features = data["text_features"]
        else:
            print(f"Tokenizing {len(prompts)} prompts...")
            text_tokens = clip.tokenize(prompts)

            batch_size = 256
            all_feats = []
            print("Encoding prompts with CLIP text encoder...")
            with torch.no_grad():
                for i in tqdm(range(0, len(text_tokens), batch_size)):
                    batch = text_tokens[i:i+batch_size]
                    batch_feats = self.clip.encode_text(batch).float()
                    batch_feats = F.normalize(batch_feats, dim=-1)
                    all_feats.append(batch_feats)

            text_features = torch.cat(all_feats, dim=0)
            torch.save({"text_features": text_features}, self.cache_path)
            print(f"Saved text embeddings to {self.cache_path}")

        # -------------------------------
        # Collapse into (C, T, D)
        # -------------------------------
        T = len(templates)
        C = self.num_classes
        D = text_features.size(-1)
        self.D = D

        text_features = text_features.view(C, T, D)

        # -------------------------------
        # Learnable weights over templates
        # -------------------------------
        self.template_logits = nn.Parameter(torch.zeros(C, T))

        # Final text embeddings computed on forward() calls
        self.register_buffer("raw_text_features", text_features)

        # -------------------------------
        # Learnable prompt embeddings
        # -------------------------------
        self.prompt_embeds = nn.Parameter(torch.randn(C, K, D) * 0.01)

        # -------------------------------
        # Learnable temperature for pooling
        # -------------------------------
        self.temp = nn.Parameter(torch.tensor(5.0))


    def get_class_text_features(self):
        """Compute final weighted text embeddings per class."""
        # template_logits: (C, T)
        weights = F.softmax(self.template_logits, dim=1).unsqueeze(-1)   # (C, T, 1)
        out = (weights * self.raw_text_features).sum(dim=1)              # (C, D)
        return F.normalize(out, dim=-1)


    def forward(self, images, labels):
        """Training forward pass."""
        B = images.size(0)
        device = images.device

        # image embeddings
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()    # (B, D)
            img_feat = F.normalize(img_feat, dim=-1)

        # get class text features (C, D)
        class_text = self.get_class_text_features()[labels]      # (B, D)

        prompts = self.prompt_embeds[labels]                     # (B, K, D)
        combined = F.normalize(class_text.unsqueeze(1) + prompts, dim=-1)  # (B,K,D)

        sims = torch.einsum("bd,bkd->bk", img_feat, combined)    # (B, K)
        gamma = F.softmax(sims, dim=1)
        return -(gamma * sims).sum(dim=1).mean()


    @torch.no_grad()
    def predict(self, images):
        """Inference over all classes."""
        img_feat = self.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)  # (B, D)
        B = img_feat.size(0)

        class_text = self.get_class_text_features()              # (C, D)
        prompts = self.prompt_embeds                              # (C, K, D)

        subprompts = F.normalize(class_text.unsqueeze(1) + prompts, dim=-1)
        sims = torch.einsum("bd,ckd->bck", img_feat, subprompts)  # (B,C,K)

        w = F.softmax(self.temp * sims, dim=2)                   # (B,C,K)
        scores = (w * sims).sum(dim=2)                           # (B,C)

        return scores.argmax(dim=1)
