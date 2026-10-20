"""Vision encoders: SigLIP2, optionally fused with DINOv2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class VisionTower(nn.Module):
    """Input: images normalised with mean=std=0.5 at 384px (SigLIP convention).
    Output: (B, 729, C) patch features.

    If a DINOv2 model name is given, its 27x27 patch features (fed at 378px, so the grid
    matches SigLIP's) are concatenated channel-wise.
    """

    def __init__(self, siglip_name: str, dinov2_name: str | None = None, dtype=torch.bfloat16):
        super().__init__()
        full = AutoModel.from_pretrained(siglip_name, dtype=dtype)
        self.siglip = full.vision_model  # drop the text tower
        del full
        cfg = self.siglip.config
        self.grid = cfg.image_size // cfg.patch_size  # 384 // 14 = 27
        self.out_dim = cfg.hidden_size  # 1152
        self.dino = None
        if dinov2_name:
            try:
                import timm
            except ImportError as e:
                raise ImportError("DINOv2 needs timm: pip install 'qwenvl-siglip[dino]'") from e
            self.dino = timm.create_model(
                dinov2_name, pretrained=True, num_classes=0, dynamic_img_size=True
            ).to(dtype)
            self.dino_res = self.grid * 14  # 378 -> 27x27 patches too
            self.out_dim += self.dino.num_features  # + 1024
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("im_mean", mean, persistent=False)
            self.register_buffer("im_std", std, persistent=False)

    def forward(self, pix: torch.Tensor) -> torch.Tensor:
        siglip_dtype = self.siglip.embeddings.patch_embedding.weight.dtype
        feats = self.siglip(pixel_values=pix.to(siglip_dtype)).last_hidden_state
        if self.dino is not None:
            x = pix.float() * 0.5 + 0.5  # undo SigLIP normalisation
            x = F.interpolate(
                x, size=(self.dino_res,) * 2, mode="bilinear", align_corners=False, antialias=True
            )
            x = ((x - self.im_mean) / self.im_std).to(next(self.dino.parameters()).dtype)
            d = self.dino.forward_features(x)[:, self.dino.num_prefix_tokens :]
            if d.shape[1] != feats.shape[1]:
                raise RuntimeError(f"token grids differ: dino {d.shape} vs siglip {feats.shape}")
            feats = torch.cat([feats, d.to(feats.dtype)], dim=-1)
        return feats
