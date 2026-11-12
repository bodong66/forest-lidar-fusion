"""Species classifier with point-only, view-only and fused variants.

Fusion strategies (the ablation axis of the project):
  * ``concat``    – late feature concatenation, the standard baseline
  * ``gated``     – learned per-channel gate deciding how much to trust each modality
  * ``attention`` – the point-cloud token queries the per-view tokens (cross-attention),
                    so the model can pick the most informative viewpoints per tree

``modality_dropout`` randomly blanks one modality during training, which makes
the fused model degrade gracefully when a sensor is missing in the field.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .multiview import MultiViewEncoder
from .pointnet2 import PointNet2Encoder

MODALITIES = ("points", "views", "fusion")
FUSIONS = ("concat", "gated", "attention")


class SpeciesClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        modality: str = "fusion",
        fusion: str = "attention",
        width: float = 1.0,
        sampling: str = "fps",
        n_views: int = 7,
        view_channels: int = 1,
        d_model: int = 256,
        use_height: bool = True,
        modality_dropout: float = 0.0,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        assert modality in MODALITIES and fusion in FUSIONS
        self.modality, self.fusion = modality, fusion
        self.use_height, self.modality_dropout = use_height, modality_dropout

        if modality in ("points", "fusion"):
            self.point_enc = PointNet2Encoder(width, sampling)
            self.point_proj = nn.Sequential(nn.Linear(self.point_enc.out_dim, d_model), nn.LayerNorm(d_model))
        if modality in ("views", "fusion"):
            self.view_enc = MultiViewEncoder(view_channels, width, n_views)
            self.view_proj = nn.Sequential(nn.Linear(self.view_enc.out_dim, d_model), nn.LayerNorm(d_model))
        if modality == "fusion":
            if fusion == "gated":
                self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
            elif fusion == "attention":
                self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
                self.attn_norm = nn.LayerNorm(d_model)

        fused_dim = 2 * d_model if modality == "fusion" and fusion in ("concat", "attention") else d_model
        if use_height:
            self.height_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
            fused_dim += 32
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def _drop_modality(self, fp: torch.Tensor, fv: torch.Tensor, tokens: torch.Tensor):
        """Per sample, blank the point OR view features with prob ``modality_dropout`` (never both)."""
        if not (self.training and self.modality_dropout > 0):
            return fp, fv, tokens
        b = fp.shape[0]
        r = torch.rand(b, device=fp.device)
        drop_p = (r < self.modality_dropout / 2).float().unsqueeze(-1)
        drop_v = ((r >= self.modality_dropout / 2) & (r < self.modality_dropout)).float().unsqueeze(-1)
        return fp * (1 - drop_p), fv * (1 - drop_v), tokens * (1 - drop_v).unsqueeze(1)

    def forward(
        self, points: torch.Tensor, views: torch.Tensor, height: torch.Tensor
    ) -> torch.Tensor:
        """Always takes all three inputs so every variant shares one interface / ONNX signature."""
        if self.modality == "points":
            f = self.point_proj(self.point_enc(points))
        elif self.modality == "views":
            f = self.view_proj(self.view_enc(views)).max(1).values
        else:
            fp = self.point_proj(self.point_enc(points))
            tokens = self.view_proj(self.view_enc(views))
            fv = tokens.max(1).values
            fp, fv, tokens = self._drop_modality(fp, fv, tokens)
            if self.fusion == "concat":
                f = torch.cat([fp, fv], -1)
            elif self.fusion == "gated":
                g = self.gate(torch.cat([fp, fv], -1))
                f = g * fp + (1 - g) * fv
            else:
                q = fp.unsqueeze(1)
                kv = torch.cat([q, tokens], 1)
                att, _ = self.attn(q, kv, kv, need_weights=False)
                f = torch.cat([self.attn_norm(fp + att.squeeze(1)), fv], -1)
        if self.use_height:
            f = torch.cat([f, self.height_embed(height)], -1)
        return self.head(f)
