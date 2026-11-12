"""Image branch: a compact ResNet shared across views (MVCNN-style).

Returns one token per view so the fusion module can attend over views rather
than only seeing a pooled vector. The same encoder can take rendered depth
views or real RGB/multispectral crops (set ``in_ch`` accordingly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity() if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class MultiViewEncoder(nn.Module):
    """(B, V, H, W) or (B, V, C, H, W) -> per-view tokens (B, V, out_dim)."""

    def __init__(self, in_ch: int = 1, width: float = 1.0, n_views: int = 7) -> None:
        super().__init__()
        chans = [max(8, int(c * width)) for c in (32, 64, 128, 256)]
        self.in_ch = in_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chans[0], 3, 1, 1, bias=False), nn.BatchNorm2d(chans[0]), nn.ReLU(inplace=True)
        )
        blocks, cin = [], chans[0]
        for i, c in enumerate(chans):
            blocks.append(BasicBlock(cin, c, 1 if i == 0 else 2))
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.view_embed = nn.Parameter(torch.zeros(1, n_views, cin))  # which viewpoint a token came from
        self.out_dim = cin

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.dim() == 4:
            views = views.unsqueeze(2)
        b, v = views.shape[:2]
        x = self.blocks(self.stem(views.flatten(0, 1)))
        tokens = self.pool(x).flatten(1).view(b, v, -1)
        return tokens + self.view_embed[:, :v]
