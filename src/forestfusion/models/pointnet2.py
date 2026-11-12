"""PointNet++ (Qi et al., 2017) in pure PyTorch.

No custom CUDA kernels, so it runs on CPU/GPU/MPS and exports to ONNX. Two
deviations from the reference implementation, both deliberate:

* Grouping uses kNN with an optional radius cap (neighbours outside the radius
  are replaced by the nearest one, as in the original ball query fallback).
* Sampling is pluggable: ``fps`` (farthest point sampling, accurate but
  sequential) or ``prefix`` (take the first S points; inputs are already randomly
  ordered by the data loader, so this is random sampling at zero cost and is the
  edge-friendly option). Comparing the two is part of the latency/accuracy study.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances. a: (B, N, C), b: (B, M, C) -> (B, N, M). ONNX friendly."""
    d = -2 * torch.matmul(a, b.transpose(1, 2))
    d = d + (a**2).sum(-1, keepdim=True) + (b**2).sum(-1).unsqueeze(1)
    return d.clamp_min(0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points. points: (B, N, C), idx: (B, ...) -> (B, ..., C)."""
    b, c = points.shape[0], points.shape[-1]
    flat = idx.reshape(b, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, c))
    return out.reshape(*idx.shape, c)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Deterministic FPS starting from point 0. xyz: (B, N, 3) -> (B, npoint) indices."""
    b, n, _ = xyz.shape
    distance = torch.full((b, n), 1e10, device=xyz.device, dtype=xyz.dtype)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    picked = []
    for _ in range(npoint):
        picked.append(farthest)
        centroid = index_points(xyz, farthest[:, None])  # (B, 1, 3)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return torch.stack(picked, dim=1)


def sample(xyz: torch.Tensor, npoint: int, method: str) -> torch.Tensor:
    if method == "fps":
        return farthest_point_sample(xyz, npoint)
    if method == "prefix":
        b = xyz.shape[0]
        return torch.arange(npoint, device=xyz.device).unsqueeze(0).expand(b, -1)
    raise ValueError(f"Unknown sampling method: {method}")


def knn_group(query: torch.Tensor, ref: torch.Tensor, k: int, radius: float | None) -> torch.Tensor:
    """Indices of k nearest ``ref`` points for each ``query`` point -> (B, S, k)."""
    d = square_distance(query, ref)
    dist, idx = torch.topk(d, k, dim=-1, largest=False)
    if radius is not None:
        idx = torch.where(dist > radius**2, idx[..., :1].expand_as(idx), idx)
    return idx


def _mlp2d(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for cin, cout in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _mlp1d(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for cin, cout in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        k: int,
        in_ch: int,
        mlp: list[int],
        sampling: str = "fps",
    ) -> None:
        super().__init__()
        self.npoint, self.radius, self.k, self.sampling = npoint, radius, k, sampling
        self.mlp = _mlp2d([in_ch + 3, *mlp])
        self.out_ch = mlp[-1]

    def forward(
        self, xyz: torch.Tensor, feats: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """xyz: (B, N, 3), feats: (B, N, C) or None -> new_xyz (B, S, 3), new_feats (B, S, C')."""
        if self.npoint is None:  # group all
            new_xyz = xyz.mean(1, keepdim=True)
            grouped_xyz = (xyz - new_xyz).unsqueeze(1)  # (B, 1, N, 3)
            grouped = grouped_xyz if feats is None else torch.cat([grouped_xyz, feats.unsqueeze(1)], -1)
        else:
            new_xyz = index_points(xyz, sample(xyz, self.npoint, self.sampling))
            idx = knn_group(new_xyz, xyz, self.k, self.radius)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # (B, S, k, 3)
            if self.radius:
                grouped_xyz = grouped_xyz / self.radius
            grouped = grouped_xyz if feats is None else torch.cat([grouped_xyz, index_points(feats, idx)], -1)
        x = self.mlp(grouped.permute(0, 3, 2, 1))  # (B, C, k, S)
        return new_xyz, x.max(dim=2).values.transpose(1, 2)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, mlp: list[int]) -> None:
        super().__init__()
        self.mlp = _mlp1d([in_ch, *mlp])
        self.out_ch = mlp[-1]

    def forward(self, xyz1, xyz2, feats1, feats2) -> torch.Tensor:
        """Interpolate feats2 (sparse, at xyz2) onto xyz1 by inverse-distance 3-NN, concat skip feats1."""
        if xyz2.shape[1] == 1:
            interp = feats2.expand(-1, xyz1.shape[1], -1)
        else:
            d = square_distance(xyz1, xyz2)
            dist, idx = torch.topk(d, 3, dim=-1, largest=False)
            w = 1.0 / (dist + 1e-8)
            w = w / w.sum(-1, keepdim=True)
            interp = (index_points(feats2, idx) * w.unsqueeze(-1)).sum(2)
        x = interp if feats1 is None else torch.cat([feats1, interp], -1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


def _w(c: int, width: float) -> int:
    return max(8, int(round(c * width)))


class PointNet2Seg(nn.Module):
    """Per-point semantic segmentation. Input (B, N, 3 + extra) -> logits (B, n_classes, N)."""

    def __init__(
        self,
        n_classes: int,
        extra_ch: int = 0,
        width: float = 1.0,
        sampling: str = "fps",
        npoints: tuple[int, ...] = (1024, 256, 64, 16),
        radii: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
        k: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.extra_ch = extra_ch
        widths = [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]]
        self.sa = nn.ModuleList()
        ch, skip_ch = extra_ch, [extra_ch]
        for npt, r, mlp in zip(npoints, radii, widths):
            sa = SetAbstraction(npt, r, k, ch, [_w(c, width) for c in mlp], sampling)
            self.sa.append(sa)
            ch = sa.out_ch
            skip_ch.append(ch)
        fp_mlps = [[256, 256], [256, 256], [256, 128], [128, 128, 128]]
        self.fp = nn.ModuleList()
        for i, mlp in enumerate(fp_mlps):
            fp = FeaturePropagation(ch + skip_ch[-(i + 2)], [_w(c, width) for c in mlp])
            self.fp.append(fp)
            ch = fp.out_ch
        self.head = nn.Sequential(
            nn.Conv1d(ch, ch, 1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Conv1d(ch, n_classes, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:] if self.extra_ch else None
        xyzs, featss = [xyz], [feats]
        for sa in self.sa:
            xyz, feats = sa(xyz, feats)
            xyzs.append(xyz)
            featss.append(feats)
        f = featss[-1]
        for i, fp in enumerate(self.fp):
            j = len(self.sa) - 1 - i
            f = fp(xyzs[j], xyzs[j + 1], featss[j], f)
        return self.head(f.transpose(1, 2))


class PointNet2Encoder(nn.Module):
    """Global shape descriptor for a single object (tree). Input (B, N, 3) -> (B, out_dim)."""

    def __init__(self, width: float = 1.0, sampling: str = "fps", npoints=(512, 128)) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(npoints[0], 0.2, 32, 0, [_w(c, width) for c in (64, 64, 128)], sampling)
        self.sa2 = SetAbstraction(
            npoints[1], 0.4, 64, self.sa1.out_ch, [_w(c, width) for c in (128, 128, 256)], sampling
        )
        self.sa3 = SetAbstraction(None, None, 0, self.sa2.out_ch, [_w(c, width) for c in (256, 512, 1024)])
        self.out_dim = self.sa3.out_ch

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz, f = self.sa1(points[..., :3].contiguous(), None)
        xyz, f = self.sa2(xyz, f)
        _, f = self.sa3(xyz, f)
        return f.squeeze(1)


def segmentation_loss(
    logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None,
    ignore_index: int = 255, label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=weight, ignore_index=ignore_index,
                           label_smoothing=label_smoothing)
