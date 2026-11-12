import numpy as np
import pytest
import torch

from forestfusion.models import PointNet2Seg, SpeciesClassifier
from forestfusion.models.pointnet2 import farthest_point_sample, knn_group, square_distance


def test_square_distance_matches_cdist():
    a, b = torch.rand(2, 50, 3), torch.rand(2, 40, 3)
    assert torch.allclose(square_distance(a, b), torch.cdist(a, b) ** 2, atol=1e-5)


def test_fps_picks_spread_out_points():
    # Two tight clusters far apart: FPS with 2 samples must pick one from each.
    xyz = torch.cat([torch.rand(1, 50, 3) * 0.01, torch.rand(1, 50, 3) * 0.01 + 10], 1)
    idx = farthest_point_sample(xyz, 2)[0]
    assert (idx < 50).sum() == 1 and (idx >= 50).sum() == 1


def test_knn_radius_cap_replaces_far_neighbours():
    ref = torch.tensor([[[0.0, 0, 0], [0.05, 0, 0], [5.0, 0, 0]]])
    idx = knn_group(ref[:, :1], ref, 3, radius=0.1)
    assert idx[0, 0].tolist() == [0, 1, 0]


@pytest.mark.parametrize("sampling", ["fps", "prefix"])
def test_seg_output_shape(sampling):
    model = PointNet2Seg(n_classes=5, width=0.25, sampling=sampling, k=8).eval()
    out = model(torch.rand(2, 1024, 3))
    assert out.shape == (2, 5, 1024)


@pytest.mark.parametrize("modality,fusion", [
    ("points", "concat"), ("views", "concat"),
    ("fusion", "concat"), ("fusion", "gated"), ("fusion", "attention"),
])
def test_species_variants(modality, fusion):
    model = SpeciesClassifier(4, modality=modality, fusion=fusion, width=0.25, n_views=7,
                              modality_dropout=0.3)
    pts, views, h = torch.rand(3, 512, 3), torch.rand(3, 7, 32, 32), torch.rand(3, 1)
    model.train()
    loss = model(pts, views, h).sum()
    loss.backward()
    model.eval()
    assert model(pts, views, h).shape == (3, 4)


def test_seg_is_permutation_invariant_with_fps_input_order_fixed():
    # Max-pooling over neighbourhoods: shuffling non-sampled points must not change outputs much.
    torch.manual_seed(0)
    model = PointNet2Seg(3, width=0.25, sampling="fps", k=8).eval()
    x = torch.rand(1, 512, 3)
    perm = torch.cat([torch.tensor([0]), torch.randperm(511) + 1])  # keep FPS start point
    out1 = model(x)
    out2 = model(x[:, perm])[:, :, torch.argsort(perm)]
    assert np.allclose(out1.detach().numpy(), out2.detach().numpy(), atol=1e-4)
