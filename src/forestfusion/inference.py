"""Full-plot inference: overlapping tiles, softmax voting, then tree instances.

    python -m forestfusion.inference --checkpoint runs/seg/best.pt --input plot.las --out pred.las

Writes a LAS file with ``pred_class`` (semantic) and ``pred_tree`` (instance)
extra dimensions, so results open directly in CloudCompare for inspection.
If the input has reference labels, semantic and instance metrics are printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data.io import read_point_cloud
from .data.segmentation import IGNORE_INDEX, build_label_lut
from .data.transforms import normalize_block
from .export import load_model
from .instance import instance_f1, segment_trees
from .metrics import ConfusionMatrix
from .utils import get_device


@torch.no_grad()
def predict_plot(
    model, xyz: np.ndarray, n_classes: int, radius: float, n_points: int,
    stride: float | None = None, batch_size: int = 8, device: torch.device | None = None, seed: int = 0,
) -> np.ndarray:
    """Per-point class probabilities (N, C) by averaging over overlapping crops."""
    device = device or torch.device("cpu")
    model.eval().to(device)
    rng = np.random.default_rng(seed)
    stride = stride or radius / 2
    tree = cKDTree(xyz[:, :2])
    probs = np.zeros((len(xyz), n_classes), dtype=np.float32)
    votes = np.zeros(len(xyz), dtype=np.int32)
    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    crops: list[tuple[np.ndarray, np.ndarray]] = []

    def flush():
        if not crops:
            return
        pts = torch.from_numpy(np.stack([c[1] for c in crops]).astype(np.float32)).to(device)
        p = torch.softmax(model(pts), 1).transpose(1, 2).cpu().numpy()
        for (idx, _), pi in zip(crops, p):
            np.add.at(probs, idx, pi)
            np.add.at(votes, idx, 1)
        crops.clear()

    for cx in np.arange(lo[0], hi[0] + stride, stride):
        for cy in np.arange(lo[1], hi[1] + stride, stride):
            center = np.array([cx, cy])
            idx = np.asarray(tree.query_ball_point(center, radius), dtype=np.int64)
            if len(idx) == 0:
                continue
            perm = rng.permutation(len(idx))
            # Several disjoint subsets so dense tiles are fully covered, not just sampled.
            for start in range(0, len(idx), n_points):
                sub = idx[perm[start:start + n_points]]
                if len(sub) < n_points:
                    sub = np.concatenate([sub, rng.choice(idx, n_points - len(sub))])
                crops.append((sub, normalize_block(xyz[sub], center, radius)))
                if len(crops) == batch_size:
                    flush()
    flush()

    missing = votes == 0
    if missing.any():  # fall back to nearest predicted neighbour
        known = np.flatnonzero(~missing)
        _, nn = cKDTree(xyz[known]).query(xyz[missing])
        probs[missing], votes[missing] = probs[known[nn]], votes[known[nn]]
    return probs / votes[:, None]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    model, cfg = load_model(args.checkpoint, "seg")
    d = cfg["data"]
    class_names = d["class_names"]
    pc = read_point_cloud(args.input, ("classification", "treeID"))
    xyz = pc["xyz"] - pc["xyz"].min(0)
    probs = predict_plot(model, xyz, len(class_names), d["radius"], d["n_points"],
                         device=get_device(args.device))
    pred = probs.argmax(1)

    inst_cfg = cfg.get("instance", {})
    stem = class_names.index(inst_cfg.get("stem_class", "stem"))
    tree_names = inst_cfg.get("tree_classes", ["stem", "live_branches", "woody_branches"])
    tree_cls = tuple(class_names.index(c) for c in tree_names)
    trees = segment_trees(xyz, pred, stem, tree_cls, eps=inst_cfg.get("eps", 0.5),
                          max_dist=inst_cfg.get("max_dist", 6.0))
    result: dict = {"n_points": len(xyz), "n_trees_pred": int(trees.max())}

    if "classification" in pc:
        lut = build_label_lut({int(k): int(v) for k, v in d["label_map"].items()})
        gt = lut[np.clip(pc["classification"].astype(np.int64), 0, 255)]
        cm = ConfusionMatrix(len(class_names), IGNORE_INDEX)
        cm.update(pred, gt)
        result.update(cm.summary())
        result.update({f"iou_{n}": float(v) for n, v in zip(class_names, cm.iou())})
    if "treeID" in pc:
        result.update({f"inst_{k}": v for k, v in instance_f1(trees, pc["treeID"].astype(np.int64)).items()})
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if args.out:
        import laspy

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets, header.scales = pc["xyz"].min(0), np.array([0.001] * 3)
        las = laspy.LasData(header)
        las.x, las.y, las.z = pc["xyz"].T
        las.add_extra_dim(laspy.ExtraBytesParams("pred_class", np.uint8))
        las.add_extra_dim(laspy.ExtraBytesParams("pred_tree", np.int32))
        las.pred_class, las.pred_tree = pred.astype(np.uint8), trees.astype(np.int32)
        las.write(Path(args.out))
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
