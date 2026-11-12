"""Individual tree segmentation baseline built on top of semantic predictions.

A deliberately simple, explainable baseline (stem-seeded region growing):
  1. Cluster predicted stem points in xy with DBSCAN -> one seed per tree.
  2. Assign every predicted tree point (stem / branches) to the nearest seed in xy,
     with a max horizontal distance so isolated points stay unassigned.

Learned instance methods (e.g. offset regression + clustering) should be
compared against this. Evaluation follows the usual matching protocol:
a predicted tree is a true positive if its point IoU with a reference tree > 0.5.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def segment_trees(
    xyz: np.ndarray,
    sem_pred: np.ndarray,
    stem_class: int,
    tree_classes: tuple[int, ...],
    eps: float = 0.5,
    min_samples: int = 10,
    max_dist: float = 6.0,
) -> np.ndarray:
    """Return per-point tree id (0 = not a tree)."""
    inst = np.zeros(len(xyz), dtype=np.int64)
    stem_idx = np.flatnonzero(sem_pred == stem_class)
    if len(stem_idx) < min_samples:
        return inst
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[stem_idx, :2])
    valid = labels >= 0
    if not valid.any():
        return inst
    ids = np.unique(labels[valid])
    seeds = np.stack([xyz[stem_idx[labels == i], :2].mean(0) for i in ids])

    tree_mask = np.isin(sem_pred, tree_classes)
    tree_idx = np.flatnonzero(tree_mask)
    dist, nearest = cKDTree(seeds).query(xyz[tree_idx, :2])
    keep = dist <= max_dist
    inst[tree_idx[keep]] = nearest[keep] + 1
    return inst


def instance_f1(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> dict[str, float]:
    """Greedy one-to-one matching of predicted and reference trees by point IoU."""
    pred_ids = [i for i in np.unique(pred) if i > 0]
    gt_ids = [i for i in np.unique(gt) if i > 0]
    if not pred_ids or not gt_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": len(pred_ids), "n_gt": len(gt_ids)}
    pairs = []
    for g in gt_ids:
        gm = gt == g
        overlap = np.unique(pred[gm], return_counts=True)
        for p, inter in zip(*overlap):
            if p == 0:
                continue
            union = gm.sum() + (pred == p).sum() - inter
            pairs.append((inter / union, g, p))
    pairs.sort(reverse=True)
    used_g, used_p, tp = set(), set(), 0
    for iou, g, p in pairs:
        if iou < iou_thr:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        tp += 1
    prec, rec = tp / len(pred_ids), tp / len(gt_ids)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "n_pred": len(pred_ids), "n_gt": len(gt_ids)}
