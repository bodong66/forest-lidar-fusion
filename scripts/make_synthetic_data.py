"""Write a small synthetic dataset in the same on-disk layout as the real ones.

    python scripts/make_synthetic_data.py --out data/synthetic

Creates:
  data/synthetic/plots/{train,val}/plot_XX.las   FOR-instance-like (classification + treeID)
  data/synthetic/trees/*.las + tree_metadata.csv FOR-species20K-like (treeID, species)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import laspy
import numpy as np

from forestfusion.data.synthetic import SPECIES, make_synthetic_plot, make_synthetic_tree


def write_las(path: Path, xyz: np.ndarray, classification=None, tree_id=None) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(0)
    header.scales = np.array([0.001, 0.001, 0.001])
    if tree_id is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    if tree_id is not None:
        las.treeID = tree_id.astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--plots", type=int, default=4)
    ap.add_argument("--trees-per-species", type=int, default=60)
    ap.add_argument("--plot-size", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)

    for i in range(args.plots):
        split = "val" if i == args.plots - 1 else "train"
        p = make_synthetic_plot(rng, size=args.plot_size, n_trees=10, points_per_tree=3000)
        # Real lidar uses projected coordinates far from the origin; mimic that.
        xyz = p["xyz"] + np.array([598_000.0, 6_615_000.0, 100.0])
        write_las(out / "plots" / split / f"plot_{i:02d}.las", xyz, p["classification"], p["treeID"])

    rows = []
    for sp in SPECIES:
        for _ in range(args.trees_per_species):
            tid = len(rows) + 1
            xyz, _ = make_synthetic_tree(sp, rng, n_points=int(rng.integers(1500, 5000)))
            write_las(out / "trees" / f"{tid}.las", xyz)
            rows.append({"treeID": tid, "species": sp, "data_type": "ULS"})
    with open(out / "tree_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["treeID", "species", "data_type"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.plots} plots and {len(rows)} trees to {out}")


if __name__ == "__main__":
    main()
