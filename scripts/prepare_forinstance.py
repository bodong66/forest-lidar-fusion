"""Create train / val / test file lists for FOR-instance.

    python scripts/prepare_forinstance.py --root data/FORinstance --out data/splits

Uses the dataset's own split table if one is found (a CSV with a path column and a
dev/test split column), and holds out a fraction of the dev plots per region for
validation so every forest type appears in both train and val. If no split table is
found, it falls back to a random plot-level split and prints a warning.

Check the dataset README after downloading: if the column names differ, pass
--path-col / --split-col.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="unzipped FOR-instance folder")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)

    all_las = {p.relative_to(root).as_posix(): p for p in root.rglob("*.la[sz]")}
    split_csv = next(iter(root.rglob("*split*.csv")), None)
    dev, test = [], []
    if split_csv:
        print(f"using split table {split_csv}")
        with open(split_csv, newline="") as f:
            for row in csv.DictReader(f):
                rel = row[args.path_col].lstrip("./")
                match = next((p for k, p in all_las.items() if k.endswith(rel)), None)
                if match is None:
                    continue
                (test if row[args.split_col].strip().lower() == "test" else dev).append(match)
    else:
        print("WARNING: no split table found, using a random 80/20 plot split as test")
        files = sorted(all_las.values())
        rnd.shuffle(files)
        n_test = max(1, len(files) // 5)
        test, dev = files[:n_test], files[n_test:]

    by_region: dict[str, list[Path]] = defaultdict(list)
    for p in dev:
        by_region[p.relative_to(root).parts[0]].append(p)
    train, val = [], []
    for region, files in sorted(by_region.items()):
        files = sorted(files)
        rnd.shuffle(files)
        n_val = max(1, round(len(files) * args.val_fraction)) if len(files) > 1 else 0
        val += files[:n_val]
        train += files[n_val:]
        print(f"{region:>10}: {len(files) - n_val} train, {n_val} val")

    for name, files in [("train", train), ("val", val), ("test", test)]:
        (out / f"forinstance_{name}.txt").write_text("\n".join(str(p) for p in sorted(files)) + "\n")
        print(f"{name}: {len(files)} plots -> {out / f'forinstance_{name}.txt'}")


if __name__ == "__main__":
    main()
