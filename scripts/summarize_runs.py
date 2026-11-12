"""Collect the best epoch of every run into a markdown table for the README.

    python scripts/summarize_runs.py runs/species_* --metric val_balanced_acc
    python scripts/summarize_runs.py runs/seg_* --metric val_miou
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default="val_miou")
    ap.add_argument("--cols", nargs="*", default=None, help="extra metric columns to show")
    args = ap.parse_args()

    rows = []
    for run in map(Path, args.runs):
        mfile, cfile = run / "metrics.jsonl", run / "config.json"
        if not mfile.exists():
            continue
        records = [json.loads(ln) for ln in mfile.read_text().splitlines() if args.metric in ln]
        if not records:
            continue
        best = max(records, key=lambda r: r[args.metric])
        cfg = json.loads(cfile.read_text()) if cfile.exists() else {}
        m = cfg.get("model", {})
        rows.append((run.name, m, best))

    cols = args.cols or sorted({k for _, _, b in rows for k in b if k.startswith("val_")})
    print("| run | modality | fusion | width | sampling | best epoch | " + " | ".join(cols) + " |")
    print("|---" * (6 + len(cols)) + "|")
    for name, m, b in sorted(rows, key=lambda r: -r[2][args.metric]):
        vals = " | ".join(f"{b.get(c, float('nan')):.3f}" for c in cols)
        print(f"| {name} | {m.get('modality', '-')} | {m.get('fusion', '-')} | {m.get('width', '-')} | "
              f"{m.get('sampling', '-')} | {b['epoch']} | {vals} |")


if __name__ == "__main__":
    main()
