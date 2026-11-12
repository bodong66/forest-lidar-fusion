"""Training utilities shared by the segmentation and species scripts."""

from __future__ import annotations

import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def resolve_files(spec: str | list[str], base: str | Path = ".") -> list[str]:
    """Expand a file spec: a list of glob patterns, or a .txt file with one path per line."""
    base = Path(base)
    specs = [spec] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for s in specs:
        p = base / s if not Path(s).is_absolute() else Path(s)
        if p.suffix == ".txt" and p.exists():
            out += [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        else:
            out += sorted(glob.glob(str(p), recursive=True))
    return out


def class_weights(freq: np.ndarray, mode: str | None) -> torch.Tensor | None:
    """Loss weights from class frequencies. 'inv', 'sqrt_inv' (gentler) or None."""
    if not mode:
        return None
    freq = np.maximum(freq, 1e-6)
    w = 1.0 / freq if mode == "inv" else 1.0 / np.sqrt(freq)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def recalibrate_bn(model: torch.nn.Module, loader, forward_fn, n_batches: int) -> None:
    """"Precise BN": recompute BatchNorm running stats with the final weights before eval.

    EMA running stats lag behind fast-changing weights (short schedules, high LR) and are
    skewed by train-only stochasticity such as modality dropout. Re-estimating them on a
    few training batches with that stochasticity switched off removes the train/eval gap.
    """
    if n_batches <= 0:
        return
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the recalibration batches
    md = getattr(model, "modality_dropout", None)
    if md is not None:
        model.modality_dropout = 0.0
    model.train()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        forward_fn(batch)
    for m, mom in zip(bns, saved):
        m.momentum = mom
    if md is not None:
        model.modality_dropout = md
    model.eval()


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class RunLogger:
    """Writes config, per-epoch metrics (JSONL) and checkpoints into one run directory."""

    def __init__(self, out_dir: str | Path, cfg: dict) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        self.best = -float("inf")
        self.t0 = time.time()

    def log(self, record: dict) -> None:
        record = {**record, "elapsed_s": round(time.time() - self.t0, 1)}
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def checkpoint(self, model, cfg: dict, score: float, extra: dict | None = None) -> bool:
        state = {"model": model.state_dict(), "config": cfg, "score": score, **(extra or {})}
        torch.save(state, self.dir / "last.pt")
        if score > self.best:
            self.best = score
            torch.save(state, self.dir / "best.pt")
            return True
        return False
