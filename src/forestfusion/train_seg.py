"""Train PointNet++ semantic segmentation on plot-level forest lidar.

    python -m forestfusion.train_seg --config configs/seg_forinstance.yaml
    python -m forestfusion.train_seg --config configs/seg_synthetic.yaml train.epochs=2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.segmentation import IGNORE_INDEX, SegmentationDataset
from .engine import RunLogger, class_weights, cosine_with_warmup, recalibrate_bn, resolve_files
from .metrics import ConfusionMatrix
from .models import PointNet2Seg
from .models.pointnet2 import segmentation_loss
from .utils import count_parameters, get_device, load_config, seed_everything


def build_model(cfg: dict) -> PointNet2Seg:
    m = cfg["model"]
    return PointNet2Seg(
        n_classes=len(cfg["data"]["class_names"]),
        width=m.get("width", 1.0),
        sampling=m.get("sampling", "fps"),
        k=m.get("k", 32),
    )


@torch.no_grad()
def evaluate(model, loader, n_classes: int, device) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(n_classes, IGNORE_INDEX)
    for batch in loader:
        logits = model(batch["points"].to(device))
        cm.update(logits.argmax(1).cpu().numpy(), batch["labels"].numpy())
    return cm


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=5")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    d, t = cfg["data"], cfg["train"]
    label_map = {int(k): int(v) for k, v in d["label_map"].items()}
    n_classes = len(d["class_names"])

    common = dict(label_map=label_map, n_points=d["n_points"], radius=d["radius"])
    train_files = resolve_files(d["train_files"], d.get("root", "."))
    val_files = resolve_files(d["val_files"], d.get("root", "."))
    print(f"train plots: {len(train_files)} | val plots: {len(val_files)} | device: {device}")
    train_ds = SegmentationDataset(train_files, split="train",
                                   samples_per_epoch=d["samples_per_epoch"], **common)
    val_ds = SegmentationDataset(val_files, split="val", **common)

    loader_kw = dict(batch_size=t["batch_size"], num_workers=t.get("num_workers", 0),
                     pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    print(f"PointNet2Seg: {count_parameters(model) / 1e6:.2f} M parameters")
    weight = class_weights(train_ds.class_frequencies(n_classes), t.get("class_weighting"))
    weight = weight.to(device) if weight is not None else None

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    total = t["epochs"] * len(train_dl)
    sched = cosine_with_warmup(opt, int(0.05 * total), total)
    use_amp = t.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger = RunLogger(cfg["output_dir"], cfg)

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_dl:
            pts, lab = batch["points"].to(device), batch["labels"].to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = segmentation_loss(model(pts), lab, weight, IGNORE_INDEX, t.get("label_smoothing", 0.0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0]}
        if epoch % t.get("eval_every", 1) == 0 or epoch == t["epochs"]:
            recalibrate_bn(model, train_dl, lambda b: model(b["points"].to(device)),
                           t.get("precise_bn_batches", 50))
            cm = evaluate(model, val_dl, n_classes, device)
            record.update({f"val_{k}": v for k, v in cm.summary().items()})
            record.update({f"iou_{n}": float(v) for n, v in zip(d["class_names"], cm.iou())})
            is_best = logger.checkpoint(model, cfg, cm.miou(), {"class_names": d["class_names"]})
            record["best"] = is_best
        logger.log(record)
    print(f"best val mIoU: {logger.best:.4f} -> {logger.dir / 'best.pt'}")
    return logger.best


if __name__ == "__main__":
    main()
