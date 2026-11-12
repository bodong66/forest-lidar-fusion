"""Train tree species classifiers: point-only, view-only or fused.

    python -m forestfusion.train_species --config configs/species_forspecies20k.yaml model.modality=points
    python -m forestfusion.train_species --config configs/species_forspecies20k.yaml model.fusion=gated

Run the three modalities (and three fusion types) with the same seed to get the
ablation table for the README.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data.species import SpeciesDataset, read_species_metadata
from .engine import RunLogger, class_weights, cosine_with_warmup, recalibrate_bn
from .metrics import ConfusionMatrix
from .models import SpeciesClassifier
from .utils import count_parameters, get_device, load_config, seed_everything


def build_model(cfg: dict, n_classes: int) -> SpeciesClassifier:
    m, d = cfg["model"], cfg["data"]
    return SpeciesClassifier(
        n_classes=n_classes,
        modality=m.get("modality", "fusion"),
        fusion=m.get("fusion", "attention"),
        width=m.get("width", 1.0),
        sampling=m.get("sampling", "fps"),
        n_views=d["n_views"] + int(d.get("top_view", True)),
        use_height=m.get("use_height", True),
        modality_dropout=m.get("modality_dropout", 0.0),
    )


def split_items(items: list[dict], val_fraction: float, seed: int, min_count: int):
    """Stratified train/val split; species with fewer than ``min_count`` trees are dropped."""
    counts: dict[str, int] = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    items = [it for it in items if counts[it["label"]] >= min_count]
    classes = sorted({it["label"] for it in items})
    labels = [it["label"] for it in items]
    tr, va = train_test_split(items, test_size=val_fraction, stratify=labels, random_state=seed)
    return tr, va, classes


@torch.no_grad()
def evaluate(model, loader, n_classes: int, device):
    model.eval()
    cm = ConfusionMatrix(n_classes)
    for b in loader:
        logits = model(b["points"].to(device), b["views"].to(device), b["height"].to(device))
        cm.update(logits.argmax(1).cpu().numpy(), b["label"].numpy())
    return cm


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))
    d, t = cfg["data"], cfg["train"]

    items = read_species_metadata(d["metadata_csv"], d["root"], d.get("id_col", "treeID"),
                                  d.get("label_col", "species"), d.get("group_col"))
    train_items, val_items, classes = split_items(items, d.get("val_fraction", 0.15), seed,
                                                  d.get("min_trees_per_class", 10))
    n_classes = len(classes)
    print(f"{len(train_items)} train / {len(val_items)} val trees, {n_classes} species | device: {device}")

    ds_kw = dict(classes=classes, n_points=d["n_points"], n_views=d["n_views"],
                 image_size=d["image_size"], top_view=d.get("top_view", True),
                 cache_dir=d.get("cache_dir"))
    train_ds = SpeciesDataset(train_items, split="train", **ds_kw)
    val_ds = SpeciesDataset(val_items, split="val", **ds_kw)

    counts = train_ds.class_counts()
    loader_kw = dict(batch_size=t["batch_size"], num_workers=t.get("num_workers", 0),
                     pin_memory=device.type == "cuda")
    if t.get("balanced_sampler", False):
        w = (1.0 / np.sqrt(counts))[[train_ds.class_to_idx[it["label"]] for it in train_items]]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(train_items))
        train_dl = DataLoader(train_ds, sampler=sampler, drop_last=True, **loader_kw)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg, n_classes).to(device)
    print(f"SpeciesClassifier[{cfg['model'].get('modality')}/{cfg['model'].get('fusion')}]: "
          f"{count_parameters(model) / 1e6:.2f} M parameters")
    weight = class_weights(counts / counts.sum(), t.get("class_weighting"))
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
        for b in train_dl:
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(b["points"].to(device), b["views"].to(device), b["height"].to(device))
                loss = F.cross_entropy(logits, b["label"].to(device), weight=weight,
                                       label_smoothing=t.get("label_smoothing", 0.0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0]}
        if epoch % t.get("eval_every", 1) == 0 or epoch == t["epochs"]:
            recalibrate_bn(model, train_dl, lambda b: model(b["points"].to(device), b["views"].to(device),
                                                             b["height"].to(device)),
                           t.get("precise_bn_batches", 50))
            cm = evaluate(model, val_dl, n_classes, device)
            s = cm.summary()
            record.update({f"val_{k}": v for k, v in s.items() if k != "miou"})
            if logger.checkpoint(model, cfg, s["balanced_acc"], {"classes": classes}):
                record["best"] = True
                np.save(logger.dir / "confusion_best.npy", cm.mat)
                (logger.dir / "classes.json").write_text(json.dumps(classes))
        logger.log(record)
    print(f"best val balanced accuracy: {logger.best:.4f} -> {logger.dir / 'best.pt'}")
    return logger.best


if __name__ == "__main__":
    main()
