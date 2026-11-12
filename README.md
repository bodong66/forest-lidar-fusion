# forest-lidar-fusion

**Deep learning for forest laser scanning: tree component segmentation, multimodal species classification and edge deployment.**

This project tackles three problems that sit at the core of digital forest inventory:

1. **Semantic segmentation of UAV lidar plots** into terrain, low vegetation, stems, live branches and woody branches, followed by **individual tree segmentation**. Evaluated on [FOR-instance](https://zenodo.org/records/8287792).
2. **Tree species classification** from single-tree point clouds, comparing a point-cloud network, a multi-view image network and **three fusion strategies** (concatenation, gating, cross-attention). Evaluated on [FOR-species20K](https://zenodo.org/records/13255198).
3. **Real-time / edge inference**: ONNX export with verified numerical parity, and an accuracy–latency study across model width, point-sampling strategy and runtime. The target is on-board use, for example on harvesters or drones.

Everything is plain PyTorch, with no custom CUDA kernels. The same code therefore trains on a GPU, runs on a laptop CPU and exports to ONNX / TensorRT.

---

## Results

> Results on the real benchmarks are filled in as experiments complete. All numbers come from `scripts/summarize_runs.py` on the run logs in `runs/`.

### Semantic segmentation – FOR-instance (validation split)

| Model | Sampling | Params | mIoU | Terrain | Low veg. | Stem | Live br. | Woody br. |
|---|---|---|---|---|---|---|---|---|
| PointNet++ (w=1.0) | FPS | 0.96 M | TBD | | | | | |
| PointNet++ (w=1.0) | prefix | 0.96 M | TBD | | | | | |
| PointNet++ (w=0.5) | prefix | 0.24 M | TBD | | | | | |

Individual trees (stem-seeded baseline, IoU > 0.5 matching): precision TBD / recall TBD / F1 TBD.

### Species classification – FOR-species20K (stratified hold-out of the dev set)

| Variant | Balanced acc. | Macro F1 | OA |
|---|---|---|---|
| Points only (PointNet++) | TBD | | |
| Views only (multi-view CNN) | TBD | | |
| Fusion – concat | TBD | | |
| Fusion – gated | TBD | | |
| Fusion – cross-attention | TBD | | |
| Fusion – cross-attention, no modality dropout | TBD | | |

### Edge latency – segmentation, 4096 points, batch 1

Measured with `python -m forestfusion.benchmark --task seg --widths 1.0 0.5 0.25 --sampling fps prefix --onnx --threads 2` on a 2-thread cloud CPU (a stand-in for a low-power edge CPU). The numbers are random-weight timings, so they are independent of accuracy.

| Width | Sampling | Runtime | Params (M) | Latency ms (mean / p90) | Samples/s |
|---|---|---|---|---|---|
| 1.0 | FPS | torch-cpu | 0.96 | 154.4 / 167.7 | 6.5 |
| 1.0 | prefix | torch-cpu | 0.96 | 63.7 / 70.3 | 15.7 |
| 1.0 | prefix | onnxruntime-cpu | 0.96 | 48.8 / 51.5 | 20.5 |
| 0.5 | FPS | torch-cpu | 0.24 | 128.4 / 145.3 | 7.8 |
| 0.5 | prefix | torch-cpu | 0.24 | 53.1 / 56.1 | 18.8 |
| 0.5 | prefix | onnxruntime-cpu | 0.24 | 36.2 / 37.1 | 27.6 |
| 0.25 | FPS | torch-cpu | 0.06 | 133.1 / 154.4 | 7.5 |
| 0.25 | prefix | torch-cpu | 0.06 | 47.4 / 54.0 | 21.1 |
| 0.25 | prefix | onnxruntime-cpu | 0.06 | 24.1 / 24.5 | 41.4 |

**Takeaway:** on CPU, farthest point sampling (FPS) dominates latency. Replacing it with prefix sampling makes the model 2–3x faster, and ONNX Runtime adds another 1.3–2x. Shrinking the network by 16x in parameters helps far less. The open question is how much accuracy prefix sampling costs, which the FOR-instance runs answer.

---

## Method

### Segmentation
- **Input:** cylindrical crops of radius 4 m sampled from the plot, normalised to the crop centre. Augmentation is gravity-aware: rotation only about z, flips, scaling and jitter.
- **Model:** PointNet++ with 4 set-abstraction and 4 feature-propagation levels. It uses kNN grouping with a radius cap, and sampling is pluggable (`fps` | `prefix`).
- **Training:** AdamW, cosine schedule with warm-up, class weights of 1/√frequency (stems and woody branches are rare), and label smoothing. BatchNorm statistics are re-estimated before each evaluation ("precise BN").
- **Inference:** tiles with 50% overlap and softmax voting, then a stem-seeded instance baseline (DBSCAN on predicted stem points, then nearest-seed assignment). The output is a LAS file with `pred_class` and `pred_tree` fields, ready to open in CloudCompare.

### Species classification
Each tree reaches the model through two modalities that are always generated from the same augmented points:
- **Points:** 2048 points → PointNet++ encoder → global descriptor.
- **Views:** 6 side views plus 1 top-down depth rendering (64×64) → shared ResNet-style CNN → one token per view with a learned viewpoint embedding.
- **Fusion:**
  - `concat` is late concatenation.
  - `gated` uses a learned sigmoid gate per channel.
  - `attention` lets the point token query the view tokens through multi-head cross-attention, so the model can weight the most informative viewpoints for each tree.
- **Height:** absolute tree height (in metres) is fed back as a feature, because unit normalisation removes size information.
- **Modality dropout:** during training, one modality is randomly blanked, which makes the fused model degrade gracefully when a sensor is missing.

The motivation is that in the FOR-species20K benchmark paper, multi-view 2D models beat point-based models on average. This project tests whether combining both beats either alone. The image branch is modality-agnostic (`in_ch`), so real RGB or multispectral crops can replace the renders.

---

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[edge,dev]"
pytest -q

# End-to-end on synthetic data (CPU, a few minutes)
python scripts/make_synthetic_data.py --out data/synthetic
python -m forestfusion.train_seg --config configs/seg_synthetic.yaml
python -m forestfusion.inference --checkpoint runs/seg_synthetic/best.pt \
    --input data/synthetic/plots/val/plot_03.las --out runs/seg_synthetic/pred.las
python -m forestfusion.train_species --config configs/species_synthetic.yaml
python -m forestfusion.export --checkpoint runs/species_synthetic/best.pt --task species
python -m forestfusion.benchmark --task seg --onnx
```

### Real data

```bash
# FOR-instance (1.6 GB): https://zenodo.org/records/8287792
unzip FORinstance_dataset.zip -d data/FORinstance
python scripts/prepare_forinstance.py --root data/FORinstance --out data/splits
python -m forestfusion.train_seg --config configs/seg_forinstance.yaml

# FOR-species20K: https://zenodo.org/records/13255198 (dev.zip + metadata CSV)
python -m forestfusion.train_species --config configs/species_forspecies20k.yaml
bash scripts/run_ablation.sh configs/species_forspecies20k.yaml     # full modality / fusion ablation
```

Any config value can be overridden from the command line, e.g. `train.epochs=50 model.width=0.5`.

## Repository layout

```
src/forestfusion/
  data/          LAS/LAZ I/O, crop sampling, augmentation, multi-view rendering, synthetic generator
  models/        PointNet++ (seg + encoder), multi-view CNN, fusion classifier
  train_seg.py   segmentation training          train_species.py  species training
  inference.py   full-plot tiled inference + tree instances
  instance.py    instance baseline + matching metric
  export.py      ONNX export + onnxruntime parity check
  benchmark.py   latency study across width / sampling / runtime
configs/         YAML configs (synthetic smoke tests and real benchmarks)
scripts/         data prep, synthetic data, ablation runner, results table
tests/           unit tests (geometry ops, invariances, datasets, metrics)
```

## Data and citation

- Puliti, S. et al. (2023). *FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees.* arXiv:2309.01279.
- Puliti, S. et al. (2025). *Benchmarking tree species classification from proximally sensed laser scanning data: introducing the FOR-species20K dataset.* Methods in Ecology and Evolution 16, 801–818.
- Qi, C. R. et al. (2017). *PointNet++: Deep hierarchical feature learning on point sets in a metric space.* NeurIPS.

Check each dataset's licence before redistributing derived data. This repository contains code only.
