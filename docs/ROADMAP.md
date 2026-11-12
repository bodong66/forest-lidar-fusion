# 3-week plan

The code base runs end-to-end on synthetic data. The three weeks go into real data, real numbers and a clear story. Each week ends with something you could show on its own.

**Compute:** FOR-instance and FOR-species20K need a GPU for full training runs. Free options are Kaggle notebooks (30 GPU-hours a week) and Google Colab. Develop locally on CPU with small `n_points` and `samples_per_epoch`, then run the configs unchanged on the GPU.

---

## Week 1 – Lidar segmentation (FOR-instance)

| Day | Task | Done when |
|---|---|---|
| 1 | Create the GitHub repo and push this code. Check that CI is green. Download FOR-instance. | Badge green, data unzipped |
| 1 | Run `prepare_forinstance.py`. Open 2–3 plots in CloudCompare and **verify the classification codes** in `configs/seg_forinstance.yaml`. | Label map confirmed |
| 2–3 | Train the PointNet++ baseline (FPS, width 1.0). Tune `radius` (3 / 4 / 6 m) and `n_points` on validation. | Baseline mIoU logged |
| 4 | Full-plot inference on validation plots, then the instance baseline and its F1. Make 2 figures: a prediction coloured by class and one coloured by tree. | Figures in `docs/figures/` |
| 5 | Train the `prefix`-sampling model and the width-0.5 model for the latency study. Fill in the segmentation table in the README. | Table filled |

**Reference point:** published FOR-instance results (e.g. Xiang et al., 2024, Remote Sensing of Environment) report about 73% mIoU over five classes with a much larger model. Being clearly below that with a 1M-parameter model is fine; explain the gap honestly.

## Week 2 – Multimodal species classification (FOR-species20K)

| Day | Task | Done when |
|---|---|---|
| 1 | Download the data and check the CSV column names. Run one epoch to build the cache. Look at rendered views for 3–4 species. | Cache built, view figure saved |
| 2–4 | `bash scripts/run_ablation.sh`: points, views, 3 fusion types, plus no-modality-dropout. | Ablation table filled |
| 4 | **Robustness test:** evaluate the fused model with the views blanked and with the points blanked. Does modality dropout help? | Short paragraph + table |
| 5 | Error analysis: confusion matrix and **per-platform accuracy (TLS / MLS / ULS)** via `group_col`, and conifer vs. broadleaf. Optional: submit to the FOR-species20K Codabench test set. | Figures + analysis written |

## Week 3 – Edge, polish and write-up

| Day | Task | Done when |
|---|---|---|
| 1 | Benchmark on your own laptop CPU and a GPU, and on a Jetson if you can borrow one (ONNX → TensorRT). Plot **accuracy vs. latency** (Pareto front). | Plot in README |
| 2 | Optional stretch: INT8 dynamic quantisation in onnxruntime, then measure the accuracy drop. | One extra row |
| 3–4 | Write `docs/report.md`, a 3–4 page paper-style write-up: motivation, method, results, limitations, next steps. | Report done |
| 5 | Clean up the README (results first, figures, one GIF of a segmented plot rotating), tag `v1.0`, and link the repo in the application. | Repo public |

---

## Talking points for the application / interview

- **Lidar segmentation:** semantic plus individual tree segmentation on the NIBIO-led FOR-instance benchmark.
- **Multimodal:** a controlled ablation of three fusion strategies for 3D + 2D data, with a robustness analysis for missing modalities.
- **Real-time / edge:** latency is dominated by point sampling, not network size. That is a concrete, measured insight that is relevant for on-machine perception in harvesters and drones.
- **Research practice:** reproducible configs, unit tests, CI, honest reporting against published baselines, and a written report.

## Ideas to mention as "next steps" (not to build now)

- Replace the stem-seeded instance baseline with a learned approach (offset regression + clustering).
- Use real RGB / multispectral drone imagery in the image branch instead of renders.
- Self-supervised pre-training on unlabelled Norwegian ALS data (Kartverket / Høydedata).
- Temporal change detection between repeated scans.
