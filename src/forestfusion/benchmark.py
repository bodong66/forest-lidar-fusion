"""Latency / throughput benchmark across model width, sampling strategy and runtime.

    python -m forestfusion.benchmark --task seg --widths 1.0 0.5 0.25 --sampling fps prefix --onnx
    python -m forestfusion.benchmark --task species --device cuda

Latency does not depend on trained weights, so this runs with random weights by
default. Combine the output table with validation accuracy of the matching
trained models to plot the accuracy/latency Pareto front for the README.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .export import dummy_inputs, export_onnx
from .models import PointNet2Seg, SpeciesClassifier
from .utils import count_parameters, get_device

SPECIES_CFG = {"data": {"n_views": 6, "top_view": True, "image_size": 64}}


def make_model(task: str, width: float, sampling: str, n_classes: int, modality: str) -> torch.nn.Module:
    if task == "seg":
        return PointNet2Seg(n_classes, width=width, sampling=sampling)
    return SpeciesClassifier(n_classes, modality=modality, width=width, sampling=sampling, n_views=7)


def time_fn(fn, warmup: int, iters: int, sync=None) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg", "species"], default="seg")
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--sampling", nargs="+", default=["fps", "prefix"])
    ap.add_argument("--modality", default="fusion")
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (simulate edge CPUs)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--onnx", action="store_true", help="also benchmark onnxruntime (CPU)")
    ap.add_argument("--onnx-fps", action="store_true",
                    help="also export FPS models to ONNX (slow: the sampling loop is unrolled)")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    n_points = args.n_points or (4096 if args.task == "seg" else 2048)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    rows = []
    for width in args.widths:
        for sampling in args.sampling:
            model = make_model(args.task, width, sampling, args.n_classes, args.modality).eval()
            inputs = dummy_inputs(args.task, SPECIES_CFG, n_points, args.batch)
            dev_inputs = tuple(x.to(device) for x in inputs)
            model_dev = model.to(device)
            with torch.inference_mode():
                t = time_fn(lambda m=model_dev, x=dev_inputs: m(*x), args.warmup, args.iters, sync)
            row = {"task": args.task, "width": width, "sampling": sampling, "runtime": f"torch-{device.type}",
                   "params_M": count_parameters(model) / 1e6, "n_points": n_points, "batch": args.batch,
                   "ms_mean": float(t.mean()), "ms_p90": float(np.percentile(t, 90)),
                   "samples_per_s": float(1000 * args.batch / t.mean())}
            rows.append(row)
            print(row)
            if args.onnx and (sampling != "fps" or args.onnx_fps):
                import onnxruntime as ort

                model_cpu = model.cpu()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "m.onnx"
                    export_onnx(model_cpu, inputs, path, args.task)
                    opts = ort.SessionOptions()
                    if args.threads:
                        opts.intra_op_num_threads = args.threads
                    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                    feeds = {i.name: x.numpy() for i, x in zip(sess.get_inputs(), inputs)}
                    t = time_fn(lambda s=sess, f=feeds: s.run(None, f), args.warmup, args.iters)
                rows.append({**row, "runtime": "onnxruntime-cpu", "ms_mean": float(t.mean()),
                             "ms_p90": float(np.percentile(t, 90)),
                             "samples_per_s": float(1000 * args.batch / t.mean())})
                print(rows[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("\n| width | sampling | runtime | params (M) | latency ms (mean / p90) | samples/s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['width']} | {r['sampling']} | {r['runtime']} | {r['params_M']:.2f} | "
              f"{r['ms_mean']:.1f} / {r['ms_p90']:.1f} | {r['samples_per_s']:.1f} |")
    return rows


if __name__ == "__main__":
    main()
