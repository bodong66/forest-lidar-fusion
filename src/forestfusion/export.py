"""Export a trained model to ONNX and check numerical parity with onnxruntime.

    python -m forestfusion.export --checkpoint runs/seg/best.pt --task seg --sampling prefix
    python -m forestfusion.export --checkpoint runs/species/best.pt --task species

The ONNX file can then go to TensorRT (Jetson) or OpenVINO / onnxruntime on
CPU-only edge devices, e.g. an on-board computer in a harvester or drone.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from .train_seg import build_model as build_seg
from .train_species import build_model as build_species


def dummy_inputs(task: str, cfg: dict, n_points: int, batch: int = 1) -> tuple[torch.Tensor, ...]:
    pts = torch.rand(batch, n_points, 3) * 2 - 1
    if task == "seg":
        return (pts,)
    d = cfg["data"]
    v = d["n_views"] + int(d.get("top_view", True))
    return pts, torch.rand(batch, v, d["image_size"], d["image_size"]), torch.rand(batch, 1)


def load_model(
    checkpoint: str | Path, task: str, sampling: str | None = None
) -> tuple[torch.nn.Module, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if sampling:
        cfg["model"]["sampling"] = sampling
    if task == "seg":
        model = build_seg(cfg)
    else:
        model = build_species(cfg, len(state["classes"]))
    model.load_state_dict(state["model"])
    return model.eval(), cfg


def export_onnx(model, inputs, out: Path, task: str) -> None:
    names = ["points"] if task == "seg" else ["points", "views", "height"]
    kw = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False  # classic TorchScript exporter: stable for this architecture
    torch.onnx.export(
        model, inputs, str(out), input_names=names, output_names=["logits"],
        dynamic_axes={n: {0: "batch"} for n in [*names, "logits"]}, opset_version=17, **kw,
    )


def check_parity(model, inputs, onnx_path: Path) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {i.name: x.numpy() for i, x in zip(sess.get_inputs(), inputs)}
    ort_out = sess.run(None, feeds)[0]
    with torch.no_grad():
        ref = model(*inputs).numpy()
    return float(np.abs(ort_out - ref).max())


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", choices=["seg", "species"], required=True)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--sampling", choices=["fps", "prefix"], default=None,
                    help="override sampling; 'prefix' gives a much smaller, faster graph")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    model, cfg = load_model(args.checkpoint, args.task, args.sampling)
    n_points = args.n_points or cfg["data"]["n_points"]
    inputs = dummy_inputs(args.task, cfg, n_points)
    out = Path(args.out or Path(args.checkpoint).with_suffix(".onnx"))
    export_onnx(model, inputs, out, args.task)
    err = check_parity(model, inputs, out)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), max |torch - onnxruntime| = {err:.2e}")
    return out


if __name__ == "__main__":
    main()
