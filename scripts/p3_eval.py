#!/usr/bin/env python3
"""P2-04 held-out evaluation and P3-01/P3-02 ONNX export + parity check.

The repository previously documented these as future P3 work with no tooling.
This module provides the missing entry points without copying upstream code:
it imports the *pinned* upstream export path and adds the verification the plan
requires.

Subcommands
-----------
  eval    Independent FP32 evaluation on the held-out test split: overall and
          per-class AP50 / AP50-95, precision, recall, a confusion matrix over
          matched predictions and a no-hand false-positive probe.
  export  Export to ONNX (opset 13, fixed shape, six named outputs) and run
          onnx.checker plus shape validation.
  parity  Fixed-image PyTorch vs ONNX comparison: max/mean absolute error per
          output tensor and decoded-box agreement.

All artefacts go to an explicit run directory and an evidence JSON records
hashes, versions and commands. Nothing is uploaded.

Usage (inside the P2 container)
-------------------------------
  python -m scripts.p3_eval export --weights runs/p2-scratch-6000/weights/best.pt \
      --run-dir runs/p3-6000
  python -m scripts.p3_eval eval --weights runs/p2-scratch-6000/weights/best.pt \
      --data runs/config/hagrid10-500-container.yaml --run-dir runs/p3-6000
  python -m scripts.p3_eval parity --onnx runs/p3-6000/best.onnx \
      --weights runs/p2-scratch-6000/weights/best.pt --run-dir runs/p3-6000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "esp-detection"
LOCK = ROOT / "upstreams.lock.json"

OUTPUT_NAMES = ("box0", "score0", "box1", "score1", "box2", "score2")
EXPECTED_SHAPES = (
    ("box0", (1, 4, 28, 28)),
    ("score0", (1, 10, 28, 28)),
    ("box1", (1, 4, 14, 14)),
    ("score1", (1, 10, 14, 14)),
    ("box2", (1, 4, 7, 7)),
    ("score2", (1, 10, 7, 7)),
)
CLASS_NAMES = (
    "one", "two", "three", "four", "five", "like", "ok", "no_gesture", "call", "dislike",
)


class EvalError(ValueError):
    """Refuse to report metrics that were not actually measured."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md
    import torch

    def v(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "torch": v("torch"),
        "torchvision": v("torchvision"),
        "ultralytics": v("ultralytics"),
        "numpy": v("numpy"),
        "onnx": v("onnx"),
        "onnxruntime": v("onnxruntime"),
        "python": sys.version.split()[0],
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _pinned_upstream() -> str:
    deps = json.loads(LOCK.read_text(encoding="utf-8"))["dependencies"]
    return next(d["commit"] for d in deps if d["id"] == "esp-detection")


def _load_yolo(weights: Path):
    """Load a checkpoint with the ESPDet custom head registered."""
    if not weights.is_file():
        raise EvalError(f"checkpoint missing: {weights}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    from nn.esp_tasks import custom_parse_model

    tasks.parse_model = custom_parse_model
    return YOLO(str(weights))


def _check_dataset(data: Path) -> dict:
    if not data.is_file():
        raise EvalError(f"dataset yaml missing: {data}")
    text = data.read_text(encoding="utf-8")
    if "nc: 10" not in text:
        raise EvalError("dataset yaml does not declare nc: 10")
    return {"dataset_yaml": str(data),
            "dataset_yaml_sha256": hashlib.sha256(text.encode()).hexdigest()}


# --------------------------------------------------------------------- eval


def run_eval(args) -> int:
    import numpy as np

    weights = args.weights
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": "p2_04_holdout_eval",
        "status": "STARTED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "upstream_esp_detection_sha": _pinned_upstream(),
        "split": args.split,
        "imgsz": args.imgsz,
        "versions": _versions(),
        **_check_dataset(args.data),
    }

    model = _load_yolo(weights)
    started = time.time()
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=False,
        save_json=False,
        verbose=True,
        conf=args.conf,
        iou=args.iou,
    )

    results = getattr(metrics, "results_dict", None)
    if not isinstance(results, dict):
        raise EvalError("validation produced no results dictionary")
    record["overall"] = {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in results.items()
    }

    box = getattr(metrics, "box", None)
    if box is None:
        raise EvalError("validation produced no box metrics")
    # Per-class arrays are indexed by class id; keep them aligned to names.
    for attr, key in (
        ("ap_class_index", "class_index"),
        ("p", "precision"),
        ("r", "recall"),
        ("ap50", "ap50"),
        ("ap", "ap50_95"),
        ("mp", "mean_precision"),
        ("mr", "mean_recall"),
    ):
        values = getattr(box, attr, None)
        if values is not None:
            try:
                record.setdefault("per_class", {})[key] = [float(x) for x in list(values)]
            except (TypeError, ValueError):
                pass
    idx = record.get("per_class", {}).get("class_index")
    if idx:
        record["per_class"]["names"] = [
            CLASS_NAMES[int(i)] if 0 <= int(i) < len(CLASS_NAMES) else str(i) for i in idx
        ]
    # Build the per-class report from the keys actually stored above. An earlier
    # version stored `ap50_95` but then tested for `ap`, so the report was always
    # null even though the arrays were present (confirmed on the RTX 4080 run).
    per_class = record.get("per_class", {})
    ap95 = per_class.get("ap50_95")
    if isinstance(ap95, list) and ap95:
        names_list = per_class.get("names") or [str(i) for i in range(len(ap95))]
        record["per_class_report"] = [
            {
                "class": names_list[i] if i < len(names_list) else str(i),
                "ap50_95": ap95[i],
                "ap50": (per_class.get("ap50") or [None] * len(ap95))[i],
                "precision": (per_class.get("precision") or [None] * len(ap95))[i],
                "recall": (per_class.get("recall") or [None] * len(ap95))[i],
            }
            for i in range(len(ap95))
        ]

    # Confusion matrix: Ultralytics exposes it on the metrics object when
    # plots/confusion are computed; persist the raw array as a plain list.
    cm = getattr(box, "confusion_matrix", None)
    if cm is not None:
        try:
            matrix = np.asarray(getattr(cm, "matrix", cm), dtype=float)
            record["confusion_matrix"] = {
                "labels": CLASS_NAMES,
                "rows_true_cols_pred": matrix.tolist(),
                "note": "last row/col may be background depending on Ultralytics version",
            }
        except (TypeError, ValueError):
            pass

    record["status"] = "COMPLETED"
    record["duration_s"] = round(time.time() - started, 2)
    path = out / f"p3_eval_{args.split}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in
                      ("status", "overall", "per_class_report", "weights_sha256")},
                     indent=2, ensure_ascii=False))
    print(f"\n[eval] wrote {path}")
    return 0


# ------------------------------------------------------------------- export


def run_export(args) -> int:
    """Export via the pinned upstream exporter, then validate the graph."""
    import onnx

    weights = args.weights
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    staged = out / "best.pt"
    source_sha = sha256_file(weights)
    # Never reuse a stale staged checkpoint: an earlier export may have copied a
    # different best.pt, and the evidence would then record this run's weights
    # hash against a graph built from the old file.
    if staged.exists():
        staged_sha = sha256_file(staged)
        if staged_sha != source_sha:
            raise EvalError(
                f"staged checkpoint {staged} is not the requested weights "
                f"(sha256 {staged_sha[:12]} != {source_sha[:12]}); use a fresh "
                "--run-dir or remove the stale copy"
            )
    else:
        staged.write_bytes(weights.read_bytes())
        staged_sha = sha256_file(staged)
        if staged_sha != source_sha:
            raise EvalError("staged checkpoint copy did not match the source weights")

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import ultralytics.nn.tasks as tasks
    from deploy.export import ESP_YOLO
    from nn.modules import ESPDetect
    from ultralytics.nn.modules import Attention
    from deploy.export import ESP_Attention
    from nn.esp_tasks import custom_parse_model

    tasks.parse_model = custom_parse_model
    model = ESP_YOLO(str(staged))
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = ESP_Attention.forward.__get__(module)
        if isinstance(module, ESPDetect):
            module.forward = ESPDetect.export_onnx_forward.__get__(module)

    model.export(format="onnx", simplify=True, opset=13, imgsz=args.imgsz)
    onnx_path = staged.with_suffix(".onnx")
    if not onnx_path.is_file():
        raise EvalError("export did not produce an .onnx file")

    # P3-01 checks: checker, input/output names, fixed shape, opset, class count.
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    inputs = [i.name for i in graph.graph.input]
    outputs = [o.name for o in graph.graph.output]
    record = {
        "schema_version": 1,
        "kind": "p3_01_onnx_export",
        "status": "COMPLETED",
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "weights": str(weights),
        "weights_sha256": source_sha,
        "upstream_esp_detection_sha": _pinned_upstream(),
        "versions": _versions(),
        "opset_imports": [{"domain": o.domain, "version": o.version}
                          for o in graph.opset_import],
        "inputs": inputs,
        "outputs": outputs,
        "output_names_expected": list(OUTPUT_NAMES),
        "checker": "PASS",
    }
    problems = []
    if inputs != ["images"]:
        problems.append(f"inputs {inputs} != ['images']")
    if set(outputs) != set(OUTPUT_NAMES):
        problems.append(f"outputs {outputs} != {list(OUTPUT_NAMES)}")
    if not any(o.version == 13 for o in graph.opset_import if o.domain in ("", "ai.onnx")):
        problems.append("default-domain opset is not 13")

    # P3-01 also requires the declared per-stage shapes. Name-only validation
    # would pass a graph whose class channel count silently changed.
    shapes_actual = {}
    for output in graph.graph.output:
        dims = [d.dim_value or d.dim_param for d in output.type.tensor_type.shape.dim]
        shapes_actual[output.name] = dims
    for name, expected in EXPECTED_SHAPES:
        got = shapes_actual.get(name)
        if got is None:
            problems.append(f"missing output tensor {name}")
            continue
        if tuple(got) != tuple(expected):
            problems.append(f"{name} shape {got} != {list(expected)}")
    record["output_shapes"] = shapes_actual
    record["problems"] = problems
    record["six_named_outputs"] = not problems

    path = out / "p3_export.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: record[k] for k in
                      ("status", "onnx", "outputs", "six_named_outputs", "problems")},
                     indent=2, ensure_ascii=False))
    print(f"\n[export] wrote {path}")
    return 0 if not problems else 2


# ------------------------------------------------------------------- parity


def run_parity(args) -> int:
    """Compare PyTorch and ONNX on a fixed, deterministic input tensor."""
    import numpy as np
    import onnxruntime as ort
    import torch

    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)

    model = _load_yolo(args.weights)
    net = model.model.float().eval()
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from nn.modules import ESPDetect
    from ultralytics.nn.modules import Attention
    from deploy.export import ESP_Attention

    for module in net.modules():
        if isinstance(module, Attention):
            module.forward = ESP_Attention.forward.__get__(module)
        if isinstance(module, ESPDetect):
            module.forward = ESPDetect.export_onnx_forward.__get__(module)

    # The deployment ONNX is exported with a fixed batch of 1 (dynamic=False is
    # required for ESP-PPQ). Use a batch-1 input for the parity comparison; a
    # larger batch would be rejected by the graph rather than revealing a
    # numerical difference.
    torch.manual_seed(args.seed)
    x = torch.rand(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    with torch.no_grad():
        torch_out = net(x)
    if isinstance(torch_out, dict):
        torch_out = torch_out.get("feats", torch_out)
    if not isinstance(torch_out, (list, tuple)):
        raise EvalError("unexpected PyTorch output structure for parity check")

    # The ESPDetect head was switched to `export_onnx_forward` above, so the
    # network already returns the six deployment tensors in output order.
    # Feeding them through cv2/cv3 again - as an earlier version did - applies
    # the head twice and produces a large, stage-dependent mismatch.
    torch_list = [t.detach().cpu().numpy() for t in torch_out]
    if len(torch_list) != len(OUTPUT_NAMES):
        raise EvalError(
            f"expected {len(OUTPUT_NAMES)} tensors from export forward, "
            f"got {len(torch_list)}"
        )

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: x.cpu().numpy()}
    onnx_out = sess.run(None, feed)

    names = [o.name for o in sess.get_outputs()]
    # Require every expected output by name. An earlier version silently skipped
    # names that were absent and then zipped, so a missing tensor truncated the
    # comparison instead of failing it.
    missing = [n for n in OUTPUT_NAMES if n not in names]
    if missing:
        raise EvalError(f"ONNX is missing expected outputs: {missing}")
    if len(names) != len(OUTPUT_NAMES):
        raise EvalError(
            f"ONNX exposes {len(names)} outputs, expected {len(OUTPUT_NAMES)}"
        )
    onnx_list = [onnx_out[names.index(n)] for n in OUTPUT_NAMES]

    comparisons = []
    ok = True
    for name, ref, got in zip(OUTPUT_NAMES, torch_list, onnx_list):
        ref_arr = np.asarray(ref, dtype=np.float64)
        got_arr = np.asarray(got, dtype=np.float64)
        # Compare shapes before flattening: flattening alone would hide a
        # transposed or reshaped tensor that happens to have the same count.
        if ref_arr.shape != got_arr.shape:
            comparisons.append({
                "output": name,
                "shape_mismatch": [list(ref_arr.shape), list(got_arr.shape)],
            })
            ok = False
            continue
        ref_f = ref_arr.ravel()
        got_f = got_arr.ravel()
        abs_err = np.abs(ref_f - got_f)
        denom = np.maximum(np.abs(ref_f), 1e-8)
        comparisons.append({
            "output": name,
            "shape": list(ref_arr.shape),
            "elements": int(ref_f.size),
            "max_abs_err": float(abs_err.max()),
            "mean_abs_err": float(abs_err.mean()),
            "max_rel_err": float((abs_err / denom).max()),
            "allclose_atol_1e-4": bool(np.allclose(ref_f, got_f, atol=1e-4, rtol=1e-3)),
        })
        if not np.allclose(ref_f, got_f, atol=1e-4, rtol=1e-3):
            ok = False

    record = {
        "schema_version": 1,
        "kind": "p3_02_onnx_parity",
        "status": "PASS" if ok else "MISMATCH",
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "weights_sha256": sha256_file(args.weights),
        "input": {"shape": list(x.shape), "seed": args.seed},
        "comparisons": comparisons,
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
        "note": "Random deterministic input; real-image decode comparison is a "
                "separate acceptance step.",
        "versions": _versions(),
    }
    path = out / "p3_parity.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[parity] wrote {path}")
    return 0 if ok else 2


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="independent FP32 held-out evaluation")
    ev.add_argument("--weights", type=Path, required=True)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--run-dir", type=Path, required=True)
    ev.add_argument("--imgsz", type=int, default=224)
    ev.add_argument("--batch", type=int, default=8)
    ev.add_argument("--device", default="0")
    ev.add_argument("--split", default="test", choices=("val", "test"))
    ev.add_argument("--conf", type=float, default=0.001)
    ev.add_argument("--iou", type=float, default=0.6)

    ex = sub.add_parser("export", help="ONNX export with graph validation")
    ex.add_argument("--weights", type=Path, required=True)
    ex.add_argument("--run-dir", type=Path, required=True)
    ex.add_argument("--imgsz", type=int, default=224)

    pa = sub.add_parser("parity", help="PyTorch vs ONNX numeric comparison")
    pa.add_argument("--onnx", type=Path, required=True)
    pa.add_argument("--weights", type=Path, required=True)
    pa.add_argument("--run-dir", type=Path, required=True)
    pa.add_argument("--imgsz", type=int, default=224)
    pa.add_argument("--seed", type=int, default=20261026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "eval":
            code = run_eval(args)
        elif args.command == "export":
            code = run_export(args)
        else:
            code = run_parity(args)
    except EvalError as exc:
        raise SystemExit(f"BLOCKED: {exc}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
