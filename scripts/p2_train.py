#!/usr/bin/env python3
"""P2 real-training entry for the pinned ESPDet-Pico nc=10 model.

The repository previously shipped only host probes plus an upstream
`train.py`; there was no auditable local entry that produced a run directory,
an evidence record and a checkpoint hash. This adapter wraps the *pinned*
upstream training path (Ultralytics trainer + `nn.esp_tasks.custom_parse_model`)
without vendoring upstream source.

Subcommands
-----------
  probe  (P2-01)  Real DataLoader -> real YOLO box/cls/dfl loss -> backward ->
                  optimizer step on the actual labelled dataset. Records loss,
                  peak VRAM, iteration timing and exit code. This is the gate
                  that the synthetic graph smoke cannot satisfy.
  train  (P2-02/03) Scratch training. Never silently loads pretrained weights.
  eval   (P2-04)  FP32 held-out evaluation of one explicit checkpoint.

No dataset download, no pretrained weights, no CI execution.

Usage (inside the P2 container):
  python -m scripts.p2_train probe --data <dataset.yaml> --run-dir runs/p2-probe-1
  python -m scripts.p2_train train --data <dataset.yaml> --run-dir runs/p2-scratch-1
  python -m scripts.p2_train eval  --weights <best.pt> --data <dataset.yaml>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "esp-detection"
LOCK = ROOT / "upstreams.lock.json"


class TrainError(ValueError):
    """Refuse to start training outside the approved P2 contract."""


# ---------------------------------------------------------------- provenance


def pinned_upstream_sha() -> str:
    deps = json.loads(LOCK.read_text(encoding="utf-8"))["dependencies"]
    return next(d["commit"] for d in deps if d["id"] == "esp-detection")


def git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if len(out) == 40 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_upstream() -> str:
    if not UPSTREAM.is_dir():
        raise TrainError(
            "Pinned upstream missing: run `bash scripts/p2_docker_host.sh source` first"
        )
    head = git_head(UPSTREAM)
    expected = pinned_upstream_sha()
    if head != expected:
        raise TrainError(f"ESP-Detection HEAD {head} != pinned {expected}")
    return head


def check_dataset(data: Path) -> dict:
    if not data.is_file() or data.is_symlink():
        raise TrainError("dataset.yaml missing or symlinked")
    text = data.read_text(encoding="utf-8")
    if "nc: 10" not in text:
        raise TrainError("dataset.yaml must declare nc: 10 for the gesture contract")
    base = data.parent
    # Resolve the declared data root so a relative path still validates.
    root_line = next(
        (ln for ln in text.splitlines() if ln.strip().startswith("path:")), None
    )
    if root_line is None:
        raise TrainError("dataset.yaml missing a `path:` root")
    declared = json.loads(root_line.split(":", 1)[1].strip().strip("'\""))
    root = Path(declared)
    if not root.is_absolute():
        root = (base / root).resolve()
    counts = {}
    for split in ("train", "val"):
        folder = root / "images" / split
        if not folder.is_dir():
            raise TrainError(f"dataset split images/{split} missing")
        n = sum(1 for _ in folder.rglob("*.jpg"))
        if n == 0:
            raise TrainError(f"dataset split images/{split} is empty")
        counts[split] = n
    return {
        "dataset_yaml_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "dataset_root": str(root),
        "images_per_split": counts,
    }


def environment_snapshot() -> dict:
    import importlib.metadata as md

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    import torch

    snap = {
        "python": platform.python_version(),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "ultralytics": version("ultralytics"),
        "numpy": version("numpy"),
        "opencv-python": version("opencv-python"),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if snap["cuda_available"]:
        snap["gpu"] = torch.cuda.get_device_name(0)
        snap["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        snap["gpu_total_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
        snap["torch_arch_list"] = torch.cuda.get_arch_list()
    snap["deterministic_algorithms_requested"] = True
    return snap


# ------------------------------------------------------------------- model


def load_model(yaml_path: Path | None = None, weights: Path | None = None):
    """Build a scratch nc=10 net or load an explicit checkpoint.

    Requires the ESPDet custom head to be registered before any YOLO build or
    checkpoint load; otherwise the pinned architecture cannot be resolved.
    """
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    from nn.esp_tasks import custom_parse_model

    tasks.parse_model = custom_parse_model
    if weights is not None:
        if not weights.is_file():
            raise TrainError("Checkpoint to evaluate does not exist")
        return YOLO(str(weights))
    if yaml_path is None or not yaml_path.is_file():
        raise TrainError("Scratch model YAML missing")
    return YOLO(str(yaml_path))


def assert_scratch(net) -> int:
    """Refuse an implicit pretrained start."""
    model = getattr(net, "model", None)
    yaml = getattr(model, "yaml", {}) or {}
    nc = int(yaml.get("nc", -1))
    if nc != 10:
        raise TrainError(f"Scratch network has nc={nc}, expected 10")
    return nc


# ------------------------------------------------------------------ runners


def _common_kwargs(args) -> dict:
    return dict(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        project=str(args.run_dir.parent),
        name=args.run_dir.name,
        exist_ok=False,
        plots=False,
        val=True,
        verbose=True,
        # Augmentation is fixed explicitly so the run is reproducible.
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=args.close_mosaic,
    )


def _finalise(run_dir: Path, record: dict) -> dict:
    """Add checkpoint identities, write evidence JSON next to the run."""
    weights = run_dir / "weights"
    artifacts = {}
    if weights.is_dir():
        for name in ("best.pt", "last.pt"):
            file = weights / name
            if file.is_file():
                artifacts[name] = {
                    "path": str(file),
                    "bytes": file.stat().st_size,
                    "sha256": sha256_file(file),
                    "mtime_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(file.stat().st_mtime)
                    ),
                }
    record["checkpoints"] = artifacts
    evidence = run_dir / "p2_evidence.json"
    evidence.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record["evidence_json"] = str(evidence)
    return record


def run_probe(args) -> dict:
    """One short, explicitly bounded real-loss run (P2-01 gate)."""
    upstream_sha = check_upstream()
    dataset = check_dataset(args.data)
    scratch_yaml = args.model_yaml
    model = load_model(yaml_path=scratch_yaml)
    nc = assert_scratch(model)

    if args.run_dir.exists():
        raise TrainError("Refusing to reuse an existing probe directory")
    args.run_dir.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "kind": "p2_01_real_batch_probe",
        "status": "STARTED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "upstream_esp_detection_sha": upstream_sha,
        "model": "ESPDet-Pico",
        "num_classes": nc,
        "input_hw": [args.imgsz, args.imgsz],
        "pretrained_weights_loaded": False,
        "environment": environment_snapshot(),
        **dataset,
    }
    started = time.time()
    try:
        result = model.train(epochs=1, **_common_kwargs(args))
    except Exception as exc:  # noqa: BLE001 - evidence must capture the failure
        record["status"] = "FAILED"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:2000]
        record["duration_s"] = round(time.time() - started, 2)
        _finalise(args.run_dir, record)
        raise
    record["status"] = "COMPLETED"
    record["duration_s"] = round(time.time() - started, 2)
    record["epochs_requested"] = 1
    record["save_dir"] = str(getattr(result, "save_dir", args.run_dir))
    try:
        import torch

        if torch.cuda.is_available():
            record["peak_vram_mib"] = round(
                torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2
            )
    except Exception:  # noqa: BLE001
        pass
    metrics = getattr(result, "results_dict", None)
    if isinstance(metrics, dict):
        record["validation_metrics"] = {
            str(k): (float(v) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
        }
    return _finalise(args.run_dir, record)


def run_train(args) -> dict:
    if args.pretrained is not None:
        raise TrainError(
            "Scratch-only entry: pass no --pretrained. The P2 contract forbids an "
            "implicit or explicit pretrained start; use a documented fine-tune entry."
        )
    upstream_sha = check_upstream()
    dataset = check_dataset(args.data)
    model = load_model(yaml_path=args.model_yaml)
    nc = assert_scratch(model)

    if args.run_dir.exists():
        raise TrainError("Refusing to reuse an existing run directory")
    args.run_dir.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "kind": "p2_scratch_training",
        "status": "STARTED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "upstream_esp_detection_sha": upstream_sha,
        "model": "ESPDet-Pico",
        "num_classes": nc,
        "input_hw": [args.imgsz, args.imgsz],
        "pretrained_weights_loaded": False,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "optimizer": "auto",
            "amp": args.amp,
            "seed": args.seed,
            "mosaic": args.mosaic,
            "close_mosaic": args.close_mosaic,
            "deterministic": True,
        },
        "environment": environment_snapshot(),
        **dataset,
    }
    started = time.time()
    try:
        result = model.train(epochs=args.epochs, **_common_kwargs(args))
    except Exception as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:2000]
        record["duration_s"] = round(time.time() - started, 2)
        _finalise(args.run_dir, record)
        raise
    record["status"] = "COMPLETED"
    record["duration_s"] = round(time.time() - started, 2)
    record["epochs_completed"] = args.epochs
    record["save_dir"] = str(getattr(result, "save_dir", args.run_dir))
    try:
        import torch

        if torch.cuda.is_available():
            record["peak_vram_mib"] = round(
                torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2
            )
    except Exception:  # noqa: BLE001
        pass
    metrics = getattr(result, "results_dict", None)
    if isinstance(metrics, dict):
        record["validation_metrics"] = {
            str(k): (float(v) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
        }
    return _finalise(args.run_dir, record)


def run_eval(args) -> dict:
    upstream_sha = check_upstream()
    dataset = check_dataset(args.data)
    model = load_model(weights=args.weights)
    record = {
        "schema_version": 1,
        "kind": "p2_fp32_holdout_eval",
        "status": "STARTED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "upstream_esp_detection_sha": upstream_sha,
        "split": args.split,
        "environment": environment_snapshot(),
        **dataset,
    }
    started = time.time()
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=False,
        verbose=True,
    )
    record["status"] = "COMPLETED"
    record["duration_s"] = round(time.time() - started, 2)
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        record["metrics"] = {
            str(k): (float(v) if isinstance(v, (int, float)) else v)
            for k, v in results.items()
        }
    box = getattr(metrics, "box", None)
    maps = getattr(box, "maps", None)
    if maps is not None:
        try:
            record["per_class_ap50_95"] = [float(x) for x in list(maps)]
        except (TypeError, ValueError):
            pass
    out = args.run_dir or (args.weights.parent.parent / "eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "p2_eval.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record["evidence_json"] = str(out / "p2_eval.json")
    return record


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--data", type=Path, required=True, help="P1-approved dataset.yaml")
        p.add_argument("--imgsz", type=int, default=224)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--device", default="0")
        p.add_argument("--seed", type=int, default=20261022)
        p.add_argument("--amp", action="store_true", help="Off by default: FP32 baseline")
        p.add_argument("--mosaic", type=float, default=1.0)
        p.add_argument("--close-mosaic", type=int, default=0)

    p_probe = sub.add_parser("probe", help="bounded real-loss probe (P2-01)")
    shared(p_probe)
    p_probe.add_argument("--model-yaml", type=Path, required=True)
    p_probe.add_argument("--run-dir", type=Path, required=True)

    p_train = sub.add_parser("train", help="scratch training run (P2-02/03)")
    shared(p_train)
    p_train.add_argument("--model-yaml", type=Path, required=True)
    p_train.add_argument("--run-dir", type=Path, required=True)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--pretrained", type=Path, default=None,
                         help="Forbidden: scratch-only entry point")

    p_eval = sub.add_parser("eval", help="FP32 held-out evaluation (P2-04)")
    shared(p_eval)
    p_eval.add_argument("--weights", type=Path, required=True)
    p_eval.add_argument("--split", default="test", choices=("val", "test"))
    p_eval.add_argument("--run-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        raise SystemExit("BLOCKED: real training is forbidden in CI")
    try:
        if args.command == "probe":
            result = run_probe(args)
        elif args.command == "train":
            result = run_train(args)
        else:
            result = run_eval(args)
    except TrainError as exc:
        raise SystemExit(f"BLOCKED: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
