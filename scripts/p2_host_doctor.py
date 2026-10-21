#!/usr/bin/env python3
"""Offline ESPDet-Pico/PyTorch host prerequisites; NEVER starts training.

Examples:
  python3 -m scripts.p2_host_doctor plan
  python3 -m scripts.p2_host_doctor check
  python3 -m scripts.p2_host_doctor check --strict --require-cuda

No PyPI/GitHub requests, dataset/weight access or host identifiers are output.
Training remains P2 BLOCKED until the separate human P1 acceptance gate.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SUPPORTED = ((3, 10), (3, 11))
# Official upstream requirements, NOT a resolved/verified lock. The
# Ultralytics version must be selected by an actual custom-head graph smoke.
CANDIDATE_PINS = {
    "torch": "2.2.0",
    "torchvision": "0.17.0",
    "numpy": "1.24.4",
}
REQUIRED_PACKAGES = ("torch", "torchvision", "numpy", "ultralytics")
MIN_ULTRALYTICS = (8, 4, 0)
TARGET_MODEL = "ESPDet-Pico"
TARGET_CLASSES = 10
TARGET_INPUT = (224, 224)


def _version_tuple(version: str) -> tuple[int, ...] | None:
    if not isinstance(version, str):
        return None
    hit = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[+._-].*)?$", version.strip())
    return tuple(int(x) for x in hit.groups()) if hit else None


def _installed_versions() -> dict[str, str | None]:
    names = REQUIRED_PACKAGES + ("opencv-python",)
    values = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _git_head(path: Path) -> str | None:
    if not path.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip().lower()
    return sha if re.fullmatch("[0-9a-f]{40}", sha) else None


def _expected_source_sha(root: Path) -> str:
    values = json.loads((root / "upstreams.lock.json").read_text(encoding="utf-8"))
    deps = values["dependencies"]
    sha = next(d["commit"] for d in deps if d["id"] == "esp-detection")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Pinned ESP-Detection revision is not a full SHA")
    return sha


def _gpu_snapshot() -> dict:
    result = {
        "cuda_available": False,
        "mps_available": False,
        "cuda_runtime_build": None,
        "vram_gib": None,
        "error": None,
    }
    try:
        torch = importlib.import_module("torch")
        result["cuda_runtime_build"] = str(torch.version.cuda) if torch.version.cuda else None
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if result["cuda_available"]:
            result["vram_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError) as exc:
        # Report only the type, never a potentially identifying file path in an exception.
        result["error"] = type(exc).__name__
    return result


def _detect_accepts_reg_max() -> bool:
    """ESPDetect in the exact upstream calls Detect(nc=..., reg_max=1, ch=...)."""
    try:
        module = importlib.import_module("ultralytics.nn.modules.head")
        signature = inspect.signature(module.Detect.__init__)
        return "reg_max" in signature.parameters
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
        return False


def offline_plan() -> dict:
    """No Python package/GPU/Git inspection, filesystem scanning or download."""
    return {
        "status": "P2_00_PLAN_ONLY",
        "stage": "P2-00 host readiness, not model training",
        "recommended_host": "Linux x86_64 with NVIDIA CUDA; Windows WSL2 Ubuntu is preferred",
        "python_candidates": ["3.10", "3.11"],
        "official_upstream_candidates_not_validated": CANDIDATE_PINS,
        "ultralytics_source_checked_candidate": "8.4.0 (CPU synthetic graph tested; CUDA still pending)",
        "ultralytics_requires_actual_reg_max_constructor_smoke": True,
        "model": TARGET_MODEL,
        "classes": TARGET_CLASSES,
        "input_hw": list(TARGET_INPUT),
        "no_second_mobilenet_classifier": True,
        "p1_real_data_and_license_gate": "PENDING",
        "downloads_or_training_started": False,
    }


def evaluate(
    package_versions: dict[str, str | None], python_version: tuple[int, int],
    upstream_head: str | None, upstream_expected: str,
    gpu: dict, detect_supports_reg_max: bool,
    *, require_cuda: bool = False,
) -> dict:
    """Pure, mockable readiness logic; never considers P1 human acceptance PASS."""
    issues = []
    notes = []
    if python_version not in PYTHON_SUPPORTED:
        issues.append("Use Python 3.10 or 3.11 for the upstream candidate Torch/NumPy pins")
    for pkg in REQUIRED_PACKAGES:
        if package_versions.get(pkg) is None:
            issues.append(f"Python package not installed: {pkg}")
    for pkg, expected in CANDIDATE_PINS.items():
        installed = package_versions.get(pkg)
        if installed and installed.split("+", 1)[0] != expected:
            notes.append(
                f"{pkg} differs from upstream candidate {expected}; prove compatibility by smoke"
            )
    ul = package_versions.get("ultralytics")
    if ul is not None:
        parsed = _version_tuple(ul)
        if parsed is None or parsed < MIN_ULTRALYTICS:
            issues.append("Ultralytics <8.4.0 lacks the ESPDetect reg_max constructor interface")
        if not detect_supports_reg_max:
            issues.append("Installed Ultralytics Detect.__init__ does not accept reg_max")
    if upstream_head != upstream_expected:
        issues.append("Pinned ESP-Detection clone missing or HEAD differs from upstreams.lock.json")
    if gpu.get("error"):
        issues.append("Cannot inspect installed PyTorch runtime")
    if require_cuda and not gpu.get("cuda_available"):
        issues.append("NVIDIA CUDA device not available to installed PyTorch")
    if not gpu.get("cuda_available"):
        notes.append("No CUDA GPU detected; offline CPU graph smoke possible, bulk training not advised")
    elif gpu.get("vram_gib") is not None and gpu["vram_gib"] < 6:
        notes.append("Low CUDA VRAM: reduce batch/workers; do not assume default batch 128")
    if gpu.get("mps_available") and not gpu.get("cuda_available"):
        notes.append("Apple MPS is an experimental alternative, not the primary CUDA baseline")
    ready = not issues
    return {
        "status": "HOST_PREREQUISITES_READY_FOR_SYNTHETIC_SMOKE" if ready
        else "HOST_SETUP_INCOMPLETE",
        "host_prerequisites_ready": ready,
        "python_minor": ".".join(map(str, python_version)),
        "packages": {pkg: package_versions.get(pkg) for pkg in REQUIRED_PACKAGES},
        "upstream_clone_matches_pin": upstream_head == upstream_expected,
        "detect_supports_reg_max": bool(detect_supports_reg_max),
        "gpu": gpu,
        "blockers": issues,
        "advisories": notes,
        "model": TARGET_MODEL,
        "model_output_classes": TARGET_CLASSES,
        "model_input_hw": list(TARGET_INPUT),
        "real_training_authorized": False,
        "p1_license_and_10_class_visual_signoff": "PENDING",
        "no_real_data_read": True,
    }


def inspect_host(root: Path = ROOT, *, require_cuda: bool = False) -> dict:
    versions = _installed_versions()
    return evaluate(
        package_versions=versions,
        python_version=sys.version_info[:2],
        upstream_head=_git_head(root / "external" / "esp-detection"),
        upstream_expected=_expected_source_sha(root),
        gpu=_gpu_snapshot() if versions["torch"] else {
            "cuda_available": False,
            "mps_available": False,
            "cuda_runtime_build": None,
            "vram_gib": None,
            "error": None,
        },
        detect_supports_reg_max=(
            _detect_accepts_reg_max() if versions["ultralytics"] else False
        ),
        require_cuda=require_cuda,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check"))
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero if host prerequisites are missing")
    parser.add_argument("--require-cuda", action="store_true",
                        help="require PyTorch CUDA, rather than CPU-only smoke")
    args = parser.parse_args()
    try:
        result = offline_plan() if args.mode == "plan" else inspect_host(
            require_cuda=args.require_cuda
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"FAIL CLOSED: {type(exc).__name__}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if args.mode == "check" and args.strict and not result["host_prerequisites_ready"]:
        parser.exit(2, "BLOCKED: host prerequisites incomplete; see reported blockers\n")


if __name__ == "__main__":
    main()
