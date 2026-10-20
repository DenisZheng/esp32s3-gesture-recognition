#!/usr/bin/env python3
"""Fail CI when user data, weights, firmware, or large objects are tracked."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".espdl", ".tflite",
    ".rknn", ".pb", ".h5", ".bin", ".elf", ".uf2", ".hex", ".zip",
    ".7z", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov"
}
BLOCK_PREFIXES = (
    "external/", "data/raw/", "data/processed/", "data/manifests/",
    "datasets/", "raw/", "annotations/", "calibration/", "runs/",
    "artifacts/", "outputs/", "checkpoints/", "weights/", "build/", "dist/",
    "firmware/build/", "firmware/managed_components/"
)
MAX_FILE_BYTES = 1024 * 1024


def check() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    tracked = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    assert tracked, "Git index is empty; stage source files before running policy guard"
    errors = []
    for name in tracked:
        fp = ROOT / name
        if name.startswith(BLOCK_PREFIXES):
            errors.append(f"Forbidden path: {name}")
        if fp.suffix.lower() in BLOCK_SUFFIXES:
            errors.append(f"Forbidden binary/media extension: {name}")
        if fp.exists() and fp.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"Exceeds 1 MiB guard: {name}")
        if fp.name in {"sdkconfig", "sdkconfig.old", ".env", "credentials.json"}:
            errors.append(f"Forbidden runtime/credential file: {name}")
    if errors:
        raise SystemExit("FAIL tracked-file policy:\n" + "\n".join(errors))
    print(f"PASS: {len(tracked)} tracked source/docs/config files; no forbidden media/weight paths")


if __name__ == "__main__":
    check()
