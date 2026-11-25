#!/usr/bin/env python3
"""向独立环境应用固定版本的 Ultralytics 补丁，可重复执行。"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path

PATCH = Path(__file__).resolve().parents[1] / "patches/ultralytics-8.4.0.patch"


def apply_patch() -> None:
    if importlib.metadata.version("ultralytics") != "8.4.0":
        raise RuntimeError("此补丁只适用于 Ultralytics 8.4.0")
    root = Path(importlib.util.find_spec("ultralytics").origin).parent.parent
    command = ["git", "apply", str(PATCH)]
    ready = subprocess.run(command + ["--check"], cwd=root, capture_output=True, text=True)
    if ready.returncode == 0:
        subprocess.run(command, cwd=root, check=True)
        print("ULTRALYTICS_PATCH_APPLIED version=8.4.0")
        return
    applied = subprocess.run(command + ["--reverse", "--check"], cwd=root, capture_output=True, text=True)
    if applied.returncode == 0:
        print("ULTRALYTICS_PATCH_ALREADY_APPLIED version=8.4.0")
        return
    raise RuntimeError("Ultralytics 源文件与补丁不匹配：" + ready.stderr.strip())


if __name__ == "__main__":
    apply_patch()
