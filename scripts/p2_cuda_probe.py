#!/usr/bin/env python3
"""Actual CUDA math/conv/backprop host probe; NOT HaGRID/model training.

Designed for NVIDIA GTX 1060 (Pascal sm_61) on an isolated CUDA 12.1
PyTorch 2.2.0 container. No data, checkpoints, model images or downloads.

Run:
  python -m scripts.p2_cuda_probe plan
  python -m scripts.p2_cuda_probe run --expect-sm-61
"""
from __future__ import annotations

import argparse
import importlib
import json


class CUDAProbeError(ValueError):
    """GPU driver/kernel/architecture did not pass strict host requirements."""


def offline_plan() -> dict:
    return {
        "status": "P2_00_GPU_PROBE_PLAN_ONLY",
        "gpu_expected": "GTX 1060 6GB Pascal sm_61",
        "torch_candidate": "2.2.0+CUDA12.1",
        "tests": [
            "PyTorch CUDA version + detected GPU compute capability",
            "CUDA floating point matrix multiplication (finite output)",
            "GPU cuDNN convolution forward + gradient and synchronization",
        ],
        "no_real_training": True,
        "no_media_or_checkpoint": True,
    }


def validate_metadata(meta: dict, *, expect_sm_61: bool = False) -> dict:
    """Pure host decision; no PyTorch import / CUDA operations here."""
    issues = []
    if not meta.get("cuda_available"):
        issues.append("PyTorch CUDA unavailable: verify NVIDIA Container Toolkit and drivers")
    if meta.get("torch_cuda_build") != "12.1":
        issues.append("Expected the pinned Torch CUDA 12.1 runtime")
    cc = meta.get("device_capability")
    if not isinstance(cc, (list, tuple)) or len(cc) != 2 or not all(
        isinstance(v, int) for v in cc
    ):
        issues.append("GPU compute capability not reported")
    elif expect_sm_61 and tuple(cc) != (6, 1):
        issues.append("Detected GPU is not the expected Pascal sm_61 GTX 1060")
    vram = meta.get("vram_gib")
    if not isinstance(vram, (int, float)) or vram <= 0:
        issues.append("GPU total memory not reported")
    elif expect_sm_61 and vram < 5:
        issues.append("GTX 1060 6GB expectation failed: less than 5GiB reported")
    return {
        "host_cuda_metadata_ready": not issues,
        "status": "GPU_METADATA_READY_TO_TEST_MATH" if not issues else "GPU_METADATA_BLOCKED",
        "blockers": issues,
        "torch_cuda_build": meta.get("torch_cuda_build"),
        "gpu": meta.get("device_name"),
        "compute_capability": list(cc) if isinstance(cc, (list, tuple)) else None,
        "vram_gib": vram,
        "real_model_training_pass": False,
    }


def run(*, expect_sm_61: bool = False) -> dict:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as exc:
        raise CUDAProbeError("PyTorch CUDA environment cannot be imported") from exc
    try:
        available = bool(torch.cuda.is_available())
        if not available:
            raise CUDAProbeError("torch.cuda.is_available() is false")
        name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        meta = {
            "cuda_available": available,
            "torch_cuda_build": torch.version.cuda,
            "device_name": name,
            "device_capability": capability,
            "vram_gib": round(props.total_memory / (1024 ** 3), 2),
        }
        decision = validate_metadata(meta, expect_sm_61=expect_sm_61)
        if not decision["host_cuda_metadata_ready"]:
            raise CUDAProbeError("; ".join(decision["blockers"]))
        torch.cuda.set_device(0)
        torch.manual_seed(20261021)
        torch.cuda.reset_peak_memory_stats(0)

        # Unlike import-only checks, these actually launch CUDA device kernels.
        x = torch.randn(256, 256, device="cuda")
        matrix = x @ x
        if not bool(torch.isfinite(matrix).all()):
            raise CUDAProbeError("CUDA matmul produced non-finite values")

        # Also exercise a real cuDNN convolution + autograd on GPU.
        conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
        pixels = torch.randn(1, 3, 64, 64, device="cuda")
        loss = conv(pixels).square().mean()
        loss.backward()
        if not (bool(torch.isfinite(loss)) and
                conv.weight.grad is not None and
                bool(torch.isfinite(conv.weight.grad).all())):
            raise CUDAProbeError("CUDA convolution/backprop result is invalid")
        torch.cuda.synchronize(0)
        return {
            "status": "GPU_CUDA_MATMUL_CONV_BACKPROP_PASS",
            "gpu": name,
            "compute_capability": list(capability),
            "vram_gib": meta["vram_gib"],
            "torch_cuda_build": str(torch.version.cuda),
            "pytorch_gpu_arch_list": torch.cuda.get_arch_list(),
            "cuda_matmul_pass": True,
            "cuda_convolution_grad_pass": True,
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2
            ),
            "no_real_training": True,
            "no_media_or_checkpoint": True,
            "actual_espdet_cuda_graph_test_still_required": True,
        }
    except CUDAProbeError:
        raise
    except (RuntimeError, ValueError, AttributeError, OSError) as exc:
        # Suppress private host paths and full driver diagnostics in error.
        raise CUDAProbeError("Actual CUDA kernels failed: " + type(exc).__name__) from exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"))
    p.add_argument("--expect-sm-61", action="store_true",
                   help="fail if active GPU is not a 6GiB-class Pascal sm_61 card")
    args = p.parse_args()
    try:
        result = offline_plan() if args.mode == "plan" else run(
            expect_sm_61=args.expect_sm_61
        )
    except CUDAProbeError as exc:
        p.exit(2, f"BLOCKED: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
