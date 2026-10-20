# Development environments and reproducibility

## Host training/quantization (P2/P3; no environment executed yet)

- Upstream ESP-Detection suggests Python 3.8, `torch==2.2.0`, `torchvision==0.17.0`, `onnx==1.17.0`, `onnxsim==0.4.36`, `numpy==1.24.4`, `opencv-python==4.11.0.86`, `ultralytics>=8.3.112`, plus a floating GitHub ESP-PPQ dependency.
- **Do not install the floating dependencies and claim reproducibility.** Clone the pinned ESP-Detection and ESP-PPQ revisions from `upstreams.lock.json`, find compatible Python/Ultralytics exact pins by actually running model instantiation + 10-class forward/backprop + ONNX export, record CUDA/OS/tool versions and freeze `pip` and hashes. This is a P2 prerequisite.
- Stage P0 `python3 scripts/check_contract.py`, `scripts/check_repo_policy.py`, `python3 -m unittest` requires only Python 3.11 standard library and git. CI is intentionally CPU/offline, no GPU, full dataset or toolchain.

## Target firmware (P4)

- ESP-IDF 5.4.4 (resolved git SHA in lock).
- ESP-DL v3.3.13 (resolved git SHA in lock); ensure the IDF Component Manager does not silently float `^3.1.3` to a newer version. Use exact registry version or a pinned local path.
- M5Stack CoreS3 ESP32-S3 with 8MB PSRAM; record exact BSP, camera sensor and clock/flash/PSRAM settings at P4.
- Board video and fixed-image runs occur on user-controlled hardware, not GitHub Actions.

## External dependencies

`scripts/fetch_upstreams.py` clones exact Git refs into ignored `external/` on a developer workstation **on explicit invocation only**. It does not run in CI and does not download HaGRID dataset archives. Review license obligations before distributing any source-derived code or binaries.
