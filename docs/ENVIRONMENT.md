# Development environments and reproducibility

## Host training/quantization (P2/P3; no environment executed yet)

- Upstream ESP-Detection suggests Python 3.8, `torch==2.2.0`, `torchvision==0.17.0`, `onnx==1.17.0`, `onnxsim==0.4.36`, `numpy==1.24.4`, `opencv-python==4.11.0.86`, `ultralytics>=8.3.112`, plus a floating GitHub ESP-PPQ dependency.
- **Do not install the floating dependencies and claim reproducibility.** Clone the pinned ESP-Detection and ESP-PPQ revisions from `upstreams.lock.json`, find compatible Python/Ultralytics exact pins by actually running model instantiation + 10-class forward/backprop + ONNX export, record CUDA/OS/tool versions and freeze `pip` and hashes. This is a P2 prerequisite.
- Stage P0 `python3 scripts/check_contract.py`, `scripts/check_repo_policy.py`, `python3 -m unittest` requires only Python 3.11 standard library and git. CI is intentionally CPU/offline, no GPU, full dataset or toolchain.

## Target firmware (P4)

- ESP-IDF 5.4.4 (resolved git SHA in lock).
- ESP-DL v3.3.14 (resolved git SHA in lock); ensure the IDF Component Manager does not silently float `^3.1.3` to a newer version. Use exact registry version `3.3.14` or a pinned local path. The `v3.3.14` release tag resolves to commit `5acd1c9992af4afb0fa80c221a676982b11381b8`. A newer *future* release must enter via source-and-device compatibility-review PR, not floating ranges.
- M5Stack CoreS3 ESP32-S3 with 8MB PSRAM; record exact BSP, camera sensor and clock/flash/PSRAM settings at P4.
- Board video and fixed-image runs occur on user-controlled hardware, not GitHub Actions.

## External dependencies

`scripts/fetch_upstreams.py` clones exact Git refs into ignored `external/` on a developer workstation **on explicit invocation only**. It does not run in CI and does not download HaGRID dataset archives. Review license obligations before distributing any source-derived code or binaries.

## Input-shape reproducibility

Baseline model input **224x224x3** (complete scene). Official 128x128 MobileNetV2 operates on cropped hand ROI in the **two-model** reference and is not used here. Freeze the model input and preprocessor shape at each experiment; see [INPUT_RESOLUTION.md](INPUT_RESOLUTION.md).


## P2-00 host bootstrapping (safe before P1 license signoff)

Use the [P2_HOST_SETUP](P2_HOST_SETUP.md) instructions to install WSL2 Ubuntu/Linux NVIDIA GPU Python 3.10 and the exact **candidate** Torch/Torchvision/NumPy/Ultralytics packages under a venv. The pinned ESP-Detection custom head calls `Detect(nc=...,reg_max=1,ch=...)`, which **Ultralytics v8.3.112/v8.3.250 lacks**; v8.4.0 exposes this parameter. `scripts/p2_host_doctor.py` checks package versions, exact git HEAD, CUDA and API shape. `scripts/p2_graph_smoke.py` builds only a **random-weight** nc10 net and tests synthetic noise gradient; P2 real data/training and P3 conversion remain blocked by P1 human acceptance. The candidate file `requirements/p2-candidate-constraints.txt` does NOT claim a proven lock until a real compatibility smoke passes.


**2026-10-21 CPU compatibility verified:** [source-only Actions run](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733) confirms Python3.10/Torch2.2.0+cpu/Torchvision0.17.0+cpu/Ultralytics8.4.0/NumPy1.24.4 can build actual pinned ESPDet-Pico `nc=10` and backpropagate a synthetic objective with 42 valid custom classification head gradients. This is better evidence than source inspection but **not** CUDA or actual YOLO detection-loss train acceptance. Version snapshot is recorded in `upstreams.lock.json`; GPU still requires local validation.
