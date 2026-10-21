# P2-00 — Local GPU training environment (NOT actual P2 training)

**Purpose:** prepare a reproducible host for future **one-model ESPDet-Pico / 224×224 / 10-class** scratch training. The ESP32-S3 device only runs the quantized `.espdl` model later. **P1 real-image mapping, custom HaGRID license acceptance and dataset QA remain BLOCKED; P2 full training is NOT authorized.**

## Recommended hardware and operating system

For a Windows workstation with an NVIDIA GPU, prefer **WSL2 Ubuntu 22.04 + Python 3.10 virtualenv**. Native Linux with NVIDIA CUDA is equally suitable. If the local GPU is missing or its VRAM is too small, use a rented GPU Linux machine. Around **8 GiB of GPU VRAM or more** is a useful starting point, not a proven minimum; actual ESPDet-10 memory use must be measured with small batches. macOS Apple Silicon MPS is an experimental alternative, not the primary validated training stack.

You do **not** need to install ESP-IDF, connect the CoreS3, or flash the board for PC model training. A PyTorch CUDA wheel includes the matching CUDA runtime libraries, but your **NVIDIA host driver** still must support it.

## Check the plan without Torch or network

From the repository root:

    python3 -m scripts.p2_host_doctor plan
    python3 -m scripts.p2_graph_smoke plan

Both are offline and neither reads human images nor creates weights.

## Windows WSL2 first-time setup

In an elevated Windows PowerShell terminal (skip if WSL already exists):

    wsl --install -d Ubuntu-22.04

Then in an Ubuntu terminal:

    nvidia-smi
    sudo apt update
    sudo apt install -y git python3.10 python3.10-venv python3-pip
    cd ~
    git clone https://github.com/DenisZheng/esp32s3-gesture-recognition.git
    cd esp32s3-gesture-recognition
    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip wheel setuptools

If `nvidia-smi` fails inside WSL, configure the supported **Windows** NVIDIA WSL driver first; do not install a conflicting Linux display driver in WSL. If the machine has no NVIDIA GPU, a cloud GPU or **CPU-only synthetic** model compatibility check is still possible.

## Python candidate dependencies

The pinned ESP-Detection upstream [requirements.txt](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/requirements.txt) specifies `torch==2.2.0` and `torchvision==0.17.0`, but **`ultralytics>=8.3.112` is too loose**. Both Ultralytics `v8.3.112` and `v8.3.250` have `Detect(nc, ch)` without the `reg_max` constructor parameter used by our pinned ESPDet head, whereas [v8.4.0](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/nn/modules/head.py) exposes it. **8.4.0 is a source-audited compatibility candidate, not yet evidence of full runtime compatibility or correct training loss.**

The repository's [candidate constraints](../requirements/p2-candidate-constraints.txt) are NOT final lockfile evidence until actual graph/backprop passes. For Linux/WSL NVIDIA with a compatible driver, PyTorch's [official previous version installation guide](https://pytorch.org/get-started/previous-versions/) documents its CUDA 12.1 wheels:

    python -m pip install numpy==1.24.4
    python -m pip install torch==2.2.0 torchvision==0.17.0 \
      --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -c requirements/p2-candidate-constraints.txt \
      ultralytics==8.4.0 opencv-python==4.11.0.86
    python -m pip check

For a **CPU-only compatibility smoke**, use `https://download.pytorch.org/whl/cpu` instead of the CUDA 12.1 index. Never confuse a CPU-only PyTorch install with a CUDA training-ready host.

If your driver/newer graphics card needs a newer Torch/CUDA wheel, do not silently replace the exact candidate version. Record the incompatibility and review an alternative dependency set through a separate PR.

## Fetch only pinned ESPDet source

    python scripts/fetch_upstreams.py --id esp-detection

The command checks out Git commit `28de5902a357d99dd9a5db0f52905255641c5a78` under ignored `external/esp-detection/`. It downloads source code, **not HaGRID or pretrained model weights**. Note the [ESP-Detection/Ultralytics license obligations](THIRD_PARTY.md) before commercial deployment.

## Local prerequisites and synthetic ten-class model smoke

    python -m scripts.p2_host_doctor check --strict --require-cuda
    python -m scripts.p2_graph_smoke run --device cuda

On a CPU-only test machine, remove `--require-cuda` and use `--device cpu`. The first tool checks Python version, packages, the pinned source checkout, `reg_max` constructor signature and GPU availability; the second uses `YOLO(temp-generated nc=10 YAML)`, **random initial weights**, a synthetic random tensor of shape `[1,3,224,224]`, and a finite backward+SGD step on a **synthetic mathematical objective**. It expects raw training feature maps `[1,14,28,28]`, `[1,14,14,14]`, `[1,14,7,7]` (4 box channels + 10 classes). The code restores the Ultralytics parser afterwards, does not download HaGRID photos or pretrained weights and writes no model checkpoint. **This is a host API compatibility test, not proper YOLO detection training or proof of accuracy.**

If the actual compatibility smoke fails, stop here and diagnose before installing or training on a GPU. A green offline unit test cannot substitute for the real Torch/Ultralytics source build.

Once your host passes, keep version evidence **locally, ignored by Git**:

    mkdir -p runs/p2-host
    python -m pip freeze > runs/p2-host/pip-freeze.txt
    python -m pip check

## Formal P1 -> P2 training gate still closed

Only after [Issue #2 license](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/2), [Issue #3 ten-class visual mapping](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/3) and [Issue #4 real-data review and subject-disjoint data](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/4) are approved may an actual training command run. Freeze the exact compatible package versions, CUDA/OS information, class mapping SHA, dataset manifest, split policy, random seed, optimizer and hyperparameters. Start with a bounded small-batch pilot, not upstream `epochs=1200,batch=128,device=cpu` defaults. Only real FP32 validation can establish hand/gesture accuracy; P3 ONNX/ESP-PPQ and P4 CoreS3 hardware acceptance are separate later milestones.


## Verified public CPU graph compatibility — 2026-10-21

In [GitHub Actions](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733), the exact pinned ESP-Detection source and **Python 3.10.22, torch 2.2.0+cpu, torchvision 0.17.0+cpu, numpy 1.24.4, Ultralytics 8.4.0 and OpenCV 4.11.0.86** passed `python -m pip check` and `python -m scripts.p2_graph_smoke run --device cpu`.

**Actual evidence:** ten-class scratch ESPDet-Pico network instantiated; stage tensor shapes `(1,14,28,28)`, `(1,14,14,14)`, `(1,14,7,7)`; custom gesture-class head received **42 finite-gradient parameter tensors**; synthetic backward and SGD step succeeded. The changed Ultralytics 8.4 output dictionary (`boxes/scores/feats`) was explicitly handled, without patching third-party dependencies. No JPGs, annotations, pretrained weights or checkpoint were used.

This supports a *CPU-verified compatibility baseline* for setting up your host. **NVIDIA CUDA on the owner's PC, a real detection training loss, and real licensed dataset training are not verified.** The candidate constraint file stays named `candidate` until these later gates pass.
