# P2-00 — Reusing a GTX 1060 6GB Linux SCRFD training host via Docker

**State:** Docker isolation files + synthetic/offline tests supplied; this exact Docker image / owner GTX 1060 CUDA execution is **NOT YET VERIFIED**. Running these commands is P2 **host preparation**, NOT genuine training. The authorized HaGRID P1 license/data/class-signoff gates are still open.

## Machine-specific rationale

The owner reports a Linux machine with **32 GiB system RAM** and **GTX 1060 6 GiB (Pascal sm_61)**. Its earlier SCRFD-500M-KPS training documentation confirms prior use of **Docker + NVIDIA Container Toolkit**; its SCRFD-specific image used **Python 3.8, PyTorch 1.8.0 + CUDA 11.1, MMCV 1.3.3**. The recorded prior NVIDIA driver was **580.173.02**, but the **current** driver must be checked on the host. SCRFD was trained with official 640×640, batch 16, workers 4, 640 epochs; those results do **not** establish equivalent ESPDet-Pico 224×224 model training speed or batch memory needs.

**Do not rebuild, upgrade, re-tag or delete the old SCRFD image.** Do not share SCRFD experiment work_dirs with the new project. The unrelated SCRFD OpenMMLab/MMCV torch1.8 environment cannot be reused to satisfy the new pinned ESPDet-Pico + Ultralytics 8.4 requirements.

The new research container is based on the official linux/amd64 PyTorch image `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`, and the Dockerfile pins its published platform manifest digest. Its approximately **3.41-GB compressed base download** needs additional unpacked disk capacity. Ensure adequate disk space before pulling. NVIDIA's CUDA 12.1 minimum/Linux driver compatibility is compatible with the **previously documented 580 driver**, assuming the actual host driver is still that revision. Pascal is supported by CUDA 12.x, but CUDA Toolkit 13 dropped Pascal code generation; **do not silently switch to CUDA 13**.

The ESPDet-Pico exact pinned source, Python 3.10, Torch 2.2.0+CPU, Torchvision 0.17.0+CPU, NumPy 1.24.4, OpenCV 4.11.0.86 and Ultralytics 8.4.0 already passed a **real ten-class 224×224 scratch forward/backward CPU check**, with 42 finite custom class-head gradients ([GitHub Actions evidence](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733)). The CUDA part **still needs your actual GPU**. We will not claim that the full Docker image builds or GPU works until you run and report it.

## 1. Host preflight (no new toolkit installation unless necessary)

Run these **on Linux**:

    nvidia-smi
    docker info
    df -h "$HOME"

Docker and NVIDIA Container Toolkit were already required for your SCRFD project; you need not reinstall them unless the old setup no longer works. You should observe GTX 1060, about 6GB VRAM and the current driver version. If host GPU visibility fails, fix the driver/runtime *before* investigating PyTorch. These are read-only commands except Docker daemon connection.

You can also test Docker GPU injection after the image is built (below). Host `nvidia-smi` passing by itself does not establish GPU access *inside* the container.

## 2. Clone or refresh the new project independently of SCRFD

Choose an independent directory in your Linux home:

    cd "$HOME"
    git clone https://github.com/DenisZheng/esp32s3-gesture-recognition.git
    cd esp32s3-gesture-recognition
    git status --short
    git rev-parse HEAD

If the repo already exists on the Linux host, `git pull --ff-only` instead of cloning again; check for uncommitted work first. Avoid placing the new project inside your existing SCRFD code or dataset directories.

**There is no WSL2 installation step for this Linux machine.**

## 3. Build a second, isolated CUDA training-container image

    bash scripts/p2_docker_host.sh build

Expected Docker image tag: `espdet-gesture:torch220-cu121`. This Dockerfile only copies the **public pinned Python constraints file** into the image. The top-level `.dockerignore` excludes all other repository contents from the build context (including `data/`, `external/`, `runs/` and model weights).

The image does NOT contain: SCRFD MMCV/old code, HaGRID private JSON/JPEGs, ESPDet model checkpoints, or copied private host paths. It checks pinned Python/PyTorch versions, CUDA 12.1 runtime and `pip check` during the build, failing rather than silently substituting a different Torch wheel. Using `--no-cache-dir` for installed pip packages is deliberate to avoid filling image layers with wheel caches.

**A Dockerfile or successful public synthetic CI is not proof the image has been built on your machine.** Capture build failures locally without uploading personal data/credentials.

## 4. Test real GTX 1060 CUDA kernels (no ESPDet source or HaGRID needed yet)

    bash scripts/p2_docker_host.sh gpu

This starts a temporary container with `--gpus all`. It requires the active GPU to be Pascal **sm_61** and at least ~5GiB VRAM to match the **6GB GTX1060**, then executes actual CUDA floating-point matrix multiplication and cuDNN convolution **with backward gradients and synchronization**. It outputs a bounded device memory/architecture record, **not a training result**.

Any `CUDA error: no kernel image is available` or `torch.cuda.is_available() is false` must block the next steps. Check `torch.cuda.get_arch_list()`, current driver and container runtime instead of assuming CUDA 12.1 is enough. If the machine has more than one NVIDIA card and CUDA selects the wrong one, do **not** silently relax the sm_61 assertion; configure/select the intended GPU explicitly.

## 5. Fetch exactly the pinned ESP-Detection source (no pretrained model)

    bash scripts/p2_docker_host.sh source

This calls the repository's `scripts/fetch_upstreams.py --id esp-detection` **inside** the new container. It checks out the exact Git SHA `28de5902a357d99dd9a5db0f52905255641c5a78` into host-owned, gitignored `external/esp-detection` under the bind-mounted new project.

Neither the SCRFD directory nor HaGRID is touched; the script fetches **source code only**, not datasets. Note that the upstream code carries AGPL-related licensing obligations for any commercial use.

## 6. Validate the installed software stack and actual nc=10 ESPDet CUDA model

    bash scripts/p2_docker_host.sh doctor
    bash scripts/p2_docker_host.sh smoke

The doctor checks the exact pinned source HEAD, Torch/Torchvision/NumPy/Ultralytics, actual CUDA device/VRAM and the required `Detect(...,reg_max=1)` API. The model smoke constructs a brand-new **random-weight** ESPDet-Pico with **ten** output classes at **224×224** and runs forward/backward + one synthetic mathematical SGD step **on GTX 1060 CUDA**.

The three training-head channel groups must be `[1,14,28,28]`, `[1,14,14,14]` and `[1,14,7,7]`, including 4 bounding-box and 10 gesture channels. No HaGRID pictures, annotations, label acceptance, pretrained weights or real optimizer checkpoint are required.

A successful `smoke` means only **ten-class CUDA graph compatibility**; it does not validate real YOLO box/class loss, training efficiency, mAP, legal data usage, ONNX, ESP-PPQ or CoreS3 inference. Preserve stdout and report any error messages without uploading paths or raw media.

The helper uses `--rm`, a fixed **2GiB container shared-memory size** rather than globally sharing host IPC, and your host UID/GID, so generated ignored source directories or later experiment outputs stay owned by you. The helper has **no train action** by design, and starts no long-running job.

## 7. Safely choose batch/workers *only after* authorized P1 data acceptance

A reasonable **unproven** performance-probe order for this 6GB Pascal card:

| Host test | Initial idea | Why |
|---|---|---|
| 224×224 single ESPDet-Pico | fixed | Compare output classes without changing sensor/model scope |
| Start batch | 4 | Conservative CUDA memory baseline, not a measured limit |
| Then batch | 8 if batch4 stable | Establish peak VRAM and actual effective batch |
| DataLoader workers | 2, then 4 | Watch RAM, `/dev/shm` and data_time |
| Precision | FP32 initially | Pascal lacks Tensor Cores; mixed precision needs measurement |
| Epochs | short explicitly named pilot, then reviewed full schedule | Never copy SCRFD 640 or ESP-Detection default 1200 blindly |

32GB system RAM helps loading/analyzing metadata but **cannot substitute for 6GB GPU VRAM**. SCRFD's measured batch16 / ~2GB probe came from a different model and image distribution; it is **not** an ESPDet memory benchmark. Likewise, SCRFD's ~54-hour 640-epoch run cannot predict ESPDet total time.

After P1 approval, mount the licensed dataset **read-only** as a separate host directory and mount outputs persistently; do not commit JPEGs, JSON subjects, checkpoints, or rendered private review cards to this **Public** repository. Use a fresh `runs/` experiment ID, preserve image hash, source/config hash, log, exit code, peak VRAM, checkpoint integrity and held-out validation scores. Do not confuse a synthetic gradient test with an actual detection-loss training step.

## 8. Existing P1 and P2 gates

- [HaGRID legal/research license human approval — #2](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/2)
- [Ten source→official gesture visual adjudication — #3](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/3)
- [Actual subject-disjoint train/val/test data and QA — #4](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/4)
- [CUDA host evidence gate — #12](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/12)

**Only a GPU probe and synthetic graph smoke may run now.** Real gesture training starts after all required human authorization and dataset controls have evidence-backed PASS.

## Upstream evidence

- [PyTorch official v2.2.0 CUDA12.1 availability](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 2.2.0 / CUDA12.1 / cuDNN8 Docker runtime image](https://hub.docker.com/layers/pytorch/pytorch/2.2.0-cuda12.1-cudnn8-runtime/images/sha256-e5c32997194edfc6fb0235223d1cd0433056045a468e7dfeaac390a3fcf8ae6d)
- [NVIDIA CUDA12.1 driver compatibility](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)
- [NVIDIA CUDA13 Pascal support removal](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
