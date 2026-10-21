#!/usr/bin/env bash
# GTX 1060 6GB Linux research host bootstrap, no actual training/data access.
# Usage: bash scripts/p2_docker_host.sh <build|source|gpu|doctor|smoke|shell>
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${P2_DOCKER_IMAGE:-espdet-gesture:torch220-cu121}"

usage() {
  cat <<'HELP'
P2-00 isolated Linux + NVIDIA Docker workflow (GTX 1060 / Pascal sm_61)
  bash scripts/p2_docker_host.sh build    # isolated Torch 2.2/CUDA 12.1
  bash scripts/p2_docker_host.sh source   # exact ESP-Detection Git source only
  bash scripts/p2_docker_host.sh gpu      # actual CUDA matmul + cuDNN backward
  bash scripts/p2_docker_host.sh doctor   # PyTorch/Ultralytics/CUDA/source check
  bash scripts/p2_docker_host.sh smoke    # nc10 random model forward/backward
  bash scripts/p2_docker_host.sh shell    # isolated diagnostic shell
There is deliberately NO 'train' command and NO HaGRID download.
The old SCRFD image and work_dirs are not modified.
HELP
}
fail() { echo "BLOCKED: $*" >&2; exit 2; }

command -v docker >/dev/null 2>&1 || fail "Docker CLI missing"
command -v id >/dev/null 2>&1 || fail "Linux id missing"
case "${1:-}" in
  build|source|gpu|doctor|smoke|shell) action="$1";;
  *) usage; exit 2;;
esac
[[ "$#" == 1 ]] || fail "Expected exactly one action"

if [[ "$action" == "build" ]]; then
  cd "$ROOT_DIR"
  docker build --progress=plain -f docker/Dockerfile.p2-espdet-cu121 -t "$IMAGE" .
  exit 0
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  fail "Image $IMAGE missing: run bash scripts/p2_docker_host.sh build first"

# Reuse existing NVIDIA Container Toolkit. Host user owns persistent repo
# output; --shm-size=2g isolates IPC instead of the older --ipc=host mode.
common=(
  --rm --gpus all --shm-size=2g
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env YOLO_AUTOINSTALL=False
  --env PYTHONUNBUFFERED=1
  --volume "$ROOT_DIR:/workspace/repo"
  --workdir /workspace/repo
)
case "$action" in
  source)
    docker run "${common[@]}" "$IMAGE" \
      python scripts/fetch_upstreams.py --id esp-detection
    ;;
  gpu)
    docker run "${common[@]}" "$IMAGE" \
      python -m scripts.p2_cuda_probe run --expect-sm-61
    ;;
  doctor)
    docker run "${common[@]}" "$IMAGE" \
      python -m scripts.p2_host_doctor check --strict --require-cuda
    ;;
  smoke)
    docker run "${common[@]}" "$IMAGE" \
      python -m scripts.p2_graph_smoke run --device cuda
    ;;
  shell)
    docker run --interactive --tty "${common[@]}" "$IMAGE" bash
    ;;
esac
