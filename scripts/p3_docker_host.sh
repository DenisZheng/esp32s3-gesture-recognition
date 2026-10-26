#!/usr/bin/env bash
# P3 tooling container: the P2 training image plus ONNX export/validation deps.
#
# Why a derived image instead of rebuilding the P2 image: rebuilding would
# re-download the 3.6 GB PyTorch base layer. This layer only adds the packages
# the P2 image deliberately omitted (it was scoped to training + host probes).
#
#   bash scripts/p3_docker_host.sh build
#   bash scripts/p3_docker_host.sh export  <weights> <run-dir>
#   bash scripts/p3_docker_host.sh eval    <weights> <data.yaml> <run-dir>
#   bash scripts/p3_docker_host.sh parity  <onnx> <weights> <run-dir>
#   bash scripts/p3_docker_host.sh shell
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE_IMAGE="${P2_DOCKER_IMAGE:-espdet-gesture:torch220-cu121}"
IMAGE="${P3_DOCKER_IMAGE:-espdet-gesture-p3:torch220-cu121}"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}
fail() { echo "BLOCKED: $*" >&2; exit 2; }

action="${1:-}"
shift || true

if [[ "$action" == "build" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || \
    fail "Base image $BASE_IMAGE missing: run scripts/p2_docker_host.sh build first"
  cat > /tmp/p3.Dockerfile <<'DOCKER'
ARG BASE
FROM ${BASE}
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Pinned to the versions ESP-Detection's deploy/export.py expects. onnxruntime
# (CPU) is enough for the parity check; no GPU inference is required.
# The link from this host to PyPI is slow and drops long reads, so pip gets an
# explicit timeout and retries rather than failing the whole layer.
RUN python -m pip install --no-cache-dir \
      --timeout 120 --retries 10 \
      "onnx==1.17.0" "onnxsim==0.4.36" "onnxruntime==1.19.2" \
    && python -m pip check \
    && python -c "import onnx,onnxruntime,onnxsim; print('onnx', onnx.__version__); print('ort', onnxruntime.__version__)"
DOCKER
  docker build --progress=plain -f /tmp/p3.Dockerfile --build-arg BASE="$BASE_IMAGE" -t "$IMAGE" /tmp
  exit 0
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  fail "Image $IMAGE missing: run bash scripts/p3_docker_host.sh build first"

common=(
  --rm --gpus all --shm-size=8g
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env YOLO_CONFIG_DIR=/workspace/repo/runs/ultralytics-cfg
  --env MPLCONFIGDIR=/workspace/repo/runs/ultralytics-cfg
  --volume "$ROOT_DIR:/workspace/repo"
  --workdir /workspace/repo
)

case "$action" in
  export)
    [[ "$#" == 2 ]] || fail "export needs <weights> <run-dir>"
    docker run "${common[@]}" "$IMAGE" python -m scripts.p3_eval export \
      --weights "$1" --run-dir "$2"
    ;;
  eval)
    [[ "$#" == 3 ]] || fail "eval needs <weights> <data.yaml> <run-dir>"
    docker run "${common[@]}" "$IMAGE" python -m scripts.p3_eval eval \
      --weights "$1" --data "$2" --run-dir "$3"
    ;;
  parity)
    [[ "$#" == 3 ]] || fail "parity needs <onnx> <weights> <run-dir>"
    docker run "${common[@]}" "$IMAGE" python -m scripts.p3_eval parity \
      --onnx "$1" --weights "$2" --run-dir "$3"
    ;;
  shell)
    docker run --interactive --tty "${common[@]}" "$IMAGE" bash
    ;;
  *)
    usage
    ;;
esac
