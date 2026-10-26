#!/usr/bin/env bash
# Weekend data download: fill the private JPEG cache, then build the dataset.
#
# Re-runnable. Does not restart training and never overwrites an existing
# dataset directory. Everything is logged under runs/ (git-ignored).
#
#   bash _run_weekend.sh            # download 500/50/50 then build
#   bash _run_weekend.sh 300 30 30  # different quota
set -uo pipefail

ROOT=/home/mnt-xm/esp32s3-gesture-recognition
cd "$ROOT" || exit 1

TRAIN=${1:-500}
VAL=${2:-50}
TEST=${3:-50}
TAG="${TRAIN}-${VAL}-${TEST}"
OUT="data/processed/hagrid10-${TAG}-v1"
LOG="runs/weekend-${TAG}.log"

exec >>"$LOG" 2>&1
echo "=== start $(date -Is) quota=${TAG} ==="

# Stage 1: fill the cache. Loops until every required JPEG is present.
python3 _prefetch_all.py --train "$TRAIN" --val "$VAL" --test "$TEST" \
  --cache data/cache/hagrid10-p1-05 --workers 16
prefetch_rc=$?
echo "prefetch_exit=$prefetch_rc"
if [ "$prefetch_rc" -ne 0 ]; then
  echo "CACHE INCOMPLETE - dataset build skipped (re-run this script to resume)"
  echo "=== end $(date -Is) ==="
  exit "$prefetch_rc"
fi

# Stage 2: build from the now-complete cache (offline, minutes).
if [ -e "$OUT" ]; then
  echo "OUTPUT_EXISTS: $OUT (refusing to overwrite)"
else
  python3 -m scripts.p1_dataset_builder build \
    --mode auto \
    --cache data/cache/hagrid10-p1-05 \
    --output "$OUT" \
    --license-ack data/reviews/hagrid-license-ack.json \
    --mapping-ack data/reviews/hagrid-mapping-ack.json \
    --train-per-source "$TRAIN" --val-per-source "$VAL" --test-per-source "$TEST" \
    --max-transfer-gib 60 --max-output-gib 40 \
    --prefetch-workers 16
  echo "build_exit=$?"
fi
echo "=== end $(date -Is) ==="
