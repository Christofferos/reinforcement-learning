#!/usr/bin/env bash
# Sequential flat -> rough -> procedural curriculum; each stage resumes the previous one.
# Usage: ./scripts/run_curriculum.sh [num_envs]
set -euo pipefail

cd "$(dirname "$0")/.."
NUM_ENVS="${1:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"

declare -a STAGES=(
  "Parkour-Flat-v0:${FLAT_STEPS:-10000000}"
  "Parkour-Rough-v0:${ROUGH_STEPS:-20000000}"
  "Parkour-Procedural-v0:${PARKOUR_STEPS:-60000000}"
)

RESUME=""
for stage in "${STAGES[@]}"; do
  TASK="${stage%%:*}"
  STEPS="${stage##*:}"
  RUN="curriculum_${STAMP}_${TASK}"
  echo "=== ${TASK}  steps=${STEPS}  envs=${NUM_ENVS}  resume=${RESUME:-none} ==="
  python scripts/train.py "$TASK" \
    --num-envs "$NUM_ENVS" \
    --timesteps "$STEPS" \
    --run-name "$RUN" \
    ${RESUME:+--resume "$RESUME"}
  RESUME="models/${RUN}/final.zip"
done

echo "done -> ${RESUME}"
