#!/usr/bin/env bash
# Sequential flat -> rough -> procedural curriculum; each stage resumes the previous one.
# Usage: ./scripts/all_curriculum.sh [num_envs]
set -euo pipefail

cd "$(dirname "$0")/.."
NUM_ENVS="${1:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"

declare -a STAGES=(
  "Flat:${FLAT_STEPS:-5000000}"
  "Rough:${ROUGH_STEPS:-10000000}"
  "Procedural:${PARKOUR_STEPS:-30000000}"
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
