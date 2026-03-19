#!/bin/bash
# ─────────────────────────────────────────────────
# Curriculum Phase 2: Complex arenas, tool use
# ─────────────────────────────────────────────────
# Run this AFTER Phase 1 completes.
#
# What changes from Phase 1:
#   - Layouts: divider, cross, L_shape (was: divider, cross)
#   - Boxes:   2-3 (was: 2 fixed)
#   - Ramps:   0-1 (was: 0)
#   - Horizon: 240 (was: 200)
#   - Entropy: 0.02→0.008 (was: 0.03→0.01)
#   - Episodes: 10,000 (was: 5,000)
#   - Network: SAME as Phase 1 (128-dim) — weights transfer!
#
# Usage:
#   bash scripts/train_phase2.sh

cd "$(dirname "$0")/.."

PHASE1_MODEL="models/hideseek_mappo_20260318_195133"

echo "============================================="
echo "  Curriculum Phase 2: Learn Tool Use"
echo "  Resuming from: $PHASE1_MODEL"
echo "============================================="

python scripts/train.py \
    --curriculum_phase 2 \
    --resume "$PHASE1_MODEL"
