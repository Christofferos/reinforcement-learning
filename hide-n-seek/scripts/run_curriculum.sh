#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Curriculum Training Pipeline for Hide & Seek MAPPO
# ──────────────────────────────────────────────────────────────
#
# Runs all 3 curriculum phases sequentially, each phase resuming
# from the previous phase's best checkpoint.
#
# Total: ~35,000 episodes across 3 phases
#   Phase 1:  5,000 episodes  — 2v2, simple arenas, no ramps
#   Phase 2: 10,000 episodes  — 2v2, complex arenas, ramps
#   Phase 3: 20,000 episodes  — 2v2, full complexity
#
# Estimated wall time (M2 Mac / single GPU):
#   Phase 1: ~1-2 hours
#   Phase 2: ~3-5 hours
#   Phase 3: ~8-14 hours
#   Total:   ~12-21 hours
#
# Usage:
#   bash scripts/run_curriculum.sh                 # full pipeline from scratch
#   bash scripts/run_curriculum.sh --skip_to 2     # skip to phase 2 (needs phase 1 checkpoint)
#   bash scripts/run_curriculum.sh --skip_to 3     # skip to phase 3 (needs phase 2 checkpoint)
#   bash scripts/run_curriculum.sh --render        # render env 0 during training
#
# To customise episodes per phase:
#   PHASE1_EPISODES=3000 PHASE2_EPISODES=8000 PHASE3_EPISODES=15000 \
#     bash scripts/run_curriculum.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Activate virtual environment ──
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# ── Parse arguments ──
SKIP_TO=1
RENDER_FLAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_to)
            SKIP_TO="$2"
            shift 2
            ;;
        --render)
            RENDER_FLAG="--render"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ── Episode counts (override with env vars) ──
PHASE1_EPISODES=${PHASE1_EPISODES:-5000}
PHASE2_EPISODES=${PHASE2_EPISODES:-10000}
PHASE3_EPISODES=${PHASE3_EPISODES:-20000}

# ── Detect device ──
DEVICE="cpu"
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null && \
    DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')") || true

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Hide & Seek — Curriculum Training Pipeline          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Phase 1: ${PHASE1_EPISODES} ep — 2v2 simple (movement + chasing)       ║"
echo "║  Phase 2: ${PHASE2_EPISODES} ep — 2v2 complex (tool use)              ║"
echo "║  Phase 3: ${PHASE3_EPISODES} ep — 2v2 full (team strategies)          ║"
echo "║  Device:  ${DEVICE}                                          ║"
echo "║  Skip to: Phase ${SKIP_TO}                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Track the latest model dir for --resume chaining
LATEST_MODEL_DIR=""

# Helper: find the most recent model directory
find_latest_model() {
    ls -td models/hideseek_mappo_* 2>/dev/null | head -1
}

# ──────────────────────────────────────────────────────────────
# PHASE 1: Simple 1v1 — learn basic movement & chasing
# ──────────────────────────────────────────────────────────────
if [ "$SKIP_TO" -le 1 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 1: 2v2 Simple — ${PHASE1_EPISODES} episodes"
    echo "  Learn: movement, chasing, fleeing, basic cover-seeking"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python scripts/train.py \
        --curriculum_phase 1 \
        --n_episodes "$PHASE1_EPISODES" \
        --n_envs 16 \
        --device "$DEVICE" \
        $RENDER_FLAG \
        $EXTRA_ARGS

    LATEST_MODEL_DIR=$(find_latest_model)
    echo ""
    echo "  [✓] Phase 1 complete → $LATEST_MODEL_DIR"
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# PHASE 2: Tool-use 1v1 — learn to grab & push objects
# ──────────────────────────────────────────────────────────────
if [ "$SKIP_TO" -le 2 ]; then
    # If skipping to phase 2, find previous checkpoint
    if [ "$SKIP_TO" -eq 2 ] || [ -z "$LATEST_MODEL_DIR" ]; then
        LATEST_MODEL_DIR=$(find_latest_model)
    fi

    RESUME_FLAG=""
    if [ -n "$LATEST_MODEL_DIR" ]; then
        RESUME_FLAG="--resume $LATEST_MODEL_DIR"
        echo "  [>>] Resuming from Phase 1: $LATEST_MODEL_DIR"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 2: 2v2 Complex — ${PHASE2_EPISODES} episodes"
    echo "  Learn: grabbing, pushing, barricade building, ramp use"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python scripts/train.py \
        --curriculum_phase 2 \
        --n_episodes "$PHASE2_EPISODES" \
        --n_envs 16 \
        --device "$DEVICE" \
        $RESUME_FLAG \
        $RENDER_FLAG \
        $EXTRA_ARGS

    LATEST_MODEL_DIR=$(find_latest_model)
    echo ""
    echo "  [✓] Phase 2 complete → $LATEST_MODEL_DIR"
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# PHASE 3: Full 2v2 — coordinated team strategies
# ──────────────────────────────────────────────────────────────
if [ "$SKIP_TO" -le 3 ]; then
    # If skipping to phase 3, find previous checkpoint
    if [ "$SKIP_TO" -eq 3 ] || [ -z "$LATEST_MODEL_DIR" ]; then
        LATEST_MODEL_DIR=$(find_latest_model)
    fi

    RESUME_FLAG=""
    if [ -n "$LATEST_MODEL_DIR" ]; then
        RESUME_FLAG="--resume $LATEST_MODEL_DIR"
        echo "  [>>] Resuming from Phase 2: $LATEST_MODEL_DIR"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 3: 2v2 Full — ${PHASE3_EPISODES} episodes"
    echo "  Learn: team coordination, counter-strategies, ramp exploitation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python scripts/train.py \
        --curriculum_phase 3 \
        --n_episodes "$PHASE3_EPISODES" \
        --n_envs 16 \
        --device "$DEVICE" \
        $RESUME_FLAG \
        $RENDER_FLAG \
        $EXTRA_ARGS

    LATEST_MODEL_DIR=$(find_latest_model)
    echo ""
    echo "  [✓] Phase 3 complete → $LATEST_MODEL_DIR"
    echo ""
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Curriculum Training Complete!                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Final model: $LATEST_MODEL_DIR"
echo "║                                                        ║"
echo "║  Evaluate with:                                        ║"
echo "║    python scripts/evaluate.py \\                        ║"
echo "║      --model_dir $LATEST_MODEL_DIR --render            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
