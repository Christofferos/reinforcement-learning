#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Auto-restart training wrapper for Hide & Seek MAPPO.
#
# If train.py exits with code 42 (CUDA/NaN crash), it restarts
# automatically with --resume pointing to the latest model dir.
# Any other exit code stops the loop.
#
# Usage:
#   bash scripts/run_training.sh               # fresh start
#   bash scripts/run_training.sh --resume models/hideseek_mappo_20260316_112935
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

MAX_RESTARTS=50
RESTART_COUNT=0
EXIT_CODE=0

# Pass all args through (e.g. --n_envs 32 --n_episodes 100000 ...)
ARGS="$@"

while true; do
    echo ""
    echo "========================================"
    echo " Training attempt $((RESTART_COUNT + 1)) / $MAX_RESTARTS"
    echo "========================================"
    echo ""

    python scripts/train.py $ARGS
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 42 ]; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "[X] Hit max restarts ($MAX_RESTARTS). Stopping."
            exit 1
        fi

        # Find the latest model directory to resume from
        LATEST_DIR=$(ls -td models/hideseek_mappo_* 2>/dev/null | head -1)
        if [ -z "$LATEST_DIR" ]; then
            echo "[X] No model directory found to resume from. Stopping."
            exit 1
        fi

        echo ""
        echo "[>>] Auto-restarting with --resume $LATEST_DIR (attempt $RESTART_COUNT)"
        echo "     Waiting 10s for GPU to settle..."
        sleep 10

        # Replace or add --resume in ARGS
        if echo "$ARGS" | grep -q -- "--resume"; then
            ARGS=$(echo "$ARGS" | sed "s|--resume [^ ]*|--resume $LATEST_DIR|")
        else
            ARGS="$ARGS --resume $LATEST_DIR"
        fi
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "[OK] Training completed successfully!"
        exit 0
    else
        echo "[X] Training exited with code $EXIT_CODE. Stopping."
        exit $EXIT_CODE
    fi
done
