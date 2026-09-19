#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-/root/lr_screen_30ep}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="$PROJECT_DIR/configs/cifar10_lr_screen_30ep_base.yaml"
LOG="$ROOT/combined.log"
PID_FILE="$ROOT/launcher.pid"

mkdir -p "$ROOT/configs" "$ROOT/runs"
echo "$$" > "$PID_FILE"
: > "$LOG"

SPECS=(
  "LR05:5.0e-5"
  "LR08:8.0e-5"
  "LR12:1.2e-4"
  "LR16:1.6e-4"
  "LR20:2.0e-4"
  "LR25:2.5e-4"
  "LR30:3.0e-4"
)

for SPEC in "${SPECS[@]}"; do
  LABEL="${SPEC%%:*}"
  LR="${SPEC#*:}"
  OUT="$ROOT/runs/$LABEL"
  CONFIG="$ROOT/configs/$LABEL.yaml"
  mkdir -p "$OUT/ckpt" "$OUT/samples"
  cp "$BASE_CONFIG" "$CONFIG"
  sed -i "s#^output_dir:.*#output_dir: $OUT#; s#^  lr:.*#  lr: $LR#" "$CONFIG"

  echo "===== START $LABEL =====" | tee -a "$LOG"
  echo "[LR_SCREEN] label=$LABEL lr=$LR output=$OUT" | tee -a "$LOG"
  set +e
  "$PYTHON_BIN" -u "$PROJECT_DIR/train.py" \
    --config "$CONFIG" \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log" | tee -a "$LOG"
  TRAIN_RC=${PIPESTATUS[0]}
  set -e
  if [ "$TRAIN_RC" -ne 0 ]; then
    echo "$LABEL training failed: exit=$TRAIN_RC" | tee -a "$LOG"
    echo "===== END $LABEL =====" | tee -a "$LOG"
    continue
  fi

  echo "[LR_SCREEN] quick FID start label=$LABEL" | tee -a "$LOG"
  set +e
  "$PYTHON_BIN" -u "$PROJECT_DIR/evaluate.py" \
    --ckpt "$OUT/ckpt/final.pt" \
    --num_samples 1000 \
    --batch_size 64 \
    --real_split train \
    --seed 44 \
    --ema_decay 0.9999 \
    --clip_denoised \
    --data_root "$PROJECT_DIR/data" \
    2>&1 | tee "$OUT/eval.log" | tee -a "$LOG"
  EVAL_RC=${PIPESTATUS[0]}
  set -e
  if [ "$EVAL_RC" -ne 0 ]; then
    echo "$LABEL evaluation failed: exit=$EVAL_RC" | tee -a "$LOG"
  fi
  echo "===== END $LABEL =====" | tee -a "$LOG"
done

echo "[LR_SCREEN] all candidates finished" | tee -a "$LOG"
