#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-/root/lr25_200ep}"
RESUME_CKPT="${RESUME_CKPT:-/root/lr_screen_50ep/runs/LR25/ckpt/final.pt}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CONFIG="$PROJECT_DIR/configs/cifar10_lr25_200ep_resume.yaml"
LOG="$ROOT/combined.log"
PID_FILE="$ROOT/launcher.pid"

mkdir -p "$ROOT/ckpt" "$ROOT/samples" "$ROOT/eval"
echo "$$" > "$PID_FILE"
: > "$LOG"
echo "===== START LR25_200EP =====" | tee -a "$LOG"

if [ ! -f "$RESUME_CKPT" ]; then
  echo "resume checkpoint missing: $RESUME_CKPT" | tee -a "$LOG"
  exit 2
fi

echo "[LR25_200EP] resume=$RESUME_CKPT output=$ROOT" | tee -a "$LOG"
set +e
"$PYTHON_BIN" -u "$PROJECT_DIR/train.py" \
  --config "$CONFIG" \
  --resume "$RESUME_CKPT" \
  --output_dir "$ROOT" \
  2>&1 | tee "$ROOT/train.log" | tee -a "$LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "[LR25_200EP] training failed: exit=$TRAIN_RC" | tee -a "$LOG"
  exit "$TRAIN_RC"
fi

echo "[LR25_200EP] formal FID comparison start" | tee -a "$LOG"
set +e
"$PYTHON_BIN" -u "$PROJECT_DIR/evaluate.py" \
  --ckpt "$ROOT/ckpt/final.pt" \
  --num_samples 5000 \
  --batch_size 64 \
  --real_split train \
  --seed 44 \
  --compare_ema \
  --clip_denoised \
  --data_root "$PROJECT_DIR/data" \
  2>&1 | tee "$ROOT/eval/fid_compare.log" | tee -a "$LOG"
EVAL_RC=${PIPESTATUS[0]}
set -e
if [ "$EVAL_RC" -ne 0 ]; then
  echo "[LR25_200EP] FID evaluation failed: exit=$EVAL_RC" | tee -a "$LOG"
  exit "$EVAL_RC"
fi

echo "[LR25_200EP] best EMA grid start" | tee -a "$LOG"
"$PYTHON_BIN" -u "$PROJECT_DIR/sample.py" \
  --ckpt "$ROOT/ckpt/final.pt" \
  --num_samples 64 \
  --batch_size 64 \
  --ema_decay 0.9999 \
  --seed 44 \
  --save_grid \
  --clip_denoised \
  --output_dir "$ROOT/samples_final_ema9999_clipx0" \
  2>&1 | tee "$ROOT/eval/sample_grid.log" | tee -a "$LOG"

echo "[LR25_200EP] all training and evaluation finished" | tee -a "$LOG"
echo "===== END LR25_200EP =====" | tee -a "$LOG"
