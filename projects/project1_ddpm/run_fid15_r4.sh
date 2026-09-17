#!/usr/bin/env bash
set -u

# R4 is the only new training run in the final plan.  It keeps the R3
# controls fixed and changes only the Min-SNR-gamma epsilon loss weighting.
cd "$(dirname "$0")"
mkdir -p logs runs/fid15_r4_min_snr/ckpt runs/fid15_r4_min_snr/samples

LOG_PATH="${R4_LOG_PATH:-logs/fid15_r4_min_snr.log}"
PID_PATH="${R4_PID_PATH:-runs/fid15_r4_min_snr.pid}"
CKPT="runs/fid15_r4_min_snr/ckpt/final.pt"
export PYTHONUNBUFFERED=1

echo $$ > "$PID_PATH"
cleanup() { rm -f "$PID_PATH"; }
trap cleanup EXIT INT TERM

run_logged() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_PATH"
  "$@" 2>&1 | tee -a "$LOG_PATH"
  local code=${PIPESTATUS[0]}
  if [[ $code -ne 0 ]]; then
    echo "[$(date '+%F %T')] command failed with exit code $code" | tee -a "$LOG_PATH"
    return "$code"
  fi
}

echo "[$(date '+%F %T')] ===== START R4_min_snr =====" | tee -a "$LOG_PATH"
if [[ ! -f "$CKPT" ]]; then
  run_logged python -u train.py --config configs/cifar10_fid15_r4_min_snr.yaml || exit $?
else
  echo "[$(date '+%F %T')] final checkpoint already exists; skip training" | tee -a "$LOG_PATH"
fi

if [[ -f "$CKPT" ]]; then
  run_logged python -u sample.py \
    --ckpt "$CKPT" --num_samples 64 --batch_size 64 --seed 44 \
    --ema_decay 0.9995 \
    --clip_denoised --save_grid \
    --output_dir runs/fid15_r4_min_snr/samples_final_ema9995_clipx0 || true
  run_logged python -u evaluate.py \
    --ckpt "$CKPT" --num_samples 5000 --batch_size 64 \
    --real_split train --seed 44 --ema_decay 0.9995 \
    --clip_denoised --data_root ./data || true
  run_logged python -u evaluate.py \
    --ckpt "$CKPT" --num_samples 5000 --batch_size 64 \
    --real_split train --seed 44 --compare_ema \
    --clip_denoised --data_root ./data || true
fi
echo "[$(date '+%F %T')] ===== END R4_min_snr =====" | tee -a "$LOG_PATH"
