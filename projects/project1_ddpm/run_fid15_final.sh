#!/usr/bin/env bash
set -u

# Final 200-epoch comparison after R3/R4:
#   R5 = uniform loss + late cosine LR decay
#   R6 = Min-SNR-gamma=5 + late cosine LR decay
# Each run has its own output directory and is never allowed to overwrite
# the earlier R3/R4 artifacts.
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1
LOG_PATH="${FID15_FINAL_LOG_PATH:-logs/fid15_final.log}"
PID_PATH="${FID15_FINAL_PID_PATH:-runs/fid15_final.pid}"
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

run_one() {
  local label="$1"
  local config="$2"
  local run_dir="$3"
  local ckpt="$run_dir/ckpt/final.pt"
  echo "[$(date '+%F %T')] ===== START $label =====" | tee -a "$LOG_PATH"
  if [[ ! -f "$ckpt" ]]; then
    run_logged python -u train.py --config "$config" || return $?
  else
    echo "[$(date '+%F %T')] final checkpoint already exists; skip training" | tee -a "$LOG_PATH"
  fi
  if [[ -f "$ckpt" ]]; then
    run_logged python -u sample.py \
      --ckpt "$ckpt" --num_samples 64 --batch_size 64 --seed 44 \
      --ema_decay 0.9995 --clip_denoised --save_grid \
      --output_dir "$run_dir/samples_final_ema9995_clipx0" || true
    run_logged python -u evaluate.py \
      --ckpt "$ckpt" --num_samples 5000 --batch_size 64 \
      --real_split train --seed 44 --ema_decay 0.9995 \
      --clip_denoised --data_root ./data || true
    run_logged python -u evaluate.py \
      --ckpt "$ckpt" --num_samples 5000 --batch_size 64 \
      --real_split train --seed 44 --compare_ema \
      --clip_denoised --data_root ./data || true
  fi
  echo "[$(date '+%F %T')] ===== END $label =====" | tee -a "$LOG_PATH"
}

run_one R5_late_decay configs/cifar10_fid15_r5_late_decay.yaml runs/fid15_r5_late_decay || exit $?
run_one R6_min_snr_late_decay configs/cifar10_fid15_r6_min_snr_late_decay.yaml runs/fid15_r6_min_snr_late_decay || exit $?
