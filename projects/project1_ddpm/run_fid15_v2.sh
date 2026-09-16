#!/usr/bin/env bash
set -u

# Run the strict 200-epoch experiments one at a time.  The script is intended
# for screen/nohup on AutoDL and keeps each experiment in its own directory.
cd "$(dirname "$0")"
mkdir -p logs runs

LOG_PATH="${FID15_LOG_PATH:-logs/fid15_v2_runner.log}"
PID_PATH="${FID15_PID_PATH:-runs/fid15_v2_runner.pid}"
TOTAL_STEPS=78000
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

run_experiment() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local checkpoint="$output_dir/ckpt/final.pt"

  echo "[$(date '+%F %T')] ===== START $label =====" | tee -a "$LOG_PATH"
  if [[ ! -f "$checkpoint" ]]; then
    if ! run_logged python -u train.py --config "$config"; then
      echo "[$(date '+%F %T')] $label training failed; continuing to the next experiment" | tee -a "$LOG_PATH"
      return 1
    fi
  else
    echo "[$(date '+%F %T')] $label checkpoint already exists; skip training" | tee -a "$LOG_PATH"
  fi

  if [[ -f "$checkpoint" ]]; then
    run_logged python -u sample.py --ckpt "$checkpoint" --num_samples 64 --batch_size 64 --seed 44 --save_grid || true
    run_logged python -u evaluate.py --ckpt "$checkpoint" --num_samples 5000 --batch_size 64 --real_split train --seed 44 || true
    run_logged python -u evaluate.py --ckpt "$checkpoint" --num_samples 5000 --batch_size 64 --real_split train --seed 44 --clip_denoised || true
  fi
  echo "[$(date '+%F %T')] ===== END $label =====" | tee -a "$LOG_PATH"
  return 0
}

run_experiment R1_full_linear_default \
  configs/cifar10_fid15_full_linear_default.yaml \
  runs/fid15_v2_full_linear_default
run_experiment R2_full_linear_cosinelr \
  configs/cifar10_fid15_full_linear_cosinelr.yaml \
  runs/fid15_v2_full_linear_cosinelr
run_experiment R3_full_cosine_default \
  configs/cifar10_fid15_full_cosine_default.yaml \
  runs/fid15_v2_full_cosine_default

echo "[$(date '+%F %T')] ===== PRIMARY EXPERIMENTS COMPLETE =====" | tee -a "$LOG_PATH"

# The primary EMA (0.9999) determines the candidate.  Compare all EMA banks
# and raw weights only for the best primary result to limit redundant FID runs.
best_score=""
best_ckpt=""
for candidate in \
  runs/fid15_v2_full_linear_default/ckpt/final.pt \
  runs/fid15_v2_full_linear_cosinelr/ckpt/final.pt \
  runs/fid15_v2_full_cosine_default/ckpt/final.pt; do
  score_file="$(dirname "$candidate")/fid_5000_EMA_0.9999_noclipx0.txt"
  if [[ -f "$score_file" ]]; then
    score="$(awk '/^FID:/{print $2}' "$score_file")"
    if [[ -n "$score" ]] && { [[ -z "$best_score" ]] || awk "BEGIN{exit !($score < $best_score)}"; }; then
      best_score="$score"
      best_ckpt="$candidate"
    fi
  fi
done

if [[ -n "$best_ckpt" ]]; then
  echo "[$(date '+%F %T')] Best primary checkpoint: $best_ckpt (FID $best_score)" | tee -a "$LOG_PATH"
  run_logged python -u evaluate.py --ckpt "$best_ckpt" --num_samples 5000 --batch_size 64 --real_split train --seed 44 --compare_ema || true
  run_logged python -u evaluate.py --ckpt "$best_ckpt" --num_samples 5000 --batch_size 64 --real_split train --seed 44 --compare_ema --clip_denoised || true
else
  echo "[$(date '+%F %T')] No primary FID result was available; EMA comparison deferred" | tee -a "$LOG_PATH"
fi

echo "[$(date '+%F %T')] ===== ALL FID15 V2 WORK COMPLETE =====" | tee -a "$LOG_PATH"
