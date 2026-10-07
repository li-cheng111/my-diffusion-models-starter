#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/project5_fair}"
# Project 4's training process can finish before its FID sweeps and packaging
# finish. Wait for the entire Project 4 workflow, not just train.py.
WAIT_PATTERN="${GPU_WAIT_PATTERN:-projects/project4_flow_matching/|post_project4_eval\.sh|package_project4_assets\.sh}"
QUEUE_DIR="$RUN_ROOT/_queue"
QUEUE_STATUS="$QUEUE_DIR/status.json"
mkdir -p "$QUEUE_DIR" "$PROJECT_DIR/logs"

write_status() {
  local state="$1" message="$2"
  printf '{"state":"%s","message":"%s","updated_at":"%s"}\n' \
    "$state" "$message" "$(date -Is)" > "$QUEUE_STATUS"
}

write_status waiting_gpu "Waiting for Project 4 evaluation and asset packaging to finish"
while pgrep -f "$WAIT_PATTERN" >/dev/null; do
  sleep 30
done

# Require three consecutive idle samples so a short gap between jobs does not
# start a second training process while another GPU task is being scheduled.
idle_samples=0
while (( idle_samples < 3 )); do
  utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ' || true)"
  if [[ "$utilization" =~ ^[0-9]+$ ]] && (( utilization < 10 )); then
    ((idle_samples += 1))
  else
    idle_samples=0
  fi
  write_status waiting_gpu "Project 4 is finished; GPU utilization ${utilization:-unknown}%; waiting for three idle samples"
  sleep 30
done

write_status starting "Launching the ten-condition Project 5 matrix with training seed 42"
cd "$PROJECT_DIR"
set +e
python -u run_fair_experiments.py --preset full --seed 42 --resume \
  > "$PROJECT_DIR/logs/fair_full_run.log" 2>&1
result=$?
set -e
if (( result == 0 )); then
  write_status launched "Experiment runner finished; see fair_summary.json"
else
  write_status failed "Experiment runner exited with code $result"
fi
exit "$result"
