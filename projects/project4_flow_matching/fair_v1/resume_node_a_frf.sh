#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the interrupted seed-44 F-RF run on AutoDL node A, then run A's
# remaining evaluation suite. Start this script in a persistent screen session.
ROOT=/root/autodl-tmp/project4-fair-work
CKPT=/root/autodl-tmp/checkpoints/project4-fair-v1
export PATH=/root/miniconda3/bin:$PATH
cd "$ROOT"

CURRENT=F-RF
write_state() {
  printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"
}
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR

if pgrep -af '[p]roject4_flow_matching/fair_v1/train.py.*fm_reflow.yaml' >/dev/null; then
  echo "F-RF training already has an active process; refusing to start a duplicate." >&2
  exit 2
fi
test -s "$ROOT/results/project4_fair_v1/noise/noise_seed1001_n5000.pt"

write_state F-RF running "resuming seed=44 from latest.pt; checkpoint step=10000 target=50000"
python3 projects/project4_flow_matching/fair_v1/train.py \
  --config projects/project4_flow_matching/fair_v1/configs/fm_reflow.yaml \
  --output "$CKPT/F-RF" --seed 44 --resume "$CKPT/F-RF/latest.pt" \
  2>&1 | tee -a "$CKPT/F-RF/console.log"

test -s "$CKPT/F-RF/step_50000.pt"
CURRENT=local_advanced_complete
write_state "$CURRENT" complete "F-RF reached step=50000; starting node A evaluations"
CURRENT=node_A_evaluation
bash projects/project4_flow_matching/fair_v1/run_eval_node_a.sh \
  2>&1 | tee -a "$CKPT/node_A_eval_runner.log"
