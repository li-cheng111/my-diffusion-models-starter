#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/project4-fair-work-git
CKPT=/root/autodl-tmp/checkpoints/project4-fair-v1
RESULT=$ROOT/results/fair_v1/F-DiT
cd "$ROOT"
CURRENT=F-DiT
write_state(){ printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"; }
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR
OUT="$CKPT/F-DiT"; mkdir -p "$OUT"
write_state F-DiT running "node=C config=fm_dit.yaml"
args=(--config projects/project4_flow_matching/fair_v1/configs/fm_dit.yaml --output "$OUT" --seed 44)
if [[ -f "$OUT/latest.pt" ]]; then args+=(--resume "$OUT/latest.pt"); fi
python3 projects/project4_flow_matching/fair_v1/train.py "${args[@]}" 2>&1 | tee -a "$OUT/console.log"
write_state F-DiT complete "step=200000"
mkdir -p "$RESULT"
CURRENT=eval-F-DiT-euler; write_state "$CURRENT" running "node=C solver=euler"
python3 projects/project4_flow_matching/fair_v1/postprocess_node_c.py
python3 projects/project4_flow_matching/fair_v1/evaluate.py --model "$OUT/step_200000.pt" --method fm --solver euler --condition unconditional --nfe 2 4 8 10 16 20 32 50 64 100 --num-samples 5000 --batch-size 64 --seed 1001 --data-root ./data --output "$RESULT/euler.json" --samples-dir "$RESULT/euler" 2>&1 | tee "$RESULT/euler.log"
write_state "$CURRENT" complete "result=$RESULT/euler.json"
python3 projects/project4_flow_matching/fair_v1/postprocess_node_c.py
write_state node_C_complete complete "F-DiT training and evaluation finished"
