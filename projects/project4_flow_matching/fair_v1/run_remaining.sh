#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CKPT="${P4_FAIR_CKPT_ROOT:-/root/autodl-tmp/checkpoints/project4-fair-v1}"
SEED="${P4_FAIR_SEED:-44}"
CURRENT=waiting
write_state(){ printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"; }
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR
while :; do
  read_state="$(python3 -c 'import json;d=json.load(open("'"$CKPT"'/current_state.json"));print(d.get("experiment",""),d.get("status",""))')"
  if [[ "$read_state" == "pair complete" ]]; then break; fi
  if [[ "$read_state" == "pair failed" ]]; then echo "core pair failed; advanced runs stopped"; exit 1; fi
  sleep 60
done
run_one(){
  local exp="$1" config="$2" steps="$3" init="${4:-}" out="$CKPT/$1"
  CURRENT="$exp"; mkdir -p "$out"; write_state "$exp" running "config=$config"
  local args=(--config "$config" --output "$out" --seed "$SEED")
  if [[ -n "$init" ]]; then args+=(--init "$init"); fi
  if [[ -f "$out/latest.pt" ]]; then args+=(--resume "$out/latest.pt"); fi
  python3 projects/project4_flow_matching/fair_v1/train.py "${args[@]}" 2>&1 | tee -a "$out/console.log"
  write_state "$exp" complete "step=$steps"
}
run_one F-C projects/project4_flow_matching/fair_v1/configs/fm_conditional.yaml 200000
run_one F-LN projects/project4_flow_matching/fair_v1/configs/fm_logit_normal.yaml 200000
run_one F-DiT projects/project4_flow_matching/fair_v1/configs/fm_dit.yaml 200000
run_one F-CT projects/project4_flow_matching/fair_v1/configs/fm_continue.yaml 50000 "$CKPT/F-U/step_200000.pt"
CURRENT=F-RF-pairs; write_state "$CURRENT" running "teacher=F-U; pairs=50000"
python3 projects/project4_flow_matching/fair_v1/make_reflow_pairs.py --teacher "$CKPT/F-U/step_200000.pt" --output "$CKPT/reflow_pairs" --num-pairs 50000 --seed 2001
run_one F-RF projects/project4_flow_matching/fair_v1/configs/fm_reflow.yaml 50000 "$CKPT/F-U/step_200000.pt"
write_state all complete "all seven training runs finished"
