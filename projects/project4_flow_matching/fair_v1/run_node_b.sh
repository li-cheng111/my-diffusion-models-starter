#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT="${P4_FAIR_CKPT_ROOT:-/root/autodl-tmp/checkpoints/project4-fair-v1}"
RESULT=$ROOT/results/fair_v1
if [[ -x /root/miniconda3/bin/python3 ]]; then export PATH=/root/miniconda3/bin:$PATH; fi
cd "$ROOT"
CURRENT=F-C
write_state(){ printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"; }
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR
run_one(){
 exp=$1; config=$2; steps=$3; out="$CKPT/$1"; CURRENT=$exp
 mkdir -p "$out"; write_state "$exp" running "node=B config=$config"
 args=(--config "$config" --output "$out" --seed 44)
 if [[ -f "$out/latest.pt" ]]; then args+=(--resume "$out/latest.pt"); fi
 python3 projects/project4_flow_matching/fair_v1/train.py "${args[@]}" 2>&1 | tee -a "$out/console.log"
 write_state "$exp" complete "step=$steps"
}
run_one F-C projects/project4_flow_matching/fair_v1/configs/fm_conditional.yaml 200000
run_one F-LN projects/project4_flow_matching/fair_v1/configs/fm_logit_normal.yaml 200000
run_eval(){
 exp=$1; name=$2; method=$3; solver=$4; condition=$5; shift 5
 local steps=200000
 local output="$RESULT/$exp/$name.json"
 mkdir -p "$(dirname "$output")"
 CURRENT="eval-$exp-$name"; write_state "$CURRENT" running "node=B solver=$solver"
 python3 projects/project4_flow_matching/fair_v1/evaluate.py --model "$CKPT/$exp/step_$steps.pt" --method "$method" --solver "$solver" --condition "$condition" --num-samples 5000 --batch-size 64 --seed 1001 --data-root ./data "$@" --output "$output" --samples-dir "$RESULT/$exp/$name" 2>&1 | tee "$output.log"
 write_state "$CURRENT" complete "result=$output"
}
run_eval F-C cfg fm euler cfg --nfe 20 --cfg 1 2 3 5 7.5
run_eval F-LN euler fm euler unconditional --nfe 2 4 8 10 16 20 32 50 64 100
write_state node_B_complete complete "F-C F-LN training and evaluation finished"
