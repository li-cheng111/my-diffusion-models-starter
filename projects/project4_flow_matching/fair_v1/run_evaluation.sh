#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CKPT="${P4_FAIR_CKPT_ROOT:-/root/autodl-tmp/checkpoints/project4-fair-v1}"
RESULT="$ROOT/results/fair_v1"
BANK="$CKPT/eval_noise_bank.pt"
CURRENT=waiting-evaluation
write_state(){ printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"; }
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR
while :; do
  read_state="$(python3 -c 'import json;d=json.load(open("'"$CKPT"'/current_state.json"));print(d.get("experiment",""),d.get("status",""))')"
  if [[ "$read_state" == "all complete" ]]; then break; fi
  if [[ "$read_state" == *" failed" ]]; then echo "training failed; evaluation stopped"; exit 1; fi
  sleep 60
done
run_eval(){
  local exp="$1" name="$2" method="$3" solver="$4" condition="$5"; shift 5
  local steps=200000
  [[ "$exp" == F-CT || "$exp" == F-RF ]] && steps=50000
  local output="$RESULT/$exp/$name.json" samples="$RESULT/$exp/$name"
  mkdir -p "$(dirname "$output")"
  CURRENT="eval-$exp-$name"; write_state "$CURRENT" running "solver=$solver"
  python3 projects/project4_flow_matching/fair_v1/evaluate.py --model "$CKPT/$exp/step_$steps.pt" --method "$method" --solver "$solver" --condition "$condition" --num-samples 5000 --batch-size 64 --seed 1001 --data-root ./data --noise-bank "$BANK" --output "$output" --samples-dir "$samples" "$@" 2>&1 | tee "$output.log"
  write_state "$CURRENT" complete "result=$output"
}
NFE=(2 4 8 10 16 20 32 50 64 100)
run_eval D-U ddim ddpm ddim unconditional --nfe "${NFE[@]}"
run_eval D-U ancestral ddpm ancestral unconditional --nfe 1000
run_eval D-U ddim-no-clip ddpm ddim unconditional --nfe 20 --no-clip
run_eval F-U euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-U heun fm heun unconditional --nfe "${NFE[@]}"
run_eval F-C cfg fm euler cfg --nfe 20 --cfg 1 2 3 5 7.5
run_eval F-LN euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-DiT euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-CT euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-RF euler fm euler unconditional --nfe "${NFE[@]}"
write_state evaluation complete "all planned evaluations finished"
