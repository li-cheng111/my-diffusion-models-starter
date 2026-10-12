#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CKPT_ROOT="${P4_FAIR_CKPT_ROOT:-/root/autodl-tmp/checkpoints/project4-fair-v1}"
SEED="${P4_FAIR_SEED:-44}"
mkdir -p "$CKPT_ROOT"
write_state() {
  local experiment="$1" status="$2" detail="${3:-}"
  printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$experiment" "$status" "$detail" "$(date -Is)" > "$CKPT_ROOT/current_state.json"
}
run_one() {
  local experiment="$1" config="$2" output="$CKPT_ROOT/$1"
  mkdir -p "$output"
  write_state "$experiment" running "config=$config"
  if [[ -f "$output/latest.pt" ]]; then
    python3 projects/project4_flow_matching/fair_v1/train.py --config "$config" --output "$output" --seed "$SEED" --resume "$output/latest.pt" 2>&1 | tee -a "$output/console.log"
  else
    python3 projects/project4_flow_matching/fair_v1/train.py --config "$config" --output "$output" --seed "$SEED" 2>&1 | tee -a "$output/console.log"
  fi
  write_state "$experiment" complete "step=200000"
}
trap 'rc=$?; write_state "pair" failed "exit=$rc"; exit "$rc"' ERR
run_one D-U projects/project4_flow_matching/fair_v1/configs/ddpm_unet.yaml
run_one F-U projects/project4_flow_matching/fair_v1/configs/fm_unet.yaml
write_state pair complete "D-U and F-U finished"
