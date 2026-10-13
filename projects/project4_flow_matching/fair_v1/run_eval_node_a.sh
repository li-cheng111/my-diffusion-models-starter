#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/project4-fair-work
CKPT=/root/autodl-tmp/checkpoints/project4-fair-v1
RESULT=$ROOT/results/fair_v1
cd "$ROOT"
CURRENT=waiting-local-advanced
write_state(){ printf '{"experiment":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$CKPT/current_state.json"; }
trap 'rc=$?; write_state "$CURRENT" failed "exit=$rc"; exit "$rc"' ERR
while :; do
 state=$(python3 -c "import json;d=json.load(open('$CKPT/current_state.json'));print(d.get('experiment',''),d.get('status',''))")
 [[ "$state" == "local_advanced_complete complete" ]] && break
 sleep 30
done
python3 - <<'PY'
import json
import os
from pathlib import Path
import torch

ckpt_dir = Path("/root/autodl-tmp/checkpoints/project4-fair-v1")
model_paths = [ckpt_dir / "F-RF/step_50000.pt", ckpt_dir / "F-RF/latest.pt"]
metadata_path = ckpt_dir / "F-RF/run_metadata.json"
teacher_sha = json.loads((ckpt_dir / "reflow_pairs/manifest.json").read_text())["teacher_sha256"]
checkpoint = torch.load(model_paths[0], map_location="cpu", weights_only=False)
if int(checkpoint.get("step", -1)) != 50000:
    raise ValueError(f"unexpected F-RF final step: {checkpoint.get('step')}")
checkpoint["parent_checkpoint_sha256"] = teacher_sha
for model_path in model_paths:
    temporary = model_path.with_suffix(".lineage.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, model_path)
metadata = json.loads(metadata_path.read_text())
metadata["parent_checkpoint_sha256"] = teacher_sha
metadata["resume_from_step"] = 10000
metadata["recovery_note"] = "The interrupted attempt logged through step 15300; the saved checkpoint resumed at step 10000, and the training log preserves both attempts."
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
print(f"F-RF parent checkpoint lineage restored: {teacher_sha}", flush=True)
PY
run_eval(){
 exp=$1; name=$2; method=$3; solver=$4; condition=$5; shift 5
 local steps=200000
 [[ "$exp" == F-CT || "$exp" == F-RF ]] && steps=50000
 local output="$RESULT/$exp/$name.json"
 mkdir -p "$(dirname "$output")"
 CURRENT="eval-$exp-$name"; write_state "$CURRENT" running "node=A solver=$solver"
 python3 projects/project4_flow_matching/fair_v1/evaluate.py --model "$CKPT/$exp/step_$steps.pt" --method "$method" --solver "$solver" --condition "$condition" --num-samples 5000 --batch-size 64 --seed 1001 --data-root ./data --noise-bank ./results/project4_fair_v1/noise/noise_seed1001_n5000.pt "$@" --output "$output" --samples-dir "$RESULT/$exp/$name" 2>&1 | tee "$output.log"
 write_state "$CURRENT" complete "result=$output"
}
NFE=(2 4 8 10 16 20 32 50 64 100)
run_eval D-U ddim ddpm ddim unconditional --nfe "${NFE[@]}"
run_eval D-U ancestral ddpm ancestral unconditional --nfe 1000
run_eval D-U ddim-no-clip ddpm ddim unconditional --nfe 20 --no-clip
run_eval F-U euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-U heun fm heun unconditional --nfe "${NFE[@]}"
run_eval F-CT euler fm euler unconditional --nfe "${NFE[@]}"
run_eval F-RF euler fm euler unconditional --nfe "${NFE[@]}"
CURRENT=node_A_archive; write_state "$CURRENT" running "packing node A source, checkpoints metadata, logs, results, and fixed noise bank"
ARCHIVE_ROOT="$CKPT/node_A_archive_work"
PKG="$ARCHIVE_ROOT/package"
if [[ -e "$PKG" ]]; then echo "Refusing to reuse existing archive staging path: $PKG" >&2; exit 2; fi
mkdir -p "$PKG/projects/project4_flow_matching/fair_v1" \
  "$PKG/projects/project4_flow_matching/logs/fair_v1" \
  "$PKG/projects/project4_flow_matching/results/fair_v1" \
  "$PKG/results/project4_fair_v1/noise" "$PKG/reference/reflow_pairs"
cp -a "$ROOT/projects/project4_flow_matching/fair_v1/." "$PKG/projects/project4_flow_matching/fair_v1/"
cp "$ROOT/results/project4_fair_v1/noise/noise_seed1001_n5000.pt" "$PKG/results/project4_fair_v1/noise/"
cp "$CKPT/reflow_pairs/manifest.json" "$CKPT/reflow_pairs/teacher_stability.json" "$PKG/reference/reflow_pairs/"
for exp in D-U F-U F-CT F-RF; do
  mkdir -p "$PKG/projects/project4_flow_matching/logs/fair_v1/$exp"
  cp "$CKPT/$exp/console.log" "$CKPT/$exp/train_log.csv" "$CKPT/$exp/run_metadata.json" \
    "$PKG/projects/project4_flow_matching/logs/fair_v1/$exp/"
  cp -a "$RESULT/$exp" "$PKG/projects/project4_flow_matching/results/fair_v1/"
done
git -C "$ROOT" diff --binary HEAD -- projects/project4_flow_matching/fair_v1 > \
  "$PKG/projects/project4_flow_matching/fair_v1/node_a_worktree.patch"
git -C "$ROOT" rev-parse HEAD > "$PKG/SOURCE_REVISION.txt"
printf '%s  %s\n' "$(sha256sum "$CKPT/D-U/step_200000.pt" | cut -d' ' -f1)" \
  "project4-fair-v1-a-d-u-seed44-step200000.pt" \
  "$(sha256sum "$CKPT/F-U/step_200000.pt" | cut -d' ' -f1)" \
  "project4-fair-v1-a-f-u-seed44-step200000.pt" \
  "$(sha256sum "$CKPT/F-CT/step_50000.pt" | cut -d' ' -f1)" \
  "project4-fair-v1-a-f-ct-seed44-step50000.pt" \
  "$(sha256sum "$CKPT/F-RF/step_50000.pt" | cut -d' ' -f1)" \
  "project4-fair-v1-a-f-rf-seed44-step50000.pt" > "$PKG/CHECKPOINT_SHA256SUMS.txt"
cat > "$PKG/PROJECT4_FAIR_V1_NODE_A_ARCHIVE.txt" <<'EOF'
AutoDL node A — Project4 fair-v1, seed 44

Extract this archive at the repository root. It restores the A source snapshot, D-U/F-U/F-CT/F-RF training logs and metadata, evaluation JSON/logs/sample grids, fixed noise bank, and the deterministic reflow-pair manifest. Final model checkpoints are separate Release assets and are listed in CHECKPOINT_SHA256SUMS.txt.

F-RF resumed from checkpoint step 10000 and completed at step 50000. The interrupted attempt log reached step 15300; the raw CSV/console logs preserve both attempts. Reflow pairs use teacher F-U, 50000 pairs, seed 2001, and Heun NFE 200. The 20 derived pair shards are omitted and can be regenerated from the included manifest/source, F-U checkpoint, and CIFAR-10 data.

The shared evaluation noise bank is results/project4_fair_v1/noise/noise_seed1001_n5000.pt.
EOF
tar -czf "$CKPT/node_A_fair_v1_repro.tar.gz.tmp" -C "$PKG" .
mv "$CKPT/node_A_fair_v1_repro.tar.gz.tmp" "$CKPT/node_A_fair_v1_repro.tar.gz"
sha256sum "$CKPT/node_A_fair_v1_repro.tar.gz"
CURRENT=node_A_complete
write_state "$CURRENT" complete "F-RF resumed, A evaluations verified, and node A repro archive created"
