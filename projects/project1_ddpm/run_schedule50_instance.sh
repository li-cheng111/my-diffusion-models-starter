#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/root/autodl-tmp/schedule50}"
SEED="${2:?Usage: run_schedule50_instance.sh ROOT SEED [linear-first|cosine-first]}"
ORDER="${3:-linear-first}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
REPO_DIR="$ROOT/repo"
PROJECT_DIR="$REPO_DIR/projects/project1_ddpm"
RUN_ROOT="$ROOT/runs/seed${SEED}"
LOG_DIR="$ROOT/logs"
MONITOR_DIR="$ROOT/monitor"
TOTAL_ATTEMPTS=19500

case "$SEED:$ORDER" in
  42:linear-first|44:linear-first|43:cosine-first) ;;
  *) echo "Invalid seed/order assignment: $SEED $ORDER" >&2; exit 2 ;;
esac

mkdir -p "$ROOT" "$RUN_ROOT" "$LOG_DIR" "$MONITOR_DIR" "$ROOT/manifest"

if [[ -f "$REPO_DIR/.source_commit" && -f "$PROJECT_DIR/train.py" ]]; then
  echo "Using pre-provisioned source snapshot $(cat "$REPO_DIR/.source_commit"); skipping remote fetch."
elif [[ -d "$REPO_DIR/.git" ]]; then
  if git -C "$REPO_DIR" fetch origin main && git -C "$REPO_DIR" checkout -q origin/main; then
    :
  elif [[ -f "$PROJECT_DIR/train.py" ]]; then
    echo "GitHub fetch failed; continuing with the pre-provisioned source snapshot." >&2
  else
    echo "Git fetch failed and no local project snapshot is available." >&2
    exit 12
  fi
elif [[ ! -f "$PROJECT_DIR/train.py" ]]; then
  git clone --filter=blob:none --sparse https://github.com/li-cheng111/my-diffusion-models-starter.git "$REPO_DIR"
  git -C "$REPO_DIR" sparse-checkout set projects/project1_ddpm
else
  echo "Using the pre-provisioned project source snapshot."
fi

cd "$PROJECT_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 10
fi

"$PYTHON_BIN" - <<'PY'
import torch
try:
    import torch_musa
except ModuleNotFoundError as exc:
    if exc.name != "torch_musa":
        raise

if torch.cuda.is_available():
    print(f"CUDA GPU ready: {torch.cuda.get_device_name(0)}")
elif hasattr(torch, "musa") and torch.musa.is_available():
    print(f"MUSA GPU ready: {torch.musa.device_count()} visible device(s)")
else:
    raise SystemExit("GPU preflight failed: neither CUDA nor MUSA is visible to PyTorch")
PY

if [[ ! -d data/cifar-10-batches-py ]]; then
  echo "CIFAR-10 will be downloaded to $PROJECT_DIR/data before training." | tee -a "$LOG_DIR/combined.log"
  "$PYTHON_BIN" -c 'from dataset import get_dataset; get_dataset("cifar10", root="./data", image_size=32, train=True, augment=False)'
fi

SOURCE_COMMIT="${SOURCE_COMMIT:-$(cat "$REPO_DIR/.source_commit" 2>/dev/null || git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo snapshot-unknown)}"
printf 'git_commit=%s\npython=%s\n' "$SOURCE_COMMIT" "$($PYTHON_BIN --version 2>&1)" > "$ROOT/manifest/seed${SEED}_environment.txt"
"$PYTHON_BIN" - <<'PY' >> "$ROOT/manifest/seed${SEED}_environment.txt"
import torch
try:
    import torch_musa
except ModuleNotFoundError as exc:
    if exc.name != "torch_musa":
        raise
if torch.cuda.is_available():
    print(f"torch={torch.__version__}; backend=cuda; gpu={torch.cuda.get_device_name(0)}")
elif hasattr(torch, "musa") and torch.musa.is_available():
    print(f"torch={torch.__version__}; backend=musa; visible_devices={torch.musa.device_count()}")
else:
    print(f"torch={torch.__version__}; backend=cpu")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >> "$ROOT/manifest/seed${SEED}_environment.txt"
elif command -v musaInfo >/dev/null 2>&1; then
  musaInfo >> "$ROOT/manifest/seed${SEED}_environment.txt"
elif [[ -x /usr/local/musa/bin/musaInfo ]]; then
  /usr/local/musa/bin/musaInfo >> "$ROOT/manifest/seed${SEED}_environment.txt"
fi

MONITOR_PID="$MONITOR_DIR/dashboard.pid"
if [[ ! -f "$MONITOR_PID" ]] || ! kill -0 "$(cat "$MONITOR_PID")" 2>/dev/null; then
  LABEL_A="linear_seed${SEED}"
  LABEL_B="cosine_seed${SEED}"
  PATH_A="$RUN_ROOT/linear"
  PATH_B="$RUN_ROOT/cosine"
  nohup "$PYTHON_BIN" -u "$PROJECT_DIR/experiment_monitor.py" \
    --run "$LABEL_A=$PATH_A" --run "$LABEL_B=$PATH_B" \
    --total-steps "$TOTAL_ATTEMPTS" \
    --pid-file "$MONITOR_DIR/runner.pid" \
    --log-path "$LOG_DIR/combined.log" \
    --host 0.0.0.0 --port 8765 \
    > "$MONITOR_DIR/dashboard.log" 2>&1 &
  echo "$!" > "$MONITOR_PID"
fi

if [[ "$ORDER" == "linear-first" ]]; then
  EXPERIMENTS=(linear cosine)
else
  EXPERIMENTS=(cosine linear)
fi

echo "$$" > "$MONITOR_DIR/runner.pid"
touch "$LOG_DIR/combined.log"

for SCHEDULE in "${EXPERIMENTS[@]}"; do
  LABEL="${SCHEDULE}_seed${SEED}"
  RUN_DIR="$RUN_ROOT/$SCHEDULE"
  CONFIG="$PROJECT_DIR/configs/cifar10_schedule50_${SCHEDULE}.yaml"
  CKPT="$RUN_DIR/ckpt/final.pt"
  RESUME="$RUN_DIR/ckpt/epoch_latest.pt"

  mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/samples"
  echo "===== START $LABEL =====" | tee -a "$LOG_DIR/combined.log"
  if [[ -f "$RUN_DIR/ckpt/final.pt" ]]; then
    echo "===== SKIP TRAINING $LABEL (final checkpoint exists) =====" | tee -a "$LOG_DIR/combined.log"
  else
    TRAIN_ARGS=(--config "$CONFIG" --output_dir "$RUN_DIR" --seed "$SEED" --num_epochs 50)
    if [[ -f "$RESUME" ]]; then
      TRAIN_ARGS+=(--resume "$RESUME")
    fi
    set +e
    "$PYTHON_BIN" -u "$PROJECT_DIR/train.py" "${TRAIN_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/train.log" | tee -a "$LOG_DIR/combined.log"
    TRAIN_RC=${PIPESTATUS[0]}
    set -e
    if [[ "$TRAIN_RC" -ne 0 ]]; then
      echo "$LABEL training failed: exit=$TRAIN_RC" | tee -a "$LOG_DIR/combined.log"
      exit "$TRAIN_RC"
    fi
  fi

  for DECAY in 0.999 0.9995 0.9999; do
    DECAY_LABEL=$(printf '%0.4f' "$DECAY")
    EVAL_LOG="$RUN_DIR/eval_ema${DECAY}.log"
    if [[ -f "$RUN_DIR/ckpt/fid_5000_EMA_${DECAY_LABEL}_clipx0.txt" ]]; then
      echo "Existing FID found for $LABEL EMA $DECAY; leaving it untouched." | tee -a "$LOG_DIR/combined.log"
      continue
    fi
    echo "===== EVALUATE $LABEL EMA $DECAY =====" | tee -a "$LOG_DIR/combined.log"
    set +e
    "$PYTHON_BIN" -u "$PROJECT_DIR/evaluate.py" \
      --ckpt "$CKPT" --num_samples 5000 --batch_size 64 \
      --real_split train --seed 44 --ema_decay "$DECAY" --clip_denoised \
      --data_root "$PROJECT_DIR/data" \
      2>&1 | tee "$EVAL_LOG" | tee -a "$LOG_DIR/combined.log"
    EVAL_RC=${PIPESTATUS[0]}
    set -e
    if [[ "$EVAL_RC" -ne 0 ]]; then
      echo "$LABEL EMA $DECAY evaluation failed: exit=$EVAL_RC" | tee -a "$LOG_DIR/combined.log"
      exit "$EVAL_RC"
    fi
  done

  if [[ ! -f "$RUN_DIR/samples/grid.png" ]]; then
    "$PYTHON_BIN" -u "$PROJECT_DIR/sample.py" \
      --ckpt "$CKPT" --num_samples 64 --batch_size 64 \
      --ema_decay 0.9999 --seed 44 --clip_denoised --save_grid \
      --output_dir "$RUN_DIR/samples" \
      2>&1 | tee "$RUN_DIR/sample_final.log" | tee -a "$LOG_DIR/combined.log"
  fi
  echo "===== END $LABEL =====" | tee -a "$LOG_DIR/combined.log"
done

echo "===== ALL EXPERIMENTS COMPLETE FOR SEED $SEED =====" | tee -a "$LOG_DIR/combined.log"
rm -f "$MONITOR_DIR/runner.pid"
