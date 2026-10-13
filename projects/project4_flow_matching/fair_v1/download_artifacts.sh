#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="li-cheng111/my-diffusion-models-starter"
TAG="project4-fair-v1.0"
BASE="https://github.com/$REPO/releases/download/$TAG"
OUT_DIR="${1:-artifacts/project4-fair-v1}"
SCOPE="${2:-C}"
SUMS="$SCRIPT_DIR/ASSET_SHA256SUMS.txt"

if [[ ! -f "$SUMS" ]]; then
  echo "Missing committed checksum manifest: $SUMS" >&2
  exit 1
fi

case "${SCOPE,,}" in
  c|fdit)
    ASSETS=(
      project4-fair-v1-fdit-seed44-step200000.pt
      project4-fair-v1-fdit-seed44-repro.tar.gz
    ) ;;
  a)
    ASSETS=(
      project4-fair-v1-a-d-u-seed44-step200000.pt
      project4-fair-v1-a-f-u-seed44-step200000.pt
      project4-fair-v1-a-f-ct-seed44-step50000.pt
      project4-fair-v1-a-f-rf-seed44-step50000.pt
    ) ;;
  b)
    ASSETS=(
      project4-fair-v1-b-f-c-seed44-step200000.pt
      project4-fair-v1-b-f-ln-seed44-step200000.pt
      project4-fair-v1-node-b-seed44-repro.tar.gz
    ) ;;
  all)
    ASSETS=(
      project4-fair-v1-fdit-seed44-step200000.pt
      project4-fair-v1-fdit-seed44-repro.tar.gz
      project4-fair-v1-a-d-u-seed44-step200000.pt
      project4-fair-v1-a-f-u-seed44-step200000.pt
      project4-fair-v1-a-f-ct-seed44-step50000.pt
      project4-fair-v1-a-f-rf-seed44-step50000.pt
      project4-fair-v1-b-f-c-seed44-step200000.pt
      project4-fair-v1-b-f-ln-seed44-step200000.pt
      project4-fair-v1-node-b-seed44-repro.tar.gz
    ) ;;
  *)
    echo "Usage: $0 [output-directory] [C|A|B|all]" >&2
    exit 2 ;;
esac

mkdir -p "$OUT_DIR" "$OUT_DIR/unpacked"
for asset in "${ASSETS[@]}"; do
  expected="$(awk -v name="$asset" '$2 == name {print $1}' "$SUMS")"
  if [[ -z "$expected" ]]; then
    echo "No SHA-256 entry for $asset in $SUMS" >&2
    exit 1
  fi
  curl -fL --retry 3 "$BASE/$asset" -o "$OUT_DIR/$asset"
  printf '%s  %s\n' "$expected" "$OUT_DIR/$asset" | sha256sum -c -
  if [[ "$asset" == *.tar.gz ]]; then
    tar -xzf "$OUT_DIR/$asset" -C "$OUT_DIR/unpacked"
  fi
done
printf 'Downloaded and verified Project 4 fair-v1 scope %s in %s\n' "$SCOPE" "$OUT_DIR"
