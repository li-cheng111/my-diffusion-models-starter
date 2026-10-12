#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  OUT_DIR="$1"
else
  OUT_DIR="artifacts/project4-fair-v1"
fi
BASE="https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-fair-v1.0"
CHECKPOINT="project4-fair-v1-fdit-seed44-step200000.pt"
BUNDLE="project4-fair-v1-fdit-seed44-repro.tar.gz"
CHECKPOINT_SHA="d0bd447c276c08a2188835c827c2f7a5c77eab58c533f99d1544c6e04ef26710"
BUNDLE_SHA="515997d69d8401917e7294d960f0df43e76174ba5dbce621ba22fd6271727072"

mkdir -p "$OUT_DIR"
curl -fL --retry 3 "$BASE/$CHECKPOINT" -o "$OUT_DIR/$CHECKPOINT"
curl -fL --retry 3 "$BASE/$BUNDLE" -o "$OUT_DIR/$BUNDLE"
printf '%s  %s\n' "$CHECKPOINT_SHA" "$OUT_DIR/$CHECKPOINT" | sha256sum -c -
printf '%s  %s\n' "$BUNDLE_SHA" "$OUT_DIR/$BUNDLE" | sha256sum -c -
mkdir -p "$OUT_DIR/unpacked"
tar -xzf "$OUT_DIR/$BUNDLE" -C "$OUT_DIR/unpacked"
printf 'Verified checkpoint and reproducibility bundle in %s\n' "$OUT_DIR"
