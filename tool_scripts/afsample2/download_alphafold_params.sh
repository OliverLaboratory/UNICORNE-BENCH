#!/bin/bash
# Download AlphaFold2 model parameters (~3.5GB).
# Required for AFsample2. Puts params in DATA_DIR/params/
#
# Usage:
#   ./download_alphafold_params.sh [DATA_DIR]
#   ./download_alphafold_params.sh ~/alphafold3-data

set -e
DATA_DIR="${1:-$HOME/alphafold3-data}"
PARAMS_DIR="${DATA_DIR}/params"
SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
BASENAME="alphafold_params_2022-12-06.tar"

mkdir -p "$PARAMS_DIR"
cd "$PARAMS_DIR"

if [[ -f params_model_1.npz ]] || [[ -f params_model_1.npy ]]; then
  echo "Params already present in $PARAMS_DIR"
  ls -la params_model*.npz params_model*.npy 2>/dev/null || true
  exit 0
fi

echo "=== Downloading AlphaFold2 params to $PARAMS_DIR ==="
if command -v aria2c &>/dev/null; then
  aria2c "$SOURCE_URL" --dir="$PARAMS_DIR" -o "$BASENAME" --max-connection-per-server=4 --continue=true
else
  wget -c --show-progress --tries=3 -O "$PARAMS_DIR/$BASENAME" "$SOURCE_URL"
fi

echo "Extracting..."
tar -xf "$PARAMS_DIR/$BASENAME" -C "$PARAMS_DIR" --preserve-permissions
rm -f "$PARAMS_DIR/$BASENAME"
echo "Done. Params in $PARAMS_DIR"
ls -la "$PARAMS_DIR"
