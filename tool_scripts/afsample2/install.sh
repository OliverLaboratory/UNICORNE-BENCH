#!/bin/bash
# Install AFsample2 (AlphaFold2-based conformational sampling) on a GPU server.
# Run on target machine (e.g. ssh user@server).
#
# Requirements: AlphaFold2 sequence databases (see https://github.com/deepmind/alphafold)
# AFsample2 uses conda; for no-sudo use --use-micromamba.
#
# Usage:
#   ./tool_scripts/afsample2/install.sh
#   ./tool_scripts/afsample2/install.sh --use-micromamba
#
# After install, download databases (one-time, ~2TB full / ~600GB reduced):
#   cd ~/AFsample2/AF_multitemplate/scripts && ./download_all_data.sh <data_path> reduced_dbs

set -e

USE_MICROMAMBA=false
for arg in "$@"; do
  [[ "$arg" == "--use-micromamba" ]] && USE_MICROMAMBA=true && break
done

AFSAMPLE2_DIR="${AFSAMPLE2_DIR:-$HOME/AFsample2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Clone if not present
if [[ ! -d "$AFSAMPLE2_DIR" ]]; then
  echo "=== Cloning AFsample2 ==="
  git clone https://github.com/wallnerlab/AFsample2.git "$AFSAMPLE2_DIR"
fi

if [[ "$USE_MICROMAMBA" == "true" ]]; then
  MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  MICROMAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
  ENV_PREFIX="$HOME/afsample2_venv"

  if [[ ! -x "$MICROMAMBA_BIN" ]]; then
    echo "=== Installing micromamba (no sudo required) ==="
    mkdir -p "$MAMBA_ROOT/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
  fi

  if [[ -d "$ENV_PREFIX" ]]; then
    echo "Removing existing env at $ENV_PREFIX"
    rm -rf "$ENV_PREFIX"
  fi
  echo "=== Creating AFsample2 env with micromamba ==="
  "$MICROMAMBA_BIN" create -y -p "$ENV_PREFIX" -f "$AFSAMPLE2_DIR/environment.yaml"
  echo "=== Installing pip requirements ==="
  "$MICROMAMBA_BIN" run -p "$ENV_PREFIX" pip install -r "$AFSAMPLE2_DIR/requirements.txt"
  echo "=== Installing benchmark package ==="
  "$MICROMAMBA_BIN" run -p "$ENV_PREFIX" pip install -e "$PROJECT_ROOT"
  echo ""
  echo "=== AFsample2 installed. Activate with: export PATH=$ENV_PREFIX/bin:\$PATH ==="
  echo "Download databases: cd $AFSAMPLE2_DIR/AF_multitemplate/scripts && ./download_all_data.sh <data_path> reduced_dbs"
  exit 0
fi

# Conda path
if command -v conda &>/dev/null; then
  echo "=== Creating AFsample2 conda env ==="
  conda env create -n afsample2 -f "$AFSAMPLE2_DIR/environment.yaml" || conda env update -n afsample2 -f "$AFSAMPLE2_DIR/environment.yaml"
  echo "=== Installing pip requirements ==="
  conda run -n afsample2 pip install -r "$AFSAMPLE2_DIR/requirements.txt"
  echo "=== Installing benchmark package ==="
  conda run -n afsample2 pip install -e "$PROJECT_ROOT"
  echo ""
  echo "=== AFsample2 installed. Activate with: conda activate afsample2 ==="
else
  echo "Conda not found. Use: ./tool_scripts/afsample2/install.sh --use-micromamba"
  exit 1
fi

echo "Download databases: cd $AFSAMPLE2_DIR/AF_multitemplate/scripts && ./download_all_data.sh <data_path> reduced_dbs"
