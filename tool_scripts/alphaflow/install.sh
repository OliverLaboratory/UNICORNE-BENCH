#!/bin/bash
# Install AlphaFlow (AlphaFold/ESMFlow for protein ensembles) on a GPU server.
# Run on target machine (e.g. ssh gonzc11@bf64.csb.vanderbilt.edu).
#
# Requirements: Python 3.9, CUDA 11
# We use ESMFlow (no MSA needed) for simpler setup.
#
# NO SUDO: Use --use-micromamba to install Python 3.9 via micromamba (user-space).
#   ./tool_scripts/alphaflow/install.sh --use-micromamba
#
# With conda: conda create -n alphaflow python=3.9 && conda activate alphaflow
# With sudo:  sudo dnf install python39-devel  (then use system python3.9)

set -e

USE_MICROMAMBA=false
for arg in "$@"; do
  [[ "$arg" == "--use-micromamba" ]] && USE_MICROMAMBA=true && break
done

setup_micromamba_env() {
  local MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  local MICROMAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
  local ENV_PREFIX="$HOME/alphaflow_venv"

  if [[ ! -x "$MICROMAMBA_BIN" ]]; then
    echo "=== Installing micromamba (no sudo required) ==="
    mkdir -p "$MAMBA_ROOT/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
  fi

  # Always recreate: existing venv may be from system python (no Python.h)
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "Removing existing env at $ENV_PREFIX (need micromamba Python with headers)"
    rm -rf "$ENV_PREFIX"
  fi
  echo "=== Creating Python 3.9 env with micromamba ==="
  "$MICROMAMBA_BIN" create -y -p "$ENV_PREFIX" python=3.9

  echo "=== Re-running install with micromamba Python 3.9 ==="
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  exec env PATH="$ENV_PREFIX/bin:$PATH" VIRTUAL_ENV="$ENV_PREFIX" bash "$SCRIPT_DIR/$(basename "$0")"
}

PYTHON="${PYTHON:-python3}"
VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
if [[ "$VER" != "3.9" ]]; then
  echo "Error: AlphaFlow requires Python 3.9. Current: $VER"
  echo "Use: ./tool_scripts/alphaflow/install.sh --use-micromamba"
  exit 1
fi

# Check for Python.h (needed for OpenFold build)
PYINC=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
if [[ -n "$PYINC" ]] && [[ ! -f "$PYINC/Python.h" ]]; then
  if [[ "$USE_MICROMAMBA" == "true" ]]; then
    echo "Python.h not found. Setting up micromamba env..."
    setup_micromamba_env
  fi
  echo "Error: Python.h not found at $PYINC/Python.h (OpenFold build needs it)"
  echo "Use: ./tool_scripts/alphaflow/install.sh --use-micromamba  (no sudo)"
  exit 1
fi

if [[ "$USE_MICROMAMBA" == "true" ]]; then
  setup_micromamba_env
fi

echo "=== Installing AlphaFlow (Python $VER) ==="

# Create venv if not in one
if [[ -z "$VIRTUAL_ENV" ]]; then
  VENV_DIR="${VENV_DIR:-$HOME/alphaflow_venv}"
  echo "Creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

pip install --upgrade pip wheel ninja
# AlphaFlow uses specific versions; adjust if needed for your CUDA
pip install "numpy>=1.21,<1.24" "pandas>=1.5"
pip install torch
pip install "biopython==1.79" dm-tree modelcif ml-collections scipy absl-py einops
pip install pytorch_lightning fair-esm wandb
# mdtraj optional (needs python3-dev for build); skip if fails
pip install mdtraj || true

# OpenFold (AlphaFlow dependency; requires CUDA)
# --no-build-isolation needed: setup.py imports torch, which must be in venv
if ! python -c "import openfold" 2>/dev/null; then
  echo "Installing OpenFold..."
  pip install --no-build-isolation 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037' || {
    echo "OpenFold install failed. May need: conda install -c nvidia cuda-toolkit=11.8"
    exit 1
  }
fi

# Patch OpenFold for numpy>=1.24 compatibility (np.object removed)
OPENFOLD_TEMPLATES=$(python -c "import openfold; import os; print(os.path.join(os.path.dirname(openfold.__file__), 'data', 'templates.py'))" 2>/dev/null)
if [[ -f "$OPENFOLD_TEMPLATES" ]] && grep -q 'np\.object' "$OPENFOLD_TEMPLATES" 2>/dev/null; then
  echo "Patching OpenFold templates.py for numpy compatibility..."
  sed -i.bak 's/np\.object/object/g' "$OPENFOLD_TEMPLATES"
fi

# Clone AlphaFlow (no setup.py; run predict.py from repo root)
ALPHAFLOW_DIR="${ALPHAFLOW_DIR:-$HOME/alphaflow}"
if [[ ! -d "$ALPHAFLOW_DIR" ]]; then
  echo "Cloning AlphaFlow..."
  git clone https://github.com/bjing2016/alphaflow.git "$ALPHAFLOW_DIR"
fi

# Download ESMFlow weights (no MSA needed)
WEIGHTS_DIR="${ALPHAFLOW_DIR}/weights"
mkdir -p "$WEIGHTS_DIR"
for name in esmflow_md_distilled esmflow_pdb_distilled; do
  if [[ ! -f "$WEIGHTS_DIR/${name}.pt" ]]; then
    echo "Downloading ${name} weights..."
    url_name="${name}_202402"
    url_name="${url_name//_/%5F}"
    curl -L -o "$WEIGHTS_DIR/${name}.pt" \
      "https://storage.googleapis.com/alphaflow/params/${url_name}.pt"
  fi
done

# ESMFold base weights (required for ESMFlow)
if [[ ! -f "$ALPHAFLOW_DIR/esmfold_3B_v1.pt" ]]; then
  echo "Downloading ESMFold 3B weights..."
  wget -q -O "$ALPHAFLOW_DIR/esmfold_3B_v1.pt" \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt" || true
fi

# Install benchmark package for run_dataset.py
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
pip install -e .

echo ""
echo "=== AlphaFlow installed. Activate with: export PATH=$VIRTUAL_ENV/bin:\$PATH ==="
echo "Weights: $WEIGHTS_DIR/esmflow_md_distilled.pt"
echo "ESMFold base: $ALPHAFLOW_DIR/esmfold_3B_v1.pt (must be in AlphaFlow repo root for --original_weights)"
