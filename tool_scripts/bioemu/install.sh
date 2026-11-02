#!/bin/bash
# Install BioEMU and dependencies for running on a GPU server.
# Run on target machine (e.g. ssh gonzc11@bf64.csb.vanderbilt.edu).
#
# Requirements: Python 3.10-3.12, CUDA (for GPU)
# BioEMU is Linux-only.
#
# If system Python is < 3.10, use conda:
#   conda create -n bioemu python=3.11
#   conda activate bioemu
#   ./tool_scripts/bioemu/install.sh

set -e

PYTHON="${PYTHON:-python3}"
VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
if [[ "$VER" < "3.10" ]]; then
  echo "Error: BioEMU requires Python 3.10-3.12. Current: $VER"
  echo "Use conda: conda create -n bioemu python=3.11 && conda activate bioemu"
  exit 1
fi

echo "=== Installing BioEMU (Python $VER) ==="

# Create venv if not in one
if [[ -z "$VIRTUAL_ENV" ]]; then
  VENV_DIR="${VENV_DIR:-$HOME/bioemu_venv}"
  echo "Creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

pip install --upgrade pip
pip install "bioemu>=0.1"
pip install mdtraj  # for converting XTC to benchmark PDB format

# Install benchmark package for run_dataset.py
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
pip install -e .

echo ""
echo "=== BioEMU installed. Activate with: source $VIRTUAL_ENV/bin/activate ==="
echo "First run will download model weights and set up ColabFold for MSA embeddings."
