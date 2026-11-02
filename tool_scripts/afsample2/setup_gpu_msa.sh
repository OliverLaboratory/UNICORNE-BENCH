#!/bin/bash
# Setup GPU-accelerated MSA for AFsample2 using ColabFold + MMseqs2-GPU.
#
# MMseqs2-GPU provides ~177x speedup over Jackhmmer for MSA search.
# ColabFold databases are different from AlphaFold2 - must be downloaded separately.
#
# Usage (run on each AFsample2 server: origen, ignatius, chesterton):
#   ./tool_scripts/afsample2/setup_gpu_msa.sh
#   ssh server 'bash -s' < tool_scripts/afsample2/setup_gpu_msa.sh
#
# After setup, use --use-gpu-msa when launching AFsample2.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths (customize if needed)
COLABFOLD_DB_DIR="${COLABFOLD_DB_DIR:-$HOME/colabfold_databases}"
MMSEQS_GPU_DIR="${MMSEQS_GPU_DIR:-$HOME/mmseqs2_gpu}"
VENV_DIR="${VENV_DIR:-$HOME/afsample2_gpu_msa_venv}"

echo "=== GPU MSA setup for AFsample2 ==="
echo "ColabFold DB dir: $COLABFOLD_DB_DIR"
echo "MMseqs2-GPU dir: $MMSEQS_GPU_DIR"
echo "Venv: $VENV_DIR"
echo ""

# 1. Create venv and install ColabFold (no AlphaFold - we only need colabfold_search)
echo "=== Installing ColabFold (colabfold_search only) ==="
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -U pip
# Install colabfold without alphafold extras (lighter)
"$VENV_DIR/bin/pip" install "colabfold[alphafold-minus-jax]" 2>/dev/null || \
  "$VENV_DIR/bin/pip" install colabfold

# 2. Install MMseqs2 (GPU or CPU)
echo ""
echo "=== Installing MMseqs2 ==="
mkdir -p "$MMSEQS_GPU_DIR"
cd "$MMSEQS_GPU_DIR"
if [[ ! -x mmseqs ]]; then
  # Try GPU build from mmseqs.com (Linux x86_64)
  if command -v wget &>/dev/null && [[ "$(uname -m)" == "x86_64" ]]; then
    echo "Downloading MMseqs2-GPU..."
    wget -q --show-progress -O mmseqs_gpu.tar.gz "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz" 2>/dev/null || true
  fi
  if [[ -f mmseqs_gpu.tar.gz ]]; then
    tar -xzf mmseqs_gpu.tar.gz
    find . -name mmseqs -type f -executable | head -1 | xargs -I {} mv {} ./mmseqs 2>/dev/null || true
    chmod +x mmseqs 2>/dev/null || true
    rm -f mmseqs_gpu.tar.gz
  fi
  if [[ ! -x mmseqs ]] && command -v micromamba &>/dev/null; then
    echo "Falling back to CPU MMseqs2 (micromamba)"
    micromamba install -y -c conda-forge mmseqs2 -p "$MMSEQS_GPU_DIR/conda_env" 2>/dev/null || true
    [[ -x "$MMSEQS_GPU_DIR/conda_env/bin/mmseqs" ]] && ln -sf "$MMSEQS_GPU_DIR/conda_env/bin/mmseqs" mmseqs
  fi
  if [[ ! -x mmseqs ]] && command -v conda &>/dev/null; then
    echo "Falling back to CPU MMseqs2 (conda)"
    conda install -y -c conda-forge mmseqs2 2>/dev/null || true
    [[ -n "$(which mmseqs 2>/dev/null)" ]] && ln -sf "$(which mmseqs)" mmseqs
  fi
fi
export PATH="$MMSEQS_GPU_DIR:$PATH"
[[ -x mmseqs ]] && echo "MMseqs2 found: $(mmseqs version 2>/dev/null || echo 'ok')" || echo "Warning: Install MMseqs2 from https://github.com/soedinglab/MMseqs2"

# 3. Download ColabFold databases (required for colabfold_search)
echo ""
echo "=== ColabFold databases ==="
if [[ ! -d "$COLABFOLD_DB_DIR" ]] || [[ ! -f "$COLABFOLD_DB_DIR/uniref30_2302_db.dbtype" ]]; then
  echo "Downloading ColabFold databases to $COLABFOLD_DB_DIR"
  echo "This is ~500GB with MMSEQS_NO_INDEX=1 (on-the-fly indexing)."
  mkdir -p "$COLABFOLD_DB_DIR"
  cd /tmp
  if [[ ! -d colabfold_setup ]]; then
    git clone --depth 1 https://github.com/sokrypton/ColabFold.git colabfold_setup
  fi
  cd colabfold_setup
  MMSEQS_NO_INDEX=1 ./setup_databases.sh "$COLABFOLD_DB_DIR"
  cd "$PROJECT_ROOT"
else
  echo "ColabFold databases already exist at $COLABFOLD_DB_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Add to your environment:"
echo "  export COLABFOLD_DB_DIR=$COLABFOLD_DB_DIR"
echo "  export PATH=$MMSEQS_GPU_DIR:\$PATH"
echo ""
echo "Launch AFsample2 with GPU MSA:"
echo "  ./tool_scripts/afsample2/launch_dataset_server.sh <server> <dataset> --use-gpu-msa --background"
echo ""
