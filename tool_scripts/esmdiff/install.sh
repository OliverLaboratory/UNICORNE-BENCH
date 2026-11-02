#!/bin/bash
# Install ESMDiff on a server.
# Usage: bash install.sh
#
# Prerequisites:
#   - pixi installed (~/.pixi/bin/pixi)
#   - CUDA-capable GPU
#   - HuggingFace token with ESM3 access (run: huggingface-cli login)

set -e

ESMDIFF_DIR="$HOME/esmdiff"

echo "=== Installing ESMDiff ==="

# Clone if not present
if [ ! -d "$ESMDIFF_DIR" ]; then
    echo "Cloning ESMDiff..."
    git clone https://github.com/lujiarui/esmdiff.git "$ESMDIFF_DIR"
fi

cd "$ESMDIFF_DIR"

# Initialize pixi environment
if [ ! -f pixi.toml ]; then
    echo "Creating pixi.toml..."
    cat > pixi.toml << 'EOF'
[workspace]
channels = ["conda-forge"]
name = "esmdiff"
platforms = ["linux-64"]
version = "0.1.0"

[tasks]

[dependencies]
python = "3.10.*"
pip = ">=26.0.1,<27"
EOF
fi

# Install pixi environment
pixi install

# Install PyTorch with CUDA
pixi run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ESMDiff dependencies
pixi run pip install -r requirements.txt

# Install ESMDiff in editable mode
pixi run pip install -e .

# Download checkpoint if not present
CKPT_DIR="$ESMDIFF_DIR/data/ckpt"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/release_v0.pt" ]; then
    echo "Downloading ESMDiff checkpoint (5.5 GB)..."
    pixi run pip install gdown
    pixi run gdown "1vvPl5SDi0K1SdGn67r_qOLPqQr8h6dGP" -O "$CKPT_DIR/release_v0.pt"
fi

echo ""
echo "=== Installation complete ==="
echo "Checkpoint: $CKPT_DIR/release_v0.pt"
echo ""
echo "IMPORTANT: You must authenticate with HuggingFace for ESM3 access:"
echo "  cd $ESMDIFF_DIR && pixi run huggingface-cli login"
echo ""
echo "Test with:"
echo "  cd $ESMDIFF_DIR && pixi run python -c 'from esm.models.esm3 import ESM3; m = ESM3.from_pretrained(\"esm3_sm_open_v1\"); print(\"OK!\")'"
