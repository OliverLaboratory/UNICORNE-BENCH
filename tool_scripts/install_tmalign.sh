#!/bin/bash
# Install TM-align C++ executable (required for MMD and TM-score metrics).
# Source: https://aideepmed.com/TM-align/TMalign.cpp
#
# Usage:
#   ./tool_scripts/install_tmalign.sh [DEST_DIR]
#   Default DEST_DIR: $HOME/bin (creates if needed)
#   Set TMALIGN_EXE or add DEST_DIR to PATH to use.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/bin}"
mkdir -p "$DEST"
cd "$DEST"

echo "=== Installing TM-align to $DEST ==="
if [[ ! -f TMalign.cpp ]]; then
    echo "Downloading TMalign.cpp..."
    curl -sL -o TMalign.cpp "https://aideepmed.com/TM-align/TMalign.cpp"
fi

# macOS: malloc.h not available, use stdlib.h
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i.bak 's|#include <malloc.h>|#include <stdlib.h>|' TMalign.cpp
fi

echo "Compiling..."
# Mac: omit -static (not supported); Linux: -static optional for portability
g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp

echo "TM-align installed: $DEST/TMalign"
echo ""
echo "To use:"
echo "  export PATH=\"$DEST:\$PATH\""
echo "  # Or: export TMALIGN_EXE=\"$DEST/TMalign\""
