#!/bin/bash
# Install TM-align C++ executable (required for all TM-score metrics).
# Compiles third_party/TMalign/TMalign.cpp: TM-align version 20220412, the
# exact source that produced the published scores
# (upstream: https://aideepmed.com/TM-align/TMalign.cpp).
#
# Usage:
#   ./tool_scripts/install_tmalign.sh [DEST_DIR]
#   Default DEST_DIR: $HOME/bin (creates if needed)
#   Set TMALIGN_EXE or add DEST_DIR to PATH to use.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/bin}"
mkdir -p "$DEST"

echo "=== Installing TM-align to $DEST ==="
cp "$SCRIPT_DIR/../third_party/TMalign/TMalign.cpp" "$DEST/TMalign.cpp"
cd "$DEST"

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
