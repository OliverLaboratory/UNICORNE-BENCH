#!/bin/bash
# Launch AlphaFlow-PDB (alphafold mode, PDB-distilled) on a dataset.
#
# Prerequisites:
#   1. MSAs generated: bash tool_scripts/alphaflow/generate_msas.sh <DATASET>
#   2. Checkpoint: ~/alphaflow/weights/alphaflow_pdb_distilled_202402.pt
#
# Usage:
#   bash tool_scripts/alphaflow/launch_af_pdb.sh DANCE [--gpu-shard 0/5] [--resume]
#   bash tool_scripts/alphaflow/launch_af_pdb.sh UNICORNE [--gpu-shard 0/3] [--resume]
#   bash tool_scripts/alphaflow/launch_af_pdb.sh CFOLD [--resume]

set -euo pipefail

DATASET="${1:?Usage: $0 <DANCE|CFOLD|UNICORNE> [extra args...]}"
shift  # remaining args passed through

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MSA_DIR="$PROJECT_ROOT/data/msas/$DATASET"
INPUT_PATH="$PROJECT_ROOT/data/inputs/$DATASET"
RUN_ID="${DATASET}"

if [ ! -d "$MSA_DIR" ]; then
    echo "Error: MSA directory not found: $MSA_DIR"
    echo "Run: bash tool_scripts/alphaflow/generate_msas.sh $DATASET"
    exit 1
fi

echo "=== AlphaFlow-PDB on $DATASET ==="
echo "MSA dir: $MSA_DIR ($(ls -d "$MSA_DIR"/*/a3m 2>/dev/null | wc -l) proteins)"
echo "Output: data/outputs/alphaflow_pdb/$RUN_ID/"
echo "Extra args: $@"
echo ""

cd "$PROJECT_ROOT"
"$HOME/alphaflow_venv/bin/python" tool_scripts/alphaflow/run_dataset.py \
    "$INPUT_PATH" \
    "$RUN_ID" \
    --model af_pdb_distilled \
    --msa-dir "$MSA_DIR" \
    --output-base "$PROJECT_ROOT/data/outputs" \
    --num-samples 250 \
    "$@"
