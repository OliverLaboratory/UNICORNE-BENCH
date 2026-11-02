#!/bin/bash
# Launch AFsample3 on a benchmark dataset (1000 samples, rotating MSA
# randomization {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} — same schedule as
# our AFsample2 runs).
#
# Prerequisites:
#   1. Apptainer SIF: ~/afsample3_run/afsample3.sif (override via SIF= env)
#      Built from https://github.com/wallnerlab/afsample3 (docker/Dockerfile)
#   2. AF3 weights: ~/af3_model/af3.bin (override via AF3_MODEL_DIR=)
#   3. MSAs: data/msas/<MSA_DATASET>/<eid>/a3m/<eid>.a3m
#
# AF3's data pipeline is skipped entirely (precomputed MSAs), so no sequence
# databases are needed at run time.
#
# Usage:
#   bash run_afsample3.sh <INPUT_DATASET> <MSA_DATASET> [extra args...]
#   bash run_afsample3.sh UNICORNE_BENCH UNICORNE --resume --gpu-shard 0/8
#   bash run_afsample3.sh UNICORNE_BENCH UNICORNE --num-samples 1000

set -euo pipefail

DATASET="${1:?Usage: $0 <INPUT_DATASET> <MSA_DATASET> [extra args...]}"
MSA_DATASET="${2:-$DATASET}"
shift 2 || shift 1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_PATH="$PROJECT_ROOT/data/inputs/$DATASET"
MSA_DIR="$PROJECT_ROOT/data/msas/$MSA_DATASET"

if [ ! -d "$MSA_DIR" ]; then
  echo "Error: MSA dir not found: $MSA_DIR" >&2
  exit 1
fi

echo "=== AFsample3 on $DATASET (MSA: $MSA_DATASET) ==="
echo "Input:     $INPUT_PATH"
echo "MSA dir:   $MSA_DIR ($(ls -d "$MSA_DIR"/*/a3m 2>/dev/null | wc -l) proteins)"
echo "Output:    data/outputs/afsample3/$DATASET/"
echo "Extra:     $*"
echo ""

python pipeline/02_compute_predictions/run_afsample3.py \
  "$INPUT_PATH" \
  "$DATASET" \
  --msa-dir "$MSA_DIR" \
  "$@"
