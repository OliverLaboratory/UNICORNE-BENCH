#!/bin/bash
# Launch AFsample2 on a benchmark dataset (paper-faithful: 1000 samples,
# rotating MSA randomization {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}).
#
# Prerequisites:
#   1. Apptainer SIF: ~/afsample2_v1.1.sif (override via SIF= env)
#      apptainer pull docker://kyogesh/afsample2:v1.1
#   2. AF databases: /data/mchaourab/alphafold3-data (override via DB_PATH=)
#   3. MSAs: data/msas/<MSA_DATASET>/<eid>/a3m/<eid>.a3m
#
# Usage:
#   bash run_afsample2.sh <INPUT_DATASET> <MSA_DATASET> [extra args...]
#   bash run_afsample2.sh DANCE_BENCH DANCE --gpu-shard 0/4 --resume
#   bash run_afsample2.sh UNICORNE_BENCH UNICORNE --num-samples 1000

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

echo "=== AFsample2 on $DATASET (MSA: $MSA_DATASET) ==="
echo "Input:     $INPUT_PATH"
echo "MSA dir:   $MSA_DIR ($(ls -d "$MSA_DIR"/*/a3m 2>/dev/null | wc -l) proteins)"
echo "Output:    data/outputs/afsample2/$DATASET/"
echo "Extra:     $*"
echo ""

python pipeline/02_compute_predictions/run_afsample2.py \
  "$INPUT_PATH" \
  "$DATASET" \
  --msa-dir "$MSA_DIR" \
  "$@"
