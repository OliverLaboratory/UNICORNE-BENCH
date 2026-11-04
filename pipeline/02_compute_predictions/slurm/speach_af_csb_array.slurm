#!/bin/bash
#SBATCH --account=csb_gpu_acc
#SBATCH --qos=csb_account_batch_gpu
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=speach_af_1k
#SBATCH --output=logs/speach_af/speach_af_%A_%a.log
# --array gets appended at submit time (e.g. --array=0-79%80)
#
# Each task runs --gpu-shard SLURM_ARRAY_TASK_ID/SLURM_ARRAY_TASK_COUNT
# against the dataset using SPEACH_AF (Stein & Mchaourab 2022): alanine
# mutagenesis on MSA sliding windows + ColabFold AF2 inference.
#
# Launch:
#   sbatch --array=0-79%80 tool_scripts/speach_af/speach_af_csb_array.slurm UNICORNE_BENCH UNICORNE
#
# --resume skips ensembles that already have >= num_samples PDBs.

set -euo pipefail

DATASET="${1:?Usage: sbatch --array=... $0 DATASET MSA_DATASET}"
MSA_DATASET="${2:-$DATASET}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/oliver_lab/gonzc11/ConformationDiscoveryBenchmark}"
cd "$PROJECT_ROOT"

mkdir -p logs/speach_af

PIXI_PY="${PIXI_PY:-/data/oliver_lab/gonzc11/localcolabfold/.pixi/envs/default/bin/python3.12}"
COLABFOLD_DATA="${COLABFOLD_DATA:-/data/oliver_lab/gonzc11/colabfold_cache}"

if [[ ! -x "$PIXI_PY" ]]; then
  echo "Error: localcolabfold env missing at $PIXI_PY"; exit 1
fi
if [[ ! -d "$COLABFOLD_DATA/params" ]]; then
  echo "Error: ColabFold params missing: $COLABFOLD_DATA/params"; exit 1
fi

SHARD_OFFSET="${SHARD_OFFSET:-0}"
SHARD_IDX=$((SLURM_ARRAY_TASK_ID + SHARD_OFFSET))
SHARD_TOT="${SHARD_TOTAL:-${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}}"

echo "=== SPEACH_AF top-up to 1000 on $DATASET (MSA=$MSA_DATASET) ==="
echo "Shard: $SHARD_IDX / $SHARD_TOT  (array_id=$SLURM_ARRAY_TASK_ID, offset=$SHARD_OFFSET)"
echo "Node : $(hostname) -- GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

"$PIXI_PY" pipeline/02_compute_predictions/runners/run_speach_af.py \
  "data/inputs/$DATASET" \
  "$DATASET" \
  --msa-dir "$PROJECT_ROOT/data/msas/$MSA_DATASET" \
  --data-dir "$COLABFOLD_DATA" \
  --num-samples 1000 \
  --seeds-per-set 9 \
  --seeds-base 3 \
  --resume \
  --gpu-shard "$SHARD_IDX/$SHARD_TOT" \
  --log-file "logs/speach_af/speach_af_shard${SHARD_IDX}_of_${SHARD_TOT}.log"
