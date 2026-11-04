#!/bin/bash
#SBATCH --account=oliver_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=afsample3
#SBATCH --output=%x_%A_%a.log
#
# AFsample3 on ACCRE. Each array task takes one --gpu-shard of the ensemble
# list, so array width == number of concurrent GPUs.
#
# Two ACCRE limits shape the sizing, both on nvidia_rtx_a6000 (the QOS rejects
# l40s / h100_80gb outright, and generic `--gres=gpu:1` is invalid here):
#   * QOS GrpTRES        gres/gpu:nvidia_rtx_a6000=80        -> 80 concurrent
#   * Assoc GrpTRESRunMins gres/gpu:nvidia_rtx_a6000=57600   -> GPU-*minutes*
# The second one is the subtle one: it caps concurrency x walltime, so a long
# walltime silently throttles width (48h requests pin you to ~20 jobs). At 8h,
# 80 tasks cost 38,400 GPU-min and stay comfortably inside the budget.
#
# Tasks are resume-safe (--resume skips ensembles already at --num-samples),
# so re-submitting the same array picks up wherever the last one stopped.
#
# Usage:
#   sbatch --array=0-79 afsample3_accre_array.slurm UNICORNE_BENCH UNICORNE

set -uo pipefail

DATASET="${1:-UNICORNE_BENCH}"
MSA_DATASET="${2:-UNICORNE}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
# Separate run id lets one input dataset feed several sampling configurations
# (e.g. UNICORNE_BENCH sampled at our schedule vs the AFsample3 paper's).
RUN_ID="${RUN_ID:-$DATASET}"
# Tool dir under data/outputs/; the results pipeline reads
# data/outputs/<tool>/<dataset>/output_pdbs, so a differently-configured run
# that should be scored separately sets TOOL, not RUN_ID.
TOOL="${TOOL:-afsample3}"
MSA_FRACTIONS="${MSA_FRACTIONS:-}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/mchaourab/gonzc11/afsample3/repo}"
export SIF="${SIF:-/data/mchaourab/gonzc11/afsample3/afsample3.sif}"
export AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/mchaourab/gonzc11/afsample3/model}"

cd "$PROJECT_ROOT"

# NSHARDS can exceed the array width so other hosts (e.g. bf65's H100s) can
# take the remaining shards of the same list and not duplicate ACCRE's work:
#   ACCRE: sbatch --array=0-79  with NSHARDS=88   -> shards 0..79
#   bf65:  --gpu-shard 80/88 .. 87/88             -> shards 80..87
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-80}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Prefer a claim pool over static shards when every task can see one shared
# directory (true within ACCRE). Static shards strand GPUs: a task that
# finishes its slice exits while other slices still have hundreds of ensembles.
# CLAIM_DIR="" falls back to the old --gpu-shard behaviour.
CLAIM_DIR="${CLAIM_DIR:-}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "dataset=${DATASET} tool=${TOOL} run_id=${RUN_ID} msa=${MSA_DATASET} samples=${NUM_SAMPLES}"
echo "fractions=${MSA_FRACTIONS:-<default>} claim_dir=${CLAIM_DIR:-<none, shard ${SHARD}/${NSHARDS}>}"

# ACCRE's system python3 is 3.9 with no pip; gemmi (for AF3 CIF -> PDB) is
# installed under /data for the conda 3.13 interpreter instead.
PYTHON="${PYTHON:-python3}"

ARGS=(
  "data/inputs/${DATASET}"
  "${RUN_ID}"
  --msa-dir "data/msas/${MSA_DATASET}"
  --num-samples "${NUM_SAMPLES}"
  --tool "${TOOL}"
  --resume
)
if [[ -n "$CLAIM_DIR" ]]; then
  mkdir -p "$CLAIM_DIR"
  ARGS+=(--claim-dir "$CLAIM_DIR")
else
  ARGS+=(--gpu-shard "${SHARD}/${NSHARDS}")
fi
# Unquoted on purpose: MSA_FRACTIONS is a space-separated list for nargs="+".
[[ -n "$MSA_FRACTIONS" ]] && ARGS+=(--msa-fractions ${MSA_FRACTIONS})

exec "$PYTHON" pipeline/02_compute_predictions/run_afsample3.py "${ARGS[@]}"
