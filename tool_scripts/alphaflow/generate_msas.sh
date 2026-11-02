#!/bin/bash
# Generate MSAs for AlphaFlow-PDB using ColabFold MMseqs2 API.
#
# For UNICORNE: symlink from existing cfrandom MSAs.
# For DANCE and CFOLD: run AlphaFlow's mmseqs_query.py.
#
# Usage:
#   bash tool_scripts/alphaflow/generate_msas.sh UNICORNE
#   bash tool_scripts/alphaflow/generate_msas.sh DANCE
#   bash tool_scripts/alphaflow/generate_msas.sh CFOLD

set -euo pipefail

DATASET="${1:?Usage: $0 <DANCE|CFOLD|UNICORNE>}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ALPHAFLOW_DIR="$HOME/alphaflow"
ALPHAFLOW_PYTHON="$HOME/alphaflow_venv/bin/python"
MSA_BASE="$PROJECT_ROOT/data/msas"
MSA_DIR="$MSA_BASE/$DATASET"
CFRANDOM_BASE="/home/vanderw/benchmarking/cfrandom"

mkdir -p "$MSA_DIR"

# Determine input CSV
INPUT_CSV="$PROJECT_ROOT/data/inputs/$DATASET/input_dataset_filtered.csv"
if [ ! -f "$INPUT_CSV" ]; then
    INPUT_CSV="$PROJECT_ROOT/data/inputs/$DATASET/input_dataset.csv"
fi
echo "Dataset: $DATASET"
echo "Input CSV: $INPUT_CSV"
echo "MSA output: $MSA_DIR"

if [ "$DATASET" = "UNICORNE" ]; then
    echo ""
    echo "=== Symlinking UNICORNE MSAs from cfrandom ==="
    # UNICORNE uses uniprot IDs as ensemble names; cfrandom has MSAs for these
    # AlphaFlow expects: {msa_dir}/{name}/a3m/{name}.a3m
    # cfrandom has: {cfrandom_base}/{uid}/{uid}-search/{uid}.a3m

    PYTHON_CMD='
import csv, os, sys

csv_path = sys.argv[1]
msa_dir = sys.argv[2]
cfrandom_base = sys.argv[3]

with open(csv_path) as f:
    reader = csv.DictReader(f)
    linked = 0
    missing = 0
    for row in reader:
        uid = (row.get("uniprot") or row.get("ensemble_ID", "")).strip()
        if not uid:
            continue
        src = os.path.join(cfrandom_base, uid, f"{uid}-search", f"{uid}.a3m")
        dst_dir = os.path.join(msa_dir, uid, "a3m")
        dst = os.path.join(dst_dir, f"{uid}.a3m")
        if os.path.exists(dst):
            linked += 1
            continue
        if not os.path.exists(src):
            print(f"  MISSING: {uid} (no MSA at {src})")
            missing += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        os.symlink(src, dst)
        linked += 1
    print(f"Linked: {linked}, Missing: {missing}")
'
    python3 -c "$PYTHON_CMD" "$INPUT_CSV" "$MSA_DIR" "$CFRANDOM_BASE"

else
    echo ""
    echo "=== Generating MSAs via ColabFold API ==="
    echo "This may take a while for large datasets..."

    # AlphaFlow's mmseqs_query.py expects a CSV with 'name' and 'seqres' columns
    # Our CSVs use 'ensemble_ID'/'uniprot' and 'ref_sequence'/'sequence'
    # Create a temp CSV in the right format
    TMPCSV=$(mktemp "$HOME/alphaflow_msa_input_XXXXXX.csv")

    python3 -c "
import csv, sys
with open(sys.argv[1]) as fin, open(sys.argv[2], 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=['name', 'seqres'])
    writer.writeheader()
    for row in reader:
        name = (row.get('ensemble_ID') or row.get('uniprot', '')).strip()
        seq = (row.get('ref_sequence') or row.get('sequence', '')).strip()
        if name and seq:
            # Skip if MSA already exists
            import os
            if os.path.exists(os.path.join(sys.argv[3], name, 'a3m', name + '.a3m')):
                continue
            writer.writerow({'name': name, 'seqres': seq})
" "$INPUT_CSV" "$TMPCSV" "$MSA_DIR"

    N_REMAINING=$(tail -n +2 "$TMPCSV" | wc -l)
    echo "Proteins needing MSAs: $N_REMAINING"

    if [ "$N_REMAINING" -gt 0 ]; then
        cd "$ALPHAFLOW_DIR"
        PYTHONNOUSERSITE=1 "$ALPHAFLOW_PYTHON" -m scripts.mmseqs_query \
            --split "$TMPCSV" \
            --outdir "$MSA_DIR"
    else
        echo "All MSAs already exist. Nothing to do."
    fi

    rm -f "$TMPCSV"
fi

echo ""
echo "=== Done ==="
echo "MSA directory: $MSA_DIR"
echo "Total MSA dirs: $(ls -d "$MSA_DIR"/*/a3m 2>/dev/null | wc -l)"
