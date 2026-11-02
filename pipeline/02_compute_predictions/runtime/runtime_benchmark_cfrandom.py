#!/usr/bin/env python3
"""
CFrandom runtime benchmark.
Calls colabfold_batch with pre-computed MSAs, timing only the prediction step.

CFrandom generates conformations by running ColabFold with randomized shallow
MSA subsampling. Each call to colabfold_batch with --num-seeds S produces
S * num_models conformations. We use --num-models 5 (default).

For N target samples: --num-seeds ceil(N/5).

Writes per-run timings to a CSV file.
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COLABFOLD_BIN = Path("/home/vanderw/benchmarking/cfrandom/localcolabfold/colabfold-conda/bin/colabfold_batch")
CFRANDOM_BASE = Path("/home/vanderw/benchmarking/cfrandom")
SAMPLE_COUNTS = [20, 40, 60, 100]
NUM_MODELS = 5
# Use shallow random MSA (max_seq=8, max_extra_seq=16) — this is the core
# CFrandom mode that generates diverse conformations
MAX_SEQ = 8
MAX_EXTRA_SEQ = 16
MODEL_TYPE = "alphafold2_ptm"


def run_colabfold(msa_dir: Path, output_dir: Path, num_seeds: int, random_seed: int = 42) -> float:
    """Run colabfold_batch and return wall-clock time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLABFOLD_BIN),
        "--num-seeds", str(num_seeds),
        "--num-models", str(NUM_MODELS),
        "--model-type", MODEL_TYPE,
        "--random-seed", str(random_seed),
        "--max-seq", str(MAX_SEQ),
        "--max-extra-seq", str(MAX_EXTRA_SEQ),
        str(msa_dir),
        str(output_dir),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-500:]}", flush=True)
        return float("nan")

    return elapsed


def main():
    config_path = Path(sys.argv[1])
    results_csv = Path(sys.argv[2])

    with open(config_path) as f:
        config = json.load(f)
    proteins = config["proteins"]

    # GPU info
    gpu_info = "unknown"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            gpu_info = out.stdout.strip()
    except Exception:
        pass

    # Warmup: run one small prediction to load models into GPU cache
    warmup_protein = proteins[0]
    warmup_msa = CFRANDOM_BASE / warmup_protein["uniprot"] / f"{warmup_protein['uniprot']}-search"
    warmup_out = Path(tempfile.mkdtemp(prefix="cfrandom_warmup_", dir=str(Path.home())))
    print(f"Warmup: {warmup_protein['uniprot']} ({warmup_protein['length']}aa), 1 seed...", flush=True)
    run_colabfold(warmup_msa, warmup_out, num_seeds=1)
    shutil.rmtree(warmup_out, ignore_errors=True)
    print("Warmup done.\n", flush=True)

    total_runs = len(proteins) * len(SAMPLE_COUNTS)
    run_idx = 0

    write_header = not results_csv.exists()
    fieldnames = ["tool", "uniprot", "sequence_length", "size_bin",
                  "num_samples", "wall_clock_seconds", "gpu", "timestamp"]

    for p in proteins:
        uid = p["uniprot"]
        seq_len = p["length"]
        size_bin = p["size_bin"]
        msa_dir = CFRANDOM_BASE / uid / f"{uid}-search"

        for n_samples in SAMPLE_COUNTS:
            run_idx += 1
            num_seeds = math.ceil(n_samples / NUM_MODELS)
            actual_samples = num_seeds * NUM_MODELS

            print(f"  [{run_idx}/{total_runs}] {uid} ({seq_len}aa, {size_bin}) | "
                  f"n={n_samples} (seeds={num_seeds}, actual={actual_samples}) ... ",
                  end="", flush=True)

            # Use a temp output dir (clean up after)
            out_dir = Path(tempfile.mkdtemp(
                prefix=f"cfrandom_{uid}_n{n_samples}_",
                dir=str(Path.home())))

            elapsed = run_colabfold(msa_dir, out_dir, num_seeds=num_seeds)

            # Clean up output PDBs
            shutil.rmtree(out_dir, ignore_errors=True)

            if math.isnan(elapsed):
                print("FAILED", flush=True)
            else:
                print(f"{elapsed:.2f}s", flush=True)

            row = {
                "tool": "cfrandom",
                "uniprot": uid,
                "sequence_length": seq_len,
                "size_bin": size_bin,
                "num_samples": n_samples,
                "wall_clock_seconds": round(elapsed, 3) if not math.isnan(elapsed) else "",
                "gpu": gpu_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)


if __name__ == "__main__":
    main()
