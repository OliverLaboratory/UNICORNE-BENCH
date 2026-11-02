#!/usr/bin/env python3
"""
Inference runtime benchmark for BioEMU, AlphaFlow, and ESMDiff.

Measures wall-clock time to generate [20, 40, 60, 100] conformers for
30 UNICORNE proteins across three size bins (~60aa, ~150aa, ~300aa).

Each tool runs in its own environment with the model loaded ONCE.
Only inference time is measured (no model loading, no I/O).

Usage:
  python scripts/runtime_benchmark.py --tool bioemu
  python scripts/runtime_benchmark.py --tool alphaflow
  python scripts/runtime_benchmark.py --tool esmdiff
  python scripts/runtime_benchmark.py --tool all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Protein selection: 10 per size bin from UNICORNE
# ---------------------------------------------------------------------------
PROTEINS = {
    "small": [
        "P01468", "P17726", "P81658", "P01426", "P01427",
        "P56790", "P0A7M6", "P81782", "P61991", "P73213",
    ],
    "medium": [
        "Q9V0G8", "P0A8A8", "E9AD06", "Q5FKC3", "P52434",
        "P59220", "Q57221", "P75540", "P00641", "P0DP25",
    ],
    "large": [
        "D3WAD3", "O41026", "O69061", "O18423", "P02925",
        "P53859", "O29280", "P9WFZ9", "Q4X158", "Q9JPG4",
    ],
}

SAMPLE_COUNTS = [20, 40, 60, 100]


def load_unicorne_sequences() -> dict[str, str]:
    csv_path = PROJECT_ROOT / "data" / "inputs" / "UNICORNE" / "input_dataset.csv"
    seqs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            uid = row.get("uniprot", "").strip()
            seq = row.get("sequence", "").strip()
            if uid and seq:
                seqs[uid] = seq
    return seqs


def get_benchmark_proteins() -> list[dict]:
    seqs = load_unicorne_sequences()
    proteins = []
    for size_bin, uniprots in PROTEINS.items():
        for uid in uniprots:
            if uid not in seqs:
                print(f"WARNING: {uid} not found in UNICORNE dataset, skipping")
                continue
            proteins.append({
                "uniprot": uid,
                "sequence": seqs[uid],
                "length": len(seqs[uid]),
                "size_bin": size_bin,
            })
    return proteins


def run_tool_benchmark(tool: str, proteins: list[dict], results_csv: Path) -> None:
    """Launch the per-tool benchmark script in its own environment."""

    # Write protein config JSON
    config_path = (PROJECT_ROOT / "results" / f"_benchmark_config_{tool}.json").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"proteins": proteins, "sample_counts": SAMPLE_COUNTS}, f, indent=2)

    if tool == "bioemu":
        python_exe = str(Path.home() / "bioemu_venv" / "bin" / "python")
        script = str(SCRIPTS_DIR / "runtime_benchmark_bioemu.py")
        cmd = [python_exe, script, str(config_path), str(results_csv)]
        cwd = str(PROJECT_ROOT)

    elif tool == "esmflow_pdb":
        python_exe = str(Path.home() / "alphaflow_venv" / "bin" / "python")
        script = str(SCRIPTS_DIR / "runtime_benchmark_alphaflow.py")
        cmd = [python_exe, script, str(config_path), str(results_csv)]
        cwd = str(Path.home() / "alphaflow")

    elif tool == "esmdiff":
        script = str(SCRIPTS_DIR / "runtime_benchmark_esmdiff.py")
        cmd = ["pixi", "run", "python", script, str(config_path), str(results_csv)]
        cwd = str(Path.home() / "esmdiff")

    elif tool == "cfrandom":
        # CFrandom uses ColabFold from vanderw's installation
        colabfold_python = str(Path("/home/vanderw/benchmarking/cfrandom/localcolabfold/colabfold-conda/bin/python"))
        script = str(SCRIPTS_DIR / "runtime_benchmark_cfrandom.py")
        cmd = [colabfold_python, script, str(config_path), str(results_csv)]
        cwd = str(Path("/home/vanderw/benchmarking/cfrandom"))

    else:
        raise ValueError(f"Unknown tool: {tool}")

    print(f"\n{'='*60}")
    print(f"Running {tool.upper()} benchmark")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  CWD: {cwd}")
    print(f"  Results: {results_csv}")
    print(f"{'='*60}\n", flush=True)

    env = os.environ.copy()
    if tool == "esmflow_pdb":
        env["PYTHONNOUSERSITE"] = "1"
    elif tool == "esmdiff":
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(cmd, cwd=cwd, env=env)

    if result.returncode != 0:
        print(f"\n{tool.upper()} benchmark FAILED (exit code {result.returncode})")
    else:
        print(f"\n{tool.upper()} benchmark completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Inference runtime benchmark")
    parser.add_argument("--tool", type=str, choices=["bioemu", "esmflow_pdb", "esmdiff", "cfrandom", "all"], required=True)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "runtime_benchmark.csv")
    args = parser.parse_args()

    args.results_csv = args.results_csv.resolve()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    proteins = get_benchmark_proteins()
    print(f"Benchmark: {len(proteins)} proteins")
    for size_bin in ["small", "medium", "large"]:
        bp = [p for p in proteins if p["size_bin"] == size_bin]
        print(f"  {size_bin}: {len(bp)} proteins, {min(p['length'] for p in bp)}-{max(p['length'] for p in bp)}aa")
    print(f"Sample counts: {SAMPLE_COUNTS}")
    print(f"Results: {args.results_csv}\n")

    tools = ["bioemu", "esmflow_pdb", "esmdiff", "cfrandom"] if args.tool == "all" else [args.tool]

    for tool in tools:
        run_tool_benchmark(tool, proteins, args.results_csv)

    print(f"\nAll done. Results: {args.results_csv}")


if __name__ == "__main__":
    main()
