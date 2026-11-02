#!/usr/bin/env python3
"""Generate MSAs locally using colabfold_search + GPU mmseqs2.

Much faster and more reliable than the ColabFold API.

Usage:
    python -u scripts/generate_msas_local.py --dataset DANCE
    python -u scripts/generate_msas_local.py --dataset DANCE --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
COLABFOLD_SEARCH = Path.home() / "localcolabfold/.pixi/envs/default/bin/colabfold_search"
MMSEQS = Path.home() / "localcolabfold/.pixi/envs/default/bin/mmseqs"
DB_DIR = Path("/aquinas/os/oliverlab/gonzc11_colabfold_databases")


def load_missing(dataset: str) -> list[dict]:
    csv_path = PROJECT / "data/inputs" / dataset / "input_dataset_filtered.csv"
    if not csv_path.exists():
        csv_path = csv_path.with_name("input_dataset.csv")
    msa_dir = PROJECT / "data/msas" / dataset
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = row["ensemble_ID"]
            seq = row["ref_sequence"]
            a3m = msa_dir / eid / "a3m" / f"{eid}.a3m"
            if not a3m.exists():
                rows.append({"ensemble_ID": eid, "ref_sequence": seq})
    return rows


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DANCE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Sequences per colabfold_search batch")
    args = parser.parse_args()

    msa_dir = PROJECT / "data/msas" / args.dataset
    work_dir = msa_dir / "_local_search"
    work_dir.mkdir(parents=True, exist_ok=True)

    missing = load_missing(args.dataset)
    log(f"Dataset: {args.dataset}")
    log(f"Missing MSAs: {len(missing)}")
    log(f"DB dir: {DB_DIR}")
    log(f"colabfold_search: {COLABFOLD_SEARCH}")
    log(f"mmseqs: {MMSEQS}")
    log(f"Batch size: {args.batch_size}")

    if args.dry_run:
        log("Dry run, exiting.")
        return

    if len(missing) == 0:
        log("Nothing to do.")
        return

    n_batches = (len(missing) + args.batch_size - 1) // args.batch_size
    total_saved = 0
    t_start = time.time()

    for b in range(n_batches):
        batch = missing[b * args.batch_size : (b + 1) * args.batch_size]
        log(f"[Batch {b+1}/{n_batches}] {len(batch)} sequences")

        # Write FASTA for this batch
        fasta_path = work_dir / f"batch_{b}.fasta"
        with open(fasta_path, "w") as f:
            for row in batch:
                f.write(f">{row['ensemble_ID']}\n{row['ref_sequence']}\n")

        # Output directory for this batch
        result_dir = work_dir / f"batch_{b}_results"
        result_dir.mkdir(exist_ok=True)

        # Run colabfold_search (GPU MMseqs2 — needs CUDA_VISIBLE_DEVICES set)
        # --db-load-mode 2 = mmap (no full-DB RAM load; uniref30 is ~134 GB,
        # system has 61 GB RAM, so mode 0/auto OOMs in result2msa).
        cmd = [
            str(COLABFOLD_SEARCH),
            "--mmseqs", str(MMSEQS),
            "--gpu", "1",
            "--threads", "16",
            "--db-load-mode", "2",
            str(fasta_path),
            str(DB_DIR),
            str(result_dir),
        ]
        log(f"  Running: {' '.join(cmd[:3])} ... {cmd[-1]}")
        t0 = time.time()

        result = subprocess.run(cmd, capture_output=False, text=True)

        elapsed = time.time() - t0
        log(f"  colabfold_search took {elapsed:.0f}s (exit code {result.returncode})")

        if result.returncode != 0:
            log(f"  ERROR: colabfold_search failed, skipping batch")
            continue

        # Move .a3m files to the MSA directory structure
        # colabfold_search outputs: {result_dir}/{name}.a3m
        saved = 0
        for row in batch:
            eid = row["ensemble_ID"]
            src = result_dir / f"{eid}.a3m"
            if not src.exists():
                # Also check 0.a3m pattern
                src = result_dir / "0.a3m"
            if not src.exists():
                log(f"  Warning: no a3m for {eid}")
                continue

            dst_dir = msa_dir / eid / "a3m"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{eid}.a3m"
            shutil.copy2(src, dst)
            saved += 1

        total_saved += saved
        log(f"  Saved {saved}/{len(batch)} MSAs (total: {total_saved}/{len(missing)})")

        # Cleanup batch work files
        shutil.rmtree(result_dir, ignore_errors=True)
        fasta_path.unlink(missing_ok=True)

    elapsed_total = time.time() - t_start
    log(f"\nDone. Saved {total_saved}/{len(missing)} MSAs in {elapsed_total:.0f}s")
    log(f"MSA dir: {msa_dir}")


if __name__ == "__main__":
    main()
