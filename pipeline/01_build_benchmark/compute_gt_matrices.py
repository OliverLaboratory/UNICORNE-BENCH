#!/usr/bin/env python3
"""
Compute GT pairwise TM-align metrics + dump full raw output as JSON.

Mirrors the GT × pred protocol in `pipeline/03_compute_results/
compute_tmalign_raw.py`: a single TMalign call per pair (gt_i, gt_j),
storing the raw stdout, parsed scores, alignment strings, residue–residue
mappings, and the rotation/translation matrix. Headline TM-score is
`tm_max2 = max(tm_norm_chain1, tm_norm_chain2)` — the min-length-
normalized convention used throughout the benchmark.

Output (per ensemble, under results/<DATASET>/gt_matrices_tmalign/<eid>/):
  - tmalign_raw.json   canonical record (same schema as compute_tmalign_raw.py),
                       with `pairs` enumerating only upper-triangle (i < j) entries.
  - tm_xx.npy          (n × n) symmetric matrix of `tm_max2`, diagonal = 1.
                       Kept for back-compat with the filter pipeline.

Usage:
  TMALIGN_EXE=~/bin/TMalign python pipeline/01_build_benchmark/compute_gt_matrices.py UNICORNE_BENCH
  python pipeline/01_build_benchmark/compute_gt_matrices.py DANCE --workers 16 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline" / "03_compute_results"))

from conformation_benchmark.datasets.folder_dataset import _parse_structure_ca
from compute_tmalign_raw import run_tmalign_one as _gt_pred_run_tmalign


def load_sequences(csv_path: Path) -> dict[str, str]:
    seqs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = (row.get("ensemble_ID") or row.get("uniprot", "")).strip()
            seq = (row.get("ref_sequence") or row.get("sequence", "")).strip()
            if eid and seq:
                seqs[eid] = seq
    return seqs


def _get_tmalign_exe() -> str:
    return os.environ.get("TMALIGN_EXE") or os.environ.get("TMALIGN") or "TMalign"


def process_ensemble(dataset, eid, gt_pdbs_dir, seq, out_dir, n_workers, timeout=120):
    """Compute all pairwise GT TM-align records (single direction) for one
    ensemble, dump JSON in the same schema as compute_tmalign_raw.py."""
    eid_dir = gt_pdbs_dir / eid
    if not eid_dir.is_dir():
        return None

    pdb_files = sorted(list(eid_dir.glob("*.pdb")) + list(eid_dir.glob("*.cif")))
    if len(pdb_files) < 2:
        return None

    exe = _get_tmalign_exe()
    pairs_idx = [(i, j) for i in range(len(pdb_files)) for j in range(i + 1, len(pdb_files))]

    # Use the same single-direction TMalign+parser as compute_tmalign_raw.py.
    # We pass file_i as "gt" and file_j as "pred"; downstream consumers should
    # treat both as GT. The fields stay identical (gt/pred, len_chain1/2,
    # tm_norm_chain1/2, mapping_*, rotation, raw, tm_max2, etc.).
    def _work(pair_idx):
        i, j = pair_idx
        rec = _gt_pred_run_tmalign((str(pdb_files[i]), str(pdb_files[j]), exe, timeout))
        # Add explicit pair indices into the record so downstream can rebuild
        # the n × n matrix without parsing filename strings.
        rec["i"] = i
        rec["j"] = j
        rec["a"] = pdb_files[i].name
        rec["b"] = pdb_files[j].name
        return rec

    t0 = time.perf_counter()
    workers = min(n_workers, len(pairs_idx))
    if workers > 1 and len(pairs_idx) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_work, pairs_idx))
    else:
        records = [_work(p) for p in pairs_idx]
    elapsed = time.perf_counter() - t0

    # Symmetric tm_max2 matrix (back-compat for the filter pipeline)
    n = len(pdb_files)
    tm_max2 = np.eye(n)
    for r in records:
        v = r.get("tm_max2")
        if v is None:
            continue
        i, j = r["i"], r["j"]
        tm_max2[i, j] = tm_max2[j, i] = float(v)

    eid_out = out_dir / eid
    eid_out.mkdir(parents=True, exist_ok=True)
    np.save(eid_out / "tm_xx.npy", tm_max2)

    payload = {
        "ensemble_id": eid,
        "n_gt": n,
        "n_pred": n,                 # symmetric: chain set is the GT itself
        "elapsed_s": round(elapsed, 2),
        "tmalign_exe": exe,
        "conformation_names": [p.name for p in pdb_files],
        "pairs": records,            # upper-triangle only (i < j)
    }
    with open(eid_out / "tmalign_raw.json", "w") as f:
        json.dump(payload, f)

    return n, len(pairs_idx)


def main():
    parser = argparse.ArgumentParser(description="Compute GT pairwise TM matrices")
    parser.add_argument("dataset", help="Dataset name (e.g. DANCE, WILL_TM)")
    parser.add_argument("--workers", type=int, default=8, help="TM-align threads per ensemble")
    parser.add_argument("--resume", action="store_true", help="Skip ensembles with existing tm_xx.npy")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N ensembles")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N ensembles")
    args = parser.parse_args()

    ds = args.dataset
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    gt_pdbs_dir = data_dir / "inputs" / ds / "input_pdbs"
    out_dir = results_dir / ds / "gt_matrices_tmalign"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(data_dir / "inputs" / ds / "input_dataset.csv")
    all_eids = sorted(d.name for d in gt_pdbs_dir.iterdir() if d.is_dir())

    # Apply offset/limit
    if args.offset:
        all_eids = all_eids[args.offset:]
    if args.limit:
        all_eids = all_eids[:args.limit]

    # Resume: skip already computed
    if args.resume:
        done = set(d.name for d in out_dir.iterdir()
                   if d.is_dir() and (d / "tm_xx.npy").exists())
        todo = [e for e in all_eids if e not in done]
        print(f"Resume: {len(done)} already done, {len(todo)} remaining")
    else:
        todo = all_eids

    print(f"Computing GT matrices for {ds}: {len(todo)} ensembles, {args.workers} workers")
    print(f"Output: {out_dir}")
    print()

    t0 = time.time()
    total_pairs = 0
    for idx, eid in enumerate(todo):
        seq = sequences.get(eid, "")
        result = process_ensemble(ds, eid, gt_pdbs_dir, seq, out_dir, args.workers)
        if result:
            n_confs, n_pairs = result
            total_pairs += n_pairs
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{len(todo)}] {eid}: {result[0] if result else 0} confs, "
              f"{result[1] if result else 0} pairs | "
              f"{rate:.1f} ens/s, ~{remaining/60:.0f}min left",
              end="\r", flush=True)

    print(f"\nDone: {len(todo)} ensembles, {total_pairs} total pairs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
