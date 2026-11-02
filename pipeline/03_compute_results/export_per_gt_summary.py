#!/usr/bin/env python3
"""
Export the per-GT summary that lets the figure scripts run without the
full GT x prediction TM-align data.

For every (ensemble, GT conformation) it stores the three quantities the
paper's figures and tables read from the full matrices:

  best_tm_chain1  TM (normalized by GT length) of the best prediction, chosen
                  by that same TM (NaN treated as 0, as in the figure scripts)
  rmsd_at_best    RMSD of that prediction
  max_tm_max2     max over predictions of max(TM_chain1, TM_chain2), skipping
                  pairs where either direction is missing

``tmalign_io.load_xy_pair`` falls back to this file when neither the raw
TM-align JSONs nor the legacy npy matrices are present.

Input:   <src-root>/results/<DATASET>/<TOOL>/{tmalign_raw,matrices_tmalign}/
Output:  results/<DATASET>/<TOOL>/per_gt_summary.csv

Usage:
  python pipeline/03_compute_results/export_per_gt_summary.py UNICORNE_BENCH bioemu
  python pipeline/03_compute_results/export_per_gt_summary.py OOD60 cfrandom --src-root /path/to/full/checkout
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tmalign_io  # noqa: E402

FIELDS = ["ensemble_ID", "gt_index", "best_tm_chain1", "rmsd_at_best", "max_tm_max2"]


def summarize(args: tuple[str, str, str]) -> list[dict]:
    ds, tool, eid = args
    out = tmalign_io.load_xy_pair(ds, tool, eid)
    if out is None:
        return []
    tm1, tm2, rmsd = out
    tm1_filled = np.where(np.isnan(tm1), 0.0, tm1)
    both = ~(np.isnan(tm1) | np.isnan(tm2))
    rows = []
    for gi in range(tm1.shape[0]):
        bi = int(tm1_filled[gi].argmax())
        valid = both[gi]
        rows.append({
            "ensemble_ID": eid,
            "gt_index": gi,
            "best_tm_chain1": repr(float(tm1_filled[gi, bi])),
            "rmsd_at_best": repr(float(rmsd[gi, bi])),
            "max_tm_max2": repr(float(np.maximum(tm1[gi], tm2[gi])[valid].max()))
                           if valid.any() else "nan",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tool")
    ap.add_argument("--src-root", type=Path, default=PROJECT_ROOT,
                    help="Checkout holding the full results/ tree (default: this repo)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tmalign_io.PROJECT_ROOT = args.src_root.resolve()
    eids = tmalign_io.list_available_eids(args.dataset, args.tool)
    with Pool(args.workers) as pool:
        chunks = pool.map(summarize, [(args.dataset, args.tool, e) for e in eids])

    out = PROJECT_ROOT / "results" / args.dataset / args.tool / "per_gt_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rows in chunks:
            w.writerows(rows)
    print(f"{args.dataset}/{args.tool}: {len(eids)} ensembles -> {out}")


if __name__ == "__main__":
    main()
