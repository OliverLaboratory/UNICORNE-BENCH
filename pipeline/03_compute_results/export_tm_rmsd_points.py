#!/usr/bin/env python3
"""
Export the data behind the TM vs. RMSD calibration figure (Fig. 2) and the
OOD60 / DOMAINMOTION tables (Tables 10-11).

Two CSVs, both read from the full GT x prediction TM-align output:

  tm_rmsd_scatter_points.csv  one row per plotted point: the best prediction
                              (by TM normalized by GT length, NaN treated as
                              0, as in fig_tm_vs_rmsd_scatter.py) for each GT
                              conformation, with its TM and RMSD
  tm_rmsd_sample_counts.csv   number of GT conformations and scored
                              predictions per (dataset, tool, ensemble)

The sample counts are only recoverable from the full output: per_gt_summary.csv
keeps the best prediction per GT but not how many predictions there were.

Input:   <src-root>/results/{OOD60,DOMAINMOTION}/<TOOL>/{tmalign_raw,matrices_tmalign}/
         data/inputs/{OOD60,DOMAINMOTION}/input_dataset.csv
Output:  results/tm_rmsd/{tm_rmsd_scatter_points,tm_rmsd_sample_counts}.csv

Usage:
  python pipeline/03_compute_results/export_tm_rmsd_points.py --src-root /path/to/full/checkout
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tmalign_io  # noqa: E402

DATASETS = ["OOD60", "DOMAINMOTION"]
TOOLS = ["bioemu", "esmflow_pdb", "alphaflow_pdb", "esmdiff", "cfrandom"]


def seq_lengths(ds: str) -> dict[str, int]:
    out = {}
    with open(PROJECT_ROOT / "data" / "inputs" / ds / "input_dataset.csv") as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or row.get("uniprot", "")).strip()
            out[eid] = len((row.get("ref_sequence") or row.get("sequence", "")).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=PROJECT_ROOT,
                    help="Checkout holding the full results/ tree (default: this repo)")
    args = ap.parse_args()
    tmalign_io.PROJECT_ROOT = args.src_root.resolve()

    points, counts = [], []
    for ds in DATASETS:
        lens = seq_lengths(ds)
        for tool in TOOLS:
            for eid in tmalign_io.list_available_eids(ds, tool):
                if lens.get(eid, 0) == 0:
                    continue
                out = tmalign_io.load_xy_pair(ds, tool, eid)
                if out is None or out[0] is None or out[2] is None:
                    continue
                tm, _tm2, rmsd = out
                counts.append([ds, tool, eid, tm.shape[0], tm.shape[1]])
                tm_filled = np.where(np.isnan(tm), 0.0, tm)
                for gi in range(tm.shape[0]):
                    bi = int(tm_filled[gi].argmax())
                    if np.isnan(rmsd[gi, bi]):
                        continue
                    points.append([ds, tool, eid, lens[eid], gi, bi,
                                   round(float(tm_filled[gi, bi]), 4), round(float(rmsd[gi, bi]), 3)])

    if counts and all(n == 1 for *_, n in counts):
        sys.exit("only per-GT summaries found; pass --src-root pointing at the full TM-align output")

    out_dir = PROJECT_ROOT / "results" / "tm_rmsd"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tm_rmsd_scatter_points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "tool", "ensemble_ID", "length", "gt_index", "pred_index", "tm", "rmsd"])
        w.writerows(points)
    with open(out_dir / "tm_rmsd_sample_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "tool", "ensemble_ID", "n_gt", "n_pred"])
        w.writerows(counts)

    n = sorted(c[-1] for c in counts)
    print(f"{len(points)} points; {sum(n)} predictions, "
          f"{n[0]}-{n[-1]} per ensemble and tool (median {n[len(n) // 2]}) -> {out_dir}")


if __name__ == "__main__":
    main()
