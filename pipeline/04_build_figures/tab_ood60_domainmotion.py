#!/usr/bin/env python3
"""
Tables: OOD60 and DOMAINMOTION per-protein results.

Generates LaTeX tables showing RMSD(TM) of the best-TM prediction per GT
conformation for each tool. Used in the paper main text (simplified, BioEMU +
CFrandom) and appendix (all tools).

Input:
  results/{OOD60,DOMAINMOTION}/{TOOL}/matrices_tmalign/{eid}/tm_xy.npy, rmsd_xy_ab.npy
  results/{OOD60,DOMAINMOTION}/gt_matrices_tmalign/{eid}/tm_xx.npy

Output:
  figures/tab_ood60_simplified.tex
  figures/tab_ood60_full.tex
  figures/tab_domainmotion_full.tex

Usage:
  python pipeline/04_build_figures/tab_ood60_domainmotion.py
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_ALL = ["bioemu", "esmflow_pdb", "alphaflow_pdb", "esmdiff", "cfrandom"]
TOOL_DISPLAY = {"bioemu":"BioEMU","esmflow_pdb":"ESMFlow","alphaflow_pdb":"AlphaFlow","esmdiff":"ESMDiff","cfrandom":"CFrandom"}

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "pipeline" / "03_compute_results"))
from tmalign_io import load_xy_pair  # type: ignore


def get_data(ds, tool, eid):
    """Load best-pred TM/RMSD per GT row. Reads new tmalign_raw JSONs first,
    falls back to legacy matrices_tmalign npy."""
    out = load_xy_pair(ds, tool, eid)
    if out is None: return None
    tm, _tm2, rmsd = out
    if tm is None or rmsd is None: return None
    tm = np.where(np.isnan(tm), 0.0, tm)
    result = {"tm":[], "rmsd":[], "n_gt": tm.shape[0]}
    for gi in range(tm.shape[0]):
        bi = int(tm[gi].argmax())
        if np.isnan(rmsd[gi, bi]): continue
        result["tm"].append(float(tm[gi, bi]))
        result["rmsd"].append(float(rmsd[gi, bi]))
    return result

def get_gt_tm(ds, eid):
    for name in ["tm_xx.npy","tm_xx_ab.npy"]:
        f = PROJECT_ROOT / "results" / ds / "gt_matrices_tmalign" / eid / name
        if f.exists():
            mat = np.load(f)
            if mat.shape[0] > 1:
                mask = ~np.eye(mat.shape[0], dtype=bool)
                return float(mat[mask].min())
    return None

def fmt_pair(d):
    if d is None or len(d["tm"]) < 2:
        if d and len(d["tm"]) == 1: return f"{d['rmsd'][0]:.1f} ({d['tm'][0]:.2f}) & --"
        return "-- & --"
    r0,t0 = d["rmsd"][0],d["tm"][0]; r1,t1 = d["rmsd"][1],d["tm"][1]
    c0 = f"{r0:.1f} ({t0:.2f})"; c1 = f"{r1:.1f} ({t1:.2f})"
    mn = min(t0, t1)
    if mn >= 0.9: return f"\\cellcolor{{lightgreen}}{c0} & \\cellcolor{{lightgreen}}{c1}"
    elif mn >= 0.85: return f"\\cellcolor{{lightyellow}}{c0} & \\cellcolor{{lightyellow}}{c1}"
    return f"{c0} & {c1}"

def build_table(ds, tools):
    seq_lens = {}
    with open(PROJECT_ROOT / "data" / "inputs" / ds / "input_dataset.csv") as f:
        for row in csv.DictReader(f):
            eid = (row.get("ensemble_ID") or row.get("uniprot","")).strip()
            seq_lens[eid] = len(row.get("ref_sequence") or row.get("sequence",""))
    rows = []
    for eid in sorted(seq_lens):
        sl = seq_lens[eid]
        gt = get_gt_tm(ds, eid)
        pts = f"{gt:.2f}" if gt else "--"
        parts = [f"{eid} & {sl} & {pts}"]
        for tool in tools:
            d = get_data(ds, tool, eid)
            parts.append(fmt_pair(d))
        rows.append(" & ".join(parts) + " \\\\")
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ds in ["OOD60", "DOMAINMOTION"]:
        rows = build_table(ds, TOOLS_ALL)
        out = args.output_dir / f"tab_{ds.lower()}_full.tex"
        with open(out, "w") as f:
            for r in rows: f.write(r + "\n")
        print(f"Wrote {out} ({len(rows)} rows)")

    # Simplified OOD60 (BioEMU + CFrandom only)
    rows = build_table("OOD60", ["bioemu", "cfrandom"])
    out = args.output_dir / "tab_ood60_simplified.tex"
    with open(out, "w") as f:
        for r in rows: f.write(r + "\n")
    print(f"Wrote {out}")

if __name__ == "__main__": main()
