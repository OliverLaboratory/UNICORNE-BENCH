#!/usr/bin/env python3
"""
Materialize legacy ``matrices_tmalign/<eid>/*.npy`` files from the new
``tmalign_raw/<eid>.json`` artifacts.

The JSON store from ``compute_tmalign_raw.py`` is the canonical TM-align
record going forward. Some 04_build_figures scripts still expect the old
per-ensemble npy layout (``tm_xy.npy``, ``tm_xy_ab.npy``, ``rmsd_xy_ab.npy``)
because they share infrastructure (``collect_ensemble_data``,
``generate_pr_curves.py``) that hasn't been ported. This script bridges
the two: read every JSON in ``tmalign_raw/`` and write the equivalent
npy files into ``matrices_tmalign/`` so figure scripts run unchanged.

Convention used by the legacy scripts:
  tm_xy.npy        TM normalized by chain1 (GT length, paper convention)
  tm_xy_ab.npy     same as tm_xy.npy (alias kept for older callers)
  tm_xy_ba.npy     TM normalized by chain2 (pred length)
  rmsd_xy_ab.npy   per-pair RMSD

Usage:
  python pipeline/03_compute_results/materialize_npy_from_raw.py UNICORNE_BENCH alphaflow_pdb
  for t in alphaflow_pdb afsample2 esmdiff esmflow_pdb bioemu cfrandom speach_af; do
    python pipeline/03_compute_results/materialize_npy_from_raw.py UNICORNE_BENCH $t
  done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def matrices_from_json(json_path: Path):
    d = json.loads(json_path.read_text())
    n_gt, n_pred = int(d["n_gt"]), int(d["n_pred"])
    tm1 = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    tm2 = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    rmsd = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    for k, p in enumerate(d.get("pairs", [])):
        i, j = divmod(k, n_pred)
        if p.get("tm_norm_chain1") is not None:
            tm1[i, j] = float(p["tm_norm_chain1"])
        if p.get("tm_norm_chain2") is not None:
            tm2[i, j] = float(p["tm_norm_chain2"])
        if p.get("rmsd") is not None:
            rmsd[i, j] = float(p["rmsd"])
    return tm1, tm2, rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tool")
    ap.add_argument(
        "--in-suffix",
        default="",
        help="Read from tmalign_raw<suffix>/ instead of tmalign_raw/",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy files (default: skip ensembles whose npy already exists)",
    )
    args = ap.parse_args()

    raw_dir = (PROJECT_ROOT / "results" / args.dataset / args.tool /
               f"tmalign_raw{args.in_suffix}")
    if not raw_dir.is_dir():
        sys.exit(f"error: tmalign_raw dir missing: {raw_dir}")

    out_root = PROJECT_ROOT / "results" / args.dataset / args.tool / "matrices_tmalign"
    out_root.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = 0
    json_files = sorted(raw_dir.glob("*.json"))
    print(f"reading {len(json_files)} JSONs from {raw_dir}")

    for f in json_files:
        eid = f.stem
        eid_dir = out_root / eid
        ndone = sum(1 for n in ("tm_xy.npy", "tm_xy_ab.npy",
                                "tm_xy_ba.npy", "rmsd_xy_ab.npy")
                    if (eid_dir / n).exists())
        if not args.overwrite and ndone == 4:
            skipped += 1
            continue
        try:
            tm1, tm2, rmsd = matrices_from_json(f)
        except Exception as e:
            print(f"  {eid}: parse error — {e}", file=sys.stderr)
            failed += 1
            continue
        eid_dir.mkdir(parents=True, exist_ok=True)
        np.save(eid_dir / "tm_xy.npy", tm1)
        np.save(eid_dir / "tm_xy_ab.npy", tm1)
        np.save(eid_dir / "tm_xy_ba.npy", tm2)
        np.save(eid_dir / "rmsd_xy_ab.npy", rmsd)
        written += 1

    print(f"wrote {written}, skipped {skipped}, failed {failed}  → {out_root}")


if __name__ == "__main__":
    main()
