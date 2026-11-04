#!/usr/bin/env python3
"""
Build the test fixtures in tests/data/ from the full benchmark output.

For each fixture ensemble this copies every GT structure and, for every tool,
the predictions that are the best match (by TM normalized by GT length) to at
least one GT. Taking the row-wise max over that sub-matrix gives the same best
prediction per GT as the full matrix, so Recall@0.9, FRR and mean best TM can be
recomputed from the fixture files alone and compared with results.csv.

The stored TM-align output for every GT x selected-prediction pair goes to
tests/data/expected.json, together with the ensemble's results.csv row.

Needs the full output (data/outputs/ and results/*/tmalign_raw/), so it is run
once by the maintainers, not by the tests:

  python tests/build_fixtures.py --src-root /path/to/full/checkout
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "data"
DATASET = "UNICORNE_BENCH"
TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb", "cfrandom",
         "afsample2", "afsample3_biorxiv", "speach_af"]

# Chosen to cover: full recovery with 2 and 3 GTs, partial recall with many
# GTs, a best TM just above and just below 0.9, mixed mmCIF/PDB GTs, and GTs
# shorter than the predictions (where GT- and prediction-length normalization
# disagree about recovery).
ENSEMBLES = {
    "A0A0H2ZNA4": "fully recovered by five tools (2 GTs)",
    "G0YP42": "fully recovered by four tools (3 GTs)",
    "G1SFR8": "partial recall, 5 GTs, best TM just above 0.9",
    "A6TF31": "mixed mmCIF/PDB GTs, best TM 0.9004",
    "P61769": "GT 98 aa vs. prediction 119 aa: normalization decides recovery",
    "G1TPG3": "best TM 0.8996, just below the threshold",
}
PAIR_FIELDS = ["tm_norm_chain1", "tm_norm_chain2", "rmsd", "len_chain1", "len_chain2", "len_aligned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True,
                    help="Checkout holding data/outputs/ and results/*/tmalign_raw/")
    args = ap.parse_args()
    src = args.src_root.resolve()

    if OUT.exists():
        shutil.rmtree(OUT)
    expected = {"dataset": DATASET, "tmalign_version": None, "ensembles": {}}
    for eid, why in ENSEMBLES.items():
        gt_src = src / "data" / "inputs" / DATASET / "input_pdbs" / eid
        gt_dst = OUT / "structures" / eid / "gt"
        gt_dst.mkdir(parents=True)
        ens = {"why": why, "tools": {}}
        for tool in TOOLS:
            d = json.loads((src / "results" / DATASET / tool / "tmalign_raw" / f"{eid}.json").read_text())
            n_gt, n_pred = d["n_gt"], d["n_pred"]
            pairs = d["pairs"]
            gt_files = [Path(pairs[i * n_pred]["gt"]).name for i in range(n_gt)]
            if "gt_files" in ens:
                assert ens["gt_files"] == gt_files, (eid, tool)
            ens["gt_files"] = gt_files
            tm1 = np.array([[p.get("tm_norm_chain1") or np.nan for p in pairs[i * n_pred:(i + 1) * n_pred]]
                            for i in range(n_gt)], dtype=float)
            best = [int(np.where(np.isnan(r), 0.0, r).argmax()) for r in tm1]
            sel = sorted(set(best))
            pred_dst = OUT / "structures" / eid / "pred" / tool
            pred_dst.mkdir(parents=True)
            sub = {f: [] for f in PAIR_FIELDS}
            for i in range(n_gt):
                for f in PAIR_FIELDS:
                    sub[f].append([pairs[i * n_pred + j].get(f) for j in sel])
            pred_files = []
            for j in sel:
                p = Path(pairs[j]["pred"])
                shutil.copy2(p, pred_dst / p.name)
                pred_files.append(p.name)
            if expected["tmalign_version"] is None:
                m = re.search(r"TM-align \(Version (\d+)\)", pairs[0]["raw"])
                expected["tmalign_version"] = m.group(1) if m else None
            row = next(r for r in csv.DictReader(open(src / "results" / DATASET / tool / "results.csv"))
                       if r["ensemble_ID"] == eid)
            ens["tools"][tool] = {
                "n_pred": n_pred,
                "pred_files": pred_files,
                "pred_indices": sel,
                "best_pred_per_gt": [sel.index(b) for b in best],
                "pairs": sub,
                "results_csv": {k: float(row[k]) for k in ("n_gt", "n_pred", "dr", "cr", "tm_bar", "emd")},
            }
        for name in ens["gt_files"]:
            shutil.copy2(gt_src / name, gt_dst / name)
        expected["ensembles"][eid] = ens
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1))
    n_files = sum(1 for p in (OUT / "structures").rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in (OUT / "structures").rglob("*") if p.is_file())
    print(f"{n_files} structure files, {size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
