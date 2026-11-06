#!/usr/bin/env python3
"""
Table: UNICORNE multi-state gold standard proteins.

Shows GT states recovered (TM >= 0.9) per tool for known conformational
switch proteins. Includes training set OOD annotation.

Input:
  results/UNICORNE_BENCH/{TOOL}/tmalign_raw/{eid}.json     # canonical TM source
  results/UNICORNE_BENCH/gt_matrices_tmalign/{eid}/        # GT-GT npys + max_disp
  data/unicorne_training_overlap.csv

Output:
  stdout: LaTeX table rows

Usage:
  python pipeline/04_build_figures/tab_unicorne_gold_standard.py
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_compute_results"))
from tmalign_io import load_per_gt_summary  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All seven tools, in the same display order as the rest of the paper
# (generative warm tools first, MSA-perturbation cool tools second).
TOOLS = [
    "bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
    "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af",
]
TOOL_DISPLAY = {
    "bioemu":        "BioEMU",
    "esmflow_pdb":   "ESMFlow",
    "esmdiff":       "ESMDiff",
    "alphaflow_pdb": "AlphaFlow",
    "cfrandom":      "CFrandom",
    "afsample2":     "AFsample2",
    "afsample3_biorxiv": "AFsample3",
    "speach_af":     "SPEACH\\_AF",
}
RESULTS_DIR = PROJECT_ROOT / "results" / "UNICORNE_BENCH"
if not RESULTS_DIR.exists():
    RESULTS_DIR = PROJECT_ROOT / "results" / "UNICORNE"

GOLD_PROTEINS = [
    ("Troponin C", "P63317", 161),
    ("IPP isomerase", "P60615", 95),
    ("Calbindin D9k", "P02633", 79),
    ("V-ATPase sub.\\ F", "P74903", 104),
    ("S100A1", "P23297", 94),
    ("Cas6/Csy4", "Q02MM2", 187),
    ("Rib.\\ prot.\\ uL29", "P0A7M6", 63),
    ("Barstar", "Q02054", 86),
    ("L11 MTase", "Q84BQ9", 254),
    ("Cytochrome $c$", "P00044", 109),
    ("FABP", "P02693", 132),
    ("Rib.\\ prot.\\ uS3", "P05750", 240),
    ("Calmodulin", "P62152", 149),
    ("HPr-like Crh", "O06976", 85),
    ("Lysozyme C", "P61626", 148),
    ("CMP kinase", "Q5SL35", 208),
    ("Adenylate kinase", "P69441", 214),
    # Amino-acid binding protein — every tool but AlphaFlow recovers
    # both states (AlphaFlow's alt-state TM drops to ~0.78).
    ("Amino-acid bind.\\ prot.", "Q5F9M1", 274),
    # Cited in the Discussion as the BioEMU/ESMFlow > MSA-perturbation example.
    ("Dehydratase activator", "Q5U925", 266),
    # Bacteriorhodopsin (7TM, light-activated) — every tool fails the
    # conformational transition.
    ("Bacteriorhodopsin", "P02945", 262),
]


def _tm_max2(p: dict) -> float | None:
    """min-length-normalized TM = max(tm_norm_chain1, tm_norm_chain2)."""
    a, b = p.get("tm_norm_chain1"), p.get("tm_norm_chain2")
    try:
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b): return None
        return max(a, b)
    except (TypeError, ValueError):
        return None


def per_gt_max_tm(eid: str, tool: str) -> np.ndarray | None:
    """Return shape (n_gt,) array: per-GT max TM_max2 across predictions.

    Reads the canonical tmalign_raw JSON (single source of truth across tools).
    Returns None if the file is missing.
    """
    jf = RESULTS_DIR / tool / "tmalign_raw" / f"{eid}.json"
    if not jf.exists():
        # Without the raw JSONs, use the exported per-GT summary.
        rows = load_per_gt_summary(RESULTS_DIR.name, tool).get(eid)
        vals = [r["max_tm_max2"] for r in rows or [] if not math.isnan(r["max_tm_max2"])]
        return np.array(vals, dtype=float) if vals else None
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    pairs = d.get("pairs") or []
    by_gt: dict[str, float] = {}
    for p in pairs:
        t = _tm_max2(p)
        if t is None: continue
        gt = p.get("gt") or ""
        prev = by_gt.get(gt)
        if prev is None or t > prev:
            by_gt[gt] = t
    if not by_gt:
        return None
    # Order by GT path so output is deterministic.
    return np.array([by_gt[k] for k in sorted(by_gt.keys())], dtype=float)


def load_gt_tm(eid: str) -> float | None:
    """Min over all GT pairs of TM_max2 (most-different GT pair)."""
    src = RESULTS_DIR / "gt_matrices_tmalign" / eid
    for name in ["tm_xx.npy", "tm_xx_ab.npy"]:
        f = src / name
        if f.exists():
            mat = np.load(f)
            if mat.shape[0] > 1:
                mask = ~np.eye(mat.shape[0], dtype=bool)
                return float(mat[mask].min())
    nr, nc = src / "tm_xx_norm_row.npy", src / "tm_xx_norm_col.npy"
    if nr.exists() and nc.exists():
        mat = np.maximum(np.load(nr), np.load(nc))
        if mat.shape[0] > 1:
            mask = ~np.eye(mat.shape[0], dtype=bool)
            return float(mat[mask].min())
    return None


def load_gt_max_disp(eid: str) -> float | None:
    """Max over all GT pairs of max per-residue CA displacement after TM-align superposition (Å)."""
    f = RESULTS_DIR / "gt_matrices_tmalign" / eid / "max_disp_xx.npy"
    if not f.exists():
        return None
    mat = np.load(f)
    if mat.shape[0] < 2:
        return None
    mask = ~np.eye(mat.shape[0], dtype=bool)
    vals = mat[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=None,
                    help="Subset of tool keys to render (default: all 7).  "
                         "Order is preserved.")
    args, _ = ap.parse_known_args()
    if args.tools:
        global TOOLS
        unknown = [t for t in args.tools if t not in TOOL_DISPLAY]
        if unknown:
            raise SystemExit(f"unknown tool(s): {unknown}")
        TOOLS = list(args.tools)

    # AF2-OOD annotation (every PDB structure for the protein is post-AF2-cutoff).
    # Star marks the cell for tools trained on AF2-derived data.
    ood_proteins = set()
    overlap_csv = PROJECT_ROOT / "data" / "unicorne_training_overlap.csv"
    if overlap_csv.exists():
        import csv
        with open(overlap_csv) as f:
            for row in csv.DictReader(f):
                if row.get("in_esmflow_train", "") == "No":
                    ood_proteins.add(row["uniprot"])

    # Fold-quality blacklist — drop UniProts whose GTs are artifactual / IDP /
    # fibril and shouldn't be evaluated as multi-state ensembles.
    blacklist = set()
    import csv as _csv
    bl_csv = PROJECT_ROOT / "data" / "inputs" / "UNICORNE_BENCH" / "fold_quality_blacklist.csv"
    if bl_csv.exists():
        for row in _csv.DictReader(open(bl_csv)):
            blacklist.add((row.get("uniprot") or "").strip())

    rows = []
    for name, uniprot, length in GOLD_PROTEINS:
        if uniprot in blacklist:
            continue
        gt_tm = load_gt_tm(uniprot)
        gt_disp = load_gt_max_disp(uniprot)
        if gt_tm is None and gt_disp is None:
            # Protein not in the curated UNICORNE_BENCH set — skip rather than
            # emit a row of "?" and "--" (happens when the benchmark filter
            # dropped a protein that's still in the hand-picked gold list).
            continue
        gt_str = f"{gt_tm:.2f}" if gt_tm else "?"
        disp_str = f"{gt_disp:.1f}" if gt_disp else "?"
        parts = []
        for tool in TOOLS:
            per_gt = per_gt_max_tm(uniprot, tool)
            if per_gt is None or per_gt.size == 0:
                parts.append("--"); continue
            n_gt = per_gt.size
            recovered = int(np.sum(per_gt >= 0.9))
            # Only star tools with AF2's training cutoff — BioEMU isn't trained on
            # AF2-derived ensembles and AFsample3 uses AlphaFold3 (later cutoff),
            # so AF2-OOD doesn't apply to either.
            star = ("$^\\star$" if uniprot in ood_proteins
                    and tool not in ("bioemu", "afsample3_biorxiv") else "")
            if recovered > 1:
                parts.append(f"\\cellcolor{{lightgreen}}{recovered}/{n_gt}{star}")
            elif recovered == 1:
                parts.append(f"\\cellcolor{{lightyellow}}{recovered}/{n_gt}{star}")
            else:
                parts.append(f"{recovered}/{n_gt}{star}")
        rows.append(f"{name:<24s}& {uniprot} & {length} & {gt_str} & {disp_str} & {' &  '.join(parts)}")
    # All rows except the last end with "\\"; last row is bare so main.tex
    # can append "\\" after the \input (avoids "Misplaced \noalign").
    for i, r in enumerate(rows):
        if i < len(rows) - 1:
            print(f"{r} \\\\")
        else:
            # End last row with "%" to suppress the EOF newline that LaTeX would
            # otherwise insert between the row content and \bottomrule.
            print(f"{r}%")


if __name__ == "__main__":
    main()
