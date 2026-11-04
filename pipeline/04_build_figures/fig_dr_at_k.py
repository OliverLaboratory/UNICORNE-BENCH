#!/usr/bin/env python3
"""
DR@k vs k line plot: how does Discovery Rate scale with prediction budget?

For each tool we plot the macro-mean of `recall_09_at_k_<k>_mean` across the
filtered ensembles, with k ∈ {50, 100, 150, 200, 250}. Same visual language
as the EU/DR/CR line plots (warm = generative, cool = MSA-subsampling AF2,
distinct markers, legend top-left).

Inputs:
  results/<DATASET>/<TOOL>/results.csv  (must already have recall_09_at_k_*_mean cols)
  data/inputs/<DATASET>/filtered_ensembles.csv

Output:
  figures/dr_at_k_lineplot_<DATASET>.png

Usage:
  python pipeline/04_build_figures/fig_dr_at_k.py
  python pipeline/04_build_figures/fig_dr_at_k.py --dataset DANCE_BENCH --y-lim 0 0.6
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paper_style  # noqa: E402  (serif font; text sized for the paper)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_DEFAULT = [
    # Generative models (warm)
    "bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
    # MSA-subsampling AF2 variants (cool)
    "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af",
]
TOOL_DISPLAY = {
    "bioemu": "BioEMU",
    "esmflow_pdb": "ESMFlow",
    "alphaflow_pdb": "AlphaFlow",
    "esmdiff": "ESMDiff",
    "cfrandom": "CFrandom",
    "afsample2": "AFsample2",
    "afsample3_biorxiv": "AFsample3",
    "speach_af": "SPEACH_AF",
}
TOOL_COLOR = {
    "bioemu":        "#C0392B",
    "esmflow_pdb":   "#E67E22",
    "esmdiff":       "#F1C40F",
    "alphaflow_pdb": "#E74C3C",
    "cfrandom":      "#1F4E79",
    "afsample2":     "#2980B9",
    "afsample3_biorxiv": "#17A589",
    "speach_af":     "#5DADE2",
}
TOOL_MARKER = {
    "bioemu": "o",
    "esmflow_pdb": "s",
    "alphaflow_pdb": "^",
    "esmdiff": "D",
    "cfrandom": "v",
    "afsample2": "P",
    "afsample3_biorxiv": "h",
    "speach_af": "X",
}

K_VALUES = list(range(5, 1001, 5))   # match compute_metrics.py


def load_filter(dataset: str) -> set[str] | None:
    f = PROJECT_ROOT / "data" / "inputs" / dataset / "filtered_ensembles.csv"
    if not f.exists():
        return None
    eids = set()
    with open(f) as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or "").strip()
            if eid:
                eids.add(eid)
    # Drop fold-quality blacklist (artifactual / IDP / fibril GTs).
    bl = PROJECT_ROOT / "data" / "inputs" / dataset / "fold_quality_blacklist.csv"
    if bl.exists():
        for row in csv.DictReader(open(bl)):
            eids.discard((row.get("uniprot") or "").strip())
    return eids


METRIC_COLUMN = {
    "dr": "recall_09_at_k_{k}_mean",
    "cr": "cr_at_k_{k}_mean",
    "tm_bar": "tm_bar_at_k_{k}_mean",
}
METRIC_LABEL = {
    "dr": "Recall@k",
    "cr": "FRR@k",
    "tm_bar": r"$\overline{\mathrm{TM}}$@k",
}
METRIC_TITLE = {
    "dr": "Recall vs sample budget",
    "cr": "Full-Recall Rate vs sample budget",
    "tm_bar": "Mean best-match TM vs sample budget",
}


def load_metric_at_k(dataset: str, tool: str, metric: str,
                     keep: set[str] | None) -> dict[int, np.ndarray]:
    """Return {k: array of per-ensemble metric@k means} for one tool."""
    csv_path = PROJECT_ROOT / "results" / dataset / tool / "results.csv"
    if not csv_path.exists():
        return {}
    col_tpl = METRIC_COLUMN[metric]
    by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or "").strip()
            if keep is not None and eid not in keep:
                continue
            for k in K_VALUES:
                v = row.get(col_tpl.format(k=k))
                if v in (None, "", "nan"):
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(fv):
                    by_k[k].append(fv)
    return {k: np.array(by_k[k]) for k in K_VALUES if by_k[k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UNICORNE_BENCH")
    ap.add_argument("--tools", nargs="+", default=TOOLS_DEFAULT)
    ap.add_argument("--metric", choices=list(METRIC_COLUMN.keys()), default="dr",
                    help="Which @k metric to plot (dr | cr | tm_bar)")
    ap.add_argument("--y-lim", type=float, nargs=2, default=None,
                    metavar=("YMIN", "YMAX"))
    ap.add_argument("--xmax", type=int, default=None,
                    help="Cap the x-axis at this k (e.g. --xmax 100)")
    ap.add_argument("--show-band", action="store_true",
                    help="Shade ±SEM around each line (sample SEM across ensembles)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    keep = load_filter(args.dataset)
    if keep is not None:
        print(f"filter: {len(keep)} ensembles")

    # Drawn at its printed size (2.7 in, side by side with its pair; the shared
    # tool legend is a separate file, see fig_dr_boxplots.py).
    fig, ax = plt.subplots(figsize=(2.70, 2.1))
    x = np.array(K_VALUES, dtype=float)
    for tool in args.tools:
        by_k = load_metric_at_k(args.dataset, tool, args.metric, keep)
        if not by_k:
            print(f"  {tool}: no data — skip")
            continue
        means, sems, ns = [], [], []
        for k in K_VALUES:
            arr = by_k.get(k)
            if arr is None or arr.size == 0:
                means.append(np.nan); sems.append(np.nan); ns.append(0)
            else:
                means.append(float(arr.mean()))
                sems.append(float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
                ns.append(int(arr.size))
        means = np.array(means); sems = np.array(sems)
        print(f"  {tool}: n={ns}  means={[f'{m:.3f}' for m in means]}")
        # With Δk=5 (200 points) drawing a marker per point overcrowds, so
        # mark every Nth point — picks roughly 10-15 markers across the curve.
        markevery = max(1, len(K_VALUES) // 12)
        ax.plot(x, means,
                marker=TOOL_MARKER.get(tool, "o"), markersize=3.5,
                markevery=markevery,
                color=TOOL_COLOR.get(tool, "#444"),
                linewidth=1.0,
                label=TOOL_DISPLAY.get(tool, tool),
                markeredgecolor="#222", markeredgewidth=0.4)
        if args.show_band:
            ax.fill_between(x, means - sems, means + sems,
                            color=TOOL_COLOR.get(tool, "#444"), alpha=0.12,
                            linewidth=0)

    ax.set_xlabel("budget k (samples)")
    ax.set_ylabel(METRIC_LABEL[args.metric])
    # No title: subcaption in the manuscript handles it.
    ax.set_xscale("log")
    if args.xmax:
        ax.set_xlim(min(K_VALUES) * 0.9, args.xmax * 1.05)
    label_candidates = (5, 10, 25, 50, 100, 250, 500, 1000)
    label_ticks = [k for k in label_candidates
                   if k in K_VALUES and (args.xmax is None or k <= args.xmax)]
    ax.set_xticks(label_ticks)
    ax.set_xticklabels([str(k) for k in label_ticks])
    if args.y_lim:
        ax.set_ylim(*args.y_lim)
    else:
        # Pick a reasonable default per metric
        ax.set_ylim(0.0, {"dr": 0.5, "cr": 0.15, "tm_bar": 1.0}[args.metric])
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    # `loc="best"` auto-picks the quadrant with the fewest overlaps with
    # plotted data — works well for these monotone-rising @k curves where
    # the lower-right corner is open for DR/CR (low ceilings) and the
    # upper-left is open for EU (lines fill the upper half).
    # Pin axes to identical fractional position across the panel pair so
    # the side-by-side rendering in the manuscript has matching plot
    # areas (saved without bbox_inches='tight' to preserve the canvas).

    suffix = f"_xmax{args.xmax}" if args.xmax else ""
    out = args.output or (
        PROJECT_ROOT / "figures" / f"{args.metric}_at_k_lineplot{suffix}_{args.dataset}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    paper_style.finalize(fig, 2.70, relayout=False)
    fig.subplots_adjust(**paper_style.AT_K_MARGINS)
    fig.savefig(out, dpi=180)
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
