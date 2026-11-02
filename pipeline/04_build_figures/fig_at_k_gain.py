#!/usr/bin/env python3
"""
Plot the marginal % gain in DR@k / CR@k / EU@k between adjacent k values —
the "diminishing returns" curve that complements the level @k plots.

For each consecutive pair (k_{i-1}, k_i) and tool, the y axis shows

    pct_gain_i = (m_i - m_{i-1}) / m_{i-1} * 100

evaluated as the macro-mean across filtered ensembles per k, then the gain
between those means. Plotted at k = k_i on a log x axis, one line per tool
with the same warm/cool color scheme used elsewhere.

Usage:
  python pipeline/04_build_figures/fig_at_k_gain.py --metric dr
  python pipeline/04_build_figures/fig_at_k_gain.py --metric cr --y-lim 0 100

Output:
  figures/<metric>_at_k_pctgain_<DATASET>.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_DEFAULT = [
    "bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
    "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af",
]
TOOL_DISPLAY = {
    "bioemu": "BioEMU", "esmflow_pdb": "ESMFlow", "alphaflow_pdb": "AlphaFlow",
    "esmdiff": "ESMDiff", "cfrandom": "CFrandom", "afsample2": "AFsample2", "afsample3_biorxiv": "AFsample3",
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
    "bioemu": "o", "esmflow_pdb": "s", "alphaflow_pdb": "^", "esmdiff": "D",
    "cfrandom": "v", "afsample2": "P", "afsample3_biorxiv": "h", "speach_af": "X",
}
K_VALUES = list(range(5, 1001, 5))   # match compute_metrics.py

METRIC_COLUMN = {
    "dr": "recall_09_at_k_{k}_mean",
    "cr": "cr_at_k_{k}_mean",
    "eu": "eu_at_k_{k}_mean",
}
METRIC_TITLE = {
    "dr": "Recall — % gain at each budget step",
    "cr": "FRR — % gain at each budget step",
    "eu": "TM-best — % gain at each budget step",
}
METRIC_DISPLAY = {"dr": "Recall", "cr": "FRR", "eu": "TM-best"}


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
    bl = PROJECT_ROOT / "data" / "inputs" / dataset / "fold_quality_blacklist.csv"
    if bl.exists():
        for row in csv.DictReader(open(bl)):
            eids.discard((row.get("uniprot") or "").strip())
    return eids


def load_metric_means(dataset: str, tool: str, metric: str,
                      keep: set[str] | None) -> dict[int, float]:
    """Macro-mean of the metric@k columns over kept ensembles."""
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
    return {k: float(np.mean(by_k[k])) for k in K_VALUES if by_k[k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UNICORNE_BENCH")
    ap.add_argument("--tools", nargs="+", default=TOOLS_DEFAULT)
    ap.add_argument("--metric", choices=list(METRIC_COLUMN.keys()), default="dr")
    ap.add_argument("--y-lim", type=float, nargs=2, default=None,
                    metavar=("YMIN", "YMAX"))
    ap.add_argument("--xmax", type=int, default=None,
                    help="Cap the x-axis at this k (e.g. --xmax 100)")
    ap.add_argument("--doubling", action="store_true",
                    help="Compute gain only between doublings of k "
                         "(5 → 10 → 20 → 40 → … → 640, plus 1000 as the final point).")
    ap.add_argument("--absolute", action="store_true",
                    help="Plot absolute Δm (in percentage points) instead of "
                         "the relative Δm/m_{i-1} %. Disentangles real scaling "
                         "from the small-baseline amplification of relative gain.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    keep = load_filter(args.dataset)
    if keep is not None:
        print(f"filter: {len(keep)} ensembles")

    # Pick the k anchors at which to evaluate gain.
    # default mode: every adjacent (k_{i-1}, k_i) pair from K_VALUES.
    # --doubling   : only doublings (and a final 1000) — answers "what does
    #                doubling the budget buy me?"
    if args.doubling:
        anchors = [5, 10, 20, 40, 80, 160, 320, 640, 1000]
        anchors = [k for k in anchors if k in K_VALUES]
    else:
        anchors = list(K_VALUES)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.array(anchors[1:], dtype=float)   # gain is from anchor[i-1] → anchor[i]

    for tool in args.tools:
        means_by_k = load_metric_means(args.dataset, tool, args.metric, keep)
        if not means_by_k:
            continue
        means = np.array([means_by_k.get(k, np.nan) for k in anchors])
        prev = means[:-1]
        cur = means[1:]
        if args.absolute:
            # Δm in percentage points (metrics are stored as 0–1 fractions).
            gains = (cur - prev) * 100.0
            print(f"  {tool:14s} abs gains (pp) @k={anchors[1:]}  =  "
                  + ", ".join(f"{g:+.2f}" for g in gains))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                gains = np.where(prev > 0, (cur - prev) / prev * 100.0, np.nan)
            print(f"  {tool:14s} pct gains @k={anchors[1:]}  =  "
                  + ", ".join(f"{g:+.1f}%" for g in gains))
        # Mark every point in doubling mode (sparse anyway), every 12th otherwise.
        markevery = 1 if args.doubling else max(1, len(x) // 12)
        ax.plot(x, gains,
                marker=TOOL_MARKER.get(tool, "o"), markersize=7,
                markevery=markevery,
                color=TOOL_COLOR.get(tool, "#444"), linewidth=1.6,
                label=TOOL_DISPLAY.get(tool, tool),
                markeredgecolor="#222", markeredgewidth=0.6)

    ax.axhline(0, color="#888", linewidth=0.7, linestyle="--")
    unit = "Δ pp" if args.absolute else "% gain"
    metric_short = METRIC_DISPLAY[args.metric]
    if args.doubling:
        ax.set_xlabel("budget k  (gain from doubling the budget vs previous k)", fontsize=16)
        ax.set_ylabel(f"{unit} in {metric_short}@k from doubling k", fontsize=16)
        title_suffix = " (per doubling of k)"
    else:
        ax.set_xlabel("budget k  (gain at step k vs previous step)", fontsize=16)
        ax.set_ylabel(f"{unit} in {metric_short}@k vs previous k", fontsize=16)
        title_suffix = ""
    ax.tick_params(axis="both", labelsize=16)
    abs_suffix = " — absolute Δ (pp)" if args.absolute else ""
    # No title: subcaption in the manuscript handles it.
    ax.set_xscale("log")
    # Match the level @k lineplot's tick distribution (5, 10, 25, …, 1000)
    # so the two figures align visually side-by-side.
    if args.xmax:
        ax.set_xlim(min(K_VALUES) * 0.9, args.xmax * 1.05)
    else:
        ax.set_xlim(min(K_VALUES) * 0.9, max(K_VALUES) * 1.1)
    if args.doubling:
        # Show every doubling anchor (so the x grid matches the data points).
        label_candidates = anchors
    else:
        label_candidates = (5, 10, 25, 50, 100, 250, 500, 1000)
    label_ticks = [k for k in label_candidates
                   if (args.xmax is None or k <= args.xmax)]
    ax.set_xticks(label_ticks)
    ax.set_xticklabels([str(k) for k in label_ticks])
    if args.doubling:
        # Suppress log-scale minor ticks at 2,3,…,9 per decade — in doubling
        # mode the only meaningful x positions are the doubling anchors.
        ax.set_xticks([], minor=True)
    if args.y_lim:
        ax.set_ylim(*args.y_lim)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    leg = ax.legend(loc="best", fontsize=16, frameon=True, ncol=2)
    leg.get_frame().set_alpha(0.9)
    # Match fig_dr_at_k axes position so the two panels render at
    # identical plot-area dimensions when placed side-by-side.
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.13)

    suffix = ""
    if args.doubling:
        suffix += "_doubling"
    if args.xmax:
        suffix += f"_xmax{args.xmax}"
    stem = "abs_gain" if args.absolute else "pctgain"
    out = args.output or (
        PROJECT_ROOT / "figures" / f"{args.metric}_at_k_{stem}{suffix}_{args.dataset}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
