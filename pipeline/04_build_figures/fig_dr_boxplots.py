#!/usr/bin/env python3
"""
Two side-by-side box plots of per-ensemble Discovery Rate (DR) by tool:

  Panel A — DR vs GT structural diversity
      For each tool, two boxes: ensembles with HIGH GT pairwise TM
      (less diverse, GT states close to each other) vs LOW GT pairwise TM
      (more diverse, multi-state proteins). Split is the median of the
      dataset's per-ensemble min-pairwise-GT-TM.

  Panel B — DR vs GT ensemble size
      For each tool, two boxes: ensembles with exactly 2 GT conformations
      vs ensembles with 3 or more GT conformations.

Inputs:
  results/<DATASET>/<TOOL>/results.csv          per-ensemble dr (and n_gt)
  results/<DATASET>/gt_matrices_tmalign/<eid>/tm_xx*.npy   GT × GT TM matrix

Output:
  figures/dr_boxplots_<DATASET>.png

Usage:
  python pipeline/04_build_figures/fig_dr_boxplots.py
  python pipeline/04_build_figures/fig_dr_boxplots.py --dataset DANCE_BENCH
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
    # Generative models — warm reds/oranges
    "bioemu":        "#C0392B",  # deep red
    "esmflow_pdb":   "#E67E22",  # orange
    "esmdiff":       "#F1C40F",  # gold
    "alphaflow_pdb": "#E74C3C",  # tomato
    # MSA-subsampling AF2 variants — cool blues
    "cfrandom":      "#1F4E79",  # navy
    "afsample2":     "#2980B9",  # mid blue
    "afsample3_biorxiv": "#17A589",
    "speach_af":     "#5DADE2",  # sky blue
}


# ---------------------------------------------------------------------------
# GT pairwise TM (per-ensemble: min over (i,j) with i<j)
# ---------------------------------------------------------------------------

def load_min_gt_tm(dataset: str) -> dict[str, float]:
    """Return {eid: min pairwise GT TM} for ensembles with ≥2 GT structures."""
    gt_root = PROJECT_ROOT / "results" / dataset / "gt_matrices_tmalign"
    out: dict[str, float] = {}
    if not gt_root.is_dir():
        return out
    for eid_dir in gt_root.iterdir():
        if not eid_dir.is_dir():
            continue
        tm = None
        for name in ("tm_xx.npy", "tm_xx_ab.npy"):
            f = eid_dir / name
            if f.exists():
                tm = np.load(f)
                break
        if tm is None or tm.shape[0] < 2:
            continue
        # min over off-diagonal
        mask = ~np.eye(tm.shape[0], dtype=bool)
        out[eid_dir.name] = float(tm[mask].min())
    return out


def load_seq_len(dataset: str) -> dict[str, int]:
    """Return {eid: ref_seq_len} from data/inputs/{dataset}/filtered_ensembles_v2.csv."""
    out: dict[str, int] = {}
    csv_path = PROJECT_ROOT / "data" / "inputs" / dataset / "filtered_ensembles_v2.csv"
    if not csv_path.exists():
        return out
    import csv as _csv
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            eid = row.get("ensemble_ID", "").strip()
            try:
                out[eid] = int(row.get("ref_seq_len", "0"))
            except (TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Per-tool DR (read straight from results.csv)
# ---------------------------------------------------------------------------

METRIC_FALLBACKS = {
    "dr": ("dr", "recall_at_tm_0.9"),
    "cr": ("cr",),
    "tm_bar": ("tm_bar",),
    "emd": ("emd",),
}

METRIC_LABEL = {
    "dr": "Recall",
    "cr": "Full-Recall Rate (FRR)",
    "tm_bar": "Mean best-match TM",
    "emd": "Earth Mover's Distance (EMD, lower = better)",
}


def load_tool_metric(dataset: str, tool: str, metric: str = "dr") -> dict[str, tuple[float, int]]:
    """Return {eid: (metric_value, n_gt)} for one tool. NaN → 0.0 for dr/cr/tm_bar;
    NaN for emd is left as NaN (skipped downstream)."""
    csv_path = PROJECT_ROOT / "results" / dataset / tool / "results.csv"
    if not csv_path.exists():
        return {}
    fallbacks = METRIC_FALLBACKS.get(metric, (metric,))
    out: dict[str, tuple[float, int]] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or "").strip()
            if not eid:
                continue
            val = None
            for k in fallbacks:
                v = row.get(k)
                if v not in (None, "", "nan"):
                    try:
                        val = float(v)
                    except (TypeError, ValueError):
                        val = None
                    if val is not None and not np.isnan(val):
                        break
                    val = None
            if val is None:
                if metric == "emd":
                    continue            # skip; can't impute
                val = 0.0                # dr/cr/tm_bar treat NaN as zero
            try:
                n_gt = int(row.get("n_gt") or 0)
            except (TypeError, ValueError):
                n_gt = 0
            out[eid] = (val, n_gt)
    return out


def load_filter(dataset: str) -> set[str] | None:
    f = PROJECT_ROOT / "data" / "inputs" / dataset / "filtered_ensembles.csv"
    if not f.exists():
        return None
    eids = set()
    with open(f) as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or row.get("uniprot") or "").strip()
            if eid:
                eids.add(eid)
    bl = PROJECT_ROOT / "data" / "inputs" / dataset / "fold_quality_blacklist.csv"
    if bl.exists():
        for row in csv.DictReader(open(bl)):
            eids.discard((row.get("uniprot") or "").strip())
    return eids


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

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


def line_plot(ax, vals_by_tool_bin, tools, bin_labels, title, x_label,
              annotate_n=False, agg="mean"):
    """One line per tool across the bins.

    vals_by_tool_bin[tool][bin_idx] = list of metric values.
    `agg` ∈ {"mean", "median"} reduces each bin to one point.
    """
    n_bins = len(bin_labels)
    x = np.arange(n_bins)
    for tool in tools:
        ys = []
        ns = []
        for bi in range(n_bins):
            data = vals_by_tool_bin[tool][bi]
            if data:
                ys.append(float(np.mean(data) if agg == "mean" else np.median(data)))
                ns.append(len(data))
            else:
                ys.append(np.nan)
                ns.append(0)
        ax.plot(
            x, ys,
            marker=TOOL_MARKER.get(tool, "o"),
            markersize=8,
            color=TOOL_COLOR.get(tool, "#444"),
            linewidth=1.6,
            label=TOOL_DISPLAY.get(tool, tool),
            markeredgecolor="#222",
            markeredgewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=15, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(getattr(line_plot, "y_label", "Ensemble Utility (EU)"))
    yscale = getattr(line_plot, "y_scale", "linear")
    if yscale == "logit":
        # Logit scale gives lots of vertical space to values near 0 *and* 1
        # — useful for zooming into the high-EU regime without clipping low ones.
        ax.set_yscale("logit")
        ax.set_ylim(getattr(line_plot, "y_lim", (1e-3, 1 - 1e-3)))
    else:
        ax.set_yscale(yscale)
        ax.set_ylim(getattr(line_plot, "y_lim", (0.0, 1.0)))
    # No title: subcaptions in the manuscript handle per-panel labelling.
    ax.grid(True, linestyle=":", alpha=0.5, which="both")


def grouped_box(ax, drs_by_tool_group, tools, group_labels, group_colors,
                title, threshold_caption=None, rng=None):
    """Box plot with individual points and a mean diamond overlaid.

    drs_by_tool_group[tool][group_idx] = list of metric values.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_groups = len(group_labels)
    width = 0.36
    positions_main = np.arange(len(tools))
    for gi in range(n_groups):
        offset = (gi - (n_groups - 1) / 2) * (width + 0.04)
        positions = positions_main + offset
        data = [drs_by_tool_group[t][gi] for t in tools]
        face = group_colors[gi]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            zorder=3,
        )
        for box in bp["boxes"]:
            box.set_facecolor(face)
            box.set_alpha(0.35)
            box.set_edgecolor("#222")
            box.set_linewidth(0.9)
        for whisker in bp["whiskers"]:
            whisker.set_color("#222")
            whisker.set_linewidth(0.9)
        for cap in bp["caps"]:
            cap.set_color("#222")
            cap.set_linewidth(0.9)

        # Jittered scatter of every ensemble's metric
        for x, ys in zip(positions, data):
            if not ys:
                continue
            jitter = rng.uniform(-width * 0.32, width * 0.32, size=len(ys))
            ax.scatter(np.full(len(ys), x) + jitter, ys,
                       s=6, alpha=0.45, color=face,
                       edgecolors="#222", linewidths=0.2,
                       zorder=2)

        # Diamond marker at the mean
        for x, ys in zip(positions, data):
            if not ys:
                continue
            ax.scatter([x], [float(np.mean(ys))],
                       marker="D", s=22, color="#111",
                       edgecolors="white", linewidths=1.0, zorder=4)

    ax.set_xticks(positions_main)
    ax.set_xticklabels([TOOL_DISPLAY.get(t, t) for t in tools], rotation=20, ha="right")
    ax.set_ylabel(getattr(grouped_box, "y_label", "Discovery Rate (DR)"))
    ax.set_ylim(getattr(grouped_box, "y_lim", (-0.04, 1.04)))
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.45, edgecolor="#222")
               for c in group_colors]
    handles.append(plt.Line2D([0], [0], marker="D", linestyle="",
                              markersize=6, color="#111",
                              markeredgecolor="white", markeredgewidth=0.8,
                              label="mean"))
    labels = list(group_labels) + ["mean"]
    leg = ax.legend(handles, labels, loc="lower left",
                    fontsize=16, frameon=True)
    leg.get_frame().set_alpha(0.9)

    if threshold_caption:
        ax.text(0.99, 0.99, threshold_caption,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=16, color="#444",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UNICORNE_BENCH")
    ap.add_argument("--tools", nargs="+", default=TOOLS_DEFAULT)
    ap.add_argument(
        "--metric",
        choices=["dr", "cr", "tm_bar", "emd"],
        default="tm_bar",
        help="Per-ensemble metric column to plot (default: tm_bar)",
    )
    ap.add_argument("--filtered-only", action="store_true", default=True)
    ap.add_argument("--no-filtered", dest="filtered_only", action="store_false")
    ap.add_argument(
        "--gt-tm-threshold",
        type=float,
        default=None,
        help="Fixed split for high/low GT TM (default: dataset median)",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--y-scale",
        choices=["linear", "logit", "log"],
        default="linear",
        help="Y-axis scale for the line plots (logit = zoom into both [0,1] tails)",
    )
    ap.add_argument(
        "--y-lim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Override y-axis limits (e.g. --y-lim 0.005 0.995 for logit zoom)",
    )
    args = ap.parse_args()

    grouped_box.y_label = METRIC_LABEL[args.metric]
    grouped_box.y_lim = (-0.04, 1.04) if args.metric != "emd" else (-0.04, 1.04)
    line_plot.y_label = METRIC_LABEL[args.metric]
    line_plot.y_scale = args.y_scale
    if args.y_lim:
        line_plot.y_lim = tuple(args.y_lim)
    elif args.y_scale == "logit":
        # Tight default so the actual data range (most metric values are
        # in [0.5, 0.95]) gets the bulk of the vertical space rather than
        # being squeezed into the middle of a (0.005, 0.995) axis where
        # logit is near-linear.
        line_plot.y_lim = (0.5, 0.99)
    else:
        line_plot.y_lim = grouped_box.y_lim

    print(f"loading GT pairwise TMs for {args.dataset}...")
    min_gt_tm = load_min_gt_tm(args.dataset)
    print(f"  {len(min_gt_tm)} ensembles with ≥2 GT structures")

    keep_set = load_filter(args.dataset) if args.filtered_only else None

    # ------------------------------------------------------------------
    # Define the bin edges
    # ------------------------------------------------------------------
    tm_vals = list(min_gt_tm.values()) if keep_set is None else [
        v for e, v in min_gt_tm.items() if e in keep_set]
    if not tm_vals:
        raise SystemExit("no GT TM data for this dataset")
    # 5 quantile bins on min-pairwise GT TM (equal-frequency)
    tm_edges = np.quantile(tm_vals, np.linspace(0.0, 1.0, 6))
    tm_edges[0] -= 1e-9   # include the min in bin 0
    tm_edges[-1] += 1e-9  # include the max in bin 4
    tm_bin_labels = [
        f"[{tm_edges[i]:.2f}-{tm_edges[i+1]:.2f}]" for i in range(5)
    ]
    print(f"GT TM quintile edges: {[round(e,3) for e in tm_edges]}")

    n_gt_bins = [(2,), (3,), (4,), (5,), tuple(range(6, 100))]
    n_gt_bin_labels = ["2", "3", "4", "5", "6+"]

    # Sequence-length bins.  Edges chosen to span the UNICORNE_BENCH
    # range (60-298 aa) without an empty trailing bucket.
    seq_len = load_seq_len(args.dataset)
    len_bins = [(0, 100), (100, 150), (150, 200), (200, 300)]
    len_bin_labels = ["≤100", "100-150", "150-200", "200-300"]

    def assign_len_bin(L: int) -> int | None:
        for i, (lo, hi) in enumerate(len_bins):
            if lo < L <= hi or (i == 0 and L <= hi):
                return i
        return None

    def assign_tm_bin(t: float) -> int:
        for i in range(5):
            if tm_edges[i] <= t < tm_edges[i+1]:
                return i
        return 4

    def assign_ngt_bin(n: int) -> int | None:
        for i, vs in enumerate(n_gt_bins):
            if n in vs:
                return i
        return None

    # ------------------------------------------------------------------
    # Collect per-tool metric values partitioned by GT TM bin and n_gt bin
    # ------------------------------------------------------------------
    by_gt_tm: dict[str, list[list[float]]] = {}
    by_n_gt: dict[str, list[list[float]]] = {}
    by_len: dict[str, list[list[float]]] = {}

    tools_with_data: list[str] = []
    for tool in args.tools:
        dr_map = load_tool_metric(args.dataset, tool, metric=args.metric)
        if not dr_map:
            continue
        tools_with_data.append(tool)
        tm_bins = [[] for _ in range(5)]
        ngt_bins = [[] for _ in range(len(n_gt_bins))]
        len_bins_v = [[] for _ in range(len(len_bins))]
        for eid, (val, n_gt) in dr_map.items():
            if keep_set is not None and eid not in keep_set:
                continue
            t = min_gt_tm.get(eid)
            if t is not None:
                tm_bins[assign_tm_bin(t)].append(val)
            ng_bin = assign_ngt_bin(n_gt)
            if ng_bin is not None:
                ngt_bins[ng_bin].append(val)
            L = seq_len.get(eid)
            if L is not None:
                lb = assign_len_bin(L)
                if lb is not None:
                    len_bins_v[lb].append(val)
        by_gt_tm[tool] = tm_bins
        by_n_gt[tool] = ngt_bins
        by_len[tool] = len_bins_v
        print(f"  {tool}: TM bins n={[len(b) for b in tm_bins]}  "
              f"n_gt bins {n_gt_bin_labels}={[len(b) for b in ngt_bins]}  "
              f"len bins {len_bin_labels}={[len(b) for b in len_bins_v]}")

    if not tools_with_data:
        raise SystemExit("no tool results.csv found")

    # ------------------------------------------------------------------
    # Plot — one line plot per x-axis dimension, written to separate files
    # ------------------------------------------------------------------
    metric_short = {"dr": "Recall", "cr": "FRR", "tm_bar": "TM-best",
                    "emd": "EMD"}.get(args.metric, args.metric.upper())
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Flip both axes high→low (least diverse / largest n_gt on the LEFT).
    tm_bin_labels_rev = list(reversed(tm_bin_labels))
    n_gt_bin_labels_rev = list(reversed(n_gt_bin_labels))
    by_gt_tm_rev = {t: list(reversed(by_gt_tm[t])) for t in by_gt_tm}
    by_n_gt_rev = {t: list(reversed(by_n_gt[t])) for t in by_n_gt}

    panels = [
        ("tm", by_gt_tm_rev, tm_bin_labels_rev,
         f"{metric_short} vs GT structural diversity (5 quantile bins)",
         "min pairwise GT TM (less diverse ←  → more diverse)"),
        ("ngt", by_n_gt_rev, n_gt_bin_labels_rev,
         f"{metric_short} vs GT ensemble size",
         "number of GT conformations (more ←  → fewer)"),
        ("len", by_len, len_bin_labels,
         f"{metric_short} vs reference sequence length",
         "reference sequence length (aa)"),
    ]

    written: list[Path] = []
    for tag, vals, bin_labels, title, x_label in panels:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        line_plot(ax, vals, tools_with_data, bin_labels=bin_labels,
                  title=title, x_label=x_label)
        # vs-GT-TM panel: lines descend left→right so lower-left is empty.
        # vs-n_gt panel: lines descend left→right too but the upper-left
        # corner is open, so put the legend up there instead.
        leg_loc = "upper left" if tag == "ngt" else "lower left"
        leg = ax.legend(loc=leg_loc, fontsize=16, frameon=True, ncol=2)
        leg.get_frame().set_alpha(0.9)
        # No suptitle: subcaptions in the manuscript handle it.
        ax.tick_params(axis="both", labelsize=16)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        fig.tight_layout()
        scale_suffix = "" if args.y_scale == "linear" else f"_{args.y_scale}"
        out_path = (args.output if args.output and tag == "tm" else None) or (
            fig_dir / f"{args.metric}_lineplot_{tag}{scale_suffix}_{args.dataset}.png"
        )
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        fig.savefig(str(out_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
