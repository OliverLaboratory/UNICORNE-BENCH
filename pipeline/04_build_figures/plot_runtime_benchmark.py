#!/usr/bin/env python3
"""Generate runtime benchmark figure: sequence length vs inference time."""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"

TOOL_STYLE = {
    "bioemu": {"color": "#1f77b4", "marker": "o", "label": "BioEMU"},
    "esmflow_pdb": {"color": "#2ca02c", "marker": "s", "label": "AlphaFlow"},
    "esmdiff": {"color": "#d62728", "marker": "^", "label": "ESMDiff"},
    "cfrandom": {"color": "#9467bd", "marker": "D", "label": "CFrandom"},
}

SIZE_BINS = {"small": 60, "medium": 150, "large": 290}


def load_data():
    """Load and aggregate benchmark results."""
    data = defaultdict(list)
    csvfiles = [
        PROJECT_ROOT / "results" / "runtime_benchmark.csv",
        PROJECT_ROOT / "results" / "runtime_benchmark_bioemu_large.csv",
        PROJECT_ROOT / "results" / "runtime_benchmark_cfrandom.csv",
    ]
    for csvfile in csvfiles:
        if not csvfile.exists():
            continue
        with open(csvfile) as f:
            for row in csv.DictReader(f):
                if not row.get("wall_clock_seconds"):
                    continue
                key = (row["tool"], row["size_bin"], int(row["num_samples"]))
                data[key].append(float(row["wall_clock_seconds"]))
    return data


def main():
    data = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left panel: seq length vs time for n=100 ---
    ax = axes[0]
    x_positions = np.array([60, 150, 290])

    for tool, style in TOOL_STYLE.items():
        medians = []
        q25s = []
        q75s = []
        for size in ["small", "medium", "large"]:
            key = (tool, size, 100)
            vals = data.get(key, [])
            if len(vals) > 1:
                vals = vals[1:]  # drop warmup
            if vals:
                medians.append(statistics.median(vals))
                sorted_v = sorted(vals)
                n = len(sorted_v)
                q25s.append(sorted_v[max(0, n // 4)])
                q75s.append(sorted_v[min(n - 1, 3 * n // 4)])
            else:
                medians.append(float("nan"))
                q25s.append(float("nan"))
                q75s.append(float("nan"))

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        ax.plot(x_positions, medians, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8, zorder=3)
        ax.fill_between(x_positions, q25s, q75s, alpha=0.15, color=style["color"])

    ax.set_xlabel("Sequence length (residues)", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("100 samples", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["~60", "~150", "~290"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 320)

    # --- Right panel: num_samples vs time for each size bin ---
    ax = axes[1]
    n_values = np.array([20, 40, 60, 100])
    linestyles = {"small": "-", "medium": "--", "large": ":"}
    size_labels = {"small": "~60aa", "medium": "~150aa", "large": "~290aa"}

    for tool, style in TOOL_STYLE.items():
        for size, ls in linestyles.items():
            medians = []
            for n in n_values:
                key = (tool, size, n)
                vals = data.get(key, [])
                if len(vals) > 1:
                    vals = vals[1:]
                medians.append(statistics.median(vals) if vals else float("nan"))

            label = f"{style['label']} ({size_labels[size]})" if tool == "bioemu" else None
            ax.plot(n_values, medians, marker=style["marker"], color=style["color"],
                    linestyle=ls, linewidth=1.5, markersize=5, alpha=0.8)

    # Add legend entries for line styles (size bins)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="-", label="~60aa"),
        Line2D([0], [0], color="gray", linestyle="--", label="~150aa"),
        Line2D([0], [0], color="gray", linestyle=":", label="~290aa"),
        Line2D([0], [0], color=TOOL_STYLE["bioemu"]["color"], marker="o",
               linestyle="", markersize=8, label="BioEMU"),
        Line2D([0], [0], color=TOOL_STYLE["esmflow_pdb"]["color"], marker="s",
               linestyle="", markersize=8, label="AlphaFlow"),
        Line2D([0], [0], color=TOOL_STYLE["esmdiff"]["color"], marker="^",
               linestyle="", markersize=8, label="ESMDiff"),
        Line2D([0], [0], color=TOOL_STYLE["cfrandom"]["color"], marker="D",
               linestyle="", markersize=8, label="CFrandom"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, ncol=2)
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("Scaling with sample count", fontsize=13)
    ax.set_xticks(n_values)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for outdir in [FIGURES_DIR]:
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "runtime_benchmark.pdf"
        fig.savefig(outpath, bbox_inches="tight", dpi=150)
        print(f"Saved: {outpath}")
        # Also PNG for preview
        fig.savefig(outdir / "runtime_benchmark.png", bbox_inches="tight", dpi=150)

    plt.close()


if __name__ == "__main__":
    main()
