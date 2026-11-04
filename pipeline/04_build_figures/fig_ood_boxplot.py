#!/usr/bin/env python3
"""
Figure: best-across-tools mean best-match TM on UNICORNE-BENCH, stratified
by AlphaFold2 OOD status.

For every UniProt in the gold-standard panel we take the maximum
mean-best-match TM (TM_bar) across all evaluated tools, then split into
two groups:
  * in AF2 train  - at least one PDB structure for the UniProt was
                    deposited before the 2018-05-01 AlphaFold2 training
                    cutoff;
  * AF2-OOD       - every deposited structure post-dates the cutoff.

Output:
  figures/unicorne_ood_boxplot.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_ood_boxplot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the per-UniProt collection from fig_unicorne_gold_standard so the
# OOD flag, fold-quality blacklist, and best-across-tools logic stay in
# one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_unicorne_gold_standard import collect_data, AF2_OOD_TOOLS, FIGURES  # noqa: E402
import paper_style  # noqa: E402  (serif font; text sized for the paper)


IN_TRAIN_COLOR = "#cad8e6"
OOD_COLOR      = "#f3c8b3"


def main():
    records = collect_data(metric="tm_bar", tools=AF2_OOD_TOOLS)
    in_train = [r["best_value"] for r in records if not r["af2_ood"]]
    ood      = [r["best_value"] for r in records if r["af2_ood"]]
    print(f"in AF2 train: n={len(in_train)}  median={np.median(in_train):.3f}  "
          f"mean={np.mean(in_train):.3f}")
    print(f"AF2-OOD:      n={len(ood)}  median={np.median(ood):.3f}  "
          f"mean={np.mean(ood):.3f}")

    plt.rcParams.update({
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 7x7 to match the scatter (fig_unicorne_gold_standard.py uses 7x7),
    # so that when the manuscript places the two panels side-by-side at
    # equal widths their plot areas match.
    # Drawn at its printed size (half the text width, appendix).
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    bp = ax.boxplot(
        [in_train, ood],
        widths=0.55,
        patch_artist=True,
        showfliers=False,           # outliers redundant with strip points
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#444"),
        capprops=dict(color="#444"),
        boxprops=dict(linewidth=1.0),
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], (IN_TRAIN_COLOR, OOD_COLOR)):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444")
        patch.set_alpha(0.6)

    # Strip plot — every UniProt as one point with horizontal jitter so
    # the underlying distribution is visible behind the box summary.
    rng = np.random.default_rng(0)
    for x_center, ys, edge in [
        (1, in_train, "#3a5a82"),
        (2, ood,      "#b04a2a"),
    ]:
        jitter = rng.uniform(-0.18, 0.18, size=len(ys))
        ax.scatter(
            x_center + jitter, ys,
            s=3, alpha=0.55,
            facecolor="white", edgecolor=edge, linewidth=0.4,
            zorder=1,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"in AF2 train\n(n = {len(in_train)})",
                        f"AF2-OOD\n(n = {len(ood)})"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_ylabel(r"Best $\overline{\mathrm{TM}}$ across tools", fontsize=16)
    ax.set_ylim(0.3, 1.02)
    # Dotted: TM = 0.9; dashed: maximum GT-GT TM (0.85) -- explained in the caption.
    ax.axhline(0.9, ls=":", color="#888", lw=0.6, alpha=0.7)
    ax.axhline(0.85, ls="--", color="#888", lw=0.7, alpha=0.7)

    out = FIGURES / "unicorne_ood_boxplot.png"
    paper_style.finalize(fig, 2.75)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
