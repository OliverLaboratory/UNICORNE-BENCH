#!/usr/bin/env python3
r"""
Two finer-grained UNICORNE-BENCH analyses for the gold-standard section.

(a) Co-failure Jaccard heatmap.  For each pair of tools we compute the
    Jaccard overlap of their per-ensemble failure sets (Recall_{0.9} = 0
    means zero GT states recovered).  High overlap ⇒ tools fail on the
    same proteins, suggesting a shared structural blind spot.  Low
    overlap ⇒ failures are tool-specific, motivating tool ensembles.

(b) Recall_{0.9} stratified by motion amplitude.  Each ensemble is
    placed into a tertile by its maximum off-diagonal pairwise GT
    Cα displacement (whole-chain Kabsch RMSD-aligned).  This goes
    beyond the macro stratifications in Section 4.3 by asking what
    KIND of conformational change a tool can recover, not just how
    diverse the GT set is by TM.

Output:
  figures/unicorne_cofail_mechanism.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_cofail_mechanism.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET   = "UNICORNE_BENCH"
RESULTS   = PROJECT_ROOT / "results" / DATASET
GT_ROOT   = RESULTS / "gt_matrices_tmalign"
FIGURES   = PROJECT_ROOT / "figures"

# Same per-tool palette / display names as the rest of the manuscript.
TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
         "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]
TOOL_DISPLAY = {
    "bioemu":        "BioEMU",
    "esmflow_pdb":   "ESMFlow",
    "esmdiff":       "ESMDiff",
    "alphaflow_pdb": "AlphaFlow",
    "cfrandom":      "CFrandom",
    "afsample2":     "AFsample2",
    "afsample3_biorxiv": "AFsample3",
    "speach_af":     "SPEACH_AF",
}


def load_recall(tool: str) -> dict[str, float]:
    csv_path = RESULTS / tool / "results.csv"
    out: dict[str, float] = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("ensemble_ID") or "").strip()
            try:
                out[eid] = float(row.get("recall_at_tm_0.9", "nan"))
            except ValueError:
                continue
    return out


def load_max_disp(eid: str) -> float | None:
    f = GT_ROOT / eid / "max_disp_xx.npy"
    if not f.exists():
        return None
    arr = np.load(f)
    if arr.size == 0 or arr.shape[0] < 2:
        return None
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return float(np.nanmax(arr[mask]))


def main():
    # ------------------------------------------------------------------
    # Per-tool failure sets and pairwise Jaccard
    # ------------------------------------------------------------------
    recall_per_tool = {t: load_recall(t) for t in TOOLS}

    # Common ensemble universe = union of evaluated ensembles across tools.
    universe = set()
    for d in recall_per_tool.values():
        universe.update(d.keys())
    print(f"Universe size: {len(universe)}")

    fail_sets: dict[str, set[str]] = {}
    for t in TOOLS:
        d = recall_per_tool[t]
        # fail = tool was evaluated on ensemble AND recovered nothing
        fail_sets[t] = {e for e, v in d.items()
                        if not np.isnan(v) and v <= 0.0}
        print(f"  {t}: {len(fail_sets[t])} / {len(d)} fully-failed ensembles")

    n = len(TOOLS)
    J = np.full((n, n), np.nan)
    for i, ti in enumerate(TOOLS):
        for j, tj in enumerate(TOOLS):
            a, b = fail_sets[ti], fail_sets[tj]
            union = len(a | b)
            J[i, j] = (len(a & b) / union) if union else np.nan

    # ------------------------------------------------------------------
    # Recall_{0.9} stratified by max-displacement tertile
    # ------------------------------------------------------------------
    eid_disp: dict[str, float] = {}
    for eid in universe:
        d = load_max_disp(eid)
        if d is not None:
            eid_disp[eid] = d
    print(f"Ensembles with max-disp: {len(eid_disp)}")

    # Three bins, biologically motivated:
    #   ≤8 Å    : local refold / loop motion
    #   8-15 Å  : hinge / domain motion
    #   >15 Å   : large-amplitude / state switch (often disorder-to-order)
    BINS = [(0, 8, "Local refold\n(<8 Å)"),
            (8, 15, "Hinge / domain\n(8–15 Å)"),
            (15, 1e6, "Large amplitude\n(>15 Å)")]

    bin_recall: dict[str, list[list[float]]] = {t: [[] for _ in BINS]
                                                for t in TOOLS}
    bin_count = [0] * len(BINS)
    for eid, disp in eid_disp.items():
        bi = next((i for i, (lo, hi, _) in enumerate(BINS)
                   if lo <= disp < hi), None)
        if bi is None:
            continue
        bin_count[bi] += 1
        for t in TOOLS:
            v = recall_per_tool[t].get(eid)
            if v is not None and not np.isnan(v):
                bin_recall[t][bi].append(v)
    print(f"Bin counts: {bin_count}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ----- (a) Co-failure Jaccard heatmap (own figure) -----
    fig_h, ax_h = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("RdPu")
    im = ax_h.imshow(J, cmap=cmap, vmin=0.0, vmax=1.0,
                     interpolation="nearest")
    ax_h.set_xticks(range(n))
    ax_h.set_yticks(range(n))
    labels = [TOOL_DISPLAY[t] for t in TOOLS]
    ax_h.set_xticklabels(labels, rotation=30, ha="right", fontsize=16)
    ax_h.set_yticklabels(labels, fontsize=16)
    for i in range(n):
        for j in range(n):
            if not np.isnan(J[i, j]):
                ax_h.text(j, i, f"{J[i, j]:.2f}",
                          ha="center", va="center",
                          color="black" if J[i, j] < 0.55 else "white",
                          fontsize=11, fontweight="bold")
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard(failure sets)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig_h.tight_layout()
    out_h = FIGURES / "unicorne_cofail.png"
    fig_h.savefig(out_h, dpi=300, bbox_inches="tight", facecolor="white")
    fig_h.savefig(str(out_h).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig_h)
    print(f"Saved {out_h}")

    # ----- (b) Recall_{0.9} stratified by motion amplitude (own figure) -----
    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    bin_labels = [b[2] for b in BINS]
    x = np.arange(len(BINS))
    bar_w = 0.10
    palette = {
        "bioemu":        "#d62728",
        "esmflow_pdb":   "#ff9800",
        "esmdiff":       "#f1c40f",
        "alphaflow_pdb": "#e84a72",
        "cfrandom":      "#1f77b4",
        "afsample2":     "#56a3df",
        "afsample3_biorxiv": "#17A589",
        "speach_af":     "#7faec0",
    }
    for ti, t in enumerate(TOOLS):
        means = [float(np.mean(v)) if v else 0.0 for v in bin_recall[t]]
        offset = (ti - (len(TOOLS) - 1) / 2) * bar_w
        ax_b.bar(x + offset, means, width=bar_w,
                 color=palette[t], edgecolor="#222", linewidth=0.4,
                 label=TOOL_DISPLAY[t])

    ax_b.set_xticks(x)
    # Three short lines per bin so the labels don't collide at this font size.
    ax_b.set_xticklabels([lab.replace(" ", "\n", 1) for lab in bin_labels], fontsize=14)
    ax_b.tick_params(axis="y", labelsize=16)
    ax_b.set_ylabel("Mean $\\text{Recall}_{0.9}$", fontsize=16)
    ax_b.set_ylim(0, 0.5)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                fontsize=11, ncol=1, framealpha=0.95,
                handlelength=1.2, handletextpad=0.4,
                borderaxespad=0.2)
    for xi, bc in enumerate(bin_count):
        ax_b.text(xi, ax_b.get_ylim()[1] * 0.97, f"$n$ = {bc}",
                  ha="center", va="top", fontsize=16, color="#555")

    fig_b.tight_layout()
    out_b = FIGURES / "unicorne_mechanism.png"
    fig_b.savefig(out_b, dpi=300, bbox_inches="tight", facecolor="white")
    fig_b.savefig(str(out_b).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved {out_b}")


if __name__ == "__main__":
    main()
