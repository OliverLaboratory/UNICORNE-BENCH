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
import paper_style  # noqa: E402  (serif font; text sized for the paper)

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
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ----- (a) Co-failure Jaccard heatmap (own figure) -----
    # Drawn at its printed size (3.4 in, appendix).
    fig_h, ax_h = plt.subplots(figsize=(3.41, 3.0))
    cmap = plt.get_cmap("RdPu")
    im = ax_h.imshow(J, cmap=cmap, vmin=0.0, vmax=1.0,
                     interpolation="nearest")
    ax_h.set_xticks(range(n))
    ax_h.set_yticks(range(n))
    labels = [TOOL_DISPLAY[t] for t in TOOLS]
    ax_h.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax_h.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            if not np.isnan(J[i, j]):
                ax_h.text(j, i, f"{J[i, j]:.2f}",
                          ha="center", va="center",
                          color="black" if J[i, j] < 0.55 else "white",
                          fontsize=7)
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard (failure sets)")

    fig_h.tight_layout()
    out_h = FIGURES / "unicorne_cofail.png"
    paper_style.finalize(fig_h, 3.41, median_pt=7.0, max_pt=8.0)
    fig_h.savefig(out_h, dpi=300, facecolor="white")
    fig_h.savefig(str(out_h).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig_h)
    print(f"Saved {out_h}")

    # ----- (b) Recall_{0.9} stratified by motion amplitude (own figure) -----
    # Drawn at its printed size (1.76 in, next to the two recall panels);
    # the shared legend comes from fig_dr_boxplots.py, so colors match it.
    fig_b, ax_b = plt.subplots(figsize=(1.76, 1.45))
    bin_labels = [b[2] for b in BINS]
    x = np.arange(len(BINS))
    bar_w = 0.10
    palette = {
        "bioemu":        "#C0392B",
        "esmflow_pdb":   "#E67E22",
        "esmdiff":       "#F1C40F",
        "alphaflow_pdb": "#E74C3C",
        "cfrandom":      "#1F4E79",
        "afsample2":     "#2980B9",
        "afsample3_biorxiv": "#17A589",
        "speach_af":     "#5DADE2",
    }
    for ti, t in enumerate(TOOLS):
        means = [float(np.mean(v)) if v else 0.0 for v in bin_recall[t]]
        offset = (ti - (len(TOOLS) - 1) / 2) * bar_w
        ax_b.bar(x + offset, means, width=bar_w,
                 color=palette[t], edgecolor="#222", linewidth=0.3,
                 label=TOOL_DISPLAY[t])

    ax_b.set_xticks(x)
    # Bin names and ensemble counts are in the figure caption.
    ax_b.set_xticklabels(["<8", "8–15", ">15"])
    ax_b.set_xlabel("GT motion amplitude (Å)")
    ax_b.set_ylabel("Recall")
    ax_b.set_ylim(0, 0.5)
    ax_b.set_yticks([0.0, 0.2, 0.4])
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    print("mechanism bins:", ", ".join(f"{lab.splitlines()[0]} n={bc}" for lab, bc in zip(bin_labels, bin_count)))

    fig_b.tight_layout()
    out_b = FIGURES / "unicorne_mechanism.png"
    paper_style.finalize(fig_b, 1.76, relayout=False)
    fig_b.subplots_adjust(**paper_style.FIG3_MARGINS)
    fig_b.savefig(out_b, dpi=300, facecolor="white")
    fig_b.savefig(str(out_b).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig_b)
    print(f"Saved {out_b}")


if __name__ == "__main__":
    main()
