#!/usr/bin/env python3
"""
Figure 2 — Dataset characterization (combined, 1 row x 3 panels).

Panel (a)  Min pairwise GT TM-score within UNICORNE ensembles.
Panel (b)  GT ensemble size (n_gt) within UNICORNE.
Panel (c)  Intra-dataset 3-mer Jaccard sequence-identity CDF for
           DANCE / UNICORNE / OOD60 / DOMAINMOTION.

Inputs:
  data/inputs/UNICORNE_BENCH/filtered_ensembles_v2.csv               (panels a, b)
  data/inputs/DANCE_BENCH/benchmarking-dataset-DANCE.csv             (panel c)
  data/inputs/UNICORNE_BENCH/benchmarking-dataset-13.csv             (panel c)
  data/inputs/OOD60/input_dataset.csv                                (panel c)
  data/inputs/DOMAINMOTION/input_dataset.csv                         (panel c)

Output:
  figures/fig_dataset_characterization.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_dataset_characterization.py
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

# Color scheme matches fig_dataset_identity_cdf.py for cross-figure consistency.
COLORS = {
    "DANCE": "#EE854A",
    "UNICORNE": "#4878CF",
    "OOD60": "#D65F5F",
    "DOMAINMOTION": "#6ACC65",
}
DISPLAY = {
    "DANCE": "DANCE",
    "UNICORNE": "UNICORNE",
    "OOD60": "OOD60",
    "DOMAINMOTION": "DOMAINMOTION",
}

UNICORNE_FILTERED = PROJECT_ROOT / "data/inputs/UNICORNE_BENCH/filtered_ensembles_v2.csv"

REDUNDANCY_DATASETS = {
    "DANCE": PROJECT_ROOT / "data/inputs/DANCE_BENCH/benchmarking-dataset-DANCE.csv",
    "UNICORNE": PROJECT_ROOT / "data/inputs/UNICORNE_BENCH/benchmarking-dataset-13.csv",
    "OOD60": PROJECT_ROOT / "data/inputs/OOD60/input_dataset.csv",
    "DOMAINMOTION": PROJECT_ROOT / "data/inputs/DOMAINMOTION/input_dataset.csv",
}

KMER_K = 3


def load_filtered(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_sequences(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            eid = (r.get("uniprot") or r.get("ensemble_ID")
                   or r.get("ensemble_id") or "").strip()
            seq = (r.get("sequence") or r.get("ref_sequence")
                   or r.get("ref_seq") or "").strip()
            if eid and seq:
                rows.append((eid, seq))
    return rows


def kmer_set(seq: str, k: int) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def pairwise_jaccard(records: list[tuple[str, str]], k: int = KMER_K) -> np.ndarray:
    n = len(records)
    if n < 2:
        return np.array([])
    ksets = [kmer_set(seq, k) for _, seq in records]
    n_pairs = n * (n - 1) // 2
    sims = np.empty(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(ksets[i] & ksets[j])
            union = len(ksets[i] | ksets[j])
            sims[idx] = inter / union if union > 0 else 0.0
            idx += 1
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins-tm", type=int, default=30,
                        help="Bins for min-pairwise-TM histogram (panel a).")
    parser.add_argument("--n-gt-cap", type=int, default=30,
                        help="Cap n_gt at this value for the histogram (panel b).")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "figures" / "fig_dataset_characterization.png")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # ---- Panel (a): UNICORNE min pairwise GT TM ----
    rows = load_filtered(UNICORNE_FILTERED)
    min_tms = [float(r["min_pairwise_tm"]) for r in rows]
    n_gts = [int(r["n_gt_dedup"]) for r in rows]

    ax = axes[0]
    ax.hist(min_tms, bins=args.bins_tm, range=(0, 1),
            color=COLORS["UNICORNE"], edgecolor="white",
            linewidth=0.5, alpha=0.9)
    ax.set_xlabel("Min pairwise GT TM-score")
    ax.set_ylabel("Number of ensembles")
    ax.set_xlim(0, 1)
    ax.axvline(0.85, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(0.86, ax.get_ylim()[1] * 0.92, "0.85",
            color="red", fontsize=16, alpha=0.8)

    # ---- Panel (b): UNICORNE n_gt distribution ----
    ax = axes[1]
    cap = args.n_gt_cap
    obs_max = max(n_gts)
    n_gts_capped = [min(n, cap) for n in n_gts]
    upper = min(obs_max, cap)
    ax.hist(n_gts_capped, bins=np.arange(2, upper + 2) - 0.5,
            color=COLORS["UNICORNE"], edgecolor="white",
            linewidth=0.5, alpha=0.9)
    median_ngt = int(np.median(n_gts))
    ax.axvline(median_ngt, color="black", linestyle="--",
               alpha=0.5, linewidth=1)
    ax.text(median_ngt + 0.4, ax.get_ylim()[1] * 0.92,
            f"median = {median_ngt}", fontsize=16, alpha=0.8)
    ax.set_xlim(1.5, upper + 0.5)
    ax.set_xlabel("Number of GT conformations")
    ax.set_ylabel("Number of ensembles")
    suffix = f" (capped at {cap})" if obs_max > cap else ""

    # ---- Panel (c): 3-mer Jaccard CDF across 4 datasets ----
    ax = axes[2]
    thresholds = np.linspace(0, 1, 200)
    for ds in ["DANCE", "UNICORNE", "OOD60", "DOMAINMOTION"]:
        records = load_sequences(REDUNDANCY_DATASETS[ds])
        n = len(records)
        sims = pairwise_jaccard(records, k=KMER_K)
        if sims.size == 0:
            print(f"{ds}: no sequences, skipping")
            continue
        frac = np.array([np.mean(sims >= t) for t in thresholds])
        ax.plot(thresholds * 100, frac * 100,
                color=COLORS[ds], linewidth=2,
                label=f"{DISPLAY[ds]} (n={n})")
        print(f"{ds}: n={n}, mean Jaccard={sims.mean():.4f}, "
              f"median={np.median(sims):.4f}")

    ax.set_xlabel(f"{KMER_K}-mer Jaccard threshold (%)")
    ax.set_ylabel("Pairs above threshold (%)")
    ax.set_xlim(0, 100)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 110)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_out = str(args.output).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.output}")
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    main()
