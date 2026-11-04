#!/usr/bin/env python3
"""
Figure: TM-score free-energy landscapes of predicted ensembles with high and
low principal angle (PA) to the GT motion (paper Fig. 5).

Each panel places every prediction of one tool for one ensemble by its
TM-score to the two GT conformations and shows the free energy
F = -log(count) of a 50 x 50 histogram on [0, 1]^2, shifted to a minimum of 0
and drawn as filled contours (20 levels, capped at 10 kT), as in the
bioemu-benchmarks plots. Dashed: TM = 0.85; dotted: TM between the two GTs.

Input:
  results/pa_heatmaps/examples.csv          panel metadata (GT pair, GT-GT TM, PA)
  results/pa_heatmaps/<UniProt>_<tool>_tm.npy   (2, K): TM of each prediction to GT x, GT y

Output:
  figures/pa_heatmaps.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_pa_heatmaps.py
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import paper_style  # noqa: E402  (serif font; text sized for the paper)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "results" / "pa_heatmaps"
PANEL_ORDER = [("G3XD64", "cfrandom"), ("P02911", "bioemu"),     # high PA
               ("O66490", "esmdiff"), ("P15454", "afsample2")]   # low PA
TOOL_DISPLAY = {"cfrandom": "CFrandom", "bioemu": "BioEMU", "esmdiff": "ESMDiff",
                "afsample2": "AFsample2"}
BINS, MAX_ENERGY, LEVELS = 50, 10.0, 20
AXIS_LO = 0.3          # all predictions in these panels have TM > 0.3


def free_energy(tm_x: np.ndarray, tm_y: np.ndarray):
    """Bin-centre grid and F = -log(count + 1e-6) - min, capped at MAX_ENERGY + 1."""
    h, xe, ye = np.histogram2d(tm_x, tm_y, bins=BINS, range=[[0, 1], [0, 1]])
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    energy = -np.log(h.T + 1e-6)
    energy -= energy.min()
    return *np.meshgrid(xc, yc), np.minimum(energy, MAX_ENERGY + 1)


def pdb_label(name: str) -> str:
    """'5J1J_A' -> '5J1J'."""
    return name.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "pa_heatmaps.png")
    args = ap.parse_args()

    meta = {(r["uniprot"], r["tool"]): r for r in csv.DictReader(open(DATA / "examples.csv"))}
    cmap = copy.copy(plt.get_cmap("turbo"))
    cmap.set_over("white")

    # Drawn at its printed size: full text width, four square panels.
    fig_h = 2.15
    fig = plt.figure(figsize=(5.5, fig_h))
    left, panel, gap, bottom = 0.065, 0.155, 0.075, 0.29
    panel_h = panel * 5.5 / fig_h
    axes = [fig.add_axes((left + i * (panel + gap), bottom, panel, panel_h)) for i in range(4)]
    cf = None
    for ax, key in zip(axes, PANEL_ORDER):
        m = meta[key]
        tm = np.load(DATA / f"{key[0]}_{key[1]}_tm.npy")
        xx, yy, energy = free_energy(tm[0], tm[1])
        cf = ax.contourf(xx, yy, energy, levels=LEVELS, cmap=cmap, vmin=0.0, vmax=MAX_ENERGY)
        gt_tm = float(m["gt_gt_tm"])
        ax.axvline(0.85, color="black", linestyle="--", linewidth=0.6)
        ax.axhline(0.85, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(gt_tm, color="green", linestyle=":", linewidth=0.9)
        ax.axhline(gt_tm, color="green", linestyle=":", linewidth=0.9)
        ax.set_xlim(AXIS_LO, 1.0); ax.set_ylim(AXIS_LO, 1.0)
        ax.set_aspect("equal")
        ticks = [0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlabel(f"TM to {pdb_label(m['gt_x'])}")
        ax.set_ylabel(f"TM to {pdb_label(m['gt_y'])}", labelpad=1)
        ax.set_title(f"{key[0]}, {TOOL_DISPLAY[key[1]]}\nPA = {float(m['principal_angle']):.1f}°",
                     pad=3, linespacing=1.1)

    # Group labels above each pair of panels.
    for (a, b), label in (((0, 1), "High PA"), ((2, 3), "Low PA")):
        x0, x1 = axes[a].get_position().x0, axes[b].get_position().x1
        fig.text((x0 + x1) / 2, 0.985, label, ha="center", va="top", fontweight="bold")
        fig.add_artist(Line2D([x0, x1], [0.915, 0.915], color="#444444", linewidth=0.6))

    cax = fig.add_axes((left + 4 * panel + 3 * gap + 0.02, bottom, 0.012, panel_h))
    cbar = fig.colorbar(cf, cax=cax, ticks=[0, 2, 4, 6, 8, 10])
    cbar.set_label("Free energy (kT)")
    fig.legend(handles=[Line2D([], [], color="black", linestyle="--", linewidth=0.6, label="TM = 0.85"),
                        Line2D([], [], color="green", linestyle=":", linewidth=0.9, label="GT–GT TM")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.8, columnspacing=1.5)

    paper_style.finalize(fig, 5.5, median_pt=7.0, max_pt=7.5, relayout=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
