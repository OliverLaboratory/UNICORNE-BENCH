#!/usr/bin/env python3
"""
Quantify per-ensemble unimodality of predicted ensembles.

For each (tool, ensemble) we represent every prediction as a vector in
``[0, 1]^{n_gt}`` whose i-th coordinate is the TM-score to GT i.  In
this space, two predictions are "close" iff they have a similar
relationship to the GT set; clusters correspond to genuinely distinct
prediction modes (some near GT_1, some near GT_2, some near neither).
We cluster via average-linkage hierarchical clustering on Euclidean
distance with a TM-distance cutoff and report the *effective number of
modes*:

    M_eff = exp(H(p_1, ..., p_C)) = exp(- sum_i p_i log p_i)

where p_i is the fraction of predictions in cluster i.  M_eff = 1 means
all predictions are in one cluster (perfectly unimodal); M_eff = K is
uniform spread across K clusters.  Smaller M_eff ⇒ more concentrated
predictions ⇒ more mode-collapsed.

Output:
  figures/unicorne_mode_count.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_mode_count.py
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import paper_style  # noqa: E402  (serif font; text sized for the paper)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "UNICORNE_BENCH"
RESULTS = PROJECT_ROOT / "results" / DATASET

TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
         "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]
TOOL_DISPLAY = {
    "bioemu": "BioEMU", "esmflow_pdb": "ESMFlow", "esmdiff": "ESMDiff",
    "alphaflow_pdb": "AlphaFlow", "cfrandom": "CFrandom",
    "afsample2": "AFsample2", "afsample3_biorxiv": "AFsample3", "speach_af": "SPEACH_AF",
}
PALETTE = {
    "bioemu":        "#C0392B",
    "esmflow_pdb":   "#E67E22",
    "esmdiff":       "#F1C40F",
    "alphaflow_pdb": "#E74C3C",
    "cfrandom":      "#1F4E79",
    "afsample2":     "#2980B9",
    "afsample3_biorxiv": "#17A589",
    "speach_af":     "#5DADE2",
}


def load_emd_means(keep: set[str] | None) -> dict[str, float]:
    """Per-tool macro-mean EMD on UNICORNE-BENCH (cheap — read from
    results.csv rather than re-loading the heavy tmalign_raw JSON)."""
    out: dict[str, float] = {}
    for t in TOOLS:
        fp = RESULTS / t / "results.csv"
        if not fp.exists():
            continue
        vals: list[float] = []
        with open(fp) as f:
            for r in csv.DictReader(f):
                eid = (r.get("ensemble_ID") or "").strip()
                if keep and eid not in keep:
                    continue
                v = r.get("emd", "")
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(fv):
                    vals.append(fv)
        out[t] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_filter() -> set[str] | None:
    fp = PROJECT_ROOT / "data" / "inputs" / DATASET / "filtered_ensembles_v2.csv"
    if not fp.exists():
        return None
    ids = set()
    with open(fp) as f:
        for r in csv.DictReader(f):
            eid = r.get("ensemble_ID", "").strip()
            if eid:
                ids.add(eid)
    bl = PROJECT_ROOT / "data" / "inputs" / DATASET / "fold_quality_blacklist.csv"
    if bl.exists():
        for r in csv.DictReader(open(bl)):
            ids.discard((r.get("uniprot") or "").strip())
    return ids


_JQ_EXPR = r'.pairs[] | "\(.gt)\t\(.pred)\t\(.tm_norm_chain1)\t\(.tm_norm_chain2)"'


def feature_matrix(json_path: Path) -> np.ndarray | None:
    """Return a (K, n_gt) matrix of TM scores for one (tool, ensemble).

    We use ``jq`` to stream-parse so we never materialize the multi-MB
    `raw` TMalign text fields in Python — purely the gt/pred/TM tuples
    we need.
    """
    if not json_path.exists():
        return None
    try:
        out = subprocess.run(
            ["jq", "-r", _JQ_EXPR, str(json_path)],
            check=True, capture_output=True, text=True, timeout=120,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    by_pred: dict[str, dict[str, float]] = {}
    gt_set: set[str] = set()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        gt, pred, t1, t2 = parts
        try:
            tm = max(float(t1), float(t2))
        except ValueError:
            continue
        by_pred.setdefault(pred, {})[gt] = tm
        gt_set.add(gt)

    if len(gt_set) < 2:
        return None
    gt_order = sorted(gt_set)
    M = []
    for pred, gts in by_pred.items():
        if len(gts) < len(gt_order):
            continue
        M.append([gts[g] for g in gt_order])
    if len(M) < 10:
        return None
    return np.asarray(M, dtype=float)


def _worker(args, cutoff):
    tool, jf = args
    F = feature_matrix(jf)
    if F is None:
        return tool, None
    M_eff, _, _ = effective_mode_count(F, cutoff=cutoff)
    return tool, M_eff


def effective_mode_count(F: np.ndarray, cutoff: float = 0.15) -> tuple[float, int, list[int]]:
    """Hierarchical clustering on F (Euclidean), cut at `cutoff` distance.

    Returns (M_eff, n_clusters, cluster_sizes).
    """
    K = F.shape[0]
    if K < 2:
        return 1.0, 1, [K]
    dists = pdist(F, metric="euclidean")
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=cutoff, criterion="distance")
    sizes = sorted(Counter(labels).values(), reverse=True)
    p = np.array(sizes) / K
    H = -float((p * np.log(p)).sum())
    return float(np.exp(H)), len(sizes), sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.15,
                    help="Euclidean cutoff in TM-vector space (default 0.15)")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "figures" / "unicorne_mode_count.png")
    ap.add_argument("--cache", type=Path,
                    default=PROJECT_ROOT / "results" / DATASET / "_mode_count_cache.npz",
                    help="Cache of computed M_eff distributions per tool. "
                         "Loaded on subsequent runs unless --rebuild is set.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Recompute M_eff distributions from the heavy "
                         "tmalign_raw JSONs even if the cache exists.")
    args = ap.parse_args()

    keep = load_filter() or set()
    print(f"filter: {len(keep)} ensembles")

    emd_means = load_emd_means(keep)
    for t, v in emd_means.items():
        print(f"  {t}: EMD = {v:.3f}")

    # ------------------------------------------------------------------
    # Cache layer: re-rendering (e.g. styling tweaks, axis range, EMD
    # tick text) does not need the slow per-(tool, ensemble) JSON parse;
    # we save M_eff and per-tool n into one npz and reload on next run.
    # ------------------------------------------------------------------
    cache_loaded = False
    per_tool_modes: dict[str, list[float]] = {t: [] for t in TOOLS}
    per_tool_n: dict[str, int] = {t: 0 for t in TOOLS}
    cache_meta = {"cutoff": float(args.cutoff)}

    if args.cache.exists() and not args.rebuild:
        try:
            data = np.load(args.cache, allow_pickle=False)
            stored_cutoff = float(data["cutoff"]) if "cutoff" in data.files else None
            if stored_cutoff is not None and stored_cutoff != args.cutoff:
                print(f"cache cutoff ({stored_cutoff}) != requested "
                      f"({args.cutoff}); rebuilding")
            else:
                for t in TOOLS:
                    key = f"modes__{t}"
                    if key in data.files:
                        per_tool_modes[t] = data[key].tolist()
                        per_tool_n[t] = len(per_tool_modes[t])
                cache_loaded = any(per_tool_modes[t] for t in TOOLS)
                if cache_loaded:
                    print(f"loaded cache from {args.cache}")
        except Exception as e:
            print(f"cache load failed ({e}); rebuilding")
            cache_loaded = False

    if not cache_loaded:
        # Build the (tool, json) work list across all tools, then parallelize.
        work: list[tuple[str, Path]] = []
        for tool in TOOLS:
            raw_dir = RESULTS / tool / "tmalign_raw"
            if not raw_dir.exists():
                print(f"  {tool}: no tmalign_raw"); continue
            for jf in raw_dir.glob("*.json"):
                eid = jf.stem
                if keep and eid not in keep:
                    continue
                work.append((tool, jf))
        print(f"work units: {len(work)}")

        n_proc = max(1, min(mp.cpu_count() - 1, 16))
        print(f"using {n_proc} workers")
        with mp.Pool(n_proc) as pool:
            for tool, m in pool.imap_unordered(
                    partial(_worker, cutoff=args.cutoff), work, chunksize=8):
                if m is None:
                    continue
                per_tool_modes[tool].append(m)
                per_tool_n[tool] += 1

        # Persist the per-tool M_eff distributions plus the cutoff used.
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.cache,
            cutoff=np.asarray(args.cutoff, dtype=float),
            **{f"modes__{t}": np.asarray(per_tool_modes[t], dtype=float)
               for t in TOOLS},
        )
        print(f"saved cache to {args.cache}")

    for tool in TOOLS:
        v = per_tool_modes[tool]
        if v:
            print(f"  {tool}: n={per_tool_n[tool]}  median M_eff={np.median(v):.2f}  "
                  f"mean={np.mean(v):.2f}  IQR=[{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]")

    # ------------------------------------------------------------------
    # Plot — strip + box per tool
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Drawn at its printed size (2.75 in, above the UMAP panel): one row per
    # tool, effective mode count on x, each tool's mean EMD on the right axis.
    fig, ax = plt.subplots(figsize=(2.75, 1.75))
    rng = np.random.default_rng(0)
    positions = list(range(len(TOOLS), 0, -1))      # first tool on top
    box_data = [per_tool_modes[t] for t in TOOLS]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(color="#444", linewidth=0.6),
        capprops=dict(color="#444", linewidth=0.6),
        zorder=2,
    )
    for patch, t in zip(bp["boxes"], TOOLS):
        patch.set_facecolor(PALETTE[t])
        patch.set_edgecolor("#222")
        patch.set_alpha(0.45)

    for ti, t in enumerate(TOOLS):
        ys = per_tool_modes[t]
        if not ys:
            continue
        y_jit = positions[ti] + rng.uniform(-0.18, 0.18, size=len(ys))
        ax.scatter(ys, y_jit, s=1.5, alpha=0.35,
                   facecolor="white", edgecolor=PALETTE[t], linewidth=0.3,
                   zorder=1)

    ax.set_yticks(positions)
    ax.set_yticklabels([TOOL_DISPLAY[t] for t in TOOLS])
    ax.set_xlabel("Effective mode count $M_{\\mathrm{eff}}$")
    ax.axvline(1.0, color="#444", linestyle=":", linewidth=0.7)
    # Cap x so the small per-tool boxes stay readable; outliers above 6
    # (a handful per tool) are still represented by jittered points but
    # don't dominate the axis.
    ax.set_xlim(0.9, 6.0)
    ax.set_ylim(0.4, len(TOOLS) + 0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    emd_ax = ax.twinx()
    emd_ax.set_ylim(ax.get_ylim())
    emd_ax.set_yticks(positions)
    emd_ax.set_yticklabels([f"{emd_means[t]:.3f}" if not np.isnan(emd_means.get(t, np.nan)) else "--"
                            for t in TOOLS])
    emd_ax.tick_params(axis="y", length=0)
    emd_ax.set_ylabel("EMD", rotation=0, va="bottom", ha="center")
    emd_ax.yaxis.set_label_coords(1.08, 1.0)
    for spine in ("top", "right"):
        emd_ax.spines[spine].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paper_style.finalize(fig, 2.75, median_pt=7.5)
    fig.savefig(args.output, dpi=300, facecolor="white")
    fig.savefig(str(args.output).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
