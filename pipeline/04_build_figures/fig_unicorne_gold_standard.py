#!/usr/bin/env python3
"""
WILL_TM dataset showcase: scatter plot of GT diversity vs recall
with well-known conformational-switch proteins highlighted.

Generates:
  will_tm_gold_standard.png — scatter + highlighted case-study proteins

Usage:
  python scripts/plot_will_tm_showcase.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
DATA = BASE / "data"
FIGURES = BASE / "figures"

TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
         "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]
# AF2-OOD panels are defined by AlphaFold2's training cutoff, so they leave out
# AFsample3, which runs AlphaFold3 (later cutoff).
AF2_OOD_TOOLS = [t for t in TOOLS if t != "afsample3_biorxiv"]
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
# Match the warm/cool grouping used in the @k and line plots.
_TOOL_PALETTE = {
    "bioemu":        "#C0392B",
    "esmflow_pdb":   "#E67E22",
    "esmdiff":       "#F1C40F",
    "alphaflow_pdb": "#E74C3C",
    "cfrandom":      "#1F4E79",
    "afsample2":     "#2980B9",
    "afsample3_biorxiv": "#17A589",
    "speach_af":     "#5DADE2",
}
# Per-tool marker symbols — same set as fig_dr_at_k.py / fig_dr_boxplots.py
# so a reader can identify the tool by shape across figures.
_TOOL_MARKER = {
    "bioemu":        "o",
    "esmflow_pdb":   "s",
    "alphaflow_pdb": "^",
    "esmdiff":       "D",
    "cfrandom":      "v",
    "afsample2":     "P",
    "afsample3_biorxiv": "h",
    "speach_af":     "X",
}

# Curated OOD callouts — biologically interesting AF2-OOD proteins,
# chosen for scientific recognizability rather than just metric extremes.
# Mix of failure cases, mid-range cases, and a few successes spanning
# different functional categories (kinetochore, mitotic spindle, IDP,
# metal-binding, chromatin, transport, chaperone, ubiquitin-ligase).
OOD_HIGHLIGHT_PROTEINS = {
    # Failure end (best EU < 0.6)
    "A0A125YJ71": "Proline switch (PDCD5)",
    "Q6INL9":     "HAUS augmin (spindle)",
    "Q9H3R5":     "Centromere protein H",
    "P33187":     "Metallothionein",
    # Mid-range
    "P62877":     "RBX1 (Cullin ligase)",
    "P06899":     "Histone H2B",
    # Success end (best EU > 0.85)
    "O60925":     "Prefoldin",
    "O95563":     "Pyruvate carrier",
}

# Curated list of well-known conformational-switch proteins
HIGHLIGHT_PROTEINS = {
    "P69441": "Adenylate kinase",
    "P63317": "Troponin C",
    "P02633": "Calbindin D9k",
    "P00044": "Cytochrome c",
    "P23297": "S100A1",
    "Q02054": "Barstar",
    "P62152": "Calmodulin",
    "Q84BQ9": "L11 MTase",
    "Q5SL35": "CMP kinase",
    "P02693": "FABP",
    "Q02MM2": "Cas6/Csy4",
    "P74903": "V-ATPase F",
    "P61626": "Lysozyme C",
    "P0A7M6": "uL29",
    "O06976": "HPr-like Crh",
}


def _load_sim(eid_dir: Path, prefix: str,
              combine: str = "mean") -> np.ndarray | None:
    # Try standard ab/ba naming first
    ab = eid_dir / f"tm_{prefix}_ab.npy"
    ba = eid_dir / f"tm_{prefix}_ba.npy"
    if ab.exists() and ba.exists():
        a, b = np.load(ab), np.load(ba)
        if a.shape == b.shape and a.size > 0:
            if combine == "max":
                return np.maximum(a, b)
            return (a + b) / 2.0
    # Try norm_row/norm_col naming (WILL_85_50 format)
    nr = eid_dir / f"tm_{prefix}_norm_row.npy"
    nc = eid_dir / f"tm_{prefix}_norm_col.npy"
    if nr.exists() and nc.exists():
        a, b = np.load(nr), np.load(nc)
        if a.shape == b.shape and a.size > 0:
            if combine == "max":
                return np.maximum(a, b)
            return (a + b) / 2.0
    return None


def _nature_rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


DATASET = "UNICORNE_BENCH" if (BASE / "results" / "UNICORNE_BENCH").exists() else "UNICORNE"
TM_COMBINE = "max"  # normalize TM to shorter chain


def _recall_from_xy(sim_xy: np.ndarray, threshold: float = 0.9) -> float:
    """Compute recall: fraction of GT conformations with at least one pred above threshold."""
    if sim_xy.shape[0] == 0 or sim_xy.shape[1] == 0:
        return 0.0
    return float(np.mean(sim_xy.max(axis=1) >= threshold))


def _eu_from_xy(sim_xy: np.ndarray) -> float:
    """Ensemble Union: mean best-match TM across GT conformations."""
    if sim_xy.shape[0] == 0 or sim_xy.shape[1] == 0:
        return 0.0
    return float(sim_xy.max(axis=1).mean())


AF2_CUTOFF = "2018-05-01"  # AlphaFold2 PDB training-set cutoff
UNICORNE_SOURCE = Path(
    "/home/vanderw/conformational-ensemble-database/databaseV1/aligned-pdbs-new"
)


UNIPROT_PDB_IDS = DATA / "inputs" / DATASET / "uniprot_pdb_ids.csv"


def _deposited_pdb_ids(uniprots) -> dict[str, set[str]]:
    """{uniprot: IDs of every deposited PDB entry}, read from the UNICORNE
    source tree when it is available, else from the exported
    uniprot_pdb_ids.csv (columns: uniprot, pdb_ids ';'-separated)."""
    out: dict[str, set[str]] = {}
    if UNICORNE_SOURCE.exists():
        for u in uniprots:
            d = UNICORNE_SOURCE / u
            if d.is_dir():
                out[u] = {f.name.split("_")[0].upper() for f in d.glob("*-aligned.pdb")}
    elif UNIPROT_PDB_IDS.exists():
        with open(UNIPROT_PDB_IDS) as fh:
            for row in csv.DictReader(fh):
                out[row["uniprot"]] = set(filter(None, row["pdb_ids"].split(";")))
    return out


def _load_af2_ood(uniprots) -> set[str]:
    """Mark a uniprot as AF2-OOD if every PDB structure for it was deposited
    after the AF2 training cutoff (2018-05-01).  Uniprots whose dates can't
    be looked up are treated as in-distribution (conservative)."""
    import json
    dates_file = BASE / "data" / "pdb_deposition_dates.json"
    if not dates_file.exists():
        return set()
    dates = json.load(open(dates_file))
    pdb_ids_by_uniprot = _deposited_pdb_ids(uniprots)
    ood = set()
    for u in uniprots:
        pdb_ids = pdb_ids_by_uniprot.get(u)
        if not pdb_ids:
            continue
        deposit = sorted(d for d in (dates.get(p) for p in pdb_ids) if d)
        if not deposit:
            continue
        if deposit[0][:10] >= AF2_CUTOFF:
            ood.add(u)
    return ood


_METRIC_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def _load_metric_csv(tool: str, base: Path,
                      metric: str = "eu") -> dict[str, float]:
    """Read pre-computed per-ensemble metric values from results.csv.
    metric ∈ {eu, dr, cr, emd}. Cached per (tool, metric)."""
    key = (tool, metric)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]
    out: dict[str, float] = {}
    f = base / tool / "results.csv"
    if not f.exists():
        _METRIC_CACHE[key] = out; return out
    import math
    with open(f) as fh:
        for r in csv.DictReader(fh):
            eid = r.get("ensemble_ID", "").strip()
            v = r.get(metric, "")
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(x): continue
            out[eid] = x
    _METRIC_CACHE[key] = out
    return out


def _metric_from_csv(eid: str, tool: str, base: Path,
                      metric: str = "eu") -> float | None:
    return _load_metric_csv(tool, base, metric).get(eid)


_PROTEIN_NAMES_CACHE: dict[str, str] | None = None


def _load_protein_names() -> dict[str, str]:
    """Load `data/inputs/UNICORNE_BENCH/protein_names.csv` (uniprot, name)."""
    global _PROTEIN_NAMES_CACHE
    if _PROTEIN_NAMES_CACHE is not None:
        return _PROTEIN_NAMES_CACHE
    out: dict[str, str] = {}
    f = DATA / "inputs" / DATASET / "protein_names.csv"
    if f.exists():
        with open(f) as fh:
            for r in csv.DictReader(fh):
                eid = (r.get("ensemble_ID") or "").strip()
                nm = (r.get("protein_name") or "").strip()
                if eid and nm:
                    out[eid] = nm
    _PROTEIN_NAMES_CACHE = out
    return out


def _short_name(name: str, max_chars: int = 28) -> str:
    """Trim a UniProt-style name to the first segment before parens / EC, and
    cap at `max_chars`. e.g.
    'Beta-glucanase (EC 3.2.1.73) (...)' -> 'Beta-glucanase'.
    """
    if not name:
        return ""
    head = name.split("(")[0].strip()
    head = head.split(",")[0].strip()
    # If still long, try splitting on common separators / take prefix.
    if len(head) > max_chars:
        head = head[: max_chars - 1].rstrip() + "…"
    return head


def _load_max_disp(eid_dir: Path) -> float | None:
    """Max over GT pairs of max per-residue CA displacement after Kabsch (Å)."""
    f = eid_dir / "max_disp_xx.npy"
    if not f.exists():
        return None
    mat = np.load(f)
    if mat.shape[0] < 2:
        return None
    mask = ~np.eye(mat.shape[0], dtype=bool)
    vals = mat[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else None


def collect_data(diversity: str = "min_tm", metric: str = "eu", tools=None):
    """Collect GT diversity and per-tool recall for all ensembles.

    diversity: 'min_tm' (default, original) or 'max_disp' (Å, max CA
    displacement after Kabsch — sensitive to small-region motions that
    whole-protein TM masks).
    """
    # Prefer the newest benchmarking CSV that covers the full 869-ensemble
    # UNICORNE_BENCH set.  -13 supersedes -12 (648 entries).
    bench_csv = DATA / "inputs" / DATASET / "benchmarking-dataset-13.csv"
    if not bench_csv.exists():
        bench_csv = DATA / "inputs" / DATASET / "benchmarking-dataset-12.csv"
    if bench_csv.exists():
        with open(bench_csv) as f:
            will_data = {row["uniprot"]: row for row in csv.DictReader(f)}
    else:
        with open(DATA / "inputs" / DATASET / "input_dataset.csv") as f:
            will_data = {
                (row.get("uniprot") or row.get("ensemble_ID")): row
                for row in csv.DictReader(f)
            }

    # Restrict to the paper-filtered subset if available so n matches Table 3.
    filt_csv = DATA / "inputs" / DATASET / "filtered_ensembles_v2.csv"
    if filt_csv.exists():
        keep = set()
        with open(filt_csv) as f:
            for r in csv.DictReader(f):
                eid = r.get("ensemble_ID", "").strip()
                if eid:
                    keep.add(eid)
        # Also drop fold-quality blacklist (artifactual / IDP / fibril GTs).
        bl_csv = DATA / "inputs" / DATASET / "fold_quality_blacklist.csv"
        if bl_csv.exists():
            for r in csv.DictReader(open(bl_csv)):
                keep.discard((r.get("uniprot") or "").strip())
        will_data = {k: v for k, v in will_data.items() if k in keep}

    base = RESULTS / DATASET
    # Also check gt/ for xx matrices
    gt_mat_dir = base / "gt" / "matrices_tmalign"

    af2_ood = _load_af2_ood(set(will_data))

    # Dedicated GT cache dir (also holds max_disp_xx.npy)
    gt_cache_dir = base / "gt_matrices_tmalign"

    records = []
    for uid, info in will_data.items():
        # Diversity metric: either min_pairwise_tm (default) or max CA
        # displacement after Kabsch.
        if diversity == "max_disp":
            div_val = _load_max_disp(gt_cache_dir / uid)
            if div_val is None:
                continue
        else:
            xx = None
            if gt_mat_dir.exists():
                xx = _load_sim(gt_mat_dir / uid, "xx", combine=TM_COMBINE)
            if xx is None:
                xx = _load_sim(gt_cache_dir / uid, "xx", combine=TM_COMBINE)
            if xx is None:
                for tool in TOOLS:
                    mat_dir = base / tool / "matrices_tmalign" / uid
                    xx = _load_sim(mat_dir, "xx", combine=TM_COMBINE)
                    if xx is not None:
                        break
            if xx is None or xx.shape[0] < 2:
                continue
            mask = ~np.eye(xx.shape[0], dtype=bool)
            div_val = float(xx[mask].min())

        # Per-tool metric value (eu / dr / cr) from canonical results.csv.
        tool_val: dict[str, float] = {}
        for tool in (tools or TOOLS):
            v = _metric_from_csv(uid, tool, base, metric=metric)
            if v is None and metric == "eu":
                # Legacy fallback: recompute EU from xy matrices for tools
                # that don't ship results.csv yet.
                mat_dir = base / tool / "matrices_tmalign" / uid
                sim_xy = _load_sim(mat_dir, "xy", combine=TM_COMBINE)
                if sim_xy is not None:
                    v = _eu_from_xy(sim_xy)
            if v is not None:
                tool_val[tool] = v
        if not tool_val:
            continue

        # "Best" = max for higher-is-better metrics (eu/dr/cr), min for emd.
        if metric == "emd":
            best_tool = min(tool_val, key=tool_val.get)
        else:
            best_tool = max(tool_val, key=tool_val.get)
        best_val = tool_val[best_tool]
        records.append({
            "uid": uid,
            "diversity": div_val,
            "best_eu": best_val,           # legacy field name; reused for any metric
            "best_tool": best_tool,
            "tool_eu": tool_val,
            "n_gt": int(info["num_clusters"]),
            "seq_len": int(info.get("seqlength", info.get("length", 0))),
            "highlight": uid in HIGHLIGHT_PROTEINS,
            "name": HIGHLIGHT_PROTEINS.get(uid, ""),
            "af2_ood": uid in af2_ood,
        })
    return records


def plot_scatter(records, output_path=None, diversity: str = "min_tm",
                 metric: str = "eu", ood_only: bool = False):
    """Scatter: GT diversity vs best per-tool metric.

    diversity='min_tm': x-axis is min pairwise TM (normalized to shorter chain).
    diversity='max_disp': x-axis is max CA displacement after Kabsch (Å).
    metric ∈ {eu, dr, cr}: y-axis metric (best across tools).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _nature_rc()

    # CR is binary 0/1 per ensemble — add a small deterministic y-jitter so
    # points don't all stack on two horizontal lines. Use uid hash so it's
    # stable across runs.
    import hashlib as _hl
    def _y_with_jitter(r):
        y = r["best_eu"]
        if metric == "cr":
            h = int(_hl.md5(r["uid"].encode()).hexdigest()[:6], 16)
            return y + (h / 0xFFFFFF - 0.5) * 0.08
        return y

    def _xy(rec_list):
        xs = np.array([r["diversity"] for r in rec_list])
        ys = np.array([_y_with_jitter(r) for r in rec_list])
        return xs, ys

    fig, ax = plt.subplots(figsize=(7, 7))

    def _draw(rec_subset, color, marker, *, size, alpha, lw, zorder, label=None):
        if not rec_subset:
            return
        xs, ys = _xy(rec_subset)
        ax.scatter(xs, ys, s=size, alpha=alpha, color=color,
                   edgecolors="black", linewidths=lw,
                   zorder=zorder, marker=marker, label=label)

    # Plot per-tool: each tool gets its own marker shape (matching the @k
    # and line plots elsewhere). AF2-OOD vs in-train is encoded by FILL:
    # solid (OOD, the interesting cases) vs translucent (in-train).
    # In OOD-only mode every point is OOD, so attach the tool label to the
    # OOD draw call instead of the in-train one.
    for tool in TOOLS:
        tool_recs = [r for r in records if r["best_tool"] == tool]
        if not tool_recs:
            continue
        color = _TOOL_PALETTE[tool]
        marker = _TOOL_MARKER.get(tool, "o")
        in_train = [r for r in tool_recs if not r["af2_ood"]]
        ood      = [r for r in tool_recs if r["af2_ood"]]
        tool_label = f"{TOOL_DISPLAY[tool]} (n={len(tool_recs)})"
        if ood_only:
            _draw(ood, color, marker, size=90, alpha=0.95, lw=0.5, zorder=3,
                  label=tool_label)
        else:
            _draw(in_train, color, marker, size=90, alpha=0.30, lw=0.5,
                  zorder=2, label=tool_label)
            _draw(ood, color, marker, size=90, alpha=0.95, lw=0.5, zorder=3)


    # Tie/zero-recall ties: any record where all tools tied (e.g. all 0). Already
    # assigned to whichever tool sorts first by `max`; that's fine for visuals.

    # Annotate highlighted proteins with solid background boxes.
    # Offsets tuned by hand so no labels overlap.
    offsets = {
        "P69441": (35, 12),       # Adenylate kinase
        "P63317": (15, -25),      # Troponin C
        "P02633": (-80, -20),     # Calbindin D9k
        "P00044": (20, -22),      # Cytochrome c
        "P23297": (-60, 15),      # S100A1
        "Q02054": (30, 15),       # Barstar
        "P62152": (-85, -8),      # Calmodulin
        "Q84BQ9": (25, 15),       # L11 MTase
        "Q5SL35": (-25, 18),      # CMP kinase
        "P02693": (-15, 20),      # FABP
        "Q02MM2": (-70, -18),     # Cas6/Csy4
        "P74903": (-70, -5),      # V-ATPase F
        "P61626": (15, 18),       # Lysozyme C
        "P0A7M6": (20, -20),      # uL29
        "O06976": (-80, 8),       # HPr-like Crh
    }
    for r in records:
        if not r["highlight"]:
            continue
        ox, oy = offsets.get(r["uid"], (15, 0))
        ax.annotate(
            r["name"], (r["diversity"], r["best_eu"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=13, color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffde7",
                      edgecolor="#ccc59e", linewidth=0.5, alpha=0.95),
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3,rad=0.1",
                color="black", lw=0.6,
                shrinkA=0, shrinkB=3,
            ),
            zorder=10,
        )

    # In OOD-only mode, label a curated handful of interesting OOD proteins
    # with the same yellow-callout style as the main highlight set.
    if ood_only:
        # Manually-tuned offsets so the callouts don't collide.
        ood_offsets = {
            "A0A125YJ71": (32,  10),     # bottom-left
            "Q6INL9":     (35,  -8),     # mid-left
            "Q9H3R5":     (-105, -8),    # right side
            "P33187":     (32, -10),     # left-mid
            "P62877":     (-100, 10),    # left-mid
            "P06899":     (28, -12),     # mid-right
            "O60925":     (-110,  6),    # mid-right (high EU)
            "O95563":     (28,  10),     # high-EU
        }
        for r in records:
            if r["uid"] not in OOD_HIGHLIGHT_PROTEINS:
                continue
            if r["highlight"]:
                continue   # already labeled via the main highlight pass
            ox, oy = ood_offsets.get(r["uid"], (28, 10))
            ax.annotate(
                OOD_HIGHLIGHT_PROTEINS[r["uid"]],
                (r["diversity"], r["best_eu"]),
                xytext=(ox, oy), textcoords="offset points",
                fontsize=13, color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffde7",
                          edgecolor="#ccc59e", linewidth=0.5, alpha=0.95),
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.1",
                    color="black", lw=0.6,
                    shrinkA=0, shrinkB=3,
                ),
                zorder=10,
            )

    # Per-metric y-axis label, range, and reference lines.
    if metric == "eu":
        ax.axhspan(0.9, 1.02, facecolor="#e8f5e9", edgecolor="none",
                   alpha=1.0, zorder=0)
        ax.axhline(0.9, ls=":", color="#888888", lw=0.6, alpha=0.6)
        ax.axhline(0.85, ls="--", color="#888888", lw=0.7, alpha=0.7)
        ax.text(0.01, 0.857, "GT-GT max TM = 0.85", fontsize=16,
                color="#555555", alpha=0.85,
                transform=ax.get_yaxis_transform())
        ax.set_ylabel("Best mean best-match TM across tools")
        ax.set_ylim(0.3, 1.02)
    elif metric == "dr":
        ax.set_ylabel("Best Recall (fraction of GT states recovered) across tools")
        ax.set_ylim(-0.04, 1.04)
        ax.axhline(1.0, ls=":", color="#888888", lw=0.6, alpha=0.6)
    elif metric == "cr":
        # FRR is per-ensemble binary {0, 1}; jittered for visibility.
        ax.set_ylabel("Best FRR (any tool fully recovers all GT states) across tools")
        ax.set_ylim(-0.15, 1.15)
        ax.axhspan(0.85, 1.15, facecolor="#e8f5e9", edgecolor="none",
                   alpha=0.7, zorder=0)
        ax.axhline(1.0, ls=":", color="#888888", lw=0.6, alpha=0.6)
        ax.axhline(0.0, ls=":", color="#888888", lw=0.6, alpha=0.6)
        ax.text(0.01, 1.02, "complete recovery", fontsize=16,
                color="#2a6b2a", alpha=0.85,
                transform=ax.get_yaxis_transform())
        ax.text(0.01, -0.02, "no GT fully recovered", fontsize=16,
                color="#555555", alpha=0.85,
                transform=ax.get_yaxis_transform())
    elif metric == "emd":
        ax.set_ylabel("Best (lowest) EMD across tools")
        ax.set_ylim(0.0, 1.0)
    ax.set_box_aspect(1)
    if diversity == "max_disp":
        # Log-ish x: max CA disp ranges from ~0 A (near-identical) to >20 A
        # (large domain motion). Use linear axis, tight to data.
        xs_all = np.array([r["diversity"] for r in records])
        xmax = float(np.nanpercentile(xs_all, 99)) * 1.05 if xs_all.size else 20.0
        ax.set_xlabel("GT conformational diversity (max CA displacement after Kabsch, \u00c5)")
        ax.set_xlim(-0.2, max(xmax, 5.0))
        ax.text(0.02, 0.907, "TM = 0.9", fontsize=16, color="#555555", alpha=0.8,
                transform=ax.get_yaxis_transform())
    else:
        ax.set_xlabel("GT diversity (min pairwise TM)", fontsize=16)
        ax.set_xlim(-0.02, 1.02)
        ax.text(0.01, 0.907, "TM = 0.9", fontsize=16, color="#555555", alpha=0.8)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.label.set_size(13)

    # Combined legend: per-tool marker shape (color + symbol), plus
    # solid-vs-light fill key for AF2-OOD.
    handles, labels = ax.get_legend_handles_labels()
    n_ood = sum(1 for r in records if r["af2_ood"])
    n_in_train = sum(1 for r in records if not r["af2_ood"])
    # Fill-convention key only when both classes are visible \u2014 in OOD-only
    # mode every point is OOD so the legend just lists tools.
    if not ood_only:
        handles.append(Line2D([], [], marker="o", color="w",
                              markerfacecolor=(0.55, 0.55, 0.55, 0.30),
                              markeredgecolor="black",
                              markersize=7, linewidth=0,
                              label=f"in AF2 train (n={n_in_train})"))
        handles.append(Line2D([], [], marker="o", color="w",
                              markerfacecolor="#555555", markeredgecolor="black",
                              markersize=7, linewidth=0,
                              label=f"AF2-OOD (n={n_ood})"))
    # Legend wraps across multiple rows so it fits under the square axes
    # without stretching the figure horizontally. With 8 tools (+ the two
    # AF2-OOD fill keys) that's 3 cols × 3-4 rows; 4 cols overflow the canvas.
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=True, framealpha=0.95, fontsize=12,
              borderaxespad=0.4, columnspacing=1.0,
              handletextpad=0.4)

    # Reserve room for the legend inside the 7x7 canvas so the saved PNG
    # has a 1:1 aspect ratio (matches Fig 8b).  We deliberately do NOT use
    # bbox_inches='tight' on save — that would re-crop and make the
    # canvas non-square.
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.32)
    out = output_path or FIGURES / "unicorne_gold_standard.png"
    fig.savefig(str(out), dpi=300, facecolor="white")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--diversity", choices=["min_tm", "max_disp"], default="min_tm",
                   help="X-axis diversity metric")
    p.add_argument("--metric", choices=["eu", "dr", "cr", "emd"], default="eu",
                   help="Y-axis metric (best across tools)")
    p.add_argument("--ood-only", action="store_true",
                   help="Restrict the scatter to AF2-OOD ensembles only "
                        "(every PDB for that UniProt deposited at or after "
                        "the AF2 training cutoff)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default depends on metric/diversity)")
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    records = collect_data(diversity=args.diversity, metric=args.metric,
                           tools=AF2_OOD_TOOLS if args.ood_only else None)
    if args.ood_only:
        records = [r for r in records if r["af2_ood"]]
    print(f"Collected {len(records)} ensembles, "
          f"{sum(1 for r in records if r['highlight'])} highlighted "
          f"(diversity={args.diversity}, metric={args.metric}, "
          f"ood_only={args.ood_only})")
    out = args.output
    if out is None:
        suffix = "_max_disp" if args.diversity == "max_disp" else ""
        metric_tag = "" if args.metric == "eu" else f"_{args.metric}"
        ood_tag = "_ood" if args.ood_only else ""
        out = FIGURES / f"unicorne_gold_standard{metric_tag}{ood_tag}{suffix}.png"
    plot_scatter(records, output_path=out, diversity=args.diversity,
                 metric=args.metric, ood_only=args.ood_only)
