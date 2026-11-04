#!/usr/bin/env python3
"""
KDE + scatter of per-prediction TM scores against the two GT states of a
hand-picked multi-state UNICORNE_BENCH protein, for two tools side-by-side
(best MSA-perturbation + best generative). PyMOL ribbon callouts mark each
GT structure.

Axes:
  x = TM(pred, GT_A)
  y = TM(pred, GT_B)
  GT_A sits at (1, gt_ab_TM); GT_B sits at (gt_ab_TM, 1).

Inputs (per protein/eid):
  results/UNICORNE_BENCH/<tool>/tmalign_raw/<eid>.json
  results/UNICORNE_BENCH/gt_matrices_tmalign/<eid>/tm_xx.npy

Output:
  figures/kde_mode_collapse/<eid>_<name>.png

Usage:
  python scripts/plot_kde_gt_callouts.py
  python scripts/plot_kde_gt_callouts.py --proteins P69441 Q5SL35
  python scripts/plot_kde_gt_callouts.py --tools bioemu afsample2 --thresh 0.3
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "UNICORNE_BENCH"

# Upstream "all-PDB-per-UniProt" mirror — every deposited structure for the
# UniProt, not just the 2-3 dedupped representatives in our benchmark. Used
# to build a richer GT-density landscape for the t-SNE KDE.
UPSTREAM_GT_DB = Path("/home/vanderw/conformational-ensemble-database/"
                     "databaseV1/aligned-pdbs-new")

# Curated multi-state proteins with n_gt=2 in the dataset.
# Names match the gold-standard table.
PROTEINS_DEFAULT = [
    ("P69441", "Adenylate kinase"),
    ("Q5SL35", "CMP kinase"),
    ("A0A125YJ71", "Proline switch"),
]

TOOL_DISPLAY = {
    "bioemu":        "BioEMU",
    "esmflow_pdb":   "ESMFlow",
    "esmdiff":       "ESMDiff",
    "alphaflow_pdb": "AlphaFlow",
    "cfrandom":      "CFrandom",
    "afsample2":     "AFsample2",
    "speach_af":     "SPEACH_AF",
}
TOOL_COLOR = {
    "bioemu":        "#C0392B",   # warm = generative (matches paper scheme)
    "esmflow_pdb":   "#E67E22",
    "esmdiff":       "#F1C40F",
    "alphaflow_pdb": "#E74C3C",
    "cfrandom":      "#1F4E79",   # cool = MSA perturbation
    "afsample2":     "#2980B9",
    "speach_af":     "#5DADE2",
}
# Ribbon colors for the GT structures — bright, saturated, easy to spot
# against the blue/red heat-KDE. Avoid yellow tones since the GT stars use
# a yellow border for contrast.
GT_COLOR_HEX = [
    "#FF00B4",   # fuchsia
    "#8A2BE2",   # bright purple
    "#00C5FF",   # bright cyan
    "#32D04A",   # lime green
    "#FF7000",   # vivid orange
    "#FF1F4F",   # hot red-pink
    "#1F4FFF",   # cobalt blue
    "#A0522D",   # sienna (last resort, brown — distinct from above)
]


def _pymol_color_name(hex_color: str) -> str:
    """Define a custom PyMOL color, return its name."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return f"set_color custom_clr, [{r:.3f}, {g:.3f}, {b:.3f}]; color custom_clr"


def render_ribbon_pymol(pdb_path: Path, color_hex: str, out_png: Path,
                         size: int = 600) -> None:
    """Render a structure as a clean cartoon ribbon with PyMOL."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    color_cmd = _pymol_color_name(color_hex)
    pml = f"""
load {pdb_path}, m
hide everything
show cartoon
{color_cmd}
bg_color white
set cartoon_smooth_loops, 1
set cartoon_loop_quality, 30
set cartoon_fancy_helices, 1
set ray_shadow, 0
set ray_opaque_background, off
set antialias, 2
orient
zoom complete=1
ray {size}, {size}
png {out_png}
quit
"""
    res = subprocess.run(
        ["pymol", "-cq", "-d", pml],
        capture_output=True, text=True
    )
    if not out_png.exists():
        raise RuntimeError(f"PyMOL render failed for {pdb_path}:\n{res.stderr}")


def crop_to_content(img_path: Path) -> np.ndarray:
    """Trim white margins from a rendered ribbon for tighter insets."""
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    # Treat as content where R/G/B not all > 240
    rgb = arr[..., :3]
    mask = ~((rgb[..., 0] > 240) & (rgb[..., 1] > 240) & (rgb[..., 2] > 240))
    if not mask.any():
        return arr
    rows = mask.any(axis=1).nonzero()[0]
    cols = mask.any(axis=0).nonzero()[0]
    pad = 6
    r0 = max(0, rows.min() - pad); r1 = min(arr.shape[0], rows.max() + pad)
    c0 = max(0, cols.min() - pad); c1 = min(arr.shape[1], cols.max() + pad)
    return arr[r0:r1, c0:c1]


def _tm_max2(p: dict) -> float | None:
    a, b = p.get("tm_norm_chain1"), p.get("tm_norm_chain2")
    try:
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b): return None
        return max(a, b)
    except (TypeError, ValueError):
        return None


def per_pred_tm_2d(eid: str, tool: str) -> tuple[np.ndarray, str, str] | None:
    """Return (arr[n_pred, 2], gt_a_path, gt_b_path) or None if not n_gt==2."""
    res = per_pred_tm_nd(eid, tool, expect_n_gt=2)
    if res is None: return None
    arr, gts = res
    return arr, gts[0], gts[1]


def per_pred_tm_nd(eid: str, tool: str,
                   expect_n_gt: int | None = None
                   ) -> tuple[np.ndarray, list[str]] | None:
    """Return (arr[n_pred, n_gt], gt_paths) where columns are sorted by GT path.

    Drops predictions that are missing a TM to any GT (so the matrix is dense).
    Returns None if the JSON is missing, or if expect_n_gt is set and doesn't
    match the file's n_gt.
    """
    jf = PROJECT_ROOT / "results" / DATASET / tool / "tmalign_raw" / f"{eid}.json"
    if not jf.exists():
        return None
    d = json.load(open(jf))
    n_gt = d.get("n_gt")
    if expect_n_gt is not None and n_gt != expect_n_gt:
        return None
    pairs = d.get("pairs") or []
    gts = sorted({p["gt"] for p in pairs})
    if len(gts) < 2:
        return None
    by_pred: dict[str, dict[str, float]] = {}
    for p in pairs:
        t = _tm_max2(p)
        if t is None: continue
        by_pred.setdefault(p["pred"], {})[p["gt"]] = t
    rows = []
    for ts in by_pred.values():
        if all(g in ts for g in gts):
            rows.append([ts[g] for g in gts])
    if not rows:
        return None
    return np.array(rows, dtype=float), gts


def gt_gt_tm_matrix(eid: str) -> np.ndarray | None:
    """Full (n_gt, n_gt) GT-GT TM matrix under max-of-2 normalization.

    Diagonal is 1.0; off-diagonal entries are the GT-GT TMs.
    """
    src = PROJECT_ROOT / "results" / DATASET / "gt_matrices_tmalign" / eid
    f = src / "tm_xx.npy"
    if f.exists():
        return np.load(f)
    return None


def gt_gt_tm_max2(eid: str) -> float | None:
    """GT-GT TM (single off-diagonal value) for n_gt==2 ensembles."""
    mat = gt_gt_tm_matrix(eid)
    if mat is None or mat.shape != (2, 2):
        return None
    return float(mat[0, 1])


def gt_gt_tm_pair_dict(eid: str) -> dict[tuple[str, str], float]:
    """Return a dict {(gt_i_path, gt_j_path): TM_max2} from the GT-GT JSON.

    The GT-GT JSON schema differs from the GT-pred one:
      - `conformation_names` lists each GT structure's filename
      - `pairs` is a dict keyed by "i_j" → {tm_max, t1_ab, t2_ab, ...}
      - tm_max is the max-of-2 TM value we want.

    Keys are the FILENAME ONLY (not the full path), so lookups must
    use Path(...).name. Both (a, b) and (b, a) orientations populated.
    """
    jf = (PROJECT_ROOT / "results" / DATASET / "gt_matrices_tmalign" / eid
          / "raw_tmalign.json")
    out: dict[tuple[str, str], float] = {}
    if not jf.exists():
        return out
    d = json.load(open(jf))
    names = d.get("conformation_names") or []
    pairs = d.get("pairs") or {}
    if not isinstance(pairs, dict):
        return out
    for key, p in pairs.items():
        try:
            i, j = (int(x) for x in key.split("_"))
        except (ValueError, AttributeError):
            continue
        if i >= len(names) or j >= len(names):
            continue
        t = p.get("tm_max")
        if t is None:
            continue
        a, b = names[i], names[j]
        out[(a, b)] = float(t)
        out[(b, a)] = float(t)
    return out


def plot_protein(eid: str, name: str, tools: list[str],
                 thresh: float, out_dir: Path, ribbon_dir: Path) -> Path | None:
    # Load per-tool data; drop tools with missing or wrong-shape data.
    data = {}
    gt_paths = None
    for t in tools:
        result = per_pred_tm_2d(eid, t)
        if result is None:
            print(f"  [{eid}] {t}: no n_gt==2 data — skip"); continue
        arr, a, b = result
        data[t] = arr
        gt_paths = (a, b)
    if not data or gt_paths is None:
        print(f"  [{eid}] no usable tool data — skipping")
        return None

    a_path, b_path = gt_paths
    g_tm = gt_gt_tm_max2(eid)
    if g_tm is None:
        print(f"  [{eid}] no GT-GT TM available — skipping")
        return None

    # Render PyMOL ribbons
    img_a = ribbon_dir / f"{eid}_GTa_{Path(a_path).stem}.png"
    img_b = ribbon_dir / f"{eid}_GTb_{Path(b_path).stem}.png"
    if not img_a.exists():
        print(f"    rendering {img_a.name}")
        render_ribbon_pymol(Path(a_path), GT_COLOR_HEX[0], img_a)
    if not img_b.exists():
        print(f"    rendering {img_b.name}")
        render_ribbon_pymol(Path(b_path), GT_COLOR_HEX[1], img_b)
    img_a_arr = crop_to_content(img_a)
    img_b_arr = crop_to_content(img_b)

    # Plot — give the figure margin around the axes so the ribbon insets
    # can sit outside the data area without overlapping points.
    fig = plt.figure(figsize=(9.5, 8.5))
    ax = fig.add_axes([0.10, 0.08, 0.62, 0.72])  # leave room above + right

    # Auto-fit the visible range to the actual data + GT markers (the
    # GT_B marker is at (g_tm, 1.0), GT_A at (1.0, g_tm); when GTs are
    # very dissimilar, like Lysozyme C with GT-GT TM=0.18, the data
    # cluster sits at extreme low TM to one of them and a fixed 0.3
    # display floor would clip it off-screen).
    all_pts = np.vstack(list(data.values()))
    data_min = float(all_pts.min())
    lo_floor = thresh
    auto_lo = min(data_min, g_tm) - 0.05
    lo = max(0.0, min(lo_floor, auto_lo))
    hi = 1.0
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")

    # KDE + scatter per tool — KDE grid uses the auto-fit lo..hi range
    grid_n = 200
    for t, arr in data.items():
        c = TOOL_COLOR[t]
        try:
            kde = gaussian_kde(arr.T, bw_method=0.20)
            xx, yy = np.mgrid[lo:hi:complex(0, grid_n),
                              lo:hi:complex(0, grid_n)]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            levels = np.linspace(zz.min(), zz.max(), 8)[1:]
            ax.contourf(xx, yy, zz, levels=levels, colors=[c]*len(levels),
                        alpha=0.10)
            ax.contour(xx, yy, zz, levels=levels[::2], colors=c,
                       alpha=0.5, linewidths=0.7)
        except Exception:
            pass
        ax.scatter(arr[:, 0], arr[:, 1], s=10, c=c, alpha=0.40,
                   label=TOOL_DISPLAY.get(t, t),
                   edgecolors="none")

    # Diagonal y=x reference
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":",
            linewidth=0.7, alpha=0.5)

    # GT markers
    ax.scatter([1.0], [g_tm], marker="*", s=350, c="white",
               edgecolors=GT_COLOR_HEX[0], linewidths=2.0, zorder=10)
    ax.scatter([g_tm], [1.0], marker="*", s=350, c="white",
               edgecolors=GT_COLOR_HEX[1], linewidths=2.0, zorder=10)

    # PyMOL ribbon insets placed OUTSIDE the data axes so they don't cover
    # points: GT_A (right-edge) → inset to the right of the plot;
    # GT_B (top-edge)   → inset above the plot.
    target_size = 150
    for img_arr, hex_c, gt_xy, corner_frac, label in [
        (img_a_arr, GT_COLOR_HEX[0], (1.0, g_tm), (1.30, 0.50),
         Path(a_path).stem),
        (img_b_arr, GT_COLOR_HEX[1], (g_tm, 1.0), (0.25, 1.28),
         Path(b_path).stem),
    ]:
        h, w = img_arr.shape[:2]
        zoom = target_size / max(h, w)
        oi = OffsetImage(img_arr, zoom=zoom, alpha=1.0)
        ab = AnnotationBbox(
            oi, xy=gt_xy, xybox=corner_frac,
            xycoords="data", boxcoords="axes fraction",
            frameon=False, clip_on=False,
            arrowprops=dict(arrowstyle="-", linestyle="dashed",
                            color=hex_c, linewidth=1.2),
            zorder=20,
        )
        ax.add_artist(ab)
        # PDB stem label just below the inset (axes-fraction coords).
        ax.annotate(label, xy=(corner_frac[0], corner_frac[1] - 0.16),
                    xycoords="axes fraction",
                    fontsize=16, color=hex_c, ha="center",
                    fontweight="bold")

    ax.set_xlabel(f"TM(pred → GT_A = {Path(a_path).stem})")
    ax.set_ylabel(f"TM(pred → GT_B = {Path(b_path).stem})")
    ax.legend(loc="lower left", fontsize=16, framealpha=0.9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eid}_{name.lower().replace(' ', '_')}_kde.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(str(out_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def plot_protein_3gt(eid: str, name: str, tools: list[str],
                     thresh: float, out_dir: Path, ribbon_dir: Path) -> Path | None:
    """3-pair scatter+KDE figure for an n_gt==3 ensemble.

    Layout: a row of 3 PyMOL ribbon panels along the top, and a row of 3
    pairwise (TM_i, TM_j) panels below them. Each pair panel has GT-marker
    stars at the (1, gt_ij_TM) and (gt_ij_TM, 1) positions, KDE+scatter for
    each tool overlaid, and an auto-fit equal-aspect axis range.
    """
    data: dict[str, np.ndarray] = {}
    gts: list[str] | None = None
    for t in tools:
        result = per_pred_tm_nd(eid, t, expect_n_gt=3)
        if result is None:
            print(f"  [{eid}] {t}: no n_gt==3 data — skip"); continue
        arr, g = result
        if gts is None:
            gts = g
        elif g != gts:
            # Sort order should match across tools (path-sorted), but be safe.
            print(f"  [{eid}] {t}: GT path order mismatch — skip"); continue
        data[t] = arr
    if not data or gts is None:
        print(f"  [{eid}] no usable n_gt==3 tool data"); return None

    pair_tm = gt_gt_tm_pair_dict(eid)
    if not pair_tm:
        print(f"  [{eid}] no GT-GT TM pair JSON — skipping"); return None

    # Render PyMOL ribbons (one per GT)
    ribbon_imgs: list[np.ndarray] = []
    for i, g in enumerate(gts):
        out_png = ribbon_dir / f"{eid}_GT{i}_{Path(g).stem}.png"
        if not out_png.exists():
            print(f"    rendering {out_png.name}")
            render_ribbon_pymol(Path(g), GT_COLOR_HEX[i], out_png)
        ribbon_imgs.append(crop_to_content(out_png))

    # ---- Layout ----
    fig = plt.figure(figsize=(15.5, 9.0))
    # Top row: 3 ribbon insets
    ribbon_y = 0.72
    ribbon_h = 0.24
    ribbon_w = 0.20
    for i, img in enumerate(ribbon_imgs):
        ax_r = fig.add_axes([0.04 + i * 0.32, ribbon_y, ribbon_w, ribbon_h])
        ax_r.imshow(img, interpolation="bilinear")
        ax_r.set_xticks([]); ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_visible(False)
    # Bottom row: 3 pair panels (a-b, a-c, b-c)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_axes([0.06 + k * 0.32, 0.08, 0.27, 0.55])
        gt_i_path, gt_j_path = gts[i], gts[j]
        # GT-GT JSON keys are filenames only; per-tool JSONs use full paths.
        gt_ij_tm = pair_tm.get((Path(gt_i_path).name, Path(gt_j_path).name), 0.0)
        # Auto-fit axes
        xs = np.concatenate([data[t][:, i] for t in tools if t in data])
        ys = np.concatenate([data[t][:, j] for t in tools if t in data])
        data_min = float(min(xs.min(), ys.min()))
        lo = max(0.0, min(thresh, min(data_min, gt_ij_tm) - 0.05))
        hi = 1.0
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        # KDE + scatter per tool
        grid_n = 150
        for t, arr in data.items():
            c = TOOL_COLOR[t]
            xy = arr[:, [i, j]]
            try:
                kde = gaussian_kde(xy.T, bw_method=0.20)
                xx, yy = np.mgrid[lo:hi:complex(0, grid_n),
                                  lo:hi:complex(0, grid_n)]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                levels = np.linspace(zz.min(), zz.max(), 8)[1:]
                ax.contourf(xx, yy, zz, levels=levels,
                            colors=[c]*len(levels), alpha=0.10)
                ax.contour(xx, yy, zz, levels=levels[::2], colors=c,
                           alpha=0.5, linewidths=0.7)
            except Exception:
                pass
            ax.scatter(xy[:, 0], xy[:, 1], s=8, c=c, alpha=0.40,
                       label=TOOL_DISPLAY.get(t, t),
                       edgecolors="none")
        # y=x reference + GT markers
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":",
                linewidth=0.7, alpha=0.5)
        ax.scatter([1.0], [gt_ij_tm], marker="*", s=240, c="white",
                   edgecolors=GT_COLOR_HEX[i], linewidths=2.0, zorder=10)
        ax.scatter([gt_ij_tm], [1.0], marker="*", s=240, c="white",
                   edgecolors=GT_COLOR_HEX[j], linewidths=2.0, zorder=10)
        ax.set_xlabel(f"TM(pred → GT {chr(65+i)})", fontsize=16)
        ax.set_ylabel(f"TM(pred → GT {chr(65+j)})", fontsize=16)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
        if k == 0:
            ax.legend(loc="lower left", fontsize=16, framealpha=0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eid}_{name.lower().replace(' ', '_')}_n3_kde.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(str(out_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def _tmalign_pair(args: tuple[int, int, str, str]) -> tuple[int, int, float]:
    """Worker: run TMalign on (pdb_a, pdb_b), return max-of-2 TM."""
    i, j, pdb_a, pdb_b = args
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from conformation_benchmark.evaluation.tmalign_exe import _run_tmalign
    try:
        tm1, tm2, _ = _run_tmalign(Path(pdb_a), Path(pdb_b))
        return i, j, max(tm1, tm2)
    except Exception:
        return i, j, float("nan")


_METRIC_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def _ensemble_metric(eid: str, tool: str, metric: str = "emd") -> float | None:
    """Look up a per-ensemble metric (emd / dr / cr / tm_bar / ...) from a tool's
    results.csv. Cached per (tool, metric) so repeat lookups are free."""
    import csv as _csv, math
    key = (tool, metric)
    if key not in _METRIC_CACHE:
        d: dict[str, float] = {}
        f = (PROJECT_ROOT / "results" / DATASET / tool / "results.csv")
        if f.exists():
            with open(f) as fh:
                for r in _csv.DictReader(fh):
                    e = r.get("ensemble_ID", "").strip()
                    v = r.get(metric, "")
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not math.isnan(x):
                        d[e] = x
        _METRIC_CACHE[key] = d
    return _METRIC_CACHE[key].get(eid)


def _gt_label(path: str) -> str:
    """Strip chain and '-aligned' suffix from a GT PDB filename for display.

    e.g. '1AKE_A-aligned' → '1AKE'.
    """
    stem = Path(path).stem
    # Strip "-aligned" if present
    if stem.endswith("-aligned"):
        stem = stem[: -len("-aligned")]
    # Strip the chain suffix "_<X>" if it's a single uppercase letter / digit
    if "_" in stem:
        head, tail = stem.rsplit("_", 1)
        if 1 <= len(tail) <= 2 and tail.isalnum():
            stem = head
    return stem


def _list_upstream_gt_pdbs(eid: str) -> list[str]:
    """All deposited aligned PDB/CIF structures for a UniProt (not just our
    dedupped representatives). Returns sorted absolute paths.

    Returns an empty list if the upstream directory is missing.
    """
    d = UPSTREAM_GT_DB / eid
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".pdb", ".cif"):
            continue
        # Prefer .cif when both are present for the same stem (avoid duplicates)
        if p.suffix.lower() == ".pdb" and (p.with_suffix(".cif")).exists():
            continue
        out.append(str(p))
    return out


def plot_protein_tsne(eid: str, name: str, tools: list[str],
                      out_dir: Path, ribbon_dir: Path,
                      n_per_tool: int = 80, perplexity: float = 30.0,
                      seed: int = 42, workers: int = 16,
                      max_extras: int = 50) -> Path | None:
    """t-SNE embedding of GTs + subsampled predictions, KDE per tool.

    Builds a full pairwise TM-distance matrix over (GTs ∪ subsampled preds)
    and t-SNE-embeds it in 2D. Pred-pred TMs are computed on the fly via
    parallel TMalign; pred-GT and GT-GT TMs are pulled from cached JSON/npy.
    """
    # 1) Subsample preds per tool (deterministic per (eid, tool, seed))
    pred_records: list[tuple[str, str]] = []  # (tool, pdb_path)
    for t in tools:
        outputs_dir = (PROJECT_ROOT / "data" / "outputs" / t / DATASET
                       / "output_pdbs" / eid)
        if not outputs_dir.exists():
            print(f"  [{eid}] {t}: no preds dir — skip"); continue
        all_preds = sorted(str(p) for p in outputs_dir.iterdir()
                           if p.suffix.lower() == ".pdb")
        if not all_preds:
            print(f"  [{eid}] {t}: no .pdb preds — skip"); continue
        # Deterministic per-(seed, tool) RNG. Python's built-in hash() is
        # randomized per-process, so use a stable hash.
        import hashlib as _hl
        tool_seed = int(_hl.md5(t.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed ^ tool_seed)
        if len(all_preds) > n_per_tool:
            idx = rng.choice(len(all_preds), n_per_tool, replace=False)
            picked = [all_preds[i] for i in sorted(idx)]
        else:
            picked = all_preds
        for p in picked:
            pred_records.append((t, p))
    if not pred_records:
        print(f"  [{eid}] no preds across any tool"); return None

    # 2) GT structure paths.
    #    Canonical = the 2-3 dedupped representatives in our benchmark.
    #    Extras    = every other deposited PDB for this UniProt, pulled from
    #                the upstream all-PDB mirror. They flesh out the GT-KDE
    #                landscape (otherwise we'd have only 2-3 points to base
    #                a density on, which gives a near-delta blob).
    gt_dir = PROJECT_ROOT / "data" / "inputs" / DATASET / "input_pdbs" / eid
    canonical_gt_paths = sorted(str(p) for p in gt_dir.iterdir()
                                if p.suffix.lower() in (".pdb", ".cif"))
    n_canonical = len(canonical_gt_paths)
    if n_canonical < 2:
        print(f"  [{eid}] need ≥2 canonical GTs, found {n_canonical}")
        return None
    # Use only the canonical (benchmark) GTs — extras dropped.
    # Building a density from 5–250 random PDBs is meaningless; instead the
    # KDE is now built from the predictions per tool (real density signal).
    gt_paths = list(canonical_gt_paths)
    n_gt = n_canonical
    n_pred = len(pred_records)
    n_total = n_gt + n_pred

    # ---- Cache: pairwise TM matrix + UMAP embedding + path layout ----
    # Re-running plotting (e.g. styling tweaks) doesn't need the slow
    # TMalign pass — load the .npz instead. Cache key includes the inputs
    # that meaningfully change the embedding: tool list, n_per_tool, seed,
    # and the deterministic pred path list (since the random subsample
    # picks specific files).
    import hashlib
    cache_key_parts = [
        eid, ",".join(tools), str(n_per_tool), str(seed),
        ";".join(p for _, p in pred_records),
        ";".join(gt_paths),
    ]
    cache_key = hashlib.sha1(
        "\n".join(cache_key_parts).encode()
    ).hexdigest()[:16]
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{eid}_{cache_key}.npz"

    if cache_path.exists():
        print(f"  [{eid}] cache hit: {cache_path.name}")
        cached = np.load(cache_path)
        tm = cached["tm"]
        embed = cached["embed"]
    else:
        print(f"  [{eid}] embedding {n_canonical} canonical GTs + "
              f"{n_pred} preds ({n_total} total)")

        # Pairwise TM matrix (1.0 on diagonal).
        tm = np.full((n_total, n_total), np.nan, dtype=float)
        np.fill_diagonal(tm, 1.0)

        # Canonical-canonical from cached tm_xx.npy
        f_xx = (PROJECT_ROOT / "results" / DATASET / "gt_matrices_tmalign"
                / eid / "tm_xx.npy")
        if f_xx.exists():
            gt_mat = np.load(f_xx)
            if gt_mat.shape == (n_canonical, n_canonical):
                tm[:n_canonical, :n_canonical] = gt_mat
        else:
            print(f"  [{eid}] WARNING: missing tm_xx.npy — "
                  "canonical-canonical defaults to fresh computation")

        # Canonical-pred from each tool's cached tmalign_raw JSON
        pred_index = {p: i for i, (_, p) in enumerate(pred_records)}
        gt_index = {g: i for i, g in enumerate(gt_paths)}
        for t in tools:
            jf = (PROJECT_ROOT / "results" / DATASET / t / "tmalign_raw"
                  / f"{eid}.json")
            if not jf.exists(): continue
            d = json.load(open(jf))
            for p in d.get("pairs") or []:
                pi = pred_index.get(p.get("pred"))
                gi = gt_index.get(p.get("gt"))
                if pi is None or gi is None: continue
                v = _tm_max2(p)
                if v is None: continue
                tm[gi, n_gt + pi] = v
                tm[n_gt + pi, gi] = v

        # Pred-pred + any remaining NaN cells: parallel TMalign.
        todo: list[tuple[int, int, str, str]] = []
        all_paths = gt_paths + [p for _, p in pred_records]
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if not math.isnan(tm[i, j]):
                    continue
                todo.append((i, j, all_paths[i], all_paths[j]))
        print(f"  [{eid}] computing {len(todo)} fresh TMalign pairs "
              f"(workers={workers})…")
        with Pool(processes=workers) as pool:
            for i, j, v in pool.imap_unordered(_tmalign_pair, todo, chunksize=32):
                if not math.isnan(v):
                    tm[i, j] = v
                    tm[j, i] = v
        tm = np.nan_to_num(tm, nan=0.0)

        # UMAP on (1 - TM) distance.
        dist = 1.0 - tm
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, None)
        try:
            import umap
            n_neighbors = max(5, min(15, n_total - 1))
            print(f"  [{eid}] running UMAP (n_neighbors={n_neighbors})…")
            embed = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors,
                              min_dist=0.10, random_state=seed,
                              n_components=2).fit_transform(dist)
        except Exception as e:
            print(f"  [{eid}] UMAP failed ({e}); falling back to t-SNE")
            from sklearn.manifold import TSNE
            embed = TSNE(n_components=2, metric="precomputed", init="random",
                         perplexity=min(perplexity,
                                        max(5.0, (n_total - 1) / 3.0)),
                         random_state=seed,
                         learning_rate="auto").fit_transform(dist)
        np.savez(cache_path, tm=tm, embed=embed)
        print(f"  [{eid}] cached → {cache_path.name}")
    canonical_xy = embed[:n_gt]
    pred_xy = embed[n_gt:]

    # 5) Render PyMOL ribbons for the canonical GTs
    ribbon_imgs: list[np.ndarray] = []
    for k in range(n_canonical):
        g = canonical_gt_paths[k]
        out_png = ribbon_dir / f"{eid}_GT{k}_{Path(g).stem}.png"
        if not out_png.exists():
            print(f"    rendering {out_png.name}")
            render_ribbon_pymol(Path(g), GT_COLOR_HEX[k], out_png)
        ribbon_imgs.append(crop_to_content(out_png))

    # 6) Plot — one panel per tool, side-by-side, sharing the same UMAP
    #    coordinate system. Each panel: KDE+scatter for that tool's preds,
    #    plus all GT stars (no PyMOL callouts inside the panels — those
    #    sit in a shared row across the top of the figure).
    n_tools = len(tools)
    fig = plt.figure(figsize=(7.0 * n_tools, 7.5))
    panel_h = 0.62
    panel_y = 0.06
    ribbon_y = 0.74
    ribbon_h = 0.18
    panel_w = 0.96 / n_tools - 0.04

    pad_x = (embed[:, 0].max() - embed[:, 0].min()) * 0.08
    pad_y = (embed[:, 1].max() - embed[:, 1].min()) * 0.08
    xlim = (embed[:, 0].min() - pad_x, embed[:, 0].max() + pad_x)
    ylim = (embed[:, 1].min() - pad_y, embed[:, 1].max() + pad_y)
    grid_n = 200
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    # Top row: PyMOL ribbon callouts spanning both panels.
    for k in range(n_canonical):
        rx = 0.04 + (k + 0.5) * (0.96 / max(1, n_canonical)) - 0.08
        rx = min(0.96 - 0.16, max(0.04, rx))
        ax_r = fig.add_axes([rx, ribbon_y, 0.16, ribbon_h])
        ax_r.imshow(ribbon_imgs[k], interpolation="bilinear")
        ax_r.set_xticks([]); ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_visible(False)

    # Two side-by-side data axes
    panel_axes = []
    for ti, tool in enumerate(tools):
        ax = fig.add_axes([0.04 + ti * (panel_w + 0.04), panel_y,
                           panel_w, panel_h])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        c = TOOL_COLOR[tool]

        # Pred KDE colored by density — energy-style: blue = high density
        # (low energy / stable), red = low density (high energy / unstable).
        # cmap RdBu maps low values → red, high values → blue, which matches.
        mask = np.array([tt == tool for (tt, _) in pred_records])
        if mask.any():
            xy = pred_xy[mask]
            try:
                kde = gaussian_kde(xy.T, bw_method=0.30)
                zz = kde(grid).reshape(xx.shape)
                levels = np.linspace(zz.min(), zz.max(), 14)
                ax.contourf(xx, yy, zz, levels=levels, cmap="RdBu",
                            alpha=0.85)
                ax.contour(xx, yy, zz, levels=levels[2::3], colors="#222222",
                           alpha=0.30, linewidths=0.4)
            except Exception:
                pass
            # Black scatter dots stay visible against any colormap
            ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#111111", alpha=0.55,
                       edgecolors="white", linewidths=0.25, zorder=5,
                       label=f"{TOOL_DISPLAY.get(tool, tool)} preds (n={len(xy)})")

        # GT stars — same positions in both panels (shared embedding).
        # Solid colored fill with a yellow outline (higher contrast than
        # black against the blue high-density centers of the heat KDE).
        for k in range(n_gt):
            ax.scatter([canonical_xy[k, 0]], [canonical_xy[k, 1]],
                       marker="*", s=380, c=GT_COLOR_HEX[k],
                       edgecolors="black", linewidths=1.4, zorder=10,
                       label=f"GT {chr(65+k)}" if ti == 0 else None)

        ax.set_xlabel("UMAP dim 1", fontsize=16)
        if ti == 0:
            ax.set_ylabel("UMAP dim 2", fontsize=16)
        # Title shows tool name + the ensemble's per-tool EMD so readers can
        # correlate the visual KDE shape with the macro metric.
        emd_v = _ensemble_metric(eid, tool, "emd")
        emd_str = f"  EMD = {emd_v:.3f}" if emd_v is not None else ""
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
        panel_axes.append(ax)


    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eid}_{name.lower().replace(' ', '_')}_tsne.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(str(out_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", nargs="+", default=None,
                    help="UniProts to plot (default: built-in list)")
    ap.add_argument("--tools", nargs=2, default=["afsample2", "bioemu"],
                    metavar=("MSA_TOOL", "GENERATIVE_TOOL"),
                    help="Two tools to overlay (best MSA + best generative)")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Lower bound of x/y axes (TM display floor)")
    ap.add_argument("--n3", action="store_true",
                    help="Use the 3-pair n_gt==3 layout instead of the n_gt==2 layout")
    ap.add_argument("--tsne", action="store_true",
                    help="t-SNE embedding of GTs+subsampled preds in a single 2D space "
                         "(works for any n_gt). Computes pred-pred TMalign on the fly.")
    ap.add_argument("--n-per-tool", type=int, default=60,
                    help="Subsample this many preds per tool when --tsne (default 60)")
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallel TMalign workers (default 32)")
    ap.add_argument("--max-extras", type=int, default=50,
                    help="Cap upstream GT extras at this many "
                         "(uniformly sampled, default 50)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "figures" / "kde_mode_collapse")
    args = ap.parse_args()

    if args.proteins:
        proteins = []
        name_lookup = dict((eid, name) for eid, name in PROTEINS_DEFAULT)
        for eid in args.proteins:
            proteins.append((eid, name_lookup.get(eid, eid)))
    else:
        proteins = PROTEINS_DEFAULT

    ribbon_dir = args.out_dir / "_pymol"
    for eid, name in proteins:
        print(f"\n[{eid}] {name}")
        if args.tsne:
            plot_protein_tsne(eid, name, list(args.tools),
                              args.out_dir, ribbon_dir,
                              n_per_tool=args.n_per_tool,
                              perplexity=args.perplexity,
                              workers=args.workers,
                              max_extras=args.max_extras)
        elif args.n3:
            plot_protein_3gt(eid, name, list(args.tools), args.thresh,
                             args.out_dir, ribbon_dir)
        else:
            plot_protein(eid, name, list(args.tools), args.thresh,
                         args.out_dir, ribbon_dir)


if __name__ == "__main__":
    main()
