"""
Common loader for figure scripts: prefer the new ``tmalign_raw/<eid>.json``
artifacts (canonical, written by ``compute_tmalign_raw.py``); fall back to
the legacy ``matrices_tmalign/<eid>/{tm_xy,tm_xy_ab,rmsd_xy_ab}.npy``.

This shim keeps 04_build_figures scripts working as the pipeline migrates
toward the JSON canonical store without forcing a regen of every figure
right away.

Usage:
    from pipeline.03_compute_results.tmalign_io import load_xy_pair
    tm1, tm2, rmsd = load_xy_pair("UNICORNE_BENCH", "alphaflow_pdb", "P23904")
    # tm1, tm2 are (n_gt, n_pred) matrices of TM normalized by GT / pred length
    # rmsd is the same shape; all NaN-filled where missing
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Internal: parse one tmalign_raw JSON into matrices
# ---------------------------------------------------------------------------

def _matrices_from_json(json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tm_chain1, tm_chain2, rmsd)`` matrices of shape (n_gt, n_pred).

    NaN entries indicate missing/failed pairs.
    """
    d = json.loads(json_path.read_text())
    n_gt, n_pred = int(d["n_gt"]), int(d["n_pred"])
    tm1 = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    tm2 = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    rmsd = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    for k, p in enumerate(d.get("pairs", [])):
        i, j = divmod(k, n_pred)
        if p.get("tm_norm_chain1") is not None:
            tm1[i, j] = float(p["tm_norm_chain1"])
        if p.get("tm_norm_chain2") is not None:
            tm2[i, j] = float(p["tm_norm_chain2"])
        if p.get("rmsd") is not None:
            rmsd[i, j] = float(p["rmsd"])
    return tm1, tm2, rmsd


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def raw_json_path(dataset: str, tool: str, eid: str) -> Path:
    return (PROJECT_ROOT / "results" / dataset / tool /
            "tmalign_raw" / f"{eid}.json")


def legacy_matrix_dir(dataset: str, tool: str, eid: str) -> Path:
    return (PROJECT_ROOT / "results" / dataset / tool /
            "matrices_tmalign" / eid)


def load_xy_pair(
    dataset: str,
    tool: str,
    eid: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load ``(tm_chain1, tm_chain2, rmsd)`` for one ensemble.

    Tries the new ``tmalign_raw/<eid>.json`` first; if that doesn't exist,
    reconstructs the same triple from the legacy ``matrices_tmalign/<eid>/``
    npy files. Returns ``None`` if neither source has data.
    """
    j = raw_json_path(dataset, tool, eid)
    if j.exists():
        return _matrices_from_json(j)

    leg = legacy_matrix_dir(dataset, tool, eid)
    if not leg.is_dir():
        return _matrices_from_summary(dataset, tool, eid)
    tm1 = tm2 = rmsd = None
    for name in ("tm_xy.npy", "tm_xy_ab.npy"):
        p = leg / name
        if p.exists():
            tm1 = np.load(p)
            break
    for name in ("tm_xy_ba.npy",):
        p = leg / name
        if p.exists():
            tm2 = np.load(p)
            break
    if tm2 is None and tm1 is not None:
        tm2 = tm1  # legacy stored only one direction in some runs
    p = leg / "rmsd_xy_ab.npy"
    if p.exists():
        rmsd = np.load(p)
    if tm1 is None and tm2 is None and rmsd is None:
        return None
    if tm1 is None:
        tm1 = np.full_like(tm2 if tm2 is not None else rmsd, np.nan)
    if tm2 is None:
        tm2 = np.full_like(tm1, np.nan)
    if rmsd is None:
        rmsd = np.full_like(tm1, np.nan)
    return tm1, tm2, rmsd


def load_tm_matrix(
    dataset: str,
    tool: str,
    eid: str,
    normalize_by: str = "chain1",
) -> Optional[np.ndarray]:
    """Just the (n_gt, n_pred) TM matrix, normalized by chain1 (GT, default)
    or chain2 (pred). NaN where missing."""
    out = load_xy_pair(dataset, tool, eid)
    if out is None:
        return None
    tm1, tm2, _ = out
    return tm1 if normalize_by == "chain1" else tm2


def load_rmsd_matrix(
    dataset: str,
    tool: str,
    eid: str,
) -> Optional[np.ndarray]:
    """Just the (n_gt, n_pred) RMSD matrix. NaN where missing."""
    out = load_xy_pair(dataset, tool, eid)
    if out is None:
        return None
    return out[2]


def list_available_eids(dataset: str, tool: str) -> list[str]:
    """All ensembles for which tmalign_raw JSON, a legacy matrix dir, or a
    per-GT summary row exists."""
    eids: set[str] = set(load_per_gt_summary(dataset, tool))
    raw = PROJECT_ROOT / "results" / dataset / tool / "tmalign_raw"
    if raw.is_dir():
        eids.update(p.stem for p in raw.glob("*.json"))
    leg = PROJECT_ROOT / "results" / dataset / tool / "matrices_tmalign"
    if leg.is_dir():
        eids.update(p.name for p in leg.iterdir() if p.is_dir())
    return sorted(eids)


# ---------------------------------------------------------------------------
# Per-GT summary fallback (results/<DATASET>/<TOOL>/per_gt_summary.csv)
# ---------------------------------------------------------------------------
# Written by export_per_gt_summary.py. Holds, per GT conformation, the best
# prediction's TM/RMSD and the max TM_max2 -- everything the figure scripts
# take from the full matrices -- so they run without the raw TM-align data.

_SUMMARIES: dict[tuple[str, str], dict[str, list[dict]]] = {}


def summary_path(dataset: str, tool: str) -> Path:
    return PROJECT_ROOT / "results" / dataset / tool / "per_gt_summary.csv"


def load_per_gt_summary(dataset: str, tool: str) -> dict[str, list[dict]]:
    """{eid: [per-GT row, ...]} in GT order; empty if no summary exists."""
    key = (dataset, tool)
    if key not in _SUMMARIES:
        by_eid: dict[str, list[dict]] = {}
        p = summary_path(dataset, tool)
        if p.exists():
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    by_eid.setdefault(row["ensemble_ID"], []).append(
                        {k: float(row[k]) for k in
                         ("best_tm_chain1", "rmsd_at_best", "max_tm_max2")})
        _SUMMARIES[key] = by_eid
    return _SUMMARIES[key]


def _matrices_from_summary(dataset: str, tool: str, eid: str):
    """(n_gt, 1) matrices holding each GT's best prediction, or None.

    Taking the row-wise argmax of these returns the same TM/RMSD as the
    full matrices would; TM_chain2 is not summarized and is NaN.
    """
    rows = load_per_gt_summary(dataset, tool).get(eid)
    if not rows:
        return None
    tm1 = np.array([[r["best_tm_chain1"]] for r in rows])
    rmsd = np.array([[r["rmsd_at_best"]] for r in rows])
    return tm1, np.full_like(tm1, np.nan), rmsd
