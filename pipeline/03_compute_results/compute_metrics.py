#!/usr/bin/env python3
"""
Compute per-ensemble benchmark metrics from compute_tmalign_raw.py JSONs.

For every JSON in ``results/<DATASET>/<TOOL>/tmalign_raw/<eid>.json``, build
the GT × pred TM matrix and derive:

  - ``recall_at_tm_<T>``     for T in {0.85, 0.9, 0.95}
        Fraction of GT structures whose best matching prediction has TM ≥ T.
  - ``precision_at_tm_<T>``  for T in {0.85, 0.9, 0.95}
        Fraction of predicted structures whose best matching GT has TM ≥ T.
  - ``recall_09_at_k_<K>_{mean,std}``  for K in {50, 100, 150, 200, 250}
        Bootstrap ``recall_at_tm_0.9`` over random subsets of K predictions,
        ``--n-bootstrap`` repeats. Captures sample-efficiency.

TM normalization defaults to GT length (``tm_norm_chain1`` field) — the paper
convention. Override with ``--normalize-by chain2`` for prediction-length
normalization.

Replaces the GT × pred metric block of the legacy ``compute_results.py``;
intentionally drops MMD (never used in the paper).

Usage:
  python pipeline/03_compute_results/compute_metrics.py UNICORNE_BENCH alphaflow_pdb
  python pipeline/03_compute_results/compute_metrics.py UNICORNE_BENCH bioemu --filtered-only

Output:
  results/<DATASET>/<TOOL>/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TM_THRESHOLDS = [0.85, 0.90, 0.95]
K_VALUES = list(range(5, 1001, 5))   # 5, 10, 15, …, 1000 (uniform Δk = 5)


# ---------------------------------------------------------------------------
# Matrix loader
# ---------------------------------------------------------------------------

def load_tm_matrix(
    json_path: Path,
    normalize_by: str = "chain1",
) -> tuple[np.ndarray, dict]:
    """Return (n_gt × n_pred TM matrix, raw payload dict).

    NaN entries indicate failed pairs (TM-align error or missing field).
    """
    d = json.loads(json_path.read_text())
    field = f"tm_norm_{normalize_by}"
    n_gt, n_pred = int(d["n_gt"]), int(d["n_pred"])
    M = np.full((n_gt, n_pred), np.nan, dtype=np.float64)
    for k, p in enumerate(d.get("pairs", [])):
        i, j = divmod(k, n_pred)
        v = p.get(field)
        if v is not None:
            M[i, j] = float(v)
    return M, d


# ---------------------------------------------------------------------------
# Per-ensemble metrics
# ---------------------------------------------------------------------------

# EMD floor: TM-scores below this are treated as fully dissimilar (cost=1)
# in the optimal-transport formulation. Matches tab_overall_performance.py.
EMD_FLOOR = 0.3


def _emd_from_tm(M: np.ndarray) -> float:
    """Optimal-transport distance between uniform distributions over GT and
    pred, with cost matrix ``1 - clip((TM - floor) / (1 - floor), 0, 1)``.

    Lower is better. Returns NaN if `pot` is unavailable.
    """
    try:
        import ot
    except ImportError:
        return float("nan")
    n_gt, n_pred = M.shape
    if n_gt == 0 or n_pred == 0:
        return float("nan")
    sim = np.clip((M - EMD_FLOOR) / (1 - EMD_FLOOR), 0.0, 1.0)
    cost = 1.0 - sim
    a = np.full(n_gt, 1.0 / n_gt)
    b = np.full(n_pred, 1.0 / n_pred)
    return float(ot.emd2(a, b, cost))


def metrics_for_ensemble(
    M: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 50,
) -> dict:
    """Compute per-ensemble benchmark metrics from a GT × pred TM matrix.

    Threshold metrics (from M):
      recall_at_tm_<T>     fraction of GT with max pred TM ≥ T
      precision_at_tm_<T>  fraction of pred with max GT TM ≥ T

    Sample-efficiency:
      recall_09_at_k_<K>_{mean,std}   bootstrap over K-subsets of predictions

    Aggregate quality scores (paper Table 3):
      dr   = recall_at_tm_0.9                (Discovery Rate, per-ensemble)
      cr   = 1.0 if dr == 1.0 else 0.0       (Complete Recovery indicator)
      eu   = mean over GT rows of max-pred-TM (Ensemble Utility)
      emd  = optimal-transport cost (lower is better)
    """
    out: dict[str, float] = {}
    n_gt, n_pred = M.shape

    if n_gt == 0 or n_pred == 0:
        for t in TM_THRESHOLDS:
            out[f"recall_at_tm_{t}"] = np.nan
            out[f"precision_at_tm_{t}"] = np.nan
        for k in K_VALUES:
            out[f"recall_09_at_k_{k}_mean"] = np.nan
            out[f"recall_09_at_k_{k}_std"] = np.nan
        out.update(dr=np.nan, cr=np.nan, eu=np.nan, emd=np.nan)
        return out

    # NaN-safe max (treat missing pairs as 0 — never recovered)
    Mfilled = np.where(np.isnan(M), 0.0, M)

    gt_max = Mfilled.max(axis=1)         # best pred per GT
    pred_max = Mfilled.max(axis=0)       # best GT per pred

    for t in TM_THRESHOLDS:
        out[f"recall_at_tm_{t}"] = float((gt_max >= t).mean())
        out[f"precision_at_tm_{t}"] = float((pred_max >= t).mean())

    # Bootstrap DR / CR / EU at sample budget k (random K-subsets of preds).
    # DR  = recall at TM ≥ 0.9                  (already used for the @k curve)
    # CR  = indicator(DR == 1.0)                 → fraction of bootstraps where
    #        every GT state is recovered with ≤ k preds
    # EU  = mean best-match TM (= mean over GTs of max TM to chosen preds)
    for k in K_VALUES:
        kk = min(k, n_pred)
        if kk <= 0:
            for prefix in ("recall_09_at_k", "cr_at_k", "eu_at_k"):
                out[f"{prefix}_{k}_mean"] = np.nan
                out[f"{prefix}_{k}_std"] = np.nan
            continue
        recalls = np.empty(n_bootstrap, dtype=np.float64)
        crs = np.empty(n_bootstrap, dtype=np.float64)
        eus = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.choice(n_pred, size=kk, replace=False)
            sub = Mfilled[:, idx]
            sub_max = sub.max(axis=1)        # best pred per GT in this subset
            r = float((sub_max >= 0.90).mean())
            recalls[b] = r
            crs[b] = 1.0 if r == 1.0 else 0.0
            eus[b] = float(sub_max.mean())
        out[f"recall_09_at_k_{k}_mean"] = float(recalls.mean())
        out[f"recall_09_at_k_{k}_std"] = float(recalls.std())
        out[f"cr_at_k_{k}_mean"] = float(crs.mean())
        out[f"cr_at_k_{k}_std"] = float(crs.std())
        out[f"eu_at_k_{k}_mean"] = float(eus.mean())
        out[f"eu_at_k_{k}_std"] = float(eus.std())

    # Aggregate quality scores
    dr = out["recall_at_tm_0.9"]
    out["dr"] = dr
    out["cr"] = 1.0 if dr == 1.0 else 0.0
    out["eu"] = float(gt_max.mean())
    out["emd"] = _emd_from_tm(Mfilled)

    return out


def column_order() -> list[str]:
    cols = ["ensemble_ID", "n_gt", "n_pred"]
    cols += ["dr", "cr", "eu", "emd"]
    cols += [f"recall_at_tm_{t}" for t in TM_THRESHOLDS]
    cols += [f"precision_at_tm_{t}" for t in TM_THRESHOLDS]
    cols += [f"recall_09_at_k_{k}_mean" for k in K_VALUES]
    cols += [f"recall_09_at_k_{k}_std" for k in K_VALUES]
    cols += [f"cr_at_k_{k}_mean" for k in K_VALUES]
    cols += [f"cr_at_k_{k}_std" for k in K_VALUES]
    cols += [f"eu_at_k_{k}_mean" for k in K_VALUES]
    cols += [f"eu_at_k_{k}_std" for k in K_VALUES]
    return cols


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_one_json(args_tuple) -> tuple[str, Optional[dict], Optional[str]]:
    """Pool worker: parse one JSON, compute per-ensemble metrics.
    Returns (eid_or_filename, row_dict_or_None, error_or_None).
    Each worker uses its own RNG seeded from (base_seed XOR hash(eid))
    so bootstrap results are deterministic and parallel-safe.
    """
    (json_path_str, normalize_by, n_bootstrap, base_seed,
     min_preds, filter_set) = args_tuple
    json_path = Path(json_path_str)
    eid = json_path.stem
    if filter_set is not None and eid not in filter_set:
        return eid, None, "filtered"
    try:
        M, d = load_tm_matrix(json_path, normalize_by=normalize_by)
    except Exception as e:
        return eid, None, f"load error — {e}"
    n_pred = int(d.get("n_pred", 0))
    if min_preds and n_pred < min_preds:
        return eid, None, "short"
    seed = (base_seed ^ (hash(d.get("ensemble_id", eid)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    try:
        m = metrics_for_ensemble(M, rng, n_bootstrap=n_bootstrap)
    except Exception as e:
        return eid, None, f"metrics error — {e}"
    m["ensemble_ID"] = d["ensemble_id"]
    m["n_gt"] = int(d["n_gt"])
    m["n_pred"] = n_pred
    return eid, m, None


def load_filter(filter_csv: Path) -> set[str]:
    eids: set[str] = set()
    with open(filter_csv) as fh:
        r = csv.DictReader(fh)
        for row in r:
            eid = (row.get("ensemble_ID") or row.get("ensemble_id")
                   or row.get("uniprot", "")).strip()
            if eid:
                eids.add(eid)
    return eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Dataset name (e.g. UNICORNE_BENCH)")
    ap.add_argument("tool", help="Tool name (e.g. alphaflow_pdb, bioemu, ...)")
    ap.add_argument(
        "--normalize-by",
        choices=["chain1", "chain2"],
        default="chain1",
        help="TM normalization: chain1=GT length (default, paper conv.), chain2=pred length",
    )
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Bootstrap replicates for recall_09_at_k metrics",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--filtered-only",
        action="store_true",
        help="Only include ensembles in data/inputs/<DATASET>/filtered_ensembles.csv",
    )
    ap.add_argument(
        "--min-preds",
        type=int,
        default=250,
        help="Minimum number of predictions required to score an ensemble. "
             "Ensembles with fewer prediction PDBs (per the JSON's n_pred) are "
             "skipped — keeps comparisons across tools fair when one tool fell "
             "short on a target. Default 250.",
    )
    ap.add_argument(
        "--in-suffix",
        default="",
        help="Read JSONs from tmalign_raw<suffix>/ instead of tmalign_raw/",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output CSV path (default: results/<DATASET>/<TOOL>/results.csv)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="Process pool size (default: half of os.cpu_count). "
             "Each worker handles one ensemble JSON; the load+parse step "
             "is the dominant cost and parallelizes ~linearly.",
    )
    args = ap.parse_args()

    raw_dir = PROJECT_ROOT / "results" / args.dataset / args.tool / f"tmalign_raw{args.in_suffix}"
    if not raw_dir.is_dir():
        sys.exit(f"error: tmalign_raw dir missing: {raw_dir}")

    filter_set: Optional[set[str]] = None
    if args.filtered_only:
        f = PROJECT_ROOT / "data" / "inputs" / args.dataset / "filtered_ensembles.csv"
        if not f.exists():
            sys.exit(f"error: --filtered-only but {f} missing")
        filter_set = load_filter(f)
        print(f"filter: keeping {len(filter_set)} ensembles from {f}")

    out_csv = args.output or (
        PROJECT_ROOT / "results" / args.dataset / args.tool / "results.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = column_order()

    json_files = sorted(raw_dir.glob("*.json"))
    print(f"found {len(json_files)} JSONs in {raw_dir}")
    print(f"writing {out_csv}  (workers={args.workers})")

    work_items = [
        (str(f), args.normalize_by, args.n_bootstrap, args.seed,
         args.min_preds, filter_set)
        for f in json_files
    ]

    rows = []
    n_skip = n_fail = n_short = 0
    if args.workers > 1 and len(work_items) > 1:
        with mp.Pool(processes=args.workers) as pool:
            for i, (eid, row, err) in enumerate(
                    pool.imap_unordered(_process_one_json, work_items, chunksize=4), 1):
                if err == "filtered":
                    n_skip += 1
                elif err == "short":
                    n_short += 1
                elif err:
                    print(f"  {eid}: {err}", file=sys.stderr)
                    n_fail += 1
                else:
                    rows.append(row)
                if i % 100 == 0:
                    print(f"  [{i}/{len(work_items)}] kept={len(rows)} "
                          f"filtered={n_skip} short={n_short} failed={n_fail}",
                          flush=True)
    else:
        for i, item in enumerate(work_items, 1):
            eid, row, err = _process_one_json(item)
            if err == "filtered":
                n_skip += 1
            elif err == "short":
                n_short += 1
            elif err:
                print(f"  {eid}: {err}", file=sys.stderr)
                n_fail += 1
            else:
                rows.append(row)

    rows.sort(key=lambda r: r["ensemble_ID"])
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows  (skipped={n_skip}, short={n_short}, failed={n_fail})")


if __name__ == "__main__":
    main()
