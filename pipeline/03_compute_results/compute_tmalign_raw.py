#!/usr/bin/env python3
"""
Compute the full GT × prediction TMalign matrix for an ensemble dataset and
store the *raw* per-pair output as JSON for downstream analysis.

For every ensemble in `data/inputs/<DATASET>/filtered_ensembles.csv` (or all
if --no-filtered) we:
  1. Enumerate GT PDBs in   data/inputs/<DATASET>/input_pdbs/<eid>/*.pdb
  2. Enumerate pred PDBs in data/outputs/<TOOL>/<DATASET>/output_pdbs/<eid>/*.pdb
  3. Run TMalign (the official C++ binary) on every (gt, pred) pair
  4. Parse TM1 / TM2 / RMSD / aligned length / seq identity / alignment strings
  5. Save the entire batch — raw stdout included — to
        results/<DATASET>/<TOOL>/tmalign_raw/<eid>.json

The schema is stable enough to recompute any metric (mmd / recall / precision /
RMSD bins / 3Di-style thresholds) from the JSON without re-running TMalign.

Usage:
  python pipeline/03_compute_results/compute_tmalign_raw.py UNICORNE_BENCH alphaflow_pdb
  python pipeline/03_compute_results/compute_tmalign_raw.py UNICORNE_BENCH bioemu --workers 16 --resume

Env: TMALIGN_EXE (defaults to `TMalign` on PATH).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_tmalign_exe() -> str:
    return os.environ.get("TMALIGN_EXE") or os.environ.get("TMALIGN") or "TMalign"


# ---------------------------------------------------------------------------
# TMalign output parsing
# ---------------------------------------------------------------------------

_RX_LEN1     = re.compile(r"Length of Chain_1:\s+(\d+)")
_RX_LEN2     = re.compile(r"Length of Chain_2:\s+(\d+)")
_RX_LALI     = re.compile(r"Aligned length=\s*(\d+)")
_RX_RMSD     = re.compile(r"RMSD=\s*([\d.]+)")
_RX_SEQID    = re.compile(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)")
_RX_TM_NORM1 = re.compile(r"TM-score=\s*([\d.]+)\s*\(if normalized by length of Chain_1")
_RX_TM_NORM2 = re.compile(r"TM-score=\s*([\d.]+)\s*\(if normalized by length of Chain_2")


def parse_tmalign_stdout(text: str) -> dict:
    """Pull structured fields out of TMalign default-format output."""
    def f(rx, cast=float):
        m = rx.search(text)
        return cast(m.group(1)) if m else None

    out = {
        "len_chain1": f(_RX_LEN1, int),
        "len_chain2": f(_RX_LEN2, int),
        "len_aligned": f(_RX_LALI, int),
        "rmsd": f(_RX_RMSD),
        "seq_id": f(_RX_SEQID),
        "tm_norm_chain1": f(_RX_TM_NORM1),
        "tm_norm_chain2": f(_RX_TM_NORM2),
    }
    # Pull the 3-line alignment block (chain1 / match / chain2)
    m = re.search(
        r'(?P<ali>[A-Z\-\*]+\n[ :\.]+\n[A-Z\-\*]+)',
        text,
    )
    if m:
        rows = m.group("ali").split("\n")
        if len(rows) == 3:
            a1, am, a2 = rows
            out["align_chain1"] = a1
            out["align_match"] = am
            out["align_chain2"] = a2
            # Derive 0-indexed residue-residue mapping from the aligned columns.
            # mapping_close: ':' columns (d < d0 after superposition).
            # mapping_loose: '.' columns (loose, d >= d0 but still aligned).
            # mapping_all: every column where neither chain has a gap.
            mapping_close, mapping_loose, mapping_all = [], [], []
            i1 = i2 = 0
            for c1, cm, c2 in zip(a1, am, a2):
                if c1 != "-" and c2 != "-":
                    mapping_all.append([i1, i2])
                    if cm == ":":
                        mapping_close.append([i1, i2])
                    elif cm == ".":
                        mapping_loose.append([i1, i2])
                if c1 != "-":
                    i1 += 1
                if c2 != "-":
                    i2 += 1
            out["mapping_close"] = mapping_close
            out["mapping_loose"] = mapping_loose
            out["mapping_all"] = mapping_all
    return out


_RX_ROT_HEADER = re.compile(
    r"-+\s*The rotation matrix to rotate Chain_1 to Chain_2\s*-+"
)


def parse_rotation_matrix(text: str) -> Optional[dict]:
    """Parse the 3-row rotation/translation matrix written by `TMalign -m`.

    File format:
        ------ The rotation matrix to rotate Chain_1 to Chain_2 ------
        m   t[m]   u[m][0]   u[m][1]   u[m][2]
        0   tx     uxx       uxy       uxz
        1   ty     uyx       uyy       uyz
        2   tz     uzx       uzy       uzz
    Returns {"t": [tx, ty, tz], "u": [[uxx,uxy,uxz], [uyx,uyy,uyz], [uzx,uzy,uzz]]}
    """
    if not _RX_ROT_HEADER.search(text):
        return None
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in ("0", "1", "2"):
            try:
                rows.append([float(p) for p in parts[1:]])
            except ValueError:
                pass
    if len(rows) != 3:
        return None
    t = [r[0] for r in rows]
    u = [r[1:] for r in rows]
    return {"t": t, "u": u}


def _pdb_has_nan_coords(path: str) -> bool:
    """Cheap grep: skip any PDB whose ATOM records carry NaN coordinates.
    Some predictors (alphaflow with bad seeds) emit ATOMs with `nan nan nan`
    in the xyz columns; TMalign infinite-loops on those instead of returning."""
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                if b"nan" in chunk or b"NaN" in chunk:
                    return True
    except OSError:
        return True
    return False


def run_tmalign_one(args: tuple) -> dict:
    """Run TMalign on one (gt, pred) pair. Worker function for multiprocessing."""
    gt_path, pred_path, exe, timeout = args
    record = {"gt": gt_path, "pred": pred_path}
    if _pdb_has_nan_coords(pred_path):
        record["error"] = "skipped: pred has NaN coordinates"
        record["raw"] = ""
        return record
    try:
        with tempfile.TemporaryDirectory(prefix="tmalign_") as td:
            mat_path = Path(td) / "m.txt"
            r = subprocess.run(
                [exe, gt_path, pred_path, "-m", str(mat_path)],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode != 0:
                record["error"] = (r.stderr or r.stdout or "non-zero exit").strip()[:200]
                record["raw"] = ""
                return record
            record["raw"] = r.stdout
            record.update(parse_tmalign_stdout(r.stdout))
            if mat_path.exists():
                rot = parse_rotation_matrix(mat_path.read_text())
                if rot:
                    record["rotation"] = rot
    except subprocess.TimeoutExpired:
        record["error"] = f"timeout after {timeout}s"
        record["raw"] = ""
    except Exception as e:
        record["error"] = str(e)[:200]
        record["raw"] = ""
    return record


# ---------------------------------------------------------------------------
# Ensemble-level driver
# ---------------------------------------------------------------------------

def list_pdbs(d: Path) -> list[Path]:
    """List structure files in a directory. TMalign reads both PDB and mmCIF."""
    files = list(d.glob("*.pdb")) + list(d.glob("*.cif"))
    return sorted(p for p in files if p.is_file())


def process_ensemble(eid: str, gt_dir: Path, pred_dir: Path, out_path: Path,
                     exe: str, workers: int, timeout: int,
                     pred_limit: Optional[int]) -> dict:
    gt_files = list_pdbs(gt_dir)
    pred_files = list_pdbs(pred_dir)
    if pred_limit:
        pred_files = pred_files[:pred_limit]

    pairs = [(str(g), str(p), exe, timeout) for g in gt_files for p in pred_files]
    t0 = time.perf_counter()
    if workers > 1 and len(pairs) > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(run_tmalign_one, pairs)
    else:
        results = [run_tmalign_one(a) for a in pairs]
    elapsed = time.perf_counter() - t0

    payload = {
        "ensemble_id": eid,
        "n_gt": len(gt_files),
        "n_pred": len(pred_files),
        "elapsed_s": round(elapsed, 2),
        "tmalign_exe": exe,
        "pairs": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(out_path)
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_ensemble_list(dataset_root: Path, filtered_only: bool) -> list[str]:
    if filtered_only:
        f = dataset_root / "filtered_ensembles.csv"
        if not f.exists():
            raise FileNotFoundError(f"filtered_ensembles.csv not found at {f}")
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        # column 1 is ensemble_ID
        col = "ensemble_ID" if "ensemble_ID" in (rows[0] if rows else {}) else next(iter(rows[0]))
        return [r[col].strip() for r in rows if r.get(col)]
    pdbs_dir = dataset_root / "input_pdbs"
    return sorted([d.name for d in pdbs_dir.iterdir() if d.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Dataset name (e.g. UNICORNE_BENCH)")
    ap.add_argument("tool", help="Tool name (e.g. alphaflow_pdb, bioemu, ...)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Process pool size for per-pair TMalign calls")
    ap.add_argument("--timeout", type=int, default=120,
                    help="TMalign timeout per pair (seconds)")
    ap.add_argument("--pred-limit", type=int, default=None,
                    help="Cap predictions per ensemble (for testing)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap ensembles (for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip ensembles whose JSON already exists")
    ap.add_argument("--no-filtered", action="store_true",
                    help="Process every ensemble in input_pdbs/, not just filtered_ensembles.csv")
    ap.add_argument("--out-suffix", default="",
                    help="Append to output dir name (e.g. _topup1k)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Process only ensembles where index %% N == I. Lets one "
                         "tool be split across several hosts writing into the "
                         "same tmalign_raw dir (each ensemble is one file, so "
                         "shards never touch the same output).")
    args = ap.parse_args()

    inputs_root = PROJECT_ROOT / "data" / "inputs" / args.dataset
    outputs_root = PROJECT_ROOT / "data" / "outputs" / args.tool / args.dataset
    if not inputs_root.is_dir():
        sys.exit(f"missing inputs: {inputs_root}")
    if not outputs_root.is_dir():
        sys.exit(f"missing outputs: {outputs_root}")

    out_dir = (PROJECT_ROOT / "results" / args.dataset / args.tool /
               f"tmalign_raw{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensembles = load_ensemble_list(inputs_root, filtered_only=not args.no_filtered)
    if args.limit:
        ensembles = ensembles[:args.limit]
    if args.shard:
        _i, _n = map(int, args.shard.split("/"))
        ensembles = [e for _idx, e in enumerate(ensembles) if _idx % _n == _i]
        print(f"shard {_i}/{_n}: {len(ensembles)} ensembles")

    exe = get_tmalign_exe()
    print(f"TMalign: {exe}")
    print(f"dataset={args.dataset} tool={args.tool} ensembles={len(ensembles)} "
          f"workers={args.workers}")
    print(f"output : {out_dir}")

    n_done = n_skip = n_fail = 0
    t0 = time.perf_counter()
    for i, eid in enumerate(ensembles, 1):
        out_path = out_dir / f"{eid}.json"
        if args.resume and out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        gt_dir = inputs_root / "input_pdbs" / eid
        pred_dir = outputs_root / "output_pdbs" / eid
        if not gt_dir.is_dir() or not list_pdbs(gt_dir):
            print(f"  [{i}/{len(ensembles)}] {eid}: no GT PDBs, skip")
            n_skip += 1
            continue
        if not pred_dir.is_dir() or not list_pdbs(pred_dir):
            print(f"  [{i}/{len(ensembles)}] {eid}: no pred PDBs, skip")
            n_skip += 1
            continue
        try:
            payload = process_ensemble(
                eid, gt_dir, pred_dir, out_path, exe,
                args.workers, args.timeout, args.pred_limit,
            )
            n_done += 1
            n_err = sum(1 for p in payload["pairs"] if p.get("error"))
            print(f"  [{i}/{len(ensembles)}] {eid}: "
                  f"{payload['n_gt']}×{payload['n_pred']} pairs in "
                  f"{payload['elapsed_s']:.1f}s"
                  + (f" ({n_err} errors)" if n_err else ""))
        except Exception as e:
            n_fail += 1
            print(f"  [{i}/{len(ensembles)}] {eid}: FAILED — {e}", file=sys.stderr)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s. "
          f"computed={n_done} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
