#!/usr/bin/env python3
"""
Build the dataset's `input_pdbs/` and `input_dataset.csv` from a curated
benchmark CSV (e.g. `benchmarking-dataset-13.csv`).

The CSV is expected to have one row per ensemble with columns:
  uniprot,sequence,seqlength,minlength,num_clusters,representative_structures

`representative_structures` is a semicolon-separated list of `<PDB>_<chain>`
identifiers. For each, we link the corresponding `.pdb` (and `.cif` if
present) file from the source database into
  data/inputs/<DATASET>/input_pdbs/<uniprot>/<rep>-aligned.{pdb,cif}

Then write
  data/inputs/<DATASET>/input_dataset.csv  (ensemble_ID, ref_sequence).

This is the new canonical Step 0: it does NO TM computation. The 0.85 GT-GT
TM dedup happens in Step 1 (`pipeline/01_build_benchmark/`).

Usage:
  python pipeline/00_build_source/build_inputs_from_csv.py \\
      --csv data/inputs/UNICORNE_BENCH/benchmarking-dataset-13.csv \\
      --dataset UNICORNE_BENCH

Options:
  --copy             Copy files instead of symlinking (slower, ~MB per file)
  --source-root PATH Override default `/home/vanderw/.../aligned-pdbs-new/`
  --overwrite        Replace existing input_pdbs/<eid>/ contents
  --exclude-dataset  Repeatable. Skip ensembles whose ensemble_ID is already
                     present in data/inputs/<NAME>/input_dataset.csv. Use this
                     to avoid redoing work when an expansion dataset (e.g.
                     UNICORNE_500) overlaps with an existing one (e.g.
                     UNICORNE_BENCH, UNICORNE).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(
    "/home/vanderw/conformational-ensemble-database/databaseV1/aligned-pdbs-new"
)


def parse_reps(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(";") if tok.strip()]


def materialize_one(uniprot: str, reps: list[str], source_root: Path,
                    out_dir: Path, copy: bool, overwrite: bool) -> tuple[int, int]:
    """Return (n_linked, n_missing) for this ensemble."""
    eid_dir = out_dir / uniprot
    if overwrite and eid_dir.exists():
        shutil.rmtree(eid_dir)
    eid_dir.mkdir(parents=True, exist_ok=True)

    src_dir = source_root / uniprot
    if not src_dir.is_dir():
        return 0, len(reps)

    n_linked = n_missing = 0
    for rep in reps:
        # Source files are <rep>-aligned.{pdb,cif}; link both if present.
        any_found = False
        for ext in (".pdb", ".cif"):
            src = src_dir / f"{rep}-aligned{ext}"
            if not src.is_file():
                continue
            dst = eid_dir / src.name
            if dst.exists() or dst.is_symlink():
                if not overwrite:
                    any_found = True
                    continue
                dst.unlink()
            if copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
            any_found = True
        if any_found:
            n_linked += 1
        else:
            n_missing += 1
    return n_linked, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Curated benchmark CSV (e.g. benchmarking-dataset-13.csv)")
    ap.add_argument("--dataset", default="UNICORNE_BENCH",
                    help="Dataset name (forms data/inputs/<DATASET>/...)")
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                    help="Source dir holding <uniprot>/<rep>-aligned.{pdb,cif}")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of symlink")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing input_pdbs/<eid>/ contents")
    ap.add_argument("--exclude-dataset", action="append", default=[],
                    metavar="NAME",
                    help="Skip ensembles already in data/inputs/<NAME>/input_dataset.csv. "
                         "Repeatable.")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"missing CSV: {args.csv}")
    if not args.source_root.is_dir():
        sys.exit(f"missing source root: {args.source_root}")

    excluded: set[str] = set()
    for name in args.exclude_dataset:
        ex_csv = PROJECT_ROOT / "data" / "inputs" / name / "input_dataset.csv"
        if not ex_csv.is_file():
            sys.exit(f"--exclude-dataset {name}: missing {ex_csv}")
        with open(ex_csv) as fh:
            for row in csv.DictReader(fh):
                eid = (row.get("ensemble_ID") or "").strip()
                if eid:
                    excluded.add(eid)
        print(f"Excluding {len(excluded)} ensembles cumulatively after reading {ex_csv}")

    ds_dir = PROJECT_ROOT / "data" / "inputs" / args.dataset
    out_pdbs = ds_dir / "input_pdbs"
    out_pdbs.mkdir(parents=True, exist_ok=True)
    out_csv = ds_dir / "input_dataset.csv"

    rows_in_raw = list(csv.DictReader(open(args.csv)))
    rows_in = [r for r in rows_in_raw
               if (r.get("uniprot") or "").strip() not in excluded]
    n_skipped = len(rows_in_raw) - len(rows_in)
    print(f"Loaded {len(rows_in_raw)} rows from {args.csv}"
          f"{f'; {n_skipped} excluded' if excluded else ''}")
    print(f"Source: {args.source_root}")
    print(f"Output: {out_pdbs}  (mode: {'copy' if args.copy else 'symlink'})")

    # Write input_dataset.csv (ensemble_ID, ref_sequence) — overwrite each run
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ensemble_ID", "ref_sequence"])
        for r in rows_in:
            eid = (r.get("uniprot") or "").strip()
            seq = (r.get("sequence") or "").strip()
            if eid and seq:
                w.writerow([eid, seq])
    print(f"Wrote {out_csv}")

    # Materialize PDB files
    total_linked = total_missing = total_no_src = 0
    for i, r in enumerate(rows_in, 1):
        eid = (r.get("uniprot") or "").strip()
        reps = parse_reps(r.get("representative_structures") or "")
        if not eid or not reps:
            continue
        n_linked, n_missing = materialize_one(
            eid, reps, args.source_root, out_pdbs,
            copy=args.copy, overwrite=args.overwrite,
        )
        total_linked += n_linked
        total_missing += n_missing
        if n_linked == 0:
            total_no_src += 1
        if i % 100 == 0 or i == len(rows_in):
            print(f"  [{i}/{len(rows_in)}] linked={total_linked} "
                  f"missing={total_missing} no_src={total_no_src}")

    print()
    print(f"Done. {len(rows_in)} ensembles processed; "
          f"{total_linked} structures linked, {total_missing} missing, "
          f"{total_no_src} ensembles with no source dir.")
    print(f"\nNext: pipeline/01_build_benchmark/compute_gt_matrices.py {args.dataset}"
          f" → filter_ensembles.py {args.dataset}")


if __name__ == "__main__":
    main()
