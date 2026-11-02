#!/usr/bin/env python3
"""
Carve a masking-fraction subset out of an existing AFsample3 run.

`run_afsample3.py` samples a target in successive passes, one MSA-masking
fraction per pass, and `harvest_cifs` converts *only that pass's* new CIFs
immediately afterwards, numbering them sequentially from the current file
count. Each pass therefore owns a contiguous, fraction-homogeneous block of
``output_NNNN.pdb``, and the run log records exactly where each block starts:

    P17726 run 2/8 (rand=0.1, seeds->50, existing=125/1000)
      +125 structures in 277s (total: 250)
      -> output_0125 .. output_0249 are all msa_rand_fraction=0.1

(The *order within* a block is lexicographic by seed and therefore arbitrary,
but every file in the block shares one fraction, which is all we need.)

This script replays the logs to recover that mapping and re-materialises the
requested fractions into a new run directory, so a differently-configured run
can reuse the overlap instead of resampling it.

Files are hard-linked when source and destination share a filesystem (free,
instant); otherwise copied.

Usage:
  # validate only -- parse logs, check against files on disk, change nothing
  python reuse_fractions.py --dataset UNICORNE_BENCH \\
      --src-run afsample3 --dst-run afsample3_biorxiv \\
      --keep 0.1 0.2 0.3 --logs 'logs/afsample3_*.log' --dry-run

  # materialise
  python reuse_fractions.py ... --keep 0.1 0.2 0.3 --logs '...' --apply
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# "[pass 3] [12/210] P17726 (len=60)"  or  "[12/210] P17726 (len=60)"
RE_ENS = re.compile(r"^(?:\[pass \d+\]\s*)?\[\d+/\d+\]\s+(\S+)\s+\(len=\d+\)")
# "  P17726 run 2/8 (rand=0.1, seeds->50, existing=125/1000, attn=triton)"
RE_RUN = re.compile(
    r"^\s+(\S+)\s+run\s+\d+/\d+\s+\(rand=([\d.]+),\s+seeds->\d+,\s+existing=(\d+)/\d+"
)
# "    +125 structures in 277s (total: 250)"
RE_ADD = re.compile(r"^\s+\+(\d+)\s+structures in")


def parse_logs(patterns: list[str]) -> dict[str, list[tuple[float, int, int]]]:
    """eid -> list of (fraction, start_index, count), in the order produced."""
    blocks: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(globmod.glob(os.path.expanduser(pat))))
    if not files:
        sys.exit(f"no log files matched: {patterns}")

    for path in files:
        pending: tuple[str, float, int] | None = None
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = RE_RUN.match(line)
                if m:
                    pending = (m.group(1), float(m.group(2)), int(m.group(3)))
                    continue
                m = RE_ADD.match(line)
                if m and pending is not None:
                    eid, frac, start = pending
                    n = int(m.group(1))
                    if n > 0:
                        blocks[eid].append((frac, start, n))
                    pending = None
                    continue
                # A new ensemble header invalidates any run line that never
                # reported structures (crashed / skipped pass).
                if RE_ENS.match(line):
                    pending = None
    print(f"parsed {len(files)} log files, {len(blocks)} ensembles with blocks")
    return blocks


def reconcile(blocks: list[tuple[float, int, int]], n_files: int):
    """Drop overlapping/stale blocks and verify coverage.

    A resumed ensemble restarts its fraction cycle, so later runs can re-report
    a start index already claimed by an earlier (pruned) attempt. Keep the last
    writer for any index, which matches what is actually on disk.
    """
    owner: dict[int, float] = {}
    for frac, start, n in blocks:
        for i in range(start, start + n):
            owner[i] = frac
    covered = sum(1 for i in owner if i < n_files)
    return owner, covered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UNICORNE_BENCH")
    ap.add_argument("--src-run", required=True, help="e.g. afsample3")
    ap.add_argument("--dst-run", required=True, help="e.g. afsample3_biorxiv")
    ap.add_argument("--keep", nargs="+", type=float, required=True,
                    help="masking fractions to carry over, e.g. 0.1 0.2 0.3")
    ap.add_argument("--logs", nargs="+", required=True,
                    help="glob(s) for the source run's logs")
    ap.add_argument("--max-per-fraction", type=int, default=None,
                    help="cap how many structures to reuse per fraction")
    ap.add_argument("--output-base", type=Path,
                    default=PROJECT_ROOT / "data" / "outputs")
    ap.add_argument("--apply", action="store_true", help="actually write files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.apply and not args.dry_run:
        sys.exit("pass --dry-run or --apply")

    keep = {round(f, 4) for f in args.keep}
    src_root = args.output_base / args.src_run / args.dataset / "output_pdbs"
    dst_root = args.output_base / args.dst_run / args.dataset / "output_pdbs"
    if not src_root.is_dir():
        sys.exit(f"missing source run: {src_root}")

    blocks = parse_logs(args.logs)

    n_ens = n_reused = n_missing_log = n_mismatch = 0
    per_frac: dict[float, int] = defaultdict(int)
    manifest: dict[str, dict] = {}

    for eid_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        eid = eid_dir.name
        files = sorted(eid_dir.glob("output_*.pdb"))
        if not files:
            continue
        n_files = len(files)
        if eid not in blocks:
            n_missing_log += 1
            continue

        owner, covered = reconcile(blocks[eid], n_files)
        if covered < n_files:
            # Logs don't account for every file on disk -> don't guess.
            n_mismatch += 1
            continue

        chosen: list[tuple[int, float]] = []
        taken: dict[float, int] = defaultdict(int)
        for idx in range(n_files):
            frac = owner.get(idx)
            if frac is None or round(frac, 4) not in keep:
                continue
            if args.max_per_fraction and taken[frac] >= args.max_per_fraction:
                continue
            taken[frac] += 1
            chosen.append((idx, frac))
        if not chosen:
            continue

        n_ens += 1
        n_reused += len(chosen)
        for _, f in chosen:
            per_frac[f] += 1
        manifest[eid] = {"n_reused": len(chosen),
                         "per_fraction": {str(k): v for k, v in sorted(taken.items())}}

        if args.apply:
            out_dir = dst_root / eid
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, (idx, _frac) in enumerate(chosen):
                src = eid_dir / f"output_{idx:04d}.pdb"
                dst = out_dir / f"output_{j:04d}.pdb"
                if dst.exists() or not src.exists():
                    continue
                try:
                    os.link(src, dst)          # same filesystem: free
                except OSError:
                    shutil.copyfile(src, dst)  # cross-device fallback

    print(f"\nensembles with reusable structures : {n_ens}")
    print(f"structures reused                  : {n_reused}")
    print(f"per fraction                       : "
          f"{ {k: v for k, v in sorted(per_frac.items())} }")
    print(f"ensembles with no log coverage     : {n_missing_log}")
    print(f"ensembles where logs != files      : {n_mismatch}  (skipped, will resample)")

    if args.apply:
        dst_root.parent.mkdir(parents=True, exist_ok=True)
        mpath = dst_root.parent / "reuse_manifest.json"
        mpath.write_text(json.dumps(
            {"src_run": args.src_run, "kept_fractions": sorted(keep),
             "ensembles": manifest}, indent=2))
        print(f"\nwrote {dst_root}")
        print(f"manifest -> {mpath}")
    else:
        print("\n(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
