#!/usr/bin/env python3
"""
Flag GT conformations whose Cα geometry looks fibrillar / massively extended
(unfolded-sheet-like rather than a compact globular protein). Heuristic uses
the gyration tensor's asphericity + the Rg/sqrt(N) ratio.

Reference outliers (P00441):
  - 2MP3_A (fibril): n=125  Rg=33  Rg/sqrt(n)=2.96  asph=0.82  e2e=110 Å
  - 2NAM_A (fibril): n=153  Rg=32  Rg/sqrt(n)=2.59  asph=0.73  e2e=100 Å
  - 1AZV_A (native): n=153  Rg=14  Rg/sqrt(n)=1.13  asph=0.13  e2e=  9 Å

Usage:
  python pipeline/01_build_benchmark/flag_fibril_like.py UNICORNE_BENCH
  python pipeline/01_build_benchmark/flag_fibril_like.py UNICORNE_BENCH --render

Outputs:
  results/<DATASET>/qc/fibril_like.csv
  results/<DATASET>/qc/renders/<eid>__<conf>.png   (with --render)
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conformation_benchmark.datasets.folder_dataset import _parse_structure_ca


def shape_metrics(ca: np.ndarray) -> dict:
    n = ca.shape[0]
    com = ca.mean(0)
    diffs = ca - com
    rg = float(np.sqrt((diffs ** 2).sum(1).mean()))
    G = (diffs[:, :, None] * diffs[:, None, :]).mean(0)
    eigs = sorted(np.linalg.eigvalsh(G), reverse=True)
    s = sum(eigs)
    asph = (eigs[0] - 0.5 * (eigs[1] + eigs[2])) / s if s > 0 else 0.0
    e2e = float(np.linalg.norm(ca[0] - ca[-1])) if n >= 2 else 0.0
    return dict(n=n, rg=rg, rg_per_sqrtn=rg / np.sqrt(max(n, 1)),
                asph=float(asph), e2e=e2e)


def render_pymol(pdb_path: Path, out_png: Path) -> bool:
    """Best-effort PyMOL cartoon render. Skips silently if pymol not on PATH."""
    if not shutil.which("pymol"):
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
load {pdb_path}, mol
hide everything
show cartoon, mol
color spectrum, mol
bg_color white
orient mol
set ray_opaque_background, off
ray 700, 500
png {out_png}
"""
    try:
        subprocess.run(["pymol", "-cqQ"], input=script, text=True,
                       check=True, timeout=60, capture_output=True)
        return out_png.exists()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--asph", type=float, default=0.5,
                    help="Asphericity threshold (default 0.5; globular usually < 0.3)")
    ap.add_argument("--rg-per-sqrtn", type=float, default=2.0,
                    help="Rg/sqrt(n) threshold (default 2.0; globular usually < 1.5)")
    ap.add_argument("--asph-delta", type=float, default=0.3,
                    help="Min asphericity excess vs ensemble median (default 0.3) — "
                         "this is the 'shape-anomalous-vs-peers' check that picks "
                         "the 2MP3_A-style outliers without flagging legitimately "
                         "elongated single-chain ensembles.")
    ap.add_argument("--render", action="store_true",
                    help="Render flagged conformations via PyMOL (-c -q)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap ensembles processed (for testing)")
    args = ap.parse_args()

    inputs_dir = PROJECT_ROOT / "data" / "inputs" / args.dataset / "input_pdbs"
    out_dir = PROJECT_ROOT / "results" / args.dataset / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "fibril_like.csv"
    render_dir = out_dir / "renders"

    ensembles = sorted(d for d in inputs_dir.iterdir() if d.is_dir())
    if args.limit:
        ensembles = ensembles[:args.limit]

    rows: list[dict] = []
    n_seen = n_flagged = 0
    for i, eid_dir in enumerate(ensembles, 1):
        eid = eid_dir.name
        # Compute shape metrics for every conformation, then flag the
        # ones that are shape-anomalous relative to their ensemble peers.
        per_conf: list[tuple[Path, dict]] = []
        for f in sorted(list(eid_dir.glob("*.pdb")) + list(eid_dir.glob("*.cif"))):
            try:
                ca = _parse_structure_ca(f)
            except Exception:
                continue
            if ca.shape[0] < 5:
                continue
            per_conf.append((f, shape_metrics(ca)))
        n_seen += len(per_conf)
        if len(per_conf) < 2:
            # Solo conformation in the ensemble — can't compare relatively.
            continue

        asph_vals = np.array([m["asph"] for _, m in per_conf])
        median_asph = float(np.median(asph_vals))
        for f, m in per_conf:
            # Flag absolute "is fibrillar" + relative "vs peers" check.
            # The relative check is what matters — it picks conformations
            # whose asphericity is dramatically larger than the ensemble's
            # median (e.g., 2MP3_A vs the four globular forms of P00441).
            if (m["asph"] >= args.asph
                    and m["rg_per_sqrtn"] >= args.rg_per_sqrtn
                    and (m["asph"] - median_asph) >= args.asph_delta):
                n_flagged += 1
                rows.append(dict(
                    ensemble_ID=eid,
                    conformation=f.name,
                    n=m["n"],
                    rg=round(m["rg"], 2),
                    rg_per_sqrtn=round(m["rg_per_sqrtn"], 3),
                    asph=round(m["asph"], 3),
                    asph_minus_median=round(m["asph"] - median_asph, 3),
                    e2e=round(m["e2e"], 1),
                    n_peers=len(per_conf) - 1,
                ))
                if args.render:
                    png = render_dir / f"{eid}__{f.stem}.png"
                    if not png.exists():
                        render_pymol(f, png)
        if i % 50 == 0 or i == len(ensembles):
            print(f"  [{i}/{len(ensembles)}] seen={n_seen}  flagged={n_flagged}",
                  flush=True)

    rows.sort(key=lambda r: (-r["asph"], r["ensemble_ID"], r["conformation"]))
    with open(out_csv, "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        else:
            fh.write("# no flagged conformations\n")
    print(f"\nTotal: {n_seen} structures inspected, {n_flagged} flagged.")
    print(f"Wrote {out_csv}")
    if args.render:
        print(f"Renders: {render_dir}")


if __name__ == "__main__":
    main()
