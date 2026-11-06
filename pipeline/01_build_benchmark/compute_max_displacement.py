#!/usr/bin/env python3
"""
Compute per-ensemble max CA displacement matrix using TMalign's structural
alignment.

For each pair (i, j) of GT conformations:
  1. Run TMalign with ``-m`` to get:
     - the rotation+translation matrix rotating Chain_1 onto Chain_2 (TM-align
       only writes it when ``-m <file>`` is given; default stdout has none), and
     - the per-residue alignment markers (":", "." for aligned pairs; "-" for
       indels/unaligned residues).
  2. Apply the TMalign transform to chain_j's CA coords -> chain_j' in chain_i's frame.
  3. For each aligned residue pair (i_a in chain_i, i_b in chain_j), compute
     ||ca_i[i_a] - ca_j_aligned[i_b]||.
  4. Take the max.  This is the largest per-residue CA displacement AMONG the
     TMalign-aligned residues (the structural anchor), so disordered / indel
     tails that TMalign chose to drop do not inflate the number.

Rationale: whole-protein TM saturates for motions localized to a small region
(the static majority dominates), masking hinge/loop dynamics. A naive
all-CA Kabsch + max displacement picks up those motions but is polluted by
disordered tails that happen to differ between PDB chains. Using TMalign's
alignment as the anchor resolves both problems: we measure motion only on
residues that both chains agree on, using TMalign's best superposition.

Cache layout (per ensemble):
  results/<DATASET>/gt_matrices_tmalign/<eid>/max_disp_xx.npy
    shape (n_gt, n_gt) float, diagonal = 0, symmetric; NaN for pairs where
    TMalign failed or zero aligned residues.

Usage:
  python pipeline/01_build_benchmark/compute_max_displacement.py DANCE_BENCH --resume
  python pipeline/01_build_benchmark/compute_max_displacement.py UNICORNE_BENCH --resume
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conformation_benchmark.datasets.folder_dataset import _parse_structure_ca
from conformation_benchmark.evaluation.tmalign_exe import _coords_to_pdb, _get_tmalign_exe


_ROT_HEADER_RE = re.compile(
    r"rotation matrix to rotate Chain_1 to Chain_2", re.IGNORECASE
)


def _parse_tmalign_output(stdout: str):
    """Return (rotation_matrix_4x3, aligned_pairs).

    rotation_matrix_4x3: np.ndarray shape (3, 4) where first col is translation
      t, remaining 3 cols are the rotation matrix u.  Applied to Chain_1 coords
      gives them in Chain_2's frame: y = u @ x + t.
    aligned_pairs: list[(i1, i2)] of 0-indexed residue pairs that TMalign
      aligned (':' or '.' column in the alignment block).
    """
    lines = stdout.splitlines()

    # --- Rotation matrix -------------------------------------------------
    # Header "... rotate Chain_1 to Chain_2 ..." then one blank line, then
    # lines with columns: i  t(i)  u(i,1) u(i,2) u(i,3)
    rot_t = np.zeros(3)
    rot_u = np.eye(3)
    for idx, line in enumerate(lines):
        if _ROT_HEADER_RE.search(line):
            for j in range(3):
                parts = lines[idx + 2 + j].split()
                if len(parts) >= 5:
                    rot_t[j] = float(parts[1])
                    rot_u[j, 0] = float(parts[2])
                    rot_u[j, 1] = float(parts[3])
                    rot_u[j, 2] = float(parts[4])
            break

    # --- Alignment block -------------------------------------------------
    # After a line like:
    #   (":" denotes residue pairs of d < 5.0 Angstrom, "." denotes other ...)
    # there are three lines: seq1, markers, seq2. Sometimes blocked into
    # multiple 60-char chunks (each chunk = 3 lines). Collect by scanning
    # for triplets where line[k+1] contains only :./space.
    aligned_pairs: list[tuple[int, int]] = []
    seq1_chunks: list[str] = []
    marker_chunks: list[str] = []
    seq2_chunks: list[str] = []

    i = 0
    while i < len(lines) - 2:
        s1 = lines[i]
        mk = lines[i + 1]
        s2 = lines[i + 2]
        # A "sequence row" is mostly letters and dashes; a "marker row" is
        # mostly ':', '.', and spaces. Detect by ratio.
        def _is_seq(s: str) -> bool:
            if not s.strip():
                return False
            n = len(s.strip())
            return sum(1 for c in s if c.isalpha() or c == "-") / max(n, 1) > 0.9

        def _is_marker(s: str) -> bool:
            if not s.strip():
                return False
            n = len(s.rstrip())
            return sum(1 for c in s.rstrip() if c in " :.") / max(n, 1) > 0.9

        if _is_seq(s1) and _is_marker(mk) and _is_seq(s2):
            w = min(len(s1), len(s2), len(mk))
            seq1_chunks.append(s1[:w])
            marker_chunks.append(mk[:w])
            seq2_chunks.append(s2[:w])
            i += 3
        else:
            i += 1

    seq1 = "".join(seq1_chunks)
    mk = "".join(marker_chunks)
    seq2 = "".join(seq2_chunks)

    # Walk the alignment, tracking residue index in each chain.
    i1 = i2 = 0
    for col in range(min(len(seq1), len(seq2), len(mk))):
        c1, cm, c2 = seq1[col], mk[col], seq2[col]
        if c1 != "-" and c2 != "-":
            if cm in (":", "."):
                aligned_pairs.append((i1, i2))
        if c1 != "-":
            i1 += 1
        if c2 != "-":
            i2 += 1

    return rot_t, rot_u, aligned_pairs


def tmalign_max_displacement(ca_i: np.ndarray, ca_j: np.ndarray, tmalign_exe: str) -> float:
    """Run TMalign, apply its rotation, measure max CA displacement over aligned residues."""
    if ca_i.shape[0] < 3 or ca_j.shape[0] < 3:
        return float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pdb_i = tmp / "i.pdb"
        pdb_j = tmp / "j.pdb"
        _coords_to_pdb(ca_i, "A" * ca_i.shape[0], pdb_i)
        _coords_to_pdb(ca_j, "A" * ca_j.shape[0], pdb_j)
        matrix_file = tmp / "matrix.txt"
        try:
            result = subprocess.run(
                [tmalign_exe, str(pdb_i), str(pdb_j), "-m", str(matrix_file)],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return float("nan")
        if result.returncode != 0 or not matrix_file.exists():
            return float("nan")
        # The -m file has the layout the parser expects (header line,
        # column-name line, three "m t u u u" rows); prepend it to stdout.
        matrix_text = matrix_file.read_text()

    rot_t, rot_u, aligned_pairs = _parse_tmalign_output(matrix_text + "\n" + result.stdout)
    if not aligned_pairs:
        return float("nan")
    if np.allclose(rot_u, np.eye(3)) and np.allclose(rot_t, 0.0):
        # No superposition parsed: would silently measure in the raw file
        # frames (the pre-fix behaviour). Treat as a failure instead.
        return float("nan")

    # TMalign's "rotate Chain_1 to Chain_2": y = u @ x + t gives chain_1
    # coords in chain_2's frame. We want both chains in a common frame so we
    # can compute per-residue displacement. Transform chain_i (Chain_1) into
    # chain_j's (Chain_2) frame.
    ca_i_in_j = ca_i @ rot_u.T + rot_t

    disps = []
    for a, b in aligned_pairs:
        if a >= ca_i.shape[0] or b >= ca_j.shape[0]:
            continue
        disps.append(np.linalg.norm(ca_i_in_j[a] - ca_j[b]))
    if not disps:
        return float("nan")
    return float(max(disps))


def process_ensemble(eid: str, gt_pdbs_dir: Path, out_dir: Path, tmalign_exe: str) -> int | None:
    eid_dir = gt_pdbs_dir / eid
    if not eid_dir.is_dir():
        return None
    pdb_files = sorted(list(eid_dir.glob("*.pdb")) + list(eid_dir.glob("*.cif")))
    confs = []
    for p in pdb_files:
        ca = _parse_structure_ca(p)
        if ca.shape[0] > 0:
            confs.append(ca)
    n = len(confs)
    if n < 2:
        return None

    M = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(a + 1, n):
            d = tmalign_max_displacement(confs[a], confs[b], tmalign_exe)
            M[a, b] = M[b, a] = d

    eid_out = out_dir / eid
    eid_out.mkdir(parents=True, exist_ok=True)
    np.save(eid_out / "max_disp_xx.npy", M)
    return n


def main():
    parser = argparse.ArgumentParser(description="Compute GT pairwise max-displacement matrices (TMalign-anchored)")
    parser.add_argument("dataset", help="Dataset name (e.g. DANCE_BENCH, UNICORNE_BENCH)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ensembles whose max_disp_xx.npy already exists")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Directory of per-ensemble GT structures "
                             "(default: data/inputs/<dataset>/input_pdbs)")
    args = parser.parse_args()

    tmalign_exe = _get_tmalign_exe()
    if not shutil.which(tmalign_exe) and not Path(tmalign_exe).exists():
        print(f"Error: TMalign executable not found at {tmalign_exe}", file=sys.stderr)
        sys.exit(1)

    gt_pdbs_dir = args.gt_dir or (PROJECT_ROOT / "data" / "inputs" / args.dataset / "input_pdbs")
    out_dir = PROJECT_ROOT / "results" / args.dataset / "gt_matrices_tmalign"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_eids = sorted(d.name for d in gt_pdbs_dir.iterdir() if d.is_dir())
    if args.offset:
        all_eids = all_eids[args.offset:]
    if args.limit:
        all_eids = all_eids[:args.limit]

    if args.resume:
        done = {d.name for d in out_dir.iterdir()
                if d.is_dir() and (d / "max_disp_xx.npy").exists()}
        todo = [e for e in all_eids if e not in done]
        print(f"Resume: {len(done)} already done, {len(todo)} remaining")
    else:
        todo = all_eids

    t0 = time.time()
    for idx, eid in enumerate(todo):
        n_confs = process_ensemble(eid, gt_pdbs_dir, out_dir, tmalign_exe)
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{len(todo)}] {eid}: n={n_confs or 0}  "
              f"{rate:.2f} ens/s, ~{remaining/60:.0f} min left",
              end="\r", flush=True)
    print(f"\nDone: {len(todo)} ensembles in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
