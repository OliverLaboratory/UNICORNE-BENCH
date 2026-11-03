"""
TM-align C++ executable wrapper.

Uses the official TM-align from https://aideepmed.com/TM-align/TMalign.cpp
instead of the Python tmtools package.

Install TMalign:
  wget https://aideepmed.com/TM-align/TMalign.cpp
  g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp   # Linux
  g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp   # Mac (omit -static)

Set TMALIGN_EXE to the path of the executable, or ensure TMalign is in PATH.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Single-letter to three-letter amino acid codes (for writing PDB)
AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK",
}


def _get_tmalign_exe() -> str:
    """Return path to TMalign executable."""
    exe = os.environ.get("TMALIGN_EXE") or os.environ.get("TMALIGN")
    if exe:
        return exe
    return "TMalign"


def _coords_to_pdb(coords: np.ndarray, sequence: str, path: Path) -> None:
    """Write CA coordinates to a minimal PDB file."""
    n = coords.shape[0]
    seq = sequence or ("A" * n)
    if len(seq) < n:
        seq = seq + "A" * (n - len(seq))
    seq = seq[:n]

    with open(path, "w") as f:
        for i in range(n):
            aa = seq[i] if i < len(seq) else "A"
            aa3 = AA_ONE_TO_THREE.get(aa.upper(), "ALA")
            x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
            # PDB ATOM format: columns 31-54 for x,y,z
            line = (
                f"ATOM  {i+1:5d}  CA  {aa3} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            f.write(line)
        f.write("END\n")


def _run_tmalign(pdb1: Path, pdb2: Path) -> Tuple[float, float, float]:
    """
    Run TMalign on two PDB files. Returns (TM1, TM2, RMSD).
    TM1 = normalized by the length of chain1 (pdb1), TM2 = by chain2 (pdb2).
    """
    exe = _get_tmalign_exe()
    result = subprocess.run(
        [exe, str(pdb1), str(pdb2), "-outfmt", "2"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TMalign failed (exit {result.returncode}): {result.stderr or result.stdout}"
        )

    # -outfmt 2: tab-separated line after header
    # Format: name1  name2  TM1  TM2  RMSD  ID1  ID2  IDali  L1  L2  Lali
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            try:
                tm1 = float(parts[2])  # TM normalized by chain1
                tm2 = float(parts[3])  # TM normalized by chain2
                rmsd = float(parts[4])
                return tm1, tm2, rmsd
            except ValueError:
                pass
    raise RuntimeError(f"Could not parse TMalign output:\n{result.stdout}")


def tm_align(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    seq_a: str = "",
    seq_b: str = "",
) -> Tuple[float, float, float]:
    """
    Compute TM-score between two CA coordinate sets using TMalign executable.

    Returns (tm_norm_chain1, tm_norm_chain2, rmsd).
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    n_a, n_b = coords_a.shape[0], coords_b.shape[0]
    seq_a = seq_a or ("A" * n_a)
    seq_b = seq_b or ("A" * n_b)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pdb_a = tmp / "a.pdb"
        pdb_b = tmp / "b.pdb"
        _coords_to_pdb(coords_a, seq_a, pdb_a)
        _coords_to_pdb(coords_b, seq_b, pdb_b)
        tm1, tm2, rmsd = _run_tmalign(pdb_a, pdb_b)
    return tm1, tm2, rmsd


def tm_score_symmetric(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    seq_a: str = "",
    seq_b: str = "",
) -> float:
    """
    Symmetric TM-score: (TM(A,B) + TM(B,A)) / 2.
    Averages both normalizations for use as kernel.
    """
    tm1, tm2, _rmsd = tm_align(coords_a, coords_b, seq_a, seq_b)
    return (tm1 + tm2) / 2.0
