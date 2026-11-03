"""
Minimal C-alpha reader for PDB and mmCIF files, independent of the pipeline.

Follows TM-align's default reading rules: first model, first chain, ATOM
records plus HETATM selenomethionine (MSE), alternate location blank or 'A',
one CA per residue number + insertion code. The tests check that this reader
recovers the same residue count and sequence TM-align reports, so TM-scores
can be recomputed from coordinates without trusting the pipeline's parsing.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}


def read_ca(path: Path) -> tuple[np.ndarray, str]:
    """Return (n x 3 CA coordinates, one-letter sequence)."""
    path = Path(path)
    if path.suffix == ".cif":
        return _read_cif(path)
    return _read_pdb(path)


def _read_pdb(path: Path) -> tuple[np.ndarray, str]:
    xyz, seq, seen, chain = [], [], set(), None
    for line in path.read_text().splitlines():
        if line.startswith(("TER", "ENDMDL", "END")) and xyz:
            break
        is_atom = line.startswith("ATOM  ") or (line.startswith("HETATM") and line[17:20] == "MSE")
        if not is_atom or line[12:16] != " CA " or line[16] not in (" ", "A"):
            continue
        if chain is None:
            chain = line[21]
        if line[21] != chain:
            break
        key = (line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        seq.append(THREE_TO_ONE.get(line[17:20], "X"))
    return np.array(xyz), "".join(seq)


def _read_cif(path: Path) -> tuple[np.ndarray, str]:
    cols, rows = [], []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif cols and line.startswith(("ATOM", "HETATM")):
            rows.append(shlex.split(line))
        elif rows:
            break
    c = {name: i for i, name in enumerate(cols)}
    atom = c.get("auth_atom_id", c["label_atom_id"])
    resnum = c.get("auth_seq_id", c["label_seq_id"])
    chain_col = c.get("auth_asym_id", c["label_asym_id"])
    model_col = c.get("pdbx_PDB_model_num")
    icode = c.get("pdbx_PDB_ins_code")

    xyz, seq, seen, chain, model = [], [], set(), None, None
    for r in rows:
        comp = r[c["label_comp_id"]]
        if r[c["group_PDB"]] == "HETATM" and comp != "MSE":
            continue
        if r[atom] != "CA" or r[c["label_alt_id"]] not in (".", "?", "A"):
            continue
        if model_col is not None:
            model = model or r[model_col]
            if r[model_col] != model:
                break
        chain = chain or r[chain_col]
        if r[chain_col] != chain:
            break
        key = (r[resnum], r[icode] if icode is not None else "")
        if key in seen:
            continue
        seen.add(key)
        xyz.append([float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]), float(r[c["Cartn_z"]])])
        seq.append(THREE_TO_ONE.get(comp, "X"))
    return np.array(xyz), "".join(seq)
