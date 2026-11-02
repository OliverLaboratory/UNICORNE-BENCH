"""
Dataset structure: INPUT --> [TOOL] --> OUTPUT --> [EVALUATOR]

INPUT (ground truth):
  <dataset_name>/
    input_pdbs/<ensemble_ID>/*.pdb   # one PDB per conformation
    input_dataset.csv                 # ensemble_ID, ref_sequence

OUTPUT (tool predictions):
  <dataset_name>_<TOOL>/
    output_pdbs/<ensemble_ID>/*.pdb
    output_dataset.csv                # ensemble_ID, ref_sequence (only succeeded)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Single-letter to three-letter amino acid codes (for writing PDB)
AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK",
}


def _parse_pdb_ca(pdb_path: Union[str, Path]) -> np.ndarray:
    """Extract CA coordinates from a PDB file. Returns (N, 3) array."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        pass
    return np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))


def _parse_cif_ca(cif_path: Union[str, Path]) -> np.ndarray:
    """Extract CA coordinates from an mmCIF file. Returns (N, 3) array."""
    coords = []
    in_loop = False
    col_names: list[str] = []
    with open(cif_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("loop_"):
                in_loop = True
                col_names = []
                continue
            if in_loop and stripped.startswith("_atom_site."):
                col_names.append(stripped)
                continue
            if in_loop and col_names and not stripped.startswith("_"):
                # Data row
                if not stripped or stripped.startswith("#"):
                    in_loop = False
                    continue
                fields = stripped.split()
                if len(fields) < len(col_names):
                    continue
                col_map = {c: i for i, c in enumerate(col_names)}
                group_idx = col_map.get("_atom_site.group_PDB")
                atom_idx = col_map.get("_atom_site.label_atom_id")
                x_idx = col_map.get("_atom_site.Cartn_x")
                y_idx = col_map.get("_atom_site.Cartn_y")
                z_idx = col_map.get("_atom_site.Cartn_z")
                if None in (group_idx, atom_idx, x_idx, y_idx, z_idx):
                    continue
                if fields[group_idx] == "ATOM" and fields[atom_idx] == "CA":
                    try:
                        coords.append([
                            float(fields[x_idx]),
                            float(fields[y_idx]),
                            float(fields[z_idx]),
                        ])
                    except ValueError:
                        pass
    return np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))


def _parse_structure_ca(path: Union[str, Path]) -> np.ndarray:
    """Extract CA coordinates from a PDB or mmCIF file (auto-detect by extension)."""
    path = Path(path)
    if path.suffix.lower() == ".cif":
        return _parse_cif_ca(path)
    return _parse_pdb_ca(path)


def _write_pdb_ca(coords: np.ndarray, sequence: str, path: Union[str, Path]) -> None:
    """Write a CA-only PDB from (N, 3) Angstrom coords and a sequence.

    Uses the biopython library (Bio.PDB.PDBIO) to emit spec-correct fixed-width
    PDB -- we no longer hand-format ATOM records. The previous hand-rolled writer
    omitted the chain-ID column, which shifted the coordinate fields one place
    left, so any coordinate that filled the 8-char field (e.g. values <= -100)
    collided with the previous coordinate and broke whitespace parsers. A
    canonical manual fallback (with a chain ID) is used only if biopython is
    unavailable."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    resnames = [AA_ONE_TO_THREE.get(sequence[i] if i < len(sequence) else "X", "UNK")
                for i in range(n)]
    try:
        from Bio.PDB import StructureBuilder, PDBIO
        sb = StructureBuilder.StructureBuilder()
        sb.init_structure("s")
        sb.init_model(0)
        sb.init_chain("A")
        for i, (x, y, z) in enumerate(coords):
            sb.init_seg(" ")
            sb.init_residue(resnames[i], " ", i + 1, " ")
            sb.init_atom("CA", (float(x), float(y), float(z)),
                         0.0, 1.0, " ", " CA ", i + 1, "C")
        io = PDBIO()
        io.set_structure(sb.get_structure())
        io.save(str(path))
    except Exception:
        with open(path, "w") as f:
            for i, (x, y, z) in enumerate(coords):
                f.write(
                    f"ATOM  {i+1:5d}  CA  {resnames[i]:>3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")


def load_input_dataset(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load INPUT dataset from folder structure.
    Returns: { ensemble_id: { 'sequence': str, 'conformations': [ (N,3) arrays ] } }
    """
    path = Path(path)
    csv_path = path / "input_dataset.csv"
    pdbs_dir = path / "input_pdbs"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    if not pdbs_dir.exists():
        raise FileNotFoundError(f"Missing {pdbs_dir}")

    # Load sequences from CSV
    sequences = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = (
                row.get("ensemble_ID")
                or row.get("ensemble_id")
                or row.get("uniprot", "")
            ).strip()
            seq = (
                row.get("ref_sequence")
                or row.get("sequence", "")
            ).strip()
            if eid and seq:
                sequences[eid] = seq

    # Load conformations from PDBs
    out = {}
    for eid, seq in sequences.items():
        ensemble_dir = pdbs_dir / eid
        if not ensemble_dir.exists():
            continue
        conformations = []
        for pdb in sorted(ensemble_dir.glob("*.pdb")):
            ca = _parse_pdb_ca(pdb)
            if ca.shape[0] > 0:
                conformations.append(ca)
        if conformations:
            out[eid] = {"sequence": seq, "conformations": conformations}
    return out


def load_output_dataset(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load OUTPUT (tool predictions) from folder structure.
    Same return format as load_input_dataset.
    """
    path = Path(path)
    csv_path = path / "output_dataset.csv"
    pdbs_dir = path / "output_pdbs"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    if not pdbs_dir.exists():
        raise FileNotFoundError(f"Missing {pdbs_dir}")

    sequences = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = (
                row.get("ensemble_ID")
                or row.get("ensemble_id")
                or row.get("uniprot", "")
            ).strip()
            seq = (
                row.get("ref_sequence")
                or row.get("sequence", "")
            ).strip()
            if eid and seq:
                sequences[eid] = seq

    out = {}
    for eid, seq in sequences.items():
        ensemble_dir = pdbs_dir / eid
        if not ensemble_dir.exists():
            continue
        conformations = []
        for pdb in sorted(ensemble_dir.glob("*.pdb")):
            ca = _parse_pdb_ca(pdb)
            if ca.shape[0] > 0:
                conformations.append(ca)
        if conformations:
            out[eid] = {"sequence": seq, "conformations": conformations}
    return out


def save_output_dataset(
    predictions: Dict[str, Dict[str, Any]],
    path: Union[str, Path],
    pdb_prefix: str = "output",
) -> None:
    """
    Save predictions to OUTPUT folder structure.

    Creates:
      path/output_pdbs/<ensemble_ID>/*.pdb
      path/output_dataset.csv
    """
    path = Path(path)
    pdbs_dir = path / "output_pdbs"
    csv_path = path / "output_dataset.csv"
    path.mkdir(parents=True, exist_ok=True)
    pdbs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for eid, rec in predictions.items():
        seq = rec["sequence"]
        conformations = rec["conformations"]
        rows.append({"ensemble_ID": eid, "ref_sequence": seq})
        ensemble_dir = pdbs_dir / eid
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        for i, coords in enumerate(conformations):
            coords = np.asarray(coords, dtype=np.float64)
            pdb_path = ensemble_dir / f"{pdb_prefix}_{i:04d}.pdb"
            _write_pdb_ca(coords, seq, pdb_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble_ID", "ref_sequence"])
        writer.writeheader()
        writer.writerows(rows)


class ConformationFolderDataset:
    """
    Dataset loaded from INPUT folder structure:
      <path>/input_pdbs/<ensemble_ID>/*.pdb
      <path>/input_dataset.csv
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._data: Optional[Dict[str, Dict[str, Any]]] = None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._data is None:
            self._data = load_input_dataset(self.path)
        return self._data

    @property
    def data(self) -> Dict[str, Dict[str, Any]]:
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def __contains__(self, ensemble_id: str) -> bool:
        return ensemble_id in self._load()

    def __getitem__(self, ensemble_id: str) -> Dict[str, Any]:
        return self._load()[ensemble_id]

    def ensemble_ids(self) -> List[str]:
        return list(self._load().keys())
