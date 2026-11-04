"""
Helpers shared by the prediction runners in runners/.

Standard library only: every runner imports this module from inside its own
tool environment (BioEMU venv, ColabFold pixi env, ...).
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Optional


def load_input_csv(path: Path) -> list[dict]:
    """Load input_dataset.csv as [{"ensemble_ID", "ref_sequence"}, ...].

    Accepts the column spellings used across datasets (ensemble_ID /
    ensemble_id / uniprot, ref_sequence / sequence); rows missing either
    field are skipped.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eid = (row.get("ensemble_ID") or row.get("ensemble_id")
                   or row.get("uniprot", "")).strip()
            seq = (row.get("ref_sequence") or row.get("sequence", "")).strip()
            if eid and seq:
                rows.append({"ensemble_ID": eid, "ref_sequence": seq})
    return rows


def count_existing_pdbs(d: Path, pattern: str = "*.pdb") -> int:
    """Number of structure files already written for an ensemble (for --resume)."""
    return len(list(d.glob(pattern))) if d.exists() else 0


def load_a3m(msa_dir: Path, eid: str) -> Optional[str]:
    """Raw A3M text for one ensemble from a ColabFold MSA directory, or None."""
    for candidate in (msa_dir / eid / "a3m" / f"{eid}.a3m",
                      msa_dir / f"{eid}-search" / "0.a3m",
                      msa_dir / eid / "0.a3m"):
        if candidate.exists():
            return candidate.read_text()
    return None


def report_gpu() -> None:
    """Print GPU utilization and memory via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    print(f"  GPU {parts[0]}: {parts[1]} | util {parts[2]}% | mem {parts[3]}/{parts[4]} MiB")
        else:
            print("  (nvidia-smi not available)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  (nvidia-smi not available)")
