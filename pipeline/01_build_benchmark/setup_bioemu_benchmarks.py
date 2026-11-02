#!/usr/bin/env python3
"""
Set up input datasets for bioemu-benchmarks multiconf benchmarks.

Downloads reference PDBs from RCSB and creates:
  data/inputs/<DATASET>/input_dataset.csv
  data/inputs/<DATASET>/input_pdbs/<ensemble_ID>/<pdbid_chain>.pdb

Usage:
  python scripts/setup_bioemu_benchmarks.py
  python scripts/setup_bioemu_benchmarks.py --datasets OOD60 DOMAINMOTION
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw GitHub URLs for bioemu-benchmarks asset CSVs
GITHUB_RAW = "https://raw.githubusercontent.com/microsoft/bioemu-benchmarks/main"
ASSET_BASE = f"{GITHUB_RAW}/bioemu_benchmarks/assets/multiconf_benchmark_0.1"

BENCHMARKS = {
    "OOD60": {
        "testcases": f"{ASSET_BASE}/ood60/testcases.csv",
        "references": f"{ASSET_BASE}/ood60/references.csv",
    },
    "DOMAINMOTION": {
        "testcases": f"{ASSET_BASE}/domainmotion/testcases.csv",
        "references": f"{ASSET_BASE}/domainmotion/references.csv",
    },
}


def fetch_csv(url: str) -> list[dict]:
    """Fetch a CSV from a URL and return rows as dicts."""
    print(f"  Fetching {url.split('/')[-1]} ...")
    resp = urllib.request.urlopen(url)
    text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def download_pdb_chain(pdb_id: str, chain: str, out_path: Path) -> bool:
    """Download a PDB chain from RCSB and extract the specified chain.

    Falls back to mmCIF if PDB format doesn't have the chain.
    """
    if out_path.exists() and out_path.stat().st_size > 100:
        return True  # already exists

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_id_lower = pdb_id.lower()
    chain_upper = chain.upper()

    # Try PDB format first
    url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
    try:
        resp = urllib.request.urlopen(url)
        pdb_text = resp.read().decode("utf-8")
        n_atoms = _extract_chain_pdb(pdb_text, chain_upper, out_path)
        if n_atoms > 0:
            return True
    except Exception:
        pass

    # Fallback: mmCIF → extract backbone as PDB
    url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif"
    try:
        resp = urllib.request.urlopen(url)
        cif_text = resp.read().decode("utf-8")
        n_atoms = _extract_chain_cif(cif_text, chain_upper, out_path)
        if n_atoms > 0:
            return True
    except Exception as e:
        print(f"    Warning: failed to download {pdb_id} (PDB+CIF): {e}")

    print(f"    Warning: no ATOM records for chain {chain} in {pdb_id}")
    out_path.unlink(missing_ok=True)
    return False


def _extract_chain_pdb(pdb_text: str, chain: str, out_path: Path) -> int:
    """Extract chain from PDB format text. Returns atom count."""
    n_atoms = 0
    with open(out_path, "w") as f:
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and len(line) > 21 and line[21] == chain:
                f.write(line + "\n")
                n_atoms += 1
        f.write("END\n")
    if n_atoms == 0:
        out_path.unlink(missing_ok=True)
    return n_atoms


def _extract_chain_cif(cif_text: str, chain: str, out_path: Path) -> int:
    """Extract chain from mmCIF format, write as PDB. Returns atom count."""
    # Parse _atom_site records from mmCIF
    n_atoms = 0
    in_atom_site = False
    col_names: list[str] = []

    with open(out_path, "w") as f:
        for line in cif_text.splitlines():
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_names.append(line.strip().split(".")[1])
                continue
            if in_atom_site and not line.startswith("_") and not line.startswith("#"):
                if not line.strip() or line.startswith("loop_"):
                    in_atom_site = False
                    continue
                parts = line.split()
                if len(parts) < len(col_names):
                    continue
                row = dict(zip(col_names, parts))

                if row.get("group_PDB") != "ATOM":
                    continue
                auth_chain = row.get("auth_asym_id", "")
                label_chain = row.get("label_asym_id", "")
                if auth_chain != chain and label_chain != chain:
                    continue

                # Write PDB ATOM line
                atom_name = row.get("auth_atom_id", row.get("label_atom_id", "CA"))
                resname = row.get("auth_comp_id", row.get("label_comp_id", "ALA"))
                resseq = row.get("auth_seq_id", row.get("label_seq_id", "1"))
                x = float(row.get("Cartn_x", 0))
                y = float(row.get("Cartn_y", 0))
                z = float(row.get("Cartn_z", 0))
                serial = n_atoms + 1

                atom_name_fmt = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
                pdb_line = (
                    f"ATOM  {serial:5d} {atom_name_fmt} {resname:>3s} {chain}{int(resseq):4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                f.write(pdb_line)
                n_atoms += 1
            elif in_atom_site and (line.startswith("#") or line.startswith("loop_")):
                in_atom_site = False
        f.write("END\n")

    if n_atoms == 0:
        out_path.unlink(missing_ok=True)
    return n_atoms


def setup_dataset(name: str, data_dir: Path) -> None:
    """Set up one dataset: fetch testcases, download reference PDBs."""
    print(f"\n=== {name} ===")

    urls = BENCHMARKS[name]
    testcases = fetch_csv(urls["testcases"])
    references = fetch_csv(urls["references"])

    ds_dir = data_dir / "inputs" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    pdbs_dir = ds_dir / "input_pdbs"

    # Deduplicate by test_case (keep first occurrence — variants differ by His-tags etc.)
    seen = set()
    unique_testcases = []
    for tc in testcases:
        tid = tc["test_case"]
        if tid not in seen:
            seen.add(tid)
            unique_testcases.append(tc)
    if len(unique_testcases) < len(testcases):
        print(f"  Deduplicated: {len(testcases)} -> {len(unique_testcases)} test cases")
    testcases = unique_testcases

    # Write input_dataset.csv
    csv_path = ds_dir / "input_dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble_ID", "ref_sequence"])
        writer.writeheader()
        for tc in testcases:
            writer.writerow({
                "ensemble_ID": tc["test_case"],
                "ref_sequence": tc["sequence"],
            })
    print(f"  Wrote {csv_path} ({len(testcases)} test cases)")

    # Download reference PDBs
    # references.csv has columns like: test_case, pdbidchain_i, pdbidchain_j
    # or: pdbidchain_i, pdbidchain_j, test_case
    n_downloaded = 0
    n_skipped = 0
    for ref in references:
        test_case = ref.get("test_case", "").strip()
        if not test_case:
            continue

        ensemble_dir = pdbs_dir / test_case
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        for col in ["pdbidchain_i", "pdbidchain_j"]:
            pdbchain = ref.get(col, "").strip()
            if not pdbchain:
                continue

            # Parse pdbid_chain (e.g., "6gts_B" or "6PWJ_A")
            parts = pdbchain.replace("-", "_").split("_")
            if len(parts) >= 2:
                pdb_id = parts[0]
                chain = parts[1]
            else:
                print(f"    Warning: can't parse {pdbchain}")
                continue

            out_file = ensemble_dir / f"{pdb_id.upper()}_{chain.upper()}.pdb"
            if download_pdb_chain(pdb_id, chain, out_file):
                n_downloaded += 1
            else:
                n_skipped += 1

    print(f"  Reference PDBs: {n_downloaded} downloaded/cached, {n_skipped} failed")

    # List ensemble counts
    n_ensembles = sum(1 for d in pdbs_dir.iterdir() if d.is_dir()) if pdbs_dir.exists() else 0
    print(f"  Total: {n_ensembles} ensembles in {pdbs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Set up input datasets for bioemu-benchmarks multiconf benchmarks")
    parser.add_argument(
        "--datasets", nargs="+", default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Which datasets to set up (default: all)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data",
        help="Root data directory",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        setup_dataset(ds, args.data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
