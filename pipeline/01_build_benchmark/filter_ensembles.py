#!/usr/bin/env python3
"""
Filter GT ensembles for the benchmark.

Pipeline:
  1. Length tolerance: drop GT conformations whose CA count differs
     by more than 20% from the reference sequence length.
  2. Max sequence length: drop ensembles with ref_sequence > 300 aa.
  3. TM deduplication: single-linkage clustering at TM >= 0.85,
     keep one medoid per cluster. Single linkage guarantees that
     NO pair of kept conformations has TM >= 0.85.
  4. Minimum ensemble size: drop ensembles with < 2 conformations
     after dedup.

Inputs:
  - data/inputs/{DATASET}/input_dataset.csv  (ensemble_ID, ref_sequence)
  - data/inputs/{DATASET}/input_pdbs/{eid}/*.pdb|*.cif
  - GT pairwise TM matrices (searched in results/{DATASET}/*/matrices_tmalign/{eid}/)

Outputs:
  - data/inputs/{DATASET}/filtered_ensembles_v2.csv
    Columns: ensemble_ID, kept_indices, n_gt_raw, n_gt_filtered,
             n_gt_dedup, ref_seq_len, avg_pairwise_tm, min_pairwise_tm

Usage:
  python scripts/filter_ensembles.py DANCE
  python scripts/filter_ensembles.py UNICORNE
  python scripts/filter_ensembles.py ALL          # process all datasets
  python scripts/filter_ensembles.py DANCE --max-len 500  # override max length
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Defaults ---
LENGTH_TOLERANCE = 0.20   # drop GT conf if |CA_count - ref_len| > tol * ref_len
DEDUP_TM = 0.85           # single-linkage dedup threshold
MIN_GT = 2                # minimum GT conformations after dedup
MAX_SEQ_LEN = 300         # maximum reference sequence length
DATASETS = ["DANCE", "UNICORNE", "DANCE_BENCH", "UNICORNE_BENCH", "UNICORNE_500", "UNICORNE_OC23", "OOD60", "DOMAINMOTION", "CFOLD"]


def count_ca_atoms(pdb_path: Path) -> int:
    """Count CA atoms in a PDB or CIF file."""
    count = 0
    text = pdb_path.read_text()
    if pdb_path.suffix == ".cif":
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "ATOM" and parts[3] == "CA":
                count += 1
    else:
        for line in text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count


def read_sequences(csv_path: Path) -> dict[str, str]:
    """Read ensemble_ID -> sequence from input_dataset.csv."""
    seqs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = (row.get("ensemble_ID") or row.get("uniprot", "")).strip()
            seq = (row.get("ref_sequence") or row.get("sequence", "")).strip()
            if eid and seq:
                seqs[eid] = seq
    return seqs


def load_gt_tm_matrix(dataset: str, eid: str, results_dir: Path) -> np.ndarray | None:
    """Load GT pairwise TM matrix, searching multiple locations.

    Returns the best available matrix using max-over-norms where possible.
    """
    search_dirs = [
        results_dir / dataset / "gt_matrices_tmalign" / eid,
        results_dir / dataset / "gt" / "matrices_tmalign" / eid,
    ]
    # Also check per-tool dirs
    ds_dir = results_dir / dataset
    if ds_dir.is_dir():
        for tool_dir in sorted(ds_dir.iterdir()):
            if tool_dir.is_dir() and not tool_dir.name.startswith("gt"):
                search_dirs.append(tool_dir / "matrices_tmalign" / eid)

    for d in search_dirs:
        if not d.is_dir():
            continue
        # Prefer combined max file
        for name in ["tm_xx.npy"]:
            f = d / name
            if f.exists():
                return np.load(f)
        # Fallback: max(ab, ba)
        ab = d / "tm_xx_ab.npy"
        ba = d / "tm_xx_ba.npy"
        if ab.exists() and ba.exists():
            return np.maximum(np.load(ab), np.load(ba))
        if ab.exists():
            return np.load(ab)
        # Fallback: max(norm_row, norm_col)
        nr = d / "tm_xx_norm_row.npy"
        nc = d / "tm_xx_norm_col.npy"
        if nr.exists() and nc.exists():
            return np.maximum(np.load(nr), np.load(nc))
    return None


def deduplicate_single_linkage(sim_xx: np.ndarray, threshold: float) -> list[int]:
    """Single-linkage clustering at TM >= threshold, keep one medoid per cluster.

    Single linkage guarantees: after dedup, NO pair of kept conformations
    has TM >= threshold. This is because single linkage merges clusters
    whenever ANY pair between them exceeds the threshold.

    Returns list of kept matrix row indices.
    """
    n = sim_xx.shape[0]
    if n <= 1:
        return list(range(n))

    dist_mat = np.clip(1.0 - sim_xx, 0.0, 1.0)
    np.fill_diagonal(dist_mat, 0.0)
    condensed = squareform(dist_mat, checks=False)

    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # Pick medoid per cluster (most representative = highest avg TM to cluster)
    kept = []
    for cid in sorted(set(labels)):
        members = [i for i in range(n) if labels[i] == cid]
        if len(members) == 1:
            kept.append(members[0])
        else:
            avg_tm = [sim_xx[i, members].mean() for i in members]
            kept.append(members[int(np.argmax(avg_tm))])

    return sorted(kept)


def process_dataset(dataset: str, data_dir: Path, results_dir: Path,
                    length_tol: float, dedup_tm: float,
                    min_gt: int, max_seq_len: int) -> Path:
    """Process one dataset and write filtered_ensembles_v2.csv."""
    ds_input = data_dir / "inputs" / dataset
    gt_pdbs_dir = ds_input / "input_pdbs"
    csv_path = ds_input / "input_dataset.csv"

    if not gt_pdbs_dir.exists():
        print(f"Error: {gt_pdbs_dir} not found", file=sys.stderr)
        sys.exit(1)

    sequences = read_sequences(csv_path)
    ensemble_dirs = sorted(d for d in gt_pdbs_dir.iterdir() if d.is_dir())

    out_path = ds_input / "filtered_ensembles_v2.csv"
    rows_written = 0
    skipped_maxlen = 0
    skipped_mingt_length = 0
    skipped_mingt_dedup = 0
    skipped_no_matrix = 0
    total = len(ensemble_dirs)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ensemble_ID", "kept_indices", "n_gt_raw", "n_gt_filtered",
            "n_gt_dedup", "ref_seq_len", "avg_pairwise_tm", "min_pairwise_tm",
        ])

        for i, edir in enumerate(ensemble_dirs):
            eid = edir.name
            seq = sequences.get(eid, "")
            ref_len = len(seq)

            # Step 0: max sequence length filter
            if ref_len > max_seq_len:
                skipped_maxlen += 1
                continue

            # Enumerate PDB/CIF files and count CA atoms
            pdb_files = sorted(list(edir.glob("*.pdb")) + list(edir.glob("*.cif")))
            ca_counts = [count_ca_atoms(p) for p in pdb_files]
            parseable = [(fi, ca) for fi, ca in enumerate(ca_counts) if ca > 0]
            n_matrix_rows = len(parseable)

            if n_matrix_rows < 1:
                continue

            # Step 1: length tolerance filter (in matrix-row space)
            if ref_len > 0:
                valid_idx = [
                    mi for mi, (fi, ca) in enumerate(parseable)
                    if abs(ca - ref_len) <= length_tol * ref_len
                ]
            else:
                valid_idx = list(range(n_matrix_rows))

            if len(valid_idx) < min_gt:
                skipped_mingt_length += 1
                continue

            # Step 2: TM deduplication (single linkage)
            avg_tm, min_tm = 1.0, 1.0
            if len(valid_idx) <= 1:
                kept_idx = valid_idx
            else:
                sim_xx = load_gt_tm_matrix(dataset, eid, results_dir)
                if sim_xx is not None and sim_xx.shape[0] == n_matrix_rows:
                    sim_sub = sim_xx[np.ix_(valid_idx, valid_idx)]
                    dedup_local = deduplicate_single_linkage(sim_sub, dedup_tm)
                    kept_idx = [valid_idx[j] for j in dedup_local]

                    # Diversity stats on deduped set
                    sim_dedup = sim_sub[np.ix_(dedup_local, dedup_local)]
                    if sim_dedup.shape[0] > 1:
                        mask = ~np.eye(sim_dedup.shape[0], dtype=bool)
                        avg_tm = float(sim_dedup[mask].mean())
                        min_tm = float(sim_dedup[mask].min())
                elif sim_xx is not None:
                    # Row count mismatch — skip dedup, keep length-filtered
                    kept_idx = valid_idx
                else:
                    # No GT matrix available — cannot dedup, skip ensemble
                    skipped_no_matrix += 1
                    continue

            n_dedup = len(kept_idx)
            if n_dedup < min_gt:
                skipped_mingt_dedup += 1
                continue

            # Verify: no pair should have TM >= threshold
            if sim_xx is not None and sim_xx.shape[0] == n_matrix_rows and n_dedup > 1:
                dedup_local = [valid_idx.index(k) for k in kept_idx]
                sim_kept = sim_xx[np.ix_(kept_idx, kept_idx)]
                mask = ~np.eye(n_dedup, dtype=bool)
                max_pair_tm = sim_kept[mask].max()
                if max_pair_tm >= dedup_tm:
                    print(f"  WARNING: {eid} has max pair TM {max_pair_tm:.3f} >= {dedup_tm} after dedup!")

            writer.writerow([
                eid,
                ";".join(str(x) for x in kept_idx),
                n_matrix_rows,
                len(valid_idx),
                n_dedup,
                ref_len,
                f"{avg_tm:.4f}",
                f"{min_tm:.4f}",
            ])
            rows_written += 1

            if (i + 1) % 100 == 0 or i + 1 == total:
                print(f"  [{i+1}/{total}] {rows_written} kept", end="\r", flush=True)

    print(f"\n{dataset}: {rows_written}/{total} ensembles kept")
    print(f"  Skipped: {skipped_maxlen} (seq > {max_seq_len}aa), "
          f"{skipped_mingt_length} (< {min_gt} after length filter), "
          f"{skipped_mingt_dedup} (< {min_gt} after dedup), "
          f"{skipped_no_matrix} (no GT matrix)")
    print(f"  Output: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter GT ensembles: length + TM dedup + min size")
    parser.add_argument("dataset",
                        help="Dataset name (e.g. DANCE) or ALL")
    parser.add_argument("--length-tol", type=float, default=LENGTH_TOLERANCE,
                        help=f"Length tolerance (default: {LENGTH_TOLERANCE})")
    parser.add_argument("--dedup-tm", type=float, default=DEDUP_TM,
                        help=f"TM dedup threshold (default: {DEDUP_TM})")
    parser.add_argument("--min-gt", type=int, default=MIN_GT,
                        help=f"Min GT after dedup (default: {MIN_GT})")
    parser.add_argument("--max-len", type=int, default=MAX_SEQ_LEN,
                        help=f"Max sequence length (default: {MAX_SEQ_LEN})")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "ALL" else [args.dataset]

    print(f"Filtering parameters:")
    print(f"  Length tolerance: {args.length_tol*100:.0f}%")
    print(f"  TM dedup: single linkage at TM >= {args.dedup_tm}")
    print(f"  Min GT: {args.min_gt}")
    print(f"  Max seq length: {args.max_len} aa")
    print()

    for ds in datasets:
        process_dataset(ds, args.data_dir, args.results_dir,
                        args.length_tol, args.dedup_tm,
                        args.min_gt, args.max_len)
        print()


if __name__ == "__main__":
    main()
