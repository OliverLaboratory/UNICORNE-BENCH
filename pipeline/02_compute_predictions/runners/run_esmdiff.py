#!/usr/bin/env python3
"""
Run ESMDiff on a benchmark dataset and write output in benchmark format.

Usage:
  python run_dataset.py <input_dataset_path> <run_id> [--num-samples N]
  python run_dataset.py data/inputs/DANCE DANCE --num-samples 250
  python run_dataset.py data/inputs/DANCE DANCE --gpu-shard 0/5 --resume

Output:
  data/outputs/esmdiff/<run_id>/output_pdbs/<ensemble_ID>/output_NNNN.pdb
  data/outputs/esmdiff/<run_id>/output_dataset.csv

The script:
  1. Prepares a flat input directory with one PDB per target
  2. Calls ESMDiff sample_esmdiff.py (model loads once, processes all targets)
  3. Splits multi-model output PDBs into individual benchmark-format files
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Shared helpers (02_compute_predictions/prediction_helpers.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prediction_helpers import load_input_csv, report_gpu
ESMDIFF_ROOT = Path.home() / "esmdiff"
ESMDIFF_CKPT = ESMDIFF_ROOT / "data" / "ckpt" / "release_v0.pt"

DEFAULT_NUM_SAMPLES = 250
DEFAULT_NUM_STEPS = 25
# Controls GPU memory: batch_size = max_batch_tokens // (seq_len^2)
# 400k works for 32GB GPUs (RTX 5090): ~16 samples/batch for 154-aa, ~4 for 300-aa
DEFAULT_MAX_BATCH_TOKENS = 400_000


def find_input_pdb(input_pdbs_dir: Path, ensemble_id: str) -> Path | None:
    """Find one representative PDB/CIF for an ensemble."""
    ens_dir = input_pdbs_dir / ensemble_id
    if not ens_dir.is_dir():
        return None
    pdbs = sorted(ens_dir.glob("*.pdb"))
    if pdbs:
        return pdbs[0]
    cifs = sorted(ens_dir.glob("*.cif"))
    if cifs:
        return cifs[0]
    return None


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """Convert CIF to PDB using simple ATOM record extraction."""
    try:
        atoms = []
        with open(cif_path) as f:
            in_atom_site = False
            col_names = []
            for line in f:
                line = line.strip()
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    col_names.append(line.split(".")[1])
                    continue
                if in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                    parts = line.split()
                    if len(parts) >= len(col_names):
                        record = dict(zip(col_names, parts))
                        atoms.append(record)
                elif in_atom_site and not line.startswith("#") and not line.startswith("_") and atoms:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        break

        if not atoms:
            return False

        with open(pdb_path, "w") as f:
            for i, a in enumerate(atoms):
                group = a.get("group_PDB", "ATOM")
                serial = i + 1
                name = a.get("auth_atom_id", a.get("label_atom_id", "CA"))
                resname = a.get("auth_comp_id", a.get("label_comp_id", "ALA"))
                chain = a.get("auth_asym_id", a.get("label_asym_id", "A"))
                resseq = a.get("auth_seq_id", a.get("label_seq_id", "1"))
                x = float(a.get("Cartn_x", 0))
                y = float(a.get("Cartn_y", 0))
                z = float(a.get("Cartn_z", 0))
                element = a.get("type_symbol", name[0])
                atom_name = f" {name:<3s}" if len(name) < 4 else name
                f.write(
                    f"{group:<6s}{serial:>5d} {atom_name:4s} {resname:>3s} {chain:1s}"
                    f"{int(resseq):>4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
            f.write("END\n")
        return True
    except Exception as e:
        print(f"  Warning: CIF conversion failed for {cif_path}: {e}")
        return False


def prepare_input_dir(
    rows: list[dict],
    input_pdbs_dir: Path,
    flat_dir: Path,
    log,
) -> list[str]:
    """Create flat input directory with one PDB per target. Returns list of prepared ensemble IDs."""
    flat_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    for row in rows:
        eid = row["ensemble_ID"]
        out_pdb = flat_dir / f"{eid}.pdb"

        if out_pdb.exists():
            prepared.append(eid)
            continue

        src = find_input_pdb(input_pdbs_dir, eid)
        if src is None:
            log(f"  Warning: no input PDB for {eid}, skipping")
            continue

        if src.suffix == ".cif":
            if cif_to_pdb(src, out_pdb):
                prepared.append(eid)
            else:
                log(f"  Warning: failed to convert CIF for {eid}")
        else:
            shutil.copy2(src, out_pdb)
            prepared.append(eid)

    return prepared


def split_multimodel_pdb(merged_pdb: Path, output_dir: Path,
                         start_index: int = 0, max_models: int | None = None) -> int:
    """Split a multi-model PDB into individual output_NNNN.pdb files.

    Args:
        start_index: File numbering offset (for appending to existing samples).
        max_models: Only write this many models (None = all).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models = []
    current_model = []
    in_model = False

    with open(merged_pdb) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_model = []
            elif line.startswith("ENDMDL"):
                if current_model:
                    models.append(current_model)
                in_model = False
                current_model = []
            elif in_model:
                current_model.append(line)
            elif not models:
                current_model.append(line)

    if not models and current_model:
        models = [current_model]

    # Only keep up to max_models
    if max_models is not None:
        models = models[:max_models]

    for i, model_lines in enumerate(models):
        pdb_path = output_dir / f"output_{start_index + i:04d}.pdb"
        with open(pdb_path, "w") as f:
            f.writelines(model_lines)
            if not any(l.startswith("END") for l in model_lines):
                f.write("END\n")

    return len(models)


def find_esmdiff_output_pdbs(output_base: Path) -> dict[str, Path]:
    """Find all ESMDiff output PDB files across timestamped directories.

    ESMDiff creates a new timestamped directory per target, so we need to
    scan all subdirectories. Returns {ensemble_id: merged_pdb_path}.
    """
    result = {}
    if not output_base.exists():
        return result
    # Check all timestamped subdirs (T*/ or step*/), sorted so newest wins
    subdirs = sorted(
        (d for d in output_base.iterdir()
         if d.is_dir() and (d.name.startswith("T") or d.name.startswith("step"))),
        key=lambda d: d.name,
    )
    for subdir in subdirs:
        for pdb in subdir.glob("*.pdb"):
            eid = pdb.stem
            result[eid] = pdb  # newer dir overwrites older
    # Also check for PDBs directly in output_base
    for pdb in output_base.glob("*.pdb"):
        eid = pdb.stem
        if eid not in result:
            result[eid] = pdb
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run ESMDiff on a benchmark dataset and write benchmark-format output."
    )
    parser.add_argument("input_path", type=Path, help="Path to input dataset (e.g. data/inputs/DANCE)")
    parser.add_argument("run_id", type=str, help="Run ID for output (e.g. DANCE, CFOLD, WILL_TM)")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS,
                        help="GPU memory control: max seq_len^2 * batch_size (default: 400000 for 32GB GPUs)")
    parser.add_argument("--output-base", type=Path, default=PROJECT_ROOT / "data" / "outputs")
    parser.add_argument("--gpu-shard", type=str, default=None, metavar="I/N",
                        help="Process only ensemble index %% N == I")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ensembles that already have >= num_samples output PDBs")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    # Parse gpu-shard
    gpu_shard_idx, gpu_shard_total = None, None
    if args.gpu_shard:
        parts = args.gpu_shard.split("/")
        gpu_shard_idx, gpu_shard_total = int(parts[0]), int(parts[1])

    input_path = args.input_path.resolve()
    # Prefer filtered CSV (only ensembles kept after dedup for the report)
    csv_path = input_path / "input_dataset_filtered.csv"
    if not csv_path.exists():
        csv_path = input_path / "input_dataset.csv"
    input_pdbs_dir = input_path / "input_pdbs"
    for p in [csv_path, input_pdbs_dir]:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    run_output = args.output_base.resolve() / "esmdiff" / args.run_id
    output_pdbs = run_output / "output_pdbs"
    # Each shard gets its own work directory to avoid race conditions
    shard_suffix = f"_shard{gpu_shard_idx}" if gpu_shard_idx is not None else ""
    work_dir = run_output / f"_work{shard_suffix}"
    esmdiff_input_dir = work_dir / "esmdiff_input"
    esmdiff_output_dir = work_dir / "esmdiff_output"

    output_pdbs.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_handle = None
    if args.log_file:
        log_handle = open(args.log_file, "a", encoding="utf-8")
        log_handle.write(f"\n{'='*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ESMDiff run started\n{'='*60}\n")

    def log(msg: str = "") -> None:
        print(msg)
        sys.stdout.flush()
        if log_handle:
            log_handle.write(msg + "\n")
            log_handle.flush()

    # Load and shard
    all_rows = load_input_csv(csv_path)
    if gpu_shard_idx is not None:
        rows = [r for i, r in enumerate(all_rows) if i % gpu_shard_total == gpu_shard_idx]
        log(f"GPU shard {args.gpu_shard}: processing {len(rows)} of {len(all_rows)} ensembles")
    else:
        rows = all_rows
        log(f"Loaded {len(rows)} ensembles from {csv_path}")

    # Count existing samples per ensemble and compute deficits
    existing_counts: dict[str, int] = {}
    for r in rows:
        eid = r["ensemble_ID"]
        ens_dir = output_pdbs / eid
        existing_counts[eid] = len(list(ens_dir.glob("*.pdb"))) if ens_dir.exists() else 0

    # Filter already-completed targets
    if args.resume:
        original = len(rows)
        rows = [r for r in rows if existing_counts[r["ensemble_ID"]] < args.num_samples]
        log(f"Resume: {original - len(rows)} already done, {len(rows)} remaining")

    if not rows:
        log("Nothing to do!")
        return

    # Compute how many new samples to generate (max deficit across remaining ensembles)
    deficits = {r["ensemble_ID"]: args.num_samples - existing_counts[r["ensemble_ID"]] for r in rows}
    max_deficit = max(deficits.values())
    log(f"Output: {run_output}")
    log(f"Target: {args.num_samples}, Generating: {max_deficit} new samples (max deficit)")
    log(f"Steps: {args.num_steps}, Max batch tokens: {args.max_batch_tokens}")
    log("GPU at start:")
    report_gpu()
    log()

    # Check ESMDiff
    if not ESMDIFF_ROOT.exists() or not ESMDIFF_CKPT.exists():
        log(f"Error: ESMDiff not found at {ESMDIFF_ROOT} or checkpoint missing")
        sys.exit(1)

    # Step 1: Prepare flat input directory
    log("Preparing input PDBs...")
    prepared_eids = prepare_input_dir(rows, input_pdbs_dir, esmdiff_input_dir, log)
    log(f"  {len(prepared_eids)} input PDBs ready in {esmdiff_input_dir}")

    if not prepared_eids:
        log("No valid inputs, exiting")
        return

    # Step 2: Run ESMDiff
    log("\nRunning ESMDiff...")
    t0 = time.perf_counter()

    cmd = [
        "pixi", "run", "python", "slm/sample_esmdiff.py",
        "--input", str(esmdiff_input_dir),
        "--output", str(esmdiff_output_dir),
        "--num_samples", str(max_deficit),
        "--num_steps", str(args.num_steps),
        "--ckpt", str(ESMDIFF_CKPT),
        "--max_batch_tokens", str(args.max_batch_tokens),
    ]
    log(f"  {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(cmd, cwd=str(ESMDIFF_ROOT), env=env)

    elapsed = time.perf_counter() - t0
    log(f"\nESMDiff finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        log(f"  WARNING: ESMDiff exited with code {result.returncode}")
        log("  Proceeding to convert any completed outputs...")

    # Step 3: Find all output PDBs and convert
    output_map = find_esmdiff_output_pdbs(esmdiff_output_dir)
    log(f"\nConverting outputs ({len(output_map)} PDBs found across timestamped dirs)...")
    succeeded = []
    row_map = {r["ensemble_ID"]: r for r in rows}

    for eid in prepared_eids:
        merged_pdb = output_map.get(eid)
        if merged_pdb is None:
            log(f"  {eid}: no output found")
            continue

        ens_out = output_pdbs / eid
        existing = existing_counts.get(eid, 0)
        needed = deficits.get(eid, args.num_samples)
        n_models = split_multimodel_pdb(merged_pdb, ens_out, start_index=existing, max_models=needed)

        if n_models > 0:
            succeeded.append(row_map[eid])
            log(f"  {eid}: {n_models} new conformations (total now {existing + n_models})")
        else:
            log(f"  {eid}: no valid models")

    # Step 4: Write output CSV
    output_csv = run_output / "output_dataset.csv"
    if gpu_shard_idx is not None:
        output_csv = run_output / f"output_dataset_shard_{gpu_shard_idx}.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble_ID", "ref_sequence"])
        writer.writeheader()
        writer.writerows(succeeded)

    total = time.perf_counter() - t0
    log(f"\nDone. {len(succeeded)}/{len(prepared_eids)} ensembles completed")
    log(f"  Total time: {total:.1f}s ({total / 60:.1f} min)")
    if succeeded:
        log(f"  Avg per ensemble: {total / len(succeeded):.1f}s")
    log(f"  Output CSV: {output_csv}")

    if log_handle:
        log_handle.close()


if __name__ == "__main__":
    main()
