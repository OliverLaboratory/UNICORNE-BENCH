#!/usr/bin/env python3
"""
Run BioEMU on a benchmark dataset and write output in benchmark format.

Usage:
  python run_dataset.py <input_dataset_path> <run_id> [--num-samples N] [--output-base PATH]
  python run_dataset.py ... --gpu-shard 0/3  # process 1/3 of ensembles on GPU 0
  python run_dataset.py ... --log-file run.log  # tee output to log file

Output:
  data/outputs/bioemu/<run_id>/output_pdbs/<ensemble_ID>/*.pdb  (all-atom)
  data/outputs/bioemu/<run_id>/output_dataset.csv

Resume: Use --resume to skip ensembles that already have >= num_samples conformations.
To add more samples later, re-run with higher --num-samples; BioEMU will append.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import mdtraj
except ImportError:
    print("mdtraj is required. Install with: pip install mdtraj")
    sys.exit(1)



def report_gpu() -> None:
    """Print GPU utilization and memory via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
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


def load_input_csv(path: Path) -> list[dict]:
    """Load input_dataset.csv, normalizing column names to ensemble_ID, ref_sequence."""
    rows = []
    with open(path, newline="") as f:
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
                rows.append({"ensemble_ID": eid, "ref_sequence": seq})
    return rows


def run_bioemu(
    sequence: str,
    output_dir: Path,
    num_samples: int = 100,
    batch_size_100: int | None = None,
    verbose: bool = True,
    a3m_path: Path | None = None,
) -> bool:
    """Run BioEMU sampling. Returns True on success.

    If ``a3m_path`` is provided, BioEMU reads the MSA directly from that file
    and skips the colabfold MSA search (saves ~5 min of CPU-bound work per
    ensemble).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # BioEMU accepts either a raw sequence string or a path to a .a3m file
    # (in which case it skips colabfold MSA generation).
    sequence_arg = str(a3m_path) if a3m_path is not None else sequence
    cmd = [
        sys.executable,
        "-m",
        "bioemu.sample",
        "--sequence",
        sequence_arg,
        "--num_samples",
        str(num_samples),
        "--output_dir",
        str(output_dir),
    ]
    if batch_size_100 is not None:
        cmd.extend(["--batch_size_100", str(batch_size_100)])
    if verbose:
        batch_info = f" --batch_size_100 {batch_size_100}" if batch_size_100 is not None else ""
        input_info = f"a3m={a3m_path.name}" if a3m_path is not None else f"<{len(sequence)} aa>"
        print(f"  Running: python -m bioemu.sample --sequence {input_info} --num_samples {num_samples}{batch_info} --output_dir {output_dir}")
        print(f"  (BioEMU output below)")
        sys.stdout.flush()
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=not verbose,
    )
    if result.returncode != 0:
        print(f"  BioEMU failed (exit code {result.returncode})", file=sys.stderr)
        if not verbose and result.stderr:
            print(result.stderr, file=sys.stderr)
        return False
    return True


def convert_bioemu_output_to_benchmark(
    bioemu_dir: Path,
    ensemble_id: str,
    sequence: str,
    output_pdbs_dir: Path,
) -> bool:
    """
    Convert BioEMU topology.pdb + samples.xtc to benchmark all-atom PDBs.
    Each frame is saved with all atoms present in BioEMU's output (N, CA, C,
    CB, O, etc.).  Downstream metrics that need only CA read them via
    ``_parse_pdb_ca``.
    Returns True if at least one conformation was written.
    """
    xtc_path = bioemu_dir / "samples.xtc"
    if not xtc_path.exists():
        xtc_path = bioemu_dir / "samples_unphysical.xtc"
    top_path = bioemu_dir / "topology.pdb"

    if not xtc_path.exists() or not top_path.exists():
        return False

    traj = mdtraj.load_xtc(str(xtc_path), top=str(top_path))
    if traj.n_frames == 0:
        return False

    ensemble_out = output_pdbs_dir / ensemble_id
    ensemble_out.mkdir(parents=True, exist_ok=True)

    for i in range(traj.n_frames):
        pdb_path = ensemble_out / f"output_{i:04d}.pdb"
        traj[i].save_pdb(str(pdb_path))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run BioEMU on a benchmark dataset and write benchmark-format output."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input dataset (e.g. data/inputs/DANCE)",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run ID for output (e.g. DANCE, CFOLD, WILL)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per sequence (default: 100)",
    )
    parser.add_argument(
        "--batch-size-100",
        type=int,
        default=500,
        metavar="N",
        help="BioEMU batch_size_100: batch size for 100-aa sequences (scales as (100/L)^2). Default: 500 (optimized for H200).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs",
        help="Base path for outputs (default: data/outputs)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for BioEMU temp outputs (default: output_base/bioemu/<run_id>/_work)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Show BioEMU output and progress (default: True)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress BioEMU subprocess output",
    )
    parser.add_argument(
        "--gpu-shard",
        type=str,
        default=None,
        metavar="I/N",
        help="Process only ensemble index %% N == I (e.g. 0/3, 1/3, 2/3 for 3 GPUs)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append all output to this log file (in addition to stdout)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ensembles that already have >= num_samples conformations in output_pdbs",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        default=None,
        help="Directory with pre-computed MSAs (per-ensemble: <msa_dir>/<eid>/a3m/<eid>.a3m). "
             "When set, BioEMU reads the MSA directly and skips the colabfold MSA search "
             "(~5 min CPU-bound work saved per ensemble).",
    )
    args = parser.parse_args()
    args.verbose = args.verbose and not args.quiet

    # Parse gpu-shard
    gpu_shard_idx, gpu_shard_total = None, None
    if args.gpu_shard:
        parts = args.gpu_shard.split("/")
        if len(parts) != 2:
            print("Error: --gpu-shard must be I/N (e.g. 0/3)", file=sys.stderr)
            sys.exit(1)
        gpu_shard_idx = int(parts[0])
        gpu_shard_total = int(parts[1])

    input_path = args.input_path.resolve()
    # Prefer filtered CSV (only ensembles kept after dedup for the report)
    csv_path = input_path / "input_dataset_filtered.csv"
    if not csv_path.exists():
        csv_path = input_path / "input_dataset.csv"
    if not csv_path.exists():
        print(f"Error: no input CSV found in {input_path}", file=sys.stderr)
        sys.exit(1)

    output_base = args.output_base.resolve()
    run_output = output_base / "bioemu" / args.run_id
    output_pdbs = run_output / "output_pdbs"
    output_csv = run_output / "output_dataset.csv"

    work_dir = args.work_dir or (run_output / "_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_pdbs.mkdir(parents=True, exist_ok=True)

    rows = load_input_csv(csv_path)

    # Set up log file (tee)
    log_file_handle = None
    if args.log_file:
        log_file_handle = open(args.log_file, "a", encoding="utf-8")
        log_file_handle.write(f"\n{'='*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Run started\n{'='*60}\n")
        log_file_handle.flush()

    def log(msg: str = "") -> None:
        print(msg)
        sys.stdout.flush()
        if log_file_handle:
            log_file_handle.write(msg + "\n")
            log_file_handle.flush()

    # Filter by GPU shard
    all_rows = load_input_csv(csv_path)
    if gpu_shard_idx is not None:
        rows = [r for i, r in enumerate(all_rows) if i % gpu_shard_total == gpu_shard_idx]
        log(f"GPU shard {args.gpu_shard}: processing {len(rows)} of {len(all_rows)} ensembles")
    else:
        log(f"Loaded {len(rows)} ensembles from {csv_path}")

    log(f"Output: {run_output}")
    log(f"Work dir: {work_dir} (BioEMU resumes from here if adding more samples)")
    log(f"Num samples: {args.num_samples}, batch_size_100: {args.batch_size_100 or 'default'}")
    log("GPU at start:")
    report_gpu()
    log()

    # Output CSV: per-shard when using gpu-shard (merge at end)
    output_csv = run_output / "output_dataset.csv"
    if gpu_shard_idx is not None:
        output_csv = run_output / f"output_dataset_shard_{gpu_shard_idx}.csv"

    succeeded = []
    total_start = time.perf_counter()
    for i, row in enumerate(rows):
        eid = row["ensemble_ID"]
        seq = row["ref_sequence"]

        # Resume: skip if already complete
        if args.resume:
            ensemble_out = output_pdbs / eid
            if ensemble_out.exists():
                n_pdbs = len(list(ensemble_out.glob("*.pdb")))
                if n_pdbs >= args.num_samples:
                    log(f"[{i + 1}/{len(rows)}] {eid} SKIP (already has {n_pdbs} >= {args.num_samples})")
                    succeeded.append(row)
                    continue

        log(f"[{i + 1}/{len(rows)}] {eid} (seq len={len(seq)})")
        sys.stdout.flush()

        ensemble_work = work_dir / eid
        # Look up pre-computed MSA if msa_dir was provided
        a3m_path: Path | None = None
        if args.msa_dir is not None:
            candidate = args.msa_dir.resolve() / eid / "a3m" / f"{eid}.a3m"
            if candidate.exists():
                a3m_path = candidate
            else:
                log(f"  WARN: no MSA at {candidate}, falling back to sequence-only "
                    f"(colabfold will regenerate)")
        t0 = time.perf_counter()
        if not run_bioemu(
            seq,
            ensemble_work,
            num_samples=args.num_samples,
            batch_size_100=args.batch_size_100,
            verbose=args.verbose,
            a3m_path=a3m_path,
        ):
            log(f"  Skipping {eid} (BioEMU failed)")
            continue

        elapsed = time.perf_counter() - t0
        log(f"  BioEMU time: {elapsed:.1f}s")
        log("  GPU after run:")
        report_gpu()

        if args.verbose:
            log(f"  Converting BioEMU output to benchmark format...")
            sys.stdout.flush()
        if convert_bioemu_output_to_benchmark(
            ensemble_work, eid, seq, output_pdbs
        ):
            succeeded.append(row)
            n_pdbs = len(list((output_pdbs / eid).glob("*.pdb")))
            log(f"  OK: wrote {n_pdbs} conformations to output_pdbs/{eid}/")
        else:
            log(f"  Skipping {eid} (no valid conformations)")
        log()
        sys.stdout.flush()

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble_ID", "ref_sequence"])
        writer.writeheader()
        writer.writerows(succeeded)

    total_elapsed = time.perf_counter() - total_start
    log(f"\nDone. {len(succeeded)}/{len(rows)} ensembles written to {run_output}")
    log(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if succeeded:
        avg_per_ensemble = total_elapsed / len(succeeded)
        log(f"  Avg per ensemble: {avg_per_ensemble:.1f}s")
    log(f"  output_pdbs/")
    log(f"  output_dataset.csv")

    if log_file_handle:
        log_file_handle.close()


if __name__ == "__main__":
    main()
