#!/usr/bin/env python3
"""
Run AlphaFlow (ESMFlow) on a benchmark dataset and write output in benchmark format.

Models: md_distilled (ESMFlow-MD), pdb_distilled (ESMFlow-PDB), af_pdb_distilled (AlphaFlow-PDB).
ESMFlow models use esmfold mode (no MSA). AlphaFlow-PDB uses alphafold mode (requires MSAs).

Usage:
  python run_dataset.py <input_dataset_path> <run_id> [--num-samples N]
  python run_dataset.py ... --model pdb_distilled --resume

Output (esmfold mode):       data/outputs/esmflow_pdb/<run_id>/output_pdbs/<eid>/*.pdb
Output (alphafold mode):      data/outputs/alphaflow_pdb/<run_id>/output_pdbs/<eid>/*.pdb
(both all-atom; output_dataset.csv alongside)

Resume: Use --resume to skip ensembles that already have >= num_samples conformations.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))



def load_input_csv(path: Path) -> list[dict]:
    """Load input_dataset.csv, normalizing column names."""
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


def split_multimodel_pdb(pdb_path: Path) -> list[str]:
    """
    Split a multi-MODEL PDB into a list of all-atom PDB strings, one per MODEL.
    Preserves every ATOM/HETATM/TER line within each MODEL block.
    """
    text = pdb_path.read_text()
    models = re.split(r"MODEL\s+\d+\s*\n", text)[1:]  # first split is pre-MODEL header
    out = []
    for block in models:
        endmdl = block.find("ENDMDL")
        if endmdl >= 0:
            block = block[:endmdl]
        # Keep only structure records (drop stray blank lines / non-atom noise)
        lines = [
            line for line in block.splitlines()
            if line.startswith(("ATOM", "HETATM", "TER"))
        ]
        if not lines:
            continue
        out.append("\n".join(lines) + "\nEND\n")
    return out


def run_alphaflow(
    input_csv: Path,
    output_dir: Path,
    num_samples: int,
    weights_path: Path,
    alphaflow_dir: Path,
    verbose: bool = True,
    tmpdir_base: Path | None = None,
    mode: str = "esmfold",
    msa_dir: Path | None = None,
) -> bool:
    """Run AlphaFlow predict.py. Returns True on success."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"  # Use venv torch, not ~/.local (avoids cuDNN mismatch)
    if tmpdir_base is not None:
        env["TMPDIR"] = str(tmpdir_base)
    # Use venv python explicitly (has tqdm, torch); nohup may not pass ALPHAFLOW_PYTHON
    venv_python = Path.home() / "alphaflow_venv" / "bin" / "python"
    python_exe = (
        os.environ.get("ALPHAFLOW_PYTHON")
        or (str(venv_python) if venv_python.is_file() else None)
        or sys.executable
    )
    cmd = [
        python_exe,
        str(alphaflow_dir / "predict.py"),
        "--mode", mode,
        "--input_csv", str(input_csv),
        "--weights", str(weights_path),
        "--samples", str(num_samples),
        "--outpdb", str(output_dir),
        "--noisy_first",
        "--no_diffusion",
    ]
    if mode == "alphafold":
        if msa_dir is None:
            raise ValueError("--msa-dir is required for alphafold mode")
        cmd.extend(["--msa_dir", str(msa_dir), "--self_cond", "--resample"])
    if verbose:
        print(f"  Running: {python_exe} predict.py --mode {mode} --samples {num_samples} --outpdb {output_dir}")
        sys.stdout.flush()
    result = subprocess.run(
        cmd,
        cwd=str(alphaflow_dir),
        env=env,
        capture_output=not verbose,
        text=not verbose,
    )
    if result.returncode != 0:
        print(f"  AlphaFlow failed (exit code {result.returncode})", file=sys.stderr)
        if not verbose and result.stderr:
            print(result.stderr, file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run AlphaFlow (ESMFlow) on a benchmark dataset."
    )
    parser.add_argument("input_path", type=Path, help="Path to input dataset")
    parser.add_argument("run_id", type=str, help="Run ID (e.g. DANCE, CFOLD)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per sequence (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N ensembles (for testing)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs",
        help="Base path for outputs",
    )
    parser.add_argument(
        "--alphaflow-dir",
        type=Path,
        default=Path.home() / "alphaflow",
        help="AlphaFlow repo directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["md_distilled", "pdb_distilled", "af_pdb_distilled"],
        default="md_distilled",
        help="Model: md_distilled (ESMFlow-MD), pdb_distilled (ESMFlow-PDB), af_pdb_distilled (AlphaFlow-PDB)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weights (overrides --model)",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        default=None,
        help="MSA directory for alphafold mode (required for af_pdb_distilled). "
             "Structure: {msa_dir}/{name}/a3m/{name}.a3m",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ensembles that already have >= num_samples conformations in output_pdbs",
    )
    parser.add_argument(
        "--gpu-shard",
        type=str,
        default=None,
        metavar="I/N",
        help="Process shard I of N (0-indexed). E.g. --gpu-shard 0/3 for first of 3 shards.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append all output to this log file (in addition to stdout)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    # Prefer filtered CSV (only ensembles kept after dedup for the report)
    csv_path = input_path / "input_dataset_filtered.csv"
    if not csv_path.exists():
        csv_path = input_path / "input_dataset.csv"
    if not csv_path.exists():
        print(f"Error: no input CSV found in {input_path}", file=sys.stderr)
        sys.exit(1)

    alphaflow_dir = args.alphaflow_dir.resolve()
    if not (alphaflow_dir / "predict.py").exists():
        print(f"Error: AlphaFlow not found at {alphaflow_dir}", file=sys.stderr)
        sys.exit(1)

    WEIGHTS_BY_MODEL = {
        "md_distilled": "esmflow_md_distilled.pt",
        "pdb_distilled": "esmflow_pdb_distilled.pt",
        "af_pdb_distilled": "alphaflow_pdb_distilled_202402.pt",
    }
    MODEL_MODE = {
        "md_distilled": "esmfold",
        "pdb_distilled": "esmfold",
        "af_pdb_distilled": "alphafold",
    }
    af_mode = MODEL_MODE[args.model]
    weights_path = args.weights or (
        alphaflow_dir / "weights" / WEIGHTS_BY_MODEL[args.model]
    )
    msa_dir = args.msa_dir
    if af_mode == "alphafold" and msa_dir is None:
        print("Error: --msa-dir is required for af_pdb_distilled model", file=sys.stderr)
        sys.exit(1)
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}", file=sys.stderr)
        print(f"  Run: curl -L -o {weights_path} {WEIGHTS_URL[args.model]}", file=sys.stderr)
        sys.exit(1)

    output_base = args.output_base.resolve()
    tool_name = "alphaflow_pdb" if af_mode == "alphafold" else "esmflow_pdb"
    run_output = output_base / tool_name / args.run_id
    output_pdbs = run_output / "output_pdbs"
    output_csv = run_output / "output_dataset.csv"
    work_base = run_output / "_work"
    output_pdbs.mkdir(parents=True, exist_ok=True)
    work_base.mkdir(parents=True, exist_ok=True)

    # Set up log file (tee)
    log_file_handle = None
    if args.log_file:
        log_file_handle = open(args.log_file, "a", encoding="utf-8")
        log_file_handle.write(
            f"\n{'='*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Run started\n{'='*60}\n"
        )
        log_file_handle.flush()

    def log(msg: str = "") -> None:
        print(msg)
        sys.stdout.flush()
        if log_file_handle:
            log_file_handle.write(msg + "\n")
            log_file_handle.flush()

    rows = load_input_csv(csv_path)
    if args.gpu_shard is not None:
        shard_idx, n_shards = map(int, args.gpu_shard.split("/"))
        rows = [r for i, r in enumerate(rows) if i % n_shards == shard_idx]
        log(f"GPU shard {shard_idx}/{n_shards}: {len(rows)} ensembles")
    if args.limit is not None:
        rows = rows[: args.limit]
        log(f"Limited to first {len(rows)} ensembles (--limit {args.limit})")
    log(f"Loaded {len(rows)} ensembles from {csv_path}")
    log(f"Output: {run_output}")
    log(f"AlphaFlow: {alphaflow_dir}, weights: {weights_path}, resume: {args.resume}")
    log()

    succeeded = []
    total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=str(work_base)) as tmpdir:
        tmp = Path(tmpdir)
        for i, row in enumerate(rows):
            eid = row["ensemble_ID"]
            seq = row["ref_sequence"]

            # Count existing samples for incremental mode
            existing_count = 0
            ensemble_out = output_pdbs / eid
            if ensemble_out.exists():
                existing_count = len(list(ensemble_out.glob("output_*.pdb")))

            if args.resume and existing_count >= args.num_samples:
                log(f"[{i + 1}/{len(rows)}] {eid} SKIP (already has {existing_count} >= {args.num_samples})")
                succeeded.append(row)
                continue

            deficit = args.num_samples - existing_count
            if deficit <= 0:
                succeeded.append(row)
                continue

            log(f"[{i + 1}/{len(rows)}] {eid} (seq len={len(seq)}, existing={existing_count}, generating={deficit})")
            sys.stdout.flush()

            # AlphaFlow CSV: name, seqres
            alphaflow_csv = tmp / "input.csv"
            with open(alphaflow_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "seqres"])
                w.writeheader()
                w.writerow({"name": eid, "seqres": seq})

            work_dir = tmp / eid
            work_dir.mkdir()
            t0 = time.perf_counter()
            if not run_alphaflow(
                alphaflow_csv,
                work_dir,
                num_samples=deficit,
                weights_path=weights_path,
                alphaflow_dir=alphaflow_dir,
                verbose=not args.quiet,
                tmpdir_base=work_base,
                mode=af_mode,
                msa_dir=msa_dir,
            ):
                log(f"  Skipping {eid} (AlphaFlow failed)")
                continue

            elapsed = time.perf_counter() - t0
            log(f"  AlphaFlow time: {elapsed:.1f}s")

            # AlphaFlow writes {outpdb}/{name}.pdb (multi-MODEL)
            pdb_file = work_dir / f"{eid}.pdb"
            if not pdb_file.exists():
                log(f"  Skipping {eid} (no output PDB)")
                continue

            model_pdbs = split_multimodel_pdb(pdb_file)
            if not model_pdbs:
                log(f"  Skipping {eid} (no MODELs in output)")
                continue

            ensemble_out.mkdir(parents=True, exist_ok=True)
            for j, pdb_text in enumerate(model_pdbs):
                # Number new files starting after existing ones
                pdb_path = ensemble_out / f"output_{existing_count + j:04d}.pdb"
                pdb_path.write_text(pdb_text)

            succeeded.append(row)
            log(f"  OK: wrote {len(model_pdbs)} new conformations to output_pdbs/{eid}/ (total now {existing_count + len(model_pdbs)})")
            log()

            # Free disk: remove intermediate PDB from work_dir (on NFS but avoids accumulation)
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass

    if log_file_handle:
        log_file_handle.close()

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble_ID", "ref_sequence"])
        writer.writeheader()
        writer.writerows(succeeded)

    total_elapsed = time.perf_counter() - total_start
    log(f"\nDone. {len(succeeded)}/{len(rows)} ensembles written to {run_output}")
    log(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if succeeded:
        log(f"  Avg per ensemble: {total_elapsed/len(succeeded):.1f}s")


if __name__ == "__main__":
    main()
