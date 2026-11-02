#!/usr/bin/env python3
"""
Run CF-random on a benchmark dataset using the direct ColabFold API.

Loads AF2 models ONCE at startup and reuses them across all ensembles/rounds.
This avoids the ~30-60s cold-start overhead of calling the ``colabfold_batch``
CLI for every random seed (~40 calls per ensemble -> ~15-30 min saved).

CF-random strategy (same as original):
  1. Full-MSA runs with varying random seeds (5 seeds x 5 models = 25 structures)
  2. Variable-depth MSA runs at max_seq=1,2,4,8,16,32,64 (1 seed x 5 models = 5 each)

Requires the ColabFold Python environment (not just the CLI). Typically
launched with ``~/localcolabfold/.pixi/envs/default/bin/python``.

Usage:
  python run_cfrandom.py <input_path> <run_id> --msa-dir <path>
  python run_cfrandom.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH \\
      --msa-dir data/msas/UNICORNE --num-samples 250 --resume --gpu-shard 0/2

Output:
  data/outputs/cfrandom/<run_id>/output_pdbs/<ensemble_ID>/output_NNNN.pdb
  (full-atom PDBs; downstream metrics extract CA via _parse_pdb_ca)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- ColabFold env setup (must precede jax/tf imports) -----------------------
os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "4.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
try:
    from _runtime_log import log_ensemble
except Exception:
    def log_ensemble(*a, **kw): pass


logger = logging.getLogger("cfrandom")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# MSA helpers
# ---------------------------------------------------------------------------

def load_a3m(msa_dir: Path, eid: str) -> str | None:
    """Read an A3M file from the MSA directory.  Returns raw text or None."""
    for candidate in [
        msa_dir / eid / "a3m" / f"{eid}.a3m",
        msa_dir / f"{eid}-search" / "0.a3m",
        msa_dir / eid / "0.a3m",
    ]:
        if candidate.exists():
            return candidate.read_text()
    return None


def truncate_a3m(a3m_text: str, max_sequences: int) -> str:
    """Keep only the first *max_sequences* entries in an A3M string."""
    lines = a3m_text.splitlines(keepends=True)
    out, count = [], 0
    for line in lines:
        if line.startswith(">"):
            count += 1
            if count > max_sequences:
                break
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(sequence: str, a3m_text: str):
    """Build a monomer feature dict from a sequence + A3M text."""
    from colabfold.batch import build_monomer_feature, mk_mock_template
    # Strip comment lines (e.g. "#153\t1") that ColabFold search prepends
    lines = a3m_text.splitlines(keepends=True)
    a3m_clean = "".join(l for l in lines if not l.startswith("#"))
    template_features = mk_mock_template(sequence)
    feature_dict = build_monomer_feature(sequence, a3m_clean, template_features)
    feature_dict["asym_id"] = np.zeros(feature_dict["aatype"].shape[0], dtype=int)
    return feature_dict


# ---------------------------------------------------------------------------
# Prediction wrapper
# ---------------------------------------------------------------------------

def predict_and_extract(
    prefix: str,
    work_dir: Path,
    feature_dict: dict,
    seq_len: int,
    pad_len: int,
    model_type: str,
    model_runner_and_params,
    random_seed: int,
    num_seeds: int,
) -> list[str]:
    """Run predict_structure and return list of unrelaxed PDB strings."""
    from colabfold.batch import predict_structure

    work_dir.mkdir(parents=True, exist_ok=True)
    results = predict_structure(
        prefix=prefix,
        result_dir=work_dir,
        feature_dict=feature_dict,
        is_complex=False,
        use_templates=False,
        sequences_lengths=[seq_len],
        pad_len=pad_len,
        model_type=model_type,
        model_runner_and_params=model_runner_and_params,
        random_seed=random_seed,
        num_seeds=num_seeds,
        num_relax=0,
        rank_by="plddt",
        stop_at_score=100,
        save_all=False,
    )
    # Collect unrelaxed PDB files written by predict_structure
    pdb_files = [
        f for f in results.get("result_files", [])
        if str(f).endswith(".pdb") and "unrelaxed" in str(f)
    ]
    pdb_texts = [f.read_text() for f in pdb_files]
    # Clean up colabfold output files (scores, etc.) to save disk
    for f in results.get("result_files", []):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    return pdb_texts


def pdb_text_has_ca(pdb_text: str) -> bool:
    """Quick sanity check: ensure the PDB string contains at least one CA atom.
    Used to skip empty/malformed predictions (parsing errors in colabfold)."""
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            return True
    return False


# ---------------------------------------------------------------------------
# Per-ensemble sampling
# ---------------------------------------------------------------------------

def sample_ensemble(
    eid: str,
    sequence: str,
    a3m_text: str,
    ensemble_out: Path,
    work_dir: Path,
    num_samples: int,
    existing_count: int,
    model_runner_and_params,
    pad_len: int,
    model_type: str,
) -> int:
    """Generate samples for one ensemble using CF-random strategy.

    1. Full-MSA run (5 seeds x 5 models = 25 structures)
    2. Variable-depth runs at max_seq=1,2,4,8,16,32,64 (1 seed x 5 models = 5 each)
    Loops until ``num_samples`` structures total exist.
    """
    deficit = num_samples - existing_count
    if deficit <= 0:
        return 0

    ensemble_out.mkdir(parents=True, exist_ok=True)
    total_new = 0
    round_idx = 0
    rng = np.random.default_rng(42)
    seq_len = len(sequence)

    while total_new < deficit:
        seed = int(rng.integers(0, 10000))

        # --- Full MSA run (5 seeds x 5 models = 25 structures) ---
        feature_dict = build_features(sequence, a3m_text)
        pdb_texts = predict_and_extract(
            prefix=f"{eid}_full_{seed}",
            work_dir=work_dir / f"{eid}_full_{seed}",
            feature_dict=feature_dict,
            seq_len=seq_len,
            pad_len=pad_len,
            model_type=model_type,
            model_runner_and_params=model_runner_and_params,
            random_seed=seed,
            num_seeds=5,
        )
        for pdb_text in pdb_texts:
            if pdb_text_has_ca(pdb_text):
                out_path = ensemble_out / f"output_{existing_count + total_new:04d}.pdb"
                out_path.write_text(pdb_text)
                total_new += 1
        logger.info(
            f"    Full-MSA seed={seed}: {len(pdb_texts)} structures "
            f"(total: {existing_count + total_new})"
        )
        del feature_dict

        if total_new >= deficit:
            break

        # --- Variable-depth MSA runs (7 depths x 1 seed x 5 models) ---
        max_seq = 1
        for _ in range(7):
            if total_new >= deficit:
                break
            truncated = truncate_a3m(a3m_text, max_seq + 1)  # +1 for query
            feature_dict = build_features(sequence, truncated)
            pdb_texts = predict_and_extract(
                prefix=f"{eid}_d{max_seq}_{seed}",
                work_dir=work_dir / f"{eid}_d{max_seq}_{seed}",
                feature_dict=feature_dict,
                seq_len=seq_len,
                pad_len=pad_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                random_seed=seed,
                num_seeds=1,
            )
            for pdb_text in pdb_texts:
                if pdb_text_has_ca(pdb_text):
                    out_path = ensemble_out / f"output_{existing_count + total_new:04d}.pdb"
                    out_path.write_text(pdb_text)
                    total_new += 1
            del feature_dict
            max_seq *= 2

        round_idx += 1
        logger.info(f"    Round {round_idx}: {total_new}/{deficit} new samples")

    return total_new


# ---------------------------------------------------------------------------
# CLI & main loop
# ---------------------------------------------------------------------------

def load_input_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eid = (
                row.get("ensemble_ID") or row.get("ensemble_id")
                or row.get("uniprot", "")
            ).strip()
            seq = (
                row.get("ref_sequence") or row.get("sequence", "")
            ).strip()
            if eid and seq:
                rows.append({"ensemble_ID": eid, "ref_sequence": seq})
    return rows


def count_existing(d: Path) -> int:
    return len(list(d.glob("output_*.pdb"))) if d.exists() else 0


def main():
    parser = argparse.ArgumentParser(
        description="Run CF-random with direct ColabFold API (AF2 loaded once).")
    parser.add_argument("input_path", type=Path,
                        help="Input dataset path (contains input_dataset.csv)")
    parser.add_argument("run_id", help="Run ID for output naming")
    parser.add_argument("--msa-dir", type=Path, required=True,
                        help="MSA directory (contains per-ensemble MSA folders)")
    parser.add_argument("--num-samples", type=int, default=250,
                        help="Target number of samples per ensemble (default 250)")
    parser.add_argument("--output-base", type=Path,
                        default=PROJECT_ROOT / "data" / "outputs",
                        help="Base path for outputs")
    parser.add_argument("--model-type", default="alphafold2_ptm")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ensembles that already have >= num_samples conformations")
    parser.add_argument("--gpu-shard", default=None, metavar="I/N",
                        help="Process shard I of N (0-indexed)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N ensembles (for testing)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Append all output to this log file")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="ColabFold params dir (default: ~/.cache/colabfold)")
    args = parser.parse_args()

    # --- Locate input CSV ---
    csv_candidates = [
        args.input_path / "input_dataset_filtered.csv",
        args.input_path / "input_dataset.csv",
    ]
    input_csv = next((c for c in csv_candidates if c.exists()), None)
    if input_csv is None:
        print(f"Error: no input CSV in {args.input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_input_csv(input_csv)
    print(f"Loaded {len(rows)} ensembles from {input_csv}")

    if args.gpu_shard:
        i, n = map(int, args.gpu_shard.split("/"))
        rows = [r for idx, r in enumerate(rows) if idx % n == i]
        print(f"GPU shard {i}/{n}: processing {len(rows)} ensembles")

    if args.limit:
        rows = rows[: args.limit]

    # Sort by sequence length so pad_len grows gradually (fewer recompiles)
    rows.sort(key=lambda r: len(r["ref_sequence"]))

    # --- Output dirs ---
    run_output = args.output_base / "cfrandom" / args.run_id
    pdbs_dir = run_output / "output_pdbs"
    work_dir = run_output / "_work_v2"
    run_output.mkdir(parents=True, exist_ok=True)
    pdbs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Log file ---
    log_fh = None
    if args.log_file:
        log_fh = open(args.log_file, "a")

    def log(msg=""):
        print(msg)
        sys.stdout.flush()
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    # --- Load AF2 models ONCE ---
    log("Loading AF2 models (one-time)...")
    t_load = time.perf_counter()

    # Disable TF GPU to avoid conflict with JAX
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from colabfold.alphafold.models import load_models_and_params
    from colabfold.download import default_data_dir

    data_dir = args.data_dir or Path(default_data_dir)
    model_runner_and_params = load_models_and_params(
        num_models=5,
        use_templates=False,
        num_recycles=None,
        num_ensemble=1,
        model_order=[1, 2, 3, 4, 5],
        model_type=args.model_type,
        data_dir=data_dir,
        stop_at_score=100,
        rank_by="plddt",
        max_seq=512,
        max_extra_seq=5120,
        use_cluster_profile=True,
        use_fuse=True,
        use_bfloat16=True,
        use_dropout=False,
    )
    log(f"Models loaded in {time.perf_counter() - t_load:.1f}s")

    # --- Compute max seq len for initial pad_len ---
    max_len = max(len(r["ref_sequence"]) for r in rows) if rows else 0
    pad_len = 0  # will grow on first query

    # --- Main loop ---
    total_start = time.perf_counter()
    succeeded = []

    for i, row in enumerate(rows):
        eid = row["ensemble_ID"]
        seq = row["ref_sequence"]
        seq_len = len(seq)

        ensemble_out = pdbs_dir / eid
        existing = count_existing(ensemble_out)

        if args.resume and existing >= args.num_samples:
            log(f"  [{i+1}/{len(rows)}] {eid}: skip (already {existing} samples)")
            log_ensemble(tool="cfrandom", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=0.0,
                         status="skip", note=f"already {existing} samples")
            succeeded.append(row)
            continue

        a3m_text = load_a3m(args.msa_dir, eid)
        if a3m_text is None:
            log(f"  [{i+1}/{len(rows)}] {eid}: SKIP (no MSA)")
            log_ensemble(tool="cfrandom", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=0.0,
                         status="skip", note="no MSA")
            continue

        # Grow pad_len to avoid recompilation
        if seq_len > pad_len:
            pad_len = math.ceil(seq_len * 1.1)
            pad_len = min(pad_len, max_len)

        t0 = time.perf_counter()
        log(f"  [{i+1}/{len(rows)}] {eid} (len={seq_len}, existing={existing})")

        try:
            n_new = sample_ensemble(
                eid, seq, a3m_text, ensemble_out,
                work_dir / eid, args.num_samples, existing,
                model_runner_and_params, pad_len, args.model_type,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log(f"  [{i+1}/{len(rows)}] {eid}: ERROR {e}")
            log_ensemble(tool="cfrandom", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=elapsed,
                         status="failed", note=str(e)[:200])
            continue

        elapsed = time.perf_counter() - t0
        log(f"  [{i+1}/{len(rows)}] {eid}: {n_new} new, "
            f"{existing + n_new} total, {elapsed:.1f}s")
        log_ensemble(tool="cfrandom", dataset=args.run_id, ensemble=eid,
                     seq_len=seq_len, structures=n_new, wall_seconds=elapsed,
                     status="ok" if n_new > 0 else "noop")
        succeeded.append(row)

    total_elapsed = time.perf_counter() - total_start
    log(f"\nDone: {len(succeeded)}/{len(rows)} ensembles, {total_elapsed:.0f}s total")


if __name__ == "__main__":
    main()
