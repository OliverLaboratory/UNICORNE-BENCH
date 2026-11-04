#!/usr/bin/env python3
"""
Run SPEACH_AF on a benchmark dataset using the direct ColabFold API.

SPEACH_AF (Stein & Mchaourab, PLoS Comput Biol 2022, doi:10.1371/journal.pcbi.1010483):
  1. Predict the structure with the *unmodified* MSA → pick best-pLDDT model
  2. Slide an 11-residue window along the structure; for each window, list the
     residues in the window that make Cα-Cα contacts (< CONTACT_DIST Å) with
     residues outside their local sequence neighborhood (separation > MIN_SEQSEP)
  3. For each "interaction set" thus found, write a modified MSA where those
     residue columns are mutated to alanine across every aligned sequence
     (non-gap positions only)
  4. Run AF2 on each modified MSA: 5 models × ``--seeds-per-set`` seeds
  5. Also run the unmodified MSA additional times to match the paper's
     ``--seeds-base`` × 5 base models

Defaults aim for ~1000 samples per protein:
  - 11-aa non-overlapping windows → ~N_seq/11 sets per target
  - --seeds-per-set 9 → 5 models × 9 seeds = 45 predictions per set
  - With ~22 sets that's 990 ≈ 1000

Loads AF2 models ONCE at startup and reuses them.

Usage:
  python run_speach_af.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH \\
      --msa-dir data/msas/UNICORNE --num-samples 1000 --resume --gpu-shard 0/4

Output:
  data/outputs/speach_af/<run_id>/output_pdbs/<ensemble_ID>/output_NNNN.pdb
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
from typing import Iterable

import numpy as np

# --- ColabFold env setup (must precede jax/tf imports) -----------------------
os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "4.0")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Shared helpers (02_compute_predictions/prediction_helpers.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prediction_helpers import count_existing_pdbs, load_a3m, load_input_csv
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
try:
    from _runtime_log import log_ensemble
except Exception:
    def log_ensemble(*a, **kw): pass


logger = logging.getLogger("speach_af")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# MSA helpers (mirrors run_cfrandom.py)
# ---------------------------------------------------------------------------

def parse_a3m(a3m_text: str) -> tuple[list[str], list[str]]:
    """Split A3M into headers + aligned sequences (a3m semantics: lowercase
    letters are insertions relative to the query and are dropped here so all
    rows align column-by-column to the query)."""
    headers, seqs = [], []
    cur_header = None
    cur_seq = []
    for line in a3m_text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if cur_header is not None:
                headers.append(cur_header)
                seqs.append("".join(cur_seq))
            cur_header = line
            cur_seq = []
        else:
            # drop lowercase (insertions) — keep uppercase + '-'
            cur_seq.append("".join(c for c in line if c.isupper() or c == "-"))
    if cur_header is not None:
        headers.append(cur_header)
        seqs.append("".join(cur_seq))
    return headers, seqs


def write_a3m(headers: list[str], seqs: list[str]) -> str:
    out = []
    for h, s in zip(headers, seqs):
        out.append(h)
        out.append(s)
    return "\n".join(out) + "\n"


def alaninate_msa(headers: list[str], seqs: list[str], positions: Iterable[int]) -> str:
    """Replace columns *positions* with 'A' across every sequence (non-gap only)."""
    pos = sorted(set(int(p) for p in positions))
    if not seqs:
        return write_a3m(headers, seqs)
    L = len(seqs[0])
    pos = [p for p in pos if 0 <= p < L]
    new_seqs = []
    for s in seqs:
        if len(s) != L:
            # malformed row — leave it
            new_seqs.append(s)
            continue
        chars = list(s)
        for p in pos:
            if chars[p] != "-":
                chars[p] = "A"
        new_seqs.append("".join(chars))
    return write_a3m(headers, new_seqs)


# ---------------------------------------------------------------------------
# Structure helpers — Cα contact analysis
# ---------------------------------------------------------------------------

def parse_pdb_ca(pdb_text: str) -> np.ndarray:
    """Return (N, 3) array of Ca coords from a single-chain PDB string (first MODEL)."""
    coords = []
    in_model = False
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            if in_model:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return np.asarray(coords, dtype=np.float64)


def parse_pdb_atoms(pdb_text: str):
    """Parse first MODEL. Return (atom_xyz [Na,3], atom_res [Na] 0-indexed residue,
    ca_bfactor [Nres] pLDDT per residue). Residues indexed in order of appearance."""
    xyz = []
    ares = []
    ca_b = []
    cur = None
    ridx = -1
    in_model = False
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            if in_model:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        key = line[21:27]  # chain + resSeq + iCode
        if key != cur:
            cur = key
            ridx += 1
            ca_b.append(0.0)
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        xyz.append([x, y, z])
        ares.append(ridx)
        if line[12:16].strip() == "CA":
            try:
                ca_b[ridx] = float(line[60:66])
            except ValueError:
                pass
    return (np.asarray(xyz, dtype=np.float64),
            np.asarray(ares, dtype=np.int64),
            np.asarray(ca_b, dtype=np.float64))


def interaction_sets(
    atom_xyz: np.ndarray,
    atom_res: np.ndarray,
    ca_bfac: np.ndarray,
    n_res: int,
    window: int = 10,
    proximity: float = 4.0,
) -> list[list[int]]:
    """SPEACH_AF alanine-masking sets, faithful to RSvan/SPEACH_AF SPEACH_AF_scan.

    - Scan only the high-confidence region (residues with pLDDT/B-factor > mean).
    - Windows of *window*+1 residues [min_res, max_res] stepping by *window*
      (1-residue overlap), exactly as the notebook's ``p*window+start`` /
      ``p*window+start+window`` bounds; plus a trailing remainder window over the
      leftover ``span % window`` residues (notebook's ``range(res_no, res_no+1)``).
    - All-atom contacts: for each window, find residues whose atoms are within
      *proximity* (A) of the window's atoms, EXCLUDING partners within +/-4 in
      sequence (long-range tertiary contacts only). Mask both the window residue
      and its distal partner. One residue set per window.
    Returns 0-indexed residue positions per window.
    """
    if atom_xyz.shape[0] == 0 or n_res == 0:
        return []
    # High-confidence (pLDDT>mean) scan region, mirroring the notebook's b-factor filter
    if ca_bfac.size and np.any(ca_bfac > 0):
        mb = ca_bfac[ca_bfac > 0].mean()
        hi = np.where(ca_bfac > mb)[0]
    else:
        hi = np.arange(n_res)
    if hi.size == 0:
        hi = np.arange(n_res)
    start = int(hi[0])
    span = int(hi[-1] - hi[0])
    res_no, res_rem = divmod(span, window)
    sets: list[list[int]] = []

    def collect(min_res: int, max_res: int) -> None:
        # atoms in residues [min_res, max_res] inclusive (notebook includes max_res)
        widx = np.where((atom_res >= min_res) & (atom_res <= max_res))[0]
        if widx.size == 0:
            return
        wxyz = atom_xyz[widx]
        d = np.sqrt(((wxyz[:, None, :] - atom_xyz[None, :, :]) ** 2).sum(axis=-1))
        wi_arr, aj_arr = np.where(d < proximity)
        res_to_use = set()
        for wi, aj in zip(wi_arr, aj_arr):
            pr = int(atom_res[aj])
            if not (min_res - 4 <= pr <= max_res + 4):
                res_to_use.add(int(atom_res[widx[wi]]))
                res_to_use.add(pr)
        if res_to_use:
            sets.append(sorted(res_to_use))

    last_max = start
    for p in range(res_no):
        min_res = p * window + start
        max_res = min_res + window          # inclusive -> window+1 residues
        collect(min_res, max_res)
        last_max = max_res
    # trailing remainder window (notebook's range(res_no, res_no+1))
    if res_rem > 0:
        rem_min = last_max + 1
        collect(rem_min, rem_min + res_rem - 1)
    return sets

# ---------------------------------------------------------------------------
# AF2 prediction (reuses run_cfrandom.py's pattern)
# ---------------------------------------------------------------------------

def build_features(sequence: str, a3m_text: str):
    from colabfold.batch import build_monomer_feature, mk_mock_template
    lines = a3m_text.splitlines(keepends=True)
    a3m_clean = "".join(l for l in lines if not l.startswith("#"))
    template_features = mk_mock_template(sequence)
    feature_dict = build_monomer_feature(sequence, a3m_clean, template_features)
    feature_dict["asym_id"] = np.zeros(feature_dict["aatype"].shape[0], dtype=int)
    return feature_dict


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
) -> tuple[list[str], list[float]]:
    """Run predict_structure. Returns (unrelaxed_pdb_texts, plddt_means)."""
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
    pdb_files = [
        f for f in results.get("result_files", [])
        if str(f).endswith(".pdb") and "unrelaxed" in str(f)
    ]
    pdb_texts = [f.read_text() for f in pdb_files]
    plddts: list[float] = []
    for f in pdb_files:
        # extract from filename ("...rank_001_..._plddt_85.32.pdb"). Fall back to None.
        name = f.name
        plddt = None
        for token in name.split("_"):
            try:
                plddt = float(token.replace(".pdb", ""))
                break
            except ValueError:
                continue
        if plddt is None:
            plddt = mean_plddt_from_pdb(f.read_text())
        plddts.append(plddt)
    for f in results.get("result_files", []):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    return pdb_texts, plddts


def mean_plddt_from_pdb(pdb_text: str) -> float:
    """Mean Cα B-factor (= pLDDT in colabfold output)."""
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else 0.0


def pdb_text_has_ca(pdb_text: str) -> bool:
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            return True
    return False


# ---------------------------------------------------------------------------
# Per-ensemble SPEACH_AF
# ---------------------------------------------------------------------------

def sample_ensemble_speach_af(
    eid: str,
    sequence: str,
    a3m_text: str,
    ensemble_out: Path,
    work_dir: Path,
    num_samples: int,
    existing_count: int,
    seeds_per_set: int,
    seeds_base: int,
    window: int,
    stride: int,
    contact_dist: float,
    min_seqsep: int,
    model_runner_and_params,
    pad_len: int,
    model_type: str,
) -> int:
    deficit = num_samples - existing_count
    if deficit <= 0:
        return 0

    ensemble_out.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    total_new = 0
    seq_len = len(sequence)
    headers, seqs = parse_a3m(a3m_text)
    if not seqs or len(seqs[0]) != seq_len:
        logger.warning(f"  {eid}: MSA query length mismatch ({len(seqs[0]) if seqs else 0} vs {seq_len}) — skipping")
        return 0

    def write_pdbs(pdb_texts: list[str]) -> int:
        nonlocal total_new
        wrote = 0
        for pdb_text in pdb_texts:
            if total_new >= deficit:
                break
            if pdb_text_has_ca(pdb_text):
                out_path = ensemble_out / f"output_{existing_count + total_new:04d}.pdb"
                out_path.write_text(pdb_text)
                total_new += 1
                wrote += 1
        return wrote

    # --- Step 1: base AF2 run on unmodified MSA (pick reference for contacts) ---
    feature_dict = build_features(sequence, a3m_text)
    base_pdbs, base_plddts = predict_and_extract(
        prefix=f"{eid}_base",
        work_dir=work_dir / "base",
        feature_dict=feature_dict,
        seq_len=seq_len,
        pad_len=pad_len,
        model_type=model_type,
        model_runner_and_params=model_runner_and_params,
        random_seed=42,
        num_seeds=seeds_base,
    )
    write_pdbs(base_pdbs)
    if not base_pdbs:
        logger.warning(f"  {eid}: base AF2 run produced no structures — skipping")
        return total_new

    # Pick best-pLDDT base model for contact analysis
    best_idx = int(np.argmax(base_plddts)) if base_plddts else 0
    atom_xyz, atom_res, ca_bfac = parse_pdb_atoms(base_pdbs[best_idx])
    del feature_dict

    # --- Step 2: derive interaction sets (SPEACH_AF all-atom masking) ---
    sets = interaction_sets(
        atom_xyz, atom_res, ca_bfac, seq_len,
        window=window, proximity=contact_dist,
    )
    logger.info(f"  {eid}: {len(sets)} interaction sets from {seq_len} residues")
    if not sets:
        return total_new

    # Coverage-first: auto-size seeds so EVERY interaction window runs within the
    # sample budget regardless of protein length (seeds_per_set<=0 => auto).
    if seeds_per_set <= 0:
        per_run = max(1, round(len(base_pdbs) / max(1, seeds_base)))
        remaining = max(0, deficit - total_new)
        seeds_per_set = max(1, remaining // (per_run * len(sets)))
        logger.info(
            f"  {eid}: auto seeds_per_set={seeds_per_set} across {len(sets)} windows "
            f"(~{per_run * seeds_per_set * len(sets)} samples; full coverage)"
        )

    # --- Step 3-4: per-set modified MSA + AF2 with seeds_per_set seeds ---
    rng = np.random.default_rng(hash(eid) & 0xFFFFFFFF)
    for s_idx, residues in enumerate(sets):
        modified = alaninate_msa(headers, seqs, residues)
        feature_dict = build_features(sequence, modified)
        seed = int(rng.integers(0, 1_000_000))
        try:
            pdbs, _plddts = predict_and_extract(
                prefix=f"{eid}_set{s_idx:02d}",
                work_dir=work_dir / f"set_{s_idx:02d}",
                feature_dict=feature_dict,
                seq_len=seq_len,
                pad_len=pad_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                random_seed=seed,
                num_seeds=seeds_per_set,
            )
        except Exception as e:
            logger.warning(f"  {eid}: set {s_idx} AF2 run failed ({e}) — skipping set")
            del feature_dict
            continue
        wrote = write_pdbs(pdbs)
        del feature_dict
        logger.info(
            f"    set {s_idx:02d}: {len(residues)} residues alaninated, "
            f"wrote {wrote} (total: {existing_count + total_new})"
        )

    return total_new


# ---------------------------------------------------------------------------
# CLI & main loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Run SPEACH_AF (Stein & Mchaourab 2022) via direct ColabFold API.")
    p.add_argument("input_path", type=Path,
                   help="Input dataset path (contains input_dataset.csv)")
    p.add_argument("run_id", help="Run ID for output naming")
    p.add_argument("--msa-dir", type=Path, required=True,
                   help="MSA directory (per-ensemble A3M files)")
    p.add_argument("--num-samples", type=int, default=1000,
                   help="Target number of samples per ensemble (default 1000)")
    p.add_argument("--output-base", type=Path,
                   default=PROJECT_ROOT / "data" / "outputs",
                   help="Base path for outputs")
    p.add_argument("--model-type", default="alphafold2_ptm")
    p.add_argument("--resume", action="store_true",
                   help="Skip ensembles that already have >= num_samples conformations")
    p.add_argument("--gpu-shard", default=None, metavar="I/N")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="ColabFold params dir (default: ~/.cache/colabfold)")
    # SPEACH_AF-specific knobs (defaults match the paper, with seeds tuned for ~1000)
    p.add_argument("--window", type=int, default=10,
                   help="Sliding window size in residues (paper: 11)")
    p.add_argument("--stride", type=int, default=11,
                   help="Window stride (paper: non-overlapping ⇒ stride==window)")
    p.add_argument("--contact-dist", type=float, default=4.0,
                   help="Cα-Cα distance threshold for contacts (Å, default 8.0)")
    p.add_argument("--min-seqsep", type=int, default=4,
                   help="Minimum |i-j| sequence separation to count a contact")
    p.add_argument("--seeds-per-set", type=int, default=0,
                   help="Random seeds per modified MSA (5 models x this = preds per window). "
                        "0 = AUTO: size seeds so EVERY window is covered within --num-samples (recommended).")
    p.add_argument("--seeds-base", type=int, default=3,
                   help="Random seeds on the unmodified MSA (paper: 3 → 15 base models)")
    args = p.parse_args()

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

    rows.sort(key=lambda r: len(r["ref_sequence"]))

    # --- Output dirs ---
    run_output = args.output_base / "speach_af" / args.run_id
    pdbs_dir = run_output / "output_pdbs"
    work_dir = run_output / "_work"
    run_output.mkdir(parents=True, exist_ok=True)
    pdbs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_fh = open(args.log_file, "a") if args.log_file else None

    def log(msg=""):
        print(msg)
        sys.stdout.flush()
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    # --- Load AF2 models ONCE ---
    log("Loading AF2 models (one-time)...")
    t_load = time.perf_counter()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.download import default_data_dir
    data_dir = args.data_dir or Path(default_data_dir)
    model_runner_and_params = load_models_and_params(
        num_models=5, use_templates=False, num_recycles=None, num_ensemble=1,
        model_order=[1, 2, 3, 4, 5], model_type=args.model_type, data_dir=data_dir,
        stop_at_score=100, rank_by="plddt", max_seq=512, max_extra_seq=5120,
        use_cluster_profile=True, use_fuse=True, use_bfloat16=True, use_dropout=False,
    )
    log(f"Models loaded in {time.perf_counter() - t_load:.1f}s")

    max_len = max(len(r["ref_sequence"]) for r in rows) if rows else 0
    pad_len = 0

    total_start = time.perf_counter()
    succeeded = []

    for i, row in enumerate(rows):
        eid = row["ensemble_ID"]
        seq = row["ref_sequence"]
        seq_len = len(seq)

        ensemble_out = pdbs_dir / eid
        existing = count_existing_pdbs(ensemble_out, "output_*.pdb")

        if args.resume and existing >= args.num_samples:
            log(f"  [{i+1}/{len(rows)}] {eid}: skip (already {existing} samples)")
            log_ensemble(tool="speach_af", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=0.0,
                         status="skip", note=f"already {existing} samples")
            succeeded.append(row)
            continue

        a3m_text = load_a3m(args.msa_dir, eid)
        if a3m_text is None:
            log(f"  [{i+1}/{len(rows)}] {eid}: SKIP (no MSA)")
            log_ensemble(tool="speach_af", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=0.0,
                         status="skip", note="no MSA")
            continue

        if seq_len > pad_len:
            pad_len = math.ceil(seq_len * 1.1)
            pad_len = min(pad_len, max_len)

        t0 = time.perf_counter()
        log(f"  [{i+1}/{len(rows)}] {eid} (len={seq_len}, existing={existing})")

        try:
            n_new = sample_ensemble_speach_af(
                eid=eid, sequence=seq, a3m_text=a3m_text,
                ensemble_out=ensemble_out, work_dir=work_dir / eid,
                num_samples=args.num_samples, existing_count=existing,
                seeds_per_set=args.seeds_per_set, seeds_base=args.seeds_base,
                window=args.window, stride=args.stride,
                contact_dist=args.contact_dist, min_seqsep=args.min_seqsep,
                model_runner_and_params=model_runner_and_params,
                pad_len=pad_len, model_type=args.model_type,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log(f"  [{i+1}/{len(rows)}] {eid}: ERROR {e}")
            log_ensemble(tool="speach_af", dataset=args.run_id, ensemble=eid,
                         seq_len=seq_len, structures=0, wall_seconds=elapsed,
                         status="failed", note=str(e)[:200])
            continue

        elapsed = time.perf_counter() - t0
        log(f"  [{i+1}/{len(rows)}] {eid}: {n_new} new, "
            f"{existing + n_new} total, {elapsed:.1f}s")
        log_ensemble(tool="speach_af", dataset=args.run_id, ensemble=eid,
                     seq_len=seq_len, structures=n_new, wall_seconds=elapsed,
                     status="ok" if n_new > 0 else "noop")
        succeeded.append(row)

    total_elapsed = time.perf_counter() - total_start
    log(f"\nDone: {len(succeeded)}/{len(rows)} ensembles, {total_elapsed:.0f}s total")
    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
