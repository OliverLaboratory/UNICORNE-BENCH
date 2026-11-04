#!/usr/bin/env python3
"""
Run AFsample3 on a benchmark dataset via its Apptainer image.

AFsample3 (Wallner lab) is an AlphaFold3 fork that adds MSA masking and seed
sampling: https://github.com/wallnerlab/afsample3

Config mirrors our AFsample2 runs so the two are comparable:
  - 1000 conformations per ensemble
  - MSA randomization rotated over {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
  - Templates off

AF2's `--dropout` has no AF3 equivalent; in AF3 the per-sample stochasticity
comes from the diffusion head, so each seed yields --num_diffusion_samples
structures.

Uses the precomputed MSAs from data/msas/<DATASET>/<eid>/a3m/<eid>.a3m (the
same colabfold-style MSAs afsample2 / alphaflow_pdb / cfrandom consume), which
lets us skip AF3's slow CPU data pipeline entirely (--run_data_pipeline=false).

Usage:
  python run_afsample3.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH \\
      --msa-dir data/msas/UNICORNE --num-samples 1000 --resume --gpu-shard 0/8

Output layout:
  data/outputs/afsample3/<run_id>/output_pdbs/<eid>/output_*.pdb
  (AF3's native CIF tree is kept under <eid>/_af3/ so AFsample3's own
   seed-resume keeps working across restarts)

SIF path defaults to ~/afsample3_run/afsample3.sif; override with $SIF or --sif.
Model weights default to ~/af3_model (must contain af3.bin); override with
$AF3_MODEL_DIR or --model-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Shared helpers (02_compute_predictions/prediction_helpers.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prediction_helpers import count_existing_pdbs, load_input_csv

# Runtime accounting (CSV row per ensemble at end). Import-safe failure if
# the shared logger isn't available — predictions take precedence.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from _runtime_log import log_ensemble
except Exception:
    def log_ensemble(*a, **kw): pass


# --- Sampling config (overridable via CLI) ----------------------------------
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_MSA_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NUM_DIFFUSION_SAMPLES = 5           # structures produced per seed
# Each apptainer invocation cold-loads AF3 + recompiles XLA (~2-5 min), so
# batch seeds per call to amortize it. 25 seeds x 5 diffusion = 125 structures
# per run, matching afsample2's STRUCTURES_PER_RUN.
SEEDS_PER_RUN = 25
STRUCTURES_PER_RUN = SEEDS_PER_RUN * NUM_DIFFUSION_SAMPLES


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def load_a3m_text(msa_dir: Path, eid: str) -> str | None:
    for cand in (msa_dir / eid / "a3m" / f"{eid}.a3m",
                 msa_dir / eid / f"{eid}.a3m",
                 msa_dir / f"{eid}.a3m"):
        if cand.is_file():
            return cand.read_text()
    return None


def a3m_query(a3m_text: str) -> str:
    """First (query) sequence of an a3m, ungapped and uppercased.

    Colabfold a3m files start with a '#<len>\\t<card>' comment line, which is
    skipped here exactly as AF3's own parser skips anything before the first
    '>' record.
    """
    out, seen = [], False
    for line in a3m_text.splitlines():
        if line.startswith(">"):
            if seen:
                break
            seen = True
            continue
        if seen:
            out.append(line.strip())
    return "".join(out).replace("-", "").replace(".", "").upper()


# --- Shared work pool -------------------------------------------------------
# Static --gpu-shard slices strand GPUs: a worker that finishes its slice exits
# while other slices still have hundreds of ensembles left. With --claim-dir,
# every worker instead walks the *whole* list and claims ensembles atomically
# via mkdir (atomic on NFS), so the fleet self-balances and nothing idles.

def claim_ensemble(claim_dir: Path | None, eid: str, stale_hours: float) -> bool:
    """Try to take exclusive ownership of an ensemble. True if we got it."""
    if claim_dir is None:
        return True
    lock = claim_dir / eid
    try:
        lock.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        # A worker that died mid-ensemble would otherwise park its claim
        # forever, so reclaim locks that have gone quiet.
        try:
            age_h = (time.time() - lock.stat().st_mtime) / 3600.0
        except OSError:
            return False
        if age_h >= stale_hours:
            try:
                lock.touch()
            except OSError:
                return False
            return True
        return False


def release_claim(claim_dir: Path | None, eid: str) -> None:
    if claim_dir is None:
        return
    try:
        (claim_dir / eid).rmdir()
    except OSError:
        pass


def looks_active(eid_dir: Path, active_window_min: float) -> bool:
    """True if some other process appears to be mid-run on this ensemble.

    Guards against colliding with workers started under the older static
    --gpu-shard scheme, which don't participate in the claim protocol.
    """
    af3 = eid_dir / "_af3"
    if not af3.is_dir():
        return False
    newest = af3.stat().st_mtime
    for p in af3.glob("*/seed-*_sample-*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return (time.time() - newest) < active_window_min * 60


def write_af3_input(eid: str, sequence: str, a3m_text: str, in_dir: Path) -> Path:
    """Stage the a3m + fold-input JSON for one ensemble.

    The a3m is written as a real file (not a symlink into the shared MSA tree)
    because the container only sees what we bind-mount.
    """
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / f"{eid}.a3m").write_text(a3m_text)
    spec = {
        "name": eid,
        "modelSeeds": [1],
        "sequences": [{
            "protein": {
                "id": "A",
                "sequence": sequence,
                "unpairedMsaPath": f"/af_input/{eid}.a3m",
                "pairedMsa": "",
                "templates": [],
            }
        }],
        "dialect": "alphafold3",
        "version": 1,
    }
    json_path = in_dir / f"{eid}.json"
    json_path.write_text(json.dumps(spec, indent=2))
    return json_path


# ----------------------------------------------------------------------------
# CIF -> PDB
# ----------------------------------------------------------------------------

_CONVERTER = None


def _get_converter():
    """Resolve a cif->pdb converter once (gemmi preferred, biopython fallback)."""
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER
    try:
        import gemmi

        def _conv(cif: Path, pdb: Path):
            st = gemmi.read_structure(str(cif))
            st.setup_entities()
            st.write_pdb(str(pdb))
        _CONVERTER = _conv
        return _CONVERTER
    except ImportError:
        pass
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.PDB.PDBIO import PDBIO

        def _conv(cif: Path, pdb: Path):
            structure = MMCIFParser(QUIET=True).get_structure("m", str(cif))
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb))
        _CONVERTER = _conv
        return _CONVERTER
    except ImportError as exc:
        raise RuntimeError("need gemmi or biopython to convert AF3 CIF output") from exc


def harvest_cifs(af3_dir: Path, eid_dir: Path, log) -> int:
    """Convert any not-yet-converted AF3 model.cif into output_*.pdb.

    A marker file records which CIFs are already converted so repeated calls
    stay cheap and idempotent.
    """
    done_marker = eid_dir / ".converted.txt"
    already = set(done_marker.read_text().split()) if done_marker.exists() else set()

    cifs = sorted(af3_dir.glob("*/seed-*_sample-*/model.cif"))
    new = [c for c in cifs if str(c.relative_to(af3_dir)) not in already]
    if not new:
        return 0

    convert = _get_converter()
    n = count_existing_pdbs(eid_dir)
    converted = 0
    for cif in new:
        pdb = eid_dir / f"output_{n:04d}.pdb"
        while pdb.exists():
            n += 1
            pdb = eid_dir / f"output_{n:04d}.pdb"
        try:
            convert(cif, pdb)
        except Exception as exc:  # noqa: BLE001 - one bad model shouldn't stop the run
            log(f"    cif->pdb failed for {cif.name}: {exc}")
            continue
        already.add(str(cif.relative_to(af3_dir)))
        converted += 1
        n += 1

    done_marker.write_text("\n".join(sorted(already)))
    return converted


# ----------------------------------------------------------------------------
# AFsample3 invocation
# ----------------------------------------------------------------------------

def prune_partial_seeds(af3_dir: Path, log) -> None:
    """Drop seeds whose sample set is incomplete.

    AFsample3 resumes by comparing models_done against
    seeds_done * num_diffusion_samples and raises ValueError on a mismatch,
    which is exactly what a job killed mid-seed leaves behind.
    """
    import shutil
    for sub in af3_dir.glob("*/"):
        seeds: dict[str, list[Path]] = {}
        for d in sub.glob("seed-*_sample-*"):
            sid = d.name.rsplit("_sample-", 1)[0]
            seeds.setdefault(sid, []).append(d)
        for sid, dirs in seeds.items():
            if sum(1 for d in dirs if (d / "model.cif").is_file()) != NUM_DIFFUSION_SAMPLES:
                log(f"    pruning incomplete seed {sid}")
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)


# AF3's Triton flash-attention kernel asks for more shared memory per block
# than some Ampere parts provide (A6000/GA102 offers ~99 KB, the kernel wants
# 108 KB for certain input shapes), so a subset of ensembles dies with
# "RESOURCE_EXHAUSTED: Shared memory size limit exceeded". A100/H100/H200 have
# enough and are unaffected. Rather than force the slower portable kernel
# everywhere, we detect that specific failure and retry the ensemble on 'xla'.
SHMEM_ERROR = "Shared memory size limit exceeded"


def build_cmd(sif: Path, in_dir: Path, model_dir: Path, af3_dir: Path,
              eid: str, cum_seeds: int, msa_rand: float,
              flash_impl: str = "triton") -> list[str]:
    return [
        "apptainer", "exec", "--nv",
        "-B", f"{in_dir}:/af_input:ro",
        "-B", f"{model_dir}:/root/models:ro",
        "-B", f"{af3_dir}:/af_output",
        str(sif),
        "python3", "/app/alphafold/run_alphafold.py",
        f"--json_path=/af_input/{eid}.json",
        "--model_dir=/root/models",
        "--output_dir=/af_output",
        "--run_data_pipeline=false",
        "--run_inference=true",
        f"--num_diffusion_samples={NUM_DIFFUSION_SAMPLES}",
        f"--num_seeds={cum_seeds}",
        f"--msa_rand_fraction={msa_rand}",
        f"--flash_attention_implementation={flash_impl}",
        "--seed_resume=true",
    ]


# ----------------------------------------------------------------------------
# Per-ensemble driver
# ----------------------------------------------------------------------------

def run_ensemble(
    eid: str,
    sequence: str,
    a3m_text: str,
    run_output: Path,
    sif: Path,
    model_dir: Path,
    num_samples: int,
    msa_fractions: list[float],
    log,
    flash_attention: str = "triton",
) -> int:
    """Sample for one ensemble. Returns number of NEW structures generated."""
    flash_impl = flash_attention
    eid_dir = run_output / "output_pdbs" / eid
    eid_dir.mkdir(parents=True, exist_ok=True)
    af3_dir = eid_dir / "_af3"
    af3_dir.mkdir(parents=True, exist_ok=True)

    ensemble_t0 = time.time()

    # Pick up anything a previous run produced but didn't get to convert.
    harvest_cifs(af3_dir, eid_dir, log)

    existing = count_existing_pdbs(eid_dir)
    deficit = num_samples - existing
    if deficit <= 0:
        log(f"  {eid}: skip (already {existing} samples)")
        log_ensemble(tool="afsample3", dataset=run_output.name, ensemble=eid,
                     seq_len=len(sequence), structures=0,
                     wall_seconds=time.time() - ensemble_t0, status="skip")
        return 0

    in_dir = eid_dir / "_af3_input"
    write_af3_input(eid, sequence, a3m_text, in_dir)

    runs_needed = (deficit + STRUCTURES_PER_RUN - 1) // STRUCTURES_PER_RUN
    n_fracs = len(msa_fractions)

    # AFsample3 resumes by slicing rng_seeds[seeds_done:], i.e. it skips by
    # *count*. So --num_seeds must be the running TOTAL, not the per-call
    # increment, or every call after the first is a no-op.
    seeds_done = len(list(af3_dir.glob("*/seed-*_sample-0/model.cif")))

    generated = 0
    cum_seeds = seeds_done
    for i in range(runs_needed):
        current = count_existing_pdbs(eid_dir)
        if current >= num_samples:
            break
        frac = msa_fractions[i % n_fracs]
        # Cap the increment at what's actually still missing, so the last run
        # (or a small --num-samples) doesn't overshoot by up to a full batch.
        remaining = num_samples - current
        inc = min(SEEDS_PER_RUN,
                  -(-remaining // NUM_DIFFUSION_SAMPLES))  # ceil division
        cum_seeds += inc

        log(f"  {eid} run {i+1}/{runs_needed} (rand={frac}, seeds->{cum_seeds}, "
            f"existing={current}/{num_samples}, attn={flash_impl})")
        t0 = time.time()
        cmd = build_cmd(sif, in_dir, model_dir, af3_dir, eid, cum_seeds, frac,
                        flash_impl)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

        if r.returncode != 0 and SHMEM_ERROR in r.stderr:
            # Measured on an A6000 (job 12907508): triton, xla, cudnn and
            # XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter
            # all fail identically, so no setting avoids this. Record it and
            # move on instead of burning GPU time on a retry that cannot work.
            log(f"    {eid} needs a GPU with more shared memory than this one "
                f"({SHMEM_ERROR}); recording and skipping")
            prune_partial_seeds(af3_dir, log)
            marker = run_output / "needs_large_shmem.txt"
            try:
                seen = set(marker.read_text().split()) if marker.exists() else set()
                if eid not in seen:
                    seen.add(eid)
                    marker.write_text("\n".join(sorted(seen)) + "\n")
            except OSError:
                pass
            break

        if r.returncode != 0:
            log(f"    apptainer FAILED (rc={r.returncode}): {r.stderr[-500:]}")
            prune_partial_seeds(af3_dir, log)
            break

        added = harvest_cifs(af3_dir, eid_dir, log)
        generated += added
        log(f"    +{added} structures in {time.time()-t0:.0f}s "
            f"(total: {count_existing_pdbs(eid_dir)})")

    final = count_existing_pdbs(eid_dir)
    log(f"  {eid}: done, {final} total ({generated} new)")
    status = "ok" if generated > 0 else ("noop" if final < num_samples else "skip")
    log_ensemble(tool="afsample3", dataset=run_output.name, ensemble=eid,
                 seq_len=len(sequence), structures=generated,
                 wall_seconds=time.time() - ensemble_t0, status=status)
    return generated


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_path", type=Path,
                   help="Input dataset path (contains input_dataset*.csv)")
    p.add_argument("run_id", help="Run ID for outputs (e.g. UNICORNE_BENCH)")
    p.add_argument("--msa-dir", type=Path, required=True,
                   help="Precomputed MSA directory (colabfold-style a3m)")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help=f"Target samples per ensemble (default: {DEFAULT_NUM_SAMPLES})")
    p.add_argument("--msa-fractions", type=float, nargs="+",
                   default=DEFAULT_MSA_FRACTIONS,
                   help="MSA randomization fractions to rotate over "
                        f"(default: {DEFAULT_MSA_FRACTIONS})")
    p.add_argument("--sif", type=Path,
                   default=Path(os.environ.get(
                       "SIF", str(Path.home() / "afsample3_run" / "afsample3.sif"))),
                   help="Path to afsample3 Apptainer SIF")
    p.add_argument("--model-dir", type=Path,
                   default=Path(os.environ.get(
                       "AF3_MODEL_DIR", str(Path.home() / "af3_model"))),
                   help="Directory containing af3.bin")
    p.add_argument("--output-base", type=Path,
                   default=PROJECT_ROOT / "data" / "outputs")
    p.add_argument("--tool", default="afsample3",
                   help="Tool directory under data/outputs/. Set this (rather "
                        "than run_id) when a differently-configured run must be "
                        "scored as its own tool, since the results pipeline "
                        "reads data/outputs/<tool>/<dataset>/output_pdbs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip ensembles already at --num-samples")
    p.add_argument("--gpu-shard", default=None, metavar="I/N",
                   help="Static slice of the ensemble list. Prefer --claim-dir "
                        "on a shared filesystem: static slices strand GPUs once "
                        "a worker finishes its slice.")
    p.add_argument("--claim-dir", type=Path, default=None,
                   help="Shared work pool. Workers walk the whole list and claim "
                        "ensembles atomically via mkdir in this dir, so the fleet "
                        "self-balances. Must be on a filesystem all workers share.")
    p.add_argument("--claim-stale-hours", type=float, default=4.0,
                   help="Reclaim a lock whose worker has gone quiet this long")
    p.add_argument("--active-window-min", type=float, default=15.0,
                   help="Treat an ensemble touched this recently as in-flight "
                        "(avoids colliding with --gpu-shard workers)")
    p.add_argument("--reverse", action="store_true",
                   help="Process longest sequences first (default is shortest "
                        "first). Use on a second site so it converges toward a "
                        "shortest-first site instead of duplicating its work.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--flash-attention", choices=["triton", "cudnn", "xla"],
                   default=os.environ.get("AF3_FLASH_ATTENTION", "triton"),
                   help="AF3 attention kernel. 'triton' is fastest but needs more "
                        "shared memory than some Ampere parts (A6000) provide for "
                        "certain shapes; the run auto-falls back to 'xla' on that "
                        "specific failure.")
    args = p.parse_args()

    if not args.sif.exists():
        print(f"Error: SIF not found: {args.sif}", file=sys.stderr)
        sys.exit(1)
    if not (args.model_dir / "af3.bin").exists():
        print(f"Error: af3.bin not found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    args.input_path = args.input_path.resolve()
    args.output_base = args.output_base.resolve()
    args.sif = args.sif.resolve()
    args.model_dir = args.model_dir.resolve()

    input_csv = next((args.input_path / n for n in ("input_dataset_filtered.csv",
                                                    "input_dataset.csv")
                      if (args.input_path / n).exists()), None)
    if input_csv is None:
        print(f"Error: no input CSV in {args.input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_input_csv(input_csv)
    print(f"Loaded {len(rows)} ensembles from {input_csv}")

    if args.gpu_shard:
        spec, n_s = args.gpu_shard.split("/")
        n = int(n_s)
        # "I/N" is one slice; "LO-HI/N" is a contiguous group of slices, which
        # lets a pool of workers share exactly the slices owned by one site
        # (e.g. 64-87/88 for the bf64+bf65 NFS tree) without straying into
        # another site's range and duplicating its compute.
        if "-" in spec:
            lo, hi = map(int, spec.split("-"))
            want = set(range(lo, hi + 1))
        else:
            want = {int(spec)}
        rows = [r for idx, r in enumerate(rows) if idx % n in want]
        print(f"GPU shard {spec}/{n}: processing {len(rows)} ensembles")

    # Sort by length — shorter first for faster early feedback.
    # --reverse exists so a second site can work the same list from the long
    # end: two pools on separate filesystems can't share a claim dir, but if
    # they converge from opposite ends they barely overlap.
    rows.sort(key=lambda r: len(r["ref_sequence"]), reverse=args.reverse)
    if args.limit:
        rows = rows[: args.limit]

    run_output = args.output_base / args.tool / args.run_id
    run_output.mkdir(parents=True, exist_ok=True)

    log_fh = open(args.log_file, "a") if args.log_file else None
    def log(msg=""):
        print(msg); sys.stdout.flush()
        if log_fh: log_fh.write(msg + "\n"); log_fh.flush()

    log(f"=== AFsample3 on {args.run_id} ===")
    log(f"SIF:            {args.sif}")
    log(f"Model dir:      {args.model_dir}")
    log(f"MSA dir:        {args.msa_dir}")
    log(f"Output:         {run_output}")
    log(f"Num samples:    {args.num_samples}")
    log(f"MSA fractions:  {args.msa_fractions}")
    log(f"Ensembles:      {len(rows)}")
    log("")

    t_start = time.time()
    done = skipped_no_msa = skipped_mismatch = 0
    interrupted = False
    idle_passes = 0
    pass_no = 0

    # Sweep repeatedly rather than once. A single pass abandons every ensemble
    # a peer happened to hold at the moment we walked past it, so a pool of
    # workers all exits with work still outstanding and the GPUs go idle --
    # exactly what stranded 13 ensembles across 24 workers. Keep sweeping until
    # a whole pass finds nothing to do.
    while not interrupted:
        pass_no += 1
        processed = 0

        for idx, row in enumerate(rows):
            eid, seq = row["ensemble_ID"], row["ref_sequence"].strip()
            eid_dir = run_output / "output_pdbs" / eid

            # Cheapest check first: skip finished work without touching the MSA.
            if count_existing_pdbs(eid_dir) >= args.num_samples:
                continue

            a3m = load_a3m_text(args.msa_dir, eid)
            if a3m is None:
                if pass_no == 1:
                    log(f"  [{idx+1}/{len(rows)}] {eid}: SKIP (no MSA)")
                    skipped_no_msa += 1
                continue
            # AF3 requires the a3m query row to match the modelled sequence.
            if a3m_query(a3m) != seq.upper():
                if pass_no == 1:
                    log(f"  [{idx+1}/{len(rows)}] {eid}: SKIP (a3m query != ref_sequence)")
                    skipped_mismatch += 1
                continue

            if args.claim_dir is not None:
                if looks_active(eid_dir, args.active_window_min):
                    continue
                if not claim_ensemble(args.claim_dir, eid, args.claim_stale_hours):
                    continue

            log(f"[pass {pass_no}] [{idx+1}/{len(rows)}] {eid} (len={len(seq)})")
            try:
                run_ensemble(
                    eid=eid, sequence=seq, a3m_text=a3m,
                    run_output=run_output,
                    sif=args.sif, model_dir=args.model_dir,
                    num_samples=args.num_samples,
                    msa_fractions=args.msa_fractions,
                    log=log,
                    flash_attention=args.flash_attention,
                )
                done += 1
                processed += 1
            except KeyboardInterrupt:
                log("Interrupted.")
                interrupted = True
                break
            except Exception as e:
                log(f"  ERROR: {e}")
            finally:
                release_claim(args.claim_dir, eid)

        if interrupted:
            break

        if processed:
            idle_passes = 0
            continue

        # A pass that did nothing means either we're done, or peers hold the
        # remainder. Wait once and re-check so we pick up anything they drop;
        # two consecutive empty passes means there's genuinely nothing left
        # (and stops us spinning on ensembles that can never complete).
        idle_passes += 1
        if idle_passes >= 2:
            break
        log(f"pass {pass_no}: nothing available; re-checking in 60s")
        time.sleep(60)

    elapsed = time.time() - t_start
    log(f"\nTotal: {done} ensembles sampled over {pass_no} pass(es), "
        f"{skipped_no_msa} skipped (no MSA), "
        f"{skipped_mismatch} skipped (MSA/seq mismatch), {elapsed/60:.1f} min")
    if log_fh: log_fh.close()


if __name__ == "__main__":
    main()
