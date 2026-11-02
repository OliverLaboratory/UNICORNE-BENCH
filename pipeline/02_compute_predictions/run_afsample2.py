#!/usr/bin/env python3
"""
Run AFsample2 on a benchmark dataset via its Apptainer image.

Paper-faithful config (Kalakoti & Wallner 2025, Comms Biology):
  - 1000 conformations per ensemble (Fig 2c: TM still rising at 1000)
  - MSA randomization rotated over {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
    (paper sweet spot = 0.15; rotation covers per-target optima, Fig 2b)
  - Templates off, dropout on

Uses precomputed MSAs from data/msas/<DATASET>/<eid>/a3m/<eid>.a3m
(the same colabfold-style MSAs alphaflow_pdb / cfrandom consume).
The script wraps that single a3m as AF2's `bfd_uniref_hits.a3m` and
stubs out the Stockholm files, matching AF2's expected MSA layout.

Usage:
  python run_afsample2.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH \\
      --msa-dir data/msas/UNICORNE --num-samples 1000 --resume --gpu-shard 0/4

Output layout:
  data/outputs/afsample2/<run_id>/output_pdbs/<eid>/unrelaxed_*.pdb
  (existing files kept; --resume skips ensembles with >= num_samples)

SIF path defaults to ~/afsample2_v1.1.sif; override with $SIF or --sif.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Runtime accounting (CSV row per ensemble at end). Import-safe failure if
# the shared logger isn't available — predictions take precedence.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from _runtime_log import log_ensemble
except Exception:
    def log_ensemble(*a, **kw): pass


# --- Paper-faithful sampling config (overridable via CLI) --------------------
DEFAULT_NUM_SAMPLES = 1000          # Kalakoti & Wallner 2025, Fig 2c
DEFAULT_MSA_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# Each apptainer invocation cold-loads AF2 (~3-5 min), so bigger batches
# per-call amortize that overhead. 5 models × 25 predictions/model = 125
# structures per run → 8 runs for 1000 samples (vs 40 at the old 25/run).
STRUCTURES_PER_RUN = 125
NSTRUCT_PER_RUN = 25                # num_monomer_predictions_per_model = 5 (kept)


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def load_input_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eid = (row.get("ensemble_ID") or row.get("ensemble_id")
                   or row.get("uniprot", "")).strip()
            seq = (row.get("ref_sequence") or row.get("sequence", "")).strip()
            if eid and seq:
                rows.append({"ensemble_ID": eid, "ref_sequence": seq})
    return rows


def load_a3m_text(msa_dir: Path, eid: str) -> str | None:
    for cand in [msa_dir / eid / "a3m" / f"{eid}.a3m",
                 msa_dir / f"{eid}-search" / "0.a3m",
                 msa_dir / eid / "0.a3m"]:
        if cand.exists():
            return cand.read_text()
    return None


def a3m_to_stockholm(a3m_text: str) -> str:
    """Convert a colabfold a3m into a Stockholm alignment.

    a3m encodes insertions relative to the query as lowercase letters and '.',
    which are *not* alignment columns. Stockholm needs every row the same
    width, so those characters are dropped, leaving each row the length of the
    query. Names are uniquified because a3m headers can repeat and Stockholm
    requires unique sequence names.
    """
    names: list[str] = []
    seqs: list[str] = []
    cur: list[str] | None = None
    for line in a3m_text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if cur is not None:
                seqs.append("".join(cur))
            cur = []
            raw = line[1:].strip().split()[0] if line[1:].strip() else "seq"
            names.append(re.sub(r"[^A-Za-z0-9_.:/-]", "_", raw) or "seq")
            continue
        if cur is not None:
            cur.append(line.strip())
    if cur is not None:
        seqs.append("".join(cur))

    aligned = ["".join(c for c in s if not c.islower() and c != ".") for s in seqs]
    if not aligned:
        return "# STOCKHOLM 1.0\n//\n"
    width = len(aligned[0])

    seen: dict[str, int] = {}
    rows: list[tuple[str, str]] = []
    for nm, sq in zip(names, aligned):
        if len(sq) != width:      # ragged row -> skip rather than corrupt the block
            continue
        n = seen.get(nm, 0)
        seen[nm] = n + 1
        rows.append((nm if n == 0 else f"{nm}_{n}", sq))

    pad = max(len(n) for n, _ in rows) + 2
    body = "".join(f"{n.ljust(pad)}{s}\n" for n, s in rows)
    return ("# STOCKHOLM 1.0\n" + body
            + f"{'#=GC RF'.ljust(pad)}{'x' * width}\n//\n")


def make_af2_msa_layout(seq_name: str, seq: str, a3m_text: str, out_msa_dir: Path):
    """Write AF2's expected msas/ layout inside a per-ensemble work dir.

    AFsample2 / AF2 monomer path reads:
      - msas/bfd_uniref_hits.a3m  (full_dbs: hhblits against BFD+UniRef30)
      - msas/small_bfd_hits.sto   (reduced_dbs: jackhmmer against small_bfd)
      - msas/uniref90_hits.sto    (stub — query-only Stockholm)
      - msas/mgnify_hits.sto      (stub)

    Both the full_dbs and reduced_dbs BFD filenames are written, because the
    preset decides which one AF2 looks for and the two invocation paths in this
    script disagree: the apptainer path uses monomer_full_dbs.flag while the
    native path generates --db_preset=reduced_dbs. Staging only the full_dbs
    name meant reduced_dbs found nothing, so --use_precomputed_msas silently
    failed to short-circuit and jackhmmer ran against small_bfd anyway
    (blowing up downstream in truncate_stockholm_msa).

    Ensures `--use_precomputed_msas=True` works for either preset.
    """
    out_msa_dir.mkdir(parents=True, exist_ok=True)
    # Strip colabfold comment headers (# nseq length)
    cleaned = "".join(l for l in a3m_text.splitlines(keepends=True)
                      if not l.startswith("#"))
    (out_msa_dir / "bfd_uniref_hits.a3m").write_text(cleaned)
    # reduced_dbs reads the same alignment, but as Stockholm under another name
    (out_msa_dir / "small_bfd_hits.sto").write_text(a3m_to_stockholm(a3m_text))

    rf_mask = "x" * len(seq)
    stockholm = (
        "# STOCKHOLM 1.0\n"
        f"#=GF ID {seq_name}\n"
        f"{seq_name}/1-{len(seq)} {seq}\n"
        f"#=GC RF {rf_mask}\n//\n"
    )
    for sto in ("uniref90_hits.sto", "mgnify_hits.sto"):
        (out_msa_dir / sto).write_text(stockholm)


def count_existing_pdbs(d: Path) -> int:
    return len(list(d.glob("*.pdb"))) if d.exists() else 0


# ----------------------------------------------------------------------------
# Apptainer invocation
# ----------------------------------------------------------------------------

def build_apptainer_cmd(
    sif: Path, db_path: Path, project_root: Path,
    dataset_dir: str, fasta_name: str, output_subpath: str,
    msa_rand: float, use_precomputed_features: bool, use_precomputed_msas: bool,
    num_monomer_predictions_per_model: int = 5,
    nstruct: int = NSTRUCT_PER_RUN,
) -> list[str]:
    """Assemble the `apptainer run` argv.

    Container bind-mounts mirror the original SLURM launcher 1:1.
    """
    cmd = [
        "apptainer", "run", "--nv", "--no-home", "--cleanenv",
        "--env", "PYTHONNOUSERSITE=1",
        "-B", f"{db_path}:/databases",
        "-B", f"{db_path}/params:/proj/beyondfold/apps/alphafold_data/params/",
        "-B", f"{db_path}/pdb70:/proj/beyondfold/apps/alphafold_data/pdb70/",
        "-B", f"{db_path}/pdb_mmcif/mmcif_files:/proj/beyondfold/apps/alphafold_data/pdb_mmcif/mmcif_files/",
        "-B", f"{db_path}/pdb_mmcif:/proj/beyondfold/apps/alphafold_data/pdb_mmcif/",
        "-B", f"{db_path}/uniref90:/proj/beyondfold/apps/alphafold_data/uniref90/",
        "-B", f"{db_path}/bfd:/proj/beyondfold/apps/alphafold_data/bfd/",
        "-B", f"{db_path}/uniref30:/proj/beyondfold/apps/alphafold_data/uniclust30/UniRef30_2021_03/",
        "-B", f"{db_path}/mgnify:/proj/wallner-b/users/x_yogka/github_repositories/AFsample2/AF_multitemplate/AF_data/mgnify/",
        "-B", f"{project_root}:/workspace",
    ]
    # If db_path is a symlink tree pointing at /data/mchaourab/alphafold3-data
    # (the ACCRE shared DB), make that path visible inside the container so the
    # symlinks resolve. Otherwise (e.g. bf64's self-contained /home/.../alphafold3-data)
    # this no-ops.
    accre_shared = Path("/data/mchaourab/alphafold3-data")
    if accre_shared.exists():
        cmd += ["-B", f"{accre_shared}:{accre_shared}"]
    cmd += [
        str(sif),
        "--method", "afsample2",
        "--fasta_paths", f"/workspace/{dataset_dir}/fastas/{fasta_name}",
        "--flagfile", "/app/alphafold/AF_multitemplate/monomer_full_dbs.flag",
        "--num_monomer_predictions_per_model", str(num_monomer_predictions_per_model),
        "--nstruct", str(nstruct),
        "--msa_rand_fraction", f"{msa_rand}",
        "--model_preset", "monomer",
        "--output_dir", f"/workspace/{output_subpath}",
        f"--use_precomputed_features={use_precomputed_features}",
        f"--use_precomputed_msas={use_precomputed_msas}",
        "--dropout=True",
    ]
    return cmd


def write_fasta(seq_name: str, seq: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f">{seq_name}\n{seq}\n")


# ----------------------------------------------------------------------------
# Native invocation (no apptainer) — for hosts without the SIF, e.g. CSB
# workstations with their own venv (afsample2_rtx5090_venv) and AFsample2 repo.
# Uses reduced_dbs (small_bfd, no uniref30/bfd) so it works on hosts that lack
# the full AF2 databases.
# ----------------------------------------------------------------------------

def make_reduced_dbs_flagfile(db_path: Path, out_path: Path):
    """Write a reduced_dbs flagfile pointing at native host AF2 data.

    Binary paths: prefer localcolabfold pixi env, fall back to /bin/true for
    missing tools. With --use_precomputed_msas=True the missing ones aren't
    actually invoked; the path just needs to exist to pass AFsample2's
    startup validation.
    """
    lcf = Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin"

    def bin_path(name: str) -> str:
        cand = lcf / name
        if cand.exists():
            return str(cand)
        # which() lookup
        from shutil import which
        w = which(name)
        if w:
            return w
        return "/bin/true"  # placeholder; only safe when not actually invoked

    content = f"""--data_dir={db_path}
--db_preset=reduced_dbs
--small_bfd_database_path={db_path}/small_bfd/bfd-first_non_consensus_sequences.fasta
--obsolete_pdbs_path={db_path}/pdb_mmcif/obsolete.dat
--pdb70_database_path={db_path}/pdb70/pdb70
--template_mmcif_dir={db_path}/pdb_mmcif/mmcif_files
--mgnify_database_path={db_path}/mgnify/mgy_clusters_2022_05.fa
--uniref90_database_path={db_path}/uniref90/uniref90.fasta
--jackhmmer_binary_path={bin_path("jackhmmer")}
--hhblits_binary_path={bin_path("hhblits")}
--hhsearch_binary_path={bin_path("hhsearch")}
--hmmsearch_binary_path={bin_path("hmmsearch")}
--hmmbuild_binary_path={bin_path("hmmbuild")}
--kalign_binary_path={bin_path("kalign")}
--max_template_date=3000-01-01
--use_gpu_relax=False
--max_recycles=3
--no_templates=True
--models_to_relax=None
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)


def build_native_cmd(
    python_bin: Path, afsample2_dir: Path, flagfile: Path,
    fasta_path: Path, output_dir: Path,
    msa_rand: float, use_precomputed_features: bool, use_precomputed_msas: bool,
    num_monomer_predictions_per_model: int = 5,
    nstruct: int = NSTRUCT_PER_RUN,
) -> list[str]:
    """Build the native (non-apptainer) afsample2 cmd. Same args as SIF version."""
    return [
        str(python_bin),
        str(afsample2_dir / "AF_multitemplate" / "run_afsample2.py"),
        f"--flagfile={flagfile}",
        f"--fasta_paths={fasta_path}",
        f"--output_dir={output_dir}",
        "--method=afsample2",
        f"--num_monomer_predictions_per_model={num_monomer_predictions_per_model}",
        f"--nstruct={nstruct}",
        f"--msa_rand_fraction={msa_rand}",
        "--model_preset=monomer",
        f"--use_precomputed_features={use_precomputed_features}",
        f"--use_precomputed_msas={use_precomputed_msas}",
        "--dropout=True",
    ]


# ----------------------------------------------------------------------------
# Per-ensemble driver
# ----------------------------------------------------------------------------

def run_ensemble(
    eid: str,
    sequence: str,
    a3m_text: str,
    run_output: Path,
    fastas_dir: Path,
    sif: Path,
    db_path: Path,
    project_root: Path,
    num_samples: int,
    msa_fractions: list[float],
    log,
    native: bool = False,
    python_bin: Path | None = None,
    afsample2_dir: Path | None = None,
) -> int:
    """Sample for one ensemble. Returns number of NEW structures generated."""
    # Write FASTA (AFsample2 reads by name from /workspace/.../fastas/<eid>.fasta)
    fasta_path = fastas_dir / f"{eid}.fasta"
    write_fasta(eid, sequence, fasta_path)

    # AFsample2's `--output_dir` points at the PARENT of <eid>/, so set it to
    # the run's output_pdbs/ — container writes output_pdbs/<eid>/afsample2/*.pdb
    # Existing on-disk files here use the flat layout output_pdbs/<eid>/*.pdb,
    # so after each apptainer call we flatten by moving *.pdb up one level.
    eid_dir = run_output / "output_pdbs" / eid
    eid_dir.mkdir(parents=True, exist_ok=True)

    ensemble_t0 = time.time()
    existing = count_existing_pdbs(eid_dir)
    deficit = num_samples - existing
    if deficit <= 0:
        log(f"  {eid}: skip (already {existing} samples)")
        log_ensemble(tool="afsample2", dataset=run_output.name, ensemble=eid,
                     seq_len=len(sequence), structures=0,
                     wall_seconds=time.time() - ensemble_t0, status="skip")
        return 0

    # Stage MSA in AF2 layout
    make_af2_msa_layout(eid, sequence, a3m_text, eid_dir / "msas")

    # Runs per ensemble — each produces ~STRUCTURES_PER_RUN
    runs_needed = (deficit + STRUCTURES_PER_RUN - 1) // STRUCTURES_PER_RUN
    n_fracs = len(msa_fractions)
    output_subpath = str(run_output.relative_to(project_root) / "output_pdbs")

    # Native mode: write a reduced_dbs flagfile once per ensemble
    native_flagfile = None
    if native:
        native_flagfile = eid_dir / ".reduced_dbs.flag"
        make_reduced_dbs_flagfile(db_path, native_flagfile)

    generated = 0
    for i in range(runs_needed):
        current = count_existing_pdbs(eid_dir)
        if current >= num_samples:
            break
        frac = msa_fractions[i % n_fracs]

        # Snapshot existing PDBs before this run so we can rename only the new ones
        pre = {p.name for p in eid_dir.glob("*.pdb")}

        if native:
            cmd = build_native_cmd(
                python_bin=python_bin, afsample2_dir=afsample2_dir,
                flagfile=native_flagfile,
                fasta_path=fastas_dir / f"{eid}.fasta",
                output_dir=run_output / "output_pdbs",
                msa_rand=frac,
                use_precomputed_features=(i > 0),
                use_precomputed_msas=True,
            )
        else:
            cmd = build_apptainer_cmd(
                sif=sif, db_path=db_path, project_root=project_root,
                dataset_dir=str(fastas_dir.parent.relative_to(project_root)),
                fasta_name=f"{eid}.fasta",
                output_subpath=output_subpath,
                msa_rand=frac,
                # First run: compute features, subsequent runs: reuse features.pkl
                use_precomputed_features=(i > 0),
                use_precomputed_msas=True,
            )

        log(f"  {eid} run {i+1}/{runs_needed} (rand={frac}, "
            f"existing={current}/{num_samples})")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            label = "native" if native else "apptainer"
            log(f"    {label} FAILED (rc={r.returncode}): {r.stderr[-500:]}")
            break

        # Copy features.pkl → msa_features.pkl after first run (matches slurm)
        if i == 0:
            fp = eid_dir / "features.pkl"
            mp = eid_dir / "msa_features.pkl"
            if fp.exists() and not mp.exists():
                shutil.copy2(fp, mp)

        # Flatten: move <eid>/afsample2/*.pdb up to <eid>/ (if container wrote
        # to the nested path) + tag new files with the next-available `_r{N}`
        # suffix so repeat (model, pred_idx, rand_frac) tuples across re-runs
        # don't silently overwrite each other.
        def _next_free_r(stem: str, suffix: str) -> Path:
            n = i
            while (eid_dir / f"{stem}_r{n}{suffix}").exists():
                n += 1
            return eid_dir / f"{stem}_r{n}{suffix}"

        nested = eid_dir / "afsample2"
        if nested.exists():
            for p in nested.glob("*.pdb"):
                dst = eid_dir / p.name
                if dst.exists():
                    dst = _next_free_r(p.stem, p.suffix)
                p.rename(dst)
            # Clean up the nested dir
            try: shutil.rmtree(nested)
            except Exception: pass

        new_files = [p for p in eid_dir.glob("*.pdb") if p.name not in pre]
        for p in new_files:
            # Tag untagged new files with the next free _r{N} suffix.
            if f"_r{i}" not in p.name:
                dst = _next_free_r(p.stem, p.suffix)
                p.rename(dst)

        generated += len(new_files)
        log(f"    +{len(new_files)} structures in {time.time()-t0:.0f}s "
            f"(total: {count_existing_pdbs(eid_dir)})")

    final = count_existing_pdbs(eid_dir)
    log(f"  {eid}: done, {final} total ({generated} new)")
    # Status: ok if we generated; "noop" if nothing produced despite needing more
    # (typically means the apptainer/native call failed and the runs loop broke).
    status = "ok" if generated > 0 else ("noop" if final < num_samples else "skip")
    log_ensemble(tool="afsample2", dataset=run_output.name, ensemble=eid,
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
    p.add_argument("run_id", help="Run ID for outputs (e.g. DANCE_BENCH)")
    p.add_argument("--msa-dir", type=Path, required=True,
                   help="Precomputed MSA directory (colabfold-style a3m)")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help=f"Target samples per ensemble (default: {DEFAULT_NUM_SAMPLES})")
    p.add_argument("--msa-fractions", type=float, nargs="+",
                   default=DEFAULT_MSA_FRACTIONS,
                   help="MSA randomization fractions to rotate over "
                        f"(default: {DEFAULT_MSA_FRACTIONS})")
    p.add_argument("--sif", type=Path,
                   default=Path(os.environ.get("SIF", str(Path.home() / "afsample2_v1.1.sif"))),
                   help="Path to afsample2 Apptainer SIF")
    p.add_argument("--db-path", type=Path,
                   default=Path(os.environ.get("DB_PATH", "/data/mchaourab/alphafold3-data")),
                   help="AlphaFold database path (must contain params/, bfd/, uniref30/)")
    p.add_argument("--output-base", type=Path,
                   default=PROJECT_ROOT / "data" / "outputs")
    p.add_argument("--resume", action="store_true",
                   help="Skip ensembles already at --num-samples")
    p.add_argument("--gpu-shard", default=None, metavar="I/N")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--native", action="store_true",
                   help="Use native AFsample2 invocation (no apptainer/SIF). "
                        "Requires --python-bin and --afsample2-dir.")
    p.add_argument("--python-bin", type=Path,
                   default=Path.home() / "afsample2_rtx5090_venv" / "bin" / "python",
                   help="Python interpreter for native mode (default: afsample2_rtx5090_venv)")
    p.add_argument("--afsample2-dir", type=Path,
                   default=Path.home() / "AFsample2",
                   help="AFsample2 repo path for native mode (default: ~/AFsample2)")
    args = p.parse_args()

    # Validate critical paths up front
    if args.native:
        if not args.python_bin.exists():
            print(f"Error: --python-bin not found: {args.python_bin}", file=sys.stderr)
            sys.exit(1)
        run_script = args.afsample2_dir / "AF_multitemplate" / "run_afsample2.py"
        if not run_script.exists():
            print(f"Error: AFsample2 run script not found: {run_script}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.sif.exists():
            print(f"Error: SIF not found: {args.sif}", file=sys.stderr)
            print("  Pull with: apptainer pull docker://kyogesh/afsample2:v1.1", file=sys.stderr)
            sys.exit(1)
    if not args.db_path.exists():
        print(f"Error: DB_PATH not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve all paths so relative_to() works regardless of how the user
    # passed them (relative vs absolute; on ACCRE /data/oliver_lab/... and
    # /panfs/accrepfs.vampire/data/oliver_lab/... are the same dir but the
    # latter is what .resolve() picks up via the symlink).
    args.input_path = args.input_path.resolve()
    args.output_base = args.output_base.resolve()
    args.db_path = args.db_path.resolve()
    args.sif = args.sif.resolve()

    input_csv = next((args.input_path / n for n in ("input_dataset_filtered.csv",
                                                    "input_dataset.csv")
                      if (args.input_path / n).exists()), None)
    if input_csv is None:
        print(f"Error: no input CSV in {args.input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_input_csv(input_csv)
    print(f"Loaded {len(rows)} ensembles from {input_csv}")

    if args.gpu_shard:
        i, n = map(int, args.gpu_shard.split("/"))
        rows = [r for idx, r in enumerate(rows) if idx % n == i]
        print(f"GPU shard {i}/{n}: processing {len(rows)} ensembles")

    # Sort by length — shorter first for faster early feedback
    rows.sort(key=lambda r: len(r["ref_sequence"]))
    if args.limit:
        rows = rows[: args.limit]

    run_output = args.output_base / "afsample2" / args.run_id
    run_output.mkdir(parents=True, exist_ok=True)
    fastas_dir = args.input_path / "fastas"

    log_fh = open(args.log_file, "a") if args.log_file else None
    def log(msg=""):
        print(msg); sys.stdout.flush()
        if log_fh: log_fh.write(msg + "\n"); log_fh.flush()

    log(f"=== AFsample2 on {args.run_id} ===")
    log(f"SIF:            {args.sif}")
    log(f"DB:             {args.db_path}")
    log(f"MSA dir:        {args.msa_dir}")
    log(f"Output:         {run_output}")
    log(f"Num samples:    {args.num_samples}")
    log(f"MSA fractions:  {args.msa_fractions}")
    log(f"Ensembles:      {len(rows)}")
    log("")

    t_start = time.time()
    done = skipped_no_msa = 0
    for idx, row in enumerate(rows):
        eid, seq = row["ensemble_ID"], row["ref_sequence"]
        a3m = load_a3m_text(args.msa_dir, eid)
        if a3m is None:
            log(f"  [{idx+1}/{len(rows)}] {eid}: SKIP (no MSA)")
            skipped_no_msa += 1
            continue

        log(f"[{idx+1}/{len(rows)}] {eid} (len={len(seq)})")
        try:
            run_ensemble(
                eid=eid, sequence=seq, a3m_text=a3m,
                run_output=run_output, fastas_dir=fastas_dir,
                sif=args.sif, db_path=args.db_path,
                project_root=PROJECT_ROOT,
                num_samples=args.num_samples,
                msa_fractions=args.msa_fractions,
                log=log,
                native=args.native,
                python_bin=args.python_bin,
                afsample2_dir=args.afsample2_dir,
            )
            done += 1
        except KeyboardInterrupt:
            log("Interrupted.")
            break
        except Exception as e:
            log(f"  ERROR: {e}")

    elapsed = time.time() - t_start
    log(f"\nTotal: {done}/{len(rows)} done, {skipped_no_msa} skipped (no MSA), "
        f"{elapsed/60:.1f} min")
    if log_fh: log_fh.close()


if __name__ == "__main__":
    main()
