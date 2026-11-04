#!/usr/bin/env python3
"""
Run one prediction tool on a benchmark dataset, configured by a YAML file.

  python pipeline/02_compute_predictions/launch.py bioemu --dataset UNICORNE_BENCH
  python pipeline/02_compute_predictions/launch.py cfrandom --dataset UNICORNE_BENCH --gpu-shard 0/4 --resume
  python pipeline/02_compute_predictions/launch.py afsample3 --dataset OOD60 --set num-samples=250 --dry-run

The tool's config, configs/<tool>.yaml, names the interpreter of the tool's
environment, its runner script, whether the tool reads MSAs, and the
parameters used in the paper. The launcher turns it into the runner's command
line and runs it from the repository root:

  <python> <script> data/inputs/<DATASET> <DATASET>
           [--msa-dir data/msas/<MSA_DATASET>]  --<param> <value> ...
           [--resume] [--gpu-shard I/N] [--limit N] [--log-file F]  [args after --]

Every runner writes data/outputs/<tool>/<DATASET>/output_pdbs/<UniProt>/*.pdb.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REQUIRED_KEYS = ("tool", "python", "script", "msa", "params")


def load_config(tool: str) -> dict:
    """Config for a tool name (configs/<tool>.yaml) or a path to a YAML file."""
    path = Path(tool) if tool.endswith((".yaml", ".yml")) else CONFIG_DIR / f"{tool}.yaml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
        sys.exit(f"no config {path} (known tools: {known})")
    cfg = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"{path}: missing keys {missing}")
    return cfg


def parse_overrides(pairs: list[str]) -> dict:
    """--set key=value pairs; values are parsed as YAML (numbers, lists, booleans)."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            sys.exit(f"--set expects key=value, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def param_args(params: dict) -> list[str]:
    """{"num-samples": 1000, "msa-fractions": [0.0, 0.1], "native": true} ->
    ["--num-samples", "1000", "--msa-fractions", "0.0", "0.1", "--native"]."""
    args = []
    for key, value in params.items():
        if value is None or value is False:
            continue
        args.append(f"--{key}")
        if value is True:
            continue
        args.extend(str(v) for v in value) if isinstance(value, list) else args.append(str(value))
    return args


def build_command(cfg: dict, dataset: str, msa_dataset: str | None = None,
                  overrides: dict | None = None, flags: list[str] | None = None,
                  extra: list[str] | None = None, python: str | None = None) -> list[str]:
    """The runner's command line (paths relative to the repository root)."""
    cmd = [os.path.expanduser(python or cfg["python"]), cfg["script"],
           f"data/inputs/{dataset}", dataset]
    if cfg["msa"]:
        cmd += ["--msa-dir", f"data/msas/{msa_dataset or dataset}"]
    cmd += param_args({**cfg["params"], **(overrides or {})})
    return cmd + (flags or []) + (extra or [])


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:                      # everything after -- goes to the runner
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", help="Tool name (configs/<tool>.yaml) or path to a YAML config")
    ap.add_argument("--dataset", default="UNICORNE_BENCH", help="Dataset under data/inputs/")
    ap.add_argument("--msa-dataset", default=None,
                    help="MSA set under data/msas/ (default: same as --dataset)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config parameter, e.g. --set num-samples=250")
    ap.add_argument("--python", default=None, help="Interpreter to use instead of the config's")
    ap.add_argument("--resume", action="store_true", help="Skip ensembles that already have enough samples")
    ap.add_argument("--gpu-shard", default=None, metavar="I/N", help="Process ensembles i, i+N, ...")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N ensembles")
    ap.add_argument("--log-file", default=None, help="Also write the runner's log to this file")
    ap.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = ap.parse_args(argv)

    cfg = load_config(args.tool)
    if args.gpu_shard and not re.fullmatch(r"\d+/\d+", args.gpu_shard):
        sys.exit("--gpu-shard must be I/N, e.g. 0/4")
    flags = (["--resume"] if args.resume else []) \
        + (["--gpu-shard", args.gpu_shard] if args.gpu_shard else []) \
        + (["--limit", str(args.limit)] if args.limit is not None else []) \
        + (["--log-file", args.log_file] if args.log_file else [])
    cmd = build_command(cfg, args.dataset, args.msa_dataset, parse_overrides(args.overrides),
                        flags, extra, args.python)

    print(f"[{cfg['tool']}] {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return
    if not (PROJECT_ROOT / "data" / "inputs" / args.dataset).is_dir():
        sys.exit(f"missing data/inputs/{args.dataset}")
    if cfg["msa"] and not (PROJECT_ROOT / "data" / "msas" / (args.msa_dataset or args.dataset)).is_dir():
        sys.exit(f"missing data/msas/{args.msa_dataset or args.dataset}: generate MSAs first "
                 "(pipeline/02_compute_predictions/generate_msas_local.py)")
    sys.exit(subprocess.run(cmd, cwd=PROJECT_ROOT).returncode)


if __name__ == "__main__":
    main()
