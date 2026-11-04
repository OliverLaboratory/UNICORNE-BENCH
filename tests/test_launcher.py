"""
Prediction launcher: every tool config builds the command used for the paper
runs, and every runner it points at exists and imports the shared helpers.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys

import pytest

from conftest import REPO

STAGE = REPO / "pipeline" / "02_compute_predictions"
sys.path.insert(0, str(STAGE))


def _load_launch():
    spec = importlib.util.spec_from_file_location("launch", STAGE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launch = _load_launch()
DATA = ["data/inputs/UNICORNE_BENCH", "UNICORNE_BENCH"]
MSA = ["--msa-dir", "data/msas/UNICORNE_BENCH"]
RUNNERS = "pipeline/02_compute_predictions/runners"

# The commands used for the paper's UNICORNE-BENCH-1 runs (1,000 samples per ensemble).
PAPER_COMMANDS = {
    "bioemu": ["~/bioemu_venv/bin/python", f"{RUNNERS}/run_bioemu.py", *DATA, "--num-samples", "1000"],
    "esmflow": ["python", f"{RUNNERS}/run_esmflow.py", *DATA, "--num-samples", "1000"],
    "esmdiff": ["python", f"{RUNNERS}/run_esmdiff.py", *DATA, "--num-samples", "1000"],
    "alphaflow": ["~/alphaflow_venv/bin/python", "tool_scripts/alphaflow/run_dataset.py", *DATA, *MSA,
                  "--model", "af_pdb_distilled", "--output-base", "data/outputs", "--num-samples", "1000"],
    "cfrandom": ["~/localcolabfold/.pixi/envs/default/bin/python", f"{RUNNERS}/run_cfrandom.py", *DATA, *MSA,
                 "--num-samples", "1000"],
    "speach_af": ["~/localcolabfold/.pixi/envs/default/bin/python", f"{RUNNERS}/run_speach_af.py", *DATA, *MSA,
                  "--num-samples", "1000", "--seeds-per-set", "9", "--seeds-base", "3"],
    "afsample2": ["python", f"{RUNNERS}/run_afsample2.py", *DATA, *MSA, "--num-samples", "1000"],
    "afsample3": ["python", f"{RUNNERS}/run_afsample3.py", *DATA, *MSA, "--num-samples", "1000",
                  "--tool", "afsample3_biorxiv", "--msa-fractions", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5"],
}


def test_every_config_is_covered():
    assert {p.stem for p in (STAGE / "configs").glob("*.yaml")} == set(PAPER_COMMANDS)


@pytest.mark.parametrize("tool", PAPER_COMMANDS)
def test_config_builds_paper_command(tool):
    cmd = launch.build_command(launch.load_config(tool), "UNICORNE_BENCH")
    expected = PAPER_COMMANDS[tool]
    assert cmd == [os.path.expanduser(expected[0]), *expected[1:]]


@pytest.mark.parametrize("tool", PAPER_COMMANDS)
def test_runner_exists_and_compiles(tool):
    script = REPO / launch.load_config(tool)["script"]
    assert script.exists(), script
    compile(script.read_text(), str(script), "exec")


def test_overrides_flags_and_passthrough():
    cfg = launch.load_config("afsample3")
    cmd = launch.build_command(cfg, "OOD60", msa_dataset="OOD60_MSA",
                               overrides=launch.parse_overrides(["num-samples=250", "msa-fractions=[0.1]"]),
                               flags=["--resume", "--gpu-shard", "0/4"], extra=["--reverse"], python="py")
    assert cmd == ["py", f"{RUNNERS}/run_afsample3.py", "data/inputs/OOD60", "OOD60",
                   "--msa-dir", "data/msas/OOD60_MSA", "--num-samples", "250",
                   "--tool", "afsample3_biorxiv", "--msa-fractions", "0.1",
                   "--resume", "--gpu-shard", "0/4", "--reverse"]


def test_boolean_params_become_flags():
    assert launch.param_args({"native": True, "skip": False, "num-samples": 5}) == ["--native", "--num-samples", "5"]


@pytest.mark.parametrize("runner", sorted((STAGE / "runners").glob("run_*.py")), ids=lambda p: p.name)
def test_runner_helpers_come_from_shared_module(runner):
    """Runners import their shared helpers instead of keeping private copies."""
    import prediction_helpers
    shared = {n for n in dir(prediction_helpers) if not n.startswith("_")}
    tree = ast.parse(runner.read_text())
    imported = {a.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "prediction_helpers"
                for a in node.names}
    assert imported, f"{runner.name} does not use prediction_helpers"
    assert imported <= shared
    local = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert not (local & {"load_input_csv", "report_gpu", "count_existing_pdbs"})


def test_load_input_csv_reads_benchmark():
    import prediction_helpers
    rows = prediction_helpers.load_input_csv(REPO / "data/inputs/UNICORNE_BENCH/input_dataset.csv")
    assert len(rows) == 869
    assert set(rows[0]) == {"ensemble_ID", "ref_sequence"}
