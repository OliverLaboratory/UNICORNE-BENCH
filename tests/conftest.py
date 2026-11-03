"""Shared fixtures: repository paths, the TM-align binary, and the stored
TM-align output for the fixture structures (tests/data/expected.json)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
sys.path.insert(0, str(REPO / "pipeline" / "03_compute_results"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_tmalign_raw import run_tmalign_one  # noqa: E402


def _find_tmalign() -> str | None:
    exe = os.environ.get("TMALIGN_EXE") or os.environ.get("TMALIGN") or "TMalign"
    return shutil.which(exe)


@pytest.fixture(scope="session")
def expected() -> dict:
    return json.loads((DATA / "expected.json").read_text())


@pytest.fixture(scope="session")
def tmalign() -> str:
    exe = _find_tmalign()
    if exe is None:
        pytest.skip("TM-align not found: run tool_scripts/install_tmalign.sh and set TMALIGN_EXE")
    return exe


def gt_path(eid: str, name: str) -> Path:
    return DATA / "structures" / eid / "gt" / name


def pred_path(eid: str, tool: str, name: str) -> Path:
    return DATA / "structures" / eid / "pred" / tool / name


def fixture_pairs(expected: dict):
    """Yield (eid, tool, gt_index, pred_index_in_fixture, gt_file, pred_file)."""
    for eid, ens in expected["ensembles"].items():
        for tool, t in ens["tools"].items():
            for gi, g in enumerate(ens["gt_files"]):
                for pj, p in enumerate(t["pred_files"]):
                    yield eid, tool, gi, pj, g, p


@pytest.fixture(scope="session")
def fresh(expected, tmalign) -> dict:
    """Run TM-align on every fixture pair once: {(eid, tool, gi, pj): record}."""
    out = {}
    for eid, tool, gi, pj, g, p in fixture_pairs(expected):
        rec = run_tmalign_one((str(gt_path(eid, g)), str(pred_path(eid, tool, p)), tmalign, 120))
        assert not rec.get("error"), (eid, tool, g, p, rec.get("error"))
        out[(eid, tool, gi, pj)] = rec
    return out


def tmalign_version(exe: str) -> str | None:
    text = subprocess.run([exe], capture_output=True, text=True).stdout
    m = re.search(r"TM-align \(Version (\d+)\)", text)
    return m.group(1) if m else None
