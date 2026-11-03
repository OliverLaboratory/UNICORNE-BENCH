"""
TM-align scoring: the C++ binary, the normalization convention, and agreement
with the stored GT x prediction TM-align output the benchmark metrics use.

Every score in the benchmark comes from `TMalign <GT> <prediction>` as run by
compute_tmalign_raw.run_tmalign_one, and the metrics read `tm_norm_chain1`:
the TM-score normalized by the length of chain 1, the ground truth.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from ca_reader import read_ca
from compute_tmalign_raw import run_tmalign_one
from conftest import fixture_pairs, gt_path, pred_path, tmalign_version

FIELDS = ["tm_norm_chain1", "tm_norm_chain2", "rmsd", "len_chain1", "len_chain2", "len_aligned"]


def d0(length: int) -> float:
    """TM-score distance scale (Zhang & Skolnick 2004), as in TM-align."""
    return max(1.24 * (length - 15) ** (1 / 3) - 1.8, 0.5) if length > 21 else 0.5


def test_tmalign_version_matches_stored_output(tmalign, expected):
    assert expected["tmalign_version"] == "20220412"
    assert tmalign_version(tmalign) == expected["tmalign_version"], (
        "stored scores come from TM-align 20220412; build third_party/TMalign/TMalign.cpp "
        "with tool_scripts/install_tmalign.sh")


def test_self_alignment_is_perfect(tmalign, expected):
    for eid, ens in expected["ensembles"].items():
        for g in ens["gt_files"]:
            rec = run_tmalign_one((str(gt_path(eid, g)), str(gt_path(eid, g)), tmalign, 120))
            assert rec["tm_norm_chain1"] == pytest.approx(1.0, abs=1e-5), (eid, g)
            assert rec["tm_norm_chain2"] == pytest.approx(1.0, abs=1e-5), (eid, g)
            assert rec["rmsd"] == pytest.approx(0.0, abs=1e-2), (eid, g)


def test_reproduces_stored_scores(expected, fresh):
    """Re-running TM-align on the shipped GT and prediction files gives the
    same TM-scores, RMSD and lengths as the stored full matrices."""
    n = 0
    for eid, tool, gi, pj, g, p in fixture_pairs(expected):
        stored = expected["ensembles"][eid]["tools"][tool]["pairs"]
        rec = fresh[(eid, tool, gi, pj)]
        for f in FIELDS:
            assert rec[f] == pytest.approx(stored[f][gi][pj], abs=1e-5), (eid, tool, g, p, f)
        n += 1
    assert n > 300


def test_chain1_is_the_ground_truth(expected, fresh):
    """Chain 1 is the GT: its length is the GT's residue count, and the
    reported chain-1 normalization uses that length and the matching d0."""
    rx = re.compile(r"TM-score=\s*([\d.]+)\s*\(if normalized by length of Chain_1, i\.e\., LN=(\d+), d0=([\d.]+)\)")
    for eid, tool, gi, pj, g, p in fixture_pairs(expected):
        rec = fresh[(eid, tool, gi, pj)]
        gt_xyz, gt_seq = read_ca(gt_path(eid, g))
        pred_xyz, pred_seq = read_ca(pred_path(eid, tool, p))
        assert rec["len_chain1"] == len(gt_xyz), (eid, g)
        assert rec["len_chain2"] == len(pred_xyz), (eid, tool, p)
        # Same residues TM-align read, in the same order.
        assert rec["align_chain1"].replace("-", "") == gt_seq, (eid, g)
        assert rec["align_chain2"].replace("-", "") == pred_seq, (eid, tool, p)
        m = rx.search(rec["raw"])
        assert m, "chain-1 normalization line missing from TM-align output"
        assert int(m.group(2)) == len(gt_xyz)
        assert float(m.group(3)) == pytest.approx(d0(len(gt_xyz)), abs=0.006)


def test_tm_score_recomputed_from_superposition(expected, fresh):
    """Recompute both TM-scores from the coordinates, TM-align's alignment and
    its rotation, with the TM-score formula normalized by each chain's length.

    TM-align's rotation maximizes the chain-2-normalized score, so that one
    matches to the printed precision. TM-align optimizes the GT-normalized
    score over its own superposition and leaves out very distant pairs, so
    that one agrees to within 0.002.
    """
    worst = 0.0
    for eid, tool, gi, pj, g, p in fixture_pairs(expected):
        rec = fresh[(eid, tool, gi, pj)]
        x, _ = read_ca(gt_path(eid, g))
        y, _ = read_ca(pred_path(eid, tool, p))
        u, t = np.array(rec["rotation"]["u"]), np.array(rec["rotation"]["t"])
        pairs = np.array(rec["mapping_all"])
        dist = np.linalg.norm((x @ u.T + t)[pairs[:, 0]] - y[pairs[:, 1]], axis=1)

        def tm(length):
            return float((1 / (1 + (dist / d0(length)) ** 2)).sum() / length)

        assert tm(len(y)) == pytest.approx(rec["tm_norm_chain2"], abs=1e-4), (eid, tool, g, p)
        gap = rec["tm_norm_chain1"] - tm(len(x))
        assert abs(gap) <= 2e-3, (eid, tool, g, p, gap)
        worst = max(worst, abs(gap))
    assert worst < 2e-3


def test_normalization_decides_recovery(expected, fresh):
    """P61769: the GT has 98 residues and the predictions 119. The best
    prediction recovers the GT under GT-length normalization (the benchmark's)
    but not under prediction-length normalization."""
    flips = [(tool, gi, pj) for (eid, tool, gi, pj), rec in fresh.items()
             if eid == "P61769" and rec["tm_norm_chain1"] >= 0.9 > rec["tm_norm_chain2"]]
    assert flips, "expected a pair recovered only under GT-length normalization"
    tool, gi, pj = flips[0]
    rec = fresh[("P61769", tool, gi, pj)]
    assert rec["len_chain1"] < rec["len_chain2"]


def test_tmalign_exe_wrapper_returns_chain1_first(expected, fresh, tmalign, monkeypatch):
    """conformation_benchmark.evaluation.tmalign_exe returns (TM by chain 1,
    TM by chain 2, RMSD), consistent with compute_tmalign_raw."""
    from conformation_benchmark.evaluation import tmalign_exe

    monkeypatch.setenv("TMALIGN_EXE", tmalign)
    for (eid, tool, gi, pj), rec in fresh.items():
        if eid != "P61769":
            continue
        ens = expected["ensembles"][eid]
        g = ens["gt_files"][gi]
        p = ens["tools"][tool]["pred_files"][pj]
        tm1, tm2, rmsd = tmalign_exe._run_tmalign(gt_path(eid, g), pred_path(eid, tool, p))
        assert tm1 == pytest.approx(rec["tm_norm_chain1"], abs=1e-4)
        assert tm2 == pytest.approx(rec["tm_norm_chain2"], abs=1e-4)
        assert rmsd == pytest.approx(rec["rmsd"], abs=1e-2)
