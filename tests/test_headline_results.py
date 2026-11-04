"""
The published Table 5 numbers (Recall@0.9, FRR, TM-bar, EMD) from the shipped
per-ensemble results, and their consistency with the per-GT best-prediction
scores (no TM-align needed).
"""
from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest

from conftest import REPO

DATASET = "UNICORNE_BENCH"

# Paper Table 5: N, Recall@0.9 % (mean, SD), FRR %, TM-bar (mean, SD), EMD (mean, SD).
TABLE4 = {
    "bioemu":            (823, 30.1, 29.2, 6.1, 0.814, 0.099, 0.384, 0.170),
    "esmflow_pdb":       (829, 23.3, 27.1, 3.4, 0.775, 0.121, 0.382, 0.192),
    "esmdiff":           (829, 22.7, 27.0, 3.3, 0.780, 0.118, 0.426, 0.209),
    "alphaflow_pdb":     (829, 27.9, 26.1, 2.4, 0.797, 0.108, 0.355, 0.178),
    "cfrandom":          (818, 30.9, 28.8, 5.9, 0.808, 0.104, 0.441, 0.165),
    "afsample2":         (828, 33.7, 30.0, 8.2, 0.818, 0.101, 0.340, 0.172),
    "afsample3_biorxiv": (828, 34.0, 30.0, 8.2, 0.819, 0.101, 0.342, 0.170),
    "speach_af":         (823, 31.3, 28.1, 5.2, 0.808, 0.104, 0.343, 0.169),
}


def _load_table_module():
    path = REPO / "pipeline" / "04_build_figures" / "tab_overall_performance.py"
    spec = importlib.util.spec_from_file_location("tab_overall_performance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def table():
    mod = _load_table_module()
    return {tool: mod.compute_metrics(DATASET, tool) for tool in TABLE4}


@pytest.fixture(scope="module")
def evaluated():
    """The 829 evaluated ensembles: filtered_ensembles_v2 minus the blacklist."""
    inputs = REPO / "data" / "inputs" / DATASET
    keep = {r["ensemble_ID"] for r in csv.DictReader(open(inputs / "filtered_ensembles_v2.csv"))}
    keep -= {r["uniprot"] for r in csv.DictReader(open(inputs / "fold_quality_blacklist.csv"))}
    return keep


def results_rows(tool):
    return {r["ensemble_ID"]: r for r in
            csv.DictReader(open(REPO / "results" / DATASET / tool / "results.csv"))}


def test_evaluation_set_has_829_ensembles(evaluated):
    assert len(evaluated) == 829


@pytest.mark.parametrize("tool", TABLE4)
def test_table4(table, tool):
    n, rec, rec_sd, frr, tm, tm_sd, emd, emd_sd = TABLE4[tool]
    m = table[tool]
    assert m["n"] == n
    assert round(m["dr"] * 100, 1) == rec
    assert round(m["dr_sd"] * 100, 1) == rec_sd
    assert round(m["cr"] * 100, 1) == frr
    assert round(m["tm_bar"], 3) == tm
    assert round(m["tm_bar_sd"], 3) == tm_sd
    assert round(m["emd"], 3) == emd
    assert round(m["emd_sd"], 3) == emd_sd


def test_headline_full_recovery_rate(table):
    """'Even the strongest tools fully recover only 8.2% of ensembles.'"""
    best = max(table[t]["cr"] for t in TABLE4)
    assert round(best * 100, 1) == 8.2
    for tool in ("afsample2", "afsample3_biorxiv"):
        n_full = sum(v == 1.0 for v in table[tool]["cr_eid"].values())
        assert (n_full, table[tool]["n"]) == (68, 828), tool


@pytest.mark.parametrize("tool", TABLE4)
def test_table4_only_uses_evaluated_ensembles(table, evaluated, tool):
    """Each tool is scored on the evaluated set; N < 829 only where the tool
    returned fewer than 250 valid predictions for an ensemble."""
    eids = set(table[tool]["dr_eid"])
    assert eids <= evaluated
    assert len(evaluated - eids) == 829 - TABLE4[tool][0]


@pytest.mark.parametrize("tool", TABLE4)
def test_results_match_per_gt_best_scores(evaluated, tool):
    """dr, cr and tm_bar in results.csv equal what the per-GT best-prediction TM
    (normalized by GT length) implies, for every evaluated ensemble."""
    best = {}
    with open(REPO / "results" / DATASET / tool / "per_gt_summary.csv") as fh:
        for r in csv.DictReader(fh):
            best.setdefault(r["ensemble_ID"], []).append(float(r["best_tm_chain1"]))
    rows = results_rows(tool)
    checked = 0
    for eid in evaluated & set(rows):
        tm = np.array(best[eid])
        r = rows[eid]
        assert len(tm) == int(r["n_gt"]), eid
        dr = float((tm >= 0.9).mean())
        assert float(r["dr"]) == pytest.approx(dr, abs=1e-12), eid
        assert float(r["cr"]) == (1.0 if dr == 1.0 else 0.0), eid
        assert float(r["tm_bar"]) == pytest.approx(tm.mean(), abs=1e-9), eid
        checked += 1
    assert checked == TABLE4[tool][0]
