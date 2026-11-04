"""
End to end on the fixture ensembles: structure files -> TM-align -> metrics.

The fixtures hold, for every tool, each GT's best prediction in the full
1,000-sample run. The row-wise max over that sub-matrix therefore equals the
row-wise max over the full matrix, and Recall@0.9 (dr), full recovery (cr) and
mean best TM (tm_bar) recomputed here must equal the published results.csv rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from compute_metrics import metrics_for_ensemble


def fresh_matrix(expected, fresh, eid, tool, field="tm_norm_chain1"):
    ens = expected["ensembles"][eid]
    n_gt, n_sel = len(ens["gt_files"]), len(ens["tools"][tool]["pred_files"])
    return np.array([[fresh[(eid, tool, gi, pj)][field] for pj in range(n_sel)] for gi in range(n_gt)])


def test_fixture_holds_each_gts_best_prediction(expected, fresh):
    for eid, ens in expected["ensembles"].items():
        for tool, t in ens["tools"].items():
            M = fresh_matrix(expected, fresh, eid, tool)
            assert list(M.argmax(axis=1)) == t["best_pred_per_gt"], (eid, tool)


@pytest.mark.parametrize("metric", ["dr", "cr", "tm_bar"])
def test_metrics_match_results_csv(expected, fresh, metric):
    rng = np.random.default_rng(0)
    for eid, ens in expected["ensembles"].items():
        for tool, t in ens["tools"].items():
            got = metrics_for_ensemble(fresh_matrix(expected, fresh, eid, tool), rng, n_bootstrap=1)[metric]
            assert got == pytest.approx(t["results_csv"][metric], abs=1e-9), (eid, tool, metric)


def test_fixtures_cover_both_outcomes(expected):
    """The fixtures include full recoveries, partial recall and misses, so the
    comparison above exercises every branch of the threshold."""
    rows = [t["results_csv"] for ens in expected["ensembles"].values() for t in ens["tools"].values()]
    assert any(r["cr"] == 1.0 for r in rows)
    assert any(0.0 < r["dr"] < 1.0 for r in rows)
    assert any(r["dr"] == 0.0 for r in rows)


def test_threshold_edge_cases(expected, fresh):
    """G1TPG3 (AFsample2): best TM 0.8996, not recovered. A6TF31 (AFsample2):
    best TM 0.9004, recovered."""
    below = fresh_matrix(expected, fresh, "G1TPG3", "afsample2").max(axis=1)
    above = fresh_matrix(expected, fresh, "A6TF31", "afsample2").max(axis=1)
    assert below.max() == pytest.approx(0.8996, abs=1e-4)
    assert any(abs(v - 0.9004) < 1e-4 for v in above)
    rng = np.random.default_rng(0)
    m_below = metrics_for_ensemble(fresh_matrix(expected, fresh, "G1TPG3", "afsample2"), rng, n_bootstrap=1)
    assert m_below["dr"] == 0.0
    m_above = metrics_for_ensemble(fresh_matrix(expected, fresh, "A6TF31", "afsample2"), rng, n_bootstrap=1)
    assert m_above["dr"] == 0.5
