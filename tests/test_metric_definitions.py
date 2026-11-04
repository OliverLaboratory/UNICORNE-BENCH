"""
Metric definitions on hand-built GT x prediction TM matrices (no TM-align).

  dr   Recall@0.9: fraction of GT conformations whose best prediction has TM >= 0.9
  cr   full recovery: 1 if every GT is recovered, else 0 (FRR is its mean)
  tm_bar   mean over GTs of the best prediction's TM (TM-bar)
  emd  optimal-transport cost between uniform GT and prediction distributions
"""
from __future__ import annotations

import numpy as np
import pytest

from compute_metrics import EMD_FLOOR, metrics_for_ensemble


def metrics(M, n_bootstrap=20):
    return metrics_for_ensemble(np.asarray(M, dtype=float), np.random.default_rng(0), n_bootstrap=n_bootstrap)


def test_recall_full_recovery_and_mean_best_tm():
    M = [[0.95, 0.40, 0.70],   # GT 0: recovered (0.95)
         [0.50, 0.89, 0.60],   # GT 1: best 0.89, missed
         [0.30, 0.20, 0.91]]   # GT 2: recovered (0.91)
    m = metrics(M)
    assert m["dr"] == pytest.approx(2 / 3)
    assert m["cr"] == 0.0
    assert m["tm_bar"] == pytest.approx((0.95 + 0.89 + 0.91) / 3)


def test_full_recovery_needs_every_gt():
    assert metrics([[0.90, 0.1], [0.2, 0.97]])["cr"] == 1.0
    assert metrics([[0.90, 0.1], [0.2, 0.8999]])["cr"] == 0.0


def test_threshold_is_inclusive():
    assert metrics([[0.9]])["dr"] == 1.0
    assert metrics([[0.89999]])["dr"] == 0.0


def test_recall_uses_best_prediction_per_gt_not_per_prediction():
    # One prediction matches both GTs well; recall counts GTs, so 2/2.
    m = metrics([[0.95, 0.1], [0.93, 0.2]])
    assert m["dr"] == 1.0
    assert m["precision_at_tm_0.9"] == 0.5


def test_failed_pairs_count_as_unrecovered():
    m = metrics([[np.nan, 0.92], [np.nan, np.nan]])
    assert m["dr"] == 0.5
    assert m["tm_bar"] == pytest.approx(0.92 / 2)


def test_recall_at_k_equals_recall_when_k_covers_all_predictions():
    M = np.array([[0.95, 0.4, 0.7], [0.5, 0.89, 0.6]])
    m = metrics(M)
    assert m["recall_09_at_k_5_mean"] == pytest.approx(m["dr"])
    assert m["recall_09_at_k_5_std"] == 0.0


def test_emd():
    pytest.importorskip("ot")
    # Perfect one-to-one match: zero cost.
    assert metrics([[1.0, 0.0], [0.0, 1.0]])["emd"] == pytest.approx(0.0, abs=1e-12)
    # Everything at or below the floor: maximal cost 1.
    assert metrics([[EMD_FLOOR, 0.1], [0.2, 0.0]])["emd"] == pytest.approx(1.0)
    # One GT, two predictions at TM 1.0 and 0.65: half the mass pays
    # 1 - (0.65 - 0.3) / 0.7 = 0.5.
    assert metrics([[1.0, 0.65]])["emd"] == pytest.approx(0.25)
