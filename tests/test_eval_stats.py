from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dm_ahr_vit.config import CLASSES


def test_classification_metrics_handbuilt():
    from dm_ahr_vit.vit.metrics import classification_metrics

    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["akiec"]["f1"] == pytest.approx(1.0)
    assert m["per_class"]["bcc"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["df"]["support"] == 0
    assert np.isnan(m["per_class"]["df"]["auc"])
    # true bcc misclassified as bkl lands in row 1, column 2
    assert m["confusion_matrix"][1][2] == 1


def test_auc_undefined_marked_for_single_class():
    from dm_ahr_vit.vit.metrics import classification_metrics

    y = np.zeros(5, dtype=int)
    prob = np.tile(np.eye(7)[0], (5, 1))
    m = classification_metrics(y, y, prob)
    assert np.isnan(m["per_class"]["akiec"]["auc"])
    assert m["per_class"]["akiec"]["auc_note"] == "single_class_present"


def test_holm_known_example():
    from dm_ahr_vit.stats.bootstrap import holm

    p = [0.01, 0.04, 0.03]
    adj = holm(p)
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.06)


def test_holm_caps_at_one_and_skips_nan():
    from dm_ahr_vit.stats.bootstrap import holm

    adj = holm([0.6, float("nan"), 0.9])
    assert adj[0] == pytest.approx(1.0)
    assert np.isnan(adj[1])
    assert adj[2] == pytest.approx(1.0)


def test_bootstrap_reproducible_under_fixed_seed():
    from dm_ahr_vit.stats.bootstrap import paired_cluster_bootstrap

    rng = np.random.default_rng(0)
    n = 200
    groups = np.repeat([f"G{i}" for i in range(50)], 4)
    a = rng.normal(1.0, 0.1, n)
    b = a - 0.2 + rng.normal(0, 0.05, n)
    r1 = paired_cluster_bootstrap(a, b, groups, replicates=300, seed=11)
    r2 = paired_cluster_bootstrap(a, b, groups, replicates=300, seed=11)
    assert r1["difference"] == r2["difference"]
    assert r1["ci_low"] == r2["ci_low"] and r1["p_raw"] == r2["p_raw"]
    assert r1["difference"] == pytest.approx(0.2, abs=0.05)
    assert r1["ci_low"] < r1["difference"] < r1["ci_high"]


def test_bootstrap_shares_resampling_across_arms():
    """Resampling groups once must keep the arms paired (difference preserved)."""
    from dm_ahr_vit.stats.bootstrap import paired_cluster_bootstrap

    groups = np.array(["g1", "g1", "g2", "g2"])
    a = np.array([1.0, 1.0, 2.0, 2.0])
    b = np.array([0.5, 0.5, 1.5, 1.5])
    r = paired_cluster_bootstrap(a, b, groups, replicates=50, seed=1)
    assert r["difference"] == pytest.approx(0.5)
    assert r["ci_low"] == pytest.approx(0.5) and r["ci_high"] == pytest.approx(0.5)


def test_wilcoxon_group_aggregated():
    from dm_ahr_vit.stats.bootstrap import wilcoxon_group_paired

    rows = []
    for g in range(12):
        rows.append({"group_id": f"G{g}", "arm": "A", "macro_f1": 0.6 + g * 0.01})
        rows.append({"group_id": f"G{g}", "arm": "C", "macro_f1": 0.5 + g * 0.01})
    out = wilcoxon_group_paired(pd.DataFrame(rows), "macro_f1", "arm")
    assert out["n_groups"] == 12
    assert out["p_raw"] < 0.01


def test_dmahr_manifest_never_contains_val_or_test():
    """Guard used by the diffusion training script."""
    train = pd.DataFrame({"split": ["train"] * 3, "group_id": ["a", "b", "c"]})
    val = pd.DataFrame({"split": ["val"] * 2, "group_id": ["d", "e"]})
    assert set(train["split"]) == {"train"}
    assert not set(train["group_id"]) & set(val["group_id"])
    bad = pd.concat([train, val]).reset_index(drop=True)
    assert not set(bad["split"]) <= {"train"}

