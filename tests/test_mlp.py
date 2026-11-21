"""MLP 的接口与对称性检查：不读真实数据，用合成特征走一遍 fit/predict。"""

import unittest

import numpy as np
import pandas as pd

from scripts.basic_team_features import DIFFERENCES
from scripts.roster_and_map_features import DIFF
from scripts.team_and_opponent_strength import EXTRA
from scripts.train_m1_m2 import design
from scripts.train_mlp import CANDIDATES, NAME, fit, predict, proba

SMALL = dict(key='test', label='测试用', hidden_layer_sizes=(8,), alpha=0.1, weighted=True)


def fake_rows(n=160, seed=0):
    rng = np.random.default_rng(seed)
    columns = DIFFERENCES + DIFF + EXTRA
    rows = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    rows["team1_prior_matches"] = rng.integers(0, 400, n)
    rows["team2_prior_matches"] = rng.integers(0, 400, n)
    for flag in ("bo1", "bo3", "bo5", "tier1", "tier2", "tier3", "bo_missing"):
        rows[flag] = 0
    rows["match_id"] = np.arange(n)
    rows["team1_won"] = rng.integers(0, 2, n)
    rows["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    rows["split"] = "train"
    rows["is_swapped"] = False
    return rows


class DesignTests(unittest.TestCase):
    def test_design_is_41_columns_and_flips_sign(self):
        rows = fake_rows()
        x = design(rows)
        self.assertEqual(len(x.columns), 41)
        self.assertTrue(np.isfinite(x.to_numpy(float)).all())
        # 交换两队：差值列变号，原始计数列互换（design 用它们算 log_experience）
        swap = rows.assign(**{c: -rows[c] for c in DIFFERENCES + DIFF + EXTRA})
        swap["team1_prior_matches"], swap["team2_prior_matches"] = (
            rows.team2_prior_matches.to_numpy(), rows.team1_prior_matches.to_numpy())
        flipped = design(swap)
        np.testing.assert_allclose(x.to_numpy(float), -flipped.to_numpy(float), atol=1e-12)


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.rows = fake_rows()
        x = design(self.rows)
        self.columns = list(x)
        self.x = x.to_numpy(float)
        self.y = self.rows.team1_won.to_numpy()
        self.w = np.linspace(1.0, 0.5, len(self.y))
        self.bundle = fit(self.x, self.y, self.w, self.columns, SMALL)

    def test_bundle_contract(self):
        self.assertEqual(self.bundle["stage"], "m3_mlp")
        self.assertEqual(self.bundle["columns"], self.columns)
        self.assertTrue(self.bundle["train_sha256"])

    def test_probabilities_are_valid_and_swapping_complements(self):
        p = predict(self.bundle, self.rows)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        # 交换两队：特征变号，概率必须互补
        mirrored = proba(self.bundle, -self.x)
        np.testing.assert_allclose(p + mirrored, 1.0, atol=1e-12)

    def test_fit_is_deterministic(self):
        again = fit(self.x, self.y, self.w, self.columns, SMALL)
        np.testing.assert_allclose(predict(self.bundle, self.rows), predict(again, self.rows), atol=1e-12)

    def test_unweighted_candidate_runs(self):
        cfg = dict(SMALL, weighted=False)
        b = fit(self.x, self.y, self.w, self.columns, cfg)
        p = predict(b, self.rows)
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_candidate_names_are_stable(self):
        keys = [c["key"] for c in CANDIDATES]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(NAME)


if __name__ == "__main__":
    unittest.main()
