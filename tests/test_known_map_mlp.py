"""单图 MLP 的输入视图检查：地图进入输入、交换两队后特征与概率都互补。"""

import unittest

import numpy as np

from known_map_v2.data_adapter import TARGET, load, swap
from scripts.train_known_map_mlp import CANDIDATES, features, fit_rows, predict_rows


class KnownMapInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = load()
        cls.small = cls.rows.iloc[:240].reset_index(drop=True)

    def test_map_is_an_input(self):
        for view, width in (('two', 2 + 7), ('diff', 71 + 7)):
            x = features(self.rows, view)
            self.assertEqual(x.shape, (len(self.rows), width))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue((x[:, -7:].sum(axis=1) == 1).all())

    def test_swap_flips_statistics_but_not_map(self):
        rows = self.small
        for view in ('two', 'diff'):
            x, y = features(rows, view), features(swap(rows), view)
            np.testing.assert_allclose(x[:, :-7], -y[:, :-7], atol=1e-12)
            np.testing.assert_array_equal(x[:, -7:], y[:, -7:])
            np.testing.assert_array_equal(1 - rows[TARGET].to_numpy(), swap(rows)[TARGET].to_numpy())

    def test_probabilities_are_valid_and_complementary(self):
        rows = self.small
        cfg = dict(CANDIDATES[0], hidden_layer_sizes=(4,))
        model = fit_rows(rows.iloc[:160], cfg)
        p = predict_rows(model, rows.iloc[160:], cfg['view'])
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        mirror = predict_rows(model, swap(rows.iloc[160:]), cfg['view'])
        np.testing.assert_allclose(p + mirror, 1.0, atol=1e-9)

    def test_candidate_views_are_declared(self):
        self.assertTrue({c['view'] for c in CANDIDATES} <= {'two', 'diff'})


if __name__ == '__main__':
    unittest.main()
