import unittest

import numpy as np
import pandas as pd

from scripts.experiment_m3_stacking import combine, fit_meta, log_loss, meta_fit_mask


class StackingTests(unittest.TestCase):
    def test_meta_eligibility_respects_fold_and_calendar_gap(self):
        rows = pd.DataFrame({'date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-02-01', '2025-02-02'])})
        fit = meta_fit_mask(rows, np.array([1, 1, 2, 3]), 2)
        np.testing.assert_array_equal(fit, [True, False, False, False])

    def test_stack_uses_a_complementary_model(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        p = np.array([[.9, .5], [.1, .5], [.5, .9], [.5, .1], [.5, .75], [.5, .25]])
        params = fit_meta(p, y)
        result = combine(p, params, p[:, 0])
        self.assertGreater(params['logit_weights'][1], 0.)
        for name in ('weighted_mean', 'logit_stack'):
            self.assertLess(log_loss(result[name], y), log_loss(result['scaled_m2'], y))

    def test_stack_discards_reversed_signal_and_preserves_team_symmetry(self):
        y = np.array([1, 0, 0, 1, 1, 0])
        p = np.array([[.9, .1], [.1, .9], [.7, .95], [.3, .05], [.65, .05], [.35, .95]])
        params = fit_meta(p, y)
        self.assertEqual(params['logit_weights'][1], 0.)
        self.assertEqual(params['lgbm_probability_weight'], 0.)
        p = np.vstack([p, [.5, .5], [1., 0.]])
        result = combine(p, params, p[:, 0])
        mirror = combine(1-p, params, 1-p[:, 0])
        for name in result:
            np.testing.assert_allclose(result[name]+mirror[name], 1., atol=1e-10, rtol=0)
            self.assertEqual(result[name][-2], .5)
