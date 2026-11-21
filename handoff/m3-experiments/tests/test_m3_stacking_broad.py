import unittest

import numpy as np
import pandas as pd

from scripts.experiment_m3_stacking_broad import ALL_STACK, fit_metas, mirrored, predict_metas, screen
from scripts.prepare_m2 import DIFF, PAIRS
from scripts.prepare_training import DIFFERENCES, TEAM_PAIRS


class BroadStackingTests(unittest.TestCase):
    def test_mirror_is_involutive_for_family_inputs(self):
        data = {'team1_won': 1, 'tier1': 0, 'match_id': 42}
        for a, b in TEAM_PAIRS+PAIRS:
            data.update({a: 1, b: 2})
        for c in DIFFERENCES+DIFF:
            data[c] = -1
        rows = pd.DataFrame([data])
        swap = mirrored(rows)
        self.assertEqual(swap.at[0, 'diff_elo'], 1)
        self.assertEqual(swap.at[0, 'team1'], rows.at[0, 'team2'])
        pd.testing.assert_frame_equal(mirrored(swap), rows)

    def test_multimodel_stack_is_symmetric_and_uses_complementary_predictions(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        p = {'second': np.array([.5, .5, .9, .1, .75, .25]),
             'm2': np.array([.9, .1, .5, .5, .6, .4]),
             'reversed': np.array([.1, .9, .1, .9, .2, .8])}
        params = fit_metas(p, y)
        self.assertGreater(params['all_weights']['second'], 0.)
        self.assertEqual(params['all_weights']['reversed'], 0.)
        pred = predict_metas(p, params, p['m2'])
        swap = predict_metas({name: 1-x for name, x in p.items()}, params, 1-p['m2'])
        self.assertIn(ALL_STACK, pred)
        for name in pred:
            np.testing.assert_allclose(pred[name] + swap[name], 1.)

    def test_roundoff_and_ties_do_not_count_as_forward_fold_wins(self):
        rows = []
        for name, changes in [('stack__pass', [-.001, -.001, -.001, .001]),
                              ('mean__tie', [-.001, -.001, -1e-12, 0.])]:
            for group, delta in zip(['all', 'fold_2', 'fold_3', 'fold_4', 'fold_5'], [np.mean(changes), *changes]):
                rows.append(dict(model=name, group=group, logloss=.65+delta,
                                 delta_vs_forward_m2=delta, delta_vs_scaled_m2=delta))
        report = screen(pd.DataFrame(rows))
        self.assertTrue(report['stack__pass']['passed'])
        self.assertFalse(report['mean__tie']['passed'])
