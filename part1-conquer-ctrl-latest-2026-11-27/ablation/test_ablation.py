"""核对分组、时间折、训练侧镜像和配对比较。"""
import unittest
from unittest.mock import patch

import numpy as np

import run_ablation as a


class AblationChecks(unittest.TestCase):
    def test_all_features_belong_to_exactly_one_group(self):
        columns = a.make_features(a.load_data()).columns
        groups = a.feature_groups(columns)
        assigned = [col for group in groups.values() for col in group]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(columns))
        self.assertEqual([len(cols) for cols in groups.values()], [5, 1, 35, 20, 10])

    def test_folds_are_forward_disjoint_and_inside_training_period(self):
        rows = a.load_data()
        rows = rows.loc[rows.day < a.TEST_START].reset_index(drop=True)
        held_indices = []
        for train, held in a.forward_folds(rows):
            self.assertLess(rows.day.iloc[train].max(), rows.day.iloc[held].min())
            self.assertLess(rows.day.iloc[held].max(), a.TEST_START)
            self.assertFalse(set(rows.matchID.iloc[train]) & set(rows.matchID.iloc[held]))
            held_indices.extend(held)
        self.assertEqual(len(held_indices), len(set(held_indices)))

    def test_mirroring_only_reaches_the_fit_side(self):
        class Recorder:
            def fit(self, x, y):
                self.x, self.y = x, y

            def predict_proba(self, x):
                return np.tile([0.4, 0.6], (len(x), 1))

        model = Recorder()
        x, y, held = np.array([[1., 2.], [3., 4.]]), np.array([0, 1]), np.array([[99., 100.]])
        with patch.object(a, "make_pipeline", return_value=model):
            p = a.fit_predict(x, y, held)
        np.testing.assert_array_equal(model.x, np.vstack([x, -x]))
        np.testing.assert_array_equal(model.y, np.r_[y, 1-y])
        self.assertEqual(len(p), len(held))

    def test_identical_predictions_have_zero_paired_interval(self):
        p = np.array([0.2, 0.7, 0.4, 0.9])
        lo, hi = a.paired_day_interval(np.array([0, 1, 1, 1]), p, p, [1, 1, 2, 3])
        self.assertEqual((lo, hi), (0, 0))


if __name__ == "__main__":
    unittest.main()
