"""融合层检查：互补性、前向拟合掩码、凸权重约束，以及 apply 与已写出的 val 预测一致。"""

import json
import unittest

import numpy as np
import pandas as pd

import inspect
from unittest import mock

from scripts import stack_m3
from scripts.stack_m3 import (FILES, GAP_DAYS, KINDS, LOCK, REPORT, ROOT, apply, apply_recipe,
                              combine, fit_convex, fit_spec, fit_weights, forward_fit_mask,
                              load_selection_context)


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(float)
    a = 1 / (1 + np.exp(-(0.7 * latent + rng.normal(scale=0.5, size=n))))
    b = 1 / (1 + np.exp(-(0.5 * latent + rng.normal(scale=0.8, size=n))))
    return y, {'m2': a, 'x': b}


class MetaLearnerTests(unittest.TestCase):
    def setUp(self):
        self.y, self.probs = synthetic()
        self.all = np.ones(len(self.y), dtype=bool)

    def test_every_meta_is_complementary_under_swap(self):
        for kind in KINDS:
            spec = dict(name=kind, kind=kind, models=['m2', 'x'])
            recipe = fit_spec(spec, self.probs, self.y, self.all)
            p = apply_recipe(recipe, self.probs)
            flipped = apply_recipe(recipe, {m: 1 - v for m, v in self.probs.items()})
            np.testing.assert_allclose(p + flipped, 1.0, atol=1e-10, err_msg=kind)
            self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_scaled_m2_control_is_complementary(self):
        recipe = fit_spec(dict(name='s', kind='logit', models=['m2']), self.probs, self.y, self.all)
        p = apply_recipe(recipe, self.probs)
        flipped = apply_recipe(recipe, {'m2': 1 - self.probs['m2']})
        np.testing.assert_allclose(p + flipped, 1.0, atol=1e-10)

    def test_convex_weights_sum_to_one_and_are_nonnegative(self):
        p = np.column_stack([self.probs['m2'], self.probs['x'], 1 - self.probs['x']])
        w = fit_convex(p, self.y)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertTrue((w >= 0).all())

    def test_logit_weights_are_nonnegative_and_mean_is_equal(self):
        p = np.column_stack([self.probs['m2'], 1 - self.probs['x']])  # 反向的模型应被压到 0 附近
        self.assertTrue((np.asarray(fit_weights('logit', p, self.y)) >= 0).all())
        self.assertEqual(fit_weights('mean', p, self.y), [0.5, 0.5])

    def test_combine_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            combine('nope', [1.0], np.ones((3, 1)) / 2)


class ForwardMaskTests(unittest.TestCase):
    def test_fit_rows_are_earlier_folds_and_a_gap_before_the_scored_fold(self):
        dates = pd.Series(pd.date_range('2024-01-01', periods=500, freq='12h'))
        fold = np.repeat(np.arange(1, 6), 100)
        for k in range(2, 6):
            mask = forward_fit_mask(dates, fold, k)
            start = dates[fold == k].min().normalize()
            self.assertFalse(mask[fold >= k].any())
            self.assertTrue(mask.any())
            self.assertTrue((dates[mask] < start - pd.Timedelta(GAP_DAYS, unit='D')).all())
            # 间隔内的更早折行必须被排除
            gap_rows = (fold < k) & (dates >= start - pd.Timedelta(GAP_DAYS, unit='D')).to_numpy()
            self.assertFalse(mask[gap_rows].any())

    def test_gap_rows_exist_and_are_excluded(self):
        dates = pd.Series(pd.date_range('2024-01-01', periods=300, freq='D'))
        fold = np.repeat(np.arange(1, 4), 100)
        mask = forward_fit_mask(dates, fold, 2)
        start = dates[fold == 2].min()
        self.assertEqual(int(mask.sum()), 70)          # 第 1 折 100 天里只有最早 70 天够远
        self.assertTrue((dates[mask] <= start - pd.Timedelta(31, unit='D')).all())


class SelectionNeverTouchesValTests(unittest.TestCase):
    def test_selection_context_requests_no_val_data(self):
        models = [m for m in ('m2', 'm1', 'mlp')
                  if all((ROOT / FILES[m].format(split=s)).is_file() for s in ('oof',))]
        if 'm2' not in models:
            self.skipTest('release predictions missing')
        read_csv, partition = pd.read_csv, stack_m3.partition
        seen_paths, seen_splits = [], []

        def spy_read(path, *a, **k):
            seen_paths.append(str(path))
            return read_csv(path, *a, **k)

        def spy_partition(split):
            seen_splits.append(split)
            return partition(split)

        with mock.patch.object(stack_m3.pd, 'read_csv', spy_read), \
                mock.patch.object(stack_m3, 'partition', spy_partition):
            ctx = load_selection_context(models)
        self.assertEqual(seen_splits, ['train'])
        self.assertTrue(seen_paths)
        self.assertFalse([p for p in seen_paths if 'val' in p.replace('\\', '/').split('/')[-1]], seen_paths)
        self.assertNotIn('val', ctx)
        self.assertNotIn('val_p', ctx)

    def test_select_source_does_not_call_validation_loader(self):
        source = inspect.getsource(stack_m3.select)
        self.assertNotIn('load_validation_context', source)
        self.assertNotIn("partition('val')", source)


@unittest.skipUnless(LOCK.is_file(), 'run select first')
class LockedPrimaryComplementarityTests(unittest.TestCase):
    def test_real_primary_is_complementary_on_oof_probabilities(self):
        lock = json.loads(LOCK.read_text(encoding='utf8'))
        ctx = load_selection_context(lock['primary']['models'])
        probs = ctx['oof']
        p = apply(lock, probs)
        flipped = apply(lock, {m: 1 - v for m, v in probs.items()})
        self.assertEqual(len(p), 6185)
        np.testing.assert_allclose(p + flipped, 1.0, rtol=0, atol=1e-10)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


@unittest.skipUnless(LOCK.is_file() and (REPORT / 'stacking_val.csv').is_file(), 'run select and validate first')
class ApplyReproducesValTests(unittest.TestCase):
    def test_apply_matches_stacking_val_csv(self):
        lock = json.loads(LOCK.read_text(encoding='utf8'))
        saved = pd.read_csv(REPORT / 'stacking_val.csv')
        probs = {}
        for m in lock['models']:
            t = pd.read_csv(ROOT / FILES[m].format(split='val')).set_index('match_id')
            probs[m] = t.loc[saved.match_id].p_team1.to_numpy(float)
        np.testing.assert_allclose(apply(lock, probs), saved.p_team1.to_numpy(float), rtol=0, atol=1e-12)
        self.assertTrue(saved.model.eq('m3_stack').all())

    def test_locked_primary_weights_are_valid(self):
        primary = json.loads(LOCK.read_text(encoding='utf8'))['primary']
        w = np.asarray(primary['weights'])
        self.assertTrue((w >= 0).all())
        if primary['kind'] in ('mean', 'convex'):
            self.assertAlmostEqual(float(w.sum()), 1.0, places=9)


if __name__ == '__main__':
    unittest.main()
