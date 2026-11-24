"""孪生 LSTM 的结构与可复现性检查：只用合成比赛，在 CPU 上小规模训练。"""

import unittest

import numpy as np
import torch

from scripts.basic_team_features import DIFFERENCES
from scripts.roster_and_map_features import DIFF
from scripts.sequence_features import build_sequences
from scripts.team_and_opponent_strength import EXTRA
from scripts.train_m1_m2 import design
from unittest import mock

import scripts.train_siamese_lstm as lstm
from scripts.match_splits_and_metrics import chronological_folds
from scripts.train_siamese_lstm import (CANDIDATES, LRS, NAME, Siamese, bundle_for, ensemble, final_choice,
    fit_side, gather, load_states, outer_oof, predict_with, scores, setup, swap)
from tests.test_sequence_features import fake_matches

SMALL = dict(key='t', label='测试', n_steps=6, hidden=8, max_epochs=4, patience=2, batch=32, es_gap_days=2)


def fake_rows(seed=0):
    m = fake_matches(n_days=40, per_day=3, seed=seed)
    rng = np.random.default_rng(seed)
    for c in DIFFERENCES + DIFF + EXTRA:
        m[c] = rng.normal(size=len(m))
    m['team1_prior_matches'] = rng.integers(0, 50, len(m))
    m['team2_prior_matches'] = rng.integers(0, 50, len(m))
    for flag in ('bo1', 'bo3', 'bo5'):
        m[flag] = 0
    for t in ('tier1', 'tier2', 'tier3'):
        m[t] = (m.tier == t).astype(int)
    m['split'] = 'train'
    m['is_swapped'] = False
    return m.sort_values(['date', 'match_id']).reset_index(drop=True)


class SiameseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup('cpu')
        cls.rows = fake_rows()
        cls.data = build_sequences(cls.rows, n_steps=6, k=32)
        cls.x = design(cls.rows).to_numpy(float)

    def test_swap_complementarity(self):
        torch.manual_seed(0)
        pos = np.arange(len(self.rows))
        for n_lin in (0, self.x.shape[1]):
            model = Siamese(8, n_lin).eval()
            inputs = gather(self.data, pos, self.x if n_lin else None)
            p = 1 / (1 + np.exp(-scores(model, inputs)))
            q = 1 / (1 + np.exp(-scores(model, swap(inputs))))
            np.testing.assert_allclose(p + q, 1.0, atol=1e-6)
            self.assertGreater(np.std(p), 0)        # 不是常数 0.5

    def test_seeded_fit_is_reproducible_and_complementary(self):
        for hybrid in (False, True):
            cfg = dict(SMALL, hybrid=hybrid, weighted=hybrid)
            a = fit_side(cfg, self.data, self.rows, self.x, 'cpu', seeds=(1, 2))
            b = fit_side(cfg, self.data, self.rows, self.x, 'cpu', seeds=(1, 2))
            self.assertEqual(a['epochs'], b['epochs'])
            ba, bb = bundle_for(cfg, a, list(design(self.rows)), 32), bundle_for(cfg, b, list(design(self.rows)), 32)
            pa, pb = predict_with(ba, self.data, self.rows), predict_with(bb, self.data, self.rows)
            np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-12)
            self.assertTrue(((pa > 0) & (pa < 1)).all())
            # 集成后的概率也必须互补
            models = load_states(ba['config'] | {'n_steps': 6}, ba['states'], self.x.shape[1] if hybrid else 0)
            inputs = gather(self.data, np.arange(len(self.rows)),
                            self.x / np.asarray(ba['x_scale']) if hybrid else None)
            np.testing.assert_allclose(ensemble(models, inputs) + ensemble(models, swap(inputs)), 1.0, atol=1e-6)

    def test_lr_selection_never_sees_outer_held_out_rows(self):
        folds = list(chronological_folds(self.rows, n_splits=2, gap_days=5))
        seen = []
        real = lstm.select_lr

        def spy(cfg, data, rows, *args, **kwargs):
            seen.append(set(rows.match_id))
            return real(cfg, data, rows, *args, **kwargs)

        cfg = dict(SMALL, hybrid=True, weighted=False)
        with mock.patch.object(lstm, 'select_lr', side_effect=spy):
            oof, fits = outer_oof(cfg, self.data, self.rows, self.x, list(design(self.rows)), folds, 'cpu',
                                  log=lambda s: None)
        self.assertEqual(len(seen), len(folds))
        for ids, (i, j) in zip(seen, folds):
            fit_ids, held_ids = set(self.rows.match_id.iloc[i]), set(self.rows.match_id.iloc[j])
            self.assertTrue(ids <= fit_ids)            # lr 选择只拿到拟合侧
            self.assertFalse(ids & held_ids)           # 外层留出折一场都没有
            # 拟合侧最晚比赛也早于留出折
            self.assertLess(self.rows.date[self.rows.match_id.isin(ids)].max(),
                            self.rows.date.iloc[j].min())
        for f in fits:
            self.assertIn(f['lr'], LRS)
            self.assertEqual(set(f['selection']['per_lr']), {str(v) for v in LRS})
        held = np.concatenate([j for _, j in folds])
        self.assertTrue(np.isfinite(oof[held]).all())

    def test_final_choice_rule(self):
        def fold(lr, e_fast, e_slow):
            return dict(lr=lr, selection=dict(per_lr={str(LRS[0]): dict(epochs=e_fast),
                                                      str(LRS[1]): dict(epochs=e_slow)}))
        fits = [fold(LRS[1], [4, 4, 4], [10, 12, 14]), fold(LRS[0], [5, 5, 5], [20, 20, 20]),
                fold(LRS[1], [6, 6, 6], [8, 8, 8])]
        self.assertEqual(final_choice(fits), (LRS[1], 12))
        # 平票取 LRS 靠前者
        self.assertEqual(final_choice(fits[:2])[0], LRS[0])

    def test_candidates_are_unique(self):
        keys = [c['key'] for c in CANDIDATES]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(4 <= len(keys) <= 6 and NAME == 'm3_siamese_lstm')


if __name__ == '__main__':
    unittest.main()
