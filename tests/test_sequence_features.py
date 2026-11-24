"""序列特征的防泄漏检查：只用合成比赛，不读真实数据。"""

import unittest

import numpy as np
import pandas as pd

from scripts.sequence_features import STEP_FEATURES, build_sequences

TEAMS = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']


def fake_matches(n_days=40, per_day=3, seed=0):
    """每天 per_day 场、同一天可能有同一支队打两场，用来检验同日不可见。"""
    rng = np.random.default_rng(seed)
    rows, mid = [], 1000
    for d in range(n_days):
        day = pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(d * 1.5))
        for k in range(per_day):
            a, b = rng.choice(len(TEAMS), 2, replace=False)
            won = int(rng.integers(0, 2))
            row = dict(match_id=mid, date=day + pd.Timedelta(hours=2 * k + 1), team1=TEAMS[a].upper(),
                       team2=TEAMS[b], tier=f'tier{rng.integers(1, 4)}', team1_won=won,
                       score1=2 if won else int(rng.integers(0, 2)), score2=int(rng.integers(0, 2)) if won else 2)
            for side, t in ((1, a), (2, b)):
                for i in range(1, 6):
                    row[f'team{side}_player{i}_id'] = np.nan if rng.random() < .1 else 100 * t + i + int(rng.integers(0, 2)) * 50
            rows.append(row)
            mid += 1
    return pd.DataFrame(rows)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.m = fake_matches()
        self.d = build_sequences(self.m, n_steps=6, k=32)

    def test_shapes_and_orientation(self):
        n = len(self.m)
        self.assertEqual(self.d['seq'].shape, (n, 2, 6, len(STEP_FEATURES)))
        self.assertTrue((self.d['team1'] == self.m.team1.str.lower().to_numpy()).all())
        self.assertTrue(np.isfinite(self.d['seq']).all())
        # 左对齐：前 length 格有效，其余为零
        for L, mk, s in zip(self.d['length'].ravel(), self.d['mask'].reshape(-1, 6), self.d['seq'].reshape(-1, 6, len(STEP_FEATURES))):
            self.assertTrue(mk[:L].all() and not mk[L:].any())
            self.assertTrue((s[L:] == 0).all())

    def test_history_strictly_earlier_day(self):
        gap = self.d['gap'][self.d['mask']]
        self.assertGreaterEqual(gap.min(), 1)
        # 逐步还原历史那场的比赛日，必须严格早于当前比赛日
        day = pd.to_datetime(self.d['day']).to_numpy()
        hist_day = day[:, None, None] - self.d['gap'].astype('timedelta64[D]')
        self.assertTrue((hist_day[self.d['mask']] < np.repeat(day, 12).reshape(-1, 2, 6)[self.d['mask']]).all())
        # 第一个比赛日没有任何历史
        first = day == day.min()
        self.assertFalse(self.d['mask'][first].any())

    def compare_until(self, changed, last_day, inclusive):
        day = pd.to_datetime(self.d['day'])
        keep = (day <= last_day) if inclusive else (day < last_day)
        for key in ('seq', 'mask', 'length', 'gap', 'team', 'ctx'):
            np.testing.assert_array_equal(self.d[key][keep], changed[key][keep], err_msg=key)
        return keep

    def test_future_and_same_day_results_do_not_leak(self):
        target = 60
        day = self.m.date.dt.normalize().iloc[target]
        m2 = self.m.copy()
        m2.loc[target, 'team1_won'] = 1 - m2.loc[target, 'team1_won']
        m2.loc[target, ['score1', 'score2']] = m2.loc[target, ['score2', 'score1']].to_numpy()
        changed = build_sequences(m2, n_steps=6, k=32)
        # 当天及之前（含同日其他比赛）不变；之后应当有变化，说明测试确实生效
        keep = self.compare_until(changed, day, inclusive=True)
        self.assertFalse(np.array_equal(self.d['seq'][~keep], changed['seq'][~keep]))

    def test_current_lineup_is_not_used(self):
        target = 60
        day = self.m.date.dt.normalize().iloc[target]
        m2 = self.m.copy()
        for i in range(1, 6):
            m2.loc[target, f'team1_player{i}_id'] = np.nan
        changed = build_sequences(m2, n_steps=6, k=32)
        self.compare_until(changed, day, inclusive=True)

    def test_row_order_does_not_matter(self):
        shuffled = self.m.sample(frac=1, random_state=3)
        again = build_sequences(shuffled, n_steps=6, k=32)
        np.testing.assert_array_equal(self.d['match_id'], again['match_id'])
        np.testing.assert_array_equal(self.d['seq'], again['seq'])


if __name__ == '__main__':
    unittest.main()
