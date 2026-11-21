"""从历史阵容、选手表现、地图和交手记录中生成 M2 特征。"""
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from scripts.basic_team_features import normalize_team

ROSTER = ['last_overlap', 'lineup_run', 'lineup_age', 'prior_adr', 'prior_kast', 'players_with_stats']
MAPS = ['maps180', 'pool_size', 'map_best', 'map_mean', 'shared_map_rate']
DUELS = ['h2h_rate', 'h2h_games']
NAMES = ROSTER + MAPS + DUELS
PAIRS = [(f'team1_{n}', f'team2_{n}') for n in NAMES if n != 'h2h_games']
DIFF = [f'diff_{n}' for n in NAMES if n != 'h2h_games']
COLUMNS = [c for pair in PAIRS for c in pair] + DIFF + ['h2h_games']


def past_features(matches, maps, events):
    rosters, stats = {}, defaultdict(lambda: deque(maxlen=10))
    pool, duels = defaultdict(deque), defaultdict(Counter)
    maps_by_id = defaultdict(list)
    for r in maps.itertuples(index=False):
        maps_by_id[r.match_id].append((r.map_name, int(r.team1_won_map)))
    rows = []
    ordered = matches.sort_values(['date','match_id'])
    for day, group in ordered.groupby(ordered.date.dt.normalize()):
        cutoff = day - pd.Timedelta(days=180)
        today = list(group.itertuples(index=False))
        for r in today:
            a,b = normalize_team(r.team1), normalize_team(r.team2)
            record = {'match_id':r.match_id}
            versus = duels[tuple(sorted((a,b)))]
            record['h2h_games'] = sum(versus.values())
            values, pools = [], []
            for team in (a,b):
                past = pool[team]
                while past and past[0][0] < cutoff:
                    past.popleft()
                wins, counts = Counter(), Counter()
                for _, name, won in past:
                    wins[name] += won
                    counts[name] += 1
                rates = {m:(wins[m]+2)/(n+4) for m,n in counts.items()}
                pools.append((counts,rates))
                ids, overlap, run, last_day = rosters.get(team,((), np.nan, np.nan,None))
                # 每位有记录的选手权重相同，避免打图多的选手主导队伍均值。
                form = [np.mean(stats[i], axis=0) for i in sorted(ids) if stats[i]]
                values.append({'last_overlap':overlap, 'lineup_run':run,
                               'lineup_age':(day-last_day).days if last_day is not None else np.nan,
                               'prior_adr':np.mean([v[0] for v in form]) if form else np.nan,
                               'prior_kast':np.mean([v[1] for v in form]) if form else np.nan,
                               'players_with_stats':len(form), 'maps180':len(past), 'pool_size':len(rates),
                               'map_best':max(rates.values(),default=0.5),
                               'map_mean':np.mean(list(rates.values())) if rates else 0.5,
                               'h2h_rate':(versus[team]+2)/(record['h2h_games']+4)})
            common = sorted(pools[0][0].keys() & pools[1][0].keys())
            mass = [min(pools[0][0][m],pools[1][0][m]) for m in common]
            for side, v in enumerate(values):
                v['shared_map_rate'] = np.average([pools[side][1][m] for m in common],weights=mass) if common else 0.5
                for n,x in v.items(): record[f'team{side+1}_{n}'] = x
            for n in values[0]: record[f'diff_{n}'] = values[0][n]-values[1][n]
            rows.append(record)
        # 当天所有特征已生成；现在才把阵容、地图和赛果写入历史。
        for r in today:
            a,b = normalize_team(r.team1), normalize_team(r.team2)
            duels[tuple(sorted((a,b)))][a if r.team1_won else b] += 1
            for t, team in enumerate((a,b),1):
                ids = frozenset(getattr(r,f'team{t}_player{i}_id') for i in range(1,6)
                                if pd.notna(getattr(r,f'team{t}_player{i}_id')))
                if len(ids)==5:
                    prior = rosters.get(team)
                    rosters[team]=(ids,len(ids & prior[0]) if prior else np.nan,
                                   prior[2]+1 if prior and prior[0]==ids else 1, day)
                for name,won in maps_by_id.get(r.match_id,()):
                    pool[team].append((day,name,won if t==1 else 1-won))
            for pid, adr, kast in events.get(r.match_id,()): stats[pid].append((adr,kast))
    return pd.DataFrame(rows,columns=['match_id',*COLUMNS])
