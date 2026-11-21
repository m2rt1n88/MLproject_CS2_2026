"""P1 features: only use completed earlier UTC days; preserve the M1 files."""
from collections import Counter, defaultdict, deque
import json

import numpy as np
import pandas as pd

from scripts.prepare_training import normalize_team
from scripts.training_protocol import ROOT, load_matches

ROSTER = ['last_overlap', 'lineup_run', 'lineup_age', 'prior_adr', 'prior_kast', 'players_with_stats']
MAPS = ['maps180', 'pool_size', 'map_best', 'map_mean', 'shared_map_rate']
DUELS = ['h2h_rate', 'h2h_games']
NAMES = ROSTER + MAPS + DUELS
PAIRS = [(f'team1_{n}', f'team2_{n}') for n in NAMES if n != 'h2h_games']
DIFF = [f'diff_{n}' for n in NAMES if n != 'h2h_games']
COLUMNS = [c for pair in PAIRS for c in pair] + DIFF + ['h2h_games']
M2_DATA = ROOT / 'data/processed/m2'


def player_events(match_ids):
    slots = [f'team{t}_player{i}' for t in (1, 2) for i in range(1, 6)]
    cols = ['match_id', 'map_name', 'game_id', 'is_total', 'score1_game', 'score2_game']
    raw = pd.read_csv(ROOT / 'data/raw/cs2_all_tiers_games.csv', usecols=cols +
                      [f'{s}_{stat}' for s in slots for stat in ('id', 'adr', 'kast')])
    raw = raw[raw.match_id.isin(match_ids) & ~raw.is_total].sort_values(['match_id', 'game_id'])
    raw = raw.drop_duplicates(['match_id', 'map_name']).dropna(subset=['map_name','score1_game','score2_game'])
    events = defaultdict(list)
    for row in raw.itertuples(index=False):
        for s in slots:
            pid, adr, kast = (getattr(row, f'{s}_{v}') for v in ('id','adr','kast'))
            if pd.notna(pid) and np.isfinite(adr) and 0 <= kast <= 100 and adr >= 0:
                events[row.match_id].append((int(pid), adr, kast))
    return events


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
                # Give each known player equal weight irrespective of available map count.
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
        # Only now do this day's lineups, player stats, maps and results become history.
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


def main():
    matches=load_matches().query("split in ['train','val']").copy()
    maps=pd.read_csv(ROOT/'data/processed/maps.csv',usecols=['match_id','map_name','team1_won_map'])
    ids=set(matches.match_id)
    extra=past_features(matches,maps[maps.match_id.isin(ids)],player_events(ids))
    M2_DATA.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        base=pd.read_csv(ROOT/f'data/processed/training/{split}.csv')
        out=base.merge(extra,on='match_id',how='left',validate='one_to_one')
        if out.h2h_games.isna().any(): raise ValueError('Missing P1 row')
        out.to_csv(M2_DATA/f'{split}.csv',index=False)
        print(split,len(out),'matches;',len(COLUMNS),'extra columns',flush=True)
    (M2_DATA/'schema.json').write_text(json.dumps({'version':'p0-v2+p1-v1','additional_columns':COLUMNS,
        'pairs':PAIRS,'differences':DIFF,'historical_policy':'strictly prior UTC days; proxy roster = last complete observed five',
        'map_window_days':180,'player_window_observed_maps':10},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
