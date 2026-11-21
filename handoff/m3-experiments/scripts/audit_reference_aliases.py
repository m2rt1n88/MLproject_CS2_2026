"""Outcome-blind audit of extra monthly VRS name matches, using earlier-day lineups only."""
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/m2_clean_v2/reference_review'
def norm(s): return ' '.join(str(s).casefold().split())
def key(s): return re.sub(r'[^\w]', '', re.sub(r'\s+(esports|gaming)$', '', norm(s)))

def main():
    oof=pd.read_csv(ROOT/'reports/m2_clean_v2/valve_monthly/train_oof.csv',usecols=['match_id','snapshot','covered']).set_index('match_id')
    cols=['match_id','date','team1','team2','split','roster_cross']+[f'team{t}_player{i}_id' for t in (1,2) for i in range(1,6)]
    matches=pd.read_csv(ROOT/'data/processed/clean_v2/matches.csv',usecols=cols,parse_dates=['date']).query("split=='train'")
    pts=pd.read_csv(ROOT/'data/reference/valve_global_monthly/points.csv',keep_default_na=False); indices={}
    for date,g in pts.groupby('date'):
        lut=defaultdict(list); exact=defaultdict(list)
        for r in g.itertuples(): lut[key(r.team)].append(r); exact[norm(r.team)].append(r)
        indices[date]=(exact,lut)
    ids={}; dictionary=pd.read_csv(ROOT/'data/raw/players.csv').dropna(subset=['player_name']); names={int(r.player_id):norm(re.split(r'[(,]',r.player_name)[0]) for r in dictionary.itertuples()}; freq=Counter(names.values()); ids={i:n for i,n in names.items() if n and freq[n]==1}
    history={}; evidence=[]
    for day,g in matches.sort_values(['date','match_id']).groupby(matches.date.dt.normalize()):
        for r in g.itertuples():
            if r.match_id not in oof.index:continue
            row=oof.loc[r.match_id]
            if row.covered or pd.isna(row.snapshot):continue
            exact,lut=indices[row.snapshot]; hits=[]
            for n in [r.team1,r.team2]:
                v=exact.get(norm(n)) or lut.get(key(n),[])
                if len(v)!=1:break
                v=v[0]; roster={norm(p) for p in v.roster.split(',')}; last=history.get(norm(n)); overlap=0
                if last and (day-last[0]).days<=30:
                    overlap=len(roster & {ids[i] for i in last[1] if i in ids})
                hits.append((n,v.team,overlap))
            if len(hits)==2 and hits[0][1]!=hits[1][1]:evidence.append((r.match_id,day.date(),row.snapshot,*hits[0],*hits[1],min(h[2] for h in hits)>=3))
        # Multiple distinct complete lineups for the same team on one UTC day are ambiguous.
        updates=defaultdict(set)
        for r in g.itertuples():
            if r.roster_cross:continue
            for t,n in [(1,r.team1),(2,r.team2)]:
                lineup=frozenset(int(x) for i in range(1,6) if pd.notna(x:=getattr(r,f'team{t}_player{i}_id')) and x>0)
                if len(lineup)==5:updates[norm(n)].add(lineup)
        for n,sets in updates.items():history[n]=(day,next(iter(sets))) if len(sets)==1 else None
    OUT.mkdir(parents=True,exist_ok=True); e=pd.DataFrame(evidence,columns='match_id day snapshot team1 source1 overlap1 team2 source2 overlap2 both_supported'.split());e.to_csv(OUT/'alias_candidates.csv',index=False); print(len(e),int(e.both_supported.sum()),len(oof),int(oof.covered.sum()));print(e[e.both_supported].head(2).to_string(index=False))
if __name__=='__main__':main()
