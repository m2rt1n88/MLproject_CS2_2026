"""Rebuild clean_v2 from frozen v9 sources. Never write to the v9 paths."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):  # Also support python scripts/build_tables.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.training_protocol import ROOT, EXPECTED_COUNTS, csv_checksum, split_matches, verify_snapshot
from scripts.prepare_training import build_features, normalize_team
from scripts.prepare_m2 import past_features

RAW = ROOT/'data/raw'
OUT = ROOT/'data/processed/clean_v2'
SLOTS = [[f'team{s}_player{i}_id' for i in range(1, 6)] for s in (1, 2)]
IDS = SLOTS[0] + SLOTS[1]


def roster_flags(left, right):
    a, b = ([p for p in side if pd.notna(p)] for side in (left, right))
    return len(set(a)) < len(a), len(set(b)) < len(b), bool(set(a) & set(b))


def classify(bo, s1, s2, n, w1):
    known = bo.isin([1, 3, 5]); need = (bo+1)/2
    maximum = np.maximum(s1, s2); minimum = np.minimum(s1, s2)
    valid = known & maximum.eq(need) & minimum.lt(need)
    conflict = known & ~valid
    verified = n.gt(0) & w1.eq(s1) & (n-w1).eq(s2)
    return conflict, verified, valid & verified


def clean(df, tiers):
    if not pd.api.types.is_bool_dtype(df.is_total) or df.is_total.isna().any():
        raise ValueError('Expected boolean row type')
    df=df.copy(); df['date']=pd.to_datetime(df.datetime,utc=True).dt.tz_localize(None)
    base_cols=['match_id','date','team1','team2','team1_id','team2_id','tournament','bestOf','score1_match','score2_match',*IDS]
    agg=df.loc[df.is_total,base_cols].copy()
    # Do not silently resolve duplicate aggregate records with different results.
    if (agg.groupby('match_id')[['score1_match','score2_match','bestOf']].nunique(dropna=False)>1).any().any():
        raise ValueError('Conflicting aggregate records')
    m=agg.drop_duplicates('match_id').rename(columns={'score1_match':'score1','score2_match':'score2','bestOf':'bo_reported'})
    m=m.loc[m.score1.notna() & m.score2.notna() & m.score1.ne(m.score2)].copy()
    if not ((m[['score1','score2']]>=0)&(m[['score1','score2']]%1==0)).all().all(): raise ValueError('Invalid series score')
    m['team1_won']=m.score1.gt(m.score2).astype(int); m['tier']=m.match_id.map(tiers)
    if m.tier.isna().any(): raise ValueError('Missing tier')
    flags=[roster_flags(row[:5],row[5:]) for row in m[IDS].itertuples(index=False,name=None)]
    m[['team1_repeat','team2_repeat','roster_cross']]=flags
    for side, cols in enumerate(SLOTS,1):
        m.loc[m[f'team{side}_repeat'] | m.roster_cross,cols]=np.nan
    m=split_matches(m)
    # Raw BO has no independently verified announcement timestamp. Keep it for audit;
    # give no row a pre-match indicator inferred from its outcome.
    m['bo']=np.nan
    r=df[~df.is_total & df.match_id.isin(m.match_id)].dropna(subset=['map_name','score1_game','score2_game']).copy()
    r=r[r.score1_game.ne(r.score2_game) & r.score1_game.ge(0) & r.score2_game.ge(0)].copy()
    r['map_name']=r.map_name.str.strip().str.casefold(); r=r.loc[r.map_name.ne('')]
    keys=['match_id','map_name']; scores=['score1_game','score2_game']
    ambiguous=r.groupby(keys)[scores].transform('nunique').gt(1).any(axis=1); rejected=r.loc[ambiguous,keys+scores].copy()
    r=r.loc[~ambiguous].sort_values(['match_id','game_id'],kind='stable').drop_duplicates(keys)
    teams=m.set_index('match_id');
    for c in ('team1','team2'):
        if not r[c].map(normalize_team).eq(r.match_id.map(teams[c]).map(normalize_team)).all(): raise ValueError('Map team orientation mismatch')
    r['team1_won_map']=r.score1_game.gt(r.score2_game).astype(int)
    maps=r[['match_id','date','team1','team2','tournament','map_name',*scores,'team1_won_map']].rename(columns=dict(zip(scores,['score1','score2'])))
    counts=maps.groupby('match_id').team1_won_map.agg(n_maps='size',map_wins_team1='sum'); m=m.join(counts,on='match_id')
    m[['n_maps','map_wins_team1']]=m[['n_maps','map_wins_team1']].fillna(0).astype(int)
    m['bo_score_conflict'],m['map_score_verified'],m['maps_complete']=classify(m.bo_reported,m.score1,m.score2,m.n_maps,m.map_wins_team1)
    m['ambiguous_map']=m.match_id.isin(rejected.match_id); m['maps_complete'] &= ~m.ambiguous_map
    return m,maps,r,rejected


def events_for_history(raw_maps, matches):
    bad={s:set(matches.loc[matches.roster_cross | matches[f'team{s}_repeat'],'match_id']) for s in (1,2)}
    out=defaultdict(list)
    for r in raw_maps.sort_values(['match_id','game_id']).itertuples(index=False):
        sides=[[getattr(r,c) for c in cols] for cols in SLOTS]; rep1,rep2,cross=roster_flags(*sides)
        for s,(slots,repeat) in enumerate(zip(SLOTS,[rep1,rep2]),1):
            if repeat or cross or r.match_id in bad[s]: continue
            for slot in slots:
                prefix=slot[:-3]; pid=getattr(r,slot); adr=getattr(r,prefix+'_adr'); kast=getattr(r,prefix+'_kast')
                if pd.notna(pid) and np.isfinite(adr) and adr>=0 and 0<=kast<=100: out[r.match_id].append((int(pid),adr,kast))
    return out


def main():
    from scripts.m2_ratings import build
    from threadpoolctl import threadpool_limits
    verify_snapshot(); parts=[]
    for tier in ('tier1','tier2','tier3'):
        ids=pd.read_csv(RAW/f'cs2_{tier}_games.csv',usecols=['match_id']).match_id.unique(); parts.append(pd.Series(tier,index=ids))
    tiers=pd.concat(parts,verify_integrity=True)
    m,maps,raw_maps,rejected=clean(pd.read_csv(RAW/'cs2_all_tiers_games.csv',low_memory=False),tiers)
    old=split_matches(pd.read_csv(ROOT/'data/processed/matches.csv'))
    pd.testing.assert_frame_equal(m[['match_id','date','team1','team2','team1_won','split']],old[['match_id','date','team1','team2','team1_won','split']],check_dtype=False)
    assert m.split.value_counts().to_dict()==EXPECTED_COUNTS
    m['was_incomplete']=~old.maps_complete.fillna(False).astype(bool)
    cols=['match_id','split','bo_reported','bo_score_conflict','n_maps','map_wins_team1','score1','score2','map_score_verified','maps_complete','team1_repeat','team2_repeat','roster_cross','ambiguous_map']
    quality=m[cols].copy(); quality['old_maps_complete']=old.maps_complete
    quality['recovered_complete']=m.maps_complete & m.was_incomplete; quality['lost_complete']=~m.maps_complete & ~m.was_incomplete
    m=m.drop(columns='was_incomplete'); m_hist=m[m.split.isin(['train','val'])].copy(); maps_hist=maps[maps.match_id.isin(m_hist.match_id)]
    print('Quality checks passed; rebuilding historical features from cleaned rows.',flush=True)
    p0=build_features(m_hist,maps_hist); p1=past_features(m_hist,maps_hist,events_for_history(raw_maps[raw_maps.match_id.isin(m_hist.match_id)],m_hist))
    with threadpool_limits(limits=1): extra=build(m_hist,maps_hist)
    features=p0.merge(p1,on='match_id',validate='one_to_one').merge(extra,on='match_id',validate='one_to_one')
    assert len(features)==len(m_hist) and features.bo_missing.eq(1).all() and not features.is_swapped.any()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'m2').mkdir(exist_ok=True)
    files={ 'matches.csv':m, 'maps.csv':maps,'match_quality.csv':quality,'rejected_map_rows.csv':rejected,
            **{f'm2/{s}.csv':features[features.split.eq(s)] for s in ('train','val')}}
    for name,frame in files.items(): frame.to_csv(OUT/name,index=False)
    summary={'version':'v9-clean-v2','source_manifest_sha256':csv_checksum(ROOT/'data/v9_manifest.json'),'counts':EXPECTED_COUNTS,
             'map_rows':len(maps),'rejected_ambiguous_map_rows':len(rejected),'bo_policy':'reported value audit only; unknown for all pre-match rows until a pre-match source is verified',
             'quality':quality.assign(roster_repeat=m.team1_repeat | m.team2_repeat).groupby('split')[['bo_score_conflict','roster_repeat','roster_cross','recovered_complete','lost_complete','ambiguous_map']].sum().to_dict(orient='index'),
             'files':{k:csv_checksum(OUT/k) for k in files}}
    (OUT/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8'); print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))


if __name__=='__main__': main()
