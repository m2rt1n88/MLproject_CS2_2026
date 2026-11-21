"""Strictly past schedule-adjusted rating features for the M2 optimization."""
from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr

from scripts.prepare_training import normalize_team

STRENGTH = ['graph90','graph270','team_player','player_team','roster_strength',
            'typical_tier','elite_share','sweep_margin','round_margin','opponent_strength','opponent_tier']
EXTRA = ['diff_'+c for c in STRENGTH]


def five(row, side):
    ids = [getattr(row,f'team{side}_player{i}_id') for i in range(1,6)]
    return tuple(sorted(map(int,ids))) if all(pd.notna(i) for i in ids) and len(set(ids))==5 else ()


def rating_fit(history, day, half_life, roster=False):
    """Ridge-adjust historical margins for schedule strength. Unknown coefficients are zero."""
    if not history: return {}
    ii=[]; jj=[]; vv=[]; target=[]; index={}; w=[]
    for n,(d,a,b,fa,fb,margin) in enumerate(history):
        weight=2**(-(day-d).days/half_life)
        entries=[(('t',a),1.),(('t',b),-1.)]
        if roster:
            for players,sign in ((fa,1.),(fb,-1.)):
                entries.extend((('p',pid),sign/3.) for pid in players)
        for key,v in entries:
            jj.append(index.setdefault(key,len(index)));ii.append(n);vv.append(v)
        target.append(margin);w.append(np.sqrt(weight))
    mat=csr_matrix((vv,(ii,jj)),shape=(len(history),len(index)))
    beta=lsqr(diags(w)@mat,np.asarray(w)*target,damp=np.sqrt(2.5),atol=1e-6,btol=1e-6,iter_lim=100)[0]
    return {k:beta[i] for k,i in index.items()}


def build(matches,maps):
    totals=maps.groupby('match_id')[['score1','score2']].sum(); rd={}
    for match_id,one,two in totals.itertuples():
        if one+two>0: rd[match_id]=(one-two)/(one+two)
    history=[]; recent=defaultdict(deque); last_five={}; next_fit=None; rows=[]
    ordered=matches.sort_values(['date','match_id']); tier_code={'tier1':1.,'tier2':2.,'tier3':3.}
    for dt,group in ordered.groupby(ordered.date.dt.normalize()):
        day=dt.date(); today=list(group.itertuples(index=False)); cutoff=day-timedelta(days=180)
        if next_fit is None or day>=next_fit:
            fast=rating_fit(history,day,90); slow=rating_fit(history,day,270); joint=rating_fit(history,day,180,True)
            next_fit=day+timedelta(days=7)
        def team_features(a):
            past=recent[a]; roster=last_five.get(a,()); player=sum(joint.get(('p',i),0.) for i in roster)/3.
            team=joint.get(('t',a),0.)
            while past and past[0][0]<cutoff: past.popleft()
            if past:
                days,tiers,margins,rounds,opps,opptiers=zip(*past); w=np.array([2**(-(day-d).days/60) for d in days]);
                def mean(values,prior):
                    v=np.asarray(values,dtype=float); ok=np.isfinite(v); return (np.dot(w[ok],v[ok])+3*prior)/(w[ok].sum()+3)
                form=[mean(tiers,2),mean(np.array(tiers)==1,.4),mean(margins,0),mean(rounds,0),mean(opps,0),mean(opptiers,2)]
            else: form=[2.,.4,0.,0.,0.,2.]
            return np.array([fast.get(('t',a),0.),slow.get(('t',a),0.),team+player,team,player,*form])
        cache={a:team_features(a) for a in sorted({normalize_team(t) for r in today for t in (r.team1,r.team2)})}
        for r in today:
            a,b=normalize_team(r.team1),normalize_team(r.team2)
            rows.append([r.match_id,*(cache[a]-cache[b])]); won=2*int(r.team1_won)-1
            series=(r.score1-r.score2)/max(r.score1+r.score2,1); round_margin=rd.get(r.match_id,np.nan)
            margin=.45*won+.35*series+.20*(np.clip(round_margin*3,-1,1) if np.isfinite(round_margin) else series)
            fa,fb=five(r,1),five(r,2)
            history.append((day,a,b,fa,fb,margin)); tier=tier_code[r.tier]
            for team,opp,sign,ids in [(a,b,1,fa),(b,a,-1,fb)]:
                recent[team].append((day,tier,sign*series,sign*round_margin,cache[opp][1],cache[opp][5]))
                if ids: last_five[team]=ids
    return pd.DataFrame(rows,columns=['match_id',*EXTRA])
