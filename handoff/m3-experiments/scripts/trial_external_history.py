"""Single independent external-history trial. No external weights, BO, ranks or target match result."""
import argparse, sqlite3, json, hashlib, joblib
from itertools import groupby
from collections import defaultdict
from datetime import date,timedelta
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.retune_m2_clean import setup,inputs,train_model,probabilities,chronological_folds,ROOT
from scripts.retrain_m2_clean import compare
from scripts.training_protocol import prediction_table
from scripts.audit_reference_aliases import key

OUT=ROOT/'reports/m2_clean_v2/roster_map_trial'; DB=ROOT/'references/csgopred/data/csgopred.db'
def external(rows):
    con=sqlite3.connect(DB.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    until=(rows.date.max().date()-timedelta(days=7)).isoformat(); raw=con.execute("select start_date,team_a,team_b,score_a,score_b from matches where source='hltv' and start_date<=? order by start_date,match_id",(until,)).fetchall();con.close(); timeline=[]
    for day,it in groupby(raw,key=lambda r:r[0]):
        pair=defaultdict(set)
        for _,a,b,sa,sb in it:
            a,b=key(a),key(b)
            if not a or not b or a==b or sa==sb:continue
            pair[tuple(sorted((a,b)))].add(a if sa>sb else b)
        timeline.append((date.fromisoformat(day),[(a,b,next(iter(ws))) for (a,b),ws in pair.items() if len(ws)==1]))
    # Use our existing Elo scale, K and half-life; aggregate updates within each external date.
    state={};pos=0;records=[]
    def rating(t,d):
        if t not in state:return 1500.
        v,last=state[t];return 1500.+(v-1500.)*2**(-(d-last).days/365)
    for day,g in rows.groupby(rows.date.dt.date,sort=True):
        while pos<len(timeline) and timeline[pos][0]<=day-timedelta(days=7):
            d,items=timeline[pos];pos+=1;changes=defaultdict(float)
            for a,b,w in items:
                delta=16*((w==a)-1/(1+10**((rating(b,d)-rating(a,d))/400)));changes[a]+=delta;changes[b]-=delta
            for t,v in changes.items():state[t]=(rating(t,d)+v,d)
        for r in g.itertuples():
            a,b=key(r.team1),key(r.team2);ok=a in state and b in state and a!=b;records.append((r.match_id,ok,rating(a,day)-rating(b,day) if ok else 0.,rating(a,day) if a in state and a!=b else np.nan,rating(b,day) if b in state and a!=b else np.nan))
    return pd.DataFrame(records,columns=['match_id','covered','external_diff','external_a','external_b']).set_index('match_id').loc[rows.match_id].reset_index()

def main(stage):
    train,x=inputs('train',*setup()); rows=train; valid=stage=='val'
    if valid:
        assert json.loads((OUT/'external_train.json').read_text())['passes'];v,vx=inputs('val',*setup());rows=pd.concat([train,v],ignore_index=True);x=pd.concat([x,vx],ignore_index=True)
    feats=external(rows);xx=x.assign(external_diff=feats.external_diff.to_numpy()); tables=[];holds=[]
    folds=[(np.arange(len(train)),np.arange(len(train),len(rows)))] if valid else chronological_folds(train)
    for f,(i,j) in enumerate(folds,1):
        h=rows.iloc[j];m=train_model(xx.iloc[i],rows.iloc[i],'linear',.001);m['temperature']=1.15;t=prediction_table(h,probabilities(m,xx.iloc[j]),'external_history');t['covered']=feats.covered.iloc[j].to_numpy();t['fold']=f;tables.append(t);holds.append(h)
    if valid:
        # Separate candidate for reproduction; do not overwrite the accepted M2 model.
        path=ROOT/'models/m2/m2_external_history_trial.joblib'; m.update(history_lag_days=7,source='csgopred.db / hltv',database_sha256=hashlib.sha256(DB.read_bytes()).hexdigest()); joblib.dump(m,path,compress=3); np.testing.assert_allclose(probabilities(joblib.load(path),xx.iloc[j]),tables[-1].p_team1.to_numpy(),rtol=0,atol=1e-12)
    tab=pd.concat(tables,ignore_index=True); held=pd.concat(holds,ignore_index=True); ref=pd.read_csv(ROOT/'reports/m2_clean_v2/retune'/('m2_clean_retuned_val.csv' if valid else 'train_oof.csv'))
    result=compare(held,tab[['match_id','y_true','p_team1','model']],ref); result['covered']=int(tab.covered.sum());result['passes']=bool(result['new']['acc']>result['old']['acc'] and result['new']['logloss']<result['old']['logloss']); print(result);tab.to_csv(OUT/f'external_{stage}.csv',index=False);(OUT/f'external_{stage}.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','val']);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.stage)


