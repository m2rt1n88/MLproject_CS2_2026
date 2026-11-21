"""Train-only retuning of clean M2; separate select and validate stages. No test data."""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from scripts.optimize_m2 import design, train_model, probabilities
from scripts.retrain_m2_clean import DATA, OLD, OUT as BASE, inputs, compare
from scripts.training_protocol import ROOT, chronological_folds, prediction_table, probability_metrics, validate_predictions

OUT=BASE/'retune'; MD=ROOT/'models/m2'; NAME='m2_clean_retuned'
CS=(.0003,.001,.003,.01,.03,.1,.3); MULT=(.7,.85,1.,1.15,1.3)

def setup():
    return json.loads((DATA/'manifest.json').read_text()), json.loads((OLD/'selection.json').read_text())['columns']

def restore_bo(rows,split):
    cols=['bo1','bo3','bo5','bo_missing']
    old=pd.read_csv(ROOT/f'data/processed/m2/{split}.csv',usecols=['match_id',*cols]).set_index('match_id')
    assert old.index.is_unique and set(old.index)==set(rows.match_id)
    r=rows.copy(); r[cols]=old.loc[r.match_id,cols].to_numpy(); return design(r)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def select():
    manifest,cols=setup(); rows,x=inputs('train',manifest,cols); folds=list(chronological_folds(rows)); h=pd.concat([rows.iloc[j] for _,j in folds],ignore_index=True)
    lengths=[len(j) for _,j in folds]; boundaries=np.cumsum([0,*lengths]); grid=[]; preds={}
    def cross(x,c):
        return np.concatenate([probabilities(train_model(x.iloc[i],rows.iloc[i],'linear',c),x.iloc[j]) for i,j in folds])
    for c in CS:
        raw=logit(cross(x,c))
        for m in MULT:
            t=prediction_table(h,expit(m*raw),NAME); preds[c,m]=t
            grid.append(dict(C=c,mult=m,**probability_metrics(t),fold_loss=[probability_metrics(t.iloc[a:b])['logloss'] for a,b in zip(boundaries,boundaries[1:])]))
    ref=validate_predictions(pd.read_csv(BASE/'train_oof.csv'),h)
    np.testing.assert_allclose(ref.p_team1,preds[.003,.85].p_team1,atol=1e-12,rtol=0)
    best=min(grid,key=lambda g:g['logloss']); c,m=best['C'],best['mult']; t=preds[c,m]
    OUT.mkdir(exist_ok=True); MD.mkdir(exist_ok=True,parents=True)
    t.to_csv(OUT/'train_oof.csv',index=False)
    oldx=restore_bo(rows,'train'); bo=prediction_table(h,expit(.85*logit(cross(oldx,.003))),'audit_bo'); bo.to_csv(OUT/'audit_bo_oof.csv',index=False)
    hashes={}
    for n,xx,cc,mm in [(NAME,x,c,m),('audit_bo',oldx,.003,.85)]:
        b=train_model(xx,rows,'linear',cc); b.update(temperature=mm,dataset_version=manifest['version'],diagnostic_only=(n=='audit_bo'))
        p=MD/(n+'.joblib'); joblib.dump(b,p,compress=3);hashes[n]=digest(p)
    old=validate_predictions(pd.read_csv(OLD/'train_oof.csv'),h)
    detail=dict(selection='minimum pooled five-fold training logloss',best=best,candidates=grid,model_hashes=hashes,train_hash=manifest['files']['m2/train.csv'],
                tuned_vs_fixed=compare(h,t,ref),bo_off_vs_on=compare(h,ref,bo),clean_bo_on_vs_old=compare(h,bo,old))
    (OUT/'selection.json').write_text(json.dumps(detail,indent=2)+'\n',encoding='utf8')

def validate():
    locked=json.loads((OUT/'selection.json').read_text()); manifest,cols=setup();rows,x=inputs('val',manifest,cols); tables={}
    assert manifest['files']['m2/train.csv']==locked['train_hash']
    for n,xx in [(NAME,x),('audit_bo',restore_bo(rows,'val'))]:
        path=MD/(n+'.joblib'); assert digest(path)==locked['model_hashes'][n]
        b=joblib.load(path);tables[n]=prediction_table(rows,probabilities(b,xx),n);tables[n].to_csv(OUT/(n+'_val.csv'),index=False)
    fixed=pd.read_csv(BASE/'val_predictions.csv');old=pd.read_csv(ROOT/'reports/predictions/val/m2_schedule.csv');t=tables[NAME];bo=tables['audit_bo'];score=probability_metrics(t)
    result=dict(tuned_vs_fixed=compare(rows,t,fixed),tuned_vs_v9=compare(rows,t,old),bo_off_vs_on=compare(rows,fixed,bo),clean_bo_on_vs_old=compare(rows,bo,old),
                meets_both=score['acc']>=.63 and score['logloss']<=.635,test_evaluated=False)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('best',locked['best']['C'],locked['best']['mult']); print('oof',[(k,round(v['new']['logloss'],6),round(v['old']['logloss'],6)) for k,v in locked.items() if isinstance(v,dict) and 'new' in v]);
    for k,v in result.items():
        if isinstance(v,dict): print(k,round(v['new']['acc'],6),round(v['new']['logloss'],6),round(v['old']['acc'],6),round(v['old']['logloss'],6),v['fixed'],v['broken'],[round(a,6) for a in v['paired_logloss']['day_bootstrap_95']])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','validate']);stage=parser.parse_args().stage
    with threadpool_limits(limits=1): (select if stage=='select' else validate)()
