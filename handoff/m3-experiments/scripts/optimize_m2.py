"""Select M2 refinements on train first. Validation is a separate, read-only-model stage."""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits

from scripts.m2_ratings import build, EXTRA
from scripts.prepare_training import normalize_team
from scripts.train_m1 import weights, accuracy_ci
from scripts.train_m2 import CONTEXT, paired_logloss, fit as baseline_fit, predict as baseline_predict, FULL
from scripts.training_protocol import (ROOT, chronological_folds, load_matches, prediction_table,
                                      probability_metrics, validate_predictions, csv_checksum)

NAME='m2_schedule'; REPORT=ROOT/'reports/m2_optimization'; MODEL=ROOT/'models/m2/m2_schedule.joblib'
PARAMS=[('linear',c) for c in (.003,.03,.3)]+[('trees',n) for n in (60,140)]
TEMPS=(.85,1.,1.15)

def read(split):
    raw=load_matches(); raw=raw[raw['split'].eq('train') if split=='train' else raw['split'].isin(['train','val'])]
    maps=pd.read_csv(ROOT/'data/processed/maps.csv',usecols=['match_id','score1','score2'])
    diffs=build(raw,maps[maps.match_id.isin(raw.match_id)])
    base=pd.read_csv(ROOT/f'data/processed/m2/{split}.csv',parse_dates=['date']); check=raw[raw.split.eq(split)].set_index('match_id').loc[base.match_id]
    if (len(base)!=int(raw.split.eq(split).sum()) or not base.match_id.is_unique or not base.split.eq(split).all() or
        base.is_swapped.any() or not np.array_equal(base.date,check.date) or not np.array_equal(base.team1_won,check.team1_won) or
        any(not np.array_equal(base[c],check[c].map(normalize_team)) for c in ('team1','team2'))):
        raise ValueError('Invalid original split rows')
    result=base.merge(diffs,on='match_id',how='inner',validate='one_to_one'); assert len(result)==len(base)
    return result


def design(rows):
    # Saturate rare, extreme workload gaps; all inputs and interactions are antisymmetric.
    x=rows[CONTEXT+EXTRA].copy(); x=x.fillna(0.)
    for field,scale in [('prior_matches',10),('rest_days',7),('lineup_age',10),('maps180',10),('lineup_run',3)]:
        col='diff_'+field;x[col]=np.arcsinh(x[col]/scale)
    x['log_experience']=np.log1p(rows.team1_prior_matches)-np.log1p(rows.team2_prior_matches)
    for col in ('diff_graph90','diff_team_player','diff_elo'):
        for context in ('bo1','bo5','tier1'):
            x[col+'_'+context]=x[col]*rows[context]
    return x


def train_model(x,rows,kind,param):
    both=np.vstack([x,-x]); y=rows.team1_won.to_numpy(); w=np.tile(weights(rows),2); scaler=None
    if kind=='linear':
        scaler=StandardScaler(with_mean=False).fit(both);both=scaler.transform(both)
        est=LogisticRegression(C=param,fit_intercept=False,max_iter=1200)
    else:
        est=LGBMClassifier(n_estimators=param,num_leaves=7,learning_rate=.04,min_child_samples=75,
                           reg_lambda=4.,n_jobs=2,verbosity=-1,random_state=2026,deterministic=True,force_col_wise=True)
    est.fit(both,np.r_[y,1-y],sample_weight=w)
    return {'model':est,'scaler':scaler,'kind':kind,'param':param,'temperature':1.,'columns':list(x.columns)}


def probabilities(bundle,x):
    if list(x.columns)!=bundle['columns']: raise ValueError('Incompatible feature schema')
    arr=x.to_numpy();s=bundle['scaler'];arr=s.transform(arr) if s is not None else arr
    model=bundle['model']
    if bundle['kind']=='linear': return expit(model.decision_function(arr)*bundle['temperature'])
    # Average the two orientations to enforce p(A,B)=1-p(B,A). LightGBM's native
    # predictor avoids sklearn feature-name warnings when using numeric arrays.
    pred=model.booster_.predict
    p=(pred(arr,num_threads=2)+1-pred(-arr,num_threads=2))/2
    return expit(logit(np.clip(p,1e-8,1-1e-8))*bundle['temperature'])


def select():
    REPORT.mkdir(parents=True,exist_ok=True);MODEL.parent.mkdir(parents=True,exist_ok=True)
    rows=read('train');x=design(rows);folds=list(chronological_folds(rows));held=pd.concat([rows.iloc[j] for _,j in folds]); records=[]; outs=[]
    prior_params={'num_leaves':5,'n_estimators':110}
    prior=np.concatenate([baseline_predict(baseline_fit(rows.iloc[i],'lightgbm',FULL,prior_params),rows.iloc[j]) for i,j in folds]);ref=probability_metrics(prediction_table(held,prior,'previous'))
    for kind,param in PARAMS:
        raw=np.concatenate([probabilities(train_model(x.iloc[i],rows.iloc[i],kind,param),x.iloc[j]) for i,j in folds]);
        for mult in TEMPS:
            p=expit(mult*logit(raw)); t=prediction_table(held,p,NAME); f=[]; off=0
            for _,j in folds:
                f.append(probability_metrics(t.iloc[off:off+len(j)])['logloss']);off+=len(j)
            records.append({'kind':kind,'param':param,'logit_multiplier':mult,**probability_metrics(t),'fold_logloss':f});outs.append(t)
    best=min(range(len(records)),key=lambda i:records[i]['logloss']); winner=records[best]
    b=train_model(x,rows,winner['kind'],winner['param']);b['temperature']=winner['logit_multiplier'];joblib.dump(b,MODEL,compress=3)
    outs[best].to_csv(REPORT/'train_oof.csv',index=False)
    detail={'selection':'minimum pooled train forward-fold logloss; no validation metrics used','training_rows':len(rows),
            'oof_rows':len(held),'previous_cv':ref,'candidates':records,'selected':winner,'columns':list(x.columns),
            'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'train_sha256':csv_checksum(ROOT/'data/processed/m2/train.csv')}
    (REPORT/'selection.json').write_text(json.dumps(detail,indent=2)+'\n',encoding='utf-8');print('train-only winner:',winner,'previous:',ref,flush=True)


def validate():
    record=json.loads((REPORT/'selection.json').read_text())
    if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=record['model_sha256']: raise ValueError('Model differs from frozen selection')
    b=joblib.load(MODEL);rows=read('val');x=design(rows);p=probabilities(b,x);t=prediction_table(rows,p,NAME)
    validate_predictions(t,rows);ref=validate_predictions(pd.read_csv(ROOT/'reports/predictions/val/lightgbm_p1_matched.csv'),rows)
    result={'selected':record['selected'],'previous':probability_metrics(ref),'validation':probability_metrics(t),
            'validation_rows':len(t),'correct':int(((p>.5)==rows.team1_won.to_numpy()).sum()),'acc_ci':accuracy_ci(t,rows.date),
            'paired_logloss':paired_logloss(p,ref.p_team1.to_numpy(),rows.team1_won.to_numpy(),rows.date),
            'test_evaluated':False}
    result['meets_both']=result['validation']['acc']>=.63 and result['validation']['logloss']<=.635
    t.to_csv(ROOT/f'reports/predictions/val/{NAME}.csv',index=False); (REPORT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','validate']);args=parser.parse_args()
    with threadpool_limits(limits=1): (select if args.stage=='select' else validate)()
