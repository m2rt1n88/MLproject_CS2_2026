"""M2: forward CV; matched P0/P1 LightGBM, regularized Bradley-Terry and ablations."""
import json
import platform
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits

from scripts import prepare_m2 as p1
from scripts.prepare_training import FEATURE_COLUMNS as P0, TEAM_PAIRS, DIFFERENCES
from scripts.train_m1 import load_inputs, weights, accuracy_ci
from scripts.training_protocol import (ROOT, chronological_folds, augment_training, prediction_table,
    probability_metrics, validate_predictions, csv_checksum)

FULL = P0 + p1.COLUMNS
CONTEXT = DIFFERENCES + p1.DIFF
SEED = 2026
TREES = [{'num_leaves':leaves,'n_estimators':rounds} for leaves in (5,12) for rounds in (50,110,180)]
REG = [{'C':c} for c in (0.05,0.25,1.,4.)]


def bt_design(bundle,x):
    enc=bundle['teams']
    a=enc.transform(x[['team1']].to_numpy()); b=enc.transform(x[['team2']].to_numpy())
    matrix=a-b
    if bundle['columns']:
        z=bundle['scale'].transform(bundle['imputer'].transform(x[bundle['columns']]))
        matrix=sparse.hstack([matrix,z],format='csr')
    return matrix


def fit(rows,kind,columns,params):
    # Only original matches from the fitting partition may enter augmentation.
    aug=augment_training(rows,TEAM_PAIRS+p1.PAIRS,DIFFERENCES+p1.DIFF)
    w=weights(aug)
    bundle={'kind':kind,'columns':columns,'params':params,'feature_version':'p0-v2+p1-v1'}
    if kind=='lightgbm':
        est=lgb.LGBMClassifier(**params,learning_rate=0.035,min_child_samples=65,reg_lambda=3,
            colsample_bytree=1.,verbosity=-1,n_jobs=2,random_state=SEED,deterministic=True,force_col_wise=True)
        est.fit(aug[columns],aug.team1_won,sample_weight=w)
    else:
        bundle['teams']=OneHotEncoder(handle_unknown='ignore').fit(aug[['team1']].to_numpy())
        if columns:
            bundle['imputer']=SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True).fit(aug[columns])
            # Differences are signed; zero means no estimated advantage when missing.
            bundle['scale']=StandardScaler(with_mean=False).fit(bundle['imputer'].transform(aug[columns]))
        est=LogisticRegression(**params,fit_intercept=False,max_iter=1200,tol=1e-6)
        est.fit(bt_design(bundle,aug),aug.team1_won,sample_weight=w/2)
    bundle['model']=est
    return bundle


def predict(bundle,frame):
    x=frame[bundle['columns']] if bundle['kind']=='lightgbm' else bt_design(bundle,frame)
    return bundle['model'].predict_proba(x)[:,1]


def select(train,folds,name,kind,cols,grid):
    records=[]; stored=[]
    for params in grid:
        preds=[]; losses=[]
        for i,j in folds:
            bundle=fit(train.iloc[i],kind,cols,params)
            held=train.iloc[j]
            out=prediction_table(held,predict(bundle,held),name)
            preds.append(out); losses.append(probability_metrics(out)['logloss'])
        combined=pd.concat(preds,ignore_index=True)
        records.append({'params':params,'fold_logloss':losses,'pooled_logloss':probability_metrics(combined)['logloss']})
        stored.append(combined)
    best=min(range(len(grid)),key=lambda i:records[i]['pooled_logloss'])
    print(name,'forward CV',round(records[best]['pooled_logloss'],5),grid[best],flush=True)
    return grid[best],records,stored[best]


def paired_logloss(a,b,y,dates):
    def loss(p):
        p=np.clip(p,1e-15,1-1e-15); return -np.where(y,np.log(p),np.log1p(-p))
    diff=loss(a)-loss(b)
    daily=pd.DataFrame({'day':dates.dt.normalize(),'delta':diff}).groupby('day').delta.agg(['sum','size']).to_numpy()
    rng=np.random.default_rng(SEED); s=daily[rng.integers(len(daily),size=(2000,len(daily)))].sum(axis=1)
    return {'delta':float(diff.mean()),'day_bootstrap_95':np.quantile(s[:,0]/s[:,1],[.025,.975]).tolist()}


def load_data():
    parts=[]
    for split,base in zip(('train','val'),load_inputs()):
        new=pd.read_csv(p1.M2_DATA/f'{split}.csv',parse_dates=['date'])
        pd.testing.assert_frame_equal(new[base.columns],base,check_dtype=False)
        if new['h2h_games'].isna().any() or np.isinf(new[p1.COLUMNS].to_numpy()).any(): raise ValueError('Bad P1')
        parts.append(new)
    return parts


def main():
    train,val=load_data(); folds=list(chronological_folds(train)); selected={}; preds={}; metrics=[]
    out=ROOT/'reports'; val_dir=out/'predictions/val'; oof_dir=out/'predictions/m2_oof'; model_dir=ROOT/'models/m2'
    for d in (val_dir,oof_dir,model_dir): d.mkdir(parents=True,exist_ok=True)
    report={'versions':{'python':platform.python_version(),'lightgbm':lgb.__version__,'sklearn':sklearn.__version__},
        'inputs':{s:csv_checksum(p1.M2_DATA/f'{s}.csv') for s in ('train','val')},'test_evaluated':False,
        'cv_gap_days':30,'n_train':len(train),'n_val':len(val),'folds':[{ 'fit':len(i),'hold':len(j),
        'fit_end':str(train.iloc[i].date.max()),'hold_start':str(train.iloc[j].date.min())} for i,j in folds],
        'models':{},'coverage':{},'paired_logloss':{}}
    variants=[('lightgbm_p0','lightgbm',P0,TREES),('lightgbm_p1','lightgbm',FULL,TREES),
              ('bradley_terry','bt',[],REG),('bradley_terry_context','bt',CONTEXT,REG)]
    def evaluate(name,kind,cols,params,cv=None,save=True):
        bundle=fit(train,kind,cols,params); prob=predict(bundle,val)
        t=prediction_table(val,prob,name); scores=probability_metrics(t); preds[name]=prob
        low,high=accuracy_ci(t,val.date); result={'model':name,'features':len(cols),'cv_logloss':cv,
            **scores,'acc_ci_low':low,'acc_ci_high':high,'meets_both':scores['acc']>=.63 and scores['logloss']<=.635}
        metrics.append(result)
        if save:
            t.to_csv(val_dir/f'{name}.csv',index=False); joblib.dump(bundle,model_dir/f'{name}.joblib',compress=3)
        print(name,'val',round(scores['acc'],5),round(scores['logloss'],5),flush=True)
        return result
    for name,kind,cols,grid in variants:
        params,records,oof=select(train,folds,name,kind,cols,grid); selected[name]=params
        oof.to_csv(oof_dir/f'{name}.csv',index=False)
        result=evaluate(name,kind,cols,params,probability_metrics(oof)['logloss'])
        report['models'][name]={'selected_params':params,'candidates':records,'validation':result}
    # Hold hyperparameters fixed to the P0 selection, including when removing feature groups.
    matched='lightgbm_p1'
    if selected['lightgbm_p0']!=selected['lightgbm_p1']:
        matched='lightgbm_p1_matched'; evaluate(matched,'lightgbm',FULL,selected['lightgbm_p0'])
    groups={'roster_form':p1.ROSTER,'maps':p1.MAPS,'h2h':p1.DUELS}
    for group,names in groups.items():
        remove={prefix+n for prefix in ('team1_','team2_','diff_','') for n in names}
        evaluate('without_'+group,'lightgbm',[c for c in FULL if c not in remove],selected['lightgbm_p0'],save=False)
    report['matched_p1']=matched
    for name,ref in [(matched,'lightgbm_p0')]+[(f'without_{g}',matched) for g in groups]:
        report['paired_logloss'][f'{name} minus {ref}']=paired_logloss(preds[name],preds[ref],val.team1_won.to_numpy(),val.date)
    for split,data in [('train',train),('val',val)]:
        report['coverage'][split]={k:float(x.mean()) for k,x in {
            'both_prior_complete_lineups':data.team1_lineup_age.notna() & data.team2_lineup_age.notna(),
            'both_any_player_stats':(data.team1_players_with_stats>0)&(data.team2_players_with_stats>0),
            'both_any_maps180':(data.team1_maps180>0)&(data.team2_maps180>0),'had_h2h':data.h2h_games>0}.items()}
    for name in ('random_forest','logistic_regression','elo'):
        table=validate_predictions(pd.read_csv(val_dir/f'{name}.csv'),val)
        report.setdefault('baselines',{})[name]=probability_metrics(table)
    pd.DataFrame(metrics).to_csv(out/'m2_validation.csv',index=False)
    (out/'m2_experiment.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
