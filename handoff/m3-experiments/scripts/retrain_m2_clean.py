"""Retrain the frozen M2 recipe on clean_v2. Does not load test features or refit on validation."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.optimize_m2 import design, probabilities, train_model
from scripts.train_m1 import accuracy_ci
from scripts.train_m2 import paired_logloss
from scripts.training_protocol import (ROOT, TEST_START, VAL_START, chronological_folds, csv_checksum,
                                      prediction_table, probability_metrics, validate_predictions)

DATA = ROOT / 'data/processed/clean_v2'
OLD = ROOT / 'reports/m2_optimization'
OUT = ROOT / 'reports/m2_clean_v2'
MODEL = ROOT / 'models/m2/m2_schedule_clean_v2.joblib'
NAME = 'm2_schedule_clean_v2'


def inputs(split, manifest, columns):
    rel = f'm2/{split}.csv'
    if csv_checksum(DATA / rel) != manifest['files'][rel]:
        raise ValueError(f'clean_v2 checksum mismatch: {rel}')
    rows = pd.read_csv(DATA / rel, parse_dates=['date'])
    dates_ok = rows.date.lt(VAL_START) if split == 'train' else rows.date.ge(VAL_START) & rows.date.lt(TEST_START)
    if not (len(rows) == manifest['counts'][split] and rows.split.eq(split).all() and dates_ok.all()
            and rows.match_id.is_unique and not rows.is_swapped.any() and rows.bo_missing.eq(1).all()
            and rows[['bo1','bo5','bo3']].eq(0).all().all()):
        raise ValueError(f'Invalid original clean_v2 partition: {split}')
    x = design(rows)
    if list(x.columns) != columns or not np.isfinite(x).all().all(): raise ValueError('Unexpected design matrix')
    return rows, x


def compare(rows, new, old):
    old = validate_predictions(old, rows); new = validate_predictions(new, rows)
    y = new.y_true.to_numpy(); a = new.p_team1.to_numpy(); b = old.p_team1.to_numpy()
    hit = (a > .5) == y; prev = (b > .5) == y
    return dict(n=len(rows), new=probability_metrics(new), old=probability_metrics(old),
                correct=int(hit.sum()), fixed=int((hit & ~prev).sum()), broken=int((~hit & prev).sum()),
                paired_logloss=paired_logloss(a,b,y,rows.date.reset_index(drop=True)))


def main():
    manifest = json.loads((DATA/'manifest.json').read_text()); recipe = json.loads((OLD/'selection.json').read_text())
    assert manifest['version']=='v9-clean-v2'
    chosen=recipe['selected']; settings={k:chosen[k] for k in ('kind','param','logit_multiplier')}
    def fit(rows,x):
        bundle=train_model(x,rows,settings['kind'],settings['param']); bundle['temperature']=settings['logit_multiplier']; return bundle
    train,x=inputs('train',manifest,recipe['columns']); folds=[]; held=[]; tables=[]
    for i,j in chronological_folds(train):
        h=train.iloc[j]; pred=prediction_table(h,probabilities(fit(train.iloc[i],x.iloc[i]),x.iloc[j]),NAME)
        folds.append(dict(train_n=len(i), hold_n=len(j), train_end=str(train.iloc[i].date.max()), hold_start=str(h.date.min()), hold_end=str(h.date.max())))
        held.append(h); tables.append(pred)
    oof_rows=pd.concat(held,ignore_index=True); oof=pd.concat(tables,ignore_index=True)
    old_oof=validate_predictions(pd.read_csv(OLD/'train_oof.csv'),oof_rows)
    for info,t,h in zip(folds,tables,held):
        info['scores']=compare(h,t,old_oof[old_oof.match_id.isin(h.match_id)])
    bundle=fit(train,x); bundle['dataset_version']=manifest['version']; bundle['train_checksum']=manifest['files']['m2/train.csv']
    MODEL.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,MODEL,compress=3)
    saved=joblib.load(MODEL); val,vx=inputs('val',manifest,recipe['columns']); a=probabilities(saved,vx)
    np.testing.assert_allclose(a,probabilities(bundle,vx),rtol=0,atol=0)
    vt=prediction_table(val,a,NAME); vc=compare(val,vt,pd.read_csv(ROOT/'reports/predictions/val/m2_schedule.csv'))
    vc['accuracy_ci_95']=accuracy_ci(vt,val.date); vc['meets_both']=vc['new']['acc']>=.63 and vc['new']['logloss']<=.635
    s=dict(recipe=settings,dataset=manifest['version'],data_checksums={s:manifest['files'][f'm2/{s}.csv'] for s in ('train','val')},
           model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),train_rows=len(train),n_columns=x.shape[1],folds=folds,
           oof=compare(oof_rows,oof,old_oof),validation=vc,test_evaluated=False)
    OUT.mkdir(exist_ok=True); oof.to_csv(OUT/'train_oof.csv',index=False); vt.to_csv(OUT/'val_predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'oof':s['oof'],'val':vc,'fold_delta':[r['scores']['paired_logloss']['delta'] for r in folds]},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
