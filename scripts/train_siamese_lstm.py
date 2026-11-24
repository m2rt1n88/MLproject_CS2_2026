"""M3 进阶模型（2/2）：孪生 LSTM，按双方各自最近 N 场历史打分，评估口径与 M1/M2/MLP 一致。

结构：一个共享 LSTM 把每队的历史序列编码成队伍向量 h；打分头 g 再读入该队当前
赛前量（Elo、经验）和本场赛事等级，s = g(h1) − g(h2)，p = sigmoid(s)。
混合版再加一个无偏置线性项 w·x（x = M2 的 41 列差值，交换两队时变号），
所以交换两队后 s 精确变号，p(swap) = 1 − p 由结构保证，无需镜像扩增。

纪律：
  * 序列由 scripts/sequence_features.py 从 clean_v2/matches.csv 在线生成，只用更早比赛日；
    训练/核验只读 2026-04-01 之前的比赛，测试期标签一次也不读。
  * 每个时间折只在拟合侧训练；学习率与轮数都在拟合侧内层选：拟合侧最近 15% 的比赛日
    （前面隔 14 天）做早停集，再用选出的 lr 和轮数在整个拟合侧重训后预测留出折。
    正式模型的 lr 取各折内层选择的多数，轮数取该 lr 下内层早停轮数的中位数。
  * 每个配置 3 个种子取概率平均；候选只按 OOF log loss 选，val 只评估一次。

用法（项目根目录）：
    python -B -m scripts.train_siamese_lstm train
    python -B -m scripts.train_siamese_lstm verify

输出：models/m3_lstm/lstm.pt（不进仓库）、reports/m3/lstm_{oof,val}.csv、
      reports/m3/lstm_metrics.json、reports/m3/LSTM.md
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn

from scripts.ablation import ece
from scripts.basic_team_features import normalize_team
from scripts.match_splits_and_metrics import (ROOT, TEST_START, chronological_folds, csv_checksum,
    prediction_table, probability_metrics, validate_predictions)
from scripts.sequence_features import (CONTEXT_FEATURES, STEP_FEATURES, TEAM_FEATURES,
    build_sequences, load_matches, positions)
from scripts.train_m1_m2 import design, partition, weights

DATA = ROOT / 'data/processed/clean_v2'
MODEL = ROOT / 'models/m3_lstm/lstm.pt'
REPORT = ROOT / 'reports/m3'
NAME = 'm3_siamese_lstm'
SEED = 2026
SEEDS = (2026, 2027, 2028)
# 学习率不是固定值：每个 (配置, 外层折) 在拟合侧内层早停集上比较 LRS，取 3 个种子
# 最佳早停 log loss 的均值最低者（并列取列表靠前者）。外层留出折从不参与。
# 正式模型：选中配置 5 个外层折的内层选择取多数（5 折为奇数，不会并列）；
# 轮数取该学习率下全部折 × 种子的内层早停轮数的中位数。
LRS = (2e-3, 1e-3)
BASE = dict(head=16, dropout=0.2, weight_decay=1e-2, lin_l2=0.02, batch=128,
            max_epochs=100, patience=10, es_frac=0.15, es_gap_days=14)
CANDIDATES = [
    dict(key='seq_n20_h16', label='仅序列，N=20，隐层 16', n_steps=20, hidden=16, hybrid=False, weighted=False),
    dict(key='seq_n20_h32', label='仅序列，N=20，隐层 32', n_steps=20, hidden=32, hybrid=False, weighted=False),
    dict(key='seq_n10_h16', label='仅序列，N=10，隐层 16', n_steps=10, hidden=16, hybrid=False, weighted=False),
    dict(key='hyb_n20_h16', label='混合（+M2 线性项），N=20，隐层 16', n_steps=20, hidden=16, hybrid=True, weighted=False),
    dict(key='hyb_n10_h16', label='混合（+M2 线性项），N=10，隐层 16', n_steps=10, hidden=16, hybrid=True, weighted=False),
    dict(key='hyb_n20_h16_w', label='混合，N=20，隐层 16，365 天半衰期加权', n_steps=20, hidden=16, hybrid=True, weighted=True),
]


class Siamese(nn.Module):
    """共享编码器 + 共享打分头；分数是两队打分之差（加可选的反对称线性项）。"""

    def __init__(self, hidden: int, n_lin: int = 0, head: int = 16, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(len(STEP_FEATURES), hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + len(TEAM_FEATURES) + len(CONTEXT_FEATURES), head),
                                  nn.Tanh(), nn.Linear(head, 1))
        self.lin = nn.Linear(n_lin, 1, bias=False) if n_lin else None
        if self.lin is not None:
            nn.init.zeros_(self.lin.weight)

    def forward(self, seq, length, team, ctx, x=None):
        b, _, n, f = seq.shape
        out, _ = self.lstm(seq.reshape(b * 2, n, f))
        flat = length.reshape(-1)
        # 左对齐序列：取最后一个有效步的输出；没有历史的队伍用零向量
        h = out[torch.arange(b * 2, device=seq.device), (flat - 1).clamp(min=0)]
        h = h * (flat > 0).unsqueeze(-1).to(h.dtype)
        h = self.drop(h.reshape(b, 2, -1))
        z = self.head(torch.cat([h, team, ctx.unsqueeze(1).expand(-1, 2, -1)], -1)).squeeze(-1)
        s = z[:, 0] - z[:, 1]
        if self.lin is not None:
            s = s + self.lin(x).squeeze(-1)
        return s


def gather(data: dict, pos: np.ndarray, x: np.ndarray | None = None) -> dict:
    out = {k: data[k][pos] for k in ('seq', 'length', 'team', 'ctx')}
    if x is not None:
        out['x'] = np.asarray(x, np.float32)
    return out


def swap(inputs: dict) -> dict:
    """交换两队：队伍轴翻转、差值特征变号、对称的 ctx 不变。"""
    out = {k: inputs[k][:, ::-1].copy() for k in ('seq', 'length', 'team')}
    out['ctx'] = inputs['ctx'].copy()
    if 'x' in inputs:
        out['x'] = -inputs['x']
    return out


def to_tensors(inputs: dict, device) -> dict:
    return {k: torch.as_tensor(np.ascontiguousarray(v), device=device) for k, v in inputs.items()}


def make_model(cfg: dict, n_lin: int) -> Siamese:
    return Siamese(cfg['hidden'], n_lin if cfg['hybrid'] else 0,
                   head=cfg.get('head', BASE['head']), dropout=cfg.get('dropout', BASE['dropout']))


def scores(model: Siamese, inputs: dict, batch: int = 4096) -> np.ndarray:
    """eval 模式在 CPU 上算 logit（float64 返回），保证同一权重给出同一结果。"""
    model = model.cpu().eval()
    t = to_tensors(inputs, 'cpu')
    return _scores(model, t, batch)


def _scores(model: Siamese, t: dict, batch: int = 4096) -> np.ndarray:
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(t['seq']), batch):
            sl = {k: v[i:i + batch] for k, v in t.items()}
            res.append(model(sl['seq'], sl['length'], sl['team'], sl['ctx'], sl.get('x')).double().cpu().numpy())
    return np.concatenate(res)


def train_one(cfg: dict, inputs: dict, y: np.ndarray, w: np.ndarray, seed: int, device,
              epochs: int | None = None, es: tuple | None = None) -> tuple[dict, int]:
    """训练一个种子。给 es=(inputs, y) 时按早停集 log loss 选最佳轮；否则固定训练 epochs 轮。"""
    p = {**BASE, **cfg}
    torch.manual_seed(seed)
    n_lin = inputs['x'].shape[1] if 'x' in inputs else 0
    model = make_model(p, n_lin).to(device)
    lin = [q for name, q in model.named_parameters() if name.startswith('lin.')]
    rest = [q for name, q in model.named_parameters() if not name.startswith('lin.')]
    groups = [dict(params=rest, weight_decay=p['weight_decay'])] + ([dict(params=lin, weight_decay=0.0)] if lin else [])
    opt = torch.optim.AdamW(groups, lr=p['lr'])
    t = to_tensors(inputs, device)
    es_t = to_tensors(es[0], device) if es is not None else None
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w / w.mean(), dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    n = len(y)
    total = epochs if es is None else p['max_epochs']
    best = (np.inf, 0, None)
    for epoch in range(1, total + 1):
        model.train()
        for idx in torch.randperm(n, generator=gen).split(p['batch']):
            idx = idx.to(device)
            s = model(t['seq'][idx], t['length'][idx], t['team'][idx], t['ctx'][idx],
                      t['x'][idx] if 'x' in t else None)
            loss = (nn.functional.binary_cross_entropy_with_logits(s, yt[idx], reduction='none') * wt[idx]).mean()
            if lin:
                loss = loss + p['lin_l2'] * lin[0].pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if es is not None:
            # 早停只用于挑轮数，在训练设备上评估即可；最终概率统一在 CPU 上重算
            ll = logloss(expit(_scores(model, es_t)), es[1])
            if ll < best[0] - 1e-5:
                best = (ll, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            elif epoch - best[1] >= p['patience']:
                break
    if es is not None:
        return best[2], best[1], best[0]          # (state, 最佳轮数, 早停集最佳 log loss)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, total


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load_states(cfg: dict, states: list[dict], n_lin: int) -> list[Siamese]:
    models = []
    for state in states:
        model = make_model({**BASE, **cfg}, n_lin)
        model.load_state_dict(state)
        models.append(model.eval())
    return models


def ensemble(models: list[Siamese], inputs: dict) -> np.ndarray:
    return np.mean([expit(scores(m, inputs)) for m in models], axis=0)


def x_scale(x: np.ndarray) -> np.ndarray:
    """只除以均方根、不减均值，保持 x → −x 的反对称（同 M2 的 with_mean=False）。"""
    s = np.sqrt(np.mean(np.asarray(x, float) ** 2, axis=0))
    return np.where(s > 1e-8, s, 1.0)


def es_split(rows: pd.DataFrame, frac: float, gap_days: int) -> tuple[np.ndarray, np.ndarray]:
    """拟合侧内部的早停切分：最近 frac 的比赛日做早停集，之前隔 gap_days 天的比赛做内层训练。"""
    days = rows.date.dt.normalize()
    unique = np.sort(days.unique())
    start = pd.Timestamp(unique[-max(1, int(round(len(unique) * frac)))])
    inner = np.flatnonzero(rows.date < start - pd.Timedelta(days=gap_days))
    held = np.flatnonzero(days >= start)
    if not len(inner) or not len(held):
        raise ValueError('Empty early-stopping partition')
    return inner, held


def prepare(cfg: dict, data: dict, rows: pd.DataFrame, x: np.ndarray | None):
    """rows 的模型输入；混合版的缩放只由 rows 自己的特征算出（不看标签）。"""
    p = {**BASE, **cfg}
    scale = x_scale(x) if p['hybrid'] else None
    inputs = gather(data, positions(data, rows.match_id), x / scale if p['hybrid'] else None)
    return inputs, scale


def select_lr(cfg: dict, data: dict, rows: pd.DataFrame, x: np.ndarray | None, device,
              seeds=SEEDS, lrs=LRS) -> dict:
    """只用 rows（= 某个外层折的拟合侧）内部的早停切分挑学习率和轮数。

    每个 lr、每个种子在内层训练集上训练、在内层早停集（rows 最近 es_frac 的比赛日，
    前隔 es_gap_days 天）上早停；lr 取 3 个种子最佳早停 log loss 的均值最低者，
    并列时取 lrs 中靠前者。返回所有 lr 的轮数与 log loss，供正式模型汇总。
    """
    p = {**BASE, **cfg}
    inputs, _ = prepare(cfg, data, rows, x)
    y = rows.team1_won.to_numpy(float)
    inner, held = es_split(rows, p['es_frac'], p['es_gap_days'])
    sub = lambda d, idx: {k: v[idx] for k, v in d.items()}
    # 内层权重以内层最晚日期为基准重新计算
    wi = weights(rows.iloc[inner]) if p['weighted'] else np.ones(len(inner))
    per_lr = {}
    for lr in lrs:
        epochs, losses = [], []
        for seed in seeds:
            _, e, ll = train_one(dict(cfg, lr=lr), sub(inputs, inner), y[inner], wi, seed, device,
                                 es=(sub(inputs, held), y[held]))
            epochs.append(max(int(e), 1))
            losses.append(float(ll))
        per_lr[str(lr)] = dict(epochs=epochs, inner_logloss=losses, mean_logloss=float(np.mean(losses)))
    best = min(lrs, key=lambda lr: (per_lr[str(lr)]['mean_logloss'], lrs.index(lr)))
    return dict(lr=float(best), epochs=per_lr[str(best)]['epochs'], per_lr=per_lr,
                n_inner=int(len(inner)), n_es=int(len(held)))


def fit_side(cfg: dict, data: dict, rows: pd.DataFrame, x: np.ndarray | None, device,
             seeds=SEEDS, epochs: int | None = None, lr: float | None = None) -> dict:
    """在一批训练比赛（原始方向）上训练 3 个种子。

    epochs=None：先用 select_lr 在 rows 内部挑 lr 和每个种子的轮数，再按它们在全部 rows 上重训。
    epochs=int, lr=float：直接固定 lr 和轮数在全部 rows 上训练（正式模型）。
    """
    p = {**BASE, **cfg}
    selection = None
    if epochs is None:
        selection = select_lr(cfg, data, rows, x, device, seeds=seeds)
        lr, per_seed = selection['lr'], selection['epochs']
    else:
        if lr is None:
            raise ValueError('A fixed-epoch fit needs a fixed learning rate')
        per_seed = [int(epochs)] * len(seeds)
    inputs, scale = prepare(cfg, data, rows, x)
    y = rows.team1_won.to_numpy(float)
    w = weights(rows) if p['weighted'] else np.ones(len(rows))
    states = []
    for seed, e in zip(seeds, per_seed):
        state, _ = train_one(dict(cfg, lr=lr), inputs, y, w, seed, device, epochs=e)
        states.append(state)
    return dict(states=states, epochs=[int(e) for e in per_seed], lr=float(lr), selection=selection,
                x_scale=None if scale is None else scale.tolist())


def final_choice(fold_fits: list[dict], lrs=LRS) -> tuple[float, int]:
    """正式模型：lr 取各外层折内层选择的多数（并列取 lrs 靠前者）；
    轮数取该 lr 在全部折 × 种子上的内层早停轮数的中位数（四舍五入）。"""
    votes = [f['lr'] for f in fold_fits]
    lr = max(lrs, key=lambda v: (votes.count(v), -lrs.index(v)))
    epochs = [e for f in fold_fits for e in f['selection']['per_lr'][str(lr)]['epochs']]
    return float(lr), int(round(np.median(epochs)))


def outer_oof(cfg: dict, data: dict, train: pd.DataFrame, x: np.ndarray, columns: list[str], folds, device,
              log=print) -> tuple[np.ndarray, list[dict]]:
    """外层时间折：每折只把拟合侧 train.iloc[i] 交给 fit_side（含 lr 选择），只对留出折预测。"""
    oof = np.full(len(train), np.nan)
    fits = []
    for f, (i, j) in enumerate(folds):
        fitted = fit_side(cfg, data, train.iloc[i], x[i], device)
        bundle = bundle_for(cfg, fitted, columns, data['elo_k'])
        oof[j] = predict_with(bundle, data, train.iloc[j])
        fits.append(dict(lr=fitted['lr'], epochs=fitted['epochs'], selection=fitted['selection']))
        log(f'  {cfg["key"]:<14} fold {f + 1} lr {fitted["lr"]:g} epochs {fitted["epochs"]} '
            f'inner ll ' + ' / '.join(f'{k}:{v["mean_logloss"]:.4f}' for k, v in fitted['selection']['per_lr'].items()))
    return oof, fits


def model_inputs(bundle: dict, data: dict, rows: pd.DataFrame) -> dict:
    pos = positions(data, rows.match_id)
    if not (np.array_equal(data['team1'][pos], rows.team1.map(normalize_team).to_numpy())
            and np.array_equal(data['team2'][pos], rows.team2.map(normalize_team).to_numpy())):
        raise ValueError('Team orientation differs between rows and the match history')
    x = None
    if bundle['hybrid']:
        d = design(rows)
        if list(d) != bundle['columns']:
            raise ValueError('M2 feature order has changed')
        x = d.to_numpy(float) / np.asarray(bundle['x_scale'])
    return gather(data, pos, x)


def predict_with(bundle: dict, data: dict, rows: pd.DataFrame) -> np.ndarray:
    models = load_states(bundle['config'], bundle['states'], len(bundle['columns']) if bundle['hybrid'] else 0)
    return ensemble(models, model_inputs(bundle, data, rows))


def predict(bundle: dict, matches_all: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """给 rows（clean_v2 赛前特征行，原始队伍方向）打分，按输入顺序返回 p_team1。

    matches_all：clean_v2/matches.csv 同结构的比赛表，须包含 rows 里的比赛及其之前的全部比赛。
    每场序列只用比赛日严格更早的比赛，所以 rows 自身及同日、之后的赛果不会进入它的输入。
    """
    data = build_sequences(matches_all, bundle['n_steps'], bundle['elo_k'])
    if data['elo_k'] != bundle['elo_k']:
        raise ValueError('Elo K differs from training')
    return predict_with(bundle, data, rows)


def bundle_for(cfg: dict, fitted: dict, columns: list[str], k: float) -> dict:
    return dict(stage=NAME, config={kk: cfg[kk] for kk in ('key', 'label', 'n_steps', 'hidden', 'hybrid', 'weighted')},
                hybrid=bool(cfg['hybrid']), n_steps=int(cfg['n_steps']), elo_k=float(k),
                columns=list(columns), **fitted)


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    table = pd.DataFrame({'match_id': np.arange(len(y)), 'y_true': y, 'p_team1': p, 'model': NAME})
    return dict(n=len(y), **probability_metrics(table), ece=ece(p, y))


def day_bootstrap(p: np.ndarray, ref: np.ndarray, y: np.ndarray, dates: pd.Series,
                  reps: int = 2000, seed: int = SEED) -> tuple[float, tuple[float, float]]:
    """逐场 log loss 差（本模型 − M2）的配对 bootstrap：整天重采样，区间 2.5%/97.5% 分位。"""
    def loss(q):
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    d = loss(p) - loss(ref)
    by_day = pd.DataFrame({'day': pd.to_datetime(dates).dt.normalize().to_numpy(), 'd': d}).groupby('day').d
    sums, counts = by_day.sum().to_numpy(), by_day.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(reps, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    return float(d.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def setup(device_name: str | None = None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))


def history_source() -> pd.DataFrame:
    """训练/核验只读测试期之前的比赛：测试期标签一次也不进内存。"""
    m = load_matches()
    return m[m.date < TEST_START].reset_index(drop=True)


def save(bundle: dict) -> None:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, MODEL)


def load_bundle() -> dict:
    return torch.load(MODEL, map_location='cpu', weights_only=True)


def main(device_name: str | None = None):
    t0 = time.time()
    device = setup(device_name)
    pre = history_source()
    train, val = partition('train'), partition('val')
    x_train, x_val = design(train), design(val)
    columns = list(x_train)
    xt = x_train.to_numpy(float)
    y = train.team1_won.to_numpy()
    folds = list(chronological_folds(train))
    order = np.concatenate([j for _, j in folds])
    held = train.iloc[order].reset_index(drop=True)
    k = None
    data = {}
    for n in sorted({c['n_steps'] for c in CANDIDATES}):
        data[n] = build_sequences(pre, n)
        k = data[n]['elo_k']
    print(f'device {device}  train {len(train):,}  val {len(val):,}  OOF {len(order):,}  Elo K={k}', flush=True)

    cand_rows, oof_by_key = [], {}
    for cfg in CANDIDATES:
        # 外层留出折的 log loss 只在本配置全部折跑完后汇总成 OOF，不回流到 lr / 轮数
        oof, fits = outer_oof(cfg, data[cfg['n_steps']], train, xt, columns, folds, device,
                              log=lambda s: print(f'{s}  ({time.time() - t0:.0f}s)', flush=True))
        final_lr, final_epochs = final_choice(fits)
        m = summary(oof[order], y[order])
        oof_by_key[cfg['key']] = oof
        cand_rows.append(dict(cfg, oof=m, fold_lr=[f['lr'] for f in fits],
                              fold_epochs=[e for f in fits for e in f['epochs']],
                              fold_selection=[f['selection'] for f in fits],
                              final_lr=final_lr, final_epochs=final_epochs))
        print(f'  {cfg["key"]:<14} OOF acc={m["acc"]:.4f} ll={m["logloss"]:.4f} ece={m["ece"]:.4f} '
              f'fold lr={[f["lr"] for f in fits]} -> final lr={final_lr:g}, epochs={final_epochs}', flush=True)

    best = min(cand_rows, key=lambda r: (r['oof']['logloss'], -r['oof']['acc']))
    print(f'\n选中 {best["key"]}：{best["label"]}，正式模型 lr={best["final_lr"]:g}、'
          f'固定 {best["final_epochs"]} 轮', flush=True)
    cfg = next(c for c in CANDIDATES if c['key'] == best['key'])
    fitted = fit_side(cfg, data[cfg['n_steps']], train, xt, device,
                      epochs=best['final_epochs'], lr=best['final_lr'])
    bundle = bundle_for(cfg, fitted, columns, k)
    bundle.update(dataset='v9-clean-v2', train_sha256=csv_checksum(DATA / 'm2/train.csv'),
                  base=dict(BASE), lrs=list(LRS), seeds=list(SEEDS))
    save(bundle)
    val_pred = predict(load_bundle(), pre, val)
    train_seconds = time.time() - t0

    oof_held = oof_by_key[best['key']][order]
    metrics = {'oof': summary(oof_held, held.team1_won.to_numpy()),
               'val': summary(val_pred, val.team1_won.to_numpy())}
    refs = {}
    for split, rows in (('oof', held), ('val', val)):
        ref = pd.read_csv(ROOT / 'reports/release' / f'm2_{split}.csv')
        refs[split] = ref.set_index('match_id').loc[rows.match_id].p_team1.to_numpy()
        metrics['m2_' + split] = summary(refs[split], rows.team1_won.to_numpy())
    REPORT.mkdir(parents=True, exist_ok=True)
    for split, rows, p in (('oof', held, oof_held), ('val', val, val_pred)):
        prediction_table(rows, p, NAME).to_csv(REPORT / f'lstm_{split}.csv', index=False)
        validate_predictions(pd.read_csv(REPORT / f'lstm_{split}.csv'), rows)
    for split, rows, p in (('oof', held, oof_held), ('val', val, val_pred)):
        d, ci = day_bootstrap(p, refs[split], rows.team1_won.to_numpy(), rows.date)
        metrics[f'vs_m2_{split}'] = dict(d_logloss=d, lo=ci[0], hi=ci[1], resample='match_day', reps=2000, seed=SEED)
    metrics['candidates'] = [dict(key=r['key'], label=r['label'], n_steps=r['n_steps'], hidden=r['hidden'],
                                  hybrid=r['hybrid'], weighted=r['weighted'], oof=r['oof'],
                                  fold_lr=r['fold_lr'], fold_epochs=r['fold_epochs'],
                                  fold_selection=r['fold_selection'],
                                  final_lr=r['final_lr'], final_epochs=r['final_epochs']) for r in cand_rows]
    metrics['selected'] = dict(bundle['config'], fold_lr=best['fold_lr'], final_lr=best['final_lr'],
                               final_epochs=best['final_epochs'], seeds=list(SEEDS))
    metrics['training'] = dict(device=str(device), seconds=round(train_seconds, 1), base=dict(BASE),
                               lrs=list(LRS), elo_k=k, torch=torch.__version__,
                               lr_rule='per (config, outer fold): lowest mean inner early-stop log loss over 3 seeds '
                                       '(tie -> earlier in lrs); final: majority of the 5 fold choices, epochs = '
                                       'median of that lr\'s inner early-stop epochs over folds x seeds')
    (REPORT / 'lstm_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    write_report(train, val, len(held), cand_rows, best, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'candidates'}, indent=2, ensure_ascii=False))


def verify():
    setup('cpu')
    val = partition('val')
    metrics = json.loads((REPORT / 'lstm_metrics.json').read_text(encoding='utf8'))
    bundle = load_bundle()
    if bundle['train_sha256'] != csv_checksum(DATA / 'm2/train.csv') or bundle['stage'] != NAME:
        raise ValueError('LSTM bundle is not trained on the current clean_v2 snapshot')
    if bundle['config']['key'] != metrics['selected']['key'] or bundle['lr'] != metrics['selected']['final_lr']:
        raise ValueError('Saved weights differ from the selected configuration')
    t = validate_predictions(pd.read_csv(REPORT / 'lstm_val.csv'), val)
    if not t.model.eq(NAME).all() or metrics['val']['n'] != len(val):
        raise ValueError('Unexpected LSTM validation predictions')
    p = predict(bundle, history_source(), val)
    np.testing.assert_allclose(p, t.p_team1, rtol=0, atol=1e-6)
    fresh = summary(p, val.team1_won.to_numpy())
    for key in ('acc', 'logloss', 'brier', 'ece'):
        np.testing.assert_allclose(fresh[key], metrics['val'][key], rtol=0, atol=1e-6)
    for k, v in probability_metrics(t).items():
        np.testing.assert_allclose(v, metrics['val'][k], rtol=0, atol=1e-12)
    print(f'Siamese LSTM verified: {len(val)} val matches recomputed from saved weights '
          f'(max |Δp| = {np.abs(p - t.p_team1.to_numpy()).max():.2e}).')


def write_report(train, val, n_held, cand_rows, best, metrics):
    v, o = metrics['vs_m2_val'], metrics['vs_m2_oof']
    if v['lo'] > 0:
        verdict = '区间整体在零以上：孪生 LSTM 比 M2 差，这个差异可以当真。'
    elif v['hi'] < 0:
        verdict = '区间整体在零以下：孪生 LSTM 比 M2 好。'
    else:
        verdict = '区间跨零，不能说它比 M2 更好或更差。'
    mv, sel = metrics['val'], metrics['selected']
    md = ['# M3（2/2）：孪生 LSTM 赛前胜负概率', '',
          '## 结论', '',
          f'- **开发验证集 {mv["acc"]:.2%} / log loss {mv["logloss"]:.4f}**，'
          f'相对冻结 M2 的 Δ log loss = {v["d_logloss"]:+.4f}（LSTM − M2，正数表示更差），'
          f'95% CI [{v["lo"]:+.5f}, {v["hi"]:+.5f}]  ',
          f'  {verdict}',
          f'- 选中配置：<u>{sel["label"]}</u>，正式模型 lr={sel["final_lr"]:g}、固定 {sel["final_epochs"]} 轮，'
          '3 个种子取平均', '',
          '## 协议', '',
          '### 序列里有什么', '',
          '- 每队取本场之前最近 N 场比赛，按时间从早到晚排列，不足 N 场补零并记录长度',
          '- 每一步是那场历史比赛从本队视角看的信息：  ',
          '  胜负、地图比分差（除以总图数）、对手赛前 Elo − 本队赛前 Elo、本队赛前 Elo、'
          '距本场天数的对数、那场的赛事等级、那场名单与本队最近一次完整五人名单的重合度（含缺失标记）',
          '- 每队另有当前赛前量：当前 Elo、历史场数的对数、是否有历史；本场赛事等级作为对称的比赛级输入',
          f'- Elo 在 `sequence_features.py` 内按天批量在线计算，K = {metrics["training"]["elo_k"]:g}'
          '（`reports/elo_validation.json` 选中值），不依赖各切分的特征 CSV',
          '- 混合版另加 M2 的 41 列差值特征，经无偏置线性项进入分数', '',
          '### 防泄漏规则', '',
          '- **只用更早比赛日**：同一天的比赛先全部生成序列，当天结束后才把赛果、Elo、名单写入历史',
          '- **本场的比分、胜负从不进入本场输入**，只在它成为之后比赛的历史时才被读取',
          '- **不用 `bo_reported` / `bo`**：clean_v2 协议下赛制赛前未知',
          '- **不用本场名单**：clean_v2 里本场五人名单是否完整与胜负强相关'
          '（只有一方名单完整时，这一方胜率约 65%–75%），属于赛后补录造成的泄漏；重合度改用本队更早的完整名单',
          '- 训练与核验只读 2026-04-01 之前的比赛，**测试期标签未读取**，也没有生成测试集预测', '',
          '### 结构与训练', '',
          '- 两队共用一个 LSTM 编码器和一个打分头，分数 s = g(队一) − g(队二)（混合版再加 w·x），p = sigmoid(s)  ',
          '  交换两队后 s 精确变号，所以 <u>p(交换) = 1 − p 由结构保证</u>，无需镜像扩增',
          f'- 训练 {len(train):,} 场 / 开发验证 {len(val):,} 场；五折按天向前、间隔 30 天，'
          f'OOF 覆盖 {n_held:,} 场',
          f'- 每折只在拟合侧训练：拟合侧最近 {BASE["es_frac"]:.0%} 的比赛日做早停集（之前隔 {BASE["es_gap_days"]} 天），'
          '挑出轮数后在整个拟合侧按该轮数重训，再预测留出折；**留出折从不参与早停**',
          f'- **学习率也在拟合侧内层选**：每个（配置, 外层折）在内层早停集上比较 lr ∈ {{{", ".join(f"{v:g}" for v in LRS)}}}，'
          '取 3 个种子最佳早停 log loss 均值最低者（并列取 2e-3）',
          '- 正式模型：lr 取选中配置 5 个外层折内层选择的多数；轮数取该 lr 在各折、各种子上内层早停轮数的中位数；'
          '在全部训练集上重训一次，val 只评估一次',
          f'- AdamW weight decay={BASE["weight_decay"]:g}、dropout={BASE["dropout"]}、'
          f'batch={BASE["batch"]}；混合版线性项 L2 = {BASE["lin_l2"]:g}',
          '- 说明：早先一版草稿在一次冒烟测试后把 lr 固定为 1e-3，那次测试打印过外层留出折的 log loss；'
          '本次运行改为只用拟合侧内层验证重新选 lr，草稿的结果已作废', '',
          '## 候选比较（只看 OOF）', '',
          '| 配置 | N | 隐层 | 混合 | 加权 | 各折 lr | OOF 准确率 | OOF log loss | OOF ECE | 正式 lr | 正式轮数 |',
          '|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|']
    for r in cand_rows:
        star = ' **← 选中**' if r['key'] == best['key'] else ''
        md.append(f'| {r["label"]}{star} | {r["n_steps"]} | {r["hidden"]} | {"是" if r["hybrid"] else "否"} | '
                  f'{"是" if r["weighted"] else "否"} | {" / ".join(f"{v:g}" for v in r["fold_lr"])} | '
                  f'{r["oof"]["acc"]:.2%} | {r["oof"]["logloss"]:.4f} | '
                  f'{r["oof"]["ece"]:.4f} | {r["final_lr"]:g} | {r["final_epochs"]} |')
    rows = [('M2（冻结）· OOF', 'm2_oof'), ('孪生 LSTM · OOF', 'oof'),
            ('M2（冻结）· val', 'm2_val'), ('**孪生 LSTM · val**', 'val')]
    md += ['', '## 与冻结 M2 的逐场对比', '',
           '| 口径 | 场次 | 准确率 | log loss | Brier | ECE |', '|---|---:|---:|---:|---:|---:|']
    for label, key in rows:
        r = metrics[key]
        md.append(f'| {label} | {r["n"]:,} | {r["acc"]:.2%} | {r["logloss"]:.4f} | {r["brier"]:.4f} | {r["ece"]:.4f} |')
    md += ['', '配对 bootstrap 按**比赛日**整天重采样（2,000 次，种子 2026）：',
           f'- val：Δ log loss = **{v["d_logloss"]:+.4f}**，95% CI **[{v["lo"]:+.5f}, {v["hi"]:+.5f}]**',
           f'- OOF：Δ log loss = {o["d_logloss"]:+.4f}，95% CI [{o["lo"]:+.5f}, {o["hi"]:+.5f}]'
           '（OOF 也用来选候选，只作参考）', '',
           '## 怎么读这个结果', '',
           '- 仅序列版只看赛果序列和自算 Elo，信息量明显少于 M2 的 41 列（阵容、选手数据、地图池、图论实力）；'
           '它和 M2 的差距说明「从赛果序列里自动学」能补回多少',
           '- 混合版等于「M2 线性项 + 序列修正」，它是否胜过 M2 才是序列模型有没有增量的直接检验',
           '- 概率未做额外校准，报告同时给出 log loss 与 ECE；融合/校准留给 M3 stacking 与 M4，'
           '且只能在更早时间折上学习',
           '- 权重 `models/m3_lstm/lstm.pt` 按 `.gitignore` 约定不进仓库，可由下方命令重建', '',
           '## 复现', '', '```bat',
           '.venv\\m3\\Scripts\\python.exe -B -m scripts.train_siamese_lstm train',
           '.venv\\m3\\Scripts\\python.exe -B -m scripts.train_siamese_lstm verify', '```', '',
           f'- 训练设备 {metrics["training"]["device"]}，耗时约 {metrics["training"]["seconds"] / 60:.0f} 分钟；'
           f'torch {metrics["training"]["torch"]}',
           '- 固定种子、确定性算法；预测统一在 CPU 上用保存的权重重算，`verify` 要求与 CSV 相差不超过 1e-6',
           '- 给新比赛打分：`predict(bundle, matches_all, rows)`，`matches_all` 要包含 rows 之前的全部比赛', '']
    (REPORT / 'LSTM.md').write_text('\n'.join(md) + '\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'verify'])
    parser.add_argument('--device', default=None, help='cuda / cpu，默认有 GPU 就用 GPU')
    args = parser.parse_args()
    main(args.device) if args.action == 'train' else verify()
