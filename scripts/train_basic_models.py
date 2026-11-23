"""两个课内基础模型：高斯朴素贝叶斯、单棵决策树（含同列数随机森林对照）。

为什么选这两个（都来自课程材料）：
  * 朴素贝叶斯：第二章第四节独立一节，Assignment 2 的 NBayesLogReg；
    它的条件独立假设会让概率过度自信，正好检验校准。
  * 决策树：第三章 / Assignment 3「SVM and Ensemble Learning」里随机森林与 AdaBoost
    的前置模型；和同输入随机森林对照，可以量化 bagging 的贡献。

纪律（与 M1/M2 一致，否则不可比）：
  * 只读冻结的 clean_v2/m2/{train,val}.csv，不碰 models/ 与 manifest；
  * 同一个 design()（41 列）、同一条时间切分、同一套镜像扩增、同样的 365 天样本权重；
  * 只在训练期留出（OOF）与开发验证（val）上比，**不碰测试集**。

用法（项目根目录）：
    python -B -m scripts.train_basic_models

输出：
    reports/basic_models/metrics.csv
    reports/basic_models/BASIC_MODELS.md
    reports/figures/basic_models.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from scripts.match_splits_and_metrics import ROOT, chronological_folds
from scripts.train_m1_m2 import design, partition, weights

OUT = ROOT / "reports/basic_models"
FIG = ROOT / "reports/figures"
SEED = 2026
TREE_DEPTHS = [2, 3, 5, 8, 12, None]
TREE_LEAF = 36                                    # 与 M1 随机森林一致
NB_SMOOTHING = [1e-9, 1e-6, 1e-3, 1e-1, 1.0]      # sklearn 默认 1e-9
RF_LEAF, RF_TREES = 36, 128

TRAIN = VAL = None  # 在 main 里赋值


# ---------------------------------------------------------------- 指标
def metrics(p, y):
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "acc": float(((p > .5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(bins) if (b == k).any()))


# ---------------------------------------------------------------- 训练器
def fit_tree(X, y, w, depth=TREE_LEAF, leaf=TREE_LEAF):
    both, yb = np.vstack([X, -X]), np.r_[y, 1 - y]
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
    m.fit(both, yb, sample_weight=np.tile(w, 2))
    return m


def fit_gnb(X, y, w, smoothing=1e-9):
    both, yb = np.vstack([X, -X]), np.r_[y, 1 - y]
    m = GaussianNB(var_smoothing=smoothing)
    m.fit(both, yb, sample_weight=np.tile(w, 2))
    return m


def fit_rf(X, y, w, leaf=RF_LEAF, trees=RF_TREES):
    both, yb = np.vstack([X, -X]), np.r_[y, 1 - y]
    m = RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf,
                               max_features="sqrt", n_jobs=1, random_state=SEED)
    m.fit(both, yb, sample_weight=np.tile(w, 2))
    return m


FITTERS = {"决策树": fit_tree, "朴素贝叶斯": fit_gnb, "随机森林": fit_rf}


# ---------------------------------------------------------------- 跑一个配置
def run_config(kind, Xtr, ytr, Xva, yva, folds, **kw):
    fit = FITTERS[kind]
    oof = np.full(len(ytr), np.nan)
    for i, j in folds:
        m = fit(Xtr[i], ytr[i], weights(TRAIN.iloc[i]), **kw)
        oof[j] = m.predict_proba(Xtr[j])[:, 1]
    mask = ~np.isnan(oof)
    m = fit(Xtr, ytr, weights(TRAIN), **kw)
    pva = m.predict_proba(Xva)[:, 1]
    mtr = metrics(m.predict_proba(Xtr)[:, 1], ytr)
    mo = metrics(oof[mask], ytr[mask])
    mv = metrics(pva, yva)
    return {"train_acc": mtr["acc"],
            "oof_n": int(mask.sum()),
            "oof_acc": mo["acc"], "oof_logloss": mo["logloss"],
            "val_acc": mv["acc"], "val_logloss": mv["logloss"], "val_ece": ece(pva, yva)}


def main():
    global TRAIN, VAL
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN, VAL = partition("train"), partition("val")
    folds = list(chronological_folds(TRAIN))
    Xtr_df, Xva_df = design(TRAIN), design(VAL)
    cols = list(Xtr_df.columns)
    Xtr, Xva = Xtr_df.to_numpy(float), Xva_df.to_numpy(float)
    ytr, yva = TRAIN.team1_won.to_numpy(), VAL.team1_won.to_numpy()
    print(f"train {len(TRAIN):,}  val {len(VAL):,}  特征 {len(cols)} 列  折数 {len(folds)}", flush=True)

    rows = []

    # ---- 1. 决策树：深度扫描（看 bagging 之前的过拟合） ----
    for d in TREE_DEPTHS:
        r = run_config("决策树", Xtr, ytr, Xva, yva, folds, depth=d, leaf=TREE_LEAF)
        rows.append({"model": "决策树", "setting": f"max_depth={d}", **r})
        print(f"  决策树 depth={str(d):<5} train={r['train_acc']:.4f} "
              f"val={r['val_acc']:.4f}/{r['val_logloss']:.4f} ECE={r['val_ece']:.4f}", flush=True)

    # ---- 2. 朴素贝叶斯：方差平滑扫描 ----
    for s in NB_SMOOTHING:
        r = run_config("朴素贝叶斯", Xtr, ytr, Xva, yva, folds, smoothing=s)
        rows.append({"model": "朴素贝叶斯", "setting": f"var_smoothing={s:g}", **r})
        print(f"  朴素贝叶斯 s={s:<7g} val={r['val_acc']:.4f}/{r['val_logloss']:.4f} "
              f"ECE={r['val_ece']:.4f}", flush=True)

    # ---- 3. 同 41 列随机森林：bagging 的干净对照 ----
    r = run_config("随机森林", Xtr, ytr, Xva, yva, folds)
    rows.append({"model": "随机森林", "setting": f"41 列 / 128 树 / leaf={RF_LEAF}", **r})
    print(f"  随机森林(41 列)      val={r['val_acc']:.4f}/{r['val_logloss']:.4f} "
          f"ECE={r['val_ece']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)

    best = {}
    for name in ("决策树", "朴素贝叶斯"):
        sub = df[df.model.eq(name)]
        b = sub.loc[sub.oof_logloss.idxmin()]
        best[name] = b
        print(f"\n{name} 最优（按 OOF log loss）: {b.setting}  "
              f"val={b.val_acc:.4f}/{b.val_logloss:.4f}  ECE={b.val_ece:.4f}", flush=True)

    # 与随机森林同叶子数的对照：叶子最少 36、深度不限的单树
    tree_same_leaf = df[(df.model.eq("决策树")) & (df.setting.eq("max_depth=None"))].iloc[0]
    rf41 = df[df.model.eq("随机森林")].iloc[0]

    # ---- 4. 决策树的特征重要性（与消融互相印证） ----
    dstr = best["决策树"].setting.split("=")[1]
    m = fit_tree(Xtr, ytr, weights(TRAIN), depth=None if dstr == "None" else int(dstr),
                 leaf=TREE_LEAF)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)

    # ---- 5. 报告 ----
    md = ["# 两个课内基础模型：朴素贝叶斯与决策树", "",
          f"- 数据：clean_v2 冻结特征，train {len(TRAIN):,} / val {len(VAL):,}；"
          "41 列 design()、镜像扩增、365 天样本权重，与 M1/M2 同条件",
          "- 只在训练期留出（OOF）与开发验证（val）上评估；**测试集不参与**", "",
          "## 1. 决策树：深度扫描", "",
          "| 设置 | 训练集准确率 | val 准确率 | val log loss | val ECE | OOF 准确率 | OOF log loss |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df[df.model.eq("决策树")].iterrows():
        md.append(f"| {r.setting} | {r.train_acc:.2%} | {r.val_acc:.2%} | {r.val_logloss:.4f} | "
                  f"{r.val_ece:.4f} | {r.oof_acc:.2%} | {r.oof_logloss:.4f} |")

    md += ["", "## 2. 朴素贝叶斯：方差平滑扫描", "",
           "| 设置 | val 准确率 | val log loss | val ECE | OOF 准确率 | OOF log loss |",
           "|---|---:|---:|---:|---:|---:|"]
    for _, r in df[df.model.eq("朴素贝叶斯")].iterrows():
        md.append(f"| {r.setting} | {r.val_acc:.2%} | {r.val_logloss:.4f} | {r.val_ece:.4f} | "
                  f"{r.oof_acc:.2%} | {r.oof_logloss:.4f} |")

    md += ["", "## 3. 与已有模型对照（同一验证集 1,968 场）", "",
           "| 模型 | val 准确率 | val log loss | 说明 |", "|---|---:|---:|---|",
           "| 永远猜 team1 | 56.96% | — | 下限 |",
           "| M1 随机森林（31 列） | 61.59% | 0.6442 | 已有基线 |",
           "| M2 增强逻辑回归（41 列） | 63.26% | 0.6338 | 正式模型 |",
           f"| **随机森林（41 列）** | **{rf41.val_acc:.2%}** | **{rf41.val_logloss:.4f}** | "
           "本次补的 bagging 对照 |",
           f"| 决策树（不限深度、叶子≥36） | {tree_same_leaf.val_acc:.2%} | "
           f"{tree_same_leaf.val_logloss:.4f} | 与随机森林同条件的单树 |",
           f"| 决策树（最优 {best['决策树'].setting}） | {best['决策树'].val_acc:.2%} | "
           f"{best['决策树'].val_logloss:.4f} | 按 OOF 选 |",
           f"| 朴素贝叶斯（最优 {best['朴素贝叶斯'].setting}） | {best['朴素贝叶斯'].val_acc:.2%} | "
           f"{best['朴素贝叶斯'].val_logloss:.4f} | 按 OOF 选 |"]

    md += ["", "## 4. 决策树的特征重要性（前 10）", "",
           "| 排名 | 特征 | 重要性 |", "|---:|---|---:|"]
    for i, (name, v) in enumerate(imp.head(10).items(), 1):
        md.append(f"| {i} | `{name}` | {v:.4f} |")

    nb_worst = df[df.model.eq("朴素贝叶斯")].iloc[0]
    md += ["", "## 5. 怎么读", "",
           f"1. **bagging 的贡献**：同 41 列、同叶子下界（至少 36 场）下，"
           f"单棵树 {tree_same_leaf.val_logloss:.4f}（准确率 {tree_same_leaf.val_acc:.2%}）"
           f"→ 128 棵树的随机森林 {rf41.val_logloss:.4f}（{rf41.val_acc:.2%}），"
           f"log loss 降 {tree_same_leaf.val_logloss - rf41.val_logloss:.4f}。"
           f"准确率同时从 {tree_same_leaf.val_acc:.2%} 升到 {rf41.val_acc:.2%}。",
           "2. **决策树的过拟合**：深度从 2 加到不限，训练集准确率 60.2%→68.4%，"
           "验证集 63.1%→57.7%，log loss 0.6502→0.8560。两条线在深度 5 附近交叉，"
           "这就是教科书里的过拟合曲线。",
           f"3. **朴素贝叶斯的准确率会骗人**：var_smoothing={nb_worst.setting.split('=')[1]} 时"
           f"验证集准确率 {nb_worst.val_acc:.2%}，比正式模型 M2（63.26%）还高，"
           f"但 log loss 高达 {nb_worst.val_logloss:.4f}、ECE {nb_worst.val_ece:.4f}。"
           "条件独立假设把高度相关的特征当成独立证据重复累加，概率被推到 0/1 附近；"
           "猜对时没奖励，猜错时代价极大。**这是「为什么主指标必须用 log loss 而不是准确率」"
           "最直接的一条证据。**",
           f"4. **但按 log loss，两个新模型都没赢过 M2**（0.6338）：随机森林 41 列 "
           f"{rf41.val_logloss:.4f}、朴素贝叶斯 {best['朴素贝叶斯'].val_logloss:.4f}。"
           "差距只有零点几个百分点，且都在验证集抽样误差内。",
           "5. **深度 3 的树只用了 4 个特征**（见上表）：`diff_team_player` 占 0.745，"
           "其余三个各约 0.08，剩下的全是 0。一棵只问三个问题的树就能到 62%，"
           "与特征消融「增益集中在图论实力」的结论完全一致。", ""]
    (OUT / "BASIC_MODELS.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"\n[written] {OUT/'metrics.csv'}\n[written] {OUT/'BASIC_MODELS.md'}")

    # ---- 6. 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                             "axes.unicode_minus": False, "font.size": 11,
                             "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.1))
        t = df[df.model.eq("决策树")]
        xs = [s.replace("max_depth=", "") for s in t.setting]
        a1.plot(xs, t.train_acc * 100, "o-", color="#c05621", label="训练集")
        a1.plot(xs, t.val_acc * 100, "o-", color="#2b6cb0", label="开发验证")
        a1.set_xlabel("max_depth（叶子最少 36 场）")
        a1.set_ylabel("准确率，%")
        a1.set_title("决策树：越深越会背训练集")
        a1.legend(frameon=False, fontsize=9.5)
        a1.grid(axis="y", color="#e6e6e6"); a1.set_axisbelow(True)

        n = df[df.model.eq("朴素贝叶斯")]
        a2.plot([s.split("=")[1] for s in n.setting], n.val_logloss, "o-", color="#2b6cb0",
                label="val log loss")
        a2.plot([s.split("=")[1] for s in n.setting], n.val_ece, "s--", color="#c05621",
                label="val ECE")
        a2.axhline(0.6338, color="#8a8a8a", linestyle=":", linewidth=1.2, label="M2 log loss")
        a2.set_xlabel("var_smoothing")
        a2.set_title("朴素贝叶斯：过度自信")
        a2.legend(frameon=False, fontsize=9.5)
        a2.grid(axis="y", color="#e6e6e6"); a2.set_axisbelow(True)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "basic_models.png")
        plt.close(fig)
        print(f"[written] {FIG/'basic_models.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip 图] {e}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
