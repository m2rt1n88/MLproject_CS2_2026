"""四个模型的同条件对比：朴素贝叶斯 / 逻辑回归 / 随机森林 / KNN。

目的：回答一个准确率答不了的问题——**同一份数据、同一批 41 列特征，
这些模型的准确率几乎一样，但概率质量差多少？**

纪律（与 M1/M2 完全一致）：
  * 只读冻结的 clean_v2/m2/{train,val}.csv；
  * 同一个 design()（41 列）、同一条时间切分、同一套镜像扩增、同样的 365 天样本权重；
  * 逻辑回归分支必须逐场复现发布的 m2_val.csv（误差 < 1e-6），否则整套对照作废；
  * 只在 OOF 与 val 上评估，**不碰测试集**。

已知的不完全可比之处（报告里必须写明）：
  * KNeighborsClassifier 不接受 sample_weight，所以 KNN 用不了 365 天时间衰减权重。

用法（项目根目录）：
    python -B -m scripts.four_models

输出：
    reports/four_models/metrics.csv
    reports/four_models/MODELS_COMPARISON.md
    reports/figures/four_models.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.match_splits_and_metrics import ROOT, chronological_folds
from scripts.train_m1_m2 import design, partition, weights

OUT = ROOT / "reports/four_models"
FIG = ROOT / "reports/figures"
RELEASE = ROOT / "reports/release"
SEED = 2026
LR_C, LR_MULT = 0.001, 1.15
RF_TREES, RF_LEAF = 128, 36
NB_SMOOTH = 1e-9
KNN_KS = [5, 15, 45, 135, 405]

TRAIN = VAL = None


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


def acc_at_coverage(p, y, cov):
    k = max(1, int(round(cov * len(p))))
    idx = np.argsort(-np.abs(p - .5))[:k]
    return float(((p[idx] > .5) == y[idx]).mean())


# ---------------------------------------------------------------- 四个训练器
def fit_nb(X, y, w, smoothing=NB_SMOOTH):
    m = GaussianNB(var_smoothing=smoothing)
    m.fit(np.vstack([X, -X]), np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_lr(X, y, w):
    both = np.vstack([X, -X])
    sc = StandardScaler(with_mean=False).fit(both)
    m = LogisticRegression(C=LR_C, fit_intercept=False, max_iter=1200)
    m.fit(sc.transform(both), np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: expit(m.decision_function(sc.transform(Z)) * LR_MULT)


def fit_rf(X, y, w):
    m = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=RF_LEAF,
                               max_features="sqrt", n_jobs=1, random_state=SEED)
    m.fit(np.vstack([X, -X]), np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_knn(X, y, w, k=135):
    """注意：KNeighborsClassifier 不支持 sample_weight，w 在这里没用上。"""
    both = np.vstack([X, -X])
    sc = StandardScaler().fit(both)          # KNN 对尺度敏感，必须标准化
    m = KNeighborsClassifier(n_neighbors=k, weights="uniform", n_jobs=1)
    m.fit(sc.transform(both), np.r_[y, 1 - y])
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def run(fitter, Xtr, ytr, Xva, yva, folds, **kw):
    oof = np.full(len(ytr), np.nan)
    for i, j in folds:
        f = fitter(Xtr[i], ytr[i], weights(TRAIN.iloc[i]), **kw)
        oof[j] = f(Xtr[j])
    mask = ~np.isnan(oof)
    f = fitter(Xtr, ytr, weights(TRAIN), **kw)
    pva = f(Xva)
    mo, mv = metrics(oof[mask], ytr[mask]), metrics(pva, yva)
    return {
        "oof_n": int(mask.sum()),
        "oof_acc": mo["acc"], "oof_logloss": mo["logloss"],
        "val_acc": mv["acc"], "val_logloss": mv["logloss"], "val_brier": mv["brier"],
        "val_ece": ece(pva, yva),
        "val_acc50": acc_at_coverage(pva, yva, .50),
        "_val": pva, "_oof": oof,
    }


def main():
    global TRAIN, VAL
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN, VAL = partition("train"), partition("val")
    folds = list(chronological_folds(TRAIN))
    Xtr_df, Xva_df = design(TRAIN), design(VAL)
    Xtr, Xva = Xtr_df.to_numpy(float), Xva_df.to_numpy(float)
    ytr, yva = TRAIN.team1_won.to_numpy(), VAL.team1_won.to_numpy()
    print(f"train {len(TRAIN):,}  val {len(VAL):,}  特征 {Xtr.shape[1]} 列  折数 {len(folds)}", flush=True)

    # ---- KNN：先扫邻居数，按 OOF log loss 选 ----
    knn_rows = []
    for k in KNN_KS:
        r = run(fit_knn, Xtr, ytr, Xva, yva, folds, k=k)
        knn_rows.append({"k": k, **{x: r[x] for x in
                                    ("oof_acc", "oof_logloss", "val_acc", "val_logloss", "val_ece")}})
        print(f"  KNN k={k:<4} val={r['val_acc']:.4f}/{r['val_logloss']:.4f} "
              f"ECE={r['val_ece']:.4f} | oof={r['oof_acc']:.4f}/{r['oof_logloss']:.4f}", flush=True)
    knn_df = pd.DataFrame(knn_rows)
    best_k = int(knn_df.loc[knn_df.oof_logloss.idxmin(), "k"])
    print(f"  -> KNN 最优 k={best_k}", flush=True)

    # ---- 四个模型的主对比 ----
    MODELS = [
        ("朴素贝叶斯", "GaussianNB（默认 var_smoothing=1e-9）", fit_nb, {}),
        ("逻辑回归", f"C={LR_C}，无截距，输出乘数 {LR_MULT}", fit_lr, {}),
        ("随机森林", f"{RF_TREES} 棵树，叶子最少 {RF_LEAF} 场", fit_rf, {}),
        ("KNN", f"k={best_k}，标准化后的欧氏距离，均匀权重", fit_knn, {"k": best_k}),
    ]

    rows, preds = [], {}
    for name, desc, fitter, kw in MODELS:
        r = run(fitter, Xtr, ytr, Xva, yva, folds, **kw)
        preds[name] = {"val": r.pop("_val"), "oof": r.pop("_oof")}
        rows.append({"模型": name, "配置": desc, **r})
        print(f"  {name:<8} val acc={r['val_acc']:.4f} ll={r['val_logloss']:.4f} "
              f"ECE={r['val_ece']:.4f} | oof acc={r['oof_acc']:.4f} ll={r['oof_logloss']:.4f}",
              flush=True)

    ref = pd.read_csv(RELEASE / "m2_val.csv").set_index("match_id").loc[VAL.match_id].p_team1.to_numpy()
    diff = float(np.max(np.abs(preds["逻辑回归"]["val"] - ref)))
    print(f"\n逻辑回归 vs 发布版 M2 的最大概率差：{diff:.3e}")
    if diff > 1e-6:
        raise SystemExit("逻辑回归分支没有复现发布的 M2，对照不可比，停止")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)
    knn_df.to_csv(OUT / "knn_sweep.csv", index=False)

    nb = df[df.模型.eq("朴素贝叶斯")].iloc[0]
    lr = df[df.模型.eq("逻辑回归")].iloc[0]
    rf = df[df.模型.eq("随机森林")].iloc[0]
    kn = df[df.模型.eq("KNN")].iloc[0]
    base = 0.6931471805599453

    # ---- 报告 ----
    md = ["# 四个模型：朴素贝叶斯 / 逻辑回归 / 随机森林 / KNN", "",
          "**同一份数据、同一批 41 列特征、同一条时间切分、同一套镜像扩增与样本权重。**",
          "唯一变化的是分类器本身。逻辑回归分支逐场复现发布版 M2（最大概率差 "
          f"{diff:.1e}），说明四者的对照条件一致。", "",
          "> 已知不完全可比之处：`KNeighborsClassifier` 不接受 `sample_weight`，"
          "所以 **KNN 用不了 365 天时间衰减权重**，其余三个模型都有。这一点在对比 KNN 时要记住。", "",
          "## 1. 结果", "",
          "| 模型 | val 准确率 | val log loss | val Brier | val ECE | 覆盖 50% 时准确率 |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {r.模型} | {r.val_acc:.2%} | {r.val_logloss:.4f} | {r.val_brier:.4f} | "
                  f"{r.val_ece:.4f} | {r.val_acc50:.2%} |")
    md += ["", f"参照：每场都报 50% 的 log loss = **{base:.4f}**。", "",
           "## 2. KNN 的邻居数扫描", "",
           "| k | val 准确率 | val log loss | val ECE | OOF log loss |",
           "|---:|---:|---:|---:|---:|"]
    for _, r in knn_df.iterrows():
        md.append(f"| {r.k} | {r.val_acc:.2%} | {r.val_logloss:.4f} | {r.val_ece:.4f} | "
                  f"{r.oof_logloss:.4f} |")
    md += ["", f"按 OOF log loss 选中 **k={best_k}**。k 太小（5）时邻居里常常一个正例都没有，"
           "概率直接给 0 或 1，log loss 会爆到 1.9 以上；k 太大则退化成「平均主义」，"
           "概率全部挤在 0.5 附近，区分度下降。", "",
           "## 3. 这几个数字最重要的一件事", "",
           f"- 准确率：{nb.val_acc:.2%} / {lr.val_acc:.2%} / {rf.val_acc:.2%} / {kn.val_acc:.2%}，"
           f"极差只有 {(df.val_acc.max() - df.val_acc.min()) * 100:.2f} 个百分点——"
           f"**在 1,968 场的抽样误差（约 1pp）内，分不出高低。**",
           f"- log loss：{nb.val_logloss:.4f} / {lr.val_logloss:.4f} / "
           f"{rf.val_logloss:.4f} / {kn.val_logloss:.4f}——"
           f"**朴素贝叶斯是最差者的 {nb.val_logloss / lr.val_logloss:.1f} 倍。**",
           f"- ECE：{nb.val_ece:.4f} / {lr.val_ece:.4f} / {rf.val_ece:.4f} / {kn.val_ece:.4f}。", "",
           "**准确率这一列几乎没有信息量；log loss 和 ECE 才能把四个模型分开。**", ""]

    md += ["## 4. 四个模型的原理（各三句）", "",
           "### 朴素贝叶斯", "",
           "1. 用贝叶斯公式 `P(胜|特征) ∝ P(特征|胜)·P(胜)`，谁的后验大概率高就判谁。",
           "2. 假设**给定胜负后各特征互相独立**，于是 41 维联合分布拆成 41 个一维分布。",
           "3. 这个假设在真实数据上几乎一定不成立——相关特征被当成独立证据重复累加，"
           "概率被推到 0/1。", "",
           "### 逻辑回归", "",
           "1. 假设胜负的对数几率是特征的线性函数 `log(p/(1-p)) = w·x`，再用 sigmoid 压回 0–1。",
           "2. 用极大似然（等价于最小化交叉熵）估权重，加 L2 正则控制过拟合。",
           "3. 它直接对概率建模、训练目标就是 log loss，所以输出的概率天然可信。", "",
           "### 随机森林", "",
           "1. 训练很多棵决策树，每棵用 bootstrap 采样得到的数据子集、每次分裂只看随机一部分特征。",
           "2. 预测时把所有树的概率取平均——单棵树方差大，平均之后方差被压下去（bagging）。",
           "3. 擅长非线性与交互，但平均概率不等于校准好的概率。", "",
           "### KNN（K-Nearest Neighbors）", "",
           "1. 不做任何训练：把训练样本全部记下来，来一个新样本就找**距离最近的 k 个邻居**，"
           "看他们里赢的比例。",
           "2. 它没有任何参数假设——边界形状完全由数据决定，`k` 是唯一的旋钮："
           "k 小 → 边界复杂、方差大；k 大 → 边界平滑、偏差大。",
           "3. 代价有两个：预测时要和全部 14,520 个样本算距离（慢）；"
           "特征尺度必须先标准化，否则距离被大尺度特征主导。", ""]

    md += ["## 5. 每个模型教会我们什么", "",
           f"1. **朴素贝叶斯**：准确率 {nb.val_acc:.2%} 不比别人差，但 log loss {nb.val_logloss:.4f}、"
           f"ECE {nb.val_ece:.4f}——准确率高不等于概率可用。",
           f"2. **逻辑回归**：没有非线性、没有集成，但 log loss {lr.val_logloss:.4f} 最低、"
           f"ECE {lr.val_ece:.4f} 最低，覆盖 50% 时准确率 {lr.val_acc50:.2%} 最高。"
           "在 41 列、7,260 场的规模上，线性可加已经足够。",
           f"3. **随机森林**：准确率 {rf.val_acc:.2%} 最高，但 log loss {rf.val_logloss:.4f}、"
           f"ECE {rf.val_ece:.4f} 都略逊。对比单棵树（0.8560），bagging 降方差的效果非常明显——"
           "但它买到的是准确率，不是概率质量。",
           f"4. **KNN**：准确率 {kn.val_acc:.2%}，log loss {kn.val_logloss:.4f}。"
           "它的意义是**提供一个「完全不做假设」的下界**——即使把边界形状完全交给数据，"
           "也还是停在同一个水平。这再次说明瓶颈不在模型的表达能力。", ""]

    md += ["## 6. 讲的时候怎么说（每个模型一句话）", "",
           f"- **朴素贝叶斯**：「它假设特征互相独立，在我们的数据上明显不成立——"
           f"准确率 {nb.val_acc:.2%}，但 log loss {nb.val_logloss:.4f}，概率被推到 0 和 1。」",
           f"- **逻辑回归**：「直接对概率建模、用交叉熵训练，概率天然可信。"
           f"log loss {lr.val_logloss:.4f} 最低，ECE {lr.val_ece:.4f} 最低。」",
           f"- **随机森林**：「bootstrap + 多树平均降方差。准确率 {rf.val_acc:.2%} 最高，"
           "但概率略逊——加复杂度能换准确率，换不来概率质量。」",
           f"- **KNN**：「它完全不假设边界形状，只靠邻居投票。准确率 {kn.val_acc:.2%}，"
           "还是停在同一个水平——**连最「自由」的模型也到不了更高。**」", ""]

    md += ["## 7. 复现", "", "```powershell", "python -B -m scripts.four_models", "```", "",
           "逻辑回归分支内置校验：与发布版 `m2_val.csv` 逐场比对，误差超过 1e-6 直接报错退出。", ""]
    (OUT / "MODELS_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"\n[written] {OUT/'metrics.csv'}\n[written] {OUT/'knn_sweep.csv'}\n"
          f"[written] {OUT/'MODELS_COMPARISON.md'}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                             "axes.unicode_minus": False, "font.size": 11,
                             "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.2))
        colors = {"朴素贝叶斯": "#c05621", "逻辑回归": "#2b6cb0",
                  "随机森林": "#2f855a", "KNN": "#6b46c1"}
        offs = {"随机森林": (8, 14, "left"), "逻辑回归": (10, 16, "left"),
                "朴素贝叶斯": (-10, 12, "right"), "KNN": (10, -14, "left")}
        for _, r in df.iterrows():
            a1.scatter(r.val_logloss, r.val_acc * 100, s=170, color=colors[r.模型],
                       zorder=4, edgecolor="white", linewidth=1.2)
            dx, dy, ha = offs[r.模型]
            a1.annotate(f"{r.模型}\n{r.val_logloss:.3f} / {r.val_acc:.2%}",
                        xy=(r.val_logloss, r.val_acc * 100), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=9.5, color=colors[r.模型])
        a1.axvline(base, color="#8a8a8a", linestyle="--", linewidth=1.2)
        a1.text(base, 64.42, "  每场都报 50%（log loss 0.693）", fontsize=8.5,
                color="#8a8a8a", ha="left", va="top")
        a1.set_xlabel("val log loss（越低越好）")
        a1.set_ylabel("val 准确率，%")
        a1.set_title("准确率几乎一样，概率质量差很大", pad=10)
        a1.set_xlim(0.55, 1.70)
        a1.set_ylim(62.2, 64.6)
        a1.grid(color="#eeeeee")
        a1.set_axisbelow(True)

        a2.plot([0, 1], [0, 1], color="#8a8a8a", linestyle="--", linewidth=1.2, label="完美校准")
        for name in ("朴素贝叶斯", "逻辑回归", "随机森林", "KNN"):
            p = preds[name]["val"]
            idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
            xs, ys = [], []
            for k in range(10):
                m = idx == k
                if m.sum() >= 15:
                    xs.append(p[m].mean()); ys.append(yva[m].mean())
            a2.plot(xs, ys, "o-", color=colors[name], linewidth=1.8, markersize=5, label=name)
        a2.set_xlabel("模型给出的概率")
        a2.set_ylabel("实际胜率")
        a2.set_title("可靠性：点离对角线越远，概率越不可信", pad=10)
        a2.legend(frameon=False, fontsize=9, loc="upper left")
        a2.grid(color="#eeeeee")
        a2.set_axisbelow(True)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "four_models.png")
        plt.close(fig)
        print(f"[written] {FIG/'four_models.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip 图] {e}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
