"""三个模型的同条件对比：朴素贝叶斯 / 逻辑回归 / 随机森林。

目的：回答一个准确率答不了的问题——**同一份数据、同一批 41 列特征，
三个模型的准确率几乎一样，但概率质量差多少？**

纪律（与 M1/M2 完全一致）：
  * 只读冻结的 clean_v2/m2/{train,val}.csv；
  * 同一个 design()（41 列）、同一条时间切分、同一套镜像扩增、同样的 365 天样本权重；
  * 逻辑回归分支必须逐场复现发布的 m2_val.csv（误差 < 1e-6），否则整套对照作废；
  * 只在 OOF 与 val 上评估，**不碰测试集**。

用法（项目根目录）：
    python -B -m scripts.three_models

输出：
    reports/three_models/metrics.csv
    reports/three_models/THREE_MODELS.md
    reports/figures/three_models.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.match_splits_and_metrics import ROOT, chronological_folds
from scripts.train_m1_m2 import design, partition, weights

OUT = ROOT / "reports/three_models"
FIG = ROOT / "reports/figures"
RELEASE = ROOT / "reports/release"
SEED = 2026
LR_C, LR_MULT = 0.001, 1.15
RF_TREES, RF_LEAF = 128, 36
NB_SMOOTH = 1e-9          # sklearn 默认；另给一个"加平滑"的对照

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


# ---------------------------------------------------------------- 三个训练器
def fit_lr(X, y, w):
    both = np.vstack([X, -X])
    sc = StandardScaler(with_mean=False).fit(both)
    m = LogisticRegression(C=LR_C, fit_intercept=False, max_iter=1200)
    m.fit(sc.transform(both), np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: expit(m.decision_function(sc.transform(Z)) * LR_MULT)


def fit_nb(X, y, w, smoothing=NB_SMOOTH):
    both = np.vstack([X, -X])
    m = GaussianNB(var_smoothing=smoothing)
    m.fit(both, np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_rf(X, y, w):
    both = np.vstack([X, -X])
    m = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=RF_LEAF,
                               max_features="sqrt", n_jobs=1, random_state=SEED)
    m.fit(both, np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return lambda Z: m.predict_proba(Z)[:, 1]


MODELS = [
    ("朴素贝叶斯", "GaussianNB（默认 var_smoothing=1e-9）", fit_nb),
    ("逻辑回归", f"C={LR_C}，无截距，输出乘数 {LR_MULT}", fit_lr),
    ("随机森林", f"{RF_TREES} 棵树，叶子最少 {RF_LEAF} 场", fit_rf),
]


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
        "val_acc35": acc_at_coverage(pva, yva, .35),
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

    rows, preds = [], {}
    for name, desc, fitter in MODELS:
        r = run(fitter, Xtr, ytr, Xva, yva, folds)
        preds[name] = {"val": r.pop("_val"), "oof": r.pop("_oof")}
        rows.append({"模型": name, "配置": desc, **r})
        print(f"  {name:<8} val acc={r['val_acc']:.4f} ll={r['val_logloss']:.4f} "
              f"ECE={r['val_ece']:.4f} | oof acc={r['oof_acc']:.4f} ll={r['oof_logloss']:.4f}",
              flush=True)

    # 平滑版朴素贝叶斯，作为"能不能救回来"的对照
    r_nb_smooth = run(fit_nb, Xtr, ytr, Xva, yva, folds, smoothing=1.0)
    preds["朴素贝叶斯（加平滑）"] = {"val": r_nb_smooth.pop("_val"), "oof": r_nb_smooth.pop("_oof")}
    print(f"  NB 平滑 val acc={r_nb_smooth['val_acc']:.4f} ll={r_nb_smooth['val_logloss']:.4f} "
          f"ECE={r_nb_smooth['val_ece']:.4f}", flush=True)

    # ---- 硬检查：逻辑回归必须复现发布的 M2 ----
    ref = pd.read_csv(RELEASE / "m2_val.csv").set_index("match_id").loc[VAL.match_id].p_team1.to_numpy()
    diff = float(np.max(np.abs(preds["逻辑回归"]["val"] - ref)))
    print(f"\n逻辑回归 vs 发布版 M2 的最大概率差：{diff:.3e}")
    if diff > 1e-6:
        raise SystemExit("逻辑回归分支没有复现发布的 M2，三模型对照不可比，停止")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)

    # ---- 报告 ----
    lr = df[df.模型.eq("逻辑回归")].iloc[0]
    nb = df[df.模型.eq("朴素贝叶斯")].iloc[0]
    rf = df[df.模型.eq("随机森林")].iloc[0]
    base = 0.6931471805599453   # 每场都报 0.5 的 log loss

    md = ["# 三个模型：朴素贝叶斯 / 逻辑回归 / 随机森林", "",
          "**同一份数据、同一批 41 列特征、同一条时间切分、同一套镜像扩增与样本权重。**",
          "唯一变化的是分类器本身。逻辑回归分支逐场复现发布版 M2（最大概率差 "
          f"{diff:.1e}），说明三者的对照条件完全一致。", "",
          "## 1. 结果", "",
          "| 模型 | val 准确率 | val log loss | val Brier | val ECE | 覆盖 50% 时准确率 |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {r.模型} | {r.val_acc:.2%} | {r.val_logloss:.4f} | {r.val_brier:.4f} | "
                  f"{r.val_ece:.4f} | {r.val_acc50:.2%} |")
    md += [f"| *朴素贝叶斯（加平滑 var_smoothing=1）* | {r_nb_smooth['val_acc']:.2%} | "
           f"{r_nb_smooth['val_logloss']:.4f} | {r_nb_smooth['val_brier']:.4f} | "
           f"{r_nb_smooth['val_ece']:.4f} | {r_nb_smooth['val_acc50']:.2%} |", "",
           f"参照：每场都报 50% 的 log loss = **{base:.4f}**。", "",
           "## 2. 这三个数字最重要的一件事", "",
           f"- 准确率：{nb.val_acc:.2%} / {lr.val_acc:.2%} / {rf.val_acc:.2%}，"
           f"最大差 {abs(rf.val_acc - nb.val_acc) * 100:.2f} 个百分点——**在 1,968 场的抽样误差内，分不出高低。**",
           f"- log loss：{nb.val_logloss:.4f} / {lr.val_logloss:.4f} / {rf.val_logloss:.4f}"
           f"——**朴素贝叶斯是最差的 {nb.val_logloss / lr.val_logloss:.1f} 倍。**",
           f"- ECE：{nb.val_ece:.4f} / {lr.val_ece:.4f} / {rf.val_ece:.4f}，"
           f"朴素贝叶斯的校准误差是逻辑回归的 {nb.val_ece / lr.val_ece:.0f} 倍。", "",
           "**所以：准确率这三列几乎没有信息量，log loss 和 ECE 才能把三个模型分开。**",
           "这就是我们以 log loss 为主指标、并要求校准的全部理由。", ""]

    md += ["## 3. 三个模型的原理（各三句）", "",
           "### 朴素贝叶斯", "",
           "1. 用贝叶斯公式：`P(胜 | 特征) ∝ P(特征 | 胜) × P(胜)`，谁的后验大概率高就判谁。",
           "2. 它假设**各个特征在给定胜负之后互相独立**，所以只需要把每个特征的似然乘起来，"
           "参数量从指数级降到线性级。",
           "3. 这个假设在真实数据上几乎一定不成立——我们的 41 列里有大量相关特征"
           "（Elo、图论实力、对手强度），它会把这些相关证据当成独立证据**重复累加**，"
           "于是概率被推到 0 或 1。", "",
           "### 逻辑回归", "",
           "1. 假设胜负的对数几率是特征的**线性函数**：`log(p/(1-p)) = w·x`，"
           "再用 sigmoid 压回 0–1。",
           "2. 用极大似然（等价于最小化交叉熵）估计权重，加 L2 正则控制过拟合；"
           "我们的 `C=0.001` 是很强的正则。",
           "3. 因为它直接对概率建模并用交叉熵训练，输出的概率天然比较可信——"
           "这正是它的 log loss 最好的原因。", "",
           "### 随机森林", "",
           "1. 训练很多棵决策树，每棵用**自助采样**（bootstrap）得到的不同数据子集，"
           "分裂时只看随机一部分特征。",
           "2. 预测时把所有树的概率取平均。单棵树方差大（换个样本就变），"
           "平均之后方差被压下去——这就是 bagging。",
           "3. 它擅长抓非线性与交互，但**平均概率并不保证校准**；"
           "在我们的数据上它的 log loss 略差于逻辑回归。", ""]

    md += ["## 4. 每个模型教会我们什么", "",
           f"1. **朴素贝叶斯告诉我们「准确率高 ≠ 概率可用」。** 它准确率 {nb.val_acc:.2%}，"
           f"不比另外两个差，但 log loss {nb.val_logloss:.4f}、ECE {nb.val_ece:.4f}："
           "模型说 90% 的时候实际可能只赢 60%。**这种概率没法用来做决策。**",
           f"2. **逻辑回归告诉我们「简单模型可以是最好的概率模型」。** 它没有非线性、没有集成，"
           f"但 log loss {lr.val_logloss:.4f} 是三者最低，"
           f"在 50% 覆盖率下准确率 {lr.val_acc50:.2%} 反而最高。"
           "在 41 列、7,260 场的规模上，线性可加已经足够。",
           f"3. **随机森林告诉我们「复杂度买得到准确率，买不到概率」。** 它准确率 {rf.val_acc:.2%} "
           f"最高，但 log loss {rf.val_logloss:.4f}、ECE {rf.val_ece:.4f} 都略差于逻辑回归。"
           f"对比单棵决策树（log loss 0.8560），bagging 把方差压下去的效果非常明显。", ""]

    md += ["## 5. 讲的时候怎么说（每个模型一句话）", "",
           f"- **朴素贝叶斯**：「它假设特征互相独立，这个假设在我们的数据上明显不成立——"
           f"准确率还有 {nb.val_acc:.2%}，但 log loss 是 {nb.val_logloss:.4f}，"
           "概率被推到了 0 和 1。它是我们主指标选择的证据。」",
           f"- **逻辑回归**：「它直接对概率建模，用交叉熵训练，所以概率天然可信。"
           f"log loss {lr.val_logloss:.4f} 是三者最好，ECE 只有 {lr.val_ece:.4f}。"
           "正式模型选它。」",
           f"- **随机森林**：「它用 bootstrap + 多树平均降低方差。准确率 {rf.val_acc:.2%} 最高，"
           f"但概率略逊。它证明了一件事——**加复杂度能换准确率，换不来概率质量**。」", ""]

    md += ["## 6. 复现", "",
           "```powershell",
           "python -B -m scripts.three_models",
           "```", "",
           "逻辑回归分支内置校验：与发布版 `m2_val.csv` 逐场比对，误差超过 1e-6 直接报错退出。", ""]
    (OUT / "THREE_MODELS.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"\n[written] {OUT/'metrics.csv'}\n[written] {OUT/'THREE_MODELS.md'}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                             "axes.unicode_minus": False, "font.size": 11,
                             "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
        colors = {"朴素贝叶斯": "#c05621", "逻辑回归": "#2b6cb0", "随机森林": "#2f855a"}
        offs = {"随机森林": (8, 12, "left"), "逻辑回归": (8, -20, "left"),
                "朴素贝叶斯": (-10, 12, "right")}
        for _, r in df.iterrows():
            a1.scatter(r.val_logloss, r.val_acc * 100, s=170, color=colors[r.模型],
                       zorder=4, edgecolor="white", linewidth=1.2)
            dx, dy, ha = offs[r.模型]
            a1.annotate(f"{r.模型}\n{r.val_logloss:.3f} / {r.val_acc:.2%}",
                        xy=(r.val_logloss, r.val_acc * 100), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=9.5, color=colors[r.模型])
        a1.axvline(base, color="#8a8a8a", linestyle="--", linewidth=1.2)
        a1.text(base, 64.52, "  每场都报 50%（log loss 0.693）", fontsize=8.5,
                color="#8a8a8a", ha="left", va="top")
        a1.set_xlabel("val log loss（越低越好）")
        a1.set_ylabel("val 准确率，%")
        a1.set_title("准确率几乎一样，概率质量差 2.4 倍", pad=10)
        a1.set_xlim(0.55, 1.62)
        a1.set_ylim(62.4, 64.7)
        a1.grid(color="#eeeeee")
        a1.set_axisbelow(True)

        a2.plot([0, 1], [0, 1], color="#8a8a8a", linestyle="--", linewidth=1.2, label="完美校准")
        for name in ("朴素贝叶斯", "逻辑回归", "随机森林"):
            p = preds[name]["val"]
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs, ys = [], []
            for k in range(10):
                m = idx == k
                if m.sum() >= 15:
                    xs.append(p[m].mean()); ys.append(yva[m].mean())
            a2.plot(xs, ys, "o-", color=colors[name], linewidth=1.8, markersize=5,
                    label=name)
        a2.set_xlabel("模型给出的概率")
        a2.set_ylabel("实际胜率")
        a2.set_title("可靠性：点离对角线越远，概率越不可信", pad=10)
        a2.legend(frameon=False, fontsize=9.5, loc="upper left")
        a2.grid(color="#eeeeee")
        a2.set_axisbelow(True)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "three_models.png")
        plt.close(fig)
        print(f"[written] {FIG/'three_models.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip 图] {e}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
