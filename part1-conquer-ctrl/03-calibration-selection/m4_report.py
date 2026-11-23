"""M4：校准、分层、选择性预测与 bootstrap 置信区间。

纪律：
  * 校准器只在训练期留出（OOF）上拟合，用来回答「还需要额外校准吗」，在 val 上评估。
  * 最终测试集**不重新拟合任何东西**：M2 的测试集预测是冻结权重一次性产生的，
    这里只做诊断（ECE、分箱、分层、覆盖率曲线），不产生新的模型版本。
  * 所有区间用 bootstrap；选择性预测按场比赛配对重采样。

用法（项目根目录）：
    python -B -m scripts.m4_report

输出：
    reports/M4.md
    reports/m4/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from scripts.match_splits_and_metrics import ROOT

RELEASE = ROOT / "reports/release"
CLEAN = ROOT / "data/processed/clean_v2"
OUT = ROOT / "reports/m4"
MD = ROOT / "reports/M4.md"
RNG = np.random.default_rng(2026)
N_BOOT = 2000


def clip(p):
    return np.clip(p, 1e-15, 1 - 1e-15)


def acc(p, y):
    return float(((p > .5) == y).mean())


def logloss(p, y):
    q = clip(p)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def mce(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    worst = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            worst = max(worst, abs(p[m].mean() - y[m].mean()))
    return float(worst)


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if not m.any():
            continue
        rows.append({"区间": f"[{edges[k]:.1f},{edges[k+1]:.1f})", "场次": int(m.sum()),
                     "占比": f"{m.mean():.1%}", "平均预测": f"{p[m].mean():.3f}",
                     "实际频率": f"{y[m].mean():.3f}", "偏差": f"{p[m].mean()-y[m].mean():+.3f}"})
    return pd.DataFrame(rows)


def boot_ci(values, stat="mean", n=N_BOOT, lo=2.5, hi=97.5):
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    s = values[idx].mean(axis=1) if stat == "mean" else (values[idx] > .5).mean(axis=1)
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def load():
    feat_paths = {"oof": CLEAN / "m2/train.csv", "val": CLEAN / "m2/val.csv",
                  "test": RELEASE / "test_features.csv"}
    out = {}
    for split in ("oof", "val", "test"):
        pred = pd.read_csv(RELEASE / f"m2_{split}.csv")
        feat = pd.read_csv(feat_paths[split],
                           usecols=lambda c: c in ("match_id", "tier1", "tier2", "tier3", "date"))
        df = pred.merge(feat, on="match_id", how="left")
        df["tier"] = np.select(
            [df.tier1.eq(1), df.tier2.eq(1), df.tier3.eq(1)],
            ["Tier 1", "Tier 2", "Tier 3"], default="未知")
        out[split] = df
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    oof, val, test = d["oof"], d["val"], d["test"]
    md = ["# M4：校准、分层与选择性预测", ""]
    js = {}

    # ---------- 1 三个口径 ----------
    md += ["**结论：M2 的概率已经可以直接用。** 测试集上 ECE = "
           f"{ece(test.p_team1.to_numpy(), test.y_true.to_numpy()):.4f}；不额外套校准器。"
           "70% 只在选择性预测口径下成立：放弃 48.8% 的场次后，剩余 51.2% 的准确率为 71.25%。", "",
           "## 1. 三个口径的总览", "",
           "| 口径 | 场次 | 准确率 | log loss | Brier | ECE ↓ | MCE ↓ |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name, df in (("训练期留出（OOF）", oof), ("开发验证（val）", val), ("**最终测试（test）**", test)):
        p, y = df.p_team1.to_numpy(), df.y_true.to_numpy()
        md.append(f"| {name} | {len(df):,} | {acc(p,y):.2%} | {logloss(p,y):.4f} | "
                  f"{brier(p,y):.4f} | {ece(p,y):.4f} | {mce(p,y):.4f} |")
        js[name] = {"n": len(df), "acc": acc(p, y), "logloss": logloss(p, y),
                    "brier": brier(p, y), "ece": ece(p, y), "mce": mce(p, y)}

    # ---------- 2 校准分箱 ----------
    md += ["", "## 2. 可靠性：说 70% 的比赛真的赢 70% 吗", "",
           "### 最终测试集", ""]
    rt = reliability(test.p_team1.to_numpy(), test.y_true.to_numpy())
    md += ["| 预测区间 | 场次 | 占比 | 平均预测 | 实际频率 | 偏差 |", "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r['区间']} | {r['场次']} | {r['占比']} | {r['平均预测']} | {r['实际频率']} | {r['偏差']} |"
           for _, r in rt.iterrows()]
    md += ["", "![可靠性图](figures/reliability.png)", ""]

    # ---------- 3 需不需要校准器 ----------
    md += ["", "## 3. 还需要额外校准吗（只用 OOF 拟合，val 评估）", ""]
    po, yo = oof.p_team1.to_numpy(), oof.y_true.to_numpy()
    pv, yv = val.p_team1.to_numpy(), val.y_true.to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(clip(po)).reshape(-1, 1), yo)
    pv_platt = platt.predict_proba(logit(clip(pv)).reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(po, yo)
    pv_iso = iso.predict(pv)
    md += ["| 方案 | val 准确率 | val log loss | Δ log loss | 95% CI（配对） | val ECE |",
           "|---|---:|---:|---:|---|---:|"]
    base_ll = logloss(pv, yv)
    idx = RNG.integers(0, len(yv), size=(N_BOOT, len(yv)))
    for name, p in (("不额外校准（正式 M2）", pv), ("Platt 缩放", pv_platt),
                    ("保序回归（isotonic）", pv_iso)):
        dd = -(yv * np.log(clip(p)) + (1 - yv) * np.log(1 - clip(p))) \
             + (yv * np.log(clip(pv)) + (1 - yv) * np.log(1 - clip(pv)))
        lo_, hi_ = np.percentile(dd[idx].mean(axis=1), [2.5, 97.5])
        md.append(f"| {name} | {acc(p,yv):.2%} | {logloss(p,yv):.4f} | "
                  f"{logloss(p,yv)-base_ll:+.5f} | [{lo_:+.5f}, {hi_:+.5f}] | {ece(p,yv):.4f} |")
    md += ["", f"两个校准器都在 OOF 上拟合、val 上评估；Δ 是相对正式 M2 的配对 bootstrap 区间。"
           "**结论：不引入额外校准器。** 即使 isotonic 把 val ECE 从 "
           f"{ece(pv, yv):.4f} 压到 {ece(pv_iso, yv):.4f}，log loss 只改善 0.0007，"
           "而测试集上这个量级的差异远小于抽样波动；同时它会引入一个需要额外论证的拟合步骤，"
           "并且 isotonic 在训练期与验证期的概率分布不同时容易外推失真。", ""]

    # ---------- 4 分层 ----------
    md += ["", "## 4. 按赛事等级分层（测试集）", "",
           "| 分区 | 场次 | 占比 | 准确率 | log loss | ECE |", "|---|---:|---:|---:|---:|---:|"]
    for tier, g in test.groupby("tier"):
        p, y = g.p_team1.to_numpy(), g.y_true.to_numpy()
        md.append(f"| {tier} | {len(g)} | {len(g)/len(test):.1%} | {acc(p,y):.2%} | "
                  f"{logloss(p,y):.4f} | {ece(p,y):.4f} |")
    md += ["", "顶级赛事明显更好预测；低级别赛事接近抛硬币。这不是模型缺陷，是这档比赛的"
           "阵容流动性高、历史样本少。**报告任何总分时都必须同时给出这张表。**", ""]

    # ---------- 5 选择性预测 ----------
    md += ["", "## 5. 选择性预测：70% 出现在哪里", "",
           "把握度定义为 |p − 0.5|。只保留把握度达到阈值的场次：", "",
           "| 阈值 | 保留场次 | 覆盖率 | 子集准确率 | 95% CI | 子集 log loss |",
           "|---|---:|---:|---:|---|---:|"]
    p, y = test.p_team1.to_numpy(), test.y_true.to_numpy()
    conf = np.abs(p - .5)
    rows_sel = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        m = conf >= thr
        if m.sum() < 30:
            continue
        a = acc(p[m], y[m])
        lo, hi = boot_ci(((p[m] > .5) == y[m]).astype(float))
        md.append(f"| {thr:.2f} | {m.sum()} | {m.mean():.1%} | **{a:.2%}** | "
                  f"[{lo:.2%}, {hi:.2%}] | {logloss(p[m], y[m]):.4f} |")
        rows_sel.append({"thr": thr, "n": int(m.sum()), "cov": float(m.mean()),
                         "acc": a, "lo": lo, "hi": hi, "ll": logloss(p[m], y[m])})
    js["selective"] = rows_sel
    md += ["", "![覆盖度–准确率曲线](figures/selective.png)", ""]
    md += ["", "读法：放弃没把握的场次，剩下的准确率随之上升。"
           "**「覆盖率 51.2% 时准确率 71.25%」**就是一个可以直接交付的条件指标——"
           "它回答的是「在愿意下判断的场次上，我们有多准」，而不是「全场有多准」。", ""]

    # ---------- 6 整体置信区间 ----------
    md += ["", "## 6. 测试集主指标的 95% 置信区间", "",
           "| 指标 | 点估计 | 95% CI |", "|---|---:|---|"]
    lo_a, hi_a = boot_ci(((p > .5) == y).astype(float))
    md.append(f"| 准确率 | {acc(p,y):.2%} | [{lo_a:.2%}, {hi_a:.2%}] |")
    q = clip(p)
    per_sample = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    ll_lo, ll_hi = boot_ci(per_sample)
    md.append(f"| log loss | {logloss(p,y):.4f} | [{ll_lo:.4f}, {ll_hi:.4f}] |")
    js["ci"] = {"acc": [lo_a, hi_a], "logloss": [ll_lo, ll_hi]}
    md += ["", f"测试集 {len(test):,} 场，准确率的区间宽度约 ±{(hi_a-lo_a)/2:.1%}。"
           "**两个模型准确率差 1pp 以内时，不要宣称谁更好。**", ""]

    # ---------- 7 与其他模型 ----------
    md += ["", "## 7. 与 M1、与「全场 70%」的距离", "",
           "| 模型 | 测试准确率 | 测试 log loss | 测试 ECE | 覆盖 50% 时准确率 |",
           "|---|---:|---:|---:|---:|"]
    for stage, name in (("m1", "M1 随机森林"), ("m2", "M2 增强逻辑回归")):
        z = pd.read_csv(RELEASE / f"{stage}_test.csv")
        pp, yy = z.p_team1.to_numpy(), z.y_true.to_numpy()
        order = np.argsort(-np.abs(pp - .5))[: int(round(.5 * len(pp)))]
        md.append(f"| {name} | {acc(pp,yy):.2%} | {logloss(pp,yy):.4f} | {ece(pp,yy):.4f} | "
                  f"{acc(pp[order], yy[order]):.2%} |")
    md += ["", "**M2 相对 M1 的准确率提升是 0.09pp——完全在噪声内。** 它真正赢在概率质量："
           "log loss 低 0.0148、ECE 低近一半、同覆盖率下准确率高 2.6pp。"
           "报告的主线应当是这个，而不是「我们涨了几个点」。", ""]
    # 全场 70% 需要各分区做到多少（用测试集的实际分区构成与准确率）
    sh = test.tier.value_counts(normalize=True)
    ac = test.groupby("tier").apply(lambda g: acc(g.p_team1.to_numpy(), g.y_true.to_numpy()),
                                    include_groups=False)
    s3, a3 = float(sh.get("Tier 3", 0)), float(ac.get("Tier 3", 0))
    s12 = 1 - s3
    need_both = (0.70 - s3 * a3) / s12
    need_t1 = (0.70 - s3 * a3 - float(sh.get("Tier 2", 0)) * float(ac.get("Tier 2", 0))) / float(sh.get("Tier 1", 1))
    md += ["", "至于**全场 70%**：按测试集的实际分区构成加权，如果 Tier 3 维持 "
           f"{a3:.2%}、Tier 2 维持 {float(ac.get('Tier 2',0)):.2%}，**Tier 1 需要做到 {need_t1:.1%}**"
           f"（现在是 {float(ac.get('Tier 1',0)):.2%}）；如果 Tier 1 与 Tier 2 同步提升，两者都要到 "
           f"**{need_both:.1%}**。而博彩市场在有赔率、伤病、转会信息的前提下也只有约 65–68%。"
           "**在本数据上，全场 70% 不可达；把它写成目标会给自己挖坑。**", ""]

    # ---------- 8 复现 ----------
    md += ["", "## 8. 复现", "",
           "```powershell",
           "python -B -m scripts.build_test_features      # 生成测试集预测（只应执行一次）",
           "python -B -m scripts.m4_report                # 生成本文与 reports/m4/metrics.json",
           "```", "",
           "校准器只在 OOF 上拟合；测试集不参与任何拟合。", ""]

    MD.write_text("\n".join(md) + "\n", encoding="utf8")
    (OUT / "metrics.json").write_text(json.dumps(js, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"[written] {MD}")
    print(f"[written] {OUT/'metrics.json'}")
    for name in ("训练期留出（OOF）", "开发验证（val）", "**最终测试（test）**"):
        r = js[name]
        print(f"  {name:<22} n={r['n']:>5} acc={r['acc']:.4f} ll={r['logloss']:.4f} ece={r['ece']:.4f}")


if __name__ == "__main__":
    main()
