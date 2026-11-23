"""生成 slides 用的三张图：消融瀑布、覆盖度-准确率、可靠性图。

用法（项目根目录）：
    python -B -m scripts.make_figures

输出：reports/figures/{ablation,selective,reliability}.png（200 dpi）
数据全部来自仓库内已冻结的产物，不重新训练、不重算预测。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from scripts.match_splits_and_metrics import ROOT

RELEASE = ROOT / "reports/release"
FIG = ROOT / "reports/figures"
BLUE, ORANGE, GRAY = "#2b6cb0", "#c05621", "#8a8a8a"
INK = "#1f2430"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def style(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b0b0b0")


def load_preds(stage: str) -> pd.DataFrame:
    return pd.read_csv(RELEASE / f"{stage}_test.csv")


# ---------------------------------------------------------------- 图 1 消融
def fig_ablation():
    df = pd.read_csv(ROOT / "reports/ablation/metrics.csv")
    d = df[df.config.str.startswith("drop_")].copy()
    d["name"] = d.label.str.extract(r"「(.+?)」")[0]
    d["ncol"] = d.label.str.extract(r"（(\d+) 列）")[0]
    d = d.sort_values("d_val_logloss")
    d = d[~d.name.isin(["Elo 差", "赛制/等级交互"]) | True]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(d))
    vals = d.d_val_logloss.to_numpy()
    colors = [BLUE if v > 0 else ORANGE for v in vals]
    ax.barh(y, vals, color=colors, height=0.62, alpha=0.9)
    for i, (v, lo, hi) in enumerate(zip(vals, d.d_val_lo, d.d_val_hi)):
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.1, solid_capstyle="butt")
        ax.plot([lo, lo], [i - .12, i + .12], color=INK, linewidth=1.1)
        ax.plot([hi, hi], [i - .12, i + .12], color=INK, linewidth=1.1)
        x = hi + 0.0004 if v >= 0 else lo - 0.0004
        ax.text(x, i, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}（{c} 列）" for n, c in zip(d.name, d.ncol)])
    ax.set_xlabel("删掉该组后 val log loss 的变化（正 = 这组有用）")
    ax.set_title("特征消融：增益集中在「图论实力」与「阵容」", pad=10)
    ax.set_xlim(min(vals.min(), d.d_val_lo.min()) - 0.0018,
                max(vals.max(), d.d_val_hi.max()) + 0.0018)
    style(ax)
    ax.text(0.99, 0.02, "误差线 = 配对 bootstrap 95% CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=GRAY)
    fig.savefig(FIG / "ablation.png")
    plt.close(fig)


# ------------------------------------------------------- 图 2 覆盖度-准确率
def coverage_curve(p, y, grid):
    conf = np.abs(p - .5)
    order = np.argsort(-conf)
    out = []
    for c in grid:
        k = max(1, int(round(c * len(p))))
        idx = order[:k]
        out.append(((p[idx] > .5) == y[idx]).mean())
    return np.array(out)


def fig_selective():
    grid = np.linspace(0.10, 1.0, 55)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    series = {}
    labels = {}
    for stage, name, color, lw in (("m1", "M1 随机森林", GRAY, 1.8),
                                   ("m2", "M2 增强逻辑回归", BLUE, 2.6)):
        d = load_preds(stage)
        p, y = d.p_team1.to_numpy(), d.y_true.to_numpy()
        acc = coverage_curve(p, y, grid)
        series[stage] = (p, y)
        labels[stage] = f"{name}（全场 {acc[-1]:.2%}）"
        ax.plot(grid * 100, acc * 100, color=color, linewidth=lw, label=labels[stage])
        ax.plot(100, acc[-1] * 100, "o", color=color, markersize=5, zorder=4)

    ax.axhline(70, color=ORANGE, linestyle="--", linewidth=1.3)
    ax.text(3.2, 70.8, "70%", color=ORANGE, fontsize=10.5, fontweight="bold", ha="left")

    p, y = series["m2"]
    conf = np.abs(p - .5)
    m = conf >= 0.10
    cov, a = m.mean(), ((p[m] > .5) == y[m]).mean()
    ax.plot(cov * 100, a * 100, "o", color=ORANGE, markersize=8, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(f"覆盖率 {cov:.1%} 时 {a:.2%}\n把握度 |p−0.5| ≥ 0.10",
                xy=(cov * 100, a * 100), xytext=(cov * 100 + 22, a * 100 - 11),
                fontsize=9.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"))

    ax.invert_xaxis()
    ax.set_xlabel("覆盖率（只保留最有把握的场次），%")
    ax.set_ylabel("子集准确率，%")
    ax.set_title("选择性预测：放弃没把握的场次，准确率随之上升", pad=10)
    ax.set_xlim(102, 8)
    ax.set_ylim(55, 92)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    style(ax)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.7)
    fig.savefig(FIG / "selective.png")
    plt.close(fig)


# ------------------------------------------------------------ 图 3 可靠性
def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if m.sum() >= 10:
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(rows)


def fig_reliability():
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(7.0, 4.8), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}, sharex=True)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, label="完美校准")
    for stage, name, color, mk in (("m1", "M1 随机森林", GRAY, "s"),
                                   ("m2", "M2 增强逻辑回归", BLUE, "o")):
        d = load_preds(stage)
        p, y = d.p_team1.to_numpy(), d.y_true.to_numpy()
        r = reliability(p, y)
        ax.plot(r[:, 0], r[:, 1], color=color, linewidth=1.8, label=name, alpha=0.95)
        ax.scatter(r[:, 0], r[:, 1], s=25 + r[:, 2] * 0.25, color=color, marker=mk,
                   edgecolor="white", linewidth=0.7, zorder=4, alpha=0.95)
    ax.set_ylabel("实际胜率")
    ax.set_title("可靠性：模型说 70% 时，实际真的赢 70% 吗（测试集 2,159 场）", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    style(ax)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.7)

    d = load_preds("m2")
    p = d.p_team1.to_numpy()
    axh.hist(p, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.35)
    axh.set_ylabel("场次", fontsize=9)
    axh.set_xlabel("模型给出的概率")
    axh.tick_params(labelsize=9)
    style(axh)
    axh.grid(axis="y", color="#f0f0f0", linewidth=0.7)
    fig.savefig(FIG / "reliability.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_ablation()
    fig_selective()
    fig_reliability()
    for f in sorted(FIG.glob("*.png")):
        print(f"  {f.name:<18}{f.stat().st_size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
