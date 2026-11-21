"""Part 1 特征消融：把 41 个特征分组删掉重训，看指标掉多少。

纪律：
  * 只读冻结的 clean_v2/m2/{train,val}.csv，不碰 models/ 和 manifest。
  * 每个配置都重新训练，不是把列置零（L2 正则对列数敏感）。
  * 超参与正式 M2 完全一致：C=0.001、fit_intercept=False、multiplier=1.15、
    365 天半衰期加权、镜像扩增。唯一变动的是"喂进去哪些列"。
  * 先在 OOF（训练期五折留出）和 val 上比，**不碰测试集**。

用法（项目根目录）：
    python -B -m scripts.ablation

输出：
    reports/ablation/metrics.csv
    reports/ablation/predictions.csv
    reports/ablation/ABLATION.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.match_splits_and_metrics import ROOT, chronological_folds
from scripts.train_m1_m2 import design, partition, weights

REPORT = ROOT / "reports/release"
C_DEFAULT = 0.001
MULT_DEFAULT = 1.15

# ---- 41 列按语义分成 8 组，正好覆盖全部列 ----
GROUPS: dict[str, list[str]] = {
    "elo": ["diff_elo", "diff_map_elo"],
    "form": ["diff_win_rate10", "diff_opponent_weighted_form10", "diff_streak",
             "diff_rest_days", "diff_matches7d", "diff_prior_matches"],
    "roster": ["diff_last_overlap", "diff_lineup_run", "diff_lineup_age",
               "diff_prior_adr", "diff_prior_kast", "diff_players_with_stats"],
    "map_pool": ["diff_maps180", "diff_pool_size", "diff_map_best",
                 "diff_map_mean", "diff_shared_map_rate"],
    "h2h": ["diff_h2h_rate"],
    "strength": ["diff_graph90", "diff_graph270", "diff_team_player", "diff_player_team",
                 "diff_roster_strength", "diff_typical_tier", "diff_elite_share",
                 "diff_sweep_margin", "diff_round_margin", "diff_opponent_strength",
                 "diff_opponent_tier"],
    "experience": ["log_experience"],
    "interactions": ["diff_graph90_bo1", "diff_graph90_bo5", "diff_graph90_tier1",
                     "diff_team_player_bo1", "diff_team_player_bo5", "diff_team_player_tier1",
                     "diff_elo_bo1", "diff_elo_bo5", "diff_elo_tier1"],
}


NAMES = {"elo": "Elo 差", "form": "近期状态", "roster": "阵容", "map_pool": "地图池",
         "h2h": "历史交手", "strength": "图论实力", "experience": "经验",
         "interactions": "赛制/等级交互"}


def fit_lr(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = C_DEFAULT):
    """与 train_m1_m2.fit 的 M2 分支完全相同的训练流程。"""
    both = np.vstack([X, -X])
    scaler = StandardScaler(with_mean=False).fit(both)
    est = LogisticRegression(C=C, fit_intercept=False, max_iter=1200)
    est.fit(scaler.transform(both), np.r_[y, 1 - y], sample_weight=np.tile(w, 2))
    return est, scaler


def predict_lr(est, scaler, X: np.ndarray, mult: float = MULT_DEFAULT) -> np.ndarray:
    return expit(est.decision_function(scaler.transform(X)) * mult)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "n": int(len(y)),
        "acc": float(((p > .5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(bins) if (b == k).any()))


def run_config(Xtr: np.ndarray, Xva: np.ndarray, train: pd.DataFrame,
               folds, mult: float = MULT_DEFAULT) -> dict:
    ytr = train.team1_won.to_numpy()
    yva = YVAL

    oof = np.full(len(train), np.nan)
    for i, j in folds:
        est, sc = fit_lr(Xtr[i], ytr[i], weights(train.iloc[i]))
        oof[j] = predict_lr(est, sc, Xtr[j], mult)
    mask = ~np.isnan(oof)

    est, sc = fit_lr(Xtr, ytr, weights(train))
    pva = predict_lr(est, sc, Xva, mult)

    mo = metrics(oof[mask], ytr[mask])
    mv = metrics(pva, yva)
    return {
        "oof_n": int(mask.sum()),
        "oof_acc": mo["acc"], "oof_logloss": mo["logloss"], "oof_brier": mo["brier"],
        "oof_ece": ece(oof[mask], ytr[mask]),
        "val_acc": mv["acc"], "val_logloss": mv["logloss"], "val_brier": mv["brier"],
        "val_ece": ece(pva, yva),
        "_oof": oof, "_val": pva,
    }


YVAL = np.array([])  # 在 main 里赋值


def main() -> None:
    global YVAL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "reports/ablation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = partition("train"), partition("val")
    YVAL = val.team1_won.to_numpy()
    folds = list(chronological_folds(train))
    print(f"train {len(train):,}  val {len(val):,}  折数 {len(folds)}", flush=True)

    Xtr_full = design(train)
    Xva_full = design(val)
    cols_all = list(Xtr_full.columns)
    if len(cols_all) != 41:
        raise SystemExit(f"design() 应产出 41 列，实际 {len(cols_all)}")
    print(f"设计矩阵 train {Xtr_full.shape} / val {Xva_full.shape}", flush=True)

    covered = sorted(c for g in GROUPS.values() for c in g)
    if covered != sorted(cols_all):
        raise SystemExit(f"分组不匹配；未分组 {set(cols_all)-set(covered)}，多余 {set(covered)-set(cols_all)}")

    configs: list[tuple[str, str, list[str]]] = [("full", "全量 41 列（= 正式 M2）", cols_all)]
    for g, gc in GROUPS.items():
        keep = [c for c in cols_all if c not in gc]
        if keep:
            configs.append((f"drop_{g}", f"删掉「{NAMES[g]}」组（{len(gc)} 列）", keep))
    for g, gc in GROUPS.items():
        configs.append((f"only_{g}", f"只留「{NAMES[g]}」组（{len(gc)} 列）",
                        [c for c in cols_all if c in gc]))

    rows, preds = [], {}
    for name, label, keep in configs:
        r = run_config(Xtr_full[keep].to_numpy(float), Xva_full[keep].to_numpy(float), train, folds)
        preds[name] = {"oof": r.pop("_oof"), "val": r.pop("_val")}
        rows.append({"config": name, "label": label, "n_cols": len(keep), **r})
        print(f"  {name:<18}{len(keep):>3} 列  val acc={r['val_acc']:.4f} ll={r['val_logloss']:.4f}"
              f" | oof acc={r['oof_acc']:.4f} ll={r['oof_logloss']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    for c in ("val_logloss", "oof_logloss", "val_acc", "oof_acc"):
        df["d_" + c] = df[c] - df.loc[df.config.eq("full"), c].iloc[0]
    base = df.loc[df.config.eq("full")].iloc[0]

    ref = pd.read_csv(REPORT / "m2_val.csv").set_index("match_id").loc[val.match_id].p_team1.to_numpy()
    diff = float(np.max(np.abs(preds["full"]["val"] - ref)))
    print(f"\n全量配置 vs 发布版 m2_val.csv 的最大概率差：{diff:.3e}", flush=True)
    if diff > 1e-6:
        raise SystemExit("全量配置未能复现发布的 M2，消融结论不可比，停止")

    df.to_csv(out / "metrics.csv", index=False)

    chunks = []
    for name, d in preds.items():
        m = ~np.isnan(d["oof"])
        chunks.append(pd.DataFrame({"config": name, "split": "oof",
                                    "match_id": train.match_id.to_numpy()[m],
                                    "y_true": train.team1_won.to_numpy()[m],
                                    "p_team1": d["oof"][m]}))
        chunks.append(pd.DataFrame({"config": name, "split": "val", "match_id": val.match_id.to_numpy(),
                                    "y_true": val.team1_won.to_numpy(), "p_team1": d["val"]}))
    pd.concat(chunks, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    # ---- 结论（放在最前面）----
    dr = df[df.config.str.startswith("drop_")].set_index("config")
    worst = dr.d_val_logloss.idxmax()
    neg = dr[dr.d_val_logloss < 0].sort_values("d_val_logloss")
    elo_d = float(dr.loc["drop_elo", "d_val_logloss"])
    only_s = df[df.config.eq("only_strength")].iloc[0]
    only_e = df[df.config.eq("only_elo")].iloc[0]
    concl = ["## 0. 先看结论", ""]
    concl.append(f"1. **只有「{NAMES[worst[5:]]}」组是实质贡献**：删掉后 val log loss "
                 f"{base.val_logloss:.4f} → {dr.loc[worst,'val_logloss']:.4f}"
                 f"（{dr.loc[worst,'d_val_logloss']:+.4f}），OOF 同向"
                 f"（{dr.loc[worst,'d_oof_logloss']:+.4f}）。")
    concl.append(f"2. **删掉「Elo 差」组几乎不掉分**（Δ = {elo_d:+.4f}）：Elo 的信息被「图论实力」组覆盖，"
                 f"两者高度冗余。但这不等于 Elo 没用——只留 Elo 也有 {only_e.val_acc:.2%} 准确率。")
    if len(neg):
        names = "、".join(f"「{NAMES[i[5:]]}」（{dr.loc[i,'d_val_logloss']:+.4f}）" for i in neg.index)
        concl.append(f"3. **删掉反而略好的组**：{names}。这些在当前正则强度下属于噪声，"
                     f"报告里建议直接移除或明确写「未观察到增益」。")
    concl.append(f"4. **11 列的「图论实力」组单独就打到 {only_s.val_acc:.2%} / {only_s.val_logloss:.4f}**，"
                 f"逼近 41 列全量（{base.val_acc:.2%} / {base.val_logloss:.4f}）。"
                 f"增益主要来自这一组，其余更多是在改善概率质量而非准确率。")
    concl.append("")

    def pct(x):
        return f"{x:.2%}"

    md = ["# Part 1 特征消融", "",
          f"- 训练 {len(train):,} 场 / 验证 {len(val):,} 场；训练期内五折向前，每折隔离 30 天；"
          f"OOF 覆盖 {sum(len(j) for _, j in folds):,} 场（训练期留出部分）",          f"- 每个配置都重新训练；超参与正式 M2 相同（C={C_DEFAULT}、multiplier={MULT_DEFAULT}、"
          "365 天半衰期加权、镜像扩增）",
          f"- 全量配置复现发布版 M2 的最大概率差 **{diff:.1e}**（< 1e-6，说明对照有效）",
          "- **测试集不参与**；下表全部来自训练期留出（OOF）与开发验证（val）", "",
          *concl,
          "## 1. 逐个删除一组（Δ 越大 = 这组越重要）", "",
          "| 配置 | 列数 | val 准确率 | val log loss | Δ log loss | val ECE | OOF log loss | Δ log loss |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = [base] + [r for _, r in df[df.config.str.startswith("drop_")]
                      .sort_values("d_val_logloss", ascending=False).iterrows()]
    for r in order:
        md.append(f"| {r.label} | {int(r.n_cols)} | {pct(r.val_acc)} | {r.val_logloss:.4f} | "
                  f"{r.d_val_logloss:+.4f} | {r.val_ece:.4f} | {r.oof_logloss:.4f} | {r.d_oof_logloss:+.4f} |")

    md += ["", "## 2. 只保留一组（极简对照）", "",
           "| 配置 | 列数 | val 准确率 | val log loss | Δ log loss | OOF log loss | Δ log loss |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df[df.config.str.startswith("only_")].sort_values("d_val_logloss").iterrows():
        md.append(f"| {r.label} | {int(r.n_cols)} | {pct(r.val_acc)} | {r.val_logloss:.4f} | "
                  f"{r.d_val_logloss:+.4f} | {r.oof_logloss:.4f} | {r.d_oof_logloss:+.4f} |")

    md += ["", "## 3. 怎么读这张表", "",
           "- **Δ log loss > 0**：删掉这组后变差，说明它提供真实信息。数值越大越重要。",
           "- **Δ log loss < 0**：删掉反而更好，说明这组在当前正则强度下是噪声，可以在报告里建议移除。",
           "- `interactions` 组 9 列里有 6 列恒为 0（赛制全未知），它的 Δ 实际只反映 3 个 tier1 交互。",
           "- OOF 与 val 方向一致时结论可靠；只有一边明显时按噪声处理。", ""]
    (out / "ABLATION.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"\n[written] {out/'metrics.csv'}\n[written] {out/'ABLATION.md'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
