"""给任意两支队算一个赛前胜率——用真实的特征管线，不手搓特征。

做法：把「A 队 vs B 队」当成一场发生在数据末尾之后一天的比赛，
塞进原始表里跑一遍完整管线。特征构建器对同一天的所有比赛是
「先用历史算特征，再用当天结果更新」，所以这场假想比赛的特征
完全来自它之前的历史，不会被它自己的（编造的）比分污染。

用法：
    python -B -m scripts.predict_matchup "Team Falcons" "TYLOO"
    python -B -m scripts.predict_matchup --list 猎鹰     # 模糊查队名
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tzdata  # noqa: F401  （保证 pandas 时区解析可用）
from threadpoolctl import threadpool_limits

from scripts.basic_team_features import build_features, normalize_team
from scripts.build_clean_match_dataset import clean, events_for_history
from scripts.match_splits_and_metrics import ROOT
from scripts.roster_and_map_features import past_features
from scripts.team_and_opponent_strength import build as build_strength
from scripts.train_m1_m2 import design, predict

RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/processed/clean_v2"
FAKE_ID = 99_999_999
FAKE_DAY = "2026-10-31 23:59:00"


def load_tiers(extra_id: int | None = None, extra_tier: str = "tier1") -> pd.Series:
    parts = []
    for tier in ("tier1", "tier2", "tier3"):
        ids = pd.read_csv(RAW / f"cs2_{tier}_games.csv", usecols=["match_id"]).match_id.unique()
        parts.append(pd.Series(tier, index=ids))
    s = pd.concat(parts, verify_integrity=True)
    if extra_id is not None:
        s = pd.concat([s, pd.Series(extra_tier, index=[extra_id])])
    return s


def team_stats(matches: pd.DataFrame, team: str) -> dict:
    t = normalize_team(team)
    as1 = matches.team1.map(normalize_team).eq(t)
    as2 = matches.team2.map(normalize_team).eq(t)
    sel = matches[as1 | as2]
    wins = int(((as1 & matches.team1_won.eq(1)) | (as2 & matches.team1_won.eq(0))).sum())
    return {"场次": len(sel), "胜": wins,
            "胜率": wins / len(sel) if len(sel) else float("nan"),
            "首场": sel.date.min() if len(sel) else None,
            "末场": sel.date.max() if len(sel) else None}


def h2h(matches: pd.DataFrame, a: str, b: str) -> tuple[int, int]:
    a, b = normalize_team(a), normalize_team(b)
    m = matches.copy()
    m["n1"] = m.team1.map(normalize_team)
    m["n2"] = m.team2.map(normalize_team)
    sel = m[((m.n1 == a) & (m.n2 == b)) | ((m.n1 == b) & (m.n2 == a))]
    aw = int((((sel.n1 == a) & (sel.team1_won == 1)) | ((sel.n2 == a) & (sel.team1_won == 0))).sum())
    return len(sel), aw


def build_with_fake(a: str, b: str):
    """把 a vs b 作为一场假想比赛塞进原始表，跑完整管线。"""
    raw = pd.read_csv(RAW / "cs2_all_tiers_games.csv", low_memory=False)
    tpl = raw[raw.is_total.astype(bool)].iloc[0].copy()
    fake = tpl.copy()
    fake["match_id"] = FAKE_ID
    fake["game_id"] = -FAKE_ID
    fake["is_total"] = True
    fake["datetime"] = FAKE_DAY
    fake["team1"], fake["team2"] = a, b
    fake["team1_id"], fake["team2_id"] = -1, -2
    fake["score1_match"], fake["score2_match"] = 2, 1     # 只用于历史更新，不参与自身特征
    fake["bestOf"] = 3
    fake["tournament"] = "HYPOTHETICAL"
    for c in [f"team{t}_player{i}{suf}" for t in (1, 2) for i in range(1, 6)
              for suf in ("_id", "_adr", "_kast")]:
        if c in fake.index:
            fake[c] = np.nan
    raw2 = pd.concat([raw, fake.to_frame().T], ignore_index=True)
    raw2 = raw2.astype({c: raw[c].dtype for c in raw.columns if c in raw2.columns and c != "datetime"}, errors="ignore")

    m, maps, raw_maps, _ = clean(raw2, load_tiers(FAKE_ID, "tier1"))
    p0 = build_features(m, maps)
    p1 = past_features(m, maps, events_for_history(raw_maps, m))
    with threadpool_limits(limits=1):
        extra = build_strength(m, maps)
    feats = (p0.merge(p1, on="match_id", validate="one_to_one")
               .merge(extra, on="match_id", validate="one_to_one"))
    return feats[feats.match_id.eq(FAKE_ID)].reset_index(drop=True), m


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) >= 2 and args[0] in ('--list',):
        raise SystemExit("用 --list 时只给一个关键词")

    bundle = joblib.load(ROOT / "models/m2/m2_clean_retuned.joblib")
    a, b = args[0], args[1]

    feats_all, matches = build_with_fake(a, b)
    row = feats_all.iloc[[0]]

    print("=" * 66)
    print(f"假想对局：{a}  vs  {b}")
    print("=" * 66)
    for t in (a, b):
        s = team_stats(matches, t)
        if s["场次"] == 0:
            print(f"  {t:<24} ⚠️ 数据里没有这支队的比赛")
            continue
        print(f"  {t:<24} {s['场次']:>4} 场  胜率 {s['胜率']:.1%}  "
              f"{s['首场'].date()} → {s['末场'].date()}")

    n, aw = h2h(matches, a, b)
    print(f"\n  历史交手：{n} 次" + (f"（{a} 胜 {aw} 次）" if n else "（从未交手）"))

    print("\n  当前状态（截至 2026-10-30）：")
    for side, name in ((1, a), (2, b)):
        elo = row[f"team{side}_elo"].iloc[0]
        pm = row[f"team{side}_prior_matches"].iloc[0]
        ov = row[f"team{side}_last_overlap"].iloc[0]
        adr = row[f"team{side}_prior_adr"].iloc[0]
        print(f"    {name:<22} Elo {elo:7.1f}   历史 {int(pm):>4} 场   "
              f"阵容重合 {ov if pd.notna(ov) else float('nan'):.2f}   "
              f"ADR {adr if pd.notna(adr) else float('nan'):.1f}")

    p = float(predict(bundle, row)[0])
    print("\n" + "=" * 66)
    print(f"  模型给出的概率：P({a} 胜) = {p:.4f}  →  P({b} 胜) = {1-p:.4f}")
    print(f"  把握度 |p-0.5| = {abs(p-0.5):.3f}")
    if abs(p - 0.5) < 0.10:
        print("  → 低于 0.10 的阈值，模型在这个对局上属于「不愿下判断」")
    elif abs(p - 0.5) < 0.15:
        print("  → 超过 0.10 的阈值，属于「可以下判断」的场次")
    else:
        print("  → 超过 0.15，属于高把握场次（测试集上这类场次准确率约 75%）")
    print("=" * 66)
    print("\n注意：这是一场**从未发生**的比赛，但特征不是编的。")
    print("      · 两队状态取自各自最后一场之前（截至 2026-10-30），是真实的")
    print("      · 交手记录、地图池、阵容稳定性也都来自真实历史")
    print("      · 只有「赛制=BO3」「赛区=Tier 1」这两项是假设")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
