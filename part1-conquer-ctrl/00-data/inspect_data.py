r"""看一眼我们的数据到底长什么样。

三层结构：原始 CSV → 清洗后的表 → 建模用的特征表。
运行后屏幕打印报告，同时写入同目录的 DATA_PROFILE.txt。

用法（项目根目录）：
    python -B -m part1.00-data.inspect_data   # 或直接 python 路径
实际请用：
    python -B F:\Git\MLproject_CS2_2026\part1-conquer-ctrl\00-data\inspect_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Git\MLproject_CS2_2026")
RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/processed/clean_v2"
OUT = Path(__file__).with_name("DATA_PROFILE.txt")

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def head(title: str) -> None:
    say()
    say("=" * 74)
    say(title)
    say("=" * 74)


def main() -> None:
    say("CS2 项目数据速查")
    say(f"仓库：{ROOT}")

    # ---------------- 第 1 层 ----------------
    head("第 1 层 · 原始数据 data/raw/（从 Kaggle 下载，不要改）")
    for f, desc in [
        ("cs2_all_tiers_games.csv", "主表：一行可能是「一整场」也可能是「一张图」"),
        ("cs2_tier1_games.csv", "Tier 1 赛事（和主表有重叠，用 match_id 筛）"),
        ("cs2_tier2_games.csv", "Tier 2 赛事"),
        ("cs2_tier3_games.csv", "Tier 3 赛事"),
        ("players.csv", "选手 id ↔ 名字"),
        ("teams.csv", "队伍 id ↔ 名字"),
        ("tournaments.csv", "赛事名单"),
    ]:
        p = RAW / f
        if not p.exists():
            continue
        n = sum(1 for _ in p.open(encoding="utf-8")) - 1
        cols = len(pd.read_csv(p, nrows=1).columns)
        say(f"  {f:<28} {n:>7,} 行  {cols:>3} 列   {desc}")

    raw = pd.read_csv(RAW / "cs2_all_tiers_games.csv", nrows=3)
    say()
    say(f"  主表为什么有 {raw.shape[1]} 列？因为它把 5 类信息堆在了一起：")
    say("    · 比赛级   match_id / datetime / tournament / team1 / team2 / score*_match")
    say("    · 地图级   is_total / map_name / score*_game / game_id")
    say("    · 赛制     bestOf / games_played")
    say("    · 阵容     20 个 team{1,2}_player{1..5}_id")
    say("    · 选手数据 60 个 ……_kills / deaths / assists / adr / kast / kddiff")
    say()
    say("  ⚠️ 关键坑：is_total=True 的行是「整场汇总」，is_total=False 是「单张地图」。")
    say("     直接用 team1_win 当标签会错（行级标记，聚合行不可靠）。")

    # ---------------- 第 2 层 ----------------
    head("第 2 层 · 清洗后的表 data/processed/clean_v2/（一行一场 / 一行一图）")
    m = pd.read_csv(CLEAN / "matches.csv", parse_dates=["date"])
    mp = pd.read_csv(CLEAN / "maps.csv", parse_dates=["date"])
    q = pd.read_csv(CLEAN / "match_quality.csv")
    say(f"  matches.csv        {len(m):>7,} 行 × {m.shape[1]:>2} 列   ← 一行一整场比赛")
    say(f"  maps.csv           {len(mp):>7,} 行 × {mp.shape[1]:>2} 列   ← 一行一张地图")
    say(f"  match_quality.csv  {len(q):>7,} 行 × {q.shape[1]:>2} 列   ← 每场的数据质量标记")
    say()
    say(f"  时间范围      {m.date.min().date()} → {m.date.max().date()}")
    say(f"  比赛总数      {len(m):,}")
    say(f"  队伍数        {pd.concat([m.team1, m.team2]).str.strip().str.casefold().nunique():,}")
    say(f"  每队比赛中位数 {int(pd.concat([m.team1, m.team2]).value_counts().median())}")
    say(f"  队 1 胜率     {m.team1_won.mean():.4f}   ← 这就是「排序偏置」的来源")
    say()
    say("  时间切分（写死，不要改）：")
    for s in ("train", "val", "test"):
        d = m[m.split.eq(s)]
        say(f"    {s:<6} {len(d):>6,} 场  {d.date.min().date()} → {d.date.max().date()}")
    say()
    say("  赛区分档：")
    for t, n in m.tier.value_counts().items():
        say(f"    {t:<6} {n:>6,} 场  ({n/len(m):.1%})")
    say()
    say("  数据完整度（这三个数字决定了能做什么特征）：")
    say(f"    有地图明细的比赛    {(m.n_maps > 0).mean():.1%}   ← 地图池特征只覆盖一半")
    say(f"    阵容完整（10 人齐） "
        f"{m[[f'team{t}_player{i}_id' for t in (1, 2) for i in range(1, 6)]].notna().all(axis=1).mean():.1%}")
    say(f"    BO 赛制已知          0.0%   ← 清洗时为了保证不泄漏，全部置为未知")

    # ---------------- 第 3 层 ----------------
    head("第 3 层 · 建模用的特征表 clean_v2/m2/{train,val}.csv（一行一场，全是赛前信息）")
    tr = pd.read_csv(CLEAN / "m2/train.csv")
    va = pd.read_csv(CLEAN / "m2/val.csv")
    say(f"  train.csv  {len(tr):>6,} 行 × {tr.shape[1]} 列")
    say(f"  val.csv    {len(va):>6,} 行 × {va.shape[1]} 列")
    say()
    say("  86 列分成四类：")
    meta = ["match_id", "date", "split", "team1", "team2", "team1_won", "is_swapped"]
    diffs = [c for c in tr.columns if c.startswith("diff_")]
    ctx = [c for c in tr.columns if c in ("bo1", "bo3", "bo5", "bo_missing", "tier1", "tier2", "tier3")]
    side = [c for c in tr.columns if c not in meta + diffs + ctx]
    say(f"    · 元数据与标签   {len(meta):>2} 列  {meta}")
    say(f"    · 单侧队伍特征   {len(side):>2} 列  如 team1_elo / team2_elo（成对出现）")
    say(f"    · 差值特征       {len(diffs):>2} 列  如 diff_elo = team1_elo − team2_elo")
    say(f"    · 赛制/赛区标记  {len(ctx):>2} 列  {ctx}")
    say()
    say("  ⚠️ 模型只吃 diff_ 和交互项，不吃单侧特征——因为交换两队时差值会变号，单侧不会。")

    # ---------------- 实际长什么样 ----------------
    head("实际长什么样：三行样本")
    say()
    say("① 一场比赛（matches.csv，取队名最短的一场看得清）")
    row = m.loc[m.team1.str.len().idxmin()]
    show = ["match_id", "date", "team1", "team2", "score1", "score2", "tier", "n_maps", "team1_won"]
    say("   " + "  ".join(f"{c}={row[c]}" for c in show if c in m.columns))
    say()
    say("② 一张地图（maps.csv）")
    r2 = mp.iloc[0]
    say("   " + "  ".join(f"{c}={r2[c]}" for c in mp.columns))
    say()
    say("③ 一条特征（m2/train.csv，挑 8 个列看）")
    r3 = tr[tr["diff_last_overlap"].notna()].iloc[0]   # 挑一条特征齐全的
    for c in ["match_id", "date", "team1", "team2", "team1_won",
              "team1_elo", "team2_elo", "diff_elo", "diff_last_overlap",
              "team1_prior_adr", "tier1", "bo_missing"]:
        if c in tr.columns:
            say(f"     {c:<22} = {r3[c]}")
    say()
    e1, e2, de = r3["team1_elo"], r3["team2_elo"], r3["diff_elo"]
    say(f"   读法：diff_elo = {e1:.0f} − {e2:.0f} = {de:.0f}，意思是队 1 的 Elo 比队 2 高 {de:.0f} 分。")
    say("   所有 ……_prior_adr / ……_last_overlap 都是「这场之前」的历史，绝不包含本场。")

    # ---------------- 特征分组 ----------------
    head("41 个建模特征的 8 个语义组（消融实验按这个分组）")
    groups = {
        "Elo 差": ["diff_elo", "diff_map_elo"],
        "近期状态": ["diff_win_rate10", "diff_opponent_weighted_form10", "diff_streak",
                     "diff_rest_days", "diff_matches7d", "diff_prior_matches"],
        "阵容": ["diff_last_overlap", "diff_lineup_run", "diff_lineup_age",
                 "diff_prior_adr", "diff_prior_kast", "diff_players_with_stats"],
        "地图池": ["diff_maps180", "diff_pool_size", "diff_map_best", "diff_map_mean",
                   "diff_shared_map_rate"],
        "历史交手": ["diff_h2h_rate"],
        "图论实力": ["diff_graph90", "diff_graph270", "diff_team_player", "diff_player_team",
                     "diff_roster_strength", "diff_typical_tier", "diff_elite_share",
                     "diff_sweep_margin", "diff_round_margin", "diff_opponent_strength",
                     "diff_opponent_tier"],
        "经验": ["log_experience"],
        "赛制/等级交互": ["diff_graph90_bo1", "diff_graph90_bo5", "diff_graph90_tier1",
                          "diff_team_player_bo1", "diff_team_player_bo5", "diff_team_player_tier1",
                          "diff_elo_bo1", "diff_elo_bo5", "diff_elo_tier1"],
    }
    total = 0
    for g, cs in groups.items():
        total += len(cs)
        say(f"  {g:<12} {len(cs):>2} 列   {', '.join(c[:28] for c in cs[:4])}"
            + ("  …" if len(cs) > 4 else ""))
    say(f"  {'合计':<12} {total:>2} 列   （design() 实际产出 41 列）")
    say()
    say("  实测覆盖度：")
    for c in ("diff_last_overlap", "diff_prior_adr", "diff_maps180"):
        if c in tr.columns:
            say(f"    {c:<22} 非空 {tr[c].notna().mean():.1%}")

    # ---------------- 三个常见误解 ----------------
    head("三个常见误解")
    say("  ① 「11,394 场」不是我们的样本量。原始 11,394，清洗掉平局/缺比分后是 11,387。")
    say("  ② 「98 列」不等于 98 个特征。模型只用 41 列，且全是「差值」。")
    say("  ③ 「有选手数据」不等于全覆盖。地图行 81% 有 ADR/KAST，比赛级只有 42%。")
    say()
    say(f"报告已写入：{OUT}")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
