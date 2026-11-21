"""共用的比赛切分与评估规则：校验数据、生成时间折、对齐逐场预测并计算指标。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-04-01")
SNAPSHOT_STOP = pd.Timestamp("2026-11-01")
EXPECTED_COUNTS = {"train": 7260, "val": 1968, "test": 2159}
PREDICTION_COLUMNS = ["match_id", "y_true", "p_team1", "model"]


def csv_checksum(path: Path) -> str:
    """Ignore only the Windows/Unix line-ending difference introduced by Git."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_snapshot(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "data/v9_manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != "v9":
        raise ValueError("Only the frozen v9 snapshot is supported")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or csv_checksum(path) != expected:
            raise ValueError(f"v9 snapshot mismatch: {name}; restore the frozen file")
    return manifest


def split_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Assign each original match once; never split augmented rows."""
    result = matches.copy()
    if result.empty or result["match_id"].isna().any() or result["match_id"].duplicated().any():
        raise ValueError("Expected nonempty, unique original match IDs")
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None)
    if result["date"].isna().any() or (result["date"] >= SNAPSHOT_STOP).any():
        raise ValueError("Invalid date or match beyond the v9 snapshot")
    if not result["team1_won"].isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    result["split"] = np.select(
        [result["date"] < VAL_START, result["date"] < TEST_START],
        ["train", "val"], default="test",
    )
    return result.sort_values(["date", "match_id"]).reset_index(drop=True)


def load_matches(root: Path = ROOT) -> pd.DataFrame:
    verify_snapshot(root)
    matches = split_matches(pd.read_csv(root / "data/processed/matches.csv"))
    if matches["split"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Split sizes differ from the frozen v9 protocol")
    return matches


def augment_training(
    frame: pd.DataFrame,
    team_pairs: list[tuple[str, str]],
    difference_columns: list[str],
) -> pd.DataFrame:
    """只在拟合侧交换两队、翻转标签；预测时保留原始队伍顺序。"""
    if frame.empty or not frame["split"].eq("train").all():
        raise ValueError("Only a nonempty training partition can be augmented")
    if frame["match_id"].duplicated().any() or frame["is_swapped"].any():
        raise ValueError("Augment original matches once, after splitting")
    original = frame.copy()
    mirrored = frame.copy()
    for left, right in team_pairs:
        mirrored[left] = original[right]
        mirrored[right] = original[left]
    mirrored[difference_columns] = -original[difference_columns]
    mirrored["team1_won"] = 1 - original["team1_won"]
    mirrored["is_swapped"] = True
    return pd.concat([original, mirrored], ignore_index=True)


def chronological_folds(frame: pd.DataFrame, n_splits: int = 5, gap_days: int = 30):
    """按比赛日向前分折，返回行位置。先切分再扩增；间隔按自然日而非场数计算。"""
    if n_splits < 1 or gap_days < 0:
        raise ValueError("Invalid fold count or time gap")
    if not frame["split"].eq("train").all() or frame["match_id"].duplicated().any():
        raise ValueError("CV requires unique original training matches")
    if "is_swapped" in frame and frame["is_swapped"].any():
        raise ValueError("Split folds before augmentation")
    dates = pd.to_datetime(frame["date"])
    days = dates.dt.normalize()
    unique_days = np.sort(days.unique())
    if len(unique_days) < n_splits + 1:
        raise ValueError("Too few days for chronological CV")
    blocks = np.array_split(unique_days, n_splits + 1)
    for block in blocks[1:]:
        start = pd.Timestamp(block[0])
        train_idx = np.flatnonzero(dates < start - pd.Timedelta(days=gap_days))
        val_idx = np.flatnonzero(days.isin(block))
        if not len(train_idx) or not len(val_idx):
            raise ValueError("Empty CV partition after applying the time gap")
        yield train_idx, val_idx


def prediction_table(matches: pd.DataFrame, probabilities, model: str) -> pd.DataFrame:
    """Probabilities must follow the supplied original-match row order."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model name is required")
    if "is_swapped" in matches and matches["is_swapped"].any():
        raise ValueError("Predictions must use original team orientation")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(matches),):
        raise ValueError("Expected one probability per match in the supplied order")
    result = pd.DataFrame({
        "match_id": matches["match_id"].to_numpy(),
        "y_true": matches["team1_won"].to_numpy(),
        "p_team1": p,
        "model": model,
    })
    return validate_predictions(result, matches)


def validate_predictions(predictions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Reject missing/extra IDs, invalid probabilities, labels or mixed models."""
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction columns must be {PREDICTION_COLUMNS}")
    if predictions.empty or matches.empty:
        raise ValueError("Predictions and matches must be nonempty")
    if matches["match_id"].isna().any() or matches["match_id"].duplicated().any():
        raise ValueError("Reference match IDs must be unique and non-null")
    if predictions["match_id"].isna().any() or predictions["match_id"].duplicated().any():
        raise ValueError("Prediction match IDs must be unique and non-null")
    if set(predictions["match_id"]) != set(matches["match_id"]):
        raise ValueError("Prediction IDs must exactly match the evaluated partition")
    if predictions["model"].isna().any() or predictions["model"].nunique() != 1:
        raise ValueError("Each prediction table must contain one model")
    model = predictions["model"].iloc[0]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model name is required")
    aligned = predictions.set_index("match_id").loc[matches["match_id"]].reset_index()
    p = aligned["p_team1"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.array_equal(aligned["y_true"].to_numpy(), matches["team1_won"].to_numpy()):
        raise ValueError("Prediction labels disagree with the original matches")
    return aligned


def probability_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y_true"].to_numpy(dtype=float)
    p = predictions["p_team1"].to_numpy(dtype=float)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "acc": float(((p > 0.5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "brier": float(np.mean((p - y) ** 2)),
    }
