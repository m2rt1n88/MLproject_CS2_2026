"""Regression checks for leakage, match identity and partition boundaries."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.basic_team_features import (
    DIFFERENCES, FEATURE_COLUMNS, TEAM_PAIRS, build_features,
)
from scripts.match_splits_and_metrics import (
    EXPECTED_COUNTS, augment_training, chronological_folds, csv_checksum,
    load_matches, prediction_table, split_matches, validate_predictions, verify_snapshot,
)
from scripts import evaluate_elo_baseline


def example_matches():
    return split_matches(pd.DataFrame({
        "match_id": [1, 2, 3, 4],
        "date": ["2025-01-01 10:00", "2025-01-01 18:00", "2025-01-02 10:00", "2025-01-03 10:00"],
        "team1": [" A ", "a", "a", "a"], "team2": ["b", "c", "b", "c"],
        "team1_won": [1, 0, 1, 0], "bo": [3, np.nan, 1, 5], "tier": ["tier1"] * 4,
        "score1": [2, 0, 1, 2], "n_maps": [2, 2, 1, 5],
    }))


class SplitTests(unittest.TestCase):
    def test_exact_timestamp_boundaries(self):
        frame = pd.DataFrame({
            "match_id": [1, 2, 3, 4, 5], "team1_won": [0, 1, 0, 1, 0],
            "date": ["2025-09-30 23:59:59", "2025-10-01 00:00:00", "2026-03-31 23:59:59", "2026-04-01 00:00:00", "2026-10-31 23:59:59"],
        })
        self.assertEqual(split_matches(frame)["split"].tolist(), ["train", "val", "val", "test", "test"])
        frame.loc[4, "date"] = "2026-11-01 00:00:00"
        with self.assertRaises(ValueError):
            split_matches(frame)

    def test_duplicate_match_is_rejected(self):
        frame = example_matches()
        with self.assertRaises(ValueError):
            split_matches(pd.concat([frame, frame.iloc[:1]]))

    def test_frozen_data_and_cv_gap(self):
        matches = load_matches()
        self.assertEqual(matches["split"].value_counts().to_dict(), EXPECTED_COUNTS)
        train = matches[matches["split"].eq("train")].reset_index(drop=True)
        folds = list(chronological_folds(train))
        self.assertEqual(len(folds), 5)
        for train_idx, val_idx in folds:
            left, right = train.iloc[train_idx], train.iloc[val_idx]
            self.assertGreater(right["date"].min() - left["date"].max(), pd.Timedelta(days=30))
            self.assertTrue(set(left["match_id"]).isdisjoint(right["match_id"]))

    def test_snapshot_drift_is_rejected_and_line_endings_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            csv = root / "data/sample.csv"
            csv.write_bytes(b"id,label\n1,0\n")
            checksum = csv_checksum(csv)
            (root / "data/v9_manifest.json").write_text(json.dumps({
                "version": "v9", "files": {"data/sample.csv": checksum},
            }), encoding="utf-8")
            csv.write_bytes(b"id,label\r\n1,0\r\n")
            verify_snapshot(root)
            csv.write_bytes(b"id,label\n1,1\n")
            with self.assertRaises(ValueError):
                verify_snapshot(root)


class FeatureTests(unittest.TestCase):
    def test_current_and_future_outcomes_do_not_change_earlier_features(self):
        matches = example_matches()
        before = build_features(matches)
        changed = matches.copy()
        changed.loc[changed["match_id"].ge(3), "team1_won"] = 1 - changed.loc[changed["match_id"].ge(3), "team1_won"]
        after = build_features(changed)
        pd.testing.assert_frame_equal(before.iloc[:3][FEATURE_COLUMNS], after.iloc[:3][FEATURE_COLUMNS])
        self.assertNotEqual(before.iloc[3]["team1_elo"], after.iloc[3]["team1_elo"])

    def test_same_day_results_do_not_leak(self):
        matches = example_matches()
        before = build_features(matches)
        matches.loc[0, "team1_won"] = 0
        after = build_features(matches)
        pd.testing.assert_frame_equal(before.iloc[:2][FEATURE_COLUMNS], after.iloc[:2][FEATURE_COLUMNS])
        self.assertEqual(before.iloc[1]["team1_prior_matches"], 0)
        self.assertEqual(before.iloc[2]["team1_prior_matches"], 2)
        self.assertEqual(before.iloc[2]["team1_rest_days"], 1)

    def test_post_match_fields_are_excluded_and_ignored(self):
        matches = example_matches()
        before = build_features(matches)
        matches["score1"] = 999
        matches["n_maps"] = 999
        matches["maps_complete"] = True
        pd.testing.assert_frame_equal(before, build_features(matches))
        self.assertTrue({"score1", "score2", "n_maps", "maps_complete", "team1_won"}.isdisjoint(FEATURE_COLUMNS))
        self.assertNotIn("score1", before)

    def test_swapping_teams_matches_the_mirrored_features(self):
        matches = example_matches()
        original = build_features(matches)
        swapped = matches.copy()
        swapped["team1"] = matches["team2"]
        swapped["team2"] = matches["team1"]
        swapped["team1_won"] = 1 - matches["team1_won"]
        recalculated = build_features(swapped)
        augmented = augment_training(original, TEAM_PAIRS, DIFFERENCES)
        mirror = augmented.iloc[len(original):].reset_index(drop=True)
        pd.testing.assert_frame_equal(mirror[FEATURE_COLUMNS], recalculated[FEATURE_COLUMNS])
        np.testing.assert_array_equal(mirror["team1_won"], 1 - original["team1_won"])
        self.assertEqual(augmented.groupby("match_id")["team1_won"].sum().tolist(), [1] * 4)

    def test_augmentation_rejects_eval_and_already_augmented_rows(self):
        features = build_features(example_matches())
        augmented = augment_training(features, TEAM_PAIRS, DIFFERENCES)
        with self.assertRaises(ValueError):
            augment_training(augmented, TEAM_PAIRS, DIFFERENCES)
        with self.assertRaises(ValueError):
            list(chronological_folds(augmented))
        features["split"] = "val"
        with self.assertRaises(ValueError):
            augment_training(features, TEAM_PAIRS, DIFFERENCES)


class PredictionTests(unittest.TestCase):
    def test_alignment_by_match_id(self):
        matches = example_matches()
        predictions = prediction_table(matches, [0.2, 0.3, 0.8, 0.9], "demo")
        aligned = validate_predictions(predictions.iloc[::-1], matches)
        pd.testing.assert_frame_equal(aligned, predictions)

    def test_invalid_exports_are_rejected(self):
        matches = example_matches()
        valid = prediction_table(matches, [0.2, 0.3, 0.8, 0.9], "demo")
        invalid = []
        for column, value in [("p_team1", np.nan), ("p_team1", 1.2), ("p_team1", np.inf), ("y_true", 0), ("match_id", 999), ("model", "different")]:
            candidate = valid.copy()
            candidate.loc[0, column] = value
            invalid.append(candidate)
        invalid.extend([valid.iloc[:-1], pd.concat([valid, valid.iloc[:1]])])
        for candidate in invalid:
            with self.subTest(candidate=candidate.to_dict("list")), self.assertRaises(ValueError):
                validate_predictions(candidate, matches)
        with self.assertRaises(ValueError):
            prediction_table(matches, [0.5], "demo")
        mirrored = build_features(matches)
        mirrored["is_swapped"] = True
        with self.assertRaises(ValueError):
            prediction_table(mirrored, [0.5] * len(mirrored), "demo")


class BaselineTests(unittest.TestCase):
    def run_baseline(self, final_test):
        toy = example_matches()
        toy.loc[2, "date"] = pd.Timestamp("2025-10-01")
        toy.loc[3, "date"] = pd.Timestamp("2026-04-01")
        toy = split_matches(toy)
        toy["t1"], toy["t2"] = toy["team1"], toy["team2"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["evaluate_elo_baseline.py"] + (["--final-test"] if final_test else [])
            with patch.object(evaluate_elo_baseline, "ROOT", root), patch.object(evaluate_elo_baseline, "load", return_value=toy), patch.object(evaluate_elo_baseline, "elo_run", wraps=evaluate_elo_baseline.elo_run) as run, patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                evaluate_elo_baseline.main()
            self.assertTrue((root / "reports/predictions/val/elo.csv").exists())
            self.assertEqual((root / "reports/predictions/test/elo.csv").exists(), final_test)
            self.assertTrue(all(call.args[1]["split"].eq("val").all() for call in run.call_args_list[:5]))
            if final_test:
                self.assertEqual(run.call_count, 6)
                self.assertTrue(run.call_args.args[1]["split"].eq("test").all())
                self.assertEqual(set(run.call_args.args[0]["split"]), {"train", "val"})
            else:
                self.assertEqual(run.call_count, 5)

    def test_default_never_scores_test(self):
        self.run_baseline(final_test=False)

    def test_final_test_uses_validation_selection(self):
        self.run_baseline(final_test=True)


if __name__ == "__main__":
    unittest.main()
