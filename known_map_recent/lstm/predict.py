"""Load a saved single-map LSTM and predict one map from history strictly before `day`.

CLI example:
    python -B -m known_map_recent.lstm.predict --team1 "team vitality" --team2 "team spirit" --map mirage --day 2027-01-15

Team names must match clean_v3 team keys exactly (lowercase); no alias guessing is done.
An unseen name is a cold-start team and is reported via team_known=False and `warnings`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from known_map_v2.runtime import ROOT
from .artifacts import load_run
from .dataset import day_blocks, encode_maps, ingest
from .model import make_batch, predict_proba
from .state import HistoryState, SEQ_LEN, SEQ_NAMES, as_day, norm_map

MODEL_DIR = ROOT / 'models/lstm_release/known_map_lstm_base_seed2026'
DEV_END = pd.Timestamp('2026-04-01')      # rows on/after this day were never trained on


def state_before(day, feature_set='base', source=None, metrics=None):
    """HistoryState with every day < `day` ingested; metric events only for non-base feature sets."""
    from known_map_recent.features import load_source
    from .metrics import load_metric_events
    source = source or load_source()
    matches, maps, players = source[:3]
    if feature_set != 'base' and metrics is None:
        metrics = load_metric_events(source)[0]
    if feature_set == 'base':
        metrics = None
    state = HistoryState(feature_set)
    for block in day_blocks(matches, maps, players, pd.Timestamp(as_day(day)), metrics):
        ingest(state, block)
    return state


class Predictor:
    def __init__(self, model_dir=MODEL_DIR, source=None, metrics=None):
        self.model, self.vocab, self.scaler, self.config, self.fingerprint = load_run(model_dir)
        self.feature_set = self.config['feature_set']
        # Resolve source early so it's available for both metric loading and state building
        if source is None:
            from known_map_recent.features import load_source
            source = load_source()
        self.source = source
        self.metrics = metrics
        self._state, self._day = None, None
        # Non-base feature sets require metrics; verify the loaded snapshot matches runtime data
        if self.feature_set != 'base':
            if self.metrics is None:
                from .metrics import load_metric_events
                self.metrics, runtime_fp = load_metric_events(self.source)
                saved_fp = self.fingerprint.get('metric_snapshot')
                if not saved_fp:
                    raise ValueError(f'{model_dir}: feature set {self.feature_set!r} has no metric_snapshot fingerprint')
                problems = []
                for key in ('folder', 'sha256'):
                    if saved_fp.get(key) != runtime_fp.get(key):
                        problems.append(f'metric_snapshot {key} differs: saved {saved_fp.get(key)} '
                                        f'vs runtime {runtime_fp.get(key)}')
                if problems:
                    raise ValueError(
                        f'{model_dir} was trained on a different metric snapshot: ' + '; '.join(problems) +
                        f'. Retrain or point to the correct snapshot folder.')


    def state(self, day):
        day = as_day(day)
        if self._day != day:
            self._state = state_before(day, self.feature_set, self.source, self.metrics)
            self._day = day
        return self._state

    def predict_map(self, team1, team2, map_name, day, lineup1=None, lineup2=None):
        state = self.state(day)
        qa = state.query(team1, map_name, day, lineup1)
        qb = state.query(team2, map_name, day, lineup2)
        n_static = len(state.static_names)
        static = np.stack([qa[0], qb[0]])[None]
        seq = np.zeros((1, 2, SEQ_LEN, len(SEQ_NAMES)), dtype=np.float32)
        for s, q in enumerate((qa, qb)):
            seq[0, s, :len(q[1])] = q[1]
        lengths = np.array([[len(qa[2]), len(qb[2])]])
        static, seq = self.scaler.transform(static, seq, lengths)
        seq_maps = encode_maps([[qa[2], qb[2]]], self.vocab)
        name = norm_map(map_name)
        target = np.array([self.vocab.index(name) if name in self.vocab else 0])
        p = float(predict_proba(self.model, make_batch(static, seq, seq_maps, lengths, target))[0])
        warnings = [f'{side} "{q[3]["team"]}" has no history before {as_day(day)}: cold start '
                    '(check the exact clean_v3 team key)' for side, q in (('team1', qa), ('team2', qb))
                    if not q[3]['team_known']]
        if name not in self.vocab:
            warnings.append(f'map "{name}" not in training vocabulary: using <unk> embedding')
        day_obj = as_day(day)
        if day_obj < DEV_END.date():
            warnings.append(f'day {day_obj} is before the test period ({DEV_END.date()}); model weights were '
                            'trained on data from this period — this is NOT an out-of-sample evaluation')
        return dict(p_team1=p, p_team2=1 - p, map_name=name, map_known=name in self.vocab,
                    feature_set=self.feature_set, history_before=str(day_obj),
                    team1=qa[3], team2=qb[3], warnings=warnings)


def predict_map(team1, team2, map_name, day, lineup1=None, lineup2=None, model_dir=MODEL_DIR):
    return Predictor(model_dir).predict_map(team1, team2, map_name, day, lineup1, lineup2)


def _ids(text):
    return None if text is None else [int(x) for x in text.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--team1', required=True)
    ap.add_argument('--team2', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--day', required=True, help='match day; only earlier days are used as history')
    ap.add_argument('--lineup1', help='comma separated player ids')
    ap.add_argument('--lineup2', help='comma separated player ids')
    ap.add_argument('--model-dir', default=str(MODEL_DIR))
    a = ap.parse_args()
    out = predict_map(a.team1, a.team2, a.map, a.day, _ids(a.lineup1), _ids(a.lineup2), a.model_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
