# lstm_release — Single-Map LSTM Checkpoints (seed 2026)

Three reproducible configurations, all trained on clean_v3 (dates < 2026-04-01).
Weights are committed here; retrain goes to a separate directory.

## Directories

| Directory | Feature set | Static features | Val logloss | Dir size |
|---|---|---|---|---|
| `known_map_lstm_base_seed2026` | base | 23 | 0.6588 | 41 KB |
| `known_map_lstm_headshots_only_seed2026` | headshots_only | 29 | 0.6665 | 44 KB |
| `known_map_lstm_rating3_headshots_seed2026` | rating3_headshots | 35 | 0.6702 | 45 KB |

Each directory contains: `checkpoint.pt`, `config.json`, `scaler.json`,
`map_vocab.json`, `fingerprint.json`, `artifact_manifest.json`.

## Recommendation

Use **base** for predictions. It performs best on val and has no retrospective-data
restrictions. The other two configurations are research-only:

- **headshots_only** adds career headshot-kill rate per player × map from HLTV +
  OpenCS2. Headshot data was captured after the fact and is not a snapshot of
  what was available before each match.
- **rating3_headshots** additionally adds HLTV Rating 3.0, which HLTV recomputed
  retroactively. It must not be treated as historically observable information
  and must not be mixed with Rating 2.0.

Both experimental configs underperformed base on val (delta +0.0077 and +0.0115
logloss). HS coverage is sparse (any-side 25.7%, both-sides 1.8% of val rows);
rating3 coverage is high (any-side 97.9%, both-sides 83.4%) but still did not help.
These are single-seed results; do not claim statistical significance.

## Quick predict

```powershell
# Default: base model, future date (no in-training-window warning)
.venv/m3/Scripts/python.exe -B -m known_map_recent.lstm.predict ^
  --team1 "team vitality" --team2 "team spirit" --map mirage --day 2027-01-15

# Explicit non-base
.venv/m3/Scripts/python.exe -B -m known_map_recent.lstm.predict ^
  --team1 "team vitality" --team2 "team spirit" --map mirage --day 2027-01-15 ^
  --model-dir models/lstm_release/known_map_lstm_headshots_only_seed2026
```

```python
from known_map_recent.lstm.predict import Predictor
pred = Predictor()   # loads known_map_lstm_base_seed2026 by default
r = pred.predict_map('team vitality', 'team spirit', 'mirage', '2027-01-15')
print(r['p_team1'], r['warnings'])
```

## Reports

Full metrics, coverage stats, coverage-bucket performance and comparison delta:

```
reports/known_map_lstm_release/
  base/metrics.json
  headshots_only/metrics.json
  rating3_headshots/metrics.json
  comparison.json          <- per-seed delta vs base, all three sets
  final_tests.txt          <- 27 synthetic tests, all OK
```

## Reproducing from scratch

```powershell
# Do NOT use --out-root models/lstm_release (would hit FileExistsError)
.venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm ^
  --feature-set base --seeds 2026 ^
  --out-root temp/lstm_reproduce --report-dir temp/lstm_reproduce_reports/base
```

Code digest covers `known_map_recent/lstm/*.py` and `scripts/train_known_map_lstm.py`.
The `requirements.txt` for this module is at `known_map_recent/lstm/requirements.txt`
(Python 3.12, m3 venv — different from the root `requirements.txt`).
