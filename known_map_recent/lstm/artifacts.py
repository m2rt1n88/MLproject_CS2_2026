"""Save/load a run directory with a cross-checked artifact manifest.

Saved files
-----------
checkpoint.pt        model weights
config.json          feature set + hyper-params + static/seq feature names
scaler.json          normalisation statistics (fitted on inner_train only)
map_vocab.json       map embedding vocabulary
fingerprint.json     data/code provenance + metric snapshot fingerprint
artifact_manifest.json  sha256 of every other file; integrity gate on load

The artifact manifest is written last, so a partial save never passes load.
On load, every file is re-hashed and compared against the manifest before any
content is inspected.  A tampered scaler.json, vocab, or config is rejected with
IncompatibleCheckpoint, not silently used.
"""
import hashlib
import json
from pathlib import Path

import torch

from .dataset import Scaler
from .model import MapLSTM
from .state import FEATURE_SETS, SEQ_NAMES, STATIC_NAMES, static_names

ARTIFACT_FILES = ('checkpoint.pt', 'config.json', 'scaler.json', 'map_vocab.json', 'fingerprint.json')
MANIFEST = 'artifact_manifest.json'


class IncompatibleCheckpoint(ValueError):
    """Saved model is incompatible with the current code; retrain instead of guessing."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def save_run(out, model, vocab, scaler, config, fingerprint) -> dict:
    """Write all artifacts and an integrity manifest; return the finalised config dict."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'checkpoint.pt')
    config = dict(config, checkpoint_sha256=_sha(out / 'checkpoint.pt'))
    _write_json(out / 'config.json', config)
    _write_json(out / 'scaler.json', scaler.to_dict())
    _write_json(out / 'map_vocab.json', vocab)
    _write_json(out / 'fingerprint.json', fingerprint)
    manifest = {f: _sha(out / f) for f in ARTIFACT_FILES}
    _write_json(out / MANIFEST, manifest)
    return config


def _verify_manifest(model_dir: Path) -> dict:
    """Hash every artifact file and compare against the saved manifest; raise on any mismatch."""
    mpath = model_dir / MANIFEST
    if not mpath.exists():
        raise IncompatibleCheckpoint(
            f'{model_dir}: artifact_manifest.json missing — run was saved with an older version or is incomplete. '
            'Retrain to get a verifiable run directory.')
    manifest = json.loads(mpath.read_text(encoding='utf8'))
    problems = []
    for f in ARTIFACT_FILES:
        fpath = model_dir / f
        if not fpath.exists():
            problems.append(f'{f} missing')
        else:
            actual = _sha(fpath)
            expected = manifest.get(f)
            if actual != expected:
                problems.append(f'{f} sha256 mismatch (expected {_fmt(expected)}… got {actual[:12]}…)')
    if problems:
        raise IncompatibleCheckpoint(f'{model_dir} integrity check failed: ' + '; '.join(problems))
    return manifest


def _fmt(h):
    """Safe 12-char prefix; h may be None if a file was absent."""
    return str(h)[:12] if h is not None else 'None'


def load_run(model_dir):
    """Return (model, vocab, scaler, config, fingerprint).

    Raises IncompatibleCheckpoint if:
    - artifact_manifest.json is missing or any file hash differs,
    - feature set / static-feature list doesn't match the installed code,
    - scaler width doesn't match the feature list,
    - metric snapshot fingerprint is absent for a non-base feature set.
    """
    model_dir = Path(model_dir)
    _verify_manifest(model_dir)   # hashes all 5 files before reading any

    config = json.loads((model_dir / 'config.json').read_text(encoding='utf8'))
    vocab = json.loads((model_dir / 'map_vocab.json').read_text(encoding='utf8'))
    scaler = Scaler.from_dict(json.loads((model_dir / 'scaler.json').read_text(encoding='utf8')))
    fingerprint = json.loads((model_dir / 'fingerprint.json').read_text(encoding='utf8'))

    legacy = 'feature_set' not in config
    feature_set = config.get('feature_set', 'base')
    if feature_set not in FEATURE_SETS:
        raise IncompatibleCheckpoint(f'unknown feature set {feature_set!r} in {model_dir}')

    names = list(static_names(feature_set))
    saved_names = list(config.get('static_names', STATIC_NAMES if legacy else ()))
    problems = []
    if saved_names != names or config.get('n_static') != len(names):
        problems.append(
            f'static features: saved {config.get("n_static")} ({saved_names[:3]}…) '
            f'vs expected {len(names)} for {feature_set!r}')
    if list(config.get('seq_names', SEQ_NAMES)) != list(SEQ_NAMES) or config.get('n_seq') != len(SEQ_NAMES):
        problems.append('sequence feature names differ')
    if len(scaler.s_mean) != len(names) or len(scaler.q_mean) != len(SEQ_NAMES):
        problems.append(f'scaler width {len(scaler.s_mean)} / {len(scaler.q_mean)} '
                        f'vs expected {len(names)} / {len(SEQ_NAMES)}')
    if feature_set != 'base' and not fingerprint.get('metric_snapshot'):
        problems.append('non-base feature set has no metric_snapshot in fingerprint — cannot verify data provenance')
    if problems:
        raise IncompatibleCheckpoint(
            f'{model_dir}: ' + '; '.join(problems) +
            f'. Retrain: scripts.train_known_map_lstm --feature-set {feature_set}')

    model = MapLSTM(len(vocab), len(names), len(SEQ_NAMES), **config['model'])
    model.load_state_dict(torch.load(model_dir / 'checkpoint.pt', map_location='cpu', weights_only=True))
    model.eval()
    config = dict(config, feature_set=feature_set, legacy_config=legacy)
    return model, vocab, scaler, config, fingerprint
