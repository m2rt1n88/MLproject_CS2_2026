"""Legacy TabPFN v1 context loader; only the pinned official weight is trusted."""
from functools import lru_cache
import inspect
import os
from pathlib import Path

from .runtime import DEPS
from .data_adapter import digest
TABPFN_CHECKPOINT_SHA256 = '3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392'
CHECKPOINT = DEPS / 'tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt'

@lru_cache(maxsize=1)
def classifier():
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.scripts import model_builder, transformer_prediction_interface as module

    if not CHECKPOINT.is_file() or digest(CHECKPOINT) != TABPFN_CHECKPOINT_SHA256:
        raise ValueError('Install the hash-verified official TabPFN v1 checkpoint at ' + str(CHECKPOINT))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for name in ('check_X_y', 'check_array'):
        original = getattr(module, name)
        if 'ensure_all_finite' in inspect.signature(original).parameters:
            def validate(*args, force_all_finite=True, _original=original, **kwargs):
                return _original(*args, ensure_all_finite=force_all_finite, **kwargs)
            setattr(module, name, validate)
    original_load = model_builder.torch.load

    def load_legacy(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == CHECKPOINT.resolve():
            kwargs['weights_only'] = False
        return original_load(path, *args, **kwargs)

    # Restore the global torch loader immediately after the trusted checkpoint is loaded.
    model_builder.torch.load = load_legacy
    try:
        return TabPFNClassifier(device='cpu', base_path=CHECKPOINT.parent.parent,
                                N_ensemble_configurations=3, subsample_features=False)
    finally:
        model_builder.torch.load = original_load
