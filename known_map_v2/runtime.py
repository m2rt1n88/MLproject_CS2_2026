"""Allow the existing portable Python to use the isolated v2 dependencies."""
import os
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
DEPS = ROOT / '.venv' / 'archotium-repro'


def configure():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MPLCONFIGDIR', str(HOME / '.cache' / 'matplotlib'))
    if (DEPS / 'sklearn').is_dir():
        sys.path.insert(0, str(DEPS))
