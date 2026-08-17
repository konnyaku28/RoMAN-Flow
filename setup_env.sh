#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-romanflow}"

python - <<'PY'
import sys
import torch

expected_python = (3, 10)
if sys.version_info[:2] != expected_python:
    raise SystemExit(
        f'RoMAN-Flow DLC environment requires Python 3.10, found {sys.version.split()[0]}.'
    )
if not torch.__version__.startswith('2.4.'):
    raise SystemExit(
        f'Expected the DLC image-provided PyTorch 2.4.x, found {torch.__version__}. '
        'Do not replace the image CUDA/PyTorch stack with pip.'
    )
if not str(torch.version.cuda or '').startswith('12.3'):
    raise SystemExit(
        f'Expected the DLC image CUDA 12.3 build, found torch CUDA {torch.version.cuda!r}.'
    )
print(f'Using image PyTorch {torch.__version__}, CUDA {torch.version.cuda}.')
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REPO_DIR}/requirements-vla.txt"
python -m pip install -r "${REPO_DIR}/requirements-robomimic.txt"

# RoboMimic 0.5 declares an older Transformers version. RoMAN-Flow uses the
# newer SmolVLM-compatible version above, so install the framework without
# allowing it to replace the resolved DLC dependency stack. RoboMimic 0.5.0
# was not published to PyPI; provide a local wheel or source checkout.
install_robomimic() {
    local wheel="${ROBOMIMIC_WHEEL:-}"
    local source_dir="${ROBOMIMIC_SOURCE_DIR:-}"
    if [[ -n "${wheel}" ]]; then
        if [[ ! -f "${wheel}" ]]; then
            echo "ROBOMIMIC_WHEEL does not exist: ${wheel}" >&2
            exit 1
        fi
        python -m pip install --no-deps "${wheel}"
        return
    fi

    if [[ -n "${source_dir}" ]]; then
        if [[ ! -f "${source_dir}/setup.py" ]]; then
            echo "ROBOMIMIC_SOURCE_DIR is not a RoboMimic source checkout: ${source_dir}" >&2
            exit 1
        fi
        python -m pip install --no-deps "${source_dir}"
        return
    fi

    echo "RoboMimic 0.5.0 must be provided locally." >&2
    echo "Set ROBOMIMIC_WHEEL=/path/to/robomimic-0.5.0-py3-none-any.whl" >&2
    echo "or ROBOMIMIC_SOURCE_DIR=/path/to/robomimic-source." >&2
    exit 1
}

INSTALL_ROBOMIMIC="${INSTALL_ROBOMIMIC:-1}"
if [[ "${INSTALL_ROBOMIMIC}" == 1 ]]; then
    install_robomimic
elif [[ "${INSTALL_ROBOMIMIC}" != 0 ]]; then
    echo "INSTALL_ROBOMIMIC must be 0 or 1, got ${INSTALL_ROBOMIMIC}." >&2
    exit 2
fi
