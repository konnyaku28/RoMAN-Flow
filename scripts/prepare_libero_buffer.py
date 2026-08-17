#!/usr/bin/env python3
"""Build a LIBERO training or rollout Zarr buffer from local HDF5 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import zarr

from utils.robomimic.dataset import _ensure_libero_buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='HDF5 path or quoted glob.')
    parser.add_argument('--buffer', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--stats-source-buffer', type=Path)
    parser.add_argument('--discount', type=float, default=0.995)
    parser.add_argument('--language-max-length', type=int, default=50)
    parser.add_argument('--no-proprioception', action='store_true')
    args = parser.parse_args()

    include_proprioception = not args.no_proprioception
    if not args.model_path.is_dir():
        raise FileNotFoundError('--model-path must be a local SmolVLM directory.')
    buffer = _ensure_libero_buffer(
        args.dataset,
        buffer_path=str(args.buffer.expanduser().resolve()),
        flip_rgb=True,
        discount=args.discount,
        language_model_path=str(args.model_path.expanduser().resolve()),
        language_max_length=args.language_max_length,
        language_tokenizer_type='smolvlm',
        include_proprioception=include_proprioception,
    )
    if args.stats_source_buffer is not None:
        if not include_proprioception:
            raise ValueError('--stats-source-buffer requires proprioception.')
        source = zarr.open_group(str(args.stats_source_buffer.expanduser().resolve()), mode='r')
        q01 = np.asarray(source.attrs.get('proprio_q01'), dtype=np.float32)
        q99 = np.asarray(source.attrs.get('proprio_q99'), dtype=np.float32)
        if q01.shape != (8,) or q99.shape != (8,) or np.any(q99 <= q01):
            raise ValueError('The source buffer does not contain valid 8-D training quantiles.')
        buffer.root.attrs.update({
            'proprio_q01': q01.tolist(),
            'proprio_q99': q99.tolist(),
            'proprio_normalization': 'q01_q99_to_minus1_plus1_clip',
        })
    print(
        f'LIBERO buffer ready: episodes={buffer.num_episodes}, '
        f'steps={buffer.num_steps}, path={args.buffer.expanduser().resolve()}'
    )


if __name__ == '__main__':
    main()
