#!/usr/bin/env python3
"""Convert a trusted local CLIP PyTorch checkpoint to Safetensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_WEIGHT_SUFFIXES = ('.bin', '.h5', '.msgpack', '.safetensors')


def _copy_model_assets(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file() or path.name.endswith(_WEIGHT_SUFFIXES):
            continue
        if path.name.endswith('.safetensors.index.json'):
            continue
        shutil.copy2(path, output / path.name)


def convert(source: Path, output: Path, *, trust_local_pickle: bool) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError('Use a separate output path to preserve the source model.')
    config_path = source / 'config.json'
    input_weights = source / 'pytorch_model.bin'
    if not config_path.is_file() or not input_weights.is_file():
        raise FileNotFoundError(f'{source} must contain config.json and pytorch_model.bin.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('model_type') != 'clip' or 'CLIPModel' not in config.get(
        'architectures', []
    ):
        raise ValueError(f'{source} is not a CLIPModel directory.')
    if not trust_local_pickle:
        raise ValueError(
            'pytorch_model.bin is pickle-based. Re-run with --trust-local-pickle '
            'only for a trusted local checkpoint.'
        )

    _copy_model_assets(source, output)
    output_weights = output / 'model.safetensors'
    temporary_weights = output / f'.model.safetensors.incomplete.{os.getpid()}'
    temporary_weights.unlink(missing_ok=True)
    try:
        state = torch.load(
            input_weights,
            map_location='cpu',
            mmap=True,
            weights_only=True,
        )
        if not isinstance(state, dict) or not state:
            raise ValueError(f'{input_weights} is not a non-empty state dictionary.')
        tensors = {}
        for key, value in state.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                raise ValueError(f'Invalid state entry in {input_weights}: {key!r}')
            tensors[key] = value.detach().cpu().contiguous()
        save_file(tensors, str(temporary_weights), metadata={'format': 'pt'})
        os.replace(temporary_weights, output_weights)
    finally:
        temporary_weights.unlink(missing_ok=True)

    with safe_open(output_weights, framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError('Safetensors output keys do not match the source checkpoint.')
    print(f'CLIP Safetensors model ready: {output}')
    return output_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-path', type=Path, required=True)
    parser.add_argument('--output-model-path', type=Path, required=True)
    parser.add_argument('--trust-local-pickle', action='store_true')
    args = parser.parse_args()
    convert(
        args.source_model_path,
        args.output_model_path,
        trust_local_pickle=args.trust_local_pickle,
    )


if __name__ == '__main__':
    main()
