#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

import h5py

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


TASKS = ('lift', 'can', 'square')


def dataset_counts(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, 'r') as file:
        if 'data' not in file or not file['data']:
            raise ValueError(f'{path} contains no demonstrations.')
        return len(file['data']), sum(
            int(demo['actions'].shape[0]) for demo in file['data'].values()
        )


def validate_image_dataset(path: Path) -> tuple[int, int]:
    episodes, steps = dataset_counts(path)
    with h5py.File(path, 'r') as file:
        for demo_key, demo in file['data'].items():
            required = (
                'actions',
                'rewards',
                'obs/agentview_image',
                'obs/robot0_eye_in_hand_image',
            )
            missing = [key for key in required if key not in demo]
            if missing:
                raise ValueError(f'{path}:{demo_key} is missing {missing}.')
            horizon = int(demo['actions'].shape[0])
            if demo['actions'].shape != (horizon, 7):
                raise ValueError(f'{path}:{demo_key} must contain [T, 7] actions.')
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                if demo['obs'][key].shape != (horizon, 84, 84, 3):
                    raise ValueError(
                        f'{path}:{demo_key}/{key} has shape {demo["obs"][key].shape}.'
                    )
    return episodes, steps


def official_converter() -> Path:
    version = importlib.metadata.version('robomimic')
    if version != '0.5.0':
        raise RuntimeError(f'RoboMimic 0.5.0 is required, found {version}.')
    spec = importlib.util.find_spec('robomimic')
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError('robomimic is not installed.')
    converter = Path(spec.origin).resolve().parent / 'scripts' / 'dataset_states_to_obs.py'
    if not converter.is_file():
        raise FileNotFoundError(converter)
    return converter


def _run_converter(converter: Path, dataset: Path, output_name: str) -> None:
    sys.argv = [
        str(converter),
        '--dataset', str(dataset),
        '--output_name', output_name,
        '--done_mode', '2',
        '--camera_names', 'agentview', 'robot0_eye_in_hand',
        '--camera_height', '84', '--camera_width', '84',
        '--exclude-next-obs',
    ]
    runpy.run_path(str(converter), run_name='__main__')


def _partition_demos(dataset: Path, num_parts: int) -> list[list[str]]:
    with h5py.File(dataset, 'r') as file:
        demos = [
            (key, int(file['data'][key]['actions'].shape[0]))
            for key in file['data'].keys()
        ]
    partitions = [[] for _ in range(min(num_parts, len(demos)))]
    totals = [0 for _ in partitions]
    for key, length in sorted(demos, key=lambda item: item[1], reverse=True):
        target = min(range(len(partitions)), key=totals.__getitem__)
        partitions[target].append(key)
        totals[target] += length
    return partitions


def _write_subset(source_path: Path, subset_path: Path, demo_keys: list[str]) -> None:
    with h5py.File(source_path, 'r') as source, h5py.File(subset_path, 'w') as target:
        target_data = target.create_group('data')
        for key, value in source['data'].attrs.items():
            target_data.attrs[key] = value
        target_data.attrs['total'] = sum(
            int(source['data'][key]['actions'].shape[0]) for key in demo_keys
        )
        for key in demo_keys:
            source.copy(source['data'][key], target_data, name=key)


def _merge_parts(source_path: Path, part_paths: list[Path], output_path: Path) -> None:
    incomplete_path = output_path.with_suffix(output_path.suffix + '.incomplete')
    incomplete_path.unlink(missing_ok=True)
    try:
        with h5py.File(incomplete_path, 'w') as target:
            target_data = target.create_group('data')
            total = 0
            for part_index, part_path in enumerate(part_paths):
                with h5py.File(part_path, 'r') as source:
                    if part_index == 0:
                        for key, value in source['data'].attrs.items():
                            target_data.attrs[key] = value
                    for demo_key in source['data']:
                        source.copy(source['data'][demo_key], target_data, name=demo_key)
                        total += int(source['data'][demo_key]['actions'].shape[0])
            target_data.attrs['total'] = total
            with h5py.File(source_path, 'r') as raw_source:
                if 'mask' in raw_source:
                    raw_source.copy('mask', target)
        os.replace(incomplete_path, output_path)
    finally:
        incomplete_path.unlink(missing_ok=True)


def _convert_partition(converter: Path, dataset: Path, output_name: str) -> None:
    os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
    from envs.robomimic import (
        _import_robomimic_without_language_download,
        _patch_robosuite_mujoco3_fullm,
    )

    _patch_robosuite_mujoco3_fullm()
    _import_robomimic_without_language_download()
    _run_converter(converter, dataset, output_name)


def _spawn_partition(converter: Path, dataset: Path, output_name: str) -> subprocess.Popen:
    command = [
        sys.executable, str(Path(__file__).resolve()), '--_convert-partition',
        '--data-root', str(dataset.parent), '--converter', str(converter),
        '--dataset', str(dataset), '--output-name', output_name,
    ]
    environment = os.environ.copy()
    environment.setdefault('MUJOCO_GL', 'osmesa')
    environment.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    environment.setdefault('NUMBA_DISABLE_JIT', '1')
    return subprocess.Popen(command, env=environment)


def convert_image_dataset(
    converter: Path,
    dataset: Path,
    output_path: Path,
    chunk_size: int,
    parallel_parts: int,
) -> None:
    with h5py.File(dataset, 'r') as source:
        num_demos = len(source['data'])
    if num_demos <= chunk_size:
        temporary_name = f'.{output_path.name}.incomplete.{os.getpid()}'
        temporary_path = output_path.parent / temporary_name
        try:
            _convert_partition(converter, dataset, temporary_name)
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return

    partitions = _partition_demos(dataset, (num_demos + chunk_size - 1) // chunk_size)
    with tempfile.TemporaryDirectory(prefix='robomimic-image-') as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        part_paths: list[Path] = []
        commands: list[tuple[int, subprocess.Popen]] = []
        for part_index, demo_keys in enumerate(partitions):
            part_dir = temp_dir / f'part_{part_index}'
            part_dir.mkdir()
            subset = part_dir / 'demo_v141.hdf5'
            _write_subset(dataset, subset, demo_keys)
            part_path = part_dir / output_path.name
            part_paths.append(part_path)
            process = _spawn_partition(converter, subset, part_path.name)
            commands.append((part_index, process))
            if len(commands) == parallel_parts:
                failed = [index for index, item in commands if item.wait() != 0]
                if failed:
                    raise RuntimeError(f'RoboMimic conversion failed for partitions {failed}.')
                commands.clear()
        failed = [index for index, item in commands if item.wait() != 0]
        if failed:
            raise RuntimeError(f'RoboMimic conversion failed for partitions {failed}.')
        _merge_parts(dataset, part_paths, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--skip-conversion', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--parallel-parts', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--_convert-partition', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--converter', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--dataset', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output-name', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._convert_partition:
        if args.converter is None or args.dataset is None or not args.output_name:
            parser.error('Internal partition conversion arguments are incomplete.')
        _convert_partition(args.converter, args.dataset, args.output_name)
        return

    if args.chunk_size <= 0 or args.parallel_parts <= 0:
        parser.error('--chunk-size and --parallel-parts must be positive.')

    from envs.robomimic import validate_robomimic_dataset_task
    from utils.robomimic.dataset import _ensure_robomimic_buffer

    for task in args.tasks:
        task_dir = args.data_root.expanduser().resolve() / task / 'mh'
        raw_path = task_dir / 'demo_v141.hdf5'
        image_path = task_dir / 'image_v141.hdf5'
        buffer_path = task_dir / 'image_v141_romanflow.zarr'
        if not image_path.is_file():
            if args.skip_conversion:
                raise FileNotFoundError(image_path)
            dataset_counts(raw_path)
            convert_image_dataset(
                official_converter(), raw_path, image_path, args.chunk_size, args.parallel_parts
            )
        image_counts = validate_image_dataset(image_path)
        validate_robomimic_dataset_task(f'robomimic_{task}_mh', image_path)
        if raw_path.is_file() and image_counts != dataset_counts(raw_path):
            raise RuntimeError(
                f'{task} image conversion is incomplete: image={image_counts}, '
                f'raw={dataset_counts(raw_path)}.'
            )
        buffer = _ensure_robomimic_buffer(
            str(image_path),
            buffer_path=str(buffer_path),
            discount=args.discount,
        )
        print(
            f'{task}-MH ready: episodes={buffer.num_episodes}, '
            f'steps={buffer.num_steps}, path={buffer_path}'
        )


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    main()
