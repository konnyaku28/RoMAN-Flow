#!/usr/bin/env python3
"""Evaluate released RoMAN-Flow checkpoints on explicit local resources.

The launcher never installs packages, infers cluster roles, or writes into the
repository.  Split a job list across machines with --shard-index and
--shard-count; after all shards finish, run again with --summarize-only.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
LIBERO_SUITES = ('libero_10', 'libero_spatial', 'libero_object', 'libero_goal')
ROBOMIMIC_TASKS = ('lift', 'can', 'square')
STAGES = ('il', 'iql', 'one_step')


@dataclass(frozen=True)
class EvaluationJob:
    benchmark: str
    suite_or_task: str
    stage: str
    dataset: Path
    buffer: Path
    checkpoint: Path
    flags: Path
    output_dir: Path
    seed: int | None
    cfg: float
    temperature: float
    biflow_guidance: float
    biflow_guidance_final: float
    action_exec_horizon: int

    @property
    def label(self) -> str:
        seed = '' if self.seed is None else f'/seed{self.seed}'
        return f'{self.suite_or_task}/{self.stage}{seed}/{self.dataset.stem}'


def _comma_separated_choices(value: str, choices: tuple[str, ...], name: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(',') if item.strip())
    if not values or any(item not in choices for item in values):
        raise argparse.ArgumentTypeError(
            f'{name} must be a comma-separated subset of {", ".join(choices)}.'
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f'{name} must not contain duplicates.')
    return values


def _device_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(',') if item.strip())
    if not values or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError('--devices must contain CUDA device indices, e.g. 0,1.')
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('--devices must not contain duplicates.')
    return values


def _path_outside_repository(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(CODE_ROOT)
    except ValueError:
        return path
    raise argparse.ArgumentTypeError('--output-root must be outside the repository.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('libero', 'robomimic'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--buffer-root', type=Path, required=True)
    parser.add_argument('--output-root', type=_path_outside_repository, required=True)
    parser.add_argument('--checkpoint-root', type=Path, default=CODE_ROOT / 'weights/checkpoints')
    parser.add_argument('--manifest', type=Path, default=CODE_ROOT / 'weights/manifest.json')
    parser.add_argument('--stages')
    parser.add_argument('--suites', default=','.join(LIBERO_SUITES))
    parser.add_argument('--tasks', default=','.join(ROBOMIMIC_TASKS))
    parser.add_argument('--devices', type=_device_list)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--episode-seed-start', type=int, default=0)
    parser.add_argument('--smolvlm-model-path', type=Path)
    parser.add_argument('--clip-model-path', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.stages is None:
        args.stages = STAGES if args.benchmark == 'libero' else ('iql', 'one_step')
    else:
        args.stages = _comma_separated_choices(args.stages, STAGES, '--stages')
    args.suites = _comma_separated_choices(args.suites, LIBERO_SUITES, '--suites')
    args.tasks = _comma_separated_choices(args.tasks, ROBOMIMIC_TASKS, '--tasks')
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error('--shard-index must be in [0, --shard-count).')
    if args.eval_episodes <= 0:
        parser.error('--eval-episodes must be positive.')
    if args.episode_seed_start < 0:
        parser.error('--episode-seed-start must be non-negative.')
    if not args.summarize_only and args.devices is None:
        parser.error('--devices is required unless --summarize-only is set.')
    if args.summarize_only and (args.shard_index != 0 or args.shard_count != 1):
        parser.error('--summarize-only requires the default shard index and count.')
    if args.benchmark == 'libero' and not args.summarize_only:
        if args.smolvlm_model_path is None or args.clip_model_path is None:
            parser.error('LIBERO evaluation requires --smolvlm-model-path and --clip-model-path.')
    return args


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing {label}: {path}')
    return path


def _require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Missing {label}: {path}')
    return path


def _load_groups(manifest_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(_require_file(manifest_path, 'weight manifest').read_text(encoding='utf-8'))
    groups = payload.get('groups')
    if not isinstance(groups, list):
        raise ValueError(f'{manifest_path} has no groups list.')
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict) or group.get('benchmark') != 'libero':
            continue
        task, stage = group.get('task'), group.get('stage')
        if isinstance(task, str) and isinstance(stage, str):
            profiles[(task, stage)] = group
    return profiles


def _load_robomimic_artifacts(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(_require_file(manifest_path, 'weight manifest').read_text(encoding='utf-8'))
    artifacts = payload.get('artifacts')
    if not isinstance(artifacts, list):
        raise ValueError(f'{manifest_path} has no artifacts list.')
    return [
        item for item in artifacts
        if isinstance(item, dict) and item.get('benchmark') == 'robomimic_mh'
    ]


def _released_checkpoint(checkpoint_root: Path, relative: Path) -> Path:
    relative_parts = relative.parts
    if relative_parts[:2] != ('weights', 'checkpoints'):
        raise ValueError(f'Unexpected release checkpoint path in manifest: {relative}')
    return _require_file(checkpoint_root.joinpath(*relative_parts[2:]), 'release checkpoint')


def _released_flags(checkpoint: Path) -> Path:
    return _require_file(checkpoint.with_name('flags.json'), 'release flags')


def build_libero_jobs(args: argparse.Namespace) -> list[EvaluationJob]:
    data_root = _require_dir(args.data_root, 'LIBERO data root')
    buffer_root = _require_dir(args.buffer_root, 'LIBERO buffer root')
    checkpoint_root = _require_dir(args.checkpoint_root, 'checkpoint root')
    if not args.summarize_only:
        _require_dir(args.smolvlm_model_path, 'SmolVLM model directory')
        _require_file(args.smolvlm_model_path / 'model.safetensors', 'SmolVLM weights')
        _require_dir(args.clip_model_path, 'CLIP model directory')
        _require_file(args.clip_model_path / 'model.safetensors', 'CLIP Safetensors weights')
    profiles = _load_groups(args.manifest)
    jobs: list[EvaluationJob] = []
    for suite in args.suites:
        task_files = sorted((data_root / suite).glob('*.hdf5'))
        if len(task_files) != 10:
            raise ValueError(f'{suite} must contain exactly 10 task HDF5 files, found {len(task_files)}.')
        weight_suite = 'libero_long' if suite == 'libero_10' else suite
        for stage in args.stages:
            profile = profiles.get((suite, stage))
            if profile is None:
                raise ValueError(f'Manifest has no LIBERO profile for {suite}/{stage}.')
            checkpoint = _require_file(
                checkpoint_root / weight_suite / stage / 'checkpoint.pt',
                f'{suite}/{stage} checkpoint',
            )
            flags = _released_flags(checkpoint)
            for dataset in task_files:
                buffer = _require_dir(
                    buffer_root / suite / f'{dataset.stem}.zarr',
                    f'{suite} rollout buffer for {dataset.stem}',
                )
                jobs.append(EvaluationJob(
                    benchmark='libero',
                    suite_or_task=suite,
                    stage=stage,
                    dataset=dataset,
                    buffer=buffer,
                    checkpoint=checkpoint,
                    flags=flags,
                    output_dir=args.output_root / 'results' / suite / stage / dataset.stem,
                    seed=None,
                    # BiFlow sampling uses biflow guidance rather than classifier-free cfg.
                    cfg=float(profile.get('cfg', 0.0)),
                    temperature=float(profile['temperature']),
                    biflow_guidance=float(profile.get('biflow_guidance', 0.0)),
                    biflow_guidance_final=float(profile.get('biflow_guidance_final', -1.0)),
                    action_exec_horizon=int(profile['action_exec_horizon']),
                ))
    return jobs


def build_robomimic_jobs(args: argparse.Namespace) -> list[EvaluationJob]:
    data_root = _require_dir(args.data_root, 'RoboMimic data root')
    buffer_root = _require_dir(args.buffer_root, 'RoboMimic buffer root')
    checkpoint_root = _require_dir(args.checkpoint_root, 'checkpoint root')
    artifacts = _load_robomimic_artifacts(args.manifest)
    jobs: list[EvaluationJob] = []
    for task in args.tasks:
        dataset = _require_file(data_root / task / 'mh' / 'image_v141.hdf5', f'{task} image HDF5')
        buffer = _require_dir(
            buffer_root / task / 'mh' / 'image_v141_romanflow.zarr',
            f'{task} Zarr buffer',
        )
        for stage in args.stages:
            selected = sorted(
                (
                    artifact for artifact in artifacts
                    if artifact.get('task') == task and artifact.get('stage') == stage
                ),
                key=lambda item: int(item['seed']),
            )
            if not selected:
                raise ValueError(f'Manifest has no RoboMimic artifacts for {task}/{stage}.')
            for artifact in selected:
                checkpoint = _released_checkpoint(
                    checkpoint_root, Path(str(artifact['release_checkpoint']))
                )
                jobs.append(EvaluationJob(
                    benchmark='robomimic',
                    suite_or_task=task,
                    stage=stage,
                    dataset=dataset,
                    buffer=buffer,
                    checkpoint=checkpoint,
                    flags=_released_flags(checkpoint),
                    output_dir=args.output_root / 'results' / task / stage / f"seed{artifact['seed']}",
                    seed=int(artifact['seed']),
                    cfg=0.0,
                    temperature=0.7,
                    biflow_guidance=0.0,
                    biflow_guidance_final=0.0,
                    action_exec_horizon=10,
                ))
    return jobs


def build_jobs(args: argparse.Namespace) -> list[EvaluationJob]:
    if args.benchmark == 'libero':
        return build_libero_jobs(args)
    return build_robomimic_jobs(args)


def _result_paths(job: EvaluationJob) -> tuple[Path, Path, Path]:
    return (
        job.output_dir / 'result.csv',
        job.output_dir / 'episodes.json',
        job.output_dir / 'eval.log',
    )


def valid_result(job: EvaluationJob, expected_episodes: int) -> bool:
    csv_path, episode_path, _ = _result_paths(job)
    try:
        with csv_path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        payload = json.loads(episode_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(rows) == 1 and len(payload.get('episodes', ())) == expected_episodes


def _command(job: EvaluationJob, args: argparse.Namespace) -> list[str]:
    backend = 'biflow' if job.stage == 'one_step' else 'simflow'
    command = [
        sys.executable,
        str(CODE_ROOT / 'eval_biflow_torch.py'),
        f'--checkpoint={job.checkpoint}',
        f'--flags_json={job.flags}',
        f'--dataset_dir={job.dataset}',
        f'--eval_episodes={args.eval_episodes}',
        f'--episode_seed_start={args.episode_seed_start}',
        f'--temperature={job.temperature}',
        f'--cfg={job.cfg}',
        f'--biflow_eval_guidance={job.biflow_guidance}',
        f'--biflow_eval_guidance_final={job.biflow_guidance_final}',
        f'--action_exec_horizon={job.action_exec_horizon}',
        f'--sample_backend={backend}',
        f'--use_biflow={str(backend == "biflow").lower()}',
        '--device=cuda:0',
        '--vec_env=dummy',
        '--num_envs=1',
        '--fixed_episode_seeds=true',
        '--clip_actions=true',
        f'--output={job.output_dir / "result.csv"}',
        f'--episode_output={job.output_dir / "episodes.json"}',
    ]
    if job.benchmark == 'libero':
        command.extend([
            f'--env_name={job.suite_or_task}',
            f'--libero_buffer_path={job.buffer}',
            f'--smolvlm_model_path={args.smolvlm_model_path.resolve()}',
            f'--language_model_path={args.clip_model_path.resolve()}',
            '--obs_horizon=2',
            '--action_horizon=16',
            '--denoising_lr=0.0',
            '--apply_denoising=true',
            '--libero_eval_visual_preprocess=auto',
        ])
    else:
        command.extend([
            f'--env_name=robomimic_{job.suite_or_task}_mh',
            f'--robomimic_buffer_path={job.buffer}',
            '--obs_horizon=1',
            '--action_horizon=10',
            '--apply_denoising=false',
        ])
    return command


def run_job(job: EvaluationJob, device: str, args: argparse.Namespace) -> str:
    csv_path, episode_path, log_path = _result_paths(job)
    if not args.overwrite and valid_result(job, args.eval_episodes):
        return f'SKIPPED gpu={device} {job.label}'
    job.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (csv_path, episode_path, log_path):
        path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = device
    environment.setdefault('OMP_NUM_THREADS', '2')
    environment.setdefault('MKL_NUM_THREADS', '2')
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            _command(job, args),
            cwd=CODE_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not valid_result(job, args.eval_episodes):
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-80:]
        raise RuntimeError(
            f'Evaluation failed on GPU {device}: {job.label}\n' + '\n'.join(tail)
        )
    return f'PASSED gpu={device} {job.label}'


def write_summary(jobs: list[EvaluationJob], args: argparse.Namespace) -> Path:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if not valid_result(job, args.eval_episodes):
            raise RuntimeError(f'Incomplete result: {job.label}')
        csv_path, _, _ = _result_paths(job)
        with csv_path.open(newline='', encoding='utf-8') as handle:
            record = next(csv.DictReader(handle))
        rows.append({
            'benchmark': job.benchmark,
            'suite_or_task': job.suite_or_task,
            'stage': job.stage,
            'seed': '' if job.seed is None else job.seed,
            'task': job.dataset.stem if job.benchmark == 'libero' else '',
            **record,
        })
    summary_path = args.output_root / 'summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'benchmark', 'suite_or_task', 'stage', 'seed', 'task', 'success',
        'eval_episodes', 'temperature', 'cfg', 'sample_backend', 'use_biflow',
        'checkpoint', 'output',
    ]
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.buffer_root = args.buffer_root.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    jobs = build_jobs(args)
    if args.summarize_only:
        summary = write_summary(jobs, args)
        print(f'Validated {len(jobs)} results. Summary: {summary}')
        return

    shard_jobs = [
        job for index, job in enumerate(jobs)
        if index % args.shard_count == args.shard_index
    ]
    print(
        f'Prepared {len(jobs)} {args.benchmark} jobs; running shard '
        f'{args.shard_index + 1}/{args.shard_count} with {len(shard_jobs)} jobs.',
        flush=True,
    )
    for offset in range(0, len(shard_jobs), len(args.devices)):
        batch = shard_jobs[offset : offset + len(args.devices)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(run_job, job, args.devices[index], args): job
                for index, job in enumerate(batch)
            }
            for future in as_completed(futures):
                print(future.result(), flush=True)
    if args.shard_count == 1:
        summary = write_summary(jobs, args)
        print(f'Validated {len(jobs)} results. Summary: {summary}')
    else:
        print('Shard complete. Run again with --summarize-only after all shards finish.')


if __name__ == '__main__':
    main()
