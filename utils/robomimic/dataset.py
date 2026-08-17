import glob
from contextlib import contextmanager
import fcntl
import os
import re
import shutil

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.trajectory_store import ZarrTrajectoryStore
from utils.robomimic.proprioception import (
    LIBERO_PROPRIO_DIM,
    LIBERO_PROPRIO_FORMAT,
    libero_hdf5_proprioception,
    proprioception_quantiles,
)
from utils.sequence_index import EpisodeSequenceIndex, sequence_right_padding


ROBOMIMIC_IMAGE_KEYS = (
    'agentview_image',
    'robot0_eye_in_hand_image',
)
ROBOMIMIC_PROPRIO_KEYS = (
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
)

LIBERO_VISUAL_RESIZE_SIZE = 240
LIBERO_VISUAL_CROP_SIZE = 224
LIBERO_VISUAL_COLOR_JITTER = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': (-0.2, 0.2),
}


@contextmanager
def _buffer_build_lock(storage_path):
    """Serialize cache publication across torchrun ranks and DLC nodes."""
    lock_path = f'{storage_path}.lock'
    os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)
    with open(lock_path, 'a+', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_complete_buffer(buffer, expected_episodes, expected_steps, label):
    if buffer.num_episodes != expected_episodes or buffer.num_steps != expected_steps:
        raise ValueError(
            f'Restored {label} buffer is incomplete: '
            f'episodes={buffer.num_episodes}/{expected_episodes}, '
            f'steps={buffer.num_steps}/{expected_steps}. Remove it and rebuild.'
        )


def _ensure_libero_proprioception_stats(buffer):
    q01 = np.asarray(buffer.root.attrs.get('proprio_q01', ()), dtype=np.float32)
    q99 = np.asarray(buffer.root.attrs.get('proprio_q99', ()), dtype=np.float32)
    if q01.shape == (LIBERO_PROPRIO_DIM,) and q99.shape == (LIBERO_PROPRIO_DIM,):
        if np.all(np.isfinite(q01)) and np.all(np.isfinite(q99)) and np.all(q99 > q01):
            return
    q01, q99 = proprioception_quantiles(np.asarray(buffer['proprioceptions']))
    buffer.root.attrs.update({
        'proprio_q01': q01.tolist(),
        'proprio_q99': q99.tolist(),
        'proprio_normalization': 'q01_q99_to_minus1_plus1_clip',
    })
def canonicalize_libero_visual_augmentation_mode(mode):
    normalized = str(mode or 'libero_224').strip().lower().replace('-', '_')
    if normalized != 'libero_224':
        raise ValueError(
            f'Unsupported visual augmentation mode {mode!r}; RoMAN-Flow uses `libero_224`.'
        )
    return normalized


def _apply_libero_visual_augmentation(observations, mode='libero_224'):
    from torchvision.transforms import ColorJitter, InterpolationMode, RandomCrop, Resize

    observations = np.asarray(observations)
    if observations.ndim != 6:
        raise ValueError(
            'LIBERO visual augmentation expects observations with shape '
            f'[B, T, V, C, H, W], got {observations.shape}.'
        )
    batch_size, sequence_length, num_views, channels, height, width = (
        observations.shape
    )
    if channels not in (1, 3):
        raise ValueError(
            f'LIBERO visual augmentation expects 1 or 3 image channels, got {channels}.'
        )

    canonicalize_libero_visual_augmentation_mode(mode)
    resize_height = resize_width = LIBERO_VISUAL_RESIZE_SIZE
    crop_height = crop_width = LIBERO_VISUAL_CROP_SIZE
    resize = Resize(
        (resize_height, resize_width),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    random_crop = RandomCrop((crop_height, crop_width))
    color_jitter = ColorJitter(**LIBERO_VISUAL_COLOR_JITTER)
    augmented = np.empty(
        (batch_size, sequence_length, num_views, channels, crop_height, crop_width),
        dtype=np.uint8,
    )

    for view_idx in range(num_views):
        images = torch.from_numpy(
            np.ascontiguousarray(observations[:, :, view_idx])
        ).reshape(batch_size * sequence_length, channels, height, width)
        images = images.float().div_(255.0)
        images = resize(images)
        images = random_crop(images)
        images = color_jitter(images)
        images = images.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        augmented[:, :, view_idx] = images.reshape(
            batch_size,
            sequence_length,
            channels,
            crop_height,
            crop_width,
        ).cpu().numpy()
    return augmented


def apply_libero_visual_eval_preprocessing(observations, mode='libero_224'):
    """Apply LIBERO's deterministic eval resize and center crop.

    Online policy observations use ``[B, V, C, T, H, W]`` layout. The
    operation is applied independently to every frame and view, preserves the
    dtype and device, deliberately omits ColorJitter, and returns 224 x 224
    images.
    """
    from torchvision.transforms import CenterCrop, InterpolationMode, Resize

    if not isinstance(observations, torch.Tensor):
        raise TypeError(
            'LIBERO eval visual preprocessing expects a torch.Tensor with shape '
            f'[B, V, C, T, H, W], got {type(observations).__name__}.'
        )
    if observations.ndim != 6:
        raise ValueError(
            'LIBERO eval visual preprocessing expects observations with shape '
            f'[B, V, C, T, H, W], got {tuple(observations.shape)}.'
        )

    batch_size, num_views, channels, sequence_length, height, width = (
        observations.shape
    )
    if channels not in (1, 3):
        raise ValueError(
            f'LIBERO eval visual preprocessing expects 1 or 3 image channels, got {channels}.'
        )

    canonicalize_libero_visual_augmentation_mode(mode)
    resize_height = resize_width = LIBERO_VISUAL_RESIZE_SIZE
    crop_height = crop_width = LIBERO_VISUAL_CROP_SIZE
    original_dtype = observations.dtype
    images = observations.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size * num_views * sequence_length,
        channels,
        height,
        width,
    )
    images = images.float()
    images = Resize(
        (resize_height, resize_width),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )(images)
    images = CenterCrop((crop_height, crop_width))(images)

    if original_dtype.is_floating_point:
        images = images.to(dtype=original_dtype)
    else:
        dtype_info = torch.iinfo(original_dtype)
        images = images.round().clamp_(dtype_info.min, dtype_info.max).to(original_dtype)

    return images.reshape(
        batch_size,
        num_views,
        sequence_length,
        channels,
        crop_height,
        crop_width,
    ).permute(0, 1, 3, 2, 4, 5).contiguous()


def _compute_return_to_go(rewards, terminals, discount):
    """Compute per-step Monte Carlo return within one trajectory."""
    rewards = rewards.astype(np.float32, copy=False)
    terminals = terminals.astype(np.float32, copy=False)
    mc_returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + discount * (1.0 - terminals[t]) * running_return
        mc_returns[t] = running_return
    return mc_returns


def _compute_rank_hubl_lambdas(trajectory_mc_returns, alpha):
    """Compute trajectory-level Rank HUBL lambda and broadcast it to each step."""
    if len(trajectory_mc_returns) == 0:
        return []

    trajectory_scores = np.array(
        [float(np.mean(mc_returns)) for mc_returns in trajectory_mc_returns],
        dtype=np.float32,
    )
    num_trajectories = len(trajectory_scores)
    trajectory_lambdas = np.array(
        [
            alpha * float(np.sum(trajectory_scores <= score)) / float(num_trajectories)
            for score in trajectory_scores
        ],
        dtype=np.float32,
    )
    return [
        np.full_like(mc_returns, fill_value=trajectory_lambdas[idx], dtype=np.float32)
        for idx, mc_returns in enumerate(trajectory_mc_returns)
    ]


def get_task_name_from_path(path):
    stem = os.path.splitext(os.path.basename(os.path.abspath(os.path.expanduser(path))))[0]
    for suffix in ('_demo', '_demos'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r'^[A-Z_]+SCENE\d+_', '', stem)
    stem = re.sub(r'_+', ' ', stem).strip()
    return stem


def _tokenize_libero_task_paths(
    hdf5_paths,
    model_path=None,
    max_length=77,
    tokenizer_type='clip',
):
    from utils.language import build_text_tokenizer

    tokenizer = build_text_tokenizer(
        tokenizer_type,
        model_path=model_path,
        max_length=max_length,
    )
    task_names = [get_task_name_from_path(path) for path in hdf5_paths]
    input_ids, attention_mask = tokenizer.tokenize(task_names)
    return input_ids.cpu().numpy().astype(np.int64), attention_mask.cpu().numpy().astype(np.int64)


def _resolve_hdf5_paths(hdf5_path):
    expanded_path = os.path.expanduser(hdf5_path)
    hdf5_paths = sorted(glob.glob(expanded_path)) if '*' in expanded_path else [expanded_path]
    hdf5_paths = [os.path.abspath(path) for path in hdf5_paths]
    if len(hdf5_paths) == 0:
        raise ValueError(f'No demonstrations found in {hdf5_path}')
    return hdf5_paths


def _resolve_libero_buffer_path(
    hdf5_path,
    buffer_path=None,
    language_tokenizer_type='clip',
):
    if buffer_path is not None:
        return os.path.abspath(os.path.expanduser(buffer_path))

    tokenizer_suffix = str(language_tokenizer_type).strip().lower().replace('_', '')
    suffix = f'_lang{tokenizer_suffix}'
    if '*' in hdf5_path:
        dataset_parent = os.path.dirname(os.path.abspath(os.path.expanduser(hdf5_path)))
        return os.path.join(dataset_parent, f'romanflow_libero_buffer{suffix}.zarr')

    resolved_hdf5 = os.path.abspath(os.path.expanduser(hdf5_path))
    stem = os.path.splitext(os.path.basename(resolved_hdf5))[0]
    return os.path.join(os.path.dirname(resolved_hdf5), f'{stem}_buffer{suffix}.zarr')


def _build_libero_episode(
    demo,
    flip_rgb,
    discount,
    language_tokens=None,
    include_proprioception=False,
):
    agentview = demo['obs']['agentview_rgb'][:]
    eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

    if flip_rgb:
        agentview = agentview[:, ::-1]
        eye_in_hand = eye_in_hand[:, ::-1]

    agentview_t = np.transpose(agentview, (0, 3, 1, 2))
    eye_in_hand_t = np.transpose(eye_in_hand, (0, 3, 1, 2))
    obs = np.stack([agentview_t, eye_in_hand_t], axis=1).astype(np.uint8, copy=False)

    actions = demo['actions'][:].astype(np.float32, copy=False)
    rewards = demo['rewards'][:].astype(np.float32)
    terminals = demo['dones'][:].astype(np.float32)
    terminals[-1] = 1.0
    masks = 1.0 - terminals
    mc_returns = _compute_return_to_go(rewards, terminals, discount)
    episode = dict(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
        mc_returns=mc_returns,
    )
    if include_proprioception:
        proprioceptions = libero_hdf5_proprioception(demo['obs'])
        if proprioceptions.shape[0] != len(rewards):
            raise ValueError(
                'LIBERO state length does not match actions/rewards: '
                f'{proprioceptions.shape[0]} != {len(rewards)}.'
            )
        episode['proprioceptions'] = proprioceptions
    if language_tokens is not None:
        input_ids, attention_mask = language_tokens
        episode['language_input_ids'] = np.repeat(input_ids[None], len(rewards), axis=0).astype(np.int64)
        episode['language_attention_mask'] = np.repeat(
            attention_mask[None],
            len(rewards),
            axis=0,
        ).astype(np.int64)
    return episode


def _get_libero_episode_infos(
    hdf5_paths,
    flip_rgb,
    discount,
    language_tokens_by_task=None,
    include_proprioception=False,
):
    episode_infos = []
    total_steps = 0
    metadata = None

    for task_index, path in enumerate(hdf5_paths):
        with h5py.File(path, 'r') as f:
            data = f['data']
            for demo_key in list(data.keys()):
                demo = data[demo_key]
                episode_len = int(demo['actions'].shape[0])
                total_steps += episode_len
                if metadata is None:
                    sample_episode = _build_libero_episode(
                        demo,
                        flip_rgb=flip_rgb,
                        discount=discount,
                        language_tokens=(
                            language_tokens_by_task[0][task_index],
                            language_tokens_by_task[1][task_index],
                        ),
                        include_proprioception=include_proprioception,
                    )
                    metadata = {
                        'observations': {
                            'shape': sample_episode['observations'].shape[1:],
                            'dtype': np.uint8,
                        },
                        'actions': {
                            'shape': sample_episode['actions'].shape[1:],
                            'dtype': np.float32,
                        },
                        'rewards': {'shape': (), 'dtype': np.float32},
                        'terminals': {'shape': (), 'dtype': np.float32},
                        'masks': {'shape': (), 'dtype': np.float32},
                        'mc_returns': {'shape': (), 'dtype': np.float32},
                    }
                    if include_proprioception:
                        metadata['proprioceptions'] = {
                            'shape': (LIBERO_PROPRIO_DIM,),
                            'dtype': np.float32,
                        }
                    metadata['language_input_ids'] = {
                        'shape': sample_episode['language_input_ids'].shape[1:],
                        'dtype': np.int64,
                    }
                    metadata['language_attention_mask'] = {
                        'shape': sample_episode['language_attention_mask'].shape[1:],
                        'dtype': np.int64,
                    }
                episode_infos.append((path, demo_key, task_index))

    if metadata is None:
        raise ValueError(f'No demonstrations found in {hdf5_paths}')
    return episode_infos, total_steps, metadata


def _ensure_libero_buffer(
    hdf5_path,
    buffer_path=None,
    flip_rgb=True,
    discount=0.99,
    language_model_path=None,
    language_max_length=77,
    language_tokenizer_type='clip',
    include_proprioception=False,
):
    hdf5_paths = _resolve_hdf5_paths(hdf5_path)
    language_tokens_by_task = _tokenize_libero_task_paths(
        hdf5_paths,
        model_path=language_model_path,
        max_length=language_max_length,
        tokenizer_type=language_tokenizer_type,
    )

    resolved_buffer_path = _resolve_libero_buffer_path(
        hdf5_path,
        buffer_path=buffer_path,
        language_tokenizer_type=language_tokenizer_type,
    )
    episode_infos, total_steps, metadata = _get_libero_episode_infos(
        hdf5_paths,
        flip_rgb=flip_rgb,
        discount=discount,
        language_tokens_by_task=language_tokens_by_task,
        include_proprioception=bool(include_proprioception),
    )

    attributes = None
    if (
        str(language_tokenizer_type).strip().lower() not in ('clip', '')
    ):
        attributes = {
            'language_tokenizer_type': str(language_tokenizer_type),
            'language_max_length': int(language_max_length),
        }
    if include_proprioception:
        attributes = dict(attributes or {})
        attributes.update({
            'include_proprioception': True,
            'proprioception_format': LIBERO_PROPRIO_FORMAT,
            'proprio_dim': LIBERO_PROPRIO_DIM,
        })

    def open_buffer(storage_path):
        return ZarrTrajectoryStore(
            storage_path=storage_path,
            schema=metadata,
            max_steps=total_steps,
            attributes=attributes,
        )

    with _buffer_build_lock(resolved_buffer_path):
        if os.path.exists(resolved_buffer_path):
            buffer = open_buffer(resolved_buffer_path)
            _validate_complete_buffer(
                buffer, len(episode_infos), total_steps, f'LIBERO at {resolved_buffer_path}'
            )
            if include_proprioception:
                _ensure_libero_proprioception_stats(buffer)
            return buffer

        temporary_buffer = f'{resolved_buffer_path}.incomplete.{os.getpid()}'
        if os.path.exists(temporary_buffer):
            shutil.rmtree(temporary_buffer)
        buffer = open_buffer(temporary_buffer)
        try:
            progress = tqdm(
                episode_infos,
                total=len(episode_infos),
                desc='Building LIBERO zarr buffer',
                dynamic_ncols=True,
            )
            for path, demo_key, task_index in progress:
                with h5py.File(path, 'r') as f:
                    demo = f['data'][demo_key]
                    episode = _build_libero_episode(
                        demo,
                        flip_rgb=flip_rgb,
                        discount=discount,
                        language_tokens=(
                            language_tokens_by_task[0][task_index],
                            language_tokens_by_task[1][task_index],
                        ),
                        include_proprioception=bool(include_proprioception),
                    )
                    buffer.append_episode(episode)
            _validate_complete_buffer(
                buffer, len(episode_infos), total_steps, f'LIBERO at {temporary_buffer}'
            )
            if include_proprioception:
                _ensure_libero_proprioception_stats(buffer)
            del buffer
            os.replace(temporary_buffer, resolved_buffer_path)
        finally:
            if os.path.exists(temporary_buffer):
                shutil.rmtree(temporary_buffer)

        return open_buffer(resolved_buffer_path)


def _resolve_robomimic_buffer_path(
    hdf5_path,
    buffer_path=None,
    include_proprioception=False,
):
    if buffer_path is not None:
        return os.path.abspath(os.path.expanduser(buffer_path))
    resolved_hdf5 = os.path.abspath(os.path.expanduser(hdf5_path))
    stem = os.path.splitext(os.path.basename(resolved_hdf5))[0]
    suffix = '_romanflow_proprio.zarr' if include_proprioception else '_romanflow.zarr'
    return os.path.join(os.path.dirname(resolved_hdf5), f'{stem}{suffix}')


def _read_robomimic_terminals(demo, horizon):
    if 'dones' in demo:
        terminals = demo['dones'][:].astype(np.float32)
    elif 'terminals' in demo:
        terminals = demo['terminals'][:].astype(np.float32)
    else:
        terminals = np.zeros((horizon,), dtype=np.float32)
    if terminals.shape != (horizon,):
        raise ValueError(
            f'RoboMimic terminal shape must be ({horizon},), got {terminals.shape}.'
        )
    if horizon > 0:
        terminals[-1] = 1.0
    return terminals


def _build_robomimic_episode(
    demo,
    discount,
    image_keys=ROBOMIMIC_IMAGE_KEYS,
    include_proprioception=False,
    proprio_keys=ROBOMIMIC_PROPRIO_KEYS,
):
    actions = demo['actions'][:].astype(np.float32, copy=False)
    horizon = int(actions.shape[0])
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f'Expected RoboMimic actions with shape [T, 7], got {actions.shape}.'
        )

    images = []
    for key in image_keys:
        if key not in demo['obs']:
            raise KeyError(f'RoboMimic demo is missing observation key {key!r}.')
        image = demo['obs'][key][:]
        if image.shape != (horizon, 84, 84, 3):
            raise ValueError(
                f'Expected {key} with shape ({horizon}, 84, 84, 3), got {image.shape}.'
            )
        images.append(np.transpose(image, (0, 3, 1, 2)))
    observations = np.stack(images, axis=1).astype(np.uint8, copy=False)

    if 'rewards' not in demo:
        raise KeyError('RoboMimic offline RL requires per-step `rewards`.')
    rewards = demo['rewards'][:].astype(np.float32)
    if rewards.shape != (horizon,):
        raise ValueError(
            f'RoboMimic reward shape must be ({horizon},), got {rewards.shape}.'
        )
    terminals = _read_robomimic_terminals(demo, horizon)
    masks = 1.0 - terminals
    mc_returns = _compute_return_to_go(rewards, terminals, discount)
    episode = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'masks': masks,
        'mc_returns': mc_returns,
    }
    if include_proprioception:
        proprio_fields = []
        for key in proprio_keys:
            if key not in demo['obs']:
                raise KeyError(f'RoboMimic demo is missing proprioception key {key!r}.')
            value = demo['obs'][key][:].astype(np.float32, copy=False)
            if value.ndim != 2 or value.shape[0] != horizon:
                raise ValueError(
                    f'Expected {key} with shape [T, D], got {value.shape}.'
                )
            proprio_fields.append(value)
        proprioceptions = np.concatenate(proprio_fields, axis=-1).astype(
            np.float32,
            copy=False,
        )
        if proprioceptions.shape != (horizon, 9):
            raise ValueError(
                'RoboMimic proprioception must contain eef_pos(3), eef_quat(4), '
                f'and gripper_qpos(2), got {proprioceptions.shape}.'
            )
        episode['proprioceptions'] = proprioceptions
    return episode


def _get_robomimic_episode_infos(
    hdf5_path,
    discount,
    image_keys=ROBOMIMIC_IMAGE_KEYS,
    include_proprioception=False,
    proprio_keys=ROBOMIMIC_PROPRIO_KEYS,
):
    resolved_hdf5 = os.path.abspath(os.path.expanduser(hdf5_path))
    if not os.path.isfile(resolved_hdf5):
        raise FileNotFoundError(f'RoboMimic dataset was not found: {resolved_hdf5}')

    with h5py.File(resolved_hdf5, 'r') as f:
        if 'data' not in f:
            raise ValueError(f'RoboMimic dataset has no `data` group: {resolved_hdf5}')
        demo_keys = list(f['data'].keys())
        if not demo_keys:
            raise ValueError(f'RoboMimic dataset contains no demonstrations: {resolved_hdf5}')
        sample_episode = _build_robomimic_episode(
            f['data'][demo_keys[0]],
            discount=discount,
            image_keys=image_keys,
            include_proprioception=include_proprioception,
            proprio_keys=proprio_keys,
        )
        total_steps = sum(int(f['data'][key]['actions'].shape[0]) for key in demo_keys)

    metadata = {
        key: {'shape': value.shape[1:], 'dtype': value.dtype}
        for key, value in sample_episode.items()
    }
    return resolved_hdf5, demo_keys, total_steps, metadata


def _ensure_robomimic_buffer(
    hdf5_path,
    buffer_path=None,
    discount=0.99,
    image_keys=ROBOMIMIC_IMAGE_KEYS,
    include_proprioception=False,
    proprio_keys=ROBOMIMIC_PROPRIO_KEYS,
):
    resolved_hdf5, demo_keys, total_steps, metadata = _get_robomimic_episode_infos(
        hdf5_path,
        discount=discount,
        image_keys=image_keys,
        include_proprioception=include_proprioception,
        proprio_keys=proprio_keys,
    )
    resolved_buffer = _resolve_robomimic_buffer_path(
        resolved_hdf5,
        buffer_path=buffer_path,
        include_proprioception=include_proprioception,
    )
    attributes = {
        'dataset_format': (
            'robomimic_mh_image_proprio_v1'
            if include_proprioception
            else 'robomimic_mh_image_v141'
        ),
        'image_keys': list(image_keys),
        'discount': float(discount),
    }
    if include_proprioception:
        attributes.update({
            'include_proprioception': True,
            'proprio_keys': list(proprio_keys),
        })

    def open_buffer(storage_path):
        return ZarrTrajectoryStore(
            storage_path=storage_path,
            schema=metadata,
            max_steps=total_steps,
            attributes=attributes,
        )

    with _buffer_build_lock(resolved_buffer):
        if os.path.exists(resolved_buffer):
            buffer = open_buffer(resolved_buffer)
            _validate_complete_buffer(
                buffer, len(demo_keys), total_steps, f'RoboMimic at {resolved_buffer}'
            )
            return buffer

        temporary_buffer = f'{resolved_buffer}.incomplete.{os.getpid()}'
        if os.path.exists(temporary_buffer):
            shutil.rmtree(temporary_buffer)
        buffer = open_buffer(temporary_buffer)
        try:
            with h5py.File(resolved_hdf5, 'r') as f:
                progress = tqdm(
                    demo_keys,
                    desc='Building RoboMimic zarr buffer',
                    dynamic_ncols=True,
                )
                for demo_key in progress:
                    buffer.append_episode(
                        _build_robomimic_episode(
                            f['data'][demo_key],
                            discount=discount,
                            image_keys=image_keys,
                            include_proprioception=include_proprioception,
                            proprio_keys=proprio_keys,
                        )
                    )
            _validate_complete_buffer(
                buffer, len(demo_keys), total_steps, f'RoboMimic at {temporary_buffer}'
            )
            del buffer
            os.replace(temporary_buffer, resolved_buffer)
        finally:
            if os.path.exists(temporary_buffer):
                shutil.rmtree(temporary_buffer)

        return open_buffer(resolved_buffer)


class LazyLiberoDataset:
    """Thin torch-compatible wrapper around an episodic Zarr store."""

    def __init__(self, buffer, episode_mask, discount, hubl_alpha=0.1):
        self.buffer = buffer
        self.episode_mask = np.asarray(episode_mask, dtype=bool)
        self.discount = float(discount)
        self.hubl_alpha = float(hubl_alpha)
        self.p_aug = None
        self.visual_augmentation_mode = 'libero_224'
        self.apply_visual_eval_preprocessing = False
        self._observation_horizon = None
        self._action_horizon = None
        self._sequence_length = 1
        self._action_clip_eps = None
        self._episode_ends = np.asarray(self.buffer.episode_ends[:], dtype=np.int64)
        self._episode_rank_lambdas = None
        self.sampler = None
        self.size = 0
        self._rebuild_sampler_and_hubl()

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = int(value)
        self._rebuild_sampler_and_hubl()

    @property
    def observation_horizon(self):
        return self._observation_horizon

    @observation_horizon.setter
    def observation_horizon(self, value):
        self._observation_horizon = None if value is None else int(value)
        if self.sampler is not None:
            self._rebuild_sampler_and_hubl()

    @property
    def action_horizon(self):
        return self._action_horizon

    @action_horizon.setter
    def action_horizon(self, value):
        self._action_horizon = None if value is None else int(value)
        if self.sampler is not None:
            self._rebuild_sampler_and_hubl()

    def _rebuild_sampler_and_hubl(self):
        pad_after = sequence_right_padding(
            self._sequence_length,
            self._observation_horizon,
            self._action_horizon,
        )
        self.sampler = EpisodeSequenceIndex(
            self.buffer,
            self._sequence_length,
            self.episode_mask,
            pad_after=pad_after,
        )
        self.size = len(self.sampler)
        self._episode_rank_lambdas = self._compute_split_episode_lambdas()

    def _compute_split_episode_lambdas(self):
        selected_episode_indices = np.nonzero(self.episode_mask)[0]
        if len(selected_episode_indices) == 0:
            return {}

        trajectory_mc_returns = []
        episode_start = 0
        for episode_idx, episode_end in enumerate(self._episode_ends):
            episode_end = int(episode_end)
            if self.episode_mask[episode_idx]:
                trajectory_mc_returns.append(
                    np.asarray(self.buffer['mc_returns'][episode_start:episode_end], dtype=np.float32)
                )
            episode_start = episode_end

        split_hubl_lambdas = _compute_rank_hubl_lambdas(trajectory_mc_returns, alpha=self.hubl_alpha)
        return {
            int(episode_idx): split_hubl_lambdas[idx]
            for idx, episode_idx in enumerate(selected_episode_indices)
        }

    def set_action_clip_eps(self, action_clip_eps):
        self._action_clip_eps = action_clip_eps

    def _get_episode_idx_for_sequence(self, sequence_idx):
        start, _ = self.sampler.indices[int(sequence_idx)]
        return min(
            int(np.searchsorted(self._episode_ends, int(start), side='right')),
            len(self._episode_ends) - 1,
        )

    def _build_hubl_fields(self, sequence_idx, batch):
        episode_idx = self._get_episode_idx_for_sequence(sequence_idx)
        episode_lambda = self._episode_rank_lambdas[int(episode_idx)]
        start, end = self.sampler.indices[int(sequence_idx)]
        episode_start = 0 if episode_idx == 0 else int(self._episode_ends[int(episode_idx) - 1])
        episode_end = int(self._episode_ends[int(episode_idx)])
        rewards = np.asarray(batch['rewards'], dtype=np.float32)
        masks = np.asarray(batch['masks'], dtype=np.float32)
        valid = np.asarray(batch['sequence_valid_mask'], dtype=np.float32)
        real_len = int(valid.sum())
        local_start = int(start) - episode_start
        hubl_lambda = np.zeros_like(rewards, dtype=np.float32)
        hubl_lambda[:real_len] = np.asarray(
            episode_lambda[local_start : local_start + real_len], dtype=np.float32
        )

        next_mc_returns = np.zeros_like(rewards, dtype=np.float32)
        next_lambda = np.zeros_like(hubl_lambda, dtype=np.float32)
        real_with_next = max(0, min(real_len, episode_end - int(start) - 1))
        if real_with_next > 0:
            next_mc_returns[:real_with_next] = np.asarray(
                self.buffer['mc_returns'][int(start) + 1 : int(start) + 1 + real_with_next],
                dtype=np.float32,
            )
            next_lambda[:real_with_next] = hubl_lambda[:real_with_next]

        hubl_rewards = rewards + self.discount * masks * next_lambda * next_mc_returns
        hubl_discounts = self.discount * masks * (1.0 - next_lambda)
        batch['hubl_lambda'] = hubl_lambda
        batch['hubl_rewards'] = hubl_rewards.astype(np.float32)
        batch['hubl_discounts'] = hubl_discounts.astype(np.float32)
        return batch

    def _clip_actions(self, batch):
        if self._action_clip_eps is None or 'actions' not in batch:
            return batch
        batch['actions'] = np.clip(
            batch['actions'],
            -1 + self._action_clip_eps,
            1 - self._action_clip_eps,
        )
        return batch

    def _to_torch_batch(self, batch):
        torch_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                torch_batch[key] = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                torch_batch[key] = torch.from_numpy(np.ascontiguousarray(value))
            else:
                torch_batch[key] = torch.as_tensor(value)
        return torch_batch

    def _maybe_apply_visual_augmentation(self, batch):
        mode = canonicalize_libero_visual_augmentation_mode(
            self.visual_augmentation_mode
        )
        if bool(getattr(self, 'apply_visual_eval_preprocessing', False)):
            if 'observations' not in batch:
                return batch
            observations = np.asarray(batch['observations'])
            if observations.ndim != 6:
                raise ValueError(
                    'LIBERO validation preprocessing expects observations with shape '
                    f'[B, T, V, C, H, W], got {observations.shape}.'
                )
            online_layout = torch.from_numpy(
                np.ascontiguousarray(observations)
            ).permute(0, 2, 3, 1, 4, 5)
            processed = apply_libero_visual_eval_preprocessing(
                online_layout,
                mode=mode,
            )
            batch['observations'] = processed.permute(0, 3, 1, 2, 4, 5).cpu().numpy()
            return batch

        probability = self.p_aug
        if probability is None:
            return batch
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f'Image augmentation probability must be in [0, 1], got {probability}.'
            )
        if probability == 0.0 or 'observations' not in batch:
            return batch
        observations = np.asarray(batch['observations'])
        batch_size = observations.shape[0]
        apply_mask = (
            np.ones((batch_size,), dtype=bool)
            if probability == 1.0
            else np.random.random(batch_size) < probability
        )
        if probability == 1.0:
            augmented = _apply_libero_visual_augmentation(observations, mode=mode)
        else:
            # Keep one tensor shape when only part of a batch is augmented.
            online_layout = torch.from_numpy(
                np.ascontiguousarray(observations)
            ).permute(0, 2, 3, 1, 4, 5)
            augmented = apply_libero_visual_eval_preprocessing(
                online_layout,
                mode=mode,
            ).permute(0, 3, 1, 2, 4, 5).cpu().numpy()
            if np.any(apply_mask):
                augmented[apply_mask] = _apply_libero_visual_augmentation(
                    observations[apply_mask],
                    mode=mode,
                )
        batch['observations'] = augmented
        return batch

    def get_subset(self, idxs):
        idxs = np.atleast_1d(np.asarray(idxs, dtype=np.int64))
        samples = []
        for idx in idxs:
            sample = self.sampler.sample_sequence(int(idx))
            sample = self._build_hubl_fields(int(idx), sample)
            sample = self._clip_actions(sample)
            samples.append(sample)
        batch = {
            key: np.stack([sample[key] for sample in samples], axis=0)
            for key in samples[0].keys()
        }
        batch = self._maybe_apply_visual_augmentation(batch)
        return self._to_torch_batch(batch)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = np.random.randint(len(self.sampler), size=batch_size)
        return self.get_subset(idxs)


def configure_libero_proprioception_stats(config, dataset) -> None:
    """Copy the buffer's training quantiles into model configuration."""
    if not bool(config.get('use_proprioception', False)):
        return
    root = getattr(getattr(dataset, 'buffer', None), 'root', None)
    if root is None:
        raise ValueError('LIBERO proprioception requires a disk-backed trajectory buffer.')
    if root.attrs.get('proprioception_format') != LIBERO_PROPRIO_FORMAT:
        raise ValueError(
            'LIBERO proprioception buffer has no compatible state format: '
            f'{root.attrs.get("proprioception_format")!r}.'
        )
    q01 = np.asarray(root.attrs.get('proprio_q01'), dtype=np.float32)
    q99 = np.asarray(root.attrs.get('proprio_q99'), dtype=np.float32)
    if q01.shape != (LIBERO_PROPRIO_DIM,) or q99.shape != (LIBERO_PROPRIO_DIM,):
        raise ValueError(
            f'Expected {LIBERO_PROPRIO_DIM}-D LIBERO state quantiles, got '
            f'{q01.shape}/{q99.shape}.'
        )
    if np.any(q99 <= q01):
        raise ValueError('LIBERO proprioception q99 must be greater than q01 in every dimension.')
    config.proprio_dim = LIBERO_PROPRIO_DIM
    config.proprio_q01 = q01.tolist()
    config.proprio_q99 = q99.tolist()


def get_libero_dataset_torch(hdf5_path, val_ratio=0.05, flip_rgb=True, discount=0.99,
                             hubl_lambda_type='rank', hubl_alpha=0.1, buffer_path=None,
                             language_model_path=None,
                             language_max_length=77,
                             language_tokenizer_type='clip',
                             include_proprioception=False):
    if hubl_lambda_type != 'rank':
        raise ValueError(f"Unsupported HUBL lambda type `{hubl_lambda_type}` for LIBERO dataset.")
    buffer = _ensure_libero_buffer(
        hdf5_path,
        buffer_path=buffer_path,
        flip_rgb=flip_rgb,
        discount=discount,
        language_model_path=language_model_path,
        language_max_length=language_max_length,
        language_tokenizer_type=language_tokenizer_type,
        include_proprioception=include_proprioception,
    )

    num_episodes = buffer.num_episodes
    num_val = max(1, int(val_ratio * num_episodes)) if val_ratio > 0 else 0
    num_train = max(1, num_episodes - num_val)

    train_mask = np.zeros((num_episodes,), dtype=bool)
    train_mask[:num_train] = True
    val_mask = np.zeros((num_episodes,), dtype=bool)
    val_mask[num_train:] = True

    train_dataset = LazyLiberoDataset(
        buffer=buffer,
        episode_mask=train_mask,
        discount=discount,
        hubl_alpha=hubl_alpha,
    )
    val_dataset = None
    if np.any(val_mask):
        val_dataset = LazyLiberoDataset(
            buffer=buffer,
            episode_mask=val_mask,
            discount=discount,
            hubl_alpha=hubl_alpha,
        )
    return train_dataset, val_dataset


def get_robomimic_dataset_torch(
    hdf5_path,
    val_ratio=0.0,
    discount=0.997,
    hubl_lambda_type='rank',
    hubl_alpha=0.15,
    buffer_path=None,
    include_proprioception=False,
):
    if hubl_lambda_type != 'rank':
        raise ValueError(
            f'Unsupported HUBL lambda type {hubl_lambda_type!r} for RoboMimic.'
        )
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f'val_ratio must be in [0, 1), got {val_ratio}.')

    buffer = _ensure_robomimic_buffer(
        hdf5_path,
        buffer_path=buffer_path,
        discount=discount,
        include_proprioception=include_proprioception,
    )
    num_episodes = buffer.num_episodes
    num_val = max(1, int(val_ratio * num_episodes)) if val_ratio > 0 else 0
    num_train = num_episodes - num_val
    if num_train <= 0:
        raise ValueError('RoboMimic split contains no training demonstrations.')

    train_mask = np.zeros((num_episodes,), dtype=bool)
    train_mask[:num_train] = True
    val_mask = ~train_mask
    train_dataset = LazyLiberoDataset(
        buffer=buffer,
        episode_mask=train_mask,
        discount=discount,
        hubl_alpha=hubl_alpha,
    )
    val_dataset = None
    if np.any(val_mask):
        val_dataset = LazyLiberoDataset(
            buffer=buffer,
            episode_mask=val_mask,
            discount=discount,
            hubl_alpha=hubl_alpha,
        )
    return train_dataset, val_dataset
