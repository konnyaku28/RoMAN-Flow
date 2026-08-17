import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import glob
import json
import random
import time
import copy
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn.parallel import DistributedDataParallel as DDP

from envs.env_utils import (
    ensure_libero_config,
    is_libero_env_name,
    is_robomimic_env_name,
    make_env,
    make_env_and_datasets,
    validate_supported_env,
)
from utils.evaluation import torch_evaluate_parallel
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb


from agents.vinf_torch import (
    VINFTorchAgent,
    uses_proprioception,
)


FLAGS = flags.FLAGS

flags.DEFINE_string('wandb_dir', 'exp/', 'Wandb dir.')
flags.DEFINE_string('wandb_name_tag', '', 'Wandb dir.')
flags.DEFINE_string('wandb_entity', '', 'Wandb entity.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_bool('tensorboard', False, 'Enable TensorBoard scalar logging on the main process.')
flags.DEFINE_string(
    'tensorboard_dir',
    None,
    'TensorBoard log directory. Defaults to <save_dir>/tensorboard when tensorboard is enabled.',
)
flags.DEFINE_integer('tensorboard_flush_secs', 30, 'TensorBoard writer flush interval in seconds.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string(
    'dataset_dir',
    '/path/to/libero_data/libero_10/*.hdf5',
    'Local LIBERO HDF5 path/glob or one RoboMimic image HDF5 file.',
)
flags.DEFINE_string('libero_buffer_path', None, 'Optional persistent zarr buffer path for LIBERO lazy loading.')
flags.DEFINE_string('robomimic_buffer_path', None, 'Optional persistent zarr buffer path for RoboMimic lazy loading.')

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'libero_10', 'Release-supported environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string(
    'restore_output_dir',
    None,
    'Optional output directory for resumed training. By default, resume writes in place.',
)
flags.DEFINE_string('pretrain_path', None, 'Optional checkpoint file to initialize model weights only.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('biflow_align_steps', 0, 'If > 0, run BiFlow alignment for this many steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')
flags.DEFINE_string(
    'save_steps',
    '',
    'Optional comma-separated absolute training steps to save in addition to save_interval.',
)
flags.DEFINE_bool(
    'save_stage_boundaries',
    True,
    'Save at BiFlow alignment/final boundaries even when save_interval is zero.',
)

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer(
    'eval_num_envs',
    0,
    'Number of parallel evaluation environments; 0 uses one environment per episode.',
)
flags.DEFINE_integer(
    'eval_action_exec_horizon',
    None,
    'Optional number of predicted action-chunk steps to execute per policy call during evaluation.',
)

flags.DEFINE_integer('obs_horizon', 2, 'Observation horizon.')
flags.DEFINE_integer('action_horizon', 16, 'Action horizon.')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_enum(
    'libero_visual_augmentation_mode',
    'libero_224',
    ('libero_224',),
    'LIBERO augmentation geometry: resize to 240 then crop to 224.',
)
flags.DEFINE_integer(
    'global_batch_size',
    0,
    'Explicit DDP-global training batch size; 0 uses the agent configuration.',
)
flags.DEFINE_bool(
    'batch_prefetch',
    False,
    'Overlap train_dataset.sample with GPU update by prefetching one training batch on a background thread.',
)

flags.DEFINE_bool('ddp', False, 'Enable DDP multi-GPU training.')
flags.DEFINE_string('ddp_backend', 'nccl', 'DDP backend.')
flags.DEFINE_integer('ddp_timeout_seconds', 3600, 'Timeout for distributed collectives in seconds.')
flags.DEFINE_bool('ddp_static_graph', True, 'Enable DDP static graph optimization.')
flags.DEFINE_bool('ddp_find_unused_parameters', False, 'Enable DDP unused parameter detection.')
flags.DEFINE_string(
    'agent_flags_json',
    None,
    'Optional release flags.json whose non-null agent values override the base agent config.',
)

config_flags.DEFINE_config_file('agent', 'agents/vinf_torch.py', lock_config=False)


def resolve_visual_augmentation_probability(p_aug=None):
    """Validate the LIBERO image augmentation probability."""
    probability = p_aug
    if probability is None:
        return None
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f'Visual augmentation probability must be in [0, 1], got {probability}.'
        )
    return probability


def parse_save_steps(value: str | None) -> frozenset[int]:
    if value is None or not str(value).strip():
        return frozenset()
    steps = set()
    for item in str(value).split(','):
        item = item.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError as exc:
            raise ValueError(f'Invalid save step {item!r} in save_steps={value!r}.') from exc
        if step <= 0:
            raise ValueError(f'save_steps must contain positive integers, got {step}.')
        steps.add(step)
    return frozenset(steps)


def resolve_training_save_dir(
    *,
    wandb_dir: str,
    run_group: str,
    exp_name: str,
    restore_path: str | None = None,
    restore_output_dir: str | None = None,
) -> str:
    """Resolve a training output directory without conflating restore input and output."""
    if restore_output_dir is not None:
        if restore_path is None:
            raise ValueError('`restore_output_dir` requires `restore_path`.')
        source = os.path.realpath(os.path.expanduser(restore_path))
        output = os.path.realpath(os.path.expanduser(restore_output_dir))
        if source == output:
            raise ValueError('`restore_output_dir` must differ from `restore_path`.')
        return restore_output_dir
    if restore_path is not None:
        return restore_path
    return os.path.join(wandb_dir, 'fql-orl-torch', run_group, exp_name)


RUNTIME = {}


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def init_distributed(rank: int, world_size: int, runtime, local_rank: int | None = None):
    os.environ.setdefault('MASTER_ADDR', runtime['master_addr'])
    os.environ.setdefault('MASTER_PORT', runtime['master_port'])
    local_rank = rank if local_rank is None else int(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=runtime['ddp_backend'],
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=int(runtime.get('ddp_timeout_seconds', 3600))),
        device_id=torch.device(f'cuda:{local_rank}') if runtime['ddp_backend'] == 'nccl' else None,
    )


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def save_agent_torch(agent, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'params_{epoch}.pt')
    state_dict = agent.state_dict()
    state_dict['model'] = agent.raw_model.state_dict()
    torch.save(state_dict, save_path)
    print(f'Saved to {save_path}')


def _match_model_state_dict_prefix(model_state_dict, expect_module_prefix: bool):
    has_module_prefix = any(key.startswith('module.') for key in model_state_dict.keys())
    if has_module_prefix == expect_module_prefix:
        return model_state_dict

    if expect_module_prefix:
        return {f'module.{key}': value for key, value in model_state_dict.items()}
    return {
        (key[len('module.'):] if key.startswith('module.') else key): value
        for key, value in model_state_dict.items()
    }


def restore_agent_torch(agent, restore_path, restore_epoch):
    load_path = os.path.join(restore_path, f'params_{restore_epoch}.pt')
    state_dict = torch.load(load_path, map_location=agent.device)
    state_dict = dict(state_dict)
    state_dict['model'] = _match_model_state_dict_prefix(
        state_dict['model'],
        expect_module_prefix=hasattr(agent.model, 'module'),
    )
    agent.load_state_dict(state_dict)
    print(f'Restored from {load_path}')
    return agent


def _normalized_state_key(key: str) -> str:
    return key[len('module.'):] if key.startswith('module.') else key


def _normalized_actor_key(key: str) -> str:
    normalized_key = _normalized_state_key(key)
    return normalized_key[len('actor.'):] if normalized_key.startswith('actor.') else normalized_key


def _agent_needs_language_tokens(config) -> bool:
    return (
        bool(config.get('use_language_conditioning', False))
        or bool(config.get('critic_use_language_conditioning', False))
        or bool(config.get('value_use_language_conditioning', False))
    )


def _language_tokenizer_spec(config) -> tuple[str, str | None, int]:
    use_smolvlm = (
        str(config.get('conditioning_mode', '')).strip().lower() == 'context'
        and bool(config.get('vlm_fuse', False))
    )
    if use_smolvlm:
        return (
            'smolvlm',
            config.get('smolvlm_model_path', None),
            int(config.get('vlm_language_max_length', 50)),
        )
    return (
        'clip',
        config.get('language_model_path', None),
        int(config.get('language_max_length', 77)),
    )


def _is_allowed_pretrain_missing_key(key: str, allow_critic_reinit: bool = False) -> bool:
    normalized_key = _normalized_state_key(key)
    actor_key = _normalized_actor_key(key)
    if normalized_key.startswith('state_text_encoder.'):
        return True
    if normalized_key.startswith((
        'critic.state_fusion.',
        'critic.proprio_fusion.',
        'target_critic.state_fusion.',
        'target_critic.proprio_fusion.',
        'value.state_fusion.',
        'value.proprio_fusion.',
        'critic_value_state_encoder.',
        'critic_value_state_fusion.',
        'critic_value_proprio_fusion.',
        'target_critic_value_state_encoder.',
        'target_critic_value_state_fusion.',
        'target_critic_value_proprio_fusion.',
        'proprio_normalizer.',
    )):
        return True
    if allow_critic_reinit and normalized_key.startswith(('critic.', 'target_critic.', 'value.')):
        return True
    if normalized_key.startswith('value.'):
        return True
    if normalized_key.startswith('reverse_model.') or normalized_key.startswith('reverse_model_ema.'):
        return True
    if normalized_key.startswith('actor.text_encoder.'):
        return True
    if normalized_key.startswith('actor_context_encoder.'):
        return True
    if normalized_key == 'actor.null_context_token':
        return True
    if normalized_key.startswith((
        'reverse_model.text_encoder.',
        'reverse_model_ema.text_encoder.',
    )) or '.null_language' in normalized_key:
        return True
    if normalized_key.startswith('actor_encoder.'):
        return True
    if actor_key == 'fake_obs_latent':
        return True
    if actor_key.startswith('blocks.') and (
        '.adaLN_modulation.1.' in actor_key
        or '.fake_latent' in actor_key
    ):
        return True
    if not normalized_key.startswith('actor.blocks.'):
        return False
    return '.prefix_pos_embed' in normalized_key


def _is_allowed_pretrain_unexpected_key(key: str) -> bool:
    normalized_key = _normalized_state_key(key)
    actor_key = _normalized_actor_key(key)
    if normalized_key.startswith('state_text_encoder.'):
        return True
    if normalized_key.startswith('actor.text_encoder.'):
        return True
    if normalized_key.startswith('actor_context_encoder.'):
        return True
    if normalized_key == 'actor.null_context_token':
        return True
    if normalized_key.startswith((
        'reverse_model.text_encoder.',
        'reverse_model_ema.text_encoder.',
    )) or (
        normalized_key.startswith(('reverse_model.', 'reverse_model_ema.'))
        and '.null_language' in normalized_key
    ):
        return True
    if normalized_key.startswith((
        'critic.state_fusion.',
        'critic.proprio_fusion.',
        'target_critic.state_fusion.',
        'target_critic.proprio_fusion.',
        'value.state_fusion.',
        'value.proprio_fusion.',
        'critic.encoder.',
        'target_critic.encoder.',
        'value.encoder.',
        'critic_value_state_encoder.',
        'critic_value_state_fusion.',
        'target_critic_value_state_encoder.',
        'target_critic_value_state_fusion.',
        'critic_value_proprio_fusion.',
        'target_critic_value_proprio_fusion.',
        'proprio_normalizer.',
    )):
        return True
    if actor_key == 'fake_obs_latent':
        return True
    if actor_key.startswith('prefix_fake_latent'):
        return True
    if not actor_key.startswith('blocks.'):
        return False
    return '.prefix_pos_embed' in actor_key


def _adapt_pretrained_action_horizon_tensor(key, value, current_value):
    """Crop SimFlow's sequence-only state when transferring to a shorter horizon."""
    normalized_key = _normalized_state_key(key)
    key_parts = normalized_key.split('.')
    is_actor_block_state = (
        len(key_parts) == 4
        and key_parts[0] == 'actor'
        and key_parts[1] == 'blocks'
        and key_parts[2].isdigit()
    )

    if normalized_key == 'actor.var' or (
        is_actor_block_state and key_parts[3] == 'pos_embed'
    ):
        if (
            value.ndim == current_value.ndim
            and value.shape[0] > current_value.shape[0]
            and tuple(value.shape[1:]) == tuple(current_value.shape[1:])
        ):
            return value[: current_value.shape[0]].clone()
        return None

    if is_actor_block_state and key_parts[3] == 'attn_mask':
        if (
            value.ndim == current_value.ndim == 2
            and value.shape[0] == value.shape[1]
            and current_value.shape[0] == current_value.shape[1]
            and value.shape[0] > current_value.shape[0]
        ):
            horizon = current_value.shape[0]
            return value[:horizon, :horizon].clone()
    return None


def _filter_allowed_pretrain_shape_mismatches(agent, model_state_dict):
    current_state = agent.model.state_dict()
    filtered_state = {}
    skipped_keys = []
    adapted_keys = []
    for key, value in model_state_dict.items():
        current_value = current_state.get(key)
        if current_value is None or tuple(current_value.shape) == tuple(value.shape):
            filtered_state[key] = value
            continue
        adapted_value = _adapt_pretrained_action_horizon_tensor(
            key,
            value,
            current_value,
        )
        if adapted_value is not None:
            filtered_state[key] = adapted_value
            adapted_keys.append(
                (key, tuple(value.shape), tuple(adapted_value.shape))
            )
            continue
        if _is_allowed_pretrain_missing_key(key):
            skipped_keys.append(key)
            continue
        filtered_state[key] = value
    return filtered_state, skipped_keys, adapted_keys


def load_pretrained_agent_torch(agent, pretrain_path):
    state_dict = torch.load(
        pretrain_path,
        map_location='cpu',
        mmap=True,
        weights_only=False,
    )
    model_state_dict = state_dict['model'] if isinstance(state_dict, dict) and 'model' in state_dict else state_dict
    model_state_dict = _match_model_state_dict_prefix(
        model_state_dict,
        expect_module_prefix=hasattr(agent.model, 'module'),
    )
    force_critic_reinit = bool(
        agent.config.get('reinitialize_critic_value_on_pretrain', False)
    )
    allow_critic_reinit = force_critic_reinit
    if allow_critic_reinit:
        model_state_dict = {
            key: value
            for key, value in model_state_dict.items()
            if not _normalized_state_key(key).startswith((
                'critic.',
                'target_critic.',
                'value.',
                'critic_value_state_encoder.',
                'critic_value_state_fusion.',
                'critic_value_proprio_fusion.',
                'target_critic_value_state_encoder.',
                'target_critic_value_state_fusion.',
                'target_critic_value_proprio_fusion.',
            ))
        }
        reason = (
            'explicit reinitialize_critic_value_on_pretrain=True'
            if force_critic_reinit
            else 'explicit release configuration'
        )
        print(f'Reinitializing all critic/value parameters from scratch ({reason}).')
    (
        model_state_dict,
        skipped_shape_keys,
        adapted_shape_keys,
    ) = _filter_allowed_pretrain_shape_mismatches(agent, model_state_dict)
    load_result = agent.model.load_state_dict(model_state_dict, strict=False)
    unexpected_keys = [
        key for key in load_result.unexpected_keys
        if not _is_allowed_pretrain_unexpected_key(key)
    ]
    disallowed_missing_keys = [
        key for key in load_result.missing_keys
        if not _is_allowed_pretrain_missing_key(key, allow_critic_reinit=allow_critic_reinit)
    ]
    if unexpected_keys or disallowed_missing_keys:
        raise RuntimeError(
            'Error(s) in loading pretrained model state_dict:\n'
            f'\tUnexpected key(s): {unexpected_keys}\n'
            f'\tMissing key(s): {disallowed_missing_keys}'
        )
    if allow_critic_reinit:
        raw_model = agent.raw_model
        raw_model.target_critic.load_state_dict(raw_model.critic.state_dict())
        for online_name, target_name in (
            ('critic_value_state_encoder', 'target_critic_value_state_encoder'),
            ('critic_value_state_fusion', 'target_critic_value_state_fusion'),
            ('critic_value_proprio_fusion', 'target_critic_value_proprio_fusion'),
        ):
            online_module = getattr(raw_model, online_name, None)
            target_module = getattr(raw_model, target_name, None)
            if online_module is not None and target_module is not None:
                target_module.load_state_dict(online_module.state_dict())
    if load_result.missing_keys:
        missing_keys = list(load_result.missing_keys)
        preview_keys = missing_keys[:8]
        suffix = '' if len(missing_keys) <= len(preview_keys) else f' ... (+{len(missing_keys) - len(preview_keys)} more)'
        print(
            'Loaded pretrained checkpoint with newly initialized parameters: '
            f'{preview_keys}{suffix}'
        )
    if skipped_shape_keys:
        preview_keys = skipped_shape_keys[:8]
        suffix = '' if len(skipped_shape_keys) <= len(preview_keys) else f' ... (+{len(skipped_shape_keys) - len(preview_keys)} more)'
        print(
            'Skipped shape-mismatched pretrained parameters initialized from the current model: '
            f'{preview_keys}{suffix}'
        )
    if adapted_shape_keys:
        preview_keys = adapted_shape_keys[:8]
        suffix = '' if len(adapted_shape_keys) <= len(preview_keys) else f' ... (+{len(adapted_shape_keys) - len(preview_keys)} more)'
        print(
            'Adapted pretrained SimFlow sequence tensors for a shorter action horizon: '
            f'{preview_keys}{suffix}'
        )
    del state_dict
    del model_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Loaded pretrained weights from {pretrain_path}')
    return agent


def _use_biflow_chained_training(runtime) -> bool:
    return int(runtime.get('biflow_align_steps', 0)) > 0


def _set_agent_train_mode(agent, config, train_mode: str):
    agent.config['train_mode'] = train_mode
    config.train_mode = train_mode
    agent.reset_optimizer_for_train_mode(train_mode)


def _sync_runtime_flag_dict(runtime):
    runtime['flag_dict']['offline_steps'] = runtime['offline_steps']
    runtime['flag_dict']['ddp_static_graph'] = runtime['ddp_static_graph']
    runtime['flag_dict']['ddp_find_unused_parameters'] = runtime['ddp_find_unused_parameters']
    runtime['flag_dict']['agent'] = runtime['agent_config'].to_dict()


def _get_libero_eval_task_paths():
    dataset_dir = RUNTIME.get('dataset_dir')
    if (
        not is_libero_env_name(RUNTIME.get('env_name', ''))
        or not dataset_dir
        or '*' not in dataset_dir
    ):
        return None

    task_paths = sorted(glob.glob(dataset_dir))
    if len(task_paths) == 0:
        raise ValueError(f'No LIBERO evaluation tasks matched dataset_dir glob: {dataset_dir}')
    return task_paths


def _get_eval_task_name(dataset_path: str) -> str:
    return Path(dataset_path).stem


def maybe_make_eval_vec_env(dataset_dir_override=None):
    if not is_main_process():
        return None
    if is_libero_env_name(RUNTIME.get('env_name', '')):
        ensure_libero_config()
    dataset_dir = dataset_dir_override if dataset_dir_override is not None else RUNTIME['dataset_dir']
    num_envs = int(RUNTIME.get('eval_num_envs', 0))
    if num_envs <= 0:
        num_envs = int(RUNTIME['eval_episodes'])
    num_envs = max(1, min(num_envs, int(RUNTIME['eval_episodes'])))
    return DummyVecEnv([
        make_env(
            RUNTIME['env_name'],
            dataset_dir=dataset_dir,
            obs_horizon=RUNTIME['obs_horizon'],
            robomimic_use_proprioception=uses_proprioception(
                RUNTIME['agent_config']
            ),
        )
        for i in range(num_envs)
    ])


def _add_eval_metrics(eval_metrics, prefix: str, eval_info):
    for key, value in eval_info.items():
        metric_prefix = f'{prefix}/{key}'
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                eval_metrics[metric_prefix] = float(value)
            else:
                flat_value = value.reshape(-1)
                for dim, dim_value in enumerate(flat_value):
                    eval_metrics[f'{metric_prefix}_{dim}'] = float(dim_value)
                eval_metrics[f'{metric_prefix}_mean'] = float(np.mean(value))
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
            if value.ndim == 0:
                eval_metrics[metric_prefix] = float(value)
            else:
                flat_value = value.reshape(-1)
                for dim, dim_value in enumerate(flat_value):
                    eval_metrics[f'{metric_prefix}_{dim}'] = float(dim_value)
                eval_metrics[f'{metric_prefix}_mean'] = float(np.mean(value))
        else:
            eval_metrics[metric_prefix] = float(value) if np.isscalar(value) else value


def _to_tensorboard_scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0].item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _make_tensorboard_writer(runtime, save_dir):
    if not bool(runtime.get('tensorboard', False)):
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            'TensorBoard logging was requested with `--tensorboard=True`, '
            'but `torch.utils.tensorboard` could not be imported. Install tensorboard first.'
        ) from exc

    log_dir = runtime.get('tensorboard_dir')
    if not log_dir:
        log_dir = os.path.join(save_dir, 'tensorboard')
    log_dir = os.path.abspath(os.path.expanduser(str(log_dir)))
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(
        log_dir=log_dir,
        flush_secs=int(runtime.get('tensorboard_flush_secs', 30)),
    )
    print(f'TensorBoard logging enabled: {log_dir}')
    return writer


def _log_tensorboard_metrics(writer, metrics, step):
    if writer is None:
        return
    for key, value in metrics.items():
        scalar = _to_tensorboard_scalar(value)
        if scalar is not None:
            writer.add_scalar(key, scalar, step)
    writer.flush()


def _is_robomimic_run() -> bool:
    return is_robomimic_env_name(RUNTIME['env_name'])

def build_training_state(device):
    rank = dist.get_rank() if is_distributed() else 0
    world_size = dist.get_world_size() if is_distributed() else 1
    config = RUNTIME['agent_config']
    if RUNTIME['global_batch_size'] > 0:
        config.batch_size = RUNTIME['global_batch_size']
    global_batch_size = int(config.batch_size)
    if global_batch_size % world_size != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} must be divisible by world size {world_size}.'
        )
    per_device_batch_size = global_batch_size // world_size
    language_tokenizer_type, language_tokenizer_path, language_tokenizer_max_length = (
        _language_tokenizer_spec(config)
    )

    dataset_kwargs = dict(
        env_name=RUNTIME['env_name'],
        dataset_dir=RUNTIME['dataset_dir'],
        discount=float(config.discount),
        hubl_lambda_type=str(config.get('hubl_lambda_type', 'rank')),
        hubl_alpha=float(config.get('hubl_alpha', 0.1)),
        libero_buffer_path=RUNTIME.get('libero_buffer_path'),
        robomimic_buffer_path=RUNTIME.get('robomimic_buffer_path'),
        robomimic_use_proprioception=uses_proprioception(config),
        language_model_path=language_tokenizer_path,
        language_max_length=language_tokenizer_max_length,
        language_tokenizer_type=language_tokenizer_type,
        action_clip_eps=1e-5,
    )

    if _is_robomimic_run():
        if rank == 0:
            from utils.robomimic.dataset import _ensure_robomimic_buffer

            _ensure_robomimic_buffer(
                RUNTIME['dataset_dir'],
                buffer_path=RUNTIME.get('robomimic_buffer_path'),
                discount=float(config.discount),
                include_proprioception=bool(
                    config.get('robomimic_use_proprioception', False)
                ),
            )
        if is_distributed():
            dist.barrier()

    train_dataset, val_dataset = make_env_and_datasets(**dataset_kwargs)

    if is_libero_env_name(RUNTIME['env_name']) and bool(
        config.get('use_proprioception', False)
    ):
        from utils.robomimic.dataset import configure_libero_proprioception_stats

        configure_libero_proprioception_stats(config, train_dataset)

    if RUNTIME['obs_horizon'] is not None:
        _, _, sequence_length = VINFTorchAgent._get_sequence_spec(config)
        for dataset in [train_dataset, val_dataset]:
            if dataset is not None:
                dataset.sequence_length = sequence_length
                dataset.observation_horizon = RUNTIME['obs_horizon']
                dataset.action_horizon = RUNTIME['action_horizon']

    is_libero_run = is_libero_env_name(RUNTIME['env_name'])
    for dataset in [train_dataset, val_dataset]:
        if dataset is not None:
            dataset.p_aug = RUNTIME['p_aug'] if dataset is train_dataset else None
            if is_libero_run:
                dataset.visual_augmentation_mode = RUNTIME['libero_visual_augmentation_mode']
                dataset.apply_visual_eval_preprocessing = bool(
                    dataset is val_dataset
                    and RUNTIME['p_aug'] is not None
                    and float(RUNTIME['p_aug']) > 0.0
                )

    example_batch = train_dataset.sample(1)
    agent = VINFTorchAgent.create(
        RUNTIME['seed'] + rank,
        example_batch['observations'],
        example_batch['actions'],
        config,
        device=device,
        ex_proprioceptions=example_batch.get('proprioceptions'),
    )
    return config, per_device_batch_size, train_dataset, val_dataset, agent


def sequential_rank_barrier(rank: int, world_size: int, stage: str):
    if not is_distributed() or world_size <= 1:
        return
    device_ids = [torch.cuda.current_device()] if dist.get_backend() == 'nccl' else None
    dist.barrier(device_ids=device_ids)
    if rank == 0:
        print(f'[distributed] entering {stage}', flush=True)


def train_worker(rank: int, world_size: int, runtime):
    global RUNTIME
    RUNTIME = runtime
    if world_size > 1:
        local_rank = int(runtime.get('local_rank', rank))
        if runtime.get('sequential_cuda_init', True):
            time.sleep(local_rank * runtime.get('rank_init_stagger_seconds', 2.0))
        init_distributed(rank, world_size, runtime, local_rank=local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')

    exp_name = runtime['exp_name']

    if is_main_process():
        setup_wandb(
            entity=runtime['wandb_entity'],
            project='fql-orl-torch',
            group=runtime['run_group'],
            name=exp_name,
            wandb_output_dir=runtime['wandb_dir'],
            mode=runtime['wandb_mode'],
            config=runtime['flag_dict'],
        )

    save_dir = runtime['save_dir']
    runtime['save_dir'] = save_dir
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
            json.dump(runtime['flag_dict'], f)

    random.seed(runtime['seed'] + rank)
    np.random.seed(runtime['seed'] + rank)
    torch.manual_seed(runtime['seed'] + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(runtime['seed'] + rank)

    sequential_rank_barrier(rank, world_size, 'before_build_training_state')
    config, per_device_batch_size, train_dataset, val_dataset, agent = build_training_state(device)
    sequential_rank_barrier(rank, world_size, 'after_build_training_state')

    if runtime['pretrain_path'] is not None:
        sequential_rank_barrier(rank, world_size, 'before_pretrain_load')
        agent = load_pretrained_agent_torch(agent, runtime['pretrain_path'])
        sequential_rank_barrier(rank, world_size, 'after_pretrain_load')

    start_step = int(runtime['restore_epoch']) + 1 if runtime['restore_path'] is not None else 1
    if _use_biflow_chained_training(runtime):
        _set_agent_train_mode(agent, config, 'biflow_align')

    if is_distributed():
        torch.cuda.empty_cache()
        sequential_rank_barrier(rank, world_size, 'before_ddp_wrap')
        ddp_local_rank = int(runtime.get('local_rank', rank))
        agent.model = DDP(
            agent.model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
            find_unused_parameters=runtime['ddp_find_unused_parameters'],
            static_graph=runtime['ddp_static_graph'],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        sequential_rank_barrier(rank, world_size, 'after_ddp_wrap')

    if _use_biflow_chained_training(runtime):
        _set_agent_train_mode(agent, config, 'biflow_align')

    if runtime['restore_path'] is not None:
        agent = restore_agent_torch(agent, runtime['restore_path'], runtime['restore_epoch'])

    if start_step > runtime['offline_steps']:
        raise ValueError(
            f"Restore epoch {runtime['restore_epoch']} is already >= offline_steps "
            f"{runtime['offline_steps']}; no training steps would run."
        )

    if is_main_process():
        total_params = sum(p.numel() for p in agent.raw_model.parameters())
        trainable_params = sum(p.numel() for p in agent.raw_model.parameters() if p.requires_grad)
        optimizer_params = sum(
            p.numel()
            for optimizer in agent.optimizers.values()
            for group in optimizer.param_groups
            for p in group['params']
        )
        print(
            f'[DDP setup] world_size={world_size}, global_batch_size={int(config.batch_size)}, '
            f'per_device_batch_size={per_device_batch_size}'
        )
        print(
            f'Model Size: {total_params / 1e6:.2f}M total parameters, '
            f'{trainable_params / 1e6:.2f}M requires_grad parameters, '
            f'{optimizer_params / 1e6:.2f}M optimizer parameters.'
        )
        print("obs_horizon:",runtime['obs_horizon'])
        print("action_horizon:",runtime['action_horizon'])
        print(
            'encoders:',
            f"actor={getattr(agent.raw_model, 'actor_encoder_name', None)}, "
            f"critic={getattr(agent.raw_model, 'critic_encoder_name', None)}"
        )
        if runtime['restore_path'] is not None:
            print(f"Resuming training from step {start_step}; checkpoints will be saved to {save_dir}.")
        if _use_biflow_chained_training(runtime):
            print(
                'BiFlow alignment: '
                f"align_steps={runtime['biflow_align_steps']}, "
                f"total_steps={runtime['offline_steps']}."
            )

    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv')) if is_main_process() else None
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv')) if is_main_process() else None
    tensorboard_writer = _make_tensorboard_writer(runtime, save_dir) if is_main_process() else None
    envs = None
    first_time = time.time()
    last_time = time.time()
    batch_prefetch = bool(runtime.get('batch_prefetch', False))
    batch_executor = None
    next_train_batch = None

    def sample_train_batch():
        return train_dataset.sample(per_device_batch_size)

    if batch_prefetch:
        batch_executor = ThreadPoolExecutor(max_workers=1)
        next_train_batch = batch_executor.submit(sample_train_batch)
        if is_main_process():
            print('Batch prefetch enabled: sampling the next training batch on a background thread.')

    pbar_iter = range(start_step, runtime['offline_steps'] + 1)
    pbar = tqdm.tqdm(pbar_iter, smoothing=0.1, dynamic_ncols=True, disable=not is_main_process())
    for i in pbar:
        if (
            _use_biflow_chained_training(runtime)
            and agent.config.get('train_mode') != 'biflow_align'
        ):
            _set_agent_train_mode(agent, config, 'biflow_align')
            if is_main_process():
                print(f'Switching train_mode to `biflow_align` at step {i}.')

        do_log = runtime['log_interval'] != 0 and i % runtime['log_interval'] == 0
        is_biflow_align_boundary = (
            _use_biflow_chained_training(runtime)
            and i == int(runtime['biflow_align_steps'])
        )
        do_eval = (
            runtime['eval_interval'] != 0
            and ((i % runtime['eval_interval'] == 0) or is_biflow_align_boundary)
        )
        do_save = (
            (runtime['save_interval'] != 0 and i % runtime['save_interval'] == 0)
            or i in runtime['save_steps']
        )
        do_stage_boundary_save = (
            runtime['save_stage_boundaries']
            and _use_biflow_chained_training(runtime)
            and i in (int(runtime['biflow_align_steps']), int(runtime['offline_steps']))
        )
        defer_prefetch_for_eval = batch_executor is not None and do_eval

        sample_wait_start = time.time()
        if batch_executor is None:
            batch = sample_train_batch()
        else:
            batch = next_train_batch.result()
            if defer_prefetch_for_eval or i >= int(runtime['offline_steps']):
                next_train_batch = None
            else:
                next_train_batch = batch_executor.submit(sample_train_batch)
        sample_wait_time = time.time() - sample_wait_start
        update_start = time.time()
        agent, update_info = agent.update(
            batch,
            step=i,
        )
        update_info['time/batch_sample_wait_sec'] = sample_wait_time
        update_info['time/update_sec'] = time.time() - update_start

        if is_main_process():
            def progress_metric(key):
                value = float(update_info.get(key, 0.0))
                return f'{value:.4f}' if np.isfinite(value) else 'n/a'

            pbar.set_description(
                f"step: {i}, mode: "
                f"{agent.config.get('train_mode', config.get('train_mode', 'unknown'))}, "
                f"actor_entropy: {progress_metric('actor/actor_entropy')}, "
                f"actor_loss: {progress_metric('actor/actor_loss')}, "
                f"bc_loss: {progress_metric('actor/bc_loss')}, "
                f"mse: {progress_metric('actor/mse')}, "
                f"mse_loss: {progress_metric('actor/mse_loss')}, "
                f"total_loss: {progress_metric('actor/total_loss')}, "
                f"critic_loss: {progress_metric('critic/critic_loss')}, "
                f"q_mean: {progress_metric('critic/q_mean')}, "
                f"next_q_mean: {progress_metric('critic/next_q_mean')}, "
                f"target_q_mean: {progress_metric('critic/target_q_mean')}, "
                f"reward_mean: {progress_metric('critic/reward_mean')}, "
                f"p_norm: {progress_metric('actor/p_norm')}, "
                f"p_logprob_mean: {progress_metric('actor/p_logprob_mean')}, "
                f"p_logdet_mean: {progress_metric('actor/p_logdet_mean')}, "
                f"b_logdet_mean: {progress_metric('actor/b_logdet_mean')}"
            )

            if do_log:
                train_metrics = {}
                for key, value in update_info.items():
                    scalar = value.item() if torch.is_tensor(value) and value.numel() == 1 else value
                    if isinstance(scalar, (int, float, np.number)) and not np.isfinite(scalar):
                        continue
                    train_metrics[f'training/{key}'] = scalar
                if val_dataset is not None:
                    val_batch = val_dataset.sample(per_device_batch_size)
                    was_training = agent.model.training
                    agent.model.eval()
                    with torch.no_grad():
                        _, _, val_info = agent.total_loss(
                            val_batch,
                            full_update=True,
                        )
                    if was_training:
                        agent.model.train()
                    for key, value in val_info.items():
                        scalar = (
                            value.item()
                            if torch.is_tensor(value) and value.numel() == 1
                            else value
                        )
                        if isinstance(scalar, (int, float, np.number)) and not np.isfinite(scalar):
                            continue
                        train_metrics[f'validation/{key}'] = scalar
                train_metrics['time/epoch_time'] = (time.time() - last_time) / runtime['log_interval']
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                wandb.log(train_metrics, step=i)
                _log_tensorboard_metrics(tensorboard_writer, train_metrics, step=i)
                train_logger.log(train_metrics, step=i)

        if is_main_process() and (do_save or do_stage_boundary_save):
            save_agent_torch(agent, save_dir, i)

        do_eval_global = do_eval

        if do_eval_global:
            if is_distributed():
                dist.barrier()
            if is_main_process():
                eval_metrics = {}
                eval_model = copy.deepcopy(agent.raw_model).to(device)
                eval_model.eval()
                language_eval_tokenizer = None
                language_eval_cache = {}
                if _agent_needs_language_tokens(config):
                    from utils.language import build_text_tokenizer
                    from utils.robomimic.dataset import get_task_name_from_path

                    tokenizer_type, tokenizer_path, tokenizer_max_length = _language_tokenizer_spec(config)
                    language_eval_tokenizer = build_text_tokenizer(
                        tokenizer_type,
                        model_path=tokenizer_path,
                        max_length=tokenizer_max_length,
                    )

                    def language_tokens_for_task(task_path):
                        cache_key = task_path or runtime.get('dataset_dir') or runtime.get('env_name')
                        if cache_key not in language_eval_cache:
                            if task_path:
                                task_text = get_task_name_from_path(task_path)
                            elif runtime.get('dataset_dir') and '*' not in str(runtime.get('dataset_dir')):
                                task_text = get_task_name_from_path(runtime.get('dataset_dir'))
                            else:
                                task_text = str(runtime.get('env_name'))
                            language_eval_cache[cache_key] = language_eval_tokenizer.tokenize(
                                task_text,
                                device=device,
                            )
                        return language_eval_cache[cache_key]
                else:
                    def language_tokens_for_task(task_path):
                        del task_path
                        return None, None

                def make_eval_actor(sample_backend='simflow', task_path=None):
                    def eval_actor(obs):
                        observations, proprioceptions = agent._prepare_policy_inputs(obs)
                        if (
                            is_libero_env_name(runtime['env_name'])
                            and runtime['p_aug'] is not None
                            and float(runtime['p_aug']) > 0.0
                        ):
                            from utils.robomimic.dataset import apply_libero_visual_eval_preprocessing

                            observations = apply_libero_visual_eval_preprocessing(
                                observations,
                                mode=runtime['libero_visual_augmentation_mode'],
                            )
                        language_input_ids, language_attention_mask = language_tokens_for_task(task_path)
                        use_biflow_backend = sample_backend == 'biflow'
                        return agent.sample_actions(
                            observations,
                            model=eval_model,
                            cfg=(
                                float(config.get('biflow_eval_guidance', 0.0))
                                if use_biflow_backend
                                else float(config.get('cfg', 1.5))
                            ),
                            cfg_final=(
                                float(config.get('biflow_eval_guidance_final', -1.0))
                                if use_biflow_backend
                                else None
                            ),
                            temperature=float(
                                config.get(
                                    'biflow_eval_temperature',
                                    config.get('eval_temperature', 1.0),
                                )
                                if use_biflow_backend
                                else config.get('eval_temperature', 1.0)
                            ),
                            apply_denoising=float(config.get('denoising_lr', 0.0)) > 0.0,
                            input_ids=language_input_ids,
                            attention_mask=language_attention_mask,
                            sample_backend=sample_backend,
                            proprioceptions=proprioceptions,
                        )

                    return eval_actor

                eval_actor = make_eval_actor(sample_backend='simflow')

                libero_task_paths = _get_libero_eval_task_paths()
                if libero_task_paths is None:
                    if envs is None:
                        envs = maybe_make_eval_vec_env()
                    eval_info = torch_evaluate_parallel(
                        eval_actor,
                        seed=runtime['seed'],
                        envs=envs,
                        num_eval_episodes=runtime['eval_episodes'],
                        clip_actions=True,
                        action_exec_horizon=runtime['eval_action_exec_horizon'],
                    )
                    _add_eval_metrics(eval_metrics, 'parallel_evaluation', eval_info)
                    if bool(config.get('use_biflow', False)):
                        biflow_eval_info = torch_evaluate_parallel(
                            make_eval_actor(sample_backend='biflow'),
                            seed=runtime['seed'],
                            envs=envs,
                            num_eval_episodes=runtime['eval_episodes'],
                            clip_actions=True,
                            action_exec_horizon=runtime['eval_action_exec_horizon'],
                        )
                        _add_eval_metrics(eval_metrics, 'parallel_evaluation/biflow', biflow_eval_info)
                else:
                    task_successes = []
                    biflow_task_successes = []
                    for task_idx, task_path in enumerate(libero_task_paths):
                        task_envs = maybe_make_eval_vec_env(task_path)
                        try:
                            eval_info = torch_evaluate_parallel(
                                make_eval_actor(sample_backend='simflow', task_path=task_path),
                                seed=runtime['seed'] + task_idx * max(runtime['eval_episodes'], 1),
                                envs=task_envs,
                                num_eval_episodes=runtime['eval_episodes'],
                                clip_actions=True,
                                action_exec_horizon=runtime['eval_action_exec_horizon'],
                            )
                            if bool(config.get('use_biflow', False)):
                                biflow_eval_info = torch_evaluate_parallel(
                                    make_eval_actor(sample_backend='biflow', task_path=task_path),
                                    seed=runtime['seed'] + task_idx * max(runtime['eval_episodes'], 1),
                                    envs=task_envs,
                                    num_eval_episodes=runtime['eval_episodes'],
                                    clip_actions=True,
                                    action_exec_horizon=runtime['eval_action_exec_horizon'],
                                )
                        finally:
                            task_envs.close()
                        task_name = _get_eval_task_name(task_path)
                        task_success = float(eval_info.get('success', 0.0))
                        eval_metrics[f'parallel_evaluation/{task_name}/success'] = task_success
                        task_successes.append(task_success)
                        if bool(config.get('use_biflow', False)):
                            biflow_task_success = float(biflow_eval_info.get('success', 0.0))
                            eval_metrics[f'parallel_evaluation/biflow/{task_name}/success'] = biflow_task_success
                            biflow_task_successes.append(biflow_task_success)
                    eval_metrics['parallel_evaluation/success'] = float(np.mean(task_successes))
                    if bool(config.get('use_biflow', False)):
                        eval_metrics['parallel_evaluation/biflow/success'] = float(np.mean(biflow_task_successes))

                eval_message = (
                    f"[{i} / {runtime['offline_steps']}] Evaluation Success Rate: "
                    f"simflow={eval_metrics.get('parallel_evaluation/success', 0.0):.4f}"
                )
                if bool(config.get('use_biflow', False)):
                    eval_message += (
                        f", biflow={eval_metrics.get('parallel_evaluation/biflow/success', 0.0):.4f}"
                    )
                print(eval_message)

                del eval_model
                wandb.log(eval_metrics, step=i)
                _log_tensorboard_metrics(tensorboard_writer, eval_metrics, step=i)
                eval_logger.log(eval_metrics, step=i)

            if is_distributed():
                dist.barrier()

        if batch_executor is not None and next_train_batch is None and i < int(runtime['offline_steps']):
            next_train_batch = batch_executor.submit(sample_train_batch)

    if batch_executor is not None:
        batch_executor.shutdown(wait=False, cancel_futures=True)

    if is_main_process():
        train_logger.close()
        eval_logger.close()
        if tensorboard_writer is not None:
            tensorboard_writer.close()
    cleanup_distributed()


def main(_):
    if FLAGS.pretrain_path is not None and FLAGS.restore_path is not None:
        raise ValueError('`pretrain_path` and `restore_path` are mutually exclusive. Please set only one.')
    if FLAGS.restore_path is not None and FLAGS.restore_epoch is None:
        raise ValueError('`restore_epoch` must be set when `restore_path` is provided.')
    if FLAGS.biflow_align_steps < 0:
        raise ValueError('`biflow_align_steps` must be non-negative.')
    use_biflow_chained_training = FLAGS.biflow_align_steps > 0
    if use_biflow_chained_training and FLAGS.restore_path is None and FLAGS.pretrain_path is None:
        raise ValueError('BiFlow training requires `pretrain_path` unless resuming from `restore_path`.')

    augmentation_probability = resolve_visual_augmentation_probability(
        p_aug=FLAGS.p_aug,
    )

    exp_name = get_exp_name(FLAGS.seed)
    exp_name = f"{FLAGS.wandb_name_tag}{FLAGS.env_name}__{FLAGS.run_group}__{exp_name}"
    save_dir = resolve_training_save_dir(
        wandb_dir=FLAGS.wandb_dir,
        run_group=FLAGS.run_group,
        exp_name=exp_name,
        restore_path=FLAGS.restore_path,
        restore_output_dir=FLAGS.restore_output_dir,
    )

    agent_config = FLAGS.agent.copy_and_resolve_references()
    if FLAGS.agent_flags_json is not None:
        with open(FLAGS.agent_flags_json, 'r', encoding='utf-8') as handle:
            released_flags = json.load(handle)
        released_agent = released_flags.get('agent')
        if not isinstance(released_agent, dict):
            raise ValueError('agent_flags_json must contain an `agent` object.')
        for key, value in released_agent.items():
            if value is not None:
                agent_config[key] = value

    master_port = None
    if FLAGS.ddp:
        master_port = os.environ.get('MASTER_PORT') or find_free_port()

    runtime = {
        'wandb_dir': FLAGS.wandb_dir,
        'wandb_name_tag': FLAGS.wandb_name_tag,
        'wandb_entity': FLAGS.wandb_entity,
        'wandb_mode': FLAGS.wandb_mode,
        'tensorboard': FLAGS.tensorboard,
        'tensorboard_dir': FLAGS.tensorboard_dir,
        'tensorboard_flush_secs': FLAGS.tensorboard_flush_secs,
        'run_group': FLAGS.run_group,
        'exp_name': exp_name,
        'save_dir': save_dir,
        'dataset_dir': FLAGS.dataset_dir,
        'libero_buffer_path': FLAGS.libero_buffer_path,
        'robomimic_buffer_path': FLAGS.robomimic_buffer_path,
        'seed': FLAGS.seed,
        'env_name': FLAGS.env_name,
        'pretrain_path': FLAGS.pretrain_path,
        'restore_path': FLAGS.restore_path,
        'restore_epoch': FLAGS.restore_epoch,
        'restore_output_dir': FLAGS.restore_output_dir,
        'offline_steps': FLAGS.offline_steps,
        'biflow_align_steps': FLAGS.biflow_align_steps,
        'buffer_size': FLAGS.buffer_size,
        'log_interval': FLAGS.log_interval,
        'eval_interval': FLAGS.eval_interval,
        'save_interval': FLAGS.save_interval,
        'save_steps': parse_save_steps(FLAGS.save_steps),
        'save_stage_boundaries': FLAGS.save_stage_boundaries,
        'eval_episodes': FLAGS.eval_episodes,
        'eval_num_envs': FLAGS.eval_num_envs,
        'eval_action_exec_horizon': FLAGS.eval_action_exec_horizon,
        'obs_horizon': FLAGS.obs_horizon,
        'action_horizon': FLAGS.action_horizon,
        'p_aug': augmentation_probability,
        'libero_visual_augmentation_mode': FLAGS.libero_visual_augmentation_mode,
        'global_batch_size': FLAGS.global_batch_size,
        'batch_prefetch': FLAGS.batch_prefetch,
        'ddp_backend': FLAGS.ddp_backend,
        'ddp_timeout_seconds': FLAGS.ddp_timeout_seconds,
        'ddp_static_graph': FLAGS.ddp_static_graph,
        'ddp_find_unused_parameters': FLAGS.ddp_find_unused_parameters,
        'sequential_cuda_init': True,
        'rank_init_stagger_seconds': 2.0,
        'master_addr': os.environ.get('MASTER_ADDR', '127.0.0.1'),
        'master_port': master_port,
        'agent_config': agent_config,
        'flag_dict': get_flag_dict(),
    }
    torchrun_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torchrun_rank = int(os.environ.get('RANK', '0'))
    torchrun_local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    using_torchrun = FLAGS.ddp and torchrun_world_size > 1
    if using_torchrun:
        runtime['local_rank'] = torchrun_local_rank
        runtime['master_addr'] = os.environ.get('MASTER_ADDR', runtime['master_addr'])
        runtime['master_port'] = os.environ.get('MASTER_PORT', runtime['master_port'])
        runtime['flag_dict']['ddp_world_size'] = torchrun_world_size
        runtime['flag_dict']['ddp_rank'] = torchrun_rank
        runtime['flag_dict']['ddp_local_rank'] = torchrun_local_rank
    if use_biflow_chained_training:
        runtime['offline_steps'] = FLAGS.biflow_align_steps
        runtime['agent_config'].use_biflow = True
        runtime['agent_config'].train_mode = 'biflow_align'
        runtime['flag_dict']['effective_offline_steps'] = runtime['offline_steps']
        runtime['flag_dict']['biflow_alignment'] = True
        if FLAGS.ddp:
            runtime['ddp_static_graph'] = False
            runtime['ddp_find_unused_parameters'] = True
            runtime['flag_dict']['ddp_static_graph'] = False
            runtime['flag_dict']['ddp_find_unused_parameters'] = True

    validate_supported_env(FLAGS.env_name)

    if (
        FLAGS.ddp
        and str(runtime['agent_config'].get('train_mode', '')).lower() == 'iql'
        and int(runtime['agent_config'].get('iql_critic_warmup_steps', 0)) > 0
    ):
        # Actor parameters are intentionally unused during critic warmup and become
        # active afterwards, so the DDP graph cannot be static.
        runtime['ddp_static_graph'] = False
        runtime['ddp_find_unused_parameters'] = True
        runtime['flag_dict']['ddp_static_graph'] = False
        runtime['flag_dict']['ddp_find_unused_parameters'] = True

    if is_libero_env_name(FLAGS.env_name):
        ensure_libero_config()
    runtime['agent_config'].action_len = runtime['action_horizon']
    runtime['agent_config'].obs_horizon = runtime['obs_horizon']
    _sync_runtime_flag_dict(runtime)
    global RUNTIME
    RUNTIME = runtime
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    visible_count = len([d for d in visible_devices.split(',') if d.strip()]) if visible_devices else torch.cuda.device_count()

    should_prepare_libero_buffer = (
        FLAGS.ddp
        and (torchrun_world_size > 1 if using_torchrun else visible_count > 1)
        and is_libero_env_name(FLAGS.env_name)
    )
    if should_prepare_libero_buffer:
        from utils.robomimic.dataset import _ensure_libero_buffer

        print(
            f'[libero-buffer] rank={torchrun_rank if using_torchrun else 0} '
            'preparing/restoring the shared LIBERO buffer...',
            flush=True,
        )
        tokenizer_type, tokenizer_path, tokenizer_max_length = _language_tokenizer_spec(
            runtime['agent_config']
        )
        _ensure_libero_buffer(
            FLAGS.dataset_dir,
            buffer_path=FLAGS.libero_buffer_path,
            flip_rgb=True,
            discount=float(runtime['agent_config'].discount),
            language_model_path=tokenizer_path,
            language_max_length=tokenizer_max_length,
            language_tokenizer_type=tokenizer_type,
            include_proprioception=bool(
                runtime['agent_config'].get('use_proprioception', False)
            ),
        )
        print('[libero-buffer] LIBERO buffer is ready. Launching DDP workers...', flush=True)

    should_prepare_robomimic_buffer = (
        FLAGS.ddp
        and (torchrun_world_size > 1 if using_torchrun else visible_count > 1)
        and is_robomimic_env_name(FLAGS.env_name)
    )
    if should_prepare_robomimic_buffer:
        from utils.robomimic.dataset import _ensure_robomimic_buffer

        print(
            f'[robomimic-buffer] rank={torchrun_rank if using_torchrun else 0} '
            'preparing/restoring the shared buffer...',
            flush=True,
        )
        _ensure_robomimic_buffer(
            FLAGS.dataset_dir,
            buffer_path=FLAGS.robomimic_buffer_path,
            discount=float(runtime['agent_config'].discount),
            include_proprioception=bool(
                runtime['agent_config'].get('robomimic_use_proprioception', False)
            ),
        )
        print('[robomimic-buffer] Buffer is ready. Launching DDP workers...', flush=True)

    if using_torchrun:
        train_worker(rank=torchrun_rank, world_size=torchrun_world_size, runtime=runtime)
    elif FLAGS.ddp and visible_count > 1:
        mp.spawn(train_worker, args=(visible_count, runtime), nprocs=visible_count, join=True)
    else:
        train_worker(rank=0, world_size=1, runtime=runtime)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    app.run(main)
