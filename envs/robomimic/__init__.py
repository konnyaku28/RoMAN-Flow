import json
import os
import sys
import types
from .wrappers import LIBEROEnvWrapper, RoboMimicEnvWrapper


ROBOMIMIC_IMAGE_KEYS = (
    'agentview_image',
    'robot0_eye_in_hand_image',
)
ROBOMIMIC_PROPRIO_KEYS = (
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
)
ROBOMIMIC_DEFAULT_ROLLOUT_HORIZON = 500
ROBOMIMIC_ENV_NAMES = {
    'robomimic_lift_mh': 'Lift',
    'robomimic_can_mh': 'PickPlaceCan',
    'robomimic_square_mh': 'NutAssemblySquare',
}
LIBERO_SUITE_NAMES = {
    'libero_10',
    'libero_spatial',
    'libero_object',
    'libero_goal',
}


def validate_robomimic_dataset_task(dataset_name, dataset_path):
    """Verify that an official image dataset matches its requested task name."""
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name not in ROBOMIMIC_ENV_NAMES:
        raise ValueError(
            f'Unsupported RoboMimic task {dataset_name!r}; expected one of '
            f'{sorted(ROBOMIMIC_ENV_NAMES)}.'
        )

    import h5py

    resolved_path = os.path.abspath(os.path.expanduser(dataset_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f'RoboMimic dataset was not found: {resolved_path}')
    with h5py.File(resolved_path, 'r') as file:
        serialized_metadata = file.get('data', {}).attrs.get('env_args') if 'data' in file else None
    if serialized_metadata is None:
        raise ValueError(
            f'RoboMimic dataset {resolved_path} is missing data.attrs["env_args"]. '
            'Use the official v1.4.1 MH state file and the provided converter.'
        )
    try:
        metadata = json.loads(serialized_metadata)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f'RoboMimic dataset {resolved_path} has invalid env_args metadata.'
        ) from exc

    expected = ROBOMIMIC_ENV_NAMES[dataset_name]
    actual = metadata.get('env_name')
    if actual != expected:
        raise ValueError(
            f'{dataset_name} requires RoboMimic env_name={expected!r}, but '
            f'{resolved_path} contains env_name={actual!r}.'
        )


def _import_robomimic_without_language_download():
    """Import RoboMimic utilities without its unused eager CLIP initialization."""
    if 'robomimic.algo' not in sys.modules:
        algo_stub = types.ModuleType('robomimic.algo')

        def _unsupported_algo_factory(*args, **kwargs):
            del args, kwargs
            raise RuntimeError('RoboMimic policy loading is not used by RoMAN-Flow.')

        class _UnusedRolloutPolicy:
            pass

        algo_stub.algo_factory = _unsupported_algo_factory
        algo_stub.RolloutPolicy = _UnusedRolloutPolicy
        sys.modules['robomimic.algo'] = algo_stub
    if 'robomimic.utils.lang_utils' not in sys.modules:
        lang_stub = types.ModuleType('robomimic.utils.lang_utils')
        lang_stub.LANG_EMB_OBS_KEY = 'lang_emb'

        def _get_lang_emb(lang):
            if lang is None:
                return None
            raise RuntimeError('Language embeddings are disabled for RoboMimic RGB tasks.')

        lang_stub.get_lang_emb = _get_lang_emb
        lang_stub.get_lang_emb_shape = lambda: []
        sys.modules['robomimic.utils.lang_utils'] = lang_stub


def _patch_robosuite_mujoco3_fullm():
    """Reject simulator stacks outside the verified DLC configuration."""
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError('MuJoCo 2.3.7 is required for RoboMimic rollout.') from exc
    if str(getattr(mujoco, '__version__', '')) != '2.3.7':
        raise RuntimeError(
            'RoMAN-Flow rollout requires mujoco==2.3.7 with robosuite==1.4.0. '
            f'Found mujoco=={getattr(mujoco, "__version__", "unknown")}. '
            'Reinstall the supported MuJoCo and Robosuite dependencies.'
        )

def _make_standard_robomimic_env(
    dataset_name,
    dataset_path,
    obs_horizon,
    record,
    use_proprioception=False,
):
    os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
    _patch_robosuite_mujoco3_fullm()
    _import_robomimic_without_language_download()
    from robomimic.utils.env_utils import get_env_class, get_env_type
    from robomimic.utils.file_utils import (
        get_env_metadata_from_dataset,
        get_shape_metadata_from_dataset,
    )
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    validate_robomimic_dataset_task(dataset_name, dataset_path)

    low_dim_keys = list(ROBOMIMIC_PROPRIO_KEYS) if use_proprioception else []
    initialize_obs_utils_with_obs_specs(
        {'obs': {'rgb': list(ROBOMIMIC_IMAGE_KEYS), 'low_dim': low_dim_keys}}
    )
    env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
    shape_meta = get_shape_metadata_from_dataset(
        dataset_config={'path': dataset_path},
        action_keys=['actions'],
        all_obs_keys=list(ROBOMIMIC_IMAGE_KEYS) + low_dim_keys,
        verbose=False,
    )
    env_type = get_env_type(env_meta=env_meta)
    env_class = get_env_class(env_type=env_type)
    env_kwargs = dict(env_meta['env_kwargs'])
    max_episode_length = int(
        os.environ.get(
            'ROBOMIMIC_ROLLOUT_HORIZON',
            env_meta.get(
                'horizon',
                env_kwargs.get('horizon', ROBOMIMIC_DEFAULT_ROLLOUT_HORIZON),
            ),
        )
    )
    if max_episode_length <= 0:
        raise ValueError(
            f'RoboMimic rollout horizon must be positive, got {max_episode_length}.'
        )
    if os.environ.get('MUJOCO_GL') in {'egl', 'glfw'}:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            env_kwargs['render_gpu_device_id'] = int(visible_devices.split(',')[0])
    env = env_class(
        env_name=env_meta['env_name'],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta['use_images'],
        use_depth_obs=shape_meta['use_depths'],
        **env_kwargs,
    )
    return RoboMimicEnvWrapper(
        env,
        obs_keys=ROBOMIMIC_IMAGE_KEYS,
        obs_horizon=obs_horizon,
        max_episode_length=max_episode_length,
        record=record,
        render_size=(
            int(os.environ.get('ROBOMIMIC_EVAL_VIDEO_SIZE', '224')),
            int(os.environ.get('ROBOMIMIC_EVAL_VIDEO_SIZE', '224')),
        ),
        video_camera=os.environ.get(
            'ROBOMIMIC_EVAL_VIDEO_CAMERA',
            'agentview',
        ),
        use_proprioception=use_proprioception,
        proprio_keys=ROBOMIMIC_PROPRIO_KEYS,
    )

def make_robomimic_env(
    dataset_name,
    dataset_path,
    obs_horizon=1,
    record=False,
    use_proprioception=False,
):
    dataset_name = dataset_name.lower()
    if dataset_name in ROBOMIMIC_ENV_NAMES:
        return _make_standard_robomimic_env(
            dataset_name,
            dataset_path,
            obs_horizon,
            record,
            use_proprioception=use_proprioception,
        )
    if dataset_name in LIBERO_SUITE_NAMES:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import benchmark, get_libero_path

        benchmark_dict = benchmark.get_benchmark_dict()
        suite_name = dataset_name
        if suite_name not in benchmark_dict or suite_name not in LIBERO_SUITE_NAMES:
            raise ValueError(
                'LIBERO benchmark is missing the requested release suite '
                f'{suite_name!r}.'
            )

        suite = benchmark_dict[suite_name]()
        dataset_basename = os.path.basename(dataset_path)
        matching_ids = [
            task_id
            for task_id in range(suite.n_tasks)
            if os.path.basename(suite.get_task_demonstration(task_id))
            == dataset_basename
        ]
        if len(matching_ids) != 1:
            raise ValueError(
                f'Could not uniquely match {dataset_basename!r} to {suite_name}; '
                f'matches={matching_ids}.'
            )
        task_id = matching_ids[0]
        task = suite.get_task(task_id)
        bddl_file_name = os.path.join(
            get_libero_path('bddl_files'),
            task.problem_folder,
            task.bddl_file,
        )
        env_kwargs = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        # Set render device if CUDA_VISIBLE_DEVICES is set
        if os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_kwargs["render_gpu_device_id"] = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            )

        # Create environment
        env = OffScreenRenderEnv(**env_kwargs)
        official_initial_states = None
        num_steps_wait = 0
        if os.environ.get('LIBERO_USE_OFFICIAL_INITIAL_STATES', '').lower() in {
            '1', 'true', 'yes'
        }:
            expected_bddl = os.path.basename(bddl_file_name)
            official_initial_states = suite.get_task_init_states(task_id)
            if len(official_initial_states) == 0:
                raise ValueError(f'LIBERO task {expected_bddl!r} has no official initial states.')
            num_steps_wait = int(os.environ.get('LIBERO_NUM_STEPS_WAIT', '10'))
        video_size = int(os.environ.get('LIBERO_EVAL_VIDEO_SIZE', '128'))
        if video_size <= 0:
            raise ValueError(
                f'LIBERO_EVAL_VIDEO_SIZE must be positive, got {video_size}.'
            )
        env = LIBEROEnvWrapper(
            env,
            obs_horizon=obs_horizon,
            record=record,
            render_size=(video_size, video_size),
            flip_rgb=True,
            video_camera=os.environ.get('LIBERO_EVAL_VIDEO_CAMERA', 'agentview'),
            use_proprioception=use_proprioception,
            official_initial_states=official_initial_states,
            num_steps_wait=num_steps_wait,
        )
    else:
        raise NotImplementedError(f"Unsupported environment: {dataset_name}")
    return env
