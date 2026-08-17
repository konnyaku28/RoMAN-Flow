"""Environment and dataset construction for the RoMAN-Flow release."""

import glob
import os
import sys
from pathlib import Path


SUPPORTED_LIBERO_SUITES = frozenset(
    {"libero_10", "libero_spatial", "libero_object", "libero_goal"}
)
SUPPORTED_ROBOMIMIC_TASKS = frozenset(
    {"robomimic_lift_mh", "robomimic_can_mh", "robomimic_square_mh"}
)


def _is_robomimic_env_name(env_name):
    return str(env_name).strip().lower() in SUPPORTED_ROBOMIMIC_TASKS


def _is_libero_env_name(env_name):
    return str(env_name).strip().lower() in SUPPORTED_LIBERO_SUITES


def _validate_supported_env(env_name):
    if not (_is_libero_env_name(env_name) or _is_robomimic_env_name(env_name)):
        raise ValueError(
            "RoMAN-Flow supports libero_10, libero_spatial, libero_object, "
            "libero_goal, robomimic_lift_mh, robomimic_can_mh, and "
            f"robomimic_square_mh; got {env_name!r}. Transport-MH is unsupported."
        )


def validate_supported_env(env_name):
    """Fail fast when a release entry point receives an unsupported task name."""
    _validate_supported_env(env_name)


def validate_libero_dataset_suite(env_name, dataset_path):
    """Verify that every HDF5 belongs to the requested official LIBERO suite."""
    suite = str(env_name).strip().lower()
    if suite not in SUPPORTED_LIBERO_SUITES:
        raise ValueError(
            f"Unsupported LIBERO suite {env_name!r}; expected one of "
            f"{sorted(SUPPORTED_LIBERO_SUITES)}."
        )
    expanded = os.path.expanduser(str(dataset_path))
    paths = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else [expanded]
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise FileNotFoundError(f"LIBERO dataset path did not resolve to files: {dataset_path}")

    import h5py

    task_names = []
    for path in paths:
        with h5py.File(path, "r") as file:
            if "data" not in file:
                raise ValueError(f"LIBERO dataset has no data group: {path}")
            attributes = file["data"].attrs
            bddl_name = attributes.get("bddl_file_name")
            tag = attributes.get("tag")
        if not isinstance(bddl_name, str):
            raise ValueError(
                f"LIBERO dataset {path} is missing data.attrs['bddl_file_name']."
            )
        path_parts = bddl_name.replace("\\", "/").split("/")
        if suite not in path_parts:
            raise ValueError(
                f"{suite} requires official {suite} demonstrations, but {path} "
                f"declares bddl_file_name={bddl_name!r}."
            )
        if tag != "libero-v1":
            raise ValueError(
                f"LIBERO dataset {path} has tag={tag!r}; expected official tag 'libero-v1'."
            )
        task_names.append(os.path.basename(bddl_name))
    if len(set(task_names)) != len(task_names):
        raise ValueError(f"LIBERO dataset selection contains duplicate tasks: {task_names}")


def is_libero_env_name(env_name):
    return _is_libero_env_name(env_name)


def is_robomimic_env_name(env_name):
    return _is_robomimic_env_name(env_name)


def ensure_libero_config():
    """Create LIBERO's local configuration when the benchmark is importable."""
    config_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", "/path/to/libero_config")
    ).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    roots = [
        Path(path).expanduser()
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path
    ]
    roots.extend(Path(path).expanduser() for path in sys.path if path)
    roots.extend((Path("/path/to/LIBERO"), Path.cwd()))
    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for benchmark_root in (root, root / "libero", root / "libero" / "libero"):
            if not (
                (benchmark_root / "__init__.py").is_file()
                and (benchmark_root / "bddl_files").is_dir()
                and (benchmark_root / "init_files").is_dir()
            ):
                continue
            config_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "benchmark_root": benchmark_root,
                "bddl_files": benchmark_root / "bddl_files",
                "init_states": benchmark_root / "init_files",
                "datasets": benchmark_root.parent / "datasets",
                "assets": benchmark_root / "assets",
            }
            with config_file.open("w", encoding="utf-8") as handle:
                for key, value in config.items():
                    handle.write(f"{key}: {value}\n")
            print(
                f"[libero-config] Wrote non-interactive LIBERO config: {config_file}",
                flush=True,
            )
            return
    print(
        "[libero-config] Could not infer LIBERO benchmark root. "
        "LIBERO may prompt for dataset path during import.",
        flush=True,
    )


def _clip_dataset_actions(dataset, action_clip_eps):
    if not hasattr(dataset, "set_action_clip_eps"):
        raise TypeError(
            "RoMAN-Flow requires a torch_lazy dataset with set_action_clip_eps()."
        )
    dataset.set_action_clip_eps(action_clip_eps)
    return dataset


def make_env(
    env_id: str,
    dataset_dir=None,
    obs_horizon: int = 1,
    robomimic_use_proprioception: bool = False,
):
    """Build one release-supported evaluation environment."""
    _validate_supported_env(env_id)
    if not dataset_dir:
        raise ValueError(
            f"{env_id} evaluation requires dataset_dir to be one task HDF5 file."
        )

    def _init():
        from envs.robomimic import make_robomimic_env

        env = make_robomimic_env(
            env_id,
            dataset_dir,
            obs_horizon=obs_horizon,
            record=False,
            use_proprioception=robomimic_use_proprioception,
        )
        return env

    return _init


def make_env_and_datasets(
    env_name,
    dataset_dir=None,
    action_clip_eps=1e-5,
    discount=None,
    hubl_lambda_type="rank",
    hubl_alpha=0.1,
    libero_buffer_path=None,
    robomimic_buffer_path=None,
    robomimic_use_proprioception=False,
    language_model_path=None,
    language_max_length=77,
    language_tokenizer_type="clip",
):
    """Build a disk-backed LIBERO or RoboMimic training dataset."""
    _validate_supported_env(env_name)
    if not dataset_dir:
        raise ValueError(f"{env_name} requires a local HDF5 path or glob.")
    if _is_robomimic_env_name(env_name):
        from envs.robomimic import validate_robomimic_dataset_task
        from utils.robomimic.dataset import get_robomimic_dataset_torch

        validate_robomimic_dataset_task(env_name, dataset_dir)

        train_dataset, val_dataset = get_robomimic_dataset_torch(
            dataset_dir,
            val_ratio=0.0,
            discount=0.99 if discount is None else discount,
            hubl_lambda_type=hubl_lambda_type,
            hubl_alpha=hubl_alpha,
            buffer_path=robomimic_buffer_path,
            include_proprioception=robomimic_use_proprioception,
        )
    else:
        validate_libero_dataset_suite(env_name, dataset_dir)
        from utils.robomimic.dataset import get_libero_dataset_torch

        val_ratio = 0.0 if "*" in str(dataset_dir) else 0.05
        train_dataset, val_dataset = get_libero_dataset_torch(
            dataset_dir,
            val_ratio=val_ratio,
            flip_rgb=True,
            discount=0.99 if discount is None else discount,
            hubl_lambda_type=hubl_lambda_type,
            hubl_alpha=hubl_alpha,
            buffer_path=libero_buffer_path,
            language_model_path=language_model_path,
            language_max_length=language_max_length,
            language_tokenizer_type=language_tokenizer_type,
            include_proprioception=robomimic_use_proprioception,
        )

    if action_clip_eps is not None:
        train_dataset = _clip_dataset_actions(train_dataset, action_clip_eps)
        if val_dataset is not None:
            val_dataset = _clip_dataset_actions(val_dataset, action_clip_eps)
    return train_dataset, val_dataset
