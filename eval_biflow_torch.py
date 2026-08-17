import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import ml_collections
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.vinf_torch import (
    VINFTorchAgent,
    get_config,
    uses_proprioception,
)
from envs.env_utils import (
    ensure_libero_config,
    is_libero_env_name,
    make_env,
    make_env_and_datasets,
    validate_supported_env,
)
from utils.evaluation import (
    torch_evaluate_parallel,
    torch_evaluate_parallel_fixed_seeds,
)
from utils.robomimic.dataset import (
    apply_libero_visual_eval_preprocessing,
    canonicalize_libero_visual_augmentation_mode,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def comma_separated_ints(value):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected comma-separated integers, got {value!r}."
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"Video episode seeds must be unique, got {value!r}."
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flags_json", default=None)
    parser.add_argument("--env_name", default="libero_10")
    parser.add_argument(
        "--dataset_dir",
        default="/path/to/libero_data/libero_10/<task>_demo.hdf5",
    )
    parser.add_argument("--libero_buffer_path", default=None)
    parser.add_argument("--robomimic_buffer_path", default=None)
    parser.add_argument(
        "--smolvlm_model_path",
        default=None,
        help="Local SmolVLM directory. Overrides the path stored in training flags.",
    )
    parser.add_argument(
        "--language_model_path",
        default=None,
        help="Local CLIP/text-model directory. Overrides the path stored in training flags.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--actor_context_obs_horizon", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--denoising_lr", type=float, default=None)
    parser.add_argument("--biflow_eval_guidance", type=float, default=None)
    parser.add_argument("--biflow_eval_guidance_final", type=float, default=None)
    parser.add_argument("--use_biflow", choices=("auto", "true", "false"), default="auto")
    parser.add_argument(
        "--sample_backend",
        choices=("auto", "simflow", "biflow"),
        default="auto",
    )
    parser.add_argument("--apply_denoising", type=str_to_bool, default=True)
    parser.add_argument("--clip_actions", type=str_to_bool, default=True)
    parser.add_argument("--action_exec_horizon", type=int, default=None)
    parser.add_argument("--vec_env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument(
        "--libero_eval_visual_preprocess",
        choices=("auto", "none", "deterministic_center_crop"),
        default="auto",
        help=(
            "LIBERO rollout image preprocessing. auto enables the deterministic "
            "counterpart of the checkpoint's augmentation geometry when p_aug > 0."
        ),
    )
    parser.add_argument(
        "--fixed_episode_seeds",
        type=str_to_bool,
        default=None,
        help="Reset every episode with an explicit consecutive seed; defaults to true for LIBERO.",
    )
    parser.add_argument(
        "--libero_use_official_initial_states",
        type=str_to_bool,
        default=None,
        help="Use benchmark-provided LIBERO initial states; defaults to true for LIBERO.",
    )
    parser.add_argument(
        "--episode_seed_start",
        type=int,
        default=0,
        help="First environment seed used by --fixed_episode_seeds.",
    )
    parser.add_argument(
        "--episode_output",
        default=None,
        help="Optional JSON path for per-episode fixed-seed results.",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        help="Optional directory for fixed-seed rollout MP4 files.",
    )
    parser.add_argument(
        "--frame_dir",
        default=None,
        help=(
            "Optional directory for per-frame PNG sequences. Uses the same "
            "fixed seeds and frame skip as --video_dir."
        ),
    )
    parser.add_argument(
        "--video_episode_seeds",
        type=comma_separated_ints,
        default=None,
        help="Comma-separated fixed episode seeds to record; defaults to all evaluated seeds.",
    )
    parser.add_argument("--video_frame_skip", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args._provided_args = {
        item.split("=", 1)[0]
        for item in sys.argv[1:]
        if item.startswith("--")
    }
    return args


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _training_visual_augmentation_probability(flags):
    if flags is None:
        return None
    p_aug = flags.get("p_aug")
    if p_aug is None:
        return None
    probability = float(p_aug)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Checkpoint visual augmentation probability must be in [0, 1], "
            f"got {probability}."
        )
    return probability


def _training_visual_augmentation_mode(flags):
    if flags is None or flags.get("libero_visual_augmentation_mode") is None:
        return "libero_224"
    return canonicalize_libero_visual_augmentation_mode(
        flags["libero_visual_augmentation_mode"]
    )


def _resolve_libero_eval_visual_preprocess(requested, env_name, training_flags=None):
    requested = str(requested).strip().lower()
    choices = ("auto", "none", "deterministic_center_crop")
    if requested not in choices:
        raise ValueError(
            f"Unsupported LIBERO eval visual preprocessing {requested!r}; expected one of {choices}."
        )

    is_libero = is_libero_env_name(env_name)
    if requested == "none":
        return (
            "none",
            "explicit",
            _training_visual_augmentation_probability(training_flags),
            _training_visual_augmentation_mode(training_flags),
        )
    if not is_libero:
        if requested == "deterministic_center_crop":
            raise ValueError(
                "--libero_eval_visual_preprocess=deterministic_center_crop is only valid for LIBERO environments."
            )
        return "none", "auto:non_libero", None, None

    probability = _training_visual_augmentation_probability(training_flags)
    augmentation_mode = _training_visual_augmentation_mode(training_flags)
    if requested == "deterministic_center_crop":
        return "deterministic_center_crop", "explicit", probability, augmentation_mode
    if probability is not None and probability > 0.0:
        return "deterministic_center_crop", "auto:p_aug_positive", probability, augmentation_mode
    return "none", "auto:p_aug_disabled", probability, augmentation_mode


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def update_config(config, values):
    for key, value in values.items():
        config[key] = value


def build_config(args, checkpoint, checkpoint_state):
    config = get_config()
    args._training_flags = None

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        update_config(config, checkpoint["config"])

    if args.flags_json is not None:
        flags = load_json(args.flags_json)
        args._training_flags = flags
        if isinstance(flags.get("agent"), dict):
            update_config(config, flags["agent"])
        provided_args = getattr(args, "_provided_args", set())
        if "--env_name" not in provided_args:
            args.env_name = flags.get("env_name", args.env_name)
        if "--dataset_dir" not in provided_args:
            args.dataset_dir = flags.get("dataset_dir", args.dataset_dir)
        if "--libero_buffer_path" not in provided_args:
            args.libero_buffer_path = flags.get("libero_buffer_path", args.libero_buffer_path)
        if "--robomimic_buffer_path" not in provided_args:
            args.robomimic_buffer_path = flags.get(
                "robomimic_buffer_path",
                args.robomimic_buffer_path,
            )
        if "--seed" not in provided_args:
            args.seed = int(flags.get("seed", args.seed))
        if "--obs_horizon" not in provided_args:
            args.obs_horizon = int(flags.get("obs_horizon", args.obs_horizon))
        if "--action_horizon" not in provided_args:
            args.action_horizon = int(flags.get("action_horizon", args.action_horizon))

    model_state = checkpoint_state
    has_biflow_weights = any(
        (key[len("module.") :] if key.startswith("module.") else key).startswith("reverse_model.")
        for key in model_state.keys()
    )
    if args.use_biflow == "auto":
        config.use_biflow = has_biflow_weights or args.sample_backend == "biflow"
    else:
        config.use_biflow = args.use_biflow == "true"
    if args.sample_backend == "biflow" and not has_biflow_weights:
        raise ValueError(f"--sample_backend={args.sample_backend} requires reverse_model weights in the checkpoint.")
    if args.sample_backend == "auto":
        args.sample_backend = "biflow" if bool(config.get("use_biflow", False)) else "simflow"

    config.train_mode = "biflow_align" if bool(config.get("use_biflow", False)) else "iql"
    config.action_len = int(args.action_horizon)
    config.obs_horizon = int(args.obs_horizon)
    # Evaluation immediately restores every backbone parameter from the
    # checkpoint. Avoid a redundant torchvision download during construction.
    if str(args.env_name).startswith("robomimic_"):
        config.robomimic_resnet_pretrained_weights = None
    if args.actor_context_obs_horizon is not None:
        config.actor_context_obs_horizon = int(args.actor_context_obs_horizon)
    if args.smolvlm_model_path is not None:
        config.smolvlm_model_path = args.smolvlm_model_path
    if args.language_model_path is not None:
        config.language_model_path = args.language_model_path
    if args.biflow_eval_guidance is not None:
        config.biflow_eval_guidance = float(args.biflow_eval_guidance)
    if args.biflow_eval_guidance_final is not None:
        config.biflow_eval_guidance_final = float(args.biflow_eval_guidance_final)
    if args.cfg is not None:
        config.cfg = float(args.cfg)
    if args.denoising_lr is not None:
        config.denoising_lr = float(args.denoising_lr)
    return ml_collections.ConfigDict(config)


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


def make_dataset_and_agent(args, config, device):
    is_libero = is_libero_env_name(args.env_name)
    tokenizer_type, tokenizer_path, tokenizer_max_length = _language_tokenizer_spec(config)
    train_dataset, val_dataset = make_env_and_datasets(
        env_name=args.env_name,
        dataset_dir=args.dataset_dir,
        discount=float(config.discount),
        hubl_lambda_type=str(config.get("hubl_lambda_type", "rank")),
        hubl_alpha=float(config.get("hubl_alpha", 0.1)),
        libero_buffer_path=args.libero_buffer_path,
        robomimic_buffer_path=args.robomimic_buffer_path,
        robomimic_use_proprioception=uses_proprioception(config),
        language_model_path=tokenizer_path,
        language_max_length=tokenizer_max_length,
        language_tokenizer_type=tokenizer_type,
        action_clip_eps=1e-5,
    )
    if is_libero and bool(config.get('use_proprioception', False)):
        from utils.robomimic.dataset import configure_libero_proprioception_stats

        configure_libero_proprioception_stats(config, train_dataset)

    if args.obs_horizon is not None:
        _, _, sequence_length = VINFTorchAgent._get_sequence_spec(config)
        for dataset in [train_dataset, val_dataset]:
            if dataset is not None:
                dataset.sequence_length = sequence_length
                dataset.observation_horizon = int(args.obs_horizon)
                dataset.action_horizon = int(args.action_horizon)
    if (
        is_libero
        and getattr(args, "_libero_eval_visual_preprocess", "none") == "deterministic_center_crop"
    ):
        for dataset in [train_dataset, val_dataset]:
            if dataset is not None:
                dataset.visual_augmentation_mode = args._libero_visual_augmentation_mode
                dataset.apply_visual_eval_preprocessing = True
    del val_dataset

    example_batch = train_dataset.sample(1)
    agent = VINFTorchAgent.create(
        args.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
        device=device,
        ex_proprioceptions=example_batch.get('proprioceptions'),
    )
    return agent, train_dataset


def load_model(agent, checkpoint_state):
    checkpoint_state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in checkpoint_state.items()
    }
    agent.model.load_state_dict(checkpoint_state, strict=True)
    return agent


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    config = build_config(args, checkpoint, checkpoint_state)
    validate_supported_env(args.env_name)

    is_libero = is_libero_env_name(args.env_name)
    if args.fixed_episode_seeds is None:
        args.fixed_episode_seeds = is_libero
    if args.libero_use_official_initial_states is None:
        args.libero_use_official_initial_states = is_libero
    if is_libero:
        os.environ['LIBERO_USE_OFFICIAL_INITIAL_STATES'] = (
            '1' if args.libero_use_official_initial_states else '0'
        )
        if args.libero_use_official_initial_states:
            os.environ.setdefault('LIBERO_NUM_STEPS_WAIT', '10')
    if args.output is not None and Path(args.output).exists() and Path(args.output).is_dir():
        raise ValueError(
            f"--output must be a CSV file path, but got an existing directory: {args.output!r}. "
            "For example: --output exp/fql-orl-torch/libero_soup_cheese_iql_nf16/result.csv"
        )
    if args.episode_output is not None and Path(args.episode_output).is_dir():
        raise ValueError(
            f"--episode_output must be a JSON file path, got directory {args.episode_output!r}."
        )
    if args.fixed_episode_seeds and args.vec_env != "dummy":
        raise ValueError(
            "--fixed_episode_seeds requires --vec_env=dummy so each completed slot can "
            "be reset with its next explicit episode seed."
        )
    if (
        args.video_dir is not None or args.frame_dir is not None
    ) and not args.fixed_episode_seeds:
        raise ValueError(
            "--video_dir and --frame_dir require --fixed_episode_seeds=true."
        )
    if (
        args.video_episode_seeds is not None
        and args.video_dir is None
        and args.frame_dir is None
    ):
        raise ValueError(
            "--video_episode_seeds requires --video_dir or --frame_dir."
        )
    if args.video_frame_skip <= 0:
        raise ValueError(
            f"--video_frame_skip must be positive, got {args.video_frame_skip}."
        )
    if args.video_fps <= 0:
        raise ValueError(f"--video_fps must be positive, got {args.video_fps}.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    training_flags = getattr(args, "_training_flags", None)
    if training_flags is None:
        sibling_flags_path = Path(args.checkpoint).resolve().parent / "flags.json"
        if sibling_flags_path.is_file():
            training_flags = load_json(sibling_flags_path)
    (
        libero_eval_visual_preprocess,
        libero_eval_visual_preprocess_source,
        training_p_aug,
        libero_visual_augmentation_mode,
    ) = _resolve_libero_eval_visual_preprocess(
        args.libero_eval_visual_preprocess,
        args.env_name,
        training_flags,
    )
    args._libero_eval_visual_preprocess = libero_eval_visual_preprocess
    args._libero_visual_augmentation_mode = libero_visual_augmentation_mode
    print(
        "[eval-visual] "
        f"requested={args.libero_eval_visual_preprocess}, "
        f"resolved={libero_eval_visual_preprocess}, "
        f"source={libero_eval_visual_preprocess_source}, "
        f"training_p_aug={training_p_aug}, "
        f"augmentation_mode={libero_visual_augmentation_mode}",
        flush=True,
    )
    tokenizer_type, tokenizer_path, tokenizer_max_length = _language_tokenizer_spec(config)
    if checkpoint_state is not checkpoint:
        del checkpoint
    if is_libero:
        ensure_libero_config()

    agent, train_dataset = make_dataset_and_agent(args, config, device)
    agent = load_model(agent, checkpoint_state)
    agent.model.eval()
    if bool(config.get("use_language_conditioning", False)):
        from utils.language import build_text_tokenizer
        from utils.robomimic.dataset import get_task_name_from_path

        tokenizer = build_text_tokenizer(
            tokenizer_type,
            model_path=tokenizer_path,
            max_length=tokenizer_max_length,
        )
        dataset_path = args.dataset_dir
        if "*" in str(dataset_path):
            matches = sorted(glob.glob(dataset_path))
            if len(matches) == 0:
                raise ValueError(f"No LIBERO task files matched dataset_dir glob: {dataset_path}")
            dataset_path = matches[0]
        if dataset_path and os.path.exists(os.path.expanduser(str(dataset_path))):
            language_text = get_task_name_from_path(dataset_path)
        else:
            language_text = str(args.env_name)
        eval_language_tokens = tokenizer.tokenize(language_text, device=device)
    else:
        eval_language_tokens = (None, None)

    num_envs = int(args.num_envs) if args.num_envs is not None else int(args.eval_episodes)
    num_envs = max(1, min(num_envs, int(args.eval_episodes)))
    env_fns = [
        make_env(
            args.env_name,
            dataset_dir=args.dataset_dir,
            obs_horizon=args.obs_horizon,
            robomimic_use_proprioception=uses_proprioception(config),
        )
        for i in range(num_envs)
    ]
    if args.vec_env == "subproc" and num_envs > 1:
        envs = SubprocVecEnv(env_fns, start_method="fork")
    else:
        envs = DummyVecEnv(env_fns)

    def eval_actor(obs):
        observations, proprioceptions = agent._prepare_policy_inputs(obs)
        if libero_eval_visual_preprocess == "deterministic_center_crop":
            observations = apply_libero_visual_eval_preprocessing(
                observations,
                mode=libero_visual_augmentation_mode,
            )
        return agent.sample_actions(
            observations,
            temperature=args.temperature,
            cfg=(
                float(config.get("biflow_eval_guidance", 0.0))
                if args.sample_backend == "biflow"
                else float(config.get("cfg", 1.1))
            ),
            cfg_final=(
                float(config.get("biflow_eval_guidance_final", -1.0))
                if args.sample_backend == "biflow"
                else None
            ),
            apply_denoising=bool(args.apply_denoising),
            input_ids=eval_language_tokens[0],
            attention_mask=eval_language_tokens[1],
            sample_backend=args.sample_backend,
            proprioceptions=proprioceptions,
        )

    episode_records = None
    try:
        if args.fixed_episode_seeds:
            metrics, episode_records = torch_evaluate_parallel_fixed_seeds(
                eval_actor,
                envs=envs,
                num_eval_episodes=args.eval_episodes,
                episode_seed_start=args.episode_seed_start,
                clip_actions=bool(args.clip_actions),
                action_exec_horizon=args.action_exec_horizon,
                video_dir=args.video_dir,
                frame_dir=args.frame_dir,
                video_episode_seeds=args.video_episode_seeds,
                video_frame_skip=args.video_frame_skip,
                video_fps=args.video_fps,
            )
        else:
            metrics = torch_evaluate_parallel(
                eval_actor,
                seed=args.seed,
                envs=envs,
                num_eval_episodes=args.eval_episodes,
                clip_actions=bool(args.clip_actions),
                action_exec_horizon=args.action_exec_horizon,
            )
    finally:
        envs.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "env_name": args.env_name,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "seed": args.seed,
        "obs_horizon": args.obs_horizon,
        "actor_context_obs_horizon": int(
            config.get("actor_context_obs_horizon", 0)
        ),
        "action_horizon": args.action_horizon,
        "eval_episodes": args.eval_episodes,
        "temperature": (
            float(args.temperature)
            if args.temperature is not None
            else float(config.get("biflow_eval_temperature", 0.0))
            if args.sample_backend == "biflow"
            else 1.0
        ),
        "cfg": float(config.get("cfg", 1.1)),
        "denoising_lr": float(config.get("denoising_lr", 0.0)),
        "clip_actions": bool(args.clip_actions),
        "use_biflow": bool(config.get("use_biflow", False)),
        "sample_backend": args.sample_backend,
        "q_agg": str(config.get("q_agg", "min")),
        "biflow_eval_guidance": float(config.get("biflow_eval_guidance", 0.0)),
        "biflow_eval_guidance_final": float(config.get("biflow_eval_guidance_final", -1.0)),
        "iql_expectile": float(config.get("iql_expectile", 0.8)),
        "num_envs": num_envs,
        "action_exec_horizon": args.action_exec_horizon,
        "vec_env": args.vec_env,
        "fixed_episode_seeds": bool(args.fixed_episode_seeds),
        "video_dir": (
            str(Path(args.video_dir).resolve()) if args.video_dir is not None else None
        ),
        "frame_dir": (
            str(Path(args.frame_dir).resolve()) if args.frame_dir is not None else None
        ),
        "video_episode_seeds": (
            ";".join(str(seed) for seed in args.video_episode_seeds)
            if args.video_episode_seeds is not None
            else None
        ),
        "video_frame_skip": int(args.video_frame_skip),
        "video_fps": int(args.video_fps),
        "videos_saved": int(
            sum("video_path" in item for item in (episode_records or []))
        ),
        "frame_sequences_saved": int(
            sum("frame_dir" in item for item in (episode_records or []))
        ),
        "frames_saved": int(
            sum(int(item.get("frame_count", 0)) for item in (episode_records or []))
        ),
        "libero_eval_visual_preprocess": libero_eval_visual_preprocess,
        "libero_eval_visual_preprocess_source": libero_eval_visual_preprocess_source,
        "libero_visual_augmentation_mode": libero_visual_augmentation_mode,
        "training_p_aug": training_p_aug,
        "episode_seed_start": (
            int(args.episode_seed_start) if args.fixed_episode_seeds else None
        ),
        "episode_seed_end": (
            int(metrics.get("fixed_episode_seed_end", args.episode_seed_start))
            if args.fixed_episode_seeds
            else None
        ),
        "accepted_episode_seeds": (
            ";".join(
                str(seed) for seed in metrics.get("accepted_episode_seeds", [])
            )
            if args.fixed_episode_seeds
            else None
        ),
        "success": float(metrics.get("success", 0.0)),
    }
    episode_output_path = None
    if episode_records is not None:
        if args.episode_output is not None:
            episode_output_path = Path(args.episode_output)
        elif args.output is not None:
            episode_output_path = Path(args.output).with_suffix(".episodes.json")
        if episode_output_path is not None:
            episode_output_path.parent.mkdir(parents=True, exist_ok=True)
            accepted_episode_seeds = [
                int(item["episode_seed"]) for item in episode_records
            ]
            episode_payload = {
                "protocol": "official fixed-seed evaluation",
                "checkpoint": os.path.abspath(args.checkpoint),
                "env_name": args.env_name,
                "policy_seed": int(args.seed),
                "libero_eval_visual_preprocess": libero_eval_visual_preprocess,
                "libero_eval_visual_preprocess_source": libero_eval_visual_preprocess_source,
                "libero_visual_augmentation_mode": libero_visual_augmentation_mode,
                "training_p_aug": training_p_aug,
                "episode_seed_start": int(args.episode_seed_start),
                "episode_seeds": accepted_episode_seeds,
                "num_envs": int(num_envs),
                "success": float(metrics.get("success", 0.0)),
                "episodes": episode_records,
            }
            with episode_output_path.open("w", encoding="utf-8") as f:
                json.dump(episode_payload, f, indent=2, sort_keys=True)
                f.write("\n")
            result["episode_output"] = str(episode_output_path.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists()
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)


if __name__ == "__main__":
    main()
