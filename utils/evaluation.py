"""Evaluation utilities for native PyTorch policies."""

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


def _stack_observation_batch(observations):
    if not observations:
        raise ValueError("Cannot stack an empty observation batch.")
    first = observations[0]
    if isinstance(first, dict):
        keys = tuple(first.keys())
        if any(tuple(observation.keys()) != keys for observation in observations):
            raise ValueError("Observation mappings must have identical ordered keys.")
        return {
            key: np.stack([observation[key] for observation in observations], axis=0)
            for key in keys
        }
    return np.stack(observations, axis=0)


def _observation_batch_size(observation):
    if isinstance(observation, dict):
        if not observation:
            raise ValueError("Observation mapping is empty.")
        sizes = {int(np.asarray(value).shape[0]) for value in observation.values()}
        if len(sizes) != 1:
            raise ValueError(
                f"Observation mapping has inconsistent batch sizes: {sizes}."
            )
        return sizes.pop()
    return int(np.asarray(observation).shape[0])


def torch_evaluate_parallel(
    actor,
    seed,
    envs,
    num_eval_episodes=50,
    clip_actions=True,
    action_exec_horizon=None,
):
    """Torch version of vectorized evaluation matching the JAX evaluate_parallel behavior."""

    if hasattr(envs, "envs"):
        results = []
        for idx, env in enumerate(envs.envs):
            results.append(env.reset(seed=seed + idx))
        obs_list, info_list = zip(*results)
        del info_list
        observation = _stack_observation_batch(obs_list)
    else:
        if hasattr(envs, "seed"):
            envs.seed(seed)
        reset_result = envs.reset()
        observation = (
            reset_result[0] if isinstance(reset_result, tuple) else reset_result
        )

    num_envs = _observation_batch_size(observation)
    if num_envs < 1:
        raise ValueError(
            f"Expected at least one evaluation environment, got {num_envs}."
        )

    episode_successes = []
    started_episodes = min(num_envs, int(num_eval_episodes))
    active = np.arange(num_envs) < started_episodes
    current_success = np.zeros((num_envs,), dtype=np.float32)
    pbar = tqdm(
        total=num_eval_episodes, desc="eval episodes", dynamic_ncols=True, leave=False
    )
    while len(episode_successes) < num_eval_episodes:
        with torch.no_grad():
            action = actor(observation)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        if action_exec_horizon is not None and action.ndim >= 3:
            action = action[:, : int(action_exec_horizon)]
        if clip_actions:
            action = np.clip(action, -1, 1)

        next_observation, reward, next_done, info = envs.step(action)
        del reward
        step_success = np.array(
            [item.get("success", 0.0) for item in info], dtype=np.float32
        )
        current_success = np.where(
            active, np.maximum(current_success, step_success), current_success
        )
        for env_idx, env_done in enumerate(np.asarray(next_done, dtype=bool)):
            if not env_done or not active[env_idx]:
                continue
            if len(episode_successes) < num_eval_episodes:
                episode_successes.append(float(current_success[env_idx]))
                pbar.update(1)
            current_success[env_idx] = 0.0
            if started_episodes < num_eval_episodes:
                started_episodes += 1
            else:
                active[env_idx] = False
        observation = next_observation

    pbar.close()

    return {"success": float(np.mean(episode_successes, dtype=np.float32))}


def _reset_observation(reset_result):
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def _find_video_recorder(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "record") and callable(getattr(current, "render", None)):
            return current
        current = getattr(current, "env", None)
    return None


def _set_video_recording(env, enabled):
    recorder = _find_video_recorder(env)
    if recorder is None:
        if enabled:
            raise RuntimeError(
                "Video recording was requested, but the environment does not expose "
                "a record flag and render method."
            )
        return None
    recorder.record = bool(enabled)
    if not hasattr(recorder, "video_buffer"):
        recorder.video_buffer = []
    else:
        recorder.video_buffer.clear()
    return recorder


def _write_episode_video(
    recorder,
    video_dir,
    episode_seed,
    success,
    frame_skip,
    fps,
):
    frames = np.asarray(recorder.video_buffer)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or len(frames) == 0:
        raise RuntimeError(
            f"Cannot save video for seed {episode_seed}: expected non-empty HWC RGB "
            f"frames, got shape {frames.shape}."
        )
    frames = frames[::frame_skip]
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / (
        f"episode_seed_{int(episode_seed):06d}_success_{int(bool(success))}.mp4"
    )
    import imageio.v2 as imageio

    imageio.mimsave(video_path, frames, fps=int(fps))
    return str(video_path.resolve())


def _write_episode_frames(
    recorder,
    frame_dir,
    episode_seed,
    success,
    frame_skip,
):
    frames = np.asarray(recorder.video_buffer)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or len(frames) == 0:
        raise RuntimeError(
            f"Cannot save frames for seed {episode_seed}: expected non-empty HWC RGB "
            f"frames, got shape {frames.shape}."
        )
    frames = frames[::frame_skip]
    episode_dir = Path(frame_dir) / (
        f"episode_seed_{int(episode_seed):06d}_success_{int(bool(success))}"
    )
    episode_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in episode_dir.glob("frame_*.png"):
        stale_path.unlink()

    import imageio.v2 as imageio

    for frame_index, frame in enumerate(frames):
        imageio.imwrite(episode_dir / f"frame_{frame_index:06d}.png", frame)
    return str(episode_dir.resolve()), int(len(frames))


def torch_evaluate_parallel_fixed_seeds(
    actor,
    envs,
    num_eval_episodes=50,
    episode_seed_start=0,
    clip_actions=True,
    action_exec_horizon=None,
    video_dir=None,
    frame_dir=None,
    video_episode_seeds=None,
    video_frame_skip=2,
    video_fps=20,
):
    """Evaluate consecutive episode seeds without vector-env automatic resets."""

    num_eval_episodes = int(num_eval_episodes)
    episode_seed_start = int(episode_seed_start)
    if num_eval_episodes <= 0:
        raise ValueError(
            f"num_eval_episodes must be positive, got {num_eval_episodes}."
        )
    if not hasattr(envs, "envs"):
        raise TypeError(
            "Fixed-seed evaluation requires DummyVecEnv (an object exposing .envs); "
            "SubprocVecEnv cannot be reset slot-by-slot with explicit seeds."
        )

    raw_envs = list(envs.envs)
    num_slots = min(len(raw_envs), num_eval_episodes)
    if num_slots <= 0:
        raise ValueError("Fixed-seed evaluation requires at least one environment.")

    if int(video_frame_skip) <= 0:
        raise ValueError(f"video_frame_skip must be positive, got {video_frame_skip}.")
    if int(video_fps) <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}.")
    requested_recording_seeds = set()
    recording_enabled = video_dir is not None or frame_dir is not None
    record_all_rollouts = recording_enabled and video_episode_seeds is None
    if recording_enabled:
        requested_recording_seeds = (
            set()
            if record_all_rollouts
            else {int(seed) for seed in video_episode_seeds}
        )
        initial_seed_range = set(
            range(episode_seed_start, episode_seed_start + num_eval_episodes)
        )
        unexpected_recording_seeds = requested_recording_seeds.difference(
            initial_seed_range
        )
        if unexpected_recording_seeds:
            raise ValueError(
                "video_episode_seeds must be part of the evaluated seed range; got "
                f"{sorted(unexpected_recording_seeds)} outside "
                f"{episode_seed_start}.."
                f"{episode_seed_start + num_eval_episodes - 1}."
            )
    elif video_episode_seeds:
        raise ValueError("video_episode_seeds requires video_dir or frame_dir.")
    slot_states = [None] * num_slots
    next_episode_seed = episode_seed_start
    completed = []

    def start_episode(slot_idx):
        nonlocal next_episode_seed
        episode_seed = int(next_episode_seed)
        next_episode_seed += 1
        reset_result = raw_envs[slot_idx].reset(seed=episode_seed)
        observation = _reset_observation(reset_result)

        record_video = record_all_rollouts or episode_seed in requested_recording_seeds
        recorder = (
            _set_video_recording(raw_envs[slot_idx], record_video)
            if recording_enabled
            else None
        )
        if record_video and frame_dir is not None:
            recorder.video_buffer.append(recorder.render())
        slot_states[slot_idx] = {
            "episode_seed": int(episode_seed),
            "observation": observation,
            "success": 0.0,
            "policy_steps": 0,
            "record_video": bool(record_video),
            "video_recorder": recorder,
        }

    for slot_idx in range(num_slots):
        start_episode(slot_idx)

    pbar = tqdm(
        total=num_eval_episodes,
        desc="eval fixed episodes",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        while len(completed) < num_eval_episodes:
            active_slots = [
                slot_idx
                for slot_idx, state in enumerate(slot_states)
                if state is not None
            ]
            if not active_slots:
                raise RuntimeError("Fixed-seed evaluator has no active episode slots.")
            observation = _stack_observation_batch(
                [slot_states[slot_idx]["observation"] for slot_idx in active_slots]
            )

            with torch.no_grad():
                action = actor(observation)

            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.asarray(action)
            if action.ndim == 0:
                raise ValueError(
                    "Actor returned a scalar action during fixed-seed evaluation."
                )
            if action.shape[0] != len(active_slots):
                if len(active_slots) == 1:
                    action = np.expand_dims(action, axis=0)
                else:
                    raise ValueError(
                        f"Actor returned batch size {action.shape[0]} for "
                        f"{len(active_slots)} active environments."
                    )
            if action_exec_horizon is not None and action.ndim >= 3:
                action = action[:, : int(action_exec_horizon)]
            if clip_actions:
                action = np.clip(action, -1, 1)

            for batch_idx, slot_idx in enumerate(active_slots):
                state = slot_states[slot_idx]
                state["policy_steps"] += 1
                step_result = raw_envs[slot_idx].step(action[batch_idx])
                if len(step_result) == 5:
                    next_observation, _, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                elif len(step_result) == 4:
                    next_observation, _, done, info = step_result
                    done = bool(done)
                else:
                    raise ValueError(
                        f"Environment step returned {len(step_result)} values; expected 4 or 5."
                    )
                info = {} if info is None else info
                state["success"] = max(
                    float(state["success"]),
                    float(info.get("success", 0.0)),
                )
                state["observation"] = next_observation
                if not done:
                    continue

                episode_record = {
                    "episode_seed": int(state["episode_seed"]),
                    "success": bool(state["success"]),
                    "policy_steps": int(state["policy_steps"]),
                }
                if state["record_video"]:
                    if video_dir is not None:
                        episode_record["video_path"] = _write_episode_video(
                            state["video_recorder"],
                            video_dir,
                            state["episode_seed"],
                            state["success"],
                            int(video_frame_skip),
                            int(video_fps),
                        )
                    if frame_dir is not None:
                        frame_path, frame_count = _write_episode_frames(
                            state["video_recorder"],
                            frame_dir,
                            state["episode_seed"],
                            state["success"],
                            int(video_frame_skip),
                        )
                        episode_record["frame_dir"] = frame_path
                        episode_record["frame_count"] = frame_count
                completed.append(episode_record)
                if recording_enabled:
                    _set_video_recording(raw_envs[slot_idx], False)
                pbar.update(1)
                slot_states[slot_idx] = None
                active_count = sum(item is not None for item in slot_states)
                if len(completed) + active_count < num_eval_episodes:
                    start_episode(slot_idx)
    finally:
        pbar.close()

    completed.sort(key=lambda item: item["episode_seed"])
    stats = {
        "success": float(np.mean([item["success"] for item in completed])),
        "fixed_episode_seed_start": float(episode_seed_start),
        "fixed_episode_seed_count": float(num_eval_episodes),
        "fixed_episode_seed_end": float(
            max(item["episode_seed"] for item in completed)
        ),
        "accepted_episode_seeds": [int(item["episode_seed"]) for item in completed],
    }
    return stats, completed
