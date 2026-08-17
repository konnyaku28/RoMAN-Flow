"""Episode-aware fixed-length sequence indexing and padding."""

from __future__ import annotations

import numpy as np


_ZERO_PAD_FIELDS = {
    "actions", "rewards", "masks", "mc_returns", "hubl_lambda",
    "hubl_rewards", "hubl_discounts",
}


def sequence_right_padding(seq_len: int, obs_horizon=None, action_horizon=None) -> int:
    """Return the unused suffix after an observation/action training window."""
    seq_len = int(seq_len)
    if obs_horizon is None or action_horizon is None:
        return 0
    obs_horizon = int(obs_horizon)
    action_horizon = int(action_horizon)
    if obs_horizon <= 0 or action_horizon <= 0:
        return 0
    used_steps = obs_horizon - 1 + action_horizon
    if used_steps > seq_len:
        raise ValueError(
            f"Sequence length {seq_len} cannot contain obs_horizon={obs_horizon} "
            f"and action_horizon={action_horizon}."
        )
    return seq_len - used_steps


class EpisodeSequenceIndex:
    """Index valid windows without allowing them to cross episode boundaries."""

    def __init__(
        self, store, seq_len: int, episode_mask: np.ndarray | None = None,
        pad_after: int = 0,
    ) -> None:
        self.store = store
        self.seq_len = int(seq_len)
        self.pad_after = int(pad_after)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if not 0 <= self.pad_after < self.seq_len:
            raise ValueError(f"pad_after must be in [0, seq_len), got {self.pad_after}.")
        self.keys = list(store.keys())
        self.arrays = {key: store[key] for key in self.keys}
        ends = np.asarray(store.episode_ends[:], dtype=np.int64)
        if episode_mask is not None:
            episode_mask = np.asarray(episode_mask, dtype=bool)
            if episode_mask.shape != ends.shape:
                raise ValueError(
                    f"episode_mask shape {episode_mask.shape} does not match "
                    f"episode count {ends.shape}."
                )

        windows: list[tuple[int, int]] = []
        uniform: list[int] = []
        selected_lengths: list[int] = []
        episode_start = 0
        required_steps = self.seq_len - self.pad_after
        for episode_idx, episode_end_value in enumerate(ends):
            episode_end = int(episode_end_value)
            selected = episode_mask is None or bool(episode_mask[episode_idx])
            if selected:
                selected_lengths.append(episode_end - episode_start)
                final_uniform_start = episode_end - required_steps
                for start in range(episode_start, final_uniform_start + 1):
                    uniform.append(len(windows))
                    windows.append((start, min(start + self.seq_len, episode_end)))
            episode_start = episode_end

        if not windows:
            raise ValueError(
                "Dataset contains no valid training sequences: "
                f"seq_len={self.seq_len}, pad_after={self.pad_after}, "
                f"required_real_steps={required_steps}, "
                f"selected_episode_lengths={selected_lengths[:20]}."
            )
        self.indices = np.asarray(windows, dtype=np.int64)
        self.uniform_indices = np.asarray(uniform, dtype=np.int64)
        print(
            f"Total number of valid sequences: {len(self.indices)} "
            f"(seq_len={self.seq_len}, pad_after={self.pad_after})",
            flush=True,
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def sample_sequence(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        start, end = self.indices[index]
        real_steps = int(end - start)
        padding = self.seq_len - real_steps
        sample = {}
        for key, source in self.arrays.items():
            value = np.asarray(source[start:end])
            if padding:
                widths = [(0, padding), *([(0, 0)] * (value.ndim - 1))]
                value = np.pad(value, widths, mode="edge")
                if key in _ZERO_PAD_FIELDS:
                    value[real_steps:] = 0
                elif key == "terminals":
                    value[real_steps:] = 1
            sample[key] = value

        valid = np.zeros(self.seq_len, dtype=np.float32)
        valid[:real_steps] = 1.0
        sample["sequence_valid_mask"] = valid
        sample["action_valid_mask"] = valid.copy()
        return sample
