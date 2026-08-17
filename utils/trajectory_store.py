"""Portable Zarr storage for fixed-schema episodic trajectories."""

from __future__ import annotations

import os
from multiprocessing import Lock
from os.path import expanduser, expandvars
from typing import Mapping

import numpy as np


def _zarr_modules():
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Disk-backed datasets require zarr and numcodecs. Run setup_env.sh "
            "before preparing LIBERO or RoboMimic data."
        ) from exc
    return zarr, numcodecs


def _chunk_shape(
    shape: tuple[int, ...], dtype: str | np.dtype, target_bytes: int = 2_000_000
) -> tuple[int, ...]:
    """Choose a transition-aligned chunk close to ``target_bytes``."""
    if not shape or shape[0] <= 0:
        raise ValueError(f"Array shape must have a positive capacity, got {shape}.")
    bytes_per_step = int(np.dtype(dtype).itemsize * np.prod(shape[1:], dtype=np.int64))
    leading = max(1, min(shape[0], target_bytes // max(1, bytes_per_step)))
    return (leading, *shape[1:])


class ZarrTrajectoryStore:
    """Fixed-capacity trajectory arrays with episode boundaries in Zarr v2."""

    def __init__(
        self,
        storage_path: str,
        schema: Mapping[str, Mapping[str, object]],
        max_steps: int | None = None,
        lock: Lock | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        zarr, numcodecs = _zarr_modules()
        storage_path = expandvars(expanduser(storage_path))
        self.restored = os.path.exists(storage_path)
        self.storage = zarr.DirectoryStore(storage_path)
        self.lock = Lock() if lock is None else lock
        self.root = zarr.open_group(store=self.storage, mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        if self.restored:
            self._validate_existing(storage_path, schema, attributes)
            lengths = {int(array.shape[0]) for _, array in self.data.arrays()}
            self.max_steps = lengths.pop()
            print(f"Restoring buffer from {storage_path}", flush=True)
        else:
            if max_steps is None or int(max_steps) <= 0:
                raise ValueError("max_steps must be positive when creating a buffer.")
            self.max_steps = int(max_steps)
            print(f"Creating new buffer at {storage_path}", flush=True)
            with self.lock:
                if attributes:
                    self.root.attrs.update(dict(attributes))
                self.meta.zeros(
                    name="episode_ends",
                    shape=(0,),
                    chunks=(1024,),
                    dtype=np.int64,
                    compressor=None,
                )
                for key, specification in schema.items():
                    item_shape = tuple(specification["shape"])
                    shape = (self.max_steps, *item_shape)
                    dtype = np.dtype(specification["dtype"])
                    compressor = numcodecs.Blosc(
                        cname="lz4",
                        clevel=5 if dtype == np.dtype(np.uint8) else 0,
                        shuffle=numcodecs.Blosc.NOSHUFFLE,
                    )
                    self.data.zeros(
                        name=key,
                        shape=shape,
                        chunks=_chunk_shape(shape, dtype),
                        dtype=dtype,
                        compressor=compressor,
                    )

        self._arrays = {key: self.data[key] for key in self.data.keys()}
        self._episode_ends = self.meta["episode_ends"]

    def _validate_existing(self, path, schema, attributes) -> None:
        if "episode_ends" not in self.meta:
            raise ValueError(f"Buffer at {path} is missing meta/episode_ends.")
        missing = sorted(set(schema) - set(self.data.keys()))
        if missing:
            raise ValueError(f"Buffer at {path} is missing datasets: {missing}.")
        for key, specification in schema.items():
            expected_shape = tuple(specification["shape"])
            expected_dtype = np.dtype(specification["dtype"])
            array = self.data[key]
            if tuple(array.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Buffer field {key!r} has item shape {array.shape[1:]}, "
                    f"expected {expected_shape}."
                )
            if np.dtype(array.dtype) != expected_dtype:
                raise ValueError(
                    f"Buffer field {key!r} has dtype {array.dtype}, "
                    f"expected {expected_dtype}."
                )
        lengths = {int(array.shape[0]) for _, array in self.data.arrays()}
        if len(lengths) != 1:
            raise ValueError(f"Buffer at {path} has inconsistent field capacities.")
        for key, expected in (attributes or {}).items():
            actual = self.root.attrs.get(key)
            if actual != expected:
                raise ValueError(
                    f"Buffer at {path} has incompatible attribute {key!r}: "
                    f"expected {expected!r}, got {actual!r}."
                )

    @property
    def episode_ends(self):
        return self._episode_ends

    @property
    def num_episodes(self) -> int:
        return int(self._episode_ends.shape[0])

    @property
    def num_steps(self) -> int:
        if self.num_episodes == 0:
            return 0
        return int(self._episode_ends[-1])

    def append_episode(self, episode: Mapping[str, np.ndarray]) -> None:
        if set(episode) != set(self._arrays):
            missing = sorted(set(self._arrays) - set(episode))
            extra = sorted(set(episode) - set(self._arrays))
            raise ValueError(f"Episode fields do not match schema; missing={missing}, extra={extra}.")
        lengths = {int(np.asarray(value).shape[0]) for value in episode.values()}
        if len(lengths) != 1:
            raise ValueError("All episode fields must have the same leading length.")
        episode_steps = lengths.pop()
        if episode_steps <= 0:
            raise ValueError("Cannot append an empty episode.")

        with self.lock:
            start = self.num_steps
            end = start + episode_steps
            if end > self.max_steps:
                raise RuntimeError(
                    f"Buffer capacity exceeded: requested {end}, capacity {self.max_steps}."
                )
            for key, value in episode.items():
                array = np.asarray(value)
                expected = self._arrays[key]
                if tuple(array.shape[1:]) != tuple(expected.shape[1:]):
                    raise ValueError(
                        f"Episode field {key!r} has item shape {array.shape[1:]}, "
                        f"expected {expected.shape[1:]}."
                    )
                self._arrays[key][start:end] = array
            self._episode_ends.resize(self.num_episodes + 1)
            self._episode_ends[-1] = end

    def keys(self):
        return self._arrays.keys()

    def __getitem__(self, key):
        return self._arrays[key]
