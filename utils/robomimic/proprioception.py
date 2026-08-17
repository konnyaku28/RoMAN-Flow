from __future__ import annotations

import math

import numpy as np


LIBERO_PROPRIO_DIM = 8
LIBERO_PROPRIO_FORMAT = 'libero_eef_pos_axis_angle_gripper_v1'


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert LIBERO's [x, y, z, w] quaternion to a rotation vector."""
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    if quaternion.shape != (4,):
        raise ValueError(f'Expected quaternion shape (4,), got {quaternion.shape}.')
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f'Invalid quaternion norm: {norm}.')
    quaternion /= norm
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quaternion[3] ** 2))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(float(quaternion[3]))
    return (quaternion[:3] * angle / denominator).astype(np.float32)


def euler_xyz_to_axis_angle(euler: np.ndarray) -> np.ndarray:
    """Match SimVLA's scipy XYZ Euler -> quaternion -> axis-angle conversion."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError('LIBERO proprioception conversion requires scipy.') from exc

    euler = np.asarray(euler, dtype=np.float64)
    if euler.ndim == 1:
        if euler.shape != (3,):
            raise ValueError(f'Expected Euler shape (3,), got {euler.shape}.')
        return quaternion_to_axis_angle(Rotation.from_euler('xyz', euler).as_quat())
    if euler.ndim != 2 or euler.shape[1] != 3:
        raise ValueError(f'Expected Euler shape [T, 3], got {euler.shape}.')
    quaternions = Rotation.from_euler('xyz', euler).as_quat()
    return np.stack(
        [quaternion_to_axis_angle(quaternion) for quaternion in quaternions],
        axis=0,
    ).astype(np.float32, copy=False)


def libero_hdf5_proprioception(obs_group) -> np.ndarray:
    if 'gripper_states' not in obs_group:
        raise KeyError("LIBERO HDF5 observation is missing state field 'gripper_states'.")
    if 'ee_pos' in obs_group and 'ee_ori' in obs_group:
        ee_pos = np.asarray(obs_group['ee_pos'][:], dtype=np.float32)
        ee_ori = np.asarray(obs_group['ee_ori'][:], dtype=np.float32)
    elif 'ee_states' in obs_group:
        ee_states = np.asarray(obs_group['ee_states'][:], dtype=np.float32)
        if ee_states.ndim != 2 or ee_states.shape[1] != 6:
            raise ValueError(
                f'Expected LIBERO ee_states with shape [T, 6], got {ee_states.shape}.'
            )
        ee_pos, ee_ori = ee_states[:, :3], ee_states[:, 3:]
    else:
        raise KeyError(
            "LIBERO HDF5 observation must contain ee_pos/ee_ori or ee_states."
        )
    ee_orientation = euler_xyz_to_axis_angle(ee_ori)
    gripper = np.asarray(obs_group['gripper_states'][:], dtype=np.float32)
    proprioception = np.concatenate([ee_pos, ee_orientation, gripper], axis=-1)
    if proprioception.ndim != 2 or proprioception.shape[1] != LIBERO_PROPRIO_DIM:
        raise ValueError(
            'LIBERO state must contain ee_pos(3), axis_angle(3), and gripper(2); '
            f'got {proprioception.shape}.'
        )
    return proprioception.astype(np.float32, copy=False)


def libero_rollout_proprioception(observation: dict) -> np.ndarray:
    required = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f'LIBERO rollout observation is missing state fields: {missing}.')
    proprioception = np.concatenate(
        [
            np.asarray(observation['robot0_eef_pos'], dtype=np.float32).reshape(-1),
            quaternion_to_axis_angle(observation['robot0_eef_quat']),
            np.asarray(observation['robot0_gripper_qpos'], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )
    if proprioception.shape != (LIBERO_PROPRIO_DIM,):
        raise ValueError(
            f'Expected {LIBERO_PROPRIO_DIM}-D LIBERO rollout state, got {proprioception.shape}.'
        )
    return proprioception.astype(np.float32, copy=False)


def proprioception_quantiles(
    proprioceptions: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    proprioceptions = np.asarray(proprioceptions, dtype=np.float32)
    if proprioceptions.ndim != 2 or proprioceptions.shape[1] != LIBERO_PROPRIO_DIM:
        raise ValueError(
            f'Expected proprioceptions [N, {LIBERO_PROPRIO_DIM}], got {proprioceptions.shape}.'
        )
    q_low = np.quantile(proprioceptions, lower, axis=0).astype(np.float32)
    q_high = np.quantile(proprioceptions, upper, axis=0).astype(np.float32)
    if not np.all(np.isfinite(q_low)) or not np.all(np.isfinite(q_high)):
        raise ValueError('LIBERO proprioception quantiles contain non-finite values.')
    if np.any(q_high <= q_low):
        dimensions = np.nonzero(q_high <= q_low)[0].tolist()
        raise ValueError(f'Degenerate LIBERO proprioception quantiles at dimensions {dimensions}.')
    return q_low, q_high
