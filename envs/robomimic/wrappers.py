from collections import deque

import numpy as np
import gymnasium as gym

from utils.robomimic.proprioception import (
    LIBERO_PROPRIO_DIM,
    libero_rollout_proprioception,
)


class RoboMimicEnvWrapper(gym.Env):
    """Expose a RoboMimic Robosuite environment using this project's image layout."""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        env,
        obs_keys,
        obs_horizon=1,
        max_episode_length=400,
        record=False,
        render_size=(224, 224),
        video_camera='agentview',
        use_proprioception=False,
        proprio_keys=(
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ),
    ):
        self.env = env
        self.obs_keys = tuple(obs_keys)
        self.obs_horizon = int(obs_horizon)
        self.obs_buffer = deque(maxlen=self.obs_horizon)
        self.max_episode_length = int(max_episode_length)
        self.record = bool(record)
        self.render_size = tuple(render_size)
        self.video_camera = str(video_camera).strip()
        self.use_proprioception = bool(use_proprioception)
        self.proprio_keys = tuple(proprio_keys)
        self._elapsed_steps = 0
        if self.obs_horizon <= 0:
            raise ValueError(f'obs_horizon must be positive, got {obs_horizon}.')
        if len(self.obs_keys) != 2:
            raise ValueError(f'Expected two RoboMimic image observations, got {self.obs_keys}.')
        if len(self.render_size) != 2 or any(int(size) <= 0 for size in self.render_size):
            raise ValueError(f'render_size must contain two positive values, got {render_size}.')
        if not self.video_camera:
            raise ValueError('video_camera must be non-empty.')
        if self.record:
            self.video_buffer = []

        action_dim = int(getattr(env, 'action_dimension', 7))
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(len(self.obs_keys), 3, self.obs_horizon, 84, 84),
            dtype=np.uint8,
        )
        if self.use_proprioception:
            self.observation_space = gym.spaces.Dict({
                'images': image_space,
                'proprioceptions': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_horizon, 9),
                    dtype=np.float32,
                ),
            })
        else:
            self.observation_space = image_space

    @staticmethod
    def _image_to_chw(image, key):
        image = np.asarray(image)
        if image.shape == (84, 84, 3):
            return np.transpose(image, (2, 0, 1))
        if image.shape == (3, 84, 84):
            return image
        raise ValueError(f'Unexpected RoboMimic image shape for {key}: {image.shape}.')

    def _format_single_images(self, raw_obs):
        images = [self._image_to_chw(raw_obs[key], key) for key in self.obs_keys]
        return np.stack(images, axis=0).astype(np.uint8, copy=False)

    def _format_single_proprioception(self, raw_obs):
        values = []
        for key in self.proprio_keys:
            if key not in raw_obs:
                raise KeyError(f'RoboMimic rollout observation is missing {key!r}.')
            values.append(np.asarray(raw_obs[key], dtype=np.float32).reshape(-1))
        proprioception = np.concatenate(values, axis=0).astype(np.float32, copy=False)
        if proprioception.shape != (9,):
            raise ValueError(
                f'Expected 9-D RoboMimic proprioception, got {proprioception.shape}.'
            )
        return proprioception

    def _format_single_obs(self, raw_obs):
        formatted = {'images': self._format_single_images(raw_obs)}
        if self.use_proprioception:
            formatted['proprioceptions'] = self._format_single_proprioception(raw_obs)
        return formatted

    def _get_obs(self):
        history = list(self.obs_buffer)
        stacked_images = np.stack([item['images'] for item in history], axis=0)
        images = np.transpose(stacked_images, (1, 2, 0, 3, 4))
        if not self.use_proprioception:
            return images
        return {
            'images': images,
            'proprioceptions': np.stack(
                [item['proprioceptions'] for item in history],
                axis=0,
            ),
        }

    def _is_success(self):
        success = self.env.is_success()
        return bool(success.get('task', False)) if isinstance(success, dict) else bool(success)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))

    def reset(self, seed=None, **kwargs):
        del kwargs
        self.seed(seed)
        self.obs_buffer.clear()
        if self.record:
            self.video_buffer.clear()
        raw_obs = self.env.reset()
        formatted = self._format_single_obs(raw_obs)
        for _ in range(self.obs_horizon):
            self.obs_buffer.append(formatted.copy())
        self._elapsed_steps = 0
        return self._get_obs(), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] != self.action_space.shape[0]:
            raise ValueError(
                f'Expected actions with shape [H, {self.action_space.shape[0]}], got {actions.shape}.'
            )

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for action in actions:
            raw_obs, reward, env_done, info = self.env.step(action)
            self._elapsed_steps += 1
            total_reward += float(reward)
            self.obs_buffer.append(self._format_single_obs(raw_obs))
            if self.record:
                self.video_buffer.append(self.render())

            success = self._is_success()
            info = dict(info)
            info['success'] = success
            terminated = success
            truncated = bool(
                not terminated
                and (env_done or self._elapsed_steps >= self.max_episode_length)
            )
            if terminated or truncated:
                break

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        return self.env.render(
            mode='rgb_array',
            width=self.render_size[0],
            height=self.render_size[1],
            camera_name=self.video_camera,
        )

    def close(self):
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()
            return
        inner_env = getattr(self.env, 'env', None)
        close = getattr(inner_env, 'close', None)
        if callable(close):
            close()

class LIBEROEnvWrapper(gym.Env):
    """Standard Gym Environment Wrapper for LIBERO"""
    def __init__(
        self,
        env,
        obs_horizon=1,
        record=False,
        render_size=(128, 128),
        flip_rgb=True,
        use_proprioception=False,
        official_initial_states=None,
        num_steps_wait=0,
        video_camera='agentview',
    ):
        self.env = env
        self.obs_horizon = obs_horizon
        self.obs_buffer = deque(maxlen=obs_horizon)
        self.record = record
        self.render_size = render_size
        self.flip_rgb = flip_rgb
        self.video_camera = str(video_camera).strip()
        self.use_proprioception = bool(use_proprioception)
        self.official_initial_states = official_initial_states
        self.num_steps_wait = int(num_steps_wait)
        if self.num_steps_wait < 0:
            raise ValueError(f'num_steps_wait must be non-negative, got {num_steps_wait}.')
        if self.video_camera not in {'agentview', 'frontview', 'robot0_eye_in_hand'}:
            raise ValueError(
                'video_camera must be `agentview`, `frontview`, or '
                f'`robot0_eye_in_hand`, got {video_camera!r}.'
            )
        if self.record:
            self.video_buffer = []

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(2, 3, self.obs_horizon, 128, 128),
            dtype=np.uint8,
        )
        if self.use_proprioception:
            self.observation_space = gym.spaces.Dict({
                'images': image_space,
                'proprioceptions': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_horizon, LIBERO_PROPRIO_DIM),
                    dtype=np.float32,
                ),
            })
        else:
            self.observation_space = image_space
        
        # Determine observation spaces theoretically if needed
        # We assume action space / obs space exist or are implicit
        
    def _is_success(self):
        return self.env.check_success()
        
    def _format_single_obs(self, sim_obs):
        agentview = sim_obs["agentview_image"]
        eye_in_hand = sim_obs["robot0_eye_in_hand_image"]
        
        if self.flip_rgb:
            agentview = agentview[::-1]
            eye_in_hand = eye_in_hand[::-1]
            
        # The single env returns (H, W, C); convert to (V, C, H, W).
        if len(agentview.shape) == 3:
            agentview_t = np.transpose(agentview, (2, 0, 1))
            eye_in_hand_t = np.transpose(eye_in_hand, (2, 0, 1))

        images = np.stack([agentview_t, eye_in_hand_t], axis=0)  # (2, 3, H, W)
        formatted = {'images': images}
        if self.use_proprioception:
            formatted['proprioceptions'] = libero_rollout_proprioception(sim_obs)
        return formatted

    def _get_obs(self):
        history = list(self.obs_buffer)
        stacked = np.stack([item['images'] for item in history], axis=0)
        images = np.transpose(stacked, (1, 2, 0, 3, 4))  # (2, 3, T, H, W)
        if not self.use_proprioception:
            return images
        return {
            'images': images,
            'proprioceptions': np.stack(
                [item['proprioceptions'] for item in history],
                axis=0,
            ),
        }

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, **kwargs):
        self.obs_buffer.clear()
        if self.record:
            self.video_buffer.clear()
            
        if seed is not None:
             self.seed(seed)
        obs = self.env.reset()
        if self.official_initial_states is not None:
            episode_index = 0 if seed is None else int(seed)
            initial_state = self.official_initial_states[
                episode_index % len(self.official_initial_states)
            ]
            obs = self.env.set_init_state(initial_state)
            dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
            for _ in range(self.num_steps_wait):
                obs, _, _, _ = self.env.step(dummy_action)
        formatted_obs = self._format_single_obs(obs)
        for _ in range(self.obs_horizon):
            self.obs_buffer.append({key: value.copy() for key, value in formatted_obs.items()})
        return self._get_obs(), {}

    def step(self, actions):
        # We allow sending a sequence of actions or a single action
        if actions.ndim == 1:
            actions = [actions]

        total_reward = 0
        for action in actions:
            # We step exactly once in standard offline RL formulation
            # NOTE: if libero returns 4 things or 5 depends on the version. We assume it returns 4 things (obs, reward, done, info) standard
            obs, reward, done, info = self.env.step(action)
            
            if self.record:
                self.video_buffer.append(self.render())
            formatted_obs = self._format_single_obs(obs)
            self.obs_buffer.append(formatted_obs)
                
            success = self._is_success()
            info["success"] = success
            
            terminated = done or success
            
            # Libero wrapper overwrites 'done' to be success, ignoring robosuite horizon limit.
            # We retrieve the actual timestep limit 'done' directly from the underlying env.
            real_done = getattr(self.env.env, "done", False)
            truncated = bool(real_done and not terminated)

            total_reward += reward

            if terminated or truncated:
                break

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        img = self.env.sim.render(
            height=self.render_size[1],
            width=self.render_size[0],
            camera_name=self.video_camera,
        )
        return img[::-1]

    def close(self):
        self.env.close()
