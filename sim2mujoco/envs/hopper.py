import mujoco
import torch
import numpy as np
import gymnasium as gym


class HopperMujocoEnv:
    def __init__(self, render=True, num_envs=1, **kwargs):
        # Make the env
        # Disable termination from angle limits, because we trained without that
        render_mode = 'human' if render else None
        self.env = gym.make('Hopper-v5', render_mode=render_mode, healthy_angle_range=[-np.inf, np.inf])

        self.num_envs = num_envs
        self.num_actions = 3
        self.num_obs = 11

        self.action_scale = 200.0
        self.episode_length = 1000

        self.device = kwargs.get("device", "cuda:0")
        self.alg = kwargs.get("alg", "shac")
        print(self.alg)

    def step(self, action):
        # Step the environment with the given action
        action = torch.clip(action, -1.0, 1.0)
        action = action.flatten().detach().cpu().numpy() * self.action_scale
        # print("Action:", action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # print(obs, reward, terminated, truncated, info)

        obs = torch.tensor(obs, dtype=torch.float32).view(self.num_envs, self.num_obs).to(self.device)
        obs = self._correct_obs(obs)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        if self.alg == "shac":
            reward = -reward
            done = torch.tensor([done], dtype=torch.bool).view(self.num_envs).to(self.device)
        else:
            done = torch.tensor(done, dtype=torch.bool).view(self.num_envs, 1).to(self.device)

        # print(done)
        termination = torch.tensor(terminated, dtype=torch.bool).view(self.num_envs, 1).to(self.device)
        truncation = torch.tensor(truncated, dtype=torch.bool).view(self.num_envs, 1).to(self.device)
        info = {
            "obs_before_reset": obs.clone(),
            "termination": termination,
            "truncation": truncation,
        }

        # Manual reset for shac
        if self.alg == "shac":
            if done.any():
                obs = self.reset()
        return obs, reward, done, info

    def reset(self, force_reset=False):
        # Reset the environment
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).view(self.num_envs, self.num_obs).to(self.device)
        obs = self._correct_obs(obs)
        return obs

    def _correct_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs[:, 0] = obs[:, 0] - 1.25
        return obs

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

