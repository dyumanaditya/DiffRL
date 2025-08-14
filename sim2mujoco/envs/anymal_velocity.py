import os
import mujoco
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple


class AnymalVelocityMujocoEnv:
    """
    Anymal velocity tracking environment using Mujoco
    Mirrors the DFlex AnymalVelocityEnv structure
    """

    def __init__(
        self,
        render=True,
        num_envs=1,
        episode_length=1000,
        stochastic_init=False,
        early_termination=True,
        termination_height=0.28,
        action_penalty=-0.005,
        up_rew_scale=0.1,
        heading_rew_scale=1.0,
        heigh_rew_scale=1.0,
        **kwargs
    ):
        self.num_envs = num_envs
        self.num_actions = 12
        self.num_obs = 49
        self.episode_length = episode_length
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.termination_height = termination_height
        
        # Device setup
        self.device = kwargs.get("device", "cuda:0")
        self.alg = kwargs.get("alg", "shac")
        
        # Action scaling
        self.action_scale = 1.0
        
        # Velocity tracking parameters
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._previous_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        
        # Reward scales (matching DFlex)
        self.lin_vel_reward_scale = 1.0
        self.yaw_rate_reward_scale = 0.5
        self.z_vel_reward_scale = -2.0
        self.ang_vel_reward_scale = -0.05
        self.joint_torque_reward_scale = -2.5e-5
        self.joint_accel_reward_scale = -2.5e-7
        self.action_rate_reward_scale = -0.01
        self.feet_air_time_reward_scale = 0.5
        self.undesired_contact_reward_scale = -4.0
        self.flat_orientation_reward_scale = -5.0
        
        # Default joint positions (matching DFlex)
        self.default_joint_pos = torch.tensor([
            0.03,  # LF_HAA
            0.4,   # LF_HFE
            -0.8,  # LF_KFE
            -0.03, # RF_HAA
            0.4,   # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,   # LH_KFE
            -0.03, # RH_HAA
            -0.4,  # RH_HFE
            0.8,   # RH_KFE
        ], device=self.device)
        
        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        
        # Contact tracking for feet air time calculation
        self._feet_contact_history = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._feet_air_time_counter = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self._current_step = 0
        
        # Feet contact links (matching DFlex)
        self.feet_contact_links = [46, 56, 66, 76]  # LF, RF, LH, RH
        
        # Initialize Mujoco
        self._init_mujoco()
        
        # Sample initial commands
        self.sample_commands(torch.arange(self.num_envs, device=self.device))
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

    def _init_mujoco(self):
        """Initialize Mujoco model and data"""
        # Load the XML file
        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'anybotics_anymal_c', 'anymal_c.xml')
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set initial state
        self._set_initial_state()
        
        # Set up renderer if needed
        if hasattr(self, 'render') and self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _set_initial_state(self):
        """Set initial state for the robot"""
        # Set initial position (floating base)
        self.data.qpos[:3] = [0.0, 0.65, 0.0]  # x, y, z
        
        # Set initial orientation (quaternion)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        
        # Set initial joint positions
        self.data.qpos[7:19] = self.default_joint_pos.cpu().numpy()
        
        # Set initial velocities to zero
        self.data.qvel[:] = 0.0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def sample_commands(self, env_ids):
        """Sample new velocity commands for specified environments"""
        # For now, use constant forward velocity (matching DFlex)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]) + torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        )

    def step(self, action):
        """Step the environment with the given action"""
        # Store previous actions for reward calculation
        self._previous_actions = self._actions.clone()
        
        # Process action
        action = torch.clip(action, -1.0, 1.0)
        self._actions = action.clone()
        
        # Convert action to target joint positions
        target_joint_positions = action + self.default_joint_pos
        
        # Apply action to Mujoco
        self._apply_action(target_joint_positions)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        
        # Check termination
        done = self._check_termination(obs)
        
        # Update step counter
        self.step_count += 1
        
        # Handle episode reset
        if done.any():
            obs = self.reset()
        
        # Prepare info dict
        info = {
            "obs_before_reset": obs.clone(),
            "termination": done.clone(),
            "truncation": torch.zeros_like(done),
        }
        
        return obs, reward, done, info

    def _apply_action(self, target_joint_positions):
        """Apply action to Mujoco model"""
        # Set joint targets (skip first 7 DoFs which are floating base)
        self.data.ctrl[:12] = target_joint_positions.cpu().numpy()

    def _get_observation(self):
        """Get observation from Mujoco state"""
        # Extract state information
        torso_pos = torch.tensor(self.data.qpos[:3], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        torso_rot = torch.tensor(self.data.qpos[3:7], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        lin_vel = torch.tensor(self.data.qvel[3:6], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ang_vel = torch.tensor(self.data.qvel[:3], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        joint_pos = torch.tensor(self.data.qpos[7:19], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        joint_vel = torch.tensor(self.data.qvel[6:18], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Torso height
        torso_height = torso_pos[:, 1].clone()
        
        # Transform velocities to body frame (simplified)
        # In a full implementation, you'd need proper quaternion operations
        lin_vel_b = lin_vel.clone()
        ang_vel_b = ang_vel.clone()
        
        # Projected gravity in body frame (simplified)
        projected_gravity_b = torch.tensor([0.0, -1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Construct observation
        obs = torch.cat([
            lin_vel_b,                           # 0:3
            ang_vel_b,                           # 3:6
            projected_gravity_b,                 # 6:9
            self._commands.clone(),              # 9:12
            joint_pos - self.default_joint_pos,  # 12:24
            joint_vel,                           # 24:36
            self._actions.clone(),               # 36:48
            torso_height.unsqueeze(1),           # 48:49
        ], dim=-1)
        
        return obs

    def _calculate_reward(self, obs, action):
        """Calculate reward based on observation and action"""
        # Extract components from observation
        lin_vel_b = obs[:, :3]
        ang_vel_b = obs[:, 3:6]
        projected_gravity_b = obs[:, 6:9]
        commands = obs[:, 9:12]
        joint_pos = obs[:, 12:24]
        joint_vel = obs[:, 24:36]
        actions = obs[:, 36:48]
        
        # Linear velocity tracking
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - lin_vel_b[:, [0, 2]]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # Yaw rate tracking
        yaw_rate_error = torch.square(commands[:, 2] - ang_vel_b[:, 1])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # Z velocity tracking
        z_vel_error = torch.square(lin_vel_b[:, 1])
        
        # Angular velocity x/z
        ang_vel_error = torch.sum(torch.square(ang_vel_b[:, [0, 2]]), dim=1)
        
        # Action rate
        action_rate = torch.sum(torch.square(actions - self._previous_actions), dim=1)
        
        # Feet air time (simplified - would need contact detection)
        feet_air_time = torch.zeros(self.num_envs, device=self.device)
        
        # Undesired contacts (simplified)
        undesired_contacts = torch.zeros(self.num_envs, device=self.device)
        
        # Flat orientation
        flat_orientation = torch.sum(torch.square(projected_gravity_b[:, [0, 2]]), dim=1)
        
        # Calculate rewards
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.yaw_rate_reward_scale,
            "lin_vel_z_l2": z_vel_error * self.z_vel_reward_scale,
            "ang_vel_xy_l2": ang_vel_error * self.ang_vel_reward_scale,
            "action_rate_l2": action_rate * self.action_rate_reward_scale,
            "feet_air_time": feet_air_time * self.feet_air_time_reward_scale,
            "undesired_contacts": undesired_contacts * self.undesired_contact_reward_scale,
            "flat_orientation_l2": flat_orientation * self.flat_orientation_reward_scale,
        }
        
        # Total reward
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Log episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _check_termination(self, obs):
        """Check if episode should terminate"""
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check for NaN values
        nonfinite_mask = ~(torch.isfinite(obs).sum(-1) > 0)
        termination = termination | nonfinite_mask
        
        # Check height termination
        if self.early_termination:
            torso_height = obs[:, 48]
            termination = termination | (torso_height < self.termination_height)
            
        # Check episode length
        if self.step_count >= self.episode_length:
            termination[:] = True
            
        return termination

    def reset(self, force_reset=False):
        """Reset the environment"""
        # Reset Mujoco state
        self._set_initial_state()
        
        # Reset episode statistics
        for key in self._episode_sums.keys():
            self._episode_sums[key][:] = 0.0
            
        # Reset contact tracking
        self._feet_contact_history[:] = False
        self._feet_air_time_counter[:] = 0.0
        
        # Reset step counter
        self.step_count = 0
        
        # Sample new commands
        self.sample_commands(torch.arange(self.num_envs, device=self.device))
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs

    def set_velocity_commands(self, commands):
        """Manually set velocity commands for all environments"""
        if commands.shape == (3,):
            self._commands[:] = commands
        elif commands.shape == (self.num_envs, 3):
            self._commands[:] = commands
        else:
            raise ValueError(f"Commands must have shape (3,) or ({self.num_envs}, 3), got {commands.shape}")

    def get_velocity_commands(self):
        """Get current velocity commands"""
        return self._commands.clone()

    @property
    def action_space(self):
        """Return action space"""
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

    @property
    def observation_space(self):
        """Return observation space"""
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
