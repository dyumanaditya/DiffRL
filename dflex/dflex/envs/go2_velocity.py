# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys
import time
import urchin
from gym import spaces

import torch

from .dflex_env import DFlexEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable logging.")


class Go2VelocityEnv(DFlexEnv):
    """
    Go2 velocity tracking environment
    """

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=64,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=16,
        early_termination=True,
        jacobian=False,
        logdir=None,
        nan_state_fix=False,
        jacobian_norm=None,
        termination_height=0.1,
        action_penalty=-0.005,
        up_rew_scale=0.1,
        heading_rew_scale=1.0,
        heigh_rew_scale=1.0,
        wandb=False,
        observation_noise=False,
        base_lin_vel_obs_noise=[-0.1, 0.1],
        base_ang_vel_obs_noise=[-0.2, 0.2],
        projected_gravity_obs_noise=[-0.05, 0.05],
        joint_pos_obs_noise=[-0.01, 0.01],
        joint_vel_obs_noise=[-1.5, 1.5],
        mass_randomization=False,
        mass_randomization_bodies=[0],
        mass_randomization_noise=[-0.5, 0.5],
        com_randomization=False,
        com_randomization_bodies=[0],
        com_randomization_noise=[-0.05, 0.05],
        **kwargs
    ):
        num_obs = 49
        num_act = 12
        self.playback_speed = 1/16

        super(Go2VelocityEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            no_grad,
            render,
            nan_state_fix,
            jacobian_norm,
            stochastic_init,
            jacobian,
            device,
            **kwargs
        )

        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -3.0, np.ones(self.num_actions) * 3.0
        )

        self.early_termination = early_termination
        self.wandb_enabled = wandb and WANDB_AVAILABLE

        # Observation noise parameters
        self.observation_noise = observation_noise
        self.base_lin_vel_obs_noise = base_lin_vel_obs_noise
        self.base_ang_vel_obs_noise = base_ang_vel_obs_noise
        self.projected_gravity_obs_noise = projected_gravity_obs_noise
        self.joint_pos_obs_noise = joint_pos_obs_noise
        self.joint_vel_obs_noise = joint_vel_obs_noise
        
        # Mass and COM randomization parameters
        self.mass_randomization = mass_randomization
        self.mass_randomization_bodies = mass_randomization_bodies
        self.mass_randomization_noise = mass_randomization_noise
        self.com_randomization = com_randomization
        self.com_randomization_bodies = com_randomization_bodies
        self.com_randomization_noise = com_randomization_noise

        self.contact_ke = kwargs["contact"]["ke"]
        self.contact_kd = kwargs["contact"]["kd"]
        self.contact_kf = kwargs["contact"]["kf"]
        self.contact_mu = kwargs["contact"]["mu"]

        self.init_sim()

        # MDP parameters
        self.action_scale = 0.5

        # Velocity tracking parameters
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._previous_actions = torch.zeros(self.num_envs, num_act, device=self.device)
        self._actions = torch.zeros(self.num_envs, num_act, device=self.device)

        # Torso height for early termination
        self.torso_height = torch.zeros(self.num_envs, device=self.device) + 0.35  # Initial height
        
        # Rewards: Following IsaacLab reward structure
        self.lin_vel_reward_scale = 5.0
        # self.lin_vel_reward_scale = 1.0
        self.yaw_rate_reward_scale = 2.0
        # self.yaw_rate_reward_scale = 0.5
        self.z_vel_reward_scale = -0.5
        self.ang_vel_reward_scale = -0.05
        self.joint_torque_reward_scale = -1e-5
        self.joint_accel_reward_scale = -2.5e-7
        self.action_rate_reward_scale = -0.01
        self.feet_air_time_reward_scale = 0.25
        self.undesired_contact_reward_scale = -4.0
        self.flat_orientation_reward_scale = -2.0

        # Early termination parameters
        self.termination_height = termination_height

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
        self._feet_contact_history = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)  # 4 feet
        self._feet_air_time_counter = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)  # Time each foot has been in air
        self._feet_last_contact_step = torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device)  # Last step each foot was in contact
        self._last_air_time = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)  # Air time when feet last made contact
        self._current_step = 0

        # Determined from print_model_info
        # These will be in contact with the ground
        self.feet_contact_links = [11, 20, 29, 38]  # FL, FR, RL, RR

        # Get all body indices (excluding feet)
        # link_cnt = self.model.link_count // self.num_envs
        # all_body_indices = list(range(link_cnt))  # 0 to num_links-1
        # self.undesired_contact_body_links = [idx for idx in all_body_indices if idx not in self.feet_contact_links]
        self.undesired_contact_body_links = [5, 14, 23, 32, 7, 16, 25, 34]     # Thigh, calf links
        self.undesired_contact_body_links.extend([1, 2, 40, 41])               # Head and camera links

        self.setup_visualizer(logdir)
        # self.print_model_info()
        # Initialize wandb if enabled
        if self.wandb_enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize wandb for logging"""
        try:
            wandb.init(
                project="go2-velocity-env",
                name="go2-velocity-training",
                config={
                    "num_envs": self.num_envs,
                    "episode_length": self.episode_length,
                    "lin_vel_reward_scale": self.lin_vel_reward_scale,
                    "yaw_rate_reward_scale": self.yaw_rate_reward_scale,
                    "z_vel_reward_scale": self.z_vel_reward_scale,
                    "ang_vel_reward_scale": self.ang_vel_reward_scale,
                    "joint_torque_reward_scale": self.joint_torque_reward_scale,
                    "joint_accel_reward_scale": self.joint_accel_reward_scale,
                    "action_rate_reward_scale": self.action_rate_reward_scale,
                    "feet_air_time_reward_scale": self.feet_air_time_reward_scale,
                    "undesired_contact_reward_scale": self.undesired_contact_reward_scale,
                    "flat_orientation_reward_scale": self.flat_orientation_reward_scale,
                    "termination_height": self.termination_height,
                    "action_scale": self.action_scale,
                    "observation_noise": self.observation_noise,
                    "base_lin_vel_obs_noise": list(self.base_lin_vel_obs_noise),
                    "base_ang_vel_obs_noise": list(self.base_ang_vel_obs_noise),
                    "projected_gravity_obs_noise": list(self.projected_gravity_obs_noise),
                    "joint_pos_obs_noise": list(self.joint_pos_obs_noise),
                    "joint_vel_obs_noise": list(self.joint_vel_obs_noise),
                    "mass_randomization": self.mass_randomization,
                    "mass_randomization_bodies": list(self.mass_randomization_bodies),
                    "mass_randomization_noise": list(self.mass_randomization_noise),
                    "com_randomization": self.com_randomization,
                    "com_randomization_bodies": list(self.com_randomization_bodies),
                    "com_randomization_noise": list(self.com_randomization_noise),
                }
            )
            print("Wandb initialized successfully for Go2 velocity environment")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            self.wandb_enabled = False

    def sample_commands(self, env_ids):
        """Sample new velocity commands for specified environments"""
        # Sample linear velocity commands (x, z) and angular velocity (y)
        # Following IsaacLab pattern: uniform distribution between -1 and 1
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-0.5, 0.5)
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]) + torch.tensor(
        #     [0.5, 0.0, 0.0], device=self.device, dtype=torch.float32
        # )

        # Optionally scale commands to more realistic ranges
        # self._commands[env_ids, :2] *= 2.0  # Scale linear velocities
        # self._commands[env_ids, 2] *= 0.5   # Scale angular velocity

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 12 + 7  # joints + root pose
        self.num_joint_qd = 12 + 6  # joints + root velocity

        self.x_unit_tensor = tu.to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)
        self.start_rotation = tu.to_torch(
            self.start_rot, device=self.device, requires_grad=False
        )

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.start_pos = []
        self.start_joint_q = [
            -0.1,  # RR_hip_joint
            1.0,  # RR_thigh_joint
            -1.5,  # RR_calf_joint
            0.1,  # RL_hip_joint
            1.0,  # RL_thigh_joint
            -1.5,  # RL_calf_joint
            -0.1,  # FR_hip_joint
            0.8,  # FR_thigh_joint
            -1.5,  # FR_calf_joint
            0.1,  # FL_hip_joint
            0.8,  # FL_thigh_joint
            -1.5,  # FL_calf_joint
        ]
        self.default_joint_pos = torch.tensor(self.start_joint_q, device=self.device)
        # Taken from IsaacGym
        # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Anymal.yaml

        self.joint_limits = [
            (-1.0472, 1.0472),
            (-0.5236, 4.5379),
            (-2.7227, -0.83776),
            (-1.0472, 1.0472),
            (-0.5236, 4.5379),
            (-2.7227, -0.83776),
            (-1.0472, 1.0472),
            (-1.5708, 3.4907),
            (-2.7227, -0.83776),
            (-1.0472, 1.0472),
            (-1.5708, 3.4907),
            (-2.7227, -0.83776)
        ]
        self.soft_limit_ratio = float(1.0)

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0.0  # set to zero for training for numerical consistency

        self.env_dist = 0.0  # set to zero for training for numerical consistency

        start_height = 0.35

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        filename = "go2/urdf/go2.urdf"
        now = time.time()
        # load URDF here for faster env initialisation when running many paralle envs
        robot = urchin.URDF.load(os.path.join(asset_folder, filename), lazy_load_meshes=False)
        for i in range(self.num_environments):
            start_pos = (0.0, start_height, 0.0 + self.env_dist * i)
            lu.urdf_load(
                self.builder,
                os.path.join(asset_folder, filename),
                df.transform(
                    start_pos,
                    self.start_rot,
                ),
                robot=robot,
                floating=True,
                stiffness=25.0,  # from config
                damping=0.2,  # from config
                # stiffness=1000.0,  # from config
                # damping=10.0,  # from config
                shape_ke=self.contact_ke,
                shape_kd=self.contact_kd,
                shape_kf=self.contact_kf,
                shape_mu=self.contact_mu,
                limit_ke=1.0e3,
                limit_kd=1.0e1,
                # limit_ke=0.0,
                # limit_kd=0.0,
                armature=0.006,
            )
            self.start_pos.append(start_pos)

        total = time.time() - now
        print(f"Loading took {total:.2f}s")

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor(
            (0.0, -9.81, 0.0), dtype=torch.float32, device=self.device
        )

        # Set ground contact
        self.model.contact_ke = self.contact_ke
        self.model.contact_kd = self.contact_kd
        self.model.contact_kf = self.contact_kf
        self.model.contact_mu = self.contact_mu

        self._set_domain_randomization()

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if self.model.ground:
            self.model.collide(self.state)

        # Set torque limits for all joints and free body
        # 18 values per environment: 6 for free body + 12 for joints
        self.torque_limits = [
            # Free body (6 values: 3 angular + 3 linear)
            -10000.0, -10000.0, -10000.0,  # Angular torque limits (x, y, z) - more reasonable values
            -10000.0, -10000.0, -10000.0,  # Linear force limits (x, y, z) - more reasonable values

            # Joint torque limits (12 values for the 12 joints)
            -20.0, -20.0, -20.0,  # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            -20.0, -20.0, -20.0,  # RL_hip_joint, RL_thigh_joint, RL_calf_joint
            -20.0, -20.0, -20.0,  # FR_hip_joint, FR_thigh_joint, FR_calf_joint
            -20.0, -20.0, -20.0,  # FL_hip_joint, FL_thigh_joint, FL_calf_joint
        ]

        # Apply torque limits to all environments
        # The model expects torque limits for ALL joints (including free body)
        # Total joints per env: 1 (free body) + 12 (actuated joints) = 13 joints
        # But free body has 6 DOFs, so total DOFs per env: 6 + 12 = 18

        # Create the full torque limit tensor for all environments
        # Shape: (num_envs * 18) flattened
        # Each environment gets its own set of 18 torque limits
        torque_limits_tensor = torch.tensor(
            self.torque_limits,
            device=self.device,
            dtype=torch.float32
        ).repeat(self.num_envs, 1).flatten()

        # Set the torque limits in the model
        self.model.joint_torque_limit_lower = torque_limits_tensor
        self.model.joint_torque_limit_upper = -torque_limits_tensor  # Upper limits are negative of lower limits

        print(f"Applied torque limits to {self.num_envs} environments:")
        print(f"  Free body limits: {self.torque_limits[:6]}")
        print(f"  Joint limits: {self.torque_limits[6:]}")
        print(f"  Total limits per env: {len(self.torque_limits)}")
        print(f"  Model tensor shape: {self.model.joint_torque_limit_lower.shape}")
        print(f"  Expected shape: ({self.num_envs * 18},)")

        # Debug: Print the first few torque limits to verify they're set correctly
        print(f"  First 18 torque limits (env 0): {self.model.joint_torque_limit_lower[:18]}")
        print(f"  Last 18 torque limits (env {self.num_envs-1}): {self.model.joint_torque_limit_lower[-18:]}")

    def step(self, actions, play=False):
        # Debug: Check for NaNs before simulation
        if hasattr(self, 'state') and self.state is not None:
            joint_q_nan = torch.isnan(self.state.joint_q).any()
            joint_qd_nan = torch.isnan(self.state.joint_qd).any()
            if joint_q_nan or joint_qd_nan:
                print(f"WARNING: NaNs detected in state before step:")
                print(f"  joint_q has NaNs: {joint_q_nan}")
                print(f"  joint_qd has NaNs: {joint_qd_nan}")
                if joint_q_nan:
                    print(f"  joint_q NaN locations: {torch.isnan(self.state.joint_q).nonzero()}")
                if joint_qd_nan:
                    print(f"  joint_qd NaN locations: {torch.isnan(self.state.joint_qd).nonzero()}")
        
        obs, rew, done, extras = super().step(actions, play)
        
        # Debug: Check for NaNs after simulation
        if obs is not None:
            obs_nan = torch.isnan(obs).any()
            if obs_nan:
                print(f"ERROR: NaNs detected in observations after step!")
                print(f"  obs NaN locations: {torch.isnan(obs).nonzero()}")
                print(f"  obs shape: {obs.shape}")
                
                # Check which observation components have NaNs
                obs_components = [
                    "lin_vel_b", "ang_vel_b", "projected_gravity_b", 
                    "commands", "joint_pos", "joint_vel", "actions"
                ]
                for i, name in enumerate(obs_components):
                    start_idx = i * 12
                    end_idx = min((i + 1) * 12, obs.shape[1])
                    if start_idx < obs.shape[1]:
                        component_nan = torch.isnan(obs[:, start_idx:end_idx]).any()
                        if component_nan:
                            print(f"    {name} (indices {start_idx}:{end_idx}) has NaNs")
        
        # Increment step counter for wandb logging
        self._current_step += 1
        
        # Log step information to wandb if enabled
        if self.wandb_enabled:
            try:
                wandb.log({
                    "env/step": self._current_step,
                    "env/episode_progress": (self._current_step % self.episode_length) / self.episode_length
                }, step=self._current_step)
            except Exception as e:
                print(f"Failed to log step info to wandb: {e}")
                self.wandb_enabled = False
        
        # print("Commands", self._commands)
        return obs, rew, done, extras

    def unscale_act(self, action):
        action = torch.clamp(action, -10, 10)
        return action * self.action_scale

    def set_act(self, action):
        # action = torch.clamp(action, -100, 100)
        self._actions = action.clone() / self.action_scale  # Because we scaled already earlier
        # Convert action to target joint positions
        target_joint_positions = action + self.default_joint_pos

        # # TODO: clamp target_joint_positions to joint limits
        # # --- Soft-limit clamp (ratio in (0, 1]; 1.0 = full limits, 0.95 = keep 5% margin) ---
        # # Pull per-env limits for the actuated joints (skip the 7 DoFs of the floating base)
        # jcount = target_joint_positions.shape[1]  # expected 12
        # lower = self.model.joint_limit_lower.view(self.num_envs, -1)[:, 7:7 + jcount]
        # upper = self.model.joint_limit_upper.view(self.num_envs, -1)[:, 7:7 + jcount]
        #
        # # Compute inner (soft) bounds centered in the range
        # soft = self.soft_limit_ratio
        # # inner = [lower + m, upper - m] with m = (1 - soft) * half_range
        # half_range = 0.5 * (upper - lower)
        # margin = (1.0 - soft) * half_range
        # inner_lower = lower + margin
        # inner_upper = upper - margin
        #
        # # If any joint has no finite limits, leave it unchanged
        # finite = torch.isfinite(inner_lower) & torch.isfinite(inner_upper)
        # # Clamp into the soft range (torch.clamp supports tensor min/max)
        # clamped = torch.where(
        #     finite,
        #     torch.clamp(target_joint_positions, min=inner_lower, max=inner_upper),
        #     target_joint_positions,
        # )
        #
        # target_joint_positions = clamped

        # print("TARGETS")
        # print(target_joint_positions)
        # print()

        # Set joint targets on the model (shape num_envs * 7+12)
        self.model.joint_target.view(self.num_envs, -1)[:, 7:] = target_joint_positions

        # Clear joint actuation
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = 0.0

    def compute_termination(self, obs, act):
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # an ugly fix for simulation nan values
        joint_q = self.state.joint_q.view(self.num_environments, -1)
        joint_qd = self.state.joint_qd.view(self.num_environments, -1)

        nonfinite_mask = ~(torch.isfinite(obs).sum(-1) > 0)
        nonfinite_mask = nonfinite_mask | ~(torch.isfinite(joint_q).sum(-1) > 0)
        nonfinite_mask = nonfinite_mask | ~(torch.isfinite(joint_qd).sum(-1) > 0)

        invalid_value_mask = (torch.abs(joint_q) > 1e6).sum(-1) > 0
        invalid_value_mask = (
            invalid_value_mask | (torch.abs(joint_qd) > 1e6).sum(-1) > 0
        )

        termination = termination | nonfinite_mask | invalid_value_mask

        if self.early_termination:
            termination = termination | (self.torso_height < self.termination_height)
            
        # Log termination statistics to wandb if enabled
        if self.wandb_enabled and termination.any():
            try:
                termination_reasons = {
                    "termination/nonfinite_obs": nonfinite_mask.sum().item(),
                    "termination/invalid_joint_q": (torch.abs(joint_q) > 1e6).sum().item(),
                    "termination/invalid_joint_qd": (torch.abs(joint_qd) > 1e6).sum().item(),
                    "termination/height": (self.torso_height < self.termination_height).sum().item() if self.early_termination else 0,
                    "termination/total": termination.sum().item()
                }
                wandb.log(termination_reasons, step=self._current_step)
            except Exception as e:
                print(f"Failed to log termination info to wandb: {e}")
                self.wandb_enabled = False
                
        return termination

    def static_init_func(self, env_ids):
        xyz = self.start_pos[env_ids]
        quat = self.start_rotation.repeat(len(env_ids), 1)
        joints = self.start_joint_q.repeat(len(env_ids), 1)
        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = torch.zeros((len(env_ids), self.num_joint_qd), device=self.device)
        
        # Sample new velocity commands for these environments
        self.sample_commands(env_ids)
        
        # Reset contact tracking for these environments
        self.reset_contact_tracking(env_ids)

        # Reset stats
        self.reset_episode_stats(env_ids)
        
        # Reset noise debug counter for these environments
        if hasattr(self, '_noise_debug_counter'):
            self._noise_debug_counter = 0
        
        # Apply mass and COM randomization
        self._apply_mass_randomization(env_ids)
        self._apply_com_randomization(env_ids)
        
        return joint_q, joint_qd

    def stochastic_init_func(self, env_ids):
        """Method for computing stochastic init state"""
        xyz = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3]
            + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.0
        )
        angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
        axis = torch.nn.functional.normalize(
            torch.rand((len(env_ids), 3), device=self.device) - 0.5
        )
        quat = tu.quat_mul(
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7],
            tu.quat_from_angle_axis(angle, axis),
        )

        joints = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:]
            + 0.2
            * (
                torch.rand(
                    size=(len(env_ids), self.num_joint_q - 7),
                    device=self.device,
                )
                - 0.5
            )
            * 2.0
        )

        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = 0.5 * (
            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5
        )
        
        # Sample new velocity commands for these environments
        self.sample_commands(env_ids)
        
        # Reset contact tracking for these environments
        self.reset_contact_tracking(env_ids)

        # Reset stats
        self.reset_episode_stats(env_ids)
        
        # Reset noise debug counter for these environments
        if hasattr(self, '_noise_debug_counter'):
            self._noise_debug_counter = 0
        
        # Apply mass and COM randomization
        self._apply_mass_randomization(env_ids)
        self._apply_com_randomization(env_ids)
        
        return joint_q, joint_qd

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

    def observation_from_state(self, state):
        # States from simulation
        torso_pos = state.joint_q.view(self.num_envs, -1)[:, 0:3].clone()
        torso_rot = state.joint_q.view(self.num_envs, -1)[:, 3:7].clone()
        lin_vel = state.joint_qd.view(self.num_envs, -1)[:, 3:6].clone()
        ang_vel = state.joint_qd.view(self.num_envs, -1)[:, 0:3].clone()
        joint_pos = state.joint_q.view(self.num_envs, -1)[:, 7:].clone()
        joint_vel = state.joint_qd.view(self.num_envs, -1)[:, 6:].clone()

        # print("JOINT POS OBS")
        # print(joint_pos)
        # print()

        # Torso height
        self.torso_height = torso_pos[:, 1].clone()

        # Velocity in body frame
        # print("Before transformation linear")
        # print(lin_vel)
        # print("Before transformation angular")
        # print(ang_vel)
        # lin_vel_b = lin_vel.clone()
        # ang_vel_b = ang_vel.clone()
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)
        lin_vel_b = tu.quat_rotate_inverse(torso_quat, lin_vel)  # world -> body
        ang_vel_b = tu.quat_rotate_inverse(torso_quat, ang_vel)
        # print("After transformation linear")
        # print(lin_vel_b)
        # print("After transformation angular")
        # print(ang_vel_b)
        # print()

        # Project gravity in body frame
        # Gravity vector in world frame (always points down)
        g_world = -self.y_unit_tensor
        projected_gravity_b = tu.quat_rotate_inverse(torso_quat, g_world)

        # print("OBSERVATIONS")
        # print("lin vel", lin_vel_b)
        # print("ang vel", ang_vel_b)
        # print("projected g", projected_gravity_b)
        # print("commands", self._commands)
        # print("joint pos", joint_pos)
        # print("joint vel", joint_vel)
        # print("joint torques", state.joint_tau.view(self.num_envs, -1)[:, 6:])
        # print("joint accel", state.joint_qdd.view(self.num_envs, -1)[:, 6:])
        # print("actions", self._actions)
        # print("torso height", torso_height)
        # print()

        obs = torch.cat(
            [
                lin_vel_b,                           # 0:3
                ang_vel_b,                           # 3:6
                projected_gravity_b,                 # 6:9
                self._commands.clone(),              # 9:12
                joint_pos - self.default_joint_pos,  # 12:24
                joint_vel,                           # 24:36
                self._actions.clone(),               # 36:48
                # self.torso_height.unsqueeze(1),           # 48:49 (optional, can be used for height tracking)
            ],
            dim=-1,
        )
        
        # Apply observation noise if enabled
        obs = self._apply_observation_noise(obs)

        # ---- Sanity check: NaNs / Infs / Huge values in observations ----
        # Threshold for "huge" values; tweak as needed
        huge_thr = getattr(self, "obs_alert_threshold", 20)
        max_print = 50  # cap how many lines we print

        # Map columns -> human-readable field names (update if you change obs layout)
        _obs_segments = [
            ("lin_vel_b", 0, 3, ("x", "y", "z")),
            ("ang_vel_b", 3, 6, ("x", "y", "z")),
            ("projected_gravity_b", 6, 9, ("x", "y", "z")),
            ("commands", 9, 12, ("vx", "vy", "yaw")),
            ("joint_pos_delta", 12, 24, None),
            ("joint_vel", 24, 36, None),
            ("actions", 36, 48, None),
            # If you add torso height back in, also add: ("torso_height", 48, 49, None),
        ]

        def _col_name(j: int) -> str:
            for name, s, e, axes in _obs_segments:
                if s <= j < e:
                    k = j - s
                    suffix = f"[{axes[k]}]" if axes and k < len(axes) else f"[{k}]"
                    return f"{name}{suffix}"
            return f"col_{j}"

        bad_mask = torch.isnan(obs) | torch.isinf(obs) | (obs.abs() > huge_thr)
        num_bad = int(bad_mask.sum().item())

        if num_bad:
            nan_count = int(torch.isnan(obs).sum().item())
            inf_count = int(torch.isinf(obs).sum().item())
            big_count = int((obs.abs() > huge_thr).sum().item())
            env_idx, col_idx = bad_mask.nonzero(as_tuple=True)

            print(f"[OBS CHECK] Found {num_bad} bad entries "
                  f"(NaN: {nan_count}, Inf: {inf_count}, >{huge_thr}: {big_count}). "
                  f"Showing up to {max_print}:")

            to_show = min(num_bad, max_print)
            for i in range(to_show):
                e = int(env_idx[i])
                j = int(col_idx[i])
                v = obs[e, j].item()  # safe on CPU/GPU
                print(f"  - env={e:04d}, obs_col={j:03d} ({_col_name(j)}), value={v}")
                if _col_name(j) == "lin_vel_b[z]" or _col_name(j) == "lin_vel_b[x]":
                    print("Before transform:", lin_vel[e, :])
                if _col_name(j) == "ang_vel_b[x]" or _col_name(j) == "ang_vel_b[y]" or _col_name(j) == "ang_vel_b[z]":
                    print("Before transform:", ang_vel[e, :])

            # Optional: show the worst offending envs
            per_env_bad = bad_mask.sum(dim=1)
            worst_count, worst_envs = torch.topk(per_env_bad, k=min(5, self.num_envs))
            print("[OBS CHECK] Worst envs by bad count:")
            for rank in range(worst_envs.numel()):
                print(f"  #{rank + 1}: env={int(worst_envs[rank])} -> {int(worst_count[rank].item())} issues")
        # -----------------------------------------------------------------

        # Log observation and action statistics to wandb if enabled
        if self.wandb_enabled:
            try:
                obs_stats = {
                    "obs/lin_vel_mean": lin_vel_b.mean().item(),
                    "obs/lin_vel_std": lin_vel_b.std().item(),
                    "obs/ang_vel_mean": ang_vel_b.mean().item(),
                    "obs/ang_vel_std": ang_vel_b.std().item(),
                    "obs/joint_pos_mean": joint_pos.mean().item(),
                    "obs/joint_pos_std": joint_pos.std().item(),
                    "obs/joint_vel_mean": joint_vel.mean().item(),
                    "obs/joint_vel_std": joint_vel.std().item(),
                    "obs/torso_height_mean": self.torso_height.mean().item(),
                    "obs/torso_height_std": self.torso_height.std().item(),
                    "action/mean": self._actions.mean().item(),
                    "action/std": self._actions.std().item(),
                }
                
                # Add noise statistics if observation noise is enabled
                if self.observation_noise:
                    obs_stats.update({
                        "noise/base_lin_vel_range": list(self.base_lin_vel_obs_noise),
                        "noise/base_ang_vel_range": list(self.base_ang_vel_obs_noise),
                        "noise/projected_gravity_range": list(self.projected_gravity_obs_noise),
                        "noise/joint_pos_range": list(self.joint_pos_obs_noise),
                        "noise/joint_vel_range": list(self.joint_vel_obs_noise),
                        "noise/obs_total_std": obs.std().item(),  # Total std of noisy observations
                    })
                
                wandb.log(obs_stats, step=self._current_step)
            except Exception as e:
                print(f"Failed to log observation/action stats to wandb: {e}")
                self.wandb_enabled = False
        
        return obs

    def feet_air_time(self):
        """Compute feet air time and update contact tracking"""
        # Get contact forces from simulation state
        # contact_f has shape (num_envs * num_links, 6) where 6 = [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        contact_f = self.state.contact_f.view(self.num_envs, -1, 6)

        # Extract contact forces for feet (only force components, not torque)
        feet_contact_forces = contact_f[:, self.feet_contact_links, 3:]  # [:, :, 3:] gets force components

        # Check if any foot has significant contact force (norm > threshold)
        # Use a small threshold to detect contact
        contact_force_threshold = 1.0  # Lower threshold to detect lighter contacts
        feet_contact_norms = torch.norm(feet_contact_forces, dim=-1)  # Norm of force vector for each foot
        feet_contact_current = feet_contact_norms > contact_force_threshold

        # Update air time counters
        # If foot is NOT in contact (force < threshold), increment air time
        # If foot IS in contact (force >= threshold), reset air time
        air_time_delta = self.dt  # Time step for this simulation step
        
        # Feet that are currently in air (not touching ground)
        feet_in_air = ~feet_contact_current  # Negate contact state to get air state

        # Isaac Gym approach: Track air time for feet that are in the air
        # Only increment air time for feet that have touched the ground before
        feet_eligible_for_air_time = feet_in_air & (self._feet_contact_history == True)
        self._feet_air_time_counter[feet_eligible_for_air_time] += air_time_delta
        
        # Detect feet that just made contact (went from air to ground)
        # This is the key insight: reward feet when they make contact based on air time
        feet_just_contacted = feet_contact_current & (~feet_in_air)  # Currently in contact AND was in air previous step
        
        # Store the air time when feet make contact (for reward calculation)
        if feet_just_contacted.any():
            # Get the air time that was accumulated before contact
            air_time_at_contact = self._feet_air_time_counter.clone()
            # Reset air time for feet that just made contact
            self._feet_air_time_counter[feet_just_contacted] = 0.0
            # Store the air time for reward calculation
            self._last_air_time[feet_just_contacted] = air_time_at_contact[feet_just_contacted]
        
        # Reset air time for feet that are currently in contact (touching ground)
        # This ensures clean state for next air time accumulation
        feet_in_contact = feet_contact_current
        self._feet_air_time_counter[feet_in_contact] = 0.0
        
        # Track feet that just made contact for the first time (for contact history)
        feet_first_contact = feet_contact_current & (~self._feet_contact_history)
        
        # Update contact history - only set to True when feet first make contact
        # Don't reset to False when feet go back in air
        # Use logical_or to ensure proper boolean operations
        old_history = self._feet_contact_history.clone()
        self._feet_contact_history = torch.logical_or(self._feet_contact_history, feet_contact_current)
        
        # # Debug: Print contact history changes
        # if torch.any(old_history != self._feet_contact_history):
        #     print(f"Contact history changed from {old_history} to {self._feet_contact_history}")
        #     print(f"Feet that made contact: {feet_contact_current}")
        
        # Update last contact step for feet that just made contact
        self._feet_last_contact_step[feet_first_contact] = self._current_step
        
        return self._feet_air_time_counter.clone()

    def get_contact_stats(self):
        """Get current contact statistics for debugging"""
        stats = {
            'feet_contact_history': self._feet_contact_history.clone(),
            'feet_air_time_counter': self._feet_air_time_counter.clone(),
            'feet_last_contact_step': self._feet_last_contact_step.clone(),
            'current_step': self._current_step
        }
        return stats

    def are_feet_in_contact(self):
        """Check which feet are currently in contact with the ground"""
        # Get contact forces from simulation state
        contact_f = self.state.contact_f.view(self.num_envs, -1, 6)
        
        # Extract contact forces for feet (only force components, not torque)
        feet_contact_forces = contact_f[:, self.feet_contact_links, 3:]  # [:, :, 3:] gets force components
        
        # Check if any foot has significant contact force (norm > threshold)
        contact_force_threshold = 1.0  # Same threshold as in feet_air_time()
        feet_contact_norms = torch.norm(feet_contact_forces, dim=-1)  # Norm of force vector for each foot
        feet_contact_current = feet_contact_norms > contact_force_threshold
        
        return feet_contact_current  # Returns boolean tensor of shape (num_envs, 4)

    def reset_contact_tracking(self, env_ids):
        """Reset contact tracking for specified environments"""
        self._feet_contact_history[env_ids] = False
        self._feet_air_time_counter[env_ids] = 0.0
        self._feet_last_contact_step[env_ids] = 0
        self._last_air_time[env_ids] = 0.0

    def calculate_reward(self, obs, act):
        # Extract data from obs
        lin_vel_b = obs[:, :3]
        ang_vel_b = obs[:, 3:6]
        projected_gravity_b = obs[:, 6:9]
        commands = obs[:, 9:12]
        joint_pos = obs[:, 12:24]
        joint_vel = obs[:, 24:36]
        actions = obs[:, 36:48]

        # Joint torques and accelerations
        # TODO: Check if finite difference for accel works better
        joint_torques = self.state.joint_tau.view(self.num_envs, -1)[:, 6:]
        joint_accel = self.state.joint_qdd.view(self.num_envs, -1)[:, 6:]

        # Linear velocity tracking
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - lin_vel_b[:, [0, 2]]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # Yaw rate tracking
        yaw_rate_error = torch.square(commands[:, 2] - ang_vel_b[:, 1])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # Z (in our case Y) velocity tracking
        z_vel_error = torch.square(lin_vel_b[:, 1])
        # Angular velocity x/y (in our case x/z)
        ang_vel_error = torch.sum(torch.square(ang_vel_b[:, [0, 2]]), dim=1)
        # Joint torques
        joint_torques = torch.sum(torch.square(joint_torques), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(joint_accel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(actions - self._previous_actions), dim=1)
        # feet air time - Isaac Gym approach
        # Reward feet when they make contact based on how long they were in the air
        # Target air time is 0.5 seconds - longer times get positive rewards, shorter get negative
        feet_air_times = self.feet_air_time()  # This updates tracking and returns current air times
        
        # Calculate air time reward: (last_air_time - 0.5) * first_contact
        # This rewards feet that stay in air for ~0.5 seconds when they make contact
        air_time_reward = torch.sum((self._last_air_time - 0.3) * (self._last_air_time > 0.0), dim=1)
        air_time_reward = torch.clamp(air_time_reward, min=0)
        # # Debug: Show air time rewards
        # if torch.any(self._last_air_time > 0.0):
        #     print(f"Last air times: {self._last_air_time}")
        #     print(f"Air time rewards: {air_time_reward}")
        
        # Only apply reward when robot is moving (prevents hopping in place)
        air_time = air_time_reward * (torch.norm(self._commands[:, :2], dim=1) > 0.1)

        # Get contact forces from simulation state
        # contact_f has shape (num_envs * num_links, 6) where 6 = [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        contact_f = self.state.contact_f.view(self.num_envs, -1, 6)

        if self.undesired_contact_body_links:
            # Extract contact forces for undesired bodies (only force components, not torque)
            undesired_contact_forces = contact_f[:, self.undesired_contact_body_links, 3:]  # [:, :, 3:] gets force components

            # Check if any undesired body has significant contact force (norm > threshold)
            contact_force_norms = torch.norm(undesired_contact_forces, dim=-1)  # Norm of force vector for each body
            contacts = (contact_force_norms > 1.0).any(dim=1).float()
        else:
            contacts = torch.zeros(self.num_envs, device=self.device)

        # flat orientation
        flat_orientation = torch.sum(torch.square(projected_gravity_b[:, [0, 2]]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.lin_vel_reward_scale * self.sim_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.yaw_rate_reward_scale * self.sim_dt,
            "lin_vel_z_l2": z_vel_error * self.z_vel_reward_scale * self.sim_dt,
            "ang_vel_xy_l2": ang_vel_error * self.ang_vel_reward_scale * self.sim_dt,
            "dof_torques_l2": joint_torques * self.joint_torque_reward_scale * self.sim_dt,
            "dof_acc_l2": joint_accel * self.joint_accel_reward_scale * self.sim_dt,
            "action_rate_l2": action_rate * self.action_rate_reward_scale * self.sim_dt,
            # "feet_air_time": air_time * self.feet_air_time_reward_scale * self.sim_dt,
            "undesired_contacts": contacts * self.undesired_contact_reward_scale * self.sim_dt,
            "flat_orientation_l2": flat_orientation * self.flat_orientation_reward_scale * self.sim_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # print("REWARDS")
        # for key, value in rewards.items():
        #     print(f"{key}: {value}")
        # print()

        # Store previous action here because it is called after get obs
        self._previous_actions = self._actions.clone()

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        # Wandb logging if enabled
        if self.wandb_enabled:
            self._log_rewards_to_wandb(rewards, reward)
            
        return reward

    def _log_rewards_to_wandb(self, rewards, total_reward):
        """Log rewards to wandb"""
        try:
            # Log total reward
            wandb.log({"reward/total": total_reward.mean().item()}, step=self._current_step)
            
            # Log individual reward terms
            for key, value in rewards.items():
                wandb.log({f"reward/{key}": value.mean().item()}, step=self._current_step)
                
            # Log episode sums for tracking cumulative performance
            for key, value in self._episode_sums.items():
                wandb.log({f"episode_sum/{key}": value.mean().item()}, step=self._current_step)
                
            # Log velocity tracking errors
            # Extract velocity tracking errors from observations
            obs = self.obs_buf  # Current observations
            if obs is not None and obs.shape[1] >= 12:
                # Linear velocity tracking error (x and z components)
                lin_vel_b = obs[:, :3]  # Body frame linear velocity
                commands = obs[:, 9:12]  # Velocity commands
                
                # Linear velocity tracking error (x and z components)
                lin_vel_x_error = torch.abs(commands[:, 0] - lin_vel_b[:, 0]).mean().item()
                lin_vel_z_error = torch.abs(commands[:, 1] - lin_vel_b[:, 2]).mean().item()
                
                # Angular velocity tracking error (yaw rate)
                ang_vel_b = obs[:, 3:6]  # Body frame angular velocity
                yaw_rate_error = torch.abs(commands[:, 2] - ang_vel_b[:, 1]).mean().item()
                
                # Log velocity tracking errors
                wandb.log({
                    "velocity_tracking/lin_vel_x_error": lin_vel_x_error,
                    "velocity_tracking/lin_vel_z_error": lin_vel_z_error,
                    "velocity_tracking/yaw_rate_error": yaw_rate_error,
                    "velocity_tracking/lin_vel_x_command": commands[:, 0].mean().item(),
                    "velocity_tracking/lin_vel_z_command": commands[:, 1].mean().item(),
                    "velocity_tracking/yaw_rate_command": commands[:, 2].mean().item(),
                    "velocity_tracking/lin_vel_x_actual": lin_vel_b[:, 0].mean().item(),
                    "velocity_tracking/lin_vel_z_actual": lin_vel_b[:, 2].mean().item(),
                    "velocity_tracking/yaw_rate_actual": ang_vel_b[:, 1].mean().item(),
                }, step=self._current_step)
                
                # Additional velocity tracking metrics
                # Normalized errors (error / command magnitude when command is non-zero)
                lin_vel_x_norm_error = torch.where(
                    torch.abs(commands[:, 0]) > 0.1,
                    torch.abs(commands[:, 0] - lin_vel_b[:, 0]) / (torch.abs(commands[:, 0]) + 1e-6),
                    torch.zeros_like(commands[:, 0])
                ).mean().item()
                
                lin_vel_z_norm_error = torch.where(
                    torch.abs(commands[:, 1]) > 0.1,
                    torch.abs(commands[:, 1] - lin_vel_b[:, 2]) / (torch.abs(commands[:, 1]) + 1e-6),
                    torch.zeros_like(commands[:, 1])
                ).mean().item()
                
                yaw_rate_norm_error = torch.where(
                    torch.abs(commands[:, 2]) > 0.1,
                    torch.abs(commands[:, 2] - ang_vel_b[:, 1]) / (torch.abs(commands[:, 2]) + 1e-6),
                    torch.zeros_like(commands[:, 2])
                ).mean().item()
                
                # Success rates (percentage of environments within error threshold)
                error_threshold = 0.1  # 10% error threshold
                lin_vel_x_success_rate = (torch.abs(commands[:, 0] - lin_vel_b[:, 0]) <= error_threshold * torch.abs(commands[:, 0])).float().mean().item()
                lin_vel_z_success_rate = (torch.abs(commands[:, 1] - lin_vel_b[:, 2]) <= error_threshold * torch.abs(commands[:, 1])).float().mean().item()
                yaw_rate_success_rate = (torch.abs(commands[:, 2] - ang_vel_b[:, 1]) <= error_threshold * torch.abs(commands[:, 2])).float().mean().item()
                
                # Log additional metrics
                wandb.log({
                    "velocity_tracking/lin_vel_x_norm_error": lin_vel_x_norm_error,
                    "velocity_tracking/lin_vel_z_norm_error": lin_vel_z_norm_error,
                    "velocity_tracking/yaw_rate_norm_error": yaw_rate_norm_error,
                    "velocity_tracking/lin_vel_x_success_rate": lin_vel_x_success_rate,
                    "velocity_tracking/lin_vel_z_success_rate": lin_vel_z_success_rate,
                    "velocity_tracking/yaw_rate_success_rate": yaw_rate_success_rate,
                }, step=self._current_step)
                
                # Aggregate velocity tracking statistics
                # RMS errors across all environments
                lin_vel_x_rms_error = torch.sqrt(torch.mean(torch.square(commands[:, 0] - lin_vel_b[:, 0]))).item()
                lin_vel_z_rms_error = torch.sqrt(torch.mean(torch.square(commands[:, 1] - lin_vel_b[:, 2]))).item()
                yaw_rate_rms_error = torch.sqrt(torch.mean(torch.square(commands[:, 2] - ang_vel_b[:, 1]))).item()
                
                # Overall velocity tracking performance
                total_lin_vel_error = torch.sqrt(torch.mean(torch.square(commands[:, :2] - lin_vel_b[:, [0, 2]]))).item()
                total_ang_vel_error = torch.sqrt(torch.mean(torch.square(commands[:, 2] - ang_vel_b[:, 1]))).item()
                
                # Log aggregate statistics
                wandb.log({
                    "velocity_tracking/lin_vel_x_rms_error": lin_vel_x_rms_error,
                    "velocity_tracking/lin_vel_z_rms_error": lin_vel_z_rms_error,
                    "velocity_tracking/yaw_rate_rms_error": yaw_rate_rms_error,
                    "velocity_tracking/total_lin_vel_rms_error": total_lin_vel_error,
                    "velocity_tracking/total_ang_vel_rms_error": total_ang_vel_error,
                }, step=self._current_step)

        except Exception as e:
            print(f"Failed to log to wandb: {e}")
            # Disable wandb logging on error to avoid repeated failures
            self.wandb_enabled = False

    def enable_wandb_logging(self):
        """Enable wandb logging if wandb is available"""
        if WANDB_AVAILABLE and not self.wandb_enabled:
            self._init_wandb()
        elif not WANDB_AVAILABLE:
            print("Warning: wandb not available. Install with 'pip install wandb' to enable logging.")

    def disable_wandb_logging(self):
        """Disable wandb logging"""
        if self.wandb_enabled:
            try:
                wandb.finish()
                self.wandb_enabled = False
                print("Wandb logging disabled")
            except Exception as e:
                print(f"Failed to disable wandb: {e}")

    def enable_observation_noise(self, base_lin_vel_range=None, base_ang_vel_range=None, 
                                projected_gravity_range=None, joint_pos_range=None, joint_vel_range=None):
        """Enable observation noise with optional custom ranges"""
        self.observation_noise = True
        if base_lin_vel_range is not None:
            self.base_lin_vel_obs_noise = base_lin_vel_range
        if base_ang_vel_range is not None:
            self.base_ang_vel_obs_noise = base_ang_vel_range
        if projected_gravity_range is not None:
            self.projected_gravity_obs_noise = projected_gravity_range
        if joint_pos_range is not None:
            self.joint_pos_obs_noise = joint_pos_range
        if joint_vel_range is not None:
            self.joint_vel_obs_noise = joint_vel_range
        print(f"Observation noise enabled: base_lin_vel_range={self.base_lin_vel_obs_noise}, "
              f"base_ang_vel_range={self.base_ang_vel_obs_noise}, "
              f"projected_gravity_range={self.projected_gravity_obs_noise}, "
              f"joint_pos_range={self.joint_pos_obs_noise}, "
              f"joint_vel_range={self.joint_vel_obs_noise}")

    def disable_observation_noise(self):
        """Disable observation noise"""
        self.observation_noise = False
        print("Observation noise disabled")

    def get_noise_config(self):
        """Get current noise configuration"""
        return {
            "observation_noise": self.observation_noise,
            "base_lin_vel_obs_noise": list(self.base_lin_vel_obs_noise),
            "base_ang_vel_obs_noise": list(self.base_ang_vel_obs_noise),
            "projected_gravity_obs_noise": list(self.projected_gravity_obs_noise),
            "joint_pos_obs_noise": list(self.joint_pos_obs_noise),
            "joint_vel_obs_noise": list(self.joint_vel_obs_noise),
            "mass_randomization": self.mass_randomization,
            "mass_randomization_bodies": list(self.mass_randomization_bodies),
            "mass_randomization_noise": list(self.mass_randomization_noise),
            "com_randomization": self.com_randomization,
            "com_randomization_bodies": list(self.com_randomization_bodies),
            "com_randomization_noise": list(self.com_randomization_noise),
        }

    def _apply_mass_randomization(self, env_ids):
        """Apply mass randomization to specified environments"""
        if not self.mass_randomization or not self.mass_randomization_bodies:
            return
        
        try:
            # Get the number of links per environment
            links_per_env = self.model.link_count // self.num_envs
            
            for body_idx in self.mass_randomization_bodies:
                # Calculate the actual link indices for the specified environments
                link_indices = [body_idx + i * links_per_env for i in env_ids]
                
                for link_idx in link_indices:
                    if link_idx < self.model.link_count:
                        # Get current mass from the inertia tensor (body_I_m[3, 3])
                        current_mass = self.model.body_I_m[link_idx, 3, 3]
                        
                        # Apply uniform noise within the specified range
                        mass_noise = torch.empty(1, device=self.device).uniform_(
                            self.mass_randomization_noise[0], 
                            self.mass_randomization_noise[1]
                        )
                        
                        # Scale the noise by the current mass and apply it
                        new_mass = current_mass * (1.0 + mass_noise.item())
                        
                        # Ensure mass stays positive
                        new_mass = max(new_mass, 0.01)  # Minimum mass of 0.01
                        
                        # Apply the new mass to the inertia tensor
                        # We need to scale the entire inertia tensor proportionally
                        mass_scale = new_mass / current_mass
                        self.model.body_I_m[link_idx] *= mass_scale

                    else:
                        print(f"Warning: Link index {link_idx} out of range (max: {self.model.link_count})")
        except Exception as e:
            print(f"Error in mass randomization: {e}")
            import traceback
            traceback.print_exc()

    def _apply_com_randomization(self, env_ids):
        """Apply COM randomization to specified environments"""
        if not self.com_randomization or not self.com_randomization_bodies:
            return
        
        try:
            # Get the number of links per environment
            links_per_env = self.model.link_count // self.num_envs
            
            for body_idx in self.com_randomization_bodies:
                # Calculate the actual link indices for the specified environments
                link_indices = [body_idx + i * links_per_env for i in env_ids]
                
                for link_idx in link_indices:
                    if link_idx < self.model.link_count:
                        # COM information is stored in joint_X_cm (joint mass frame in child frame)
                        # This describes where the center of mass is relative to the joint frame
                        if hasattr(self.model, 'joint_X_cm') and self.model.joint_X_cm is not None:
                            # Get current COM from joint_X_cm (center of mass in joint frame)
                            # joint_X_cm has shape [joint_count, 7] where 7 = [x, y, z, qx, qy, qz, qw]
                            current_com = self.model.joint_X_cm[link_idx, :3]  # First 3 elements are x, y, z
                            
                            # Generate uniform noise for x, y, z COM components
                            com_noise = torch.empty(3, device=self.device).uniform_(
                                self.com_randomization_noise[0], 
                                self.com_randomization_noise[1]
                            )
                            
                            # Apply noise to COM
                            new_com = current_com + com_noise
                            
                            # Update the COM in the model
                            self.model.joint_X_cm[link_idx, :3] = new_com

                            # Note: When COM changes, we should also update the inertia tensor
                            # This is a simplified approach - in practice, you might need to recompute
                            # the inertia tensor using Steiner's theorem or similar methods
                        else:
                            print(f"Warning: joint_X_cm not available for link {link_idx}")
                            print(f"Available attributes: {[attr for attr in dir(self.model) if 'cm' in attr or 'com' in attr]}")
                        
                    else:
                        print(f"Warning: Link index {link_idx} out of range (max: {self.model.link_count})")
        except Exception as e:
            print(f"Error in COM randomization: {e}")
            import traceback
            traceback.print_exc()

    def enable_mass_randomization(self, bodies=None, noise_range=None):
        """Enable mass randomization with optional custom parameters"""
        self.mass_randomization = True
        if bodies is not None:
            self.mass_randomization_bodies = bodies
        if noise_range is not None:
            self.mass_randomization_noise = noise_range
        print(f"Mass randomization enabled: bodies={self.mass_randomization_bodies}, "
              f"noise_range={self.mass_randomization_noise}")

    def disable_mass_randomization(self):
        """Disable mass randomization"""
        self.mass_randomization = False
        print("Mass randomization disabled")

    def enable_com_randomization(self, bodies=None, noise_range=None):
        """Enable COM randomization with optional custom parameters"""
        self.com_randomization = True
        if bodies is not None:
            self.com_randomization_bodies = bodies
        if noise_range is not None:
            self.com_randomization_noise = noise_range
        print(f"COM randomization enabled: bodies={self.com_randomization_bodies}, "
              f"noise_range={self.com_randomization_noise}")

    def disable_com_randomization(self):
        """Disable COM randomization"""
        self.com_randomization = False
        print("COM randomization disabled")

    def inspect_model_structure(self):
        """Inspect the model structure to understand available attributes"""
        print(f"\n=== Model Structure Inspection ===")
        print(f"Model type: {type(self.model)}")
        print(f"Link count: {self.model.link_count}")
        print(f"Available attributes:")
        
        # Check key attributes
        key_attrs = [
            'body_X_cm', 'body_I_m', 'joint_X_cm', 'shape_transform', 'shape_body',
            'joint_q', 'joint_qd', 'particle_q', 'particle_mass'
        ]
        
        for attr in key_attrs:
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if value is not None:
                    if torch.is_tensor(value):
                        print(f"  {attr}: {type(value)}, shape: {value.shape}, dtype: {value.dtype}")
                    else:
                        print(f"  {attr}: {type(value)}, value: {value}")
                else:
                    print(f"  {attr}: {type(value)}, value: None")
            else:
                print(f"  {attr}: Not available")
        
        # Check if body_X_cm exists and show its structure
        if hasattr(self.model, 'body_X_cm') and self.model.body_X_cm is not None:
            print(f"\nbody_X_cm details:")
            print(f"  Shape: {self.model.body_X_cm.shape}")
            print(f"  First few values:")
            for i in range(min(3, self.model.link_count)):
                com = self.model.body_X_cm[i, :3].tolist()
                print(f"    Link {i}: {com}")
        else:
            print(f"\nbody_X_cm: Not available or None")
        
        # Check if joint_X_cm exists and show its structure
        if hasattr(self.model, 'joint_X_cm') and self.model.joint_X_cm is not None:
            print(f"\njoint_X_cm details:")
            print(f"  Shape: {self.model.joint_X_cm.shape}")
            print(f"  First few values:")
            for i in range(min(3, self.model.link_count)):
                com = self.model.joint_X_cm[i, :3].tolist()
                print(f"    Link {i}: {com}")
        else:
            print(f"\njoint_X_cm: Not available or None")
        
        print("=" * 50)

    def reset_episode_stats(self, env_ids):
        """Reset episode statistics for specified environments"""
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0
            
        # Log episode reset to wandb if enabled
        if self.wandb_enabled:
            try:
                wandb.log({
                    "env/episode_reset": len(env_ids),
                    "env/active_envs": self.num_envs - len(env_ids)
                }, step=self._current_step)
            except Exception as e:
                print(f"Failed to log episode reset to wandb: {e}")
                self.wandb_enabled = False

    def _apply_observation_noise(self, obs):
        """Apply noise to observations if observation_noise is enabled"""
        if not self.observation_noise:
            return obs
        
        # Create a copy to avoid modifying the original
        noisy_obs = obs.clone()
        
        # Apply noise to base linear velocity observations (indices 0:3)
        if self.base_lin_vel_obs_noise[1] > self.base_lin_vel_obs_noise[0]:
            lin_vel_noise = torch.empty_like(obs[:, :3]).uniform_(
                self.base_lin_vel_obs_noise[0], 
                self.base_lin_vel_obs_noise[1]
            )
            noisy_obs[:, :3] += lin_vel_noise
        
        # Apply noise to base angular velocity observations (indices 3:6)
        if self.base_ang_vel_obs_noise[1] > self.base_ang_vel_obs_noise[0]:
            ang_vel_noise = torch.empty_like(obs[:, 3:6]).uniform_(
                self.base_ang_vel_obs_noise[0], 
                self.base_ang_vel_obs_noise[1]
            )
            noisy_obs[:, 3:6] += ang_vel_noise
        
        # Apply noise to projected gravity observations (indices 6:9)
        if self.projected_gravity_obs_noise[1] > self.projected_gravity_obs_noise[0]:
            proj_gravity_noise = torch.empty_like(obs[:, 6:9]).uniform_(
                self.projected_gravity_obs_noise[0], 
                self.projected_gravity_obs_noise[1]
            )
            noisy_obs[:, 6:9] += proj_gravity_noise
        
        # Apply noise to joint position observations (indices 12:24)
        if self.joint_pos_obs_noise[1] > self.joint_pos_obs_noise[0]:
            joint_pos_noise = torch.empty_like(obs[:, 12:24]).uniform_(
                self.joint_pos_obs_noise[0], 
                self.joint_pos_obs_noise[1]
            )
            noisy_obs[:, 12:24] += joint_pos_noise
        
        # Apply noise to joint velocity observations (indices 24:36)
        if self.joint_vel_obs_noise[1] > self.joint_vel_obs_noise[0]:
            joint_vel_noise = torch.empty_like(obs[:, 24:36]).uniform_(
                self.joint_vel_obs_noise[0], 
                self.joint_vel_obs_noise[1]
            )
            noisy_obs[:, 24:36] += joint_vel_noise
        
        return noisy_obs

    def close(self):
        """Close the environment and wandb logging"""
        if self.wandb_enabled:
            try:
                wandb.finish()
                print("Wandb logging finished")
            except Exception as e:
                print(f"Failed to close wandb: {e}")
