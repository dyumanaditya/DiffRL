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
    ):
        num_obs = 49
        num_act = 12
        self.playback_speed = 0.005

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
        )

        self.early_termination = early_termination
        self.wandb_enabled = wandb and WANDB_AVAILABLE

        self.init_sim()

        # MDP parameters
        self.action_scale = 1.0

        # Velocity tracking parameters
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._previous_actions = torch.zeros(self.num_envs, num_act, device=self.device)
        self._actions = torch.zeros(self.num_envs, num_act, device=self.device)
        
        # Rewards: Following IsaacLab reward structure
        self.lin_vel_reward_scale = 8.0
        # self.lin_vel_reward_scale = 1.0
        self.yaw_rate_reward_scale = 4.0
        # self.yaw_rate_reward_scale = 0.5
        self.z_vel_reward_scale = -4.0
        self.ang_vel_reward_scale = -0.05
        self.joint_torque_reward_scale = -2.5e-5
        self.joint_accel_reward_scale = -2.5e-7
        self.action_rate_reward_scale = -0.01
        self.feet_air_time_reward_scale = 0.25
        self.undesired_contact_reward_scale = -2.0
        self.flat_orientation_reward_scale = -2.5

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
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-2.0, 2.0)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]) + torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        )

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
        self.soft_limit_ratio = float(0.95)

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0.0  # set to zero for training for numerical consistency

        start_height = 0.45

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
                stiffness=85.0,  # from config
                damping=2.0,  # from config
                # stiffness=1000.0,  # from config
                # damping=10.0,  # from config
                shape_ke=2.0e4,
                shape_kd=5.0e3,
                shape_kf=1.0e3,
                shape_mu=1.0,
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

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if self.model.ground:
            self.model.collide(self.state)

    def step(self, actions, play=False):
        obs, rew, done, extras = super().step(actions, play)
        
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
        return action * self.action_scale

    def set_act(self, action):
        self._actions = action.clone() / self.action_scale  # Because we scaled already earlier
        # Convert action to target joint positions
        target_joint_positions = action + self.default_joint_pos

        # TODO: clamp target_joint_positions to joint limits
        # --- Soft-limit clamp (ratio in (0, 1]; 1.0 = full limits, 0.95 = keep 5% margin) ---
        # Pull per-env limits for the actuated joints (skip the 7 DoFs of the floating base)
        jcount = target_joint_positions.shape[1]  # expected 12
        lower = self.model.joint_limit_lower.view(self.num_envs, -1)[:, 7:7 + jcount]
        upper = self.model.joint_limit_upper.view(self.num_envs, -1)[:, 7:7 + jcount]

        # Compute inner (soft) bounds centered in the range
        soft = self.soft_limit_ratio
        # inner = [lower + m, upper - m] with m = (1 - soft) * half_range
        half_range = 0.5 * (upper - lower)
        margin = (1.0 - soft) * half_range
        inner_lower = lower + margin
        inner_upper = upper - margin

        # If any joint has no finite limits, leave it unchanged
        finite = torch.isfinite(inner_lower) & torch.isfinite(inner_upper)
        # Clamp into the soft range (torch.clamp supports tensor min/max)
        clamped = torch.where(
            finite,
            torch.clamp(target_joint_positions, min=inner_lower, max=inner_upper),
            target_joint_positions,
        )

        target_joint_positions = clamped

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
            termination = termination | (obs[:, 48] < self.termination_height)
            
        # Log termination statistics to wandb if enabled
        if self.wandb_enabled and termination.any():
            try:
                termination_reasons = {
                    "termination/nonfinite_obs": nonfinite_mask.sum().item(),
                    "termination/invalid_joint_q": (torch.abs(joint_q) > 1e6).sum().item(),
                    "termination/invalid_joint_qd": (torch.abs(joint_qd) > 1e6).sum().item(),
                    "termination/height": (obs[:, 48] < self.termination_height).sum().item() if self.early_termination else 0,
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
        torso_height = torso_pos[:, 1].clone()

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
                torso_height.unsqueeze(1),           # 48:49 (optional, can be used for height tracking)
            ],
            dim=-1,
        )
        
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
                    "obs/torso_height_mean": torso_height.mean().item(),
                    "obs/torso_height_std": torso_height.std().item(),
                    "action/mean": self._actions.mean().item(),
                    "action/std": self._actions.std().item(),
                }
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
        contact_force_threshold = 0.1  # Same threshold as in feet_air_time()
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
        air_time_reward = torch.sum((self._last_air_time - 0.5) * (self._last_air_time > 0.0), dim=1)
        
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
            "feet_air_time": air_time * self.feet_air_time_reward_scale * self.sim_dt,
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

    def close(self):
        """Close the environment and wandb logging"""
        if self.wandb_enabled:
            try:
                wandb.finish()
                print("Wandb logging finished")
            except Exception as e:
                print(f"Failed to close wandb: {e}")
