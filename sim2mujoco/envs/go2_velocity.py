import os
import math
import dflex as df
import mujoco
import torch
import numpy as np
import gymnasium as gym
import dflex.envs.torch_utils as tu

# Try different mujoco viewer imports
try:
    from mujoco import viewer
    MUJOCO_VIEWER_AVAILABLE = True
    MUJOCO_VIEWER_TYPE = "mujoco.viewer"
except ImportError:
    try:
        import mujoco_viewer
        MUJOCO_VIEWER_AVAILABLE = True
        MUJOCO_VIEWER_TYPE = "mujoco_viewer"
    except ImportError:
        try:
            from mujoco_py import MjViewer
            MUJOCO_VIEWER_AVAILABLE = True
            MUJOCO_VIEWER_TYPE = "mujoco_py"
        except ImportError:
            MUJOCO_VIEWER_AVAILABLE = False
            MUJOCO_VIEWER_TYPE = "none"


class Go2VelocityMujocoEnv:
    """
    Go2 velocity tracking environment using Mujoco
    """

    def __init__(
        self,
        render=True,
        num_envs=1,
        episode_length=1000,
        stochastic_init=False,
        early_termination=True,
        termination_height=0.10,
        action_penalty=-0.005,
        up_rew_scale=0.1,
        heading_rew_scale=1.0,
        heigh_rew_scale=1.0,
        render_delay=0.01,  # Delay between steps for visualization
        **kwargs
    ):
        self.num_envs = num_envs
        self.num_actions = 12
        self.num_obs = 49
        self.episode_length = episode_length
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.termination_height = termination_height
        self.render = render
        self.render_delay = render_delay
        
        # Set timestep to match DFlex (1/60 seconds per control step)
        # self.dt = 1.0 / 60.0  # 0.01667 seconds
        self.dt = 0.02

        # Step counter
        self.step_count = 0
        
        # Device setup - ensure consistent dtype
        self.device = kwargs.get("device", "cpu")
        self.dtype = torch.float32  # Ensure consistent dtype
        
        # Action scaling
        self.action_scale = 1.0
        
        # Velocity tracking parameters - ensure consistent dtype
        self._commands = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.num_actions, dtype=self.dtype, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.num_actions, dtype=self.dtype, device=self.device)
        
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
        
        # Default joint positions (matching DFlex) - ensure consistent dtype
        self.default_joint_pos = torch.tensor([
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
        ], dtype=self.dtype, device=self.device)
        
        # Episode logging - ensure consistent dtype
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
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
        self._feet_air_time_counter = torch.zeros((self.num_envs, 4), dtype=self.dtype, device=self.device)
        self._current_step = 0
        
        # Feet contact links (matching DFlex)
        self.feet_contact_links = [11, 20, 29, 38]  # FL, FR, RL, RR
        
        # Initialize Mujoco
        self._init_mujoco()
        
        # Sample initial commands
        self.sample_commands(torch.arange(self.num_envs, device=self.device))
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)
        self.start_rotation = tu.to_torch(
            self.start_rot, device=self.device, requires_grad=False
        )
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )

    def _init_mujoco(self):
        """Initialize Mujoco model and data"""
        # Load the XML file with position actuators instead of motors
        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'unitree_go2', 'scene.xml')
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set Mujoco timestep to match DFlex (1/60 seconds)
        # self.model.opt.timestep = self.dt
        
        # Set initial state
        self._set_initial_state()
        
        # Set up renderer if needed
        if self.render:
            self._setup_viewer()

    def _setup_viewer(self):
        """Set up the appropriate viewer based on available options"""
        if not MUJOCO_VIEWER_AVAILABLE:
            print("Warning: No Mujoco viewer available. Rendering disabled.")
            self.render = False
            return
        
        try:
            if MUJOCO_VIEWER_TYPE == "mujoco.viewer":
                # Modern mujoco viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
                print("Using mujoco.viewer")
                # Show axis triads (choose one of: GLOBAL, BODY, GEOM, SITE, CAMERA, LIGHT)
                # self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY  # body-local axes on every body
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD   # world axes at the origin

                # # Make them visible enough
                # self.viewer.opt.framelength = 0.20  # meters
                # self.viewer.opt.framewidth = 0.006  # thickness
                # Set viewer properties for better performance
                if hasattr(self.viewer, 'cam'):
                    self.viewer.cam.distance = 3.0
                    self.viewer.cam.azimuth = 45.0
                    self.viewer.cam.elevation = -20.0
            elif MUJOCO_VIEWER_TYPE == "mujoco_viewer":
                # mujoco-viewer package
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                print("Using mujoco_viewer")
            elif MUJOCO_VIEWER_TYPE == "mujoco_py":
                # Legacy mujoco-py
                self.viewer = MjViewer(self.model, self.data)
                print("Using mujoco_py")
            else:
                print("Warning: No compatible viewer found. Rendering disabled.")
                self.render = False
        except Exception as e:
            print(f"Warning: Could not launch viewer ({MUJOCO_VIEWER_TYPE}): {e}")
            print("Rendering disabled. Continuing without visualization.")
            self.render = False

    def _set_initial_state(self):
        """Set initial state for the robot"""
        # Set initial position (floating base)
        self.data.qpos[:3] = [0.0, 0.0, 0.40]  # x, y, z
        
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
            [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        )

    def _force_viewer_update(self):
        """Force a viewer update - useful for debugging"""
        if self.render and hasattr(self, 'viewer'):
            try:
                if MUJOCO_VIEWER_TYPE == "mujoco.viewer":
                    # Force sync for modern viewer
                    if hasattr(self.viewer, 'sync'):
                        self.viewer.sync()
                    # Also try to render if available
                    if hasattr(self.viewer, 'render'):
                        self.viewer.render()
                elif MUJOCO_VIEWER_TYPE == "mujoco_viewer":
                    self.viewer.render()
                elif MUJOCO_VIEWER_TYPE == "mujoco_py":
                    self.viewer.render()
            except Exception as e:
                print(f"Force viewer update failed: {e}")

    def _update_viewer(self):
        """Update the viewer based on the viewer type"""
        try:
            if MUJOCO_VIEWER_TYPE == "mujoco.viewer":
                # Modern mujoco viewer - needs explicit sync
                if hasattr(self.viewer, 'sync'):
                    self.viewer.sync()
                # Also try to render if available
                if hasattr(self.viewer, 'render'):
                    self.viewer.render()
            elif MUJOCO_VIEWER_TYPE == "mujoco_viewer":
                # mujoco-viewer package
                self.viewer.render()
            elif MUJOCO_VIEWER_TYPE == "mujoco_py":
                # Legacy mujoco-py
                self.viewer.render()
        except Exception as e:
            # If viewer fails, disable rendering
            print(f"Viewer update failed: {e}")
            self.render = False

    def step(self, action):
        """Step the environment with the given action"""
        # Ensure action has correct dtype and device
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=self.dtype, device=self.device)
        elif action.dtype != self.dtype:
            action = action.to(dtype=self.dtype)
        if action.device != self.device:
            action = action.to(device=self.device)
        
        # Store previous actions for reward calculation
        self._previous_actions = self._actions.clone()
        
        # Process action - ensure it's the right shape
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Add batch dimension if needed
        
        action = torch.clamp(action, -1.0, 1.0)
        self._actions = action.clone()
        
        # Convert action to target joint positions
        target_joint_positions = action + self.default_joint_pos
        
        # Apply action to Mujoco
        self._apply_action(target_joint_positions)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update viewer if rendering - do this after simulation step
        if self.render and hasattr(self, 'viewer'):
            self._update_viewer()
            # Add small delay for visualization
            if self.render_delay > 0:
                import time
                time.sleep(self.render_delay)
                # time.sleep(0.3)

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
            "dt": self.dt,
            "step_count": self.step_count,
        }
        
        return obs, reward, done, info

    def get_timestep_info(self):
        """Get information about the current timestep"""
        return {
            "dt": self.dt,
            "control_freq": 1.0 / self.dt,  # 60 Hz
            "step_count": self.step_count
        }

    def _apply_action(self, target_joint_positions):
        """Apply action to Mujoco model"""
        # Ensure we have the right number of actions
        if target_joint_positions.shape[1] != 12:
            raise ValueError(f"Expected 12 actions, got {target_joint_positions.shape[1]}")
        
        # Set joint targets (skip first 7 DoFs which are floating base)
        # Use the first environment's actions if multiple envs
        actions_to_apply = target_joint_positions[0] if target_joint_positions.dim() > 1 else target_joint_positions
        self.data.ctrl[:12] = actions_to_apply.cpu().numpy()

    def _get_observation(self):
        """Get observation from MuJoCo state and reproduce DFLEX-style observations."""
        dtype = self.dtype
        device = self.device
        N = self.num_envs

        # ----- 1) Read MuJoCo state (world frame, Z-up) -----
        torso_pos_mj = torch.tensor(self.data.qpos[:3], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)  # [N,3]
        torso_rot_mj = torch.tensor(self.data.qpos[3:7], dtype=dtype, device=device).unsqueeze(0).repeat(N,
                                                                                                         1)  # [N,4] (MuJoCo)
        lin_vel_mj = torch.tensor(self.data.qvel[3:6], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)  # [N,3]
        ang_vel_mj = torch.tensor(self.data.qvel[:3], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)  # [N,3]
        joint_pos = torch.tensor(self.data.qpos[7:], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)
        joint_vel = torch.tensor(self.data.qvel[6:], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)

        # ----- 2) MuJoCo quaternion is [w,x,y,z] -> convert to [x,y,z,w] (required by tu.quat_*) -----
        # If your MuJoCo build already provides [x,y,z,w], remove this line.
        torso_rot_mj = torso_rot_mj[:, [1, 2, 3, 0]]

        # ----- 3) Z-up (MuJoCo) -> Y-up (DFLEX) via Rx(-90°). You already created start_rotation = Rx(-90°)
        # Use the SAME start_rotation / inv_start_rot you set up outside.
        q_world_to_dflex = self.start_rotation.to(dtype=dtype, device=device)
        if q_world_to_dflex.dim() == 1:
            q_world_to_dflex = q_world_to_dflex.unsqueeze(0).repeat(N, 1)

        # Rotate pose/orientation/velocities into DFLEX world (Y-up)
        torso_pos_df_w = tu.quat_apply(q_world_to_dflex, torso_pos_mj)
        torso_rot_df_w = tu.quat_mul(q_world_to_dflex, torso_rot_mj)
        lin_vel_df_w = tu.quat_rotate(q_world_to_dflex, lin_vel_mj)
        ang_vel_df_w = tu.quat_rotate(q_world_to_dflex, ang_vel_mj)  # pseudovector -> rotate like a vector

        # ----- 4) DFLEX canonicalization (exact DFLEX line): torso_quat = quat_mul(torso_rot, inv_start_rot)
        inv_start_rot = self.inv_start_rot.to(dtype=dtype, device=device)  # you already repeat to [N,4] outside
        torso_quat = tu.quat_mul(torso_rot_df_w, inv_start_rot)

        # ----- 5) World -> body projection (exact DFLEX ops) -----
        lin_vel_b = tu.quat_rotate_inverse(torso_quat, lin_vel_df_w)
        ang_vel_b = tu.quat_rotate_inverse(torso_quat, ang_vel_df_w)

        # print("lin vel before projection", lin_vel_df_w)
        # print("lin vel after projection", lin_vel_b)
        # print("ang vel before projection", ang_vel_df_w)
        # print("ang vel after projection", ang_vel_b)
        # print()

        # Gravity in DFLEX world points DOWN along -Y
        y_unit = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        if y_unit.dim() == 1:
            y_unit = y_unit.unsqueeze(0).repeat(N, 1)

        g_world = -y_unit
        projected_gravity_b = tu.quat_rotate_inverse(torso_quat, g_world)

        # Torso height in DFLEX (Y-up)
        torso_height = torso_pos_df_w[:, 1].clone().unsqueeze(1)

        # Ensure joint dims match defaults (DFLEX obs layout)
        default_joint_pos = self.default_joint_pos.to(dtype=dtype, device=device)
        if default_joint_pos.dim() == 1:
            default_joint_pos = default_joint_pos.unsqueeze(0).repeat(N, 1)
        if joint_pos.shape[1] != default_joint_pos.shape[1]:
            joint_pos = joint_pos[:, : default_joint_pos.shape[1]]
            joint_vel = joint_vel[:, : default_joint_pos.shape[1]]
        #
        # print("OBSERVATIONS")
        # print("lin vel", lin_vel_b)
        # print("ang vel", ang_vel_b)
        # print("projected g", projected_gravity_b)
        # print("commands", self._commands)
        # print("joint pos", joint_pos)
        # print("joint vel", joint_vel)
        # print("actions", self._actions)
        # print("torso height", torso_height)
        # print("torso pos", torso_pos_df_w)
        # print()

        # ----- 6) Build obs exactly like DFLEX -----
        obs = torch.cat(
            [
                lin_vel_b,  # 0:3
                ang_vel_b,  # 3:6
                projected_gravity_b,  # 6:9
                self._commands.clone().to(dtype),  # 9:12
                joint_pos - default_joint_pos,  # 12:...
                joint_vel,  # ...
                self._actions.clone().to(dtype),  # 36:48
                torso_height,  # 48:49
            ],
            dim=-1,
        )

        return obs.to(dtype=dtype)

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
        feet_air_time = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        
        # Undesired contacts (simplified)
        undesired_contacts = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        
        # Flat orientation
        flat_orientation = torch.sum(torch.square(projected_gravity_b[:, [0, 2]]), dim=1)
        
        # Calculate rewards - ensure consistent dtype
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
        
        # Total reward - ensure consistent dtype
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0).to(dtype=self.dtype)
        
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
        if not isinstance(commands, torch.Tensor):
            commands = torch.tensor(commands, dtype=self.dtype, device=self.device)
        
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
