# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from abc import abstractmethod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

import dflex as df

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from gym import spaces


def jacobian(output, input, max_out_dim=None):
    """Computes the jacobian of output tensor with respect to the input"""
    num_envs, input_dim = input.shape
    output_dim = output.shape[1]
    if max_out_dim:
        output_dim = min(output_dim, max_out_dim)
    jacobians = torch.zeros((num_envs, output_dim, input_dim), dtype=torch.float32)
    for out_idx in range(output_dim):
        select_index = torch.zeros(output.shape[1])
        select_index[out_idx] = 1.0
        e = torch.tile(select_index, (num_envs, 1)).cuda()
        # retain = out_idx != output_dim - 1  # NOTE: experimental
        try:
            (grad,) = torch.autograd.grad(
                outputs=output, inputs=input, grad_outputs=e, retain_graph=True
            )
            jacobians[:, out_idx, :] = grad.view(num_envs, input_dim)
        except RuntimeError as err:
            print(f"WARN: Couldn't compute jacobian for {out_idx} index")
            print(err)


class DFlexEnv:
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        MM_caching_frequency=1,
        no_grad=True,
        render=False,
        nan_state_fix=False,
        jacobian_norm=None,
        stochastic_init=False,
        jacobian=False,
        device="cuda:0",
        **kwargs,
    ):
        self.no_grad = no_grad
        df.config.no_grad = self.no_grad
        self.nan_state_fix = nan_state_fix
        self.jacobian_norm = jacobian_norm
        # if true resets all envs on earfly termination
        self.stochastic_init = stochastic_init
        self.jacobian = jacobian
        self.jacobians = []

        self.episode_length = episode_length
        self.max_episode_steps = episode_length

        self.device = device

        self.visualize = render

        self.sim_time = 0.0

        self.num_frames = 1  # record the number of frames for rendering

        self.num_environments = num_envs
        self.num_agents = 1

        self.MM_caching_frequency = MM_caching_frequency

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act

        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        # --------------------------------------------------------------
        # Convenience pre-compute counts for DR helper
        # These will be re-computed lazily if the model is not yet built.
        # --------------------------------------------------------------
        # will be filled lazily once the model is built
        self._links_per_env = None
        self._shapes_per_env = None

        # allocate buffers
        self.obs_buf: torch.Tensor = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )

        # --------------------------------------------------------------
        # Set up domain-randomisation parameters
        # --------------------------------------------------------------
        # YAML files can supply either
        #   dr_params: {...}
        # or (older)  domain_randomization: {...}
        # We also require a top-level flag `dr: true` to enable it so that
        # users can keep the dict around but disable sampling with a single
        # switch.

        if kwargs.get("dr", False):
            self.dr_params = (
                kwargs.get("domain_randomization")
            )
        else:
            self.dr_params = None

    @abstractmethod
    def observation_from_state(self, state):
        pass

    @abstractmethod
    def set_act(self, act):
        pass

    @abstractmethod
    def set_state_act(self, obs, act):
        pass

    @abstractmethod
    def unscale_act(self, action):
        pass

    @abstractmethod
    def calculate_reward(self, obs, act):
        pass

    @abstractmethod
    def static_init_func(self, env_ids):
        pass

    @abstractmethod
    def stochastic_init_func(self, env_ids):
        pass

    def _apply_domain_randomization(self, env_ids):
        """Randomise physical parameters for the given environments.

        The YAML dictionary passed in `self.dr_params` can contain items of
        the form

        ```yaml
        ke_range: [1e3, 1e5]
        mu_range: [0.5, 1.0]
        mass_range: [0.8, 1.2]
        ```

        Supported keys map to internal tensors as follows

        • *_range  → uniform sampling in that range  (low, high)
        • ke → shape material column 0
        • kd → shape material column 1
        • kf → shape material column 2
        • mu → shape material column 3
        • mass → body_mass and body_inertia scale
        """

        if self.dr_params is None or len(env_ids) == 0:
            return

        # print("Applying domain randomization for envs:", env_ids)
        # print(self.dr_params)

        # Make sure counts are available (model might be built after __init__)
        if self._links_per_env is None:
            self._links_per_env = (
                self.model.link_count // self.num_envs if self.model.link_count else 0
            )
        if self._shapes_per_env is None:
            self._shapes_per_env = (
                self.model.shape_count // self.num_envs if self.model.shape_count else 0
            )

        # Mapping from parameter name → (type, index)
        # type == "shape_mat" → self.model.shape_materials[:, col]
        # type == "mass"      → body_mass + inertia scaling
        _map = {
            "ke_range": ("shape_mat", 0),
            "kd_range": ("shape_mat", 1),
            "kf_range": ("shape_mat", 2),
            "mu_range": ("shape_mat", 3),
            # "contact_ke": ("shape_mat", 0),
            # "contact_kd": ("shape_mat", 1),
            # "contact_kf": ("shape_mat", 2),
            # "friction": ("shape_mat", 3),
            # "mass_range": ("mass", None),
            # "mass": ("mass", None),
        }

        with torch.no_grad():

            # Target bodies to add DR to
            if "bodies" in self.dr_params:
                target_bodies = self.dr_params["bodies"]

                if target_bodies is not None and len(target_bodies) > 0:
                    target_bodies = torch.tensor(
                        target_bodies, device=self.device, dtype=torch.long
                    )
                else:
                    target_bodies = None

            for name, val in self.dr_params.items():
                if name not in _map:
                    continue

                # Determine low / high bounds
                val = list(val)
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    low, high = val
                elif isinstance(val, dict):
                    low, high = val.get("min"), val.get("max")
                else:
                    # unsupported spec, skip (for bodies)
                    continue

                param_type, col = _map[name]

                # Sample one value per environment being reset
                rnd_body = (
                    torch.rand(len(env_ids), device=self.device) * (high - low) + low
                )

                if param_type == "shape_mat" and self._shapes_per_env > 0:
                    for i, env_id in enumerate(env_ids):
                        s0 = env_id * self._shapes_per_env
                        s1 = s0 + self._shapes_per_env
                        if target_bodies is None:
                            # apply to all shapes in env
                            self.model.shape_materials[s0:s1, col] = rnd_body[i]
                        else:
                            # Filter shapes whose (link-id relative to env) is in target_bodies
                            body_ids = self.model.shape_body[s0:s1] - (
                                    env_id * self._links_per_env
                            )
                            mask = (body_ids[..., None] == target_bodies).any(-1)
                            self.model.shape_materials[s0:s1][mask, col] = rnd_body[i]

                # elif param_type == "mass" and self._links_per_env > 0:
                #     for i, env_id in enumerate(env_ids):
                #         l0 = env_id * self._links_per_env
                #         l1 = l0 + self._links_per_env
                #         scale = rnd[i]
                #         self.model.body_mass[l0:l1] *= scale
                #         self.model.body_inertia[l0:l1] *= scale
                #         # no need to update other derived quantities – they
                #         # will be re-computed during stepping

                # Randomize the ground parameters as well
                rnd_ground = (
                    torch.rand(len(env_ids), device=self.device) * (high - low) + low
                )
                setattr(self.model, f"contact_{name}", rnd_ground)

    def compute_termination(self, obs, act):
        # Never terminate; needs to be overriden if we need termination
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return termination

    def step(self, actions, play=False):
        actions = actions.view((self.num_envs, self.num_actions))
        actions = torch.clip(actions, -1.0, 1.0)
        unscaled_actions = self.unscale_act(actions)
        self.set_act(unscaled_actions)

        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        if self.nan_state_fix:

            def create_hook():
                def hook(grad):
                    torch.nan_to_num(grad, 0.0, 0.0, 0.0, out=grad)

                return hook

            if self.state.joint_q.requires_grad:
                self.state.joint_q.register_hook(create_hook())
            if self.state.joint_qd.requires_grad:
                self.state.joint_qd.register_hook(create_hook())
            if actions.requires_grad:
                actions.register_hook(create_hook())

        if self.jacobian_norm:

            def create_hook():
                def hook(grad):
                    mask = torch.norm(grad, dim=1) > self.jacobian_norm
                    if torch.any(mask):
                        grad[mask] /= (torch.norm(grad, dim=1)[mask] + 1e-9).view(
                            (-1, 1)
                        ) * self.jacobian_norm
                    return grad

                return hook

            if self.obs_buf.requires_grad:
                self.obs_buf.register_hook(create_hook())
            if actions.requires_grad:
                actions.register_hook(create_hook())

        next_state = self.integrator.forward(
            self.model,
            self.state,
            self.sim_dt,
            self.sim_substeps,
            self.MM_caching_frequency,
        )

        # compute dynamics jacobians if requested
        if self.jacobian and not play:
            inputs = torch.cat((self.obs_buf.clone(), unscaled_actions.clone()), dim=1)
            inputs.requires_grad_(True)
            last_obs = inputs[:, : self.num_obs]
            act = inputs[:, self.num_obs :]
            self.set_state_act(last_obs, act)
            output = self.integrator.forward(
                self.model,
                self.state,
                self.sim_dt,
                self.sim_substeps,
                self.MM_caching_frequency,
                False,
            )
            outputs = self.observation_from_state(output)
            # TODO for some reason can only compute up to 11th dim
            jac = jacobian(outputs, inputs, max_out_dim=11)
            self.jacobians.append(jac.cpu().numpy())

        self.state = next_state
        self.sim_time += self.sim_dt

        self.progress_buf += 1
        self.num_frames += 1

        rew = self.calculate_reward(self.obs_buf, actions)
        self.obs_buf = self.observation_from_state(self.state)

        # Reset environments if agent has ended in a bad state based on heuristics
        termination = self.compute_termination(self.obs_buf, actions)

        # Reset environments if exseeded horizon
        truncation = self.progress_buf > self.episode_length - 1

        extras = {
            "obs_before_reset": self.obs_buf.clone(),
            "termination": termination,
            "truncation": truncation,
        }
        if hasattr(self, "primal"):
            extras.update({"primal": self.primal})
            
        if self.no_grad == False:
            extras.update(
                {
                    "contact_count": self.state.contact_count.clone().detach(),
                    "contact_forces": self.state.contact_f.clone()
                    .detach()
                    .view(self.num_envs, -1, 6),
                    "body_forces": self.state.body_f_s.clone()
                    .detach()
                    .view(self.num_envs, -1, 6),
                    "accelerations": self.state.body_a_s.clone()
                    .detach()
                    .view(self.num_envs, -1, 6),
                }
            )

            if self.jacobian and not play:
                extras.update({"jacobian": jac.cpu().numpy()})

        # reset all environments which have been terminated
        done = termination | truncation
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.render()

        return self.obs_buf, rew, done, extras

    def reset(self, env_ids=None, grads=False, force_reset=False):
        if grads:
            """
            This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
            It has to be called every time the algorithm starts an episode and it returns the observation vectors

            Note: force_reset resets all envs and is here for compatibility with rl_games
            """
            self.clear_grad()
            self.obs_buf = self.observation_from_state(self.state)
            return self.obs_buf

        if env_ids is None or force_reset:
            # reset all environments
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # Domain randomisation (must happen before we overwrite the
            # state so that the very first observations come from the
            # new physics parameters)
            self._apply_domain_randomization(env_ids)

            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            joint_q, joint_qd = self.static_init_func(env_ids)
            self.state.joint_q.view(self.num_envs, -1)[env_ids] = joint_q
            self.state.joint_qd.view(self.num_envs, -1)[env_ids] = joint_qd

            # randomization
            if self.stochastic_init:
                joint_q, joint_qd = self.stochastic_init_func(env_ids)
                self.state.joint_q.view(self.num_envs, -1)[env_ids] = joint_q
                self.state.joint_qd.view(self.num_envs, -1)[env_ids] = joint_qd

            # clear action
            self.state.joint_act.view(self.num_envs, -1)[env_ids, :] = 0.0

            self.progress_buf[env_ids] = 0

            self.obs_buf = self.observation_from_state(self.state)

        return self.obs_buf

    def reset_with_state(self, init_joint_q, init_joint_qd, env_ids=None):
        if env_ids is None:
            # reset all environemnts
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = init_joint_q.view(
                -1, self.num_joint_q
            )[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[
                env_ids, :
            ] = init_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()

            self.progress_buf[env_ids] = 0

            self.obs_buf = self.observation_from_state(self.state)

        return self.obs_buf

    def setup_visualizer(self, logdir=None):
        if self.visualize:
            filename = f"{logdir}/{self.__class__.__name__}_{self.num_envs}.usd"
            self.stage = Usd.Stage.CreateNew(filename)
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def render(self, mode="human"):
        if self.visualize:
            # Advance the simulation time
            self.render_time += self.dt

            # Playback-speed scaling: <1 → slower, >1 → faster.
            time_scale = getattr(self, "playback_speed", 1.0)
            if time_scale <= 0:
                time_scale = 1.0  # guard against invalid values
            scaled_time = self.render_time / time_scale

            # Update USD renderer with the scaled time-code
            self.renderer.update(self.state, scaled_time)

            render_interval = 1
            # if self.num_frames == render_interval:
            try:
                self.stage.Save()
            except Exception as e:
                print("USD save error:", e)

                self.num_frames -= render_interval

    def clear_grad(self, checkpoint=None):
        """cut off the gradient from the current state to previous states"""

        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint["joint_q"] = self.state.joint_q.clone()
                checkpoint["joint_qd"] = self.state.joint_qd.clone()
                checkpoint["joint_act"] = self.state.joint_act.clone()
                checkpoint["progress_buf"] = self.progress_buf.clone()

            self.state = self.model.state()
            self.state.joint_q = checkpoint["joint_q"]
            self.state.joint_qd = checkpoint["joint_qd"]
            self.state.joint_act = checkpoint["joint_act"]
            self.progress_buf = checkpoint["progress_buf"]

    def initialize_trajectory(self):
        """
        This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
        It has to be called every time the algorithm starts an episode and it returns the observation vectors
        """
        self.clear_grad()
        self.obs_buf = self.observation_from_state(self.state)
        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint["joint_q"] = self.state.joint_q.clone()
        checkpoint["joint_qd"] = self.state.joint_qd.clone()
        checkpoint["joint_act"] = self.state.joint_act.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()

        return checkpoint
    
    def rand_act(self):
        return torch.rand((self.num_envs, self.num_actions), device=self.device)*2.0 - 1.0

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def get_state(self):
        return self.state.joint_q.clone(), self.state.joint_qd.clone()
