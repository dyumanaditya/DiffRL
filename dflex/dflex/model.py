# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state.
"""

import math
import torch
import numpy as np
import copy

from typing import Tuple
from typing import List

Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]

from dflex.util import *

# shape geometry types
GEO_SPHERE = 0
GEO_BOX = 1
GEO_CAPSULE = 2
GEO_MESH = 3
GEO_SDF = 4
GEO_PLANE = 5
GEO_NONE = 6

# body joint types
JOINT_PRISMATIC = 0
JOINT_REVOLUTE = 1
JOINT_BALL = 2
JOINT_FIXED = 3
JOINT_FREE = 4


class Mesh:
    """Describes a triangle collision mesh for simulation

    Attributes:

        vertices (List[Vec3]): Mesh vertices
        indices (List[int]): Mesh indices
        I (Mat33): Inertia tensor of the mesh assuming density of 1.0 (around the center of mass)
        mass (float): The total mass of the body assuming density of 1.0
        com (Vec3): The center of mass of the body
    """

    def __init__(self, vertices: List[Vec3], indices: List[int]):
        """Construct a Mesh object from a triangle mesh

        The mesh center of mass and inertia tensor will automatically be
        calculated using a density of 1.0. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List of vertices in the mesh
            indices: List of triangle indices, 3 per-element
        """

        self.vertices = vertices
        self.indices = indices

        # compute com and inertia (using density=1.0)
        com = np.mean(vertices, 0)

        num_tris = int(len(indices) / 3)

        # compute signed inertia for each tetrahedron
        # formed with the interior point, using an order-2
        # quadrature: https://www.sciencedirect.com/science/article/pii/S0377042712001604#br000040

        weight = 0.25
        alpha = math.sqrt(5.0) / 5.0

        I = np.zeros((3, 3))
        mass = 0.0

        for i in range(num_tris):
            p = np.array(vertices[indices[i * 3 + 0]])
            q = np.array(vertices[indices[i * 3 + 1]])
            r = np.array(vertices[indices[i * 3 + 2]])

            mid = (com + p + q + r) / 4.0

            pcom = p - com
            qcom = q - com
            rcom = r - com

            Dm = np.matrix((pcom, qcom, rcom)).T
            volume = np.linalg.det(Dm) / 6.0

            # quadrature points lie on the line between the
            # centroid and each vertex of the tetrahedron
            quads = (
                mid + (p - mid) * alpha,
                mid + (q - mid) * alpha,
                mid + (r - mid) * alpha,
                mid + (com - mid) * alpha,
            )

            for j in range(4):
                # displacement of quadrature point from COM
                d = quads[j] - com

                I += weight * volume * (length_sq(d) * np.eye(3, 3) - np.outer(d, d))
                mass += weight * volume

        self.I = I
        self.mass = mass
        self.com = com


class State:
    """The State object holds all *time-varying* data for a model.

    Time-varying data includes particle positions, velocities, rigid body states, and
    anything that is output from the integrator as derived data, e.g.: forces.

    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`Model.state()` function.

    Attributes:

        particle_q (torch.Tensor): Tensor of particle positions
        particle_qd (torch.Tensor): Tensor of particle velocities

        joint_q (torch.Tensor): Tensor of joint coordinates
        joint_qd (torch.Tensor): Tensor of joint velocities
        joint_act (torch.Tensor): Tensor of joint actuation values

    """

    def __init__(self):
        self.particle_count = 0
        self.link_count = 0

    # def flatten(self):
    #     """Returns a list of Tensors stored by the state

    #     This function is intended to be used internal-only but can be used to obtain
    #     a set of all tensors owned by the state.
    #     """

    #     tensors = []

    #     # particles
    #     if (self.particle_count):
    #         tensors.append(self.particle_q)
    #         tensors.append(self.particle_qd)

    #     # articulations
    #     if (self.link_count):
    #         tensors.append(self.joint_q)
    #         tensors.append(self.joint_qd)
    #         tensors.append(self.joint_act)

    #     return tensors

    def flatten(self):
        """Returns a list of Tensors stored by the state

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the state.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if torch.is_tensor(value):
                tensors.append(value)

        return tensors

    def clone(self):
        s = State()
        for attr, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(s, attr, torch.clone(value))
            else:
                setattr(s, attr, copy.deepcopy(value))
        return s

    @staticmethod
    def _slice_by_env(flat, num_envs, env_ids):
        """
        Reshape a tensor laid out as [num_envs * per_env, …] so we can
        index environments, then flatten again.
        """
        if flat is None or not torch.is_tensor(flat):
            return flat
        if flat.numel() == 0 or flat.shape[0] % num_envs != 0:
            return flat  # globals stay unchanged

        per_env = flat.shape[0] // num_envs
        new_shape = (num_envs, per_env, *flat.shape[1:])
        return (flat.view(*new_shape)[env_ids]  # keep envs
                .reshape(-1, *flat.shape[1:]))  # back to flat

    def select_envs(self, num_envs: int, env_ids: torch.Tensor) -> "State":
        """
        Returns a *new* State containing only the environments in `env_ids`.

        Args
        ----
        num_envs   : how many environments the *original* batch held
        env_ids    : 1‑D LongTensor with the indices to keep, on any device
        """

        env_ids = env_ids.to(torch.long)
        assert env_ids.ndim == 1, "env_ids must be a 1‑D LongTensor"

        # deep‑copy Python containers & tensors first
        sub = self.clone()
        keep = len(env_ids)  # new batch size
        ratio = keep / num_envs

        # update the per‑env counters if they exist
        for name in ("particle_count", "link_count", "contact_count"):
            if hasattr(sub, name) and not torch.is_tensor(getattr(sub, name)):
                setattr(sub, name, int(getattr(sub, name) * ratio))

        # slice every tensor attribute
        for attr, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(sub, attr,
                        self._slice_by_env(val, num_envs, env_ids).clone())
            elif isinstance(val, list) and val and torch.is_tensor(val[0]):
                setattr(sub, attr,
                        [self._slice_by_env(v, num_envs, env_ids).clone() for v in val])

            # setattr(sub, attr, val)

        return sub


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Attributes:
        particle_q (torch.Tensor): Particle positions, shape [particle_count, 3], float
        particle_qd (torch.Tensor): Particle velocities, shape [particle_count, 3], float
        particle_mass (torch.Tensor): Particle mass, shape [particle_count], float
        particle_inv_mass (torch.Tensor): Particle inverse mass, shape [particle_count], float

        shape_transform (torch.Tensor): Rigid shape transforms, shape [shape_count, 7], float
        shape_body (torch.Tensor): Rigid shape body index, shape [shape_count], int
        shape_geo_type (torch.Tensor): Rigid shape geometry type, [shape_count], int
        shape_geo_src (torch.Tensor): Rigid shape geometry source, shape [shape_count], int
        shape_geo_scale (torch.Tensor): Rigid shape geometry scale, shape [shape_count, 3], float
        shape_materials (torch.Tensor): Rigid shape contact materials, shape [shape_count, 4], float

        spring_indices (torch.Tensor): Particle spring indices, shape [spring_count*2], int
        spring_rest_length (torch.Tensor): Particle spring rest length, shape [spring_count], float
        spring_stiffness (torch.Tensor): Particle spring stiffness, shape [spring_count], float
        spring_damping (torch.Tensor): Particle spring damping, shape [spring_count], float
        spring_control (torch.Tensor): Particle spring activation, shape [spring_count], float

        tri_indices (torch.Tensor): Triangle element indices, shape [tri_count*3], int
        tri_poses (torch.Tensor): Triangle element rest pose, shape [tri_count, 2, 2], float
        tri_activations (torch.Tensor): Triangle element activations, shape [tri_count], float

        edge_indices (torch.Tensor): Bending edge indices, shape [edge_count*2], int
        edge_rest_angle (torch.Tensor): Bending edge rest angle, shape [edge_count], float

        tet_indices (torch.Tensor): Tetrahedral element indices, shape [tet_count*4], int
        tet_poses (torch.Tensor): Tetrahedral rest poses, shape [tet_count, 3, 3], float
        tet_activations (torch.Tensor): Tetrahedral volumetric activations, shape [tet_count], float
        tet_materials (torch.Tensor): Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]

        body_X_cm (torch.Tensor): Rigid body center of mass (in local frame), shape [link_count, 7], float
        body_I_m (torch.Tensor): Rigid body inertia tensor (relative to COM), shape [link_count, 3, 3], float

        articulation_start (torch.Tensor): Articulation start offset, shape [num_articulations], int

        joint_q (torch.Tensor): Joint coordinate, shape [joint_coord_count], float
        joint_qd (torch.Tensor): Joint velocity, shape [joint_dof_count], float
        joint_type (torch.Tensor): Joint type, shape [joint_count], int
        joint_parent (torch.Tensor): Joint parent, shape [joint_count], int
        joint_X_pj (torch.Tensor): Joint transform in parent frame, shape [joint_count, 7], float
        joint_X_cm (torch.Tensor): Joint mass frame in child frame, shape [joint_count, 7], float
        joint_axis (torch.Tensor): Joint axis in child frame, shape [joint_count, 3], float
        joint_q_start (torch.Tensor): Joint coordinate offset, shape [joint_count], int
        joint_qd_start (torch.Tensor): Joint velocity offset, shape [joint_count], int

        joint_armature (torch.Tensor): Armature for each joint, shape [joint_count], float
        joint_target_ke (torch.Tensor): Joint stiffness, shape [joint_count], float
        joint_target_kd (torch.Tensor): Joint damping, shape [joint_count], float
        joint_target (torch.Tensor): Joint target, shape [joint_count], float

        particle_count (int): Total number of particles in the system
        joint_coord_count (int): Total number of joint coordinates in the system
        joint_dof_count (int): Total number of joint dofs in the system
        link_count (int): Total number of links in the system
        shape_count (int): Total number of shapes in the system
        tri_count (int): Total number of triangles in the system
        tet_count (int): Total number of tetrahedra in the system
        edge_count (int): Total number of edges in the system
        spring_count (int): Total number of springs in the system
        contact_count (int): Total number of contacts in the system

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    def __init__(self, adapter):
        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None

        self.shape_transform = None
        self.shape_body = None
        self.shape_geo_type = None
        self.shape_geo_src = None
        self.shape_geo_scale = None
        self.shape_materials = None

        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_control = None

        self.tri_indices = None
        self.tri_poses = None
        self.tri_activations = None

        self.edge_indices = None
        self.edge_rest_angle = None

        self.tet_indices = None
        self.tet_poses = None
        self.tet_activations = None
        self.tet_materials = None

        self.body_X_cm = None
        self.body_I_m = None

        self.articulation_start = None

        self.joint_q = None
        self.joint_qd = None
        self.joint_type = None
        self.joint_parent = None
        self.joint_X_pj = None
        self.joint_X_cm = None
        self.joint_axis = None
        self.joint_q_start = None
        self.joint_qd_start = None

        self.joint_armature = None
        self.joint_target_ke = None
        self.joint_target_kd = None
        self.joint_target = None

        self.particle_count = 0
        self.joint_coord_count = 0
        self.joint_dof_count = 0
        self.link_count = 0
        self.shape_count = 0
        self.tri_count = 0
        self.tet_count = 0
        self.edge_count = 0
        self.spring_count = 0
        self.contact_count = 0

        self.gravity = torch.tensor(
            (0.0, -9.8, 0.0), dtype=torch.float32, device=adapter
        )

        self.contact_distance = 0.1
        self.contact_ke = 1.0e3
        self.contact_kd = 0.0
        self.contact_kf = 1.0e3
        self.contact_mu = 0.5

        self.tri_ke = 100.0
        self.tri_ka = 100.0
        self.tri_kd = 10.0
        self.tri_kb = 100.0
        self.tri_drag = 0.0
        self.tri_lift = 0.0

        self.edge_ke = 100.0
        self.edge_kd = 0.0

        self.particle_radius = 0.1
        self.adapter = adapter

    def state(self) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.
        """

        s = State()

        s.particle_count = self.particle_count
        s.link_count = self.link_count

        # --------------------------------
        # dynamic state (input, output)

        # particles
        if self.particle_count:
            s.particle_q = torch.clone(self.particle_q)
            s.particle_qd = torch.clone(self.particle_qd)

        # articulations
        if self.link_count:
            s.joint_q = torch.clone(self.joint_q)
            s.joint_qd = torch.clone(self.joint_qd)
            s.joint_act = torch.zeros_like(self.joint_qd)

            s.joint_q.requires_grad = True
            s.joint_qd.requires_grad = True

        # --------------------------------
        # derived state (output only)

        if self.particle_count:
            s.particle_f = torch.empty_like(self.particle_qd, requires_grad=True)

        if self.link_count:
            # joints
            s.joint_qdd = torch.empty_like(self.joint_qd, requires_grad=True)
            s.joint_tau = torch.empty_like(self.joint_qd, requires_grad=True)
            s.joint_S_s = torch.empty(
                (self.joint_dof_count, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )

            # derived rigid body data (maximal coordinates)
            # body transform in world coordinates
            s.body_X_sc = torch.empty(
                (self.link_count, 7),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # relative transform from body frame to its center of mass
            s.body_X_sm = torch.empty(
                (self.link_count, 7),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # spatial inertia matrix 6x6 that encodes the center of mass, mass, and local 3x3 inertia matrix in one
            s.body_I_s = torch.empty(
                (self.link_count, 6, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # spatial body velocity in world coordinates
            s.body_v_s = torch.empty(
                (self.link_count, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # spatial body acceleration in world coordinates
            s.body_a_s = torch.empty(
                (self.link_count, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # external forces applied to the bodies in world coordinates
            s.body_f_s = torch.zeros(
                (self.link_count, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # contact forces
            s.contact_f = torch.zeros(
                (self.link_count, 6),
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            s.contact_count = torch.zeros(
                self.link_count,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            # s.body_ft_s = torch.zeros((self.link_count, 6), dtype=torch.float32, device=self.adapter, requires_grad=True)
            # s.body_f_ext_s = torch.zeros((self.link_count, 6), dtype=torch.float32, device=self.adapter, requires_grad=True)

        return s

    def alloc_mass_matrix(self):
        if self.link_count:
            # system matrices
            self.M = torch.zeros(
                self.M_size,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            self.J = torch.zeros(
                self.J_size,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            self.P = torch.empty(
                self.J_size,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )
            self.H = torch.empty(
                self.H_size,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = torch.zeros(
                self.H_size,
                dtype=torch.float32,
                device=self.adapter,
                requires_grad=True,
            )

    def flatten(self):
        """Returns a list of Tensors stored by the model

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the model.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if torch.is_tensor(value):
                tensors.append(value)

        return tensors

    # builds contacts
    def collide(self, state: State):
        """Constructs a set of contacts between rigid bodies and ground

        This method performs collision detection between rigid body vertices in the scene and updates
        the model's set of contacts stored as the following attributes:

            * **contact_body0**: Tensor of ints with first rigid body index
            * **contact_body1**: Tensor of ints with second rigid body index (currently always -1 to indicate ground)
            * **contact_point0**: Tensor of Vec3 representing contact point in local frame of body0
            * **contact_dist**: Tensor of float values representing the distance to maintain
            * **contact_material**: Tensor contact material indices

        Args:
            state: The state of the simulation at which to perform collision detection

        Note:
            Currently this method uses an 'all pairs' approach to contact generation that is
            state indepdendent. In the future this will change and will create a node in
            the computational graph to propagate gradients as a function of state.

        Todo:

            Only ground-plane collision is currently implemented. Since the ground is static
            it is acceptable to call this method once at initialization time.
        """

        body0 = []
        body1 = []
        point = []
        dist = []
        mat = []

        def add_contact(b0, b1, t, p0, d, m):
            body0.append(b0)
            body1.append(b1)
            point.append(transform_point(t, np.array(p0)))
            dist.append(d)
            mat.append(m)

        for i in range(self.shape_count):
            # transform from shape to body
            X_bs = transform_expand(self.shape_transform[i].tolist())

            geo_type = self.shape_geo_type[i].item()

            if geo_type == GEO_SPHERE:
                radius = self.shape_geo_scale[i][0].item()

                add_contact(self.shape_body[i], -1, X_bs, (0.0, 0.0, 0.0), radius, i)

            elif geo_type == GEO_CAPSULE:
                radius = self.shape_geo_scale[i][0].item()
                half_width = self.shape_geo_scale[i][1].item()

                add_contact(
                    self.shape_body[i], -1, X_bs, (-half_width, 0.0, 0.0), radius, i
                )
                add_contact(
                    self.shape_body[i], -1, X_bs, (half_width, 0.0, 0.0), radius, i
                )

            elif geo_type == GEO_BOX:
                edges = self.shape_geo_scale[i].tolist()

                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (-edges[0], -edges[1], -edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (edges[0], -edges[1], -edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (-edges[0], edges[1], -edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (edges[0], edges[1], -edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (-edges[0], -edges[1], edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (edges[0], -edges[1], edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i],
                    -1,
                    X_bs,
                    (-edges[0], edges[1], edges[2]),
                    0.0,
                    i,
                )
                add_contact(
                    self.shape_body[i], -1, X_bs, (edges[0], edges[1], edges[2]), 0.0, i
                )

            elif geo_type == GEO_MESH:
                mesh = self.shape_geo_src[i]
                scale = self.shape_geo_scale[i]

                for v in mesh.vertices:
                    p = (v[0] * scale[0], v[1] * scale[1], v[2] * scale[2])

                    add_contact(self.shape_body[i], -1, X_bs, p, 0.0, i)

        # send to torch
        self.contact_body0 = torch.tensor(body0, dtype=torch.int32, device=self.adapter)
        self.contact_body1 = torch.tensor(body1, dtype=torch.int32, device=self.adapter)
        self.contact_point0 = torch.tensor(
            point, dtype=torch.float32, device=self.adapter
        )
        self.contact_dist = torch.tensor(dist, dtype=torch.float32, device=self.adapter)
        self.contact_material = torch.tensor(
            mat, dtype=torch.int32, device=self.adapter
        )

        self.contact_count = len(body0)

    def clone(self) -> "Model":
        """
        Return a copy of this Model whose Tensor attributes have been .clone()'d
        so you can mutate one without affecting the other, while still preserving
        full gradient flow back to any original leaf Tensors.
        """
        # shallow‐copy the Python object (new.__dict__ refs same values)
        # new = copy.copy(self)
        m = Model(self.adapter)

        # for every attr that is a torch.Tensor, replace with a .clone()
        for attr, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(m, attr, torch.clone(value))
            else:
                setattr(m, attr, copy.deepcopy(value))

        return m

    @staticmethod
    def _slice_by_env(flat, num_envs, env_ids):
        """
        Utility: reshape a 1‑D or N‑D tensor whose first dimension is
        (num_envs * something) so that we can index on envs.
        """
        if flat is None or not torch.is_tensor(flat):
            return flat  # lists, dicts, scalars …
        if flat.numel() == 0 or flat.shape[0] % num_envs != 0:
            return flat  # tensor is global (gravity, KE, …)
        per_env = flat.shape[0] // num_envs
        new_shape = (num_envs, per_env, *flat.shape[1:])
        return (flat
                .view(*new_shape)  # [E, per_env, ...]
                [env_ids]  # keep only chosen envs
                .reshape(-1, *flat.shape[1:]))  # back to flat

    def select_envs(self, env_ids: torch.Tensor) -> "Model":
        """Return a *new* Model that contains *only* the environments in
        ``env_ids``.

        The current implementation assumes that the full model is a batch
        of identical environments that were concatenated one after another
        (which is exactly how the ModelBuilder assembles multi-env training
        scenes in dflex).  Under this assumption each environment
        contributes *fixed* amounts of particles, links, shapes … so the
        slicing rules are straightforward:

        1.  Figure out the *per-environment* counts for everything
            (particles, links, …) by simply dividing the global counts by
            ``articulation_count``.
        2.  For every *data* tensor whose first dimension is a concatenation
            of per-environment blocks (e.g. ``particle_q``) we gather the
            corresponding blocks and concatenate them.
        3.  For every *index* tensor that references the objects above
            (e.g. ``spring_indices`` which refers to ``particle_q``) we
            gather the blocks *and* shift the integer values so they point
            into the *new* (smaller) tensors.
        4.  Re-compute all *start/offset* arrays such that they are
            consistent with the reduced model.
        5.  Update the scalar counters (``particle_count``,
            ``link_count`` …) so they reflect the size of the new model.

        Note:  The implementation focuses on the attributes that are
        actually used at runtime by the simulator.  If you add new arrays
        to the Model in the future you may have to extend the slicing logic
        here as well.
        """

        env_ids = env_ids.to(torch.long)
        assert env_ids.ndim == 1, "env_ids must be 1-D LongTensor"

        # ---------- helper lambdas ----------

        num_envs_full = int(self.articulation_count)
        keep_envs = env_ids.tolist()
        n_keep = len(keep_envs)

        def _gather_block(tensor: torch.Tensor, per_env: int, dim: int = 0):
            """Gather *per-environment* blocks from a 1-D or N-D tensor.

            ``per_env`` is the size of one block along ``dim``.  Returns a
            tensor that contains *only* the requested environments and is
            contiguous in the original order of ``env_ids``.
            """
            if tensor is None or not torch.is_tensor(tensor):
                return tensor
            if per_env == 0:
                return tensor.clone()  # nothing to slice

            slices = []
            for e in keep_envs:
                start = e * per_env
                end = start + per_env
                slices.append(tensor.narrow(dim, start, per_env))
            return torch.cat(slices, dim=dim).clone()

        def _shift_index_block(index_tensor: torch.Tensor, per_env: int, offset: int):
            """Shift an *index* tensor so it becomes local to the sub-model.

            ``offset`` is the value that needs to be *subtracted* from all
            indices coming from the *current* environment.
            """
            if index_tensor is None or not torch.is_tensor(index_tensor):
                return index_tensor
            return (index_tensor - offset).clone()

        # ---------- per-environment breakdown ----------

        def _per_env(total: int):
            return int(total // num_envs_full) if total else 0

        n_particles_env = _per_env(self.particle_count)
        n_links_env = _per_env(self.link_count)
        n_shapes_env = _per_env(self.shape_count)
        n_springs_env = _per_env(self.spring_count)
        n_edges_env = _per_env(self.edge_count)
        n_tris_env = _per_env(self.tri_count)
        n_tets_env = _per_env(self.tet_count)
        n_muscles_env = _per_env(getattr(self, "muscle_count", 0))

        # n_dofs_env = _per_env(self.joint_dof_count)
        # n_coords_env = _per_env(self.joint_coord_count)
        # ------------------------------------------------------------
        # robust per-environment coordinate / dof counts
        # ------------------------------------------------------------
        if hasattr(self, "articulation_coord_start") and len(self.articulation_coord_start) > 1:
            n_coords_env = int(
                (self.articulation_coord_start[1] -
                 self.articulation_coord_start[0]).item())
        else:
            n_coords_env = self.joint_coord_count // num_envs_full

        if hasattr(self, "articulation_dof_start") and len(self.articulation_dof_start) > 1:
            n_dofs_env = int(
                (self.articulation_dof_start[1] -
                 self.articulation_dof_start[0]).item())
        else:
            n_dofs_env = self.joint_dof_count // num_envs_full

        # ---------- build new model ----------

        sub = Model(self.adapter)

        # copy over *global* (not per-env) configuration values verbatim
        for attr, val in self.__dict__.items():
            if torch.is_tensor(val):
                continue  # handled below
            if attr.endswith("_count"):
                continue  # handled later
            if attr in ("particle_q", "particle_qd", "particle_mass", "particle_inv_mass",
                        "spring_indices", "spring_rest_length", "spring_stiffness", "spring_damping", "spring_control",
                        "tri_indices", "tri_poses", "tri_activations",
                        "edge_indices", "edge_rest_angle",
                        "tet_indices", "tet_poses", "tet_activations", "tet_materials",
                        "shape_transform", "shape_body", "shape_geo_type", "shape_geo_src", "shape_geo_scale", "shape_materials",
                        "body_I_m", "joint_*", "articulation_*", "muscle_*"):
                # tensors handled later – skip placeholder text matches
                continue
            setattr(sub, attr, copy.deepcopy(val))

        # ---------- particles ----------

        sub.particle_q = _gather_block(self.particle_q, n_particles_env, 0)
        sub.particle_qd = _gather_block(self.particle_qd, n_particles_env, 0)
        sub.particle_mass = _gather_block(self.particle_mass, n_particles_env, 0)
        sub.particle_inv_mass = _gather_block(self.particle_inv_mass, n_particles_env, 0)

        # ---------- shapes ----------

        sub.shape_transform = _gather_block(self.shape_transform, n_shapes_env, 0)

        # shape_body needs *index* correction because it refers to link indices
        if self.shape_body is not None and self.shape_body.numel():
            shape_body_blocks = []
            for k, e in enumerate(keep_envs):  # k = 0…len(keep_envs)-1
                start = e * n_shapes_env
                end = start + n_shapes_env

                block = self.shape_body[start:end].clone()  # copy this env’s slice
                shift = (e - k) * n_links_env  # how far it moves left
                block -= shift  # re‑index links

                shape_body_blocks.append(block)

            sub.shape_body = torch.cat(shape_body_blocks, dim=0)
        else:
            sub.shape_body = (self.shape_body.clone()
                              if torch.is_tensor(self.shape_body) else None)

        sub.shape_geo_type = _gather_block(self.shape_geo_type, n_shapes_env, 0)
        # geo_src is a python list of meshes – slice it, deep-copy for safety
        if isinstance(self.shape_geo_src, list):
            sub.shape_geo_src = [copy.deepcopy(self.shape_geo_src[e * n_shapes_env:(e + 1) * n_shapes_env]) for e in keep_envs]
            sub.shape_geo_src = [item for sublist in sub.shape_geo_src for item in sublist]
        else:
            sub.shape_geo_src = copy.deepcopy(self.shape_geo_src)

        sub.shape_geo_scale = _gather_block(self.shape_geo_scale, n_shapes_env, 0)
        sub.shape_materials = _gather_block(self.shape_materials, n_shapes_env, 0)

        # ---------- springs ----------

        if self.spring_count:
            # scalar spring attributes
            sub.spring_rest_length = _gather_block(self.spring_rest_length, n_springs_env, 0)
            sub.spring_stiffness = _gather_block(self.spring_stiffness, n_springs_env, 0)
            sub.spring_damping = _gather_block(self.spring_damping, n_springs_env, 0)
            sub.spring_control = _gather_block(self.spring_control, n_springs_env, 0)

            # spring_indices is length 2*spring_count (ij pairs)
            spring_idx_blocks = []
            for e in keep_envs:
                start = e * n_springs_env * 2
                end = start + n_springs_env * 2
                idx_block = self.spring_indices[start:end].clone()
                idx_block -= e * n_particles_env  # make local
                spring_idx_blocks.append(idx_block)
            sub.spring_indices = torch.cat(spring_idx_blocks, dim=0)
        else:
            sub.spring_rest_length = sub.spring_stiffness = sub.spring_damping = sub.spring_control = sub.spring_indices = None

        # ---------- triangles ----------

        if self.tri_count:
            sub.tri_poses = _gather_block(self.tri_poses, n_tris_env, 0)
            sub.tri_activations = _gather_block(self.tri_activations, n_tris_env, 0)

            tri_idx_blocks = []
            for e in keep_envs:
                start = e * n_tris_env * 3
                end = start + n_tris_env * 3
                idx_block = self.tri_indices[start:end].clone()
                idx_block -= e * n_particles_env
                tri_idx_blocks.append(idx_block)
            sub.tri_indices = torch.cat(tri_idx_blocks, dim=0)
        else:
            sub.tri_indices = sub.tri_poses = sub.tri_activations = None

        # ---------- edges ----------

        if self.edge_count:
            sub.edge_rest_angle = _gather_block(self.edge_rest_angle, n_edges_env, 0)

            edge_idx_blocks = []
            for e in keep_envs:
                start = e * n_edges_env * 4
                end = start + n_edges_env * 4
                idx_block = self.edge_indices[start:end].clone()
                idx_block -= e * n_particles_env
                edge_idx_blocks.append(idx_block)
            sub.edge_indices = torch.cat(edge_idx_blocks, dim=0)
        else:
            sub.edge_indices = sub.edge_rest_angle = None

        # ---------- tetrahedra ----------

        if self.tet_count:
            sub.tet_poses = _gather_block(self.tet_poses, n_tets_env, 0)
            sub.tet_activations = _gather_block(self.tet_activations, n_tets_env, 0)
            sub.tet_materials = _gather_block(self.tet_materials, n_tets_env, 0)

            tet_idx_blocks = []
            for e in keep_envs:
                start = e * n_tets_env * 4
                end = start + n_tets_env * 4
                idx_block = self.tet_indices[start:end].clone()
                idx_block -= e * n_particles_env
                tet_idx_blocks.append(idx_block)
            sub.tet_indices = torch.cat(tet_idx_blocks, dim=0)
        else:
            sub.tet_indices = sub.tet_poses = sub.tet_activations = sub.tet_materials = None

        # ---------- links / articulations ----------

        # body_I_m is per-link (6x6)
        sub.body_I_m = _gather_block(self.body_I_m, n_links_env, 0)

        # joint arrays – all are per-link or per-dof, so gathering and/or shifting is needed
        sub.joint_type = _gather_block(self.joint_type, n_links_env, 0)

        # parent indices need shifting as well – parent link lies in same env
        parent_blocks = []
        for k, e in enumerate(keep_envs):  # k = position in the *kept* list
            start = e * n_links_env
            end = start + n_links_env

            block = self.joint_parent[start:end].clone()

            shift = (e - k) * n_links_env  # how much this env moves left
            mask = block >= 0  # -1 means “root”, keep as is
            block[mask] -= shift

            parent_blocks.append(block)

        sub.joint_parent = torch.cat(parent_blocks, dim=0)

        # transforms & axes
        sub.joint_X_pj = _gather_block(self.joint_X_pj, n_links_env, 0)
        sub.joint_X_cm = _gather_block(self.joint_X_cm, n_links_env, 0)
        sub.joint_axis = _gather_block(self.joint_axis, n_links_env, 0)

        # armature & target arrays (per-link)
        sub.joint_armature = _gather_block(self.joint_armature, n_dofs_env, 0)
        sub.joint_target_ke = _gather_block(self.joint_target_ke, n_links_env, 0)
        sub.joint_target_kd = _gather_block(self.joint_target_kd, n_links_env, 0)
        sub.joint_target = _gather_block(self.joint_target, n_coords_env, 0)
        # sub.joint_target = _gather_block(self.joint_target, n_links_env, 0)
        # sub.joint_target = self.joint_target.clone()

        sub.joint_limit_lower = _gather_block(self.joint_limit_lower, n_coords_env, 0)
        sub.joint_limit_upper = _gather_block(self.joint_limit_upper, n_coords_env, 0)
        sub.joint_limit_ke = _gather_block(self.joint_limit_ke, n_coords_env, 0)
        sub.joint_limit_kd = _gather_block(self.joint_limit_kd, n_coords_env, 0)

        # q / qd (state) – they will be filled by caller when it builds a State, but keep tensors consistent
        sub.joint_q = _gather_block(self.joint_q, n_coords_env, 0)
        sub.joint_qd = _gather_block(self.joint_qd, n_dofs_env, 0)

        # ----- offset arrays -----

        # articulation offsets (one per env + sentinel)
        # We simply build a new contiguous start array: 0, n_links_env, 2*n_links_env, ...
        new_artic_start = [i * n_links_env for i in range(n_keep + 1)]
        sub.articulation_joint_start = torch.tensor(new_artic_start, dtype=self.articulation_joint_start.dtype, device=self.adapter)
        sub.articulation_count = n_keep

        # joint_q_start / joint_qd_start  (per-joint sentinel list) —
        # rebuild by taking original pattern from first env and offsetting
        # NOTE: we assume identical per-link layout across envs.
        if n_links_env:
            base_q_start = self.joint_q_start[: n_links_env + 1].clone()
            base_qd_start = self.joint_qd_start[: n_links_env + 1].clone()

            q_start_blocks = []
            qd_start_blocks = []

            for k in range(n_keep):  # k = position in the *kept* batch
                q_shift = k * n_coords_env
                qd_shift = k * n_dofs_env

                if k == 0:
                    # keep the leading 0 for the first env
                    q_start_blocks.append(base_q_start + q_shift)
                    qd_start_blocks.append(base_qd_start + qd_shift)
                else:
                    # drop the first element to avoid duplication
                    q_start_blocks.append(base_q_start[1:] + q_shift)
                    qd_start_blocks.append(base_qd_start[1:] + qd_shift)

            sub.joint_q_start = torch.cat(q_start_blocks, dim=0).to(self.joint_q_start.dtype)
            sub.joint_qd_start = torch.cat(qd_start_blocks, dim=0).to(self.joint_qd_start.dtype)
        else:
            sub.joint_q_start = self.joint_q_start.clone()
            sub.joint_qd_start = self.joint_qd_start.clone()

        sub.joint_q_start = sub.joint_q_start.to(self.adapter)
        sub.joint_qd_start = sub.joint_qd_start.to(self.adapter)

        # articulation-level coordinate/dof starts (one per env)
        sub.articulation_coord_start = torch.tensor([i * n_coords_env for i in range(n_keep)], dtype=self.articulation_coord_start.dtype if hasattr(self, 'articulation_coord_start') else torch.int32, device=self.adapter)
        sub.articulation_dof_start = torch.tensor([i * n_dofs_env for i in range(n_keep)], dtype=self.articulation_dof_start.dtype, device=self.adapter)

        # Extract the relevant portions of mass matrix tensors from the main model
        if self.J is not None and self.M is not None and self.P is not None and self.H is not None and self.L is not None:
            # First, allocate the tensors with the correct sizes
            J_size_sub = 0
            M_size_sub = 0
            H_size_sub = 0
            
            for e in keep_envs:
                # Calculate sizes for each environment
                J_start = self.articulation_J_start[e]
                J_end = self.articulation_J_start[e + 1] if e + 1 < len(self.articulation_J_start) else self.J_size
                J_size_sub += J_end - J_start
                
                M_start = self.articulation_M_start[e]
                M_end = self.articulation_M_start[e + 1] if e + 1 < len(self.articulation_M_start) else self.M_size
                M_size_sub += M_end - M_start
                
                H_start = self.articulation_H_start[e]
                H_end = self.articulation_H_start[e + 1] if e + 1 < len(self.articulation_H_start) else self.H_size
                H_size_sub += H_end - H_start
            
            # Allocate tensors
            sub.J = torch.zeros(J_size_sub, dtype=torch.float32, device=self.adapter, requires_grad=True)
            sub.M = torch.zeros(M_size_sub, dtype=torch.float32, device=self.adapter, requires_grad=True)
            sub.P = torch.zeros(J_size_sub, dtype=torch.float32, device=self.adapter, requires_grad=True)  # Same size as J
            sub.H = torch.zeros(H_size_sub, dtype=torch.float32, device=self.adapter, requires_grad=True)
            sub.L = torch.zeros(H_size_sub, dtype=torch.float32, device=self.adapter, requires_grad=True)  # Same size as H
            
            # Now copy the data in a way that maintains gradient connections
            J_offset = 0
            M_offset = 0
            H_offset = 0
            
            for e in keep_envs:
                # J matrix blocks
                J_start = self.articulation_J_start[e]
                J_end = self.articulation_J_start[e + 1] if e + 1 < len(self.articulation_J_start) else self.J_size
                J_size = J_end - J_start
                sub.J[J_offset:J_offset + J_size] = torch.clone(self.J[J_start:J_end])
                
                # M matrix blocks
                M_start = self.articulation_M_start[e]
                M_end = self.articulation_M_start[e + 1] if e + 1 < len(self.articulation_M_start) else self.M_size
                M_size = M_end - M_start
                sub.M[M_offset:M_offset + M_size] = torch.clone(self.M[M_start:M_end])
                
                # P matrix blocks (same structure as J)
                P_start = self.articulation_J_start[e]  # P uses same start indices as J
                P_end = self.articulation_J_start[e + 1] if e + 1 < len(self.articulation_J_start) else self.J_size
                P_size = P_end - P_start
                sub.P[J_offset:J_offset + P_size] = torch.clone(self.P[P_start:P_end])
                
                # H matrix blocks
                H_start = self.articulation_H_start[e]
                H_end = self.articulation_H_start[e + 1] if e + 1 < len(self.articulation_H_start) else self.H_size
                H_size = H_end - H_start
                sub.H[H_offset:H_offset + H_size] = torch.clone(self.H[H_start:H_end])
                
                # L matrix blocks (same structure as H)
                L_start = self.articulation_H_start[e]  # L uses same start indices as H
                L_end = self.articulation_H_start[e + 1] if e + 1 < len(self.articulation_H_start) else self.H_size
                L_size = L_end - L_start
                sub.L[H_offset:H_offset + L_size] = torch.clone(self.L[L_start:L_end])
                
                # Update offsets
                J_offset += J_size
                M_offset += M_size
                H_offset += H_size
        else:
            # If mass matrices haven't been allocated yet, set to None
            sub.J = sub.M = sub.P = sub.H = sub.L = None

        # ---------- scalar counters ----------

        sub.particle_count = n_particles_env * n_keep
        sub.link_count = n_links_env * n_keep
        sub.shape_count = n_shapes_env * n_keep
        sub.spring_count = n_springs_env * n_keep
        sub.edge_count = n_edges_env * n_keep
        sub.tri_count = n_tris_env * n_keep
        sub.tet_count = n_tets_env * n_keep
        sub.joint_dof_count = n_dofs_env * n_keep
        sub.joint_coord_count = n_coords_env * n_keep
        sub.muscle_count = n_muscles_env * n_keep if hasattr(self, "muscle_count") else 0
        # sub.contact_count = _per_env(self.contact_count) * n_keep

        # ---------- contacts ----------
        n_contacts_env = _per_env(self.contact_count)

        if self.contact_count and n_contacts_env:
            # contact_body0 needs index shift
            body0_blocks = []
            for e in keep_envs:
                start = e * n_contacts_env
                end = start + n_contacts_env
                block = self.contact_body0[start:end].clone()
                mask = block >= 0
                block[mask] -= e * n_links_env
                body0_blocks.append(block)

            sub.contact_body0 = torch.cat(body0_blocks, dim=0)
            sub.contact_body1 = _gather_block(self.contact_body1, n_contacts_env, 0)
            sub.contact_point0 = _gather_block(self.contact_point0, n_contacts_env, 0)
            sub.contact_dist = _gather_block(self.contact_dist, n_contacts_env, 0)
            sub.contact_material = _gather_block(self.contact_material, n_contacts_env, 0)
            sub.contact_count = n_contacts_env * n_keep
        else:
            sub.contact_body0 = sub.contact_body1 = sub.contact_point0 = None
            sub.contact_dist = sub.contact_material = None
            sub.contact_count = 0

        # ---------- muscles ----------
        if hasattr(self, "muscle_count") and self.muscle_count and n_muscles_env:
            # muscle arrays
            sub.muscle_start = _gather_block(self.muscle_start, n_muscles_env, 0)
            sub.muscle_params = _gather_block(self.muscle_params, n_muscles_env, 0)
            sub.muscle_activation = _gather_block(self.muscle_activation, n_muscles_env, 0)

            # muscle_links and muscle_points need index shifting
            # muscle_start gives us the ranges
            muscle_links_blocks = []
            muscle_points_blocks = []
            for e in keep_envs:
                start_idx = e * n_muscles_env
                end_idx = (e + 1) * n_muscles_env
                start_offset = self.muscle_start[start_idx]
                end_offset = self.muscle_start[end_idx] if end_idx < len(self.muscle_start) else len(
                    self.muscle_links)

                # gather links and points for this env
                links_block = self.muscle_links[start_offset:end_offset].clone()
                points_block = self.muscle_points[start_offset:end_offset].clone()

                # shift link indices to be local
                mask = links_block >= 0
                links_block[mask] -= e * n_links_env

                muscle_links_blocks.append(links_block)
                muscle_points_blocks.append(points_block)

            sub.muscle_links = torch.cat(muscle_links_blocks, dim=0)
            sub.muscle_points = torch.cat(muscle_points_blocks, dim=0)
        else:
            sub.muscle_start = sub.muscle_params = sub.muscle_activation = None
            sub.muscle_links = sub.muscle_points = None

        # ----------- recompute global matrix sizes -----------
        if n_keep:
            J_rows_val = n_links_env * 6
            J_cols_val = n_dofs_env
            M_rows_val = n_links_env * 6
            H_rows_val = n_dofs_env

            sub.articulation_J_rows = torch.full((n_keep,), J_rows_val, dtype=self.articulation_J_rows.dtype if hasattr(self,"articulation_J_rows") else torch.int32, device=self.adapter)
            sub.articulation_J_cols = torch.full((n_keep,), J_cols_val, dtype=self.articulation_J_cols.dtype if hasattr(self,"articulation_J_cols") else torch.int32, device=self.adapter)
            sub.articulation_M_rows = torch.full((n_keep,), M_rows_val, dtype=self.articulation_M_rows.dtype if hasattr(self,"articulation_M_rows") else torch.int32, device=self.adapter)
            sub.articulation_H_rows = torch.full((n_keep,), H_rows_val, dtype=self.articulation_H_rows.dtype if hasattr(self,"articulation_H_rows") else torch.int32, device=self.adapter)

            J_env_size = J_rows_val * J_cols_val
            M_env_size = M_rows_val * M_rows_val
            H_env_size = H_rows_val * H_rows_val

            sub.articulation_J_start = torch.tensor([i*J_env_size for i in range(n_keep)], dtype=self.articulation_J_start.dtype if hasattr(self,"articulation_J_start") else torch.int32, device=self.adapter)
            sub.articulation_M_start = torch.tensor([i*M_env_size for i in range(n_keep)], dtype=self.articulation_M_start.dtype if hasattr(self,"articulation_M_start") else torch.int32, device=self.adapter)
            sub.articulation_H_start = torch.tensor([i*H_env_size for i in range(n_keep)], dtype=self.articulation_H_start.dtype if hasattr(self,"articulation_H_start") else torch.int32, device=self.adapter)
        else:
            sub.articulation_J_rows = sub.articulation_J_cols = None
            sub.articulation_M_rows = sub.articulation_H_rows = None
            sub.articulation_J_start = sub.articulation_M_start = sub.articulation_H_start = None

        # articulation_J_rows/cols already gathered; recompute sizes
        if sub.articulation_J_rows is not None and sub.articulation_J_cols is not None:
            sub.J_size = int((sub.articulation_J_rows * sub.articulation_J_cols).sum().item())
        else:
            sub.J_size = 0

        if sub.articulation_M_rows is not None:
            # M is square (rows x rows)
            sub.M_size = int((sub.articulation_M_rows * sub.articulation_M_rows).sum().item())
        else:
            sub.M_size = 0

        if sub.articulation_H_rows is not None:
            sub.H_size = int((sub.articulation_H_rows * sub.articulation_H_rows).sum().item())
        else:
            sub.H_size = 0

        # mass-matrix caches will be re-computed lazily when needed

        # for attr, val in self.__dict__.items():
        #     print(attr)
        #     print(val)
        #     if hasattr(sub, attr):
        #         print("SLICED")
        #         print(getattr(sub, attr))
        #         print()

        # sub.shape_body = self.shape_body
        # sub.joint_parent = self.joint_parent
        # sub.joint_q_start = self.joint_q_start
        # sub.joint_qd_start = self.joint_qd_start
        sub.gravity = self.gravity
        sub.contact_randomization = getattr(self, "contact_randomization", None)

        # sub.particle_q = self.particle_q
        # sub.particle_qd = self.particle_qd
        # sub.particle_mass = self.particle_mass
        # sub.particle_inv_mass = self.particle_inv_mass
        #
        # sub.shape_transform = self.shape_transform
        # sub.shape_body = self.shape_body
        # sub.shape_geo_type = self.shape_geo_type
        # sub.shape_geo_src = self.shape_geo_src
        # sub.shape_geo_scale = self.shape_geo_scale
        # sub.shape_materials = self.shape_materials
        #
        # sub.spring_indices = self.spring_indices
        # sub.spring_rest_length = self.spring_rest_length
        # sub.spring_stiffness = self.spring_stiffness
        # sub.spring_damping = self.spring_damping
        # sub.spring_control = self.spring_control
        #
        # sub.tri_indices = self.tri_indices
        # sub.tri_poses = self.tri_poses
        # sub.tri_activations = self.tri_activations
        #
        # sub.edge_indices = self.edge_indices
        # sub.edge_rest_angle = self.edge_rest_angle
        #
        # sub.tet_indices = self.tet_indices
        # sub.tet_poses = self.tet_poses
        # sub.tet_activations = self.tet_activations
        # sub.tet_materials = self.tet_materials
        #
        # sub.body_X_cm = self.body_X_cm
        # sub.body_I_m = self.body_I_m
        #
        # sub.articulation_start = self.articulation_start
        #
        # sub.joint_q = self.joint_q
        # sub.joint_qd = self.joint_qd
        # sub.joint_type = self.joint_type
        # sub.joint_parent = self.joint_parent
        # sub.joint_X_pj = self.joint_X_pj
        # sub.joint_X_cm = self.joint_X_cm
        # sub.joint_axis = self.joint_axis
        # sub.joint_q_start = self.joint_q_start
        # sub.joint_qd_start = self.joint_qd_start
        # sub.joint_armature = self.joint_armature
        # sub.joint_target_ke = self.joint_target_ke
        # sub.joint_target_kd = self.joint_target_kd
        # sub.joint_target = self.joint_target
        #
        # sub.particle_count = self.particle_count
        # sub.joint_coord_count = self.joint_coord_count
        # sub.joint_dof_count = self.joint_dof_count
        # sub.link_count = self.link_count
        # sub.shape_count = self.shape_count
        # sub.tri_count = self.tri_count
        # sub.tet_count = self.tet_count
        # sub.edge_count = self.edge_count
        # sub.spring_count = self.spring_count
        # sub.contact_count = self.contact_count
        #
        # sub.gravity = self.gravity
        # sub.contact_distance = self.contact_distance
        # sub.contact_ke = self.contact_ke
        # sub.contact_kd = self.contact_kd
        # sub.contact_kf = self.contact_kf
        # sub.contact_mu = self.contact_mu
        #
        # sub.tri_ke = self.tri_ke
        # sub.tri_ka = self.tri_ka
        # sub.tri_kd = self.tri_kd
        # sub.tri_kb = self.tri_kb
        # sub.tri_drag = self.tri_drag
        # sub.tri_lift = self.tri_lift
        #
        # sub.edge_ke = self.edge_ke
        # sub.edge_kd = self.edge_kd
        #
        # sub.particle_radius = self.particle_radius
        # sub.adapter = self.adapter
        #
        # sub.muscle_start = self.muscle_start
        # sub.muscle_params = self.muscle_params
        # sub.muscle_links = self.muscle_links
        # sub.muscle_points = self.muscle_points
        # sub.muscle_activation = self.muscle_activation
        #
        # sub.J_size = self.J_size
        # sub.M_size = self.M_size
        # sub.H_size = self.H_size
        #
        # sub.articulation_joint_start = self.articulation_joint_start
        # sub.articulation_J_start = self.articulation_J_start
        # sub.articulation_M_start = self.articulation_M_start
        # sub.articulation_H_start = self.articulation_H_start
        # sub.articulation_M_rows = self.articulation_M_rows
        # sub.articulation_H_rows = self.articulation_H_rows
        # sub.articulation_J_rows = self.articulation_J_rows
        # sub.articulation_J_cols = self.articulation_J_cols
        # sub.articulation_dof_start = self.articulation_dof_start
        # sub.articulation_coord_start = self.articulation_coord_start
        #
        # sub.joint_limit_lower = self.joint_limit_lower
        # sub.joint_limit_upper = self.joint_limit_upper
        # sub.joint_limit_ke = self.joint_limit_ke
        # sub.joint_limit_kd = self.joint_limit_kd
        #
        # sub.articulation_count = self.articulation_count
        # sub.muscle_count = self.muscle_count
        #
        # sub.geo_meshes = self.geo_meshes
        # sub.geo_sdfs = self.geo_sdfs
        # sub.ground = self.ground
        # sub.enable_tri_collisions = self.enable_tri_collisions
        #
        # sub.M = self.M
        # sub.J = self.J
        # sub.P = self.P
        # sub.H = self.H
        # sub.L = self.L
        #
        # sub.contact_body0 = self.contact_body0
        # sub.contact_body1 = self.contact_body1
        # sub.contact_point0 = self.contact_point0
        # sub.contact_dist = self.contact_dist
        # sub.contact_material = self.contact_material

        return sub

    def set_contact_randomization_params(self, ke_range, kd_range, kf_range, mu_range, target_bodies):
        self.contact_randomization = {
            "ke_range": ke_range,
            "kd_range": kd_range,
            "kf_range": kf_range,
            "mu_range": mu_range,
            "target_bodies": target_bodies
        }

    def randomize_contact_params(self) -> None:
        if getattr(self, "contact_randomization", None) is None:
            return

        ke_low, ke_high = self.contact_randomization["ke_range"]
        kd_low, kd_high = self.contact_randomization["kd_range"]
        kf_low, kf_high = self.contact_randomization["kf_range"]
        mu_low, mu_high = self.contact_randomization["mu_range"]
        dr_target_shape = self.contact_randomization["target_bodies"]

        dev = self.adapter  # CUDA / CPU device handle
        E = int(self.articulation_count)  # number of environments

        shapes_per_env = self.shape_count // E

        links_per_env = (
            self.link_count // E if self.link_count else 0
        )

        if dr_target_shape is not None and len(dr_target_shape):
            dr_target_shape = torch.as_tensor(dr_target_shape, device=dev, dtype=torch.long)
        else:
            dr_target_shape = None  # means “all shapes”

        # -------- helper --------------------------------------------------
        def _rand_vec(low, high):
            return torch.rand(E, device=dev) * (high - low) + low

        def _rand_scalar(low, high):  # one scalar
            return (torch.rand(1, device=dev) * (high - low) + low).item()

        ke = _rand_vec(ke_low, ke_high)
        kd = _rand_vec(kd_low, kd_high)
        kf = _rand_vec(kf_low, kf_high)
        mu = _rand_vec(mu_low, mu_high)

        ground_ke = _rand_scalar(ke_low, ke_high)
        ground_kd = _rand_scalar(kd_low, kd_high)
        ground_kf = _rand_scalar(kf_low, kf_high)
        ground_mu = _rand_scalar(mu_low, mu_high)

        # -------- write per‑shape materials ------------------------------
        with torch.no_grad():
            # view into storage: (E, Senv, 4)
            mat = self.shape_materials.view(E, shapes_per_env, 4)

            # (E, 1, 4) → broadcast along shapes_per_env
            param = torch.stack([ke, kd, kf, mu], dim=-1)[:, None, :]

            if dr_target_shape is None:
                # update *all* shapes: broadcast does the work
                mat.copy_(param.expand(-1, shapes_per_env, -1))
            else:
                # build mask of shapes whose *local* body‑id is in target_bodies
                local_body_ids = (self.shape_body
                                  .view(E, shapes_per_env)  # (E,Senv)
                                  - torch.arange(E, device=dev)[:, None] * links_per_env)

                mask = (local_body_ids[..., None] == dr_target_shape).any(-1)  # (E,Senv)

                # same shape as mat — boolean advanced indexing keeps mapping
                mat[mask] = param.expand(-1, shapes_per_env, -1)[mask]

            # -------- update ground / global contact tensors -------------
            self.contact_ke = ground_ke
            self.contact_kd = ground_kd
            self.contact_kf = ground_kf
            self.contact_mu = ground_mu

        # print("ground contact")
        # print(self.contact_ke)
        # print(self.contact_kd)
        # print(self.contact_kf)
        # print(self.contact_mu)
        #
        # print("materials")
        # print(self.shape_materials)
        # print()


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    is independent of PyTorch and builds the scene representation using
    standard Python data structures, this means it is not differentiable. Once :func:`finalize()`
    has been called the ModelBuilder transfers all data to Torch tensors and returns
    an object that may be used for simulation.

    Example:

        >>> import dflex as df
        >>>
        >>> builder = df.ModelBuilder()
        >>>
        >>> # anchor point (zero mass)
        >>> builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        >>>
        >>> # build chain
        >>> for i in range(1,10):
        >>>     builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        >>>     builder.add_spring(i-1, i, 1.e+3, 0.0, 0)
        >>>
        >>> # create model
        >>> model = builder.finalize()

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    def __init__(self):
        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []

        # shapes
        self.shape_transform = []
        self.shape_body = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_materials = []

        # geometry
        self.geo_meshes = []
        self.geo_sdfs = []

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activation = []
        self.muscle_links = []
        self.muscle_points = []

        # rigid bodies
        self.joint_parent = (
            []
        )  # index of the parent body                      (constant)
        self.joint_child = (
            []
        )  # index of the child body                       (constant)
        self.joint_axis = []  # joint axis in child joint frame               (constant)
        self.joint_X_pj = []  # frame of joint in parent                      (constant)
        self.joint_X_cm = []  # frame of child com (in child coordinates)     (constant)

        self.joint_q_start = []  # joint offset in the q array
        self.joint_qd_start = []  # joint offset in the qd array
        self.joint_type = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_target = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []

        self.joint_q = []  # generalized coordinates       (input)
        self.joint_qd = []  # generalized velocities        (input)
        self.joint_qdd = []  # generalized accelerations     (id,fd)
        self.joint_tau = []  # generalized actuation         (input)
        self.joint_u = []  # generalized total torque      (fd)

        self.body_mass = []
        self.body_inertia = []
        self.body_com = []

        self.articulation_start = []

    def add_articulation(self) -> int:
        """Add an articulation object, all subsequently added links (see: :func:`add_link`) will belong to this articulation object.
        Calling this method multiple times 'closes' any previous articulations and begins a new one.

        Returns:
            The index of the articulation
        """
        self.articulation_start.append(len(self.joint_type))
        return len(self.articulation_start) - 1

    # rigids, register a rigid body and return its index.
    def add_link(
        self,
        parent: int,
        X_pj: Transform,
        axis: Vec3,
        type: int,
        armature: float = 0.01,
        stiffness: float = 0.0,
        damping: float = 0.0,
        limit_lower: float = -1.0e3,
        limit_upper: float = 1.0e3,
        limit_ke: float = 100.0,
        limit_kd: float = 10.0,
        com: Vec3 = np.zeros(3),
        I_m: Mat33 = np.zeros((3, 3)),
        m: float = 0.0,
    ) -> int:
        """Adds a rigid body to the model.

        Args:
            parent: The index of the parent body
            X_pj: The location of the joint in the parent's local frame connecting this body
            axis: The joint axis
            type: The type of joint, should be one of: JOINT_PRISMATIC, JOINT_REVOLUTE, JOINT_BALL, JOINT_FIXED, or JOINT_FREE
            armature: Additional inertia around the joint axis
            stiffness: Spring stiffness that attempts to return joint to zero position
            damping: Spring damping that attempts to remove joint velocity
            com: The center of mass of the body w.r.t its origin
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass)
            m: The mass of the body

        Returns:
            The index of the body in the model

        Note:
            If the mass (m) is zero then the body is treated as kinematic with no dynamics

        """

        # joint data
        self.joint_type.append(type)
        self.joint_axis.append(np.array(axis))
        self.joint_parent.append(parent)
        self.joint_X_pj.append(X_pj)

        self.joint_target_ke.append(stiffness)
        self.joint_target_kd.append(damping)
        self.joint_limit_ke.append(limit_ke)
        self.joint_limit_kd.append(limit_kd)

        self.joint_q_start.append(len(self.joint_q))
        self.joint_qd_start.append(len(self.joint_qd))

        if type == JOINT_PRISMATIC:
            self.joint_q.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_target.append(0.0)
            self.joint_armature.append(armature)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_upper.append(limit_upper)

        elif type == JOINT_REVOLUTE:
            self.joint_q.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_target.append(0.0)
            self.joint_armature.append(armature)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_upper.append(limit_upper)

        elif type == JOINT_BALL:
            # quaternion
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(1.0)

            # angular velocity
            self.joint_qd.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_qd.append(0.0)

            # pd targets
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)

            self.joint_armature.append(armature)
            self.joint_armature.append(armature)
            self.joint_armature.append(armature)

            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(0.0)

            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(0.0)

        elif type == JOINT_FIXED:
            pass
        elif type == JOINT_FREE:
            # translation
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)

            # quaternion
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(1.0)

            # note armature for free joints should always be zero, better to modify the body inertia directly
            self.joint_armature.append(0.0)
            self.joint_armature.append(0.0)
            self.joint_armature.append(0.0)
            self.joint_armature.append(0.0)
            self.joint_armature.append(0.0)
            self.joint_armature.append(0.0)

            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)

            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)

            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)

            # joint velocities
            for i in range(6):
                self.joint_qd.append(0.0)

        self.body_inertia.append(np.zeros((3, 3)))
        self.body_mass.append(0.0)
        self.body_com.append(np.zeros(3))

        # return index of body
        return len(self.joint_type) - 1

    # muscles
    def add_muscle(
        self,
        links: List[int],
        positions: List[Vec3],
        f0: float,
        lm: float,
        lt: float,
        lmax: float,
        pen: float,
    ) -> float:
        """Adds a muscle-tendon activation unit

        Args:
            links: A list of link indices for each waypoint
            positions: A list of positions of each waypoint in the link's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        """

        n = len(links)

        self.muscle_start.append(len(self.muscle_links))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activation.append(0.0)

        for i in range(n):
            self.muscle_links.append(links[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start) - 1

    # shapes
    def add_shape_plane(
        self,
        plane: Vec4 = (0.0, 1.0, 0.0, 0.0),
        ke: float = 1.0e5,
        kd: float = 1000.0,
        kf: float = 1000.0,
        mu: float = 0.5,
    ):
        """Adds a plane collision shape

        Args:
            plane: The plane equation in form a*x + b*y + c*z + d = 0
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """
        self._add_shape(
            -1,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            GEO_PLANE,
            plane,
            None,
            0.0,
            ke,
            kd,
            kf,
            mu,
        )

    def add_shape_sphere(
        self,
        body,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        density: float = 1000.0,
        ke: float = 1.0e5,
        kd: float = 1000.0,
        kf: float = 1000.0,
        mu: float = 0.5,
    ):
        """Adds a sphere collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the sphere
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(
            body,
            pos,
            rot,
            GEO_SPHERE,
            (radius, 0.0, 0.0, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            mu,
        )

    def add_shape_box(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        density: float = 1000.0,
        ke: float = 1.0e5,
        kd: float = 1000.0,
        kf: float = 1000.0,
        mu: float = 0.5,
    ):
        """Adds a box collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            hx: The half-extents along the x-axis
            hy: The half-extents along the y-axis
            hz: The half-extents along the z-axis
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(
            body, pos, rot, GEO_BOX, (hx, hy, hz, 0.0), None, density, ke, kd, kf, mu
        )

    def add_shape_capsule(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        half_width: float = 0.5,
        density: float = 1000.0,
        ke: float = 1.0e5,
        kd: float = 1000.0,
        kf: float = 1000.0,
        mu: float = 0.5,
    ):
        """Adds a capsule collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the capsule
            half_width: The half length of the center cylinder along the x-axis
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(
            body,
            pos,
            rot,
            GEO_CAPSULE,
            (radius, half_width, 0.0, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            mu,
        )

    def add_shape_mesh(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        mesh: Mesh = None,
        scale: Vec3 = (1.0, 1.0, 1.0),
        density: float = 1000.0,
        ke: float = 1.0e5,
        kd: float = 1000.0,
        kf: float = 1000.0,
        mu: float = 0.5,
    ):
        """Adds a triangle mesh collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            mesh: The mesh object
            scale: Scale to use for the collider
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(
            body,
            pos,
            rot,
            GEO_MESH,
            (scale[0], scale[1], scale[2], 0.0),
            mesh,
            density,
            ke,
            kd,
            kf,
            mu,
        )

    def _add_shape(self, body, pos, rot, type, scale, src, density, ke, kd, kf, mu):
        self.shape_body.append(body)
        self.shape_transform.append(transform(pos, rot))
        self.shape_geo_type.append(type)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2]))
        self.shape_geo_src.append(src)
        self.shape_materials.append((ke, kd, kf, mu))

        (m, I) = self._compute_shape_mass(type, scale, src, density)

        self._update_body_mass(body, m, I, np.array(pos), np.array(rot))

    # particles
    def add_particle(self, pos: Vec3, vel: Vec3, mass: float) -> int:
        """Adds a single particle to the model

        Args:
            pos: The initial position of the particle
            vel: The initial velocity of the particle
            mass: The mass of the particle

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that does is not subject to dynamics.

        Returns:
            The index of the particle in the system
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)

        return len(self.particle_q) - 1

    def add_spring(self, i: int, j, ke: float, kd: float, control: float):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            control: The actuation level of the spring

        Note:
            The spring is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_q[i]
        q = self.particle_q[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

    def add_triangle(self, i: int, j: int, k: int) -> float:
        """Adds a trianglular FEM element between three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specfied on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        Todo:
            * Expose elastic paramters on a per-element basis

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = normalize(np.cross(qp, rp))
        e1 = normalize(qp)
        e2 = normalize(np.cross(n, e1))

        R = np.matrix((e1, e2))
        M = np.matrix((qp, rp))

        D = R * M.T
        inv_D = np.linalg.inv(D)

        area = np.linalg.det(D) / 2.0

        if area < 0.0:
            print("inverted triangle element")

        self.tri_indices.append((i, j, k))
        self.tri_poses.append(inv_D.tolist())
        self.tri_activations.append(0.0)

        return area

    def add_tetrahedron(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        k_mu: float = 1.0e3,
        k_lambda: float = 1.0e3,
        k_damp: float = 0.0,
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahdera are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configruation

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.matrix((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(self, i: int, j: int, k: int, l: int, rest: float = None):
        """Adds a bending edge element between four particles in the system.

        Bending elements are designed to be between two connected triangles. Then
        bending energy is based of [Bridson et al. 2002]. Bending stiffness is controlled
        by the `model.tri_kb` parameter.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            rest: The rest angle across the edge in radians, if not specified it will be computed

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counter clockwise
            winding: (i, k, l), (j, l, k).

        """
        # compute rest angle
        if rest == None:
            x1 = np.array(self.particle_q[i])
            x2 = np.array(self.particle_q[j])
            x3 = np.array(self.particle_q[k])
            x4 = np.array(self.particle_q[l])

            n1 = normalize(np.cross(x3 - x1, x4 - x1))
            n2 = normalize(np.cross(x4 - x2, x3 - x2))
            e = normalize(x4 - x3)

            d = np.clip(np.dot(n2, n1), -1.0, 1.0)

            angle = math.acos(d)
            sign = np.sign(np.dot(np.cross(n2, n1), e))

            rest = angle * sign

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)

    def add_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
    ):
        """Helper to create a regular planar cloth grid

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic

        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                g = np.array((x * cell_x, y * cell_y, 0.0))
                p = quat_rotate(rot, g) + pos
                m = mass

                if x == 0 and fix_left:
                    m = 0.0
                elif x == dim_x and fix_right:
                    m = 0.0
                elif y == 0 and fix_bottom:
                    m = 0.0
                elif y == dim_y and fix_top:
                    m = 0.0

                self.add_particle(p, vel, m)

                if x > 0 and y > 0:
                    if reverse_winding:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                        )

                        tri2 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

                    else:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        tri2 = (
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        # bending constraints, could create these explicitly for a grid but this
        # is a good test of the adjacency structure
        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        for k, e in adj.edges.items():
            # skip open edges
            if e.f0 == -1 or e.f1 == -1:
                continue

            self.add_edge(
                e.o0, e.o1, e.v0, e.v1
            )  # opposite 0, opposite 1, vertex 0, vertex 1

    def add_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: List[Vec3],
        indices: List[int],
        density: float,
        edge_callback=None,
        face_callback=None,
    ):
        """Helper to create a cloth model from a regular triangle mesh

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            edge_callback: A user callback when an edge is created
            face_callback: A user callback when a face is created

        Note:

            The mesh should be two manifold.
        """

        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        for i, v in enumerate(vertices):
            p = quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # triangles
        for t in range(num_tris):
            i = start_vertex + indices[t * 3 + 0]
            j = start_vertex + indices[t * 3 + 1]
            k = start_vertex + indices[t * 3 + 2]

            if face_callback:
                face_callback(i, j, k)

            area = self.add_triangle(i, j, k)

            # add area fraction to particles
            if area > 0.0:
                self.particle_mass[i] += density * area / 3.0
                self.particle_mass[j] += density * area / 3.0
                self.particle_mass[k] += density * area / 3.0

        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        # bend constraints
        for k, e in adj.edges.items():
            # skip open edges
            if e.f0 == -1 or e.f1 == -1:
                continue

            if edge_callback:
                edge_callback(e.f0, e.f1)

            self.add_edge(e.o0, e.o1, e.v0, e.v1)

    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrhedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = np.array((x * cell_x, y * cell_y, z * cell_z))
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2])

    def add_soft_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: List[Vec3],
        indices: List[int],
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
    ):
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions
            indices: A list of tetrahedron indices, 4 entries per-element
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
        """
        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        # add particles
        for v in vertices:
            p = quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):
            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if volume > 0.0:
                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add triangles
        for k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2])
            except np.linalg.LinAlgError:
                continue

    def compute_sphere_inertia(self, density: float, r: float) -> tuple:
        """Helper to compute mass and inertia of a sphere

        Args:
            density: The sphere density
            r: The sphere radius

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        v = 4.0 / 3.0 * math.pi * r * r * r

        m = density * v
        Ia = 2.0 / 5.0 * m * r * r

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_capsule_inertia(self, density: float, r: float, l: float) -> tuple:
        """Helper to compute mass and inertia of a capsule

        Args:
            density: The capsule density
            r: The capsule radius
            l: The capsule length (full width of the interior cylinder)

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        ms = density * (4.0 / 3.0) * math.pi * r * r * r
        mc = density * math.pi * r * r * l

        # total mass
        m = ms + mc

        # adapted from ODE
        Ia = mc * (0.25 * r * r + (1.0 / 12.0) * l * l) + ms * (
            0.4 * r * r + 0.375 * r * l + 0.25 * l * l
        )
        Ib = (mc * 0.5 + ms * 0.4) * r * r

        I = np.array([[Ib, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_box_inertia(
        self, density: float, w: float, h: float, d: float
    ) -> tuple:
        """Helper to compute mass and inertia of a box

        Args:
            density: The box density
            w: The box width along the x-axis
            h: The box height along the y-axis
            d: The box depth along the z-axis

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        v = w * h * d
        m = density * v

        Ia = 1.0 / 12.0 * m * (h * h + d * d)
        Ib = 1.0 / 12.0 * m * (w * w + d * d)
        Ic = 1.0 / 12.0 * m * (w * w + h * h)

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

        return (m, I)

    def _compute_shape_mass(self, type, scale, src, density):
        if density == 0:  # zero density means fixed
            return 0, np.zeros((3, 3))

        if type == GEO_SPHERE:
            return self.compute_sphere_inertia(density, scale[0])
        elif type == GEO_BOX:
            return self.compute_box_inertia(
                density, scale[0] * 2.0, scale[1] * 2.0, scale[2] * 2.0
            )
        elif type == GEO_CAPSULE:
            return self.compute_capsule_inertia(density, scale[0], scale[1] * 2.0)
        elif type == GEO_MESH:
            # todo: non-uniform scale of inertia tensor
            s = scale[0]  # eventually want to compute moment of inertia for mesh.
            return (density * src.mass * s * s * s, density * src.I * s * s * s * s * s)

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        if i == -1:
            return

        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:  # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = transform_inertia(
            self.body_mass[i], self.body_inertia[i], com_offset, quat_identity()
        ) + transform_inertia(m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

    # returns a (model, state) pair given the description
    def finalize(self, adapter: str) -> Model:
        """Convert this builder object to a concrete model for simulation.

        After building simulation elements this method should be called to transfer
        all data to PyTorch tensors ready for simulation.

        Args:
            adapter: The simulation adapter to use, e.g.: 'cpu', 'cuda'

        Returns:

            A model object.
        """

        # construct particle inv masses
        particle_inv_mass = []
        for m in self.particle_mass:
            if m > 0.0:
                particle_inv_mass.append(1.0 / m)
            else:
                particle_inv_mass.append(0.0)

        # -------------------------------------
        # construct Model (non-time varying) data

        m = Model(adapter)

        # ---------------------
        # particles

        # state (initial)
        m.particle_q = torch.tensor(
            self.particle_q, dtype=torch.float32, device=adapter
        )
        m.particle_qd = torch.tensor(
            self.particle_qd, dtype=torch.float32, device=adapter
        )

        # model
        m.particle_mass = torch.tensor(
            self.particle_mass, dtype=torch.float32, device=adapter
        )
        m.particle_inv_mass = torch.tensor(
            particle_inv_mass, dtype=torch.float32, device=adapter
        )

        # ---------------------
        # collision geometry

        m.shape_transform = torch.tensor(
            transform_flatten_list(self.shape_transform),
            dtype=torch.float32,
            device=adapter,
        )
        m.shape_body = torch.tensor(self.shape_body, dtype=torch.int32, device=adapter)
        m.shape_geo_type = torch.tensor(
            self.shape_geo_type, dtype=torch.int32, device=adapter
        )
        m.shape_geo_src = self.shape_geo_src
        m.shape_geo_scale = torch.tensor(
            self.shape_geo_scale, dtype=torch.float32, device=adapter
        )
        m.shape_materials = torch.tensor(
            self.shape_materials, dtype=torch.float32, device=adapter
        )

        # ---------------------
        # springs

        m.spring_indices = torch.tensor(
            self.spring_indices, dtype=torch.int32, device=adapter
        )
        m.spring_rest_length = torch.tensor(
            self.spring_rest_length, dtype=torch.float32, device=adapter
        )
        m.spring_stiffness = torch.tensor(
            self.spring_stiffness, dtype=torch.float32, device=adapter
        )
        m.spring_damping = torch.tensor(
            self.spring_damping, dtype=torch.float32, device=adapter
        )
        m.spring_control = torch.tensor(
            self.spring_control, dtype=torch.float32, device=adapter
        )

        # ---------------------
        # triangles

        m.tri_indices = torch.tensor(
            self.tri_indices, dtype=torch.int32, device=adapter
        )
        m.tri_poses = torch.tensor(self.tri_poses, dtype=torch.float32, device=adapter)
        m.tri_activations = torch.tensor(
            self.tri_activations, dtype=torch.float32, device=adapter
        )

        # ---------------------
        # edges

        m.edge_indices = torch.tensor(
            self.edge_indices, dtype=torch.int32, device=adapter
        )
        m.edge_rest_angle = torch.tensor(
            self.edge_rest_angle, dtype=torch.float32, device=adapter
        )

        # ---------------------
        # tetrahedra

        m.tet_indices = torch.tensor(
            self.tet_indices, dtype=torch.int32, device=adapter
        )
        m.tet_poses = torch.tensor(self.tet_poses, dtype=torch.float32, device=adapter)
        m.tet_activations = torch.tensor(
            self.tet_activations, dtype=torch.float32, device=adapter
        )
        m.tet_materials = torch.tensor(
            self.tet_materials, dtype=torch.float32, device=adapter
        )

        # -----------------------
        # muscles

        muscle_count = len(self.muscle_start)

        # close the muscle waypoint indices
        self.muscle_start.append(len(self.muscle_links))

        m.muscle_start = torch.tensor(
            self.muscle_start, dtype=torch.int32, device=adapter
        )
        m.muscle_params = torch.tensor(
            self.muscle_params, dtype=torch.float32, device=adapter
        )
        m.muscle_links = torch.tensor(
            self.muscle_links, dtype=torch.int32, device=adapter
        )
        m.muscle_points = torch.tensor(
            self.muscle_points, dtype=torch.float32, device=adapter
        )
        m.muscle_activation = torch.tensor(
            self.muscle_activation, dtype=torch.float32, device=adapter
        )

        # --------------------------------------
        # articulations

        # build 6x6 spatial inertia and COM transform
        body_X_cm = []
        body_I_m = []

        for i in range(len(self.body_inertia)):
            body_I_m.append(
                spatial_matrix_from_inertia(self.body_inertia[i], self.body_mass[i])
            )
            body_X_cm.append(transform(self.body_com[i], quat_identity()))

        m.body_I_m = torch.tensor(body_I_m, dtype=torch.float32, device=adapter)

        articulation_count = len(self.articulation_start)
        joint_coord_count = len(self.joint_q)
        joint_dof_count = len(self.joint_qd)

        # 'close' the start index arrays with a sentinel value
        self.joint_q_start.append(len(self.joint_q))
        self.joint_qd_start.append(len(self.joint_qd))
        self.articulation_start.append(len(self.joint_type))

        # calculate total size and offsets of Jacobian and mass matrices for entire system
        m.J_size = 0
        m.M_size = 0
        m.H_size = 0

        articulation_J_start = []
        articulation_M_start = []
        articulation_H_start = []

        articulation_M_rows = []
        articulation_H_rows = []
        articulation_J_rows = []
        articulation_J_cols = []

        articulation_dof_start = []
        articulation_coord_start = []

        for i in range(articulation_count):
            first_joint = self.articulation_start[i]
            last_joint = self.articulation_start[i + 1]

            first_coord = self.joint_q_start[first_joint]
            last_coord = self.joint_q_start[last_joint]

            first_dof = self.joint_qd_start[first_joint]
            last_dof = self.joint_qd_start[last_joint]

            joint_count = last_joint - first_joint
            dof_count = last_dof - first_dof
            coord_count = last_coord - first_coord

            articulation_J_start.append(m.J_size)
            articulation_M_start.append(m.M_size)
            articulation_H_start.append(m.H_size)
            articulation_dof_start.append(first_dof)
            articulation_coord_start.append(first_coord)

            # bit of data duplication here, but will leave it as such for clarity
            articulation_M_rows.append(joint_count * 6)
            articulation_H_rows.append(dof_count)
            articulation_J_rows.append(joint_count * 6)
            articulation_J_cols.append(dof_count)

            m.J_size += 6 * joint_count * dof_count
            m.M_size += 6 * joint_count * 6 * joint_count
            m.H_size += dof_count * dof_count

        m.articulation_joint_start = torch.tensor(
            self.articulation_start, dtype=torch.int32, device=adapter
        )

        # matrix offsets for batched gemm
        m.articulation_J_start = torch.tensor(
            articulation_J_start, dtype=torch.int32, device=adapter
        )
        m.articulation_M_start = torch.tensor(
            articulation_M_start, dtype=torch.int32, device=adapter
        )
        m.articulation_H_start = torch.tensor(
            articulation_H_start, dtype=torch.int32, device=adapter
        )

        m.articulation_M_rows = torch.tensor(
            articulation_M_rows, dtype=torch.int32, device=adapter
        )
        m.articulation_H_rows = torch.tensor(
            articulation_H_rows, dtype=torch.int32, device=adapter
        )
        m.articulation_J_rows = torch.tensor(
            articulation_J_rows, dtype=torch.int32, device=adapter
        )
        m.articulation_J_cols = torch.tensor(
            articulation_J_cols, dtype=torch.int32, device=adapter
        )

        m.articulation_dof_start = torch.tensor(
            articulation_dof_start, dtype=torch.int32, device=adapter
        )
        m.articulation_coord_start = torch.tensor(
            articulation_coord_start, dtype=torch.int32, device=adapter
        )

        # state (initial)
        m.joint_q = torch.tensor(self.joint_q, dtype=torch.float32, device=adapter)
        m.joint_qd = torch.tensor(self.joint_qd, dtype=torch.float32, device=adapter)

        # model
        m.joint_type = torch.tensor(self.joint_type, dtype=torch.int32, device=adapter)
        m.joint_parent = torch.tensor(
            self.joint_parent, dtype=torch.int32, device=adapter
        )
        m.joint_X_pj = torch.tensor(
            transform_flatten_list(self.joint_X_pj), dtype=torch.float32, device=adapter
        )
        m.joint_X_cm = torch.tensor(
            transform_flatten_list(body_X_cm), dtype=torch.float32, device=adapter
        )
        m.joint_axis = torch.tensor(
            self.joint_axis, dtype=torch.float32, device=adapter
        )
        m.joint_q_start = torch.tensor(
            self.joint_q_start, dtype=torch.int32, device=adapter
        )
        m.joint_qd_start = torch.tensor(
            self.joint_qd_start, dtype=torch.int32, device=adapter
        )

        # dynamics properties
        m.joint_armature = torch.tensor(
            self.joint_armature, dtype=torch.float32, device=adapter
        )

        m.joint_target = torch.tensor(
            self.joint_target, dtype=torch.float32, device=adapter
        )
        m.joint_target_ke = torch.tensor(
            self.joint_target_ke, dtype=torch.float32, device=adapter
        )
        m.joint_target_kd = torch.tensor(
            self.joint_target_kd, dtype=torch.float32, device=adapter
        )

        m.joint_limit_lower = torch.tensor(
            self.joint_limit_lower, dtype=torch.float32, device=adapter
        )
        m.joint_limit_upper = torch.tensor(
            self.joint_limit_upper, dtype=torch.float32, device=adapter
        )
        m.joint_limit_ke = torch.tensor(
            self.joint_limit_ke, dtype=torch.float32, device=adapter
        )
        m.joint_limit_kd = torch.tensor(
            self.joint_limit_kd, dtype=torch.float32, device=adapter
        )

        # counts
        m.particle_count = len(self.particle_q)

        m.articulation_count = articulation_count
        m.joint_coord_count = joint_coord_count
        m.joint_dof_count = joint_dof_count
        m.muscle_count = muscle_count

        m.link_count = len(self.joint_type)
        m.shape_count = len(self.shape_geo_type)
        m.tri_count = len(self.tri_poses)
        m.tet_count = len(self.tet_poses)
        m.edge_count = len(self.edge_rest_angle)
        m.spring_count = len(self.spring_rest_length)
        m.contact_count = 0

        # store refs to geometry
        m.geo_meshes = self.geo_meshes
        m.geo_sdfs = self.geo_sdfs

        # enable ground plane
        m.ground = True
        m.enable_tri_collisions = False

        m.gravity = torch.tensor((0.0, -9.8, 0.0), dtype=torch.float32, device=adapter)

        # allocate space for mass / jacobian matrices
        m.alloc_mass_matrix()

        return m
