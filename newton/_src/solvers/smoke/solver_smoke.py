# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import GeoType, ShapeFlags
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from . import kernels as k

__all__ = ["SolverSmoke"]

_NEVER_EMITTED_AGE = 1.0e30
"""Age [s] of a tracer that has not been emitted yet; larger than any lifetime."""

_TILE = 8
_TILE_CELLS = _TILE**3
_MAX_COLLIDER_TILES = 4
"""Bricks activated per axis around one moving collider."""


@wp.kernel
def _record_rebuild_status(status: wp.array[wp.uint32], accumulated: wp.array[wp.uint32]):
    """Retain and report rebuild failures without a host synchronization."""
    new_status = status[0] & ~accumulated[0]
    if new_status != wp.uint32(0):
        accumulated[0] = accumulated[0] | status[0]
        wp.printf("Warning: SolverSmoke sparse grid rebuild failed with status %u.\n", status[0])


def _csr_by_world(worlds: np.ndarray, world_count: int, include_global: bool) -> tuple[np.ndarray, np.ndarray]:
    """Group item indices by world. Global items (world -1) join every world, or only world 0."""
    start = [0]
    items: list[int] = []
    for world in range(world_count):
        members = np.flatnonzero(worlds == world)
        if include_global or world == 0:
            members = np.union1d(members, np.flatnonzero(worlds < 0))
        items.extend(int(m) for m in members)
        start.append(len(items))
    return np.asarray(start, dtype=np.int32), np.asarray(items, dtype=np.int32)


class SolverSmoke(SolverBase):
    """Eulerian smoke on a sparse, staggered grid, driven by emitters and pushed by colliders.

    .. experimental::

        :class:`SolverSmoke`, its :class:`Config`, and the ``smoke:*`` custom attributes may
        change without the normal deprecation period.

    The solver follows the stable-fluids method: semi-Lagrangian (RK2) advection
    of velocity, smoke density, and temperature; buoyancy and vorticity
    confinement; and a red-black SOR pressure projection that makes the air
    incompressible. Space is Z up and unbounded except for the ground plane at
    :attr:`Config.ground_height`.

    **Sparse grid.** Cells are stored in bricks of 8 x 8 x 8, and only bricks
    near live tracers, active emitters, moving colliders, or existing smoke and
    moving air are active. The active set is rebuilt in place every step within
    a fixed capacity (:attr:`Config.max_active_cell_count`), so :meth:`step`
    needs no host synchronization; call :meth:`check_sparse_grid_rebuild_status`
    to detect overflow. Outside the active bricks the air is still, clean, and at
    ambient pressure.

    **Worlds.** Each world has its own grid, so smoke never crosses worlds.
    Tracers and emitters belong to their builder world (global ones to world 0);
    global colliders act in every world.

    **Emitters.** Smoke sources are entries of the ``smoke:emitter`` custom
    frequency (see :meth:`register_custom_attributes`). Their position,
    velocity, and emission rates are :class:`~newton.Control` attributes, so they
    can follow anything, such as the contact point of a skidding tire.

    **Colliders.** Shapes with :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` block
    the air and drag it with their velocity. By default they come from the
    solver's model; :meth:`setup_collider` reads them from another model
    instead. Collider bodies are posed from ``state_in.body_q`` and
    ``state_in.body_qd``. Coupling is one way: smoke never pushes bodies.
    Planes and heightfields are ignored; the ground plane is built in. A moving
    collider keeps the air around it active, up to four bricks per axis.

    **Tracers.** The model's particles are a pool of tracers carried by the air
    for rendering. Emitters spawn them, reusing the oldest slot of their world
    when the pool is full. A tracer's age is the ``smoke:particle_age`` state
    attribute; a tracer whose age reaches :attr:`Config.tracer_lifetime` is
    inactive and no longer keeps the grid active.
    :meth:`sample_render_grains` and :meth:`update_render_grains` draw many more
    render points than there are tracers.

    Limitations: each world's grid spans about ``2**19 * voxel_size`` along x;
    no two-way coupling; not differentiable; tracer emission order is not
    deterministic.

    Args:
        model: Model holding the tracer particles and the ``smoke:*`` attributes.
        config: Solver configuration. ``None`` uses the defaults.
    """

    EMITTER_FREQUENCY = "smoke:emitter"
    """Custom frequency of smoke emitters."""

    @dataclass
    class Config:
        """Configuration for :class:`SolverSmoke`."""

        voxel_size: float = 0.05
        """Edge length of a grid cell [m]."""
        ground_height: float = 0.0
        """Height of the ground plane [m]."""
        max_active_cell_count: int = 1 << 21
        """Capacity of the sparse grid in cells across all worlds, rounded up to whole 8 x 8 x 8 bricks."""
        max_lower_node_count: int = -1
        """NanoVDB lower-node capacity of the brick index. ``-1`` picks a default from the world count."""
        max_upper_node_count: int = -1
        """NanoVDB upper-node capacity of the brick index. ``-1`` picks a default from the world count."""
        activation_padding: int = 2
        """Cells of air kept active around tracers, emitters, and moving colliders."""
        activation_threshold: float = 1.0e-3
        """Smoke density or temperature [K] above which a brick and its neighbors stay active."""
        activation_speed: float = 0.05
        """Air speed [m/s] above which a brick stays active, without activating its neighbors."""
        density_weight: float = 0.05
        """Downward acceleration per unit of smoke density [m/s²]."""
        temperature_lift: float = 4.0
        """Upward acceleration per kelvin above ambient [m/(s²·K)]."""
        density_decay_rate: float = 0.15
        """Exponential decay rate of smoke density [1/s]."""
        temperature_decay_rate: float = 0.4
        """Exponential cooling rate toward ambient [1/s]."""
        velocity_damping: float = 0.1
        """Exponential damping rate of the air velocity [1/s]."""
        vorticity_confinement: float = 2.0
        """Strength of vorticity confinement where there is smoke. Zero disables it."""
        pressure_iterations: int = 30
        """Red-black SOR iterations per step. The pressure is warm-started across steps."""
        pressure_relaxation: float = 1.7
        """SOR over-relaxation factor, in ``(0, 2)``."""
        emitter_velocity_coupling: float = 0.5
        """How strongly an active emitter pulls nearby air toward its velocity, in ``[0, 1]``."""
        tracer_lifetime: float = 4.0
        """Time [s] after which a tracer becomes inactive."""
        tracer_rate_scale: float = 60.0
        """Tracers emitted per second for each unit of ``smoke:emitter_rate`` [1/s per (1/s)]."""
        max_tracers_per_emitter_step: int = 64
        """Upper bound on the tracers one emitter spawns in one step."""
        tracer_spread_rate: float = 0.05
        """Growth rate [m/s] of a tracer's puff radius, used to confine its render grains."""

    def __init__(self, model: Model, config: Config | None = None):
        super().__init__(model)
        self.config = config if config is not None else SolverSmoke.Config()
        cfg = self.config
        if cfg.voxel_size <= 0.0:
            raise ValueError(f"SolverSmoke.Config.voxel_size must be positive, got {cfg.voxel_size}.")
        if not 0.0 < cfg.pressure_relaxation < 2.0:
            raise ValueError(
                f"SolverSmoke.Config.pressure_relaxation must be in (0, 2), got {cfg.pressure_relaxation}."
            )
        if cfg.activation_padding < 0:
            raise ValueError(f"SolverSmoke.Config.activation_padding must be >= 0, got {cfg.activation_padding}.")

        device = self.device
        h = float(cfg.voxel_size)
        self._h = h
        self._world_count = max(int(model.world_count), 1)
        world_count = self._world_count

        # An activation box may span at most three bricks per axis.
        self._padding = cfg.activation_padding * h
        reach_limit = _TILE * h - self._padding
        self._check_reach(model.particle_radius, reach_limit, "particle radius")

        # Tracers, stored contiguously by world.
        particle_world = np.zeros(model.particle_count, dtype=np.int32)
        if model.particle_count > 0:
            particle_world = np.maximum(model.particle_world.numpy(), 0).astype(np.int32)
            if np.any(np.diff(particle_world) < 0) or np.any(particle_world >= world_count):
                raise ValueError("SolverSmoke requires tracer particles to be stored contiguously by world.")
        world_tracer_start = np.searchsorted(particle_world, np.arange(world_count + 1)).astype(np.int32)
        self._particle_world = wp.array(particle_world, dtype=wp.int32, device=device)
        self._world_tracer_start = wp.array(world_tracer_start, dtype=wp.int32, device=device)

        # Emitters, grouped by world.
        smoke_ns = getattr(model, "smoke", None)
        radius = getattr(smoke_ns, "emitter_radius", None)
        self._emitter_count = 0 if radius is None else int(radius.shape[0])
        if self._emitter_count > 0:
            self._check_reach(radius, reach_limit, "smoke:emitter_radius")
            start, items = _csr_by_world(smoke_ns.emitter_world.numpy(), world_count, include_global=False)
        else:
            start, items = np.zeros(world_count + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)
        self._world_emitter_start = wp.array(start, dtype=wp.int32, device=device)
        self._world_emitters = wp.array(items, dtype=wp.int32, device=device)

        # Sparse brick index: a rebuildable index volume whose voxels are bricks.
        max_tiles = max(world_count, math.ceil(cfg.max_active_cell_count / _TILE_CELLS))
        self._max_tiles = max_tiles
        if cfg.max_lower_node_count > 0:
            lower = cfg.max_lower_node_count
        else:
            lower = min(max_tiles, 16 * world_count + 16)
        upper = cfg.max_upper_node_count if cfg.max_upper_node_count > 0 else min(lower, 2 * world_count + 2)
        anchors = wp.array(
            [wp.vec3i(w * int(k.WORLD_TILE_STRIDE), 0, 0) for w in range(world_count)], dtype=wp.vec3i, device=device
        )
        self._rebuild_status = wp.zeros(1, dtype=wp.uint32, device=device)
        self._accumulated_status = wp.zeros(1, dtype=wp.uint32, device=device)
        self._volume = wp.Volume.allocate_by_voxels(
            voxel_points=anchors,
            voxel_size=_TILE * h,
            device=device,
            rebuildable=True,
            max_active_voxels=max_tiles,
            max_leaf_nodes=max_tiles,
            max_lower_nodes=lower,
            max_upper_nodes=upper,
            status=self._rebuild_status,
        )
        self._tile_coord = wp.full(max_tiles, k.INVALID_TILE, dtype=wp.vec3i, device=device)
        self._old_tile_coord = wp.full(max_tiles, k.INVALID_TILE, dtype=wp.vec3i, device=device)
        self._neighbor = wp.full((max_tiles, 27), -1, dtype=wp.int32, device=device)
        self._grid = k.SparseGrid()
        self._grid.volume = self._volume.id
        self._grid.tile_coord = self._tile_coord
        self._grid.neighbor = self._neighbor
        self._grid.origin = wp.vec3(0.0, 0.0, cfg.ground_height)
        self._grid.h = h
        self._launch(
            k.assign_tile_coords_kernel,
            world_count,
            [self._volume.id, anchors, wp.ones(world_count, dtype=wp.int32, device=device)],
            [self._tile_coord],
        )
        self._launch(k.build_neighbors_kernel, (max_tiles, 27), [self._grid], [self._neighbor])

        cells = (max_tiles, _TILE, _TILE, _TILE)
        face_shapes = (
            (max_tiles, _TILE + 1, _TILE, _TILE),
            (max_tiles, _TILE, _TILE + 1, _TILE),
            (max_tiles, _TILE, _TILE, _TILE + 1),
        )

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=device)

        self._u, self._v, self._w = (zeros(s) for s in face_shapes)
        self._u_tmp, self._v_tmp, self._w_tmp = (zeros(s) for s in face_shapes)
        self._density = zeros(cells)
        self._temperature = zeros(cells)
        self._pressure = zeros(cells)
        self._divergence = zeros(cells)
        self._scalar_tmp = zeros(cells)
        self._solid = zeros(cells, dtype=wp.int32)
        self._solid_velocity = zeros(cells, dtype=wp.vec3)
        self._center_velocity = zeros(cells, dtype=wp.vec3)
        self._curl = zeros(cells, dtype=wp.vec3)
        self._curl_magnitude = zeros(cells)
        self._force = zeros(cells, dtype=wp.vec3)
        self._content = zeros(max_tiles, dtype=wp.int32)
        self._tracer_active = zeros(model.particle_count, dtype=wp.int32)
        self._emitter_active = zeros(self._emitter_count, dtype=wp.int32)

        self._seed = wp.zeros(1, dtype=wp.int32, device=device)
        self._tracer_head = wp.zeros(world_count, dtype=wp.int32, device=device)
        self._default_control: Control | None = None
        self._grain_seed = 0

        self.setup_collider()

    @staticmethod
    def _check_reach(radii: wp.array | None, limit: float, name: str):
        if radii is None or radii.shape[0] == 0:
            return
        largest = float(radii.numpy().max())
        if largest > limit:
            raise ValueError(
                f"SolverSmoke supports a {name} of at most 8 * voxel_size minus the activation padding "
                f"({limit:.4g} m), got {largest:.4g} m."
            )

    def _launch(self, kernel, dim, inputs, outputs=()):
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=list(outputs), device=self.device)

    # ------------------------------------------------------------------
    # Custom attributes
    # ------------------------------------------------------------------

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the ``smoke`` custom frequency and attributes.

        Add emitters with :meth:`ModelBuilder.add_custom_values`, for example
        ``builder.add_custom_values(**{"smoke:emitter_radius": 0.05})``. An
        emitter belongs to the builder's current world.

        Attributes registered on Model (per emitter):
            - ``smoke:emitter_radius``: Radius of the emitter's linear falloff [m].
            - ``smoke:emitter_world``: World of the emitter, filled from the builder's current world.

        Attributes registered on Control (per emitter):
            - ``smoke:emitter_position``: Emitter center [m].
            - ``smoke:emitter_velocity``: Velocity the emitter gives nearby air [m/s].
            - ``smoke:emitter_rate``: Smoke density added per second at the center [1/s].
              The emitter is off while this and the heat rate are zero.
            - ``smoke:emitter_heat_rate``: Temperature added per second at the center [K/s].

        Attributes registered on State (per particle):
            - ``smoke:particle_age``: Tracer age since emission [s]. Tracers that were
              never emitted start at a very large age, so they are inactive.

        Args:
            builder: Model builder to register the attributes on.
        """
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="emitter", namespace="smoke"))

        def emitter_attribute(name, dtype, default, assignment, references=None):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=cls.EMITTER_FREQUENCY,
                    assignment=assignment,
                    dtype=dtype,
                    default=default,
                    namespace="smoke",
                    references=references,
                )
            )

        emitter_attribute("emitter_radius", wp.float32, 0.05, Model.AttributeAssignment.MODEL)
        emitter_attribute("emitter_world", wp.int32, -1, Model.AttributeAssignment.MODEL, references="world")
        emitter_attribute("emitter_position", wp.vec3, wp.vec3(0.0), Model.AttributeAssignment.CONTROL)
        emitter_attribute("emitter_velocity", wp.vec3, wp.vec3(0.0), Model.AttributeAssignment.CONTROL)
        emitter_attribute("emitter_rate", wp.float32, 0.0, Model.AttributeAssignment.CONTROL)
        emitter_attribute("emitter_heat_rate", wp.float32, 0.0, Model.AttributeAssignment.CONTROL)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="particle_age",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=_NEVER_EMITTED_AGE,
                namespace="smoke",
            )
        )

    # ------------------------------------------------------------------
    # Colliders
    # ------------------------------------------------------------------

    def setup_collider(self, *, model: Model | None = None, shape_ids: list[int] | None = None) -> None:
        """Choose the shapes that block the air.

        Args:
            model: Model to read collider shapes and body properties from. ``None``
                uses the solver's model. The bodies are posed from ``state_in.body_q``
                and ``state_in.body_qd`` in :meth:`step`, so those arrays must index
                this model's bodies, and its worlds must match the solver's model.
            shape_ids: Shapes of ``model`` to use. ``None`` selects every shape with
                :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`, except planes and
                heightfields.
        """
        model = self.model if model is None else model
        self._collider_model = model
        if shape_ids is None:
            shape_ids = []
            if model.shape_count > 0:
                flags = model.shape_flags.numpy()
                types = model.shape_type.numpy()
                excluded = (int(GeoType.PLANE), int(GeoType.HFIELD))
                shape_ids = [
                    s
                    for s in range(model.shape_count)
                    if flags[s] & int(ShapeFlags.COLLIDE_PARTICLES) and int(types[s]) not in excluded
                ]
        shape_ids = np.asarray(shape_ids, dtype=np.int32)
        shape_world = model.shape_world.numpy()[shape_ids] if shape_ids.size else np.zeros(0, np.int32)
        shape_body = model.shape_body.numpy()[shape_ids] if shape_ids.size else np.zeros(0, np.int32)
        start, items = _csr_by_world(shape_world, self._world_count, include_global=True)
        self._world_shape_start = wp.array(start, dtype=wp.int32, device=self.device)
        self._world_shapes = wp.array(shape_ids[items], dtype=wp.int32, device=self.device)
        self._moving_shapes = wp.array(shape_ids[shape_body >= 0], dtype=wp.int32, device=self.device)
        self._has_colliders = shape_ids.size > 0

        # Activation points: world anchors, tracers, emitters, moving colliders, busy bricks.
        self._segment_tracers = self._world_count
        self._segment_emitters = self._segment_tracers + self.model.particle_count * 27
        self._segment_colliders = self._segment_emitters + self._emitter_count * 27
        self._segment_content = self._segment_colliders + len(self._moving_shapes) * _MAX_COLLIDER_TILES**3
        point_count = self._segment_content + self._max_tiles * 27
        self._points = wp.zeros(point_count, dtype=wp.vec3i, device=self.device)
        self._point_mask = wp.zeros(point_count, dtype=wp.int32, device=self.device)

    # ------------------------------------------------------------------
    # Grid queries
    # ------------------------------------------------------------------

    def sample_grid(self, points: wp.array[wp.vec3], *, world: int = 0) -> tuple[wp.array[float], wp.array[wp.vec3]]:
        """Sample the smoke density and air velocity at world positions.

        Args:
            points: Positions to sample [m].
            world: World whose grid to sample.

        Returns:
            Smoke density and air velocity [m/s] at each point; both are zero
            outside the active grid.
        """
        density = wp.empty(points.shape[0], dtype=float, device=self.device)
        velocity = wp.empty(points.shape[0], dtype=wp.vec3, device=self.device)
        self._launch(
            k.sample_points_kernel,
            points.shape[0],
            [self._grid, self._u, self._v, self._w, self._density, int(world), points],
            [velocity, density],
        )
        return density, velocity

    def active_cell_count(self) -> int:
        """Number of active cells across all worlds. Synchronizes with the device."""
        return int(self._volume.get_active_stats().voxel_count) * _TILE_CELLS

    def check_sparse_grid_rebuild_status(self) -> None:
        """Raise if the sparse grid ran out of capacity since construction or :meth:`reset`.

        Raises:
            RuntimeError: If a rebuild exceeded :attr:`Config.max_active_cell_count` or a node capacity.
        """
        status = int(self._accumulated_status.numpy()[0])
        if status != 0:
            raise RuntimeError(
                f"SolverSmoke sparse grid rebuild failed (status {status}); increase "
                "Config.max_active_cell_count or the node capacities."
            )

    def reset(self, state: State | None = None) -> None:
        """Clear the grid to still, clean, ambient air and clear the rebuild status.

        Args:
            state: If given, also deactivate all of its tracers.
        """
        for array in (self._u, self._v, self._w, self._density, self._temperature, self._pressure):
            array.zero_()
        self._tracer_head.zero_()
        self._accumulated_status.zero_()
        if state is not None and state.particle_count > 0:
            state.smoke.particle_age.fill_(_NEVER_EMITTED_AGE)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _faces(self):
        return ((self._u, 0), (self._v, 1), (self._w, 2))

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the smoke by ``dt``.

        Reads emitters from ``control`` (``None`` uses the model's default control
        values) and collider poses from ``state_in``. Writes tracers to ``state_out``,
        which may be the same object as ``state_in``. ``contacts`` is unused.
        """
        del contacts, observables
        if control is None:
            if self._default_control is None:
                self._default_control = self.model.control(clone_variables=False)
            control = self._default_control
        if self._collider_model.body_count > 0 and (state_in.body_q is None or state_in.body_qd is None):
            raise ValueError(
                "SolverSmoke colliders are attached to bodies, but state_in has no body_q or body_qd; "
                "assign the collider model's body state to state_in (see setup_collider)."
            )
        emitters = self._emitters(control)
        self._update_topology(state_in, emitters)
        self._rasterize_colliders(state_in)
        self._advect(dt)
        self._apply_emitters(emitters, dt)
        self._apply_forces(dt)
        self._project()
        self._step_tracers(state_in, state_out, emitters, dt)

    def _emitters(self, control: Control):
        if self._emitter_count == 0:
            return None
        ns = control.smoke
        return (
            ns.emitter_position,
            self.model.smoke.emitter_radius,
            ns.emitter_rate,
            ns.emitter_heat_rate,
            ns.emitter_velocity,
            self.model.smoke.emitter_world,
        )

    def _update_topology(self, state_in: State, emitters):
        cfg = self.config
        grid = self._grid
        points, mask = self._points, self._point_mask

        # Collect the bricks to keep or activate.
        self._content.zero_()
        self._launch(
            k.mark_content_kernel,
            self._density.shape,
            [
                grid,
                self._density,
                self._temperature,
                self._u,
                self._v,
                self._w,
                cfg.activation_threshold,
                cfg.activation_speed,
            ],
            [self._content],
        )
        self._launch(k.anchor_worlds_kernel, self._world_count, [0], [points, mask])
        if self.model.particle_count > 0:
            self._launch(
                k.tracer_activation_inputs_kernel,
                self.model.particle_count,
                [state_in.smoke.particle_age, cfg.tracer_lifetime],
                [self._tracer_active],
            )
            self._launch(
                k.activate_boxes_kernel,
                (self.model.particle_count, 27),
                [
                    grid,
                    state_in.particle_q,
                    self.model.particle_radius,
                    self._particle_world,
                    self._tracer_active,
                    self._padding,
                    self._segment_tracers,
                ],
                [points, mask],
            )
        if emitters is not None:
            position, radius, rate, heat_rate, _velocity, world = emitters
            self._launch(
                k.emitter_activation_inputs_kernel, self._emitter_count, [rate, heat_rate], [self._emitter_active]
            )
            self._launch(
                k.activate_boxes_kernel,
                (self._emitter_count, 27),
                [grid, position, radius, world, self._emitter_active, self._padding, self._segment_emitters],
                [points, mask],
            )
        if len(self._moving_shapes) > 0:
            collider = self._collider_model
            n = _MAX_COLLIDER_TILES
            self._launch(
                k.activate_colliders_kernel,
                (len(self._moving_shapes), n, n, n),
                [
                    grid,
                    self._moving_shapes,
                    collider.shape_body,
                    collider.shape_transform,
                    collider.shape_collision_radius,
                    collider.shape_world,
                    state_in.body_q,
                    self._padding,
                    self._segment_colliders,
                ],
                [points, mask],
            )
        self._launch(
            k.activate_content_kernel,
            (self._max_tiles, 27),
            [grid, self._content, self._segment_content],
            [points, mask],
        )

        # Rebuild the brick index in place and move surviving bricks to their new slots.
        wp.copy(self._old_tile_coord, self._tile_coord)
        self._volume.rebuild(points, status=self._rebuild_status, point_mask=mask)
        self._launch(_record_rebuild_status, 1, [self._rebuild_status, self._accumulated_status])
        self._tile_coord.fill_(k.INVALID_TILE)
        self._launch(k.assign_tile_coords_kernel, points.shape[0], [self._volume.id, points, mask], [self._tile_coord])
        self._launch(k.build_neighbors_kernel, (self._max_tiles, 27), [grid], [self._neighbor])
        for field, tmp in (
            (self._u, self._u_tmp),
            (self._v, self._v_tmp),
            (self._w, self._w_tmp),
            (self._density, self._scalar_tmp),
            (self._temperature, self._scalar_tmp),
            (self._pressure, self._scalar_tmp),
        ):
            wp.copy(tmp, field)
            field.zero_()
            self._launch(k.remap_field_kernel, field.shape, [self._volume.id, self._old_tile_coord, tmp], [field])

    def _rasterize_colliders(self, state_in: State):
        if not self._has_colliders:
            self._solid.zero_()
            self._solid_velocity.zero_()
            return
        collider = self._collider_model
        has_bodies = collider.body_count > 0
        self._launch(
            k.rasterize_colliders_kernel,
            self._solid.shape,
            [
                self._grid,
                self._world_shape_start,
                self._world_shapes,
                collider.shape_body,
                collider.shape_transform,
                collider.shape_type,
                collider.shape_scale,
                collider.shape_source_ptr,
                collider.shape_collision_radius,
                state_in.body_q if has_bodies else None,
                state_in.body_qd if has_bodies else None,
                collider.body_com if has_bodies else None,
            ],
            [self._solid, self._solid_velocity],
        )

    def _advect(self, dt: float):
        cfg = self.config
        grid = self._grid
        velocity_decay = math.exp(-cfg.velocity_damping * dt)
        tmps = (self._u_tmp, self._v_tmp, self._w_tmp)
        for (src, axis), dst in zip(self._faces(), tmps, strict=True):
            self._launch(
                k.advect_face_kernel,
                src.shape,
                [grid, src, axis, self._u, self._v, self._w, dt, velocity_decay],
                [dst],
            )
        # Copy back rather than swap, so a captured step always reads the same buffers.
        for (field, _axis), tmp in zip(self._faces(), tmps, strict=True):
            wp.copy(field, tmp)
        for field, rate in ((self._density, cfg.density_decay_rate), (self._temperature, cfg.temperature_decay_rate)):
            self._launch(
                k.advect_cell_kernel,
                field.shape,
                [grid, field, self._u, self._v, self._w, dt, math.exp(-rate * dt)],
                [self._scalar_tmp],
            )
            wp.copy(field, self._scalar_tmp)

    def _apply_emitters(self, emitters, dt: float):
        if emitters is None:
            return
        position, radius, rate, heat_rate, velocity, _world = emitters
        grid = self._grid
        self._launch(
            k.emit_scalars_kernel,
            self._density.shape,
            [grid, self._world_emitter_start, self._world_emitters, position, radius, rate, heat_rate, dt],
            [self._density, self._temperature],
        )
        coupling = self.config.emitter_velocity_coupling
        if coupling <= 0.0:
            return
        for vel, axis in self._faces():
            self._launch(
                k.emit_velocity_kernel,
                vel.shape,
                [
                    grid,
                    vel,
                    axis,
                    self._world_emitter_start,
                    self._world_emitters,
                    position,
                    radius,
                    rate,
                    velocity,
                    coupling,
                ],
            )

    def _apply_forces(self, dt: float):
        cfg = self.config
        grid = self._grid
        cells = self._density.shape
        self._launch(
            k.buoyancy_kernel,
            self._w.shape,
            [grid, self._density, self._temperature, cfg.density_weight, cfg.temperature_lift, dt],
            [self._w],
        )
        if cfg.vorticity_confinement <= 0.0:
            return
        self._launch(k.center_velocity_kernel, cells, [grid, self._u, self._v, self._w], [self._center_velocity])
        self._launch(k.curl_kernel, cells, [grid, self._center_velocity], [self._curl, self._curl_magnitude])
        self._launch(
            k.confinement_kernel,
            cells,
            [grid, self._curl, self._curl_magnitude, self._density, cfg.vorticity_confinement],
            [self._force],
        )
        for vel, axis in self._faces():
            self._launch(k.apply_cell_force_kernel, vel.shape, [grid, self._force, axis, dt], [vel])

    def _project(self):
        grid = self._grid
        cells = self._density.shape
        for vel, axis in self._faces():
            self._launch(k.enforce_solid_faces_kernel, vel.shape, [grid, vel, axis, self._solid, self._solid_velocity])
        self._launch(k.divergence_kernel, cells, [grid, self._u, self._v, self._w, self._solid], [self._divergence])
        for _ in range(self.config.pressure_iterations):
            for parity in (0, 1):
                self._launch(
                    k.pressure_rbsor_kernel,
                    cells,
                    [grid, self._divergence, self._solid, self.config.pressure_relaxation, parity],
                    [self._pressure],
                )
        for vel, axis in self._faces():
            self._launch(k.project_kernel, vel.shape, [grid, vel, axis, self._pressure, self._solid])

    def _step_tracers(self, state_in: State, state_out: State, emitters, dt: float):
        if self.model.particle_count == 0:
            return
        cfg = self.config
        self._launch(
            k.advect_tracers_kernel,
            self.model.particle_count,
            [
                self._grid,
                self._u,
                self._v,
                self._w,
                self._particle_world,
                dt,
                cfg.tracer_lifetime,
                state_in.particle_q,
                state_in.smoke.particle_age,
            ],
            [state_out.particle_q, state_out.particle_qd, state_out.smoke.particle_age],
        )
        if emitters is not None and cfg.max_tracers_per_emitter_step > 0:
            position, radius, rate, _heat_rate, _velocity, world = emitters
            self._launch(
                k.emit_tracers_kernel,
                self._emitter_count * cfg.max_tracers_per_emitter_step,
                [
                    position,
                    radius,
                    rate,
                    world,
                    self._world_tracer_start,
                    cfg.tracer_rate_scale,
                    cfg.max_tracers_per_emitter_step,
                    dt,
                    self._seed,
                    self._tracer_head,
                ],
                [state_out.particle_q, state_out.particle_qd, state_out.smoke.particle_age],
            )
        self._launch(
            k.advance_seed_kernel, self._world_count, [self._seed, self._tracer_head, self._world_tracer_start]
        )

    # ------------------------------------------------------------------
    # Render grains
    # ------------------------------------------------------------------

    def _next_grain_seed(self) -> int:
        self._grain_seed += 1
        return self._grain_seed

    def sample_render_grains(self, state: State, grains_per_particle: int) -> wp.array2d[wp.vec3]:
        """Scatter render points around each tracer.

        Each tracer gets ``grains_per_particle`` points within its puff, whose radius
        is the particle radius plus :attr:`Config.tracer_spread_rate` times its age.
        Keep them moving with :meth:`update_render_grains`.

        Args:
            state: State whose tracers to sample around.
            grains_per_particle: Number of render points per tracer.

        Returns:
            Render point positions [m], shape ``(particle_count, grains_per_particle)``.
        """
        grains = wp.empty((state.particle_count, grains_per_particle), dtype=wp.vec3, device=self.device)
        if grains.size > 0:
            self._launch(
                k.sample_grains_kernel,
                grains.shape,
                [
                    state.particle_q,
                    self.model.particle_radius,
                    state.smoke.particle_age,
                    self.config.tracer_spread_rate,
                    self._next_grain_seed(),
                ],
                [grains],
            )
        return grains

    def update_render_grains(self, state_prev: State, state: State, grains: wp.array2d[wp.vec3], dt: float) -> None:
        """Carry render points with the air and keep them inside their tracer's puff.

        Call once after each :meth:`step`. Points of newly emitted tracers are
        scattered around their new position.

        Args:
            state_prev: State before the step. Unused; kept for parity with
                :meth:`SolverImplicitMPM.update_render_grains <newton.solvers.SolverImplicitMPM.update_render_grains>`.
            state: State after the step.
            grains: Render points from :meth:`sample_render_grains`, updated in place [m].
            dt: Time step of the last :meth:`step` [s].
        """
        del state_prev
        if grains.size == 0:
            return
        self._launch(
            k.update_grains_kernel,
            grains.shape,
            [
                self._grid,
                self._u,
                self._v,
                self._w,
                self._particle_world,
                dt,
                self.config.tracer_lifetime,
                self.config.tracer_spread_rate,
                self._next_grain_seed(),
                state.particle_q,
                self.model.particle_radius,
                state.smoke.particle_age,
            ],
            [grains],
        )
