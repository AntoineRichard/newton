# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from ...core.types import override
from ...geometry import GeoType, ShapeFlags
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from . import kernels as k

__all__ = ["SolverSmoke"]

_NEVER_EMITTED_AGE = 1.0e30
"""Age [s] of a tracer that has not been emitted yet; larger than any lifetime."""


class SolverSmoke(SolverBase):
    """Eulerian smoke on a dense, staggered grid, driven by emitters and pushed by colliders.

    .. experimental::

        :class:`SolverSmoke`, its :class:`Config`, and the ``smoke:*`` custom attributes may
        change without the normal deprecation period.

    The solver follows the stable-fluids method: semi-Lagrangian (RK2) advection
    of velocity, smoke density, and temperature; buoyancy and vorticity
    confinement; and a red-black SOR pressure projection that makes the air
    incompressible. The grid is axis aligned and Z up. Its bottom face is solid
    ground; its sides and top are open.

    **Grid.** The grid, its velocity, and its scalar fields belong to the
    solver, like the background grid of :class:`~newton.solvers.SolverImplicitMPM`.
    Read them through :attr:`density`, :attr:`temperature`, and
    :attr:`face_velocity`; clear them with :meth:`reset`.

    **Emitters.** Smoke sources are entries of the ``smoke:emitter`` custom
    frequency (see :meth:`register_custom_attributes`). Their position,
    velocity, and emission rates are :class:`~newton.Control` attributes, so they
    can follow anything, such as the contact point of a skidding tire.

    **Colliders.** Shapes with :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` block
    the air and drag it with their velocity. By default they come from the
    solver's model; :meth:`setup_collider` reads them from another model
    instead. Collider bodies are posed from ``state_in.body_q`` and
    ``state_in.body_qd``. Coupling is one way: smoke never pushes bodies.
    Planes and heightfields are ignored; the grid floor is the ground.

    **Tracers.** The model's particles are a pool of tracers carried by the air
    for rendering. Emitters spawn them, reusing the oldest slot when the pool is
    full. A tracer's age is the ``smoke:particle_age`` state attribute; a tracer
    whose age reaches :attr:`Config.tracer_lifetime` is inactive.
    :meth:`sample_render_grains` and :meth:`update_render_grains` draw many more
    render points than there are tracers.

    Limitations: one grid shared by all worlds; no two-way coupling; not
    differentiable.

    Args:
        model: Model holding the tracer particles and the ``smoke:*`` attributes.
        config: Solver configuration. ``None`` uses the defaults.
    """

    EMITTER_FREQUENCY = "smoke:emitter"
    """Custom frequency of smoke emitters."""

    @dataclass
    class Config:
        """Configuration for :class:`SolverSmoke`."""

        grid_lower: tuple[float, float, float] = (-1.0, -1.0, 0.0)
        """Lower corner of the grid [m]. Its z value is the ground height."""
        grid_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)
        """Upper corner of the grid [m], rounded up to a whole number of voxels."""
        voxel_size: float = 0.05
        """Edge length of a grid cell [m]."""
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

        h = float(cfg.voxel_size)
        dims = tuple(max(2, math.ceil((cfg.grid_upper[a] - cfg.grid_lower[a]) / h - 1.0e-6)) for a in range(3))
        self._h = h
        self._lower = wp.vec3(*cfg.grid_lower)
        self._upper = wp.vec3(*(cfg.grid_lower[a] + dims[a] * h for a in range(3)))
        self._dims = dims
        nx, ny, nz = dims

        device = self.device

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=device)

        self._u = zeros((nx + 1, ny, nz))
        self._v = zeros((nx, ny + 1, nz))
        self._w = zeros((nx, ny, nz + 1))
        self._u_tmp = zeros(self._u.shape)
        self._v_tmp = zeros(self._v.shape)
        self._w_tmp = zeros(self._w.shape)
        self._density = zeros(dims)
        self._temperature = zeros(dims)
        self._scalar_tmp = zeros(dims)
        self._pressure = zeros(dims)
        self._divergence = zeros(dims)
        self._solid = zeros(dims, dtype=wp.int32)
        self._solid_velocity = zeros(dims, dtype=wp.vec3)
        self._center_velocity = zeros(dims, dtype=wp.vec3)
        self._curl = zeros(dims, dtype=wp.vec3)
        self._curl_magnitude = zeros(dims)
        self._force = zeros(dims, dtype=wp.vec3)

        self._seed = wp.zeros(1, dtype=wp.int32, device=device)
        self._tracer_head = wp.zeros(1, dtype=wp.int32, device=device)
        self._default_control: Control | None = None
        self._grain_seed = 0

        self.setup_collider()

    # ------------------------------------------------------------------
    # Custom attributes
    # ------------------------------------------------------------------

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the ``smoke`` custom frequency and attributes.

        Add emitters with :meth:`ModelBuilder.add_custom_values`, for example
        ``builder.add_custom_values(**{"smoke:emitter_radius": 0.05})``.

        Attributes registered on Model (per emitter):
            - ``smoke:emitter_radius``: Radius of the emitter's linear falloff [m].

        Attributes registered on Control (per emitter):
            - ``smoke:emitter_position``: Emitter center [m].
            - ``smoke:emitter_velocity``: Velocity the emitter gives nearby air [m/s].
            - ``smoke:emitter_rate``: Smoke density added per second at the center [1/s].
              Zero disables the emitter.
            - ``smoke:emitter_heat_rate``: Temperature added per second at the center [K/s].

        Attributes registered on State (per particle):
            - ``smoke:particle_age``: Tracer age since emission [s]. Tracers that were
              never emitted start at a very large age, so they are inactive.

        Args:
            builder: Model builder to register the attributes on.
        """
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="emitter", namespace="smoke"))

        def emitter_attribute(name, dtype, default, assignment):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=cls.EMITTER_FREQUENCY,
                    assignment=assignment,
                    dtype=dtype,
                    default=default,
                    namespace="smoke",
                )
            )

        emitter_attribute("emitter_radius", wp.float32, 0.05, Model.AttributeAssignment.MODEL)
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
                this model's bodies.
            shape_ids: Shapes of ``model`` to use. ``None`` selects every shape with
                :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`, except planes and
                heightfields.
        """
        model = self.model if model is None else model
        self._collider_model = model
        if shape_ids is None:
            if model.shape_count == 0:
                shape_ids = []
            else:
                flags = model.shape_flags.numpy()
                types = model.shape_type.numpy()
                excluded = (int(GeoType.PLANE), int(GeoType.HFIELD))
                shape_ids = [
                    s
                    for s in range(model.shape_count)
                    if flags[s] & int(ShapeFlags.COLLIDE_PARTICLES) and int(types[s]) not in excluded
                ]
        self._collider_shapes = wp.array(shape_ids, dtype=wp.int32, device=self.device)

    # ------------------------------------------------------------------
    # Grid access
    # ------------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Number of cells along x, y, and z."""
        return self._dims

    @property
    def grid_lower(self) -> wp.vec3:
        """Lower corner of the grid [m]."""
        return self._lower

    @property
    def grid_upper(self) -> wp.vec3:
        """Upper corner of the grid [m]."""
        return self._upper

    @property
    def density(self) -> wp.array3d[float]:
        """Smoke density at cell centers, shape ``grid_shape``."""
        return self._density

    @property
    def temperature(self) -> wp.array3d[float]:
        """Temperature above ambient at cell centers [K], shape ``grid_shape``."""
        return self._temperature

    @property
    def face_velocity(self) -> tuple[wp.array3d[float], wp.array3d[float], wp.array3d[float]]:
        """Air velocity on cell faces [m/s] as ``(u, v, w)``.

        ``u[i, j, k]`` is the x velocity on the face between cells ``i - 1`` and ``i``,
        shape ``(nx + 1, ny, nz)``; ``v`` and ``w`` follow the same pattern.
        """
        return self._u, self._v, self._w

    def reset(self, state: State | None = None) -> None:
        """Clear the grid to still, clean, ambient air.

        Args:
            state: If given, also deactivate all of its tracers.
        """
        for array in (self._u, self._v, self._w, self._density, self._temperature, self._pressure):
            array.zero_()
        self._tracer_head.zero_()
        if state is not None and state.particle_count > 0:
            state.smoke.particle_age.fill_(_NEVER_EMITTED_AGE)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _faces(self):
        return ((self._u, 0), (self._v, 1), (self._w, 2))

    def _emitters(self, control: Control | None):
        model_ns = getattr(self.model, "smoke", None)
        control_ns = getattr(control, "smoke", None) if control is not None else None
        radius = getattr(model_ns, "emitter_radius", None)
        if radius is None or control_ns is None or radius.shape[0] == 0:
            return None
        return (
            control_ns.emitter_position,
            radius,
            control_ns.emitter_rate,
            control_ns.emitter_heat_rate,
            control_ns.emitter_velocity,
        )

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
        cfg = self.config
        h = self._h
        dev = self.device

        # Colliders.
        collider_model = self._collider_model
        has_bodies = collider_model.body_count > 0
        if has_bodies and (state_in.body_q is None or state_in.body_qd is None):
            raise ValueError(
                "SolverSmoke colliders are attached to bodies, but state_in has no body_q or body_qd; "
                "assign the collider model's body state to state_in (see setup_collider)."
            )
        if self._collider_shapes.shape[0] > 0:
            wp.launch(
                k.rasterize_colliders_kernel,
                dim=self._dims,
                inputs=[
                    self._lower,
                    h,
                    self._collider_shapes,
                    collider_model.shape_body,
                    collider_model.shape_transform,
                    collider_model.shape_type,
                    collider_model.shape_scale,
                    collider_model.shape_source_ptr,
                    collider_model.shape_collision_radius,
                    state_in.body_q if has_bodies else None,
                    state_in.body_qd if has_bodies else None,
                    collider_model.body_com if has_bodies else None,
                ],
                outputs=[self._solid, self._solid_velocity],
                device=dev,
            )

        # Advection.
        velocity_decay = math.exp(-cfg.velocity_damping * dt)
        for (src, axis), dst in zip(self._faces(), (self._u_tmp, self._v_tmp, self._w_tmp), strict=True):
            wp.launch(
                k.advect_kernel,
                dim=src.shape,
                inputs=[src, self._u, self._v, self._w, self._lower, h, _AXIS_OFFSETS[axis], dt, velocity_decay],
                outputs=[dst],
                device=dev,
            )
        self._u, self._u_tmp = self._u_tmp, self._u
        self._v, self._v_tmp = self._v_tmp, self._v
        self._w, self._w_tmp = self._w_tmp, self._w
        for name, rate in (("_density", cfg.density_decay_rate), ("_temperature", cfg.temperature_decay_rate)):
            src = getattr(self, name)
            wp.launch(
                k.advect_kernel,
                dim=self._dims,
                inputs=[src, self._u, self._v, self._w, self._lower, h, k.OFFSET_C, dt, math.exp(-rate * dt)],
                outputs=[self._scalar_tmp],
                device=dev,
            )
            setattr(self, name, self._scalar_tmp)
            self._scalar_tmp = src

        # Emitters.
        emitters = self._emitters(control)
        if emitters is not None:
            position, radius, rate, heat_rate, velocity = emitters
            wp.launch(
                k.emit_scalars_kernel,
                dim=self._dims,
                inputs=[self._lower, h, position, radius, rate, heat_rate, dt],
                outputs=[self._density, self._temperature],
                device=dev,
            )
            if cfg.emitter_velocity_coupling > 0.0:
                for vel, axis in self._faces():
                    wp.launch(
                        k.emit_velocity_kernel,
                        dim=vel.shape,
                        inputs=[
                            vel,
                            axis,
                            self._lower,
                            h,
                            position,
                            radius,
                            rate,
                            velocity,
                            cfg.emitter_velocity_coupling,
                        ],
                        device=dev,
                    )

        # Body forces.
        wp.launch(
            k.buoyancy_kernel,
            dim=self._w.shape,
            inputs=[self._density, self._temperature, cfg.density_weight, cfg.temperature_lift, dt],
            outputs=[self._w],
            device=dev,
        )
        if cfg.vorticity_confinement > 0.0:
            wp.launch(
                k.center_velocity_kernel,
                dim=self._dims,
                inputs=[self._u, self._v, self._w],
                outputs=[self._center_velocity],
                device=dev,
            )
            wp.launch(
                k.curl_kernel,
                dim=self._dims,
                inputs=[self._center_velocity, h],
                outputs=[self._curl, self._curl_magnitude],
                device=dev,
            )
            wp.launch(
                k.confinement_kernel,
                dim=self._dims,
                inputs=[self._curl, self._curl_magnitude, self._density, h, cfg.vorticity_confinement],
                outputs=[self._force],
                device=dev,
            )
            for vel, axis in self._faces():
                wp.launch(
                    k.apply_cell_force_kernel, dim=vel.shape, inputs=[self._force, axis, dt], outputs=[vel], device=dev
                )

        self._project()
        self._step_tracers(state_in, state_out, emitters, dt)

    def _project(self):
        h = self._h
        dev = self.device
        for vel, axis in self._faces():
            wp.launch(
                k.enforce_solid_faces_kernel,
                dim=vel.shape,
                inputs=[vel, axis, self._solid, self._solid_velocity],
                device=dev,
            )
        wp.launch(
            k.divergence_kernel,
            dim=self._dims,
            inputs=[self._u, self._v, self._w, self._solid, h],
            outputs=[self._divergence],
            device=dev,
        )
        for _ in range(self.config.pressure_iterations):
            for parity in (0, 1):
                wp.launch(
                    k.pressure_rbsor_kernel,
                    dim=self._dims,
                    inputs=[self._divergence, self._solid, h, self.config.pressure_relaxation, parity],
                    outputs=[self._pressure],
                    device=dev,
                )
        for vel, axis in self._faces():
            wp.launch(k.project_kernel, dim=vel.shape, inputs=[vel, axis, self._pressure, self._solid, h], device=dev)

    def _step_tracers(self, state_in: State, state_out: State, emitters, dt: float):
        if self.model.particle_count == 0:
            return
        cfg = self.config
        dev = self.device
        wp.launch(
            k.advect_tracers_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._u,
                self._v,
                self._w,
                self._lower,
                self._upper,
                self._h,
                dt,
                cfg.tracer_lifetime,
                state_in.particle_q,
                state_in.smoke.particle_age,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd, state_out.smoke.particle_age],
            device=dev,
        )
        if emitters is not None and cfg.max_tracers_per_emitter_step > 0:
            position, radius, rate, _heat_rate, _velocity = emitters
            wp.launch(
                k.emit_tracers_kernel,
                dim=position.shape[0] * cfg.max_tracers_per_emitter_step,
                inputs=[
                    position,
                    radius,
                    rate,
                    cfg.tracer_rate_scale,
                    cfg.max_tracers_per_emitter_step,
                    dt,
                    self._seed,
                    self._tracer_head,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd, state_out.smoke.particle_age],
                device=dev,
            )
        wp.launch(
            k.advance_seed_kernel,
            dim=1,
            inputs=[self._seed, self._tracer_head, self.model.particle_count],
            device=dev,
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
            wp.launch(
                k.sample_grains_kernel,
                dim=grains.shape,
                inputs=[
                    state.particle_q,
                    self.model.particle_radius,
                    state.smoke.particle_age,
                    self.config.tracer_spread_rate,
                    self._next_grain_seed(),
                ],
                outputs=[grains],
                device=self.device,
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
        wp.launch(
            k.update_grains_kernel,
            dim=grains.shape,
            inputs=[
                self._u,
                self._v,
                self._w,
                self._lower,
                self._h,
                dt,
                self.config.tracer_lifetime,
                self.config.tracer_spread_rate,
                self._next_grain_seed(),
                state.particle_q,
                self.model.particle_radius,
                state.smoke.particle_age,
            ],
            outputs=[grains],
            device=self.device,
        )


_AXIS_OFFSETS = (k.OFFSET_U, k.OFFSET_V, k.OFFSET_W)
