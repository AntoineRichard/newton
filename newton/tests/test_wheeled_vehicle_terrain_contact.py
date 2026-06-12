# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE

WHEEL_RADIUS = 0.5
WHEEL_WIDTH = 0.2
TERRAIN_MU = 0.77
_VEHICLE_TERRAIN_ANGLE = -0.05
_VEHICLE_TERRAIN_Z_OFFSET = -0.002
_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"


@dataclass(frozen=True)
class TerrainContactFixture:
    model: newton.Model
    state: newton.State
    contacts: newton.Contacts
    metadata: newton.wheeled.WheeledModelMetadata
    patch_state: newton.wheeled.WheelContactPatchState
    terrain_shape: int
    wheel_body: int
    wheel_shape: int


@dataclass(frozen=True)
class ProjectionPatchSnapshot:
    contact_count: int
    wheel_contact_point_count: int
    active: np.ndarray
    patch_contact_count: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    patch_u_extent: np.ndarray
    patch_v_extent: np.ndarray
    patch_area: np.ndarray
    terrain_shape_index: np.ndarray
    friction_mu_seed: np.ndarray


def _shape_config(*, gap: float = 0.0, margin: float = 0.0, mu: float = TERRAIN_MU) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.gap = gap
    cfg.margin = margin
    cfg.mu = mu
    return cfg


def _wheel_attributes(wheel_id: int = 0) -> dict[str, object]:
    return {
        "wheeled:is_wheel": True,
        "wheeled:wheel_id": wheel_id,
        "wheeled:vehicle_id": wheel_id,
        "wheeled:wheel_radius": WHEEL_RADIUS,
        "wheeled:wheel_width": WHEEL_WIDTH,
    }


def _wheel_body_attributes(wheel_id: int = 0) -> dict[str, object]:
    return {"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": wheel_id}


def _wheel_rotation() -> wp.quat:
    # Cylinder local Z is the axle; rotate it to world Y so the cylinder rolls on world Z terrain.
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)


def _add_cylinder_wheel(
    builder: newton.ModelBuilder,
    *,
    wheel_id: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.49,
    gap: float = 0.0,
) -> tuple[int, int]:
    body = builder.add_body(
        xform=wp.transform(wp.vec3(x, y, z), _wheel_rotation()),
        label=f"wheel_body_{wheel_id}",
        custom_attributes=_wheel_body_attributes(wheel_id),
    )
    shape = builder.add_shape_cylinder(
        body,
        radius=WHEEL_RADIUS,
        half_height=0.5 * WHEEL_WIDTH,
        cfg=_shape_config(gap=gap),
        label=f"wheel_shape_{wheel_id}",
        custom_attributes=_wheel_attributes(wheel_id),
    )
    return body, shape


def _build_fixture(kind: str, *, gap: float = 0.0, wheel_z: float | None = None) -> TerrainContactFixture:
    builder = newton.ModelBuilder(gravity=0.0)
    newton.wheeled.register_wheeled_custom_attributes(builder)
    terrain_cfg = _shape_config(gap=gap)
    wheel_x = 0.0

    if kind == "plane":
        terrain_shape = builder.add_ground_plane(cfg=terrain_cfg, label="terrain")
        z = 0.49 if wheel_z is None else wheel_z
    elif kind == "flat_box":
        terrain_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            hx=2.0,
            hy=2.0,
            hz=0.05,
            cfg=terrain_cfg,
            label="terrain",
        )
        z = 0.49 if wheel_z is None else wheel_z
    elif kind == "ramp":
        ramp_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.2)
        terrain_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), ramp_rotation),
            hx=2.0,
            hy=2.0,
            hz=0.05,
            cfg=terrain_cfg,
            label="terrain",
        )
        z = 0.49 if wheel_z is None else wheel_z
    elif kind == "ridge":
        terrain_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.08,
            hy=1.0,
            hz=0.05,
            cfg=terrain_cfg,
            label="terrain",
        )
        z = 0.54 if wheel_z is None else wheel_z
    elif kind == "mesh_ripple":
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [2.0, 2.0, 0.08],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        terrain_mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        terrain_shape = builder.add_shape_mesh(-1, mesh=terrain_mesh, cfg=terrain_cfg, label="terrain")
        z = 0.49 if wheel_z is None else wheel_z
    elif kind == "mesh_jump":
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [0.0, -2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, -2.0, 0.2],
                [2.0, 2.0, 0.2],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2, 1, 4, 3, 4, 5, 3], dtype=np.int32)
        terrain_mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        terrain_shape = builder.add_shape_mesh(-1, mesh=terrain_mesh, cfg=terrain_cfg, label="terrain")
        z = 0.59 if wheel_z is None else wheel_z
        wheel_x = 1.0
    else:
        raise ValueError(f"unknown terrain fixture kind: {kind}")

    wheel_body, wheel_shape = _add_cylinder_wheel(builder, x=wheel_x, z=z, gap=gap)
    model = builder.finalize(device="cpu")
    metadata = newton.wheeled.build_wheeled_metadata(model)
    state = model.state()
    contacts = model.contacts()
    model.collide(state, contacts)
    patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
    newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)
    return TerrainContactFixture(model, state, contacts, metadata, patch_state, terrain_shape, wheel_body, wheel_shape)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _point_world(model: newton.Model, state: newton.State, shape_index: int, point: np.ndarray) -> np.ndarray:
    body_index = int(model.shape_body.numpy()[shape_index])
    if body_index < 0:
        return np.asarray(point, dtype=np.float32)
    body_q = state.body_q.numpy()[body_index]
    return body_q[:3] + _quat_rotate(body_q[3:7], np.asarray(point, dtype=np.float32))


def _wheel_contact_points_from_contacts(
    model: newton.Model,
    state: newton.State,
    wheel_shape: int,
    contacts: newton.Contacts,
) -> np.ndarray:
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
    point0 = contacts.rigid_contact_point0.numpy()[:contact_count]
    point1 = contacts.rigid_contact_point1.numpy()[:contact_count]

    points = []
    for contact_id in range(contact_count):
        if int(shape0[contact_id]) == wheel_shape:
            points.append(_point_world(model, state, wheel_shape, point0[contact_id]))
        elif int(shape1[contact_id]) == wheel_shape:
            points.append(_point_world(model, state, wheel_shape, point1[contact_id]))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _wheel_contact_points(fixture: TerrainContactFixture) -> np.ndarray:
    return _wheel_contact_points_from_contacts(fixture.model, fixture.state, fixture.wheel_shape, fixture.contacts)


def _classify_patch(patch_state: newton.wheeled.WheelContactPatchState) -> str:
    u_extent = float(patch_state.patch_u_extent.numpy()[0])
    v_extent = float(patch_state.patch_v_extent.numpy()[0])
    larger = max(u_extent, v_extent)
    smaller = min(u_extent, v_extent)
    if larger < 1.0e-5:
        return "point"
    if smaller <= max(1.0e-5, larger * 0.02):
        return "line"
    return "area"


def _classify_projection_snapshot(snapshot: ProjectionPatchSnapshot) -> str:
    u_extent = float(snapshot.patch_u_extent[0])
    v_extent = float(snapshot.patch_v_extent[0])
    larger = max(u_extent, v_extent)
    smaller = min(u_extent, v_extent)
    if larger < 1.0e-5:
        return "point"
    if smaller <= max(1.0e-5, larger * 0.02):
        return "line"
    return "area"


def _line_residual_ratio(points: np.ndarray) -> float:
    if points.shape[0] <= 2:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] <= 1.0e-8:
        return 0.0
    return float(singular_values[1] / singular_values[0])


def _projection_patch_snapshot(
    fixture: TerrainContactFixture,
    *,
    enable_axial_contact_projection: bool,
    enable_plane_cylinder_contact_collapse: bool = True,
    enable_analytic_plane_patches: bool = False,
) -> ProjectionPatchSnapshot:
    pipeline = newton.CollisionPipeline(
        fixture.model,
        broad_phase="explicit",
        reduce_contacts=False,
        deterministic=True,
        enable_axial_contact_projection=enable_axial_contact_projection,
        enable_plane_cylinder_contact_collapse=enable_plane_cylinder_contact_collapse,
        verify_buffers=False,
    )
    contacts = pipeline.contacts()
    pipeline.collide(fixture.state, contacts)
    patch_state = newton.wheeled.WheelContactPatchState(fixture.model, fixture.metadata)
    newton.wheeled.update_wheel_contact_patches(
        fixture.model,
        fixture.state,
        contacts,
        fixture.metadata,
        patch_state,
        enable_analytic_plane_patches=enable_analytic_plane_patches,
    )
    points = _wheel_contact_points_from_contacts(fixture.model, fixture.state, fixture.wheel_shape, contacts)
    return ProjectionPatchSnapshot(
        contact_count=int(contacts.rigid_contact_count.numpy()[0]),
        wheel_contact_point_count=points.shape[0],
        active=patch_state.active.numpy().copy(),
        patch_contact_count=patch_state.contact_count.numpy().copy(),
        center=patch_state.center.numpy().copy(),
        normal=patch_state.normal.numpy().copy(),
        patch_u_extent=patch_state.patch_u_extent.numpy().copy(),
        patch_v_extent=patch_state.patch_v_extent.numpy().copy(),
        patch_area=patch_state.patch_area.numpy().copy(),
        terrain_shape_index=patch_state.terrain_shape_index.numpy().copy(),
        friction_mu_seed=patch_state.friction_mu_seed.numpy().copy(),
    )


def _assert_projection_snapshot_finite(test: unittest.TestCase, snapshot: ProjectionPatchSnapshot) -> None:
    test.assertGreater(snapshot.contact_count, 0)
    test.assertGreater(snapshot.wheel_contact_point_count, 0)
    test.assertTrue(bool(snapshot.active[0]))
    test.assertGreater(int(snapshot.patch_contact_count[0]), 0)
    for values in (
        snapshot.center,
        snapshot.normal,
        snapshot.patch_u_extent,
        snapshot.patch_v_extent,
        snapshot.patch_area,
        snapshot.friction_mu_seed,
    ):
        test.assertTrue(np.isfinite(values).all())
    test.assertGreater(float(np.linalg.norm(snapshot.normal[0])), 0.9)


def _assert_finite_patch(test: unittest.TestCase, fixture: TerrainContactFixture) -> None:
    patch = fixture.patch_state
    test.assertTrue(bool(patch.active.numpy()[0]))
    test.assertGreater(int(patch.contact_count.numpy()[0]), 0)
    test.assertEqual(int(patch.terrain_shape_index.numpy()[0]), fixture.terrain_shape)
    test.assertAlmostEqual(float(patch.friction_mu_seed.numpy()[0]), TERRAIN_MU, delta=1.0e-6)
    for values in (
        patch.center.numpy(),
        patch.normal.numpy(),
        patch.patch_u_extent.numpy(),
        patch.patch_v_extent.numpy(),
        patch.patch_area.numpy(),
    ):
        test.assertTrue(np.isfinite(values).all())
    test.assertGreater(float(np.linalg.norm(patch.normal.numpy()[0])), 0.9)
    test.assertGreater(float(patch.normal.numpy()[0, 2]), 0.5)


@dataclass
class VehicleTerrainFixture:
    model: newton.Model
    state_0: newton.State
    state_1: newton.State
    control: object
    contacts: newton.Contacts
    metadata: newton.wheeled.WheeledModelMetadata
    patch_state: newton.wheeled.WheelContactPatchState
    tire_control: newton.wheeled.WheelTireControl
    tire_state: newton.wheeled.WheelTireState
    vehicle_layout: newton.wheeled.WheeledVehicleLayout
    vehicle_control: newton.wheeled.WheeledVehicleControl
    vehicle_state: newton.wheeled.WheeledVehicleState
    motor_config: newton.wheeled.WheeledMotorConfig
    steering_config: newton.wheeled.WheeledSteeringConfig
    solver: object
    terrain_shape: int


def _configure_zero_gap_builder(builder: newton.ModelBuilder) -> None:
    builder.rigid_gap = 0.0
    builder.default_shape_cfg.gap = 0.0
    builder.default_shape_cfg.margin = 0.0


def _vehicle_asset(vehicle_name: str) -> newton.wheeled.WheeledAssetMetadata:
    assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(_MANIFEST_PATH)}
    return assets[vehicle_name]


def _vehicle_tire_parameters(vehicle_name: str) -> tuple[float, float, float, float, float]:
    if vehicle_name == "husky":
        return 1.2, 200.0, 900.0, 1200.0, 8.0
    return 0.8, 20.0, 40.0, 30.0, 14.0


def _vehicle_ramp_xform() -> wp.transform:
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), _VEHICLE_TERRAIN_ANGLE)
    return wp.transform(wp.vec3(0.0, 0.0, _VEHICLE_TERRAIN_Z_OFFSET), rotation)


def _add_vehicle_terrain(scene: newton.ModelBuilder, terrain_kind: str) -> int:
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), _VEHICLE_TERRAIN_ANGLE)
    if terrain_kind == "ramp_box":
        return scene.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), rotation),
            hx=2.0,
            hy=2.0,
            hz=0.05,
            cfg=_shape_config(gap=0.0),
            label="terrain_ramp_box",
        )
    if terrain_kind == "mesh_slope":
        vertices = np.array(
            [
                [x, y, -math.tan(_VEHICLE_TERRAIN_ANGLE) * x]
                for x, y in ((-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0))
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        return scene.add_shape_mesh(
            -1,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=_shape_config(gap=0.0),
            label="terrain_mesh_slope",
        )
    raise ValueError(f"unknown vehicle terrain kind: {terrain_kind}")


def _build_vehicle_terrain_fixture(
    vehicle_name: str,
    terrain_kind: str,
    *,
    world_count: int = 1,
) -> VehicleTerrainFixture:
    asset = _vehicle_asset(vehicle_name)

    world = newton.ModelBuilder()
    _configure_zero_gap_builder(world)
    newton.solvers.SolverMuJoCo.register_custom_attributes(world)
    newton.wheeled.register_wheeled_custom_attributes(world)
    world.add_usd(
        str(asset.file),
        xform=_vehicle_ramp_xform(),
        enable_self_collisions=False,
        schema_resolvers=[newton.usd.SchemaResolverPhysx()],
    )
    newton.wheeled.apply_wheeled_manifest(world, _MANIFEST_PATH, asset_names=(vehicle_name,))
    newton.wheeled.configure_wheel_axle_joints(world, axle_joint_labels=asset.axle_joint_labels)

    scene = newton.ModelBuilder()
    _configure_zero_gap_builder(scene)
    scene.replicate(world, world_count)
    terrain_shape = _add_vehicle_terrain(scene, terrain_kind)
    model = scene.finalize(device="cpu")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    metadata = newton.wheeled.build_wheeled_metadata(model)
    patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
    tire_control = newton.wheeled.WheelTireControl(model, metadata)
    tire_state = newton.wheeled.WheelTireState(model, metadata)
    vehicle_layout = newton.wheeled.build_wheeled_vehicle_layout(
        model,
        metadata,
        manifest_path=_MANIFEST_PATH,
        asset_names=(vehicle_name,),
    )
    vehicle_control = newton.wheeled.WheeledVehicleControl(vehicle_layout)
    vehicle_state = newton.wheeled.WheeledVehicleState(vehicle_layout)
    tire_parameters = _vehicle_tire_parameters(vehicle_name)
    tire_mu, fallback_normal_load, longitudinal_stiffness, lateral_stiffness, max_wheel_speed = tire_parameters
    motor_config = newton.wheeled.WheeledMotorConfig(vehicle_layout, max_wheel_angular_speed=max_wheel_speed)
    steering_config = newton.wheeled.WheeledSteeringConfig(vehicle_layout)
    newton.wheeled.configure_wheel_tire_control(
        tire_control,
        friction_mu=tire_mu,
        fallback_normal_load=fallback_normal_load,
        longitudinal_stiffness=longitudinal_stiffness,
        lateral_stiffness=lateral_stiffness,
    )
    newton.wheeled.configure_mujoco_wheel_contacts(model, metadata)
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        disable_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        njmax=max(256 * world_count, model.rigid_contact_max),
        nconmax=max(128 * world_count, model.rigid_contact_max),
        iterations=10,
        ls_iterations=50,
    )

    return VehicleTerrainFixture(
        model,
        state_0,
        state_1,
        control,
        contacts,
        metadata,
        patch_state,
        tire_control,
        tire_state,
        vehicle_layout,
        vehicle_control,
        vehicle_state,
        motor_config,
        steering_config,
        solver,
        terrain_shape,
    )


def _skid_steer_drive_commands(
    layout: newton.wheeled.WheeledVehicleLayout,
    *,
    left: float,
    right: float,
) -> np.ndarray:
    commands = np.zeros(layout.drive_channel_count, dtype=np.float32)
    channel_sides = np.zeros(layout.drive_channel_count, dtype=np.int32)
    for wheel_id, channel in enumerate(layout.wheel_drive_channel_host):
        if channel >= 0 and channel_sides[channel] == 0:
            channel_sides[channel] = layout.wheel_side_host[wheel_id]
    for channel, side in enumerate(channel_sides):
        if side == newton.wheeled.WheeledVehicleLayout.WheelSide.LEFT:
            commands[channel] = left
        else:
            commands[channel] = right
    return commands


def _simulate_vehicle_terrain(
    fixture: VehicleTerrainFixture,
    *,
    drive_command: np.ndarray,
    steering_command: np.ndarray | None = None,
    step_count: int = 12,
) -> None:
    if steering_command is None:
        steering_command = np.zeros(fixture.vehicle_layout.steering_channel_count, dtype=np.float32)
    newton.wheeled.configure_wheeled_vehicle_control(
        fixture.vehicle_control,
        drive_command=drive_command,
        steering_command=steering_command,
    )

    for _ in range(step_count):
        fixture.state_0.clear_forces()
        fixture.control.clear()
        newton.wheeled.update_wheeled_vehicle_controls(
            fixture.model,
            fixture.control,
            fixture.metadata,
            fixture.vehicle_layout,
            fixture.vehicle_control,
            fixture.vehicle_state,
            fixture.tire_control,
            motor_config=fixture.motor_config,
            steering_config=fixture.steering_config,
        )
        fixture.model.collide(fixture.state_0, fixture.contacts)
        newton.wheeled.update_wheel_contact_patches(
            fixture.model,
            fixture.state_0,
            fixture.contacts,
            fixture.metadata,
            fixture.patch_state,
        )
        newton.wheeled.apply_wheel_tire_forces(
            fixture.model,
            fixture.state_0,
            fixture.metadata,
            fixture.patch_state,
            fixture.tire_control,
            fixture.tire_state,
        )
        fixture.solver.step(fixture.state_0, fixture.state_1, fixture.control, fixture.contacts, 1.0 / 240.0)
        fixture.state_0, fixture.state_1 = fixture.state_1, fixture.state_0

    fixture.model.collide(fixture.state_0, fixture.contacts)
    newton.wheeled.update_wheel_contact_patches(
        fixture.model,
        fixture.state_0,
        fixture.contacts,
        fixture.metadata,
        fixture.patch_state,
    )


def _assert_vehicle_terrain_patch_state(test: unittest.TestCase, fixture: VehicleTerrainFixture) -> None:
    test.assertTrue(np.isfinite(fixture.state_0.body_q.numpy()).all())
    test.assertTrue(np.isfinite(fixture.state_0.body_qd.numpy()).all())
    np.testing.assert_array_equal(fixture.patch_state.active.numpy(), np.ones(fixture.metadata.wheel_count, dtype=bool))
    test.assertTrue(np.all(fixture.patch_state.contact_count.numpy() > 0))
    test.assertTrue(np.isfinite(fixture.patch_state.center.numpy()).all())
    test.assertTrue(np.isfinite(fixture.patch_state.normal.numpy()).all())
    test.assertTrue(np.isfinite(fixture.patch_state.patch_u_extent.numpy()).all())
    test.assertTrue(np.isfinite(fixture.patch_state.patch_v_extent.numpy()).all())
    test.assertTrue(np.isfinite(fixture.patch_state.patch_area.numpy()).all())
    np.testing.assert_array_equal(
        fixture.patch_state.terrain_shape_index.numpy(),
        np.full(fixture.metadata.wheel_count, fixture.terrain_shape, dtype=np.int32),
    )
    np.testing.assert_allclose(
        fixture.patch_state.friction_mu_seed.numpy(),
        np.full(fixture.metadata.wheel_count, TERRAIN_MU, dtype=np.float32),
        rtol=1.0e-6,
    )
    normals = fixture.patch_state.normal.numpy()
    test.assertGreater(float(np.min(normals[:, 2])), 0.9)
    test.assertGreater(abs(float(np.mean(normals[:, 0]))), 0.02)
    gaps = fixture.model.shape_gap.numpy()
    test.assertAlmostEqual(float(gaps[fixture.terrain_shape]), 0.0, delta=1.0e-9)
    np.testing.assert_allclose(gaps[list(fixture.metadata.wheel_shape_indices)], 0.0, atol=1.0e-9)


class TestWheelTerrainContactFixture(unittest.TestCase):
    def test_gap_zero_cylinder_plane_patch_and_material(self):
        fixture = _build_fixture("plane", gap=0.0)

        gaps = fixture.model.shape_gap.numpy()
        margins = fixture.model.shape_margin.numpy()
        self.assertAlmostEqual(float(gaps[fixture.terrain_shape]), 0.0, delta=1.0e-9)
        self.assertAlmostEqual(float(gaps[fixture.wheel_shape]), 0.0, delta=1.0e-9)
        self.assertAlmostEqual(float(margins[fixture.terrain_shape]), 0.0, delta=1.0e-9)
        self.assertAlmostEqual(float(margins[fixture.wheel_shape]), 0.0, delta=1.0e-9)
        self.assertLess(float(fixture.state.body_q.numpy()[fixture.wheel_body, 2]), WHEEL_RADIUS)

        _assert_finite_patch(self, fixture)
        self.assertGreaterEqual(int(fixture.patch_state.contact_count.numpy()[0]), 2)
        self.assertEqual(_classify_patch(fixture.patch_state), "line")
        self.assertGreater(float(fixture.patch_state.patch_v_extent.numpy()[0]), 0.5 * WHEEL_WIDTH)
        self.assertAlmostEqual(float(fixture.patch_state.patch_area.numpy()[0]), 0.0, delta=1.0e-6)

    def test_nonzero_gap_can_create_separated_contacts_but_baseline_uses_zero(self):
        separated_zero_gap = _build_fixture("plane", gap=0.0, wheel_z=WHEEL_RADIUS + 0.005)
        separated_nonzero_gap = _build_fixture("plane", gap=0.01, wheel_z=WHEEL_RADIUS + 0.005)

        self.assertFalse(bool(separated_zero_gap.patch_state.active.numpy()[0]))
        self.assertTrue(bool(separated_nonzero_gap.patch_state.active.numpy()[0]))
        self.assertAlmostEqual(
            float(separated_nonzero_gap.model.shape_gap.numpy()[separated_nonzero_gap.wheel_shape]),
            0.01,
            delta=1.0e-9,
        )

    def test_cylinder_plane_raw_contact_cloud_is_line_like(self):
        fixture = _build_fixture("plane", gap=0.0)
        points = _wheel_contact_points(fixture)

        self.assertGreaterEqual(points.shape[0], 2)
        self.assertEqual(_classify_patch(fixture.patch_state), "line")
        self.assertLessEqual(_line_residual_ratio(points), 1.0e-5)

    def test_analytic_plane_patch_mode_matches_cylinder_plane_footprint(self):
        fixture = _build_fixture("plane", gap=0.0)

        raw = _projection_patch_snapshot(
            fixture,
            enable_axial_contact_projection=True,
            enable_analytic_plane_patches=False,
        )
        analytic = _projection_patch_snapshot(
            fixture,
            enable_axial_contact_projection=True,
            enable_analytic_plane_patches=True,
        )

        self.assertEqual(_classify_projection_snapshot(raw), "line")
        self.assertEqual(_classify_projection_snapshot(analytic), "area")
        self.assertEqual(int(analytic.terrain_shape_index[0]), fixture.terrain_shape)
        self.assertAlmostEqual(float(analytic.friction_mu_seed[0]), TERRAIN_MU, delta=1.0e-6)
        np.testing.assert_allclose(analytic.normal[0], np.array([0.0, 0.0, 1.0]), atol=1.0e-6)
        self.assertAlmostEqual(float(analytic.center[0, 2]), 0.0, delta=1.0e-6)

        wheel_z = float(fixture.state.body_q.numpy()[fixture.wheel_body, 2])
        sink_depth = WHEEL_RADIUS - wheel_z
        expected_chord = 2.0 * math.sqrt(max(0.0, 2.0 * WHEEL_RADIUS * sink_depth - sink_depth * sink_depth))
        expected_area = expected_chord * WHEEL_WIDTH

        self.assertAlmostEqual(float(analytic.patch_u_extent[0]), expected_chord, delta=1.0e-5)
        self.assertAlmostEqual(float(analytic.patch_v_extent[0]), WHEEL_WIDTH, delta=1.0e-6)
        self.assertAlmostEqual(float(analytic.patch_area[0]), expected_area, delta=1.0e-5)
        self.assertGreater(float(analytic.patch_area[0]), float(raw.patch_area[0]) + 1.0e-3)

    def test_primitive_and_mesh_terrain_patches_remain_finite(self):
        expected_classifications = {
            "flat_box": "line",
            "ramp": "line",
            "ridge": "line",
            "mesh_ripple": "area",
            "mesh_jump": "line",
        }
        for terrain_kind, expected_classification in expected_classifications.items():
            with self.subTest(terrain_kind=terrain_kind):
                fixture = _build_fixture(terrain_kind, gap=0.0)
                _assert_finite_patch(self, fixture)
                self.assertEqual(_classify_patch(fixture.patch_state), expected_classification)
                if expected_classification == "area":
                    self.assertGreater(float(fixture.patch_state.patch_area.numpy()[0]), 0.0)

    def test_axial_projection_diagnostic_leaves_plane_cylinder_baseline_unchanged(self):
        fixture = _build_fixture("plane", gap=0.0)

        projected = _projection_patch_snapshot(fixture, enable_axial_contact_projection=True)
        no_projection = _projection_patch_snapshot(fixture, enable_axial_contact_projection=False)

        _assert_projection_snapshot_finite(self, projected)
        _assert_projection_snapshot_finite(self, no_projection)
        np.testing.assert_allclose(projected.center, no_projection.center, atol=1.0e-6)
        np.testing.assert_allclose(projected.normal, no_projection.normal, atol=1.0e-6)
        np.testing.assert_allclose(projected.patch_u_extent, no_projection.patch_u_extent, atol=1.0e-6)
        np.testing.assert_allclose(projected.patch_v_extent, no_projection.patch_v_extent, atol=1.0e-6)
        np.testing.assert_allclose(projected.patch_area, no_projection.patch_area, atol=1.0e-6)

    def test_plane_cylinder_collapse_diagnostic_compares_flat_plane_gjk_path(self):
        fixture = _build_fixture("plane", gap=0.0)

        collapsed = _projection_patch_snapshot(fixture, enable_axial_contact_projection=True)
        gjk_projected = _projection_patch_snapshot(
            fixture,
            enable_axial_contact_projection=True,
            enable_plane_cylinder_contact_collapse=False,
        )
        gjk_no_projection = _projection_patch_snapshot(
            fixture,
            enable_axial_contact_projection=False,
            enable_plane_cylinder_contact_collapse=False,
        )

        _assert_projection_snapshot_finite(self, collapsed)
        _assert_projection_snapshot_finite(self, gjk_projected)
        _assert_projection_snapshot_finite(self, gjk_no_projection)
        np.testing.assert_array_equal(collapsed.terrain_shape_index, gjk_no_projection.terrain_shape_index)
        np.testing.assert_allclose(collapsed.friction_mu_seed, gjk_no_projection.friction_mu_seed, atol=1.0e-6)
        np.testing.assert_allclose(collapsed.normal, gjk_no_projection.normal, atol=1.0e-6)

        self.assertEqual(_classify_projection_snapshot(collapsed), "line")
        self.assertEqual(_classify_projection_snapshot(gjk_projected), "line")
        self.assertEqual(_classify_projection_snapshot(gjk_no_projection), "area")
        self.assertGreater(gjk_no_projection.contact_count, collapsed.contact_count)
        self.assertGreater(float(gjk_no_projection.patch_area[0]), float(collapsed.patch_area[0]) + 1.0e-3)

    def test_axial_projection_diagnostic_compares_primitive_discrete_terrain(self):
        fixture = _build_fixture("ramp", gap=0.0)

        projected = _projection_patch_snapshot(fixture, enable_axial_contact_projection=True)
        no_projection = _projection_patch_snapshot(fixture, enable_axial_contact_projection=False)

        _assert_projection_snapshot_finite(self, projected)
        _assert_projection_snapshot_finite(self, no_projection)
        np.testing.assert_array_equal(projected.terrain_shape_index, no_projection.terrain_shape_index)
        np.testing.assert_allclose(projected.friction_mu_seed, no_projection.friction_mu_seed, atol=1.0e-6)
        self.assertEqual(int(projected.terrain_shape_index[0]), fixture.terrain_shape)
        self.assertEqual(_classify_projection_snapshot(projected), "line")
        self.assertEqual(_classify_projection_snapshot(no_projection), "area")
        self.assertGreater(float(no_projection.patch_area[0]), float(projected.patch_area[0]) + 1.0e-3)

    def test_axial_projection_diagnostic_compares_triangle_mesh_terrain(self):
        fixture = _build_fixture("mesh_ripple", gap=0.0)

        projected = _projection_patch_snapshot(fixture, enable_axial_contact_projection=True)
        no_projection = _projection_patch_snapshot(fixture, enable_axial_contact_projection=False)

        _assert_projection_snapshot_finite(self, projected)
        _assert_projection_snapshot_finite(self, no_projection)
        np.testing.assert_array_equal(projected.terrain_shape_index, no_projection.terrain_shape_index)
        np.testing.assert_allclose(projected.friction_mu_seed, no_projection.friction_mu_seed, atol=1.0e-6)
        self.assertEqual(int(projected.terrain_shape_index[0]), fixture.terrain_shape)
        self.assertEqual(_classify_projection_snapshot(projected), "area")
        self.assertEqual(_classify_projection_snapshot(no_projection), "area")
        self.assertGreater(float(no_projection.patch_area[0]), float(projected.patch_area[0]) + 1.0e-3)

    def test_mesh_ripple_patch_stability_over_low_speed_offsets(self):
        fixture = _build_fixture("mesh_ripple", gap=0.0)
        base_body_q = fixture.state.body_q.numpy().copy()
        centers = []
        normals = []
        areas = []

        for x_offset in np.linspace(-0.02, 0.02, 5, dtype=np.float32):
            body_q = base_body_q.copy()
            body_q[fixture.wheel_body, 0] += x_offset
            fixture.state.body_q.assign(body_q)
            fixture.model.collide(fixture.state, fixture.contacts)
            newton.wheeled.update_wheel_contact_patches(
                fixture.model,
                fixture.state,
                fixture.contacts,
                fixture.metadata,
                fixture.patch_state,
            )
            _assert_finite_patch(self, fixture)
            centers.append(fixture.patch_state.center.numpy()[0].copy())
            normals.append(fixture.patch_state.normal.numpy()[0].copy())
            areas.append(float(fixture.patch_state.patch_area.numpy()[0]))

        centers = np.asarray(centers, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        areas = np.asarray(areas, dtype=np.float32)
        normal_deviation = 1.0 - np.clip(normals @ normals[0], -1.0, 1.0)

        self.assertLess(float(np.max(normal_deviation)), 1.0e-3)
        self.assertLess(float(np.ptp(centers[:, 2])), 5.0e-3)
        self.assertLess(float(np.ptp(areas)), 5.0e-3)

    def test_tire_forces_consume_terrain_patch_without_nonfinite_state(self):
        fixture = _build_fixture("mesh_ripple", gap=0.0)
        _assert_finite_patch(self, fixture)

        control = newton.wheeled.WheelTireControl(fixture.model, fixture.metadata)
        tire_state = newton.wheeled.WheelTireState(fixture.model, fixture.metadata)
        newton.wheeled.configure_wheel_tire_control(
            control,
            friction_mu=-1.0,
            fallback_normal_load=50.0,
            longitudinal_stiffness=20.0,
            lateral_stiffness=10.0,
        )
        control.wheel_angular_speed.assign(np.array([4.0], dtype=np.float32))

        fixture.state.clear_forces()
        newton.wheeled.apply_wheel_tire_forces(
            fixture.model,
            fixture.state,
            fixture.metadata,
            fixture.patch_state,
            control,
            tire_state,
        )

        self.assertTrue(np.isfinite(fixture.state.body_f.numpy()).all())
        self.assertTrue(np.isfinite(tire_state.applied_longitudinal_force.numpy()).all())
        self.assertGreater(float(tire_state.friction_limit.numpy()[0]), 0.0)
        self.assertGreater(abs(float(tire_state.applied_longitudinal_force.numpy()[0])), 0.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_rc_car_ramp_run_uses_vehicle_controls_and_tire_patches(self):
        fixture = _build_vehicle_terrain_fixture("rc_car", "ramp_box")
        drive_command = np.full(fixture.vehicle_layout.drive_channel_count, 0.2, dtype=np.float32)
        steering_command = np.full(fixture.vehicle_layout.steering_channel_count, 0.1, dtype=np.float32)

        _simulate_vehicle_terrain(fixture, drive_command=drive_command, steering_command=steering_command)

        _assert_vehicle_terrain_patch_state(self, fixture)
        self.assertGreater(float(np.max(np.abs(fixture.vehicle_state.wheel_angular_speed.numpy()))), 0.0)
        self.assertGreater(float(np.max(np.abs(fixture.vehicle_state.steering_angle.numpy()))), 0.0)
        self.assertGreater(float(np.max(np.abs(fixture.tire_state.applied_longitudinal_force.numpy()))), 0.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_husky_mesh_slope_run_accepts_left_right_drive_commands(self):
        fixture = _build_vehicle_terrain_fixture("husky", "mesh_slope")
        drive_command = _skid_steer_drive_commands(fixture.vehicle_layout, left=0.2, right=0.25)

        _simulate_vehicle_terrain(fixture, drive_command=drive_command)

        _assert_vehicle_terrain_patch_state(self, fixture)
        np.testing.assert_allclose(fixture.vehicle_state.drive_command.numpy(), drive_command)
        self.assertGreater(float(np.max(np.abs(fixture.vehicle_state.wheel_angular_speed.numpy()))), 0.0)
        self.assertGreater(float(np.max(np.abs(fixture.tire_state.applied_longitudinal_force.numpy()))), 0.0)
        self.assertGreater(float(np.max(fixture.vehicle_state.wheel_angular_speed.numpy())), 0.0)
        self.assertGreater(float(np.ptp(fixture.vehicle_state.wheel_angular_speed.numpy())), 0.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_replicated_vehicle_mesh_slope_contacts_batch_by_wheel(self):
        fixture = _build_vehicle_terrain_fixture("husky", "mesh_slope", world_count=2)
        fixture.model.collide(fixture.state_0, fixture.contacts)
        newton.wheeled.update_wheel_contact_patches(
            fixture.model,
            fixture.state_0,
            fixture.contacts,
            fixture.metadata,
            fixture.patch_state,
        )

        self.assertEqual(fixture.model.world_count, 2)
        self.assertEqual(fixture.vehicle_layout.vehicle_count, 2)
        self.assertEqual(fixture.metadata.wheel_count, 8)
        _assert_vehicle_terrain_patch_state(self, fixture)

    def test_multi_world_cylinder_plane_contacts_batch_by_wheel(self):
        builder = newton.ModelBuilder(gravity=0.0)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        terrain_shapes = []
        for wheel_id, x in enumerate((0.0, 1.0)):
            builder.begin_world()
            terrain_shapes.append(builder.add_ground_plane(cfg=_shape_config(gap=0.0), label=f"terrain_{wheel_id}"))
            _add_cylinder_wheel(builder, wheel_id=wheel_id, x=x, z=0.49, gap=0.0)
            builder.end_world()

        model = builder.finalize(device="cpu")
        metadata = newton.wheeled.build_wheeled_metadata(model)
        state = model.state()
        contacts = model.contacts()
        model.collide(state, contacts)
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)

        self.assertEqual(model.world_count, 2)
        np.testing.assert_array_equal(patch_state.active.numpy(), np.array([True, True]))
        np.testing.assert_array_equal(patch_state.terrain_shape_index.numpy(), np.array(terrain_shapes, dtype=np.int32))
        self.assertTrue(np.isfinite(patch_state.center.numpy()).all())
        self.assertTrue(np.all(patch_state.contact_count.numpy() >= 2))


if __name__ == "__main__":
    unittest.main()
