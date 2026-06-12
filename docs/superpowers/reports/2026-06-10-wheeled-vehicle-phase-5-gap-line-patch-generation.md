# Wheeled Vehicle Phase 5 Gap, Line, And Patch Generation Report

## Scope

This report records the first Phase 5 contact-patch measurements for cylinder
wheels on primitive and mesh terrain. Heightfields are intentionally out of
scope for this pass. Non-flat support should be tested with boxes, jumps, and
small triangle meshes so we exercise the same collision contacts used by the
wheel fixtures.

## Test Setup

The focused regression fixture is
`newton/tests/test_wheeled_vehicle_terrain_contact.py`.

- Wheel geometry: cylinder radius `0.5 m`, width `0.2 m`, axle aligned with
  world `Y`.
- Baseline contact setup: wheel and terrain `shape_gap = 0.0`,
  `shape_margin = 0.0`.
- Material setup: terrain friction seed `mu = 0.77`.
- Contact path: `Model.collide()` followed by
  `newton.wheeled.update_wheel_contact_patches()`.
- Classification: local test thresholds classify contact clouds as point-like,
  line-like, or area-like from the two tangent extents.

The baseline wheel placement sinks the cylinder slightly into the support
surface. This matches the current solver guidance: use zero gap and allow a
small physical overlap instead of relying on an inflated contact shell.

## Measurements

| Terrain case | Raw contacts | Wheel contacts | Normal | Extents [m] | Area [m2] | Class | Line residual |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| Plane | 2 | 2 | `[0.000, 0.000, 1.000]` | `(0, 0.200)` | `0` | line | `0` |
| Flat box | 5 | 5 | `[-0.000, -0.000, 1.000]` | `(1.75e-10, 0.200)` | `3.50e-11` | line | `4.03e-4` |
| Ramp box | 4 | 4 | `[-0.199, 0.000, 0.980]` | `(3.05e-5, 0.200)` | `6.09e-6` | line | `1.69e-4` |
| Low box ridge | 5 | 5 | `[-0.000, -0.000, 1.000]` | `(4.61e-11, 0.200)` | `9.21e-12` | line | `4.05e-4` |
| Mesh ripple | 6 | 6 | `[-0.010, -0.010, 1.000]` | `(0.010, 0.200)` | `2.00e-3` | area | `4.08e-2` |
| Mesh jump | 4 | 4 | `[-0.100, -0.000, 0.995]` | `(1.04e-5, 0.200)` | `2.07e-6` | line | `4.18e-4` |

The gap sweep used a wheel positioned `5 mm` above the radius. With
`shape_gap = 0.0`, no contact patch was active. With `shape_gap = 0.01`, the
same separated wheel produced two contacts. This confirms the zero-gap baseline
is not creating support through an inflated detection shell.

A low-speed mesh-ripple stability test offsets the same cylinder wheel across
five nearby positions. The accepted bounds are maximum normal deviation below
`1.0e-3`, patch-center `z` range below `5 mm`, and patch-area range below
`5.0e-3 m2`. This keeps the first stability check deterministic without turning
Phase 5 into a high-fidelity tire-footprint model.

## Vehicle-Level Validation

The same Phase 5 test file now includes short vehicle-level terrain runs using
the simplified Phase 00 assets:

- RC car on a shallow ramp box: the asset is pitched with the ramp, wheel and
  terrain gaps are zero, all four wheel patches remain active after a 12-step
  command-mapped tire-force run, and steering plus longitudinal tire force
  targets are nonzero.
- Husky on a shallow triangle-mesh slope: direct left/right skid-steer drive
  commands produce different wheel speed targets, all four patches remain
  active, and patch normals track the non-flat mesh normal.
- Replicated Husky mesh-slope scene: two worlds produce eight active wheel
  patches with the shared terrain shape id and material friction seed.

See also
`docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-contact-observability.md`
for the MuJoCo-native contact observability comparison.

## Source Interpretation

Flat plane-cylinder contacts use an analytical path. The narrow phase routes
plane-cylinder pairs to `collide_plane_cylinder()` in
`newton/_src/geometry/narrow_phase.py`. That helper explicitly describes two
modes: a near-upright flat-surface mode and a rolling mode. Rolling mode emits a
deepest rim point plus side-generator contacts; one generator often merges with
the deepest point. For a wheel cylinder with axle perpendicular to the ground
normal, line-like contact across tire width is the expected rigid geometry, not
an obvious patch-estimator bug.

The wheel patch estimator in `newton/_src/wheeled/contact_patch.py` consumes raw
Newton rigid contacts directly. It averages wheel-side contact points for the
patch center, averages support normals, records the minimum counterpart terrain
shape index, seeds friction from `shape_material_mu`, and computes tangent
extents from min/max projections. It does not synthesize an area, inflate a line
into a rectangle, or estimate a compliant footprint.

There is also an axial rolling stabilization post-process in
`newton/_src/geometry/collision_core.py`. For cylinder/cone contacts against
discrete surfaces, when the axial shape is in rolling configuration, it projects
the contact point onto a rolling stabilization plane. This may affect box, mesh,
or other discrete terrain contacts. The current measurements do not show a
blocking failure from that projection: primitive and mesh jump contacts remain
finite, material/shape data are preserved, and tire forces can consume the mesh
ripple patch without non-finite state. The projection should therefore be kept
as-is until we have a concrete case where it damages wheel tire inputs.

## Current Decision

We can get useful wheel patches from Newton rigid contacts today, but they are
not guaranteed to have nonzero area. For rigid cylinder wheels:

- center, normal, contact count, terrain shape id, material friction seed, and
  tire-width extent are usable;
- flat plane, flat box, ramp, ridge, and mesh jump support are usually line-like;
- a triangulated non-flat ripple can produce a small area-like patch;
- patch area should be treated as diagnostic only for now, not as an input that
  tire force computation depends on.

No collision-pipeline opt-out should be added yet. The better next step is to
harden tests around moving vehicle cases and keep the tire model independent of
patch area. If we later need a finite physical contact area, it should probably
come from a wheel-specific patch estimator or a compliant/hydroelastic contact
path, not from pretending that rigid cylinder-plane contact naturally produces a
rectangle.

## Follow-Up

- Broaden frame-to-frame stability tests while wheels move over ramps, ridges,
  jumps, and simple triangle meshes.
- Extend vehicle-level terrain runs beyond the current short RC car and Husky
  smoke tests if a future tire model depends on patch area or contact jitter.
- If tire models need contact area, add a documented area fallback based on tire
  width and penetration/compression diagnostics, with tests that prove why it is
  needed.
- If axial rolling projection is suspected to harm wheel patches, build an A/B
  diagnostic before adding a public option. A scoped wheel-terrain contact mode
  is preferable to changing all cylinder contacts.
- Keep heightfield work out of Phase 5; revisit it only if a later terrain
  asset path requires it.
