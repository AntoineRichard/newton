# Wheeled Vehicle Phase 5A Cylinder Contact Projection And Collapse Diagnostics

## Scope

This report records A/B diagnostics for cylinder wheel contacts with and without
axial rolling projection, and with the analytical plane-cylinder contact collapse
enabled or disabled. The diagnostics do not change the default collision path.
They use:

- `CollisionPipeline(enable_axial_contact_projection=False)` to bypass the GJK/MPR
  axial rolling post-process.
- `CollisionPipeline(enable_plane_cylinder_contact_collapse=False)` to bypass the
  analytical plane-cylinder primitive helper and route flat plane-cylinder pairs
  through GJK/MPR.

The comparison targets representative discrete terrain contacts and the flat
plane-cylinder case that matters for wheel footprint estimation.

## Setup

The focused tests live in
`newton/tests/test_wheeled_vehicle_terrain_contact.py`.

- Wheel geometry: cylinder radius `0.5 m`, width `0.2 m`, axle aligned with
  world `Y`.
- Contact setup: wheel and terrain `shape_gap = 0.0`, `shape_margin = 0.0`.
- Diagnostic collision setup: `CollisionPipeline(broad_phase="explicit",
  reduce_contacts=False, deterministic=True)`.
- Projected mode: `enable_axial_contact_projection=True`.
- No-projection mode: `enable_axial_contact_projection=False`.
- Analytical plane-cylinder mode: `enable_plane_cylinder_contact_collapse=True`.
- GJK plane-cylinder diagnostic mode:
  `enable_plane_cylinder_contact_collapse=False`.
- Tire-force probe: one wheel with `4.0 rad/s` analytical angular speed,
  `50.0 N` fallback normal load, and the current saturated linear tire model.

## Measurements

| Terrain | Mode | Contacts | Wheel contacts | Class | Normal | Extents [m] | Area [m^2] | Tire force [N] |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| Plane | analytical, projected | 2 | 2 | line | `[0.0000, 0.0000, 1.0000]` | `(0, 0.200)` | `0` | `38.5` |
| Plane | analytical, no projection | 2 | 2 | line | `[0.0000, 0.0000, 1.0000]` | `(0, 0.200)` | `0` | `38.5` |
| Plane | GJK, projected | 5 | 5 | line | `[0.0000, 0.0000, 1.0000]` | `(0, 0.200)` | `0` | `38.5` |
| Plane | GJK, no projection | 5 | 5 | area | `[0.0000, 0.0000, 1.0000]` | `(0.0316, 0.200)` | `6.31e-3` | `38.5` |
| Ramp box | projected | 4 | 4 | line | `[-0.1987, 0.0000, 0.9801]` | `(3.05e-5, 0.200)` | `6.09e-6` | `38.5` |
| Ramp box | no projection | 4 | 4 | area | `[-0.1987, 0.0000, 0.9801]` | `(0.0323, 0.200)` | `6.46e-3` | `38.5` |
| Mesh ripple | projected | 10 | 10 | area | `[-0.0100, -0.0100, 0.9999]` | `(0.0100, 0.200)` | `2.00e-3` | `38.5` |
| Mesh ripple | no projection | 10 | 10 | area | `[-0.0100, -0.0100, 0.9999]` | `(0.0416, 0.200)` | `8.32e-3` | `38.5` |
| Mesh jump | projected | 9 | 9 | line | `[-0.0995, -0.0000, 0.9950]` | `(1.04e-5, 0.200)` | `2.07e-6` | `38.5` |
| Mesh jump | no projection | 9 | 9 | area | `[-0.0995, -0.0000, 0.9950]` | `(0.0314, 0.200)` | `6.28e-3` | `38.5` |

The simple tire-force probe is unchanged because the current saturated linear
tire model does not use patch area. It uses contact activity, patch normal,
kinematics, normal load, and friction limit.

## Interpretation

The diagnostic confirms two separate sources of lost footprint information:

- The analytical plane-cylinder helper emits a line-like contact set for a flat
  wheel on a plane. Disabling axial projection alone cannot recover any hidden
  area because GJK/MPR is never used for that pair.
- When the plane-cylinder helper is bypassed, GJK/MPR produces a wider flat-plane
  contact set, but axial rolling projection still collapses it back to a line.
  The flat-plane area appears only with both
  `enable_plane_cylinder_contact_collapse=False` and
  `enable_axial_contact_projection=False`.

The non-plane discrete terrain result remains the same as the first diagnostic:
contact count, terrain shape identity, material friction seed, and patch normal
stay stable, while the smaller tangent extent changes from near-zero to roughly
`3-4 cm` when axial projection is disabled.

This does not yet justify changing default cylinder contacts globally. The
analytical plane-cylinder path and axial rolling projection are general collision
stabilization behaviors, and the current tire model does not consume patch area.

## Decision

Keep the default collision path unchanged. Use
`CollisionPipeline(enable_plane_cylinder_contact_collapse=False,
enable_axial_contact_projection=False)` as an internal diagnostic tool for flat
wheel-cylinder footprint studies, and use
`CollisionPipeline(enable_axial_contact_projection=False)` for non-plane terrain
projection studies.

Do not add raw and projected contacts to the public `Contacts` structure yet. If
a later tire model needs finite rigid patch area, the next design should be a
scoped wheel-terrain collision mode or wheeled diagnostic helper, not a global
change to all cylinder/cone contacts.

## Follow-Up

- Re-run this diagnostic once a tire model consumes patch area or footprint
  length directly.
- If no-collapse/no-projection contacts improve tire output stability on moving
  vehicles, design a per-shape or per-pair wheel-terrain contact mode.
- Keep the hydroelastic study separate; hydroelastic contacts may still be the
  better source of physical patch area on uneven terrain.
