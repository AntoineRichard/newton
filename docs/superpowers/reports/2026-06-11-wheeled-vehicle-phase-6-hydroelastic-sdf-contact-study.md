# Wheeled Vehicle Phase 6: Hydroelastic/SDF Contact Patch Study

Date: 2026-06-11

## Summary

Hydroelastic/SDF contact is a promising optional patch source for wheel-terrain
contact, especially once the terrain is represented as volumetric SDF geometry
instead of a plane or heightfield. In a simple cylinder-over-box probe, the
hydroelastic contact surface produced an area-bearing patch whose longitudinal
span and total area grew with sink depth. The ordinary rigid cylinder contact
path stayed at four contacts and mostly represented a line.

This should not replace the current rigid-contact path by default. The current
implementation requires CUDA for SDF texture construction, cannot use planes or
heightfields as hydroelastic shapes, and needs explicit volumetric terrain
geometry. It is better treated as an optional higher-quality patch mode and as a
study target for non-flat terrain.

## Code Path Findings

- User-facing configuration is available through
  `newton.ModelBuilder.ShapeConfig` and `newton.geometry.HydroelasticSDF`.
- Hydroelastic contact is selected only for SDF-vs-SDF shape pairs where both
  shapes have the hydroelastic flag.
- Primitive hydroelastic shapes require `sdf_max_resolution` or
  `sdf_target_voxel_size`; the builder then creates a watertight primitive mesh
  and builds a texture SDF from it.
- Mesh hydroelastic shapes require a prebuilt mesh SDF via `mesh.build_sdf(...)`.
- Planes and heightfields are not hydroelastic shapes, so wheel/terrain probes
  must use volumetric boxes, ramps, obstacle meshes, or terrain meshes rather
  than `add_ground_plane()` or heightfields.
- The SDF/hydroelastic path is CUDA-only today. Builder finalization rejects SDF
  collision paths on CPU because texture SDFs require CUDA.
- `HydroelasticSDF.Config(output_contact_surface=True)` exposes
  `ContactSurfaceData` with world-space triangle vertices, per-face depth, shape
  pairs, and face count.
- Hydroelastic contacts are exported as ordinary Newton contacts. The export
  path writes hydro-derived contact stiffness and optional friction scale into
  the regular contact buffer; the downstream solver still resolves ordinary
  contacts. It does not bypass MuJoCo friction or install a separate friction
  solver.

## Probe Setup

The probes were run as throwaway `uv run --extra dev python` scripts. No examples
were modified.

Common setup:

- Device: `cuda:0`.
- Wheel: cylinder, radius `0.055 m`, width `0.045 m`.
- Terrain: static volumetric box, top surface at `z = 0`.
- Wheel axis: cylinder local Z rotated to world Y.
- Shape gap: `0.0 m` unless otherwise noted.
- Hydroelastic material stiffness: `kh = 1.0e9`.
- Hydroelastic SDF resolution: `64` unless otherwise noted.
- Hydroelastic narrow band: `[-0.03, 0.03] m`.

## Rigid Contact Baseline

Rigid cylinder-over-box contacts did not provide a useful patch area. With the
default rigid path, every penetrating case produced four contacts spread across
the wheel width but nearly collapsed in the longitudinal rolling direction.
Disabling the later axial projection widened the longitudinal extent to about
`3.47 mm`, but that extent did not grow with sink depth.

| Sink depth | Mode | Contacts | Longitudinal span | Width span |
| --- | --- | ---: | ---: | ---: |
| `1 mm` | default rigid | 4 | `~0.00 mm` | `45.00 mm` |
| `3 mm` | default rigid | 4 | `~0.00 mm` | `45.00 mm` |
| `6 mm` | default rigid | 4 | `~0.00 mm` | `45.00 mm` |
| `10 mm` | default rigid | 4 | `~0.00 mm` | `45.00 mm` |
| `1 mm` | no axial projection | 4 | `3.47 mm` | `45.00 mm` |
| `3 mm` | no axial projection | 4 | `3.47 mm` | `45.00 mm` |
| `6 mm` | no axial projection | 4 | `3.47 mm` | `45.00 mm` |
| `10 mm` | no axial projection | 4 | `3.47 mm` | `45.00 mm` |

## Hydroelastic Contact Surface

Hydroelastic contact generated a real area patch. The contact surface face count
and area grew with sink depth.

| Sink depth | Solver contacts, unreduced | Surface faces | Surface area | Longitudinal span | Width span | Mean depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0 mm` | 0 | 0 | `0 mm^2` | `0.00 mm` | `0.00 mm` | `0.000 mm` |
| `1 mm` | 599 | 599 | `859 mm^2` | `20.62 mm` | `45.72 mm` | `-0.260 mm` |
| `3 mm` | 1233 | 1233 | `1587 mm^2` | `34.38 mm` | `45.44 mm` | `-0.950 mm` |
| `6 mm` | 2017 | 2017 | `2468 mm^2` | `51.56 mm` | `45.87 mm` | `-1.807 mm` |
| `10 mm` | 2637 | 2637 | `3134 mm^2` | `64.64 mm` | `45.31 mm` | `-2.884 mm` |

With contact reduction enabled, solver contacts dropped to roughly `19-34`
contacts in the same probe while `output_contact_surface=True` still exposed the
larger generated contact surface. This is useful: the solver can use reduced
contacts while the wheel patch estimator reads the richer surface data.

## Gap Behavior

Using `gap=0` behaved as expected for tire sink-in studies:

- At exact touch (`0 mm` sink), no hydroelastic contact surface was generated.
- After small penetration, the contact patch appeared and grew coherently.

Positive gaps generated shallow non-penetrating or near-margin contacts at exact
touch. For example, `gap=1 mm` at `0 mm` sink produced 46 faces with only about
`70 mm^2` of area and positive contact depth. That may be useful for early
collision detection, but it is not ideal if the tire patch should be based on
actual sink depth. For wheel patch estimation, `gap=0` is the better first
configuration to test.

## Resolution Sensitivity

At `3 mm` sink and `gap=0`, SDF resolution changed tessellation density much
more than the aggregate patch dimensions.

| SDF resolution | Faces | Surface area | Longitudinal span | Width span |
| --- | ---: | ---: | ---: | ---: |
| 32 | 281 | `1558 mm^2` | `34.38 mm` | `45.44 mm` |
| 64 | 1233 | `1587 mm^2` | `34.38 mm` | `45.44 mm` |
| 128 | 4897 | `1596 mm^2` | `34.38 mm` | `45.44 mm` |

This suggests a moderate resolution may be enough for patch aggregate estimates,
while higher resolution mostly buys smoother visualization and per-face detail.

## Implications For Wheel Contact

- Hydroelastic/SDF gives direct patch geometry: area, centroid, span, and depth
  can be computed from the contact surface triangles without inventing an area
  from a sparse point cloud.
- It is particularly relevant for non-flat volumetric terrain where rigid
  cylinder contacts may remain sparse or projection-biased.
- The terrain representation matters. Flat planes and heightfields are outside
  the current hydroelastic path, so study fixtures should use volumetric boxes,
  ramps, step boxes, or watertight terrain meshes.
- The current tire-force path should keep rigid-contact support. Hydroelastic
  should be an optional patch source because of CUDA, setup, memory, and
  geometry constraints.
- If reduced hydroelastic solver contacts are used, consider
  `moment_matching=True` only after measuring friction moment preservation for
  wheel patches. For our custom tire model, the richer contact surface may be
  more useful than relying on solver friction contacts.

## Follow-Up Plan

1. Add a small non-example study or test utility that builds hydroelastic
   wheel-over-terrain fixtures without changing interactive examples.
2. Compare rigid and hydroelastic patch estimates on:
   - flat volumetric box,
   - box bump or step,
   - ramp box,
   - watertight non-flat mesh.
3. Compute patch aggregates from the hydro surface:
   centroid, normal, tangent basis, projected area, longitudinal and lateral
   extents, min/mean/max depth, and area-weighted normal load proxy.
4. Decide whether to add an optional hydroelastic patch extractor behind the
   same wheel-contact interface as the rigid patch estimator.
5. Before documenting tuning guidance, verify the hydroelastic stiffness formula
   in code and docs. The current code path uses `(ka * kb) / (ka + kb)` for
   effective stiffness, while some docs may describe a doubled harmonic-mean
   form.
