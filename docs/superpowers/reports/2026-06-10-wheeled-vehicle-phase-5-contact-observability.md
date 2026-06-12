# Wheeled Vehicle Phase 5 Contact Observability Report

## Scope

This report records the Phase 5 observability comparison between Newton-generated
contacts converted into MuJoCo Warp and MuJoCo-native normal contacts. It is a
stability and ownership check only. The wheeled tire path still builds
`WheelContactPatchState` from Newton `Contacts`; MuJoCo-native contacts are not
used as the patch source.

## Scenario

The comparison uses the simplified RC car asset on a shallow ramp-box terrain.
The car and ramp are both pitched by `-0.05 rad`, wheel and terrain
`shape_gap = 0.0`, and the command mapper drives all wheels with a normalized
`0.2` drive command and a `0.1` steering command for 12 steps at `1/240 s`.
Wheel contacts are configured normal-only in MuJoCo so the solver owns normal
support while the wheeled tire path owns longitudinal and lateral tire forces.

## Measurements

- `use_mujoco_contacts=False`: finite state; 4 / 4 active wheel patches;
  patch counts `[4, 4, 4, 4]`; mean normal `[-0.050, 0.000, 0.999]`;
  friction seed `0.77`; max wheel speed `2.80 rad/s`; max tire force `1.64 N`;
  minimum body height `0.046 m`.
- `use_mujoco_contacts=True`: finite state; 4 / 4 active wheel patches;
  patch counts `[4, 4, 4, 4]`; mean normal `[-0.050, 0.000, 0.999]`;
  friction seed `0.77`; max wheel speed `2.80 rad/s`; max tire force `1.80 N`;
  minimum body height `0.045 m`.

Patch counts, normals, and material seeds are from Newton `Model.collide()` in
both rows. The difference is which contact source the MuJoCo Warp solver uses
for normal support during stepping.

## Decision

No architecture change is justified by this comparison. Newton-generated
contacts converted into MuJoCo Warp remain the main path for wheeled examples
and tests because they expose the contact buffers consumed by
`WheelContactPatchState`. MuJoCo-native contacts are viable as a short stability
comparison on the primitive ramp scene, but they do not replace the Newton patch
source and do not provide a reason to change the tire-force ownership split.

## Limits

This is intentionally narrow. It does not prove MuJoCo-native contacts are a
better patch source, and it does not cover every mesh or jump terrain case. If a
future solver issue appears only with `use_mujoco_contacts=False`, add a focused
A/B test for that terrain and vehicle before changing the default path.
