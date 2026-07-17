# tirecal HDF5 trajectory session format — schema v1

This document is the field-level contract for `.h5` files produced by
`tirecal.format.write_session` and consumed by `tirecal.format.read_session`
(implemented in `tirecal/format.py`). It covers every field, unit, and
convention a reader or writer needs to agree on. The format is deliberately
generic: it is shared by simulator recordings (Newton) and, eventually, logs
captured from a real vehicle.

## Root attributes

| Attribute        | Type   | Meaning                                                          |
|-------------------|--------|-------------------------------------------------------------------|
| `schema_version`  | int    | Format version. Currently `1`.                                    |
| `source`          | string | `"newton-sim"` or `"real"` — where the session was recorded.      |
| `created`          | string | ISO-8601 UTC timestamp of when the file was written (e.g. `2026-07-17T12:00:00+00:00`). |
| `notes`           | string | Free-form human notes. May be empty.                              |

## `/vehicle` group

The `/vehicle` group carries no datasets; its `meta` attribute is a JSON
string describing the vehicle and, for simulated sources, how it was
generated. Readers must treat `meta` as a name-keyed dict and tolerate
unknown keys.

`meta` JSON fields:

- `mass` — vehicle mass [kg].
- `wheelbase` — wheelbase [m].
- `track_width` — track width [m].
- `wheel_radius` — wheel radius [m].
- `tire_model` — name of the generator tire model (e.g. `"brush"`, `"pacejka"`).
- `tire_params` — named dict of tire model parameters (e.g. `{"c_long": 20.0}`). Keys are model-specific; readers must not assume a fixed parameter set.
- For sim sources only: per-world layout counts (e.g. number of worlds, bodies per world, wheels per world) needed to interpret `sim_state` shapes.

## `/laps/NNNN` groups

Each lap is stored under a zero-padded four-digit group name (`0000`, `0001`,
...) so that lexicographic and lap-index ordering agree. Readers iterate
`sorted(f["laps"])`.

### Lap attributes

- `meta` — JSON dict. For sim sources, the following keys are required:
  - `sim_dt` — simulator integration timestep [s].
  - `sim_substeps` — number of substeps per control step.
  - `control_rate_hz` — control-loop rate [Hz].
  - `track_seed` — RNG seed used to generate the track.
  - `track_generator` — name of the track generator used.

  Real-vehicle sources may omit sim-specific keys; readers must not require
  keys beyond what they explicitly consume.
- `control_names` — JSON list of strings naming each column of `controls`, in
  order (e.g. `["drive", "steer", "brake"]`).

### Lap datasets

State datasets have `T+1` rows, samples `0..T`. `controls` has `T` rows.
**Alignment rule:** `controls[t]` is the command applied to drive the
transition from state sample `t` to state sample `t+1`.

| Dataset        | Shape        | Units          | Meaning                                                                 |
|----------------|--------------|----------------|--------------------------------------------------------------------------|
| `time`         | `[T+1]`      | s              | Sample timestamps.                                                       |
| `body_q`       | `[T+1, 7]`   | m (position)   | Chassis position + quaternion `(x, y, z, qx, qy, qz, qw)`, world frame.  |
| `body_qd`      | `[T+1, 6]`   | rad/s, m/s     | Chassis spatial velocity: angular then linear, world frame, Newton spatial convention. |
| `wheel_omega`  | `[T+1, W]`   | rad/s          | Per-wheel angular velocity, `W` wheels.                                  |
| `controls`     | `[T, A]`     | normalized     | Normalized control commands, `A` actions, columns named by `control_names`. |

### `sim_state/` group (optional)

Present only for simulator-recorded sessions; never required by readers,
since real-vehicle logs will not have it. When present, all arrays have
`T+1` rows aligned with the lap's other state datasets.

| Dataset          | Shape         | Units | Meaning                                              |
|------------------|---------------|-------|-------------------------------------------------------|
| `joint_q`        | `[T+1, D]`    | m or rad | Full generalized joint coordinates.                |
| `joint_qd`       | `[T+1, Dv]`   | m/s or rad/s | Full generalized joint velocities.             |
| `body_q_full`    | `[T+1, B, 7]` | m (position) | Position + quaternion for all `B` bodies in the world. |
| `body_qd_full`   | `[T+1, B, 6]` | rad/s, m/s | Spatial velocity for all `B` bodies in the world. |
| `trans_long`     | `[T+1, W]`    | m/s   | Per-wheel longitudinal slip/contact-patch translation velocity. |
| `trans_lat`      | `[T+1, W]`    | m/s   | Per-wheel lateral slip/contact-patch translation velocity. |
| `patch_fz`       | `[T+1, W]`    | N     | Per-wheel vertical (normal) contact-patch force.       |

`sim_state` is a name-keyed dict; a writer may store a superset or subset of
the keys above and readers must handle whichever keys are present rather than
assuming a fixed set.

## Versioning rule

`schema_version` is a simple integer. Readers **reject** files whose
`schema_version` is greater than the reader's `SCHEMA_VERSION`, raising
`ValueError`. Additions of new optional fields, datasets, or `meta`/
`sim_state` keys are backward-compatible and do not require a version bump —
readers must ignore keys and datasets they don't recognize. This is why
`control_names`, `tire_params`, and `sim_state` are all name-keyed dicts
rather than fixed-position arrays: richer channel sets or parameter sets
never break older or simpler readers.
