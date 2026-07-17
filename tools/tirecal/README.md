# tirecal

Trajectory-based tire-model calibration tooling for Newton vehicles.
`tirecal` records driving trajectories (from Newton simulation or, later,
real-vehicle logs) to a standalone HDF5 format, then fits tire-model
parameters (e.g. brush, Pacejka) against those recordings. The trajectory
format is intentionally independent of `newton` — see `tirecal/format.py` —
so it can also read logs that never touched a simulator.

## Install

```bash
uv sync --extra tirecal --extra sim
```

`tirecal` itself only needs `numpy` and `h5py` (the `tirecal` extra); the
`sim` extra is required for the Newton-facing generation and calibration
modules added by later tasks.

### External requirement: `track_gen`

The recorder (`tirecal.generate`) — and the later closed-loop validation —
drive the MPPI track example, which imports the external `track_gen` package.
It is **not** part of the `tirecal` or `sim` extras: a 2D-centerline version
exposing the `CollisionChecker(method=...)` / `PropSampler(mode=...)` API must
be on `PYTHONPATH`. Newer `track_gen` builds with a `vec3f` (3D gate-course)
data model are incompatible with the example's `vec2f` track buffers. When
`track_gen` is absent or incompatible, `test_generate` skips rather than
fails.

## Usage

`tirecal` is a standalone package under `tools/tirecal`, not part of the
`newton` distribution, so point `PYTHONPATH` at its directory when invoking
it as a module:

```bash
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.generate ...
```

## Tests

```bash
uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v
```

## Further reading

- [`SCHEMA.md`](SCHEMA.md) — the HDF5 trajectory session format: root/lap
  attributes, dataset shapes and units, and the versioning rule.
- `docs/superpowers/plans/2026-07-17-pacejka-tire-ga-calibration.md` — the
  full calibration pipeline spec this package implements.
