# Extracting `newton.vehicles` into a standalone `newton-vehicles` package

This document is the move recipe for pulling the wheeled-vehicle layer out of
the newton tree into its own repository (`newton-vehicles`, import name
`newton_vehicles`) with newton as a plain dependency. The layer is already
extraction-ready: it imports only public `newton` API, except for two
forward-compat shims isolated in `_compat.py` (see below).

## 1. File inventory

Proposed standalone layout is flat — the package is small enough that an
`_src/` split adds no value; keep implementation modules private with a
leading-underscore-free name but document that only `newton_vehicles/__init__.py`
re-exports are public.

| Current path | Target path |
| --- | --- |
| `newton/vehicles.py` (public facade) | `newton_vehicles/__init__.py` (keep the same `__all__`; change `from ._src.vehicles import ...` to relative imports of the sibling modules) |
| `newton/_src/vehicles/__init__.py` | folded into `newton_vehicles/__init__.py` |
| `newton/_src/vehicles/_compat.py` | `newton_vehicles/_compat.py` |
| `newton/_src/vehicles/contact.py` | `newton_vehicles/contact.py` |
| `newton/_src/vehicles/controller.py` | `newton_vehicles/controller.py` |
| `newton/_src/vehicles/impulse.py` | `newton_vehicles/impulse.py` |
| `newton/_src/vehicles/joints.py` | `newton_vehicles/joints.py` |
| `newton/_src/vehicles/metadata.py` | `newton_vehicles/metadata.py` |
| `newton/_src/vehicles/mppi.py` | `newton_vehicles/mppi.py` |
| `newton/_src/vehicles/tire.py` | `newton_vehicles/tire.py` |
| `newton/_src/vehicles/vehicle.py` | `newton_vehicles/vehicle.py` |
| `newton/_src/vehicles/wheel.py` | `newton_vehicles/wheel.py` |
| `newton/tests/test_vehicles_*.py` (10 files) | `tests/test_vehicles_*.py` |
| `newton/tests/vehicles_test_utils.py` | `tests/vehicles_test_utils.py` — replace the re-export with a **vendored copy** of `add_function_test`, `get_test_devices`, and `USD_AVAILABLE` from `newton/tests/unittest_utils.py` (test-only helpers, not public newton API; test imports stay unchanged) |
| `newton/examples/vehicles/*.py` | `examples/*.py` |
| `newton/examples/assets/wheeled/**` (`manifest.json`, `husky.usda`, `rc_car.usda`) | `examples/assets/wheeled/**` |
| `newton/examples/assets/cone.usda` | `examples/assets/cone.usda` (used by the MPPI track example) |
| `newton/tests/test_examples.py` `TestVehicleExamples` block | port into a small `tests/test_examples.py` in the new repo (or drop and rely on CI running the examples headless) |

Also remove from newton itself: the `vehicles` entries in
`newton/__init__.py` (submodule import + `__all__`), the vehicle rows in
`newton/examples/README.md`, the `TestVehicleExamples` block in
`newton/tests/test_examples.py`, and the API docs entry (re-run
`docs/generate_api.py`). Deprecate `newton.vehicles` with a stub that raises
or forwards to `newton_vehicles` for one release rather than deleting it
outright.

## 2. pyproject skeleton

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "newton-vehicles"
version = "0.1.0"
description = "Wheeled-vehicle simulation layer for the Newton physics engine"
requires-python = ">=3.10"
dependencies = [
    # Bump the floor to the first newton release that ships
    # ModelBuilder.get_custom_frequency_count and ModelBuilder.set_joint_type
    # (newton-physics/newton#3361 / #3362); until then any newton release the
    # _compat.py fallbacks support works. Distribution name per newton's own
    # pyproject.toml ("newton").
    "newton",
]

[project.optional-dependencies]
examples = ["usd-core"]  # examples/tests load .usda assets

[tool.setuptools.packages.find]
include = ["newton_vehicles*"]
```

`warp-lang` and `numpy` are used directly by the layer but arrive
transitively as hard dependencies of newton; listing them explicitly is
optional (do it if you want to survive newton dependency refactors).

## 3. What the examples still use from newton

The examples import only public modules: `newton`, `newton.vehicles` (becomes
`newton_vehicles`), and `newton.examples`. From `newton.examples` they use:

- `newton.examples.create_parser()` / `add_world_count_arg()` — argparse setup
- `newton.examples.init(parser)` — returns `(viewer, args)` (viewer selection,
  device setup)
- `newton.examples.run(example, args)` — the standard example loop, which also
  drives `test_final()` / `test_post_step()` hooks
- `newton.examples.get_asset(name)` — resolves files under
  `newton/examples/assets/`

`newton.examples` is a public helper module, so the extracted examples may
keep using everything except `get_asset`, which resolves paths inside the
newton install. Required adaptation: replace `newton.examples.get_asset(...)`
with a tiny local helper resolving against the new repo's `examples/assets/`
directory (3 call sites: manifest + robot USD in each example, cone.usda in
the MPPI track example). If newton's example-browser registration is not
wanted, drop the `newton/examples/README.md` registration step; otherwise the
new repo's README documents `python -m` invocations directly.

## 4. The two compat shims (`_compat.py`)

`_compat.py` is the only module that touches anything non-public in newton:

1. `get_custom_frequency_count(builder, frequency)` — falls back to reading
   `builder._custom_frequency_counts`. Upstream ask:
   newton-physics/newton#3361 (fork PR 3).
2. `set_joint_type_fixed(builder, joint)` — falls back to vendored builder
   array surgery converting a revolute joint to `JointType.FIXED`. Upstream
   ask: newton-physics/newton#3362 (fork PR 4).

Each shim tests `hasattr(builder, ...)` and prefers the public method, so the
package upgrades automatically the moment a newton release ships the APIs.
Once the dependency floor in `pyproject.toml` is raised past that release,
delete the fallback bodies (or the whole module, inlining the two public
calls) — after that the package uses zero newton private API.

## 5. What intentionally stays in newton

These are engine features the vehicles layer *uses* but that belong upstream;
do **not** move them:

- Narrow-phase wheel contact footprint options (fork PRs 1–2, upstream issues
  newton-physics/newton#3359 / #3360): rounded-cylinder wheel footprint
  support in the collision pipeline and hydroelastic-plane contact support.
  These are core collision-engine capabilities; `configure_wheel_solver_contacts`
  merely selects them through public per-shape settings.
- The custom-attribute system (`Model.AttributeFrequency`, builder custom
  frequencies) that the metadata layer builds on.
- Solver integration points (MuJoCo Warp solver, body-force application) —
  the layer only consumes public `Model`/`State`/solver API.
