# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the wheeled-vehicle test suite.

Indirection over :mod:`newton.tests.unittest_utils` re-exporting exactly the
helpers the ``test_vehicles_*`` modules use. On extraction of the vehicles
layer into a standalone ``newton-vehicles`` package, this file is replaced by a
vendored copy of these helpers (they are test-only utilities, not public
newton API), and the vehicle tests keep importing from this module unchanged.
"""

from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_test_devices,
)

__all__ = [
    "USD_AVAILABLE",
    "add_function_test",
    "get_test_devices",
]
