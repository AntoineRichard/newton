.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.vehicles
===============

Wheeled-vehicle simulation layer.

A cohesive :class:`WheeledVehicles` controller wraps a rigid solver (MuJoCo Warp
first): the solver owns collision and normal support while this layer owns the
analytical wheel spin and a brush combined-slip tire model, supporting
heterogeneous vehicles (Ackermann, skid-steer, generic) in a single model.

.. py:module:: newton.vehicles
.. currentmodule:: newton.vehicles

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   DriveInput
   DriveMode
   TireModel
   VehicleAssetMetadata
   VehicleModelData
   WheeledConfig
   WheeledVehicles

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   add_wheel
   apply_vehicle_manifest
   configure_wheel_axle_joints
   configure_wheel_solver_contacts
   load_vehicle_manifest
   read_vehicle_model_data
   register_vehicle_attributes
   set_vehicle

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``VEHICLE_NAMESPACE``
     - ``vehicle``
