.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.wheeled
==============

Wheeled-vehicle metadata, contact patch, drive, and tire helpers.

.. py:module:: newton.wheeled
.. currentmodule:: newton.wheeled

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   WheelContactPatchState
   WheelDriveControl
   WheelDriveState
   WheelMetadata
   WheelMomentControl
   WheelMomentState
   WheelTireControl
   WheelTireState
   WheeledAssetMetadata
   WheeledModelMetadata
   WheeledMotorConfig
   WheeledSteeringConfig
   WheeledVehicleControl
   WheeledVehicleLayout
   WheeledVehicleState

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   apply_wheel_drive_forces
   apply_wheel_tire_forces
   apply_wheeled_manifest
   apply_wheeled_manifest_metadata
   build_wheeled_metadata
   build_wheeled_vehicle_layout
   configure_mujoco_wheel_contacts
   configure_wheel_axle_joints
   configure_wheel_moment_control
   configure_wheel_tire_control
   configure_wheeled_vehicle_control
   load_wheeled_manifest
   read_wheeled_metadata
   register_wheeled_custom_attributes
   update_wheel_contact_patches
   update_wheel_drive_normal_loads
   update_wheel_moments
   update_wheel_tire_normal_loads
   update_wheeled_vehicle_controls
