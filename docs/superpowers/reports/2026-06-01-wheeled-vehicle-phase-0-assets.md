# Wheeled Vehicle Phase 0 Asset Report

## rc_car (ackermann)

- Path: `newton/examples/assets/wheeled/rc_car.usda`
- Bodies: 11
- Joints: 11
- Shapes: 5

### Candidate Labels

- Wheel bodies: ['/rc_car/RigidBodies/rc_front_left_wheel', '/rc_car/RigidBodies/rc_front_right_wheel', '/rc_car/RigidBodies/rc_rear_left_wheel', '/rc_car/RigidBodies/rc_rear_right_wheel']
- Wheel shapes: ['/rc_car/RigidBodies/rc_front_left_wheel/Geometry/rc_front_left_wheel', '/rc_car/RigidBodies/rc_rear_left_wheel/Geometry/rc_rear_left_wheel', '/rc_car/RigidBodies/rc_front_right_wheel/Geometry/rc_front_right_wheel', '/rc_car/RigidBodies/rc_rear_right_wheel/Geometry/rc_rear_right_wheel']
- Suspension joints: ['/rc_car/Joints/rc_front_left_suspension', '/rc_car/Joints/rc_front_right_suspension', '/rc_car/Joints/rc_rear_left_suspension', '/rc_car/Joints/rc_rear_right_suspension']
- Steering joints: ['/rc_car/Joints/rc_front_left_steering', '/rc_car/Joints/rc_front_right_steering']

### Manifest Labels

- Wheel bodies: ['/rc_car/RigidBodies/rc_front_left_wheel', '/rc_car/RigidBodies/rc_rear_left_wheel', '/rc_car/RigidBodies/rc_front_right_wheel', '/rc_car/RigidBodies/rc_rear_right_wheel']
- Wheel shapes: ['/rc_car/RigidBodies/rc_front_left_wheel/Geometry/rc_front_left_wheel', '/rc_car/RigidBodies/rc_rear_left_wheel/Geometry/rc_rear_left_wheel', '/rc_car/RigidBodies/rc_front_right_wheel/Geometry/rc_front_right_wheel', '/rc_car/RigidBodies/rc_rear_right_wheel/Geometry/rc_rear_right_wheel']
- Suspension joints: ['/rc_car/Joints/rc_front_left_suspension', '/rc_car/Joints/rc_rear_left_suspension', '/rc_car/Joints/rc_front_right_suspension', '/rc_car/Joints/rc_rear_right_suspension']
- Steering joints: ['/rc_car/Joints/rc_front_left_steering', '/rc_car/Joints/rc_front_right_steering']

## husky (skid_steer)

- Path: `newton/examples/assets/wheeled/husky.usda`
- Bodies: 5
- Joints: 5
- Shapes: 5

### Candidate Labels

- Wheel bodies: ['/husky/RigidBodies/husky_front_left_wheel', '/husky/RigidBodies/husky_front_right_wheel', '/husky/RigidBodies/husky_rear_left_wheel', '/husky/RigidBodies/husky_rear_right_wheel']
- Wheel shapes: ['/husky/RigidBodies/husky_front_left_wheel/Geometry/husky_front_left_wheel', '/husky/RigidBodies/husky_rear_left_wheel/Geometry/husky_rear_left_wheel', '/husky/RigidBodies/husky_front_right_wheel/Geometry/husky_front_right_wheel', '/husky/RigidBodies/husky_rear_right_wheel/Geometry/husky_rear_right_wheel']
- Suspension joints: []
- Steering joints: []

### Manifest Labels

- Wheel bodies: ['/husky/RigidBodies/husky_front_left_wheel', '/husky/RigidBodies/husky_rear_left_wheel', '/husky/RigidBodies/husky_front_right_wheel', '/husky/RigidBodies/husky_rear_right_wheel']
- Wheel shapes: ['/husky/RigidBodies/husky_front_left_wheel/Geometry/husky_front_left_wheel', '/husky/RigidBodies/husky_rear_left_wheel/Geometry/husky_rear_left_wheel', '/husky/RigidBodies/husky_front_right_wheel/Geometry/husky_front_right_wheel', '/husky/RigidBodies/husky_rear_right_wheel/Geometry/husky_rear_right_wheel']
- Suspension joints: []
- Steering joints: []

## Phase 1A Metadata Decisions

- Mark every `wheel_shape_labels` entry with `wheeled:is_wheel = 1`.
- Use `wheel_shape_labels` as the source of explicit `wheeled:wheel_radius` values when shape inference is ambiguous.
- Use `wheel_body_labels` as the receiving bodies for wheel support forces.
- Keep Husky `steering_joint_labels` empty for skid-steer control.
- Treat missing suspension labels as an asset limitation, not a Phase 1A blocker, because the Phase 1B contact wrapper can use shape/body metadata only.
