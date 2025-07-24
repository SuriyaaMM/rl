import pybullet as p
import pybullet_data
import time
import numpy as np
import random

def setup_pybullet_environment():
    """Initializes the PyBullet physics engine and sets up the basic environment."""
    # Connect to the physics server (GUI mode for visualization)
    physics_client = p.connect(p.GUI, options="--render_device=egl")
    # Add PyBullet's default data path for loading models (e.g., plane)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Set gravity
    p.setGravity(0, 0, -9.81)
    # Set simulation time step
    p.setTimeStep(1./240.) # 240 steps per second

    # Create a reddish-brown ground plane (Mars surface)
    # createCollisionShape for physics, createMultiBody for combining shape and visual
    plane_collision_shape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_plane_id = p.createMultiBody(
        baseCollisionShapeIndex=plane_collision_shape,
        baseVisualShapeIndex=-1, # No separate visual shape, uses collision shape's default
        basePosition=[0, 0, 0]
    )
    # Set a reddish-brown color for the ground plane
    p.changeVisualShape(ground_plane_id, -1, rgbaColor=[0.6, 0.3, 0.1, 1])

    print("PyBullet environment set up.")
    return physics_client

def create_simple_rover():
    """Creates a basic wheeled rover model from primitive shapes."""
    # Define rover dimensions
    chassis_half_extents = [0.25, 0.2, 0.08] # x, y, z (half dimensions)
    wheel_radius = 0.08
    wheel_width = 0.05
    wheel_offset_x = chassis_half_extents[0] # Aligned with chassis edge
    wheel_offset_y = chassis_half_extents[1] + wheel_width/2 # Slightly outside chassis
    wheel_offset_z = -chassis_half_extents[2] # Bottom of chassis

    # Chassis visual and collision shape
    chassis_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=chassis_half_extents,
        rgbaColor=[0.7, 0.3, 0.1, 1] # Reddish-brown for rover body
    )
    chassis_collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=chassis_half_extents
    )

    # Wheel visual and collision shape
    # Wheels are cylinders, rotated to stand upright
    wheel_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=wheel_radius,
        length=wheel_width,
        rgbaColor=[0.2, 0.2, 0.2, 1] # Dark grey for wheels
    )
    wheel_collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=wheel_radius,
        length=wheel_width
    )

    base_mass = 5 # kg
    base_position = [0, 0, 0.2] # Start slightly above ground to avoid initial collision
    base_orientation = p.getQuaternionFromEuler([0, 0, 0]) # No initial rotation

    # Define link parameters for the four wheels
    # Each wheel is a link connected to the base (chassis)
    # linkMass, linkCollisionShapeIndex, linkVisualShapeIndex, linkPosition (relative to parent),
    # linkOrientation (relative to parent), linkInertialFramePosition, linkInertialFrameOrientation,
    # parentIndex, jointType, jointAxis (relative to parent), parentFramePosition, parentFrameOrientation
    link_masses = [0.5] * 4 # Mass for each wheel
    link_collision_shape_indices = [wheel_collision_shape_id] * 4
    link_visual_shape_indices = [wheel_visual_shape_id] * 4

    # Positions of wheels relative to the chassis's center
    # Front Right, Front Left, Rear Right, Rear Left
    link_positions = [
        [wheel_offset_x, wheel_offset_y, wheel_offset_z],  # Front Right
        [-wheel_offset_x, wheel_offset_y, wheel_offset_z], # Front Left
        [wheel_offset_x, -wheel_offset_y, wheel_offset_z], # Rear Right
        [-wheel_offset_x, -wheel_offset_y, wheel_offset_z]  # Rear Left
    ]
    # Orientations of wheels relative to the chassis (rotate cylinders to be wheels)
    link_orientations = [p.getQuaternionFromEuler([np.pi/2, 0, 0])] * 4 # Rotate 90 deg around X-axis
    link_inertial_frame_positions = [[0, 0, 0]] * 4
    link_inertial_frame_orientations = [[0, 0, 0, 1]] * 4
    parent_indices = [0, 0, 0, 0] # All wheels connected to the base (link index 0)
    joint_types = [p.JOINT_REVOLUTE] * 4 # Wheels rotate
    joint_axes = [[0, 0, 1]] * 4 # Axis of rotation for wheels (Z-axis in wheel's local frame)

    # Create the multi-body rover
    rover_id = p.createMultiBody(
        baseMass=base_mass,
        baseCollisionShapeIndex=chassis_collision_shape_id,
        baseVisualShapeIndex=chassis_visual_shape_id,
        basePosition=base_position,
        baseOrientation=base_orientation,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shape_indices,
        linkVisualShapeIndices=link_visual_shape_indices,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_frame_positions,
        linkInertialFrameOrientations=link_inertial_frame_orientations,
        linkParentIndices=parent_indices,
        linkJointTypes=joint_types,
        linkJointAxes=joint_axes
    )

    # Disable motors and set friction for wheels
    # Joint indices correspond to the order they were added (0, 1, 2, 3 for the 4 wheels)
    for i in range(p.getNumJoints(rover_id)):
        p.setJointMotorControl2(rover_id, i, p.VELOCITY_CONTROL, force=0) # Disable motors initially
        # Increase friction to prevent excessive slipping on the 'Mars' surface
        p.changeDynamics(rover_id, i, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

    print("Rover created.")
    return rover_id

def create_obstacles(num_obstacles=10):
    """Creates random rock-like obstacles on the terrain."""
    obstacle_ids = []
    for i in range(num_obstacles):
        # Randomly choose between sphere or box
        shape_type = random.choice([p.GEOM_SPHERE, p.GEOM_BOX])
        
        # Random dimensions
        if shape_type == p.GEOM_SPHERE:
            radius = random.uniform(0.1, 0.4)
            collision_shape = p.createCollisionShape(shapeType=shape_type, radius=radius)
            visual_shape = p.createVisualShape(shapeType=shape_type, radius=radius, rgbaColor=[0.4, 0.2, 0.1, 1])
            height = radius # For positioning
        else: # p.GEOM_BOX
            half_extents = [random.uniform(0.1, 0.3) for _ in range(3)]
            collision_shape = p.createCollisionShape(shapeType=shape_type, halfExtents=half_extents)
            visual_shape = p.createVisualShape(shapeType=shape_type, halfExtents=half_extents, rgbaColor=[0.4, 0.2, 0.1, 1])
            height = half_extents[2] # For positioning

        # Random position within a certain range
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        z = height # Place on top of the ground

        # Random orientation
        orientation = p.getQuaternionFromEuler([random.uniform(0, np.pi), random.uniform(0, np.pi), random.uniform(0, np.pi)])

        obstacle_id = p.createMultiBody(
            baseMass=0, # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z],
            baseOrientation=orientation
        )
        obstacle_ids.append(obstacle_id)
    print(f"{num_obstacles} obstacles created.")
    return obstacle_ids

def main():
    """Main function to run the Mars rover simulation."""
    physics_client = setup_pybullet_environment()

    # Create the rover
    rover_id = create_simple_rover()

    # Create obstacles
    create_obstacles(num_obstacles=20) # Create 20 random obstacles

    # Set up camera to view the scene
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )

    # Add debug parameters for manual control of wheel velocities
    # These will appear as sliders in the PyBullet GUI
    max_velocity = 10 # rad/s
    # Front Right, Front Left, Rear Right, Rear Left
    fr_wheel_vel = p.addUserDebugParameter("FR Wheel Vel", -max_velocity, max_velocity, 0)
    fl_wheel_vel = p.addUserDebugParameter("FL Wheel Vel", -max_velocity, max_velocity, 0)
    rr_wheel_vel = p.addUserDebugParameter("RR Wheel Vel", -max_velocity, max_velocity, 0)
    rl_wheel_vel = p.addUserDebugParameter("RL Wheel Vel", -max_velocity, max_velocity, 0)

    # Simulation loop
    try:
        while True:
            # Get current values from debug sliders
            target_fr_vel = p.readUserDebugParameter(fr_wheel_vel)
            target_fl_vel = p.readUserDebugParameter(fl_wheel_vel)
            target_rr_vel = p.readUserDebugParameter(rr_wheel_vel)
            target_rl_vel = p.readUserDebugParameter(rl_wheel_vel)

            # Apply velocities to the wheels
            # Joint indices are 0, 1, 2, 3 for FR, FL, RR, RL wheels respectively
            p.setJointMotorControl2(rover_id, 0, p.VELOCITY_CONTROL, targetVelocity=target_fr_vel, force=50)
            p.setJointMotorControl2(rover_id, 1, p.VELOCITY_CONTROL, targetVelocity=target_fl_vel, force=50)
            p.setJointMotorControl2(rover_id, 2, p.VELOCITY_CONTROL, targetVelocity=target_rr_vel, force=50)
            p.setJointMotorControl2(rover_id, 3, p.VELOCITY_CONTROL, targetVelocity=target_rl_vel, force=50)

            # Step the simulation
            p.stepSimulation()
            time.sleep(1./240.) # Visual delay to make it observable

    except p.error as e:
        print(f"PyBullet error: {e}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Disconnect from the physics server
        p.disconnect()
        print("Simulation ended.")

if __name__ == "__main__":
    main()
