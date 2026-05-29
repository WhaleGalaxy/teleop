## Mujoco simulation
- Robot definition files: both `.xml` and `.urdf` files are required and they should be consistent with each other (same link names and joint names). The `.xml` file is for mujoco simulation, and the `.urdf` is for placo. Optionally, there should be 1 additional free floating body per end effector defined in the `.xml` file for visualization of commanded teleop targets in mujoco.
- Teleoperation task is defined by a config dict
  - link_name: the name of end effector link as defined in mujoco .xml & .urdf files
  - pose_source: name of the source of pose to be used by XrClient (e.g., `left_controller`, `right_controller`)
  - control_trigger: the key to define whether an arm control is active
  - control_mode: optional field to specify tracking mode - "pose" (default, full 6DOF) or "position" (3DOF position only)
  - vis_target: name of the body for teleop target visualization
  - motion_tracker: optional config for additional motion tracker to control another link in the manipulator (not recommended for 6DOF arms like UR5)
    - serial: serial number of the motion tracker device
    - link_target: name of the robot link to be controlled by the motion tracker
    ```python
    config = {
        "right_hand": {
            "link_name": "flange",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "control_mode": "position", # optional: "pose" (default) or "position"
            "motion_tracker": {
                "serial": "PC2310BLH9020740B",
                "link_target": "link4",
            },
            "vis_target": "right_target", # optional, only used in mujoco
        },
        "left_hand": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_trigger": "left_trigger",
            "vis_target": "left_target", # optional, only used in mujoco
        },
    }
    ```

- Run mujoco demo for dual UR5e with the following script
    ```bash
    python scripts/simulation/teleop_dual_ur5e_mujoco.py
    ```

- Controlling parallel gripper in mujoco simulation
  - Users can add an optional gripper configuration in the end effector config dict
    - joint_name: the actuated mujoco joint within the gripper
    - gripper_trigger: name of the key mapped to this gripper from the controller
    - open_pos: the value of the actuated joint when fully opened
    - close_pos: the value of the actuated joint when fully closed
    ```python
    config = {
        "right_hand": {
            # other configs,
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["right_gripper_finger_joint1",],
                "open_pos": [0.05,],
                "close_pos": [0.0,],
            },
        },
        "left_hand": {
            # other configs,
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["left_gripper_finger_joint1",],
                "open_pos": [0.05,],
                "close_pos": [0.0,],
            },
        },
    }
    ```
  - Note that the parallel gripper might contain multiple joints in the `.xml` file, but only 1 of the joints should be actuated, the others should be controlled by additional equality constraints in the xml. The `.urdf` file supplied to Placo does not have to contain the gripper dof.
    ```xml
    <equality>
    <joint name="right_gripper_constraint" joint1="right_gripper_finger_joint1" joint2="right_gripper_finger_joint2" polycoef="0 -1 0 0 0" />
    <joint name="left_gripper_constraint" joint1="left_gripper_finger_joint1" joint2="left_gripper_finger_joint2" polycoef="0 -1 0 0 0" />
    </equality>
    ```
- Example of mujoco teleoperation with gripper control using dual A1X arm
    ```bash
    python scripts/simulation/teleop_dual_a1x_mujoco.py
    ```

- Marvin M6 dual-arm teleop (URDF-based)
    - The script builds a combined dual-arm URDF at runtime, fixes mesh paths, and uses it for both MuJoCo and Placo.
    - End-effector link names are `Link7_L` and `Link7_R` (from the provided URDFs).
    - Use `--left-base-xyz` and `--right-base-xyz` to separate the bases in the scene.
    - Script: `teleop_dual_marvin_m6_mujoco.py`