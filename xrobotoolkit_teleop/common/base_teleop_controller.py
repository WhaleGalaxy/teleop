import abc
import threading
import time
import webbrowser
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.data_logger import DataLogger
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    apply_delta_pose,
    is_valid_quaternion,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)


class BaseTeleopController(abc.ABC):
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base: bool,
        R_headset_world: np.ndarray,
        scale_factor: float,
        axis_map: list[str] | None = None,
        r_vr_to_robot: list[float] | None = None,
        rpy_offset_deg: list[float] | None = None,
        scale_rpy: float = 1.0,
        use_relative: bool = True,
        use_ee_local_rotation: bool = False,
        input_mode: str = "xr",
        manual_delta_xyz: list[float] | None = None,
        manual_delta_rpy_deg: list[float] | None = None,
        q_init: np.ndarray | None = None,
        dt: float = 0.01,
        debug_log_interval: float = 0.5,
        require_alignment: bool = True,
        align_pos_tol: float = 0.05,
        align_rot_tol_deg: float = 15.0,
        align_pos_only: bool = False,
        visualize_robot_body: bool = True,
        enable_log_data: bool = False,
        log_dir: str = "logs",
        log_freq: float = 50,
    ):
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.axis_map = axis_map or ["x", "y", "z"]
        self.r_vr_to_robot = (
            np.array(r_vr_to_robot, dtype=np.float64).reshape(3, 3)
            if r_vr_to_robot is not None
            else np.eye(3, dtype=np.float64)
        )
        self.rpy_offset_deg = rpy_offset_deg or [0.0, 0.0, 0.0]
        self.scale_rpy = float(scale_rpy)
        self.use_relative = bool(use_relative)
        self.use_ee_local_rotation = bool(use_ee_local_rotation)
        self.input_mode = str(input_mode).lower()
        self.manual_delta_xyz = np.array(manual_delta_xyz or [0.0, 0.0, 0.0], dtype=np.float64)
        self.manual_delta_rpy_deg = manual_delta_rpy_deg or [0.0, 0.0, 0.0]
        self.q_init = q_init
        self.dt = dt
        self.debug_log_interval = debug_log_interval
        self.require_alignment = bool(require_alignment)
        self.align_pos_tol = float(align_pos_tol)
        self.align_rot_tol_rad = float(np.deg2rad(align_rot_tol_deg))
        self.align_pos_only = bool(align_pos_only)
        self.visualize_robot_body = bool(visualize_robot_body)
        self.xr_client = XrClient()

        self.enable_log_data = enable_log_data
        self.log_dir = log_dir
        self.log_freq = log_freq
        if enable_log_data:
            self.data_logger = DataLogger(log_dir=log_dir)

        # Initial poses
        self.ref_ee_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_ee_quat = {name: None for name in manipulator_config.keys()}
        self.ref_controller_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_controller_quat = {name: None for name in manipulator_config.keys()}
        self.effector_task = {}
        self.effector_control_mode = {}  # Store control mode for each end effector
        self.active = {}
        self.gripper_pos_target = {}
        self._last_debug_log_time = {}
        self._last_align_log_time = {}
        self.aligned = {name: False for name in manipulator_config.keys()}
        self.align_target_xyz = {name: None for name in manipulator_config.keys()}
        self.align_target_quat = {name: None for name in manipulator_config.keys()}
        self.controller_cursor_xyz = {name: None for name in manipulator_config.keys()}
        self.controller_cursor_quat = {name: None for name in manipulator_config.keys()}
        self._aligned_marker_shown = {name: False for name in manipulator_config.keys()}
        self._placo_viz_enabled = False
        self._last_b_button = False

        # Motion tracker support
        self.motion_tracker_task = {}
        self.ref_tracker_xyz = {}  # Store initial tracker positions
        self.ref_robot_xyz = {}  # Store initial robot end-effector positions
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[name] = {
                    joint_name: joint_pos
                    for joint_name, joint_pos in zip(gripper_config["joint_names"], gripper_config["open_pos"])
                }

        self._stop_event = threading.Event()

        self._robot_setup()
        self._placo_setup()

    def _apply_axis_map(self, vec: np.ndarray) -> np.ndarray:
        axes = {"x": vec[0], "y": vec[1], "z": vec[2]}
        out = []
        for key in self.axis_map:
            sign = 1.0
            axis = key
            if key.startswith("-"):
                sign = -1.0
                axis = key[1:]
            out.append(sign * axes[axis])
        return np.array(out, dtype=np.float64)

    def _rot_matrix_from_rpy_deg(self, rpy_deg: list[float]) -> np.ndarray:
        roll, pitch, yaw = [np.deg2rad(v) for v in rpy_deg]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cp * cy, -cp * sy, sp],
                [sr * sp * cy + cr * sy, -sr * sp * sy + cr * cy, -sr * cp],
                [-cr * sp * cy + sr * sy, cr * sp * sy + sr * cy, cr * cp],
            ],
            dtype=np.float64,
        )

    def _quat_from_rpy_deg(self, rpy_deg: list[float]) -> np.ndarray:
        r_mat = self._rot_matrix_from_rpy_deg(rpy_deg)
        r_transform = np.eye(4, dtype=np.float64)
        r_transform[:3, :3] = r_mat
        return tf.quaternion_from_matrix(r_transform)

    def _apply_rot_matrix_to_quat(self, quat: np.ndarray, rot: np.ndarray) -> np.ndarray:
        r_transform = np.eye(4, dtype=np.float64)
        r_transform[:3, :3] = rot
        r_quat = tf.quaternion_from_matrix(r_transform)
        return tf.quaternion_multiply(
            tf.quaternion_multiply(r_quat, quat),
            tf.quaternion_conjugate(r_quat),
        )

    def _normalize_quat(self, quat: np.ndarray) -> tuple[np.ndarray, bool]:
        if quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), False
        q = np.array(quat, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(q)):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), False
        norm = float(np.linalg.norm(q))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), False
        q = q / norm
        return q, is_valid_quaternion(q)

    def _map_xr_pose_to_world(self, xr_pose) -> tuple[np.ndarray, np.ndarray]:
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]], dtype=np.float64)
        controller_quat = np.array(
            [
                xr_pose[6],
                xr_pose[3],
                xr_pose[4],
                xr_pose[5],
            ],
            dtype=np.float64,
        )
        controller_quat, _ = self._normalize_quat(controller_quat)

        controller_xyz = self._apply_axis_map(controller_xyz)
        if self.R_headset_world is not None:
            controller_xyz = self.R_headset_world @ controller_xyz
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, self.R_headset_world)

        if self.r_vr_to_robot is not None:
            controller_xyz = self.r_vr_to_robot @ controller_xyz
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, self.r_vr_to_robot)

        if self.rpy_offset_deg is not None:
            r_offset = self._rot_matrix_from_rpy_deg(self.rpy_offset_deg)
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, r_offset)

        return controller_xyz, controller_quat

    def _process_xr_pose(self, xr_pose, src_name):
        """Process the current XR controller pose."""
        # Get position and orientation
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]], dtype=np.float64)
        controller_quat = np.array(
            [
                xr_pose[6],  # w
                xr_pose[3],  # x
                xr_pose[4],  # y
                xr_pose[5],  # z
            ],
            dtype=np.float64,
        )
        controller_quat, valid_quat = self._normalize_quat(controller_quat)

        controller_xyz = self._apply_axis_map(controller_xyz)
        if self.R_headset_world is not None:
            controller_xyz = self.R_headset_world @ controller_xyz
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, self.R_headset_world)

        if self.r_vr_to_robot is not None:
            controller_xyz = self.r_vr_to_robot @ controller_xyz
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, self.r_vr_to_robot)

        if self.rpy_offset_deg is not None:
            r_offset = self._rot_matrix_from_rpy_deg(self.rpy_offset_deg)
            controller_quat = self._apply_rot_matrix_to_quat(controller_quat, r_offset)

        self.controller_cursor_xyz[src_name] = controller_xyz.copy()
        self.controller_cursor_quat[src_name] = controller_quat.copy()
        if not valid_quat:
            if self.ref_controller_quat[src_name] is not None:
                controller_quat = self.ref_controller_quat[src_name]
            else:
                controller_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self.use_relative:
            if self.ref_controller_xyz[src_name] is None:
                self.ref_controller_xyz[src_name] = controller_xyz
                self.ref_controller_quat[src_name] = controller_quat

                delta_xyz = np.zeros(3)
                delta_rot = np.array([0.0, 0.0, 0.0])
            else:
                delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
                delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)
        else:
            delta_xyz = controller_xyz * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(np.array([1.0, 0.0, 0.0, 0.0]), controller_quat)

        delta_rot = delta_rot * self.scale_rpy

        return delta_xyz, delta_rot

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver."""
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            print(f"  {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        # self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
            self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion for base

        self.placo_robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.manipulator_config.items():
            # Get control mode (default to "pose" for backward compatibility)
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            if control_mode == "position":
                # Position-only control
                self.effector_task[name] = self.solver.add_position_task(config["link_name"], ee_xyz)
                print(f"Created position task for {name} -> {config['link_name']}")
            else:
                # Full pose control (default)
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
                print(f"Created pose task for {name} -> {config['link_name']}")

            # Alignment target uses current end-effector pose
            self.align_target_xyz[name] = ee_xyz.copy()
            self.align_target_quat[name] = ee_quat.copy()
            
            self.effector_task[name].configure(name, "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure("manipulability", "soft", 1e-2)

            # Set up motion tracker tasks if configured (position only)
            if "motion_tracker" in config:
                tracker_config = config["motion_tracker"]
                link_target = tracker_config["link_target"]

                # Get current position of the target link
                target_xyz, _ = self._get_link_pose(link_target)

                # Create position task for motion tracker target (xyz only)
                tracker_task_name = f"{name}_tracker"
                self.motion_tracker_task[name] = self.solver.add_position_task(link_target, target_xyz)
                self.motion_tracker_task[name].configure(tracker_task_name, "soft", 1.0)

                print(f"Motion tracker position task created for {name} -> {link_target}")

        self.placo_robot.update_kinematics()

    def _update_ik(self):
        """
        This is the core IK logic block. It reads from XR, updates Placo tasks,
        and solves the kinematics.
        """
        self._handle_viz_shortcuts()
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        for src_name, config in self.manipulator_config.items():
            xr_pose = None
            if self.input_mode != "manual":
                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])

            if self.input_mode == "manual":
                self.active[src_name] = True
                self.aligned[src_name] = True
            else:
                if self.require_alignment and not self._check_alignment(src_name, config, xr_pose):
                    self.active[src_name] = False
                else:
                    xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
                    self.active[src_name] = xr_grip_val > 0.9

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                if self.input_mode == "manual":
                    delta_xyz = self.manual_delta_xyz * self.scale_factor
                    target_quat = self._quat_from_rpy_deg(self.manual_delta_rpy_deg)
                    delta_rot = quat_diff_as_angle_axis(np.array([1.0, 0.0, 0.0, 0.0]), target_quat)
                    delta_rot = delta_rot * self.scale_rpy
                else:
                    delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                
                if self.effector_control_mode[src_name] == "position":
                    # Position-only control: only apply position delta
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    self.effector_task[src_name].target_world = target_xyz
                    self._maybe_debug_log(src_name, xr_pose, target_xyz, None)
                else:
                    # Full pose control: apply both position and orientation deltas
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                        local_frame=self.use_ee_local_rotation,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
                    self._maybe_debug_log(src_name, xr_pose, target_xyz, target_quat)
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_ee_quat[src_name] = None
                    self.ref_controller_xyz[src_name] = None
                    self.ref_controller_quat[src_name] = None
                    if self.input_mode == "manual":
                        self.aligned[src_name] = True

        # Process motion tracker data
        self._update_motion_tracker_tasks()

        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _update_motion_tracker_tasks(self):
        """Process motion tracker data and update corresponding Placo tasks."""
        motion_tracker_data = self.xr_client.get_motion_tracker_data()

        for src_name, config in self.manipulator_config.items():
            # Skip if no motion tracker configured for this end effector
            if "motion_tracker" not in config:
                continue

            # Skip if main controller is not active
            if not self.active.get(src_name, False):
                # Reset motion tracker references when controller is inactive
                if src_name in self.ref_tracker_xyz:
                    del self.ref_tracker_xyz[src_name]
                    del self.ref_robot_xyz[src_name]
                continue

            tracker_config = config["motion_tracker"]
            serial = tracker_config["serial"]

            # Skip if this tracker is not available
            if serial not in motion_tracker_data:
                continue

            # Get motion tracker pose
            tracker_pose = motion_tracker_data[serial]["pose"]
            tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])

            # Initialize reference positions on first detection
            if src_name not in self.ref_tracker_xyz:
                self.ref_tracker_xyz[src_name] = tracker_xyz.copy()
                # Get current robot end-effector position as baseline
                robot_xyz, _ = self._get_link_pose(config["motion_tracker"]["link_target"])
                self.ref_robot_xyz[src_name] = robot_xyz.copy()
                continue

            # Calculate movement delta from tracker's initial position
            tracker_delta = tracker_xyz - self.ref_tracker_xyz[src_name]

            # Apply scaled tracker movement to robot's initial position
            final_target_xyz = self.ref_robot_xyz[src_name] + tracker_delta * self.scale_factor

            # Update motion tracker task target position
            if src_name in self.motion_tracker_task:
                self.motion_tracker_task[src_name].target_world = final_target_xyz

    def _init_placo_viz(self):
        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        self._placo_viz_enabled = True
        if not self.visualize_robot_body:
            self._hide_robot_viz()
        for name, config in self.manipulator_config.items():
            if self.visualize_robot_body:
                robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            if self.require_alignment and self.align_target_xyz.get(name) is not None:
                align_frame = np.eye(4)
                align_frame[:3, 3] = self.align_target_xyz[name]
                frame_viz(f"vis_align_{name}", align_frame)

            cursor_xyz = self.controller_cursor_xyz.get(name)
            cursor_quat = self.controller_cursor_quat.get(name)
            if cursor_xyz is not None:
                cursor_frame = tf.quaternion_matrix(cursor_quat) if cursor_quat is not None else np.eye(4)
                cursor_frame[:3, 3] = cursor_xyz
                frame_viz(f"vis_controller_{name}", cursor_frame)

            self._update_alignment_marker(name)
            # Visualize motion tracker target if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                if self.visualize_robot_body:
                    robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def _update_placo_viz(self):
        if not self._placo_viz_enabled:
            return
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            if self.visualize_robot_body:
                robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            if self.require_alignment and self.align_target_xyz.get(name) is not None:
                align_frame = np.eye(4)
                align_frame[:3, 3] = self.align_target_xyz[name]
                frame_viz(f"vis_align_{name}", align_frame)

            cursor_xyz = self.controller_cursor_xyz.get(name)
            cursor_quat = self.controller_cursor_quat.get(name)
            if cursor_xyz is not None:
                cursor_frame = tf.quaternion_matrix(cursor_quat) if cursor_quat is not None else np.eye(4)
                cursor_frame[:3, 3] = cursor_xyz
                frame_viz(f"vis_controller_{name}", cursor_frame)

            self._update_alignment_marker(name)
            # Update motion tracker target visualization if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                if self.visualize_robot_body:
                    robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def sync_end_effector_poses_to_placo_tasks(self):
        """
        Syncs the current end effector link poses to their corresponding placo tasks.
        This is useful for initializing or resetting task targets to current robot state.
        """
        for name, config in self.manipulator_config.items():
            # Get current link pose
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            # Update the corresponding placo task
            if self.effector_control_mode[name] == "position":
                # Position-only control: update target position
                self.effector_task[name].target_world = ee_xyz
            else:
                # Full pose control: update target pose
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name].T_world_frame = ee_target
            
            print(f"Synced {name} end effector pose to placo task: {config['link_name']}")

    def _update_gripper_target(self):
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                for joint_name, open_pos, close_pos in zip(
                    gripper_config["joint_names"],
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                ):
                    # Calculate the target position based on the trigger value
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
            else:
                # TODO: add dexterous hand support
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def _maybe_debug_log(self, src_name: str, xr_pose, target_xyz, target_quat):
        now = time.time()
        last = self._last_debug_log_time.get(src_name, 0.0)
        if now - last < self.debug_log_interval:
            return
        try:
            if target_quat is None:
                print(
                    f"[{src_name}] XR pose: {xr_pose} | target_xyz: "
                    f"({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
                )
            else:
                print(
                    f"[{src_name}] XR pose: {xr_pose} | target_xyz: "
                    f"({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}) "
                    f"target_quat: ({target_quat[0]:.4f}, {target_quat[1]:.4f}, "
                    f"{target_quat[2]:.4f}, {target_quat[3]:.4f})"
                )
        except Exception:
            print(f"[{src_name}] XR pose: {xr_pose} | target_xyz: {target_xyz}")
        self._last_debug_log_time[src_name] = now

    def _update_alignment_marker(self, name: str) -> None:
        if not self.require_alignment:
            return
        if not self.placo_vis:
            return
        if self.aligned.get(name, False):
            marker_frame = np.eye(4)
            marker_xyz = self.align_target_xyz.get(name)
            if marker_xyz is None:
                return
            marker_frame[:3, 3] = marker_xyz
            frame_viz(f"vis_aligned_{name}", marker_frame)
            self._aligned_marker_shown[name] = True
        else:
            if self._aligned_marker_shown.get(name, False):
                try:
                    self.placo_vis.viewer[f"vis_aligned_{name}"].delete()
                except Exception:
                    try:
                        self.placo_vis.viewer[f"vis_aligned_{name}"].set_property("visible", False)
                    except Exception:
                        pass
                self._aligned_marker_shown[name] = False

    def _hide_robot_viz(self) -> None:
        try:
            self.placo_vis.viewer["robot"].delete()
        except Exception:
            try:
                self.placo_vis.viewer["robot"].set_property("visible", False)
            except Exception:
                pass

    def _close_placo_viz(self) -> None:
        if not self._placo_viz_enabled:
            return
        self._placo_viz_enabled = False
        print("Placo visualizer paused.")

    def _open_placo_viz(self) -> None:
        if self._placo_viz_enabled:
            return
        self._placo_viz_enabled = True
        for name in self.manipulator_config.keys():
            self.aligned[name] = False
            self.ref_ee_xyz[name] = None
            self.ref_ee_quat[name] = None
            self.ref_controller_xyz[name] = None
            self.ref_controller_quat[name] = None
            self._aligned_marker_shown[name] = False
        print("Placo visualizer resumed. Alignment reset.")

    def _handle_viz_shortcuts(self) -> None:
        if not self.placo_vis:
            return
        try:
            b_pressed = bool(self.xr_client.get_button_state_by_name("B"))
        except Exception:
            return
        if b_pressed and not self._last_b_button:
            if self._placo_viz_enabled:
                self._close_placo_viz()
            else:
                self._open_placo_viz()
        self._last_b_button = b_pressed

    def _check_alignment(self, src_name: str, config: Dict[str, Any], xr_pose) -> bool:
        if not self.require_alignment:
            return True
        if self.aligned.get(src_name, False):
            return True
        if xr_pose is None:
            return False

        target_xyz = self.align_target_xyz.get(src_name)
        target_quat = self.align_target_quat.get(src_name)
        if target_xyz is None or target_quat is None:
            return False

        controller_xyz, controller_quat = self._map_xr_pose_to_world(xr_pose)
        controller_quat, valid_quat = self._normalize_quat(controller_quat)
        self.controller_cursor_xyz[src_name] = controller_xyz.copy()
        self.controller_cursor_quat[src_name] = controller_quat.copy()
        pos_err = np.linalg.norm(controller_xyz - target_xyz)
        if valid_quat and is_valid_quaternion(target_quat):
            rot_err = np.linalg.norm(quat_diff_as_angle_axis(target_quat, controller_quat))
        else:
            rot_err = 0.0

        if pos_err <= self.align_pos_tol and (self.align_pos_only or rot_err <= self.align_rot_tol_rad):
            self.aligned[src_name] = True
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat
            self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])
            print(
                f"{src_name} aligned ✅ pos_err={pos_err:.3f} m, rot_err={np.rad2deg(rot_err):.1f} deg"
            )
            return True

        now = time.time()
        last = self._last_align_log_time.get(src_name, 0.0)
        if now - last >= max(0.5, self.debug_log_interval):
            if self.align_pos_only:
                print(f"{src_name} waiting for alignment. pos_err={pos_err:.3f} m")
            else:
                print(
                    f"{src_name} waiting for alignment. pos_err={pos_err:.3f} m, rot_err={np.rad2deg(rot_err):.1f} deg"
                )
            self._last_align_log_time[src_name] = now
        return False

    def _log_data(self):
        """
        Logs the current state of the robot, including joint positions, end effector poses,
        and any other relevant data
        """
        if self.enable_log_data:
            raise NotImplementedError

    # ---------------------------------------------------------
    # --- Abstract Methods (to be implemented by subclasses) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """Initializes the specific backend (connects to robot, starts sim, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """Reads the current joint states from the robot/sim and updates self.placo_robot.state.q."""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """Sends the calculated target joint positions from self.placo_robot.state.q to the robot/sim."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_link_pose(self, link_name):
        """Gets the current world pose for a given link name."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        The main entry point. Subclasses must implement this to define their
        execution model (single-threaded or multi-threaded).
        """
        raise NotImplementedError
