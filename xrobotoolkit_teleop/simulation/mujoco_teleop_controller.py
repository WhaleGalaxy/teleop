from typing import Any, Dict
import os
import tempfile
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    import os
    import sys

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import mujoco
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.mujoco_utils import (
    calc_mujoco_ctrl_from_qpos,
    calc_mujoco_qpos_from_placo_q,
    calc_placo_q_from_mujoco_qpos,
    set_mujoco_joint_pos_by_name,
)


class MujocoTeleopController(BaseTeleopController):
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
        axis_map=None,
        r_vr_to_robot=None,
        rpy_offset_deg=None,
        scale_rpy=1.0,
        use_relative=True,
        use_ee_local_rotation: bool = False,
        input_mode: str = "xr",
        manual_delta_xyz: list[float] | None = None,
        manual_delta_rpy_deg: list[float] | None = None,
        dt=0.01,
        debug_log_interval: float = 0.5,
        require_alignment: bool = True,
        align_pos_tol: float = 0.05,
        align_rot_tol_deg: float = 15.0,
        align_pos_only: bool = False,
        visualize_robot_body: bool = True,
        disable_gravity: bool = False,
        disable_contact: bool = False,
        joint_damping: float = 0.0,
        joint_armature: float = 0.0,
        auto_actuators: bool = True,
        actuator_kp: float = 50.0,
        actuator_kv: float = 5.0,
        control_source: str = "teleop",
        direct_qpos: bool = False,
        gui_direct_qpos: bool = False,
        ctrlrange_from_limits: bool = False,
        mj_qpos_init=None,
        q_init=None,
    ):
        self.visualize_placo = visualize_placo
        self.xml_path = xml_path
        self.mj_qpos_init = mj_qpos_init
        self.disable_gravity = disable_gravity
        self.disable_contact = disable_contact
        self.joint_damping = joint_damping
        self.joint_armature = joint_armature
        self.auto_actuators = auto_actuators
        self.actuator_kp = actuator_kp
        self.actuator_kv = actuator_kv
        self.control_source = str(control_source).lower()
        self.direct_qpos = bool(direct_qpos)
        self.gui_direct_qpos = bool(gui_direct_qpos)
        self.ctrlrange_from_limits = bool(ctrlrange_from_limits) or self.control_source in ("gui", "teleop")
        self._joint_to_ctrl_index: dict[str, int] = {}
        self._debug_joint_ids: dict[str, int] = {}
        self._last_gui_log_time = 0.0

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.target_mocap_idx = {name: -1 for name in manipulator_config.keys()}

        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            axis_map=axis_map,
            r_vr_to_robot=r_vr_to_robot,
            rpy_offset_deg=rpy_offset_deg,
            scale_rpy=scale_rpy,
            use_relative=use_relative,
            use_ee_local_rotation=use_ee_local_rotation,
            input_mode=input_mode,
            manual_delta_xyz=manual_delta_xyz,
            manual_delta_rpy_deg=manual_delta_rpy_deg,
            q_init=q_init,
            dt=dt,
            debug_log_interval=debug_log_interval,
            require_alignment=require_alignment,
            align_pos_tol=align_pos_tol,
            align_rot_tol_deg=align_rot_tol_deg,
            align_pos_only=align_pos_only,
            visualize_robot_body=visualize_robot_body,
        )

        if visualize_placo:
            self._init_placo_viz()

    def _robot_setup(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        if self.auto_actuators and self.mj_model.nu == 0:
            actuated_xml = self._build_actuated_xml(self.mj_model)
            if actuated_xml:
                self.mj_model = mujoco.MjModel.from_xml_path(actuated_xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        print("Joint names in the Mujoco model:")
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {joint_name}")
        print(f"MuJoCo actuators (nu): {self.mj_model.nu}")
        if self.mj_model.nu > 0:
            print("Actuator mapping (ctrl index -> joint):")
            for i in range(self.mj_model.nu):
                trn_type = self.mj_model.actuator_trntype[i]
                if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
                    print(f"  ctrl{i}: (non-joint actuator)")
                    continue
                joint_id = int(self.mj_model.actuator_trnid[i][0])
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name:
                    self._joint_to_ctrl_index[joint_name] = i
                print(f"  ctrl{i}: {joint_name}")
            if self.ctrlrange_from_limits:
                # Ensure actuator ctrlrange covers joint limits (avoid clipping at [-1, 1])
                for i in range(self.mj_model.nu):
                    trn_type = self.mj_model.actuator_trntype[i]
                    if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
                        continue
                    joint_id = int(self.mj_model.actuator_trnid[i][0])
                    jmin, jmax = self.mj_model.jnt_range[joint_id]
                    if jmin != 0.0 or jmax != 0.0:
                        self.mj_model.actuator_ctrllimited[i] = 1
                        self.mj_model.actuator_ctrlrange[i][0] = jmin
                        self.mj_model.actuator_ctrlrange[i][1] = jmax
            print("Actuator parameters (ctrl index -> joint, trn, ctrlrange, gainprm, biasprm, gear):")
            for i in range(self.mj_model.nu):
                trn_type = self.mj_model.actuator_trntype[i]
                if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
                    continue
                joint_id = int(self.mj_model.actuator_trnid[i][0])
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                ctrlrange = self.mj_model.actuator_ctrlrange[i]
                ctrllimited = int(self.mj_model.actuator_ctrllimited[i])
                gainprm = self.mj_model.actuator_gainprm[i]
                biasprm = self.mj_model.actuator_biasprm[i]
                gear = self.mj_model.actuator_gear[i]
                jmin, jmax = self.mj_model.jnt_range[joint_id]
                print(
                    "  ctrl{idx}: {jname} trn={trn} ctrllimited={limited} ctrlrange=[{cmin:.4f}, {cmax:.4f}] "
                    "gainprm[0:3]=[{g0:.4f}, {g1:.4f}, {g2:.4f}] biasprm[0:3]=[{b0:.4f}, {b1:.4f}, {b2:.4f}] "
                    "gear[0:3]=[{r0:.4f}, {r1:.4f}, {r2:.4f}] jrange=[{jmin:.4f}, {jmax:.4f}]"
                    .format(
                        idx=i,
                        jname=joint_name,
                        trn=int(trn_type),
                        limited=ctrllimited,
                        cmin=float(ctrlrange[0]),
                        cmax=float(ctrlrange[1]),
                        g0=float(gainprm[0]),
                        g1=float(gainprm[1]),
                        g2=float(gainprm[2]),
                        b0=float(biasprm[0]),
                        b1=float(biasprm[1]),
                        b2=float(biasprm[2]),
                        r0=float(gear[0]),
                        r1=float(gear[1]),
                        r2=float(gear[2]),
                        jmin=float(jmin),
                        jmax=float(jmax),
                    )
                )
        if self.mj_model.nu == 0:
            print("Warning: MuJoCo model has no actuators. Falling back to direct qpos update.")

        for joint_name in ("Joint1_L", "Joint1_R"):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                self._debug_joint_ids[joint_name] = int(joint_id)

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        if self.disable_gravity:
            self.mj_model.opt.gravity[:] = 0.0

        if self.disable_contact:
            self.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        if self.joint_damping > 0.0:
            self.mj_model.dof_damping[:] = self.joint_damping

        if self.joint_armature > 0.0:
            self.mj_model.dof_armature[:] = self.joint_armature

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_qpos_init is None:
            if self.mj_model.nkey > 0:
                try:
                    key_id = self.mj_model.key("home").id
                except KeyError:
                    key_id = 0
                mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, key_id)
        else:
            self.mj_data.qpos[:] = self.mj_qpos_init
            self.mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(self.mj_model, self.mj_qpos_init)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # setup mocap target
        for name, config in self.manipulator_config.items():
            if "vis_target" not in config:
                print(f"Warning: 'vis_target' not found in config for {name}. Skipping mocap setup.")
                continue
            vis_target = config["vis_target"]
            mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis_target)
            if mocap_id == -1:
                raise ValueError(f"Mocap body '{vis_target}' not found in the model.")

            if self.mj_model.body_mocapid[mocap_id] == -1:
                raise ValueError(f"Body '{self.vis_target}' is not configured for mocap.")
            else:
                self.target_mocap_idx[name] = self.mj_model.body_mocapid[mocap_id]

            print(f"Mocap ID for '{vis_target}' body: {self.target_mocap_idx[name]}")

    def _build_actuated_xml(self, model: mujoco.MjModel) -> str | None:
        """Create a MJCF with position actuators for all non-free joints."""
        try:
            tmp_dir = os.path.join(os.path.dirname(self.xml_path), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.xml_path))[0]
            saved_xml = os.path.join(tmp_dir, f"{base_name}_mjcf.xml")
            mujoco.mj_saveLastXML(saved_xml, model)
        except Exception as exc:
            print(f"Failed to save MJCF from model: {exc}")
            return None

        try:
            tree = ET.parse(saved_xml)
            root = tree.getroot()
        except Exception as exc:
            print(f"Failed to parse MJCF: {exc}")
            return None

        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")

        existing = {elem.get("joint") for elem in actuator.findall("position") if elem.get("joint")}
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not joint_name or joint_name in existing:
                continue
            jmin, jmax = model.jnt_range[i]
            ctrl_attrs = {
                "joint": joint_name,
                "kp": f"{self.actuator_kp}",
                "kv": f"{self.actuator_kv}",
            }
            if jmin != 0.0 or jmax != 0.0:
                ctrl_attrs["ctrllimited"] = "true"
                ctrl_attrs["ctrlrange"] = f"{jmin} {jmax}"
            ET.SubElement(
                actuator,
                "position",
                ctrl_attrs,
            )

        actuated_xml = os.path.join(tmp_dir, f"{base_name}_actuated.xml")
        tree.write(actuated_xml, encoding="utf-8", xml_declaration=True)
        print(f"Generated actuated MJCF: {actuated_xml}")
        return actuated_xml

    def _send_command(self):
        qpos_desired = calc_mujoco_qpos_from_placo_q(
            self.mj_model,
            self.placo_robot,
            self.placo_robot.state.q,
            floating_base=self.floating_base,
        )

        for gripper_name, gripper_target in self.gripper_pos_target.items():
            for joint_name, joint_pos in gripper_target.items():
                success = set_mujoco_joint_pos_by_name(
                    self.mj_model,
                    qpos_desired,
                    joint_name,
                    joint_pos,
                )
                if not success:
                    raise ValueError(f"Joint '{gripper_name}' not found in MuJoCo model.")

        if self.direct_qpos or self.mj_model.nu == 0:
            self.mj_data.qpos[:] = qpos_desired
            self.mj_data.qvel[:] = 0.0
            self.mj_data.qacc[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            self.mj_data.ctrl = calc_mujoco_ctrl_from_qpos(self.mj_model, qpos_desired)

        if self.visualize_placo:
            self._update_placo_viz()

    def _apply_gui_ctrl_to_qpos(self) -> None:
        if self.mj_model.nu == 0:
            return
        for i in range(self.mj_model.nu):
            trn_type = self.mj_model.actuator_trntype[i]
            if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint_id = int(self.mj_model.actuator_trnid[i][0])
            if self.mj_model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            qpos_adr = int(self.mj_model.jnt_qposadr[joint_id])
            ctrl_val = float(self.mj_data.ctrl[i])
            jmin, jmax = self.mj_model.jnt_range[joint_id]
            if jmin != 0.0 or jmax != 0.0:
                ctrl_val = max(-1.0, min(1.0, ctrl_val))
                self.mj_data.qpos[qpos_adr] = jmin + (ctrl_val + 1.0) * 0.5 * (jmax - jmin)
            else:
                self.mj_data.qpos[qpos_adr] = ctrl_val
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _update_robot_state(self):
        mj_qpos = self.mj_data.qpos.copy()
        self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
            self.mj_model,
            self.placo_robot,
            mj_qpos,
            floating_base=self.floating_base,
        )
        self.placo_robot.update_kinematics()

    def _update_mocap_target(self):
        for name, task in self.effector_task.items():
            mocap_idx = self.target_mocap_idx.get(name)
            if mocap_idx is None or mocap_idx == -1:
                continue

            if hasattr(task, "T_world_frame"):
                T_world_target = task.T_world_frame
                self.mj_data.mocap_pos[mocap_idx] = T_world_target[:3, 3]
                self.mj_data.mocap_quat[mocap_idx] = tf.quaternion_from_matrix(T_world_target)
            else:
                self.mj_data.mocap_pos[mocap_idx] = task.target_world

    def _get_link_pose(self, ee_name):
        """Get the end effector position and orientation."""
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if ee_id == -1:
            raise ValueError(f"End effector body '{ee_name}' not found in the model.")

        ee_xyz = self.mj_data.xpos[ee_id].copy()
        ee_quat = self.mj_data.xquat[ee_id].copy()

        return ee_xyz, ee_quat

    def run(self):
        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Set up viewer camera
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0.2, 0, 0]

            while not self._stop_event.is_set():
                try:
                    if self.control_source == "gui":
                        if self.gui_direct_qpos:
                            self._apply_gui_ctrl_to_qpos()
                        mujoco.mj_step(self.mj_model, self.mj_data)
                        viewer.sync()
                        if self.debug_log_interval > 0.0:
                            now = float(self.mj_data.time)
                            if now - self._last_gui_log_time >= self.debug_log_interval:
                                self._last_gui_log_time = now
                                for joint_name, joint_id in self._debug_joint_ids.items():
                                    qpos_adr = int(self.mj_model.jnt_qposadr[joint_id])
                                    dof_adr = int(self.mj_model.jnt_dofadr[joint_id])
                                    qpos = float(self.mj_data.qpos[qpos_adr])
                                    qvel = float(self.mj_data.qvel[dof_adr])
                                    ctrl_idx = self._joint_to_ctrl_index.get(joint_name)
                                    ctrl_val = None if ctrl_idx is None else float(self.mj_data.ctrl[ctrl_idx])
                                    print(
                                        f"[GUI] t={now:.2f}s {joint_name} qpos={qpos:.4f} qvel={qvel:.4f} ctrl={ctrl_val}"
                                    )
                        continue

                    self._update_robot_state()
                    self._update_ik()
                    self._update_gripper_target()
                    self._update_mocap_target()
                    self._send_command()

                    # Step simulation and update viewer
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    viewer.sync()
                except KeyboardInterrupt:
                    print("\nTeleoperation stopped.")
                    self._stop_event.set()
