import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Tuple

import mujoco
import numpy as np

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

# python teleop_dual_marvin_m6_mujoco.py --scale-factor 4.0 --scale-rpy 1.5 --use-relative --debug-log-interval 0.2
# python teleop_dual_marvin_m6_mujoco.py --control-source gui
# python teleop_dual_marvin_m6_mujoco.py --scale-factor 4.0 --scale-rpy 1.5 --use-relative --debug-log-interval 0.2 --visualize-placo --placo-targets-only --align-pos-only

@dataclass
class ArmSpec:
    name: str
    urdf_path: str
    mesh_dir: str
    base_link: str


def _resolve_mesh_paths(root: ET.Element, mesh_dir: str) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            basename = os.path.basename(filename)
            mesh.set("filename", os.path.join(mesh_dir, basename))
        elif filename.startswith("file://"):
            mesh.set("filename", filename.replace("file://", "", 1))


def _collect_robot_children(root: ET.Element) -> Iterable[ET.Element]:
    for child in list(root):
        if child.tag in {"link", "joint", "material", "transmission", "gazebo"}:
            yield child


def _add_fixed_joint(
    parent: ET.Element,
    name: str,
    parent_link: str,
    child_link: str,
    xyz: Tuple[float, float, float],
    rpy: Tuple[float, float, float],
) -> None:
    joint = ET.SubElement(parent, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": f"{xyz[0]} {xyz[1]} {xyz[2]}",
            "rpy": f"{rpy[0]} {rpy[1]} {rpy[2]}",
        },
    )
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})


def _build_dual_urdf(
    left: ArmSpec,
    right: ArmSpec,
    out_path: str,
    left_xyz: Tuple[float, float, float],
    right_xyz: Tuple[float, float, float],
    left_rpy: Tuple[float, float, float],
    right_rpy: Tuple[float, float, float],
) -> str:
    left_tree = ET.parse(left.urdf_path)
    right_tree = ET.parse(right.urdf_path)
    left_root = left_tree.getroot()
    right_root = right_tree.getroot()

    _resolve_mesh_paths(left_root, left.mesh_dir)
    _resolve_mesh_paths(right_root, right.mesh_dir)

    combined = ET.Element("robot", {"name": "marvin_m6_dual"})
    ET.SubElement(combined, "link", {"name": "world"})

    for child in _collect_robot_children(left_root):
        combined.append(child)
    for child in _collect_robot_children(right_root):
        combined.append(child)

    _add_fixed_joint(
        combined,
        "world_to_base_left",
        "world",
        left.base_link,
        left_xyz,
        left_rpy,
    )
    _add_fixed_joint(
        combined,
        "world_to_base_right",
        "world",
        right.base_link,
        right_xyz,
        right_rpy,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(combined).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _default_paths(repo_root: str) -> Tuple[ArmSpec, ArmSpec]:
    left_root = os.path.join(
        repo_root,
        "Marvin M6-CCS-696-urdf_V4.0",
        "Marvin M6-S-L-CCS-696-V4.0 urdf (FUSION)",
    )
    right_root = os.path.join(
        repo_root,
        "Marvin M6-CCS-696-urdf_V4.0",
        "Marvin M6-S-R-CCS-696-V4.0 urdf (FUSION)",
    )
    left = ArmSpec(
        name="left",
        urdf_path=os.path.join(left_root, "urdf", "Marvin M6-S-L-CCS-696-V4.0 urdf.urdf"),
        mesh_dir=os.path.join(left_root, "meshes"),
        base_link="Base_L",
    )
    right = ArmSpec(
        name="right",
        urdf_path=os.path.join(right_root, "urdf", "Marvin M6-S-R-CCS-696-V4.0 urdf.urdf"),
        mesh_dir=os.path.join(right_root, "meshes"),
        base_link="Base_R",
    )
    return left, right


def _human_arm_joint_values() -> dict:
    return {
        # 弧度
        # 抬平
        # "Joint1_L": 0.0,
        # "Joint2_L": 0.0,
        # "Joint3_L": 0.0,
        # "Joint4_L": 0.0,
        # "Joint5_L": 0.0,
        # "Joint6_L": 0.0,
        # "Joint7_L": 0.0,
        # "Joint1_R": 0.0,
        # "Joint2_R": 0.0,
        # "Joint3_R": 0.0,
        # "Joint4_R": 0.0,
        # "Joint5_R": 0.0,
        # "Joint6_R": 0.0,
        # "Joint7_R": 0.0,
        "Joint1_L": 1.08,
        "Joint2_L": -1.26,
        "Joint3_L": -0.895,
        "Joint4_L": -2.14,
        "Joint5_L": 0.0,
        "Joint6_L": 0.0,
        "Joint7_L": 0.0,
        "Joint1_R": -1.08,
        "Joint2_R": -1.26,
        "Joint3_R": 0.895,
        "Joint4_R": -2.14,
        "Joint5_R": 0.0,
        "Joint6_R": 0.0,
        "Joint7_R": 0.0,
        
    }


def _build_qpos_from_joint_names(joint_names: Iterable[str], joint_values: dict) -> list:
    qpos = []
    for name in joint_names:
        qpos.append(float(joint_values.get(name, 0.0)))
    return qpos


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    left_spec, right_spec = _default_paths(repo_root)

    parser = argparse.ArgumentParser(description="Dual Marvin M6 teleop in MuJoCo (URDF-based)")
    parser.add_argument("--left-urdf", default=left_spec.urdf_path)
    parser.add_argument("--right-urdf", default=right_spec.urdf_path)
    parser.add_argument("--disable-contact", action="store_true", default=True)
    parser.add_argument("--enable-contact", dest="disable_contact", action="store_false")
    parser.add_argument("--left-mesh-dir", default=left_spec.mesh_dir)
    parser.add_argument("--right-mesh-dir", default=right_spec.mesh_dir)
    parser.add_argument("--left-base-link", default=left_spec.base_link)
    parser.add_argument("--right-base-link", default=right_spec.base_link)
    parser.add_argument("--left-base-xyz", nargs=3, type=float, default=[0.0, 0.2, 0.0])
    parser.add_argument("--right-base-xyz", nargs=3, type=float, default=[0.0, -0.2, 0.0])
    parser.add_argument("--left-base-rpy", nargs=3, type=float, default=[-1.5708, 0.0, 0.0])# 1.5708
    parser.add_argument("--right-base-rpy", nargs=3, type=float, default=[1.5708, 0.0, 0.0])
    parser.add_argument("--scale-factor", type=float, default=1)
    parser.add_argument("--axis-map", nargs=3, default=["-z", "-x", "y"])
    parser.add_argument(
        "--r-vr-to-robot", # never 做两次旋转
        nargs=9,
        type=float,
        default=[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    )
    parser.add_argument("--rpy-offset-deg", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--scale-rpy", type=float, default=0.7) # 把控制器的旋转增量（角轴）按比例放大/缩小
    parser.add_argument("--use-relative", action="store_true", default=True)
    parser.add_argument("--ee-local-rotation", action="store_true", default=True)
    parser.add_argument("--debug-log-interval", type=float, default=0.5)
    parser.add_argument("--no-gravity", action="store_true", default=False)
    parser.add_argument("--joint-damping", type=float, default=0.5)
    parser.add_argument("--joint-armature", type=float, default=0.0)
    parser.add_argument("--auto-actuators", action="store_true", default=True)
    parser.add_argument("--no-auto-actuators", dest="auto_actuators", action="store_false")
    parser.add_argument("--actuator-kp", type=float, default=50.0)# K 位置伺服的 PD 增益
    parser.add_argument("--actuator-kv", type=float, default=5.0) # D
    parser.add_argument("--control-mode", choices=["pose", "position"], default="pose") # position only for end position, pose for both position and pose
    parser.add_argument("--control-source", choices=["teleop", "gui"], default="teleop")
    parser.add_argument("--require-alignment", action="store_true", default=True)
    parser.add_argument("--no-alignment", dest="require_alignment", action="store_false")
    parser.add_argument("--align-pos-tol", type=float, default=0.05)
    parser.add_argument("--align-rot-tol-deg", type=float, default=15.0)
    parser.add_argument("--align-pos-only", action="store_true", default=False)
    parser.add_argument("--direct-qpos", action="store_true", default=False)
    parser.add_argument("--gui-direct-qpos", action="store_true", default=False)
    parser.add_argument("--ctrlrange-from-limits", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--visualize-placo", action="store_true", default=False)
    parser.add_argument("--placo-targets-only", action="store_true", default=False)
    parser.add_argument("--init-pose", choices=["neutral", "human"], default="human")
    args = parser.parse_args()

    left_spec = ArmSpec(
        name="left",
        urdf_path=args.left_urdf,
        mesh_dir=args.left_mesh_dir,
        base_link=args.left_base_link,
    )
    right_spec = ArmSpec(
        name="right",
        urdf_path=args.right_urdf,
        mesh_dir=args.right_mesh_dir,
        base_link=args.right_base_link,
    )

    combined_urdf_path = os.path.join(repo_root, "tmp", "marvin_m6_dual.urdf")
    _build_dual_urdf(
        left_spec,
        right_spec,
        combined_urdf_path,
        tuple(args.left_base_xyz),
        tuple(args.right_base_xyz),
        tuple(args.left_base_rpy),
        tuple(args.right_base_rpy),
    )

    xml_path = combined_urdf_path

    mj_qpos_init = None
    q_init = None
    if args.init_pose == "human":
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
        joint_values = _human_arm_joint_values()
        mj_qpos_init = _build_qpos_from_joint_names(joint_names, joint_values)
        q_init = np.array(mj_qpos_init, dtype=float)

    # 按键配置
    config = {
        "right_hand": {
            "link_name": "Link7_R",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "control_mode": args.control_mode,
        },
        "left_hand": {
            "link_name": "Link7_L",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "control_mode": args.control_mode,
        },
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=combined_urdf_path,
        manipulator_config=config,
        scale_factor=args.scale_factor,
        axis_map=args.axis_map,
        r_vr_to_robot=args.r_vr_to_robot,
        rpy_offset_deg=args.rpy_offset_deg,
        scale_rpy=args.scale_rpy,
        use_relative=args.use_relative,
        use_ee_local_rotation=args.ee_local_rotation,
        R_headset_world=np.eye(3),
        visualize_placo=args.visualize_placo,
        visualize_robot_body=not args.placo_targets_only,
        mj_qpos_init=mj_qpos_init,
        q_init=q_init,
        dt=args.dt,
        debug_log_interval=args.debug_log_interval,
        disable_gravity=args.no_gravity,
        disable_contact=args.disable_contact,
        joint_damping=args.joint_damping,
        joint_armature=args.joint_armature,
        auto_actuators=args.auto_actuators,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        control_source=args.control_source,
        direct_qpos=args.direct_qpos,
        gui_direct_qpos=args.gui_direct_qpos,
        ctrlrange_from_limits=args.ctrlrange_from_limits,
        require_alignment=args.require_alignment,
        align_pos_tol=args.align_pos_tol,
        align_rot_tol_deg=args.align_rot_tol_deg,
        align_pos_only=args.align_pos_only,
    )

    controller.run()


if __name__ == "__main__":
    main()
