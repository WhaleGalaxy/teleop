"""
读取 PICO 控制器位姿并发布 ROS 2 /vr/controller_pose，
用于配合 vr_teleop_node 进行机器人遥操作。

使用方法（示例）：
  ros2 run <your_pkg> pico_teleop_publisher.py

参数（ROS 2 参数）：
  - controller: "right" | "left"（默认 right）
  - topic: 发布话题（默认 /vr/controller_pose）
  - frame_id: PoseStamped.frame_id（默认 vr）
  - rate_hz: 发布频率（默认 60）
  - scale_pos: 位置缩放（默认 1.0）
  - pos_offset_m: 位置偏移(m)，长度 3（默认 [0,0,0]）
  - axis_map: 坐标轴映射，例如 ["x","-z","y"]（默认 ["x","y","z"]）
"""

from typing import Iterable, Tuple, Optional
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import xrobotoolkit_sdk as xrt


def _apply_axis_map(vec, axis_map):
    axes = {"x": vec[0], "y": vec[1], "z": vec[2]}
    out = []
    for key in axis_map:
        sign = 1.0
        axis = key
        if isinstance(key, str) and key.startswith("-"):
            sign = -1.0
            axis = key[1:]
        out.append(sign * axes[axis])
    return out


def _parse_pose(pose) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """兼容多种返回格式，统一为 (x,y,z,qx,qy,qz,qw)。"""
    if pose is None:
        return None

    # dict 形式
    if isinstance(pose, dict):
        keys = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if all(k in pose for k in keys):
            return (
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
                float(pose["qx"]),
                float(pose["qy"]),
                float(pose["qz"]),
                float(pose["qw"]),
            )

    # list/tuple 形式
    if isinstance(pose, (list, tuple)) and len(pose) >= 7:
        return (
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
            float(pose[6]),
        )

    # 对象属性形式
    attrs = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if all(hasattr(pose, a) for a in attrs):
        return (
            float(getattr(pose, "x")),
            float(getattr(pose, "y")),
            float(getattr(pose, "z")),
            float(getattr(pose, "qx")),
            float(getattr(pose, "qy")),
            float(getattr(pose, "qz")),
            float(getattr(pose, "qw")),
        )

    return None


class PicoTeleopPublisher(Node):
    def __init__(self):
        super().__init__("pico_teleop_publisher")

        self.declare_parameter("controller", "right")
        self.declare_parameter("topic", "/vr/controller_pose")
        self.declare_parameter("frame_id", "vr")
        self.declare_parameter("rate_hz", 60)
        self.declare_parameter("scale_pos", 1.0)
        self.declare_parameter("pos_offset_m", [0.0, 0.0, 0.0])
        self.declare_parameter("axis_map", ["x", "y", "z"])

        self.controller = self.get_parameter("controller").value
        self.topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.scale_pos = float(self.get_parameter("scale_pos").value)
        self.pos_offset_m = list(self.get_parameter("pos_offset_m").value)
        self.axis_map = list(self.get_parameter("axis_map").value)

        self.pub = self.create_publisher(PoseStamped, self.topic, 10)

        xrt.init()
        self.get_logger().info("xrobotoolkit_sdk init OK")

        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._tick)

    def _get_pose(self):
        if str(self.controller).lower() == "left":
            return xrt.get_left_controller_pose()
        return xrt.get_right_controller_pose()

    def _tick(self):
        raw_pose = self._get_pose()
        parsed = _parse_pose(raw_pose)
        if parsed is None:
            self.get_logger().warning("无法解析控制器位姿，跳过发布")
            return

        x, y, z, qx, qy, qz, qw = parsed

        pos = [x * self.scale_pos, y * self.scale_pos, z * self.scale_pos]
        pos = _apply_axis_map(pos, self.axis_map)
        pos = [pos[i] + self.pos_offset_m[i] for i in range(3)]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)

        self.pub.publish(msg)

    def destroy_node(self):
        try:
            xrt.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = PicoTeleopPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
