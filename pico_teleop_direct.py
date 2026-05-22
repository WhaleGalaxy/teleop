"""
直接读取 PICO 位姿并调用机器人 SDK 实现遥操作（不走 ROS 话题）。
python pico_teleop_direct.py --controller both

依赖：
- xrobotoolkit_sdk（来自 xr-robotics 环境）
- Marvin 机器人 SDK（需可被 import）

参数（命令行）：
  --robot-ip          机器人 IP
  --arm               机械臂标识（默认 A）
  --rate-hz           控制频率
  --vel-mm-s          速度
  --acc-mm-s2         加速度
    --controller        left/right/both
    --arm-left          双臂模式下左臂标识（默认 A）
    --arm-right         双臂模式下右臂标识（默认 B）
  --scale-xyz         位置缩放
  --offset-xyz-mm     位置偏移(mm)，3个数
  --axis-map          坐标轴映射，例如 x -z y
  --rpy-offset-deg    姿态偏移(deg)，3个数
  --filter-alpha      一阶低通滤波系数
  --max-delta-mm      单步最大位置变化
  --max-delta-deg     单步最大角度变化
    --log-interval      日志输出间隔（秒）
    --enable-robot      是否连接机器人（默认 False）
"""

from __future__ import annotations

import argparse
import ctypes
import math
import time
from typing import List, Optional, Tuple

import xrobotoolkit_sdk as xrt

try:
    from vr_teleop.SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
    from vr_teleop.SDK_PYTHON.fx_kine import Marvin_Kine
except Exception as exc:  # pragma: no cover
    Marvin_Robot = None
    DCSS = None
    Marvin_Kine = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None


def _normalize_quat(qx: float, qy: float, qz: float, qw: float) -> List[float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    qx, qy, qz, qw = _normalize_quat(qx, qy, qz, qw)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _rot_xyz_to_euler_xyz_deg(r: List[List[float]]) -> List[float]:
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    roll = math.atan2(r[2][1], r[2][2])
    yaw = math.atan2(r[1][0], r[0][0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _euler_xyz_deg_to_rot(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    rx = math.radians(roll)
    ry = math.radians(pitch)
    rz = math.radians(yaw)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
    ]


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return out


def _apply_axis_map(vec: List[float], axis_map: List[str]) -> List[float]:
    axes = {"x": vec[0], "y": vec[1], "z": vec[2]}
    out = []
    for key in axis_map:
        sign = 1.0
        axis = key
        if key.startswith("-"):
            sign = -1.0
            axis = key[1:]
        out.append(sign * axes[axis])
    return out


def _clamp(value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _limit_delta(current: float, target: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return target
    delta = target - current
    if abs(delta) > max_delta:
        return current + math.copysign(max_delta, delta)
    return target


def _parse_pose(pose) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    if pose is None:
        return None
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


class PicoTeleopDirect:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # 机器人 SDK 相关句柄
        self._robot = None
        self._dcss = None
        self._kine = None
        # 上一帧目标（用于滤波/限速），按手柄独立保存
        self._last_xyzabc = {}
        # 日志限频用（按手柄独立限频，避免互相抑制）
        self._last_raw_log_time = {}
        self._last_target_log_time = {}

        # 初始化 PICO SDK
        xrt.init()

        # 只有在明确开启机器人时才连接
        if args.enable_robot:
            self._init_robot()

    def _patch_kine_prototypes(self) -> None:
        # 修正 ctypes 原型，避免不同构建导致的参数不匹配
        if self._kine is None:
            return
        try:
            self._kine.kine.FX_Robot_PLN_MOVLA_C.argtypes = [
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_long,
                ctypes.c_void_p,
            ]
            self._kine.kine.FX_Robot_PLN_MOVLA_C.restype = ctypes.c_bool
        except AttributeError:
            pass
        try:
            self._kine.kine.FX_Robot_PLN_MOVL_KeepJA_C.argtypes = [
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_long,
                ctypes.c_void_p,
            ]
            self._kine.kine.FX_Robot_PLN_MOVL_KeepJA_C.restype = ctypes.c_bool
        except AttributeError:
            pass

    def _init_robot(self) -> None:
        # 连接机器人并设置为笛卡尔阻抗模式
        if Marvin_Robot is None or DCSS is None or Marvin_Kine is None:
            raise RuntimeError(f"机器人 SDK 不可用: {_SDK_IMPORT_ERROR}")
        self._robot = Marvin_Robot()
        self._dcss = DCSS()
        self._kine = Marvin_Kine()
        self._patch_kine_prototypes()

        init = self._robot.connect(self.args.robot_ip)
        if init == 0:
            raise RuntimeError("failed to connect to robot")
        self._robot.check_error_and_clear(self._dcss)
        self._robot.clear_set()
        if self.args.controller == "both":
            arms = [self.args.arm_left, self.args.arm_right]
        else:
            arms = [self.args.arm]
        for arm in arms:
            self._robot.set_state(arm=arm, state=3)
            self._robot.set_impedance_type(arm=arm, type=2)
            self._robot.set_vel_acc(arm=arm, velRatio=10, AccRatio=10)
        self._robot.send_cmd()
        time.sleep(0.2)

    def _apply_limits(self, xyz_mm: List[float]) -> List[float]:
        # 对 XYZ 做上下限限制（0 表示不限制）
        out = []
        for i in range(3):
            min_v = self.args.min_xyz_mm[i] if self.args.min_xyz_mm[i] != 0.0 else None
            max_v = self.args.max_xyz_mm[i] if self.args.max_xyz_mm[i] != 0.0 else None
            out.append(_clamp(xyz_mm[i], min_v, max_v))
        return out

    def _filter(self, key: str, xyzabc: List[float]) -> List[float]:
        # 一阶低通滤波，减少抖动
        last = self._last_xyzabc.get(key)
        if last is None:
            return xyzabc
        alpha = max(0.0, min(1.0, self.args.filter_alpha))
        if alpha >= 1.0:
            return xyzabc
        return [
            alpha * xyzabc[i] + (1.0 - alpha) * last[i]
            for i in range(6)
        ]

    def _limit_step(self, key: str, xyzabc: List[float]) -> List[float]:
        # 单步变化限制，避免跳变过大
        last = self._last_xyzabc.get(key)
        if last is None:
            return xyzabc
        xyz = [
            _limit_delta(last[i], xyzabc[i], self.args.max_delta_mm)
            for i in range(3)
        ]
        rpy = [
            _limit_delta(last[i + 3], xyzabc[i + 3], self.args.max_delta_deg)
            for i in range(3)
        ]
        return xyz + rpy

    def _get_pose(self, controller: str):
        # 读取指定手柄的位姿
        if controller == "left":
            return xrt.get_left_controller_pose()
        return xrt.get_right_controller_pose()

    def _send_cartesian_target(self, arm: str, xyzabc: List[float], key: str) -> None:
        # 通过 SDK 生成规划点集并下发
        if self._robot is None or self._kine is None or self._dcss is None:
            return
        last = self._last_xyzabc.get(key)
        if last is None:
            start_xyzabc = xyzabc
        else:
            start_xyzabc = last
        sub_data = self._robot.subscribe(self._dcss)
        ref_joints = sub_data["outputs"][0]["fb_joint_pos"]
        _, pset = self._kine.movLA(
            start_xyzabc=start_xyzabc,
            end_xyzabc=xyzabc,
            ref_joints=ref_joints,
            vel=self.args.vel_mm_s,
            acc=self.args.acc_mm_s2,
            freq_hz=self.args.freq_hz,
        )
        if not pset:
            return
        self._robot.clear_set()
        self._robot.setPln_Cart(arm, pset)
        self._robot.send_cmd()
        self._kine.destroy_point_set(pset)

    def step(self) -> None:
        if self.args.controller == "both":
            self._step_one("left", self.args.arm_left)
            self._step_one("right", self.args.arm_right)
        else:
            self._step_one(self.args.controller, self.args.arm)

    def _step_one(self, controller: str, arm: str) -> None:
        # 读取指定手柄的位姿并处理
        raw_pose = self._get_pose(controller)
        parsed = _parse_pose(raw_pose)
        if parsed is None:
            self._log(controller, "无法解析 PICO 位姿，raw_pose=%s", raw_pose)
            return

        x, y, z, qx, qy, qz, qw = parsed

        # 原始位姿限频输出
        self._log_raw(
            controller,
            "[{0}] PICO 原始: pos=({1:.3f},{2:.3f},{3:.3f}) quat=({4:.4f},{5:.4f},{6:.4f},{7:.4f})",
            controller, x, y, z, qx, qy, qz, qw,
        )

        # 位置：米 -> 毫米 + 缩放
        pos_mm = [x * 1000.0 * self.args.scale_xyz,
              y * 1000.0 * self.args.scale_xyz,
              z * 1000.0 * self.args.scale_xyz]
        # 轴映射（适配不同坐标系）
        pos_mm = _apply_axis_map(pos_mm, self.args.axis_map)
        # 位置偏移
        pos_mm = [pos_mm[i] + self.args.offset_xyz_mm[i] for i in range(3)]
        # 位置限幅
        pos_mm = self._apply_limits(pos_mm)

        # 姿态：四元数 -> 旋转矩阵
        r_vr = _quat_to_rot(qx, qy, qz, qw)
        r_map = [
            self.args.r_vr_to_robot[0:3],
            self.args.r_vr_to_robot[3:6],
            self.args.r_vr_to_robot[6:9],
        ]
        # 额外姿态偏移（欧拉角 -> 旋转矩阵）
        r_offset = _euler_xyz_deg_to_rot(
            self.args.rpy_offset_deg[0],
            self.args.rpy_offset_deg[1],
            self.args.rpy_offset_deg[2],
        )
        # 组合映射 + 偏移
        r_target = _mat_mul(r_offset, _mat_mul(r_map, r_vr))
        # 旋转矩阵 -> 欧拉角（度）
        rpy_deg = _rot_xyz_to_euler_xyz_deg(r_target)

        # 合成机器人目标：XYZ(mm) + RPY(deg)
        xyzabc = pos_mm + rpy_deg
        # 滤波 + 限速
        xyzabc = self._filter(controller, xyzabc)
        xyzabc = self._limit_step(controller, xyzabc)

        # 目标限频输出（XYZ + RPY）
        self._log_target(
            controller,
            "[{0}] 目标: xyz_mm=({1:.1f},{2:.1f},{3:.1f}) rpy_deg=({4:.1f},{5:.1f},{6:.1f})",
            controller, xyzabc[0], xyzabc[1], xyzabc[2], xyzabc[3], xyzabc[4], xyzabc[5],
        )

        if self.args.enable_robot:
            self._send_cartesian_target(arm, xyzabc, controller)

        self._last_xyzabc[controller] = xyzabc

    def _log(self, key: str, fmt: str, *args) -> None:
        # 兼容旧调用（统一到目标限频）
        self._log_target(key, fmt, *args)

    def _log_raw(self, key: str, fmt: str, *args) -> None:
        # 原始位姿专用限频日志
        now = time.time()
        last = self._last_raw_log_time.get(key, 0.0)
        if now - last < self.args.log_interval:
            return
        try:
            print(fmt.format(*args))
        except Exception:
            print(fmt, *args)
        self._last_raw_log_time[key] = now

    def _log_target(self, key: str, fmt: str, *args) -> None:
        # 目标专用限频日志
        now = time.time()
        last = self._last_target_log_time.get(key, 0.0)
        if now - last < self.args.log_interval:
            return
        try:
            print(fmt.format(*args))
        except Exception:
            print(fmt, *args)
        self._last_target_log_time[key] = now

    def close(self) -> None:
        try:
            xrt.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICO 直连机器人遥操作")
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--arm", default="A")
    parser.add_argument("--rate-hz", type=float, default=90.0)
    parser.add_argument("--freq-hz", type=int, default=500)
    parser.add_argument("--vel-mm-s", type=float, default=150.0)
    parser.add_argument("--acc-mm-s2", type=float, default=300.0)
    parser.add_argument("--controller", choices=["left", "right", "both"], default="right")
    parser.add_argument("--arm-left", default="A") # 双手模式下分别控制左右臂
    parser.add_argument("--arm-right", default="B")
    parser.add_argument("--scale-xyz", type=float, default=1.0)
    parser.add_argument("--offset-xyz-mm", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--axis-map", nargs=3, default=["x", "y", "z"])
    parser.add_argument(
        "--r-vr-to-robot",
        nargs=9,
        type=float,
        default=[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        help="3x3 旋转矩阵（按行展开），用于 VR->机器人 坐标映射",
    )
    parser.add_argument("--rpy-offset-deg", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--filter-alpha", type=float, default=1.0)
    parser.add_argument("--max-delta-mm", type=float, default=0.0)
    parser.add_argument("--max-delta-deg", type=float, default=0.0)
    parser.add_argument("--max-xyz-mm", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--min-xyz-mm", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--enable-robot", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    teleop = PicoTeleopDirect(args)
    try:
        period = 1.0 / max(1.0, args.rate_hz)
        while True:
            start = time.time()
            teleop.step()
            dt = time.time() - start
            sleep_t = period - dt
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
