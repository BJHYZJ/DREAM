#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm Controller (always-ready version)
- 自动 ensure_ready(): motion_enable -> set_mode(0) -> set_state(0)
- 角度单位自适应：若疑似“度”(abs>6.3)，自动转弧度
- 默认速度/加速度使用官方示例级别：speed=0.35 rad/s, acc=10 rad/s^2
"""

import logging
from typing import Optional, List
import time
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState

# xArm service types
from xarm_msgs.srv import (
    SetInt16, SetInt16ById, SetFloat32List,
    MoveCartesian, MoveJoint, MoveHome,
    GetFloat32, GetFloat32List, GetInt16,
    GripperMove, Call
)

# ---------------- Constants ----------------
CONTROL_HZ = 10.0

# 关节默认（rad 系）
JOINT_DEFAULT_SPEED_RAD_S = 0.35    # rad/s
JOINT_DEFAULT_ACC_RAD_S2   = 10.0   # rad/s^2

# 笛卡尔默认（mm 系）
CART_DEFAULT_SPEED_MM_S = 200.0     # mm/s（可按需要调整）
CART_DEFAULT_ACC_MM_S2  = 2000.0    # mm/s^2

DEFAULT_TIMEOUT = 10.0  # 秒

QUICK_TIMEOUT = 2.0
NORMAL_TIMEOUT = 5.0
LONG_TIMEOUT = 15.0


log = logging.getLogger(__name__)


class ArmControllerNode(Node):
    def __init__(self, hz: float = CONTROL_HZ):
        super().__init__("arm_controller")
        self.hz = hz

        # Controller state
        self.active = False
        self.is_moving = False

        # Robot state cache
        self.robot_state = -1
        self.robot_mode = -1
        self.robot_error = 0

        # Defaults（两套，不混用）
        self.joint_speed = JOINT_DEFAULT_SPEED_RAD_S
        self.joint_acc   = JOINT_DEFAULT_ACC_RAD_S2
        self.cart_speed  = CART_DEFAULT_SPEED_MM_S
        self.cart_acc    = CART_DEFAULT_ACC_MM_S2
        self.timeout     = DEFAULT_TIMEOUT

        # Clients & IO
        self.create_service_clients()
        self.create_pubs_subs_services()
        # Init sequence
        self.initialize_robot()
        self.get_logger().info("Arm Controller (always-ready, safe Cartesian) initialized.")

    # ------------------------- helpers -------------------------
    def create_service_clients(self):
        # Basic control
        self.motion_enable_client = self.create_client(SetInt16ById, "/xarm/motion_enable")
        self.set_mode_client = self.create_client(SetInt16, "/xarm/set_mode")
        self.set_state_client = self.create_client(SetInt16, "/xarm/set_state")
        self.clear_error_client = self.create_client(Call, "/xarm/clean_error")

        # Motion
        self.set_position_client = self.create_client(MoveCartesian, "/xarm/set_position")
        self.set_position_aa_client = self.create_client(MoveCartesian, "/xarm/set_position_aa")
        self.set_servo_angle_client = self.create_client(MoveJoint, "/xarm/set_servo_angle")
        self.move_gohome_client = self.create_client(MoveHome, "/xarm/move_gohome")

        # Gripper
        self.set_gripper_enable_client = self.create_client(SetInt16, "/xarm/set_gripper_enable")
        self.set_gripper_mode_client = self.create_client(SetInt16, "/xarm/set_gripper_mode")
        self.set_gripper_position_client = self.create_client(GripperMove, "/xarm/set_gripper_position")
        self.get_gripper_position_client = self.create_client(GetFloat32, "/xarm/get_gripper_position")
        self.clean_gripper_error_client = self.create_client(Call, "/xarm/clean_gripper_error")

        # Sensors
        self.get_position_client = self.create_client(GetFloat32List, "/xarm/get_position")
        self.get_servo_angle_client = self.create_client(GetFloat32List, "/xarm/get_servo_angle")
        self.get_state_client = self.create_client(GetInt16, "/xarm/get_state")

        # Safety
        self.set_collision_sensitivity_client = self.create_client(SetInt16, "/xarm/set_collision_sensitivity")
        self.set_tcp_offset_client = self.create_client(SetFloat32List, "/xarm/set_tcp_offset")
        self.set_tcp_load_client = self.create_client(SetFloat32List, "/xarm/set_tcp_load")

    def create_pubs_subs_services(self):
        # Pubs
        self.moving_pub = self.create_publisher(Bool, "arm_controller/moving", 1)

        # Subs
        self.create_subscription(Float32MultiArray, "arm_controller/goal_pose", self._goal_pose_callback, 1)
        self.create_subscription(Pose, "arm_controller/goal_pose_aa", self._goal_pose_aa_callback, 1)
        self.create_subscription(JointState, "arm_controller/goal_joints", self._goal_joints_callback, 1)

        self.create_subscription(Float32, "arm_controller/speed", self._set_joint_speed_callback, 1)
        self.create_subscription(Float32, "arm_controller/acceleration", self._set_joint_acc_callback, 1)
        self.create_subscription(Float32, "arm_controller/cart_speed", self._set_cart_speed_callback, 1)
        self.create_subscription(Float32, "arm_controller/cart_acceleration", self._set_cart_acc_callback, 1)

        # Srvs
        self.create_service(Trigger, "arm_controller/enable", self._enable_service)
        self.create_service(Trigger, "arm_controller/disable", self._disable_service)
        self.create_service(Trigger, "arm_controller/home", self._home_service)
        self.create_service(Trigger, "arm_controller/clear_error", self._clear_error_service)
        # Timer
        self.create_timer(1 / self.hz, self.status_update_callback)

    def wait_for_services(self, timeout: float = 5.0):
        essentials = [
            self.motion_enable_client,
            self.set_mode_client,
            self.set_state_client,
            self.clear_error_client,
            self.set_position_aa_client,
            self.set_servo_angle_client,
            self.set_gripper_enable_client,
        ]
        for c in essentials:
            if not c.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f"Essential service {c.srv_name} not available!")
                raise RuntimeError(f"Required service not available: {c.srv_name}")

    def _call_and_wait(self, client, request, timeout=NORMAL_TIMEOUT):
        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                self.get_logger().error(f"{client.srv_name} timeout")
                return False, None
            resp = future.result()
            ok = True
            msg = getattr(resp, 'message', '')
            if hasattr(resp, 'ret'):
                ok = (resp.ret == 0)
            log_fn = self.get_logger().info if ok else self.get_logger().error
            log_fn(f"{client.srv_name} -> {'ok' if ok else 'failed'}: {msg}")
            return ok, resp
        except Exception as e:
            self.get_logger().error(f"{client.srv_name} exception: {e}")
            return False, None

    def ensure_ready(self) -> bool:
        """Idempotent: 多次调用也安全。"""
        ok1, _ = self._call_and_wait(self.motion_enable_client, SetInt16ById.Request(id=8, data=1))
        ok2, _ = self._call_and_wait(self.set_mode_client, SetInt16.Request(data=0))
        ok3, _ = self._call_and_wait(self.set_state_client, SetInt16.Request(data=0))
        if ok2: self.robot_mode = 0
        if ok3: self.robot_state = 0
        return ok1 and ok2 and ok3


    def initialize_robot(self):
        self.get_logger().info("Initializing robot...")
        self.wait_for_services()
        # 清错
        self.clean_error()
        self.clean_gripper_error()
        # 就绪
        self.ensure_ready()

        self.set_gripper_enable(True)
        self.set_gripper_mode(0)

        # 安全参数
        self._call_and_wait(self.set_collision_sensitivity_client, SetInt16.Request(data=1), timeout=QUICK_TIMEOUT)
        
        init_deg = [0, -45, -90, 0, 90, 0]  # 会根据 DOF 截断
        self.init_servo_angle = [math.radians(a) for a in init_deg]

        time.sleep(1.0)

        self.active = True  # 初始化时启用
        self.set_servo_angle(angle=self.init_servo_angle, wait=True)
        self.open_gripper()
        self.active = False
        self.get_logger().info("Robot initialization completed.")

    # ------------------------- basic control -------------------------
    def motion_enable(self, enable: bool, servo_id: int = 8) -> bool:
        ok, _ = self._call_and_wait(self.motion_enable_client, SetInt16ById.Request(id=servo_id, data=1 if enable else 0))
        return ok

    def set_mode(self, mode: int) -> bool:
        ok, _ = self._call_and_wait(self.set_mode_client, SetInt16.Request(data=mode))
        if ok:
            self.robot_mode = mode
        return ok

    def set_state(self, state: int) -> bool:
        ok, _ = self._call_and_wait(self.set_state_client, SetInt16.Request(data=state))
        if ok:
            self.robot_state = state
        return ok

    def clean_error(self) -> bool:
        ok, _ = self._call_and_wait(self.clear_error_client, Call.Request())
        if ok:
            self.robot_error = 0
        return ok

    def clean_gripper_error(self) -> bool:
        ok, _ = self._call_and_wait(self.clean_gripper_error_client, Call.Request())
        return ok

    # ------------------------- motion -------------------------
    def set_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None,
                     speed: float = None, acc: float = None, wait: bool = True, timeout: float = None) -> bool:

        if not self.ensure_ready():
            self.get_logger().error("Robot not ready (enable/mode/state)")
            return False

        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot move")
            return False
            
        speed = (speed if speed is not None else self.cart_speed)
        acc = (acc if acc is not None else self.cart_acc)
        timeout = timeout or NORMAL_TIMEOUT

        ok, resp = self._call_and_wait(self.get_position_client, GetFloat32List.Request(), timeout=QUICK_TIMEOUT)
        if not ok or resp is None:
            self.get_logger().error("Cannot get current position")
            return False
        cur = list(resp.datas)
        pose = [
            cur[0] if x is None else x,
            cur[1] if y is None else y,
            cur[2] if z is None else z,
            cur[3] if roll is None else roll,
            cur[4] if pitch is None else pitch,
            cur[5] if yaw is None else yaw,
        ]

        req = MoveCartesian.Request()
        req.pose = pose
        req.speed = speed
        req.acc = acc
        req.wait = wait
        req.timeout = timeout

        self.is_moving = True
        ok, resp = self._call_and_wait(self.set_position_client, req, timeout=timeout)
        self.is_moving = False
        return ok

    def set_position_aa(self, pose: List[float], speed: float = None, acc: float = None,
                        is_tool_coord: bool = False, relative: bool = False,
                        wait: bool = True, timeout: float = None) -> bool:
        if not self.ensure_ready():
            self.get_logger().error("Robot not ready (enable/mode/state)")
            return False

        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot move")
            return False

        speed = (speed if speed is not None else self.cart_speed)
        acc = (acc if acc is not None else self.cart_acc)
        timeout = timeout or NORMAL_TIMEOUT

        req = MoveCartesian.Request()
        req.pose = pose
        req.speed = speed
        req.acc = acc
        req.is_tool_coord = is_tool_coord
        req.relative = relative
        req.wait = wait
        req.timeout = timeout

        self.is_moving = True
        ok, _ = self._call_and_wait(self.set_position_aa_client, req, timeout=timeout)
        self.is_moving = False
        return ok

    def set_servo_angle(self, servo_id: int = None, angle: float = None,
                        speed: float = None, acc: float = None,
                        wait: bool = True, timeout: float = None) -> bool:

        if not self.ensure_ready():
            self.get_logger().error("Robot not ready (enable/mode/state)")
            return False

        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot move")
            return False

        speed = (speed if speed is not None else self.joint_speed)
        acc = (acc if acc is not None else self.joint_acc)
        timeout = timeout or NORMAL_TIMEOUT

        req = MoveJoint.Request()
        if servo_id is None or servo_id == 8:
            if not isinstance(angle, (list, tuple)):
                self.get_logger().error("For all joints, angle must be a list")
                return False

            # 与当前 DOF 对齐
            ok, resp = self._call_and_wait(self.get_servo_angle_client, GetFloat32List.Request(), timeout=QUICK_TIMEOUT)
            if not ok or resp is None:
                self.get_logger().error("Cannot read current joint angles")
                return False

            req.angles = [float(a) for a in angle]
        else:
            ok, resp = self._call_and_wait(self.get_servo_angle_client, GetFloat32List.Request(), timeout=QUICK_TIMEOUT)
            if not ok or resp is None:
                self.get_logger().error("Cannot get current joint angles")
                return False
            current = list(resp.datas)
            if servo_id < 1 or servo_id > len(current):
                self.get_logger().error(f"Invalid servo_id: {servo_id}")
                return False
            # 单关节角度单位自适应
            current[servo_id - 1] = float(angle)
            req.angles = current

        req.speed = speed
        req.acc = acc
        req.wait = wait
        req.timeout = timeout

        self.is_moving = True
        ok, resp = self._call_and_wait(self.set_servo_angle_client, req, timeout=timeout)
        self.is_moving = False
        return ok

    def move_gohome(self, speed: float = None, acc: float = None,
                    wait: bool = True, timeout: float = None) -> bool:

        if not self.ensure_ready():
            self.get_logger().error("Robot not ready (enable/mode/state)")
            return False
            
        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot move")
            return False

        speed = (speed if speed is not None else self.joint_speed)
        acc = (acc if acc is not None else self.joint_acc)
        timeout = timeout or LONG_TIMEOUT

        req = MoveHome.Request()
        req.speed = speed
        req.acc = acc
        req.wait = wait
        req.timeout = timeout
        self.is_moving = True
        ok, _ = self._call_and_wait(self.move_gohome_client, req, timeout=timeout)
        self.is_moving = False
        return ok

    # ------------------------- gripper -------------------------
    def set_gripper_enable(self, enable: bool) -> bool:
        ok, _ = self._call_and_wait(self.set_gripper_enable_client, SetInt16.Request(data=1 if enable else 0))
        return ok

    def set_gripper_mode(self, mode: int) -> bool:
        ok, _ = self._call_and_wait(self.set_gripper_mode_client, SetInt16.Request(data=mode))
        return ok

    def set_gripper_position(self, pos: float, wait: bool = False, timeout: float = None) -> bool:
        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot control gripper")
            return False
        
        req = GripperMove.Request()
        req.pos = float(pos)
        req.wait = wait
        req.timeout = timeout or NORMAL_TIMEOUT
        ok, _ = self._call_and_wait(self.set_gripper_position_client, req, timeout=req.timeout)
        return ok

    def open_gripper(self, wait: bool = True, half_open: bool = False) -> bool:
        return self.set_gripper_position(400.0 if half_open else 800.0, wait)

    def close_gripper(self, wait: bool = True) -> bool:
        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot close gripper")
            return False

        return self.set_gripper_position(0.0, wait)

    def get_gripper_position(self) -> Optional[float]:
        ok, resp = self._call_and_wait(self.get_gripper_position_client, GetFloat32.Request(), timeout=LONG_TIMEOUT)
        if ok and resp:
            return resp.data
        return None

    # ------------------------- sensors -------------------------
    def get_position(self) -> Optional[List[float]]:
        ok, resp = self._call_and_wait(self.get_position_client, GetFloat32List.Request(), timeout=QUICK_TIMEOUT)
        return list(resp.datas) if ok and resp else None

    def get_servo_angle(self) -> Optional[List[float]]:
        ok, resp = self._call_and_wait(self.get_servo_angle_client, GetFloat32List.Request(), timeout=QUICK_TIMEOUT)
        return list(resp.datas) if ok and resp else None

    def get_state(self) -> Optional[int]:
        ok, resp = self._call_and_wait(self.get_state_client, GetInt16.Request(), timeout=QUICK_TIMEOUT)
        if ok and resp:
            self.robot_state = resp.data
            return resp.data
        return None

    # ------------------------- utils -------------------------
    def is_robot_ready(self) -> bool:
        return (self.robot_state == 0 and self.robot_mode == 0 and self.robot_error == 0)

    # ------------------------- callbacks -------------------------
    def _goal_pose_aa_callback(self, msg: Pose):
        if not self.active:
            self.get_logger().warn("Arm controller not active, cannot move")
            return False
            
        pose_list = [
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ]
        self.set_position_aa(pose_list)

    def _goal_pose_callback(self, msg: Float32MultiArray):
        if not self.active:
            self.get_logger().warn("Received goal pose while NOT active. Auto-enable & proceed.")
            return False

        if len(msg.data) != 6:
            self.get_logger().error(f"Invalid xyzrpy length: {len(msg.data)}, expected 6")
            return
        x, y, z, roll, pitch, yaw = msg.data
        self.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def _goal_joints_callback(self, msg: JointState):

        if not self.active:
            self.get_logger().warn("Received goal joints while NOT active. Auto-enable & proceed.")
            return False

        if msg.position:
            self.set_servo_angle(angle=list(msg.position))
        else:
            self.get_logger().warn("Empty JointState.position")


    # speed/acc setters
    def _set_joint_speed_callback(self, msg: Float32):
        self.joint_speed = float(msg.data)
        self.get_logger().info(f"Joint speed set to {self.joint_speed} rad/s")

    def _set_joint_acc_callback(self, msg: Float32):
        self.joint_acc = float(msg.data)
        self.get_logger().info(f"Joint acc set to {self.joint_acc} rad/s^2")

    def _set_cart_speed_callback(self, msg: Float32):
        self.cart_speed = float(msg.data)
        self.get_logger().info(f"Cartesian speed set to {self.cart_speed} mm/s")

    def _set_cart_acc_callback(self, msg: Float32):
        self.cart_acc = float(msg.data)
        self.get_logger().info(f"Cartesian acc set to {self.cart_acc} mm/s^2")


    # services
    def _enable_service(self, request, response):
        self.active = True
        response.success = True
        response.message = "Arm controller ENABLED"
        return response

    def _disable_service(self, request, response):
        self.active = False
        self.is_moving = False
        response.success = True
        response.message = "Arm controller DISABLED"
        return response

    def _home_service(self, request, response):
        ok = self.move_gohome()
        response.success = ok
        response.message = "Home done" if ok else "Home failed"
        return response

    def _clear_error_service(self, request, response):
        ok1 = self.clean_error()
        ok2 = self.clean_gripper_error()
        response.success = ok1 and ok2
        response.message = "Error cleared" if (ok1 and ok2) else "Failed to clear error"
        return response

    def status_update_callback(self):
        self.moving_pub.publish(Bool(data=self.is_moving))


def main():
    rclpy.init()
    node = ArmControllerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
