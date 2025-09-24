#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Hello Robot
# All rights reserved.

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import tf2_ros
from tf_transformations import quaternion_multiply, quaternion_inverse

from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry


# =========================
# ---- Helper Functions ----
# =========================
def quat_to_list(q: Quaternion):
    # geometry_msgs/Quaternion -> [x, y, z, w] (tf_transformations 期望的顺序)
    return [q.x, q.y, q.z, q.w]

def list_to_quat(lst):
    q = Quaternion()
    q.x, q.y, q.z, q.w = lst
    return q

def normalize_quat_inplace(q: Quaternion):
    n = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
    if n > 1e-12:
        q.x /= n; q.y /= n; q.z /= n; q.w /= n

def rotate_vec3_by_quat(vx, vy, vz, q_xyzw):
    # 旋转 3D 向量 v by 四元数 q： v' = q * v * q^{-1}，其中 v 当作纯虚四元数 [x,y,z,0]
    v = [vx, vy, vz, 0.0]
    vq = quaternion_multiply(quaternion_multiply(q_xyzw, v),
                             quaternion_inverse(q_xyzw))
    return vq[0], vq[1], vq[2]

def quat_to_rot(qxyzw):
    # 四元数 [x,y,z,w] 转 3x3 旋转矩阵
    x, y, z, w = qxyzw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),   2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),     1 - 2*(xx + yy)]
    ])

def adjoint_of_rotation(R):
    # 纯旋转的 6x6 Adjoint：blockdiag(R, R)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R     # 对位置/线速度部分
    Ad[3:, 3:] = R     # 对姿态/角速度部分
    return Ad

def compose_transforms(a: TransformStamped, b: TransformStamped) -> TransformStamped:
    """
    连乘刚体变换：T_out = a * b
    其中 a: P -> C1， b: C1 -> C2，输出: P -> C2
    旋转用四元数，平移用 t_out = t_a + R_a * t_b
    """
    out = TransformStamped()
    out.header = a.header
    out.child_frame_id = b.child_frame_id  # 最终 child

    ta = np.array([a.transform.translation.x,
                   a.transform.translation.y,
                   a.transform.translation.z])
    tb = np.array([b.transform.translation.x,
                   b.transform.translation.y,
                   b.transform.translation.z])

    qa = quat_to_list(a.transform.rotation)
    qb = quat_to_list(b.transform.rotation)

    # R_a * t_b 通过四元数作用实现
    t_rot = rotate_vec3_by_quat(tb[0], tb[1], tb[2], qa)
    t_out = ta + np.array(t_rot)

    q_out = quaternion_multiply(qa, qb)

    out.transform.translation.x = float(t_out[0])
    out.transform.translation.y = float(t_out[1])
    out.transform.translation.z = float(t_out[2])
    out.transform.rotation = list_to_quat(q_out)
    return out


# =======================
# ---- Main ROS Node ----
# =======================
class OdometryTFPublisher(Node):
    """
    将 FAST-LIO2 的 odom（camera_init → sensor_frame）
    迁移为（fast_lio2_odom → base_link），并发布新的 Odometry + TF。
    同时对 twist 和 pose 协方差做坐标旋转，避免信息丢失/方向错乱。
    """

    def __init__(self):
        super().__init__("odometry_tf_publisher")

        # ---- 参数区 ----
        self.declare_parameter('input_odom_topic', '/fast_lio2/Odometry')
        self.declare_parameter('output_odom_topic', '/fast_lio2/Odometry_base_link')
        self.declare_parameter('odom_frame', 'fast_lio2_odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('sensor_frame', 'body')  # FAST-LIO2 的 child（传感器/惯导/雷达固连帧）
        self.declare_parameter('timeout_seconds', 0.2)
        self.declare_parameter('preserve_input_timestamp', True)
        self.declare_parameter('rotate_twist', True)
        self.declare_parameter('rotate_covariance', True)

        self.input_odom_topic = self.get_parameter('input_odom_topic').get_parameter_value().string_value
        self.output_odom_topic = self.get_parameter('output_odom_topic').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.sensor_frame = self.get_parameter('sensor_frame').get_parameter_value().string_value
        self.timeout_seconds = self.get_parameter('timeout_seconds').get_parameter_value().double_value
        self.preserve_input_timestamp = self.get_parameter('preserve_input_timestamp').get_parameter_value().bool_value
        self.rotate_twist_flag = self.get_parameter('rotate_twist').get_parameter_value().bool_value
        self.rotate_cov_flag = self.get_parameter('rotate_covariance').get_parameter_value().bool_value

        # ---- TF buffer/listener ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- QoS 设置（可按需调整）----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        pub_qos = QoSProfile(depth=10)

        # ---- Pub/Sub ----
        self.odom_pub = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)
        self.odom_subscriber = self.create_subscription(Odometry, self.input_odom_topic, self.odom_callback, sensor_qos)

        # ---- TF Broadcaster ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- 获取静态外参：sensor_frame → base_link ----
        self.get_logger().info(f"Waiting for static TF transform {self.sensor_frame} -> {self.base_frame} ...")
        while rclpy.ok():
            try:
                # 注意：lookup_transform(target, source, time) 返回的是 target<-source 的 TransformStamped，
                # 在 ROS2 的实现里，这个 TransformStamped 的 header.frame_id=target, child_frame_id=source，
                # transform 表示 “source 在 target 下”的位姿（也常被视作 target<-source）。
                # 为了右乘 (odom->sensor) * (sensor->base)，我们要的就是 sensor->base，
                # 因此使用 target=self.sensor_frame, source=self.base_frame。
                self.sensor_to_base = self.tf_buffer.lookup_transform(
                    self.sensor_frame, self.base_frame, Time(), timeout=Duration(seconds=self.timeout_seconds)
                )
                self.get_logger().info(f"Static TF ready: {self.sensor_frame} -> {self.base_frame}")
                break
            except Exception as e:
                self.get_logger().warn(f"Waiting for static TF transform: {e}")
                rclpy.spin_once(self, timeout_sec=0.1)

        # 预缓存外参的向量与四元数/旋转矩阵（回调中复用，降开销）
        self.t_sb = np.array([self.sensor_to_base.transform.translation.x,
                              self.sensor_to_base.transform.translation.y,
                              self.sensor_to_base.transform.translation.z])
        self.q_sb = quat_to_list(self.sensor_to_base.transform.rotation)
        self.R_sb = quat_to_rot(self.q_sb)

        self.get_logger().info(f"zhijie output {self.t_sb}")

        self.get_logger().info(
            "OdometryTFPublisher initialized:\n"
            f"  Input topic : {self.input_odom_topic}\n"
            f"  Output topic: {self.output_odom_topic}\n"
            f"  Frames      : {self.odom_frame} -> {self.base_frame} (via {self.sensor_frame})"
        )

    def odom_callback(self, msg: Odometry):
        try:
            # 1) 把上游 odom 里的位姿当作刚体变换：T_os = odom(=msg.header.frame_id) → sensor_frame
            T_os = TransformStamped()
            T_os.header = msg.header
            T_os.child_frame_id = self.sensor_frame
            T_os.transform.translation.x = msg.pose.pose.position.x
            T_os.transform.translation.y = msg.pose.pose.position.y
            T_os.transform.translation.z = msg.pose.pose.position.z
            T_os.transform.rotation = msg.pose.pose.orientation

            # 2) 右乘静态外参：T_ob = T_os * T_sb  (odom → base_link)
            #    注意这里我们直接使用 self.sensor_to_base（sensor->base），
            #    与上一步的 child 对接一致
            T_ob = compose_transforms(T_os, self.sensor_to_base)

            # 3) 构造新的 Odometry，保留时间戳（或用 now）
            new_odom = Odometry()
            new_odom.header.frame_id = self.odom_frame
            new_odom.header.stamp = msg.header.stamp if self.preserve_input_timestamp else self.get_clock().now().to_msg()
            new_odom.child_frame_id = self.base_frame

            # Pose
            new_odom.pose.pose.position.x = T_ob.transform.translation.x
            new_odom.pose.pose.position.y = T_ob.transform.translation.y
            new_odom.pose.pose.position.z = T_ob.transform.translation.z
            new_odom.pose.pose.orientation = T_ob.transform.rotation
            normalize_quat_inplace(new_odom.pose.pose.orientation)

            # 4) 协方差：对 6×6 进行旋转（不引入平移），避免信息方向错乱
            if self.rotate_cov_flag:
                try:
                    cov = np.array(msg.pose.covariance).reshape(6, 6)
                    Ad = adjoint_of_rotation(self.R_sb)  # 纯旋转的 Adjoint
                    cov_bl = Ad @ cov @ Ad.T
                    new_odom.pose.covariance = cov_bl.reshape(36).tolist()
                except Exception:
                    # 如果上游没填满 36 项/维度错等，保底沿用原值
                    new_odom.pose.covariance = msg.pose.covariance
            else:
                new_odom.pose.covariance = msg.pose.covariance

            # 5) Twist：从 sensor_frame 旋到 base_link（线/角速度）
            if self.rotate_twist_flag:
                vb = msg.twist.twist.linear
                wb = msg.twist.twist.angular
                vx, vy, vz = rotate_vec3_by_quat(vb.x, vb.y, vb.z, self.q_sb)
                wx, wy, wz = rotate_vec3_by_quat(wb.x, wb.y, wb.z, self.q_sb)
                new_odom.twist.twist.linear.x = vx
                new_odom.twist.twist.linear.y = vy
                new_odom.twist.twist.linear.z = vz
                new_odom.twist.twist.angular.x = wx
                new_odom.twist.twist.angular.y = wy
                new_odom.twist.twist.angular.z = wz
            else:
                new_odom.twist = msg.twist  # 原样拷贝（若 sensor 与 base 未对齐，会方向错乱）

            # 6) 发布新的里程计
            self.odom_pub.publish(new_odom)

            # 7) 广播 TF：odom_frame → base_link（与 new_odom 对齐）
            tf_msg = TransformStamped()
            tf_msg.header.stamp = new_odom.header.stamp
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame
            tf_msg.transform.translation.x = new_odom.pose.pose.position.x
            tf_msg.transform.translation.y = new_odom.pose.pose.position.y
            tf_msg.transform.translation.z = new_odom.pose.pose.position.z
            tf_msg.transform.rotation = new_odom.pose.pose.orientation
            normalize_quat_inplace(tf_msg.transform.rotation)

            self.tf_broadcaster.sendTransform(tf_msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to transform odometry: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = OdometryTFPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
