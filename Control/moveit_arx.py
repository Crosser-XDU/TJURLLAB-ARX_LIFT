#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory

from arx5_arm_msg.msg import RobotCmd


class MoveItToARXBridge(Node):
    """
    MoveIt 规划 → 厂家单帧 joint 指令
    """

    def __init__(self, arm_mode: str = "right"):
        super().__init__('moveit_to_arx_bridge')

        # ===== 参数 =====
        self.declare_parameter('send_rate_hz', 20.0)
        self.send_rate = self.get_parameter('send_rate_hz').value
        # arm_mode 优先用命令行传入，否则用参数服务器
        self.declare_parameter('arm_mode', arm_mode)  # right / left / double
        self.arm_mode = str(self.get_parameter('arm_mode').value).lower()
        if self.arm_mode not in {'left', 'right', 'double'}:
            self.get_logger().warn(f"arm_mode={self.arm_mode} 无效，改用 right")
            self.arm_mode = 'right'
        self.dt = 1.0 / self.send_rate

        # ===== publisher =====
        self.left_pub = self.create_publisher(RobotCmd, 'arm_cmd_l', 5)
        self.right_pub = self.create_publisher(RobotCmd, 'arm_cmd_r', 5)

        # ===== subscriber =====
        self.create_subscription(
            DisplayTrajectory,
            '/display_planned_path',
            self.on_display_trajectory,
            20
        )

        self.get_logger().info(
            f"MoveIt → ARX bridge started (mode={self.arm_mode}, rate={self.send_rate} Hz)"
        )

        self._lock = threading.Lock()
        self._executing = False

    # ===================== callback =====================
    def on_display_trajectory(self, msg: DisplayTrajectory):
        """
        RViz / MoveIt 发出来的规划结果
        """
        if self._executing:
            self.get_logger().warn("Still executing previous trajectory, skip")
            return

        if not msg.trajectory:
            self.get_logger().warn("Empty trajectory")
            return

        traj: JointTrajectory = msg.trajectory[0].joint_trajectory

        if len(traj.points) == 0:
            self.get_logger().warn("Trajectory has no points")
            return

        self.get_logger().info(
            f"Received trajectory: {len(traj.points)} points"
        )

        threading.Thread(
            target=self.execute_trajectory,
            args=(traj,),
            daemon=True
        ).start()

    # ===================== execution =====================
    def execute_trajectory(self, traj: JointTrajectory):
        with self._lock:
            self._executing = True

        try:
            for idx, point in enumerate(traj.points):
                positions = list(point.positions)
                # ===== joint 拆分 =====
                if self.arm_mode == 'left':
                    if len(positions) < 6:
                        self.get_logger().warn("轨迹长度<6，无法拆左臂，跳过该点")
                        continue
                    left_joints = positions[0:6]
                    left_cmd = RobotCmd()
                    left_cmd.mode = 5
                    left_cmd.joint_pos = left_joints
                    left_cmd.gripper = -0.0
                    self.left_pub.publish(left_cmd)
                    self.get_logger().debug(f"[{idx}] L={left_joints}")
                elif self.arm_mode == 'right':
                    if len(positions) >= 12:
                        right_joints = positions[6:12]
                    elif len(positions) >= 6:
                        right_joints = positions[0:6]
                    else:
                        self.get_logger().warn("轨迹长度<6，无法拆右臂，跳过该点")
                        continue
                    right_cmd = RobotCmd()
                    right_cmd.mode = 5
                    right_cmd.joint_pos = right_joints
                    right_cmd.gripper = -1.5
                    self.right_pub.publish(right_cmd)
                    self.get_logger().debug(f"[{idx}] R={right_joints}")
                else:  # double
                    if len(positions) < 12:
                        self.get_logger().warn("轨迹长度<12，无法拆双臂，跳过该点")
                        continue
                    left_joints = positions[0:6]
                    right_joints = positions[6:12]
                    left_cmd = RobotCmd()
                    left_cmd.mode = 5
                    left_cmd.joint_pos = left_joints
                    left_cmd.gripper = -0.0
                    right_cmd = RobotCmd()
                    right_cmd.mode = 5
                    right_cmd.joint_pos = right_joints
                    right_cmd.gripper = -1.5
                    self.left_pub.publish(left_cmd)
                    self.right_pub.publish(right_cmd)
                    self.get_logger().debug(
                        f"[{idx}] L={left_joints} R={right_joints}")

                time.sleep(self.dt)

            self.get_logger().info("Trajectory execution finished")

        finally:
            with self._lock:
                self._executing = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MoveIt → ARX 轨迹桥接")
    parser.add_argument(
        "--arm",
        choices=["left", "right", "double"],
        default="right",
        help="选择拆分/下发的手臂，默认 right",
    )
    args = parser.parse_args()

    rclpy.init()
    node = MoveItToARXBridge(arm_mode=args.arm)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
