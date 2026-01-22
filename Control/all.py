#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arx5_arm_msg.msg._robot_status import RobotStatus
from arm_control.msg._pos_cmd import PosCmd


class WholeBodyControlNode(Node):
    def __init__(self):
        super().__init__('whole_body_control')

        self.left_cmd_pub = self.create_publisher(RobotCmd, 'arm_cmd_l', 1)
        self.right_cmd_pub = self.create_publisher(RobotCmd, 'arm_cmd_r', 1)
        self.base_cmd_pub = self.create_publisher(PosCmd, '/ARX_VR_L', 1)

        self.left_status_sub = self.create_subscription(
            RobotStatus, 'arm_status_l', self.on_left_status, 10)
        self.right_status_sub = self.create_subscription(
            RobotStatus, 'arm_status_r', self.on_right_status, 10)
        self.base_status_sub = self.create_subscription(
            PosCmd, '/body_information', self.on_base_status, 10)

        time.sleep(2)
        self.get_logger().info(
            f"Left subs: {self.left_cmd_pub.get_subscription_count()}, "
            f"Right subs: {self.right_cmd_pub.get_subscription_count()}"
        )
        for _ in range(3):
            self.send_first()
            time.sleep(2)

            self.send_second()
            time.sleep(2)

            self.send_third()
            time.sleep(2)

        self.send_home()

    def publish_all(self, left_cmd: RobotCmd, right_cmd: RobotCmd, base_cmd: PosCmd) -> None:
        self.left_cmd_pub.publish(left_cmd)
        self.right_cmd_pub.publish(right_cmd)
        self.base_cmd_pub.publish(base_cmd)

    def send_first(self):
        left = RobotCmd()
        left.end_pos = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        left.gripper = -1.0
        left.mode = 4

        right = RobotCmd()
        right.end_pos = [0.05, -0.05, 0.05, -0.05,-0.05, -0.05]
        right.gripper = -1.0
        right.mode = 4

        base = PosCmd()
        base.chx = 0.0
        base.chy = 0.0
        base.chz = 0.25
        base.height = 3.0
        base.mode1 = 1

        self.publish_all(left, right, base)
        self.get_logger().info("Sent first step (arms + base).")

    def send_second(self):
        left = RobotCmd()
        left.end_pos = [0.055, 0.055, 0.055, 0.055, 0.055, 0.055]
        left.gripper = -1.5
        left.mode = 4

        right = RobotCmd()
        right.end_pos = [0.055, -0.055, 0.055, -0.055, -0.055, -0.055]
        right.gripper = -1.5
        right.mode = 4

        base = PosCmd()
        base.chx = 0.0
        base.chy = 0.0
        base.chz = 0.25
        base.height = 6.0
        base.mode1 = 1
        self.publish_all(left, right, base)
        self.get_logger().info("Sent second step (arms + base).")

    def send_third(self):
        left = RobotCmd()
        left.end_pos = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
        left.gripper = -2.0
        left.mode = 4

        right = RobotCmd()
        right.end_pos = [0.06,-0.06, 0.06, -0.06, -0.06, -0.06]
        right.gripper = -2.0
        right.mode = 4

        base = PosCmd()
        base.chx = 0.0
        base.chy = 0.0
        base.chz = 0.25
        base.height = 3.0
        base.mode1 = 1

        self.publish_all(left, right, base)
        self.get_logger().info("Sent second step (arms + base).")

    def send_home(self):
        left = RobotCmd()
        left.mode = 1

        right = RobotCmd()
        right.mode = 1

        base = PosCmd()
        base.mode1 = 2

        self.publish_all(left, right, base)
        self.get_logger().info("Sent home step (arms + base).")

    def on_left_status(self, msg: RobotStatus):
        self.get_logger().info(
            f"[L] end_pos={list(msg.end_pos)} "
            f"joint_pos={list(msg.joint_pos)} "
            f"gripper={msg.joint_pos[6]}"
        )

    def on_right_status(self, msg: RobotStatus):
        self.get_logger().info(
            f"[R] end_pos={list(msg.end_pos)} "
            f"joint_pos={list(msg.joint_pos)} "
            f"gripper={msg.joint_pos[6]}"
        )

    def on_base_status(self, msg: PosCmd):
        self.get_logger().info(
            f"[B] height={msg.height:.3f} "
            f"head_pit={msg.head_pit:.3f} head_yaw={msg.head_yaw:.3f}"
        )


def main():
    rclpy.init()
    node = WholeBodyControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
