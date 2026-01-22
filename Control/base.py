#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from arm_control.msg._pos_cmd import PosCmd


class BaseControlNode(Node):
    def __init__(self):
        super().__init__('base_control')

        self.cmd_pub = self.create_publisher(PosCmd, '/ARX_VR_L', 1)

        self.status_sub = self.create_subscription(
            PosCmd, '/body_information', self.on_status, 10)

        time.sleep(2.0)
        self.get_logger().info(
            f"Subscribers on /ARX_VR_L: {self.cmd_pub.get_subscription_count()}"
        )

        self.send_move()
        time.sleep(5.0)
        self.send_stop()

    def send_move(self):
        msg = PosCmd()
        # msg.chx = -1.0 # 约为0.24 m/s
        msg.chy = 0.0
        msg.chz = 0.5
        # msg.temp_float_data = [0.01]*4
        msg.height = 5.0
        msg.mode1 = 1
        self.cmd_pub.publish(msg)
        self.get_logger().info("Sent base move command.")

    def send_stop(self):
        msg = PosCmd()
        # msg.chx = 2.5
        msg.chy = 0.0
        msg.chz = 0.0
        msg.height = 0.0
        msg.mode1 = 2
        self.cmd_pub.publish(msg)
        self.get_logger().info("Sent base stop command.")

    def on_status(self, msg: PosCmd):
        self.get_logger().info(
            f"body_information height={msg.height:.3f} "
            f"head_pit={msg.head_pit:.3f} head_yaw={msg.head_yaw:.3f}"
        )


def main():
    rclpy.init()
    node = BaseControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
