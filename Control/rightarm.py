#!/usr/bin/env python3
import rclpy                             # ROS2 Python 客户端库
from rclpy.node import Node              # 基类：ROS2 节点
from arx5_arm_msg.msg import RobotCmd, RobotStatus  # 控制命令与状态消息类型

class CmdAndMonitor(Node):
    def __init__(self):
        super().__init__('cmd_and_monitor')      # 初始化节点名
        # 发布 RobotCmd 到 arm_cmd 话题（默认 X5 normal 模式订阅）
        self.cmd_pub = self.create_publisher(RobotCmd, 'arm_cmd', 10)
        # 订阅 arm_status 话题，查看本体状态反馈
        self.status_sub = self.create_subscription(
            RobotStatus, 'arm_status', self.on_status, 10)
        self.sent = False                        # 标记只发一条命令
        # 每 0.1s 调用一次 send_once（只会执行一次）
        self.create_timer(0.1, self.send_once)

    def send_once(self):
        if self.sent:
            return
        cmd = RobotCmd()
        cmd.mode = 4  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        cmd.end_pos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 末端目标 x,y,z,roll,pitch,yaw
        cmd.joint_pos = [0.0]*6                        # 关节目标（此处未用，填 0）
        cmd.gripper = 0.0                             # 夹爪开合
        self.cmd_pub.publish(cmd)                     # 发布命令
        self.get_logger().info(f'Sent RobotCmd: {cmd}')
        self.sent = True                              # 只发一次

    def on_status(self, msg: RobotStatus):
        # 回调：收到状态时打印末端、关节、夹爪（关节数组第 7 个为夹爪）
        self.get_logger().info(
            f'Recv status end_pos={list(msg.end_pos)} '
            f'joint_pos={list(msg.joint_pos)} gripper={msg.joint_pos[6]}'
        )

def main():
    rclpy.init()          # 初始化 ROS2
    node = CmdAndMonitor()
    try:
        rclpy.spin(node)  # 进入事件循环
    finally:
        node.destroy_node()
        rclpy.shutdown()  # 退出 ROS2

if __name__ == '__main__':
    main()
