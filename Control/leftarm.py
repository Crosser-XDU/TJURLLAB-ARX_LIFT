#!/usr/bin/env python3
import rclpy                             # ROS2 Python 客户端库
from rclpy.node import Node              # 基类：ROS2 节点
from arx5_arm_msg.msg._robot_cmd import RobotCmd # 控制命令
from arx5_arm_msg.msg._robot_status import RobotStatus #状态消息类型
import time
class LeftArmEEF4ControlNode(Node):
    def __init__(self):
        super().__init__('left_arm_eef4control')      # 初始化节点名
        # 发布 RobotCmd 到 arm_cmd 话题
        self.cmd_pub = self.create_publisher(RobotCmd, 'arm_cmd', 1)
        # 订阅 arm_status 话题，查看本体状态反馈
        self.status_sub = self.create_subscription(
            RobotStatus, 'arm_status', self.on_status, 10)


        print(rclpy.ok())
        # 为啥要sleep，你得保证订阅者链接成功
        time.sleep(2)
        print(self.cmd_pub.get_subscription_count())

        # 动作之后要sleep一下为啥，因为动作需要时间完成，也就是说如果你把sleep拉太小，动作可能没完成就下一个动作了
        self.send_first()
        time.sleep(5)
        
        # 最极端就是sleep(0),这样的话基本上就是发完命令立马发下一个命令，你会发现只有最后一个命令生效了
        self.send_second()
        time.sleep(5)

        self.send_third()
        time.sleep(5)

        self.send_home()

        
        
    def send_first(self):
        cmd = RobotCmd()
        cmd.mode = 4  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        cmd.end_pos = [0.1]*6  # 末端目标 x,y,z,roll,pitch,yaw
        # cmd.joint_pos = [0.0]*6                        # 关节目标（此处未用，填 0）
        cmd.gripper = -0.5                           # 夹爪开合
        self.cmd_pub.publish(cmd)                     # 发布命令
        self.get_logger().info(f'Sent RobotCmd: {cmd}')
        
    def send_second(self):
        cmd = RobotCmd()
        cmd.mode = 4  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        cmd.end_pos = [0.12]*6 # 末端目标 x,y,z,roll,pitch,yaw
        # cmd.joint_pos = [0.3, -0.2, 0.1, 0.1, 0.1, -0.1]  # 关节目标
        cmd.gripper = -1.5                           # 夹爪开合
        self.cmd_pub.publish(cmd)                     # 发布命令
        self.get_logger().info(f'Sent RobotCmd: {cmd}')

    def send_third(self):
        cmd = RobotCmd()
        cmd.mode = 4  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        cmd.end_pos = [0.14]*6 # 末端目标 x,y,z,roll,pitch,yaw
        # cmd.joint_pos = [0.0]*6                        # 关节目标（此处未用，填 0）
        cmd.gripper = -3.0                            # 夹爪开合
        self.cmd_pub.publish(cmd)                     # 发布命令
        self.get_logger().info(f'Sent RobotCmd: {cmd}')
    
    def send_home(self):
        cmd = RobotCmd()
        cmd.mode = 1  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        cmd.end_pos = [0.18]*6  # 末端目标 x,y,z,roll,pitch,yaw
        # cmd.joint_pos = [0.0]*6                        # 关节目标（此处未用，填 0）
        cmd.gripper = 0.3                             # 夹爪开合
        self.cmd_pub.publish(cmd)                     # 发布命令
        self.get_logger().info(f'Sent RobotCmd: {cmd}')

    def on_status(self, msg: RobotStatus):
        # 回调：收到状态时打印末端、关节、夹爪（关节数组第 7 个为夹爪）
        self.get_logger().info(
            f'Recv status end_pos={list(msg.end_pos)}\n '
            f'joint_pos={list(msg.joint_pos)}\n '
            f'gripper={msg.joint_pos[6]}\n'
        )

def main():
    rclpy.init()          # 初始化 ROS2
    node = LeftArmEEF4ControlNode()
    try:
        rclpy.spin(node)  # 进入事件循环
    finally:
        node.destroy_node()
        rclpy.shutdown()  # 退出 ROS2

if __name__ == '__main__':
    main()
