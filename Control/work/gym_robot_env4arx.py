from typing import Any, Dict, Tuple, Literal, Iterable, Optional

import numpy as np
import rclpy
from gym import spaces
from arx5_arm_msg.msg._robot_cmd import RobotCmd  # 控制命令
from arx5_arm_msg.msg._robot_status import RobotStatus  # 状态消息
from Control.work.arx_utils import *
from gym_robot_env import BaseRobotEnv
import time


class ARXRobotEnv():
    """
    Concrete implementation of BaseRobotEnv for a single-arm ARX robot.

    Action Space :
    abs/delta x, y, z, roll, pitch, yaw, and gripper opening
    y正左，x正前，z正,单位米;
    roll/pitch/yaw，单位弧度;
    joint 关节，单位弧度
    Shape: (7,)
    """

    def __init__(self, control_type: Literal["end", "joint"] = "end", control_mode: Literal["abs", "delta"] = "abs", freq: int = 20,
                 camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h"),
                 dir: Optional[str] = None, target_size: Optional[Tuple[int, int]] = (224, 224)):
        super().__init__()
        self._setup_spaces()
        self.control_type = control_type  # end or joint控制
        self.control_mode = control_mode  # abs or delta控制
        self.camera_view = camera_view  # 使用哪些相机
        self.camera_type = camera_type  # 相机类型
        self.dir = dir  # 保存执行demo路径（图片，#TODO 视频）
        self.target_size = target_size  # 保存图片大小
        self.freq = freq  # HZ

    def _setup_spaces(self):
        """
        Configure Action and Observation spaces for ARX single-arm robot.
        """

        # 1. Action Space: Delta x, y, z, roll, pitch, yaw, and gripper opening
        # Shape: (7,)
        # TODO 问下客服end是给范围还是得IK解到joint去比（py有接口。。。）
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # 2. Observation Space:
        # Two cameras: [camera_0, camera_1]
        # x, y, z, roll, pitch, yaw, gripper opening
        # Shape: (14,)
        observation_space = dict()

        # end末端位姿的6个自由度：x, y, z, roll, pitch, yaw
        observation_space["end_pos"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # 各个关节的可用范围（同厂家给出的软件范围），比物理限位要小一点，第7个就是对应夹爪
        joint_low = np.array([-2.62, -0.1, -0.1, -1.29, -1.48, -1.74, -3.4])
        joint_high = np.array([2.62, 3.6, 3.0, 1.29, 1.48, 1.74, 0.1])
        observation_space["joint_pos"] = spaces.Box(
            low=joint_low, high=joint_high, shape=(7,), dtype=np.float32)

        # 各个关节的力矩
        observation_space["joint_cur"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # 各个关节的速度
        observation_space["joint_vel"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # 左右中的相机彩色图像
        observation_space["camera_l_color"] = spaces.Box(
            low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        # observation_space["camera_r_color"] = spaces.Box(
        #     low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        observation_space["camera_h_color"] = spaces.Box(
            low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        observation_space["camera_l_depth"] = spaces.Box(
            low=0, high=65535, shape=(224, 224, 1), dtype=np.uint16)
        # observation_space["camera_r_depth"] = spaces.Box(
        #     low=0, high=65535, shape=(224, 224, 1), dtype=np.uint16)
        observation_space["camera_h_depth"] = spaces.Box(
            low=0, high=65535, shape=(224, 224, 1), dtype=np.uint16)

        self.observation_space = spaces.Dict(observation_space)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the robot to the initial pose.

        Returns:
            observation (Dict[str, np.ndarray]): The initial state of the robot.
        """

        # 1. Enable the robot（拉起节点后已经enbale）
        success, error_message = self._enable_robot()
        if not success:
            raise RuntimeError(f"Failed to enable the robot: {error_message}")

        # 2. Go to the initial pose
        success, error_message = self._go_to_initial_pose()
        if not success:
            raise RuntimeError(
                f"Failed to go to the initial pose: {error_message}")

        # 3. Get the initial observation
        obs = self._get_observation()
        # 返回观测
        return obs

    def step(self, action: np.ndarray):
        """
        Execute one time step in the environment.

        Args:
            action (np.ndarray): Action provided by the agent. Shape (action_dim,).

        Returns:
            observation (np.ndarray): New state of the robot.
            reward (float): Scalar reward value.
            done (bool): Whether the episode has ended.
            info (dict): Diagnostic information.
        """

        # 1. Apply action
        success, error_message = self._apply_action(action)
        if not success:
            raise RuntimeError(f"Failed to apply action: {error_message}")

        # 2. State Observation
        # Retrieve latest data from ROS topics
        obs = self._get_observation()  # 这个就是下一个step推理动作的观测

        # TODO 我先留空（仿真才能拿到应该）
        reward = 0.0
        is_done = False
        info = dict()
        # 3. Reward Calculation
        # reward = self._get_reward(obs, action)

        # 4. Termination Logic
        # is_done = self._get_termination(obs, action)

        # 5. Metadata（这个拿啥？）
        # info = self._get_info()

        return obs, reward, is_done, info

    def close(self):
        """
        Clean up resources and shut down ROS nodes.
        """

        # 1. Go to initial pose
        success, error_message = self._go_to_initial_pose()
        if not success:
            raise RuntimeError(
                f"Failed to go to the initial pose: {error_message}")

        # 2. Disable the robot

        success, error_message = self._disable_robot()
        if not success:
            raise RuntimeError(f"Failed to disable the robot: {error_message}")

    # Enable和disable应该更像拉起对应控制节点和关闭对应控制节点而不是现在的io节点 #TODO shell的哪些命令，拉起和挂掉能不能写进来

    def _enable_robot(self) -> Tuple[bool, str | None]:
        """
        Enables the robot.
        挂起一个通讯节点，env中会利用它发送控制信号，收回本体信息

        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is enabled, (False, error_message) otherwise.
        """
        rclpy.init()  # 也放外面统一管理吧
        self.node, self.executor = start_robot_io(
            self.camera_type, self.camera_view)
        if self.node and self.executor:
            return (True, None)
        erorr = []
        if not self.node:
            erorr.append("控制io节点挂起失败")
        if not self.executor:
            erorr.append("控制io节点读写锁初始化失败")
        if not rclpy.ok():
            erorr.append("ROS2开启失败")
        return (False, "原因：".join(erorr) if erorr else "其他错误")

    def _disable_robot(self) -> Tuple[bool, str | None]:
        """
        Disables the robot.
        销毁通讯节点
        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is disabled, (False, error_message) otherwise.
        """
        self.node.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()  # TODO check一下有没有关闭所有ros2窗口能力

        if self.node is None and self.executor is None:
            return (True, None)
        erorr = []
        if self.node:
            erorr.append("控制io节点销毁失败")
        if self.executor:
            erorr.append("控制io节点读写锁销毁失败")
        if rclpy.ok():
            erorr.append("ROS2关闭失败")
        return (False, "原因：".join(erorr) if erorr else "其他错误")

    def _success_check(self, target: np.ndarray) -> Tuple[bool, str | None]:
        """
        Check if the robot has reached the target within a threshold after executing the action.
        Args:
            target (np.array): The target position to reach. 
        """
        threshold_xyz = 0.01  # 1cm
        threshold_rpy = 0.05   # 约2.86度
        threshold_joint = 0.05  # 关节空间阈值 #TODO 这里的关节角度，和上面的欧拉角比，需要限制的更严格吗，因为转动幅度没有上面大
        threshold_gripper = 0.01  # TODO 这里依旧是电机转动角度呢，还是映射到张开的长度呢
        # 取arm_status的最新一条
        current_status = self.node.get_robot_status()

        if self.control_type == "end":
            curr_end_xyz = current_status.end_pos[:3]
            curr_end_rpy = current_status.end_pos[3:]
            curr_gripper = current_status.joint_pos[6]
            # 计算误差
            xyz_diff = np.linalg.norm(curr_end_xyz - target[:3])
            rpy_diff = np.linalg.norm(curr_end_rpy - target[3:])
            gripper_diff = abs(curr_gripper - target[6])

            if xyz_diff <= threshold_xyz and rpy_diff <= threshold_rpy and gripper_diff <= threshold_gripper:
                return True, None
            reasons = []
            if xyz_diff > threshold_xyz:
                reasons.append(f"位置误差 {xyz_diff:.4f} 超阈值 {threshold_xyz}")
            if rpy_diff > threshold_rpy:
                reasons.append(f"欧拉角误差 {rpy_diff:.4f} 超阈值 {threshold_rpy}")
            if gripper_diff > threshold_gripper:
                reasons.append(
                    f"夹爪误差 {gripper_diff:.4f} 超阈值 {threshold_gripper}")
            return False, "原因：".join(reasons) if reasons else "其他错误"

        else:  # joint pos模式
            curr_joint = current_status.joint_pos[:6]
            curr_gripper = current_status.joint_pos[6]
            # 计算误差
            joint_diff = np.linalg.norm(curr_joint - target[:6])
            gripper_diff = abs(curr_gripper - target[6])

            if joint_diff <= threshold_joint and gripper_diff <= threshold_gripper:
                return True, None
            reasons = []
            if joint_diff > threshold_joint:
                reasons.append(f"关节误差 {joint_diff:.4f} 超阈值 {threshold_joint}")
            if gripper_diff > threshold_gripper:
                reasons.append(
                    f"夹爪误差 {gripper_diff:.4f} 超阈值 {threshold_gripper}")
            return False, "原因：".join(reasons) if reasons else "其他错误"

    def _go_to_initial_pose(self) -> Tuple[bool, str | None]:
        """
        Goes to the initial pose(home pose).

        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is at the initial pose, (False, error_message) otherwise.
        """
        cmd = RobotCmd()
        cmd.mode = 1
        self.node.send_control_msg(cmd)
        time.sleep(5)
        success, error_message = self._success_check([0, 0, 0, 0, 0, 0, 0])
        if not success:
            return (False, error_message)
        else:
            return (True, None)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Polls ROS topics for sensor data.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the observation.
        """
        obs = dict()
        beigin = time.time()
        status = self.node.get_robot_status()
        if not status:
            print(f"状态获取失败")
            return None

        # 1. end末端位姿的6个自由度：x, y, z, roll, pitch, yaw
        if status.end_pos and self.observation_space["end_pos"].contains(np.array(status.end_pos, dtype=np.float32)):
            obs["end_pos"] = status.end_pos
        else:
            print(f"end_pos获取失败")
        # 2. 各个关节的位置
        if status.joint_pos and self.observation_space["joint_pos"].contains(np.array(status.joint_pos, dtype=np.float32)):
            obs["joint_pos"] = status.joint_pos
        else:
            print(f"joint_pos获取失败")
        # 3. 各个关节的力矩
        if status.joint_cur and self.observation_space["joint_cur"].contains(np.array(status.joint_cur, dtype=np.float32)):
            obs["joint_cur"] = status.joint_cur
        else:
            print(f"joint_cur获取失败")
        # 4. 各个关节的速度
        if status.joint_vel and self.observation_space["joint_vel"].contains(np.array(status.joint_vel, dtype=np.float32)):
            obs["joint_vel"] = status.joint_vel
        else:
            print(f"joint_vel获取失败")

        # 5. 相机图像
        camera_all = self.node.get_camera(self.dir, self.target_size)
        end = time.time()
        # TODO check一下会不会通信要花很久，然后考虑要不要camera和status topic放一起近似时间拿
        print(f"4debug！！获取本体信息的时间间隙：{end-beigin}")

        cam_label = ["camera_l_color", "camera_l_depth",
                     "camera_h_color", "camera_h_depth"]
        for key in cam_label:
            img = camera_all.get(key)
            if img is None:
                print(f"{img}获取失败")
                continue
            if 'color' in key:
                img = np.asarray(img, dtype=np.uint8)
            else:
                img = np.asarray(img, dtype=np.uint16)
            
            space = self.observation_space[key]
            if space.contains(img):
                obs[key] = img

        return obs

    def _apply_action(self, action: np.ndarray) -> Tuple[bool, str | None]:
        """
        apply to the robot single step control command. # TODO 接入motion planning算法或者是进行一个简单的增加中间step（插值？）
        - Args:
            action: the action provieded by the agent（step level）. 
            end: x,y,z,roll,pitch,yaw,gripper
            joint: joint0,joint1,...,joint5,gripper

        """
        msg = RobotCmd()
        target = None
        if self.control_type == "end":
            msg.mode = 4
            target = action
            if self.control_mode == "delta":
                curr = self._get_observation()
                curr_end = np.concatenate(
                    (curr["end_pos"], curr["joint_pos"][6:]), axis=None, dtype=np.float32)
                target = curr_end + action
            msg.end_pos = target[:6].tolist()
            msg.gripper = float(target[6])
            if self.action_space.contains(target):
                begin = time.time()
                self.node.send_control_msg(msg)
                dur = time.time() - begin
                if dur < 1 / self.freq:
                    time.sleep(1 / self.freq - dur)
                success, error_message = self._success_check(target)
                if not success:
                    return (False, error_message)
                return (True, None)
            return (False, "end动作超出定义")
        else:  # joint控制模式
            msg.mode = 5
            target = action
            if self.control_mode == "delta":
                curr = self._get_observation()
                target = curr["joint_pos"] + action
            msg.joint_pos = target[:6].tolist()
            msg.gripper = float(target[6])
            if self.observation_space["joint_pos"].contains(target):
                begin = time.time()
                self.node.send_control_msg(msg)
                dur = time.time() - begin
                if dur < 1 / self.freq:
                    time.sleep(1 / self.freq - dur)
                success, error_message = self._success_check(target)
                if not success:
                    return (False, error_message)
                return (True, None)
            return (False, "joint动作超出定义")
