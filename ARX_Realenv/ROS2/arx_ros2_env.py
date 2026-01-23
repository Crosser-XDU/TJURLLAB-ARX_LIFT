from typing import Any, Dict, Tuple, Iterable, Literal, Optional
import numpy as np
import subprocess
import time
from pathlib import Path
import shlex
import rclpy
from arx_ros2_env_utils import start_robot_io, success_check, interpolate_action

# 本体控制相关的msg类
from arx5_arm_msg.msg._robot_cmd import RobotCmd  # 双臂控制命令
from arx5_arm_msg.msg._robot_status import RobotStatus  # 状态消息
from arm_control.msg._pos_cmd import PosCmd  # 底盘控制命令


WORKSPACE = Path(__file__).resolve().parent
CONTROL_DIR = Path(__file__).resolve().parents[3] / "Control"


def open_terminal(title: str, cmd: str, use_sudo: bool = False, cwd: Path | None = None):
    """Launch a gnome-terminal tab/window running the given command."""
    base_path = cwd if cwd is not None else WORKSPACE
    cd_workspace = f"cd {shlex.quote(str(base_path))}"
    full_cmd = f"{cd_workspace}; {cmd}; exec bash"

    base = ["gnome-terminal", "-t", title, "-x"]
    if use_sudo:
        base.append("sudo")
    base += ["bash", "-c", full_cmd]
    subprocess.Popen(base)


class ARXRobotEnv():
    """
    Concrete implementation of BasetEnv for ARX robot.
    """

    def __init__(self, duration_per_step: float = 0.02, min_steps_per_action: int = 2,
                 max_v_xyz: float = 0.05, max_v_rpy: float = 0.3,
                 camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h", "camera_r"),
                 dir: Optional[str] = None, img_size: Optional[Tuple[int, int]] = (224, 224),
                 min_steps_gripper: int = 10):
        super().__init__()
        self.camera_view = camera_view  # 使用哪些相机
        self.camera_type = camera_type  # 相机类型
        self.dir = dir  # 保存执行demo路径（图片，#TODO 视频）
        self.img_size = img_size  # 保存图片大小
        self.duration_per_step = duration_per_step  # 每个动作插值步的持续时间（秒）
        self.min_steps_per_action = min_steps_per_action  # 每个动作至少插值步数
        # 速度上限用于自适应步数
        self.max_v_xyz = max_v_xyz
        self.max_v_rpy = max_v_rpy
        self.min_steps_gripper = min_steps_gripper  # 夹爪插值步最小值

        # 1. Enable the robot
        success, error_message = self._enable_robot()
        if not success:
            raise RuntimeError(f"Failed to enable the robot: {error_message}")

    def _setup_space(self):
        """设置动作空间与观测空间,提供查询"""
        # TODO 按照动作空间进行一些check。

        pass

    def _enable_robot(self) -> Tuple[bool, str | None]:
        # 开启通讯
        rclpy.init()
        # 启动ros的io节点与 executor与本体发布的topic建立联系
        self.node, self.executor, self.executor_thread = start_robot_io(
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
        """销毁通讯节点与 executor。"""
        # 安全销毁并置空，避免重复检查误报
        if getattr(self, "node", None) is not None:
            try:
                self.node.stop_saver()
            except Exception:
                pass
            self.node.destroy_node()
            self.node = None
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown()
            self.executor = None
        if getattr(self, "executor_thread", None) is not None:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        if rclpy.ok():
            rclpy.shutdown()
        # 再次检查
        if self.node is None and self.executor is None and not rclpy.ok():
            return (True, None)
        erorr = []
        if self.node is not None:
            erorr.append("控制io节点销毁失败")
        if self.executor is not None:
            erorr.append("控制io节点读写锁销毁失败")
        if rclpy.ok():
            erorr.append("ROS2关闭失败")
        return (False, "原因：".join(erorr) if erorr else "其他错误")

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the robot to the initial pose.

        Returns:
            observation (Dict[str, np.ndarray]): The initial state of the robot.
        """

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

    def _success_check(self, side: str, target: np.ndarray) -> Tuple[bool, str | None]:
        """委托给通用阈值检查。"""
        status_all = self.node.get_robot_status()
        return success_check(side, target, status_all)

    def _go_to_initial_pose(self) -> Tuple[bool, str | None]:
        """机械臂回初始位。"""
        home_action = {
            "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),

        }
        success, error_message = self._apply_action(home_action)
        if not success:
            print(f"回初始位失败: {error_message},强行切模式回 home")
            self._set_special_mode(1)  # 回 home 模式
            time.sleep(5.0)  # 等待模式切换完成
            return (False, f"回初始位失败: {error_message}")
        else:
            print(f"左右臂回初始位成功")
        return (True, None)

    def _set_special_mode(self, mode: int) -> Tuple[bool, str | None]:
        """设置特殊模式，如重力补偿等。"""
        mode_type = {0: "soft", 1: "home", 2: "protect", 3: "gravity"}
        cmd = RobotCmd()
        cmd.mode = mode
        # 0-soft,1-home,2-protect,3-gravity
        self.node.send_control_msg("left", cmd)
        self.node.send_control_msg("right", cmd)
        print(f"左右臂设置特殊模式 {mode_type.get(mode, 'unknown')} 完成")
        return (True, None)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Polls ROS topics for sensor data.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the observation.
        """
        obs = dict()
        # 使用与相机帧近似同刻的状态快照；若无则去拿最新状态
        time.sleep(0.05)  # 等待状态刷新
        camera_all, status_snapshot = self.node.get_camera(
            self.dir, self.img_size, return_status=True)
        status_all = status_snapshot if status_snapshot else self.node.get_robot_status()
        # 如果回传状态为空，或者光有键值没有内容，则视为获取失败
        if (not status_all) or (isinstance(status_all, dict) and not any(status_all.values())):
            print("状态获取失败，关闭节点退出")
            try:
                # 机械臂回位后，关闭与本体的通讯节点
                self._go_to_initial_pose()
                self._disable_robot()
            except Exception:
                pass
            raise RuntimeError("未获取到机器人状态，已关闭节点")
        lstatus = status_all.get("left")
        rstatus = status_all.get("right")
        if lstatus is not None and rstatus is not None:
            obs["left_end_pos"] = np.array(lstatus.end_pos, dtype=np.float32)
            obs["left_joint_pos"] = np.array(
                lstatus.joint_pos, dtype=np.float32)
            obs["left_joint_cur"] = np.array(
                lstatus.joint_cur, dtype=np.float32)
            obs["left_joint_vel"] = np.array(
                lstatus.joint_vel, dtype=np.float32)
            obs["right_end_pos"] = np.array(rstatus.end_pos, dtype=np.float32)
            obs["right_joint_pos"] = np.array(
                rstatus.joint_pos, dtype=np.float32)
            obs["right_joint_cur"] = np.array(
                rstatus.joint_cur, dtype=np.float32)
            obs["right_joint_vel"] = np.array(
                rstatus.joint_vel, dtype=np.float32)
        # 如果需要camera
        if not camera_all:
            keys = []
            try:
                keys = self.node.get_camera_keys()
            except Exception:
                pass
            print(
                f"相机帧为空，订阅话题: {getattr(self.node, 'subscribed_topics', [])}, 缓存keys: {keys}")
        else:
            for key, img in camera_all.items():
                if img is None:
                    print(f"{key} 获取失败")
                    continue
                # 保留原始深度精度，彩色统一为 uint8
                if 'color' in key:
                    img = np.asarray(img, dtype=np.uint8)
                else:
                    img = np.asarray(img)
                obs[key] = img
        # 输出观测包含的状态键与图像键，便于调试时查看
        state_keys = [k for k in obs.keys() if k.endswith(
            ("_end_pos", "_joint_pos", "_joint_cur", "_joint_vel"))]
        img_keys = list(camera_all.keys()) if camera_all else []
        print(f"obs状态键: {state_keys}, obs图像键: {img_keys}")
        return obs

    def _apply_action(self, action: Dict[str, np.ndarray]) -> Tuple[bool, str | None]:
        """
        apply to the robot single step control command.
            action: the action provieded by the agent（step level）.
            end: x,y,z,roll,pitch,yaw,gripper
            joint: joint0,joint1,...,joint5,gripper

        """
        curr_obs = self._get_observation()
        # 按位移和速度上限自适应步数，必要时回退到默认步数
        steps_by_side: Dict[str, int] = {}
        pose_changed: Dict[str, bool] = {}
        # 计算每个动作插值的步数
        for side in ("left", "right"):
            target = action.get(side)
            curr_end = curr_obs.get(f"{side}_end_pos")
            curr_joint = curr_obs.get(f"{side}_joint_pos")
            if target is None or curr_end is None or curr_joint is None:
                continue
            start = np.concatenate([np.array(curr_end, dtype=np.float32),
                                    [float(curr_joint[6])]])
            # 对齐以下类型
            target_arr = target if isinstance(
                target, np.ndarray) else np.array(target, dtype=np.float32)
            # 计算目标和现在其实位姿的差距
            diff = np.abs(target_arr - start)
            need_steps = []
            need_steps.append(
                int(np.ceil(diff[:3].max() / (self.max_v_xyz * self.duration_per_step))))
            need_steps.append(
                int(np.ceil(diff[3:6].max() / (self.max_v_rpy * self.duration_per_step))))
            pose_steps = max(self.min_steps_per_action, max(need_steps))
            # 标记是否有末端位姿变更，纯夹爪动作则不做成功检查
            pose_changed[side] = bool(np.any(diff[:6] > 1e-6))

            # 夹爪不按速度限制，压缩在前半段完成，步数至少 min_steps_gripper
            # 若夹爪变化极小，则直接一步到位，避免微抖
            grip_steps = 0
            delta_g = diff[6]
            if delta_g > 0:
                if delta_g <= 1e-3:
                    grip_steps = 1
                else:
                    grip_steps = max(self.min_steps_gripper,
                                     max(1, pose_steps // 3))
            # print(f"{side}臂动作插值步数计算: 位置步数 {pose_steps}, 夹爪步数 {grip_steps}")

            steps_by_side[side] = (pose_steps, grip_steps)

        sequences = interpolate_action(curr_obs, action, steps_by_side)
        lsequence = sequences.get("left") or []
        rsequence = sequences.get("right") or []
        max_len = max(len(lsequence), len(rsequence))
        for i in range(max_len):
            has_left = i < len(lsequence)
            has_right = i < len(rsequence)
            if not has_left and not has_right:
                continue
            t0 = time.time()
            if has_left:
                lmsg = RobotCmd()
                lmsg.mode = 4
                lmsg.end_pos = [float(x) for x in lsequence[i][:6]]
                lmsg.gripper = float(lsequence[i][6])
                lok = self.node.send_control_msg("left", lmsg)
                if not lok:
                    return (False, f"left: 指令未发送")
            if has_right:
                rmsg = RobotCmd()
                rmsg.mode = 4
                rmsg.end_pos = [float(x) for x in rsequence[i][:6]]
                rmsg.gripper = float(rsequence[i][6])
                rok = self.node.send_control_msg("right", rmsg)
                if not rok:
                    return (False, f"right: 指令未发送")
            dt = time.time() - t0
            sleep_need = self.duration_per_step - dt
            if sleep_need > 0:
                time.sleep(sleep_need)
        # 发完最后一步后等状态刷新，避免还在动作过程中就检查成功；只检查末端，不校验夹爪
        target_l = lsequence[-1] if lsequence else None
        target_r = rsequence[-1] if rsequence else None
        success = True
        lerror_message = rerror_message = None
        for _ in range(20):  # 最多等待约1秒（20*0.05）
            if target_l is not None and pose_changed.get("left", False):
                lsuccess, lerror_message = self._success_check(
                    "left", target_l)
            else:
                lsuccess = True
            if target_r is not None and pose_changed.get("right", False):
                rsuccess, rerror_message = self._success_check(
                    "right", target_r)
            else:
                rsuccess = True
            success = bool(lsuccess) and bool(rsuccess)
            if success:
                break
            time.sleep(0.05)
        if not success:
            return (False, f"left: {lerror_message}, right: {rerror_message}")
        return (True, None)


def main():
    arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                      min_steps_per_action=20,  # 每个动作至少插值20步，理论上来说越大越好
                      min_steps_gripper=10,  # 夹爪插值步数最少10步
                      # 就改速度参数其实就可以
                      max_v_xyz=0.1,
                      max_v_rpy=0.1,
                      camera_type="all",
                      camera_view=("camera_h",),
                      dir="testdata",
                      img_size=(640, 480))

    time.sleep(1.5)
    obs = arx.reset()

    actions = []
    x = 0.00
    for i in range(3):
        x += 0.05
        if i % 2 == 0:
            g = -3.4
        else:
            g = 0.0
        actions.append({
            "left": np.array([0, 0, x, 0, 0, 0, g], dtype=np.float32),
            "right": np.array([0, 0, x, 0, 0, 0, g], dtype=np.float32),
        })

    for i in range(3):
        obs, _, _, _ = arx.step(actions[i])
    arx.close()


if __name__ == "__main__":
    main()
