import os
import time
import threading
import queue
from typing import Dict, Iterable, Optional, Literal

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arx5_arm_msg.msg._robot_status import RobotStatus

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RobotIO(Node):
    def __init__(self, camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
        super().__init__('robot_io')
        self.bridge = CvBridge() if CvBridge is not None else None
        # 左右臂命令发布
        self.cmd_pub_l = self.create_publisher(RobotCmd, 'arm_cmd_l', 5)
        self.cmd_pub_r = self.create_publisher(RobotCmd, 'arm_cmd_r', 5)
        self.latest_status: Dict[str, Optional[RobotStatus]] = {
            "left": None, "right": None}
        # 近似与相机帧同步的状态快照
        self.status_snapshot: Optional[Dict[str, Optional[RobotStatus]]] = None
        self.status_lock = threading.Lock()
        # 左右臂状态订阅
        self.create_subscription(
            RobotStatus, 'arm_status_l', lambda msg: self._on_status('left', msg), 5)
        self.create_subscription(
            RobotStatus, 'arm_status_r', lambda msg: self._on_status('right', msg), 5)
        # 相机订阅（近似同步多路）
        self.camera_type = camera_type  # color/depth/all
        self.camera_view = list(camera_view) if camera_view else []
        self.cam_lock = threading.Lock()
        self.latest_images: Dict[str, Image] = {}
        self.subscribed_topics = []
        self.save_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self.saver_thread = threading.Thread(
            target=self._save_worker, daemon=True)
        self.saver_thread.start()

        subs: list[Subscriber] = []
        labels: list[str] = []
        # 这里订阅的深度图片的topic是对齐彩色像素的
        types = [
            "color", "aligned_depth_to_color"] if camera_type == "all" else [camera_type]
        for cam in self.camera_view:
            for typ in types:
                if "aligned" in typ:
                    topic = f"/{cam}_namespace/{cam}/aligned_depth_to_color/image_raw"
                else:
                    topic = f"/{cam}_namespace/{cam}/{typ}/image_rect_raw"
                subs.append(Subscriber(self, Image, topic, qos_profile=5))
                labels.append(f"{cam}_{typ}")
                self.subscribed_topics.append(topic)
        self.labels = labels
        if subs:
            self.sync = ApproximateTimeSynchronizer(
                subs, queue_size=5, slop=0.02)
            self.sync.registerCallback(self._on_images_status)
            self.get_logger().info(f"订阅相机话题: {self.subscribed_topics}")
        else:
            self.get_logger().warn("未配置相机订阅，camera_view 为空。")
        self.get_logger().info(
            f"初始化 camera_view={self.camera_view}, types={types}")

    def _on_status(self, side: str, msg: RobotStatus):
        with self.status_lock:
            self.latest_status[side] = msg

    def _on_images_status(self, *msgs):
        # 先记录状态快照，再更新相机帧，尽量对齐时间
        # 先通过最新状态拿到快照，再更新图像
        with self.status_lock:
            self.status_snapshot = dict(self.latest_status)
        with self.cam_lock:
            for label, msg in zip(self.labels, msgs):
                self.latest_images[label] = msg

    def send_control_msg(self, side: str, cmd: RobotCmd):
        pub = self.cmd_pub_l if side == "left" else self.cmd_pub_r
        if not rclpy.ok():
            try:
                self.get_logger().warn("ROS 未就绪，指令未发送")
            except Exception:
                pass
            return False
        sub_count = pub.get_subscription_count()
        if sub_count == 0:
            try:
                self.get_logger().warn(f"{side} 没有订阅者，指令未发送")
            except Exception:
                pass
            return False
        pub.publish(cmd)
        return True

    def get_robot_status(self):
        with self.status_lock:
            return self.latest_status

    def get_camera(self, save_dir: Optional[str] = None, target_size: Optional[tuple[int, int]] = None,
                   return_status: bool = False):
        """返回最新近似同步的相机帧，可选返回拍摄时的状态快照。"""
        if self.bridge is None:
            print("CvBridge 未初始化，无法获取图像。")
            return (dict(), self.status_snapshot) if return_status else dict()
        frames = dict()
        with self.cam_lock:
            items = list(self.latest_images.items())
        for key, msg in items:
            try:
                img = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
            except Exception as exc:  # pragma: no cover
                self.get_logger().warn(f"{key} 解码失败: {exc}")
                continue
            if target_size:
                img = cv2.resize(img, target_size)
            frames[key] = img
            if save_dir:
                stamp = getattr(msg, "header", None)
                if stamp:
                    ts = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
                else:
                    ts = time.time()
                self.save_queue.put((save_dir, key, ts, img))
        if return_status:
            with self.status_lock:
                snap = dict(
                    self.status_snapshot) if self.status_snapshot is not None else None
            return frames, snap
        return frames

    def get_camera_keys(self):
        with self.cam_lock:
            return list(self.latest_images.keys())

    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                break
            save_dir, key, ts, img = task
            try:
                os.makedirs(save_dir, exist_ok=True)
                base = os.path.join(save_dir, f"{key}_{ts}")
                if "depth" in key:
                    np.save(base + ".npy", img)
                    depth_float = img.astype(np.float32)
                    finite = depth_float[np.isfinite(depth_float)]
                    if finite.size == 0:
                        continue
                    vmax = np.percentile(finite, 99)
                    if vmax <= 1e-6:
                        continue
                    vis = cv2.convertScaleAbs(depth_float, alpha=255.0 / vmax)
                    cv2.imwrite(base + "_vis.png", vis,
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(base + ".png", img,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as exc:  # pragma: no cover
                try:
                    self.get_logger().warn(f"保存 {key} 失败: {exc}")
                except Exception:
                    pass
            finally:
                self.save_queue.task_done()

    def stop_saver(self):
        self.save_queue.put(None)
        if getattr(self, "saver_thread", None) is not None:
            self.saver_thread.join(timeout=2.0)


def start_robot_io(camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
    node = RobotIO(camera_type=camera_type, camera_view=camera_view)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return node, executor, t


def success_check(side: str, target: np.ndarray, status_all: Dict[str, Optional[RobotStatus]],
                  threshold_xyz: float = 1.0, threshold_rpy: float = 1.5) -> tuple[bool, str | None]:
    current_status = status_all.get(
        side) if isinstance(status_all, dict) else None
    if current_status is None:
        return False, f"{side} 无状态数据"

    curr_end_xyz = current_status.end_pos[:3]
    curr_end_rpy = current_status.end_pos[3:6]

    diff_xyz = np.abs(curr_end_xyz - target[:3])
    diff_rpy = np.abs(curr_end_rpy - target[3:6])

    reasons = []
    axis_xyz = ["x", "y", "z"]
    axis_rpy = ["roll", "pitch", "yaw"]
    for idx, axis in enumerate(axis_xyz):
        if diff_xyz[idx] > threshold_xyz:
            reasons.append(f"{side} {axis}超阈值({diff_xyz[idx]:.4f})")
    for idx, axis in enumerate(axis_rpy):
        if diff_rpy[idx] > threshold_rpy:
            reasons.append(f"{side} {axis}超阈值({diff_rpy[idx]:.4f})")

    if not reasons:
        return True, None
    return False, "; ".join(reasons)


def interpolate_action(curr_obs: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
                       steps: Dict[str, tuple[int, int]]) -> Dict[str, list[np.ndarray]]:
    """插值生成左右臂的末端+夹爪序列。

    steps: 每侧一个 (pose_steps, gripper_steps)，二者独立，最终长度取二者最大；超出各自长度时保持最后一个值。
    """
    results: Dict[str, list[np.ndarray]] = {}

    def smoothstep5(n: int) -> np.ndarray:
        """五次平滑插值 6t^5-15t^4+10t^3，起停更柔和。"""
        if n <= 1:
            return np.array([1.0], dtype=np.float32)
        t = np.linspace(0.0, 1.0, n)
        return t**3 * (10 - 15 * t + 6 * t * t)

    for side in ("left", "right"):
        target = action.get(side)
        if target is None:
            continue
        if not isinstance(target, np.ndarray):
            target = np.array(target, dtype=np.float32)
        curr_end = curr_obs.get(f"{side}_end_pos")
        curr_joint = curr_obs.get(f"{side}_joint_pos")
        if curr_end is None or curr_joint is None:
            continue

        curr_gripper = float(curr_joint[6])
        start_pose = np.array(curr_end, dtype=np.float32)
        start_grip = curr_gripper

        pose_steps, grip_steps = steps.get(side, (0, 0))
        max_steps = max(pose_steps, grip_steps)
        if max_steps <= 0:
            continue

        pose_alphas = smoothstep5(pose_steps)
        grip_alphas = smoothstep5(grip_steps)
        seq: list[np.ndarray] = []
        for i in range(max_steps):
            pose_alpha = pose_alphas[min(
                i, pose_steps - 1)] if pose_steps > 0 else 1.0
            grip_alpha = grip_alphas[min(
                i, grip_steps - 1)] if grip_steps > 0 else 1.0
            pose_interp = start_pose + (target[:6] - start_pose) * pose_alpha
            grip_interp = start_grip + \
                (float(target[6]) - start_grip) * grip_alpha
            seq.append(np.concatenate([pose_interp, [grip_interp]]))
        results[side] = seq
    return results
