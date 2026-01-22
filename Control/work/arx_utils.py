import os
import time
import threading
from typing import Dict, Iterable, Optional, Literal

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# 本体节点消息
from arx5_arm_msg.msg._robot_cmd import RobotCmd  # 双臂控制命令
from arx5_arm_msg.msg._robot_status import RobotStatus  # 状态消息
from arm_control.msg._pos_cmd import PosCmd # 底盘控制命令

from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
from cv_bridge import CvBridge


class RobotIO(Node):
    def __init__(self, camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
        super().__init__('robot_io')
        self.bridge = CvBridge() if CvBridge is not None else None
        # 机器人命令发布到节点arm_cmd
        self.cmd_pub = self.create_publisher(RobotCmd, 'arm_cmd', 5)
        self.latest_status = None
        self.status_lock = threading.Lock()
        # 机器人状态从arm_status订阅
        self.create_subscription(RobotStatus, 'arm_status', self._on_status, 5)
        # 相机订阅
        self.camera_type = camera_type  # color/depth/all，传输一律用 compressed
        self.camera_view = list(camera_view) if camera_view else []
        self.cam_lock = threading.Lock()
        self.latest_images: Dict[str, CompressedImage] = {}

        # 使用近似时间同步多路相机
        subs = []
        labels = []
        types = ["color", "depth"] if camera_type == "all" else [camera_type]
        for cam in self.camera_view:
            for typ in types:
                # TODO: 确认具体话题名
                topic = f"/camera/{cam}/{typ}/image_rect_raw/compresse"
                subs.append(Subscriber(
                    self, CompressedImage, topic, qos_profile=5))
                labels.append(f"{cam}_{typ}")
        # 缓存5张，允许 0.02s 的时间偏差，和现在默认realsense的90hz对得上
        self.labels = labels
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.02)
        self.sync.registerCallback(self._on_images)

    def _on_status(self, msg):
        """收到机器人状态消息的回调函数。"""
        # 可以理解为一个持续更新的缓存，但带读写锁
        with self.status_lock:
            self.latest_status = msg

    def _on_images(self, *msgs):
        """现在这个回调是只有满足近似时间才触发"""
        with self.cam_lock:
            for label, msg in zip(self.labels, msgs):
                self.latest_images[label] = msg

    def send_control_msg(self, cmd: RobotCmd):
        """发送控制命令到机器人。"""
        # begin = time.time()
        # # 等待至少一个订阅者连上再发
        # while self.cmd_pub.get_subscription_count() > 0 and rclpy.ok():
        #     print(f"等待订阅链接时间{time.time()-begin}")
        #     self.cmd_pub.publish(cmd)
        #     break
        if self.cmd_pub.get_subscription_count() > 0 and rclpy.ok():
            self.cmd_pub.publish(cmd)
        return True

    def get_robot_status(self):
        """返回最新的机器人状态。"""
        with self.status_lock:
            return self.latest_status

    def get_camera(self, save_dir: Optional[str] = None, target_size: Optional[tuple[int, int]] = None):
        """返回最新近似同步的相机帧，可选保存到目录；target_size 传 (w, h) 时会 resize。"""
        if self.bridge is None:
            return dict()
        frames = dict()
        with self.cam_lock:
            items = list(self.latest_images.items())
        for key, msg in items:
            img = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
            if target_size:
                img = cv2.resize(img, target_size)
            frames[key] = img
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ts = time.time()
                cv2.imwrite(os.path.join(save_dir, f"{key}_{ts}.png"), img)
        return frames


def start_robot_io(camera_type: Literal["co", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
    # rclpy.init()
    node = RobotIO(camera_type=camera_type, camera_view=camera_view)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return node, executor
