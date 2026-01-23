import sys
import time
import threading
import termios
import tty
import imageio
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from arm_control.msg._pos_cmd import PosCmd
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer

import math

from datetime import datetime

import argparse
import heapq
from typing import List, Tuple
from er1_5_tool import predict_point_from_rgb

# ! 测试，先固定Pw试试能不能到
STABLE_Pw = [1.0, 0.6, 0.0, 1.0]

# ===============================
# 相机内参
# ===============================
K = np.array([
    [391.9335632324219, 0.0, 320.5389099121094],
    [0.0, 391.6839294433594, 239.18849182128906],
    [0.0, 0.0, 1.0]
])

FX, FY = K[0, 0], K[1, 1]
CX, CY = K[0, 2], K[1, 2]

# ===============================
# 相机 → base_link 外参
# ===============================

T_CAM2REF = np.array([
    [-0.00573023442903553, -0.7003284264578072, 0.7137977721375433, 0.018783642268891315],
    [ -0.9997359307205717, 0.01989737860714924, 0.011496223329259703, 0.2696572130336537],
    [ -0.02225383651554841, -0.713543404019472, -0.7002575078788116, 0.138034882000128],
    [0.0, 0.0, 0.0, 1.0]
]
)

BIAS_REF2CAM = np.array([-0.28, -0.24, 0.0, 0.0])

# T_RIGHT2BASE = np.array([
#     [1.0, 0.0, 0.0, -0.252],
#     [0.0, 1.0, 0.0, 0.249],
#     [0.0, 0.0, 1.0, -0.391],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# T_CAM2BASE = T_RIGHT2BASE @ T_CAM2RIGHT

# ===============================
# 栅格参数
# ===============================
GRID_RES = 0.05      # m / cell 一个栅格是0.05米
GRID_SIZE = 200
MAX_DEPTH = 5.0

from enum import Enum

class AutoState(Enum):
    FORWARD_DETECT_CORNER = 1
    FORWARD_EXTRA = 2
    TURN_RIGHT_SEARCH_CUP = 3
    STOP = 4

class AutoControl(Node):
    def __init__(self):
        super().__init__('base_control')

        self.latest_height = 0.0

        self.running = True

        # -------- emergency stop ------
        self.kb_thread = threading.Thread(
            target=self.keyboard_listener,
            daemon=True
        )
        self.kb_thread.start()

        # ---------- auto state --------
        self.auto_state = AutoState.FORWARD_DETECT_CORNER

        # ---------- save path ----------
        self.save_root = Path(
            "/home/arx/Robotbase_base/data/camera_record"
        )
        self.save_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = self.save_root / ts
        self.save_dir.mkdir()

        self.rgb_dir = self.save_dir / "rgb"
        self.depth_dir = self.save_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()

        # -- rgb depth grid start goal --
        self.rgb = None
        self.depth = None
        self.grid = None
        self.start = None
        self.goal = None

        # -- ros --
        self.cmd_pub = self.create_publisher(PosCmd, '/ARX_VR_L', 1)
        self.sta_cmd_sub = self.create_subscription(PosCmd, '/ARX_VR_L', self.on_status, 1)
        self.bridge = CvBridge()

        self.action_log = []   # [(chx, chy, chz, duration)]
        self.rgb_frames = []
        self.frame_id = 0

        self.sub_rgb = Subscriber(
            self,
            Image,
            '/camera_h_namespace/camera_h/color/image_rect_raw',
            qos_profile=qos_profile_sensor_data
        )

        self.sub_depth = Subscriber(
            self,
            Image,
            '/camera_h_namespace/camera_h/aligned_depth_to_color/image_raw',
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=10,
            slop=0.03  # 30ms
        )
        self.sync.registerCallback(self.on_rgb_depth)

        # 默认悬浮高度（厘米或米，取决于底盘协议，保持与之前脚本一致）
        self.default_height = 0.0

        self.frame_id = 0

        self.get_logger().info(
            "Keyboard control:\n"
            " w: forward\n"
            " a: left\n"
            " d: right\n"
            " s: back\n"
            " q: save data\n"
            " f: replay forward\n"
            " r: replay reverse\n"
            " e: exit program\n"
            f"Save dir: {self.save_dir}"
        )

        self.lift_to_default_height(duration=1.0)

    def on_status(self, msg: PosCmd):
        self.latest_height = msg.height

    def lift_to_default_height(self, duration=19.0):
        msg = PosCmd()
        msg.height = 0.0
        while (msg.height < self.default_height):
            msg.chx = msg.chy = msg.chz = 0.0
            msg.height += 0.1
            msg.mode1 = 1
            self.cmd_pub.publish(msg)
            time.sleep(0.03)
        self.stop()


    def on_rgb_depth(self, msg_rgb: Image, msg_depth: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding='bgr8')

        self.depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')
        if self.depth.dtype == np.uint16:
            self.depth = self.depth.astype(np.float32)

        fid = self.frame_id

        cv2.imwrite(
            str(self.rgb_dir / f"{fid:06d}.png"),
            self.rgb
        )

        np.save(
            self.depth_dir / f"{fid:06d}.npy",
            self.depth
        )

        self.rgb_frames.append(self.rgb)
        self.frame_id += 1

    def get_pixel_pw(self, pixel):
        # depth = self.depth.astype(np.float32) / 1000.0
        Pw = self.detect_and_project(pixel, self.depth)
        # self.grid, self.start, self.goal = self.build_grid(Pw)
        self.start, self.goal = (0, 0), (Pw[0], -Pw[1])
        return Pw


    def detect_and_project(self, pixel, depth):
        u, v = pixel

        z = depth_to_meters(float(depth[int(v), int(u)]))
        if z <= 0 or z > MAX_DEPTH:
            return None, None

        # 像素 → 相机坐标
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        Pc = np.array([x, y, z, 1.0], dtype=np.float64)

        # 相机 → ref → base_link
        Pw_right = T_CAM2REF @ Pc
        Pw = Pw_right + BIAS_REF2CAM
        return Pw

    def build_grid(self, Pw):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        center = GRID_SIZE // 2

        start = (center, center)

        gx = int(center + Pw[0] / GRID_RES)
        # gy = int(center - Pw[1] / GRID_RES)
        gy = int(center - Pw[1] / GRID_RES)

        gx = np.clip(gx, 0, GRID_SIZE - 1)
        gy = np.clip(gy, 0, GRID_SIZE - 1)

        goal = (gx, gy)
        return grid, start, goal
    
    def draw_grid(self, grid, start, goal, path):
        img = grid.copy()

        for x, y in path:
            img[y, x] = (255, 0, 0)

        img[start[1], start[0]] = (0, 255, 0)
        img[goal[1], goal[0]] = (0, 0, 255)
        return img

    # -- motion --
    def stop(self):
        msg = PosCmd()
        msg.chx = msg.chy = msg.chz = 0.0
        msg.mode1 = 2
        msg.height = self.default_height
        self.cmd_pub.publish(msg)

    def run_for_1s(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        start = time.time()
        # while time.time() - start < duration:
        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        # 每次发送控制指令时，都附带期望的高度，这样底盘就会一直维持该高度
        msg.height = self.default_height
        msg.mode1 = 1

        self.cmd_pub.publish(msg)

        while self.running and time.time() - start < duration:
            time.sleep(0.03)   # 小步 sleep，方便中断

        self.stop()

        if record and self.running:
            self.action_log.append((chx, chy, chz, duration))

    def forward(self, distance=0.15):
        duration_time = distance / 0.12
        self.run_for_1s(chx=0.5, duration=duration_time)       # 前进0.24m/s
        return duration_time

    def forward_1_speed(self, distance=0.15):
        duration_time = distance / 0.24
        self.run_for_1s(chx=1.0, duration=duration_time)       # 前进0.24m/s
        return duration_time

    def back(self, distance=0.15):
        duration_time = distance / 0.24
        self.run_for_1s(chx=-1.0, duration=duration_time)       # 前进0.24m/s
        return duration_time

    def turn_left(self, angle=math.pi/4):
        angle_speed = (2 * math.pi) / 20.60
        duration_time = angle / angle_speed
        self.run_for_1s(chz=1.0, duration=duration_time)
        if angle < math.pi/8:
            self.run_for_1s(chz=duration_time)
        else:
            self.run_for_1s(chz=0.5, duration=duration_time*2)
        return duration_time

    def turn_right(self, angle=math.pi/4):
        angle_speed = (2 * math.pi) / 20.60
        duration_time = -angle / angle_speed
        if abs(angle) < math.pi/8:
            self.run_for_1s(chz=-duration_time)
        else:
            self.run_for_1s(chz=-0.5, duration=duration_time*2)
        return duration_time

    def forward_until_corner_disappears(self):
        self.get_logger().info("Start forward & detect table corner")


        max_distance = 3.5

        max_duration_time = max_distance / 0.12
        start_time = time.time()
        msg = PosCmd()
        msg.chx = 0.5
        msg.chy = 0.0
        msg.chz = 0.0
        msg.height = self.default_height
        msg.mode1 = 1
        self.cmd_pub.publish(msg)

        # self.forward(1.2)

        while self.rgb is None:
            print("No rgb!")
            time.sleep(0.1)
        color = self.rgb.copy()

        # 桌面曝光太高，以桌角下的笔来作为标志物
        point = predict_point_from_rgb(
            color,
            text_prompt="Where is the pen on the ground?",
            detect_goal="pen",
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-v1",
            api_key="EMPTY"
        )

        # 没有笔
        if point == (-1, -1):
            color = self.rgb.copy()
            # 再检测一下
            point = predict_point_from_rgb(
                color,
                text_prompt="Where is the pen on the ground?",
                detect_goal="pen",
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-v1",
                api_key="EMPTY"
            )
            
            # 两次都没有笔
            if point == (-1, -1):
                self.get_logger().info("Corner disappeared!")
                self.stop()
                return

        print("桌角下笔的像素:", point)

        while self.running and rclpy.ok() and point != (-1, -1) and time.time() - start_time < max_duration_time:

            color = self.rgb.copy()

            point = predict_point_from_rgb(
                color,
                text_prompt="Where is the pen on the table corner?",
                detect_goal="pen",
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-v1",
                api_key="EMPTY"
            )

            print("桌角下笔的像素:", point)

            if point == (-1, -1):
                self.get_logger().info("Corner disappeared!")
                self.stop()
                return
    
    def forward_extra_distance(self, distance=1.3):
        self.get_logger().info("Go forward extra distance!")
        self.forward(distance)
    
    def turn_right_until_see_cup(self):
        self.get_logger().info("Turn right and search for cup")

        angle = math.pi

        duration_time = math.pi / ((2 * math.pi) / 20.60)
        start_time = time.time()
        msg = PosCmd()
        msg.chx = 0.0
        msg.chy = 0.0
        msg.chz = -0.5
        msg.height = self.default_height
        msg.mode1 = 1
        self.cmd_pub.publish(msg)

        # self.turn_right(-math.pi)

        while self.rgb is None:
            print("No rgb!")
            time.sleep(0.5)
        color = self.rgb.copy()

        cup_point = predict_point_from_rgb(
            color,
            text_prompt="Where is the cup on the table?",
            detect_goal="cup",
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-v1",
            api_key="EMPTY"
        )

        print("杯子像素:", cup_point)

        if cup_point != (-1, -1):
            self.get_logger().info("Cup detected!")
            self.stop()
            return cup_point

        while self.running and rclpy.ok() and cup_point == (-1, -1) and time.time() - start_time < 2 * duration_time:

            color = self.rgb.copy()

            cup_point = predict_point_from_rgb(
                color,
                text_prompt="Where is the cup on the table?",
                detect_goal="cup",
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-v1",
                api_key="EMPTY"
            )

            print("杯子像素:", cup_point)

            if cup_point != (-1, -1):
                self.get_logger().info("Cup detected!")
                self.stop()
                return cup_point
        
        return (-1, -1)

    def forward_until_see_cup(self):
        self.get_logger().info("Go forward and search for cup")

        max_distance = 2.0

        max_duration_time = max_distance / 0.12

        start_time = time.time()
        msg = PosCmd()
        msg.chx = 0.5
        msg.chy = 0.0
        msg.chz = 0.0
        msg.height = self.default_height
        msg.mode1 = 1
        self.cmd_pub.publish(msg)

        # self.turn_right(-math.pi)

        color = self.rgb.copy()

        cup_point = predict_point_from_rgb(
            color,
            text_prompt="Where is the cup on the table?",
            detect_goal="cup",
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-v1",
            api_key="EMPTY"
        )

        print("杯子像素:", cup_point)

        if cup_point != (-1, -1):
            self.get_logger().info("Cup detected!")
            self.stop()
            return cup_point

        while self.running and rclpy.ok() and cup_point == (-1, -1) and time.time() - start_time < max_duration_time:

            color = self.rgb.copy()

            cup_point = predict_point_from_rgb(
                color,
                text_prompt="Where is the cup on the table?",
                detect_goal="cup",
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-v1",
                api_key="EMPTY"
            )

            print("杯子像素:", cup_point)

            if cup_point != (-1, -1):
                self.get_logger().info("Cup detected!")
                self.stop()
                return cup_point
        
        return (-1, -1)


    # ---------- save ----------
    def save_video(self):
        if not self.rgb_frames:
            self.get_logger().warn("No RGB frames collected")
            return

        h, w, _ = self.rgb_frames[0].shape
        video_path = self.save_dir / "rgb.mp4"

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,
            (w, h)
        )

        for frame in self.rgb_frames:
            writer.write(frame)
        writer.release()

        self.get_logger().info(f"Saved RGB video: {video_path}")

    def save_gif(self):
        if not self.rgb_frames:
            self.get_logger().warn("No RGB frames collected")
            return

        gif_path = self.save_dir / "rgb.gif"

        frames_rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in self.rgb_frames
        ]

        imageio.mimsave(
            gif_path,
            frames_rgb,
            fps=20  
        )

        self.get_logger().info(f"Saved RGB GIF: {gif_path}")

    def save_actions(self):
        """保存动作序列到 txt + json 文件"""
        actions_path = self.save_dir / "actions.txt"
        actions_json_path = self.save_dir / "actions.json"

        # ---------- 1. 保存 TXT（保持你原来的格式） ----------
        with open(actions_path, 'w', encoding='utf-8') as f:
            f.write("# 动作序列记录\n")
            f.write("# 格式: chx, chy, chz, duration(秒)\n")
            f.write("# chx: 前进/后退速度, chy: 左右平移速度, chz: 旋转速度, duration: 持续时间\n")
            f.write(f"# 总动作数: {len(self.action_log)}\n\n")

            for i, (chx, chy, chz, duration) in enumerate(self.action_log, 1):
                if chx > 0:
                    action_desc = "前进"
                elif chx < 0:
                    action_desc = "后退"
                elif chz > 0:
                    action_desc = "左转"
                elif chz < 0:
                    action_desc = "右转"
                else:
                    action_desc = "停止"

                f.write(
                    f"动作 {i:04d}: {action_desc:4s} | "
                    f"chx={chx:6.2f}, chy={chy:6.2f}, "
                    f"chz={chz:6.2f}, duration={duration:.2f}s\n"
                )

        # ---------- 2. 保存 JSON（结构化） ----------
        actions_json = {
            "meta": {
                "description": "动作序列记录",
                "format": {
                    "chx": "前进/后退速度",
                    "chy": "左右平移速度",
                    "chz": "旋转速度",
                    "duration": "持续时间（秒）"
                },
                "total_actions": len(self.action_log)
            },
            "actions": []
        }

        for i, (chx, chy, chz, duration) in enumerate(self.action_log, 1):
            if chx > 0:
                action_type = "前进"
            elif chx < 0:
                action_type = "后退"
            elif chz > 0:
                action_type = "左转"
            elif chz < 0:
                action_type = "右转"
            else:
                action_type = "停止"

            actions_json["actions"].append({
                "id": i,
                "type": action_type,
                "chx": float(chx),
                "chy": float(chy),
                "chz": float(chz),
                "duration": float(duration)
            })

        with open(actions_json_path, "w", encoding="utf-8") as f:
            json.dump(actions_json, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f"Saved actions log: {actions_path}")
        self.get_logger().info(f"Saved actions json: {actions_json_path}")

    def save_data(self):
        self.get_logger().info("Saving data...")
        # self.save_video()        # 可选择保存为mp4格式或gif格式
        self.save_gif()
        self.save_actions()  # 保存动作序列
        self.get_logger().info("Data saved successfully!")

    # 回放动作
    def replay_forward(self):
        self.get_logger().info("Replaying actions forward...")
        for chx, chy, chz, duration in self.action_log:
            self.run_for_1s(chx, chy, chz, duration, record=False)
        self.get_logger().info("Replaying actions forward successfully!")
    # 回溯动作
    def replay_reverse(self):
        self.get_logger().info("Replaying actions in reverse...")
        for chx, chy, chz, duration in reversed(self.action_log):
            self.run_for_1s(-chx, -chy, -chz, duration, record=False)
        self.get_logger().info("Replaying actions in reverse successfully!")

    # 监听键盘
    def keyboard_listener(self):
        while self.running:
            try:
                ch = get_key()
            except Exception:
                continue

            if ch == 'q':
                self.get_logger().warn("Key 'q' pressed! Emergency stop!")
                self.running = False
                self.stop()
                break

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """启发函数：曼哈顿距离"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    use_diagonal: bool = False,
) -> List[Tuple[int, int]]:

    h, w = grid.shape

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    def passable(x, y):
        return grid[y, x] == 0

    if use_diagonal:
        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
    else:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    open_heap = []
    g_cost = {start: 0.0}
    came_from = {}

    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    while open_heap:
        _, g_cur, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if g_cur > g_cost.get(current, float("inf")):
            continue

        x, y = current
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or not passable(nx, ny):
                continue

            step = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
            new_g = g_cur + step

            if new_g < g_cost.get((nx, ny), float("inf")):
                g_cost[(nx, ny)] = new_g
                came_from[(nx, ny)] = current
                f = new_g + heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (f, new_g, (nx, ny)))

    return []

def path_to_actions(
    path: List[Tuple[int, int]],
    cell_size: float,
    init_yaw: float = 0.0,
):
    actions = []
    cur_yaw = init_yaw

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        # dx = (x1 - x0) * cell_size
        # dy = (y1 - y0) * cell_size
        dx = x1 - x0
        dy = y1 - y0

        target_yaw = math.atan2(dy, dx)
        d_yaw = target_yaw - cur_yaw

        # 归一化到 [-pi, pi]
        while d_yaw > math.pi:
            d_yaw -= 2 * math.pi
        while d_yaw < -math.pi:
            d_yaw += 2 * math.pi

        dist = math.hypot(dx, dy)

        if abs(d_yaw) > 1e-3:
            actions.append(("rotate", -d_yaw))
            cur_yaw = target_yaw

        if dist > 1e-3:
            actions.append(("forward", dist))

    return actions

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

Action = Tuple[str, float]

def merge_forward_actions(actions: List[Action]) -> List[Action]:
    merged = []
    forward_acc = 0.0

    for act, val in actions:
        if act == "forward":
            forward_acc += val
        else:
            # 遇到非 forward，先把累计的 forward 放进去
            if forward_acc > 0:
                merged.append(("forward", forward_acc))
                forward_acc = 0.0
            merged.append((act, val))

    # 结尾如果是 forward
    if forward_acc > 0:
        merged.append(("forward", forward_acc))

    return merged

def depth_to_meters(raw_depth: float) -> float:
    """兼容毫米与米的深度值。"""
    if not np.isfinite(raw_depth) or raw_depth <= 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)

def main():

    rclpy.init()
    node = AutoControl()
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    spin_thread.start()

    Pw = None

    try:
        time.sleep(3)

        # if not node.running:
        #     return

        # node.forward_until_corner_disappears()
        # if not node.running:
        #     return

        # node.forward_extra_distance(2.8)
        # if not node.running:
        #     return

        
        # # 方式一：距桌角比较近
        # cup_point = node.turn_right_until_see_cup()
        # way_num = 1

        # # # 方式二：距桌角比较远
        # # node.turn_right(-math.pi/2)
        # # cup_point = node.forward_until_see_cup()
        # # way_num = 2

        # node.get_logger().info("Auto turn finished")

        # Pw = node.get_pixel_pw(cup_point)
        # print(f"奶茶杯 {cup_point} -> 基坐标系 3D 点: {Pw.tolist()}")

        # # grid
        # grid = node.grid
        # start = node.start
        # goal = node.goal

        # # if grid is None:
        # #     print("No grid!")
        # #     node.running = False

        # if start is None:
        #     print("No start!")
        #     node.running = False
        
        # if goal is None:
        #     print("No goal!")
        #     node.running = False

        # print(grid)

        # grid_2d = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        # # 路径规划
        # # path = astar(grid=grid_2d, start=start, goal=goal)
        # path = [start, goal]
        
        # # grid_img = node.draw_grid(grid, start, goal, path)
        # # cv2.imwrite(os.path.join(node.save_dir, 'grid_path.png'), grid_img)

        # actions = path_to_actions(path, cell_size=GRID_RES)
        # actions = merge_forward_actions(actions)
        # # if actions[0][0] == "rotate":
        # #     last_action_reverse = (actions[0][0], -actions[0][1])
        # #     actions.append(last_action_reverse)
        # if way_num == 2:
        #     actions = [actions[0]]
        # print(actions)
        
        # # while node.running:
        # #     key = get_key()
        # #     if key == 'e':
        # #         node.running = False
        # #         break
        # #     elif key == 'w':
        # #         for action, action_content in actions:
        # #             if action == "forward":
        # #                 duration_time = node.forward(action_content*2)
        # #                 # time.sleep(duration_time)
        # #             elif action == "rotate":
        # #                 if action_content <= 0:
        # #                     duration_time = node.turn_right(angle=action_content)
        # #                     # time.sleep(duration_time)
        # #                 else:
        # #                     duration_time = node.turn_left(angle=action_content)
        # #                     # time.sleep(duration_time)
        # #     elif key == 'q':
        # #         node.save_data()


    finally:
        # 退出程序前，将机器人高度缓慢降回为 0
        msg = PosCmd()
        msg.height = node.latest_height
        while (msg.height > 0.0):
            msg.chx = msg.chy = msg.chz = 0.0
            msg.height -= 0.1
            msg.mode1 = 1
            node.cmd_pub.publish(msg)
            time.sleep(0.03)

        node.get_logger().info("Shutting down ROS...")

        rclpy.shutdown()
        node.destroy_node()
        spin_thread.join()
        
        
if __name__ == '__main__':
    main()

