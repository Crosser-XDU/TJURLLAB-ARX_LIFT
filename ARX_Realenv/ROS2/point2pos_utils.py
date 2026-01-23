"""
将 2D 像素点 + 对齐深度转换为基坐标系下的 6 维末端姿态。

默认使用右手高位相机的标定文件：
- 内参: instrics_right4camerah.json
- 外参: final_extrinsics_cam_h_right.json (T_cam2ref)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INTRINSICS = WORKSPACE / "instrinsics_right4camerah.json"
DEFAULT_EXTRINSICS = WORKSPACE / "final_extrinsics_cam_h_right.json"


def _depth_to_meters(raw_depth: float) -> float:
    """将深度值转换为米单位，兼容毫米与米输入。"""
    if not np.isfinite(raw_depth) or raw_depth <= 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)


def _load_intrinsics(path: Path | str | None = None) -> np.ndarray:
    """读取 3x3 相机内参矩阵。"""
    intr_path = Path(path) if path else DEFAULT_INTRINSICS
    data = json.loads(intr_path.read_text())
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"内参矩阵形状异常: {K.shape}")
    return K


def _load_cam2ref(path: Path | str | None = None) -> np.ndarray:
    """读取 4x4 齐次矩阵 T_cam2ref。"""
    ext_path = Path(path) if path else DEFAULT_EXTRINSICS
    payload = json.loads(ext_path.read_text())
    T = np.asarray(payload.get("T_cam2ref"), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"外参矩阵形状异常: {T.shape}")
    return T


def point2pos(
    pixel: Tuple[float, float],
    depth_image: np.ndarray,
    current_end_pos: Iterable[float],
    intrinsics_path: Path | str | None = None,
    extrinsics_path: Path | str | None = None,
) -> np.ndarray:
    """
    像素 + 深度 -> 基坐标系下的末端 6DoF（位置来自反投影，姿态沿用 current_end_pos 的 rpy）。

    Args:
        pixel: (u, v) 像素坐标。
        depth_image: 与彩色对齐的深度图（单通道）。
        current_end_pos: 末端当前 6 维姿态，用于保留 roll/pitch/yaw。
        intrinsics_path: 可选内参路径，默认 instrics_right4camerah.json。
        extrinsics_path: 可选外参路径，默认 final_extrinsics_cam_h_right.json。
    Returns:
        np.ndarray, shape (6,) -> [x, y, z, roll, pitch, yaw]（ref/基坐标系）。
    """
    if depth_image is None or depth_image.ndim != 2:
        raise ValueError(
            f"深度图必须为单通道，当前: {None if depth_image is None else depth_image.shape}")

    u, v = int(round(pixel[0])), int(round(pixel[1]))
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")

    z = _depth_to_meters(float(depth_image[v, u]))

    K = _load_intrinsics(intrinsics_path)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    T_cam2ref = _load_cam2ref(extrinsics_path)
    ref_point = T_cam2ref @ cam_point

    curr = np.asarray(list(current_end_pos), dtype=np.float64).flatten()
    if curr.shape[0] < 6:
        raise ValueError(f"current_end_pos 维度不足: {curr.shape}")

    return np.concatenate([ref_point[:3], curr[3:6]])


def load_intrinsics(path: Path | str | None = None) -> np.ndarray:
    """公开版内参读取，默认读取右手高位相机参数。"""
    return _load_intrinsics(path)


def load_cam2ref(path: Path | str | None = None) -> np.ndarray:
    """公开版外参读取，默认读取右手高位相机 T_cam2ref。"""
    return _load_cam2ref(path)


def depth_to_meters(raw_depth: float) -> float:
    """公开版深度单位转换，兼容 mm / m。"""
    return _depth_to_meters(raw_depth)


def pixel_to_ref_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray:
    """像素 + 深度 -> 基坐标系 3D 点。"""
    u, v = pixel
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")
    z = _depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    ref_point = T_cam2ref @ cam_point
    return ref_point[:3]


__all__ = [
    "point2pos",
    "load_intrinsics",
    "load_cam2ref",
    "depth_to_meters",
    "pixel_to_ref_point",
]
