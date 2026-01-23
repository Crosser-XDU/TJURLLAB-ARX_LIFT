import numpy as np
from typing import Dict, Optional

CLOSE = 0.1
OPEN = -3.4


def make_pick_move_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """张开夹爪偏移到目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0] - 0.12, base[1], base[2], 0, 0, 0, -3.4],
            dtype=np.float32,
        ),
    }


def make_pick_robust_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒夹取位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0] - 0.09, base[1], base[2], 0, 0, 0, -3.4],
            dtype=np.float32,
        ),
    }


def make_close_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行夹紧动作，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0]-0.08, base[1], base[2], 0, 0, 0, 0.0],
            dtype=np.float32,
        ),
    }


def make_pick_stop_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """回撤一点抓回位置，抬高保证一个重力对抗，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0] - 0.16, base[1], base[2] + 0.15, 0, 0, 0, 0.0],
            dtype=np.float32,
        ),
    }


def make_pick_back_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹住回到初始位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {"left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "right": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)}


def make_place_move_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """保持抓取偏移到放置目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0] - 0.12, base[1], base[2]+0.08, 0, 0, 0, -2.6],
            dtype=np.float32,
        ),
    }


def make_place_robust_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒放置位置。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0]-0.12, base[1], base[2]+0.08, 0, 0, 0, -2.6],
            dtype=np.float32,
        ),
    }


def make_down_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """下降到放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0]-0.12, base[1], base[2]+0.06, 0, 0, 0, -2.6],
            dtype=np.float32,
        ),
    }


def make_open_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹爪张开放置"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array([base[0]-0.12, base[1], base[2]+0.06, 0, 0, 0, -3.4], dtype=np.float32)}


def make_place_stop_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """回撤一点放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array(
            [base[0] - 0.16, base[1], base[2]+0.06, 0, 0, 0, -3.4],
            dtype=np.float32,
        ),
    }


def make_release_acqion(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹爪home位置张开放置"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32)}
