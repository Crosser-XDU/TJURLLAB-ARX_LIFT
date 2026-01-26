"""
基于 collect_calibration.py 采集的数据，使用 ChArUco 板进行角点识别并计算眼在手外外参。

与 calibrate_extrinsics.py 的主要区别：
- 标定板从棋盘格改为 ChArUco（ArUco + 棋盘角点），可在遮挡下保持鲁棒检测
- 仍然只做角点识别与 Hand-Eye 求解，不做可视化调参

使用前提：
- 样本是单张 RGB 图 + end_pos（相对于 ref 坐标系），ChArUco 规格由参数指定
- 相机内参通过 JSON/XML 提供
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 roll/pitch/yaw(ZYX) 转为旋转矩阵。"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


DEFAULT_INTR_PATH = Path(__file__).resolve().parent / "right4camerah.json"


def load_intrinsics(path: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    """从 JSON（camera_matrix, dist_coeffs）或 OpenCV XML 读取 K 与 D；默认 right4camerah.json。"""
    use_path = path if path is not None else DEFAULT_INTR_PATH
    if use_path.suffix.lower() == ".json":
        data = json.loads(use_path.read_text())
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
        return K, dist
    fs = cv2.FileStorage(str(use_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开内参文件: {use_path}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError(f"内参文件缺少 camera_matrix 或 dist_coeffs: {use_path}")
    return K, dist


def pick_image(meta: Dict, sample_dir: Path) -> Tuple[Path, np.ndarray] | None:
    """返回 (图像路径, image)。当前目录只存单张彩色图，若有 color 关键字则优先。"""
    files = sorted(sample_dir.glob("*.png"))
    if not files:
        return None
    color_files = [f for f in files if "color" in f.name]
    if color_files:
        files = color_files
    img_path = files[0]
    if img_path.suffix == ".npy":
        img = np.load(img_path)
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img_path, img


def get_aruco_dictionary(dict_name: str) -> cv2.aruco.Dictionary:
    """根据名称获取 ArUco 字典。"""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 未包含 aruco 模块（需要 opencv-contrib-python）")
    if hasattr(cv2.aruco, dict_name):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    raise ValueError(f"未知的 ArUco 字典名称: {dict_name}")


def detect_charuco(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
) -> Tuple[np.ndarray, np.ndarray, tuple] | None:
    """
    ChArUco 角点 -> PnP 估计 board -> cam 的位姿。
    需至少 4 个插值角点。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=parameters)
    if not corners:
        return None
    if rejected:
        cv2.aruco.refineDetectedMarkers(
            gray, board, corners, ids, rejected,
            cameraMatrix=K, distCoeffs=dist)
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
        cameraMatrix=K,
        distCoeffs=dist,
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None
    obj_pts = board.chessboardCorners[charuco_ids.flatten(), :]
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, charuco_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), (corners, ids, charuco_corners, charuco_ids)


def draw_axes_bgr(
    img: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    scale: float = 0.05,
    thickness: int = 4,
    warn_prefix: str | None = None,
) -> np.ndarray:
    """在 BGR 图像上绘制物体坐标系 XYZ 轴（OpenCV 风格，X=红，Y=绿，Z=蓝）。"""
    axes = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [scale, 0.0, 0.0, 1.0],
            [0.0, scale, 0.0, 1.0],
            [0.0, 0.0, scale, 1.0],
        ],
        dtype=np.float64,
    )
    pts_cam = (T_cam_obj @ axes.T).T
    if pts_cam[0, 2] <= 1e-6:
        return img
    origin_px = K @ (pts_cam[0, :3] / pts_cam[0, 2])
    origin_px = (int(origin_px[0]), int(origin_px[1]))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z
    skipped = []
    for p, color in zip(pts_cam[1:], colors):
        if p[2] <= 1e-6:
            skipped.append(color)
            continue
        uvw = K @ (p[:3] / p[2])
        pt = (int(uvw[0]), int(uvw[1]))
        cv2.arrowedLine(img, origin_px, pt, color, thickness, tipLength=0.25)
        cv2.circle(img, pt, 4, color, -1)
    if skipped and warn_prefix:
        color_map = {(0, 0, 255): "X(red)", (0, 255, 0): "Y(green)", (255, 0, 0): "Z(blue)"}
        print(
            f"{warn_prefix}: 轴未绘制 {', '.join(color_map[c] for c in skipped)} (Z<=0 或投影失败)")
    return img


def ref_gripper_transforms(end_pos: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (R_g2ref, t_g2ref, R_ref2g, t_ref2g)。
    end_pos 记录的是“gripper 在 ref 下的位姿”（常见的基坐标系下末端姿态），因此需要取逆得到 gripper->ref。
    """
    pose = np.asarray(end_pos, dtype=np.float64).flatten()
    R_ref2g = rpy_to_matrix(pose[3], pose[4], pose[5])
    t_ref2g = pose[:3]
    R_g2ref = R_ref2g.T
    t_g2ref = -R_g2ref @ t_ref2g
    return R_g2ref, t_g2ref, R_ref2g, t_ref2g


def project_ref_point_to_image(
    img: np.ndarray,
    point_ref: np.ndarray,
    R_ref2cam: np.ndarray,
    t_ref2cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int] | None, bool]:
    """将 ref 坐标系下的 3D 点投影到图像；返回绘制后的图、像素坐标及是否发生过边框填充。"""
    point_cam = R_ref2cam @ point_ref.reshape(3, 1) + t_ref2cam.reshape(3, 1)
    if point_cam[2, 0] <= 1e-6:
        return img, None, False
    uvw = K @ (point_cam[:3] / point_cam[2, 0])
    u, v = int(uvw[0]), int(uvw[1])
    H, W = img.shape[:2]
    pad_top = max(0, -v)
    pad_left = max(0, -u)
    pad_bottom = max(0, v - (H - 1))
    pad_right = max(0, u - (W - 1))
    padded = bool(pad_top or pad_bottom or pad_left or pad_right)
    if pad_top or pad_bottom or pad_left or pad_right:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        u += pad_left
        v += pad_top
    cv2.circle(img, (u, v), 10, (255, 0, 255), -1)
    return img, (u, v), padded


def main():
    parser = argparse.ArgumentParser(
        description="眼在手外数据解析（ChArUco 版本，仅输出可知的 R/T）")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("right_calibration_data"), help="采集数据目录"
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("instrinsics_camerah.json"),
        help="相机内参（json 或 xml），默认使用 right4camerah.json",
    )
    parser.add_argument(
        "--camera-label",
        type=str,
        default="cam_h",
        help="相机标签（如 h/l/r），用于输出文件名和键名",
    )
    parser.add_argument(
        "--charuco-squares-x",
        type=int,
        default=6,
        help="ChArUco 方格列数（X 方向，总格子数，不是内角点数）",
    )
    parser.add_argument(
        "--charuco-squares-y",
        type=int,
        default=5,
        help="ChArUco 方格行数（Y 方向，总格子数，不是内角点数）",
    )
    parser.add_argument(
        "--square-length",
        type=float,
        default=0.03,
        help="ChArUco 单个方格边长（米）",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.02,
        help="ChArUco 内部 ArUco marker 边长（米），需小于 square-length",
    )
    parser.add_argument(
        "--anchor-marker-id",
        type=int,
        default=None,
        help="将坐标系原点移动到指定 marker 的中心（保持与板一致的朝向），默认不移动",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="ArUco 字典名称（例如 DICT_4X4_50, DICT_5X5_100 等）",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="right",
        help="使用哪个末端的姿态（与采集时一致），默认 right",
    )
    args = parser.parse_args()

    dictionary = get_aruco_dictionary(args.aruco_dict)
    board = cv2.aruco.CharucoBoard_create(
        squaresX=args.charuco_squares_x,
        squaresY=args.charuco_squares_y,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=dictionary,
    )
    anchor_center = None
    payload_anchor = None
    if args.anchor_marker_id is not None:
        board_ids = board.ids.flatten().tolist()
        if args.anchor_marker_id not in board_ids:
            print(
                f"警告：anchor_marker_id={args.anchor_marker_id} 不在当前板的 ID 列表 {board_ids} 中，忽略平移。")
        else:
            idx = board_ids.index(args.anchor_marker_id)
            anchor_corners = np.asarray(
                board.objPoints[idx], dtype=np.float64).reshape(4, 3)
            anchor_center = anchor_corners.mean(axis=0)
            payload_anchor = {
                "anchor_marker_id": args.anchor_marker_id,
                "anchor_center_board": anchor_center.tolist(),
            }

    K, dist = load_intrinsics(args.intrinsics)

    sample_metas = sorted((args.data_dir).glob("sample_*/meta.json"))
    if not sample_metas:
        raise RuntimeError(f"未在 {args.data_dir} 找到 sample_*/meta.json")
    total_samples = len(sample_metas)
    detected_samples = 0
    detected_corners = 0

    records: list[Dict] = []
    R_t2c_all: list[np.ndarray] = []
    t_t2c_all: list[np.ndarray] = []
    R_g2r_all: list[np.ndarray] = []
    t_g2r_all: list[np.ndarray] = []
    R_r2g_all: list[np.ndarray] = []
    t_r2g_all: list[np.ndarray] = []
    vis_records: list[tuple[Path, np.ndarray]] = []

    for meta_path in sample_metas:
        meta = json.loads(meta_path.read_text())
        sample_dir = meta_path.parent
        end_pos = meta.get("end_pos")
        if end_pos is None:
            print(f"{meta_path} 缺少 end_pos，跳过")
            continue
        img_pick = pick_image(meta, sample_dir)
        if img_pick is None:
            print(f"{meta_path} 无可用相机帧，跳过")
            continue
        img_path, img = img_pick
        cam_key = img_path.stem
        det = detect_charuco(
            img,
            K,
            dist,
            board=board,
            dictionary=dictionary,
        )
        if det is None:
            print(f"{meta_path} [{cam_key}] 未检测到 ChArUco 角点，跳过")
            continue

        R_t2c, t_t2c, det_info = det  # board->cam
        corners, ids, charuco_corners, charuco_ids = det_info
        t_t2c = np.asarray(t_t2c, dtype=np.float64).reshape(3)
        # 可选：将原点移到 anchor marker 中心，保持旋转不变
        if anchor_center is not None:
            t_t2c = (R_t2c @ anchor_center + t_t2c).reshape(3)
        debug_dir = sample_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(
            dbg, charuco_corners, charuco_ids, (0, 255, 0))
        cv2.imwrite(str(debug_dir / f"{cam_key}_charuco.jpg"), dbg)
        T_cam_target = np.eye(4, dtype=np.float64)
        T_cam_target[:3, :3] = R_t2c
        T_cam_target[:3, 3] = t_t2c
        axes_vis = draw_axes_bgr(img.copy(), T_cam_target, K, scale=0.05)
        cv2.imwrite(str(debug_dir / f"{cam_key}_axes.jpg"), axes_vis)

        R_g2ref, t_g2ref, R_ref2g, t_ref2g = ref_gripper_transforms(end_pos)
        t_g2ref = np.asarray(t_g2ref, dtype=np.float64).reshape(3)

        records.append({
            "sample_dir": str(sample_dir),
            "image": str(img_path),
            "camera_key": cam_key,
            "R_target2cam": R_t2c.tolist(),
            "t_target2cam": t_t2c.tolist(),
            "R_gripper2ref": R_g2ref.tolist(),
            "t_gripper2ref": t_g2ref.tolist(),
            "R_ref2gripper": R_ref2g.tolist(),
            "t_ref2gripper": t_ref2g.tolist(),
            "charuco_corner_count": len(charuco_corners) if charuco_corners is not None else 0,
            "charuco_ids_detected": charuco_ids.flatten().tolist() if charuco_ids is not None else [],
            "charuco_squares_x": args.charuco_squares_x,
            "charuco_squares_y": args.charuco_squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "aruco_dict": args.aruco_dict,
        })
        R_t2c_all.append(R_t2c)
        t_t2c_all.append(t_t2c)
        R_g2r_all.append(R_g2ref)
        t_g2r_all.append(t_g2ref)
        R_r2g_all.append(R_ref2g)
        t_r2g_all.append(t_ref2g)
        vis_records.append((img_path, t_ref2g))
        detected_samples += 1
        detected_corners += len(charuco_corners) if charuco_corners is not None else 0

    if len(R_g2r_all) < 2:
        raise RuntimeError("有效样本不足（<2），无法估计相机外参")

    R_cam2ref, t_cam2ref = cv2.calibrateHandEye(
        R_g2r_all,
        t_g2r_all,
        R_t2c_all,
        t_t2c_all,
    )
    t_cam2ref = np.asarray(t_cam2ref, dtype=np.float64).reshape(3)
    R_cam2ref = np.asarray(R_cam2ref, dtype=np.float64)
    R_ref2cam = R_cam2ref.T
    t_ref2cam = -R_ref2cam @ t_cam2ref
    T_ref2cam = np.eye(4, dtype=np.float64)
    T_ref2cam[:3, :3] = R_ref2cam
    T_ref2cam[:3, 3] = t_ref2cam
    T_cam2ref = np.eye(4, dtype=np.float64)
    T_cam2ref[:3, :3] = R_cam2ref
    T_cam2ref[:3, 3] = t_cam2ref

    failed_samples = total_samples - detected_samples
    out_path = Path(f"extrinsics_charuco_{args.camera_label}_{args.side}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_label": args.camera_label,
        "side": args.side,
        "R_cam2ref": R_cam2ref.tolist(),
        "t_cam2ref": t_cam2ref.tolist(),
        "R_ref2cam": R_ref2cam.tolist(),
        "t_ref2cam": t_ref2cam.tolist(),
        "T_cam2ref": T_cam2ref.tolist(),
        "T_ref2cam": T_ref2cam.tolist(),
        "charuco": {
            "squares_x": args.charuco_squares_x,
            "squares_y": args.charuco_squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "aruco_dict": args.aruco_dict,
        },
    }
    if payload_anchor:
        payload.update(payload_anchor)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("标定完成：")
    print(
        f"样本总数: {total_samples}, 成功检测: {detected_samples}, 失败: {failed_samples}, 总 ChArUco 角点数: {detected_corners}")
    print(f"结果已保存到 {out_path}")
    print("齐次矩阵字段已写入 JSON: T_ref2cam, T_cam2ref")
    for img_path, t_ref2g in vis_records:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_vis, uv, padded = project_ref_point_to_image(
            img, t_ref2g, R_ref2cam, t_ref2cam, K
        )
        vis_dir = img_path.parent / "visual"
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{img_path.stem}_ref_origin.jpg"), img_vis)
        msg = f"{img_path}: ref->gripper 平移投影到 {uv}" if uv else f"{img_path}: ref->gripper 平移未能投影"
        if uv and padded:
            msg += "（已添加黑边）"
        print(msg)


if __name__ == "__main__":
    main()
