#!/usr/bin/env python3
"""
用 MoveItPy 做目标位姿规划，起点固定为 base_link 下 [0,0,0,0,0,0,1]，
目标固定为 z 抬高 0.1 的 [0,0,0.1,0,0,0,1]，只打印轨迹。
需要提前启动你的 lift2 MoveIt 运行环境（joint_states/TF/控制器）。
"""
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder


def build_moveit_config_dict():
    """构造 moveit_py 所需的扁平参数字典。"""
    moveit_config = (
        MoveItConfigsBuilder(
            "lift2",
            package_name="lift2_moveit_config",
        )
        .robot_description(file_path="config/lift2.urdf.xacro")
        .robot_description_semantic(file_path="config/lift2.srdf")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )

    config = moveit_config.to_dict()
    # rclcpp 需要扁平键，这里显式指定 OMPL 规划管线参数
    config["planning_pipelines.pipeline_names"] = ["ompl"]
    config["planning_pipelines.namespace"] = ""
    config["ompl.planning_plugin"] = "ompl_interface/OMPLPlanner"
    config["ompl.request_adapters"] = [
        "default_planning_request_adapters/ResolveConstraintFrames",
        "default_planning_request_adapters/ValidateWorkspaceBounds",
        "default_planning_request_adapters/CheckStartStateBounds",
        "default_planning_request_adapters/CheckStartStateCollision",
    ]
    config["ompl.response_adapters"] = [
        "default_planning_response_adapters/AddTimeOptimalParameterization",
        "default_planning_response_adapters/ValidateSolution",
        "default_planning_response_adapters/DisplayMotionPath",
    ]
    config["ompl.start_state_max_bounds_error"] = 0.1
    return config


def plan_and_output(moveit: MoveItPy, group_name: str, pose_link: str, target: PoseStamped):
    """规划单个规划组并打印轨迹。"""
    pc = moveit.get_planning_component(group_name)  # 取对应规划组件
    pc.set_start_state_to_current_state()  # 起点设为当前状态
    pc.set_goal_state(pose_stamped_msg=target, pose_link=pose_link)  # 设末端位姿目标

    params = PlanRequestParameters(moveit, group_name)  # 规划参数对象
    params.planning_pipeline = "ompl"  # 选用 OMPL 管线

    result = pc.plan(params)
    if not result:
        print(f"[{group_name}] 规划失败")
        return False

    print(f"[{group_name}] 规划成功: planner={result.planner_id} time={result.planning_time:.3f}s")

    jt = result.trajectory.joint_trajectory  # 取关节轨迹
    names = list(jt.joint_names)  # 关节名顺序
    print(f"[{group_name}] 轨迹点数: {len(jt.points)} (关节顺序: {names})")
    for idx, point in enumerate(jt.points):  # 逐点打印关节角
        positions = [round(p, 5) for p in point.positions]
        print(f"  [{group_name} #{idx:02d}] {positions}")

    return True


def make_pose(vals):
    """将 [x y z qx qy qz qw] 转成 Pose。"""
    if len(vals) != 7:
        raise ValueError("位姿需要 7 个数：x y z qx qy qz qw")
    p = Pose()
    p.position.x, p.position.y, p.position.z = vals[0:3]
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = vals[3:7]
    return p


def set_start_from_pose(moveit: MoveItPy, group_name: str, pose_link: str, pose: Pose):
    """
    尝试用 IK 把给定末端位姿设置为起始状态，返回 (是否成功, RobotState 或 None)。
    """
    psm = moveit.get_planning_scene_monitor()
    psm.update_frame_transforms()
    robot_model = moveit.get_robot_model()
    jmg = robot_model.get_joint_model_group(group_name)
    if jmg is None:
        print(f"[{group_name}] 找不到关节组")
        return False, None

    with psm.read_write() as scene:
        rs = scene.current_state
        ok = rs.set_from_ik(jmg, pose, pose_link)
        if not ok:
            return False, None
        return True, rs


def main():
    rclpy.init()  # 初始化 ROS2
    config_dict = build_moveit_config_dict()  # 获取参数
    moveit = MoveItPy(node_name="moveit_py_all_mp",
                      config_dict=config_dict)  # 启动 MoveItCpp 节点

    group_name = "left_arm"
    pose_link = "left_link16"
    frame = "base_link"

    # 固定起点/终点位姿
    start_pose = make_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    target_pose = make_pose([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])

    # 起点：尝试 IK，不行就用当前状态
    start_state = None
    ok, rs = set_start_from_pose(moveit, group_name, pose_link, start_pose)
    if ok:
        start_state = rs
        print(f"[{group_name}] 起点已通过 IK 设置为全 0 位姿")
    else:
        print(f"[{group_name}] 起点 IK 失败，改用当前状态")

    # 目标位姿
    target = PoseStamped()
    target.header.frame_id = frame
    target.pose = target_pose

    pc = moveit.get_planning_component(group_name)
    if start_state:
        pc.set_start_state(start_state)
    else:
        pc.set_start_state_to_current_state()
    pc.set_goal_state(pose_stamped_msg=target, pose_link=pose_link)

    params = PlanRequestParameters(moveit, group_name)
    params.planning_pipeline = "ompl"
    result = pc.plan(params)
    if not result:
        print(f"[{group_name}] 规划失败")
        moveit.shutdown()
        rclpy.shutdown()
        return

    print(f"[{group_name}] 规划成功: planner={result.planner_id} time={result.planning_time:.3f}s")
    jt = result.trajectory.joint_trajectory
    names = list(jt.joint_names)
    print(f"[{group_name}] 轨迹点数: {len(jt.points)} (关节顺序: {names})")
    for idx, point in enumerate(jt.points):
        positions = [round(p, 4) for p in point.positions]
        print(f"  [{group_name} #{idx:02d}] {positions}")

    moveit.shutdown()  # 关闭 MoveItCpp 节点
    rclpy.shutdown()  # 关闭 ROS2


if __name__ == "__main__":
    main()
