#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from moveit.planning import PlanRequestParameters
from moveit.planning import PlanningComponent
from moveit.planning import TrajectoryExecutionManager
from moveit.planning import MoveItPy
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from builtin_interfaces.msg import Time
from moveit_configs_utils import MoveItConfigsBuilder


def make_box(box_id, size_xyz, pos_xyz):
    co = CollisionObject()
    co.id = box_id
    co.header.frame_id = "base_link"
    co.header.stamp = Time()
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = list(size_xyz)
    co.primitives.append(box)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = pos_xyz
    pose.orientation.w = 1.0
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD
    return co

def plan_and_execute(
        robot,
        planning_component,
        logger,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        ):
        """A helper function to plan and execute a motion."""
        # plan to goal
        logger.info("Planning trajectory")
        if multi_plan_parameters is not None:
                plan_result = planning_component.plan(
                        multi_plan_parameters=multi_plan_parameters
                )
        elif single_plan_parameters is not None:
                plan_result = planning_component.plan(
                        single_plan_parameters=single_plan_parameters
                )
        else:
                plan_result = planning_component.plan()

        # execute the plan
        if plan_result:
                logger.info("Executing plan")
                robot_trajectory = plan_result.trajectory
                robot.execute(robot_trajectory, controllers=[])
        else:
                logger.error("Planning failed")


def main():
    rclpy.init()
    moveit_config = (
        MoveItConfigsBuilder("lift2", package_name="lift_moveit_config")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    ).to_dict()
    config_dict = moveit_config
    # rclcpp 参数是扁平的，这里显式设置规划管线相关的扁平键
    config_dict["planning_pipelines.pipeline_names"] = ["ompl"]
    config_dict["planning_pipelines.namespace"] = ""
    config_dict["ompl.planning_plugin"] = "ompl_interface/OMPLPlanner"
    config_dict["ompl.request_adapters"] = [
        "default_planning_request_adapters/ResolveConstraintFrames",
        "default_planning_request_adapters/ValidateWorkspaceBounds",
        "default_planning_request_adapters/CheckStartStateBounds",
        "default_planning_request_adapters/CheckStartStateCollision",
    ]
    config_dict["ompl.response_adapters"] = [
        "default_planning_response_adapters/AddTimeOptimalParameterization",
        "default_planning_response_adapters/ValidateSolution",
        "default_planning_response_adapters/DisplayMotionPath",
    ]
    config_dict["ompl.start_state_max_bounds_error"] = 0.1

    # Demo模式下不执行轨迹，避免控制器依赖

    moveit = MoveItPy(
        node_name="moveit_py_demo",
        config_dict=config_dict,
    )
    psm = moveit.get_planning_scene_monitor()

    # 添加两个方块 ok的
    with psm.read_write() as scene:
        scene.apply_collision_object(
            make_box("boxr", [0.1, 0.1, 0.1], [0.23, -0.25, 0.59]))
        scene.apply_collision_object(
            make_box("boxl", [0.1, 0.1, 0.1], [0.23, 0.25, 0.59]))
    psm.update_frame_transforms()

    group_name = "double_arm"  # SRDF 中的分组，可换 right_arm/double_arm
    pc = moveit.get_planning_component(group_name)

    # 目标位姿示例
    pc.set_start_state_to_current_state()
    target = PoseStamped()
    target.header.frame_id = "base_link"
    target.pose.position.x = 0.5
    target.pose.position.y = 0.0
    target.pose.position.z = 0.4
    target.pose.orientation.w = 1.0

    # left arm末端link，根据 SRDF 是 left_link16
    pc.set_goal_state(pose_stamped_msg=target, pose_link="left_link16")

    params = PlanRequestParameters(moveit, group_name)
    params.planning_pipeline = "ompl"
    plan_and_execute(
        robot="lift2",
        planning_component=pc,
        logger=moveit.get_logger(),
        single_plan_parameters=params,
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()