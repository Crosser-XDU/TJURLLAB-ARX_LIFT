#!/usr/bin/env python3
"""
Code snippet commanding MoveIt planning and execution on Tiago from Python

# Reminder of available MoveItPy functions:
#   print(dir(robot_state)) # 'clear_attached_bodies', 'dirty', 'get_frame_transform', 'get_global_link_transform', 'get_jacobian', 'get_joint_group_accelerations', 'get_joint_group_positions', 'get_joint_group_velocities', 'get_pose', 'joint_accelerations', 'joint_efforts', 'joint_positions', 'joint_velocities', 'robot_model', 'set_from_ik', 'set_joint_group_accelerations', 'set_joint_group_active_positions', 'set_joint_group_positions', 'set_joint_group_velocities', 'set_to_default_values', 'set_to_random_positions', 'state_info', 'state_tree', 'update'
#   print(dir(tiago_arm))   # 'get_named_target_state_values', 'get_start_state', 'named_target_states', 'plan', 'planning_group_name', 'set_goal_state', 'set_path_constraints', 'set_start_state', 'set_start_state_to_current_state', 'set_workspace', 'unset_workspace'
#   print(dir(tiago))       # 'execute', 'get_planning_component', 'get_planning_scene_monitor', 'get_robot_model', 'get_trajectory_execution_manager', 'shutdown'
#   print(dir(plan_result)) # 'error_code', 'planner_id', 'planning_time', 'start_state', 'trajectory'
"""
import rclpy
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped


class MoveItExecutorNode(Node):
    def __init__(self):
        super().__init__('moveit_executor_node')
        self.logger = self.get_logger()
        self.logger.info("Initializing MoveItPy and executing motion...")
        
        # Initialize MoveItPy instance (already manages its own executor)
        self.tiago = MoveItPy(node_name="moveit_py_pick")
        self.robot_model = self.tiago.get_robot_model()
        self.tiago_arm = self.tiago.get_planning_component("arm")
        self.logger.info("MoveItPy connected!")

        # Store joint values of open/close gripper positions
        self.gripper_open_joints = {"left_finger": 0.04, "right_finger": 0.04}
        self.gripper_closed_joints = {"left_finger": 0.0, "right_finger": 0.0}


    
    def plan1(self):
        """
        Plan 1 - set goal state with RobotState object (using its joint states)
        """
        # First go to the joint state with default values (zero angles)
        robot_initial_state = RobotState(self.robot_model)
        robot_state = RobotState(self.robot_model)
        robot_state.set_to_default_values()
        self.tiago_arm.set_start_state_to_current_state()
        self.logger.info("Set goal state to the initialized robot state")
        self.tiago_arm.set_goal_state(robot_state=robot_state)
        plan_result = self.tiago_arm.plan()
        if plan_result:
            self.logger.info("Executing plan")
            self.tiago.execute(plan_result.trajectory, controllers=[])
        else:
            self.logger.error("Planning failed")
        
        # Then go back to the initial state
        self.tiago_arm.set_start_state_to_current_state()
        self.tiago_arm.set_goal_state(robot_state=robot_initial_state)
        plan_result = self.tiago_arm.plan()
        if plan_result:
            self.logger.warning("Moving arm to the joints goal (RobotState goal)...")
            self.tiago.execute(plan_result.trajectory, controllers=[])
    
    def plan2(self):
        """
        Plan 2 - set cartesian goal for the gripper with PoseStamped message
        """
        self.tiago_arm.set_start_state_to_current_state()
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = 0.7
        pose_goal.pose.position.y = 0.1
        pose_goal.pose.position.z = 0.4
        pose_goal.pose.orientation.x = 0.0
        pose_goal.pose.orientation.y = 0.0
        pose_goal.pose.orientation.z = 0.0
        pose_goal.pose.orientation.w = 1.0
        self.tiago_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="gripper_grasping_frame")
        plan_result = self.tiago_arm.plan()
        if plan_result:
            self.logger.warning("Moving arm to the cartesian goal...")
            self.tiago.execute(plan_result.trajectory, controllers=[])
    
    def plan3(self):
        """
        Plan 3 - Open and close gripper
        """
        # First open fingers
        tiago_gripper = self.tiago.get_planning_component("gripper")
        gripper_open = RobotState(self.robot_model)
        gripper_open.set_joint_group_positions("gripper", list(self.gripper_open_joints.values()))
        tiago_gripper.set_goal_state(robot_state=gripper_open)
        gripper_opening_result = tiago_gripper.plan()
        if gripper_opening_result:
            self.logger.warning("Opening Tiago's gripper...")
            self.tiago.execute(gripper_opening_result.trajectory, controllers=[])
        # Now close fingers
        gripper_closed = RobotState(self.robot_model)
        gripper_closed.set_joint_group_positions("gripper", list(self.gripper_closed_joints.values()))
        tiago_gripper.set_goal_state(robot_state=gripper_closed)
        gripper_closing_result = tiago_gripper.plan()
        if gripper_closing_result:
            self.logger.warning("Closing Tiago's gripper...")
            self.tiago.execute(gripper_closing_result.trajectory, controllers=[])

def main():
    rclpy.init()
    node = MoveItExecutorNode()
    # Uncomment one of the following function calls to execute the respective plan
    node.plan1()
    # node.plan2()
    # node.plan3()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()