from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym import spaces


class BaseRobotEnv(gym.Env):
    """
    Abstract Base Class for Robot Control using OpenAI Gym.
    This class handles the interface between RL algorithms and ROS hardware/simulation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the environment parameters.
        """
        super(BaseRobotEnv, self).__init__()

        self._setup_spaces()

    def _setup_spaces(self):
        """
        Configure Action and Observation spaces separately.
        Standardizes the input/output shapes for agents.
        """

        # EXAMPLE:

        """
        # 1. Action Space: Delta x, y, z, roll, pitch, yaw, and gripper opening
        # Shape: (7,)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # 2. Observation Space:
        # Two cameras: [camera_0, camera_1]
        # x, y, z, roll, pitch, yaw, gripper opening
        # Shape: (14,)
        observation_space = dict()
        observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        observation_space["camera_0"] = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        observation_space["camera_1"] = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(observation_space)
        """
        raise NotImplementedError

    def reset(self, **kwargs: Any) -> Dict[str, np.ndarray]:
        """
        Resets the robot to the initial pose.

        Returns:
            observation (Dict[str, np.ndarray]): The initial state of the robot.
        """

        # 1. Enable the robot
        success, error_message = self._enable_robot()
        if not success:
            raise RuntimeError(f"Failed to enable the robot: {error_message}")

        # 2. Go to the initial pose
        success, error_message = self._go_to_initial_pose()
        if not success:
            raise RuntimeError(
                f"Failed to go to the initial pose: {error_message}")

        # 3. Get the initial observation
        obs = self._get_observation()
        return obs

    def step(self, action):
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
        obs = self._get_observation()

        # 3. Reward Calculation
        reward = self._get_reward(obs, action)

        # 4. Termination Logic
        is_done = self._get_termination(obs, action)

        # 5. Metadata
        info = self._get_info()

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

    # ==========================================================
    # Private / Abstract Methods (To be implemented by Subclass)
    # ==========================================================

    def _enable_robot(self) -> Tuple[bool, str | None]:
        """
        Enables the robot.

        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is enabled, (False, error_message) otherwise.
        """
        raise NotImplementedError

    def _disable_robot(self) -> Tuple[bool, str | None]:
        """
        Disables the robot.

        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is disabled, (False, error_message) otherwise.
        """
        raise NotImplementedError

    def _go_to_initial_pose(self) -> Tuple[bool, str | None]:
        """
        Goes to the initial pose.

        Returns:
            Tuple[bool, str | None]: (True, None) if the robot is at the initial pose, (False, error_message) otherwise.
        """
        raise NotImplementedError

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Polls ROS topics for sensor data.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the observation.
        """
        raise NotImplementedError

    def _apply_action(self, action: np.ndarra, **kwargs) -> Tuple[bool, str | None]:
        """
        Converts action to ROS messages and executes them.

        Args:
            action (np.ndarray): Action provided by the agent. Shape (action_dim,).
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple[bool, str | None]: (True, None) if the action is applied, (False, error_message) otherwise.
        """
        raise NotImplementedError

    def _get_reward(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
        """
        Calculates the reward (e.g., based on distance to target).

        Args:
            obs (Dict[str, np.ndarray]): Dictionary containing the observation.
            action (np.ndarray): Action provided by the agent. Shape (action_dim,).

        Returns:
            float: Reward value.
        """
        raise NotImplementedError

    def _get_termination(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> bool:
        """
        Checks for success conditions or safety violations (collisions).

        Args:
            obs (Dict[str, np.ndarray]): Dictionary containing the observation.
            action (np.ndarray): Action provided by the agent. Shape (action_dim,).

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        raise NotImplementedError

    def _get_info(self) -> Dict[str, Any]:
        """
        Returns extra metadata for debugging.

        Returns:
            Dict[str, Any]: Dictionary containing the extra metadata.
        """
        raise NotImplementedError
