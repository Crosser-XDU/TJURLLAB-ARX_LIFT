gnome-terminal -t "state_pub" -- bash -c 'ros2 run robot_state_publisher robot_state_publisher /home/arx/Robotbase_yxz/assets/urdf/lift2.urdf; exec bash' 

sleep 1

gnome-terminal -t "joint_pub" -- bash -c 'ros2 run joint_state_publisher_gui joint_state_publisher_gui; exec bash'  --title="rviz2" -- bash -ic 'source ~/Robotbase_yxz/LIFT/body/ROS2/install/setup.bash; rviz2; exec bash'

rviz2