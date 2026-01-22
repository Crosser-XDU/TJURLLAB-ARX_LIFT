workspace=$(pwd)
source ~/.bashrc

# 中间相机拉起
gnome-terminal -t "h_camera" -x  bash -c "cd ${workspace}; cd .. ; cd realsense; \
  source install/setup.bash && ros2 launch realsense2_camera rs_launch.py\
  align_depth.enable:=true \
  pointcloud.enable:=true\
  publish_tf:=true \
  tf_publish_rate:=50.0 \
  camera_name:=camera_h \
  camera_namespace:=camera_h_namespace \
  serial_no:=_409122274317 \
  depth_module.color_profile:=640x480x90 \
  depth_module.depth_profile:=640x480x90; exec bash;"
