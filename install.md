setuptools==59.8.0


cd DREAM_ws/src/stretch_mujoco/third_party/robocasa/
pip install -e .
cd ../robosuit
pip install -e .

python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
python robocasa/scripts/setup_macros.py              # Set up system variables.

cd ../../
pip install -e .  # install stretch_mujoco



https://github.com/hello-robot/stretch_tutorials/blob/2ccee6617dc8bb7bfa5b2c8438185dec5f99e053/ros2/remote_compute.md



```bash
# mkdir -p ~/ament_ws/src
# cd ~/ament_ws/src/
git clone https://github.com/hello-robot/stretch_ros2
git clone https://github.com/hello-binit/ros2_numpy -b humble
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master  # default ros2-development
# git clone https://github.com/Slamtec/sllidar_ros2.git -b main
# git clone https://github.com/hello-binit/respeaker_ros2.git -b humble
# git clone https://github.com/hello-binit/audio_common.git -b humble
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
git clone https://github.com/Ericsii/FAST_LIO_ROS2.git --recursive
git clone https://github.com/westonrobot/ugv_sdk.git
git clone https://github.com/westonrobot/ranger_ros2.git
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b humbleccc

git clone https://github.com/introlab/rtabmap.git
git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git


# Make sure to uninstall any rtabmap binaries:
sudo apt remove ros-$ROS_DISTRO-rtabmap*

git clone https://github.com/introlab/rtabmap.git
git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git


```


```bash
cd livox_ros_driver2
cp package_ROS2.xml package.xml
cp -rf launch_ROS2/ launch/

cd ../..
source /opt/ros/humble/setup.bash
rosdep install --rosdistro=humble -iyr --skip-keys="librealsense2" --from-paths 
rosdep update && rosdep install --from-paths src --ignore-src -r -y

source /opt/ros/humble/setup.bash
export MAKEFLAGS="-j3" # Can be ignored if you have a lot of RAM (>16GB)
colcon build --symlink-install --cmake-args -DHUMBLE_ROS=humble -DCMAKE_BUILD_TYPE=Release

# To build with rgbd_cameras>1 support and/or subscribe_user_data support:
colcon build --symlink-install --cmake-args -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -DCMAKE_BUILD_TYPE=Release

# cd src/livox_ros_driver2
# rm package.xml
# rm -rf launch/
cd ../..
```

anygrasp environment setup (cuda-12.1)
```bash
git clone git@github.com:BJHYZJ/DREAM.git
# 更新submodule
git submodule update --remote --recursive

export PYTHONNOUSERSITE=1
conda create -n anygrasp python=3.10 -y
conda activate anygrasp
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install ipython cmake pybind11 ninja scipy==1.10.1 scikit-learn==1.4.0 pandas==2.0.3 hydra-core opencv-python openai-clip timm matplotlib==3.7.2 imageio timm open3d numpy-quaternion more-itertools pyliblzfse einops transformers pytorch-lightning wget gdown tqdm zmq torch_geometric numpy==1.23.0  # -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install protobuf==3.19.0

conda install "setuptools <65"
pip install git+https://github.com/pccws/MinkowskiEngine  # -i https://pypi.tuna.tsinghua.edu.cn/simple

# if you meet ` No module named 'distutils.msvccompiler` error, use: conda install "setuptools <65" 
# pip install graspnetAPI
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True  # reference https://github.com/graspnet/graspnetAPI/issues/43
pip install git+https://github.com/graspnet/graspnetAPI.git

pip install git+https://github.com/luca-medeiros/lang-segment-anything.git  # -i https://pypi.tuna.tsinghua.edu.cn/simple

rerun-sdk==0.26.1
pip install numpy==1.23.5
pip install pip install transforms3d==0.3.1

# set lib_cxx.so and gsnet.so
```bash
# set {python version} to 310 when you use python==3.10
cp third_party/anygrasp_sdk/grasp_detection/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so src/anygrasp_manipulation/gsnet.so
cp third_party/anygrasp_sdk/license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so src/anygrasp_manipulation/lib_cxx.so

```bash
cd src/anygrasp_manipulation/pointnet2
pip install -e .
```



# use anygrasp
get computer id
```bash
cd src/anygrasp_manipulation
./anygrasp_license_registration/license_checker -f  
```
get license from [src/anygrasp_manipulation/anygrasp_license_registration/README.md](src/anygrasp_manipulation/anygrasp_license_registration/README.md)

put license to `src/anygrasp_manipulation/license`
and copy anygrasp checkpoints to ./checkpoints/anygrasp

You can check license states via
```bash
./anygrasp_license_registration/license_checker -c license/licenseCfg.json
```

run anygrasp_manipulation
```bash
conda activate anygrasp
python demo.py --open_communication --port 5557
```


# install Segment-anything-2
```bash
git submodule add https://github.com/facebookresearch/sam2.git third_party/segment-anything-2
cd third_party/segment-anything-2
pip install -e .
pip install numpy==1.23.5  # numpy的版本会被覆盖，重新安装一次
```


```bash
# dream 环境的构建
# 1) 给该环境永久关闭用户 site-packages 注入
conda activate dream
conda env config vars set PYTHONNOUSERSITE=1

# 让设置生效
conda deactivate && conda activate dream

# 2) 验证：应当看到 False，且 sys.path 里不再有 ~/.local/...
python -c "import site, sys; print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('\n'.join(sys.path))"
```




```bash
# Terminal 1: Start the Stretch Driver Node
ros2 launch stretch_core stretch_driver.launch.py
# Terminal 2: Start the realsense D435i stream.
ros2 launch stretch_core d435i_high_resolution.launch.py
# Terminal 3: Start lidar.
ros2 launch stretch_core rplidar.launch.py
```



```bash
ros2 daemon stop
ros2 daemon start
# kill node by node name
ps -ef | grep camera | grep -v grep
ps -ef | grep d435i_accel_correction_node
ps -ef | grep spawner_xarm6_traj_controller
ps -ef | grep ros2
pgrep -a -f 'realsense|d435|d435i|accel|component_container|camera|xarm|livox|robot_description|ros2 launch|joint_states|parametr_events'

pkill -9 -f ros2
pkill -9 -f rviz
pkill -9 -f move_group

pkill -f "ros2 launch"

ros2 lifecycle nodes
ros2 lifecycle set /d435i_accel_correction_node shutdown



ps aux | grep ros2
ros2 daemon stop && sleep 2 && ros2 daemon start


ps aux | grep -E "(ufactory|xarm|ranger)" | grep -v grep


pkill -9 static_transform_publisher
pkill -9 -f static

pkill -9 -f ranger_base_node 强制清除node


# 查看 topic 下的 某个key的数据
ros2 topic echo /rtabmap_ranger_xarm/mapData --field graph --once
ros2 topic echo /rtabmap_ranger_xarm/mapData --field nodes --once
# ros2 interface 用于查看操作消息接口定义的工具
ros2 interface list | grep rtabmap_msgs
ros2 interface show rtabmap_msgs/msg/MapData


# 查看带宽是实际内存占用大小
ros2 topic bw /rtabmap_ranger_xarm/mapData
# 输出里会给你：
  # 平均频率 (Hz)
  # 平均带宽 (MB/s)
  # 平均消息大小 (kB 或 MB)


# 只看节点数量
ros2 topic echo /rtabmap/mapData --once | grep -c "id:"
# 只看 graph 的部分
ros2 topic echo /rtabmap_rangerrtabmap_xarm/mapData --once --field graph
# 只看 nodes 数组的长度（ROS 2 Iron 之后的 colcon-extensions支持 --field，更早版本要写脚本）
ros2 topic echo /rtabmap/mapData --once --field nodes[0].id

# 用 ros2 interface查看节点结构
ros2 interface show rtabmap_msgs/msg/MapData
# 你会看到：
  # std_msgs/Header header
  # rtabmap_msgs/MapGraph graph
  # rtabmap_msgs/Node[] nodes
# 再逐步展开
ros2 interface show rtabmap_msgs/msg/MapGraph
ros2 interface show rtabmap_msgs/msg/Node



# 查看rtabmap数据库中的信息

rtabmap-info ~/.ros/rtabmap.db 

```




```bash
ros2 service call /save_map nav2_msgs/srv/SaveMap "{map_url: 'orbslam3_map'}"
ros2 service call /load_map nav2_msgs/srv/LoadMap "{map_url: 'orbslam3_map'}"
```


view tf frames
```bash
ros2 run tf2_tools view_frames


# vie params from rtqbmap
rtabmap --params | grep Grid
ros2 run rtabmap_slam rtabmap --params

# 查看frame之间的转换
ros2 run tf2_ros tf2_echo livox_frame base_link
```


start ranger

```bash
sudo apt install -y can-utils
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000
````

```bash
# rtabmap config in ~/.bashrc


# RTAM-Map
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
# Optional, but if you like colored logs:
export RCUTILS_COLORIZED_OUTPUT=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp     
# Cyclone prefers multicast by default, if your router got too much spammed,
# disable multicast with (https://github.com/ros2/rmw_cyclonedds/issues/489):
export CYCLONEDDS_URI="<Disc><DefaultMulticastAddress>0.0.0.0</></>"

```



```bash
# 通过LivoxViewer2来修改雷达的范围
https://github.com/Livox-SDK/livox_ros_driver2/issues/199?utm_source=chatgpt.com
# 建议在windows中设置fov，因为ubuntu22.04使用LivoxViewer2会报错，官方也没有给出解释
https://github.com/Livox-SDK/livox_ros_driver2/issues/145
```



从源码构建rtabmap运行是报错：
```bash
[rtabmap-4] terminate called after throwing an instance of 'cv::Exception'
[rtabmap-4]   what():  OpenCV(4.13.0-dev) /home/yanzj/opencv_build/opencv/modules/core/src/alloc.cpp:73: error: (-4:Insufficient memory) Failed to allocate 126573934692672 bytes in function 'OutOfMemoryError'
[rtabmap-4] 
[ERROR] [rtabmap-4]: process has died [pid 31704, exit code -6, cmd '/home/yanzj/DREAM_ws/DREAM_ws/install/rtabmap_slam/lib/rtabmap_slam/rtabmap -d --ros-args -r __ns:=/rtabmap_ranger_xarm --params-file /tmp/launch_params_tnddmdv1 --params-file /tmp/launch_params_xw0tduiw --params-file /tmp/launch_params_ljkfxpl2 -r odom:=/fast_lio2/Odometry -r imu:=imu/data -r rgb/image:=/camera/color/image_raw -r rgb/camera_info:=/camera/color/camera_info -r depth/image:=/camera/aligned_depth_to_color/image_raw -r scan_cloud:=/fast_lio2/cloud_registered_body'].
```

解决方案：
```bash
https://github.com/introlab/rtabmap_ros/issues/917
https://github.com/introlab/rtabmap_ros/issues/1295

```




""" BUG
[ERROR] [planner_server-5]: process has died [pid 176791, exit code -6, cmd '/opt/ros/humble/lib/nav2_planner/planner_server --ros-args -r __node:=planner_server --params-file /tmp/tmp0hsm3e3s -r /tf:=tf -r /tf_static:=tf_static'].
```bash

ros2 run nav2_planner planner_server \
  --ros-args \
  --params-file /home/yanzj/DREAM_ws/DREAM_ws/src/dream_ros2_bridge/config/nav2_params.yaml \
  -r /tf:=tf -r /tf_static:=tf_static \
  --log-level DEBUG \
  --log-level nav2_planner:=DEBUG \
  --log-level nav2_costmap_2d:=DEBUG \
  --log-level pluginlib.ClassLoader:=DEBUG
```
log:

[INFO] [1757588886.632799941] [global_costmap.global_costmap]: Creating Costmap
terminate called after throwing an instance of 'rclcpp::exceptions::InvalidParameterTypeException'
  what():  parameter 'height' has invalid type: Wrong parameter type, parameter {height} is of type {integer}, setting it to {double} is not allowed.
[ros2run]: Aborted
""" 


```bash
# 关闭笔记本的usb自动省电功能
sudo vim /etc/default/grub
# 找到：GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
# 在里面加上：
usbcore.autosuspend=-1
# 比如：
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.autosuspend=-1"
# 更新grub并重启
sudo update-grub
sudo reboot
```


```bash
# 查看点云
cloudcompare.CloudCompare map.ply

```

rerun-sdk==0.26.1
numpy==1.24.3


网络端口被占用时

```bash
sudo lsof -i:4403
```

暂停rtabmap
```bash
ros2 service call /rtabmap/rtabmap/pause std_srvs/srv/Empty {}
```

恢复rtabmap
```bash
ros2 service call /rtabmap/rtabmap/resume std_srvs/srv/Empty {}
```



vscode debug for ros2 环境生成
```bash
bash -lc 'source /opt/ros/humble/setup.bash && source ./DREAM_ws/install/setup.bash && env' \
        | awk -F= 'BEGIN{OFS="="} /^[A-Za-z_][A-Za-z0-9_]*=/{gsub(/\0/,"",$2); print $1"="$2}' \
        > .vscode/ros2.env
```

随后用下面json
```bash

{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: run read_mapData_nodes_only.py (ROS2 env)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/read_mapData_nodes_only.py",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.vscode/ros2.env",
      "justMyCode": false
    }
  ]
}

```


rvl深度图形压缩与解压：

https://github.com/introlab/rtabmap/issues/1362



```
ros2 bag record \
  /tf /tf_static \
  /parameter_events \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/color/metadata \
  /camera/aligned_depth_to_color/image_raw \
  /camera/aligned_depth_to_color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/depth/metadata \
  /camera/depth/color/points \
  /camera/accel/sample \
  /camera/accel/imu_info \
  /camera/accel/metadata \
  /camera/gyro/sample \
  /camera/gyro/imu_info \
  /camera/gyro/metadata \
  /camera/extrinsics/depth_to_accel \
  /camera/extrinsics/depth_to_color \
  /camera/extrinsics/depth_to_depth \
  /camera/extrinsics/depth_to_gyro \
  /camera/imu \
  /livox/imu \
  /livox/lidar \
  /ranger/odom \
  /ranger/cmd_vel \
  /ranger/motion_state \
  /ranger/system_state \
  /ranger/battery_state \
  /ranger/rc_state \
  /ranger/actuator_state \
  /joint_states \
  /ranger/joint_states \
  /xarm/joint_states \
  /xarm/robot_states \
  /xarm/uf_ftsensor_ext_states \
  /xarm/uf_ftsensor_raw_states \
  /xarm/vc_set_cartesian_velocity \
  /xarm/vc_set_joint_velocity \
  /robot_description


bag paly
```
ros2 bag play /home/yanzj/DREAM_ws/DREAM_ws/rosbag2_2025_09_15-10_47_03 --clock --rate 1.0
```



rtabmap/mapData
```
std_msgs/Header header
        builtin_interfaces/Time stamp
                int32 sec
                uint32 nanosec
        string frame_id

##################
# Optimized graph
##################
MapGraph graph
        std_msgs/Header header
                builtin_interfaces/Time stamp
                        int32 sec
                        uint32 nanosec
                string frame_id
        ##
        ##
        geometry_msgs/Transform map_to_odom
                Vector3 translation
                        float64 x
                        float64 y
                        float64 z
                Quaternion rotation
                        float64 x 0
                        float64 y 0
                        float64 z 0
                        float64 w 1
        int32[] poses_id
        geometry_msgs/Pose[] poses
                Point position
                        float64 x
                        float64 y
                        float64 z
                Quaternion orientation
                        float64 x 0
                        float64 y 0
                        float64 z 0
                        float64 w 1
        Link[] links
                int32 from_id
                int32 to_id
                int32 type
                geometry_msgs/Transform transform
                        Vector3 translation
                                float64 x
                                float64 y
                                float64 z
                        Quaternion rotation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                float64[36] information

##################
# Graph data
##################
Node[] nodes
        int32 id
        int32 map_id
        int32 weight
        float64 stamp
        string label
        geometry_msgs/Pose pose
                Point position
                        float64 x
                        float64 y
                        float64 z
                Quaternion orientation
                        float64 x 0
                        float64 y 0
                        float64 z 0
                        float64 w 1
        int32[] word_id_keys
        int32[] word_id_values
        KeyPoint[] word_kpts
                Point2f pt
                        float32 x
                        float32 y
                float32 size
                float32 angle
                float32 response
                int32 octave
                int32 class_id
        Point3f[] word_pts
                float32 x
                float32 y
                float32 z
        uint8[] word_descriptors
        SensorData data
                std_msgs/Header header
                        builtin_interfaces/Time stamp
                                int32 sec
                                uint32 nanosec
                        string frame_id
                sensor_msgs/Image left
                        std_msgs/Header header #
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                                                     # Header frame_id should be optical frame of camera
                                                     # origin of frame should be optical center of cameara
                                                     # +x should point to the right in the image
                                                     # +y should point down in the image
                                                     # +z should point into to plane of the image
                                                     # If the frame_id here and the frame_id of the CameraInfo
                                                     # message associated with the image conflict
                                                     # the behavior is undefined
                        uint32 height                #
                        uint32 width                 #
                        string encoding       #
                                              # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
                        uint8 is_bigendian    #
                        uint32 step           #
                        uint8[] data          #
                sensor_msgs/Image right
                        std_msgs/Header header #
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                                                     # Header frame_id should be optical frame of camera
                                                     # origin of frame should be optical center of cameara
                                                     # +x should point to the right in the image
                                                     # +y should point down in the image
                                                     # +z should point into to plane of the image
                                                     # If the frame_id here and the frame_id of the CameraInfo
                                                     # message associated with the image conflict
                                                     # the behavior is undefined
                        uint32 height                #
                        uint32 width                 #
                        string encoding       #
                                              # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
                        uint8 is_bigendian    #
                        uint32 step           #
                        uint8[] data          #
                uint8[] left_compressed
                uint8[] right_compressed
                sensor_msgs/CameraInfo[] left_camera_info
                        #
                        #
                        #
                        #
                        #######################################################################
                        #######################################################################
                        std_msgs/Header header #
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                                                     # Header frame_id should be optical frame of camera
                                                     # origin of frame should be optical center of camera
                                                     # +x should point to the right in the image
                                                     # +y should point down in the image
                                                     # +z should point into the plane of the image
                        #######################################################################
                        #######################################################################
                        #######################################################################
                        uint32 height
                        uint32 width
                        string distortion_model
                        float64[] d
                        #     [fx  0 cx]
                        #     [ 0  0  1]
                        float64[9]  k #
                        float64[9]  r #
                        #     [fx'  0  cx' Tx]
                        #     [ 0   0   1   0]
                        float64[12] p #
                        #######################################################################
                        #######################################################################
                        #######################################################################
                        uint32 binning_x
                        uint32 binning_y
                        RegionOfInterest roi
                                #
                                uint32 x_offset  #
                                                 # (0 if the ROI includes the left edge of the image)
                                uint32 y_offset  #
                                                 # (0 if the ROI includes the top edge of the image)
                                uint32 height    #
                                uint32 width     #
                                bool do_rectify
                sensor_msgs/CameraInfo[] right_camera_info
                        #
                        #
                        #
                        #
                        #######################################################################
                        #######################################################################
                        std_msgs/Header header #
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                                                     # Header frame_id should be optical frame of camera
                                                     # origin of frame should be optical center of camera
                                                     # +x should point to the right in the image
                                                     # +y should point down in the image
                                                     # +z should point into the plane of the image
                        #######################################################################
                        #######################################################################
                        #######################################################################
                        uint32 height
                        uint32 width
                        string distortion_model
                        float64[] d
                        #     [fx  0 cx]
                        #     [ 0  0  1]
                        float64[9]  k #
                        float64[9]  r #
                        #     [fx'  0  cx' Tx]
                        #     [ 0   0   1   0]
                        float64[12] p #
                        #######################################################################
                        #######################################################################
                        #######################################################################
                        uint32 binning_x
                        uint32 binning_y
                        RegionOfInterest roi
                                #
                                uint32 x_offset  #
                                                 # (0 if the ROI includes the left edge of the image)
                                uint32 y_offset  #
                                                 # (0 if the ROI includes the top edge of the image)
                                uint32 height    #
                                uint32 width     #
                                bool do_rectify
                geometry_msgs/Transform[] local_transform
                        Vector3 translation
                                float64 x
                                float64 y
                                float64 z
                        Quaternion rotation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                sensor_msgs/PointCloud2 laser_scan
                        #
                        std_msgs/Header header
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                        uint32 height
                        uint32 width
                        PointField[] fields
                                uint8 INT8    = 1
                                uint8 UINT8   = 2
                                uint8 INT16   = 3
                                uint8 UINT16  = 4
                                uint8 INT32   = 5
                                uint8 UINT32  = 6
                                uint8 FLOAT32 = 7
                                uint8 FLOAT64 = 8
                                string name      #
                                uint32 offset    #
                                uint8  datatype  #
                                uint32 count     #
                        bool    is_bigendian #
                        uint32  point_step   #
                        uint32  row_step     #
                        uint8[] data         #
                        bool is_dense        #
                uint8[] laser_scan_compressed
                int32 laser_scan_max_pts
                float32 laser_scan_max_range
                int32 laser_scan_format
                geometry_msgs/Transform laser_scan_local_transform
                        Vector3 translation
                                float64 x
                                float64 y
                                float64 z
                        Quaternion rotation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                uint8[] user_data
                uint8[] grid_ground
                uint8[] grid_obstacles
                uint8[] grid_empty_cells
                float32 grid_cell_size
                Point3f grid_view_point
                        float32 x
                        float32 y
                        float32 z
                KeyPoint[] key_points
                        Point2f pt
                                float32 x
                                float32 y
                        float32 size
                        float32 angle
                        float32 response
                        int32 octave
                        int32 class_id
                Point3f[] points
                        float32 x
                        float32 y
                        float32 z
                uint8[] descriptors
                GlobalDescriptor[] global_descriptors
                        std_msgs/Header header
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                        int32 type
                        uint8[] info
                        uint8[] data
                EnvSensor[] env_sensors
                        std_msgs/Header header
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                        int32 type
                        float64 value
                sensor_msgs/Imu imu
                        #
                        #
                        #
                        std_msgs/Header header
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                        geometry_msgs/Quaternion orientation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                        float64[9] orientation_covariance #
                        geometry_msgs/Vector3 angular_velocity
                                float64 x
                                float64 y
                                float64 z
                        float64[9] angular_velocity_covariance #
                        geometry_msgs/Vector3 linear_acceleration
                                float64 x
                                float64 y
                                float64 z
                        float64[9] linear_acceleration_covariance #
                geometry_msgs/Transform imu_local_transform
                        Vector3 translation
                                float64 x
                                float64 y
                                float64 z
                        Quaternion rotation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                LandmarkDetection[] landmarks
                        std_msgs/Header header
                                builtin_interfaces/Time stamp
                                        int32 sec
                                        uint32 nanosec
                                string frame_id
                        string landmark_frame_id
                        int32 id
                        float32 size
                        geometry_msgs/PoseWithCovariance pose
                                Pose pose
                                        Point position
                                                float64 x
                                                float64 y
                                                float64 z
                                        Quaternion orientation
                                                float64 x 0
                                                float64 y 0
                                                float64 z 0
                                                float64 w 1
                                float64[36] covariance
                geometry_msgs/Pose ground_truth_pose
                        Point position
                                float64 x
                                float64 y
                                float64 z
                        Quaternion orientation
                                float64 x 0
                                float64 y 0
                                float64 z 0
                                float64 w 1
                GPS gps
                        float64 stamp      #
                        float64 longitude  #
                        float64 latitude   #
                        float64 altitude   #
                        float64 error      #
                        float64 bearing    #

```


回环检测
```bash
# link type
enum Type {
    kNeighbor,           // 0: 相邻节点（里程计边）
    kGlobalClosure,      // 1: 全局回环
    kLocalSpaceClosure,  // 2: 局部空间回环
    kLocalTimeClosure,   // 3: 局部时间回环
    kUserClosure,        // 4: 用户定义回环
    kVirtualClosure,     // 5: 虚拟回环
    kNeighborMerged,     // 6: 合并的相邻节点
    kPosePrior,          // 7: 绝对位姿先验
    kLandmark,           // 8: 地标
    kGravity,            // 9: 重力约束
    kEnd
};
```

```bash
# goto_contrller
ros2 run dream_ros2_bridge goto_controller
```


```bash
Param: BRIEF/Bytes = "32"                                  [Bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.]
Param: BRISK/Octaves = "3"                                 [Detection octaves. Use 0 to do single scale.]
Param: BRISK/PatternScale = "1"                            [Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.]
Param: BRISK/Thresh = "30"                                 [FAST/AGAST detection threshold score.]
Param: Bayes/FullPredictionUpdate = "false"                [Regenerate all the prediction matrix on each iteration (otherwise only removed/added ids are updated).]
Param: Bayes/PredictionLC = "0.1 0.36 0.30 0.16 0.062 0.0151 0.00255 0.000324 2.5e-05 1.3e-06 4.8e-08 1.2e-09 1.9e-11 2.2e-13 1.7e-15 8.5e-18 2.9e-20 6.9e-23" [Prediction of loop closures (Gaussian-like, here with sigma=1.6) - Format: {VirtualPlaceProb, LoopClosureProb, NeighborLvl1, NeighborLvl2, ...}.]
Param: Bayes/VirtualPlacePriorThr = "0.9"                  [Virtual place prior]
Param: Db/TargetVersion = ""                               [Target database version for backward compatibility purpose. Only Major and minor versions are used and should be set (e.g., 0.19 vs 0.20 or 1.0 vs 2.0). Patch version is ignored (e.g., 0.20.1 and 0.20.3 will generate a 0.20 database).]
Param: DbSqlite3/CacheSize = "10000"                       [Sqlite cache size (default is 2000).]
Param: DbSqlite3/InMemory = "false"                        [Using database in the memory instead of a file on the hard disk.]
Param: DbSqlite3/JournalMode = "3"                         [0=DELETE, 1=TRUNCATE, 2=PERSIST, 3=MEMORY, 4=OFF (see sqlite3 doc : "PRAGMA journal_mode")]
Param: DbSqlite3/Synchronous = "0"                         [0=OFF, 1=NORMAL, 2=FULL (see sqlite3 doc : "PRAGMA synchronous")]
Param: DbSqlite3/TempStore = "2"                           [0=DEFAULT, 1=FILE, 2=MEMORY (see sqlite3 doc : "PRAGMA temp_store")]
Param: FAST/CV = "0"                                       [Enable FastCV implementation if non-zero (and RTAB-Map is built with FastCV support). Values should be 9 and 10.]
Param: FAST/Gpu = "false"                                  [GPU-FAST: Use GPU version of FAST. This option is enabled only if OpenCV is built with CUDA and GPUs are detected.]
Param: FAST/GpuKeypointsRatio = "0.05"                     [Used with FAST GPU.]
Param: FAST/GridCols = "0"                                 [Grid cols (0 to disable). Adapts the detector to partition the source image into a grid and detect points in each cell.]
Param: FAST/GridRows = "0"                                 [Grid rows (0 to disable). Adapts the detector to partition the source image into a grid and detect points in each cell.]
Param: FAST/MaxThreshold = "200"                           [Maximum threshold. Used only when FAST/GridRows and FAST/GridCols are set.]
Param: FAST/MinThreshold = "7"                             [Minimum threshold. Used only when FAST/GridRows and FAST/GridCols are set.]
Param: FAST/NonmaxSuppression = "true"                     [If true, non-maximum suppression is applied to detected corners (keypoints).]
Param: FAST/Threshold = "20"                               [Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.]
Param: FREAK/NOctaves = "4"                                [Number of octaves covered by the detected keypoints.]
Param: FREAK/OrientationNormalized = "true"                [Enable orientation normalization.]
Param: FREAK/PatternScale = "22"                           [Scaling of the description pattern.]
Param: FREAK/ScaleNormalized = "true"                      [Enable scale normalization.]
Param: GFTT/BlockSize = "3"                                []
Param: GFTT/Gpu = "false"                                  [GPU-GFTT: Use GPU version of GFTT. This option is enabled only if OpenCV>=3 is built with CUDA and GPUs are detected.]
Param: GFTT/K = "0.04"                                     []
Param: GFTT/MinDistance = "7"                              []
Param: GFTT/QualityLevel = "0.001"                         []
Param: GFTT/UseHarrisDetector = "false"                    []
Param: GMS/ThresholdFactor = "6.0"                         [The higher, the less matches.]
Param: GMS/WithRotation = "false"                          [Take rotation transformation into account.]
Param: GMS/WithScale = "false"                             [Take scale transformation into account.]
Param: GTSAM/IncRelinearizeSkip = "1"                      [Only relinearize any variables every X calls to ISAM2::update(). See GTSAM::ISAM2 doc for more info.]
Param: GTSAM/IncRelinearizeThreshold = "0.01"              [Only relinearize variables whose linear delta magnitude is greater than this threshold. See GTSAM::ISAM2 doc for more info.]
Param: GTSAM/Incremental = "false"                         [Do graph optimization incrementally (iSAM2) to increase optimization speed on loop closures. Note that only GaussNewton and Dogleg optimization algorithms are supported (GTSAM/Optimizer) in this mode.]
Param: GTSAM/Optimizer = "1"                               [0=Levenberg 1=GaussNewton 2=Dogleg]
Param: Grid/3D = "true"                                    [A 3D occupancy grid is required if you want an OctoMap (3D ray tracing). Set to false if you want only a 2D map, the cloud will be projected on xy plane. A 2D map can be still generated if checked, but it requires more memory and time to generate it. Ignored if laser scan is 2D and "Grid/Sensor" is 0.]
Param: Grid/CellSize = "0.05"                              [Resolution of the occupancy grid.]
Param: Grid/ClusterRadius = "0.1"                          [[Grid/NormalsSegmentation=true] Cluster maximum radius.]
Param: Grid/DepthDecimation = "4"                          [[Grid/DepthDecimation=true] Decimation of the depth image before creating cloud.]
Param: Grid/DepthRoiRatios = "0.0 0.0 0.0 0.0"             [[Grid/Sensor>=1] Region of interest ratios [left, right, top, bottom].]
Param: Grid/FlatObstacleDetected = "true"                  [[Grid/NormalsSegmentation=true] Flat obstacles detected.]
Param: Grid/FootprintHeight = "0.0"                        [Footprint height used to filter points over the footprint of the robot. Footprint length and width should be set.]
Param: Grid/FootprintLength = "0.0"                        [Footprint length used to filter points over the footprint of the robot.]
Param: Grid/FootprintWidth = "0.0"                         [Footprint width used to filter points over the footprint of the robot. Footprint length should be set.]
Param: Grid/GroundIsObstacle = "false"                     [[Grid/3D=true] Ground segmentation (Grid/NormalsSegmentation) is ignored, all points are obstacles. Use this only if you want an OctoMap with ground identified as an obstacle (e.g., with an UAV).]
Param: Grid/MapFrameProjection = "false"                   [Projection in map frame. On a 3D terrain and a fixed local camera transform (the cloud is created relative to ground), you may want to disable this to do the projection in robot frame instead.]
Param: Grid/MaxGroundAngle = "45"                          [[Grid/NormalsSegmentation=true] Maximum angle (degrees) between point's normal to ground's normal to label it as ground. Points with higher angle difference are considered as obstacles.]
Param: Grid/MaxGroundHeight = "0.0"                        [Maximum ground height (0=disabled). Should be set if "Grid/NormalsSegmentation" is false.]
Param: Grid/MaxObstacleHeight = "0.0"                      [Maximum obstacles height (0=disabled).]
Param: Grid/MinClusterSize = "10"                          [[Grid/NormalsSegmentation=true] Minimum cluster size to project the points.]
Param: Grid/MinGroundHeight = "0.0"                        [Minimum ground height (0=disabled).]
Param: Grid/NoiseFilteringMinNeighbors = "5"               [Noise filtering minimum neighbors.]
Param: Grid/NoiseFilteringRadius = "0.0"                   [Noise filtering radius (0=disabled). Done after segmentation.]
Param: Grid/NormalK = "20"                                 [[Grid/NormalsSegmentation=true] K neighbors to compute normals.]
Param: Grid/NormalsSegmentation = "true"                   [Segment ground from obstacles using point normals, otherwise a fast passthrough is used.]
Param: Grid/PreVoxelFiltering = "true"                     [Input cloud is downsampled by voxel filter (voxel size is "Grid/CellSize") before doing segmentation of obstacles and ground.]
Param: Grid/RangeMax = "5.0"                               [Maximum range from sensor. 0=inf.]
Param: Grid/RangeMin = "0.0"                               [Minimum range from sensor.]
Param: Grid/RayTracing = "false"                           [Ray tracing is done for each occupied cell, filling unknown space between the sensor and occupied cells. If Grid/3D=true, RTAB-Map should be built with OctoMap support, otherwise 3D ray tracing is ignored.]
Param: Grid/Scan2dUnknownSpaceFilled = "false"             [Unknown space filled. Only used with 2D laser scans. Use Grid/RangeMax to set maximum range if laser scan max range is to set.]
Param: Grid/ScanDecimation = "1"                           [[Grid/Sensor=0 or 2] Decimation of the laser scan before creating cloud.]
Param: Grid/Sensor = "1"                                   [Create occupancy grid from selected sensor: 0=laser scan, 1=depth image(s) or 2=both laser scan and depth image(s).]
Param: GridGlobal/AltitudeDelta = "0"                      [Assemble only nodes that have the same altitude of +-delta meters of the current pose (0=disabled). This is used to generate 2D occupancy grid based on the current altitude (e.g., multi-floor building).]
Param: GridGlobal/Eroded = "false"                         [Erode obstacle cells.]
Param: GridGlobal/FloodFillDepth = "0"                     [Flood fill filter (0=disabled), used to remove empty cells outside the map. The flood fill is done at the specified depth (between 1 and 16) of the OctoMap.]
Param: GridGlobal/FootprintRadius = "0.0"                  [Footprint radius (m) used to clear all obstacles under the graph.]
Param: GridGlobal/MaxNodes = "0"                           [Maximum nodes assembled in the map starting from the last node (0=unlimited).]
Param: GridGlobal/MinSize = "0.0"                          [Minimum map size (m).]
Param: GridGlobal/OccupancyThr = "0.5"                     [Occupancy threshold (value between 0 and 1).]
Param: GridGlobal/ProbClampingMax = "0.971"                [Probability clamping maximum (value between 0 and 1).]
Param: GridGlobal/ProbClampingMin = "0.1192"               [Probability clamping minimum (value between 0 and 1).]
Param: GridGlobal/ProbHit = "0.7"                          [Probability of a hit (value between 0.5 and 1).]
Param: GridGlobal/ProbMiss = "0.4"                         [Probability of a miss (value between 0 and 0.5).]
Param: GridGlobal/UpdateError = "0.01"                     [Graph changed detection error (m). Update map only if poses in new optimized graph have moved more than this value.]
Param: Icp/CCFilterOutFarthestPoints = "false"             [If true, the algorithm will automatically ignore farthest points from the reference, for better convergence.]
Param: Icp/CCMaxFinalRMS = "0.2"                           [Maximum final RMS error.]
Param: Icp/CCSamplingLimit = "50000"                       [Maximum number of points per cloud (they are randomly resampled below this limit otherwise).]
Param: Icp/CorrespondenceRatio = "0.1"                     [Ratio of matching correspondences to accept the transform.]
Param: Icp/DebugExportFormat = ""                          [Export scans used for ICP in the specified format (a warning on terminal will be shown with the file paths used). Supported formats are "pcd", "ply" or "vtk". If logger level is debug, from and to scans will stamped, so previous files won't be overwritten.]
Param: Icp/DownsamplingStep = "1"                          [Downsampling step size (1=no sampling). This is done before uniform sampling.]
Param: Icp/Epsilon = "0"                                   [Set the transformation epsilon (maximum allowable difference between two consecutive transformations) in order for an optimization to be considered as having converged to the final solution.]
Param: Icp/FiltersEnabled = "3"                            [Flag to enable filters: 1="from" cloud only, 2="to" cloud only, 3=both.]
Param: Icp/Force4DoF = "false"                             [Limit ICP to x, y, z and yaw DoF. Available if Icp/Strategy > 0.]
Param: Icp/Iterations = "30"                               [Max iterations.]
Param: Icp/MaxCorrespondenceDistance = "0.1"               [Max distance for point correspondences.]
Param: Icp/MaxRotation = "0.78"                            [Maximum ICP rotation correction accepted (rad).]
Param: Icp/MaxTranslation = "0.2"                          [Maximum ICP translation correction accepted (m).]
Param: Icp/OutlierRatio = "0.85"                           [Outlier ratio used with Icp/Strategy>0. For libpointmatcher, this parameter set TrimmedDistOutlierFilter/ratio for convenience when configuration file is not set. For CCCoreLib, this parameter set the "finalOverlapRatio". The value should be between 0 and 1.]
Param: Icp/PMConfig = ""                                   [Configuration file (*.yaml) used by libpointmatcher. Note that data filters set for libpointmatcher are done after filtering done by rtabmap (i.e., Icp/VoxelSize, Icp/DownsamplingStep), so make sure to disable those in rtabmap if you want to use only those from libpointmatcher. Parameters Icp/Iterations, Icp/Epsilon and Icp/MaxCorrespondenceDistance are also ignored if configuration file is set.]
Param: Icp/PMMatcherEpsilon = "0.0"                        [KDTreeMatcher/epsilon: approximation to use for the nearest-neighbor search. For convenience when configuration file is not set.]
Param: Icp/PMMatcherIntensity = "false"                    [KDTreeMatcher:  among nearest neighbors, keep only the one with the most similar intensity. This only work with Icp/PMMatcherKnn>1.]
Param: Icp/PMMatcherKnn = "1"                              [KDTreeMatcher/knn: number of nearest neighbors to consider it the reference. For convenience when configuration file is not set.]
Param: Icp/PointToPlane = "true"                           [Use point to plane ICP.]
Param: Icp/PointToPlaneGroundNormalsUp = "0.0"             [Invert normals on ground if they are pointing down (useful for ring-like 3D LiDARs). 0 means disabled, 1 means only normals perfectly aligned with -z axis. This is only done with 3D scans.]
Param: Icp/PointToPlaneK = "5"                             [Number of neighbors to compute normals for point to plane if the cloud doesn't have already normals.]
Param: Icp/PointToPlaneLowComplexityStrategy = "1"         [If structural complexity is below Icp/PointToPlaneMinComplexity: set to 0 to so that the transform is automatically rejected, set to 1 to limit ICP correction in axes with most constraints (e.g., for a corridor-like environment, the resulting transform will be limited in y and yaw, x will taken from the guess), set to 2 to accept "as is" the transform computed by PointToPoint.]
Param: Icp/PointToPlaneMinComplexity = "0.02"              [Minimum structural complexity (0.0=low, 1.0=high) of the scan to do PointToPlane registration, otherwise PointToPoint registration is done instead and strategy from Icp/PointToPlaneLowComplexityStrategy is used. This check is done only when Icp/PointToPlane=true.]
Param: Icp/PointToPlaneRadius = "0.0"                      [Search radius to compute normals for point to plane if the cloud doesn't have already normals.]
Param: Icp/RangeMax = "0"                                  [Maximum range filtering (0=disabled).]
Param: Icp/RangeMin = "0"                                  [Minimum range filtering (0=disabled).]
Param: Icp/ReciprocalCorrespondences = "true"              [To be a valid correspondence, the corresponding point in target cloud to point in source cloud should be both their closest closest correspondence.]
Param: Icp/Strategy = "1"                                  [ICP implementation: 0=Point Cloud Library, 1=libpointmatcher, 2=CCCoreLib (CloudCompare).]
Param: Icp/VoxelSize = "0.05"                              [Uniform sampling voxel size (0=disabled).]
Param: ImuFilter/ComplementaryBiasAlpha = "0.01"           [Bias estimation gain parameter, belongs in [0, 1].]
Param: ImuFilter/ComplementaryDoAdpativeGain = "true"      [Parameter whether to do adaptive gain or not.]
Param: ImuFilter/ComplementaryDoBiasEstimation = "true"    [Parameter whether to do bias estimation or not.]
Param: ImuFilter/ComplementaryGainAcc = "0.01"             [Gain parameter for the complementary filter, belongs in [0, 1].]
Param: ImuFilter/MadgwickGain = "0.1"                      [Gain of the filter. Higher values lead to faster convergence but more noise. Lower values lead to slower convergence but smoother signal, belongs in [0, 1].]
Param: ImuFilter/MadgwickZeta = "0.0"                      [Gyro drift gain (approx. rad/s), belongs in [-1, 1].]
Param: KAZE/Diffusivity = "1"                              [Diffusivity type: 0=DIFF_PM_G1, 1=DIFF_PM_G2, 2=DIFF_WEICKERT or 3=DIFF_CHARBONNIER.]
Param: KAZE/Extended = "false"                             [Set to enable extraction of extended (128-byte) descriptor.]
Param: KAZE/NOctaveLayers = "4"                            [Default number of sublevels per scale level.]
Param: KAZE/NOctaves = "4"                                 [Maximum octave evolution of the image.]
Param: KAZE/Threshold = "0.001"                            [Detector response threshold to accept keypoint.]
Param: KAZE/Upright = "false"                              [Set to enable use of upright descriptors (non rotation-invariant).]
Param: Kp/BadSignRatio = "0.5"                             [Bad signature ratio (less than Ratio x AverageWordsPerImage = bad).]
Param: Kp/ByteToFloat = "false"                            [For Kp/NNStrategy=1, binary descriptors are converted to float by converting each byte to float instead of converting each bit to float. When converting bytes instead of bits, less memory is used and search is faster at the cost of slightly less accurate matching.]
Param: Kp/DetectorStrategy = "8"                           [0=SURF 1=SIFT 2=ORB 3=FAST/FREAK 4=FAST/BRIEF 5=GFTT/FREAK 6=GFTT/BRIEF 7=BRISK 8=GFTT/ORB 9=KAZE 10=ORB-OCTREE 11=SuperPoint 12=SURF/FREAK 13=GFTT/DAISY 14=SURF/DAISY 15=PyDetector]
Param: Kp/DictionaryPath = ""                              [Path of the pre-computed dictionary]
Param: Kp/FlannIndexSaved = "false"                        [Save FLANN index during localization session (when Mem/IncrementalMemory=false). The FLANN index will be saved to database after the first time localization mode is used, then on next sessions, the index is reloaded from the database instead of being rebuilt again. This can save significant loading time when the visual word dictionary is big (>1M words). Note that if the dictionary is modified (parameters or data), the index will be rebuilt and saved again on the next session.]
Param: Kp/FlannRebalancingFactor = "2.0"                   [Factor used when rebuilding the incremental FLANN index (see "Kp/IncrementalFlann"). Set <=1 to disable.]
Param: Kp/GridCols = "1"                                   [Number of columns of the grid used to extract uniformly "Kp/MaxFeatures / grid cells" features from each cell.]
Param: Kp/GridRows = "1"                                   [Number of rows of the grid used to extract uniformly "Kp/MaxFeatures / grid cells" features from each cell.]
Param: Kp/IncrementalDictionary = "true"                   []
Param: Kp/IncrementalFlann = "true"                        [When using FLANN based strategy, add/remove points to its index without always rebuilding the index (the index is built only when the dictionary increases of the factor "Kp/FlannRebalancingFactor" in size).]
Param: Kp/MaxDepth = "0"                                   [Filter extracted keypoints by depth (0=inf).]
Param: Kp/MaxFeatures = "500"                              [Maximum features extracted from the images (0 means not bounded, <0 means no extraction).]
Param: Kp/MinDepth = "0"                                   [Filter extracted keypoints by depth.]
Param: Kp/NNStrategy = "1"                                 [kNNFlannNaive=0, kNNFlannKdTree=1, kNNFlannLSH=2, kNNBruteForce=3, kNNBruteForceGPU=4]
Param: Kp/NewWordsComparedTogether = "true"                [When adding new words to dictionary, they are compared also with each other (to detect same words in the same signature).]
Param: Kp/NndrRatio = "0.8"                                [NNDR ratio (A matching pair is detected, if its distance is closer than X times the distance of the second nearest neighbor.)]
Param: Kp/Parallelized = "true"                            [If the dictionary update and signature creation were parallelized.]
Param: Kp/RoiRatios = "0.0 0.0 0.0 0.0"                    [Region of interest ratios [left, right, top, bottom].]
Param: Kp/SSC = "false"                                    [If true, SSC (Suppression via Square Covering) is applied to limit keypoints.]
Param: Kp/SerializeWithChecksum = "true"                   [On serialization of the FLANN index, compute checksum of the data used by the FLANN index. This adds a slight overhead on serialization/deserialization to make sure that the dictionary data correspond to same data used when the index was built.]
Param: Kp/SubPixEps = "0.02"                               [See cv::cornerSubPix().]
Param: Kp/SubPixIterations = "0"                           [See cv::cornerSubPix(). 0 disables sub pixel refining.]
Param: Kp/SubPixWinSize = "3"                              [See cv::cornerSubPix().]
Param: Kp/TfIdfLikelihoodUsed = "true"                     [Use of the td-idf strategy to compute the likelihood.]
Param: Marker/CornerRefinementMethod = "0"                 [Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag2). For OpenCV <3.3.0, this is "doCornerRefinement" parameter: set 0 for false and 1 for true.]
Param: Marker/Dictionary = "0"                             [Dictionary to use: DICT_ARUCO_4X4_50=0, DICT_ARUCO_4X4_100=1, DICT_ARUCO_4X4_250=2, DICT_ARUCO_4X4_1000=3, DICT_ARUCO_5X5_50=4, DICT_ARUCO_5X5_100=5, DICT_ARUCO_5X5_250=6, DICT_ARUCO_5X5_1000=7, DICT_ARUCO_6X6_50=8, DICT_ARUCO_6X6_100=9, DICT_ARUCO_6X6_250=10, DICT_ARUCO_6X6_1000=11, DICT_ARUCO_7X7_50=12, DICT_ARUCO_7X7_100=13, DICT_ARUCO_7X7_250=14, DICT_ARUCO_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16, DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20]
Param: Marker/Length = "0"                                 [The length (m) of the markers' side. 0 means automatic marker length estimation using the depth image (the camera should look at the marker perpendicularly for initialization).]
Param: Marker/MaxDepthError = "0.01"                       [Maximum depth error between all corners of a marker when estimating the marker length (when Marker/Length is 0). The smaller it is, the more perpendicular the camera should be toward the marker to initialize the length.]
Param: Marker/MaxRange = "0.0"                             [Maximum range in which markers will be detected. <=0 for unlimited range.]
Param: Marker/MinRange = "0.0"                             [Miniminum range in which markers will be detected. <=0 for unlimited range.]
Param: Marker/Priors = ""                                  [World prior locations of the markers. The map will be transformed in marker's world frame when a tag is detected. Format is the marker's ID followed by its position (angles in rad), markers are separated by vertical line ("id1 x y z roll pitch yaw|id2 x y z roll pitch yaw"). Example:  "1 0 0 1 0 0 0|2 1 0 1 0 0 1.57" (marker 2 is 1 meter forward than marker 1 with 90 deg yaw rotation).]
Param: Marker/PriorsVarianceAngular = "0.001"              [Angular variance to set on marker priors.]
Param: Marker/PriorsVarianceLinear = "0.001"               [Linear variance to set on marker priors.]
Param: Marker/VarianceAngular = "0.01"                     [Angular variance to set on marker detections. If Marker/VarianceOrientationIgnored is enabled, it is ignored with Optimizer/Strategy=1 (g2o) and it corresponds to bearing variance with Optimizer/Strategy=2 (GTSAM).]
Param: Marker/VarianceLinear = "0.001"                     [Linear variance to set on marker detections. If Marker/VarianceOrientationIgnored is enabled and Optimizer/Strategy=2 (GTSAM): it is the variance of the range factor, with 9999 to disable range factor and to do only bearing.]
Param: Marker/VarianceOrientationIgnored = "false"         [When this setting is false, the landmark's orientation is optimized during graph optimization. When this setting is true, only the position of the landmark is optimized. This can be useful when the landmark's orientation estimation is not reliable. Note that for Optimizer/Strategy=1 (g2o), only Marker/VarianceLinear needs be set if we ignore orientation. For Optimizer/Strategy=2 (GTSAM), instead of optimizing the landmark's position directly, a bearing/range factor is used, with Marker/VarianceLinear as the variance of the range factor (with 9999 to optimize the position with only a bearing factor) and Marker/VarianceAngular as the variance of the bearing factor (pitch/yaw).]
Param: Mem/BadSignaturesIgnored = "false"                  [Bad signatures are ignored.]
Param: Mem/BinDataKept = "true"                            [Keep binary data in db.]
Param: Mem/CompressionParallelized = "true"                [Compression of sensor data is multi-threaded.]
Param: Mem/CovOffDiagIgnored = "true"                      [Ignore off diagonal values of the covariance matrix.]
Param: Mem/DepthAsMask = "true"                            [Use depth image as mask when extracting features for vocabulary.]
Param: Mem/DepthCompressionFormat = ".rvl"                 [Depth image compression format for 16UC1 depth type. It should be ".png" or ".rvl". If depth type is 32FC1, ".png" is used.]
Param: Mem/DepthMaskFloorThr = "0.0"                       [Filter floor from depth mask below specified threshold (m) before extracting features. 0 means disabled, negative means remove all objects above the floor threshold instead. Ignored if Mem/DepthAsMask is false.]
Param: Mem/GenerateIds = "true"                            [True=Generate location IDs, False=use input image IDs.]
Param: Mem/GlobalDescriptorStrategy = "0"                  [Extract global descriptor from sensor data. 0=disabled, 1=PyDescriptor]
Param: Mem/ImageCompressionFormat = ".jpg"                 [RGB image compression format. It should be ".jpg" or ".png".]
Param: Mem/ImageKept = "false"                             [Keep raw images in RAM.]
Param: Mem/ImagePostDecimation = "1"                       [Decimation of the RGB image before saving it to database. If depth size is larger than decimated RGB size, depth is decimated to be always at most equal to RGB size. Decimation is done from the original image. If set to same value than Mem/ImagePreDecimation, data already decimated is saved (no need to re-decimate the image).]
Param: Mem/ImagePreDecimation = "1"                        [Decimation of the RGB image before visual feature detection. If depth size is larger than decimated RGB size, depth is decimated to be always at most equal to RGB size. If Mem/DepthAsMask is true and if depth is smaller than decimated RGB, depth may be interpolated to match RGB size for feature detection.]
Param: Mem/IncrementalMemory = "true"                      [SLAM mode, otherwise it is Localization mode.]
Param: Mem/InitWMWithAllNodes = "false"                    [Initialize the Working Memory with all nodes in Long-Term Memory. When false, it is initialized with nodes of the previous session.]
Param: Mem/IntermediateNodeDataKept = "false"              [Keep intermediate node data in db.]
Param: Mem/LaserScanDownsampleStepSize = "1"               [If > 1, downsample the laser scans when creating a signature.]
Param: Mem/LaserScanNormalK = "0"                          [If > 0 and laser scans don't have normals, normals will be computed with K search neighbors when creating a signature.]
Param: Mem/LaserScanNormalRadius = "0.0"                   [If > 0 m and laser scans don't have normals, normals will be computed with radius search neighbors when creating a signature.]
Param: Mem/LaserScanVoxelSize = "0.0"                      [If > 0 m, voxel filtering is done on laser scans when creating a signature. If the laser scan had normals, they will be removed. To recompute the normals, make sure to use "Mem/LaserScanNormalK" or "Mem/LaserScanNormalRadius" parameters.]
Param: Mem/LoadVisualLocalFeaturesOnInit = "true"          [Load all local visual features (keypoints, descriptors and 3D points) in RAM when loading an existing database. This can add significant time to initialize the memory but the features will be already loaded before computing loop closure transforms. If false, the features are loaded on-demand from the database when a loop closure transformation should be estimated.]
Param: Mem/LocalizationDataSaved = "false"                 [Save localization data during localization session (when Mem/IncrementalMemory=false). When enabled, the database will then also grow in localization mode. This mode would be used only for debugging purpose.]
Param: Mem/MapLabelsAdded = "true"                         [Create map labels. The first node of a map will be labeled as "map#" where # is the map ID.]
Param: Mem/NotLinkedNodesKept = "true"                     [Keep not linked nodes in db (rehearsed nodes and deleted nodes).]
Param: Mem/RawDescriptorsKept = "true"                     [Raw descriptors kept in memory.]
Param: Mem/RecentWmRatio = "0.2"                           [Ratio of locations after the last loop closure in WM that cannot be transferred.]
Param: Mem/ReduceGraph = "false"                           [Reduce graph. Merge nodes when loop closures are added (ignoring those with user data set).]
Param: Mem/RehearsalIdUpdatedToNewOne = "false"            [On merge, update to new id. When false, no copy.]
Param: Mem/RehearsalSimilarity = "0.6"                     [Rehearsal similarity.]
Param: Mem/RehearsalWeightIgnoredWhileMoving = "false"     [When the robot is moving, weights are not updated on rehearsal.]
Param: Mem/RotateImagesUpsideUp = "false"                  [Rotate images so that upside is up if they are not already. This can be useful in case the robots don't have all same camera orientation but are using the same map, so that not rotation-invariant visual features can still be used across the fleet.]
Param: Mem/STMSize = "10"                                  [Short-term memory size.]
Param: Mem/SaveDepth16Format = "false"                     [Save depth image into 16 bits format to reduce memory used. Warning: values over ~65 meters are ignored (maximum 65535 millimeters).]
Param: Mem/StereoFromMotion = "false"                      [Triangulate features without depth using stereo from motion (odometry). It would be ignored if Mem/DepthAsMask is true and the feature detector used supports masking.]
Param: Mem/TransferSortingByWeightId = "false"             [On transfer, signatures are sorted by weight->ID only (i.e. the oldest of the lowest weighted signatures are transferred first). If false, the signatures are sorted by weight->Age->ID (i.e. the oldest inserted in WM of the lowest weighted signatures are transferred first). Note that retrieval updates the age, not the ID.]
Param: Mem/UseOdomFeatures = "true"                        [Use odometry features instead of regenerating them.]
Param: Mem/UseOdomGravity = "false"                        [Use odometry instead of IMU orientation to add gravity links to new nodes created. We assume that odometry is already aligned with gravity (e.g., we are using a VIO approach). Gravity constraints are used by graph optimization only if "Optimizer/GravitySigma" is not zero.]
Param: ORB/EdgeThreshold = "19"                            [This is size of the border where the features are not detected. It should roughly match the patchSize parameter.]
Param: ORB/FirstLevel = "0"                                [It should be 0 in the current implementation.]
Param: ORB/Gpu = "false"                                   [GPU-ORB: Use GPU version of ORB. This option is enabled only if OpenCV is built with CUDA and GPUs are detected.]
Param: ORB/NLevels = "3"                                   [The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels).]
Param: ORB/PatchSize = "31"                                [size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.]
Param: ORB/ScaleFactor = "2"                               [Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.]
Param: ORB/ScoreType = "0"                                 [The default HARRIS_SCORE=0 means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE=1 is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.]
Param: ORB/WTA_K = "2"                                     [The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).]
Param: Optimizer/Epsilon = "0.00001"                       [Stop optimizing when the error improvement is less than this value.]
Param: Optimizer/GravitySigma = "0.3"                      [Gravity sigma value (>=0, typically between 0.1 and 0.3). Optimization is done while preserving gravity orientation of the poses. This should be used only with visual/lidar inertial odometry approaches, for which we assume that all odometry poses are aligned with gravity. Set to 0 to disable gravity constraints. Currently supported only with g2o and GTSAM optimization strategies (see Optimizer/Strategy).]
Param: Optimizer/Iterations = "20"                         [Optimization iterations.]
Param: Optimizer/LandmarksIgnored = "false"                [Ignore landmark constraints while optimizing. Currently only g2o and gtsam optimization supports this.]
Param: Optimizer/PriorsIgnored = "true"                    [Ignore prior constraints (global pose or GPS) while optimizing. Currently only g2o and gtsam optimization supports this.]
Param: Optimizer/Robust = "false"                          [Robust graph optimization using Vertigo (only work for g2o and GTSAM optimization strategies). Not compatible with "RGBD/OptimizeMaxError" if enabled.]
Param: Optimizer/Strategy = "2"                            [Graph optimization strategy: 0=TORO, 1=g2o, 2=GTSAM and 3=Ceres.]
Param: Optimizer/VarianceIgnored = "false"                 [Ignore constraints' variance. If checked, identity information matrix is used for each constraint. Otherwise, an information matrix is generated from the variance saved in the links.]
Param: PyDescriptor/Dim = "4096"                           [Descriptor dimension.]
Param: PyDescriptor/Path = ""                              [Path to python script file (see available ones in rtabmap/corelib/src/pydescriptor/*). See the header to see where the script should be used.]
Param: PyDetector/Cuda = "true"                            [Use cuda.]
Param: PyDetector/Path = ""                                [Path to python script file (see available ones in rtabmap/corelib/src/python/*). See the header to see where the script should be copied.]
Param: PyMatcher/Cuda = "true"                             [Used by SuperGlue.]
Param: PyMatcher/Iterations = "20"                         [Sinkhorn iterations. Used by SuperGlue.]
Param: PyMatcher/Model = "indoor"                          [For SuperGlue, set only "indoor" or "outdoor". For OANet, set path to one of the pth file (e.g., "OANet/model/gl3d/sift-4000/model_best.pth").]
Param: PyMatcher/Path = ""                                 [Path to python script file (see available ones in rtabmap/corelib/src/python/*). See the header to see where the script should be copied.]
Param: PyMatcher/Threshold = "0.2"                         [Used by SuperGlue.]
Param: RGBD/AggressiveLoopThr = "0.05"                     [Loop closure threshold used (overriding Rtabmap/LoopThr) when a new mapping session is not yet linked to a map of the highest loop closure hypothesis. In localization mode, this threshold is used when there are no loop closure constraints with any map in the cache (RGBD/MaxOdomCacheSize). In all cases, the goal is to aggressively loop on a previous map in the database. Only used when RGBD/Enabled is enabled. Set 1 to disable.]
Param: RGBD/AngularSpeedUpdate = "0.0"                     [Maximum angular speed (rad/s) to update the map (0 means not limit).]
Param: RGBD/AngularUpdate = "0.1"                          [Minimum angular displacement (rad) to update the map. Rehearsal is done prior to this, so weights are still updated.]
Param: RGBD/CreateOccupancyGrid = "true"                   [Create local occupancy grid maps. See "Grid" group for parameters.]
Param: RGBD/Enabled = "true"                               [Activate metric SLAM. If set to false, classic RTAB-Map loop closure detection is done using only images and without any metric information.]
Param: RGBD/ForceOdom3DoF = "true"                         [Force odometry pose to be 3DoF if Reg/Force3DoF=true.]
Param: RGBD/GoalReachedRadius = "0.5"                      [Goal reached radius (m).]
Param: RGBD/GoalsSavedInUserData = "false"                 [When a goal is received and processed with success, it is saved in user data of the location with this format: "GOAL:#".]
Param: RGBD/InvertedReg = "false"                          [On loop closure, do registration from the target to reference instead of reference to target.]
Param: RGBD/LinearSpeedUpdate = "0.0"                      [Maximum linear speed (m/s) to update the map (0 means not limit).]
Param: RGBD/LinearUpdate = "0.1"                           [Minimum linear displacement (m) to update the map. Rehearsal is done prior to this, so weights are still updated.]
Param: RGBD/LocalBundleOnLoopClosure = "false"             [Do local bundle adjustment with neighborhood of the loop closure.]
Param: RGBD/LocalImmunizationRatio = "0.25"                [Ratio of working memory for which local nodes are immunized from transfer.]
Param: RGBD/LocalRadius = "10"                             [Local radius (m) for nodes selection in the local map. This parameter is used in some approaches about the local map management.]
Param: RGBD/LocalizationPriorError = "0.001"               [The corresponding variance (error x error) set to priors of the map's poses during localization (when RGBD/MaxOdomCacheSize>0).]
Param: RGBD/LocalizationSecondTryWithoutProximityLinks = "true" [When localization is rejected by graph optimization validation, try a second time without proximity links if landmark or loop closure links are also present in odometry cache (see RGBD/MaxOdomCacheSize). If it succeeds, the proximity links are removed. This assumes that global loop closure and landmark links are more accurate than proximity links.]
Param: RGBD/LocalizationSmoothing = "true"                 [Adjust localization constraints based on optimized odometry cache poses (when RGBD/MaxOdomCacheSize>0).]
Param: RGBD/LoopClosureIdentityGuess = "false"             [Use Identity matrix as guess when computing loop closure transform, otherwise no guess is used, thus assuming that registration strategy selected (Reg/Strategy) can deal with transformation estimation without guess.]
Param: RGBD/LoopClosureReextractFeatures = "false"         [Extract features even if there are some already in the nodes. Raw features are not saved in database.]
Param: RGBD/LoopCovLimited = "false"                       [Limit covariance of non-neighbor links to minimum covariance of neighbor links. In other words, if covariance of a loop closure link is smaller than the minimum covariance of odometry links, its covariance is set to minimum covariance of odometry links.]
Param: RGBD/MarkerDetection = "false"                      [Detect static markers to be added as landmarks for graph optimization. If input data have already landmarks, this will be ignored. See "Marker" group for parameters.]
Param: RGBD/MaxLocalRetrieved = "2"                        [Maximum local locations retrieved (0=disabled) near the current pose in the local map or on the current planned path (those on the planned path have priority).]
Param: RGBD/MaxLoopClosureDistance = "0.0"                 [Reject loop closures/localizations if the distance from the map is over this distance (0=disabled).]
Param: RGBD/MaxOdomCacheSize = "10"                        [Maximum odometry cache size. Used only in localization mode (when Mem/IncrementalMemory=false). This is used to get smoother localizations and to verify localization transforms (when RGBD/OptimizeMaxError!=0) to make sure we don't teleport to a location very similar to one we previously localized on. Set 0 to disable caching.]
Param: RGBD/NeighborLinkRefining = "false"                 [When a new node is added to the graph, the transformation of its neighbor link to the previous node is refined using registration approach selected (Reg/Strategy).]
Param: RGBD/NewMapOdomChangeDistance = "0"                 [A new map is created if a change of odometry translation greater than X m is detected (0 m = disabled).]
Param: RGBD/OptimizeFromGraphEnd = "false"                 [Optimize graph from the newest node. If false, the graph is optimized from the oldest node of the current graph (this adds an overhead computation to detect to oldest node of the current graph, but it can be useful to preserve the map referential from the oldest node). Warning when set to false: when some nodes are transferred, the first referential of the local map may change, resulting in momentary changes in robot/map position (which are annoying in teleoperation).]
Param: RGBD/OptimizeMaxError = "3.0"                       [Reject loop closures if optimization error ratio is greater than this value (0=disabled). Ratio is computed as absolute error over standard deviation of each link. This will help to detect when a wrong loop closure is added to the graph. Not compatible with "Optimizer/Robust" if enabled.]
Param: RGBD/PlanAngularVelocity = "0"                      [Angular velocity (rad/sec) used to compute path weights.]
Param: RGBD/PlanLinearVelocity = "0"                       [Linear velocity (m/sec) used to compute path weights.]
Param: RGBD/PlanStuckIterations = "0"                      [Mark the current goal node on the path as unreachable if it is not updated after X iterations (0=disabled). If all upcoming nodes on the path are unreachabled, the plan fails.]
Param: RGBD/ProximityAngle = "45"                          [Maximum angle (degrees) for one-to-one proximity detection.]
Param: RGBD/ProximityBySpace = "true"                      [Detection over locations (in Working Memory) near in space.]
Param: RGBD/ProximityByTime = "false"                      [Detection over all locations in STM.]
Param: RGBD/ProximityGlobalScanMap = "false"               [Create a global assembled map from laser scans for one-to-many proximity detection, replacing the original one-to-many proximity detection (i.e., detection against local paths). Only used in localization mode (Mem/IncrementalMemory=false), otherwise original one-to-many proximity detection is done. Note also that if graph is modified (i.e., memory management is enabled or robot jumps from one disjoint session to another in same database), the global scan map is cleared and one-to-many proximity detection is reverted to original approach.]
Param: RGBD/ProximityMaxGraphDepth = "50"                  [Maximum depth from the current/last loop closure location and the local loop closure hypotheses. Set 0 to ignore.]
Param: RGBD/ProximityMaxPaths = "3"                        [Maximum paths compared (from the most recent) for proximity detection. 0 means no limit.]
Param: RGBD/ProximityMergedScanCovFactor = "100.0"         [Covariance factor for one-to-many proximity detection (when RGBD/ProximityPathMaxNeighbors>0 and scans are used).]
Param: RGBD/ProximityOdomGuess = "false"                   [Use odometry as motion guess for one-to-one proximity detection.]
Param: RGBD/ProximityPathFilteringRadius = "1"             [Path filtering radius to reduce the number of nodes to compare in a path in one-to-many proximity detection. The nearest node in a path should be inside that radius to be considered for one-to-one proximity detection.]
Param: RGBD/ProximityPathMaxNeighbors = "0"                [Maximum neighbor nodes compared on each path for one-to-many proximity detection. Set to 0 to disable one-to-many proximity detection (by merging the laser scans).]
Param: RGBD/ProximityPathRawPosesUsed = "true"             [When comparing to a local path for one-to-many proximity detection, merge the scans using the odometry poses (with neighbor link optimizations) instead of the ones in the optimized local graph.]
Param: RGBD/ScanMatchingIdsSavedInLinks = "true"           [Save scan matching IDs from one-to-many proximity detection in link's user data.]
Param: RGBD/StartAtOrigin = "false"                        [If true, rtabmap will assume the robot is starting from origin of the map. If false, rtabmap will assume the robot is restarting from the last saved localization pose from previous session (the place where it shut down previously). Used only in localization mode (Mem/IncrementalMemory=false).]
Param: Reg/Force3DoF = "false"                             [Force 3 degrees-of-freedom transform (3Dof: x,y and yaw). Parameters z, roll and pitch will be set to 0.]
Param: Reg/RepeatOnce = "true"                             [Do a second registration with the output of the first registration as guess. Only done if no guess was provided for the first registration (like on loop closure). It can be useful if the registration approach used can use a guess to get better matches.]
Param: Reg/Strategy = "0"                                  [0=Vis, 1=Icp, 2=VisIcp]
Param: Rtabmap/ComputeRMSE = "true"                        [Compute root mean square error (RMSE) and publish it in statistics, if ground truth is provided.]
Param: Rtabmap/CreateIntermediateNodes = "false"           [Create intermediate nodes between loop closure detection. Only used when Rtabmap/DetectionRate>0.]
Param: Rtabmap/DetectionRate = "1"                         [Detection rate (Hz). RTAB-Map will filter input images to satisfy this rate.]
Param: Rtabmap/ImageBufferSize = "1"                       [Data buffer size (0 min inf).]
Param: Rtabmap/ImagesAlreadyRectified = "true"             [Images are already rectified. By default RTAB-Map assumes that received images are rectified. If they are not, they can be rectified by RTAB-Map if this parameter is false.]
Param: Rtabmap/LoopGPS = "true"                            [Use GPS to filter likelihood (if GPS is recorded). Only locations inside the local radius "RGBD/LocalRadius" of the current GPS location are considered for loop closure detection.]
Param: Rtabmap/LoopRatio = "0"                             [The loop closure hypothesis must be over LoopRatio x lastHypothesisValue.]
Param: Rtabmap/LoopThr = "0.11"                            [Loop closing threshold.]
Param: Rtabmap/MaxRepublished = "2"                        [Maximum nodes republished when requesting missing data. When RGBD/Enabled=false, only loop closure data is republished, otherwise the closest nodes from the current localization are republished first. Ignored if Rtabmap/PublishLastSignature=false.]
Param: Rtabmap/MaxRetrieved = "2"                          [Maximum nodes retrieved at the same time from LTM.]
Param: Rtabmap/MemoryThr = "0"                             [Maximum nodes in the Working Memory (0 means infinity). Similar to "Rtabmap/TimeThr", when the number of nodes in Working Memory (WM) exceeds this treshold, some nodes are transferred to Long-Term Memory to keep WM size fixed.]
Param: Rtabmap/PublishLastSignature = "true"               [Publishing last signature.]
Param: Rtabmap/PublishLikelihood = "true"                  [Publishing likelihood.]
Param: Rtabmap/PublishPdf = "true"                         [Publishing pdf.]
Param: Rtabmap/PublishRAMUsage = "false"                   [Publishing RAM usage in statistics (may add a small overhead to get info from the system).]
Param: Rtabmap/PublishStats = "true"                       [Publishing statistics.]
Param: Rtabmap/RectifyOnlyFeatures = "false"               [If "Rtabmap/ImagesAlreadyRectified" is false and this parameter is true, the whole RGB image will not be rectified, only the features. Warning: As projection of RGB-D image to point cloud is assuming that images are rectified, the generated point cloud map will have wrong colors if this parameter is true.]
Param: Rtabmap/SaveWMState = "false"                       [Save working memory state after each update in statistics.]
Param: Rtabmap/StartNewMapOnGoodSignature = "false"        [Start a new map only if the first signature is not bad (i.e., has enough features, see Kp/BadSignRatio).]
Param: Rtabmap/StartNewMapOnLoopClosure = "false"          [Start a new map only if there is a global loop closure with a previous map.]
Param: Rtabmap/StatisticLogged = "false"                   [Logging enabled.]
Param: Rtabmap/StatisticLoggedHeaders = "true"             [Add column header description to log files.]
Param: Rtabmap/StatisticLogsBufferedInRAM = "true"         [Statistic logs buffered in RAM instead of written to hard drive after each iteration.]
Param: Rtabmap/TimeThr = "0"                               [Maximum time allowed for map update (ms) (0 means infinity). When map update time exceeds this fixed time threshold, some nodes in Working Memory (WM) are transferred to Long-Term Memory to limit the size of the WM and decrease the update time.]
Param: Rtabmap/VirtualPlaceLikelihoodRatio = "0"           [Likelihood ratio for virtual place (for no loop closure hypothesis): 0=Mean / StdDev, 1=StdDev / (Max-Mean)]
Param: Rtabmap/WorkingDirectory = "/home/yanzj/.ros"       [Working directory.]
Param: SIFT/ContrastThreshold = "0.04"                     [The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector. Not used by CudaSift (see SIFT/GaussianThreshold instead).]
Param: SIFT/EdgeThreshold = "10"                           [The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).]
Param: SIFT/GaussianThreshold = "2.0"                      [CudaSift: Threshold on difference of Gaussians for feature pruning. The higher the threshold, the less features are produced by the detector.]
Param: SIFT/Gpu = "false"                                  [CudaSift: Use GPU version of SIFT. This option is enabled only if RTAB-Map is built with CudaSift dependency and GPUs are detected.]
Param: SIFT/NOctaveLayers = "3"                            [The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution. Not used by CudaSift, the number of octaves is still computed automatically.]
Param: SIFT/PreciseUpscale = "false"                       [Whether to enable precise upscaling in the scale pyramid (OpenCV >= 4.8).]
Param: SIFT/RootSIFT = "false"                             [Apply RootSIFT normalization of the descriptors.]
Param: SIFT/Sigma = "1.6"                                  [The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.]
Param: SIFT/Upscale = "false"                              [CudaSift: Whether to enable upscaling.]
Param: SURF/Extended = "false"                             [Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).]
Param: SURF/GpuKeypointsRatio = "0.01"                     [Used with SURF GPU.]
Param: SURF/GpuVersion = "false"                           [GPU-SURF: Use GPU version of SURF. This option is enabled only if OpenCV is built with CUDA and GPUs are detected.]
Param: SURF/HessianThreshold = "500"                       [Threshold for hessian keypoint detector used in SURF.]
Param: SURF/OctaveLayers = "2"                             [Number of octave layers within each octave.]
Param: SURF/Octaves = "4"                                  [Number of pyramid octaves the keypoint detector will use.]
Param: SURF/Upright = "false"                              [Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).]
Param: Stereo/DenseStrategy = "0"                          [0=cv::StereoBM, 1=cv::StereoSGBM]
Param: Stereo/Eps = "0.01"                                 [[Stereo/OpticalFlow=true] Epsilon stop criterion.]
Param: Stereo/Gpu = "false"                                [[Stereo/OpticalFlow=true] Enable GPU version of the optical flow approach (only available if OpenCV is built with CUDA).]
Param: Stereo/Iterations = "30"                            [Maximum iterations.]
Param: Stereo/MaxDisparity = "128.0"                       [Maximum disparity.]
Param: Stereo/MaxLevel = "5"                               [Maximum pyramid level.]
Param: Stereo/MinDisparity = "0.5"                         [Minimum disparity.]
Param: Stereo/OpticalFlow = "true"                         [Use optical flow to find stereo correspondences, otherwise a simple block matching approach is used.]
Param: Stereo/SSD = "true"                                 [[Stereo/OpticalFlow=false] Use Sum of Squared Differences (SSD) window, otherwise Sum of Absolute Differences (SAD) window is used.]
Param: Stereo/WinHeight = "3"                              [Window height.]
Param: Stereo/WinWidth = "15"                              [Window width.]
Param: StereoBM/BlockSize = "15"                           [See cv::StereoBM]
Param: StereoBM/Disp12MaxDiff = "-1"                       [See cv::StereoBM]
Param: StereoBM/MinDisparity = "0"                         [See cv::StereoBM]
Param: StereoBM/NumDisparities = "128"                     [See cv::StereoBM]
Param: StereoBM/PreFilterCap = "31"                        [See cv::StereoBM]
Param: StereoBM/PreFilterSize = "9"                        [See cv::StereoBM]
Param: StereoBM/SpeckleRange = "4"                         [See cv::StereoBM]
Param: StereoBM/SpeckleWindowSize = "100"                  [See cv::StereoBM]
Param: StereoBM/TextureThreshold = "10"                    [See cv::StereoBM]
Param: StereoBM/UniquenessRatio = "15"                     [See cv::StereoBM]
Param: StereoSGBM/BlockSize = "15"                         [See cv::StereoSGBM]
Param: StereoSGBM/Disp12MaxDiff = "1"                      [See cv::StereoSGBM]
Param: StereoSGBM/MinDisparity = "0"                       [See cv::StereoSGBM]
Param: StereoSGBM/Mode = "2"                               [See cv::StereoSGBM]
Param: StereoSGBM/NumDisparities = "128"                   [See cv::StereoSGBM]
Param: StereoSGBM/P1 = "2"                                 [See cv::StereoSGBM]
Param: StereoSGBM/P2 = "5"                                 [See cv::StereoSGBM]
Param: StereoSGBM/PreFilterCap = "31"                      [See cv::StereoSGBM]
Param: StereoSGBM/SpeckleRange = "4"                       [See cv::StereoSGBM]
Param: StereoSGBM/SpeckleWindowSize = "100"                [See cv::StereoSGBM]
Param: StereoSGBM/UniquenessRatio = "20"                   [See cv::StereoSGBM]
Param: SuperPoint/Cuda = "true"                            [Use Cuda device for Torch, otherwise CPU device is used by default.]
Param: SuperPoint/ModelPath = ""                           [[Required] Path to pre-trained weights Torch file of SuperPoint (*.pt).]
Param: SuperPoint/NMS = "true"                             [If true, non-maximum suppression is applied to detected keypoints.]
Param: SuperPoint/NMSRadius = "4"                          [[SuperPoint/NMS=true] Minimum distance (pixels) between keypoints.]
Param: SuperPoint/Threshold = "0.010"                      [Detector response threshold to accept keypoint.]
Param: VhEp/Enabled = "false"                              [Verify visual loop closure hypothesis by computing a fundamental matrix. This is done prior to transformation computation when RGBD/Enabled is enabled.]
Param: VhEp/MatchCountMin = "8"                            [Minimum of matching visual words pairs to accept the loop hypothesis.]
Param: VhEp/RansacParam1 = "3"                             [Fundamental matrix (see cvFindFundamentalMat()): Max distance (in pixels) from the epipolar line for a point to be inlier.]
Param: VhEp/RansacParam2 = "0.99"                          [Fundamental matrix (see cvFindFundamentalMat()): Performance of RANSAC.]
Param: Vis/BundleAdjustment = "1"                          [Optimization with bundle adjustment: 0=disabled, 1=g2o, 2=cvsba, 3=Ceres.]
Param: Vis/CorFlowEps = "0.01"                             [[Vis/CorType=1] See cv::calcOpticalFlowPyrLK(). Used for optical flow approach.]
Param: Vis/CorFlowGpu = "false"                            [[Vis/CorType=1] Enable GPU version of the optical flow approach (only available if OpenCV is built with CUDA).]
Param: Vis/CorFlowIterations = "30"                        [[Vis/CorType=1] See cv::calcOpticalFlowPyrLK(). Used for optical flow approach.]
Param: Vis/CorFlowMaxLevel = "3"                           [[Vis/CorType=1] See cv::calcOpticalFlowPyrLK(). Used for optical flow approach.]
Param: Vis/CorFlowWinSize = "16"                           [[Vis/CorType=1] See cv::calcOpticalFlowPyrLK(). Used for optical flow approach.]
Param: Vis/CorGuessMatchToProjection = "false"             [[Vis/CorType=0] Match frame's corners to source's projected points (when guess transform is provided) instead of projected points to frame's corners.]
Param: Vis/CorGuessWinSize = "40"                          [[Vis/CorType=0] Matching window size (pixels) around projected points when a guess transform is provided to find correspondences. 0 means disabled.]
Param: Vis/CorNNDR = "0.8"                                 [[Vis/CorType=0] NNDR: nearest neighbor distance ratio. Used for knn features matching approach.]
Param: Vis/CorNNType = "1"                                 [[Vis/CorType=0] kNNFlannNaive=0, kNNFlannKdTree=1, kNNFlannLSH=2, kNNBruteForce=3, kNNBruteForceGPU=4, BruteForceCrossCheck=5, SuperGlue=6, GMS=7. Used for features matching approach.]
Param: Vis/CorType = "0"                                   [Correspondences computation approach: 0=Features Matching, 1=Optical Flow]
Param: Vis/DepthAsMask = "true"                            [Use depth image as mask when extracting features.]
Param: Vis/DepthMaskFloorThr = "0.0"                       [Filter floor from depth mask below specified threshold (m) before extracting features. 0 means disabled, negative means remove all objects above the floor threshold instead. Ignored if Vis/DepthAsMask is false.]
Param: Vis/EpipolarGeometryVar = "0.1"                     [[Vis/EstimationType = 2] Epipolar geometry maximum variance to accept the transformation.]
Param: Vis/EstimationType = "1"                            [Motion estimation approach: 0:3D->3D, 1:3D->2D (PnP), 2:2D->2D (Epipolar Geometry)]
Param: Vis/FeatureType = "8"                               [0=SURF 1=SIFT 2=ORB 3=FAST/FREAK 4=FAST/BRIEF 5=GFTT/FREAK 6=GFTT/BRIEF 7=BRISK 8=GFTT/ORB 9=KAZE 10=ORB-OCTREE 11=SuperPoint 12=SURF/FREAK 13=GFTT/DAISY 14=SURF/DAISY 15=PyDetector]
Param: Vis/GridCols = "1"                                  [Number of columns of the grid used to extract uniformly "Vis/MaxFeatures / grid cells" features from each cell.]
Param: Vis/GridRows = "1"                                  [Number of rows of the grid used to extract uniformly "Vis/MaxFeatures / grid cells" features from each cell.]
Param: Vis/InlierDistance = "0.1"                          [[Vis/EstimationType = 0] Maximum distance for feature correspondences. Used by 3D->3D estimation approach.]
Param: Vis/Iterations = "300"                              [Maximum iterations to compute the transform.]
Param: Vis/MaxDepth = "0"                                  [Max depth of the features (0 means no limit).]
Param: Vis/MaxFeatures = "1000"                            [0 no limits.]
Param: Vis/MeanInliersDistance = "0.0"                     [Maximum distance (m) of the mean distance of inliers from the camera to accept the transformation. 0 means disabled.]
Param: Vis/MinDepth = "0"                                  [Min depth of the features (0 means no limit).]
Param: Vis/MinInliers = "20"                               [Minimum feature correspondences to compute/accept the transformation.]
Param: Vis/MinInliersDistribution = "0.0"                  [Minimum distribution value of the inliers in the image to accept the transformation. The distribution is the second eigen value of the PCA (Principal Component Analysis) on the keypoints of the normalized image [-0.5, 0.5]. The value would be between 0 and 0.5. 0 means disabled.]
Param: Vis/PnPFlags = "0"                                  [[Vis/EstimationType = 1] PnP flags: 0=Iterative, 1=EPNP, 2=P3P]
Param: Vis/PnPMaxVariance = "0.0"                          [[Vis/EstimationType = 1] Max linear variance between 3D point correspondences after PnP. 0 means disabled.]
Param: Vis/PnPRefineIterations = "0"                       [[Vis/EstimationType = 1] Refine iterations. Set to 0 if "Vis/BundleAdjustment" is also used.]
Param: Vis/PnPReprojError = "2"                            [[Vis/EstimationType = 1] PnP reprojection error.]
Param: Vis/PnPSamplingPolicy = "1"                         [[Vis/EstimationType = 1] Multi-camera random sampling policy: 0=AUTO, 1=ANY, 2=HOMOGENEOUS. With HOMOGENEOUS policy, RANSAC will be done uniformly against all cameras, so at least 2 matches per camera are required. With ANY policy, RANSAC is not constraint to sample on all cameras at the same time. AUTO policy will use HOMOGENEOUS if there are at least 2 matches per camera, otherwise it will fallback to ANY policy.]
Param: Vis/PnPSplitLinearCovComponents = "false"           [[Vis/EstimationType = 1] Compute variance for each linear component instead of using the combined XYZ variance for all linear components.]
Param: Vis/PnPVarianceMedianRatio = "4"                    [[Vis/EstimationType = 1] Ratio used to compute variance of the estimated transformation if 3D correspondences are provided (should be > 1). The higher it is, the smaller the covariance will be. With accurate depth estimation, this could be set to 2. For depth estimated by stereo, 4 or more maybe used to ignore large errors of very far points.]
Param: Vis/RefineIterations = "5"                          [[Vis/EstimationType = 0] Number of iterations used to refine the transformation found by RANSAC. 0 means that the transformation is not refined.]
Param: Vis/RoiRatios = "0.0 0.0 0.0 0.0"                   [Region of interest ratios [left, right, top, bottom].]
Param: Vis/SSC = "false"                                   [If true, SSC (Suppression via Square Covering) is applied to limit keypoints.]
Param: Vis/SubPixEps = "0.02"                              [See cv::cornerSubPix().]
Param: Vis/SubPixIterations = "0"                          [See cv::cornerSubPix(). 0 disables sub pixel refining.]
Param: Vis/SubPixWinSize = "3"                             [See cv::cornerSubPix().]
Param: g2o/Baseline = "0.075"                              [When doing bundle adjustment with RGB-D data, we can set a fake baseline (m) to do stereo bundle adjustment (if 0, mono bundle adjustment is done). For stereo data, the baseline in the calibration is used directly.]
Param: g2o/Optimizer = "0"                                 [0=Levenberg 1=GaussNewton]
Param: g2o/PixelVariance = "1.0"                           [Pixel variance used for bundle adjustment.]
Param: g2o/RobustKernelDelta = "8"                         [Robust kernel delta used for bundle adjustment (0 means don't use robust kernel). Observations with chi2 over this threshold will be ignored in the second optimization pass.]
Param: g2o/Solver = "0"                                    [0=csparse 1=pcg 2=cholmod 3=Eigen]

```



```
ros2 service call /goto_controller/enable std_srvs/srv/Trigger

ros2 topic pub --once /goto_controller/goal geometry_msgs/msg/Pose "
position:
  x: 0.0
  y: 0.0
  z: 0.0
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
"




ros2 service call /arm_controller/enable std_srvs/srv/Trigger "{}"

ros2 topic pub --once /arm_controller/goal_pose std_msgs/msg/Float32MultiArray \
  "{data: [265.3, -7.6, 665.3, 177.9, -45, 0.2]}"
```