# DREAM Hardware-Side Setup (ROS2 Humble)

This document covers hardware-side installation for robot drivers, sensors, and
ROS2 workspace build. The commands below assume the reference workspace layout
under `~/DREAM_ws`.

## 0. Hardware, robot model, and calibration assumptions

Read this section before installing. The provided launch files and configs are
prepared for our reference hardware:

- AgileX Ranger Mini V3 mobile base
- UFACTORY xArm6 arm with gripper
- Livox MID-360 LiDAR/IMU
- Intel RealSense D435i RGB-D camera

DREAM's SLAM and mapping pipeline is built from LiDAR + IMU + RGB-D input:

- FAST-LIO2 uses the LiDAR/IMU stream for odometry.
- RTAB-Map consumes the LiDAR point cloud, FAST-LIO2 odometry, and synchronized
  RGB-D observations.
- The DREAM ROS2 bridge reads the resulting TF, RGB-D, map, navigation, and
  manipulation state.

The RGB-D camera does not need to be mounted exactly like ours. It only needs a
valid calibrated TF chain to the LiDAR/tracking frame:

- Fixed camera layout: publish a calibrated static transform between the RGB-D
  camera and the LiDAR/base frame, and adjust the launch/URDF if you are not
  using the reference xArm-mounted camera layout.
- Arm-mounted camera layout: publish arm joint states and a URDF, then provide
  the calibrated transform from the arm link to the camera. Our system uses this
  layout: the D435i is mounted on xArm6 `link6`, so `link6_to_camera` is static
  while the camera-to-LiDAR/base transform changes with the arm joints.

You can build DREAM on different hardware or a different physical mounting if
you provide matching ROS drivers, topics, robot model, and extrinsics. If your
hardware is the same as ours but the mounting is different, the system is still
usable, but you must redo the calibration/model values below.

Reference model and calibration files:

- CAD source-of-truth files:
  - [ROBOT.zip](ROBOT.zip)
  - [ROBOT.jpg](ROBOT.jpg)
- URDF files:
  - [`src/dream_ros2_bridge/urdf/rangerminiv3_with_xarm6.urdf`](../src/dream_ros2_bridge/urdf/rangerminiv3_with_xarm6.urdf)
  - [`src/dream_ros2_bridge/urdf/rangerminiv3_with_xarm6_simple.urdf`](../src/dream_ros2_bridge/urdf/rangerminiv3_with_xarm6_simple.urdf)
- Runtime extrinsics:
  - [`src/dream_ros2_bridge/config/extrinsics_cad.yaml`](../src/dream_ros2_bridge/config/extrinsics_cad.yaml)

The provided extrinsics are exported from CAD rather than from a separate
calibration procedure.
[`dream_node_start.launch.py`](../src/dream_ros2_bridge/launch/dream_node_start.launch.py)
expects these transform keys by default: `link6_to_camera`, `body_to_livox`,
`livox_to_base`, and `base_to_footprint`. If your setup differs, update the URDF
and extrinsics file or pass another file with
`extrinsics_file:=your_extrinsics.yaml`.

Calibration accuracy note: our experiments did not use high-precision
vision-based hand-eye calibration, and we did not add complex hardware time
synchronization beyond the standard ROS sensor timestamps and approximate
synchronization used by RTAB-Map. We measured the transforms directly in CAD and
exported them to
[`extrinsics_cad.yaml`](../src/dream_ros2_bridge/config/extrinsics_cad.yaml).
In our indoor navigation and
pick-and-place experiments, this coarse CAD-based calibration was sufficient;
small residual transform errors were tolerated well by RTAB-Map's multi-sensor
fusion. You can see a large-scale SLAM example here:
[slam_test.mp4](https://bjhyzj.github.io/dream-web/media/videos/slam_test.mp4).
We still encourage higher-precision calibration when available, especially if
your robot has a different mounting layout or you need tighter manipulation
accuracy.

## Target environment

- Ubuntu 22.04
- ROS2 Humble
- CUDA 12.1
- System RAM >= 16 GB

## 1. Install system dependencies (`ROS2`)

Please install ROS 2 Humble first by following the official Ubuntu installation guide: [ROS 2 Humble Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).

Then install additional package dependencies used by this stack:

```bash
sudo apt update
sudo apt install -y \
  libasio-dev libboost-all-dev \
  portaudio19-dev libfuse2 \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-robot-state-publisher
```


## 2. Install Livox SDK2

```bash
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## 3. Install librealsense

```bash
# Build librealsense from source to get realsense-viewer and aligned runtime behavior.
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.5

# Install build dependencies required by librealsense.
sudo apt update
sudo apt install -y cmake make g++ libglfw3-dev libusb-1.0-0-dev libgtk-3-dev pkg-config

mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=true
make -j$(nproc)
sudo make install
sudo ldconfig
```

Optional:

```bash
echo 'alias realsense-viewer=/usr/local/bin/realsense-viewer' >> ~/.bashrc
source ~/.bashrc
```

## 4. Livox MID-360 FOV / range mask notes

- Driver runtime config file:
  - [`MID360_config.json`](../src/dream_ros2_bridge/config/MID360_config.json)
- [`livox_mid360_driver.launch.py`](../src/dream_ros2_bridge/launch/livox_mid360_driver.launch.py)
  loads this JSON as `user_config_path`.
- We configure MID-360 FOV with LivoxViewer2 and mask the rear 90 degrees of the robot.
- This mask is intentional: it removes points from the robot back side to reduce self-observation noise.
- In practice, LivoxViewer2 configuration is more stable on Windows 11 than Ubuntu 22.04.
- Relevant references:
  - https://github.com/Livox-SDK/livox_ros_driver2/issues/199
  - https://github.com/Livox-SDK/livox_ros_driver2/issues/145
- If you need to reproduce exactly the same behavior, keep the same rear 90-degree mask policy during deployment.

## 5. Create workspace and clone repositories

```bash
mkdir -p ~/DREAM_ws
cd ~/DREAM_ws
git clone https://github.com/BJHYZJ/DREAM.git --recursive

mkdir -p ~/DREAM_ws/DREAM_ws/src
cd ~/DREAM_ws/DREAM_ws/src
ln -s ~/DREAM_ws/DREAM/src/dream_ros2_bridge .

git clone https://github.com/hello-binit/ros2_numpy -b humble
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
git clone https://github.com/Ericsii/FAST_LIO_ROS2.git --recursive
git clone https://github.com/BJHYZJ/ugv_sdk.git
git clone https://github.com/BJHYZJ/ranger_ros2.git
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b humble

git clone https://github.com/introlab/rtabmap.git
git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git
```

If needed, remove binary `rtabmap` packages to avoid conflicts:

```bash
sudo apt remove ros-$ROS_DISTRO-rtabmap*
```

## 6. Patch livox ROS2 package files

```bash
cd ~/DREAM_ws/DREAM_ws/src/livox_ros_driver2
cp package_ROS2.xml package.xml
cp -rf launch_ROS2/ launch/
```

## 7. Install ROS dependencies

```bash
cd ~/DREAM_ws/DREAM_ws
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro=humble --skip-keys="librealsense2"
```

If `rosdep` times out:

```bash
sudo mkdir -p /etc/ros/rosdep/sources.list.d/
sudo curl -o /etc/ros/rosdep/sources.list.d/20-default.list -L \
  https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list
rosdep update
```

## 8. Build

```bash
cd ~/DREAM_ws/DREAM_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DHUMBLE_ROS=humble -DCMAKE_BUILD_TYPE=Release
```

Optional:

```bash
# To build with rgbd_cameras>1 support and/or subscribe_user_data support:
colcon build --symlink-install --cmake-args \
  -DRTABMAP_SYNC_MULTI_RGBD=ON \
  -DRTABMAP_SYNC_USER_DATA=ON \
  -DCMAKE_BUILD_TYPE=Release
```

## 9. Runtime environment

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
```

Recommended `~/.bashrc` additions:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# for RTAB-Map
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
# Optional, but if you like colored logs:
export RCUTILS_COLORIZED_OUTPUT=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Cyclone prefers multicast by default, if your router got too much spammed,
# disable multicast with (https://github.com/ros2/rmw_cyclonedds/issues/489):
export CYCLONEDDS_URI="<Disc><DefaultMulticastAddress>0.0.0.0</></>"
export PYDEVD_WARN_EVALUATION_TIMEOUT=10
```

## 10. Validate robot model and extrinsics

Before running the full system, verify that the TF tree matches your physical
robot:

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
ros2 launch dream_ros2_bridge dream_node_start.launch.py use_rviz:=true
```

In RViz, check that the mobile base, arm, LiDAR, RGB-D camera, and
`base_footprint` frames are in the expected positions. If you updated the CAD
assembly, sensor mounting, or robot model, update the URDF and
[`extrinsics_cad.yaml`](../src/dream_ros2_bridge/config/extrinsics_cad.yaml),
then rebuild and source the ROS2 workspace again.
