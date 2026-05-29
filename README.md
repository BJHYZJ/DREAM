# DREAM: Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation

## Introduction
Reliable mobile manipulation in dynamic indoor environments requires a 3D semantic representation that remains consistent with the evolving real world. Most existing systems rely on pre-built maps, assume static environments, or presuppose highly accurate camera poses; when these assumptions break, navigation and manipulation operate on stale information.

DREAM is a mobile manipulation framework for previously unseen indoor environments without any pre-built map. It integrates a lightweight indoor LiDAR-Inertial-Visual SLAM backend with dynamic spatio-semantic memory, Redundancy-Aware Memory Pruning, hybrid localization, task-oriented navigation, and robust grasping and placement strategies.

![log](docs/dream.png)

## Requirements

- Hardware side: Ubuntu 22.04, ROS2 Humble, CUDA 12.1, system RAM >= 16 GB
- Service machine: Ubuntu 22.04, no ROS distro restriction, CUDA 12.1, at least 24 GB VRAM on one GPU or two GPUs with at least 16 GB VRAM each

## Installation Guides

- Hardware-side installation: [hardware_install.md](docs/hardware_install.md)
- Service-machine installation: [service_machine_install.md](docs/service_machine_install.md)

## Run

### Hardware machine (ROS runtime)

Always source ROS2 + workspace first in each terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
```

Start Ranger mini v3
```bash
sudo apt install -y can-utils
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000
```


Terminal 1 (Start Sensors and Fast-LIO2):
```bash
ros2 launch dream_ros2_bridge dream_node_start.launch.py use_rviz:=false
```

Common options:

```bash
ros2 launch dream_ros2_bridge dream_node_start.launch.py use_rviz:=true
ros2 launch dream_ros2_bridge dream_node_start.launch.py use_simple_urdf:=false
ros2 launch dream_ros2_bridge dream_node_start.launch.py robot_ip:=192.168.1.233 joint_states_rate:=50
```


Terminal 2 (RTAB-Map SLAM):

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
ros2 launch dream_ros2_bridge dream_rtabmap_slam.launch.py
```

RTAB-Map's `frame_id` is the tracking frame used by the bridge pose names:
`tracking_in_map_pose` is `map_T_frame_id`, while `camera_in_tracking_pose`
and `base_in_tracking_pose` are local transforms relative to that frame. The
default is `body`, matching FAST-LIO2's body frame. If you are not using
FAST-LIO2, change `frame_id` in `dream_rtabmap_slam.launch.py` and make sure TF
from that frame to the camera and base exists.

Terminal 3 (DREAM ROS2 bridge server):

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
ros2 launch dream_ros2_bridge dream_server.launch.py
```

You can also start the hardware-side ROS runtime with one script. It uses
Terminator to open one terminal window with three split panes for the node start
launch, RTAB-Map SLAM, and DREAM ROS2 bridge server:

```bash
sudo apt install -y terminator
```

```bash
cd ~/DREAM_ws/DREAM_ws
bash src/dream_ros2_bridge/run_hardware_system.sh
```

If a machine needs more time for sensors or RTAB-Map startup, increase the waits:

```bash
NODE_START_WAIT=25 RTABMAP_WAIT=15 bash src/dream_ros2_bridge/run_hardware_system.sh
```

If `can0` is already up, skip CAN setup:

```bash
SKIP_CAN_SETUP=1 bash src/dream_ros2_bridge/run_hardware_system.sh
```

### Service machine (AnyGrasp server)

#### Run Anygrasp

If you deploy manipulation service on a separate server, run:

```bash
conda activate anygrasp
cd src/anygrasp_manipulation
python demo.py --open_communication --port 5557
```

Optional

If you want to debug AnyGrasp locally without running the robot, disable socket communication and enable debug visualization:

```bash
conda activate anygrasp
cd ~/DREAM_ws/DREAM/src/anygrasp_manipulation
python demo.py --debug
```

This runs `demo.py` with `open_communication` disabled, so it will load the local example data instead of waiting for the robot and lets you inspect the AnyGrasp outputs directly.

#### Run Dream
```bash
cd ~/path/to/DREAM
conda activate dream
python src/dream/app/run_dream.py --robot_ip 10.33.140.5 --server_ip 127.0.0.1  --skip_confirmations
```

Change `--robot_ip` to your hardware machine.
If you run anygrasp in other machine, you should change `--server_ip`


## Reference
- Dynamem: [https://dynamem.github.io/](https://dynamem.github.io/)
- DovSG: [https://bjhyzj.github.io/dovsg-web/](https://bjhyzj.github.io/dovsg-web/)
- Ok-Robot: [https://ok-robot.github.io/](https://ok-robot.github.io/)
