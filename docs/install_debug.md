# Troubleshooting

Use the [hardware setup](hardware_install.md) and
[service-machine setup](service_machine_install.md) guides for installation.
The checks below cover sensor input, transforms, mapping, and Python runtime
issues in the reference ROS2 Humble system.

## ROS environment and node discovery

Source ROS and the built workspace in each hardware-side terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/DREAM_ws/DREAM_ws/install/setup.bash
ros2 node list
ros2 topic list
ros2 service list
```

If the workspace is elsewhere, use its `install/setup.bash`. Keep
`ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` consistent between terminals that need
to communicate. The hardware setup uses `rmw_cyclonedds_cpp`.

After changing the ROS environment, refresh the CLI discovery daemon if its
node list is stale:

```bash
ros2 daemon stop
ros2 daemon start
```

## Sensor input and transforms

Check the topics consumed by
[`dream_rtabmap_slam.launch.py`](../src/dream_ros2_bridge/launch/dream_rtabmap_slam.launch.py):

```bash
ros2 topic hz /fast_lio2/Odometry
ros2 topic hz /fast_lio2/cloud_registered_body
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info --once
```

Run each rate check separately and stop it with Ctrl+C. If a topic is absent,
check its driver and the topic arguments in the launch file. For RGB-D input,
inspect the camera images and aligned depth in RViz alongside the timestamps
and camera calibration.

Inspect the transform tree:

```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo livox_frame base_link
```

RTAB-Map uses `body` as its default tracking frame. It needs transforms from
that frame to the camera and base. For the reference arm-mounted camera, check
that robot joint states are arriving and that the URDF and
[`extrinsics_cad.yaml`](../src/dream_ros2_bridge/config/extrinsics_cad.yaml)
match the sensor mounting. See the calibration section of the
[hardware guide](hardware_install.md#0-hardware-robot-model-and-calibration-assumptions).

## Map data and RTAB-Map

Inspect the graph and node observations published for DREAM:

```bash
ros2 topic echo /dream/mapData --field graph --once
ros2 topic echo /dream/mapData --field nodes --once
ros2 topic bw /dream/mapData
```

The node array can be large. Use the installed message definitions to inspect
its fields:

```bash
ros2 interface show rtabmap_msgs/msg/MapData
ros2 interface show rtabmap_msgs/msg/MapGraph
ros2 interface show rtabmap_msgs/msg/Node
```

To inspect mapping parameters and an existing database:

```bash
ros2 param list /dream_rtabmap/rtabmap
rtabmap --params
rtabmap-info ~/.ros/rtabmap.db
```

Substitute the node name from `ros2 node list` or the configured database path
if you changed the launch defaults. Mapping can be paused and resumed through
the RTAB-Map services:

```bash
ros2 service call /dream_rtabmap/rtabmap/pause std_srvs/srv/Empty '{}'
ros2 service call /dream_rtabmap/rtabmap/resume std_srvs/srv/Empty '{}'
```

## Python and CUDA dependencies

For the service machine, activate the intended Conda environment and inspect
the interpreter and import paths:

```bash
conda activate dream
python -c "import sys; print(sys.executable); print('\n'.join(sys.path))"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If packages from `~/.local` are being imported into the Conda environment,
disable user-site imports for that environment:

```bash
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate dream
python -c "import site; print(site.ENABLE_USER_SITE)"
```

The last command should print `False`. Build native extensions in the same
Python and CUDA environment used to run them. The
[service-machine guide](service_machine_install.md) lists the reference
versions and build steps for MinkowskiEngine, PointNet2, and AnyGrasp.

## Common startup errors

| Symptom | What to check |
| --- | --- |
| RTAB-Map exits with an OpenCV allocation error | Check which OpenCV libraries the executable loads and whether RTAB-Map and its ROS packages were built against the same installation. An implausibly large allocation request can indicate a library mismatch; the error alone does not establish the cause. |
| Nav2 reports `InvalidParameterTypeException` for `height` | Match the YAML value to the parameter's declared type. The recorded error expected an integer but received a floating-point value. Check the parameter file named in the launch output. |
| AnyGrasp cannot load `libcrypto.so.1.1` or validate its license | Follow the [AnyGrasp license setup](service_machine_install.md#23-anygrasp-license), then run the license checker in the `anygrasp` environment. |
| A service cannot bind its port | Use `ss -ltnp` to identify the listening process and check the server/client port settings. The reference AnyGrasp command uses port `5557`. |
| The camera disconnects or stops publishing | Check the USB connection, power, driver output, and whether another process is using the device. Inspect the camera in `realsense-viewer` before restarting the ROS camera driver. |

For an OpenCV build mismatch, compare `pkg-config --modversion opencv4` with
the libraries shown by `ldd` for the affected executable, then rebuild the
RTAB-Map packages against the intended installation using the
[hardware setup guide](hardware_install.md).
