# Copy from: https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/lidar3d.launch.py

# Description:
#   In this example, we keep only minimal data to do LiDAR SLAM.
#
# Example:
#   Launch your lidar sensor:
#   $ ros2 launch velodyne_driver velodyne_driver_node-VLP16-launch.py
#   $ ros2 launch velodyne_pointcloud velodyne_transform_node-VLP16-launch.py
#
#   If an IMU is used, make sure TF between lidar/base frame and imu is
#     already calibrated. In this example, we assume the imu topic has 
#     already the orientation estimated, if not, you can use 
#     imu_filter_madgwick_node (with use_mag:=false publish_tf:=false)
#     and set imu_topic to output topic of the filter.
#
#   If a camera is used, make sure TF between lidar/base frame and camera is
#     already calibrated. To provide image data to this example, you should use
#     rtabmap_sync's rgbd_sync or stereo_sync node.
#
#   Launch the example by adjusting the lidar topic and base frame:
#   $ ros2 launch rtabmap_examples lidar3d.launch.py lidar_topic:=/velodyne_points frame_id:=velodyne

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context: LaunchContext, *args, **kwargs):
  
  frame_id = LaunchConfiguration('frame_id')
  
  imu_topic = LaunchConfiguration('imu_topic')
  imu_used =  imu_topic.perform(context) != ''
  
  robot_ns = LaunchConfiguration('robot_ns')
  
  rgb_image_topic = LaunchConfiguration('rgb_image_topic')
  rgb_camera_info_topic = LaunchConfiguration('rgb_camera_info_topic')
  depth_image_topic = LaunchConfiguration('depth_image_topic')

  voxel_size = LaunchConfiguration('voxel_size')
  voxel_size_value = float(voxel_size.perform(context))
  
  use_sim_time = LaunchConfiguration('use_sim_time')
  
  lidar_topic = LaunchConfiguration('lidar_topic').perform(context)
  
  # Get odom topic from launch parameter
  odom_topic = LaunchConfiguration('odom_topic').perform(context)
  
  localization = LaunchConfiguration('localization').perform(context)
  localization = localization == 'true' or localization == 'True'

  force_3dof = LaunchConfiguration('force3dof').perform(context)
  
  # Rule of thumb:
  max_correspondence_distance = voxel_size_value * 20.0  # default * 10

  shared_parameters = {
    'use_sim_time': use_sim_time,
    'frame_id': frame_id,
    'qos': LaunchConfiguration('qos'),
    'wait_for_transform': 0.5,
    # Synchronization parameters # 控制rgbd和雷达的同步参数
    'approx_sync': True,  
    'approx_sync_max_interval': '0.1',
    # 'queue_size': 20,
    'topic_queue_size': 20,
    'sync_queue_size': 20,
    # RTAB-Map's internal parameters are strings:
    'Icp/PointToPlane': 'true',
    'Icp/Iterations': '20',
    'Icp/VoxelSize': str(voxel_size_value),
    'Icp/Epsilon': '0.001',
    'Icp/PointToPlaneK': '20',
    'Icp/PointToPlaneRadius': '0',
    'Icp/MaxTranslation': '3',
    'Icp/MaxCorrespondenceDistance': str(max_correspondence_distance),
    # 'Icp/MaxCorrespondenceDistance': '1',
    'Icp/Strategy': '1',
    'Icp/OutlierRatio': '0.7',
  }


  rtabmap_parameters = {
    'subscribe_depth': False,
    'subscribe_rgb': False,
    'subscribe_rgbd': True,
    'subscribe_odom_info': False,  # 只有使用rtabmap中的odom才打开，这里使用了fast_lio2的odometry，所以不能设置为true
    'subscribe_scan_cloud': True,
    'odom_sensor_sync': True, # This will adjust camera position based on difference between lidar and camera stamps.
    'map_frame_id': 'map',
    'odom_frame_id': 'camera_init',  # FAST-LIO's odometry frame
    
    'Rtabmap/DetectionRate': '1.0', 

    # RTAB-Map's internal parameters are strings:
    'RGBD/ProximityMaxGraphDepth': '0',
    'RGBD/ProximityPathMaxNeighbors': '2',
    'RGBD/AngularUpdate': '0.1',
    'RGBD/LinearUpdate': '0.1',
    'RGBD/CreateOccupancyGrid': 'true',
    'RGBD/ForceOdom3DoF': force_3dof,       # 默认: true - Force odometry pose to be 3DoF if Reg/Force3DoF=true.

    'Grid/3D': 'true',   # 显示设置为false，不需要grid，同时设置'Grid/Sensor'为0是为了避免warn: [rtabmap_ranger_xarm.rtabmap]: Setting "Grid/Sensor" parameter to 0 (default 1) as "subscribe_scan" or "subscribe_scan_cloud" or "gen_scan" is true. The occupancy grid map will be constructed from laser scans. To get occupancy grid map from cloud projection, set "Grid/Sensor" to true. To suppress this warning, add <param name="Grid/Sensor" type="string" value="0"/>
    'Grid/Sensor': "1",  # Create occupancy grid from selected sensor: 0=laser scan, 1=depth image(s) or 2=both laser scan and depth image(s).

    'Mem/NotLinkedNodesKept': 'false',
    'Mem/STMSize': '60',
    'Mem/DepthCompressionFormat': '.png',  # 将深度图压缩格式从RVL改为PNG

    'Reg/Force3DoF': force_3dof,
    'Reg/RepeatOnce': 'true',
    'Reg/Strategy': '2',  # 0=Vis, 1=Icp, 2=VisIcp
    'Icp/CorrespondenceRatio': str(LaunchConfiguration('min_loop_closure_overlap').perform(context))
  }
  
  arguments = []
  if localization:
    rtabmap_parameters['Mem/IncrementalMemory'] = 'False'
    rtabmap_parameters['Mem/InitWMWithAllNodes'] = 'True'
  else:
    arguments.append('-d') # This will delete the previous database (~/.ros/rtabmap.db)
  
  # Use FAST-LIO odometry instead of ICP odometry
  remappings = [('odom', odom_topic)]
  if imu_used:
    remappings.append(('imu', 'imu/data'))  # notice namespace problem
  else:
    remappings.append(('imu', 'imu_not_used'))
  
  remappings.append(('rgb/image', rgb_image_topic))
  remappings.append(('rgb/camera_info', rgb_camera_info_topic))
  remappings.append(('depth/image', depth_image_topic))
  
  nodes = [
      Node(
          package='rtabmap_sync', executable='rgbd_sync', output='screen',
          namespace=robot_ns,
          parameters=[{'approx_sync': False, 'use_sim_time': use_sim_time}],  # approx_sync，控制 RGB 图像和深度图像的同步
          remappings=remappings),

      Node(
        package='rtabmap_slam', executable='rtabmap', output='screen',
        namespace=robot_ns,
        parameters=[shared_parameters, rtabmap_parameters, 
                    {'rgbd_cameras': 1}],
        remappings=remappings + [('scan_cloud', lidar_topic)],
        arguments=arguments),
      
      Node(
        condition=IfCondition(LaunchConfiguration('use_rtabmap_viz')),
        package='rtabmap_viz', executable='rtabmap_viz', output='screen',
        namespace=robot_ns,
        parameters=[shared_parameters, rtabmap_parameters, 
                    {'subscribe_depth': False,
                     'subscribe_rgb': False, 
                     'subscribe_rgbd': False,
                     'subscribe_scan_cloud': False}],  # Disable real-time sensor data to avoid coordinate mismatch
        remappings=remappings),  # Use FAST-LIO2 output directly

      Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(
          get_package_share_directory('dream_ros2_bridge'), 'rviz', 'rtabmap_vis.rviz')],
        condition=IfCondition(LaunchConfiguration('use_rviz')))
  ]

  print(imu_used)
  if imu_used:
    nodes.append(
      Node(
        package='imu_filter_madgwick', executable='imu_filter_madgwick_node', output='screen',
        namespace=robot_ns,
        parameters=[{'use_mag': False, 
                      'world_frame':'enu', 
                      'publish_tf':False}],
        remappings=[('imu/data_raw', imu_topic)]))

      
  return nodes
  
def generate_launch_description():
  return LaunchDescription([

    # Launch arguments
    DeclareLaunchArgument(
      'use_sim_time', default_value='false',
      description='Use simulated clock.'),
    
    DeclareLaunchArgument(
      'use_rviz', default_value='true',
      description='Use rviz2'),

    DeclareLaunchArgument(
      'use_rtabmap_viz', default_value='false',
      description='Use rtabmap_viz'),

    DeclareLaunchArgument(
      'frame_id', default_value='base_footprint',
      description='Base frame of the robot.'),
    
    DeclareLaunchArgument(
      'localization', default_value='false',
      description='Localization mode.'),

    DeclareLaunchArgument(
      'lidar_topic', default_value='/fast_lio2/cloud_registered_body',  # /fast_lio2/cloud_registered_body在lidar_frame坐标系下
      description='FAST-LIO2 registered point cloud topic.'),

    DeclareLaunchArgument(
      'odom_topic', default_value='/fast_lio2/Odometry',
      description='Odometry topic from SLAM system (e.g., FAST-LIO2).'),

    DeclareLaunchArgument(
      'imu_topic', default_value='/camera/imu',  # with realsense-d435i imu information
      description='IMU topic (ignored if empty).'),
    
    DeclareLaunchArgument(
      'robot_ns', default_value='rtabmap',
      description='Robot namespace.'),

    DeclareLaunchArgument(
      'rgb_image_topic', default_value='/camera/color/image_raw',
      description='RGB image topic.'),

    DeclareLaunchArgument(
      'rgb_camera_info_topic', default_value='/camera/color/camera_info',
      description='RGB camera info topic.'),

    DeclareLaunchArgument(
      'depth_image_topic', default_value='/camera/aligned_depth_to_color/image_raw',
      description='Depth image topic.'),

    DeclareLaunchArgument(
      'voxel_size', default_value='0.05',
      description='Voxel size (m) of the downsampled lidar point cloud. For indoor, set it between 0.1 and 0.3. For outdoor, set it to 0.5 or over.'),
    
    DeclareLaunchArgument(
      'min_loop_closure_overlap', default_value='0.1',
      description='Minimum scan overlap pourcentage to accept a loop closure.'),

    DeclareLaunchArgument(
      'qos', default_value='1',
      description='Quality of Service: 0=system default, 1=reliable, 2=best effort.'),

    DeclareLaunchArgument(
      'force3dof', default_value="false",
      description='Force 3 degrees-of-freedom transform (3Dof: x,y and yaw). Parameters z, roll and pitch will be set to 0'),

    OpaqueFunction(function=launch_setup),
  ])

    