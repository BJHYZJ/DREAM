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
  
  namespace = LaunchConfiguration('namespace')
  
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
  slam_profile = LaunchConfiguration('slam_profile').perform(context)
  
  # Rule of thumb:
  # max_correspondence_distance = voxel_size_value * 20.0  # default * 10

  # Conservative defaults for mobile-manipulator setup with external odometry.
  icp_max_translation = '0.6'
  icp_max_correspondence_distance = '0.2'
  rtabmap_detection_rate = '3.0'
  rgbd_linear_update = '0.05'
  rgbd_angular_update = '0.05'
  min_loop_overlap = str(LaunchConfiguration('min_loop_closure_overlap').perform(context))

  if slam_profile == 'fast':
    icp_max_translation = '1.0'
    icp_max_correspondence_distance = '0.3'
    rtabmap_detection_rate = '5.0'
    rgbd_linear_update = '0.1'
    rgbd_angular_update = '0.1'
  elif slam_profile == 'robust':
    icp_max_translation = '0.4'
    icp_max_correspondence_distance = '0.15'
    rtabmap_detection_rate = '2.0'
    rgbd_linear_update = '0.03'
    rgbd_angular_update = '0.03'
    min_loop_overlap = '0.2'

  shared_parameters = {
    'use_sim_time': use_sim_time,
    'frame_id': frame_id,
    'qos': LaunchConfiguration('qos'),
    'wait_for_transform': 0.5,
    # Synchronization parameters for RGB-D and LiDAR.
    'approx_sync': True,  
    'approx_sync_max_interval': '0.1',
    # 'queue_size': 20,
    'topic_queue_size': 20,
    'sync_queue_size': 20,
    # RTAB-Map's internal parameters are strings:
    'Icp/PointToPlane': 'true',
    'Icp/Iterations': '30',
    'Icp/VoxelSize': str(voxel_size_value),
    'Icp/Epsilon': '0.001',
    'Icp/PointToPlaneK': '5',
    'Icp/PointToPlaneRadius': '0',
    'Icp/MaxTranslation': icp_max_translation,
    # 'Icp/MaxCorrespondenceDistance': str(max_correspondence_distance),
    'Icp/MaxCorrespondenceDistance': icp_max_correspondence_distance,
    # 'Icp/MaxCorrespondenceDistance': '1',
    'Icp/Strategy': '1',
    'Icp/OutlierRatio': '0.85',
  }


  rtabmap_parameters = {
    'subscribe_depth': False,
    'subscribe_rgb': False,
    'subscribe_rgbd': True,
    'subscribe_odom_info': False,  # Enable only when using RTAB-Map odometry. FAST-LIO2 odometry is used here.
    'subscribe_scan_cloud': True,
    # With external odometry (FAST-LIO), timestamp offsets are often better handled
    # by upstream drivers/sync. Enabling this may introduce pose jitter.
    'odom_sensor_sync': False,
    'map_frame_id': 'map',
    'odom_frame_id': 'camera_init',  # FAST-LIO's odometry frame
    
    'Rtabmap/DetectionRate': rtabmap_detection_rate,

    # RTAB-Map's internal parameters are strings:
    'RGBD/ProximityMaxGraphDepth': '0',
    'RGBD/ProximityPathMaxNeighbors': '0',
    'RGBD/AngularUpdate': rgbd_angular_update,
    'RGBD/LinearUpdate': rgbd_linear_update,
    'RGBD/CreateOccupancyGrid': 'false',
    'RGBD/ForceOdom3DoF': force_3dof,       # Default: true - Force odometry pose to be 3DoF if Reg/Force3DoF=true.

    'Grid/3D': 'false',   # Explicitly disable the 3D grid and set Grid/Sensor to 0 to avoid RTAB-Map occupancy grid warnings.
    'Grid/Sensor': "0",  # Create occupancy grid from selected sensor: 0=laser scan, 1=depth image(s) or 2=both laser scan and depth image(s).

    'Mem/NotLinkedNodesKept': 'false',
    'Mem/STMSize': '60',
    'Mem/DepthCompressionFormat': '.png',  # Use PNG instead of RVL for depth image compression.

    'Reg/Force3DoF': force_3dof,
    'Reg/RepeatOnce': 'true',
    'Reg/Strategy': '2',  # 0=Vis, 1=Icp, 2=VisIcp
    'Icp/CorrespondenceRatio': min_loop_overlap
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
          namespace=namespace,
          parameters=[{'approx_sync': True, 'approx_sync_max_interval': 0.03, 'use_sim_time': use_sim_time}],
          remappings=remappings),

      Node(
        package='rtabmap_slam', executable='rtabmap', output='screen',
        namespace=namespace,
        parameters=[shared_parameters, rtabmap_parameters, 
                    {'rgbd_cameras': 1}],
        remappings=remappings + [('scan_cloud', lidar_topic)],
        arguments=arguments),
      
      Node(
        condition=IfCondition(LaunchConfiguration('use_rtabmap_viz')),
        package='rtabmap_viz', executable='rtabmap_viz', output='screen',
        namespace=namespace,
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
        namespace=namespace,
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

    # The bridge calls RTAB-Map's frame_id the "tracking" frame.
    # Pose names such as tracking_in_map_pose and camera_in_tracking_pose come from this.
    DeclareLaunchArgument(
      'frame_id', default_value='body',
      description='Tracking frame used by RTAB-Map. Recommended: body (FAST-LIO body frame). '
                  'Bridge pose names like tracking_in_map_pose, camera_in_tracking_pose, '
                  'and base_in_tracking_pose use this frame as "tracking". '
                  'TF chain is typically map->camera_init->body->livox_frame->base_link.'),
    
    DeclareLaunchArgument(
      'localization', default_value='false',
      description='Localization mode.'),

    DeclareLaunchArgument(
      'lidar_topic', default_value='/fast_lio2/cloud_registered_body',  # /fast_lio2/cloud_registered_body is in the lidar_frame coordinate frame.
      description='FAST-LIO2 registered point cloud topic.'),

    DeclareLaunchArgument(
      'odom_topic', default_value='/fast_lio2/Odometry',
      description='Odometry topic from SLAM system (e.g., FAST-LIO2).'),

    DeclareLaunchArgument(
      # Keep empty by default when using external odometry (FAST-LIO) to avoid
      # introducing extra orientation constraints from another IMU stream.
      'imu_topic', default_value='',
      description='IMU topic (ignored if empty).'),
    
    DeclareLaunchArgument(
      'namespace', default_value='dream_rtabmap',
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
      'voxel_size', default_value='0.1',
      description='Voxel size (m) of the downsampled lidar point cloud. For indoor, set it between 0.1 and 0.3. For outdoor, set it to 0.5 or over.'),
    
    DeclareLaunchArgument(
      'min_loop_closure_overlap', default_value='0.1',
      description='Minimum scan overlap pourcentage to accept a loop closure.'),

    DeclareLaunchArgument(
      'slam_profile', default_value='balanced',
      description='Preset for SLAM tuning: balanced (default), fast (higher update rate), robust (more conservative and stable).'),

    DeclareLaunchArgument(
      'qos', default_value='1',
      description='Quality of Service: 0=system default, 1=reliable, 2=best effort.'),

    DeclareLaunchArgument(
      'force3dof', default_value="true",
      description='Force 3 degrees-of-freedom transform (3Dof: x,y and yaw). Parameters z, roll and pitch will be set to 0'),

    OpaqueFunction(function=launch_setup),
  ])
