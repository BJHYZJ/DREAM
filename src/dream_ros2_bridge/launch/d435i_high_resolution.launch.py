import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Starting with `realsense_ros` 4.55.1, the `.profile`` parameter is split by stream type.
# Here, we keep both the old and new parameters for backwards compatibility.
# https://github.com/IntelRealSense/realsense-ros/pull/3052
configurable_parameters = [{'name': 'depth_module.profile',         'default': '1280x720x15', 'description': 'depth module profile'},
                           {'name': 'depth_module.depth_profile',   'default': '1280x720x15', 'description': 'depth module profile'},
                           {'name': 'depth_module.infra_profile',   'default': '1280x720x15', 'description': 'depth module profile'},
                           {'name': 'rgb_camera.profile',           'default': '1280x720x15', 'description': 'color image width'},
                           {'name': 'rgb_camera.color_profile',     'default': '1280x720x15', 'description': 'color image width'},
                           {'name': 'align_depth.enable',           'default': 'true',        'description': 'whether to publish aligned_depth_to_color feed'},
                           {'name': 'device_type',                  'default': 'd435', 'description': "''"}
                           ]
                           
def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def generate_launch_description():
     d435i_basic_launch = IncludeLaunchDescription(
          PythonLaunchDescriptionSource([os.path.join(
               get_package_share_directory('dream_ros2_bridge'), 'launch'),
               '/d435i_basic.launch.py'])
          )

     logger = LogInfo(msg='D435i launched in high resolution')

     return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
          d435i_basic_launch,
          logger,
          ])
