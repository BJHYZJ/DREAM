import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    dream_ros2_bridge_path = get_package_share_directory('dream_ros2_bridge')

    rviz_param = DeclareLaunchArgument('use_rviz', default_value='true', choices=['true', 'false'])
    rviz_config = DeclareLaunchArgument('rviz_config',
                                        default_value=dream_ros2_bridge_path + '/' + 'rviz/rtabmap_nav.rviz')
     
    # node_start_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(dream_ros2_bridge_path, 'launch', 'yanzj_rtabmap_node_start.launch.py')
    #     ))
    
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dream_ros2_bridge_path, 'launch', 'yanzj_rtabmap_slam.launch.py')
        ))

    params_file_param = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(dream_ros2_bridge_path, 'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')
    
    nav2_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(dream_ros2_bridge_path, 'launch', 'navigation.launch.py')),
            launch_arguments={'params_file': LaunchConfiguration('params_file')}.items())
    
    
    rviz_launch = Node(package='rviz2', executable='rviz2',
        output='log',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        respawn=True,
        arguments=['-d', LaunchConfiguration('rviz_config')],
        )

    return LaunchDescription([
        # teleop_type_param,
        rviz_param,
        rviz_config,
        # node_start_launch,
        slam_launch,
        params_file_param,
        nav2_launch,
        rviz_launch,
    ])
