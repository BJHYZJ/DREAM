import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def launch_setup(context, *args, **kwargs):
    """Launch setup function to handle dynamic parameters"""
    
    # Get the use_sim_time parameter value for consistent time synchronization
    use_sim_time = LaunchConfiguration('use_sim_time')
    print(f"use_sim_time parameter value: {use_sim_time.perform(context)}")

    start_server = Node(
        package="dream_ros2_bridge",
        executable="server_dream",
        namespace="ros2_zmq_server",
        output="screen",
        on_exit=launch.actions.Shutdown(),
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    return [
        start_server,
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulated clock for consistent time synchronization across all nodes.'
        ),
        OpaqueFunction(function=launch_setup),
    ])