#!/usr/bin/env python3
# Integrated ranger driver launch script
# Contains all necessary components for joint states and TF transforms without rviz

import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def launch_setup(context, *args, **kwargs):
    """Launch setup function to handle dynamic parameters"""
    
    # Launch Arguments
    port_name = LaunchConfiguration("port_name")
    robot_model = LaunchConfiguration("robot_model")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    update_rate = LaunchConfiguration("update_rate")
    odom_topic_name = LaunchConfiguration("odom_topic_name")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf")
    use_rviz = LaunchConfiguration("use_rviz")

    # Get the actual values of parameters
    port_name_value = port_name.perform(context)
    robot_model_value = robot_model.perform(context)
    odom_frame_value = odom_frame.perform(context)
    base_frame_value = base_frame.perform(context)
    update_rate_value = update_rate.perform(context)
    odom_topic_name_value = odom_topic_name.perform(context)
    publish_odom_tf_value = publish_odom_tf.perform(context)
    use_rviz_value = use_rviz.perform(context)

    print(f"ranger_driver parameters:")
    print(f"  port_name: {port_name_value}")
    print(f"  robot_model: {robot_model_value}")
    print(f"  odom_frame: {odom_frame_value}")
    print(f"  base_frame: {base_frame_value}")
    print(f"  update_rate: {update_rate_value}")
    print(f"  odom_topic_name: {odom_topic_name_value}")
    print(f"  publish_odom_tf: {publish_odom_tf_value}")
    print(f"  use_rviz: {use_rviz_value}")

    # Ranger base node
    ranger_base_node = Node(
        package="ranger_base",
        executable="ranger_base_node",
        name="ranger_base_node",
        output="screen",
        parameters=[{
            "port_name": port_name_value,
            "robot_model": robot_model_value,
            "odom_frame": odom_frame_value,
            "base_frame": base_frame_value,
            "update_rate": int(update_rate_value),
            "odom_topic_name": odom_topic_name_value,
            "publish_odom_tf": publish_odom_tf_value == 'true',
        }],
        namespace='ranger',
        remappings=[
            ('/system_state', '/ranger/system_state'),
            ('/motion_state', '/ranger/motion_state'),
            ('/actuator_state', '/ranger/actuator_state'),
            ('/odom', '/ranger/odom'),
            ('/battery_state', '/ranger/battery_state'),
            ('/rc_state', '/ranger/rc_state'),
            ('/cmd_vel', '/ranger/cmd_vel'),
        ]
    )


    # Custom joint state publisher for ranger
    joint_state_publisher_node = Node(
        package='ranger_base',
        executable='ranger_joint_state_publisher',
        name='ranger_joint_state_publisher',
        output='screen',
    )


    nodes_to_return = [
        ranger_base_node, 
        joint_state_publisher_node
    ]
    
    
    return nodes_to_return


def generate_launch_description():
    return LaunchDescription([
        # CAN port name
        DeclareLaunchArgument(
            "port_name", 
            default_value="can0",
            description="CAN port name for ranger robot communication."
        ),
        # Robot model
        DeclareLaunchArgument(
            "robot_model", 
            default_value="ranger",
            description="Robot model name."
        ),
        # Odometry frame
        DeclareLaunchArgument(
            "odom_frame", 
            default_value="odom",
            description="Odometry frame name."
        ),
        # Base frame
        DeclareLaunchArgument(
            "base_frame", 
            default_value="base_link",
            description="Base frame name."
        ),
        # Update rate
        DeclareLaunchArgument(
            "update_rate", 
            default_value="50",
            description="Update rate for odometry publishing (Hz)."
        ),
        # Odometry topic name
        DeclareLaunchArgument(
            "odom_topic_name", 
            default_value="/odom",
            description="Odometry topic name."
        ),
        # Publish odometry TF
        DeclareLaunchArgument(
            "publish_odom_tf", 
            default_value="false",
            description="Whether to publish odometry TF transform."
        ),
        # Use RViz2
        DeclareLaunchArgument(
            "use_rviz", 
            default_value="false",
            description="Whether to launch RViz2 for visualization."
        ),
        
        OpaqueFunction(function=launch_setup),
    ])














# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, OpaqueFunction
# from launch.conditions import IfCondition
# from launch.substitutions import LaunchConfiguration, Command
# from launch_ros.actions import Node
# from ament_index_python.packages import get_package_share_path

# def launch_setup(context, *args, **kwargs):
#     """Launch setup function to handle dynamic parameters"""
    
#     # Get parameter values
#     port_name = LaunchConfiguration("port_name").perform(context)
#     robot_model = LaunchConfiguration("robot_model").perform(context)
#     odom_frame = LaunchConfiguration("odom_frame").perform(context)
#     base_frame = LaunchConfiguration("base_frame").perform(context)
#     update_rate = LaunchConfiguration("update_rate").perform(context)
#     odom_topic_name = LaunchConfiguration("odom_topic_name").perform(context)
#     publish_odom_tf = LaunchConfiguration("publish_odom_tf").perform(context)
    
#     print(f"Ranger parameters:")
#     print(f"  port_name: {port_name}")
#     print(f"  robot_model: {robot_model}")
#     print(f"  odom_frame: {odom_frame}")
#     print(f"  base_frame: {base_frame}")
#     print(f"  update_rate: {update_rate}")
#     print(f"  odom_topic_name: {odom_topic_name}")
#     print(f"  publish_odom_tf: {publish_odom_tf}")

#     # Ranger base node
#     ranger_base_node = Node(
#         package="ranger_base",
#         executable="ranger_base_node",
#         name="ranger_base_node",
#         output="screen",
#         parameters=[{
#             "port_name": port_name,
#             "robot_model": robot_model,
#             "odom_frame": odom_frame,
#             "base_frame": base_frame,
#             "update_rate": int(update_rate),
#             "odom_topic_name": odom_topic_name,
#             "publish_odom_tf": publish_odom_tf == 'true',
#         }],
#     )

#     joint_state_publisher_node = Node(
#         package='ranger_base',
#         executable='ranger_joint_state_publisher',
#         name='ranger_joint_state_publisher',
#         output='screen',
#     )

#     return [
#         ranger_base_node, 
#         joint_state_publisher_node
#     ]

# def generate_launch_description():
#     declare_args = [
#         DeclareLaunchArgument("port_name", default_value="can0"),
#         DeclareLaunchArgument("robot_model", default_value="ranger"),
#         DeclareLaunchArgument("odom_frame", default_value="odom"),
#         DeclareLaunchArgument("base_frame", default_value="base_link"),
#         DeclareLaunchArgument("update_rate", default_value="50"),
#         DeclareLaunchArgument("odom_topic_name", default_value="ranger/odom"),
#         DeclareLaunchArgument("publish_odom_tf", default_value="false"),
#     ]

#     return LaunchDescription(declare_args + [
#         OpaqueFunction(function=launch_setup),
#     ])