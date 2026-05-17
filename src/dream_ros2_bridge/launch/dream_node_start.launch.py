import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


REQUIRED_EXTRINSIC_KEYS = (
    "link6_to_camera",
    "body_to_livox",
    "livox_to_base",
    "base_to_footprint",
)

def _to_tf_args(tf_cfg):
    vals = tf_cfg["translation"] + tf_cfg["rotation_ypr"] + [tf_cfg["parent"], tf_cfg["child"]]
    return [str(v) for v in vals]


def _load_extrinsics_required(extrinsics_file):
    if not os.path.isfile(extrinsics_file):
        raise FileNotFoundError(f"Required extrinsics file not found: {extrinsics_file}")

    with open(extrinsics_file, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    extrinsics = loaded.get("extrinsics", loaded)

    for key in REQUIRED_EXTRINSIC_KEYS:
        if key not in extrinsics:
            raise KeyError(f"Missing extrinsics key: {key}")
        item = extrinsics[key]
        for field in ("translation", "rotation_ypr", "parent", "child"):
            if field not in item:
                raise KeyError(f"Missing field '{field}' in extrinsics['{key}']")

    print(f"Loaded extrinsics from: {extrinsics_file}")
    return extrinsics


def launch_setup(context, *args, **kwargs):
    """Launch setup function to handle dynamic parameters"""
    
    dream_ros2_bridge_path = get_package_share_directory('dream_ros2_bridge')
    
    # Get the use_simple_urdf parameter value
    use_simple_urdf = LaunchConfiguration('use_simple_urdf')
    simple_urdf = use_simple_urdf.perform(context)
    print(f"use_simple_urdf parameter value: {simple_urdf}")
    
    # Get the use_sim_time parameter value for consistent time synchronization
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_time = use_sim_time.perform(context)
    print(f"use_sim_time parameter value: {sim_time}")

    config_path = LaunchConfiguration('config_path')
    extrinsics_file = PathJoinSubstitution([
        config_path,
        LaunchConfiguration('extrinsics_file'),
    ]).perform(context)
    extrinsics = _load_extrinsics_required(extrinsics_file)

    ranger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                dream_ros2_bridge_path,
                'launch',
                "ranger_mini_v3_driver.launch.py"
            )
        ]),
    )

    xarm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                dream_ros2_bridge_path,
                'launch',
                "xarm6_driver.launch.py"
            )
        ]),
        launch_arguments={
            'robot_ip': LaunchConfiguration('xarm_ip'),
            'joint_states_rate': LaunchConfiguration('joint_states_rate'),
            'hw_ns': 'xarm',
            'add_gripper': 'true'
        }.items(),
    )

    camera_launch = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                dream_ros2_bridge_path, 
                '/launch/d435i_high_resolution.launch.py'])),  
    ])

    lidar_launch = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                dream_ros2_bridge_path, 
                '/launch/livox_mid360_driver.launch.py'])),  
    ])

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[
            PathJoinSubstitution([config_path, LaunchConfiguration('fast_lio_config_file')]),
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        namespace='fast_lio2',
        remappings=[
            ('/cloud_registered', '/fast_lio2/cloud_registered'),
            ('/cloud_registered_body', '/fast_lio2/cloud_registered_body'),
            ('/cloud_effected', '/fast_lio2/cloud_effected'),
            ('/Laser_map', '/fast_lio2/Laser_map'),
            ('/Odometry', '/fast_lio2/Odometry'),
            ('/path', '/fast_lio2/path'),
        ]
    )


    if simple_urdf == 'true':
        urdf_file_path = os.path.join(dream_ros2_bridge_path, 'urdf', "rangerminiv3_with_xarm6_simple.urdf")
    else:
        urdf_file_path = os.path.join(dream_ros2_bridge_path, 'urdf', "rangerminiv3_with_xarm6.urdf")
    
    # Read the combined URDF file
    with open(urdf_file_path, 'r') as f:
        robot_description_content = f.read()

    # Robot state publisher for the combined robot
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='combined_robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'publish_frequency': 50.0,
            'use_sim_time': use_sim_time,
        }],
        
    )

    # Custom joint state aggregator that handles time synchronization
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='combined_joint_state_publisher',
        output='screen',
        parameters=[{
            'source_list': ['/ranger/joint_states', '/xarm/joint_states'],
            'use_sim_time': use_sim_time,
            'rate': 30.0,
        }]
    )


    # Static transform from camera to end effector
    static_tf_link6_to_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_link6_to_camera",
        arguments=_to_tf_args(extrinsics["link6_to_camera"]),
        output="screen",
        parameters=[{'use_sim_time': use_sim_time}]
    )


    # Static transform from fast lio2 lidar frame to livox lidar frame
    static_tf_lidar_to_livox = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_lidar_to_livox",
        arguments=_to_tf_args(extrinsics["body_to_livox"]),
        output="screen",
        parameters=[{'use_sim_time': use_sim_time}]
    )


    # Static transform from livox frame to robot base
    static_tf_livox_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_livox_to_base",
        arguments=_to_tf_args(extrinsics["livox_to_base"]),
        output="screen",
        parameters=[{'use_sim_time': use_sim_time}]
    )


    static_tf_base_to_footprint = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_base_to_footprint",
        arguments=_to_tf_args(extrinsics["base_to_footprint"]),
        output="screen",
        parameters=[{'use_sim_time': use_sim_time}]
    )


    tf_pose_publisher_node = Node(
        package='dream_ros2_bridge',
        executable='tf_pose_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    # controller
    goto_controller_node = Node(
        package='dream_ros2_bridge',
        executable='goto_controller',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    # Custom RViz for TF visualization only
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(dream_ros2_bridge_path, 'rviz', 'show_node_start.rviz')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return [
        camera_launch,
        lidar_launch,
        fast_lio_node,
        static_tf_link6_to_camera,
        static_tf_lidar_to_livox,
        static_tf_livox_to_base,
        static_tf_base_to_footprint,
        xarm_launch,
        ranger_launch,
        robot_state_publisher_node,
        joint_state_publisher_node,
        tf_pose_publisher_node,
        goto_controller_node,
        rviz_node,
    ]

def generate_launch_description():
    package_path = get_package_share_directory('dream_ros2_bridge')
    default_config_path = os.path.join(package_path, 'config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_simple_urdf',
            default_value='true',
            description='Whether to use simple URDF (true) or full URDF (false) for both xarm6 and ranger robots.'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulated clock for consistent time synchronization across all nodes.'
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Use rviz2'
        ),
        DeclareLaunchArgument(
            'xarm_ip', default_value='192.168.1.233',
            description='xArm controller IP address.'
        ),
        DeclareLaunchArgument(
            'joint_states_rate', default_value='50',
            description='xArm joint states publish rate (Hz).'
        ),
        DeclareLaunchArgument(
            'config_path', default_value=default_config_path,
            description='Directory path for config files.'
        ),
        DeclareLaunchArgument(
            'extrinsics_file', default_value='extrinsics_cad.yaml',
            description='Extrinsics YAML filename under config_path.'
        ),
        DeclareLaunchArgument(
            'fast_lio_config_file', default_value='fast_lio2_mid360.yaml',
            description='FAST-LIO config filename under config_path.'
        ),
        OpaqueFunction(function=launch_setup),
    ])
