from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_rotation_node = Node(
        package="dream_ros2_bridge",
        executable="rotate_images",
        name="rotate_images_from_stretch_head",
    )

    return LaunchDescription([image_rotation_node])
