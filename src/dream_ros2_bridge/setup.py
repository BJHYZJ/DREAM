import os
from glob import glob

from setuptools import find_packages, setup

package_name = "dream_ros2_bridge"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*")),
        ),
        (
            os.path.join("share", package_name, "urdf"),
            glob(os.path.join("urdf", "*.urdf")),
        ),
        (
            os.path.join("share", package_name, "meshes"),
            glob(os.path.join("meshes", "*.*")),
        ),
    ],
    install_requires=["setuptools", "dream"],
    zip_safe=True,
    maintainer="Zhijie Yan",
    maintainer_email="yanzhijie@buaa.edu.cn",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_pose_publisher = dream_ros2_bridge.nodes.camera_pose_publisher:main",
            "goto_controller = dream_ros2_bridge.nodes.goto_controller:main",
            "odom_tf_publisher = dream_ros2_bridge.nodes.odom_tf_publisher:main",
            "tf_pose_publisher = dream_ros2_bridge.nodes.tf_pose_publisher:main",
            'd435i_accel_correction = dream_ros2_bridge.nodes.d435i_accel_correction:main',
            "server = dream_ros2_bridge.remote.server:main",
        ],
    },
)
