from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # -----------------------------
    # Nav2 TB4 Simulation Launch
    # -----------------------------
    nav2_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('nav2_bringup'),
            'launch',
            'tb4_simulation_launch.py'
        )
    ),
    launch_arguments={
        'headless': 'False',
        'rviz': 'True'
    }.items()
    )

    rviz_config = os.path.join(
    get_package_share_directory('turtlebot_state_estimation'),
    'rviz',
    'rviz_config_default.rviz'
    )

    Node(
    package='rviz2',
    executable='rviz2',
    arguments=['-d', rviz_config],
    output='screen'
    )

    # -----------------------------
    # KF Node
    # -----------------------------
    kf_node = Node(
        package='turtlebot_state_estimation',
        executable='KF_node',
        name='kf_node',
        output='screen'
    )

    # -----------------------------
    # EKF Node
    # -----------------------------
    ekf_node = Node(
        package='turtlebot_state_estimation',
        executable='EKF_node',
        name='ekf_node',
        output='screen'
    )

    return LaunchDescription([
        nav2_launch,
        kf_node,
        ekf_node
    ])