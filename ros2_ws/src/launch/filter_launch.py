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
    
    rviz_config = os.path.join(
    get_package_share_directory('turtlebot_state_estimation'),
    'rviz',
    'rviz_config_default.rviz'
    )

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
            'use_rviz': 'True',
            'rviz_config_file': rviz_config,
        }.items()
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

    # -----------------------------
    # Eval Node
    # -----------------------------
    eval_node = Node(
        package='turtlebot_state_estimation',
        executable='eval_node',
        name='eval_node',
        output='screen',
        parameters=[{
            'ground_truth_topic': '/world/depot/dynamic_pose/info',
            'ground_truth_index': 1,
        }]
    )

    return LaunchDescription([
        nav2_launch,
        kf_node,
        ekf_node,
        eval_node,
    ])