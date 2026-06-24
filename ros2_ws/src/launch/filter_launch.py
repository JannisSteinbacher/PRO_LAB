from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # -----------------------------
    # Launch arguments
    # -----------------------------
    # Set eval:=false to skip writing the evaluation PNG on shutdown
    # (e.g. `ros2 launch ... filter_launch.py eval:=false` for quick testing).
    # The eval node still runs and collects data; only the PNG is suppressed.
    eval_arg = DeclareLaunchArgument(
        'eval',
        default_value='true',
        description='Generate the evaluation PNG on shutdown.'
    )

    # -----------------------------
    # Nav2 TB4 Simulation Launch
    # -----------------------------
    
    rviz_config = os.path.join(
    get_package_share_directory('turtlebot_state_estimation'),
    'rviz',
    'rviz_config_default.rviz'
    )

    # Central noise tuning (R / Q) shared by both KF and EKF
    filter_params = os.path.join(
    get_package_share_directory('turtlebot_state_estimation'),
    'config',
    'filter_params.yaml'
    )

    pkg_share = get_package_share_directory('turtlebot_state_estimation')

    # Our customized depot world with AprilTag landmarks
    world = os.path.join(pkg_share, 'worlds', 'depot_apriltags.sdf')

    # Let gz sim resolve the model:// AprilTag textures.
    # nav2 also uses AppendEnvironmentVariable, so this stacks alongside it.
    set_models_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_share, 'models')
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
            'world': world,
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
        output='screen',
        parameters=[filter_params]
    )

    # -----------------------------
    # EKF Node
    # -----------------------------
    ekf_node = Node(
        package='turtlebot_state_estimation',
        executable='EKF_node',
        name='ekf_node',
        output='screen',
        parameters=[filter_params]
    )

    # -----------------------------
    # PF Node
    # -----------------------------
    # NOTE: PF_node does not yet declare the central R/Q parameters,
    # so filter_params is intentionally not passed here.
    pf_node = Node(
        package='turtlebot_state_estimation',
        executable='PF_node',
        name='pf_node',
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
            'save_plots': ParameterValue(
                LaunchConfiguration('eval'), value_type=bool),
        }]
    )

    return LaunchDescription([
        eval_arg,
        set_models_path,
        nav2_launch,
        kf_node,
        ekf_node,
        pf_node,
        eval_node,
    ])