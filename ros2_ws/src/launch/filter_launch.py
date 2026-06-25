from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    RegisterEventHandler,
    EmitEvent,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
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
    # Set eval:=false to skip writing the evaluation outputs (PNG + CSVs) on
    # shutdown (e.g. `ros2 launch ... filter_launch.py eval:=false` for quick
    # testing). The eval node still runs and collects data; only the files are
    # suppressed.
    eval_arg = DeclareLaunchArgument(
        'eval',
        default_value='true',
        description='Write the evaluation PNG and CSV files on shutdown.'
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
    # PF reads its own 'pf_node:' block from filter_params.yaml (num_particles
    # and the resampling/tempering knobs). The shared '/**' R/Q values are also
    # passed but harmlessly ignored — the PF does not declare them.
    pf_node = Node(
        package='turtlebot_state_estimation',
        executable='PF_node',
        name='pf_node',
        output='screen',
        parameters=[filter_params]
    )

    # -----------------------------
    # AprilTag Detector Node
    # -----------------------------
    # Detects tag36h11 landmarks in the TurtleBot4 camera feed and publishes
    # their position (robot frame) on /apriltag/detections for the EKF to use.
    apriltag_node = Node(
        package='turtlebot_state_estimation',
        executable='apriltag_detector_node',
        name='apriltag_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'image_topic': '/rgbd_camera/image',
            'camera_info_topic': '/rgbd_camera/camera_info',
            'detections_topic': '/apriltag/detections',
            'base_frame': 'base_link',
            'tag_size': 0.16,
        }]
    )

    # -----------------------------
    # Ground-truth pose bridge
    # -----------------------------
    # Gazebo only publishes entity poses on the gz-transport side
    # (gz.msgs.Pose_V on /world/depot/dynamic_pose/info). This parameter_bridge
    # relays it into ROS 2 as a geometry_msgs/PoseArray so the eval node can
    # read ground truth — replacing the manual `ros2 run ros_gz_bridge
    # parameter_bridge ...` we used to start in a second terminal.
    #   '[' = gz -> ROS only (one-way). NOTE: the bridge drops each entity's
    #   name (PoseArray has no names; the TFMessage variant leaves child_frame_id
    #   empty for this topic), so the eval node picks the robot out of the array
    #   by matching its spawn position — see ground_truth_spawn_xy below.
    gt_topic = '/world/depot/dynamic_pose/info'
    gt_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gt_pose_bridge',
        output='screen',
        arguments=[
            f'{gt_topic}@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V',
        ],
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
            'ground_truth_topic': gt_topic,
            # Robot spawn in the world frame; must match map_origin[:2] in
            # filter_params.yaml. The eval node locks onto the array entry
            # nearest this point, so it stays correct if the entity order shifts.
            'ground_truth_spawn_xy': [-8.0, 0.0],
            'save_plots': ParameterValue(
                LaunchConfiguration('eval'), value_type=bool),
            'save_csv': ParameterValue(
                LaunchConfiguration('eval'), value_type=bool),
        }]
    )

    # -----------------------------
    # Patrol Node
    # -----------------------------
    # Drives the TurtleBot through the 4 waypoints once via Nav2 (replaces the
    # manual `./patrol_node.py` we used to start in a second terminal). It blocks
    # on waitUntilNav2Active(), so it's safe to launch alongside the rest of the
    # stack — it just waits for Nav2 to come up before sending goals.
    patrol_node = Node(
        package='turtlebot_state_estimation',
        executable='patrol_node',
        output='screen',
    )

    # When the patrol reaches the 4th/last waypoint it exits. That triggers this
    # handler, which shuts the whole launch down — the eval node writes its
    # PNG/CSV on shutdown, and `ros2 launch` returns to the prompt.
    shutdown_on_patrol_done = RegisterEventHandler(
        OnProcessExit(
            target_action=patrol_node,
            on_exit=[EmitEvent(event=Shutdown(reason='Patrol route complete'))],
        )
    )

    return LaunchDescription([
        eval_arg,
        set_models_path,
        nav2_launch,
        kf_node,
        ekf_node,
        pf_node,
        apriltag_node,
        gt_pose_bridge,
        eval_node,
        patrol_node,
        shutdown_on_patrol_done,
    ])