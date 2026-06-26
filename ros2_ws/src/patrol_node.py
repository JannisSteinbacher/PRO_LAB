#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

# Import the Nav2 Simple Commander API
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy,
)

def euler_to_quaternion(roll, pitch, yaw):
    """
    Helper function to convert Euler angles (in radians) to Quaternions.
    Nav2 requires orientations in Quaternion format.
    """
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return qx, qy, qz, qw

def create_waypoint(navigator, x, y, yaw_deg):
    """
    Helper function to generate a PoseStamped message for a waypoint.
    """
    wp = PoseStamped()
    wp.header.frame_id = 'map'
    wp.header.stamp = navigator.get_clock().now().to_msg()
    
    # Set position
    wp.pose.position.x = float(x)
    wp.pose.position.y = float(y)
    wp.pose.position.z = 0.0
    
    # Convert yaw from degrees to radians, then to quaternion
    yaw_rad = math.radians(yaw_deg)
    qx, qy, qz, qw = euler_to_quaternion(0, 0, yaw_rad)
    wp.pose.orientation.x = qx
    wp.pose.orientation.y = qy
    wp.pose.orientation.z = qz
    wp.pose.orientation.w = qw
    
    return wp

def main(args=None):
    rclpy.init(args=args)

    # 1. Initialize the Navigator
    navigator = BasicNavigator()

    # 2. Wait for Nav2 to become active
    print("Waiting for Nav2 to become active...")
    navigator.waitUntilNav2Active()
    print("Nav2 is active! Preparing waypoints.")

    # Latched publisher that tells the eval node when to start recording, so it
    # skips the long Gazebo/Nav2 startup where the robot is parked at spawn.
    # Transient-local keeps the signal available even if eval subscribes late.
    start_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    eval_start_pub = navigator.create_publisher(Empty, '/eval/start', start_qos)

    # 3. Define the Waypoints
    # create_waypoint(navigator, x_meters, y_meters, yaw_degrees)
    wp1 = create_waypoint(navigator, 5.2, 3.58, 0.0)
    wp2 = create_waypoint(navigator, 8.25, 5.166, 180.0)
    wp3 = create_waypoint(navigator, -6.11, 2.625, 180.0)
    wp4 = create_waypoint(navigator, 0.0, 0.0, 0.0)

    # Put them in a list
    route = [wp1, wp2, wp3, wp4]

    # 4. Drive the route once, then exit.
    print("--- Starting patrol route ---")

    # Send the route to Nav2
    navigator.followWaypoints(route)

    # Signal the eval node so tracking starts the moment the robot begins
    # driving (publish once; latched for late joiners).
    eval_start_pub.publish(Empty())
    print("Published /eval/start — evaluation tracking begins.")

    # Wait while the robot is driving
    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            print(f"Executing waypoint {feedback.current_waypoint} / {len(route)}", end="\r")

    # Check the result of the task
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print("\nReached the final waypoint! Patrol complete — shutting down.")
    elif result == TaskResult.CANCELED:
        print("\nTask was canceled. Exiting.")
    elif result == TaskResult.FAILED:
        print("\nTask failed! Exiting.")

    # Clean up. Exiting this process triggers the launch file's OnProcessExit
    # handler, which shuts the whole stack down (and the eval node writes its
    # PNG/CSV outputs on that shutdown).
    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()