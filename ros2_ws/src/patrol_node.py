#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

# Import the Nav2 Simple Commander API
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped

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

    # 3. Define the Waypoints (Adjust these to fit your example map!)
    # create_waypoint(navigator, x_meters, y_meters, yaw_degrees)
    wp1 = create_waypoint(navigator, 5.2, 3.58, 0.0)
    wp2 = create_waypoint(navigator, 8.25, 5.166, 180.0)
    wp3 = create_waypoint(navigator, -6.11, 2.625, 180.0)
    wp4 = create_waypoint(navigator, 1.26, 0.78, 0.0)

    # Put them in a list
    route = [wp1, wp2, wp3, wp4]

    # 4. Infinite Loop for Patrolling
    loop_count = 1
    while rclpy.ok():
        print(f"--- Starting Patrol Loop #{loop_count} ---")
        
        # Send the route to Nav2
        navigator.followWaypoints(route)

        # Wait while the robot is driving
        while not navigator.isTaskComplete():
            # You can do other things here, like print feedback or check sensors
            feedback = navigator.getFeedback()
            if feedback:
                print(f"Executing waypoint {feedback.current_waypoint} / {len(route)}", end="\r")

        # Check the result of the task
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print("\nLoop completed successfully! Restarting...")
        elif result == TaskResult.CANCELED:
            print("\nTask was canceled. Exiting.")
            break
        elif result == TaskResult.FAILED:
            print("\nTask failed! Exiting.")
            break
            
        loop_count += 1

    # Clean up
    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()