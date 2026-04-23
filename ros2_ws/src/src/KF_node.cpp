//Node for Kalman Filter implementation

//Subscribes to:
//  /odom, /cmd_vel /imu, /scan (optional)

//Publishes:
//  /kf/estimated_pose (type geometry_msgs/msg/PoseWithCovarianceStamped
//  or nav_msgs/msg/Odometry)

#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("my_node");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
