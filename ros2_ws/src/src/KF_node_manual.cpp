#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <cmath>
#include <cstdint>
#include <memory>


double x = 0; 
double y = 0;
double theta = 0;

double robot_state[3] = {x,y,theta};