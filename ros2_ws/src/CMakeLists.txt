cmake_minimum_required(VERSION 3.8)
project(turtlebot_state_estimation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. FIND DEPENDENCIES
# ==========================================
# Tell CMake which ROS 2 packages you need to compile your code.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# ==========================================
# 2. CREATE EXECUTABLES (YOUR NODES)
# ==========================================
# Add the Kalman Filter Node
add_executable(KF_node src/KF_node.cpp)
ament_target_dependencies(KF_node rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros)

# Add the Extended Kalman Filter Node
add_executable(EKF_node src/EKF_node.cpp)
ament_target_dependencies(EKF_node rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros)

# Add the Particle Filter Node (Your specific focus area)
add_executable(PF_node src/PF_node.cpp)
ament_target_dependencies(PF_node rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros)

# ==========================================
# 3. INSTALL RULES
# ==========================================
# Tell ROS 2 where to install the compiled nodes so `ros2 run` can find them.
install(TARGETS
  KF_node
  EKF_node
  PF_node
  DESTINATION lib/${PROJECT_NAME}
)

# Tell ROS 2 where to install your launch files, RViz configs, and parameter YAMLs 
# so `ros2 launch` can find them.
install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# ==========================================
# 4. AMENT PACKAGE BUILD
# ==========================================
# Required by ROS 2 to finalize the build process
ament_package()