cmake_minimum_required(VERSION 3.8)
project(turtlebot_state_estimation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. FIND DEPENDENCIES
# ==========================================
find_package(ament_cmake        REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(nav_msgs           REQUIRED)
find_package(geometry_msgs      REQUIRED)
find_package(sensor_msgs        REQUIRED)
find_package(std_msgs           REQUIRED)
find_package(message_filters    REQUIRED)
find_package(tf2                REQUIRED)
find_package(tf2_ros            REQUIRED)
find_package(tf2_geometry_msgs  REQUIRED)
# ==========================================
# 2. CREATE EXECUTABLES (YOUR NODES)
# ==========================================
# Add the Kalman Filter Node
add_executable(KF_node src/KF_node.cpp)
ament_target_dependencies(KF_node
  rclcpp
  std_msgs 
  tf2_geometry_msgs 
  geometry_msgs 
  nav_msgs sensor_msgs 
  tf2_ros 
  message_filters)

# Add the Extended Kalman Filter Node
add_executable(EKF_node src/EKF_node.cpp)
ament_target_dependencies(EKF_node 
  rclcpp
  std_msgs 
  tf2_geometry_msgs 
  geometry_msgs 
  nav_msgs sensor_msgs 
  tf2_ros 
  message_filters)

# Add the Particle Filter Node (Your specific focus area)
add_executable(PF_node src/PF_node.cpp)
ament_target_dependencies(PF_node 
  rclcpp 
  std_msgs 
  geometry_msgs 
  nav_msgs 
  sensor_msgs 
  tf2_ros)

# ==========================================
# 3. INSTALL RULES
# ==========================================
install(TARGETS
  KF_node
  EKF_node
  PF_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}
)

# ==========================================
# 4. AMENT PACKAGE BUILD
# ==========================================
ament_package()
