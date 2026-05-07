#include <cmath>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>

// ============================================================
//  KalmanFilterNode
//
//  State vector:   x = [x,  y,  theta]^T   (3x1)
//
//  Motion model (differential-drive, linearised each step):
//    x'     = x + v*cos(theta)*dt
//    y'     = y + v*sin(theta)*dt
//    theta' = theta + omega*dt
//
//  Jacobian F (used for covariance propagation):
//    F = I + [[ 0,  0, -v*sin(theta)*dt ],
//             [ 0,  0,  v*cos(theta)*dt ],
//             [ 0,  0,  0              ]]
//
//  Measurements
//    * Odometry  -> z = [x, y, theta]   H = I3
//    * IMU       -> z = [theta]         H = [0, 0, 1]
//    * LaserScan -> landmark stub (see scanCallback)
// ============================================================

class KalmanFilterNode : public rclcpp::Node
{
public:
  KalmanFilterNode() : Node("kalman_filter_node")
  {
    // ----------------------------------------------------------
    // Initial state  x = [0, 0, 0]
    // ----------------------------------------------------------
    x_ = Eigen::Vector3d::Zero();

    // ----------------------------------------------------------
    // Initial state covariance P  (high uncertainty at start)
    // ----------------------------------------------------------
    P_ = Eigen::Matrix3d::Identity() * 1.0;

    // ----------------------------------------------------------
    // Process noise Q
    // Tune to reflect how well cmd_vel represents true motion.
    // Larger values -> trust the motion model less.
    // ----------------------------------------------------------
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = 0.05;   // position x  [m^2]
    Q_(1, 1) = 0.05;   // position y  [m^2]
    Q_(2, 2) = 0.01;   // heading  theta  [rad^2]

    // ----------------------------------------------------------
    // Measurement noise R  (per sensor)
    // Larger values -> trust that sensor less.
    // ----------------------------------------------------------
    R_odom_ = Eigen::Matrix3d::Zero();
    R_odom_(0, 0) = 0.1;   // odom x      [m^2]
    R_odom_(1, 1) = 0.1;   // odom y      [m^2]
    R_odom_(2, 2) = 0.05;  // odom theta  [rad^2]

    R_imu_(0, 0) = 0.02;   // IMU theta   [rad^2]

    // ----------------------------------------------------------
    // Subscribers
    // ----------------------------------------------------------
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&KalmanFilterNode::odomCallback, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", 10,
      std::bind(&KalmanFilterNode::imuCallback, this, std::placeholders::_1));

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&KalmanFilterNode::scanCallback, this, std::placeholders::_1));

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&KalmanFilterNode::cmdVelCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Publisher
    // ----------------------------------------------------------
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/kf/pose_estimate", 10);

    // ----------------------------------------------------------
    // Timer – publish at 10 Hz
    // ----------------------------------------------------------
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&KalmanFilterNode::publishEstimate, this));

    last_time_ = this->get_clock()->now();

    RCLCPP_INFO(get_logger(), "Kalman Filter node started.");
  }

private:
  // ============================================================
  //  KF Core
  // ============================================================

  // ------------------------------------------------------------
  //  Predict step
  //  Propagates state and covariance forward by dt seconds
  //  using the latest cmd_vel (v_, omega_).
  // ------------------------------------------------------------
  void predict(double dt)
  {
    const double theta = x_(2);

    // --- Nonlinear state propagation (differential drive) ---
    x_(0) += v_ * std::cos(theta) * dt;
    x_(1) += v_ * std::sin(theta) * dt;
    x_(2) += omega_ * dt;
    x_(2)  = normalizeAngle(x_(2));

    // --- Linearised state-transition Jacobian F ---
    //
    //       d(x') / d(x, y, theta)
    //
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -v_ * std::sin(theta) * dt;
    F(1, 2) =  v_ * std::cos(theta) * dt;

    // --- Covariance prediction ---
    //   P = F * P * F^T + Q
    P_ = F * P_ * F.transpose() + Q_;
  }

  // ------------------------------------------------------------
  //  Correct step  (templated on measurement dimension M)
  //
  //    z  – measurement vector              (Mx1)
  //    H  – measurement matrix              (Mx3)
  //    R  – measurement noise covariance    (MxM)
  // ------------------------------------------------------------
  template<int M>
  void correct(const Eigen::Matrix<double, M, 1>& z,
               const Eigen::Matrix<double, M, 3>& H,
               const Eigen::Matrix<double, M, M>& R)
  {
    // Innovation:  y = z - H * x
    Eigen::Matrix<double, M, 1> y = z - H * x_;

    // Normalise angle channel(s) to [-pi, pi]
    for (int i = 0; i < M; ++i) {
      if (std::abs(y(i)) > M_PI && std::abs(z(i)) < M_PI + 0.5) {
        y(i) = normalizeAngle(y(i));
      }
    }

    // Innovation covariance:  S = H * P * H^T + R
    const Eigen::Matrix<double, M, M> S = H * P_ * H.transpose() + R;

    // Kalman gain:  K = P * H^T * S^{-1}
    const Eigen::Matrix<double, 3, M> K = P_ * H.transpose() * S.inverse();

    // State update:  x = x + K * y
    x_ = x_ + K * y;
    x_(2) = normalizeAngle(x_(2));

    // Covariance update (Joseph / symmetric form for numerical stability):
    //   P = (I - K*H) * P * (I - K*H)^T  +  K * R * K^T
    const Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
  }

  // ============================================================
  //  Callbacks
  // ============================================================

  // ------------------------------------------------------------
  //  cmd_vel  ->  Predict step
  //  The control input (v, omega) arrives here and triggers the
  //  prediction step with the elapsed time since the last call.
  // ------------------------------------------------------------
  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    v_     = msg->linear.x;
    omega_ = msg->angular.z;

    if (!initialized_) return;

    const auto now = this->get_clock()->now();
    const double dt = (now - last_time_).seconds();
    last_time_ = now;

    if (dt > 0.0 && dt < 1.0) {   // guard against large/zero dt
      predict(dt);
    }
  }

  // ------------------------------------------------------------
  //  Odometry  ->  Correct with z = [x, y, theta]
  //  Also initialises the filter state on the very first message.
  // ------------------------------------------------------------
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double yaw = quaternionToYaw(msg->pose.pose.orientation);

    if (!initialized_) {
      x_(0)        = msg->pose.pose.position.x;
      x_(1)        = msg->pose.pose.position.y;
      x_(2)        = yaw;
      last_time_   = this->get_clock()->now();
      initialized_ = true;
      RCLCPP_INFO(get_logger(),
                  "KF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  x_(0), x_(1), x_(2));
      return;
    }

    // Measurement:  z = [x, y, theta]
    const Eigen::Vector3d z(msg->pose.pose.position.x,
                            msg->pose.pose.position.y,
                            yaw);

    // H maps state [x, y, theta] directly to measurement -> identity
    const Eigen::Matrix3d H = Eigen::Matrix3d::Identity();

    correct<3>(z, H, R_odom_);
  }

  // ------------------------------------------------------------
  //  IMU  ->  Correct with z = [theta]
  //  IMU provides a good absolute heading reference.
  // ------------------------------------------------------------
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!initialized_) return;

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = quaternionToYaw(msg->orientation);

    // H extracts theta from [x, y, theta]
    Eigen::Matrix<double, 1, 3> H;
    H << 0.0, 0.0, 1.0;

    correct<1>(z, H, R_imu_);
  }

  // ------------------------------------------------------------
  //  LaserScan  ->  Landmark-based correction  (stub)
  //
  //  TODO: implement a range-and-bearing measurement model.
  //
  //  Suggested approach:
  //    1. Declare a known landmark position (lx, ly) in map frame.
  //    2. Identify the landmark return in msg->ranges (e.g. the
  //       minimum range within a narrow angular window).
  //    3. Build the expected measurement from the current state:
  //         r_hat = hypot(lx - x_(0),  ly - x_(1))
  //         a_hat = atan2(ly - x_(1),  lx - x_(0)) - x_(2)
  //    4. Compute the Jacobian H (2x3) of (r, bearing) w.r.t. state.
  //    5. Call correct<2>(z, H, R_scan_) with measured (r, bearing).
  // ------------------------------------------------------------
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    if (!initialized_) return;
    (void)msg;   // remove once landmark detection is implemented
  }

  // ============================================================
  //  Publisher
  // ============================================================
  void publishEstimate()
  {
    if (!initialized_) return;

    geometry_msgs::msg::PoseWithCovarianceStamped out;
    out.header.stamp    = this->get_clock()->now();
    out.header.frame_id = "map";

    // Pose
    out.pose.pose.position.x = x_(0);
    out.pose.pose.position.y = x_(1);
    out.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x_(2));
    out.pose.pose.orientation = tf2::toMsg(q);

    // Covariance: 6x6 row-major  (x, y, z, roll, pitch, yaw)
    // Map 3x3 P_ into the relevant rows/cols (x->0, y->1, yaw->5)
    out.pose.covariance.fill(0.0);
    out.pose.covariance[0]  = P_(0, 0);   // var(x)
    out.pose.covariance[1]  = P_(0, 1);   // cov(x, y)
    out.pose.covariance[5]  = P_(0, 2);   // cov(x, yaw)
    out.pose.covariance[6]  = P_(1, 0);   // cov(y, x)
    out.pose.covariance[7]  = P_(1, 1);   // var(y)
    out.pose.covariance[11] = P_(1, 2);   // cov(y, yaw)
    out.pose.covariance[30] = P_(2, 0);   // cov(yaw, x)
    out.pose.covariance[31] = P_(2, 1);   // cov(yaw, y)
    out.pose.covariance[35] = P_(2, 2);   // var(yaw)

    pose_pub_->publish(out);
  }

  // ============================================================
  //  Helpers
  // ============================================================
  double quaternionToYaw(const geometry_msgs::msg::Quaternion& q) const
  {
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    return yaw;
  }

  static double normalizeAngle(double a)
  {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  // ============================================================
  //  Members
  // ============================================================

  // KF matrices
  Eigen::Vector3d x_;                       // state  [x, y, theta]
  Eigen::Matrix3d P_;                       // state covariance
  Eigen::Matrix3d Q_;                       // process noise covariance
  Eigen::Matrix3d R_odom_;                  // odometry measurement noise
  Eigen::Matrix<double, 1, 1> R_imu_;       // IMU measurement noise

  // Control input (updated from /cmd_vel)
  double v_     {0.0};    // linear  velocity [m/s]
  double omega_ {0.0};    // angular velocity [rad/s]

  // Bookkeeping
  rclcpp::Time last_time_;
  bool initialized_ {false};

  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr       imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   cmd_vel_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

// ============================================================
//  Entry point
// ============================================================
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}