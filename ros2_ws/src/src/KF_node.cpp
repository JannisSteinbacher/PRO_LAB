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

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// ============================================================
//  Linear KalmanFilterNode
// ============================================================

class KalmanFilterNode : public rclcpp::Node
{
public:
  KalmanFilterNode() : Node("kalman_filter_node")
  {
    // ----------------------------------------------------------
    // Initial state  mu = [0, 0, 0]
    // ----------------------------------------------------------
    mu_ = Eigen::Vector3d::Zero();

    // ----------------------------------------------------------
    // Initial state covariance Sigma (high uncertainty at start)
    // ----------------------------------------------------------
    Sigma_ = Eigen::Matrix3d::Identity() * 1.0;

    // ----------------------------------------------------------
    // Process noise R 
    // ----------------------------------------------------------
    R_ = Eigen::Matrix3d::Zero();
    R_(0, 0) = 0.05;   
    R_(1, 1) = 0.05;   
    R_(2, 2) = 0.01;   

    // ----------------------------------------------------------
    // Measurement noise Q 
    // ----------------------------------------------------------
    Q_odom_ = Eigen::Matrix3d::Zero();
    Q_odom_(0, 0) = 0.1;   
    Q_odom_(1, 1) = 0.1;   
    Q_odom_(2, 2) = 0.05;  

    Q_imu_(0, 0) = 0.02;   

    // ----------------------------------------------------------
    // Independent Subscribers
    // ----------------------------------------------------------
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&KalmanFilterNode::scanCallback, this, std::placeholders::_1));

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&KalmanFilterNode::cmdVelCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Synchronized Subscribers (Odom & IMU)
    // ----------------------------------------------------------
    odom_filter_.subscribe(this, "/odom");
    imu_filter_.subscribe(this, "/imu");

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), odom_filter_, imu_filter_
    );

    sync_->registerCallback(
      std::bind(&KalmanFilterNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // ----------------------------------------------------------
    // Publisher & Timer
    // ----------------------------------------------------------
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/kf/pose_estimate", 10);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&KalmanFilterNode::publishEstimate, this));

    last_time_ = this->get_clock()->now();

    RCLCPP_INFO(get_logger(), "Linear Kalman Filter node started.");
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry, 
    sensor_msgs::msg::Imu
  > SyncPolicy;


  // ============================================================
  //  Prediction Step (linear KF model from the slides)
  // ============================================================

  void predict(double dt)
  {
    // 1. Control vector u_t = [v, omega]^T directly from cmd_vel.
    Eigen::Vector2d u_t(v_, omega_);

    // 2. Linear state prediction:
    //
    //   mu_t = A_t * mu_{t-1} + B_t * u_t
    //
    // This is intentionally linear. It does not rotate the translational
    // velocity with cos(theta)/sin(theta); that nonlinear model belongs to
    // the EKF.
    Eigen::Matrix3d A_t = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 3, 2> B_t;
    B_t << dt, 0.0,
           0.0, 0.0,
           0.0, dt;

    mu_ = A_t * mu_ + B_t * u_t;
    mu_(2) = normalizeAngle(mu_(2));

    // 3. Linear covariance prediction:
    //
    //   Sigma_t = A_t * Sigma_{t-1} * A_t^T + R_t
    Sigma_ = A_t * Sigma_ * A_t.transpose() + R_;
  }

  template<int M>
  void correct(const Eigen::Matrix<double, M, 1>& z,
               const Eigen::Matrix<double, M, 3>& C_t,
               const Eigen::Matrix<double, M, M>& R)
  {
    // Calculate Innovation
    Eigen::Matrix<double, M, 1> innovation = z - C_t * mu_;

    // Handle angle wrap-around for yaw
    for (int i = 0; i < M; ++i) {
      if (std::abs(innovation(i)) > M_PI && std::abs(z(i)) < M_PI + 0.5) {
        innovation(i) = normalizeAngle(innovation(i));
      }
    }

    // Line 4: Kalman Gain K_t = \bar{Sigma}_t * C_t^T * (C_t * \bar{Sigma}_t * C_t^T + Measurement Noise)^-1
    const Eigen::Matrix<double, M, M> S = C_t * Sigma_ * C_t.transpose() + R;
    const Eigen::Matrix<double, 3, M> K_t = Sigma_ * C_t.transpose() * S.inverse();

    // Line 5: mu_t = \bar{mu}_t + K_t * (z_t - C_t * \bar{mu}_t)
    mu_ = mu_ + K_t * innovation;
    mu_(2) = normalizeAngle(mu_(2));

    // Line 6: Sigma_t = (I - K_t * C_t) * \bar{Sigma}_t
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Sigma_ = (I - K_t * C_t) * Sigma_;
  }

  // ============================================================
  //  Callbacks
  // ============================================================

  void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg,
                    const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
  {
    processOdom(odom_msg);
    processImu(imu_msg);
  }

  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    v_     = msg->linear.x;
    omega_ = msg->angular.z;

    if (!initialized_) return;

    const auto now = this->get_clock()->now();
    const double dt = (now - last_time_).seconds();
    last_time_ = now;

    if (dt > 0.0 && dt < 1.0) { 
      predict(dt);
    }
  }

  void processOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
  {
    const double yaw = quaternionToYaw(msg->pose.pose.orientation);

    if (!initialized_) {
      mu_(0)       = msg->pose.pose.position.x;
      mu_(1)       = msg->pose.pose.position.y;
      mu_(2)       = yaw;
      last_time_   = this->get_clock()->now();
      initialized_ = true;
      RCLCPP_INFO(get_logger(),
                  "KF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    const Eigen::Vector3d z(msg->pose.pose.position.x,
                            msg->pose.pose.position.y,
                            yaw);

    // Measurement Matrix C_t for Odom (maps state [x, y, theta] directly to [x, y, theta])
    const Eigen::Matrix3d C_t = Eigen::Matrix3d::Identity();
    correct<3>(z, C_t, Q_odom_);
  }

  void processImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
  {
    if (!initialized_) return;

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = quaternionToYaw(msg->orientation);

    // Measurement Matrix C_t for IMU (maps state [x, y, theta] to [theta])
    Eigen::Matrix<double, 1, 3> C_t;
    C_t << 0.0, 0.0, 1.0;

    correct<1>(z, C_t, Q_imu_);
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    if (!initialized_) return;
    (void)msg;   
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

    out.pose.pose.position.x = mu_(0);
    out.pose.pose.position.y = mu_(1);
    out.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, mu_(2));
    out.pose.pose.orientation = tf2::toMsg(q);

    out.pose.covariance.fill(0.0);
    out.pose.covariance[0]  = Sigma_(0, 0);   
    out.pose.covariance[1]  = Sigma_(0, 1);   
    out.pose.covariance[5]  = Sigma_(0, 2);   
    out.pose.covariance[6]  = Sigma_(1, 0);   
    out.pose.covariance[7]  = Sigma_(1, 1);   
    out.pose.covariance[11] = Sigma_(1, 2);   
    out.pose.covariance[30] = Sigma_(2, 0);   
    out.pose.covariance[31] = Sigma_(2, 1);   
    out.pose.covariance[35] = Sigma_(2, 2);   

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

  Eigen::Vector3d mu_;                       
  Eigen::Matrix3d Sigma_;                       
  Eigen::Matrix3d R_;                       
  Eigen::Matrix3d Q_odom_;                  
  Eigen::Matrix<double, 1, 1> Q_imu_;       

  double v_     {0.0};    
  double omega_ {0.0};    

  rclcpp::Time last_time_;
  bool initialized_ {false};

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

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
