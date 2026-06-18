#include <cmath>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
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
    // Measurement noise Q (IMU yaw)
    // ----------------------------------------------------------
    Q_imu_(0, 0) = 0.02;

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

    RCLCPP_INFO(get_logger(), "Linear Kalman Filter node started.");
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::Imu
  > SyncPolicy;


  // ============================================================
  //  Prediction Step (linear KF, slide 17)
  //
  //  Control: u_t = [Δx, Δy, Δθ]  (odometry pose deltas)
  //  Model:   A = I,  B = I
  //
  //  Line 2: mu_bar_t  = A_t * mu_{t-1} + B_t * u_t
  //  Line 3: Sigma_bar_t = A_t * Sigma_{t-1} * A_t^T + R_t
  // ============================================================

  void predict(double delta_x, double delta_y, double delta_theta)
  {
    const Eigen::Vector3d u_t(delta_x, delta_y, delta_theta);

    // A = I, B = I  →  mu_bar_t = mu_{t-1} + u_t
    const Eigen::Matrix3d A_t = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d B_t = Eigen::Matrix3d::Identity();

    mu_ = A_t * mu_ + B_t * u_t;
    mu_(2) = normalizeAngle(mu_(2));

    Sigma_ = A_t * Sigma_ * A_t.transpose() + R_;
  }

  // ============================================================
  //  Correction Step
  //
  //  Line 4: K_t  = Sigma_bar_t * C_t^T * (C_t * Sigma_bar_t * C_t^T + Q_t)^{-1}
  //  Line 5: mu_t = mu_bar_t + K_t * (z_t - C_t * mu_bar_t)
  //  Line 6: Sigma_t = (I - K_t * C_t) * Sigma_bar_t
  // ============================================================

  template<int M>
  void correct(const Eigen::Matrix<double, M, 1>& z,
               const Eigen::Matrix<double, M, 3>& C_t,
               const Eigen::Matrix<double, M, M>& Q)
  {
    Eigen::Matrix<double, M, 1> innovation = z - C_t * mu_;

    // Handle angle wrap-around for yaw
    for (int i = 0; i < M; ++i) {
      if (std::abs(innovation(i)) > M_PI && std::abs(z(i)) < M_PI + 0.5) {
        innovation(i) = normalizeAngle(innovation(i));
      }
    }

    // Line 4: Kalman Gain
    const Eigen::Matrix<double, M, M> S = C_t * Sigma_ * C_t.transpose() + Q;
    const Eigen::Matrix<double, 3, M> K_t = Sigma_ * C_t.transpose() * S.inverse();

    // Line 5: correct mean
    mu_ = mu_ + K_t * innovation;
    mu_(2) = normalizeAngle(mu_(2));

    // Line 6: correct covariance
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

  void processOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
  {
    const double x   = msg->pose.pose.position.x;
    const double y   = msg->pose.pose.position.y;
    const double yaw = quaternionToYaw(msg->pose.pose.orientation);

    if (!initialized_) {
      mu_(0)         = x;
      mu_(1)         = y;
      mu_(2)         = yaw;
      prev_odom_x_   = x;
      prev_odom_y_   = y;
      prev_odom_yaw_ = yaw;
      initialized_   = true;
      RCLCPP_INFO(get_logger(),
                  "KF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    // Compute pose deltas as control input u_t = [Δx, Δy, Δθ]
    const double delta_x   = x   - prev_odom_x_;
    const double delta_y   = y   - prev_odom_y_;
    const double delta_yaw = normalizeAngle(yaw - prev_odom_yaw_);

    prev_odom_x_   = x;
    prev_odom_y_   = y;
    prev_odom_yaw_ = yaw;

    predict(delta_x, delta_y, delta_yaw);
  }

  void processImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
  {
    if (!initialized_) return;

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = quaternionToYaw(msg->orientation);

    // C_t maps state [x, y, theta] to measurement [theta]
    Eigen::Matrix<double, 1, 3> C_t;
    C_t << 0.0, 0.0, 1.0;

    correct<1>(z, C_t, Q_imu_);
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
  Eigen::Matrix<double, 1, 1> Q_imu_;

  double prev_odom_x_ = 0.0;
  double prev_odom_y_ = 0.0;
  double prev_odom_yaw_ = 0.0;

  bool initialized_ {false};

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu>   imu_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

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
