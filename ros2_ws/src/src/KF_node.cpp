#include <cmath>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
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
    // Noise matrices loaded from parameters (see config/filter_params.yaml).
    // Hardcoded values below are fallback defaults if no YAML is supplied.
    //   R      : process noise         (predict step, Line 3)
    //   Q_odom : odom pose meas. noise (correct step, Line 4)
    //   Q_imu  : IMU yaw meas. noise   (correct step, Line 4)
    // ----------------------------------------------------------
    R_      = diagMatrix3FromParam("R_diag",      {0.05, 0.05, 0.01});
    Q_odom_ = diagMatrix3FromParam("Q_odom_diag", {0.1, 0.1, 0.05});
    Q_imu_(0, 0) = declare_parameter<double>("Q_imu", 0.02);

    // ----------------------------------------------------------
    // Control Subscriber (/cmd_vel drives the prediction step)
    // ----------------------------------------------------------
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&KalmanFilterNode::cmdVelCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Synchronized Subscribers (Odom & IMU) for the correction step
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
  //  Prediction Step (linear KF, slide 17)
  //
  //  Driven by /cmd_vel (v, omega). The body-frame velocity is
  //  rotated into a world-frame control increment using the current
  //  heading, so the state update stays linear (A = I, B = I):
  //
  //    u_t = [ v*cos(theta)*dt,  v*sin(theta)*dt,  omega*dt ]
  //
  //  Line 2: mu_bar_t    = A_t * mu_{t-1} + B_t * u_t
  //  Line 3: Sigma_bar_t = A_t * Sigma_{t-1} * A_t^T + R_t
  //
  //  NOTE: keeping A_t = I (instead of the EKF Jacobian G_t) ignores
  //  the theta-coupling term in the covariance propagation. This is the
  //  deliberate linear-KF approximation that distinguishes it from the
  //  EKF; the mean update itself uses the full velocity motion model.
  // ============================================================

  void predict(double dt)
  {
    const double theta = mu_(2);

    const Eigen::Vector3d u_t(v_ * std::cos(theta) * dt,
                              v_ * std::sin(theta) * dt,
                              omega_ * dt);

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
      mu_(0)       = x;
      mu_(1)       = y;
      mu_(2)       = yaw;
      last_time_   = this->get_clock()->now();
      initialized_ = true;
      RCLCPP_INFO(get_logger(),
                  "KF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    // Full-pose measurement z = [x, y, theta]; C_t = I observes the whole state
    const Eigen::Vector3d z(x, y, yaw);
    const Eigen::Matrix3d C_t = Eigen::Matrix3d::Identity();
    correct<3>(z, C_t, Q_odom_);
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
  // Declare a [d0, d1, d2] parameter and build a 3x3 diagonal noise matrix.
  Eigen::Matrix3d diagMatrix3FromParam(const std::string& name,
                                       const std::vector<double>& def)
  {
    std::vector<double> v = declare_parameter<std::vector<double>>(name, def);
    if (v.size() != 3) {
      RCLCPP_WARN(get_logger(),
                  "Parameter '%s' must have 3 elements; falling back to defaults.",
                  name.c_str());
      v = def;
    }
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    m(0, 0) = v[0];
    m(1, 1) = v[1];
    m(2, 2) = v[2];
    return m;
  }

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

  Eigen::Vector3d mu_;                  // State mean  [x, y, theta]
  Eigen::Matrix3d Sigma_;               // State covariance
  Eigen::Matrix3d R_;                   // Process noise        (predict step, Line 3)
  Eigen::Matrix3d Q_odom_;              // Odometry pose noise   (correct step, Line 4)
  Eigen::Matrix<double, 1, 1> Q_imu_;  // IMU yaw noise         (correct step, Line 4)

  double v_     {0.0};   // Linear  velocity from /cmd_vel
  double omega_ {0.0};   // Angular velocity from /cmd_vel

  rclcpp::Time last_time_;
  bool initialized_ {false};

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

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
