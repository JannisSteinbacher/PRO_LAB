#include <cmath>
#include <string>
#include <vector>
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
//  Extended Kalman Filter Node
//
//  Follows the EKF algorithm (Thrun, 2006):
//
//  Predict  (nonlinear motion model g(.)):
//    Line 2:  mu_bar_t  = g(u_t, mu_{t-1})
//    Line 3:  Sig_bar_t = G_t * Sigma_{t-1} * G_t^T + R_t
//
//  Correct  (linear measurement model via H_t):
//    Line 4:  K_t       = Sig_bar_t * H_t^T * (H_t * Sig_bar_t * H_t^T + Q_t)^-1
//    Line 5:  mu_t      = mu_bar_t  + K_t * (z_t - H_t * mu_bar_t)
//    Line 6:  Sigma_t   = (I - K_t * H_t) * Sig_bar_t          (Joseph form used)
//
//  Differences vs. linear KF:
//    - predict() applies the nonlinear g(.) directly for the state update
//      instead of A_t * mu + B_t * u
//    - predict() builds the Jacobian G_t analytically and uses it for
//      covariance propagation instead of A_t * Sigma * A_t^T
//    - correct() is structurally identical to the KF but uses H_t
//      (same role as C_t in the KF) and the Joseph form for Sigma
//
//  EKF Localization (landmark correction via scanCallback):
//    - For each observed landmark i:
//        delta   = [m_j.x - mu_x, m_j.y - mu_y]   (landmark - robot, map frame)
//        q       = delta^T * delta
//        h(.)    = [sqrt(q),  atan2(dy,dx) - theta]   <- nonlinear [range, bearing]
//        H_t^i   = (1/q) * [[-sqrt(q)*dx, -sqrt(q)*dy,  0],        <- Jacobian
//                            [         dy,          -dx, -q]]
//        K       = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q_lm)^-1
//        mu      = mu_bar + K * (z - h(mu_bar))
//        Sigma   = (I - K*H) * Sigma_bar * (I - K*H)^T + K*Q_lm*K^T
// ============================================================

// ------------------------------------------------------------
//  Landmark definition
//
//  A landmark is a known feature in the map with a fixed
//  world-frame position and an optional integer signature (id).
//
//  How to define a good landmark:
//    - Landmarks must be UNIQUELY IDENTIFIABLE from sensor data
//      so data association (observed -> map) is unambiguous.
//    - Choose features that are STABLE over time (poles, pillars,
//      corner reflectors) — not dynamic objects like furniture.
//    - Space them so at least 2-3 are ALWAYS VISIBLE for good
//      observability of x, y, and theta simultaneously.
//    - Record world positions precisely (survey or SLAM-built map).
//    - Use 'id' to encode color, reflectance class, or shape so
//      association can be signature-based rather than nearest-neighbour.
// ------------------------------------------------------------
struct Landmark {
    double x;   // world-frame x  [m]
    double y;   // world-frame y  [m]
    int    id;  // signature / class label  (0 = unknown)
};

class ExtendedKalmanFilterNode : public rclcpp::Node
{
public:
  ExtendedKalmanFilterNode() : Node("extended_kalman_filter_node")
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
    //   Q_landmark : [range, bearing] landmark meas. noise
    // ----------------------------------------------------------
    R_      = diagMatrix3FromParam("R_diag",      {0.05, 0.05, 0.01});
    Q_odom_ = diagMatrix3FromParam("Q_odom_diag", {0.1, 0.1, 0.05});
    Q_imu_(0, 0) = declare_parameter<double>("Q_imu", 0.02);

    const std::vector<double> q_lm =
      declare_parameter<std::vector<double>>("Q_landmark_diag", {0.1, 0.05});
    Q_landmark_ = Eigen::Matrix2d::Zero();
    Q_landmark_(0, 0) = q_lm.size() > 0 ? q_lm[0] : 0.1;   // range   [m^2]
    Q_landmark_(1, 1) = q_lm.size() > 1 ? q_lm[1] : 0.05;  // bearing [rad^2]

    // ----------------------------------------------------------
    // Map landmarks  (edit to match your physical environment)
    //
    // Rules for good placement:
    //   - At least 3 landmarks, not collinear
    //   - Spread around the expected robot workspace
    //   - Each landmark visible from multiple robot poses
    // ----------------------------------------------------------
    map_landmarks_ = {
      { 1.0,  1.0, 1},
      { 1.0, -1.0, 2},
      {-1.0,  0.0, 3},
    };

    // Reject scan hit <-> map landmark pairs further apart than this [m]
    association_threshold_ = 0.5;

    // ----------------------------------------------------------
    // Independent Subscribers
    // ----------------------------------------------------------
    /*scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ExtendedKalmanFilterNode::scanCallback, this, std::placeholders::_1));*/

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&ExtendedKalmanFilterNode::cmdVelCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Synchronized Subscribers (Odom & IMU)
    // ----------------------------------------------------------
    odom_filter_.subscribe(this, "/odom");
    imu_filter_.subscribe(this, "/imu");

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), odom_filter_, imu_filter_
    );

    sync_->registerCallback(
      std::bind(&ExtendedKalmanFilterNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // ----------------------------------------------------------
    // Publisher & Timer
    // ----------------------------------------------------------
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/ekf/pose_estimate", 10);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&ExtendedKalmanFilterNode::publishEstimate, this));

    last_time_ = this->get_clock()->now();

    RCLCPP_INFO(get_logger(), "Extended Kalman Filter node started.");
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::Imu
  > SyncPolicy;


  // ============================================================
  //  EKF Core Algorithm
  // ============================================================

  void predict(double dt)
  {
    const double theta = mu_(2);

    // -------------------------------------------------------
    // Line 2: mu_bar_t = g(u_t, mu_{t-1})
    //
    // Nonlinear motion model g(.) for a differential-drive robot
    // (slide 37):
    //   x_t     = x_{t-1}     + v * cos(theta_{t-1}) * dt
    //   y_t     = y_{t-1}     + v * sin(theta_{t-1}) * dt
    //   theta_t = theta_{t-1} + omega * dt
    //
    // This replaces the linear  A_t * mu + B_t * u  of the KF.
    // -------------------------------------------------------
    mu_(0) += v_ * std::cos(theta) * dt;
    mu_(1) += v_ * std::sin(theta) * dt;
    mu_(2) += omega_ * dt;
    mu_(2)  = normalizeAngle(mu_(2));

    // -------------------------------------------------------
    // Line 3: Sigma_bar_t = G_t * Sigma_{t-1} * G_t^T + R_t
    //
    // G_t is the Jacobian of g(.) w.r.t. the state x,
    // evaluated at mu_{t-1}  (slide 41):
    //
    //       | 1   0   -v*sin(theta)*dt |
    //  G_t =| 0   1    v*cos(theta)*dt |
    //       | 0   0         1          |
    //
    // This replaces  A_t * Sigma * A_t^T  of the KF.
    // -------------------------------------------------------
    Eigen::Matrix3d G_t = Eigen::Matrix3d::Identity();
    G_t(0, 2) = -v_ * std::sin(theta) * dt;
    G_t(1, 2) =  v_ * std::cos(theta) * dt;

    Sigma_ = G_t * Sigma_ * G_t.transpose() + R_;
  }

  // ----------------------------------------------------------
  // correct()  —  Lines 4-6 of the EKF algorithm.
  //
  // Structurally identical to the linear KF correct() step;
  // H_t plays the same role as C_t in the KF.
  //
  //   Line 4: K_t     = Sigma_bar * H_t^T * (H_t * Sigma_bar * H_t^T + Q)^-1
  //   Line 5: mu_t    = mu_bar + K_t * (z_t - H_t * mu_bar)
  //   Line 6: Sigma_t = (I - K_t * H_t) * Sigma_bar   [Joseph form]
  // ----------------------------------------------------------
  template<int M>
  void correct(const Eigen::Matrix<double, M, 1>& z,
               const Eigen::Matrix<double, M, 3>& H_t,
               const Eigen::Matrix<double, M, M>& Q)
  {
    // Innovation:  z_t - H_t * mu_bar_t
    Eigen::Matrix<double, M, 1> innovation = z - H_t * mu_;

    // Handle angle wrap-around for yaw components
    for (int i = 0; i < M; ++i) {
      if (std::abs(innovation(i)) > M_PI && std::abs(z(i)) < M_PI + 0.5) {
        innovation(i) = normalizeAngle(innovation(i));
      }
    }

    // Line 4: Kalman Gain
    const Eigen::Matrix<double, M, M> S  = H_t * Sigma_ * H_t.transpose() + Q;
    const Eigen::Matrix<double, 3, M> K_t = Sigma_ * H_t.transpose() * S.inverse();

    // Line 5: State update
    mu_ = mu_ + K_t * innovation;
    mu_(2) = normalizeAngle(mu_(2));

    // Line 6: Covariance update  (Joseph form for numerical stability)
    const Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K_t * H_t;
    Sigma_ = I_KH * Sigma_ * I_KH.transpose() + K_t * Q * K_t.transpose();
  }

  // ----------------------------------------------------------
  // correctLandmark()  —  EKF correction for one landmark observation.
  //
  // Uses the NONLINEAR measurement function h(mu_bar, m_j) and its
  // Jacobian H_t^i (Thrun 2006, EKF_localization):
  //
  //   delta = [m_j.x - mu_x,  m_j.y - mu_y]
  //   q     = delta^T * delta
  //   h(.)  = [sqrt(q),  atan2(dy, dx) - theta]   <- [range, bearing]
  //
  //   H_t^i = (1/q) * [[-sqrt(q)*dx, -sqrt(q)*dy,  0],
  //                     [         dy,          -dx, -q]]
  // ----------------------------------------------------------

  /*void correctLandmark(double z_range, double z_bearing, const Landmark& lm)
  {
    const double dx = lm.x - mu_(0);
    const double dy = lm.y - mu_(1);
    const double q  = dx * dx + dy * dy;
    const double r  = std::sqrt(q);

    if (r < 1e-6) return; // singularity guard

    // Expected measurement  h(mu_bar, m_j)
    Eigen::Vector2d z_hat;
    z_hat(0) = r;
    z_hat(1) = normalizeAngle(std::atan2(dy, dx) - mu_(2));

    // Actual observation
    Eigen::Vector2d z_obs;
    z_obs(0) = z_range;
    z_obs(1) = normalizeAngle(z_bearing);

    // Innovation  (bearing always normalized)
    Eigen::Vector2d innovation;
    innovation(0) = z_obs(0) - z_hat(0);
    innovation(1) = normalizeAngle(z_obs(1) - z_hat(1));

    // Jacobian H_t^i  (2 x 3)
    Eigen::Matrix<double, 2, 3> H;
    H << -dx / r,  -dy / r,   0.0,
          dy / q,  -dx / q,  -1.0;

    // Kalman gain  (Line 4)
    const Eigen::Matrix2d S = H * Sigma_ * H.transpose() + Q_landmark_;
    const Eigen::Matrix<double, 3, 2> K = Sigma_ * H.transpose() * S.inverse();

    // State update  (Line 5)
    mu_ += K * innovation;
    mu_(2) = normalizeAngle(mu_(2));

    // Covariance update  (Line 6, Joseph form)
    const Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K * H;
    Sigma_ = I_KH * Sigma_ * I_KH.transpose() + K * Q_landmark_ * K.transpose();
  }

  // ----------------------------------------------------------
  // detectPoles()
  //
  // Finds pole-like landmarks in a LaserScan by locating range
  // local minima that are significantly closer than both neighbours
  // (an isolated pole appears as a dip in the range profile).
  //
  // Returns (range [m], bearing [rad]) in the robot frame.
  // jump_threshold: minimum step [m] to call a point a pole tip.
  // ----------------------------------------------------------
  std::vector<std::pair<double, double>>
  detectPoles(const sensor_msgs::msg::LaserScan::SharedPtr& scan,
              float jump_threshold = 0.25f)
  {
    std::vector<std::pair<double, double>> poles;
    const int N = static_cast<int>(scan->ranges.size());

    auto valid = [&](int i) {
      const float r = scan->ranges[i];
      return std::isfinite(r) && r >= scan->range_min && r <= scan->range_max;
    };

    for (int i = 1; i < N - 1; ++i) {
      if (!valid(i - 1) || !valid(i) || !valid(i + 1)) continue;
      const float r_prev = scan->ranges[i - 1];
      const float r_curr = scan->ranges[i];
      const float r_next = scan->ranges[i + 1];

      if (r_curr < r_prev - jump_threshold && r_curr < r_next - jump_threshold) {
        const double bearing = scan->angle_min + i * scan->angle_increment;
        poles.emplace_back(static_cast<double>(r_curr), bearing);
      }
    }
    return poles;
  }*/

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
                  "EKF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    const Eigen::Vector3d z(msg->pose.pose.position.x,
                            msg->pose.pose.position.y,
                            yaw);

    // H_t maps the full state [x, y, theta] directly to [x, y, theta]
    const Eigen::Matrix3d H_t = Eigen::Matrix3d::Identity();
    correct<3>(z, H_t, Q_odom_);
  }

  void processImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
  {
    if (!initialized_) return;

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = quaternionToYaw(msg->orientation);

    // H_t maps the state [x, y, theta] to [theta] only (slide 21)
    Eigen::Matrix<double, 1, 3> H_t;
    H_t << 0.0, 0.0, 1.0;

    correct<1>(z, H_t, Q_imu_);
  }

  // ----------------------------------------------------------
  // scanCallback  —  EKF localization with landmarks
  //
  // For each scan frame:
  //   1. Detect pole candidates  (robot frame: range, bearing)
  //   2. Project each pole into the map frame using current mu_
  //   3. Associate with nearest map landmark within threshold
  //   4. Run correctLandmark() for each accepted association
  // ----------------------------------------------------------
  
  /*void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    if (!initialized_) return;

    for (const auto& [range, bearing] : detectPoles(msg)) {
      // Project scan hit into map frame
      const double obs_x = mu_(0) + range * std::cos(mu_(2) + bearing);
      const double obs_y = mu_(1) + range * std::sin(mu_(2) + bearing);

      // Nearest-neighbour data association
      const Landmark* match = nullptr;
      double best_dist = association_threshold_;
      for (const auto& lm : map_landmarks_) {
        const double dx   = lm.x - obs_x;
        const double dy   = lm.y - obs_y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < best_dist) {
          best_dist = dist;
          match     = &lm;
        }
      }

      if (match == nullptr) continue;

      correctLandmark(range, bearing, *match);
    }
  }*/

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

  Eigen::Vector3d mu_;                      // State mean  [x, y, theta]
  Eigen::Matrix3d Sigma_;                   // State covariance
  Eigen::Matrix3d R_;                       // Process noise (predict step, Line 3)
  Eigen::Matrix3d Q_odom_;                  // Odometry measurement noise (correct step, Line 4)
  Eigen::Matrix<double, 1, 1> Q_imu_;      // IMU measurement noise     (correct step, Line 4)
  Eigen::Matrix2d Q_landmark_;             // Landmark [range, bearing] noise
  std::vector<Landmark> map_landmarks_;    // Known landmark positions in map frame
  double association_threshold_;           // Max map-projection error [m] for association

  double v_     {0.0};   // Linear  velocity from /cmd_vel
  double omega_ {0.0};   // Angular velocity from /cmd_vel

  rclcpp::Time last_time_;
  bool initialized_ {false};

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu>   imu_filter_;
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
  rclcpp::spin(std::make_shared<ExtendedKalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}