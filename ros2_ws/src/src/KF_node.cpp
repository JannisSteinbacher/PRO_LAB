#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/float64.hpp>

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
    // Initial state  mu_ = [0, 0, 0]
    // ----------------------------------------------------------
    mu_ = Eigen::Vector3d::Zero();

    // ----------------------------------------------------------
    // Initial state covariance Sigma 
    // ----------------------------------------------------------
    Sigma_ = Eigen::Matrix3d::Identity() * 0.0;

    // ----------------------------------------------------------
    // Noise matrices — loaded from parameters (config/filter_params.yaml).
    // ----------------------------------------------------------
    R_ = diagMatrix3FromParam("R_diag");

    declare_parameter("Q_imu", rclcpp::PARAMETER_DOUBLE);
    Q_imu_(0, 0) = get_parameter("Q_imu").as_double();

    // ----------------------------------------------------------
    // Synchronized Subscribers (Odom & IMU)
    //   /odom drives the PREDICT step (odometry motion model, relative
    //   increments); /imu drives a yaw CORRECTION. They are synchronized so
    //   each cycle predicts then corrects in the proper KF order.
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

    // Scalar Kalman gain applied to theta from the IMU yaw correction, for the
    // eval node to log
    kgain_pub_ = create_publisher<std_msgs::msg::Float64>(
      "/kf/kalman_gain_theta", 10);

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
  //  Prediction Step  —  LINEAR motion model.
  //
  //  The control input u_t = [dx, dy, dtheta] is the WORLD-FRAME pose
  //  increment. The body->world rotation (the cos/sin of the heading) is the
  //  nonlinearity, and it is performed OUTSIDE this node by the odometry
  //  source, which already integrates the motion into world-frame x/y. The
  //  node therefore only superimposes increments and stays fully linear.
  //
  //  Line 2: mu_bar_t    = A_t * mu_{t-1} + B_t * u_t
  //  Line 3: Sigma_bar_t = A_t * Sigma_{t-1} * A_t^T + R_t
  // ============================================================

  void predict(const Eigen::Vector3d& u)
  {
    const Eigen::Matrix3d A_t = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d B_t = Eigen::Matrix3d::Identity();

    // Line 2: linear mean update (no trigonometry of the state).
    mu_ = A_t * mu_ + B_t * u;
    mu_(2) = normalizeAngle(mu_(2));

    // Line 3: covariance update.
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

    // Record the theta-row gain w.r.t. the first measurement for logging.
    // correct<>() is only ever called for the IMU yaw update (M=1), so
    // K_t(2, 0) is the Kalman gain applied to theta from the yaw measurement.
    k_theta_imu_ = K_t(2, 0);

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
    predictFromOdom(odom_msg);
    processImu(imu_msg);         
  }

  // ----------------------------------------------------------
  // predictFromOdom  —  turn the raw /odom pose into a world-frame motion input.
  //
  // The first message anchors the filter (and the odom-delta reference). Every
  // later message yields a world-frame pose INCREMENT u = [dx, dy, dtheta] that
  // is handed straight to the linear predict(). Only the delta is used, so the
  // absolute odom drift never re-enters the estimate, and because odometry has
  // already projected the body motion into world x/y, no nonlinearity remains
  // in the node.
  // ----------------------------------------------------------
  void predictFromOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
  {
    const double ox   = msg->pose.pose.position.x;
    const double oy   = msg->pose.pose.position.y;
    const double oyaw = quaternionToYaw(msg->pose.pose.orientation);

    if (!initialized_) {
      mu_ << ox, oy, oyaw;          // start at the first odom pose
      last_odom_ << ox, oy, oyaw;   // reference for the first delta
      initialized_ = true;
      RCLCPP_INFO(get_logger(),
                  "KF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    // World-frame pose increment reported by odometry since the last message.
    Eigen::Vector3d u;
    u(0) = ox - last_odom_(0);
    u(1) = oy - last_odom_(1);
    u(2) = normalizeAngle(oyaw - last_odom_(2));
    last_odom_ << ox, oy, oyaw;

    predict(u);
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

    // Publish the latest theta Kalman gain alongside the pose (same rate).
    std_msgs::msg::Float64 kg;
    kg.data = k_theta_imu_;
    kgain_pub_->publish(kg);
  }

  // ============================================================
  //  Helpers
  // ============================================================
  // Read a REQUIRED [d0, d1, d2] parameter and build a 3x3 diagonal noise
  // matrix. No in-code default: if the value is not provided via parameters
  // (filter_params.yaml) the node throws on start, so the YAML is always used.
  Eigen::Matrix3d diagMatrix3FromParam(const std::string& name)
  {
    declare_parameter(name, rclcpp::PARAMETER_DOUBLE_ARRAY);
    const std::vector<double> v = get_parameter(name).as_double_array();
    if (v.size() != 3) {
      throw std::runtime_error("Parameter '" + name + "' must have exactly 3 "
                               "elements [x, y, theta].");
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
  Eigen::Matrix3d R_;                   // Process noise        
  Eigen::Matrix<double, 1, 1> Q_imu_;  // IMU yaw noise       

  // Latest theta Kalman gain from the IMU yaw correction (published for evaluation).
  double k_theta_imu_ {0.0};

  // Last raw /odom pose [x, y, theta]; the reference for the next motion delta.
  Eigen::Vector3d last_odom_ {Eigen::Vector3d::Zero()};
  bool initialized_ = false;

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu>   imu_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr kgain_pub_;
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
