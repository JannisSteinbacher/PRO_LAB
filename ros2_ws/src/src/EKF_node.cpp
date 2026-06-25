#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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
//  Predict  (nonlinear odometry motion model g(.)):
//    Line 2:  mu_bar_t  = g(u_t, mu_{t-1})
//    Line 3:  Sig_bar_t = G_t * Sigma_{t-1} * G_t^T + R_t
//
//    The motion input u_t is the INCREMENTAL pose change reported by /odom
//    (the delta between two consecutive odom messages), decomposed into
//    rot1 / trans / rot2 and re-applied from the filter's CURRENT pose.
//    We deliberately use only the odometry *increment*, never its absolute
//    pose: raw odometry drifts, so feeding its absolute pose back in as a
//    measurement would continuously drag the estimate onto the drifted track
//    and undo every landmark fix the moment a tag leaves view. Landmarks
//    (and IMU yaw) are therefore the ONLY absolute corrections — once a tag
//    snaps the pose, the next odom increments build forward from that
//    corrected anchor instead of snapping back to where odom thinks it is.
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
//  EKF Localization with AprilTag landmarks (aprilTagCallback):
//    Implements EKF_localization_known_correspondences (Thrun 2006, slide 25).
//    The TurtleBot4 camera feed is processed by apriltag_detector_node, which
//    publishes each visible tag's position in the robot (base_link) frame on
//    /apriltag/detections (visualization_msgs/MarkerArray, marker.id = tag id).
//
//    - Known correspondence: the AprilTag id IS the landmark signature, so each
//      detection maps unambiguously to one map landmark (no nearest-neighbour).
//    - For each observed landmark i:
//        z       = [range, bearing] from the detection (robot frame)
//        delta   = [m_j.x - mu_x, m_j.y - mu_y]   (landmark - robot, map frame)
//        q       = delta^T * delta
//        h(.)    = [sqrt(q),  atan2(dy,dx) - theta]   <- nonlinear [range, bearing]
//        H_t^i   = (1/q) * [[-sqrt(q)*dx, -sqrt(q)*dy,  0],        <- Jacobian
//                            [         dy,          -dx, -q]]
//        K       = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q_lm)^-1
//        mu      = mu_bar + K * (z - h(mu_bar))
//        Sigma   = (I - K*H) * Sigma_bar * (I - K*H)^T + K*Q_lm*K^T
//
//    Frame note: the filter state lives in the spawn-relative 'odom' frame
//    (mu starts at 0,0 where the robot spawns), while the AprilTag world
//    positions come from the Gazebo world frame. The map is therefore shifted
//    by the spawn pose 'map_origin' so landmarks and state share one frame.
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
    Sigma_ = Eigen::Matrix3d::Identity() * 0.2;

    // ----------------------------------------------------------
    // Noise matrices — loaded ONLY from parameters (config/filter_params.yaml).
    // These are required: there are no in-code defaults, so if the YAML is not
    // supplied the node fails to start. That guarantees the tuning always comes
    // from filter_params.yaml (single source of truth).
    //   R      : process noise         (predict step, Line 3 — odom motion)
    //   Q_imu  : IMU yaw meas. noise   (correct step, Line 4)
    //   Q_landmark : [range, bearing] landmark meas. noise
    //
    // Note: there is no Q_odom here. Odometry is the motion INPUT to predict(),
    // not a correction, so it has no measurement-noise matrix. (The shared
    // filter_params.yaml still defines Q_odom_diag for the linear KF node.)
    // ----------------------------------------------------------
    R_      = diagMatrix3FromParam("R_diag");

    declare_parameter("Q_imu", rclcpp::PARAMETER_DOUBLE);
    Q_imu_(0, 0) = get_parameter("Q_imu").as_double();

    declare_parameter("Q_landmark_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);
    const std::vector<double> q_lm = get_parameter("Q_landmark_diag").as_double_array();
    if (q_lm.size() != 2) {
      throw std::runtime_error("Parameter 'Q_landmark_diag' must have exactly 2 "
                               "elements [range, bearing].");
    }
    Q_landmark_ = Eigen::Matrix2d::Zero();
    Q_landmark_(0, 0) = q_lm[0];   // range   [m^2]
    Q_landmark_(1, 1) = q_lm[1];   // bearing [rad^2]

    // ----------------------------------------------------------
    // AprilTag landmark map.
    //
    // The tag world positions (parameters below) match the Gazebo world SDF
    // (depot_apriltags.sdf). 'map_origin' is the robot spawn pose in that world
    // frame; each landmark is rotated/translated into the filter's odom frame:
    //
    //   delta = (world_xy - origin_xy)
    //   lm    = R(-yaw0) * delta            (R(-yaw0) = [[c, s], [-s, c]])
    //
    // The signature 'id' equals the AprilTag id, giving known correspondence.
    // Defaults are the 5 tags in depot_apriltags.sdf with spawn pose (-8, 0, 0).
    // ----------------------------------------------------------
    const std::vector<int64_t> ids =
      declare_parameter<std::vector<int64_t>>("landmark_ids", {0, 1, 2, 3, 4});
    const std::vector<double> wx = declare_parameter<std::vector<double>>(
      "landmark_world_x", {-7.307, -0.9047, -15.0213, -7.66, -1.589});
    const std::vector<double> wy = declare_parameter<std::vector<double>>(
      "landmark_world_y", {3.6816, 4.5724, 2.811, -3.654, 3.5964});
    const std::vector<double> origin =
      declare_parameter<std::vector<double>>("map_origin", {-8.0, 0.0, 0.0});

    const double x0 = origin.size() > 0 ? origin[0] : 0.0;
    const double y0 = origin.size() > 1 ? origin[1] : 0.0;
    const double yaw0 = origin.size() > 2 ? origin[2] : 0.0;
    const double c = std::cos(yaw0), s = std::sin(yaw0);

    map_landmarks_.clear();
    const size_t n = std::min(ids.size(), std::min(wx.size(), wy.size()));
    for (size_t i = 0; i < n; ++i) {
      const double dx = wx[i] - x0;
      const double dy = wy[i] - y0;
      Landmark lm;
      lm.x  =  c * dx + s * dy;   // world -> odom frame
      lm.y  = -s * dx + c * dy;
      lm.id = static_cast<int>(ids[i]);
      map_landmarks_.push_back(lm);
      RCLCPP_INFO(get_logger(),
                  "Landmark id=%d  world(%.3f, %.3f) -> odom(%.3f, %.3f)",
                  lm.id, wx[i], wy[i], lm.x, lm.y);
    }

    // ----------------------------------------------------------
    // Independent Subscribers
    // ----------------------------------------------------------
    apriltag_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "/apriltag/detections", 10,
      std::bind(&ExtendedKalmanFilterNode::aprilTagCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Synchronized Subscribers (Odom & IMU)
    //   /odom drives the PREDICT step (odometry motion model, relative
    //   increments); /imu drives a yaw CORRECTION. They are synchronized so
    //   each cycle predicts then corrects in the proper EKF order.
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

  // predict()  —  Lines 2-3 of the EKF, driven by the ODOMETRY motion model.
  //
  // Instead of integrating a commanded velocity, the motion input u_t is the
  // pose INCREMENT reported by /odom between the previous and current message.
  // The increment is decomposed (Thrun's odometry motion model) into
  //   rot1  : initial turn toward the direction of travel
  //   trans : straight-line distance travelled
  //   rot2  : final turn to the new heading
  // and then re-applied starting from the filter's CURRENT pose mu_. This is
  // the whole point of the fix: we consume only the odom *delta*, never its
  // absolute (drifting) pose, so any landmark correction already baked into
  // mu_ is preserved and the next motion builds forward from that anchor.
  void predict(double rot1, double trans, double rot2)
  {
    // -------------------------------------------------------
    // Line 2: mu_bar_t = g(u_t, mu_{t-1})
    //
    //   heading = theta_{t-1} + rot1   (direction of travel in the filter frame)
    //   x_t     = x_{t-1}     + trans * cos(heading)
    //   y_t     = y_{t-1}     + trans * sin(heading)
    //   theta_t = theta_{t-1} + rot1 + rot2
    // -------------------------------------------------------
    const double heading = mu_(2) + rot1;
    mu_(0) += trans * std::cos(heading);
    mu_(1) += trans * std::sin(heading);
    mu_(2)  = normalizeAngle(mu_(2) + rot1 + rot2);

    // -------------------------------------------------------
    // Line 3: Sigma_bar_t = G_t * Sigma_{t-1} * G_t^T + R_t
    //
    // G_t is the Jacobian of g(.) w.r.t. the state x, evaluated at mu_{t-1}:
    //
    //       | 1   0   -trans*sin(heading) |
    //  G_t =| 0   1    trans*cos(heading) |
    //       | 0   0          1            |
    // -------------------------------------------------------
    Eigen::Matrix3d G_t = Eigen::Matrix3d::Identity();
    G_t(0, 2) = -trans * std::sin(heading);
    G_t(1, 2) =  trans * std::cos(heading);

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

  void correctLandmark(double z_range, double z_bearing, const Landmark& lm)
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

  // ============================================================
  //  Callbacks
  // ============================================================

  void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg,
                    const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
  {
    predictFromOdom(odom_msg);   // Lines 2-3: motion model from the odom delta
    processImu(imu_msg);         // Lines 4-6: yaw correction
  }

  // ----------------------------------------------------------
  // predictFromOdom  —  turn the raw /odom pose into a relative motion input.
  //
  // The first message anchors the filter (and the odom-delta reference). Every
  // later message yields a pose INCREMENT, which is decomposed into the
  // odometry motion model (rot1, trans, rot2) and handed to predict(). Only the
  // delta is used, so the absolute odom drift never re-enters the estimate.
  // ----------------------------------------------------------
  void predictFromOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
  {
    const double ox  = msg->pose.pose.position.x;
    const double oy  = msg->pose.pose.position.y;
    const double oyaw = quaternionToYaw(msg->pose.pose.orientation);

    if (!initialized_) {
      mu_ << ox, oy, oyaw;          // start at the first odom pose
      last_odom_ << ox, oy, oyaw;   // reference for the first delta
      initialized_ = true;
      RCLCPP_INFO(get_logger(),
                  "EKF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  mu_(0), mu_(1), mu_(2));
      return;
    }

    // Pose change reported by odometry since the last message (odom frame).
    const double prev_yaw = last_odom_(2);
    const double dx   = ox - last_odom_(0);
    const double dy   = oy - last_odom_(1);
    const double dyaw = normalizeAngle(oyaw - prev_yaw);
    last_odom_ << ox, oy, oyaw;

    // Decompose into rot1 / trans / rot2 (Thrun's odometry motion model).
    // For a (near-)pure rotation the travel direction is ill-defined, so put
    // the whole turn into rot2 instead of letting atan2 amplify odom jitter.
    const double trans = std::hypot(dx, dy);
    double rot1, rot2;
    if (trans < 1e-4) {
      rot1 = 0.0;
      rot2 = dyaw;
    } else {
      rot1 = normalizeAngle(std::atan2(dy, dx) - prev_yaw);
      rot2 = normalizeAngle(dyaw - rot1);
    }

    predict(rot1, trans, rot2);
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
  // aprilTagCallback  —  EKF localization with AprilTag landmarks
  //
  // apriltag_detector_node publishes each visible tag's position in the
  // robot (base_link) frame as a Marker (marker.id = AprilTag id). For each:
  //   1. Known correspondence: match marker.id to a map landmark signature.
  //   2. Convert the body-frame position to an observation (range, bearing).
  //   3. Run correctLandmark() — one EKF correction per detected tag.
  // ----------------------------------------------------------
  void aprilTagCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    if (!initialized_) return;

    for (const auto& marker : msg->markers) {
      // Known correspondence: the AprilTag id IS the landmark signature.
      const Landmark* match = nullptr;
      for (const auto& lm : map_landmarks_) {
        if (lm.id == marker.id) { match = &lm; break; }
      }
      if (match == nullptr) continue;  // unknown tag — not in the map

      // Detection position in the robot body frame -> (range, bearing).
      const double mx      = marker.pose.position.x;
      const double my      = marker.pose.position.y;
      const double range   = std::hypot(mx, my);
      const double bearing = std::atan2(my, mx);

      if (range < 1e-3) continue;  // degenerate / on top of the robot

      correctLandmark(range, bearing, *match);
    }
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

  Eigen::Vector3d mu_;                      // State mean  [x, y, theta]
  Eigen::Matrix3d Sigma_;                   // State covariance
  Eigen::Matrix3d R_;                       // Process noise (predict step, Line 3)
  Eigen::Matrix<double, 1, 1> Q_imu_;      // IMU measurement noise     (correct step, Line 4)
  Eigen::Matrix2d Q_landmark_;             // Landmark [range, bearing] noise
  std::vector<Landmark> map_landmarks_;    // Known landmark positions in odom frame

  // Last raw /odom pose [x, y, theta]; the reference for the next motion delta.
  Eigen::Vector3d last_odom_ {Eigen::Vector3d::Zero()};
  bool initialized_ {false};

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu>   imu_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr apriltag_sub_;

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