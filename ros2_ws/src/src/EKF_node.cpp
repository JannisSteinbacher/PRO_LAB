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
//  KalmanFilterNode
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
    // ----------------------------------------------------------
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = 0.05;   
    Q_(1, 1) = 0.05;   
    Q_(2, 2) = 0.01;   

    // ----------------------------------------------------------
    // Measurement noise R
    // ----------------------------------------------------------
    R_odom_ = Eigen::Matrix3d::Zero();
    R_odom_(0, 0) = 0.1;   
    R_odom_(1, 1) = 0.1;   
    R_odom_(2, 2) = 0.05;  

    R_imu_(0, 0) = 0.02;   

    // ----------------------------------------------------------
    // Unabhängige Subscriber (Keine Synchronisation nötig)
    // ----------------------------------------------------------
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&KalmanFilterNode::scanCallback, this, std::placeholders::_1));

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&KalmanFilterNode::cmdVelCallback, this, std::placeholders::_1));

    // ----------------------------------------------------------
    // Synchronisierte Subscriber (Odom & IMU)
    // ----------------------------------------------------------
    
    // 1. Initialisiere die message_filters Subscriber
    // Sie fangen die Nachrichten ab und leiten sie an den Synchronizer weiter.
    odom_filter_.subscribe(this, "/odom");
    imu_filter_.subscribe(this, "/imu");

    // 2. Erstelle den Synchronizer mit unserer definierten Policy (siehe private Member)
    // Der erste Parameter (10) ist die Größe der Warteschlange (Queue Size).
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), odom_filter_, imu_filter_
    );

    // 3. Registriere den gemeinsamen Callback
    // Diese Funktion wird NUR aufgerufen, wenn Odom und IMU ungefähr den 
    // gleichen Zeitstempel haben.
    sync_->registerCallback(
      std::bind(&KalmanFilterNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // ----------------------------------------------------------
    // Publisher & Timer
    // ----------------------------------------------------------
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/ekf/pose_estimate", 10);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&KalmanFilterNode::publishEstimate, this));

    last_time_ = this->get_clock()->now();

    RCLCPP_INFO(get_logger(), "Kalman Filter node started (mit Synchronisierung).");
  }

private:
  // ============================================================
  //  Typendefinition für die Synchronisierung
  // ============================================================
  // Dies definiert die Regeln für den ApproximateTimeSynchronizer.
  // Er vergleicht die Zeitstempel von nav_msgs::msg::Odometry und sensor_msgs::msg::Imu.
  typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry, 
    sensor_msgs::msg::Imu
  > SyncPolicy;


  // ============================================================
  //  KF Core (Predict & Correct bleiben exakt gleich!)
  // ============================================================

  void predict(double dt)
  {
    const double theta = x_(2);

    x_(0) += v_ * std::cos(theta) * dt;
    x_(1) += v_ * std::sin(theta) * dt;
    x_(2) += omega_ * dt;
    x_(2)  = normalizeAngle(x_(2));

    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -v_ * std::sin(theta) * dt;
    F(1, 2) =  v_ * std::cos(theta) * dt;

    P_ = F * P_ * F.transpose() + Q_;
  }

  template<int M>
  void correct(const Eigen::Matrix<double, M, 1>& z,
               const Eigen::Matrix<double, M, 3>& H,
               const Eigen::Matrix<double, M, M>& R)
  {
    Eigen::Matrix<double, M, 1> y = z - H * x_;

    for (int i = 0; i < M; ++i) {
      if (std::abs(y(i)) > M_PI && std::abs(z(i)) < M_PI + 0.5) {
        y(i) = normalizeAngle(y(i));
      }
    }

    const Eigen::Matrix<double, M, M> S = H * P_ * H.transpose() + R;
    const Eigen::Matrix<double, 3, M> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    x_(2) = normalizeAngle(x_(2));

    const Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
  }

  // ============================================================
  //  Callbacks
  // ============================================================

  // --- NEUER GEMEINSAMER CALLBACK ---
  // Diese Funktion ersetzt die individuellen Aufrufe durch ROS.
  void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg,
                    const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
  {
    // Wir rufen hier einfach nacheinander deine bestehenden Logiken auf.
    // Da wir nun wissen, dass diese Nachrichten quasi zeitgleich entstanden sind,
    // explodiert die Kovarianz nicht mehr.
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

  // Dies war vorher "odomCallback" - umbenannt, da es nun aus syncCallback aufgerufen wird
  void processOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
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

    const Eigen::Vector3d z(msg->pose.pose.position.x,
                            msg->pose.pose.position.y,
                            yaw);

    const Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    correct<3>(z, H, R_odom_);
  }

  // Dies war vorher "imuCallback" - umbenannt, da es nun aus syncCallback aufgerufen wird
  void processImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
  {
    if (!initialized_) return;

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = quaternionToYaw(msg->orientation);

    Eigen::Matrix<double, 1, 3> H;
    H << 0.0, 0.0, 1.0;

    correct<1>(z, H, R_imu_);
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

    out.pose.pose.position.x = x_(0);
    out.pose.pose.position.y = x_(1);
    out.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x_(2));
    out.pose.pose.orientation = tf2::toMsg(q);

    out.pose.covariance.fill(0.0);
    out.pose.covariance[0]  = P_(0, 0);   
    out.pose.covariance[1]  = P_(0, 1);   
    out.pose.covariance[5]  = P_(0, 2);   
    out.pose.covariance[6]  = P_(1, 0);   
    out.pose.covariance[7]  = P_(1, 1);   
    out.pose.covariance[11] = P_(1, 2);   
    out.pose.covariance[30] = P_(2, 0);   
    out.pose.covariance[31] = P_(2, 1);   
    out.pose.covariance[35] = P_(2, 2);   

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

  Eigen::Vector3d x_;                       
  Eigen::Matrix3d P_;                       
  Eigen::Matrix3d Q_;                       
  Eigen::Matrix3d R_odom_;                  
  Eigen::Matrix<double, 1, 1> R_imu_;       

  double v_     {0.0};    
  double omega_ {0.0};    

  rclcpp::Time last_time_;
  bool initialized_ {false};

  // --- NEUE MEMBER FÜR MESSAGE FILTERS ---
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_filter_;
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Alte Standard-Subscriber bleiben für Scan und Cmd_Vel
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