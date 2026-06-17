// Node for Particle Filter implementation

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// ============================================================
//  Particle Filter Node
//
//  Implements the slide algorithm:
//    1. sample particles from the motion model
//    2. weight moved particles with a measurement model
//    3. resample particles proportional to their weights
//
//  Motion model:
//    Odometry delta with sampled Gaussian translation/rotation noise.
//
//  Measurement model:
//    IMU yaw likelihood. The /scan subscriber is kept for the map/laser
//    measurement model that the slides describe as the next step.
// ============================================================

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode()
  : Node("particle_filter_node"),
    rng_(std::random_device{}())
  {
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&ParticleFilterNode::odomCallback, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", 10,
      std::bind(&ParticleFilterNode::imuCallback, this, std::placeholders::_1));

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ParticleFilterNode::scanCallback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/pf/pose_estimate", 10);

    particle_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      "/particle_cloud", 10);

    RCLCPP_INFO(get_logger(), "Particle Filter node started with %zu particles.", num_particles_);
  }

private:
  struct Particle
  {
    double x {0.0};
    double y {0.0};
    double theta {0.0};
    double weight {1.0};
  };

  struct EigenLikeCovariance
  {
    double xx {0.0};
    double xy {0.0};
    double xt {0.0};
    double yy {0.0};
    double yt {0.0};
    double tt {0.0};
  };

  // ============================================================
  //  Particle Filter Core Algorithm
  // ============================================================

  void initializeParticles(double x, double y, double theta)
  {
    particles_.clear();
    particles_.reserve(num_particles_);

    for (size_t i = 0; i < num_particles_; ++i) {
      Particle p;
      p.x = x + sampleGaussian(init_position_stddev_);
      p.y = y + sampleGaussian(init_position_stddev_);
      p.theta = normalizeAngle(theta + sampleGaussian(init_yaw_stddev_));
      p.weight = 1.0 / static_cast<double>(num_particles_);
      particles_.push_back(p);
    }

    initialized_ = true;
  }

  void sampleMotionModel(double delta_rot1, double delta_trans, double delta_rot2)
  {
    for (auto& p : particles_) {
      const double trans_stddev = trans_noise_stddev_ * std::abs(delta_trans) + trans_noise_min_stddev_;
      const double rot1_stddev = rot_noise_stddev_ * std::abs(delta_rot1) + rot_noise_min_stddev_;
      const double rot2_stddev = rot_noise_stddev_ * std::abs(delta_rot2) + rot_noise_min_stddev_;

      const double delta_rot1_hat = delta_rot1 + sampleGaussian(rot1_stddev);
      const double delta_trans_hat = delta_trans + sampleGaussian(trans_stddev);
      const double delta_rot2_hat = delta_rot2 + sampleGaussian(rot2_stddev);

      p.x += delta_trans_hat * std::cos(p.theta + delta_rot1_hat);
      p.y += delta_trans_hat * std::sin(p.theta + delta_rot1_hat);
      p.theta = normalizeAngle(p.theta + delta_rot1_hat + delta_rot2_hat);
    }
  }

  void measurementModel()
  {
    if (!have_imu_) {
      setUniformWeights();
      return;
    }

    double weight_sum = 0.0;

    for (auto& p : particles_) {
      const double yaw_error = normalizeAngle(latest_imu_yaw_ - p.theta);
      p.weight = gaussianLikelihood(yaw_error, imu_yaw_stddev_);
      weight_sum += p.weight;
    }

    normalizeWeights(weight_sum);
  }

  void resample()
  {
    if (particles_.empty()) return;

    std::vector<Particle> resampled;
    resampled.reserve(num_particles_);

    const double step = 1.0 / static_cast<double>(num_particles_);
    std::uniform_real_distribution<double> uniform_dist(0.0, step);
    double target = uniform_dist(rng_);
    double cumulative_weight = particles_.front().weight;
    size_t index = 0;

    for (size_t i = 0; i < num_particles_; ++i) {
      while (target > cumulative_weight && index + 1 < particles_.size()) {
        ++index;
        cumulative_weight += particles_[index].weight;
      }

      Particle p = particles_[index];
      p.weight = 1.0 / static_cast<double>(num_particles_);
      resampled.push_back(p);
      target += step;
    }

    particles_ = std::move(resampled);
  }

  Particle estimatePose() const
  {
    Particle estimate;
    double sin_sum = 0.0;
    double cos_sum = 0.0;

    for (const auto& p : particles_) {
      estimate.x += p.weight * p.x;
      estimate.y += p.weight * p.y;
      sin_sum += p.weight * std::sin(p.theta);
      cos_sum += p.weight * std::cos(p.theta);
    }

    estimate.theta = std::atan2(sin_sum, cos_sum);
    estimate.weight = 1.0;
    return estimate;
  }

  EigenLikeCovariance estimateCovariance(const Particle& estimate) const
  {
    EigenLikeCovariance covariance;

    for (const auto& p : particles_) {
      const double dx = p.x - estimate.x;
      const double dy = p.y - estimate.y;
      const double dtheta = normalizeAngle(p.theta - estimate.theta);

      covariance.xx += p.weight * dx * dx;
      covariance.xy += p.weight * dx * dy;
      covariance.xt += p.weight * dx * dtheta;
      covariance.yy += p.weight * dy * dy;
      covariance.yt += p.weight * dy * dtheta;
      covariance.tt += p.weight * dtheta * dtheta;
    }

    return covariance;
  }

  // ============================================================
  //  Callbacks
  // ============================================================

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double odom_x = msg->pose.pose.position.x;
    const double odom_y = msg->pose.pose.position.y;
    const double odom_theta = quaternionToYaw(msg->pose.pose.orientation);

    if (!have_last_odom_) {
      last_odom_x_ = odom_x;
      last_odom_y_ = odom_y;
      last_odom_theta_ = odom_theta;
      have_last_odom_ = true;

      initializeParticles(odom_x, odom_y, odom_theta);
      publishEstimate();

      RCLCPP_INFO(get_logger(),
                  "PF initialised from odometry: x=%.3f  y=%.3f  theta=%.3f",
                  odom_x, odom_y, odom_theta);
      return;
    }

    const double dx = odom_x - last_odom_x_;
    const double dy = odom_y - last_odom_y_;
    const double delta_trans = std::hypot(dx, dy);
    const double delta_rot1 = normalizeAngle(std::atan2(dy, dx) - last_odom_theta_);
    const double delta_rot2 = normalizeAngle(odom_theta - last_odom_theta_ - delta_rot1);

    last_odom_x_ = odom_x;
    last_odom_y_ = odom_y;
    last_odom_theta_ = odom_theta;

    sampleMotionModel(delta_rot1, delta_trans, delta_rot2);
    measurementModel();
    resample();
    publishEstimate();
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    latest_imu_yaw_ = quaternionToYaw(msg->orientation);
    have_imu_ = true;
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    if (!initialized_) return;
    latest_scan_ = msg;
  }

  // ============================================================
  //  Publisher
  // ============================================================

  void publishEstimate()
  {
    if (!initialized_ || particles_.empty()) return;

    const Particle estimate = estimatePose();
    const EigenLikeCovariance covariance = estimateCovariance(estimate);

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = estimate.x;
    pose_msg.pose.pose.position.y = estimate.y;
    pose_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, estimate.theta);
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    pose_msg.pose.covariance.fill(0.0);
    pose_msg.pose.covariance[0] = covariance.xx;
    pose_msg.pose.covariance[1] = covariance.xy;
    pose_msg.pose.covariance[5] = covariance.xt;
    pose_msg.pose.covariance[6] = covariance.xy;
    pose_msg.pose.covariance[7] = covariance.yy;
    pose_msg.pose.covariance[11] = covariance.yt;
    pose_msg.pose.covariance[30] = covariance.xt;
    pose_msg.pose.covariance[31] = covariance.yt;
    pose_msg.pose.covariance[35] = covariance.tt;

    pose_pub_->publish(pose_msg);

    geometry_msgs::msg::PoseArray particle_msg;
    particle_msg.header = pose_msg.header;
    particle_msg.poses.reserve(particles_.size());

    for (const auto& p : particles_) {
      geometry_msgs::msg::Pose particle_pose;
      particle_pose.position.x = p.x;
      particle_pose.position.y = p.y;
      particle_pose.position.z = 0.0;

      tf2::Quaternion particle_q;
      particle_q.setRPY(0.0, 0.0, p.theta);
      particle_pose.orientation = tf2::toMsg(particle_q);

      particle_msg.poses.push_back(particle_pose);
    }

    particle_pub_->publish(particle_msg);
  }

  // ============================================================
  //  Helpers
  // ============================================================

  double sampleGaussian(double stddev)
  {
    if (stddev <= 0.0) return 0.0;
    std::normal_distribution<double> normal_dist(0.0, stddev);
    return normal_dist(rng_);
  }

  static double gaussianLikelihood(double error, double stddev)
  {
    const double variance = stddev * stddev;
    if (variance <= 0.0) return 0.0;
    return std::exp(-0.5 * error * error / variance);
  }

  void normalizeWeights(double weight_sum)
  {
    if (weight_sum <= 1e-12 || !std::isfinite(weight_sum)) {
      setUniformWeights();
      return;
    }

    for (auto& p : particles_) {
      p.weight /= weight_sum;
    }
  }

  void setUniformWeights()
  {
    if (particles_.empty()) return;

    const double uniform_weight = 1.0 / static_cast<double>(particles_.size());
    for (auto& p : particles_) {
      p.weight = uniform_weight;
    }
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

  const size_t num_particles_ {500};

  const double init_position_stddev_ {0.20};
  const double init_yaw_stddev_ {0.15};
  const double trans_noise_stddev_ {0.10};
  const double trans_noise_min_stddev_ {0.01};
  const double rot_noise_stddev_ {0.15};
  const double rot_noise_min_stddev_ {0.01};
  const double imu_yaw_stddev_ {0.08};

  std::vector<Particle> particles_;
  std::mt19937 rng_;

  bool initialized_ {false};
  bool have_last_odom_ {false};
  bool have_imu_ {false};

  double last_odom_x_ {0.0};
  double last_odom_y_ {0.0};
  double last_odom_theta_ {0.0};
  double latest_imu_yaw_ {0.0};

  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
};

// ============================================================
//  Entry point
// ============================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParticleFilterNode>());
  rclcpp::shutdown();
  return 0;
}
