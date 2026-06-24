#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// ============================================================
//  Monte Carlo Localization (Particle Filter) Node
//
//  Full MCL (Thrun, Probabilistic Robotics, Table 8.2):
//
//    MCL(X_{t-1}, u_t, z_t, m):
//      for m = 1..M:
//        x_t^[m] = sample_motion_model(u_t, x_{t-1}^[m])      <- PREDICT
//        w_t^[m] = measurement_model(z_t, x_t^[m], m)         <- CORRECT
//      X_t = low_variance_resample(X_t, w_t)                  <- RESAMPLE
//
//  1) PREDICT  — odometry motion model, sample_motion_model_odometry
//                (Table 5.6) driven by the pose delta from nav_msgs/Odometry
//                (/odom).  Uses the physical wheel odometry, not /cmd_vel.
//
//  2) CORRECT  — likelihood-field range-finder model (Table 6.3) using
//                sensor_msgs/LaserScan (/scan) against nav_msgs/OccupancyGrid
//                (/map).  A Euclidean distance transform of the map is
//                precomputed once so each beam endpoint is scored by its
//                distance to the nearest occupied cell.
//
//  3) RESAMPLE — normalize weights, compute the effective sample size
//                N_eff = 1 / Σ w², and systematically resample whenever
//                N_eff drops below half the particle count.  The surviving
//                cloud is published as geometry_msgs/PoseArray on
//                /particlecloud for RViz.
//
//  Updates (correct + resample) are gated on a minimum amount of motion
//  since the last update (AMCL style) to avoid particle depletion while the
//  robot is standing still.
// ============================================================

struct Particle {
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilterNode : public rclcpp::Node
{
public:
    ParticleFilterNode()
    : Node("particle_filter_node"), gen_(std::random_device{}())
    {
        // ---- Parameters (all tunable, sensible AMCL-like defaults) ----
        M_              = declare_parameter<int>("num_particles", 2000);

        // Odometry motion-noise parameters alpha_1..alpha_4 (Table 5.6)
        alpha1_         = declare_parameter<double>("alpha1", 0.2);  // rot   noise from rotation
        alpha2_         = declare_parameter<double>("alpha2", 0.2);  // rot   noise from translation
        alpha3_         = declare_parameter<double>("alpha3", 0.2);  // trans noise from translation
        alpha4_         = declare_parameter<double>("alpha4", 0.2);  // trans noise from rotation

        // Likelihood-field measurement-model parameters (Table 6.3)
        z_hit_          = declare_parameter<double>("z_hit", 0.95);
        z_rand_         = declare_parameter<double>("z_rand", 0.05);
        sigma_hit_      = declare_parameter<double>("sigma_hit", 0.2);   // [m]
        laser_max_dist_ = declare_parameter<double>("laser_likelihood_max_dist", 2.0); // [m]
        max_beams_      = declare_parameter<int>("max_beams", 60);

        // Update gating + resampling
        update_min_d_   = declare_parameter<double>("update_min_d", 0.20); // [m]
        update_min_a_   = declare_parameter<double>("update_min_a", 0.30); // [rad]
        resample_ratio_ = declare_parameter<double>("resample_neff_ratio", 0.5);

        // Occupancy threshold (grid value >= this == occupied, percent 0..100)
        occ_threshold_  = declare_parameter<int>("occupied_threshold", 50);

        // Fallback map bounds for the pre-map cloud (depot map extent;
        // superseded by the real grid bounds once /map is received).
        map_x_min_ = -7.14;  map_x_max_ = 23.06;
        map_y_min_ = -7.83;  map_y_max_ =  7.52;

        // Show an initial (uniform) cloud before the map arrives.
        initParticlesUniform();

        // ---- TF (to find the laser pose relative to the robot base) ----
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ---- Subscriptions ----
        // Map is latched -> transient_local durability is required to get it.
        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", map_qos,
            std::bind(&ParticleFilterNode::mapCallback, this, std::placeholders::_1));

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", rclcpp::QoS(50),
            std::bind(&ParticleFilterNode::odomCallback, this, std::placeholders::_1));

        // Scans are usually best-effort sensor data.
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&ParticleFilterNode::scanCallback, this, std::placeholders::_1));

        // ---- Publishers ----
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/pf/pose_estimate", 10);
        particles_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
            "/particlecloud", 10);

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ParticleFilterNode::publishEstimate, this));

        RCLCPP_INFO(get_logger(),
            "Particle Filter (MCL) started.  M=%d  waiting for /map, /odom, /scan ...", M_);
    }

private:
    // ============================================================
    //  Initialization
    // ============================================================

    // Uniform over the fallback bounding box (used before the map arrives).
    void initParticlesUniform()
    {
        particles_.resize(M_);
        std::uniform_real_distribution<double> dx(map_x_min_, map_x_max_);
        std::uniform_real_distribution<double> dy(map_y_min_, map_y_max_);
        std::uniform_real_distribution<double> dth(-M_PI, M_PI);
        for (auto& p : particles_) {
            p.x = dx(gen_);  p.y = dy(gen_);  p.theta = dth(gen_);
            p.weight = 1.0 / M_;
        }
    }

    // Uniform over the FREE cells of the real occupancy grid (global init).
    void initParticlesFreeSpace()
    {
        particles_.resize(M_);
        std::uniform_real_distribution<double> dth(-M_PI, M_PI);
        std::uniform_int_distribution<int> di(0, width_  - 1);
        std::uniform_int_distribution<int> dj(0, height_ - 1);

        int placed = 0;
        const long max_attempts = static_cast<long>(M_) * 1000;
        for (long a = 0; a < max_attempts && placed < M_; ++a) {
            const int i = di(gen_);
            const int j = dj(gen_);
            const int8_t v = occ_[static_cast<size_t>(j) * width_ + i];
            if (v < 0 || v >= occ_threshold_) continue;   // unknown or occupied
            particles_[placed].x     = origin_x_ + (i + 0.5) * resolution_;
            particles_[placed].y     = origin_y_ + (j + 0.5) * resolution_;
            particles_[placed].theta = dth(gen_);
            particles_[placed].weight = 1.0 / M_;
            ++placed;
        }
        // Degenerate fallback (map essentially has no free space).
        for (; placed < M_; ++placed) {
            std::uniform_real_distribution<double> dx(map_x_min_, map_x_max_);
            std::uniform_real_distribution<double> dy(map_y_min_, map_y_max_);
            particles_[placed].x = dx(gen_); particles_[placed].y = dy(gen_);
            particles_[placed].theta = dth(gen_); particles_[placed].weight = 1.0 / M_;
        }
        RCLCPP_INFO(get_logger(), "Initialized %d particles uniformly over map free space.", M_);
    }

    // ============================================================
    //  PREDICT — sample_motion_model_odometry (Thrun Table 5.6)
    //
    //  Applies the same control (delta of odometry pose) to every particle,
    //  each with independently sampled noise.
    // ============================================================
    void applyMotionModel(double d_rot1, double d_trans, double d_rot2)
    {
        // Backward-motion fix (as in AMCL): use the smaller of the rotation and
        // its pi-complement when scaling the rotation noise, so reversing does
        // not inject huge spurious heading noise.
        const double r1_noise = std::min(std::fabs(normalizeAngle(d_rot1)),
                                         std::fabs(normalizeAngle(d_rot1 - M_PI)));
        const double r2_noise = std::min(std::fabs(normalizeAngle(d_rot2)),
                                         std::fabs(normalizeAngle(d_rot2 - M_PI)));

        const double s_rot1  = std::sqrt(alpha1_ * r1_noise * r1_noise +
                                         alpha2_ * d_trans  * d_trans);
        const double s_trans = std::sqrt(alpha3_ * d_trans  * d_trans  +
                                         alpha4_ * r1_noise * r1_noise +
                                         alpha4_ * r2_noise * r2_noise);
        const double s_rot2  = std::sqrt(alpha1_ * r2_noise * r2_noise +
                                         alpha2_ * d_trans  * d_trans);

        for (auto& p : particles_) {
            const double dr1 = d_rot1  - sampleNormal(s_rot1);
            const double dt  = d_trans - sampleNormal(s_trans);
            const double dr2 = d_rot2  - sampleNormal(s_rot2);

            p.x    += dt * std::cos(p.theta + dr1);
            p.y    += dt * std::sin(p.theta + dr1);
            p.theta = normalizeAngle(p.theta + dr1 + dr2);
        }
    }

    // ============================================================
    //  CORRECT — likelihood-field model (Thrun Table 6.3)
    //
    //  For each particle, project a subsample of scan endpoints into the map
    //  using the particle's pose (+ the fixed laser mounting offset) and score
    //  each by a Gaussian over its distance to the nearest occupied cell.
    //  Accumulated in log-space for numerical stability, then normalized.
    // ============================================================
    void measurementUpdate(const sensor_msgs::msg::LaserScan::SharedPtr& scan)
    {
        const int n = static_cast<int>(scan->ranges.size());
        if (n == 0) return;

        const int step          = std::max(1, n / std::max(1, max_beams_));
        const double two_sigma2 = 2.0 * sigma_hit_ * sigma_hit_;
        const double z_rand_term = (scan->range_max > 0.0) ? z_rand_ / scan->range_max : 0.0;

        for (auto& p : particles_) {
            // Laser origin/orientation in the map frame for this particle.
            const double c = std::cos(p.theta), s = std::sin(p.theta);
            const double lx = p.x + laser_x_ * c - laser_y_ * s;
            const double ly = p.y + laser_x_ * s + laser_y_ * c;
            const double lth = p.theta + laser_yaw_;

            double log_w = 0.0;
            for (int k = 0; k < n; k += step) {
                const double z = scan->ranges[k];
                if (!std::isfinite(z) || z <= scan->range_min || z >= scan->range_max)
                    continue;   // skip invalid / no-return / max-range beams

                const double ang = lth + scan->angle_min + k * scan->angle_increment;
                const double ex  = lx + z * std::cos(ang);
                const double ey  = ly + z * std::sin(ang);

                const double d  = distanceToObstacle(ex, ey);   // [m], capped
                const double pz = z_hit_ * std::exp(-(d * d) / two_sigma2) + z_rand_term;
                log_w += std::log(pz);                          // pz > 0 always
            }
            p.weight = log_w;   // store log-weight temporarily
        }

        // Normalize from log-space via max-subtraction.
        double max_lw = -std::numeric_limits<double>::infinity();
        for (const auto& p : particles_) max_lw = std::max(max_lw, p.weight);

        double sum = 0.0;
        for (auto& p : particles_) { p.weight = std::exp(p.weight - max_lw); sum += p.weight; }

        if (sum > 0.0) for (auto& p : particles_) p.weight /= sum;
        else           for (auto& p : particles_) p.weight = 1.0 / M_;
    }

    // Distance [m] from a world point to the nearest occupied cell.
    double distanceToObstacle(double wx, double wy) const
    {
        const int i = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
        const int j = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
        if (i < 0 || i >= width_ || j < 0 || j >= height_)
            return laser_max_dist_;   // off-map -> treat as far (random floor)
        return dist_field_[static_cast<size_t>(j) * width_ + i];
    }

    // ============================================================
    //  RESAMPLE — systematic (low-variance) resampling, gated on N_eff.
    // ============================================================
    void maybeResample()
    {
        double sum_sq = 0.0;
        for (const auto& p : particles_) sum_sq += p.weight * p.weight;
        const double n_eff = (sum_sq > 0.0) ? 1.0 / sum_sq : 0.0;

        if (n_eff < resample_ratio_ * M_)
            particles_ = systematicResample(particles_);
    }

    std::vector<Particle> systematicResample(const std::vector<Particle>& weighted)
    {
        std::vector<double> cdf(M_);
        cdf[0] = weighted[0].weight;
        for (int i = 1; i < M_; ++i)
            cdf[i] = cdf[i - 1] + weighted[i].weight;

        std::uniform_real_distribution<double> ud(0.0, 1.0 / M_);
        const double u0 = ud(gen_);

        std::vector<Particle> resampled;
        resampled.reserve(M_);

        int j = 0;
        for (int m = 0; m < M_; ++m) {
            const double threshold = u0 + static_cast<double>(m) / M_;
            while (j < M_ - 1 && cdf[j] < threshold) ++j;
            Particle p = weighted[j];
            p.weight   = 1.0 / M_;
            resampled.push_back(p);
        }
        return resampled;
    }

    // ============================================================
    //  Callbacks
    // ============================================================

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        buildLikelihoodField(*msg);
        initParticlesFreeSpace();   // global init over real free space
        map_received_  = true;
        odom_ref_set_  = false;     // force re-sync of the odom reference pose
        RCLCPP_INFO(get_logger(),
            "Map received: %dx%d @ %.3f m/cell, origin (%.2f, %.2f). Likelihood field built.",
            width_, height_, resolution_, origin_x_, origin_y_);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        ox_  = msg->pose.pose.position.x;
        oy_  = msg->pose.pose.position.y;
        oth_ = yawFromQuat(msg->pose.pose.orientation);
        if (base_frame_.empty())
            base_frame_ = msg->child_frame_id.empty() ? "base_link" : msg->child_frame_id;
        have_odom_ = true;
    }

    // The "brain": predict from accumulated odometry, then correct + resample.
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!map_received_ || !have_odom_) return;
        if (!ensureLaserPose(msg->header.frame_id)) return;

        const double cx = ox_, cy = oy_, cth = oth_;   // latest odom snapshot

        // First scan after (re)init: just latch the odom reference, no update.
        if (!odom_ref_set_) {
            rx_ = cx; ry_ = cy; rth_ = cth;
            odom_ref_set_ = true;
            return;
        }

        // Odometry delta since the last filter update, as the (rot1, trans, rot2)
        // decomposition of sample_motion_model_odometry.
        const double dx = cx - rx_;
        const double dy = cy - ry_;
        const double d_trans = std::hypot(dx, dy);
        const double d_rot1  = (d_trans < 0.01) ? 0.0
                                                : normalizeAngle(std::atan2(dy, dx) - rth_);
        const double d_rot2  = normalizeAngle(cth - rth_ - d_rot1);

        // Gate: only update after enough motion (avoids particle depletion).
        if (d_trans < update_min_d_ &&
            std::fabs(normalizeAngle(cth - rth_)) < update_min_a_)
            return;

        applyMotionModel(d_rot1, d_trans, d_rot2);   // PREDICT
        rx_ = cx; ry_ = cy; rth_ = cth;              // advance reference

        measurementUpdate(msg);                      // CORRECT
        maybeResample();                             // RESAMPLE
    }

    // ============================================================
    //  Publisher — weighted-mean pose + particle cloud
    // ============================================================
    void publishEstimate()
    {
        if (particles_.empty()) return;

        double mx = 0.0, my = 0.0, mcos = 0.0, msin = 0.0, wsum = 0.0;
        for (const auto& p : particles_) {
            mx   += p.weight * p.x;
            my   += p.weight * p.y;
            mcos += p.weight * std::cos(p.theta);
            msin += p.weight * std::sin(p.theta);
            wsum += p.weight;
        }
        if (wsum <= 0.0) wsum = 1.0;
        mx /= wsum; my /= wsum;
        const double mth = std::atan2(msin, mcos);

        double vx = 0.0, vy = 0.0, vth = 0.0;
        for (const auto& p : particles_) {
            const double w = p.weight / wsum;
            vx  += w * (p.x - mx) * (p.x - mx);
            vy  += w * (p.y - my) * (p.y - my);
            const double dth = normalizeAngle(p.theta - mth);
            vth += w * dth * dth;
        }

        const auto stamp = this->get_clock()->now();

        geometry_msgs::msg::PoseWithCovarianceStamped out;
        out.header.stamp    = stamp;
        out.header.frame_id = "map";
        out.pose.pose.position.x = mx;
        out.pose.pose.position.y = my;
        out.pose.pose.position.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, mth);
        out.pose.pose.orientation = tf2::toMsg(q);
        out.pose.covariance.fill(0.0);
        out.pose.covariance[0]  = vx;
        out.pose.covariance[7]  = vy;
        out.pose.covariance[35] = vth;
        pose_pub_->publish(out);

        geometry_msgs::msg::PoseArray pa;
        pa.header = out.header;
        pa.poses.reserve(particles_.size());
        for (const auto& p : particles_) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = p.x;
            pose.position.y = p.y;
            pose.position.z = 0.0;
            tf2::Quaternion pq;
            pq.setRPY(0.0, 0.0, p.theta);
            pose.orientation = tf2::toMsg(pq);
            pa.poses.push_back(pose);
        }
        particles_pub_->publish(pa);
    }

    // ============================================================
    //  Map preprocessing — exact Euclidean distance transform
    //  (Felzenszwalb & Huttenlocher, 2-pass squared-distance EDT).
    // ============================================================
    void buildLikelihoodField(const nav_msgs::msg::OccupancyGrid& grid)
    {
        width_      = static_cast<int>(grid.info.width);
        height_     = static_cast<int>(grid.info.height);
        resolution_ = grid.info.resolution;
        origin_x_   = grid.info.origin.position.x;
        origin_y_   = grid.info.origin.position.y;
        occ_        = grid.data;

        // Update bounds to the real map extent.
        map_x_min_ = origin_x_;
        map_y_min_ = origin_y_;
        map_x_max_ = origin_x_ + width_  * resolution_;
        map_y_max_ = origin_y_ + height_ * resolution_;

        const size_t N = static_cast<size_t>(width_) * height_;
        const double BIG = 1e12;                 // sentinel for "no obstacle yet"
        std::vector<double> f(N);
        for (size_t idx = 0; idx < N; ++idx)
            f[idx] = (occ_[idx] >= occ_threshold_) ? 0.0 : BIG;   // obstacle=0, else inf

        std::vector<double> col(height_), row(width_);
        // Pass 1: transform along columns (y).
        for (int i = 0; i < width_; ++i) {
            for (int j = 0; j < height_; ++j) col[j] = f[static_cast<size_t>(j) * width_ + i];
            dt1d(col);
            for (int j = 0; j < height_; ++j) f[static_cast<size_t>(j) * width_ + i] = col[j];
        }
        // Pass 2: transform along rows (x).
        for (int j = 0; j < height_; ++j) {
            for (int i = 0; i < width_; ++i) row[i] = f[static_cast<size_t>(j) * width_ + i];
            dt1d(row);
            for (int i = 0; i < width_; ++i) f[static_cast<size_t>(j) * width_ + i] = row[i];
        }

        // f now holds squared distance in cells^2 -> meters, capped.
        dist_field_.resize(N);
        for (size_t idx = 0; idx < N; ++idx) {
            const double d_m = std::sqrt(f[idx]) * resolution_;
            dist_field_[idx] = static_cast<float>(std::min(d_m, laser_max_dist_));
        }
    }

    // 1-D squared distance transform of a sampled function (in place).
    static void dt1d(std::vector<double>& f)
    {
        const int n = static_cast<int>(f.size());
        const double INF = std::numeric_limits<double>::infinity();

        std::vector<double> d(n);
        std::vector<int>    v(n);
        std::vector<double> z(n + 1);

        int k = 0;
        v[0] = 0;
        z[0] = -INF;
        z[1] =  INF;
        for (int q = 1; q < n; ++q) {
            double s = ((f[q] + static_cast<double>(q) * q) -
                        (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
                       (2.0 * q - 2.0 * v[k]);
            while (s <= z[k]) {
                --k;
                s = ((f[q] + static_cast<double>(q) * q) -
                     (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
                    (2.0 * q - 2.0 * v[k]);
            }
            ++k;
            v[k]     = q;
            z[k]     = s;
            z[k + 1] = INF;
        }
        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[k + 1] < q) ++k;
            const double dq = static_cast<double>(q - v[k]);
            d[q] = dq * dq + f[v[k]];
        }
        f = d;
    }

    // ============================================================
    //  Helpers
    // ============================================================

    // Look up (and cache) the laser pose relative to the robot base via TF.
    bool ensureLaserPose(const std::string& laser_frame)
    {
        if (have_laser_pose_) return true;
        if (base_frame_.empty()) return false;
        try {
            const auto tf = tf_buffer_->lookupTransform(
                base_frame_, laser_frame, tf2::TimePointZero);
            laser_x_   = tf.transform.translation.x;
            laser_y_   = tf.transform.translation.y;
            laser_yaw_ = yawFromQuat(tf.transform.rotation);
            have_laser_pose_ = true;
            RCLCPP_INFO(get_logger(),
                "Laser pose %s->%s = (%.3f, %.3f, %.3f rad).",
                base_frame_.c_str(), laser_frame.c_str(), laser_x_, laser_y_, laser_yaw_);
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for TF %s -> %s: %s",
                base_frame_.c_str(), laser_frame.c_str(), e.what());
            return false;
        }
    }

    double sampleNormal(double sigma)
    {
        if (sigma <= 0.0) return 0.0;
        std::normal_distribution<double> nd(0.0, sigma);
        return nd(gen_);
    }

    static double yawFromQuat(const geometry_msgs::msg::Quaternion& q)
    {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z));
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
    int    M_;
    double alpha1_, alpha2_, alpha3_, alpha4_;
    double z_hit_, z_rand_, sigma_hit_, laser_max_dist_;
    int    max_beams_;
    double update_min_d_, update_min_a_, resample_ratio_;
    int    occ_threshold_;

    std::vector<Particle> particles_;        // current particle set X_t

    // Map / likelihood field
    bool   map_received_ {false};
    int    width_  {0}, height_ {0};
    double resolution_ {0.05};
    double origin_x_ {0.0}, origin_y_ {0.0};
    std::vector<int8_t> occ_;                // raw occupancy (for free-space init)
    std::vector<float>  dist_field_;         // distance [m] to nearest obstacle
    double map_x_min_, map_x_max_, map_y_min_, map_y_max_;

    // Odometry tracking
    bool   have_odom_ {false};
    bool   odom_ref_set_ {false};
    double ox_ {0.0}, oy_ {0.0}, oth_ {0.0};   // latest odom pose
    double rx_ {0.0}, ry_ {0.0}, rth_ {0.0};   // odom pose at last filter update
    std::string base_frame_;                   // odom child frame (robot base)

    // Laser mounting offset (base_frame -> laser_frame)
    bool   have_laser_pose_ {false};
    double laser_x_ {0.0}, laser_y_ {0.0}, laser_yaw_ {0.0};

    std::mt19937 gen_;

    std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr  scan_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr                 particles_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

// ============================================================
//  Entry point
// ============================================================
int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilterNode>());
    rclcpp::shutdown();
    return 0;
}
