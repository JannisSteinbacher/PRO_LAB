#include <cmath>
#include <random>
#include <vector>
#include <array>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// ============================================================
//  Monte Carlo Localization (Particle Filter) Node
//
//  Implements Algorithm MCL (Thrun 2006, slide 22):
//
//  MCL(X_{t-1}, u_t, z_t, m):
//    Line 3-7:  for each particle m:
//                 x_t^[m] = sample_motion_model(u_t, x_{t-1}^[m])   <- predict
//    Line 8-11: resample M particles with probability ∝ w_t^[i]      <- resample
//
//  Motion Model — sample_motion_model_velocity (slide 19, 04 PDF):
//    v_hat     = v     + N(0, (alpha1*|v| + alpha2*|w|)^2 )
//    w_hat     = omega + N(0, (alpha3*|v| + alpha4*|w|)^2 )
//    gamma_hat =         N(0, (alpha5*|v| + alpha6*|w|)^2 )
//    x'     = x - (v_hat/w_hat)*sin(theta) + (v_hat/w_hat)*sin(theta + w_hat*dt)
//    y'     = y + (v_hat/w_hat)*cos(theta) - (v_hat/w_hat)*cos(theta + w_hat*dt)
//    theta' = theta + w_hat*dt + gamma_hat*dt
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
        // Number of particles
        M_ = 500;

        // Map bounds for uniform random initialization over the whole map
        map_x_min_ = -3.0;
        map_x_max_ =  3.0;
        map_y_min_ = -3.0;
        map_y_max_ =  3.0;

        // Motion noise parameters alpha_1 .. alpha_6 (slide 19, 04 PDF)
        // sigma_v     = alpha[0]*|v| + alpha[1]*|omega|
        // sigma_omega = alpha[2]*|v| + alpha[3]*|omega|
        // sigma_gamma = alpha[4]*|v| + alpha[5]*|omega|
        alpha_ = {0.1, 0.05, 0.05, 0.1, 0.02, 0.02};

        // Initialize: uniform distribution over whole map (slide 07, 05 PDF)
        initParticles();

        // Subscriptions
        cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&ParticleFilterNode::cmdVelCallback, this, std::placeholders::_1));

        // Publishers
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/pf/pose_estimate", 10);

        particles_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
            "/pf/particles", 10);

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ParticleFilterNode::publishEstimate, this));

        last_time_ = this->get_clock()->now();

        RCLCPP_INFO(get_logger(),
            "Particle Filter node started.  M=%d  map=[%.1f,%.1f]x[%.1f,%.1f]",
            M_, map_x_min_, map_x_max_, map_y_min_, map_y_max_);
    }

private:

    // ============================================================
    //  Initialization — uniform random distribution over the whole map.
    //  (slide 07, 05 PDF: "hundreds of particles placed randomly in the map")
    // ============================================================
    void initParticles()
    {
        particles_.resize(M_);
        std::uniform_real_distribution<double> dx(map_x_min_, map_x_max_);
        std::uniform_real_distribution<double> dy(map_y_min_, map_y_max_);
        std::uniform_real_distribution<double> dth(-M_PI, M_PI);

        for (auto& p : particles_) {
            p.x      = dx(gen_);
            p.y      = dy(gen_);
            p.theta  = dth(gen_);
            p.weight = 1.0 / M_;
        }
    }

    // ============================================================
    //  Motion Model — sample_motion_model_velocity (slide 19, 04 PDF)
    //
    //  Lines 2-3: Add Gaussian noise to v and omega at the source
    //             (wheels), propagated through the kinematic model.
    //  Line  4:   Additional heading drift noise gamma.
    //  Lines 5-7: Apply ICC-based kinematic model with noisy inputs.
    // ============================================================
    Particle sampleMotionModel(const Particle& prev,
                               double v, double omega, double dt) const
    {
        // Noise standard deviations (lines 2-4 of algorithm, slide 19)
        const double sv = alpha_[0] * std::abs(v) + alpha_[1] * std::abs(omega);
        const double sw = alpha_[2] * std::abs(v) + alpha_[3] * std::abs(omega);
        const double sg = alpha_[4] * std::abs(v) + alpha_[5] * std::abs(omega);

        // Guard: always give a minimum spread so particles don't freeze
        auto noise = [&](double sigma) -> double {
            std::normal_distribution<double> nd(0.0, std::max(sigma, 1e-9));
            return nd(gen_);
        };

        // Line 2: v_hat     = v     + sample(sv)
        // Line 3: omega_hat = omega + sample(sw)
        // Line 4: gamma_hat =         sample(sg)
        const double v_hat     = v     + noise(sv);
        const double omega_hat = omega + noise(sw);
        const double gamma_hat =         noise(sg);

        Particle next;
        next.weight = prev.weight;

        if (std::abs(omega_hat) > 1e-6) {
            // Lines 5-7: general curved-path model (ICC / circular arc)
            const double r = v_hat / omega_hat;
            next.x     = prev.x - r * std::sin(prev.theta)
                                 + r * std::sin(prev.theta + omega_hat * dt);
            next.y     = prev.y + r * std::cos(prev.theta)
                                 - r * std::cos(prev.theta + omega_hat * dt);
            next.theta = prev.theta + omega_hat * dt + gamma_hat * dt;
        } else {
            // Degenerate case: straight-line motion (omega ≈ 0)
            next.x     = prev.x + v_hat * std::cos(prev.theta) * dt;
            next.y     = prev.y + v_hat * std::sin(prev.theta) * dt;
            next.theta = prev.theta + gamma_hat * dt;
        }

        next.theta = normalizeAngle(next.theta);
        return next;
    }

    // ============================================================
    //  Systematic resampling — Lines 8-11 of MCL algorithm (slide 22, 05 PDF)
    //
    //  Draws M particles from the weighted set with probability ∝ weight.
    //  Systematic resampling has lower variance than multinomial and O(M) cost.
    // ============================================================
    std::vector<Particle> systematicResample(const std::vector<Particle>& weighted)
    {
        // Build cumulative weight vector
        std::vector<double> cdf(M_);
        cdf[0] = weighted[0].weight;
        for (int i = 1; i < M_; ++i)
            cdf[i] = cdf[i-1] + weighted[i].weight;

        // Single random offset u ~ U[0, 1/M]
        std::uniform_real_distribution<double> ud(0.0, 1.0 / M_);
        const double u0 = ud(gen_);

        std::vector<Particle> resampled;
        resampled.reserve(M_);

        int j = 0;
        for (int m = 0; m < M_; ++m) {
            const double threshold = u0 + static_cast<double>(m) / M_;
            while (j < M_ - 1 && cdf[j] < threshold) ++j;
            Particle p  = weighted[j];
            p.weight    = 1.0 / M_;           // reset weight after resampling
            resampled.push_back(p);
        }

        return resampled;
    }

    // ============================================================
    //  Callbacks
    // ============================================================

    // MCL Line 4: sample_motion_model — called on each cmd_vel
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        v_     = msg->linear.x;
        omega_ = msg->angular.z;

        const auto now = this->get_clock()->now();
        const double dt = (now - last_time_).seconds();
        last_time_ = now;

        if (dt <= 0.0 || dt >= 1.0) return;

        // Line 3-4 of MCL (slide 22): propagate each particle through motion model
        for (auto& p : particles_)
            p = sampleMotionModel(p, v_, omega_, dt);
    }

    // ============================================================
    //  Publisher — weighted mean pose + particle cloud
    // ============================================================
    void publishEstimate()
    {
        // Weighted mean position (after resampling all weights equal 1/M,
        // so this degenerates to the arithmetic mean)
        double mx = 0.0, my = 0.0, mcos = 0.0, msin = 0.0;
        for (const auto& p : particles_) {
            mx   += p.weight * p.x;
            my   += p.weight * p.y;
            mcos += p.weight * std::cos(p.theta);
            msin += p.weight * std::sin(p.theta);
        }
        const double mth = std::atan2(msin, mcos);

        // Variance from particle spread (used as covariance diagonal)
        double vx = 0.0, vy = 0.0, vth = 0.0;
        for (const auto& p : particles_) {
            vx  += p.weight * (p.x - mx) * (p.x - mx);
            vy  += p.weight * (p.y - my) * (p.y - my);
            const double dth = normalizeAngle(p.theta - mth);
            vth += p.weight * dth * dth;
        }

        // PoseWithCovarianceStamped
        geometry_msgs::msg::PoseWithCovarianceStamped out;
        out.header.stamp    = this->get_clock()->now();
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

        // PoseArray for RViz particle visualization
        geometry_msgs::msg::PoseArray pa;
        pa.header = out.header;
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
    //  Helpers
    // ============================================================

    static double normalizeAngle(double a)
    {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    // ============================================================
    //  Members
    // ============================================================
    int M_;
    std::vector<Particle> particles_;      // current particle set X_t

    double map_x_min_, map_x_max_;         // map bounds for uniform init
    double map_y_min_, map_y_max_;

    std::array<double, 6> alpha_;          // motion noise params alpha_1..alpha_6

    double v_     {0.0};
    double omega_ {0.0};
    rclcpp::Time last_time_;

    mutable std::mt19937 gen_;             // mutable: used inside const helpers

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   cmd_vel_sub_;

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
