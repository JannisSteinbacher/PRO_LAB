AUTHOR: Jannis Steinbacher
Code implementation for Probabilistic Lab

Specific user Task: Disable or modify resampling and analyze particle degeneration.


# turtlebot_state_estimation

Probabilistic state estimation (KF, EKF, PF) for TurtleBot4 in Gazebo simulation.

## Demonstration
Click to go to youtube video

[![Watch the demonstration](https://img.youtube.com/vi/0IarFzx-WKA/maxresdefault.jpg)](https://youtu.be/0IarFzx-WKA)


---

## Overview

This package implements and compares three probabilistic localization filters — a **linear Kalman Filter (KF)**, an **Extended Kalman Filter (EKF)** with AprilTag landmark corrections, and a **Monte Carlo Particle Filter (PF)** — running simultaneously on a TurtleBot4 navigating a depot environment. An evaluation node records each filter's trajectory and covariance against Gazebo ground truth and saves plots and CSVs on shutdown.

---

## Prerequisites

- ROS 2 (Humble or later)
- Nav2 (`nav2_bringup`, `nav2_simple_commander`)
- Gazebo with TurtleBot4 simulation (`turtlebot4_gazebo`)
- `ros_gz_bridge`
- Python packages: `opencv-python`, `cv_bridge`, `matplotlib`, `numpy`
- C++ dependencies: Eigen3, `message_filters`, `tf2`

Build the package from the workspace root:

```bash
cd ~/Documents/MRE/S2/PRO_LAB/ros2_ws
colcon build --packages-select turtlebot_state_estimation
source install/setup.bash
```

---

## Quick Start — Running the Full Stack

A single launch file starts everything: Gazebo, Nav2, RViz, all three filters, the AprilTag detector, the ground-truth bridge, the evaluation node, and the patrol node.

```bash
ros2 launch turtlebot_state_estimation filter_launch.py
```

The robot will wait for Nav2 to become active, then autonomously drive a 4-waypoint patrol route. When the patrol completes, the entire stack shuts down and the evaluation outputs are written automatically.

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `eval` | `true` | Write evaluation PNG and CSV files on shutdown. Set `false` for quick testing without saving outputs. |

Example — run without saving evaluation files:

```bash
ros2 launch turtlebot_state_estimation filter_launch.py eval:=false
```

### Evaluation outputs

Written to `/PRO_LAB/ros2_ws/eval_results/` on shutdown:

| File | Contents |
|------|----------|
| `filter_eval_<timestamp>.png` | 4-panel plot: trajectory, Euclidean position error, position covariance (σ²_x, σ²_y), yaw covariance (σ²_θ) |
| `csv/kf_eval_<timestamp>.csv` | KF time series (time, estimated pose, ground truth, covariance, Kalman gain) |
| `csv/ekf_eval_<timestamp>.csv` | EKF time series (same columns) |
| `csv/pf_eval_<timestamp>.csv` | PF time series (K_theta column is NaN — PF has no Kalman gain) |

All three CSVs are resampled onto a shared 20 Hz time grid so their rows align 1-to-1. Angles are exported in degrees; position values in metres.

---

## Tuning Filter Parameters

All noise parameters are in [config/filter_params.yaml](config/filter_params.yaml). Edit this file and relaunch — runtime `ros2 param set` has no effect (parameters are read once at node startup).

| Parameter | Affects | Meaning |
|-----------|---------|---------|
| `R_diag` | KF, EKF | Process noise diagonal `[σ²_x, σ²_y, σ²_θ]` (predict step) |
| `Q_imu` | KF, EKF | IMU yaw measurement noise variance (correct step) |
| `Q_landmark_diag` | EKF | Landmark measurement noise `[σ²_range, σ²_bearing]` |
| `landmark_ids/world_x/world_y` | EKF | AprilTag map (must match `depot_apriltags.sdf`) |
| `map_origin` | EKF | Robot spawn pose in the world frame `[x, y, yaw]` |
| `num_particles` | PF | Number of particles (more = better coverage, higher CPU) |
| `weight_lambda` | PF | Likelihood tempering exponent (lower = flatter weights, more hypotheses survive) |
| `roughen_xy` / `roughen_yaw` | PF | Post-resample Gaussian jitter to maintain particle diversity |
| `resample_neff_ratio` | PF | Resample when N_eff < ratio × N (lower = less frequent resampling) |

---

## Node Reference

### `kf_node` — Linear Kalman Filter

**Purpose:** Tracks the robot pose `[x, y, θ]` using a linear motion model driven by odometry increments, corrected by IMU yaw.

**Subscribes to:**

| Topic | Type | Role |
|-------|------|------|
| `/odom` | `nav_msgs/Odometry` | Predict step — world-frame pose increment drives `mu_bar` update |
| `/imu` | `sensor_msgs/Imu` | Correct step — absolute yaw measurement fuses into the estimate |

> `/odom` and `/imu` are time-synchronized (ApproximateTime policy).

**Publishes:**

| Topic | Type | Description |
|-------|------|-------------|
| `/kf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | Estimated pose with 3×3 covariance, 10 Hz |
| `/kf/kalman_gain_theta` | `std_msgs/Float64` | Scalar Kalman gain applied to θ from the IMU correction (for evaluation) |

---

### `ekf_node` — Extended Kalman Filter

**Purpose:** Tracks the robot pose `[x, y, θ]` using Thrun's nonlinear odometry motion model (rot1/trans/rot2 decomposition). Corrected by IMU yaw and, when visible, by AprilTag landmark observations (range + bearing). The nonlinear measurement function and its Jacobian H_t are used for landmark updates; the IMU correction is identical in structure to the linear KF.

**Subscribes to:**

| Topic | Type | Role |
|-------|------|------|
| `/odom` | `nav_msgs/Odometry` | Predict step — odometry increment decomposed into rot1/trans/rot2 |
| `/imu` | `sensor_msgs/Imu` | Correct step — absolute yaw measurement |
| `/apriltag/detections` | `visualization_msgs/MarkerArray` | Correct step — per-tag range/bearing landmark corrections |

> `/odom` and `/imu` are time-synchronized; `/apriltag/detections` is processed independently.

**Publishes:**

| Topic | Type | Description |
|-------|------|-------------|
| `/ekf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | Estimated pose with covariance, 10 Hz |
| `/ekf/kalman_gain_theta` | `std_msgs/Float64` | Scalar Kalman gain applied to θ from the IMU correction (for evaluation) |

---

### `pf_node` — Particle Filter (Monte Carlo Localization)

**Purpose:** Global localization using a particle cloud (MCL, Thrun Table 8.2). Predicts particle poses via the stochastic odometry motion model, weights them against laser scan observations using a likelihood-field model, and resamples when the effective sample size drops below a threshold. Requires a map from Nav2 SLAM/AMCL.

**Subscribes to:**

| Topic | Type | Role |
|-------|------|------|
| `/map` | `nav_msgs/OccupancyGrid` | Builds a Euclidean distance transform (likelihood field) once on receipt |
| `/odom` | `nav_msgs/Odometry` | Predict step — odometry delta propagated through all particles |
| `/scan` | `sensor_msgs/LaserScan` | Correct step — beam endpoints scored against the likelihood field |

**Publishes:**

| Topic | Type | Description |
|-------|------|-------------|
| `/pf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | Weighted-mean pose with particle-cloud variance, 10 Hz |
| `/particlecloud` | `geometry_msgs/PoseArray` | Full particle set for visualization in RViz |

---

### `apriltag_detector_node` — AprilTag Detector

**Purpose:** Detects AprilTag 36h11 family tags in the TurtleBot4 RGB-D camera feed. For each detected tag it estimates its 3D position in the camera optical frame, transforms it into `base_link`, and publishes it as a `Marker`. The EKF node consumes these to derive (range, bearing) observations.

**Subscribes to:**

| Topic | Type | Role |
|-------|------|------|
| `/rgbd_camera/image` | `sensor_msgs/Image` | Camera image feed |
| `/rgbd_camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (used once on first message) |

**Publishes:**

| Topic | Type | Description |
|-------|------|-------------|
| `/apriltag/detections` | `visualization_msgs/MarkerArray` | One `Marker` per visible tag; `marker.id` = AprilTag ID; `marker.pose.position` = tag position in `base_link` frame [m] |

---

### `patrol_node` — Autonomous Patrol

**Purpose:** Drives the TurtleBot4 through a fixed 4-waypoint route using the Nav2 `BasicNavigator` API. Waits for Nav2 to become active before sending goals. Publishes a start signal to the eval node when navigation begins. When the final waypoint is reached the process exits, which triggers the launch file to shut down the entire stack (causing the eval node to write its outputs).

**Subscribes to:** *(none directly — uses the Nav2 action interface internally)*

**Publishes:**

| Topic | Type | Description |
|-------|------|-------------|
| `/eval/start` | `std_msgs/Empty` | Latched signal sent once when the patrol route begins; triggers evaluation recording |

**Waypoints (map frame):**

| # | x [m] | y [m] | yaw [°] |
|---|-------|-------|---------|
| 1 | 5.20 | 3.58 | 0 |
| 2 | 8.25 | 5.17 | 180 |
| 3 | −6.11 | 2.63 | 180 |
| 4 | 0.00 | 0.00 | 0 |

---

### `eval_node` — Evaluation

**Purpose:** Subscribes to all three filter estimates and Gazebo ground truth simultaneously. Buffers the time series and, on shutdown (Ctrl+C or end of patrol), writes synchronized CSVs and a 4-panel evaluation PNG comparing trajectories, position error, and covariance evolution.

**Subscribes to:**

| Topic | Type | Role |
|-------|------|------|
| `/kf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | KF estimate |
| `/ekf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | EKF estimate |
| `/pf/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | PF estimate |
| `/kf/kalman_gain_theta` | `std_msgs/Float64` | KF theta gain (logged to CSV) |
| `/ekf/kalman_gain_theta` | `std_msgs/Float64` | EKF theta gain (logged to CSV) |
| `/world/depot/dynamic_pose/info` | `geometry_msgs/PoseArray` | Gazebo ground truth (bridged by `gt_pose_bridge`) |
| `/eval/start` | `std_msgs/Empty` | Latched start signal from `patrol_node`; recording begins on receipt |

**Publishes:** *(none — output is files on shutdown)*

---

## Topic Graph Summary

```
/odom ──────────────┬──► kf_node  ──► /kf/pose_estimate
/imu ───────────────┤    (linear)      /kf/kalman_gain_theta
                    │
                    └──► ekf_node ──► /ekf/pose_estimate
                              ▲        /ekf/kalman_gain_theta
/apriltag/detections ─────────┘

/rgbd_camera/image ──┐
/rgbd_camera/info ───┴► apriltag_detector_node ──► /apriltag/detections

/map ────────────────┐
/odom ───────────────┼──► pf_node  ──► /pf/pose_estimate
/scan ───────────────┘                 /particlecloud

/kf/pose_estimate ──────┐
/ekf/pose_estimate ─────┤
/pf/pose_estimate ──────┼──► eval_node ──► (PNG + CSV on shutdown)
/kf/kalman_gain_theta ──┤
/ekf/kalman_gain_theta ─┤
/world/depot/...info ───┘

Nav2 ◄──── patrol_node ────► /eval/start
```
