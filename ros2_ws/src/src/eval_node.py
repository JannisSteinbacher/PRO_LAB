#!/usr/bin/env python3
"""
eval_node.py — Real-time evaluation of KF and EKF filter performance.

Subscribes to:
  /kf/pose_estimate              (PoseWithCovarianceStamped)
  /ekf/pose_estimate             (PoseWithCovarianceStamped)
  <ground_truth_topic param>     (nav_msgs/Odometry) — Gazebo ground truth

Default ground truth topic: /model/turtlebot4/odometry
  -> If that topic doesn't exist in your sim, check available topics with:
       ros2 topic list | grep -i "odom\|truth\|pose"
  -> Override at runtime:
       ros2 run turtlebot_state_estimation eval_node \
         --ros-args -p ground_truth_topic:=/your/topic

On shutdown (Ctrl+C): saves a 4-panel PNG to /tmp/filter_eval_<timestamp>.png
  Panel 1  — Trajectory: ground truth vs KF vs EKF
  Panel 2  — Euclidean position error over time
  Panel 3  — Position variance (σ_x², σ_y²) over time
  Panel 4  — Yaw variance (σ_θ²) over time
"""

import os
import signal
from datetime import datetime
from math import atan2

import matplotlib
matplotlib.use('Agg')  # file-only backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def _quat_to_yaw(q) -> float:
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EvalNode(Node):
    def __init__(self):
        super().__init__('eval_node')

        self.declare_parameter('ground_truth_topic', '/world/depot/dynamic_pose/info')
        # Index of turtlebot4 in the Pose_V array (confirmed via gz topic echo).
        self.declare_parameter('ground_truth_index', 1)
        # Set false to collect data but skip writing the PNG on shutdown.
        self.declare_parameter('save_plots', True)
        gt_topic = (self.get_parameter('ground_truth_topic')
                    .get_parameter_value().string_value)
        self._gt_index = (self.get_parameter('ground_truth_index')
                          .get_parameter_value().integer_value)
        self._save_plots = (self.get_parameter('save_plots')
                            .get_parameter_value().bool_value)

        # Each list stores tuples; converted to np.array on shutdown.
        # GT:      (t, x, y, yaw)
        # KF/EKF: (t, x, y, yaw, var_x, var_y, var_yaw)
        self._gt: list  = []
        self._kf: list  = []
        self._ekf: list = []
        self._t0: float | None = None

        # ros_gz_bridge publishes with BEST_EFFORT — use matching QoS.
        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseArray, gt_topic, self._gt_cb, gz_qos)
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/kf/pose_estimate', self._kf_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/ekf/pose_estimate', self._ekf_cb, 10)

        self.get_logger().info(f'EvalNode ready.  Ground truth: {gt_topic} '
                               f'(pose index: {self._gt_index})')
        self.get_logger().info('Press Ctrl+C to stop and save plots.')

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        # Use node receive time (wall clock) for all messages so that
        # gz sim-time stamps and ROS wall-clock stamps stay on the same axis.
        t = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = t
        return t - self._t0

    # ------------------------------------------------------------------ #
    #  Subscribers                                                         #
    # ------------------------------------------------------------------ #

    def _gt_cb(self, msg: PoseArray):
        # Pose_V bridged as PoseArray; turtlebot4 is at index 1 (verified via
        # gz topic echo — Depot@0, turtlebot4@1, main@2, ...).
        if len(msg.poses) <= self._gt_index:
            return
        t = self._now()
        p = msg.poses[self._gt_index]
        self._gt.append((t, p.position.x, p.position.y,
                         _quat_to_yaw(p.orientation)))

    def _filter_cb(self, msg: PoseWithCovarianceStamped, store: list):
        t = self._now()
        p = msg.pose.pose
        c = msg.pose.covariance  # row-major 6×6; indices [0,7,35] → x,y,θ
        store.append((t, p.position.x, p.position.y,
                      _quat_to_yaw(p.orientation),
                      c[0], c[7], c[35]))

    def _kf_cb(self, msg: PoseWithCovarianceStamped):
        self._filter_cb(msg, self._kf)

    def _ekf_cb(self, msg: PoseWithCovarianceStamped):
        self._filter_cb(msg, self._ekf)

    # ------------------------------------------------------------------ #
    #  Plot generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_plots(self):
        if not self._save_plots:
            self.get_logger().info(
                'save_plots=false — skipping evaluation PNG.')
            return

        gt  = np.array(self._gt)  if self._gt  else None
        kf  = np.array(self._kf)  if self._kf  else None
        ekf = np.array(self._ekf) if self._ekf else None

        # Align ground truth to the odom frame origin (filters start at 0,0;
        # GT is in Gazebo world frame where the robot spawns at e.g. (-8, 0)).
        if gt is not None:
            gt[:, 1] -= gt[0, 1]
            gt[:, 2] -= gt[0, 2]

        if gt is None:
            self.get_logger().warn(
                'No ground-truth data received — cannot compute errors. '
                'Check that the ground_truth_topic parameter is correct.')
        if kf is None and ekf is None:
            self.get_logger().warn('No filter data received. Nothing to plot.')
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Filter Performance Evaluation', fontsize=14, fontweight='bold')

        self._plot_trajectory(axes[0, 0], gt, kf, ekf)
        self._plot_error(axes[0, 1], gt, kf, ekf)
        self._plot_pos_variance(axes[1, 0], kf, ekf)
        self._plot_yaw_variance(axes[1, 1], kf, ekf)

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.expanduser('~/Documents/MRE/PRO_LAB/ros2_ws/eval_results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'filter_eval_{timestamp}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f'Saved → {out_path}')

    # ---- individual panels ------------------------------------------- #

    @staticmethod
    def _plot_trajectory(ax, gt, kf, ekf):
        if gt is not None:
            ax.plot(gt[:, 1], gt[:, 2], 'k-',
                    label='Ground Truth', linewidth=2.0, zorder=3)
        if kf is not None:
            ax.plot(kf[:, 1], kf[:, 2], 'b--',
                    label='KF', linewidth=1.4)
        if ekf is not None:
            ax.plot(ekf[:, 1], ekf[:, 2], 'r--',
                    label='EKF', linewidth=1.4)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Trajectory')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend()
        ax.grid(True)

    @staticmethod
    def _plot_error(ax, gt, kf, ekf):
        if gt is None:
            ax.text(0.5, 0.5, 'No ground truth available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Error')
            return

        def euclidean_error(filt):
            gt_x = np.interp(filt[:, 0], gt[:, 0], gt[:, 1])
            gt_y = np.interp(filt[:, 0], gt[:, 0], gt[:, 2])
            return np.sqrt((filt[:, 1] - gt_x) ** 2 + (filt[:, 2] - gt_y) ** 2)

        if kf is not None:
            e = euclidean_error(kf)
            ax.plot(kf[:, 0], e, 'b-',
                    label=f'KF  (mean={e.mean():.3f} m, max={e.max():.3f} m)',
                    linewidth=1.2)
        if ekf is not None:
            e = euclidean_error(ekf)
            ax.plot(ekf[:, 0], e, 'r-',
                    label=f'EKF (mean={e.mean():.3f} m, max={e.max():.3f} m)',
                    linewidth=1.2)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Error [m]')
        ax.set_title('Euclidean Position Error')
        ax.legend()
        ax.grid(True)

    @staticmethod
    def _plot_pos_variance(ax, kf, ekf):
        if kf is not None:
            ax.plot(kf[:, 0], kf[:, 4], 'b-',  label='KF  σ_x²', linewidth=1.2)
            ax.plot(kf[:, 0], kf[:, 5], 'b--', label='KF  σ_y²', linewidth=1.2)
        if ekf is not None:
            ax.plot(ekf[:, 0], ekf[:, 4], 'r-',  label='EKF σ_x²', linewidth=1.2)
            ax.plot(ekf[:, 0], ekf[:, 5], 'r--', label='EKF σ_y²', linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Variance [m²]')
        ax.set_title('Position Covariance  (σ_x², σ_y²)')
        ax.legend()
        ax.grid(True)

    @staticmethod
    def _plot_yaw_variance(ax, kf, ekf):
        if kf is not None:
            ax.plot(kf[:, 0], kf[:, 6], 'b-', label='KF  σ_θ²', linewidth=1.2)
        if ekf is not None:
            ax.plot(ekf[:, 0], ekf[:, 6], 'r-', label='EKF σ_θ²', linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Variance [rad²]')
        ax.set_title('Yaw Covariance  (σ_θ²)')
        ax.legend()
        ax.grid(True)


# ---------------------------------------------------------------------- #
#  Entry point                                                             #
# ---------------------------------------------------------------------- #

def main():
    rclpy.init()
    node = EvalNode()

    # SIGTERM is sent by `ros2 launch` to children; rclpy only handles SIGINT.
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Reached on both Ctrl+C (SIGINT) and SIGTERM shutdown paths.
        node.generate_plots()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
