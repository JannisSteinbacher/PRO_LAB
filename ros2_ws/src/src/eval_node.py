#!/usr/bin/env python3
"""
eval_node.py — Real-time evaluation of KF, EKF and PF filter performance.

Subscribes to:
  /kf/pose_estimate              (PoseWithCovarianceStamped)
  /ekf/pose_estimate             (PoseWithCovarianceStamped)
  /pf/pose_estimate              (PoseWithCovarianceStamped)
  /kf/kalman_gain_theta          (std_msgs/Float64) — theta Kalman gain
  /ekf/kalman_gain_theta         (std_msgs/Float64) — theta Kalman gain
  <ground_truth_topic param>     (geometry_msgs/PoseArray) — Gazebo ground truth

Default ground truth topic: /world/depot/dynamic_pose/info
  A ros_gz_bridge (started by filter_launch.py) relays the gz.msgs.Pose_V
  ground-truth poses onto this topic as a PoseArray. The bridge drops entity
  names, so the robot is found by matching the 'ground_truth_spawn_xy' param
  (world-frame spawn, default [-8.0, 0.0]) and that array index is then locked.
  -> If no ground truth shows up, check the bridge with:
       ros2 topic echo <ground_truth_topic> --once
  -> Override at runtime:
       ros2 run turtlebot_state_estimation eval_node --ros-args \
         -p ground_truth_topic:=/your/topic \
         -p ground_truth_spawn_xy:='[-8.0, 0.0]'

On shutdown (Ctrl+C / SIGTERM), writes to ros2_ws/eval_results/:
  filter_eval_<timestamp>.png   — 4-panel plot (trajectory, error, σ²_pos, σ²_yaw)
                                   with ground truth vs KF / EKF (PF is hidden
                                   unless plot_pf:=true; CSV always has PF).
  csv/{kf,ekf,pf}_eval_<timestamp>.csv — columns:
       time_s, est_x, est_y, est_theta_deg, gt_x, gt_y, gt_theta_deg,
       cov_x, cov_y, cov_theta_deg2, K_theta
  Positions/variances are metres / m²; angles are DEGREES and the yaw variance
  is deg² (so sqrt(cov_theta_deg2) is a std-dev in degrees). K_theta is the
  scalar Kalman gain applied to theta from the IMU yaw correction (KF/EKF only;
  NaN for the PF, which has no Kalman gain).
  All three CSVs are resampled onto ONE shared time grid (sync_rate_hz, default
  20 Hz) spanning where all filters overlap, so their rows line up 1:1 and share
  identical time_s / gt_* columns. time_s is elapsed seconds since the robot
  started driving (see wait_for_start); est_*/cov_* are each filter resampled
  onto the grid and gt_* is the ground truth on the same grid, in the odom frame
  the filters report in.
"""

import csv
import os
import signal
from datetime import datetime
from math import atan2, cos, sin

import matplotlib
matplotlib.use('Agg')  # file-only backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import Empty, Float64
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


_OUT_DIR = os.path.expanduser('~/Documents/MRE/PRO_LAB/ros2_ws/eval_results')
_CSV_DIR = os.path.join(_OUT_DIR, 'csv')   # CSVs kept separate from the PNGs
_RAD2DEG = 180.0 / np.pi
# Angles are exported in degrees (variance in deg^2); units are in the names so
# the CSV is unambiguous. x/y are metres, their variances m^2.
_CSV_HEADER = ['time_s', 'est_x', 'est_y', 'est_theta_deg',
               'gt_x', 'gt_y', 'gt_theta_deg',
               'cov_x', 'cov_y', 'cov_theta_deg2', 'K_theta']


def _quat_to_yaw(q) -> float:
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]; works on scalars and numpy arrays."""
    return np.arctan2(np.sin(a), np.cos(a))


def _interp_angle(times, ts, angles):
    """Wrap-safe interpolation of angle samples onto `times` (via sin/cos), so
    the result stays correct across the +/-pi wrap."""
    c = np.interp(times, ts, np.cos(angles))
    s = np.interp(times, ts, np.sin(angles))
    return np.arctan2(s, c)


def _interp_gt(times, gt):
    """Resample aligned ground truth (gt columns t, x, y, yaw) onto `times`."""
    gx = np.interp(times, gt[:, 0], gt[:, 1])
    gy = np.interp(times, gt[:, 0], gt[:, 2])
    gth = _interp_angle(times, gt[:, 0], gt[:, 3])
    return gx, gy, gth


class EvalNode(Node):
    def __init__(self):
        super().__init__('eval_node')

        self.declare_parameter('ground_truth_topic', '/world/depot/dynamic_pose/info')
        # The ros_gz_bridge drops entity names from the Pose_V, so we can't match
        # by name. Instead we lock onto the array entry nearest the robot's spawn
        # position (world frame) on the first message and reuse that index — this
        # survives a shift in entity order that would break a hardcoded index.
        self.declare_parameter('ground_truth_spawn_xy', [-8.0, 0.0])
        # Hold off recording until the patrol node sends its first goal (it
        # publishes std_msgs/Empty on this topic). This skips the long Gazebo /
        # Nav2 startup where the robot just sits at spawn. Set wait_for_start
        # false to record immediately (e.g. running eval without patrol_node).
        self.declare_parameter('wait_for_start', True)
        self.declare_parameter('start_topic', '/eval/start')
        # Set false to collect data but skip writing the PNG on shutdown.
        self.declare_parameter('save_plots', True)
        # Set false to skip writing the per-filter CSV files on shutdown.
        self.declare_parameter('save_csv', True)
        # All three filter CSVs are resampled onto one shared time grid at this
        # rate [Hz] so their rows line up 1:1 for direct comparison.
        self.declare_parameter('sync_rate_hz', 20.0)
        # Draw the PF series in the PNG plots. CSV output is unaffected (PF is
        # always written); this only controls the plot for now.
        self.declare_parameter('plot_pf', False)
        gt_topic = (self.get_parameter('ground_truth_topic')
                    .get_parameter_value().string_value)
        spawn = (self.get_parameter('ground_truth_spawn_xy')
                 .get_parameter_value().double_array_value)
        self._gt_spawn = (float(spawn[0]), float(spawn[1]))
        self._gt_index: int | None = None  # locked on the first GT message
        start_topic = (self.get_parameter('start_topic')
                       .get_parameter_value().string_value)
        # When wait_for_start is false, tracking is "already started".
        self._tracking = not (self.get_parameter('wait_for_start')
                              .get_parameter_value().bool_value)
        self._save_plots = (self.get_parameter('save_plots')
                            .get_parameter_value().bool_value)
        self._save_csv = (self.get_parameter('save_csv')
                          .get_parameter_value().bool_value)
        self._sync_rate_hz = (self.get_parameter('sync_rate_hz')
                              .get_parameter_value().double_value)
        self._plot_pf = (self.get_parameter('plot_pf')
                         .get_parameter_value().bool_value)

        # Each list stores tuples; converted to np.array on shutdown.
        # GT:         (t, x, y, yaw)
        # KF/EKF/PF:  (t, x, y, yaw, var_x, var_y, var_yaw)
        self._gt: list  = []
        self._kf: list  = []
        self._ekf: list = []
        self._pf: list  = []
        # Theta Kalman gain time series (t, K_theta) per filter. Only the KF and
        # EKF have one; the PF is a particle filter, so it never gets a column.
        self._kgain: dict = {'kf': [], 'ekf': []}
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
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/pf/pose_estimate', self._pf_cb, 10)
        self.create_subscription(Float64, '/kf/kalman_gain_theta',
                                 lambda m: self._kgain_cb(m, 'kf'), 10)
        self.create_subscription(Float64, '/ekf/kalman_gain_theta',
                                 lambda m: self._kgain_cb(m, 'ekf'), 10)

        # Latched (transient-local) so the signal is still delivered if patrol
        # publishes it before this subscription is matched.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        if not self._tracking:
            self.create_subscription(Empty, start_topic, self._start_cb,
                                     latched_qos)

        self.get_logger().info(f'EvalNode ready.  Ground truth: {gt_topic} '
                               f'(spawn match: {self._gt_spawn})')
        if self._tracking:
            self.get_logger().info('Recording immediately (wait_for_start=false).')
        else:
            self.get_logger().info(
                f'Waiting for start signal on "{start_topic}" before recording.')
        self.get_logger().info('Press Ctrl+C to stop and save plots.')

    def _start_cb(self, _msg: Empty):
        if self._tracking:
            return
        self._tracking = True
        self.get_logger().info('Start signal received — recording begins now.')

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
        # Pose_V bridged as PoseArray (entity names are dropped by the bridge).
        # On the first message, find the pose nearest the known spawn position
        # and lock its index; the bridge keeps a stable entity order per session,
        # so that index stays valid for the rest of the run.
        if not self._tracking or not msg.poses:
            return
        if self._gt_index is None:
            sx, sy = self._gt_spawn
            i = min(range(len(msg.poses)),
                    key=lambda k: (msg.poses[k].position.x - sx) ** 2
                                + (msg.poses[k].position.y - sy) ** 2)
            d = ((msg.poses[i].position.x - sx) ** 2
                 + (msg.poses[i].position.y - sy) ** 2) ** 0.5
            # Guard: don't lock before the robot actually shows up in the array.
            if d > 1.0:
                return
            self._gt_index = i
            self.get_logger().info(
                f'Ground-truth entity locked at index {i} '
                f'({d:.3f} m from spawn {self._gt_spawn}).')

        if len(msg.poses) <= self._gt_index:
            return
        t = self._now()
        p = msg.poses[self._gt_index]
        self._gt.append((t, p.position.x, p.position.y,
                         _quat_to_yaw(p.orientation)))

    def _filter_cb(self, msg: PoseWithCovarianceStamped, store: list):
        if not self._tracking:
            return
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

    def _pf_cb(self, msg: PoseWithCovarianceStamped):
        self._filter_cb(msg, self._pf)

    def _kgain_cb(self, msg: Float64, name: str):
        if not self._tracking:
            return
        self._kgain[name].append((self._now(), msg.data))

    # ------------------------------------------------------------------ #
    #  Output generation  (CSV + plots)                                    #
    # ------------------------------------------------------------------ #

    def write_outputs(self):
        """Build the aligned arrays once, then write per-filter CSVs and a PNG."""
        gt, filters, kgain = self._collect_aligned()

        if all(f is None for f in filters.values()):
            self.get_logger().warn('No filter data received. Nothing to write.')
            return
        if gt is None:
            self.get_logger().warn(
                'No ground-truth data received — GT columns will be NaN and the '
                'error panel is skipped. Check the ground_truth_topic parameter.')

        os.makedirs(_OUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.generate_csv(gt, filters, kgain, timestamp)
        self.generate_plots(gt, filters, timestamp)

    def _collect_aligned(self):
        """np.array each buffer and shift GT into the robot's start (odom) frame."""
        gt = np.array(self._gt) if self._gt else None
        filters = {
            'kf':  np.array(self._kf)  if self._kf  else None,
            'ekf': np.array(self._ekf) if self._ekf else None,
            'pf':  np.array(self._pf)  if self._pf  else None,
        }
        kgain = {k: (np.array(v) if v else None)
                 for k, v in self._kgain.items()}
        if gt is not None:
            # Filters start at (0, 0, 0) in the odom frame; the first GT sample
            # defines that frame's origin and heading in the Gazebo world frame,
            # so apply the inverse rigid transform to every GT sample.
            x0, y0, th0 = gt[0, 1], gt[0, 2], gt[0, 3]
            c, s = cos(th0), sin(th0)
            dx, dy = gt[:, 1] - x0, gt[:, 2] - y0
            gt[:, 1] = c * dx + s * dy
            gt[:, 2] = -s * dx + c * dy
            gt[:, 3] = _wrap(gt[:, 3] - th0)
        return gt, filters, kgain

    # ---- CSV ---------------------------------------------------------- #

    def generate_csv(self, gt, filters, kgain, timestamp):
        if not self._save_csv:
            self.get_logger().info('save_csv=false — skipping CSV files.')
            return

        # One shared time grid so every filter CSV lines up row-for-row.
        grid = self._common_time_grid(filters, gt)
        if grid is None:
            self.get_logger().warn(
                'Filters share no overlapping time span — cannot sync CSVs.')
            return

        os.makedirs(_CSV_DIR, exist_ok=True)

        if gt is not None:
            gx, gy, gth = _interp_gt(grid, gt)   # GT resampled once for all
        else:
            nan = np.full(len(grid), np.nan)
            gx, gy, gth = nan, nan, nan
        gth_deg = np.degrees(gth)                # angles exported in degrees

        for name, filt in filters.items():
            if filt is None:
                continue
            # Resample this filter onto the shared grid (yaw is wrap-safe).
            ex = np.interp(grid, filt[:, 0], filt[:, 1])
            ey = np.interp(grid, filt[:, 0], filt[:, 2])
            eth_deg = np.degrees(_interp_angle(grid, filt[:, 0], filt[:, 3]))
            cx = np.interp(grid, filt[:, 0], filt[:, 4])
            cy = np.interp(grid, filt[:, 0], filt[:, 5])
            # Yaw variance rad^2 -> deg^2, so its sqrt is a std-dev in degrees.
            cth_deg2 = np.interp(grid, filt[:, 0], filt[:, 6]) * _RAD2DEG ** 2
            # Theta Kalman gain (KF/EKF only; NaN for the PF, which has none).
            kg = kgain.get(name)
            if kg is not None and len(kg):
                kth = np.interp(grid, kg[:, 0], kg[:, 1])
            else:
                kth = np.full(len(grid), np.nan)

            path = os.path.join(_CSV_DIR, f'{name}_eval_{timestamp}.csv')
            with open(path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(_CSV_HEADER)
                for i in range(len(grid)):
                    writer.writerow([
                        f'{grid[i]:.6f}',                  # time_s
                        f'{ex[i]:.6f}', f'{ey[i]:.6f}',    # est_x, est_y [m]
                        f'{eth_deg[i]:.6f}',               # est_theta [deg]
                        f'{gx[i]:.6f}', f'{gy[i]:.6f}',    # gt_x, gt_y [m]
                        f'{gth_deg[i]:.6f}',               # gt_theta [deg]
                        f'{cx[i]:.9f}', f'{cy[i]:.9f}',    # cov_x, cov_y [m^2]
                        f'{cth_deg2[i]:.6f}',              # cov_theta [deg^2]
                        f'{kth[i]:.6f}',                   # K_theta (IMU yaw gain)
                    ])
            self.get_logger().info(f'Saved → {path}  ({len(grid)} rows)')

    def _common_time_grid(self, filters, gt):
        """Uniform time grid (at sync_rate_hz) over the span where every present
        filter — and ground truth, if any — has data. Restricting to the overlap
        avoids extrapolating any series. Returns None if there is no overlap."""
        present = [f for f in filters.values() if f is not None]
        if not present:
            return None
        t_start = max(float(f[0, 0]) for f in present)
        t_end = min(float(f[-1, 0]) for f in present)
        if gt is not None:
            t_start = max(t_start, float(gt[0, 0]))
            t_end = min(t_end, float(gt[-1, 0]))
        if t_end <= t_start:
            return None
        n = int(round((t_end - t_start) * self._sync_rate_hz)) + 1
        return t_start + np.arange(n) / self._sync_rate_hz

    # ---- Plots -------------------------------------------------------- #

    def generate_plots(self, gt, filters, timestamp):
        if not self._save_plots:
            self.get_logger().info('save_plots=false — skipping evaluation PNG.')
            return
        kf, ekf, pf = filters['kf'], filters['ekf'], filters['pf']
        if not self._plot_pf:
            pf = None   # keep PF out of the PNG for now (still in the CSV)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Filter Performance Evaluation', fontsize=14, fontweight='bold')

        self._plot_trajectory(axes[0, 0], gt, kf, ekf, pf)
        self._plot_error(axes[0, 1], gt, kf, ekf, pf)
        self._plot_pos_variance(axes[1, 0], kf, ekf, pf)
        self._plot_yaw_variance(axes[1, 1], kf, ekf, pf)

        plt.tight_layout()
        out_path = os.path.join(_OUT_DIR, f'filter_eval_{timestamp}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f'Saved → {out_path}')

    # ---- individual panels ------------------------------------------- #

    # (filter array, matplotlib color, legend label) for KF / EKF / PF.
    # Color convention: KF=red, EKF=green. PF takes blue so it doesn't collide
    # with EKF's green (PF is hidden from the PNG by default; see plot_pf).
    _SERIES = (('kf', 'r', 'KF'), ('ekf', 'g', 'EKF'), ('pf', 'b', 'PF'))

    @classmethod
    def _series(cls, kf, ekf, pf):
        arrays = {'kf': kf, 'ekf': ekf, 'pf': pf}
        return [(arrays[k], color, label) for k, color, label in cls._SERIES
                if arrays[k] is not None]

    @classmethod
    def _plot_trajectory(cls, ax, gt, kf, ekf, pf):
        if gt is not None:
            ax.plot(gt[:, 1], gt[:, 2], 'k-',
                    label='Ground Truth', linewidth=2.0, zorder=3)
        for filt, color, label in cls._series(kf, ekf, pf):
            ax.plot(filt[:, 1], filt[:, 2], color + '--', label=label, linewidth=1.4)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Trajectory')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend()
        ax.grid(True)

    @classmethod
    def _plot_error(cls, ax, gt, kf, ekf, pf):
        if gt is None:
            ax.text(0.5, 0.5, 'No ground truth available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Error')
            return
        for filt, color, label in cls._series(kf, ekf, pf):
            gx, gy, _ = _interp_gt(filt[:, 0], gt)
            e = np.sqrt((filt[:, 1] - gx) ** 2 + (filt[:, 2] - gy) ** 2)
            ax.plot(filt[:, 0], e, color + '-',
                    label=f'{label} (mean={e.mean():.3f} m, max={e.max():.3f} m)',
                    linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Error [m]')
        ax.set_title('Euclidean Position Error')
        ax.legend()
        ax.grid(True)

    @classmethod
    def _plot_pos_variance(cls, ax, kf, ekf, pf):
        for filt, color, label in cls._series(kf, ekf, pf):
            ax.plot(filt[:, 0], filt[:, 4], color + '-',
                    label=f'{label} σ_x²', linewidth=1.2)
            ax.plot(filt[:, 0], filt[:, 5], color + '--',
                    label=f'{label} σ_y²', linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Variance [m²]')
        ax.set_title('Position Covariance  (σ_x², σ_y²)')
        ax.legend()
        ax.grid(True)

    @classmethod
    def _plot_yaw_variance(cls, ax, kf, ekf, pf):
        for filt, color, label in cls._series(kf, ekf, pf):
            ax.plot(filt[:, 0], filt[:, 6], color + '-',
                    label=f'{label} σ_θ²', linewidth=1.2)
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
        node.write_outputs()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
