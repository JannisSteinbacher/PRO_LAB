#!/usr/bin/env python3
"""
apriltag_detector_node.py — Detect AprilTag landmarks from the TurtleBot4 camera.

  /rgbd_camera/image        (sensor_msgs/Image)       --\
  /rgbd_camera/camera_info  (sensor_msgs/CameraInfo)  ---> cv2.aruco (tag36h11)
                                                            -> per-tag 3D pose
                                                            -> transform into base_link
                                                            -> /apriltag/detections (MarkerArray)

Each detected tag is published as one visualization_msgs/Marker:
  * marker.id              = AprilTag id        (signature / known correspondence)
  * marker.header.frame_id = base_link          (robot body frame)
  * marker.pose.position   = tag position in the robot frame [m]

The EKF node consumes this, derives (range, bearing) = (hypot(x, y), atan2(y, x))
and runs the landmark correction (Thrun, EKF_localization_known_correspondences).
Publishing in base_link also makes the detections show up directly in RViz.


Tag size note
  The textured quad is 0.2 m, but the *black square* the detector keys on is the
  inner 8/10 of it = 0.16 m (a 1-cell white margin on every side). 'tag_size' is
  that black-square edge length; getting it right keeps the range estimate true.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from a (x, y, z, w) quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('apriltag_detector_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('image_topic', '/rgbd_camera/image')
        self.declare_parameter('camera_info_topic', '/rgbd_camera/camera_info')
        self.declare_parameter('detections_topic', '/apriltag/detections')
        self.declare_parameter('base_frame', 'base_link')
        # Fallback only — the camera-image header.frame_id is preferred.
        self.declare_parameter('optical_frame', 'oakd_rgb_camera_optical_frame')
        self.declare_parameter('tag_size', 0.16)      # black-square edge [m]
        self.declare_parameter('max_range', 8.0)      # drop detections beyond this [m]
        self.declare_parameter('marker_lifetime', 0.3)

        self._base_frame = self.get_parameter('base_frame').value
        self._optical_frame_param = self.get_parameter('optical_frame').value
        self._tag_size = float(self.get_parameter('tag_size').value)
        self._max_range = float(self.get_parameter('max_range').value)
        self._lifetime = float(self.get_parameter('marker_lifetime').value)

        image_topic = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        det_topic = self.get_parameter('detections_topic').value

        # ---------------- Detector ----------------
        self._bridge = CvBridge()
        self._dictionary = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        self._det_params = aruco.DetectorParameters_create()

        # ---------------- Camera intrinsics (filled by camera_info) ----------
        self._K = None          # 3x3 camera matrix
        self._D = None          # distortion (zeros in sim)

        # ---------------- Static camera -> base transform (cached once) ------
        self._R_bo = None       # rotation base_link <- optical
        self._t_bo = None       # translation base_link <- optical
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---------------- I/O ----------------
        # sensor_data QoS (best effort) is compatible with both the reliable
        # parameter_bridge (camera_info) and the image_bridge stream.
        self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self._pub = self.create_publisher(MarkerArray, det_topic, 10)

        self.get_logger().info(
            f'AprilTag detector ready. image="{image_topic}" '
            f'info="{info_topic}" -> "{det_topic}" '
            f'(family=36h11, tag_size={self._tag_size} m)')

    # ------------------------------------------------------------------ #
    def _info_cb(self, msg: CameraInfo):
        if self._K is not None:
            return
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        self._D = d.reshape(1, -1) if d.size else np.zeros((1, 5))
        self.get_logger().info(
            f'Camera intrinsics received (fx={self._K[0, 0]:.1f}, '
            f'fy={self._K[1, 1]:.1f}).')

    # ------------------------------------------------------------------ #
    def _ensure_static_tf(self, optical_frame: str) -> bool:
        """Look up (once) and cache the static base_link <- optical transform."""
        if self._R_bo is not None:
            return True
        src = optical_frame if optical_frame else self._optical_frame_param
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, src, rclpy.time.Time())
        except tf2_ros.TransformException as ex:  # not available yet
            self.get_logger().warn(
                f'Waiting for transform {self._base_frame} <- {src}: {ex}',
                throttle_duration_sec=5.0)
            return False
        q = tf.transform.rotation
        t = tf.transform.translation
        self._R_bo = _quat_to_rot(q.x, q.y, q.z, q.w)
        self._t_bo = np.array([t.x, t.y, t.z])
        self.get_logger().info(
            f'Cached static transform {self._base_frame} <- {src}.')
        return True

    # ------------------------------------------------------------------ #
    def _image_cb(self, msg: Image):
        if self._K is None:
            return  # need intrinsics first
        if not self._ensure_static_tf(msg.header.frame_id):
            return

        gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        corners, ids, _ = aruco.detectMarkers(
            gray, self._dictionary, parameters=self._det_params)
        if ids is None or len(ids) == 0:
            return

        # tvec[i] = tag-centre position in the camera (optical) frame.
        _, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self._tag_size, self._K, self._D)

        out = MarkerArray()
        for i, tag_id in enumerate(ids.flatten()):
            p_opt = tvecs[i].reshape(3)
            p_base = self._R_bo @ p_opt + self._t_bo   # optical -> base_link

            rng = math.hypot(p_base[0], p_base[1])      # horizontal range
            if rng < 1e-3 or rng > self._max_range:
                continue

            m = Marker()
            m.header.frame_id = self._base_frame
            m.header.stamp = msg.header.stamp
            m.ns = 'apriltag'
            m.id = int(tag_id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p_base[0])
            m.pose.position.y = float(p_base[1])
            m.pose.position.z = float(p_base[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.g = 1.0
            m.color.a = 1.0
            m.lifetime = rclpy.duration.Duration(
                seconds=self._lifetime).to_msg()
            out.markers.append(m)

        if out.markers:
            self._pub.publish(out)


def main():
    rclpy.init()
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
