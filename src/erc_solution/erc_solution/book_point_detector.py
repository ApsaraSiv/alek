"""
book_point_detector.py

Publishes /erc/target_book_point (geometry_msgs/PointStamped) for
manipulation_node's grasp_book -- the real 3D position of the
target-coloured book, from the head camera's RGB + depth, not a
hardcoded placeholder and not a world-geometry guess.

Why this exists: navigation_node's approach_shelf only rotates to center
the column marker in view, then drives straight forward -- it never
strafes sideways, so the robot ends up facing the right column but is
not guaranteed to be laterally in front of that column's books (measured
1.83m lateral miss in testing). A world-geometry-based grasp target
(shelf_column_number + row) inherits that error and reaches for the
wrong spot. Detecting the book directly in the camera and back-projecting
its real depth sidesteps this: wherever the robot actually ended up,
this publishes where the book actually is, not where it "should" be.

Same HSV colour ranges as erc_perception/book_color_detector.py (kept in
sync manually -- no shared constants module between packages). Detection
runs on the colour image; depth is read from the depth image at the same
pixel (colour and depth camera_info report identical resolution/intrinsics
in this sim, so they're pixel-aligned -- no separate registration needed).

STILL TO TUNE / KNOWN LIMITATIONS:
  - Publishes only while the target colour is actually visible and has a
    valid (non-zero, non-NaN) depth reading at its centroid. If the book
    is fully out of frame (possible given the lateral-miss issue above),
    grasp_book will simply time out waiting for a point -- that's an
    honest failure, not a silent wrong-target grasp.
  - No temporal smoothing across frames -- publishes the latest single
    detection whenever the colour is in view. MAX_PLAUSIBLE_DISTANCE /
    MIN_PLAUSIBLE_Z reject obviously-wrong single-frame outliers (seen: a
    point 2.4m away for a book on a nearby shelf) but won't catch a bad
    reading that happens to land inside plausible bounds.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point

COLOR_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
DEPTH_INFO_TOPIC = '/head_front_camera/head_front_camera/depth/camera_info'

# Same ranges as erc_perception/book_color_detector.py.
COLOUR_RANGES = {
    'red': [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    'blue': [((100, 100, 60), (130, 255, 255))],
    'green': [((40, 70, 60), (85, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
}

MIN_BLOB_AREA = 200

# Sanity bounds on the published point (output_frame, i.e. base_footprint) --
# a bad depth/colour reading (motion blur, edge noise, a reflection) can
# back-project to a wildly implausible position that then dooms a grasp
# attempt outright (seen: (1.59, 0.76, 1.60), ~2.4m from the robot, for a
# book physically mounted on a nearby shelf). Reject and keep waiting for a
# better frame instead of publishing something that can't be real.
#
# 1.4 (first value tried) turned out too tight: rejected a real, stable
# detection at 1.64m (unlike the drifting/inconsistent bad ones, which were
# all 2.2m+) right as manipulation_node's widened orientation/position
# tolerances and torso lift made that distance plausibly reachable. Raised
# to keep a margin below the ~2.2m the bad readings clustered at, while
# accepting genuine detections like that one.
#
# 1.9 still rejected another real, stable detection (consistent across
# dozens of frames, not drifting like the bad ones) at ~2.0-2.02m during a
# middle-shelf approach via the debug column-search bypass, which doesn't
# laterally centre on a column so ends up slightly farther off. Raised
# again, still keeping margin below the ~2.2m+ bad-reading cluster.
#
# 2.1 still rejected a real detection during a full column-2 run: values
# moved smoothly and monotonically frame to frame (2.85m -> 2.49m over
# ~9s, x/y/z all drifting consistently in the same direction each frame)
# -- a physically real, settling object, not noise (the genuinely bad
# readings seen elsewhere jump erratically between unrelated positions
# frame to frame, not a smooth trend). The 10s point_wait_timeout window
# closed before it crossed under the old 2.1m cutoff. Raised again to
# give a still-converging real detection room to be accepted.
MAX_PLAUSIBLE_DISTANCE = 2.7  # m, straight-line from base_footprint origin
MIN_PLAUSIBLE_Z = 0.3         # m -- below this is basically the floor


class BookPointDetector(Node):
    def __init__(self):
        super().__init__('book_point_detector', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])
        self.declare_parameter('target_colour', 'red')
        self.declare_parameter('output_frame', 'base_footprint')
        target_colour = self.get_parameter('target_colour').value
        if target_colour not in COLOUR_RANGES:
            self.get_logger().error(
                f'Unknown target_colour {target_colour!r}, defaulting to "red".')
            target_colour = 'red'
        self.target_colour = target_colour
        self.output_frame = self.get_parameter('output_frame').value

        cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()

        self.depth_frame = None
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_frame_id = None

        self.point_pub = self.create_publisher(PointStamped, '/erc/target_book_point', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, DEPTH_INFO_TOPIC, self._on_depth_info, 10, callback_group=cb_group)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._on_depth, 10, callback_group=cb_group)
        self.create_subscription(
            Image, COLOR_TOPIC, self._on_color, 10, callback_group=cb_group)

        self.get_logger().info(
            f'book_point_detector ready, target_colour={self.target_colour}, '
            f'publishing in frame {self.output_frame}')

    def _on_depth_info(self, msg: CameraInfo):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]

    def _on_depth(self, msg: Image):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.depth_frame_id = msg.header.frame_id
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert depth frame: {exc}')

    def _on_color(self, msg: Image):
        if self.depth_frame is None or self.fx is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert colour frame: {exc}')
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOUR_RANGES[self.target_colour]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BLOB_AREA:
                continue
            if best is None or area > best[1]:
                best = (contour, area)
        if best is None:
            return

        x, y, w, h = cv2.boundingRect(best[0])
        px, py = x + w // 2, y + h // 2

        if not (0 <= py < self.depth_frame.shape[0] and 0 <= px < self.depth_frame.shape[1]):
            return
        depth = float(self.depth_frame[py, px])
        if not np.isfinite(depth) or depth <= 0.0:
            return

        # Pinhole back-projection: camera optical frame convention
        # (X right, Y down, Z forward).
        cam_x = (px - self.cx) * depth / self.fx
        cam_y = (py - self.cy) * depth / self.fy
        cam_z = depth

        point = PointStamped()
        point.header.stamp = msg.header.stamp
        point.header.frame_id = self.depth_frame_id
        point.point.x, point.point.y, point.point.z = cam_x, cam_y, cam_z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, self.depth_frame_id, rclpy.time.Time())
            point = do_transform_point(point, transform)
        except Exception as exc:
            self.get_logger().warn(
                f'TF transform {self.depth_frame_id} -> {self.output_frame} failed: {exc}')
            return

        p = point.point
        distance = (p.x ** 2 + p.y ** 2 + p.z ** 2) ** 0.5
        if distance > MAX_PLAUSIBLE_DISTANCE or p.z < MIN_PLAUSIBLE_Z:
            self.get_logger().warn(
                f'rejecting implausible book point ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}), '
                f'distance={distance:.2f}m -- likely a bad depth/colour reading')
            return

        self.point_pub.publish(point)


def main(args=None):
    rclpy.init(args=args)
    node = BookPointDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
