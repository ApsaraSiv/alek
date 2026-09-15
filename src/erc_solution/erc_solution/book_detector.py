import math

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

COLOR_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
COLOR_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
DEPTH_INFO_TOPIC = '/head_front_camera/head_front_camera/depth/camera_info'
FRONT_SCAN_TOPIC = '/scan_front_raw'

# Same colour ranges as erc_perception's book_color_detector (kept in sync
# manually -- erc_solution doesn't depend on erc_perception).
COLOUR_RANGES = {
    'red': [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    'blue': [((100, 100, 60), (130, 255, 255))],
    'green': [((40, 70, 60), (85, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
}

MIN_BLOB_AREA = 200
DEPTH_PATCH_RADIUS = 3  # px, half-width of the window averaged for depth at the centroid
STALE_SEC = 1.0  # how old a depth frame / lidar scan can be and still be trusted

# Front-facing cone to pull a single "distance to what's ahead" scalar out
# of the LiDAR, same convention navigation_node uses for its own
# front_min_range so the two stay consistent.
FRONT_CONE_HALF_ANGLE = math.radians(15.0)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class BookDetector(Node):

    def __init__(self):
        super().__init__('book_detector')
        self.declare_parameter('target_book_colour', 'red')
        target_colour = self.get_parameter('target_book_colour').value
        if target_colour not in COLOUR_RANGES:
            self.get_logger().error(
                f'Unknown target_book_colour {target_colour!r}, must be one of '
                f'{list(COLOUR_RANGES)}. Defaulting to "red".')
            target_colour = 'red'
        self.target_colour = target_colour

        self.bridge = CvBridge()
        self.color_info = None
        self.depth_info = None
        self.latest_depth = None       # (frame, header) or None
        self.front_range = None        # float, metres
        self.front_range_stamp = None  # rclpy Time

        self.create_subscription(CameraInfo, COLOR_INFO_TOPIC, self._on_color_info, 10)
        self.create_subscription(CameraInfo, DEPTH_INFO_TOPIC, self._on_depth_info, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth_image, 10)
        self.create_subscription(LaserScan, FRONT_SCAN_TOPIC, self._on_front_scan, 10)
        self.create_subscription(Image, COLOR_TOPIC, self._on_color_image, 10)

        self.row_pub = self.create_publisher(
            Int32, '/erc/shelf_row_identification', 10)
        self.point_pub = self.create_publisher(
            PointStamped, '/erc/target_book_point', 10)

        self.get_logger().info(
            f'book_detector looking for target_book_colour={self.target_colour}, '
            'back-projecting via the depth camera when available, else falling '
            'back to the front LiDAR range, to publish /erc/target_book_point')

    # ------------------------------------------------------------------
    def _on_color_info(self, msg: CameraInfo):
        self.color_info = msg

    def _on_depth_info(self, msg: CameraInfo):
        self.depth_info = msg

    def _on_depth_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert depth image, skipping frame: {exc}')
            return
        self.latest_depth = (frame, msg.header)

    def _on_front_scan(self, msg: LaserScan):
        ranges = [
            distance for index, distance in enumerate(msg.ranges)
            if abs(normalize_angle(msg.angle_min + index * msg.angle_increment))
            <= FRONT_CONE_HALF_ANGLE
            and msg.range_min <= distance <= msg.range_max
        ]
        self.front_range = min(ranges) if ranges else None
        self.front_range_stamp = self.get_clock().now()

    # ------------------------------------------------------------------
    def _find_target_blob(self, color_frame):
        """Picks the target-colour blob closest to the image's horizontal
        centre, not the largest. The shelf can have several books of the
        same colour visible at once (one per column, colours are only
        unique *within* a column -- see simulation.launch.py's book spawn
        logic), and the largest-blob heuristic used to flip between two of
        them from one frame to the next depending on which was fractionally
        bigger. That fed manipulation_node's base-centering loop a moving
        target it could never converge on. Whatever book the robot is
        actually squared up to (the one nearest dead ahead) is the one we
        actually approached/tucked the arm for, so that's the stable choice."""
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOUR_RANGES[self.target_colour]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_center_u = color_frame.shape[1] / 2.0
        best = None
        best_center_dist = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BLOB_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center_dist = abs((x + w / 2.0) - image_center_u)
            if best is None or center_dist < best_center_dist:
                best = (x, y, w, h)
                best_center_dist = center_dist
        return best

    def _depth_at(self, depth_frame, u, v):
        """Median of finite depth samples in a small window around (u, v),
        so a single noisy/zero pixel at the centroid doesn't kill detection."""
        h, w = depth_frame.shape[:2]
        u0, u1 = max(0, u - DEPTH_PATCH_RADIUS), min(w, u + DEPTH_PATCH_RADIUS + 1)
        v0, v1 = max(0, v - DEPTH_PATCH_RADIUS), min(h, v + DEPTH_PATCH_RADIUS + 1)
        patch = depth_frame[v0:v1, u0:u1].astype(np.float32).ravel()
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _resolve_depth(self, du, dv, now):
        """Best available distance-to-book estimate, plus the intrinsics/
        frame it should be back-projected with. Depth camera first (it's
        the accurate one when it's actually producing frames); if it's
        missing/stale/invalid at this pixel, fall back to the front LiDAR
        range -- coarser (it assumes the shelf front face is a flat plane
        square-on to the robot, since the LiDAR gives one distance, not a
        per-pixel one), but it's a real reading instead of nothing, and
        this sim's software-rendered depth camera doesn't reliably
        produce frames at all on GPU-less hosts.
        """
        stale = Duration(seconds=STALE_SEC)

        if (self.latest_depth is not None and self.depth_info is not None
                and now - rclpy.time.Time.from_msg(self.latest_depth[1].stamp) <= stale):
            frame, header = self.latest_depth
            depth = self._depth_at(frame, du, dv)
            if depth is not None:
                return depth, self.depth_info, header.frame_id, 'depth_camera'

        if (self.front_range is not None and self.color_info is not None
                and self.front_range_stamp is not None
                and now - self.front_range_stamp <= stale):
            return self.front_range, self.color_info, self.color_info.header.frame_id, 'lidar_fallback'

        return None, None, None, None

    # ------------------------------------------------------------------
    def _on_color_image(self, color_msg: Image):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert colour image, skipping frame: {exc}')
            return

        blob = self._find_target_blob(color_frame)
        if blob is None:
            return

        x, y, w, h = blob
        cu, cv_ = x + w / 2.0, y + h / 2.0

        image_height = color_frame.shape[0]
        row = min(4, max(1, int(cv_ / (image_height / 4)) + 1))
        self.row_pub.publish(Int32(data=row))

        du, dv = int(round(cu)), int(round(cv_))
        now = rclpy.time.Time.from_msg(color_msg.header.stamp)
        depth, info, frame_id, source = self._resolve_depth(du, dv, now)
        if depth is None:
            self.get_logger().warn(
                'target blob found but no depth camera or LiDAR reading '
                'available to estimate distance, skipping this frame')
            return

        fx = info.k[0]
        fy = info.k[4]
        cx = info.k[2]
        cy = info.k[5]

        point_x = (du - cx) * depth / fx
        point_y = (dv - cy) * depth / fy
        point_z = depth

        msg = PointStamped()
        msg.header.stamp = color_msg.header.stamp
        msg.header.frame_id = frame_id
        msg.point.x, msg.point.y, msg.point.z = point_x, point_y, point_z
        self.point_pub.publish(msg)
        self.get_logger().info(
            f'target_book_point via {source}: '
            f'({point_x:.2f}, {point_y:.2f}, {point_z:.2f}) in {frame_id}',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = BookDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
