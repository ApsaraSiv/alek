import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
CAMERA_TIMEOUT_SEC = 5.0

# Same red HSV range used for books - the bin and books share this colour,
# so shape/size below is what actually tells them apart.
RED_RANGES = [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))]

# The bin is a wide box (50cm w x 21cm h), so its aspect ratio (h/w) is well
# below 1, opposite of a book standing on a shelf (h/w well above 1).
MIN_BIN_AREA = 2500
MAX_BIN_ASPECT_RATIO = 0.75
SAVE_INTERVAL_SEC = 2.0


class BinDetector(Node):
    def __init__(self):
        super().__init__('bin_detector')

        self.declare_parameter('enable_display', False)
        self.enable_display = bool(self.get_parameter('enable_display').value)
        self.display_available = self.enable_display

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.bin_publisher = self.create_publisher(Bool, '/erc/bin_identification', 10)

        # Shared top-level folder (sibling of all src/ packages) so images
        # land in the git repo on the host, not buried in one package.
        self.save_dir = '/opt/erc_ws/src/erc_images'
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.bin_currently_visible = False

        self.last_frame_time = None
        self.create_timer(1.0, self.check_camera_alive)

        self.get_logger().info('Looking for the red collection bin')

    def check_camera_alive(self):
        if self.last_frame_time is None:
            return
        elapsed = time.time() - self.last_frame_time
        if elapsed > CAMERA_TIMEOUT_SEC:
            self.get_logger().warn(
                f'No camera frames received for {elapsed:.1f}s - is {CAMERA_TOPIC} still publishing?'
            )

    def image_callback(self, msg: Image):
        self.last_frame_time = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert camera image, skipping frame: {exc}')
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        annotated = frame.copy()

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in RED_RANGES:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bin_box = None  # largest wide/squat red blob seen this frame
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / float(w)
            if aspect > MAX_BIN_ASPECT_RATIO:
                continue  # too tall/narrow to be the bin - likely a book
            if bin_box is None or w * h > bin_box[2] * bin_box[3]:
                bin_box = (x, y, w, h)

        bin_found = bin_box is not None
        if bin_found:
            x, y, w, h = bin_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(annotated, 'COLLECTION BIN', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Publish once per detection streak rather than every frame.
            if not self.bin_currently_visible:
                self.bin_publisher.publish(Bool(data=True))
        self.bin_currently_visible = bin_found

        if self.display_available:
            try:
                cv2.imshow('Bin detection', annotated)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().warn(f'Display unavailable, disabling further preview windows: {exc}')
                self.display_available = False

        now = time.time()
        if bin_found and now - self.last_save_time >= SAVE_INTERVAL_SEC:
            self.last_save_time = now
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            cv2.putText(annotated, timestamp, (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            filename = os.path.join(self.save_dir, f'bin_{timestamp}.png')
            cv2.imwrite(filename, annotated)
            self.get_logger().info(f'Saved {filename}')


def main():
    rclpy.init()
    node = BinDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
