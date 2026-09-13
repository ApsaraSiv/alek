import os
import time

import cv2
import pytesseract
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
# Vertical band (as a fraction of image height) where the marker plate
# lands at navigation's wide-scan standoff/tilt (see WIDE_SHELF_STANDOFF
# and MARKER_SCAN_HEAD_TILT in navigation_node.py) - eyeballed off a real
# captured frame, not calculated.
MARKER_BAND_TOP_FRACTION = 0.20
MARKER_BAND_BOTTOM_FRACTION = 0.50
SAVE_INTERVAL_SEC = 2.0


class ShelfNumberDetector(Node):

    def __init__(self):
        super().__init__('shelf_number_detector')
        self.declare_parameter('target_shelf_column_number', 0)
        self.target = int(self.get_parameter('target_shelf_column_number').value)

        self.bridge = CvBridge()
        self.column_pub = self.create_publisher(
            Int32, '/erc/shelf_column_identification', 10)
        self.create_subscription(Image, CAMERA_TOPIC, self.image_callback, 10)

        # Shared top-level folder, same convention as book_color_detector.py.
        self.save_dir = '/opt/erc_ws/src/erc_images'
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.target_currently_visible = False

        self.get_logger().info(
            f'Subscribed to {CAMERA_TOPIC}, target_shelf_column_number={self.target}')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert camera image, skipping frame: {exc}')
            return

        height, width = frame.shape[:2]
        band_top = max(0, int(height * MARKER_BAND_TOP_FRACTION))
        band_bottom = min(height, int(height * MARKER_BAND_BOTTOM_FRACTION))
        band = frame[band_top:band_bottom, :]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        try:
            boxes = pytesseract.image_to_boxes(
                thresh, config='--psm 6 -c tessedit_char_whitelist=12345')
        except Exception as exc:
            self.get_logger().warn(f'OCR failed, skipping frame: {exc}')
            boxes = ''

        # only trust a marker near the horizontal centre - wide FOV means
        # the NEXT column's marker can peek in at the edge and falsely
        # confirm a match while parked at the wrong slot (saw this happen:
        # target "1" showed up at the frame edge from a different slot)
        centre_x = width / 2
        centre_tolerance = width * 0.2

        annotated = frame.copy()
        target_found = False
        for line in boxes.splitlines():
            parts = line.split()
            if len(parts) != 6 or not parts[0].isdigit():
                continue
            digit, left, _bottom, right, _top, _page = parts
            left, right = int(left), int(right)
            box_centre = (left + right) / 2
            near_centre = abs(box_centre - centre_x) <= centre_tolerance
            is_target = int(digit) == self.target and near_centre
            colour = (0, 255, 0) if is_target else (0, 0, 255)
            cv2.rectangle(annotated, (left, band_top), (right, band_bottom), colour, 2)
            cv2.putText(annotated, digit, (left, band_bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if is_target:
                target_found = True

        # Publish once per detection streak rather than every frame.
        if target_found and not self.target_currently_visible:
            self.column_pub.publish(Int32(data=self.target))
        self.target_currently_visible = target_found

        now = time.time()
        if target_found and now - self.last_save_time >= SAVE_INTERVAL_SEC:
            self.last_save_time = now
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            cv2.putText(annotated, timestamp, (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            filename = os.path.join(self.save_dir, f'shelf_{timestamp}.png')
            cv2.imwrite(filename, annotated)
            self.get_logger().info(f'Saved {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = ShelfNumberDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
