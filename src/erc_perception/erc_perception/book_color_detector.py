import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'

# HSV lower/upper bounds for each book colour. HSV = Hue/Saturation/Value,
# a colour format that makes "find all red pixels" much easier than RGB.
COLOUR_RANGES = {
    'red': [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    'blue': [((100, 100, 60), (130, 255, 255))],
    'green': [((40, 70, 60), (85, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
}

BOX_COLOUR_BGR = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
}

MIN_BLOB_AREA = 200
SAVE_INTERVAL_SEC = 2.0


CAMERA_TIMEOUT_SEC = 5.0


class BookColorDetector(Node):
    def __init__(self):
        super().__init__('book_color_detector')
        self.declare_parameter('target_colour', 'red')
        target_colour = self.get_parameter('target_colour').value
        if target_colour not in COLOUR_RANGES:
            self.get_logger().error(
                f'Unknown target_colour {target_colour!r}, must be one of '
                f'{list(COLOUR_RANGES)}. Defaulting to "red".'
            )
            target_colour = 'red'
        self.target_colour = target_colour

        # GUI windows require a display; disable for headless evaluation runs.
        self.declare_parameter('enable_display', False)
        self.enable_display = bool(self.get_parameter('enable_display').value)
        self.display_available = self.enable_display

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.row_publisher = self.create_publisher(Int32, '/erc/shelf_row_identification', 10)

        # Shared top-level folder (sibling of all src/ packages) so images
        # land in the git repo on the host, not in the container-only
        # build/ directory, and aren't buried inside one specific package.
        self.save_dir = '/opt/erc_ws/src/erc_images'
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.target_currently_visible = False

        self.last_frame_time = None
        self.create_timer(1.0, self.check_camera_alive)

        self.get_logger().info(
            f'Subscribed to {CAMERA_TOPIC}, saving images to {self.save_dir}, '
            f'target_colour={self.target_colour}'
        )

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
        image_height = frame.shape[0]

        target_box = None  # (x, y, w, h) of the largest target-coloured blob

        for colour_name, ranges in COLOUR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < MIN_BLOB_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                colour = BOX_COLOUR_BGR[colour_name]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(annotated, colour_name, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

                if colour_name == self.target_colour:
                    if target_box is None or w * h > target_box[2] * target_box[3]:
                        target_box = (x, y, w, h)

        if target_box is not None:
            x, y, w, h = target_box
            # Shelf rows are numbered 1 (top) to 4 (bottom); estimate which
            # quarter of the image the blob's vertical centre falls in.
            centre_y = y + h / 2
            row = min(4, max(1, int(centre_y / (image_height / 4)) + 1))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cv2.putText(annotated, f'TARGET row {row}', (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Publish once per detection streak rather than every frame, so
            # the topic isn't flooded while the book stays in view.
            if not self.target_currently_visible:
                self.row_publisher.publish(Int32(data=row))
            self.target_currently_visible = True
        else:
            self.target_currently_visible = False

        if self.display_available:
            try:
                cv2.imshow('Book colour detection', annotated)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().warn(f'Display unavailable, disabling further preview windows: {exc}')
                self.display_available = False

        now = time.time()
        if target_box is not None and now - self.last_save_time >= SAVE_INTERVAL_SEC:
            self.last_save_time = now
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            cv2.putText(annotated, timestamp, (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            filename = os.path.join(self.save_dir, f'books_{timestamp}.png')
            cv2.imwrite(filename, annotated)
            self.get_logger().info(f'Saved {filename}')


def main():
    rclpy.init()
    node = BookColorDetector()
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
