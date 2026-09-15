import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'

MARKER_BAND_TOP_FRACTION = 0.20
MARKER_BAND_BOTTOM_FRACTION = 0.50
SAVE_INTERVAL_SEC = 2.0
CENTER_TOLERANCE_FRACTION = 0.06  # matches state_machine_node's CENTER_LOCK_TOLERANCE.
                                   # Was 0.035 -- live testing showed the state
                                   # machine's tracking correction (even after
                                   # speeding it up) rarely lands two
                                   # consecutive OCR frames inside that tight a
                                   # band under this sim's wheel slip, causing
                                   # column-lock to keep timing out and
                                   # restarting the search sweep.
REQUIRED_CENTERED_FRAMES = 2
GLYPH_THRESHOLD = 160
MIN_GLYPH_AREA = 80
NORMALIZED_GLYPH_SIZE = (64, 96)
# 0.27/0.06 (first values tried) worked for digits 1/3/5 but never passed for
# digit "2" at all -- measured directly: captured 15 live frames of the real
# "2" marker mid-search and ran this exact scoring offline. At the range
# SEEK_COLUMN operates from, the live glyph crop is only ~20x19px (vs. the
# template's native ~111x133px), and "2"'s curved strokes lose far more
# shape fidelity than "1"'s straight stroke at that resolution -- its best
# score consistently landed at 0.35-0.44, never once under 0.27, even though
# it was *still always the correct top-ranked digit* (large margin over the
# 2nd-best candidate in nearly every frame). Tried smoothing the resize
# (INTER_AREA on a continuous-valued mask instead of hard threshold +
# INTER_NEAREST) first -- barely moved the score, confirming this is a real
# pixel-resolution limit, not a quantization artifact. Widened both
# thresholds to comfortably cover digit "2"'s observed range while staying
# below the ~0.5+ scores seen for genuinely wrong/junk candidates.
MAX_TEMPLATE_DIFFERENCE = 0.40
MIN_TEMPLATE_MARGIN = 0.03

MAX_DIGIT_WIDTH_FRACTION = 0.15
MAX_DIGIT_HEIGHT_FRACTION = 0.25
MIN_DIGIT_HEIGHT_PX = 12
DARK_VALUE_MAX = 160
MAX_BLACK_INK_SATURATION = 70
MIN_LOW_SATURATION_INK_FRACTION = 0.75
PLATE_VALUE_MIN = 220
PLATE_SATURATION_MAX = 45

MIN_BRIGHT_PLATE_FRACTION = 0.18


def _normalize_glyph(mask):
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    crop = mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    return cv2.resize(
        crop.astype(np.uint8), NORMALIZED_GLYPH_SIZE,
        interpolation=cv2.INTER_NEAREST).astype(bool)


def _load_digit_templates():
    texture_dir = os.path.join(
        get_package_share_directory('erc_description'),
        'models', 'number_marker', 'textures')
    templates = {}
    for digit in range(1, 6):
        image = cv2.imread(
            os.path.join(texture_dir, f'{digit}.png'), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f'Unable to load marker template {digit}')
        templates[digit] = _normalize_glyph(image < GLYPH_THRESHOLD)
    return templates


def _box_is_on_number_plate(frame, left, top, right, bottom):
    """Reject OCR boxes produced by coloured book spines or large structures."""
    height, width = frame.shape[:2]
    box_width = right - left
    box_height = bottom - top
    if (box_width <= 0 or box_height < MIN_DIGIT_HEIGHT_PX
            or box_width > width * MAX_DIGIT_WIDTH_FRACTION
            or box_height > height * MAX_DIGIT_HEIGHT_FRACTION):
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    digit = hsv[top:bottom, left:right]
    dark_pixels = digit[:, :, 2] <= DARK_VALUE_MAX
    if not dark_pixels.any():
        return False

    # Real printed ink is black/grey (low saturation). Red, green and blue
    # book spines remain highly saturated even when grayscale thresholding
    # makes them look black to Tesseract.
    low_saturation_ink = digit[:, :, 1][dark_pixels] <= MAX_BLACK_INK_SATURATION
    if low_saturation_ink.mean() < MIN_LOW_SATURATION_INK_FRACTION:
        return False

    pad_x = max(8, int(box_width * 1.5))
    pad_y = max(5, int(box_height * 0.35))
    x0, x1 = max(0, left - pad_x), min(width, right + pad_x)
    y0, y1 = max(0, top - pad_y), min(height, bottom + pad_y)
    surround = hsv[y0:y1, x0:x1]
    bright_plate = ((surround[:, :, 2] >= PLATE_VALUE_MIN)
                    & (surround[:, :, 1] <= PLATE_SATURATION_MAX))
    return bright_plate.mean() >= MIN_BRIGHT_PLATE_FRACTION


class ShelfNumberDetector(Node):

    def __init__(self):
        super().__init__('shelf_number_detector')
        self.declare_parameter('target_shelf_column_number', 0)
        self.target = int(self.get_parameter('target_shelf_column_number').value)

        self.bridge = CvBridge()
        self.digit_templates = _load_digit_templates()
        self.column_pub = self.create_publisher(
            Int32, '/erc/shelf_column_identification', 10)
        self.column_error_pub = self.create_publisher(
            Float32, '/erc/shelf_column_horizontal_error', 10)
        # OCR is slower than the camera. A deep queue makes steering react
        # to frames captured several seconds ago and causes repeated overshoot.
        camera_qos = QoSProfile(depth=1)
        camera_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, camera_qos)

        # Shared top-level folder, same convention as book_color_detector.py.
        self.save_dir = '/opt/erc_ws/src/erc_images'
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.centered_frame_count = 0

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
        dark_mask = (gray < GLYPH_THRESHOLD).astype(np.uint8)
        component_count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(dark_mask))

        detections = []
        for label_id in range(1, component_count):
            left, local_top, box_width, box_height, area = stats[label_id]
            if (area < MIN_GLYPH_AREA or box_height < MIN_DIGIT_HEIGHT_PX
                    or box_width <= 0):
                continue
            top = band_top + local_top
            right = left + box_width
            bottom = top + box_height
            if not _box_is_on_number_plate(frame, left, top, right, bottom):
                continue

            component = labels[
                local_top:local_top + box_height,
                left:left + box_width] == label_id
            normalized = _normalize_glyph(component)
            if normalized is None:
                continue
            scores = sorted(
                (float(np.mean(normalized != template)), digit)
                for digit, template in self.digit_templates.items())
            best_score, digit = scores[0]
            margin = scores[1][0] - best_score
            if (best_score <= MAX_TEMPLATE_DIFFERENCE
                    and margin >= MIN_TEMPLATE_MARGIN):
                detections.append(
                    (digit, int(left), int(top), int(right), int(bottom),
                     best_score))

        centre_x = width / 2
        centre_tolerance = width * CENTER_TOLERANCE_FRACTION

        annotated = frame.copy()
        target_found = False
        for digit, left, top, right, bottom, score in detections:
            box_centre = (left + right) / 2
            near_centre = abs(box_centre - centre_x) <= centre_tolerance
            is_target = digit == self.target and near_centre
            if digit == self.target:
                normalized_error = (box_centre - centre_x) / width
                self.column_error_pub.publish(Float32(data=float(normalized_error)))
            colour = (0, 255, 0) if is_target else (0, 0, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), colour, 2)
            cv2.putText(annotated, str(digit), (left, min(height - 5, bottom + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if is_target:
                target_found = True

        self.centered_frame_count = (
            self.centered_frame_count + 1 if target_found else 0)
        confirmed = self.centered_frame_count >= REQUIRED_CENTERED_FRAMES

        
        if confirmed:
            self.column_pub.publish(Int32(data=self.target))

        now = time.time()
        if confirmed and now - self.last_save_time >= SAVE_INTERVAL_SEC:
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
