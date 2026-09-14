import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

# solution.launch.py uses erc_perception's book_color_detector for row
# identification (the real implementation) -- this node isn't launched for
# that. It exists solely to fill the /erc/target_book_point gap: nothing
# currently publishes it (book_color_detector only does row), so
# manipulation_node's grasp_book can never proceed past waiting for a
# point. PLACEHOLDER, not real detection -- see bin_detector.py for the
# same pattern on the bin side.
PLACEHOLDER_BOOK_POINT = (0.6, 0.0, 0.9)  # base_link


class BookDetector(Node):

    def __init__(self):
        super().__init__('book_detector')
        self.declare_parameter('target_book_colour', '')

        self.row_pub = self.create_publisher(
            Int32, '/erc/shelf_row_identification', 10)
        self.point_pub = self.create_publisher(
            PointStamped, '/erc/target_book_point', 10)

        self.create_subscription(
            Image, '/head_front_camera/head_front_camera/color/image_raw',
            self._on_image, 10)
        self.create_timer(1.0, self._publish_placeholder_point)

        self.get_logger().warn(
            'book_detector is publishing a PLACEHOLDER target_book_point, '
            'not a real detection. Replace once book-point perception exists.')

    def _publish_placeholder_point(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x, msg.point.y, msg.point.z = PLACEHOLDER_BOOK_POINT
        self.point_pub.publish(msg)

    def _on_image(self, msg: Image):
        # TODO:
        # 1. cv_bridge to cv2, HSV threshold for target_book_colour
        # 2. find largest matching contour -> bbox, pick row from y-position
        # 3. once confident, publish row number
        # 4. back-project centroid -> 3D via depth + camera_info, TF to
        #    base_link, publish on /erc/target_book_point (replaces the
        #    placeholder timer above)
        # 5. cv2.imwrite annotated bbox + timestamp to /erc_images/
        pass


def main(args=None):
    rclpy.init(args=args)
    node = BookDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
