import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped


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

    def _on_image(self, msg: Image):
        # TODO:
        # 1. cv_bridge to cv2, HSV threshold for target_book_colour
        # 2. find largest matching contour -> bbox, pick row from y-position
        # 3. once confident, publish row number
        # 4. back-project centroid -> 3D via depth + camera_info, TF to
        #    base_link, publish on /erc/target_book_point
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
