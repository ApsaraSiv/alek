"""
bin_detector.py

setup.py and solution.launch.py already reference erc_solution.bin_detector
(ModuleNotFoundError on launch -- the file itself was never committed). This
fills that gap with a placeholder, in the same spirit as book_detector.py's
TODOs, so the pipeline can reach PLACE and manipulation_node can be
exercised end to end while real bin perception doesn't exist yet.

TODO (bin perception owner): replace the hardcoded point below with a real
detection -- back-project the bin opening's centroid from a camera + depth,
TF to base_link, and publish that instead. Contract (INTERFACES.md):
/erc/collection_bin_point, geometry_msgs/PointStamped, frame_id=base_link.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class BinDetector(Node):
    def __init__(self):
        super().__init__('bin_detector')
        self.declare_parameter('placeholder_x', 0.5)
        self.declare_parameter('placeholder_y', -0.4)
        self.declare_parameter('placeholder_z', 0.9)

        self.point_pub = self.create_publisher(
            PointStamped, '/erc/collection_bin_point', 10)
        self.create_timer(1.0, self._publish_placeholder)

        self.get_logger().warn(
            'bin_detector is a PLACEHOLDER -- publishing a hardcoded '
            'collection_bin_point, not a real detection. Replace once bin '
            'perception exists.')

    def _publish_placeholder(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x = self.get_parameter('placeholder_x').value
        msg.point.y = self.get_parameter('placeholder_y').value
        msg.point.z = self.get_parameter('placeholder_z').value
        self.point_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BinDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
