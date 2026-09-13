import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from erc_interfaces.srv import GraspBook, PlaceInBin


class ManipulationNode(Node):

    def __init__(self):
        super().__init__('manipulation_node')
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_left_controller/joint_trajectory', 10)
        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)

        # just wiring for now, no actual motion - lets state_machine_node
        # call a real service instead of one that doesn't exist yet
        self.create_service(GraspBook, '/erc/grasp_book', self._on_grasp_book)
        self.create_service(PlaceInBin, '/erc/place_in_bin', self._on_place_in_bin)

    def _on_grasp_book(self, request, response):
        # TODO:
        # 1. raise torso to the lift height for request.row
        # 2. move arm to the pre-grasp config for request.row (+ offset
        #    from /erc/target_book_point once book_color_detector publishes it)
        # 3. close gripper using the effort-feedback closed loop
        # 4. check the grasp actually worked (contacts or gripper position)
        response.success = False
        response.message = 'grasp_book not implemented yet'
        return response

    def _on_place_in_bin(self, request, response):
        # TODO: position over /erc/collection_bin_point once bin_detector
        # exists, lower, open gripper. gently placed = +4 vs dropped = +2,
        # worth doing properly.
        response.success = False
        response.message = 'place_in_bin not implemented yet'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
