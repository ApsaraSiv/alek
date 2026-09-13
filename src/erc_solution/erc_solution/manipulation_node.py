"""
manipulation_node.py

Right-arm pick-and-place for the ERC 2026 Phase 1 solution. Implements the
Manipulation side of the erc_solution service contract (see INTERFACES.md):
exposes /erc/grasp_book and /erc/place_in_bin, called by state_machine_node.

Talks directly to move_group via the low-level moveit_msgs/action/MoveGroup
action interface (moveit_py is not installed in this image). MoveIt config
lives in src/tiago_pro_right_arm_moveit_config (Setup Assistant-generated,
since the base repo ships no SRDF/MoveIt config despite claiming one).
Confirmed working values:
  - planning group: arm_right
  - end-effector link: arm_right_tool_link
  - planning frame: base_footprint
  - gripper: /gripper_right_controller/joint_trajectory,
    joint gripper_right_finger_joint

Ported from this team's aleksandria/arm_manipulation_node.py standalone
prototype (verified end-to-end against move_group in Gazebo -- see that
package for the original dev/test harness). The prototype ran its sequence
autonomously off a single pose topic; this version is restructured as
service handlers callable from state_machine_node, consuming the contract's
/erc/target_book_point and /erc/collection_bin_point instead.

Runs under a MultiThreadedExecutor with a ReentrantCallbackGroup so a
blocking service call (grasp_book/place_in_bin) doesn't starve the
move_group action client's own callbacks or the point subscriptions --
same pattern state_machine_node and navigation_node use. Blocking on a
future is done with a plain poll loop (not spin_until_future_complete,
which tries to attach this node to a second executor and fails since the
node is already spinning under executor.spin()).

STILL TO TUNE:
  - approach_offset (dx, dy, dz): vector from the target point to a safe
    hover point before/after grasping / placing. Books sit on a shelf and
    the bin has its own opening geometry, so these are likely different in
    practice -- currently sharing one offset for both, revisit once
    book_detector/bin_detector are real and we can see actual geometry.
  - request.row (GraspBook) is accepted (contract requires it) but not
    used yet -- book height comes entirely from /erc/target_book_point's
    z-coordinate. If arm_right can't reach all shelf rows once perception
    is real and this gets tested end to end, add a torso-lift step keyed
    on row here.

TEST TIP: with move_group running, publish a fake point and call the
service directly (no perception, no state machine, needed):
  ros2 topic pub /erc/target_book_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: 'base_footprint'}, point: {x: 0.6, y: 0.0, z: 0.9}}"
  ros2 service call /erc/grasp_book erc_interfaces/srv/GraspBook "{row: 1}"
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
)

from erc_interfaces.srv import GraspBook, PlaceInBin

import tf2_ros
from tf2_geometry_msgs import do_transform_point

MOVEIT_SUCCESS = 1  # moveit_msgs/msg/MoveItErrorCodes.SUCCESS


class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])

        # --- Tunable parameters (confirmed names from the team's MoveIt setup) ---
        self.declare_parameter('planning_group', 'arm_right')
        self.declare_parameter('eef_link', 'arm_right_tool_link')
        self.declare_parameter('planning_frame', 'base_footprint')
        self.declare_parameter('approach_dx', -0.15)
        self.declare_parameter('approach_dy', 0.0)
        self.declare_parameter('approach_dz', 0.0)
        self.declare_parameter('gripper_open', 0.04)
        self.declare_parameter('gripper_closed', 0.0)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('orientation_tolerance', 0.05)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('point_wait_timeout', 5.0)

        self.planning_group = self.get_parameter('planning_group').value
        self.eef_link = self.get_parameter('eef_link').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.approach_offset = (
            self.get_parameter('approach_dx').value,
            self.get_parameter('approach_dy').value,
            self.get_parameter('approach_dz').value,
        )
        self.gripper_open = self.get_parameter('gripper_open').value
        self.gripper_closed = self.get_parameter('gripper_closed').value
        self.pos_tol = self.get_parameter('position_tolerance').value
        self.orient_tol = self.get_parameter('orientation_tolerance').value
        self.planning_time = self.get_parameter('planning_time').value
        self.point_wait_timeout = self.get_parameter('point_wait_timeout').value

        cb_group = ReentrantCallbackGroup()

        # --- MoveGroup action client (talks to the already-running move_group) ---
        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=cb_group)

        # --- Gripper: direct topic publish, matching the controller interface ---
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_right_controller/joint_trajectory', 10)

        # --- TF for transforming perception output into the planning frame ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Perception input (contract: geometry_msgs/PointStamped, frame base_link) ---
        self.latest_book_point = None
        self.latest_bin_point = None
        self.create_subscription(
            PointStamped, '/erc/target_book_point', self._on_book_point, 10,
            callback_group=cb_group)
        self.create_subscription(
            PointStamped, '/erc/collection_bin_point', self._on_bin_point, 10,
            callback_group=cb_group)

        # --- Manipulation service contract (state_machine_node calls these) ---
        self.create_service(
            GraspBook, '/erc/grasp_book', self._on_grasp_book, callback_group=cb_group)
        self.create_service(
            PlaceInBin, '/erc/place_in_bin', self._on_place_in_bin, callback_group=cb_group)

        self.get_logger().info(
            'Manipulation node ready (right arm) -- waiting for grasp_book/place_in_bin calls')

    # ------------------------------------------------------------------
    def _on_book_point(self, msg: PointStamped):
        self.latest_book_point = msg

    def _on_bin_point(self, msg: PointStamped):
        self.latest_bin_point = msg

    def _wait_for_point(self, attr_name):
        deadline = time.monotonic() + self.point_wait_timeout
        while time.monotonic() < deadline:
            point = getattr(self, attr_name)
            if point is not None:
                return point
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------
    def transform_to_planning_frame(self, point_stamped: PointStamped):
        if point_stamped.header.frame_id == self.planning_frame:
            return point_stamped
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame, point_stamped.header.frame_id, rclpy.time.Time())
            return do_transform_point(point_stamped, transform)
        except Exception as e:
            self.get_logger().warn(f'TF transform to {self.planning_frame} failed: {e}')
            return None

    def make_pose(self, x, y, z, frame_id, qx=0.0, qy=0.0, qz=0.0, qw=1.0) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.x = qx
        p.pose.orientation.y = qy
        p.pose.orientation.z = qz
        p.pose.orientation.w = qw
        return p

    def point_to_pose(self, point_stamped: PointStamped) -> PoseStamped:
        return self.make_pose(
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z,
            point_stamped.header.frame_id)

    def offset_pose(self, pose_stamped: PoseStamped, dx, dy, dz) -> PoseStamped:
        return self.make_pose(
            pose_stamped.pose.position.x + dx,
            pose_stamped.pose.position.y + dy,
            pose_stamped.pose.position.z + dz,
            pose_stamped.header.frame_id,
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w,
        )

    # ------------------------------------------------------------------
    def pose_to_constraints(self, pose_stamped: PoseStamped, link_name: str) -> Constraints:
        constraints = Constraints()

        pos_constraint = PositionConstraint()
        pos_constraint.header = pose_stamped.header
        pos_constraint.link_name = link_name
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.pos_tol]
        pos_constraint.constraint_region.primitives.append(sphere)
        pos_constraint.constraint_region.primitive_poses.append(pose_stamped.pose)
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        orient_constraint = OrientationConstraint()
        orient_constraint.header = pose_stamped.header
        orient_constraint.link_name = link_name
        orient_constraint.orientation = pose_stamped.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = self.orient_tol
        orient_constraint.absolute_y_axis_tolerance = self.orient_tol
        orient_constraint.absolute_z_axis_tolerance = self.orient_tol
        orient_constraint.weight = 1.0
        constraints.orientation_constraints.append(orient_constraint)

        return constraints

    @staticmethod
    def _block_on_future(future, timeout_sec=30.0, poll=0.02):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(poll)
        return future.result()

    def move_to_pose(self, pose_stamped: PoseStamped) -> bool:
        """Blocking helper: plan AND execute a move to the given pose. Safe to
        call from a service callback under MultiThreadedExecutor -- polls the
        futures instead of calling spin_until_future_complete (see module
        docstring for why)."""
        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.goal_constraints = [
            self.pose_to_constraints(pose_stamped, self.eef_link)
        ]
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = self.planning_time
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5

        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False  # plan AND execute

        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/move_action server not available')
            return False

        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        goal_handle = self._block_on_future(send_goal_future)

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('MoveGroup goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        result_wrapper = self._block_on_future(result_future)
        if result_wrapper is None:
            self.get_logger().warn('MoveGroup result timed out')
            return False
        result = result_wrapper.result

        if result.error_code.val == MOVEIT_SUCCESS:
            return True
        self.get_logger().warn(
            f'MoveGroup failed, error code {result.error_code.val} '
            '(see moveit_msgs/msg/MoveItErrorCodes.msg for what it means)')
        return False

    def set_gripper(self, opening: float, duration_sec: float = 1.0):
        msg = JointTrajectory()
        msg.joint_names = ['gripper_right_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [opening]
        point.time_from_start.sec = int(duration_sec)
        msg.points = [point]
        self.gripper_pub.publish(msg)
        time.sleep(duration_sec + 0.5)  # crude wait for the motion to finish

    # ------------------------------------------------------------------
    def _on_grasp_book(self, request, response):
        book_point = self._wait_for_point('latest_book_point')
        if book_point is None:
            response.success = False
            response.message = 'timed out waiting for /erc/target_book_point'
            return response

        book_point = self.transform_to_planning_frame(book_point)
        if book_point is None:
            response.success = False
            response.message = f'TF transform of target_book_point to {self.planning_frame} failed'
            return response

        book_pose = self.point_to_pose(book_point)
        dx, dy, dz = self.approach_offset
        pregrasp = self.offset_pose(book_pose, dx, dy, dz)

        if not self.move_to_pose(pregrasp):
            response.success = False
            response.message = 'failed to plan/move to pre-grasp hover pose'
            return response

        self.set_gripper(self.gripper_open)

        if not self.move_to_pose(book_pose):
            response.success = False
            response.message = 'failed to plan/move to grasp pose'
            return response

        self.set_gripper(self.gripper_closed)
        self.move_to_pose(pregrasp)  # retreat clear of the shelf; best-effort

        response.success = True
        response.message = 'book grasped'
        return response

    def _on_place_in_bin(self, request, response):
        bin_point = self._wait_for_point('latest_bin_point')
        if bin_point is None:
            response.success = False
            response.message = 'timed out waiting for /erc/collection_bin_point'
            return response

        bin_point = self.transform_to_planning_frame(bin_point)
        if bin_point is None:
            response.success = False
            response.message = f'TF transform of collection_bin_point to {self.planning_frame} failed'
            return response

        bin_pose = self.point_to_pose(bin_point)
        dx, dy, dz = self.approach_offset
        hover = self.offset_pose(bin_pose, dx, dy, dz)

        if not self.move_to_pose(hover):
            response.success = False
            response.message = 'failed to plan/move above the bin'
            return response

        if not self.move_to_pose(bin_pose):
            response.success = False
            response.message = 'failed to plan/move down into the bin'
            return response

        self.set_gripper(self.gripper_open)
        self.move_to_pose(hover)  # retreat; best-effort

        response.success = True
        response.message = 'book placed in bin'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
