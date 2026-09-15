#!/usr/bin/env python3
"""
arm_manipulation_node.py

Right-arm pick-and-place node for the ERC 2026 Simulation Phase.

Talks directly to the running `move_group` node via the low-level
moveit_msgs/action/MoveGroup action interface (NOT moveit_py, which
is not installed in this image -- confirmed via
`python3 -c "from moveit.planning import MoveItPy"` failing with
ModuleNotFoundError).

Confirmed working values from this team's own MoveIt Setup Assistant
run (see chat history):
  - planning group: arm_right
  - end-effector link: arm_right_tool_link
  - base frame / virtual joint child link: base_footprint
  - gripper controller joint (from repo README): gripper_right_finger_joint

Assumes a separate perception node publishes the target book's pose
(geometry_msgs/PoseStamped) on /erc/target_book_pose. This node then:
  1. Moves the right arm to a pre-grasp "hover" pose in front of the book
  2. Opens the gripper
  3. Moves in to the grasp pose
  4. Closes the gripper on the book
  5. Retreats to the hover pose (lifting the book clear of the shelf)
  6. Moves to the collection bin drop pose
  7. Opens the gripper to release the book

STILL TO TUNE (same as before -- unrelated to the MoveIt backend swap):
  - approach_offset (dx, dy, dz): the vector (in base_frame) from the
    book's pose to a safe hover point before/after grasping. Books sit
    on a shelf, so this is likely a horizontal standoff, not vertical.
  - bin_pose: fixed pose in base_frame for the collection bin. Measure
    this once and hard-code it.

TEST TIP: before wiring this to real perception, publish a fake pose
by hand and watch it work in isolation:
  ros2 topic pub /erc/target_book_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'base_footprint'}, pose: {position: {x: 0.6, y: 0.0, z: 0.9}, orientation: {w: 1.0}}}"
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Vector3
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
from tf2_geometry_msgs import do_transform_pose_stamped

# MoveItErrorCodes.SUCCESS
MOVEIT_SUCCESS = 1


class ArmManipulationNode(Node):
    def __init__(self):
        super().__init__('arm_manipulation_node')

        # --- Tunable parameters (confirmed names from your own setup) ---
        self.declare_parameter('planning_group', 'arm_right')
        self.declare_parameter('eef_link', 'arm_right_tool_link')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('approach_dx', -0.15)
        self.declare_parameter('approach_dy', 0.0)
        self.declare_parameter('approach_dz', 0.0)
        self.declare_parameter('gripper_open', 0.04)
        self.declare_parameter('gripper_closed', 0.0)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('orientation_tolerance', 0.05)
        self.declare_parameter('planning_time', 5.0)
        # Placeholder -- measure the real bin pose in base_frame and set these.
        self.declare_parameter('bin_x', 0.5)
        self.declare_parameter('bin_y', -0.4)
        self.declare_parameter('bin_z', 0.9)

        self.planning_group = self.get_parameter('planning_group').value
        self.eef_link = self.get_parameter('eef_link').value
        self.base_frame = self.get_parameter('base_frame').value
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
        self.bin_pose = self.make_pose(
            self.get_parameter('bin_x').value,
            self.get_parameter('bin_y').value,
            self.get_parameter('bin_z').value,
            self.base_frame,
        )

        # --- MoveGroup action client (talks to the already-running move_group) ---
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')

        # --- Gripper: direct topic publish, matching the controller interface ---
        # NOTE: the spawned controller is gripper_right_controller_raw (see
        # erc_bringup/config/controller_params.yaml) -- publishing to the
        # non-_raw topic silently goes nowhere and the gripper never moves.
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_right_controller_raw/joint_trajectory', 10)

        # --- TF for transforming perception output into base_frame ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Perception input ---
        # Latest transformed book pose available to a grasp_book call. Kept
        # up to date continuously (not latched to the first message) so
        # repeated calls across different rows/books pick up fresh poses.
        self.latest_book_pose = None
        self.create_subscription(
            PoseStamped, '/erc/target_book_pose', self.book_pose_callback, 10)

        # --- Service interface used by state_machine_node ---
        # Calls block on MoveGroup goals (spin_until_future_complete), so
        # these must run on their own callback group under a
        # MultiThreadedExecutor -- see main() below.
        cb_group = ReentrantCallbackGroup()
        self.create_service(
            GraspBook, '/erc/grasp_book', self._on_grasp_book, callback_group=cb_group)
        self.create_service(
            PlaceInBin, '/erc/place_in_bin', self._on_place_in_bin, callback_group=cb_group)

        # Retained across the grasp_book -> place_in_bin call pair so the
        # arm knows where to retreat from before heading to the bin.
        self._pregrasp_pose = None

        self.get_logger().info(
            'Arm manipulation node ready — /erc/grasp_book and /erc/place_in_bin online')

    # ------------------------------------------------------------------
    def make_pose(self, x, y, z, frame_id, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
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

    def transform_to_base(self, pose_stamped: PoseStamped):
        if pose_stamped.header.frame_id == self.base_frame:
            return pose_stamped
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, pose_stamped.header.frame_id, rclpy.time.Time())
            return do_transform_pose_stamped(pose_stamped, transform)
        except Exception as e:
            self.get_logger().warn(f'TF transform to {self.base_frame} failed: {e}')
            return None

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

    def move_to_pose(self, pose_stamped: PoseStamped) -> bool:
        """Blocking helper: plan AND execute a move to the given pose.
        Must be called from the main thread (not from inside a
        subscription callback) -- see note in main() below."""
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

        self.move_group_client.wait_for_server()
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('MoveGroup goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

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
    def book_pose_callback(self, msg: PoseStamped):
        """Keeps the latest transformed book pose available for the next
        grasp_book service call. Runs on the node's default callback
        group; transform_to_base() is non-blocking (a single TF lookup),
        so this is safe to do inline."""
        book_pose = self.transform_to_base(msg)
        if book_pose is None:
            return
        self.latest_book_pose = book_pose

    # ------------------------------------------------------------------
    # Service handlers. These run under a ReentrantCallbackGroup /
    # MultiThreadedExecutor (see main()) so their blocking move_to_pose()
    # calls don't deadlock the executor the way they would under a single
    # spin() thread.
    def _on_grasp_book(self, request, response):
        if self.latest_book_pose is None:
            response.success = False
            response.message = 'no target book pose available on /erc/target_book_pose'
            return response

        book_pose = self.latest_book_pose
        dx, dy, dz = self.approach_offset
        pregrasp = self.offset_pose(book_pose, dx, dy, dz)

        # 1. Hover in front of the book
        if not self.move_to_pose(pregrasp):
            response.success = False
            response.message = f'failed to plan/execute pregrasp move for row {request.row}'
            return response

        # 2. Open gripper before moving in
        self.set_gripper(self.gripper_open)

        # 3. Move in to the book
        if not self.move_to_pose(book_pose):
            response.success = False
            response.message = f'failed to plan/execute grasp move for row {request.row}'
            return response

        # 4. Close gripper on the book
        self.set_gripper(self.gripper_closed)

        # 5. Retreat with the book, keeping the pregrasp pose around so
        # place_in_bin can be reasoned about relative to it if needed later.
        if not self.move_to_pose(pregrasp):
            response.success = False
            response.message = f'failed to retreat after grasping row {request.row}'
            return response
        self._pregrasp_pose = pregrasp

        response.success = True
        response.message = f'grasped book at row {request.row}'
        return response

    def _on_place_in_bin(self, request, response):
        # 6. Move to the bin
        if not self.move_to_pose(self.bin_pose):
            response.success = False
            response.message = 'failed to plan/execute move to collection bin'
            return response

        # 7. Release the book
        self.set_gripper(self.gripper_open)

        response.success = True
        response.message = 'book placed in bin'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ArmManipulationNode()
    # MultiThreadedExecutor lets the blocking move_to_pose() calls inside
    # a service callback run without stalling the /erc/target_book_pose
    # subscription (same pattern as state_machine_node).
    executor = rclpy.executors.MultiThreadedExecutor()
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
