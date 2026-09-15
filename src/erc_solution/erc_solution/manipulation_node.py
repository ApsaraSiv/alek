#!/usr/bin/env python3
"""
manipulation_node.py

Right-arm pick-and-place for the ERC 2026 Phase 1 solution. Implements the
Manipulation side of the erc_solution service contract (see INTERFACES.md):
exposes /erc/grasp_book and /erc/place_in_bin, called by state_machine_node.

Talks directly to move_group via the low-level moveit_msgs/action/MoveGroup
action interface (moveit_py is not installed in this image). MoveIt config
lives in src/tiago_pro_right_arm_moveit_config.
Confirmed working values:
  - planning group: arm_right
  - end-effector link: arm_right_tool_link
  - planning frame: base_footprint
  - gripper: /gripper_right_controller_raw/joint_trajectory,
    joint gripper_right_finger_joint

This is the service-shaped wrapper the rest of the pipeline (state_machine_node,
INTERFACES.md) depends on, rebuilt on top of the pose-based grasp logic from
aleksandria/arm_manipulation_node.py (upstream's rewrite, merged in
2026-09-15 -- see that node's own history for the single-shot standalone
version this was adapted from). Differences from a straight adoption of that
node:
  - GraspBook/PlaceInBin stay separate services instead of one run-once
    sequence, so state_machine_node can drive navigate_to_bin in between --
    the upstream node's fixed bin_pose only makes sense if grasp and place
    happen back-to-back from the same base pose, which isn't true once the
    robot actually drives to the bin.
  - Subscribes to /erc/target_book_pose (geometry_msgs/PoseStamped) per
    upstream, transformed via tf2 like before; book_detector still publishes
    /erc/target_book_point (PointStamped) per INTERFACES.md's perception
    contract, so this node converts point->pose on receipt.
  - Runs under a MultiThreadedExecutor + ReentrantCallbackGroup (same as
    before) so a blocking service call doesn't starve the MoveGroup action
    client's own callbacks or the pose subscription. move_to_pose() polls
    futures instead of calling rclpy.spin_until_future_complete(), which
    tries to attach this node to a second executor and fails since it's
    already spinning under executor.spin().
  - Keeps the base-centering fix below (MAX_REACHABLE_ABS_Y etc.): arm_right
    is mounted on the right side of the torso and simply can't reach a book
    more than ~0.15m to the left of base_footprint, so grasp_book rotates
    the base in small steps to bring the book onto the arm's reachable axis
    before planning, instead of MoveGroup failing outright with "unable to
    sample any valid states for goal tree".

STILL TO TUNE:
  - approach_offset (dx, dy, dz): vector from the grasp point to a safe
    hover point before/after grasping.
  - place_in_bin measures the bin rim, bin centre and table top from the
    head camera (see _perceive_bin); it assumes the robot is already parked
    facing the bin within arm reach (navigate_to_bin's job).
  - request.row (GraspBook) is accepted (contract requires it) but not
    used yet -- book height comes entirely from /erc/target_book_pose's
    z-coordinate.

TEST TIP: with move_group running, publish a fake pose and call the
service directly (no perception, no state machine, needed):
  ros2 topic pub /erc/target_book_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'base_footprint'}, pose: {position: {x: 0.6, y: 0.0, z: 0.9}, orientation: {w: 1.0}}}"
  ros2 service call /erc/grasp_book erc_interfaces/srv/GraspBook "{row: 1}"
"""

import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath
from rclpy.duration import Duration
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
)

from erc_interfaces.srv import GraspBook, PlaceInBin

import tf2_ros
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point, do_transform_pose_stamped

MOVEIT_SUCCESS = 1  # moveit_msgs/msg/MoveItErrorCodes.SUCCESS
# Gazebo runs at ~0.13x real time on this GPU-less host; a 30s wall-clock wait
# gave up on moves that were still executing.
MOTION_TIMEOUT_SEC = 300.0
LINEAR_TIME_SCALE = 3.0  # stretch straight-line trajectories near books/bin

CAMERA_OPTICAL_FRAME = 'head_front_camera_color_optical_frame'
BOOK_SETTLE_SEC = 1.5            # s, let head/arm stop before trusting a detection
BOOK_REFRESH_TIMEOUT_SEC = 10.0
BOOK_REAIM_MAX_SHIFT = 0.05      # m, bigger jump than this = not the same book
# Pads straddle a 20mm book with ~18mm to spare each side; a book that moves
# more than this while the hand goes in is being pushed, not straddled.
BOOK_PUSH_TOLERANCE = 0.012      # m

# arm_right is mounted on the right side of the torso -- /compute_ik testing
# in Gazebo shows it can only reach roughly up to y=0.15m to the *left* of
# base_footprint (past that IK returns NO_IK_SOLUTION even with a wide-open
# orientation tolerance). book_detector's point is wherever the book actually
# sits in its shelf slot, which is often well past that, so grasp planning
# would just fail outright with "unable to sample any valid states for goal
# tree." Rotate the base to bring the book onto the arm's forward axis first.
# Grasp orientation: fingertip TF (arm_right_tool_link -> gripper_right_
# fingertip_{left,right}_link) shows the fingers sit ~0.13m out along the
# tool_link's local +Z, separated along local Y. At the identity
# orientation used previously, tool_link's local Z lines up with
# base_footprint's Z (straight *up*), not toward the shelf -- so grasp
# planning was reaching the right XYZ with the fingers pointed skyward,
# never actually closing around the book. This is a +90 deg pitch (about Y)
# so local Z -> base_footprint's +X (forward, into the book) while local Y
# (finger spread) stays on base_footprint's Y (spans the book's stacking
# direction on the shelf). Also verified via /compute_ik to reach much
# further sideways than identity did (~0.35m vs ~0.15m), since the wrist
# doesn't have to twist itself into an awkward pose to hit the position
# constraint anymore.
GRASP_ORIENTATION = (0.0, 0.70710678, 0.0, 0.70710678)  # qx, qy, qz, qw

# The MoveGroup position constraint targets arm_right_tool_link, but the
# point between the finger pads is PAL's gripper_right_grasping_link, 0.157m
# further out along the tool's local +Z (base_footprint +X under
# GRASP_ORIENTATION) -- read from TF. Sending the book's point straight
# through as the tool_link target used to drive the fingers past the book
# into the shelf frame (/contacts: fingertip vs erc_shelf::shelf_base_link).
# Pull the tool_link target back so the grasp point, not the tool, lands on it.
GRASPING_LINK_OFFSET = 0.157  # m

# book_detector's point is the book's *front face* (nearest depth pixel).
# Books are 0.02m thick across the finger-spread axis and 0.16m deep, so the
# pads (0.114-0.170m out from arm_right_tool_link) need to slide past the
# front face to straddle the book rather than stopping on it.
GRASP_DEPTH = 0.04  # m past the front face

MAX_REACHABLE_ABS_Y = 0.35      # m, planning-frame y, see above

# Vertical reach: /compute_ik with GRASP_ORIENTATION at x~0.76 tops out
# around z=1.10 (base_footprint) with the torso down, and reaches past 1.40
# with it fully raised. Rows 2-3 of the shelf (z~1.25-1.6) are only
# reachable with lift, so raise it just enough for the target, then drop
# back to default after the grasp so the robot drives with a low CoG.
TORSO_JOINT = 'torso_lift_joint'
TORSO_MIN, TORSO_MAX = 0.0, 0.35
# PAL's home torso height. At 0.0 MoveIt flags the tucked right arm as
# colliding with base_link (/check_state_validity), so planning out of the
# tuck fails with INVALID_MOTION_PLAN; anything >=0.05 is clear.
TORSO_DEFAULT = 0.10
REACH_Z_AT_ZERO_LIFT = 1.05     # m, a little under the measured 1.10 limit
TORSO_LIFT_MARGIN = 0.05        # m
# The sim's torso tracks far slower than its URDF velocity limit (~60s for
# the full 0.35m in testing), so wait on /joint_states, not a fixed sleep.
TORSO_TOLERANCE = 0.01          # m
TORSO_TIMEOUT_SEC = 300.0  # wall time; sim runs ~0.13x real time

GRIPPER_JOINT = 'gripper_right_finger_joint'
GRIPPER_TOLERANCE = 0.003       # joint units
GRIPPER_STALL_DELTA = 0.0005    # per 0.1s poll
GRIPPER_STALL_SEC = 1.0
GRIPPER_TIMEOUT_SEC = 10.0
# Closing on a book: gz reports fingertip<->book contact at only ~0.02mm
# depth, so the position controller still reaches its target -- joint
# position can't tell "holding" from "empty" at the target. What does break
# the grasp is slamming to 0.0 in 1s: the fingers squeezed the book out of
# the pads. A slow close to a slight squeeze (0.010 ~ a few mm past the
# book's 20mm edge) held it, then it lifted cleanly with the gripper.
GRIPPER_CLOSE_DURATION = 3      # s
LIFT_AFTER_GRASP = 0.03         # m, raise off the shelf before pulling out

# --- place_in_bin: perceived, not hardcoded ---
# The bin is red (same hue as red books) and much bigger in view; it sits on
# the table, so its rim and the table top are both measured from the head
# camera's colour+depth before choosing a release height.
COLOR_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
DEPTH_INFO_TOPIC = '/head_front_camera/head_front_camera/depth/camera_info'
BIN_RED_RANGES = [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))]
BIN_MIN_AREA_PX = 2500
BIN_DEPTH = 0.56                # m, bin extent along the robot's forward axis (mesh)
BIN_HEAD_TILT = -0.4            # rad, rim at ~0.95m a metre ahead is in view
BOOK_HALF_HEIGHT = 0.125        # m, book hangs this far below the grasp point
BOOK_CENTRE_AHEAD = 0.04        # m, book centre ahead of the grasp point (0.16m deep, gripped 0.04 in)
DROP_CLEARANCE = 0.05           # m, book bottom above the rim at release
BIN_HOVER_BACK = 0.15           # m, approach over the rim from this far back...
BIN_HOVER_UP = 0.05             # m, ...and this far up
BIN_RELEASE_TOOL_X = 0.60       # m, comfortable forward reach for arm_right_tool_link
BIN_APPROACH_SPEED = 0.1        # m/s
BIN_MIN_FRONT_RANGE = 0.12      # m, front LiDAR stop while closing in on the table

# PAL's home for arm_right (tiago_pro_motions_general_arm_right.yaml); the
# "default position" after the drop.
ARM_RIGHT_HOME_WAYPOINTS = (
    ((-1.8614, -1.6008, -0.34892, -1.9818, 0.10153, -1.2, 0.0), 3),
    ((-0.26, -1.6008, -0.3489, -1.9818, 0.0, -1.2, 0.0), 6),
    ((-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0), 9),
)
CENTERING_LINEAR_SPEED = 0.2    # m/s, mecanum-base lateral strafe command
# Calibrated against odom: commanding 0.4m of strafe (at the speed/duration
# above) only produced ~0.094m of *actual* lateral displacement -- severe
# slip, much worse than the forward-drive slip navigation_node already
# works around. Open-loop speed*time strafing was tried first and rejected:
# it under-shot by ~4x and never converged (see git history). Drive against
# /odom instead, like navigation_node does for forward motion.
CENTERING_POSITION_TOLERANCE = 0.05  # m, odom-measured lateral displacement
CENTERING_TIMEOUT_SEC = 8.0     # s, safety cap per strafe attempt
CENTERING_SETTLE_SEC = 1.0      # s, let the wheels stop and a fresh point arrive


def clamp_angle(angle, limit):
    return math.copysign(min(abs(angle), limit), angle) if angle else 0.0


class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])

        # --- Tunable parameters (confirmed names from the team's MoveIt setup) ---
        self.declare_parameter('planning_group', 'arm_right')
        self.declare_parameter('eef_link', 'arm_right_tool_link')
        self.declare_parameter('planning_frame', 'base_footprint')
        self.declare_parameter('approach_dx', -0.20)
        self.declare_parameter('approach_dy', 0.0)
        self.declare_parameter('approach_dz', 0.0)
        # 0.069 is the clamp's max; 0.04 only opened the fingertips ~61mm,
        # tight for sliding around a 20mm book with any lateral error.
        self.declare_parameter('gripper_open', 0.065)
        self.declare_parameter('gripper_closed', 0.010)
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
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=cb_group)
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb_group)

        # --- Gripper: direct topic publish, matching the controller interface ---
        # NOTE: the spawned controller is gripper_right_controller_raw (see
        # erc_bringup/config/controller_params.yaml) -- the non-_raw topic
        # has no subscriber, so publishing there silently does nothing.
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_right_controller_raw/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.torso_position = None
        self.gripper_position = None
        self.head_positions = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10, callback_group=cb_group)
        self.odom_pose = None  # (x, y, yaw), for closed-loop strafing -- see _strafe_base
        self.create_subscription(Odometry, '/odom', self._on_odom, 10, callback_group=cb_group)

        # --- TF for transforming perception output into the planning frame ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Perception input ---
        # book_detector (INTERFACES.md contract) publishes a PointStamped on
        # /erc/target_book_point; converted to a PoseStamped (identity
        # orientation) on /erc/target_book_pose here so the rest of this
        # node can share upstream's pose-based grasp logic.
        self.latest_book_pose = None
        self.create_subscription(
            PointStamped, '/erc/target_book_point', self._on_book_point, 10,
            callback_group=cb_group)
        self.book_pose_pub = self.create_publisher(PoseStamped, '/erc/target_book_pose', 10)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.depth_info = None
        self.create_subscription(
            Image, COLOR_TOPIC, lambda m: setattr(self, 'latest_color', m), 2, callback_group=cb_group)
        self.create_subscription(
            Image, DEPTH_TOPIC, lambda m: setattr(self, 'latest_depth', m), 2, callback_group=cb_group)
        self.create_subscription(
            CameraInfo, DEPTH_INFO_TOPIC, lambda m: setattr(self, 'depth_info', m), 2, callback_group=cb_group)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.front_range = math.inf
        self.create_subscription(LaserScan, '/scan_front_raw', self._on_scan, 10, callback_group=cb_group)
        self.arm_right_pub = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)

        # --- Manipulation service contract (state_machine_node calls these) ---
        self.create_service(
            GraspBook, '/erc/grasp_book', self._on_grasp_book, callback_group=cb_group)
        self.create_service(
            PlaceInBin, '/erc/place_in_bin', self._on_place_in_bin, callback_group=cb_group)

        self.get_logger().info(
            'Manipulation node ready (right arm) -- waiting for grasp_book/place_in_bin calls')

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState):
        if TORSO_JOINT in msg.name:
            self.torso_position = msg.position[msg.name.index(TORSO_JOINT)]
        if GRIPPER_JOINT in msg.name:
            self.gripper_position = msg.position[msg.name.index(GRIPPER_JOINT)]
        if 'head_1_joint' in msg.name and 'head_2_joint' in msg.name:
            self.head_positions = (msg.position[msg.name.index('head_1_joint')],
                                   msg.position[msg.name.index('head_2_joint')])

    def _set_torso(self, target) -> bool:
        target = max(TORSO_MIN, min(TORSO_MAX, target))
        if self.torso_position is not None and abs(self.torso_position - target) <= TORSO_TOLERANCE:
            return True
        msg = JointTrajectory()
        msg.joint_names = [TORSO_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start.sec = 12
        msg.points = [point]
        self.torso_pub.publish(msg)
        self.get_logger().info(f'moving torso to {target:.2f}m')
        deadline = time.monotonic() + TORSO_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self.torso_position is not None and abs(self.torso_position - target) <= TORSO_TOLERANCE:
                return True
            time.sleep(0.2)
        self.get_logger().warn(
            f'torso did not reach {target:.2f}m (at {self.torso_position}) within {TORSO_TIMEOUT_SEC}s')
        return False

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_pose = (p.x, p.y, yaw)

    def _on_book_point(self, msg: PointStamped):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position = msg.point
        pose.pose.orientation.w = 1.0
        self.latest_book_pose = pose
        self.book_pose_pub.publish(pose)

    def _wait_for_pose(self, attr_name):
        deadline = time.monotonic() + self.point_wait_timeout
        while time.monotonic() < deadline:
            pose = getattr(self, attr_name)
            if pose is not None:
                return pose
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------
    def transform_to_planning_frame(self, pose_stamped: PoseStamped):
        if pose_stamped.header.frame_id == self.planning_frame:
            return pose_stamped
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame, pose_stamped.header.frame_id, rclpy.time.Time())
            return do_transform_pose_stamped(pose_stamped, transform)
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

    def _strafe_base(self, dy):
        """Lateral move via the mecanum base (cmd_vel.linear.y) -- unlike
        rotating, this keeps the camera pointed the same direction, so it
        doesn't swing a *different* column's same-coloured book into frame
        (see _center_on_book).

        Closed-loop against /odom: an open-loop speed*time version was
        tried first and rejected after calibration showed commanding 0.4m
        of strafe only produced ~0.094m of actual displacement (severe
        lateral slip) -- it never converged, just oscillated. This drives
        until the odom-measured displacement (rotated into the pose the
        base had when the move started) matches the request."""
        if self.odom_pose is None:
            self.get_logger().warn('no /odom yet -- cannot strafe closed-loop')
            return
        start_x, start_y, start_yaw = self.odom_pose
        target = abs(dy)
        sign = math.copysign(1.0, dy)
        cmd = Twist()
        cmd.linear.y = sign * CENTERING_LINEAR_SPEED
        # Sim-time deadline: base motion covers distance at sim speed, so a
        # wall-clock cap cuts the move short whenever the real-time factor
        # drops (seen at ~7%). Wall-clock cap kept only as a hang backstop.
        deadline = self.get_clock().now().nanoseconds * 1e-9 + CENTERING_TIMEOUT_SEC
        wall_cap = time.monotonic() + 600.0
        c, s = math.cos(-start_yaw), math.sin(-start_yaw)
        while self.get_clock().now().nanoseconds * 1e-9 < deadline and time.monotonic() < wall_cap:
            x, y, _ = self.odom_pose
            ddx, ddy = x - start_x, y - start_y
            # rotate the odom-frame delta back into the base's strafe axis
            # at the moment the move started
            moved = ddx * s + ddy * c
            if abs(moved) >= target - CENTERING_POSITION_TOLERANCE:
                break
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)
        self.cmd_vel_pub.publish(Twist())

    def _center_on_book(self, book_pose: PoseStamped, max_attempts=4):
        """Strafe the base sideways so the book point lands on arm_right's
        forward reach axis (small |y| in the planning frame) instead of off
        to the side where no IK solution exists. Open-loop (no odom
        feedback, just the measured y offset in the base-fixed planning
        frame), re-checked against a fresh book pose after each move.

        Rotating the base to fix this was tried first and rejected: turning
        by the full bearing angle routinely swung a *different* column's
        book of the same target colour into the frame instead of centering
        the original one (column spacing is only ~1m at ~0.7m standoff, so
        even fairly small turns cross into the next column), and the
        tracked point would jump to a wildly different offset instead of
        converging. Strafing keeps the camera facing the same direction the
        whole time, so the same book stays the biggest/most-recently-seen
        blob across attempts."""
        for attempt in range(max_attempts):
            y = book_pose.pose.position.y
            # Slight overshoot margin so we land inside the reach envelope
            # rather than exactly on its edge.
            step = clamp_angle(y - math.copysign(0.05, y), 2.0)
            self.get_logger().info(
                f'book at y={y:.2f}m (> {MAX_REACHABLE_ABS_Y}m reach limit) -- '
                f'strafing base {step:.2f}m to center it (attempt {attempt + 1})')
            self._strafe_base(step)
            time.sleep(CENTERING_SETTLE_SEC)

            self.latest_book_pose = None
            fresh_pose = self._wait_for_pose('latest_book_pose')
            if fresh_pose is None:
                self.get_logger().warn('no fresh target_book_pose after centering turn')
                return None
            fresh_pose = self.transform_to_planning_frame(fresh_pose)
            if fresh_pose is None:
                return None
            book_pose = fresh_pose
            if abs(book_pose.pose.position.y) <= MAX_REACHABLE_ABS_Y:
                return book_pose
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

    @staticmethod
    def _block_on_future(future, timeout_sec=MOTION_TIMEOUT_SEC, poll=0.02):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(poll)
        return future.result()

    def move_linear(self, pose_stamped: PoseStamped) -> bool:
        """Straight-line tool motion to pose_stamped (via /compute_cartesian_path
        + /execute_trajectory). Used for every move near a book or the bin:
        OMPL's joint-space plans between two nearby poses can still swing the
        hand along a curve, which is how the gripper swept books over."""
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/compute_cartesian_path not available')
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id = pose_stamped.header.frame_id
        req.group_name = self.planning_group
        req.link_name = self.eef_link
        req.waypoints = [pose_stamped.pose]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        res = self._block_on_future(self.cartesian_client.call_async(req))
        if res is None or res.fraction < 0.99:
            self.get_logger().warn(
                f'straight-line path only {0.0 if res is None else res.fraction:.0%} feasible')
            return False

        traj = res.solution
        for point in traj.joint_trajectory.points:  # slow it down near objects
            t = (point.time_from_start.sec + point.time_from_start.nanosec * 1e-9) * LINEAR_TIME_SCALE
            point.time_from_start = Duration(seconds=t).to_msg()
            point.velocities = [v / LINEAR_TIME_SCALE for v in point.velocities]
            point.accelerations = [a / LINEAR_TIME_SCALE ** 2 for a in point.accelerations]

        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/execute_trajectory not available')
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        handle = self._block_on_future(self.execute_client.send_goal_async(goal))
        if handle is None or not handle.accepted:
            self.get_logger().warn('ExecuteTrajectory goal rejected')
            return False
        result = self._block_on_future(handle.get_result_async())
        if result is None or result.result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().warn(
                f'straight-line execution failed ({None if result is None else result.result.error_code.val})')
            return False
        return True

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
        """Command the gripper and block until it either reaches `opening` or
        stops moving (closed on something). Returns the settled position.

        The fingers track far slower than the 1s trajectory in this sim, and
        a fixed sleep let the arm drive in with the fingers still half shut --
        which knocked the book over instead of straddling it."""
        msg = JointTrajectory()
        msg.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [opening]
        point.time_from_start.sec = int(duration_sec)
        msg.points = [point]
        self.gripper_pub.publish(msg)

        deadline = time.monotonic() + GRIPPER_TIMEOUT_SEC
        last_pos, still_since = None, None
        time.sleep(duration_sec)
        while time.monotonic() < deadline:
            pos = self.gripper_position
            if pos is not None:
                if abs(pos - opening) <= GRIPPER_TOLERANCE:
                    return pos
                if last_pos is not None and abs(pos - last_pos) < GRIPPER_STALL_DELTA:
                    still_since = still_since or time.monotonic()
                    if time.monotonic() - still_since >= GRIPPER_STALL_SEC:
                        return pos
                else:
                    still_since = None
                last_pos = pos
            time.sleep(0.1)
        return self.gripper_position

    # ------------------------------------------------------------------
    def _fresh_book_face(self, timeout_sec=BOOK_REFRESH_TIMEOUT_SEC):
        """Book front face (x, y, z) in the planning frame from a detection
        taken *after* this call, or None. Waits BOOK_SETTLE_SEC first so the
        camera isn't mid-motion."""
        time.sleep(BOOK_SETTLE_SEC)
        since = self.get_clock().now().nanoseconds
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            pose = self.latest_book_pose
            if pose is not None and rclpy.time.Time.from_msg(pose.header.stamp).nanoseconds >= since:
                pose = self.transform_to_planning_frame(pose)
                if pose is not None:
                    p = pose.pose.position
                    return (p.x, p.y, p.z)
            time.sleep(0.05)
        return None

    def _look_at(self, point):
        """Pan/tilt the head so `point` (planning frame) is at image centre --
        detections are most accurate there, and book_detector picks the blob
        nearest centre, so this also locks onto the intended book."""
        if self.head_positions is None:
            return
        target = PointStamped()
        target.header.frame_id = self.planning_frame
        target.point.x, target.point.y, target.point.z = point
        try:
            tf = self.tf_buffer.lookup_transform(CAMERA_OPTICAL_FRAME, self.planning_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'look_at TF failed: {e}')
            return
        c = do_transform_point(target, tf).point
        pan = self.head_positions[0] - math.atan2(c.x, c.z)
        tilt = self.head_positions[1] - math.atan2(c.y, c.z)
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [pan, tilt]
        pt.time_from_start.sec = 1
        msg.points = [pt]
        self.head_pub.publish(msg)

    def _grasp_poses(self, face):
        """(pregrasp, grasp) tool_link poses for a book front face."""
        grasp = self.make_pose(
            face[0] + GRASP_DEPTH - GRASPING_LINK_OFFSET, face[1], face[2],
            self.planning_frame, *GRASP_ORIENTATION)
        dx, dy, dz = self.approach_offset
        return self.offset_pose(grasp, dx, dy, dz), grasp

    def _on_grasp_book(self, request, response):
        book_pose = self._wait_for_pose('latest_book_pose')
        if book_pose is None:
            response.success = False
            response.message = 'timed out waiting for /erc/target_book_pose'
            return response

        book_pose = self.transform_to_planning_frame(book_pose)
        if book_pose is None:
            response.success = False
            response.message = f'TF transform of target_book_pose to {self.planning_frame} failed'
            return response

        if abs(book_pose.pose.position.y) > MAX_REACHABLE_ABS_Y:
            book_pose = self._center_on_book(book_pose)
            if book_pose is None:
                response.success = False
                response.message = (
                    'book stayed out of arm_right reach after centering the base on it')
                return response
        face = (book_pose.pose.position.x, book_pose.pose.position.y, book_pose.pose.position.z)

        # 1. Raise the body for the book's row before reaching.
        lift = max(TORSO_DEFAULT, face[2] - REACH_Z_AT_ZERO_LIFT + TORSO_LIFT_MARGIN)
        if lift > TORSO_MAX:
            response.success = False
            response.message = f'book at z={face[2]:.2f}m is above reach even at full torso lift'
            return response
        if not self._set_torso(lift):
            response.success = False
            response.message = 'torso failed to reach the lift height for this row'
            return response

        try:
            # 2. Re-measure with the head aimed at the book (camera moved with the torso).
            self._look_at(face)
            refined = self._fresh_book_face()
            if refined is None:
                response.success = False
                response.message = 'lost the book after raising the torso'
                return response
            self.get_logger().info(f'book face {face} -> refined {tuple(round(v, 3) for v in refined)}')
            face = refined

            opened = self.set_gripper(self.gripper_open)
            if opened is None or abs(opened - self.gripper_open) > GRIPPER_TOLERANCE * 3:
                response.success = False
                response.message = f'gripper did not open (at {opened}); not approaching the book'
                return response

            pregrasp, grasp = self._grasp_poses(face)
            if not self.move_to_pose(pregrasp):
                response.success = False
                response.message = 'failed to plan/move to pre-grasp hover pose'
                return response

            # 3. Close-range check from the hover point; re-aim if it's the
            # same book, abort if it isn't where we thought.
            self._look_at(face)
            close = self._fresh_book_face()
            if close is not None:
                shift = math.dist(close, face)
                if shift > BOOK_REAIM_MAX_SHIFT:
                    # At the hover pose the gripper sits between the head camera
                    # and the book, so this detection can be clipped or read depth
                    # off the gripper (seen: 0.241m jump with the book unmoved).
                    # Treat it like "not visible" and keep the clear-view fix from
                    # before the hover, which matched ground truth to 1-3cm.
                    self.get_logger().warn(
                        f'close-range detection {shift:.3f}m off (dx={close[0] - face[0]:+.3f} '
                        f'dy={close[1] - face[1]:+.3f} dz={close[2] - face[2]:+.3f}); '
                        'likely occluded by the gripper -- using the earlier fix')
                else:
                    face = close
                    pregrasp, grasp = self._grasp_poses(face)
                    if not self.move_linear(pregrasp):
                        response.success = False
                        response.message = 'failed to re-align at the pre-grasp pose'
                        return response
            else:
                self.get_logger().warn('book not visible from the hover pose; using the earlier fix')

            # 4. Straight in, then confirm the book didn't get pushed before closing.
            if not self.move_linear(grasp):
                response.success = False
                response.message = 'failed to move in to the grasp pose'
                return response
            after = self._fresh_book_face()
            # With the fingers around the book the gripper hides it from the
            # head camera, so this detection can land on the same-coloured
            # book in the next column instead (seen: dy=+0.716 with the
            # target still straddled). A shift past BOOK_REAIM_MAX_SHIFT is a
            # different book, not a push -- close on the one we reached.
            if after is not None and math.dist(after[:2], face[:2]) > BOOK_REAIM_MAX_SHIFT:
                self.get_logger().warn(
                    f'post-reach detection {math.dist(after[:2], face[:2]):.3f}m away is a '
                    'different book (target hidden by the gripper); closing on the reached pose')
                after = None
            if after is not None and (abs(after[0] - face[0]) > BOOK_PUSH_TOLERANCE
                                      or abs(after[1] - face[1]) > BOOK_PUSH_TOLERANCE):
                self.move_linear(pregrasp)
                response.success = False
                response.message = (
                    f'book moved while reaching in (dx={after[0] - face[0]:+.3f} '
                    f'dy={after[1] - face[1]:+.3f}); backed out without closing')
                return response

            # 5. Squeeze gently, lift off the shelf, pull straight out.
            closed = self.set_gripper(self.gripper_closed, duration_sec=GRIPPER_CLOSE_DURATION)
            if closed is None:
                response.success = False
                response.message = 'no gripper feedback after closing'
                return response
            if not self.move_linear(self.offset_pose(grasp, 0.0, 0.0, LIFT_AFTER_GRASP)):
                response.success = False
                response.message = 'closed on the book but failed to lift it off the shelf'
                return response
            self.move_linear(self.offset_pose(pregrasp, 0.0, 0.0, LIFT_AFTER_GRASP))
        finally:
            self._set_torso(TORSO_DEFAULT)
            self._set_head_tilt(0.0)

        response.success = True
        response.message = 'book grasped'
        return response

    def _perceive_bin(self):
        """Find the red bin in the head camera and measure it in the planning
        frame: rim height, centre, and the table top it sits on. Returns a
        dict or None. Uses the same colour+depth back-projection as
        book_detector (depth and colour are both 640x360 and line up)."""
        self._set_head_tilt(BIN_HEAD_TILT)
        time.sleep(2.0)
        color_msg, depth_msg, info = self.latest_color, self.latest_depth, self.depth_info
        if color_msg is None or depth_msg is None or info is None:
            self.get_logger().warn('no colour/depth frames for bin perception')
            return None
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
        if depth_msg.encoding == '16UC1':
            depth /= 1000.0

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in BIN_RED_RANGES:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= BIN_MIN_AREA_PX]
        if not contours:
            self.get_logger().warn('bin not visible (no large red blob)')
            return None
        blob = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(blob)
        bin_mask = np.zeros_like(mask)
        cv2.drawContours(bin_mask, [blob], -1, 255, thickness=cv2.FILLED)
        bin_mask &= mask

        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame, depth_msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF for bin perception failed: {e}')
            return None
        q = tf.transform.rotation
        rot = np.array([
            [1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
            [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w)],
            [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y)]])
        trans = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]

        def project(pixel_mask):
            vs, us = np.nonzero(pixel_mask)
            d = depth[vs, us]
            ok = np.isfinite(d) & (d > 0.1) & (d < 4.0)
            vs, us, d = vs[ok], us[ok], d[ok]
            cam = np.stack([(us - cx) * d / fx, (vs - cy) * d / fy, d], axis=1)
            return cam @ rot.T + trans

        bin_pts = project(bin_mask)
        if len(bin_pts) < 50:
            self.get_logger().warn(f'too few bin depth points ({len(bin_pts)})')
            return None

        # Table: the non-red pixels just below the bin in the image are the
        # table top in front of it.
        below = np.zeros_like(mask)
        below[by + bh:min(by + bh + max(bh, 20), below.shape[0]), bx:bx + bw] = 255
        below &= cv2.bitwise_not(mask)
        table_pts = project(below)

        rim_z = float(np.percentile(bin_pts[:, 2], 98))
        bin_y = float(np.median(bin_pts[:, 1]))
        front_x = float(np.percentile(bin_pts[:, 0], 3))
        back_x = float(np.percentile(bin_pts[:, 0], 97))
        centre_x = (front_x + back_x) / 2 if back_x - front_x > BIN_DEPTH * 0.7 else front_x + BIN_DEPTH / 2
        table_z = None
        if len(table_pts) >= 20:
            candidates = table_pts[(table_pts[:, 2] > 0.3) & (table_pts[:, 2] < rim_z - 0.05)]
            if len(candidates) >= 20:
                table_z = float(np.percentile(candidates[:, 2], 90))
        self.get_logger().info(
            f'bin perceived: rim z={rim_z:.3f} centre=({centre_x:.3f},{bin_y:.3f}) '
            f'front x={front_x:.3f} table z={table_z if table_z is None else round(table_z, 3)} '
            f'({len(bin_pts)} bin pts)')
        return {'rim_z': rim_z, 'centre_x': centre_x, 'centre_y': bin_y, 'table_z': table_z}

    def _set_head_tilt(self, tilt):
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, tilt]
        point.time_from_start.sec = 1
        msg.points = [point]
        self.head_pub.publish(msg)

    def _go_home(self):
        """Right arm back to PAL's home pose (left arm is already tucked by
        navigation_node), torso to default, base held still."""
        msg = JointTrajectory()
        msg.joint_names = [f'arm_right_{i}_joint' for i in range(1, 8)]
        for positions, t in ARM_RIGHT_HOME_WAYPOINTS:
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.time_from_start.sec = t
            msg.points.append(point)
        self.arm_right_pub.publish(msg)
        time.sleep(ARM_RIGHT_HOME_WAYPOINTS[-1][1] + 2.0)
        self._set_torso(TORSO_DEFAULT)
        self._set_head_tilt(0.0)
        self.cmd_vel_pub.publish(Twist())

    def _on_scan(self, msg: LaserScan):
        ranges = [
            d for i, d in enumerate(msg.ranges)
            if abs(math.atan2(math.sin(msg.angle_min + i * msg.angle_increment),
                              math.cos(msg.angle_min + i * msg.angle_increment))) <= math.radians(20)
            and msg.range_min <= d <= msg.range_max]
        self.front_range = min(ranges) if ranges else math.inf

    def _drive_straight(self, distance) -> float:
        """Closed-loop on /odom (same reason as _strafe_base), stopping early
        if the front LiDAR gets within BIN_MIN_FRONT_RANGE. Returns the
        distance actually covered (signed)."""
        if self.odom_pose is None:
            return 0.0
        sx, sy, syaw = self.odom_pose
        cmd = Twist()
        cmd.linear.x = math.copysign(BIN_APPROACH_SPEED, distance)
        # Sim-time deadline (see _strafe_base): at ~7% RTF a wall-clock cap of
        # 8 + distance/speed*3 seconds let the base cover only 0.151m of a
        # 0.485m approach, leaving the bin out of reach.
        deadline = (self.get_clock().now().nanoseconds * 1e-9
                    + CENTERING_TIMEOUT_SEC + abs(distance) / BIN_APPROACH_SPEED * 3)
        wall_cap = time.monotonic() + 600.0
        moved = 0.0
        while self.get_clock().now().nanoseconds * 1e-9 < deadline and time.monotonic() < wall_cap:
            x, y, _ = self.odom_pose
            moved = (x - sx) * math.cos(syaw) + (y - sy) * math.sin(syaw)
            if abs(moved) >= abs(distance) - 0.01:
                break
            if distance > 0 and self.front_range < BIN_MIN_FRONT_RANGE:
                self.get_logger().warn(f'front range {self.front_range:.2f}m -- stopping approach to the bin')
                break
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)
        for _ in range(3):
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.05)
        return moved

    def _on_place_in_bin(self, request, response):
        self.cmd_vel_pub.publish(Twist())
        bin_info = self._perceive_bin()
        if bin_info is None:
            response.success = False
            response.message = 'could not perceive the bin (need it in the head camera)'
            return response
        table_z = bin_info['table_z']
        if table_z is not None and bin_info['rim_z'] < table_z + 0.08:
            response.success = False
            response.message = f"bin rim z={bin_info['rim_z']:.2f} not clearly above table z={table_z:.2f}"
            return response

        # The book hangs about half its height below the grasp point, so hold
        # the grasp point that far above the rim, plus clearance -- this is
        # also well above the table top, which the rim sits on.
        release_z = bin_info['rim_z'] + BOOK_HALF_HEIGHT + DROP_CLEARANCE
        # Book centre is BOOK_CENTRE_AHEAD in front of the grasp point.
        release_tool_x = BIN_RELEASE_TOOL_X
        book_centre_x = release_tool_x + GRASPING_LINK_OFFSET + BOOK_CENTRE_AHEAD
        advance = bin_info['centre_x'] - book_centre_x

        def tool_pose(x, z):
            return self.make_pose(x, bin_info['centre_y'], z, self.planning_frame, *GRASP_ORIENTATION)

        # 1. Lift the book above the rim *before* getting near the table.
        carry = tool_pose(release_tool_x - BIN_HOVER_BACK, release_z + BIN_HOVER_UP)
        lift = max(TORSO_DEFAULT, carry.pose.position.z - REACH_Z_AT_ZERO_LIFT + TORSO_LIFT_MARGIN)
        if lift > TORSO_MAX:
            response.success = False
            response.message = f'release height z={release_z:.2f}m is above reach even at full torso lift'
            return response
        if not self._set_torso(lift):
            response.success = False
            response.message = 'torso failed to reach the lift height for the bin'
            return response
        if not (self.move_linear(carry) or self.move_to_pose(carry)):
            response.success = False
            response.message = 'failed to raise the book above bin height'
            return response

        # 2. Now close the gap to the table with the book held high.
        moved = 0.0
        if advance > 0.02:
            moved = self._drive_straight(advance)
            self.get_logger().info(f'advanced {moved:.3f}m of {advance:.3f}m toward the bin')
        shortfall = advance - moved  # bin is still this much further than planned
        release_tool_x += max(0.0, shortfall)

        # 3. Over the rim, drop, and pull the hand back out.
        over = tool_pose(release_tool_x, release_z)
        if not self.move_linear(over):
            self._drive_straight(-moved)
            self._go_home()
            response.success = False
            response.message = 'failed to reach over the bin (backed off, arm home)'
            return response
        self.set_gripper(self.gripper_open, duration_sec=2)
        time.sleep(1.0)  # let the book fall clear before moving the hand
        self.move_linear(tool_pose(release_tool_x - BIN_HOVER_BACK, release_z + BIN_HOVER_UP))

        # 4. Back away from the table, return to default, and stay put.
        self._drive_straight(-moved)
        self._go_home()

        response.success = True
        response.message = (
            f"book released over bin (rim z={bin_info['rim_z']:.2f}, table z="
            f"{'?' if table_z is None else round(table_z, 2)}); arm home, stopped")
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
