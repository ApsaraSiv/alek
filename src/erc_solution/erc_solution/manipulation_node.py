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
service handlers callable from state_machine_node.

grasp_book consumes /erc/target_book_point (geometry_msgs/PointStamped),
published by book_point_detector.py -- a real detection (HSV colour blob
+ depth back-projection at that pixel), not a hardcoded placeholder and
not a world-geometry guess. An earlier version computed the book's pose
from SHELF_X/Y/Z + shelf_column_number + row instead of using a live
point; that was scrapped after live testing showed navigation_node's
approach_shelf only rotates to center the column marker in view then
drives straight (no lateral strafing), so the robot ends up facing the
right column without being laterally in front of its books -- measured a
1.83m lateral miss. A precomputed geometry target inherits that error;
detecting the book directly in the camera, wherever the robot actually
ended up, does not.

Since a MoveGroup "success" only means the commanded pose was reached, not
that anything was actually grasped, _on_grasp_book polls /contacts after
closing the gripper for a real collision between a gripper_right_* link
and the target book's link before reporting success -- otherwise this
would happily report "grasped" after closing on nothing, which is exactly
what an earlier version (no verification) did the one time this ran
against a real book placement and was watched live.

place_in_bin does NOT use a live point: INTERFACES.md still describes
/erc/collection_bin_point (geometry_msgs/PointStamped) from bin_detector,
but the real bin_detector (erc_perception, merged after that doc was
written) only publishes /erc/bin_identification -- a visibility Bool, not
a point. Nothing publishes a bin point. Instead this uses a fixed pose in
the planning frame, since navigate_to_bin (navigation_node) already parks
the robot at a known standoff facing the bin -- see STILL TO TUNE below.

Runs under a MultiThreadedExecutor with a ReentrantCallbackGroup so a
blocking service call (grasp_book/place_in_bin) doesn't starve the
move_group action client's own callbacks or the point subscriptions --
same pattern state_machine_node and navigation_node use. Blocking on a
future is done with a plain poll loop (not spin_until_future_complete,
which tries to attach this node to a second executor and fails since the
node is already spinning under executor.spin()).

Also tucks arm_right (joint-space goal, TUCK_JOINTS below -- mirrors the
"tuck" group_state added to tiago_pro.srdf, duplicated here since the
low-level MoveGroup interface has no way to resolve a named state without
moveit_py) once on startup and again after place_in_bin, so it isn't left
sitting at the URDF's all-zero-joint default (not a folded pose by
convention) through navigation. Also stows arm_left once at startup
(direct joint command, ARM_LEFT_DOWN_JOINTS -- see there for why) since
nothing else manages that arm and its default pose collides with a table
near spawn. Also raises torso_lift_joint (direct command, not part of the
arm_right MoveIt group) before a grasp if the detected book is above
TORSO_REACH_BASELINE_Z, trading bottom-of-workspace reach for top-of
-workspace reach -- see raise_torso_for_height -- and lowers it back to 0
afterward so place_in_bin's fixed-height pose assumption stays valid.

STILL TO TUNE:
  - TUCK_JOINTS: first guess, not yet visually verified against the mesh.
  - TORSO_REACH_BASELINE_Z: the one height (0.9) this node has actually
    confirmed reachable at torso=0; everything above that gets torso lift
    on a straight-line guess, not a measured reach envelope.
  - approach_offset (dx, dy, dz): vector from the grasp point to a safe
    hover point before/after grasping.
  - place_x/y/z: hardcoded pose (planning frame) for place_in_bin, guessed
    from navigation_node's BIN_STANDOFF (0.7m) and BOOK_APPROACH_HEAD_TILT.
    Not measured against the actual bin model, and place_in_bin has no
    equivalent contact verification yet (only grasp_book does).
  - Even with a real detected point, the robot's lateral-alignment issue
    (see above) means the book may simply be out of arm reach for some
    columns -- book_point_detector publishing a correct point doesn't by
    itself fix navigation not driving close enough. grasp_book will
    correctly fail (pre-grasp hover pose unreachable) in that case rather
    than reaching for a wrong point, but it still won't grasp anything
    until navigation adds lateral positioning.

TEST TIP: with move_group and book_point_detector running (needs a real
book visible in the head camera -- point the robot/camera at one, or run
against the full pipeline), call the service directly:
  ros2 service call /erc/grasp_book erc_interfaces/srv/GraspBook "{row: 1}"
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState

from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    PositionIKRequest,
    RobotState,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
)

from erc_interfaces.srv import GraspBook, PlaceInBin

import tf2_ros
from tf2_geometry_msgs import do_transform_point

MOVEIT_SUCCESS = 1  # moveit_msgs/msg/MoveItErrorCodes.SUCCESS

# Keep in sync with the "tuck" group_state in tiago_pro.srdf.
#
# arm_right_1_joint was -1.5 (a ~86deg shoulder yaw) originally. Found via
# /check_state_validity (moveit_msgs/GetStateValidity) that this SWEEPS
# arm_right's wrist almost all the way across the body to the LEFT side --
# it doesn't fold in close to the torso the way "tuck" suggests. Confirmed
# this collides with arm_left_1_link (fixed near the left shoulder -- a
# revolute joint only rotates that link in place, it never translates, so
# no arm_left joint choice can dodge it) AND has a self-collision between
# gripper_right_base_link and torso_lift_link -- both present even with
# arm_left back at the plain URDF-zero default, so this was never actually
# about arm_left at all: TUCK_JOINTS itself has always reached into that
# collision volume, just never checked with a formal state-validity call
# before. Reduced the yaw to -0.8 (keeps the wrist on arm_right's own side
# instead of sweeping across the body) -- confirmed valid=True via
# /check_state_validity with arm_left at rest.
#
# That fixed arm_left/torso, but a live scripted run then hit a DIFFERENT
# self-collision: head_2_link vs the gripper fingertips. The -0.8/0.9/-2.0
# fold still brings the wrist up right next to the head, and once the head
# tilts down for SEEK_BOOK's scan (or approach_shelf's fixed
# BOOK_APPROACH_HEAD_TILT), they intersect -- visually confirmed via a
# /gui/screenshot capture showing the gripper tip touching the head.
# Reduced arm_right_2_joint (shoulder lift) from 0.9 to 0.5, which keeps
# the wrist lower and clear of the head -- confirmed valid=True via
# /check_state_validity across every head tilt this codebase actually
# uses (0.3 SPIN_HEAD_TILT, -0.15 BOOK_APPROACH_HEAD_TILT, -0.05/-0.25
# ROW_SEARCH_HEAD_TILTS, and 0.0 neutral), with arm_left simultaneously
# at its own real rest pose each time.
#
# User-directed follow-up: tried joint 1 (shoulder yaw) and joint 2
# (shoulder lift) at 22deg (0.384rad) on both arms, collision-free
# (verified across every head tilt) -- but a live screenshot showed the
# real problem: at only 22deg of yaw, arm_left barely rotates away from
# facing forward at all, so with the elbow still nearly straight it
# reaches out toward the shelf right alongside arm_right instead of
# clearing the workspace. Not a collision bug -- it just doesn't
# accomplish "out of the way" at that shallow an angle. Reverted to the
# ~80-92deg yaw values below, which have an actual track record of
# keeping the arm clear in live runs.
#
# Follow-up (live-run investigation): even with the head_2_link/gripper
# self-collision above resolved, book/column detection kept losing lock or
# reading corrupted positions across many live runs -- confirmed via a
# camera-frame grab (not just /check_state_validity, which only checks
# self-collision, not what the camera can actually see) that the
# arm_right_2_joint=0.5 pose sits DIRECTLY in the head camera's forward
# view once the head tilts down for book search (BOOK_APPROACH_HEAD_TILT
# and the ROW_SEARCH_HEAD_TILTS band): the gripper filled most of the
# frame, leaving perception blind to most of the shelf regardless of
# self-collision status. Yaw alone (tried -1.4, -1.7, -2.4 rad) barely
# changed how much of the frame it blocked -- at this close range to the
# camera, small angular sweeps around the shoulder yaw axis don't move the
# gripper much in the image; changing the arm's DEPTH/HEIGHT relative to
# the camera did. Dropping arm_right_2_joint (shoulder lift) to -0.5 (down
# and back, instead of up toward the head) confirmed via live screenshot to
# clear the view almost entirely (only a small fingertip sliver remains,
# off the shelf) -- re-verified valid=True via /check_state_validity across
# every head tilt this codebase uses, arm_left simultaneously at its own
# rest pose.
TUCK_JOINTS = {
    'arm_right_1_joint': -0.8,
    'arm_right_2_joint': -0.5,
    'arm_right_3_joint': 0.0,
    'arm_right_4_joint': -1.8,
    'arm_right_5_joint': 0.0,
    'arm_right_6_joint': 0.0,
    'arm_right_7_joint': 0.0,
}

# arm_left isn't used by this task and has no MoveIt planning group (this
# MoveIt config only covers arm_right), so it's commanded directly via its
# controller topic rather than through move_group -- no collision-aware
# planning needed, just get it out of the way. Left at the URDF's all-zero
# default, it hangs low enough to collide with erc_table (world pose
# (-1, 0, 0.7), close to spawn).
#
# Every downward angle tried (-1.2, -1.6, -2.2) either hit the table, hit
# the ground plane, or grazed the floor transiently during the 2s motion
# itself even when the final rest pose checked clean via /contacts (a
# single-waypoint trajectory interpolates straight from wherever the arm
# currently is to the target, so a clean endpoint doesn't guarantee a
# clean path).
#
# Next tried mirroring arm_right's TUCK_JOINTS exactly (lift=0.9,
# elbow=-2.0, yaw=0.0) -- clean against the table/floor in isolation, but
# that was only ever checked with arm_right NOT also at its own tuck pose
# at the same time. With both arms tucked simultaneously (the real
# startup scenario) move_group's own start-state collision check found
# arm_left and arm_right intersecting -- both arms fold their elbow up
# and back over the same central overhead space above the torso, so
# mirroring the same fold on both sides just makes them meet in the
# middle. Confirmed live: rotating arm_left's shoulder yaw alone (tried
# 1.2, 2.4, -0.5 rad, keeping the same 0.9/-2.0 lift+fold) only moved
# *which* link pair collided (arm_left_2/arm_right_6, then
# arm_left_1/arm_right_6, then back to arm_left_2/arm_right_6) -- yaw
# alone can't fix it because the fold amount, not the direction, is what
# puts arm_left's forearm into that shared space.
#
# Shoulder swung out to the side (yaw=1.6) with a mild lift and a
# nearly-straight elbow (0.3/-0.3). Doesn't visually read as a compact
# "tuck" -- looks like the arm is just sticking out to the side -- but
# this is the pose actually confirmed by a real successful live grasp
# run with zero collisions.
#
# Repeatedly tried the fuller, more visually "tucked" 0.9/-2.0 fold at
# this same yaw, re-tuning TUCK_JOINTS each time a new collision turned
# up, and each time /check_state_validity passed across every head tilt
# this codebase uses -- and each time a live run then hit a DIFFERENT
# collision pair anyway: arm_left_2/arm_right_6, then arm_left_1/
# gripper_right + gripper_right/torso (fixed by lowering TUCK_JOINTS'
# shoulder lift), then arm_left_5/arm_right_1. Three distinct pairs
# across the two kinematic chains -- the full fold genuinely overlaps
# arm_right's tuck volume across a wide region, not at one narrow spot a
# few spot-checks can fully map. Given that pattern, don't re-attempt the
# full fold again without a systematic sweep (e.g. checking validity at
# many points along arm_right's actual tuck->pregrasp trajectory, not
# just the two static endpoints) -- the mild pose is the one with an
# actual track record of working.
#
# Also tried the user-directed 22deg (0.384rad) yaw here, matching
# TUCK_JOINTS' 22deg experiment above -- collision-free but visually
# useless: 22deg isn't enough shoulder rotation to turn the arm away from
# facing forward, so it reached out toward the shelf right alongside
# arm_right instead of staying clear (confirmed via a live screenshot).
# Reverted to yaw=1.6 (~92deg), which actually swings the arm out to the
# side.
#
# User then asked for a properly-folded look instead: close to the body,
# not swung out to the side, not touching the floor. Tried modest yaw
# (0.6rad) with a full 0.9/-2.0-style fold at various lift heights first
# -- consistently collided with arm_right's wrist/gripper (arm_right_5/6/7
# and gripper_right_base), because at that little yaw separation arm_left's
# folded forearm reaches into the same roughly-chest-height zone arm_right's
# tuck already occupies, regardless of arm_left's own lift.
#
# Fix: fold LOW instead of trying to out-yaw arm_right. Small yaw (0.2rad,
# barely off centre -- not "going to the side"), low lift (0.15, just
# enough to clear the floor -- previous floor hits all had lift=0), and a
# full elbow fold (-2.0) to keep the forearm pulled in close to the body
# rather than extended. This puts the whole forearm near hip height,
# vertically clear of arm_right's wrist zone instead of needing lateral
# separation from it. Confirmed valid=True across every head tilt this
# codebase uses, zero /contacts through both the rest pose and the full
# URDF-zero -> this pose transition.
ARM_LEFT_DOWN_JOINTS = {
    'arm_left_1_joint': 0.2,
    'arm_left_2_joint': 0.15,
    'arm_left_3_joint': 0.0,
    'arm_left_4_joint': -2.0,
    'arm_left_5_joint': 0.0,
    'arm_left_6_joint': 0.0,
    'arm_left_7_joint': 0.0,
}

# torso_lift_joint range is [0, 0.35] (URDF limit). Raising it moves the
# whole torso (and both arms' mount point) up, directly trading reach at
# the bottom of the workspace for reach at the top -- useful since a
# z=0.9 book point was already confirmed reachable at torso=0 (this
# node's very first successful test), so only books ABOVE that need
# compensation. TORSO_REACH_BASELINE_Z is that "no lift needed" reference,
# not a measured limit of the arm's real workspace -- STILL TO TUNE if
# grasps at the top rows keep failing even with this.
TORSO_REACH_BASELINE_Z = 1.0
TORSO_LIFT_MAX = 0.35
TORSO_LIFT_VELOCITY = 0.035  # m/s, URDF velocity limit -- used to size the wait

# Lateral pre-grasp alignment. navigation_node's approach_shelf never strafes
# (see INTERFACES.md's "1.83m lateral miss" note and book_point_detector's own
# module docstring) -- it only rotates to center the column marker, then
# drives straight in, so the base can stop laterally off-center from the
# actual book by an amount the arm's own reach can't absorb. Confirmed live
# via /compute_ik sweeps against a real failed grasp (top shelf row, book at
# y=0.38m off centerline): the arm_right + full torso lift combination
# reached y=0.0 out to x=0.7m forward, but couldn't reach y=0.38m at ANY
# forward distance, including much closer ones. Height wasn't the limiting
# factor -- (x=0.6, y=0.0, z=1.58) planned fine -- lateral offset was. Since
# the base is a holonomic mecanum drive (can strafe directly, not just
# rotate+drive), close that gap here, right before planning, instead of
# reaching for a pose the arm genuinely cannot attain. This does mean
# manipulation_node commands /cmd_vel directly, normally navigation_node's
# job per INTERFACES.md -- justified because this correction is specifically
# about making the arm's own target reachable, not general navigation.
LATERAL_ALIGN_TOLERANCE = 0.12  # m -- how close to the book's y=0 (base-centered) counts as aligned
LATERAL_KP = 1.0
MAX_LATERAL_SPEED = 0.15        # m/s -- slow, precise strafe, not a navigation-speed drive
LATERAL_ALIGN_TIMEOUT = 8.0     # s
LATERAL_CONTROL_PERIOD = 0.1    # s
# Real physical motion per cycle at MAX_LATERAL_SPEED is ~0.015m -- a jump
# far beyond that within one cycle means a different book (see the
# rejection comment in _align_laterally_to_book), not real tracked motion.
MAX_PLAUSIBLE_Y_JUMP = 0.15     # m per LATERAL_CONTROL_PERIOD


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
        # Fixed place pose (planning frame) -- see STILL TO TUNE above.
        self.declare_parameter('place_x', 0.6)
        self.declare_parameter('place_y', 0.0)
        self.declare_parameter('place_z', 0.9)
        self.declare_parameter('gripper_open', 0.04)
        self.declare_parameter('gripper_closed', 0.0)
        # Loose on purpose, same reasoning as orientation_tolerance below --
        # the gripper doesn't need millimeter precision to close around a
        # book, and a too-tight position tolerance shrinks the goal region
        # OMPL has to sample into, on top of the orientation constraint.
        #
        # 0.03 still failed ("Unable to sample any valid states for goal
        # tree") for a pregrasp hover reached via the middle-shelf debug
        # bypass, which doesn't laterally centre on any specific column
        # the way a real column-search approach does -- the book ends up
        # at a more awkward angle/distance than the straight-on case this
        # was tuned against. Widened further.
        self.declare_parameter('position_tolerance', 0.06)
        # Tighter tolerance used ONLY for the final grasp-pose move (closing
        # in on the book itself), not the pregrasp hover above. The hover's
        # 0.06 has to stay loose to keep OMPL solvable from an imperfect
        # approach, but that same looseness on the actual grasp let the
        # gripper land close-but-off-centre on the book -- observed live via
        # screenshot: gripper closed near the book but knocked it onto the
        # floor instead of enclosing it, and the old contact check (any
        # touch, once, right after closing) treated that knock as a
        # successful grasp.
        #
        # 0.015 was too tight in practice -- "Unable to sample any valid
        # states for goal tree" even with orientation_tolerance still at
        # 1.2, since the position sphere alone shrank too far for OMPL to
        # find a sample inside it. Backed off to 0.03: still meaningfully
        # tighter than the pregrasp's 0.06 (halves the positional slop for
        # the precision-critical final approach) while remaining solvable.
        self.declare_parameter('grasp_position_tolerance', 0.03)
        # Loose on purpose: a tight tolerance around the fixed identity
        # orientation this node always requests ("gripper pointing however
        # identity means in the planning frame") turns a merely-awkward but
        # physically reachable position into "Unable to sample any valid
        # states for goal tree" (OMPL can't find ANY IK solution satisfying
        # both position and a narrow orientation band). 0.05 failed a
        # pregrasp hover at ~0.77m forward, ~0.93m up (well within reach);
        # 0.4 fixed that but still failed the grasp pose itself (~0.15m
        # deeper than a hover that succeeded at 0.4) -- widened further.
        # 0.8 still failed for an off-centre approach (see position_tolerance
        # above) -- widened again.
        self.declare_parameter('orientation_tolerance', 1.2)
        # 5.0 repeatedly hit "Unable to sample any valid states for goal
        # tree" on live runs where the target wasn't a pure reach-limit
        # case (torso well under TORSO_LIFT_MAX, book height moderate) --
        # RRTConnect's random sampling needs more attempts/time for some
        # approach geometries even within otherwise-reachable tolerances.
        # Widened to give it more chances before giving up.
        self.declare_parameter('planning_time', 10.0)
        # 5.0 sometimes too short: observed live, GRASP can start right as
        # SEEK_BOOK leaves the head mid-scan (see ROW_SEARCH_HEAD_TILTS in
        # state_machine_node.py), so the first several frames in this window
        # can be noisy/off-target before a genuinely good view arrives.
        # Widened to give book_point_detector more chances at a clean frame
        # rather than giving up on what's actually just early noise.
        #
        # 10.0 still timed out on a run where book_point_detector's own
        # distance readings were smoothly converging (2.85m -> 2.49m,
        # trending toward book_point_detector's MAX_PLAUSIBLE_DISTANCE) but
        # hadn't crossed the threshold by the 10s mark -- widened again to
        # give a still-converging detection room to finish.
        self.declare_parameter('point_wait_timeout', 15.0)
        self.declare_parameter('grasp_contact_timeout', 2.0)
        # Generous on purpose -- this bounds plan+execute together, not
        # just planning, and execution time depends on Gazebo's real-time
        # factor (can be well under 1x), not just trajectory length.
        self.declare_parameter('result_timeout', 120.0)

        self.planning_group = self.get_parameter('planning_group').value
        self.eef_link = self.get_parameter('eef_link').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.approach_offset = (
            self.get_parameter('approach_dx').value,
            self.get_parameter('approach_dy').value,
            self.get_parameter('approach_dz').value,
        )
        self.place_xyz = (
            self.get_parameter('place_x').value,
            self.get_parameter('place_y').value,
            self.get_parameter('place_z').value,
        )
        self.gripper_open = self.get_parameter('gripper_open').value
        self.gripper_closed = self.get_parameter('gripper_closed').value
        self.pos_tol = self.get_parameter('position_tolerance').value
        self.grasp_pos_tol = self.get_parameter('grasp_position_tolerance').value
        self.orient_tol = self.get_parameter('orientation_tolerance').value
        self.planning_time = self.get_parameter('planning_time').value
        self.point_wait_timeout = self.get_parameter('point_wait_timeout').value
        self.result_timeout = self.get_parameter('result_timeout').value
        self.grasp_contact_timeout = self.get_parameter('grasp_contact_timeout').value

        cb_group = ReentrantCallbackGroup()

        # --- MoveGroup action client (talks to the already-running move_group) ---
        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=cb_group)

        # --- IK client + joint-state tracking: see move_to_pose_via_ik ---
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=cb_group)
        self.latest_joint_state = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10,
            callback_group=cb_group)

        # --- Gripper: direct topic publish, matching the controller interface ---
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_right_controller/joint_trajectory', 10)

        # --- Lateral pre-grasp alignment: direct /cmd_vel strafe -- see
        # LATERAL_ALIGN_TOLERANCE above for why this lives here ---
        self.strafe_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- arm_left: direct topic publish too -- see ARM_LEFT_DOWN_JOINTS ---
        self.arm_left_pub = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)

        # --- torso_lift: direct topic publish, for high-book reach compensation ---
        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self._torso_lift_current = 0.0  # tracked so _command_torso knows travel distance

        # --- TF for transforming perception output into the planning frame ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Perception input: real book position from book_point_detector.py ---
        self.latest_book_point = None
        self.create_subscription(
            PointStamped, '/erc/target_book_point', self._on_book_point, 10,
            callback_group=cb_group)

        # --- Contact sensing, for verifying a grasp actually touched the book
        # (see module docstring for why this exists) ---
        self.latest_contacts = None
        self.create_subscription(
            Contacts, '/contacts', self._on_contacts, 10, callback_group=cb_group)

        # --- Manipulation service contract (state_machine_node calls these) ---
        self.create_service(
            GraspBook, '/erc/grasp_book', self._on_grasp_book, callback_group=cb_group)
        self.create_service(
            PlaceInBin, '/erc/place_in_bin', self._on_place_in_bin, callback_group=cb_group)

        self.get_logger().info(
            'Manipulation node ready (right arm) -- waiting for grasp_book/place_in_bin calls')

        # One-shot: tuck once move_group is actually up. Can't just call
        # move_to_tuck() here -- nothing is spinning yet at __init__ time,
        # so the action client's futures would never resolve.
        self._startup_tuck_done = False
        self._startup_tuck_timer = self.create_timer(
            1.0, self._startup_tuck, callback_group=cb_group)

    def _startup_tuck(self):
        if self._startup_tuck_done:
            return
        self._startup_tuck_done = True
        self._startup_tuck_timer.cancel()
        # Strictly sequential: stow_arm_left() now blocks until that motion
        # is actually complete before returning, so arm_right's tuck can't
        # start moving until arm_left is already clear -- previously these
        # overlapped (stow_arm_left published and returned immediately,
        # then move_to_tuck's own MoveGroup planning delay meant arm_right
        # sometimes started moving while arm_left was still mid-motion).
        self.stow_arm_left()
        if not self.move_to_tuck():
            self.get_logger().warn('startup tuck failed -- arm may still be at the default pose')

    def stow_arm_left(self, duration_sec: float = 2.0):
        """Direct joint command, not MoveGroup -- see ARM_LEFT_DOWN_JOINTS.
        Blocks until the motion is actually complete (see _startup_tuck)."""
        msg = JointTrajectory()
        msg.joint_names = list(ARM_LEFT_DOWN_JOINTS.keys())
        point = JointTrajectoryPoint()
        point.positions = list(ARM_LEFT_DOWN_JOINTS.values())
        point.time_from_start.sec = int(duration_sec)
        msg.points = [point]
        self.arm_left_pub.publish(msg)
        time.sleep(duration_sec + 0.5)

    def _command_torso(self, position: float):
        """Direct joint command, not MoveGroup (torso_lift_joint isn't part
        of the arm_right planning group either). Blocks until the torso
        has had time to actually get there -- see TORSO_LIFT_VELOCITY --
        so the subsequent MoveGroup plan is against the robot's real
        current state, not a mid-motion one."""
        travel = abs(position - self._torso_lift_current)
        if travel <= 1e-3:
            return
        msg = JointTrajectory()
        msg.joint_names = ['torso_lift_joint']
        point = JointTrajectoryPoint()
        point.positions = [position]
        travel_time = travel / TORSO_LIFT_VELOCITY
        point.time_from_start.sec = int(travel_time) + 1
        msg.points = [point]
        self.torso_pub.publish(msg)
        time.sleep(travel_time + 0.5)
        self._torso_lift_current = position

    def raise_torso_for_height(self, target_z: float):
        lift = max(0.0, min(TORSO_LIFT_MAX, target_z - TORSO_REACH_BASELINE_Z))
        if lift <= 0.0:
            return
        self.get_logger().info(
            f'raising torso {lift:.2f}m to reach a book at z={target_z:.2f}')
        self._command_torso(lift)

    def reset_torso(self):
        """Back to baseline -- place_in_bin's fixed pose (place_x/y/z) is
        tuned assuming torso=0, so a raise left over from grasp_book would
        throw it off by however much lift was applied."""
        self._command_torso(0.0)

    def _recover_to_tuck(self):
        """Best-effort cleanup on a failed grasp_book: a pregrasp hover that
        DID succeed but was then followed by a failed grasp-pose plan left
        the arm sitting at whatever the IK/OMPL solver picked for that
        hover -- confirmed live via /joint_states this can be a valid but
        wildly contorted configuration (e.g. arm_right_1_joint solved to
        -3.56 rad, visually reads as the arm having spun most of the way
        around), not just an awkward-looking rest. Also resets the torso,
        which raise_torso_for_height may have left lifted. Never leave the
        robot sitting in that state indefinitely -- return to the known-
        clear tuck pose so a failed attempt doesn't look (or behave) like
        the robot is stuck/broken."""
        self.set_gripper(self.gripper_open)
        self.reset_torso()
        if not self.move_to_tuck():
            self.get_logger().warn('recovery tuck after failed grasp did not complete')

    def _align_laterally_to_book(self):
        """Blocking: strafe sideways on the holonomic base until the book's
        y (in planning_frame, i.e. how far off the robot's centerline it
        is) is within LATERAL_ALIGN_TOLERANCE, or LATERAL_ALIGN_TIMEOUT
        elapses. Best-effort -- see LATERAL_ALIGN_TOLERANCE above for why
        this exists. No-ops immediately if already aligned or if the point
        stream/TF isn't available."""
        start = time.monotonic()
        deadline = start + LATERAL_ALIGN_TIMEOUT
        first_y = None
        last_y = None
        iterations = 0
        rejected = 0
        last_log = start
        while time.monotonic() < deadline:
            point = self.latest_book_point
            if point is None:
                self.get_logger().warn('lateral align: latest_book_point went None mid-loop -- stopping')
                break
            transformed = self.transform_to_planning_frame(point)
            if transformed is None:
                self.get_logger().warn('lateral align: TF transform failed mid-loop -- stopping')
                break
            y = transformed.point.y

            # Outlier rejection: a genuinely tracked book's y can only drift
            # by a tiny amount per control cycle (at MAX_LATERAL_SPEED, real
            # physical motion is ~0.015m per LATERAL_CONTROL_PERIOD). Seen
            # live: as the base strafes past roughly a column's width, an
            # ADJACENT column's same-coloured book can enter frame and
            # book_point_detector's largest-blob-of-target-colour logic can
            # jump to tracking THAT book instead -- observed as a ~0.7m
            # single-cycle jump in y that then dragged the base the wrong
            # way and never converged. A jump this large in one cycle is
            # physically impossible for the same tracked book, so treat it
            # as a bad sample and hold the previous command rather than
            # reacting to what's very likely a different book entirely.
            if last_y is not None and abs(y - last_y) > MAX_PLAUSIBLE_Y_JUMP:
                rejected += 1
                time.sleep(LATERAL_CONTROL_PERIOD)
                continue

            if first_y is None:
                first_y = y
            last_y = y
            iterations += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                last_log = now
                self.get_logger().info(
                    f'lateral align: t={now - start:.1f}s y={y:.3f} iterations={iterations} rejected={rejected}')
            if abs(y) <= LATERAL_ALIGN_TOLERANCE:
                break
            cmd = Twist()
            cmd.linear.y = max(-MAX_LATERAL_SPEED, min(MAX_LATERAL_SPEED, LATERAL_KP * y))
            self.strafe_pub.publish(cmd)
            time.sleep(LATERAL_CONTROL_PERIOD)
        self.strafe_pub.publish(Twist())
        time.sleep(0.3)  # let residual motion settle before replanning
        self.get_logger().info(
            f'lateral align: first_y={first_y} last_y={last_y} iterations={iterations} '
            f'elapsed={time.monotonic() - start:.1f}s '
            f'{"within" if last_y is not None and abs(last_y) <= LATERAL_ALIGN_TOLERANCE else "OUTSIDE"} '
            f'tolerance={LATERAL_ALIGN_TOLERANCE}')

    # ------------------------------------------------------------------
    def _on_book_point(self, msg: PointStamped):
        self.latest_book_point = msg

    def _wait_for_point(self, attr_name):
        deadline = time.monotonic() + self.point_wait_timeout
        while time.monotonic() < deadline:
            point = getattr(self, attr_name)
            if point is not None:
                return point
            time.sleep(0.05)
        return None

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

    # ------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg

    # ------------------------------------------------------------------
    def _on_contacts(self, msg: Contacts):
        self.latest_contacts = msg

    def check_grasp_contact(self) -> bool:
        """Poll /contacts for a real collision between a gripper_right_*
        link and any book link, within grasp_contact_timeout. This is the
        actual "did we grab something" check -- move_to_pose succeeding
        only means the commanded pose was reached, not that anything is
        between the fingers."""
        deadline = time.monotonic() + self.grasp_contact_timeout
        while time.monotonic() < deadline:
            contacts = self.latest_contacts
            if contacts is not None:
                for contact in contacts.contacts:
                    n1 = contact.collision1.name
                    n2 = contact.collision2.name
                    names = (n1, n2)
                    if (any('gripper_right' in n for n in names)
                            and any('book_' in n for n in names)):
                        self.get_logger().info(f'grasp contact confirmed: {n1} <-> {n2}')
                        return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    def point_to_pose(self, point_stamped: PointStamped) -> PoseStamped:
        return self.make_pose(
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z,
            point_stamped.header.frame_id)

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

    # ------------------------------------------------------------------
    def pose_to_constraints(self, pose_stamped: PoseStamped, link_name: str,
                             position_tolerance: float = None) -> Constraints:
        constraints = Constraints()
        pos_tol = self.pos_tol if position_tolerance is None else position_tolerance

        pos_constraint = PositionConstraint()
        pos_constraint.header = pose_stamped.header
        pos_constraint.link_name = link_name
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
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

    def move_to_pose(self, pose_stamped: PoseStamped, position_tolerance: float = None) -> bool:
        """Blocking helper: plan AND execute a move to the given pose."""
        return self._send_goal_and_wait(
            [self.pose_to_constraints(pose_stamped, self.eef_link, position_tolerance)])

    def move_to_pose_via_ik(self, pose_stamped: PoseStamped,
                             position_tolerance: float = None) -> bool:
        """Prefer this over move_to_pose for precision-critical moves
        (the pregrasp hover and the grasp pose itself).

        move_to_pose hands OMPL/RRTConnect a whole region (a position
        sphere + a wide orientation cone) and lets it sample ANY state
        inside -- it has no notion of "closest to where the arm already
        is". Checked live with /compute_ik on a pose that had been
        failing with "Unable to sample any valid states for goal tree":
        an IK solution existed (error_code=SUCCESS) but was a wildly
        different, contorted configuration from the current tucked pose
        (e.g. arm_right_1_joint solved to -3.76 rad vs. the tuck's -0.8).
        RRTConnect burning its whole sampling budget on distant, awkward
        solutions like that -- instead of the natural nearby one -- is
        exactly what "not using proper inverse kinematics" looks like.

        Seeds /compute_ik with the arm's actual current joint state so it
        returns a solution *near* where the arm already is, then plans a
        joint-space goal directly to that solution (tight tolerance,
        short path) instead of leaving OMPL to explore a whole pose
        region. Falls back to the old pose-region move_to_pose if IK
        itself fails to find any solution (a genuine reachability limit,
        not a sampling problem)."""
        if self.latest_joint_state is None or not self.ik_client.wait_for_service(timeout_sec=2.0):
            return self.move_to_pose(pose_stamped, position_tolerance)

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.eef_link
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.robot_state = RobotState(joint_state=self.latest_joint_state)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 2

        future = self.ik_client.call_async(req)
        result = self._block_on_future(future, timeout_sec=5.0)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().warn(
                'move_to_pose_via_ik: IK failed '
                f'(error_code={result.error_code.val if result else "no response"}), '
                'falling back to pose-region planning')
            return self.move_to_pose(pose_stamped, position_tolerance)

        solution_names = list(result.solution.joint_state.name)
        solution_positions = list(result.solution.joint_state.position)
        joint_positions = {
            name: pos for name, pos in zip(solution_names, solution_positions)
            if name.startswith('arm_right_')
        }
        return self.move_to_joint_positions(joint_positions)

    def joint_positions_to_constraints(self, joint_positions: dict) -> Constraints:
        constraints = Constraints()
        for joint_name, position in joint_positions.items():
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        return constraints

    def move_to_joint_positions(self, joint_positions: dict) -> bool:
        """Blocking helper: plan AND execute a move to a joint-space goal
        (e.g. TUCK_JOINTS), instead of a Cartesian pose."""
        return self._send_goal_and_wait(
            [self.joint_positions_to_constraints(joint_positions)])

    def move_to_tuck(self) -> bool:
        return self.move_to_joint_positions(TUCK_JOINTS)

    def _send_goal_and_wait(self, goal_constraints: list) -> bool:
        """Shared by move_to_pose and move_to_joint_positions: send a
        MoveGroup goal, block for acceptance then for the plan+execute
        result. Safe to call from a service callback under
        MultiThreadedExecutor -- polls the futures instead of calling
        spin_until_future_complete (see module docstring for why)."""
        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.goal_constraints = goal_constraints
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
        # Separate, much longer timeout than the 30s default: that default
        # is sized for a goal-acceptance handshake, not a full plan+execute.
        # Seen this cut off a real, successful execution at 30.35s under
        # GUI-mode Gazebo (slower than headless) -- move_group's own log
        # showed "Solution was found and executed" a third of a second
        # after we'd already given up and reported failure.
        result_wrapper = self._block_on_future(result_future, timeout_sec=self.result_timeout)
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
            response.message = (
                'timed out waiting for /erc/target_book_point -- book_point_detector '
                'has not detected the target colour (out of view?)')
            return response

        book_point = self.transform_to_planning_frame(book_point)
        if book_point is None:
            response.success = False
            response.message = f'TF transform of target_book_point to {self.planning_frame} failed'
            return response

        self._align_laterally_to_book()

        # Strafing moved the base, so the point above is stale (it was
        # relative to the pre-strafe base_footprint) -- wait for a fresh
        # detection against the post-strafe pose rather than reusing it.
        self.latest_book_point = None
        book_point = self._wait_for_point('latest_book_point')
        if book_point is None:
            response.success = False
            response.message = (
                'lost /erc/target_book_point after lateral alignment strafe')
            return response
        book_point = self.transform_to_planning_frame(book_point)
        if book_point is None:
            response.success = False
            response.message = (
                f'TF transform of target_book_point to {self.planning_frame} '
                'failed (post-align)')
            return response

        book_pose = self.point_to_pose(book_point)
        dx, dy, dz = self.approach_offset
        pregrasp = self.offset_pose(book_pose, dx, dy, dz)
        bp = book_pose.pose.position
        self.get_logger().info(
            f'grasp target (post-align, {self.planning_frame}): '
            f'x={bp.x:.3f} y={bp.y:.3f} z={bp.z:.3f}')

        # Target coordinates are in the ground-referenced planning frame, so
        # they don't change as the torso rises -- this only buys the arm
        # more relative reach for a high target, see raise_torso_for_height.
        self.raise_torso_for_height(book_pose.pose.position.z)

        if not self.move_to_pose_via_ik(pregrasp):
            response.success = False
            response.message = 'failed to plan/move to pre-grasp hover pose'
            self._recover_to_tuck()
            return response

        self.set_gripper(self.gripper_open)

        # Tighter tolerance than the pregrasp hover for this final approach
        # -- see grasp_position_tolerance declaration above for why.
        if not self.move_to_pose_via_ik(book_pose, position_tolerance=self.grasp_pos_tol):
            response.success = False
            response.message = 'failed to plan/move to grasp pose'
            self._recover_to_tuck()
            return response

        self.set_gripper(self.gripper_closed)
        contact_at_close = self.check_grasp_contact()
        self.move_to_pose_via_ik(pregrasp)  # retreat clear of the shelf; best-effort
        # Re-check AFTER retreating, not just right after closing. A book
        # that's genuinely held moves with the gripper through the retreat
        # and is still in contact; one that was only grazed/knocked (the
        # actual failure seen live -- book ended up on the floor while the
        # service still reported success) gets left behind and the contact
        # disappears the moment the arm starts moving away. Checking only
        # once, right at closing, can't tell these apart.
        contact_after_retreat = self.check_grasp_contact()
        grasped = contact_at_close and contact_after_retreat
        self.reset_torso()  # back to baseline before navigate_to_bin/place_in_bin

        if not grasped:
            self.set_gripper(self.gripper_open)
            response.success = False
            response.message = (
                'gripper closed but the book was not carried through the retreat '
                f'(contact at close={contact_at_close}, after retreat={contact_after_retreat}) '
                '-- grasp missed, likely knocked rather than held')
            return response

        response.success = True
        response.message = 'book grasped (contact confirmed)'
        return response

    def _on_place_in_bin(self, request, response):
        # Fixed pose, not a live point -- see module docstring.
        px, py, pz = self.place_xyz
        bin_pose = self.make_pose(px, py, pz, self.planning_frame)
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
        self.move_to_tuck()  # back to resting pose; best-effort

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
