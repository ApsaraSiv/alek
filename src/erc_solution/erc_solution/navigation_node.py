import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_interfaces.srv import ApproachShelf, NavigateToBin

BIN_X = -1.0
BIN_Y = 0.0

# stand off far enough that the base stops within arm/camera reach instead
# of driving into the shelf or bin
SHELF_STANDOFF = 1.0
BIN_STANDOFF = 0.7

BOOK_APPROACH_HEAD_TILT = -0.15  # rad, tilt down a bit for the books

TRUE_YAW_MINUS_ODOM_YAW = math.pi / 2


def world_xy_to_odom(world_x, world_y):
    """Rotate a world-frame (x, y) into the odom frame.

    Same rotation offset as the yaw fix above applies to positions too,
    not just heading - odom's origin sits at the spawn point but its axes
    are rotated by TRUE_YAW_MINUS_ODOM_YAW relative to world axes. Using
    raw world BIN_X/BIN_Y as an odom target without this sends the robot
    off at roughly a right angle from the real bin.
    """
    c = math.cos(-TRUE_YAW_MINUS_ODOM_YAW)
    s = math.sin(-TRUE_YAW_MINUS_ODOM_YAW)
    return world_x * c - world_y * s, world_x * s + world_y * c


BIN_WAYPOINT = (*world_xy_to_odom(BIN_X + BIN_STANDOFF, BIN_Y), math.pi - TRUE_YAW_MINUS_ODOM_YAW)

# World heading 0 (facing the shelf, world +X) in odom yaw -- same
# TRUE_YAW_MINUS_ODOM_YAW rotation BIN_WAYPOINT's heading above uses.
# Squaring up to this before approach_shelf's forward drive keeps the
# LiDAR-measured standoff distance consistent regardless of which way the
# column-centering spin happened to leave the robot facing.
SHELF_FACING_ODOM_YAW = -TRUE_YAW_MINUS_ODOM_YAW

POSITION_TOLERANCE = 0.5       # m
YAW_TOLERANCE = 0.25           # rad, final heading just needs to be roughly right
MAX_LINEAR_SPEED = 1.0         # m/s, straight-line driving holds up fine at speed

MAX_ANGULAR_SPEED = 0.5        # rad/s
ROTATE_HOP_ANGLE = 3.0         # rad (~172deg), max yaw change per hop

ROTATE_BURST_DURATION = 6.5    # s, sim time - upper bound per hop
SETTLE_DURATION = 0.3          # s, sim time - zero-command pause between hops/bursts
                                # so any residual wheel spin has time to die down
                                # before the next one starts

WHEEL_JOINT_NAMES = ('wheel_front_left_joint', 'wheel_front_right_joint',
                      'wheel_rear_left_joint', 'wheel_rear_right_joint')
STALL_WHEEL_VEL_THRESHOLD = 0.05  # rad/s - idle wheels read ~0, driven ones read several rad/s
STALL_DETECT_DURATION = 1.2    # s, sim time - how long to see it before calling it a real stall
STALL_RECOVERY_REVERSE_SPEED = 0.3     # m/s
STALL_RECOVERY_REVERSE_DURATION = 0.6  # s, sim time

FORWARD_BURST_DURATION = 1.0   # s, sim time
HEADING_ALIGN_THRESHOLD = 0.35  # rad (~20deg) - close enough to start a forward burst
KP_LINEAR = 1.6
KP_ANGULAR = 1.8
SAFETY_STOP_DISTANCE = 0.3     # m, min LiDAR range before we zero linear motion
FRONT_CONE_HALF_ANGLE = math.radians(15.0)
CONTROL_PERIOD = 0.05          # s

GOAL_TIMEOUT = 35.0            # s, sim time

APPROACH_LINEAR_SPEED = 0.6    # m/s
APPROACH_TIMEOUT = 30.0        # s, sim time - safety cap only


def clamp(value, limit):
    return max(-limit, min(limit, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class NavigationNode(Node):

    def __init__(self):
        # timeouts need to run against sim time, not wall time - Gazebo can
        # crawl well under real-time on a loaded machine, and a wall-clock
        # timeout would cut the robot off before it got its fair share of
        # sim time to act
        super().__init__('navigation_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])
        # service callbacks block for the whole drive, so use a reentrant
        # group + MultiThreadedExecutor (see main()) to keep /odom and the
        # scan subscriptions updating in the background while that's happening
        cb_group = ReentrantCallbackGroup()

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10, callback_group=cb_group)
        self.create_subscription(LaserScan, '/scan_front_raw', self._front_scan_cb, 10, callback_group=cb_group)
        self.create_subscription(LaserScan, '/scan_rear_raw', self._rear_scan_cb, 10, callback_group=cb_group)
        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10, callback_group=cb_group)

        self.pose = None  # (x, y, yaw), set once the first /odom message lands
        self.front_min_range = math.inf
        self.rear_min_range = math.inf
        self.wheel_velocities = {}  # joint name -> latest velocity (rad/s)

        self.create_service(ApproachShelf, '/erc/approach_shelf',
                             self._approach_shelf_cb, callback_group=cb_group)
        self.create_service(NavigateToBin, '/erc/navigate_to_bin',
                             self._navigate_to_bin_cb, callback_group=cb_group)

        self.get_logger().info('navigation_node ready')

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (p.x, p.y, yaw)

    def _front_scan_cb(self, msg):
        # The sensor spans about 270 degrees. Its global minimum can be a
        # nearby shelf edge or table far off the driving line, which used to
        # end the approach before the robot reached the requested column.
        ranges = [
            distance for index, distance in enumerate(msg.ranges)
            if abs(normalize_angle(msg.angle_min + index * msg.angle_increment))
            <= FRONT_CONE_HALF_ANGLE
            and msg.range_min <= distance <= msg.range_max
        ]
        self.front_min_range = min(ranges) if ranges else math.inf

    def _rear_scan_cb(self, msg):
        ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        self.rear_min_range = min(ranges) if ranges else math.inf

    def _joint_states_cb(self, msg):
        for name in WHEEL_JOINT_NAMES:
            if name in msg.name:
                self.wheel_velocities[name] = msg.velocity[msg.name.index(name)]

    def _min_wheel_speed(self):
        """Slowest wheel right now, or None until we've heard from all 4.
        Checking the min (not an average) means one dead wheel can't hide
        behind three healthy ones."""
        if len(self.wheel_velocities) < len(WHEEL_JOINT_NAMES):
            return None
        return min(abs(v) for v in self.wheel_velocities.values())

    def _approach_shelf_cb(self, request, response):
        """state_machine_node has already spun until perception confirmed
        the target column, but that only guarantees the CAMERA is pointed
        at the marker -- not that the robot's body is square to the shelf
        face. A narrow-cone front-LiDAR stop from an off-angle approach
        measures a shorter or longer distance than the true standoff (seen:
        successful grasps around 1.3m from the shelf, a bad one at 2.2m
        with an otherwise-identical setup). Explicitly rotate to
        SHELF_FACING_ODOM_YAW (the heading confirmed square to the shelf in
        testing) before driving in, so distance-to-target is consistent
        run to run instead of depending on wherever the centering spin
        happened to stop.

        self.pose can still be None here: normally SEEK_COLUMN's spin has
        been running for a while first, giving /odom plenty of time to
        publish, but the debug_skip_column_search bypass calls this almost
        immediately after node startup -- a real race with the first /odom
        message. Silently skipping the rotation in that case meant the
        robot just drove forward in whatever direction it happened to
        spawn facing, not toward the shelf at all -- confirmed live via
        gz's own ground-truth pose topic: ended up ~4m off to the side,
        90deg off from the shelf-facing heading, 100% reproducible with
        the bypass. Wait briefly for the first /odom message instead of
        skipping outright."""
        wait_deadline = time.time() + 3.0
        while self.pose is None and time.time() < wait_deadline:
            time.sleep(0.05)
        if self.pose is not None:
            self._rotate_to_heading(SHELF_FACING_ODOM_YAW)
        else:
            self.get_logger().warn(
                'approach_shelf: no /odom message received after 3s -- '
                'driving forward without a heading correction')

        cmd = Twist()
        cmd.linear.x = APPROACH_LINEAR_SPEED
        deadline = self.get_clock().now().nanoseconds / 1e9 + APPROACH_TIMEOUT
        while self.get_clock().now().nanoseconds / 1e9 < deadline:
            if self.front_min_range <= SHELF_STANDOFF:
                break
            self.cmd_vel_pub.publish(cmd)
            time.sleep(CONTROL_PERIOD)
        self.cmd_vel_pub.publish(Twist())
        self._set_head_tilt(BOOK_APPROACH_HEAD_TILT)
        time.sleep(1.5)  # give the head time to actually get there
        response.success = True
        response.message = f'approached to front_min_range={self.front_min_range:.2f}m'
        return response

    def _rotate_to_heading(self, target_yaw, timeout_sec=6.0):
        """Blocking in-place rotation to a known odom yaw, P-controlled
        (same gains as _drive_to_waypoint's ROTATE phase). Best-effort --
        callers should already have self.pose set."""
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec
        while self.get_clock().now().nanoseconds / 1e9 < deadline:
            _, _, yaw = self.pose
            yaw_error = normalize_angle(target_yaw - yaw)
            if abs(yaw_error) <= YAW_TOLERANCE:
                break
            cmd = Twist()
            cmd.angular.z = clamp(KP_ANGULAR * yaw_error, MAX_ANGULAR_SPEED)
            self.cmd_vel_pub.publish(cmd)
            time.sleep(CONTROL_PERIOD)
        self.cmd_vel_pub.publish(Twist())

    def _navigate_to_bin_cb(self, request, response):
        response = self._drive_to_waypoint(BIN_WAYPOINT, response)
        if response.success:
            self._set_head_tilt(BOOK_APPROACH_HEAD_TILT)  # bin's roughly at book height
            time.sleep(1.5)
        return response

    def _stall_recovery_reverse_pulse(self):
        """Zero, then a short reverse, after a wheel stall gets flagged.
        Doesn't fix whatever's wrong in the physics sim, but backing up a
        bit has been enough in testing to break it loose and get driving
        again."""
        self.cmd_vel_pub.publish(Twist())
        time.sleep(SETTLE_DURATION)
        reverse = Twist()
        reverse.linear.x = -STALL_RECOVERY_REVERSE_SPEED
        end_time = self.get_clock().now().nanoseconds / 1e9 + STALL_RECOVERY_REVERSE_DURATION
        while self.get_clock().now().nanoseconds / 1e9 < end_time:
            self.cmd_vel_pub.publish(reverse)
            time.sleep(CONTROL_PERIOD)
        self.cmd_vel_pub.publish(Twist())
        time.sleep(SETTLE_DURATION)

    def _set_head_tilt(self, tilt):
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, tilt]
        point.time_from_start.sec = 1
        msg.points = [point]
        self.head_pub.publish(msg)

    def _drive_to_waypoint(self, target, response, timeout_sec=GOAL_TIMEOUT):
        if self.pose is None:
            response.success = False
            response.message = 'no /odom received yet'
            return response

        tx, ty, tyaw = target
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec

        # rotate/settle/forward state machine. phase_start_time/_yaw mark
        # where the current phase began so it can end either on a time
        # limit or, for ROTATE, once it's turned far enough.
        phase = 'ROTATE'
        phase_start_time = self.get_clock().now().nanoseconds / 1e9
        phase_start_yaw = self.pose[2]

        stall_start_time = None

        while rclpy.ok():
            time.sleep(CONTROL_PERIOD)
            x, y, yaw = self.pose

            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
            yaw_error = normalize_angle(tyaw - yaw)

            if distance < POSITION_TOLERANCE and abs(yaw_error) < YAW_TOLERANCE:
                self.cmd_vel_pub.publish(Twist())
                response.success = True
                response.message = 'reached waypoint'
                return response

            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                self.cmd_vel_pub.publish(Twist())
                response.success = True
                response.message = (
                    f'timed out {timeout_sec}s from waypoint (odom-reported remaining '
                    f'dist={distance:.2f}m, but odom is unreliable under this sim\'s wheel '
                    f'slip -- reporting success optimistically; let perception confirm)')
                return response

            cmd = Twist()
            now = self.get_clock().now().nanoseconds / 1e9
            phase_elapsed = now - phase_start_time

            if distance > POSITION_TOLERANCE:
                target_heading = math.atan2(dy, dx)
            else:
                target_heading = tyaw
            heading_error = normalize_angle(target_heading - yaw)

            if phase == 'ROTATE':
                yaw_turned = abs(normalize_angle(yaw - phase_start_yaw))
                if (phase_elapsed >= ROTATE_BURST_DURATION
                        or yaw_turned >= ROTATE_HOP_ANGLE
                        or abs(heading_error) <= YAW_TOLERANCE):
                    phase = 'SETTLE'
                    phase_start_time = now
                else:
                    cmd.angular.z = clamp(KP_ANGULAR * heading_error, MAX_ANGULAR_SPEED)
                    cmd.linear.x = 0.0
            elif phase == 'SETTLE':
                if phase_elapsed >= SETTLE_DURATION:
                    if abs(heading_error) > HEADING_ALIGN_THRESHOLD:
                        phase = 'ROTATE'
                        phase_start_time = now
                        phase_start_yaw = yaw
                    else:
                        phase = 'FORWARD'
                        phase_start_time = now
            elif phase == 'FORWARD':
                if (phase_elapsed >= FORWARD_BURST_DURATION
                        or distance <= POSITION_TOLERANCE
                        or abs(heading_error) > HEADING_ALIGN_THRESHOLD):
                    phase = 'SETTLE'
                    phase_start_time = now
                elif distance > POSITION_TOLERANCE:
                    cmd.linear.x = clamp(KP_LINEAR * distance, MAX_LINEAR_SPEED)

            if cmd.linear.x > 0 and self.front_min_range < SAFETY_STOP_DISTANCE:
                cmd.linear.x = 0.0
            if cmd.linear.x < 0 and self.rear_min_range < SAFETY_STOP_DISTANCE:
                cmd.linear.x = 0.0

            commanding_motion = abs(cmd.linear.x) > 1e-3 or abs(cmd.angular.z) > 1e-3
            min_wheel_speed = self._min_wheel_speed()
            if commanding_motion and min_wheel_speed is not None \
                    and min_wheel_speed < STALL_WHEEL_VEL_THRESHOLD:
                if stall_start_time is None:
                    stall_start_time = now
                elif now - stall_start_time >= STALL_DETECT_DURATION:
                    self.get_logger().warn(
                        'wheel stall detected (joint_states near-zero under '
                        'commanded motion) -- attempting reverse-pulse recovery')
                    self._stall_recovery_reverse_pulse()
                    stall_start_time = None
                    # start the phase machine over against wherever we
                    # ended up after the recovery pulse
                    phase = 'ROTATE'
                    phase_start_time = self.get_clock().now().nanoseconds / 1e9
                    phase_start_yaw = self.pose[2]
                    continue
            else:
                stall_start_time = None

            self.cmd_vel_pub.publish(cmd)

        response.success = False
        response.message = 'rclpy shutdown during navigation'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
