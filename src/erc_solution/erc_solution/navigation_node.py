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
# 1.0 parked the base ~0.98m from the shelf front, putting the book face
# ~1.04m ahead: the grasp pose then needs the tool ~0.92m out, just past
# arm_right's ~0.9m forward reach, and the final straight-line move in was
# only 67% feasible (seed 11, yellow row 4). 0.85 closes that gap.
SHELF_STANDOFF = 0.85
# Park well back from the bin: the book is carried below table height, and
# place_in_bin lifts it above the rim before driving the last stretch in.
BIN_STANDOFF = 1.1

BOOK_APPROACH_HEAD_TILT = -0.15  # rad, tilt down a bit for the books

# Only arm_right is in the MoveIt planning group (see tiago_pro.srdf) --
# arm_left has no group, so it can't be planned around and just sits in
# whatever pose it was left in. At spawn/zero that's stretched out to the
# side, which both juts into the shelf approach and can clip the head
# camera's view of the books. Tuck it down alongside the torso (elbow
# bent, forearm roughly vertical) before the right arm goes to work, so
# it's physically out of the way instead of relying on MoveIt collision
# checking to route around it.
ARM_LEFT_JOINT_NAMES = (
    'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
    'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint')
# PAL's own "home" motion for arm_left (tiago_pro_bringup/config/motions/
# tiago_pro_motions_general_arm_left.yaml), same waypoints PAL uses to get
# there without self-collision. The previous hand-picked (0,-1.5,0,-2,0,0,0)
# left the fingertips 0.62m ahead of base_footprint -- ~0.34m past the front
# LiDAR, so a LiDAR-based standoff still let the hand hit the shelf (seen
# on /contacts as arm_left_6_link vs erc_shelf). /compute_fk puts this pose's
# furthest point at x=0.34m.
ARM_LEFT_TUCK_WAYPOINTS = (
    ((1.8557, -1.5919, 0.35538, -2.0502, 0.10524, -1.5976, 0.0), 3),
    ((0.26, -1.6008, 0.3489, -1.9818, 0.0, -1.2, 0.0), 6),
    ((0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0), 9),
)

# arm_right's spawn pose also reaches 0.97m forward at shelf-bottom height,
# so it gets PAL's mirrored home too before the drive in. manipulation_node
# plans from wherever it's left, so this doesn't fight MoveIt.
ARM_RIGHT_JOINT_NAMES = tuple(name.replace('left', 'right') for name in ARM_LEFT_JOINT_NAMES)
ARM_RIGHT_TUCK_WAYPOINTS = (
    ((-1.8614, -1.6008, -0.34892, -1.9818, 0.10153, -1.2, 0.0), 3),
    ((-0.26, -1.6008, -0.3489, -1.9818, 0.0, -1.2, 0.0), 6),
    ((-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0), 9),
)

# PAL's home torso height -- at 0.0 the tucked arms press into base_link
# (MoveIt's /check_state_validity agrees), so raise to this before tucking.
TORSO_HOME = 0.10          # m
TORSO_TOLERANCE = 0.01     # m
TORSO_TIMEOUT = 300.0      # s, wall time; the sim torso tracks slowly (~0.13x real time)

# In Gazebo the arms settle ~0.1-0.15 rad short of PAL's home on joints 2/4
# (pressed against the torso), which still leaves both hands within ~0.4m
# of base_footprint -- close enough. The 9s trajectory is in sim time, and
# the sim runs well under real time here, so the wall-clock wait is generous.
ARM_TUCK_TOLERANCE = 0.2   # rad, per joint
ARM_TUCK_TIMEOUT = 300.0   # s, wall time

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
        self.arm_left_pub = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right_pub = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10, callback_group=cb_group)
        self.create_subscription(LaserScan, '/scan_front_raw', self._front_scan_cb, 10, callback_group=cb_group)
        self.create_subscription(LaserScan, '/scan_rear_raw', self._rear_scan_cb, 10, callback_group=cb_group)
        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10, callback_group=cb_group)

        self.pose = None  # (x, y, yaw), set once the first /odom message lands
        self.front_min_range = math.inf
        self.rear_min_range = math.inf
        self.wheel_velocities = {}  # joint name -> latest velocity (rad/s)
        self.arm_positions = {}  # arm joint name -> latest position (rad)

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
        for name in ARM_LEFT_JOINT_NAMES + ARM_RIGHT_JOINT_NAMES + ('torso_lift_joint',):
            if name in msg.name:
                self.arm_positions[name] = msg.position[msg.name.index(name)]

    def _min_wheel_speed(self):
        """Slowest wheel right now, or None until we've heard from all 4.
        Checking the min (not an average) means one dead wheel can't hide
        behind three healthy ones."""
        if len(self.wheel_velocities) < len(WHEEL_JOINT_NAMES):
            return None
        return min(abs(v) for v in self.wheel_velocities.values())

    def _approach_shelf_cb(self, request, response):
        """No waypoint math, no odom position target - state_machine_node
        has already spun until perception confirmed the target column, so
        the shelf is roughly dead ahead. Just drive straight forward until
        the front LiDAR says we're within SHELF_STANDOFF or the safety
        timeout hits, then tilt the head down for book detection."""
        # Tuck before driving in, not after -- the arm has to be clear while
        # the base closes the gap, or it's the thing that hits the shelf.
        if not self._tuck_arms():
            response.success = False
            response.message = 'arms did not reach the tuck pose'
            return response

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

    def _arm_at(self, joint_names, positions):
        return all(
            name in self.arm_positions
            and abs(self.arm_positions[name] - target) <= ARM_TUCK_TOLERANCE
            for name, target in zip(joint_names, positions))

    def _set_torso_home(self):
        def at_home():
            pos = self.arm_positions.get('torso_lift_joint')
            return pos is not None and abs(pos - TORSO_HOME) <= TORSO_TOLERANCE
        if at_home():
            return True
        msg = JointTrajectory()
        msg.joint_names = ['torso_lift_joint']
        point = JointTrajectoryPoint()
        point.positions = [TORSO_HOME]
        point.time_from_start.sec = 5
        msg.points = [point]
        self.torso_pub.publish(msg)
        deadline = time.monotonic() + TORSO_TIMEOUT
        while time.monotonic() < deadline:
            if at_home():
                return True
            time.sleep(0.2)
        self.get_logger().warn(
            f'torso not at {TORSO_HOME}m after {TORSO_TIMEOUT}s: {self.arm_positions.get("torso_lift_joint")}')
        return False

    def _tuck_arms(self):
        """Fold both arms into PAL's home poses -- see ARM_*_TUCK_WAYPOINTS
        above. Blocks until /joint_states confirms both got there."""
        if not self._set_torso_home():
            return False
        arms = (
            (self.arm_left_pub, ARM_LEFT_JOINT_NAMES, ARM_LEFT_TUCK_WAYPOINTS),
            (self.arm_right_pub, ARM_RIGHT_JOINT_NAMES, ARM_RIGHT_TUCK_WAYPOINTS),
        )
        pending = []
        for pub, names, waypoints in arms:
            if self._arm_at(names, waypoints[-1][0]):
                continue
            msg = JointTrajectory()
            msg.joint_names = list(names)
            for positions, t in waypoints:
                point = JointTrajectoryPoint()
                point.positions = list(positions)
                point.time_from_start.sec = t
                msg.points.append(point)
            pub.publish(msg)
            pending.append((names, waypoints[-1][0]))
        deadline = time.monotonic() + ARM_TUCK_TIMEOUT
        while time.monotonic() < deadline:
            if all(self._arm_at(names, target) for names, target in pending):
                return True
            time.sleep(0.2)
        self.get_logger().warn(f'arms not tucked after {ARM_TUCK_TIMEOUT}s: {self.arm_positions}')
        return False

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
