import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_interfaces.srv import ApproachShelf, NavigateToBin, GraspBook, PlaceInBin

SERVICE_WAIT_TIMEOUT = 15.0     # s -- how long to wait for a service server to appear
MANIPULATION_WAIT_TIMEOUT = 3.0  # s -- shorter: manipulation_node may not exist yet
ROW_WAIT_TIMEOUT = 30.0        # s
POLL_PERIOD = 0.1               # s

# SEEK_BOOK previously just polled at whatever fixed head tilt
# navigation_node's approach_shelf left it at (BOOK_APPROACH_HEAD_TILT,
# -0.15) -- if the target-coloured book isn't in view at exactly that
# tilt (approach standoff/alignment varies run to run), book_color_detector
# never sees it and shelf_row_identification never arrives, so the wait
# just times out. Actively nudge tilt while waiting instead of sitting at
# one fixed angle forever.
#
# book_color_detector's row number comes from *which quarter of the
# current frame* the blob's centre falls in -- it assumes the whole
# shelf column (all 4 rows) is visible in one frame at the calibrated
# tilt, not an absolute vertical measurement. A wide sweep would report
# wrong row numbers at tilts where that assumption breaks (e.g. only
# rows 2-4 in frame), so this only nudges in a narrow band around the
# known-working tilt rather than sweeping the full range.
ROW_SEARCH_HEAD_TILTS = [-0.15, -0.05, -0.25]  # rad, small deviations
                                                 # around BOOK_APPROACH_HEAD_TILT
ROW_SEARCH_TILT_SETTLE = 1.0    # s -- let the head actually get there and
                                 # book_color_detector process a frame
                                 # before moving on to the next tilt

SPIN_ANGULAR_SPEED = -0.4      # rad/s; initial turn toward the shelf from spawn
TRACKING_ANGULAR_SPEED = 0.25  # rad/s; was 0.12, but live testing showed the
                                # column error decreasing only ~0.006/s at that
                                # speed under this sim's wheel slip -- far too
                                # slow to reach CENTER_LOCK_TOLERANCE inside
                                # LOST_TRACK_TIMEOUT, so it kept losing lock
                                # and restarting the search before ever
                                # converging
SPIN_HEAD_TILT = 0.3           # rad, tilt up so the overhead marker is in frame
SPIN_CONTROL_PERIOD = 0.05     # s
COLUMN_ERROR_STALE_TIME = 1.0  # s, wall time; OCR is intentionally throttled
LOST_TRACK_TIMEOUT = 3.0       # s, wall time -- if the marker has been out of
                                # view this long while crawling toward its last
                                # known side, it's genuinely lost (not just
                                # between OCR frames): resume a full search
                                # sweep instead of creeping blindly forever
BRAKE_SETTLE_DURATION = 0.4    # s, wall time -- kill momentum when first
                                # catching the marker, before switching from
                                # SPIN_ANGULAR_SPEED to the much slower
                                # TRACKING_ANGULAR_SPEED
CENTER_LOCK_TOLERANCE = 0.06   # normalized image width, matches detector's
                                # CENTER_TOLERANCE_FRACTION -- keep these two in
                                # sync, see that file for why this was widened
CENTER_SETTLE_DURATION = 0.5   # s, wall time - let residual rotation die down once
                                # centred so two consecutive OCR frames can land there
SPIN_MAX_DURATION = 180.0       # s, wall time - safety cap only, not a normal stop
                                 # condition. Was sim-time, but under this sim's real-time
                                 # factor (measured as low as ~8-38% in GUI mode) a
                                 # sim-time deadline can take many minutes of real time to
                                 # elapse -- switched to wall-clock so the cap (and the
                                 # analogous ROW_WAIT_TIMEOUT wait) actually behaves like
                                 # the number of real seconds it says.

STEP_BURST_DURATION = 0.6      # s, wall time -- ~14deg of rotation per burst at
                                # SPIN_ANGULAR_SPEED. While still just searching
                                # (no fresh column error), rotate in short bursts
                                # with a full stop between them instead of one
                                # long continuous spin. Measured via gz's own
                                # ground-truth pose topic: a long uninterrupted
                                # spin physically walks this skid-steer base
                                # off its start point (>1.8m drift observed
                                # after a search that never caught the marker
                                # even once) -- it isn't just odom yaw error,
                                # the robot really translates from sustained
                                # wheel scrub while rotating. That drift can
                                # walk the camera into a nearby wall partway
                                # through the sweep, permanently losing line
                                # of sight to the marker for the rest of the
                                # search. Short bursts + full stops bound the
                                # drift per burst and also hand OCR a
                                # motion-blur-free stationary frame each time.
STEP_SETTLE_DURATION = 0.5     # s, wall time -- pause after each burst, stationary,
                                # so a clean OCR frame has time to arrive & be processed
WALL_STANDOFF_MIN = 0.35       # m -- front LiDAR minimum range below this means the
                                # base has drifted close enough to a wall that further
                                # rotation risks getting physically wedged against it
                                # (observed live: two camera frames grabbed several
                                # seconds apart during a step-scan were pixel-identical
                                # -- the base was commanded to rotate but wasn't
                                # actually turning, front LiDAR read ~0.58m into a
                                # corner at the time). Step bursts bound drift per
                                # burst but don't prevent it from *accumulating* over
                                # many bursts across a long search, so this is a
                                # separate, explicit recovery rather than relying on
                                # burst size alone.
                                #
                                # First tried 0.55m -- immediately over-triggered at
                                # spawn (still basically at (0,0) by ground truth,
                                # not drifted at all) because erc_table sits close
                                # enough to spawn to read ~0.5-0.55m on some sweep
                                # headings even though it's not actually a collision
                                # risk while just rotating in place. Tightened to
                                # 0.35m so a merely-nearby fixture doesn't trigger
                                # this, only genuine near-contact.
BACKUP_LINEAR_SPEED = -0.15    # m/s -- reverse speed for the wall recovery backup
BACKUP_DURATION = 1.0          # s, wall time -- enough to clear WALL_STANDOFF_MIN
                                # from a corner at BACKUP_LINEAR_SPEED with margin
MAX_CONSECUTIVE_BACKUPS = 3    # safety cap -- if backing up repeatedly doesn't
                                # clear WALL_STANDOFF_MIN (e.g. a fixture that's
                                # just always nearby, not actually blocking), give
                                # up backing away and resume the sweep anyway
                                # rather than looping forever


class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])
        self.declare_parameter('shelf_column_number', 0)
        self.declare_parameter('book_colour', '')
        
        self.declare_parameter('skip_manipulation_if_unavailable', True)

        self.declare_parameter('debug_skip_to_bin_after_column', False)
        # Bypasses _spin_until_column_found() entirely and drives straight to
        # approach_shelf -- for testing the approach/grasp/place pipeline when
        # column-marker perception itself is the thing under investigation.
        # navigation_node's approach_shelf squares up to a fixed absolute
        # heading (SHELF_FACING_ODOM_YAW) and drives forward regardless of
        # which column was found, so skipping the search still lands the
        # robot centred on the shelf -- with the robot spawning centred on
        # the shelf's own centre column, that's the middle column.
        self.declare_parameter('debug_skip_column_search', False)

        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value
        self.skip_manip = self.get_parameter('skip_manipulation_if_unavailable').value
        self.debug_skip_to_bin = self.get_parameter('debug_skip_to_bin_after_column').value
        self.debug_skip_column_search = self.get_parameter('debug_skip_column_search').value

        cb_group = ReentrantCallbackGroup()

        self.target_row = None
        self.column_confirmed_at = None  # timestamp of last shelf_number_detector match
        self.column_error = None
        self.column_error_at = None

        self.create_subscription(
            Int32, '/erc/shelf_column_identification',
            self._on_column_identified, 10, callback_group=cb_group)
        self.create_subscription(
            Float32, '/erc/shelf_column_horizontal_error',
            self._on_column_error, 10, callback_group=cb_group)
        self.create_subscription(
            Int32, '/erc/shelf_row_identification',
            self._on_row_identified, 10, callback_group=cb_group)

        self.front_scan_min_range = None
        self.create_subscription(
            LaserScan, '/scan_front_raw',
            self._on_front_scan, 10, callback_group=cb_group)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)

        self.approach_client = self.create_client(
            ApproachShelf, '/erc/approach_shelf', callback_group=cb_group)
        self.nav_bin_client = self.create_client(
            NavigateToBin, '/erc/navigate_to_bin', callback_group=cb_group)
        self.grasp_client = self.create_client(
            GraspBook, '/erc/grasp_book', callback_group=cb_group)
        self.place_client = self.create_client(
            PlaceInBin, '/erc/place_in_bin', callback_group=cb_group)

        self.state = 'INIT'
        self.get_logger().info(
            f'Starting solution: shelf_column_number={self.target_column} '
            f'book_colour={self.target_colour}')

        self._started = False
        self.create_timer(0.5, self._run_once, callback_group=cb_group)

    def _on_column_identified(self, msg: Int32):
        if msg.data != self.target_column:
            self.get_logger().warn(
                f'ignoring unexpected shelf_column_identification={msg.data}; '
                f'target is {self.target_column}')
            return
        self.column_confirmed_at = time.time()
        self.get_logger().info(f'[perception] shelf_column_identification={msg.data}')

    def _on_row_identified(self, msg: Int32):
        if self.target_row is None:
            self.target_row = msg.data
            self.get_logger().info(f'[perception] shelf_row_identification={msg.data}')

    def _on_column_error(self, msg: Float32):
        self.column_error = msg.data
        self.column_error_at = time.time()

    def _on_front_scan(self, msg: LaserScan):
        finite = [r for r in msg.ranges if r == r and r > 0.0]
        self.front_scan_min_range = min(finite) if finite else None

    def _run_once(self):
        if self._started:
            return
        self._started = True
        self._run()

    def _run(self):
        if self.debug_skip_column_search:
            self.get_logger().warn(
                'debug_skip_column_search=true -- skipping SEEK_COLUMN entirely, '
                'driving straight to approach_shelf')
        else:
            self._set_state('SEEK_COLUMN')
            if not self._spin_until_column_found():
                self._fail(f'did not find shelf_column_number={self.target_column} while spinning')
                return

        if not self.approach_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT):
            self._fail('approach_shelf service unavailable')
            return
        result = self._call(self.approach_client, ApproachShelf.Request())
        if not result.success:
            self._fail(f'approach_shelf failed: {result.message}')
            return

        if self.debug_skip_to_bin:
            self.get_logger().warn(
                'debug_skip_to_bin_after_column=true -- skipping SEEK_BOOK/GRASP/PLACE, '
                'waiting 10s then heading straight to the bin to exercise bin_detector')
            time.sleep(10.0)
            self._goto_bin_debug_only()
            return

        self._set_state('SEEK_BOOK')
        if not self._wait_for_row(ROW_WAIT_TIMEOUT):
            self._fail('timed out waiting for /erc/shelf_row_identification')
            return

        self._set_state('GRASP')
        self.get_logger().info(f'target_row={self.target_row} book_colour={self.target_colour}')
        if self.grasp_client.wait_for_service(timeout_sec=MANIPULATION_WAIT_TIMEOUT):
            result = self._call(self.grasp_client, GraspBook.Request(row=self.target_row))
            if not result.success:
                self._fail(f'grasp_book failed: {result.message}')
                return
        elif self.skip_manip:
            self.get_logger().warn(
                'manipulation_node not up (grasp_book unavailable) -- skipping GRASP '
                '(skip_manipulation_if_unavailable=true) to exercise the rest of the '
                'pipeline anyway')
        else:
            self._fail('grasp_book service unavailable')
            return

        self._set_state('NAV_TO_BIN')
        if not self.nav_bin_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT):
            self._fail('navigate_to_bin service unavailable')
            return
        result = self._call(self.nav_bin_client, NavigateToBin.Request())
        if not result.success:
            self._fail(f'navigate_to_bin failed: {result.message}')
            return

        self._set_state('PLACE')
        if self.place_client.wait_for_service(timeout_sec=MANIPULATION_WAIT_TIMEOUT):
            result = self._call(self.place_client, PlaceInBin.Request())
            if not result.success:
                self._fail(f'place_in_bin failed: {result.message}')
                return
        elif self.skip_manip:
            self.get_logger().warn(
                'manipulation_node not up (place_in_bin unavailable) -- skipping PLACE')
        else:
            self._fail('place_in_bin service unavailable')
            return

        self._set_state('DONE')
        self.get_logger().info('Solution finished.')

    def _goto_bin_debug_only(self):
        """debug_skip_to_bin_after_column path: just drive to the bin and
        stop there (no grasp, no place, no book in hand) so bin_detector
        can be watched against a live camera feed of the actual bin."""
        self._set_state('NAV_TO_BIN')
        if not self.nav_bin_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT):
            self._fail('navigate_to_bin service unavailable')
            return
        result = self._call(self.nav_bin_client, NavigateToBin.Request())
        if not result.success:
            self._fail(f'navigate_to_bin failed: {result.message}')
            return
        self._set_state('DONE')
        self.get_logger().info(
            'Debug run finished: parked at the bin for bin_detector testing.')

    def _spin_until_column_found(self):
        """Bypasses navigation_node and odom entirely: severe wheel slip
        corrupts odom's own yaw estimate badly enough that a convergence
        check against it can self-report "aligned" long before the robot
        has actually turned that far (established separately). Command a
        spin directly and stop the INSTANT perception confirms the marker
        - never based on any position/yaw estimate.

        Search sweeping is step-and-stop, not one continuous spin: checked
        against gz sim's own ground-truth pose topic, a long uninterrupted
        spin genuinely translates this skid-steer base (>1.8m measured
        after a search that never caught the marker), not just odom's yaw
        -- confirmed by comparing that same drift against odom's own
        position estimate, which matched ground truth closely (it's
        specifically yaw that odom gets wrong here, not position). That
        drift can walk the camera into a nearby wall mid-sweep and
        permanently lose the marker for the remainder of the search. See
        STEP_BURST_DURATION above."""
        self._publish_head_tilt(SPIN_HEAD_TILT)
        time.sleep(1.0)  # give the head a moment to actually get there

        spin_start = time.time()
        deadline = spin_start + SPIN_MAX_DURATION
        twist = Twist()
        twist.angular.z = SPIN_ANGULAR_SPEED
        last_correction_sign = None
        consecutive_backups = 0

        self.get_logger().info('SEEK_COLUMN: spinning until shelf_column_number is seen')

        while time.time() < deadline:
            if (self.column_error_at is not None
                    and time.time() - self.column_error_at <= COLUMN_ERROR_STALE_TIME):
                if abs(self.column_error) <= CENTER_LOCK_TOLERANCE:
                    
                    self.cmd_vel_pub.publish(Twist())
                    time.sleep(CENTER_SETTLE_DURATION)
                    if self.column_confirmed_at is not None and self.column_confirmed_at >= spin_start:
                        break
                    continue
                if last_correction_sign is None:
                    # Just caught the marker while still sweeping at full
                    # SPIN_ANGULAR_SPEED -- brake first. Without this, the
                    # robot's momentum (this sim has severe wheel slip, see
                    # docstring above) carries it straight past the marker
                    # before the much slower TRACKING_ANGULAR_SPEED command
                    # actually takes effect, so it loses lock almost
                    # immediately -- observed as a catch-lose cycle
                    # repeating every ~14s without ever converging.
                    self.cmd_vel_pub.publish(Twist())
                    time.sleep(BRAKE_SETTLE_DURATION)
                last_correction_sign = math.copysign(1.0, self.column_error)
                twist.angular.z = -math.copysign(TRACKING_ANGULAR_SPEED, self.column_error)
            elif (last_correction_sign is not None
                    and time.time() - self.column_error_at <= LOST_TRACK_TIMEOUT):
                twist.angular.z = -math.copysign(TRACKING_ANGULAR_SPEED, last_correction_sign)
            else:
                if last_correction_sign is not None:
                    self.get_logger().info(
                        'SEEK_COLUMN: lost the marker while tracking -- '
                        'resuming step-scan sweep')
                    last_correction_sign = None
                # Wall recovery: if drift (bounded per-burst, but still
                # cumulative over a long search) has carried the base close
                # to an obstacle, back away before rotating further instead
                # of risking getting physically wedged against it -- see
                # WALL_STANDOFF_MIN docstring above. Capped at
                # MAX_CONSECUTIVE_BACKUPS: a fixture that's just always
                # somewhat nearby (not actually blocking) would otherwise
                # keep re-triggering this forever without ever clearing it.
                if (self.front_scan_min_range is not None
                        and self.front_scan_min_range < WALL_STANDOFF_MIN
                        and consecutive_backups < MAX_CONSECUTIVE_BACKUPS):
                    consecutive_backups += 1
                    self.get_logger().info(
                        f'SEEK_COLUMN: front range {self.front_scan_min_range:.2f}m < '
                        f'{WALL_STANDOFF_MIN}m -- backing away before continuing sweep '
                        f'(attempt {consecutive_backups}/{MAX_CONSECUTIVE_BACKUPS})')
                    backup = Twist()
                    backup.linear.x = BACKUP_LINEAR_SPEED
                    self.cmd_vel_pub.publish(backup)
                    time.sleep(BACKUP_DURATION)
                    self.cmd_vel_pub.publish(Twist())
                    time.sleep(STEP_SETTLE_DURATION)
                    continue
                consecutive_backups = 0
                # Step-and-stop instead of continuous spin -- see
                # STEP_BURST_DURATION docstring above for why.
                twist.angular.z = SPIN_ANGULAR_SPEED
                self.cmd_vel_pub.publish(twist)
                time.sleep(STEP_BURST_DURATION)
                self.cmd_vel_pub.publish(Twist())
                if self.column_confirmed_at is not None and self.column_confirmed_at >= spin_start:
                    break
                time.sleep(STEP_SETTLE_DURATION)
                continue
            self.cmd_vel_pub.publish(twist)
            if self.column_confirmed_at is not None and self.column_confirmed_at >= spin_start:
                break
            time.sleep(SPIN_CONTROL_PERIOD)
        else:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().warn(
                f'shelf_column_number={self.target_column} not confirmed after spinning for '
                f'{SPIN_MAX_DURATION}s')
            return False

        for _ in range(10):
            self.cmd_vel_pub.publish(Twist())
            time.sleep(SPIN_CONTROL_PERIOD)
        self.get_logger().info(
            f'shelf_column_number={self.target_column} found -- stopping and approaching')
        return True

    def _publish_head_tilt(self, tilt):
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, tilt]
        point.time_from_start.sec = 1
        msg.points = [point]
        self.head_pub.publish(msg)

    def _call(self, client, request):
        future = client.call_async(request)
        while not future.done():
            time.sleep(POLL_PERIOD)
        return future.result()

    def _wait_for_row(self, timeout_sec):
        # target_row can arrive opportunistically at ANY time -- book_color_detector
        # publishes whenever it happens to see the target colour, even mid-drive
        # during approach_shelf, well before the robot has settled into its final
        # resting position. If that already set target_row before this is even
        # called, the scan loop below never runs at all, and the head is left
        # wherever approach_shelf put it (a fixed tilt, not re-verified) -- which
        # may no longer have the book in view once the robot has actually
        # stopped. Observed live: row captured during the drive, GRASP started
        # ~85s later, book_point_detector never saw the book at all (timed out)
        # because nothing re-checked the head was actually pointed at it once
        # the robot had come to rest. Always do at least one settle at the
        # primary tilt before returning, regardless of whether target_row was
        # already known, so the head is verified-current before GRASP begins.
        deadline = time.time() + timeout_sec
        tilt_index = 0
        while time.time() < deadline:
            tilt = ROW_SEARCH_HEAD_TILTS[tilt_index % len(ROW_SEARCH_HEAD_TILTS)]
            self.get_logger().info(f'SEEK_BOOK: scanning at head tilt {tilt:.2f}rad')
            self._publish_head_tilt(tilt)
            tilt_index += 1
            settle_deadline = time.time() + ROW_SEARCH_TILT_SETTLE
            while time.time() < settle_deadline and time.time() < deadline:
                time.sleep(POLL_PERIOD)
            # Always complete at least this one settle before returning --
            # even if target_row was already known walking in, this confirms
            # the head is actually pointed at the book right now, not just
            # wherever it happened to be left.
            if self.target_row is not None:
                break
        return self.target_row is not None

    def _set_state(self, state):
        self.state = state
        self.get_logger().info(f'state={state}')

    def _fail(self, reason):
        self.state = 'FAILED'
        self.get_logger().error(f'state=FAILED: {reason}')


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
