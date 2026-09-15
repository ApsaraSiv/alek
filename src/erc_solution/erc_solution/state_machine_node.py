import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_interfaces.srv import ApproachShelf, NavigateToBin, GraspBook, PlaceInBin

SERVICE_WAIT_TIMEOUT = 15.0     # s -- how long to wait for a service server to appear
MANIPULATION_WAIT_TIMEOUT = 3.0  # s -- shorter: manipulation_node may not exist yet
ROW_WAIT_TIMEOUT = 30.0        # s
POLL_PERIOD = 0.1               # s

SPIN_ANGULAR_SPEED = -0.4      # rad/s; initial turn toward the shelf from spawn
TRACKING_ANGULAR_SPEED = 0.12  # rad/s; correction stays below one column per OCR frame
SPIN_HEAD_TILT = 0.3           # rad, tilt up so the overhead marker is in frame
SPIN_CONTROL_PERIOD = 0.05     # s
COLUMN_ERROR_STALE_TIME = 1.0  # s, wall time; OCR is intentionally throttled
CENTER_LOCK_TOLERANCE = 0.035  # normalized image width, matches detector's tolerance
CENTER_SETTLE_DURATION = 0.5   # s, wall time - let residual rotation die down once
                                # centred so two consecutive OCR frames can land there
SPIN_MAX_DURATION = 180.0       # s, sim time - safety cap only, not a normal stop condition


class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])
        self.declare_parameter('shelf_column_number', 0)
        self.declare_parameter('book_colour', '')
        # manipulation_node (erc_solution/manipulation_node.py) is wired in
        # via solution.launch.py, so a missing grasp/place service means
        # something actually broke rather than "not implemented yet" --
        # fail the run instead of silently skipping. Override back to true
        # for pipeline-only testing without the arm/MoveIt stack running.
        self.declare_parameter('skip_manipulation_if_unavailable', False)

        self.declare_parameter('debug_skip_to_bin_after_column', False)

        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value
        self.skip_manip = self.get_parameter('skip_manipulation_if_unavailable').value
        self.debug_skip_to_bin = self.get_parameter('debug_skip_to_bin_after_column').value

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

    def _run_once(self):
        if self._started:
            return
        self._started = True
        self._run()

    def _run(self):
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
        continuous spin directly and stop the INSTANT perception confirms
        the marker - never based on any position/yaw estimate."""
        self._publish_head_tilt(SPIN_HEAD_TILT)
        time.sleep(1.0)  # give the head a moment to actually get there

        spin_start = time.time()
        deadline = self.get_clock().now().nanoseconds / 1e9 + SPIN_MAX_DURATION
        twist = Twist()
        twist.angular.z = SPIN_ANGULAR_SPEED
        last_correction_sign = None

        self.get_logger().info('SEEK_COLUMN: spinning until shelf_column_number is seen')

        while self.get_clock().now().nanoseconds / 1e9 < deadline:
            if (self.column_error_at is not None
                    and time.time() - self.column_error_at <= COLUMN_ERROR_STALE_TIME):
                if abs(self.column_error) <= CENTER_LOCK_TOLERANCE:
                    
                    self.cmd_vel_pub.publish(Twist())
                    time.sleep(CENTER_SETTLE_DURATION)
                    if self.column_confirmed_at is not None and self.column_confirmed_at >= spin_start:
                        break
                    continue
                last_correction_sign = math.copysign(1.0, self.column_error)
                twist.angular.z = -math.copysign(TRACKING_ANGULAR_SPEED, self.column_error)
            elif last_correction_sign is not None:
                twist.angular.z = -math.copysign(TRACKING_ANGULAR_SPEED, last_correction_sign)
            else:
                twist.angular.z = SPIN_ANGULAR_SPEED
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
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec
        while self.target_row is None and self.get_clock().now().nanoseconds / 1e9 < deadline:
            time.sleep(POLL_PERIOD)
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
