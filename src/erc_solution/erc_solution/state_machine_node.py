import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_msgs.msg import Int32

from erc_interfaces.srv import NavigateToColumn, NavigateToBin, GraspBook, PlaceInBin

SERVICE_WAIT_TIMEOUT = 15.0     # s -- how long to wait for a service server to appear
MANIPULATION_WAIT_TIMEOUT = 3.0  # s -- shorter: manipulation_node may not exist yet
ROW_WAIT_TIMEOUT = 30.0        # s
SLOT_CONFIRM_TIMEOUT = 6.0     # s -- how long to wait, per slot, for shelf_number_detector
NUM_PHYSICAL_SLOTS = 5
POLL_PERIOD = 0.1               # s


class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine_node', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True),
        ])
        self.declare_parameter('shelf_column_number', 0)
        self.declare_parameter('book_colour', '')
        # lets us test the rest of the pipeline before manipulation_node is
        # done - GRASP/PLACE get skipped with a warning instead of failing
        # the run if the service isn't up. flip to false once manipulation
        # actually works so a real failure stops the run properly.
        self.declare_parameter('skip_manipulation_if_unavailable', True)

        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value
        self.skip_manip = self.get_parameter('skip_manipulation_if_unavailable').value

        cb_group = ReentrantCallbackGroup()

        self.target_row = None
        self.column_confirmed_at = None  # timestamp of last shelf_number_detector match

        self.create_subscription(
            Int32, '/erc/shelf_column_identification',
            self._on_column_identified, 10, callback_group=cb_group)
        self.create_subscription(
            Int32, '/erc/shelf_row_identification',
            self._on_row_identified, 10, callback_group=cb_group)

        self.nav_column_client = self.create_client(
            NavigateToColumn, '/erc/navigate_to_shelf_column', callback_group=cb_group)
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

        # runs the whole sequence once. ReentrantCallbackGroup +
        # MultiThreadedExecutor let this block on each service call while
        # perception subscriptions keep updating in the background - same
        # trick navigation_node uses, keeps this readable top-to-bottom
        # instead of a pile of async callbacks.
        self._started = False
        self.create_timer(0.5, self._run_once, callback_group=cb_group)

    def _on_column_identified(self, msg: Int32):
        # only publishes on a real match, so any message = confirmed at
        # whatever slot we're currently parked at
        self.column_confirmed_at = time.time()
        self.get_logger().info(f'[perception] shelf_column_identification={msg.data}')

    def _on_row_identified(self, msg: Int32):
        if self.target_row is None:
            self.target_row = msg.data
            self.get_logger().info(f'[perception] shelf_row_identification={msg.data}')

    def _run_once(self):
        if self._started:
            return
        self._started = True
        self._run()

    def _run(self):
        self._set_state('SEEK_COLUMN')
        if not self.nav_column_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT):
            self._fail('navigate_to_shelf_column service unavailable')
            return

        found_slot = None
        for slot in range(1, NUM_PHYSICAL_SLOTS + 1):
            self.get_logger().info(f'SEEK_COLUMN: trying physical slot {slot}')
            # grab the timestamp before driving there, and never reset
            # column_confirmed_at to None - resetting it raced with the
            # perception callback and dropped real confirmations. comparing
            # against a start time set once avoids that.
            attempt_start = time.time()
            # wide_scan backs off farther and tilts the head up - the
            # overhead marker is too high up to reliably see at the close
            # book-detection distance
            result = self._call(self.nav_column_client, NavigateToColumn.Request(
                column_index=slot, wide_scan=True))
            if not result.success:
                self.get_logger().warn(
                    f'navigate_to_shelf_column({slot}) failed: {result.message} -- '
                    f'trying next slot')
                continue

            deadline = self.get_clock().now().nanoseconds / 1e9 + SLOT_CONFIRM_TIMEOUT
            while self.get_clock().now().nanoseconds / 1e9 < deadline:
                if self.column_confirmed_at is not None and self.column_confirmed_at >= attempt_start:
                    found_slot = slot
                    break
                time.sleep(POLL_PERIOD)
            if found_slot is not None:
                self.get_logger().info(
                    f'shelf_column_number={self.target_column} found at physical slot {slot}')
                break
            self.get_logger().info(f'slot {slot} did not match, trying next')

        if found_slot is None:
            self._fail(
                f'scanned all {NUM_PHYSICAL_SLOTS} physical slots, none matched '
                f'shelf_column_number={self.target_column}')
            return

        # Close in from the wide scan standoff to the book-detection
        # standoff now that we know which physical slot is correct.
        result = self._call(self.nav_column_client, NavigateToColumn.Request(
            column_index=found_slot, wide_scan=False))
        if not result.success:
            self._fail(f'close approach to slot {found_slot} failed: {result.message}')
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
