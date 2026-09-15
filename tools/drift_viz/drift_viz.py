#!/usr/bin/env python3
"""Overlay planned waypoints, odometry path, and Gazebo ground truth in the
odom frame so odometry drift is visible in RViz. Read-only: subscribes to
/odom and samples Gazebo's pose topic; publishes only /drift_viz/*."""
import math
import re
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from erc_solution import navigation_node as nav

GZ_POSE_TOPIC = '/world/erc_world/dynamic_pose/info'
SHELF_FRONT_WORLD_X = 2.755
SHELF_HALF_WIDTH = 2.5
TIAGO_BLOCK = re.compile(
    r'name:\s*"tiago_pro".*?position\s*\{(?P<pos>[^}]*)\}', re.S)
AXIS_VALUE = re.compile(r'([xyz]):\s*(-?[\d.eE+-]+)')
STAMP = re.compile(r'stamp\s*\{\s*(?:sec:\s*(\d+))?\s*(?:nsec:\s*(\d+))?')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DriftViz(Node):
    def __init__(self):
        super().__init__('drift_viz', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('target_marker_world_y', 0.0)
        self.target_y = self.get_parameter('target_marker_world_y').value

        self.odom_path = Path()
        self.odom_path.header.frame_id = 'odom'
        self.true_path = Path()
        self.true_path.header.frame_id = 'odom'
        self.latest_odom = None
        self.latest_true = None
        self.cleared = False
        self.lock = threading.Lock()

        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.odom_pub = self.create_publisher(Path, '/drift_viz/odom_path', 5)
        self.true_pub = self.create_publisher(Path, '/drift_viz/true_path', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/drift_viz/markers', 5)
        self.create_timer(0.5, self._publish)
        self.create_timer(5.0, self._log_drift)

        threading.Thread(target=self._sample_ground_truth, daemon=True).start()
        bx, by, byaw = nav.BIN_WAYPOINT
        self.get_logger().info(
            f'drift_viz up: BIN_WAYPOINT(odom)=({bx:.2f},{by:.2f},{byaw:.2f}) '
            f'tolerance={nav.POSITION_TOLERANCE}m, world->odom rotation '
            f'from navigation_node.world_xy_to_odom')

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            self.latest_odom = (p.x, p.y, yaw_of(msg.pose.pose.orientation))
            self._append(self.odom_path, p.x, p.y, msg.header.stamp)

    def _append(self, path, x, y, stamp, min_step=0.02):
        if path.poses:
            last = path.poses[-1].pose.position
            if math.hypot(x - last.x, y - last.y) < min_step:
                return
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.header.stamp = stamp
        ps.pose.position.x, ps.pose.position.y = x, y
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)

    def _sample_ground_truth(self):
        while rclpy.ok():
            try:
                out = subprocess.run(
                    ['gz', 'topic', '-e', '-t', GZ_POSE_TOPIC, '-n', '1'],
                    capture_output=True, text=True, timeout=5).stdout
            except subprocess.TimeoutExpired:
                out = ''
            m = TIAGO_BLOCK.search(out)
            if m:
                axes = {k: float(v) for k, v in AXIS_VALUE.findall(m.group('pos'))}
                wx, wy = axes.get('x', 0.0), axes.get('y', 0.0)
                ox, oy = nav.world_xy_to_odom(wx, wy)
                with self.lock:
                    self.latest_true = (ox, oy, wx, wy)
                    self._append(self.true_path, ox, oy, self.get_clock().now().to_msg())
            time.sleep(1.0)

    def _publish(self):
        now = self.get_clock().now().to_msg()
        with self.lock:
            self.odom_path.header.stamp = now
            self.true_path.header.stamp = now
            self.odom_pub.publish(self.odom_path)
            self.true_pub.publish(self.true_path)
            odom, true = self.latest_odom, self.latest_true
        arr = self._markers(now, odom, true)
        if not self.cleared:
            # RViz keeps markers from earlier publishers (old text labels) until told otherwise.
            wipe = Marker()
            wipe.header.frame_id = 'odom'
            wipe.action = Marker.DELETEALL
            arr.markers.insert(0, wipe)
            self.cleared = True
        self.marker_pub.publish(arr)

    def _marker(self, now, mid, mtype, ns, rgba, scale=(0.1, 0.1, 0.1)):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = now
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.pose.orientation.w = 1.0
        return m

    def _markers(self, now, odom, true):
        arr = MarkerArray()
        bx, by, byaw = nav.BIN_WAYPOINT

        goal = self._marker(now, 0, Marker.ARROW, 'planned', (1.0, 0.25, 0.25, 1.0), (0.6, 0.1, 0.1))
        goal.pose.position.x, goal.pose.position.y = bx, by
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(byaw / 2), math.cos(byaw / 2)
        arr.markers.append(goal)
        ring = self._marker(now, 1, Marker.CYLINDER, 'planned', (1.0, 0.25, 0.25, 0.18),
                            (2 * nav.POSITION_TOLERANCE, 2 * nav.POSITION_TOLERANCE, 0.01))
        ring.pose.position.x, ring.pose.position.y = bx, by
        arr.markers.append(ring)

        shelf = self._marker(now, 3, Marker.LINE_STRIP, 'planned', (0.75, 0.75, 0.8, 1.0), (0.05, 0.0, 0.0))
        for wy in (-SHELF_HALF_WIDTH, SHELF_HALF_WIDTH):
            ox, oy = nav.world_xy_to_odom(SHELF_FRONT_WORLD_X, wy)
            shelf.points.append(Point(x=ox, y=oy, z=0.0))
        arr.markers.append(shelf)

        tx, ty = nav.world_xy_to_odom(SHELF_FRONT_WORLD_X, self.target_y)
        target = self._marker(now, 5, Marker.SPHERE, 'planned', (0.3, 0.6, 1.0, 1.0), (0.18, 0.18, 0.18))
        target.pose.position.x, target.pose.position.y = tx, ty
        arr.markers.append(target)

        spawn = self._marker(now, 7, Marker.SPHERE, 'planned', (1.0, 1.0, 1.0, 0.9), (0.12, 0.12, 0.12))
        arr.markers.append(spawn)

        if odom and true:
            gap = self._marker(now, 9, Marker.LINE_LIST, 'drift', (1.0, 0.1, 0.1, 1.0), (0.035, 0.0, 0.0))
            gap.points = [Point(x=odom[0], y=odom[1], z=0.05), Point(x=true[0], y=true[1], z=0.05)]
            arr.markers.append(gap)
        return arr

    def _log_drift(self):
        with self.lock:
            odom, true = self.latest_odom, self.latest_true
        if not (odom and true):
            self.get_logger().info(f'waiting for data: odom={odom is not None} truth={true is not None}')
            return
        drift = math.hypot(odom[0] - true[0], odom[1] - true[1])
        d_odom = math.hypot(odom[0], odom[1])
        d_true = math.hypot(true[0], true[1])
        angle = float('nan')
        if d_odom > 0.3 and d_true > 0.3:
            angle = math.degrees(math.atan2(odom[1], odom[0]) - math.atan2(true[1], true[0]))
            angle = (angle + 180.0) % 360.0 - 180.0
        self.get_logger().info(
            f'odom=({odom[0]:+.2f},{odom[1]:+.2f}) truth(odom frame)=({true[0]:+.2f},{true[1]:+.2f}) '
            f'truth(world)=({true[2]:+.2f},{true[3]:+.2f}) drift={drift:.3f}m '
            f'bearing_diff_from_spawn={angle:+.1f}deg')


def main():
    rclpy.init()
    node = DriftViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
