#!/usr/bin/env python3
"""
Clamps incoming gripper JointTrajectory commands to a safe position range
before they reach the JointTrajectoryController, preventing commands past
the physical/URDF limit from ever reaching Gazebo.

Public topics  (subscribed, unchanged for teams):
    /gripper_left_controller/joint_trajectory
    /gripper_right_controller/joint_trajectory

Internal topics (published, feed the real controllers):
    /gripper_left_controller_raw/joint_trajectory
    /gripper_right_controller_raw/joint_trajectory

The controllers themselves are renamed to *_raw in controller_manager.yaml
and controller_params.yaml, freeing up the public topic names for this
node to own.
"""

import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from trajectory_msgs.msg import JointTrajectory

# public_controller_name -> (raw_controller_name, joint_name, (min, max))
GRIPPERS = {
    "gripper_left_controller": (
        "gripper_left_controller_raw",
        "gripper_left_finger_joint",
        (0.00, 0.069),
    ),
    "gripper_right_controller": (
        "gripper_right_controller_raw",
        "gripper_right_finger_joint",
        (0.00, 0.069),
    ),
}


class GripperClampChannel:
    """Owns one public subscription -> raw publisher pair for one gripper."""

    def __init__(self, node: Node, public_name: str, raw_name: str,
                 joint_name: str, limits: tuple):
        self._node = node
        self._joint_name = joint_name
        self._lo, self._hi = limits

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._pub = node.create_publisher(
            JointTrajectory, f"/{raw_name}/joint_trajectory", qos
        )
        node.create_subscription(
            JointTrajectory, f"/{public_name}/joint_trajectory",
            self._callback, qos
        )

    def _callback(self, msg: JointTrajectory):
        clamped = JointTrajectory()
        clamped.header = msg.header
        clamped.joint_names = msg.joint_names

        for point in msg.points:
            new_point = copy.deepcopy(point)
            reject = False

            for i, jname in enumerate(msg.joint_names):
                if jname != self._joint_name:
                    continue
                value = point.positions[i]

                if value > self._hi or value < self._lo:
                    self._node.get_logger().warn(
                        f"{jname}: value {value:.4f} out of range, enter a "
                        f"value between {self._lo:.2f} and {self._hi:.3f}. "
                        f"Command rejected."
                    )
                    reject = True

            if not reject:
                clamped.points.append(new_point)

        if clamped.points:
            self._pub.publish(clamped)


class GripperCommandClamp(Node):
    def __init__(self):
        super().__init__("gripper_command_clamp")
        self._channels = [
            GripperClampChannel(self, public_name, raw_name, joint_name, limits)
            for public_name, (raw_name, joint_name, limits) in GRIPPERS.items()
        ]
        self.get_logger().info(
            "Gripper command clamp active for: %s" % ", ".join(GRIPPERS.keys())
        )


def main():
    rclpy.init()
    node = GripperCommandClamp()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()