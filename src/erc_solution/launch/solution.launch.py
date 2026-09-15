import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')

    # arm_manipulation_node talks to move_group over the MoveGroup action
    # interface -- nothing else in the bringup starts it, so it has to be
    # brought up alongside the rest of the solution.
    move_group_launch = os.path.join(
        get_package_share_directory('tiago_pro_right_arm_moveit_config'),
        'launch', 'move_group.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'shelf_column_number',
            description='Target shelf column number (1-5), provided by the evaluator'),
        DeclareLaunchArgument(
            'book_colour',
            description='Target book colour (red|blue|green|yellow), provided by the evaluator'),

        # Perception -- shelf_number_detector and book_color_detector are the
        # real (erc_perception) implementations. bin_detector still points
        # at erc_solution's empty stub since erc_perception's bin_detector.py
        # hasn't been delivered yet -- swap this back to erc_perception once
        # it exists.
        Node(
            package='erc_perception',
            executable='shelf_number_detector',
            name='shelf_number_detector',
            output='screen',
            parameters=[{'target_shelf_column_number': shelf_column_number}],
        ),
        Node(
            package='erc_perception',
            executable='book_color_detector',
            name='book_color_detector',
            output='screen',
            parameters=[{'target_colour': book_colour}],
        ),
        Node(
            package='erc_solution',
            executable='bin_detector',
            name='bin_detector',
            output='screen',
        ),

        # Navigation
        Node(
            package='erc_solution',
            executable='navigation_node',
            name='navigation_node',
            output='screen',
        ),

        # Manipulation -- move_group (MoveIt) must be up before
        # arm_manipulation_node's first grasp_book call, which blocks on
        # the /move_action action server.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(move_group_launch),
        ),
        Node(
            package='aleksandria',
            executable='arm_manipulation_node',
            name='arm_manipulation_node',
            output='screen',
        ),

        # Integration / orchestrator — owns the overall task state machine
        Node(
            package='erc_solution',
            executable='state_machine_node',
            name='state_machine_node',
            output='screen',
            parameters=[{
                'shelf_column_number': shelf_column_number,
                'book_colour': book_colour,
            }],
        ),
    ])
