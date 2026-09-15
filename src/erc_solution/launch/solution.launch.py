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
    debug_skip_to_bin = LaunchConfiguration('debug_skip_to_bin_after_column')

    # manipulation_node talks to move_group over the MoveGroup action
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
        DeclareLaunchArgument(
            'debug_skip_to_bin_after_column', default_value='false',
            description='Dev-only: skip SEEK_BOOK/GRASP/PLACE and drive straight to the '
                        'bin after reaching the column, to test bin_detector standalone.'),

        # Perception -- all three (shelf_number_detector, book_color_detector,
        # bin_detector) are the real erc_perception implementations.
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
            package='erc_perception',
            executable='bin_detector',
            name='bin_detector',
            output='screen',
        ),
        # book_color_detector only publishes shelf_row_identification, not a
        # 3D point -- this fills the /erc/target_book_point gap manipulation
        # needs, by back-projecting the same colour-blob detection through
        # the depth camera (see book_detector.py).
        Node(
            package='erc_solution',
            executable='book_detector',
            name='book_detector',
            output='screen',
            parameters=[{'target_book_colour': book_colour}],
        ),

        # Navigation
        Node(
            package='erc_solution',
            executable='navigation_node',
            name='navigation_node',
            output='screen',
        ),

        # Manipulation -- move_group (MoveIt) must be up before
        # manipulation_node's first grasp_book call, which blocks on
        # the /move_action action server.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(move_group_launch),
        ),
        Node(
            package='erc_solution',
            executable='manipulation_node',
            name='manipulation_node',
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
                'debug_skip_to_bin_after_column': debug_skip_to_bin,
            }],
        ),
    ])
