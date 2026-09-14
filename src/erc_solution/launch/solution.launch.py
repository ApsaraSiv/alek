from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')

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
        # book_color_detector only publishes shelf_row_identification, not a
        # 3D point -- this fills the /erc/target_book_point gap manipulation
        # needs. Was missing from this launch file entirely. PLACEHOLDER
        # point, not real detection (see book_detector.py).
        Node(
            package='erc_solution',
            executable='book_detector',
            name='book_detector',
            output='screen',
        ),

        # Navigation
        Node(
            package='erc_solution',
            executable='navigation_node',
            name='navigation_node',
            output='screen',
        ),

        # Manipulation
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
            }],
        ),
    ])
