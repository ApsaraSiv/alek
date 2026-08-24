import os
import random
import tempfile
import re as _re
import yaml as _yaml

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# RGBA values for each book colour
BOOK_COLOURS = {
    'red':    '1 0 0 1',
    'green':  '0 1 0 1',
    'yellow': '1 1 0 1',
    'blue':   '0 0 1 1',
}

# Shelf world pose (matches the shelf's own spawn call in erc_world.sdf)
SHELF_X = 3.0
SHELF_Y = 0.0
SHELF_Z = 1.1

# Shelf geometry (origin is at the shelf's physical center)
NUM_COLUMNS = 5
NUM_ROWS = 6
ACTIVE_ROWS = [1, 2, 3, 4]          # 0-indexed; shelf has rows 0-5, so this skips top (0) and bottom (5)
COLUMN_WIDTH = 1.0                  # meters
SHELF_HEIGHT = 2.10                 # meters, total across all rows
ROW_HEIGHT = SHELF_HEIGHT / NUM_ROWS

# Book position and orientation on shelf
COLUMN_JITTER_RANGE = COLUMN_WIDTH * 0.25
ROTATE_90_DEGREES_RAD = 1.5708

# Column/row offsets from center of shelf
COLUMN_Y_OFFSETS = [((NUM_COLUMNS - 1) / 2 - col) * COLUMN_WIDTH for col in range(NUM_COLUMNS)]
TOP_BOOK_Z = 0.825                  # Position of top row's book, and spacing between rows
ROW_SPACING = 0.33
ROW_FLOOR_Z_OFFSETS = [TOP_BOOK_Z - (i * ROW_SPACING) for i in range(NUM_ROWS)]

# Number marker positioning
NUMBER_MARKER_PLATE_X = SHELF_X - 0.245 # To be flush with shelf edge
NUMBER_MARKER_PLATE_Z = 2.26


def generate_launch_description():

    # ── Reproducible layout ──
    # Layout is randomized via ERC_SEED
    _seed = os.environ.get('ERC_SEED')
    if _seed:
        random.seed(int(_seed))

    headless = LaunchConfiguration('headless')

    bringup_dir = get_package_share_directory('erc_bringup')
    arena_dir = get_package_share_directory('erc_description')

    controller_params = os.path.join(bringup_dir, 'config', 'controller_params.yaml')
    world_file = os.path.join(arena_dir, 'worlds', 'erc_world.sdf')

    # ── Load pre-generated URDF ──
    # Generated once by scripts/generate_urdf.py (PAL xacro + Harmonic patches)
    urdf_path = os.path.join(arena_dir, 'urdf', 'tiago_pro.urdf')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(
            f'URDF not found at {urdf_path}. '
            f'Run: python3 src/erc_bringup/scripts/generate_urdf.py')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # ── Gazebo Harmonic ──
    # GUI when headless:=false (default), server-only when headless:=true.
    # Headless mode: unset DISPLAY to force ogre2 onto the surfaceless EGL path
    # (otherwise sensors silently produce no frames) and pin the NVIDIA ICD
    # when present.
    gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        condition=UnlessCondition(headless),
        output='screen',
    )
    _nv_icd = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    gazebo_headless = ExecuteProcess(
        cmd=['bash', '-c',
             'unset DISPLAY; '
             f'[ -f {_nv_icd} ] && export __EGL_VENDOR_LIBRARY_FILENAMES={_nv_icd}; '
             f'exec gz sim -r -s "{world_file}"'],
        condition=IfCondition(headless),
        output='screen',
    )

    # ── Robot state publisher ──
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

   # ── Bridge: gz-transport ↔ ROS 2 ──
    # CLI args handle topics where the gz and ROS names are identical.
    # The YAML config (below) handles topics that need renaming.
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # Mobile base (gz MecanumDrive plugin)
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/odom_tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',  # remapped → /tf below
        # LiDAR
        '/scan_front_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/scan_rear_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        # IMU
        '/base_imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        # Collection bin contact
        '/world/erc_world/model/erc_collection_bin/link/collection_bin_base_link/sensor/bin_contact_sensor/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
    ]

    #  Contact sensors
    #  Map all links to a single /contacts topic
    _contact_topic_map = {}
    for _link in _re.findall(r'name="([^"]+)_contact" type="contact"', robot_description):
        _contact_topic_map[_link] = '/contacts'

    contact_bridge_config = []
    for _link, _ros_topic in _contact_topic_map.items():
        sensor_name = f'{_link}_contact'
        if f'name="{sensor_name}"' not in robot_description:
            continue
        gz_topic = f'/world/erc_world/model/tiago_pro/link/{_link}/sensor/{sensor_name}/contact'
        contact_bridge_config.append({
            'ros_topic_name': _ros_topic,
            'gz_topic_name': gz_topic,
            'ros_type_name': 'ros_gz_interfaces/msg/Contacts',
            'gz_type_name': 'gz.msgs.Contacts',
            'direction': 'GZ_TO_ROS',
        })

    contact_bridge_yaml = '/tmp/erc_contact_bridge.yaml'
    with open(contact_bridge_yaml, 'w') as f:
        _yaml.dump(contact_bridge_config, f)

    print(f'[simulation.launch] {len(contact_bridge_config)} contact bridges configured')

    contact_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='contact_bridge',
        parameters=[{'use_sim_time': True, 'config_file': contact_bridge_yaml}],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'use_sim_time': True}],
        arguments=bridge_args,
        remappings=[
            # Route the bridged odom transform onto the standard /tf topic so it
            # merges with robot_state_publisher's TF tree (odom -> base_footprint).
            ('/odom_tf', '/tf'),
            ('/world/erc_world/model/erc_collection_bin/link/collection_bin_base_link/sensor/bin_contact_sensor/contact', '/bin_contacts'),
        ],
        output='screen',
    )

    #  Camera bridge - use ROS2 topic names that match the ones on the physical TIAGo Pro
    _head_cam_cfg = os.path.join(
    get_package_share_directory('erc_bringup'), 'config', 'head_camera_bridge.yaml')

    head_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='head_camera_bridge',
        parameters=[{'use_sim_time': True, 'config_file': _head_cam_cfg}],
        output='screen',
    )

    # ── Spawn robot at start zone ──
    spawn_tiago_pro = TimerAction(period=3.0, actions=[
        Node(package='ros_gz_sim', executable='create',
             parameters=[{'use_sim_time': True}],
             arguments=[
                 '-name', 'tiago_pro',
                 '-topic', 'robot_description',
                 '-x', '0', '-y', '0', '-z', '0.15',
                 '-R', '0', '-P', '0', '-Y', str(ROTATE_90_DEGREES_RAD),
             ],
             output='screen'),
    ])

    # ── Controllers ──
    def spawner(name, params=None):
        args = [name, '-c', '/controller_manager']
        if params:
            args += ['--param-file', params]
        return Node(package='controller_manager', executable='spawner',
                    arguments=args,
                    parameters=[{'use_sim_time': True}],
                    output='screen')

    controllers = TimerAction(period=8.0, actions=[
        spawner('joint_state_broadcaster'),
        # Base is driven by the gz MecanumDrive plugin, not ros2_control
        spawner('arm_left_controller', controller_params),
        spawner('arm_right_controller', controller_params),
        spawner('head_controller', controller_params),
        spawner('torso_controller', controller_params),
        spawner('gripper_left_controller_raw', controller_params),
        spawner('gripper_right_controller_raw', controller_params),
    ])

    # ── Gripper command clamp ──
    gripper_clamp = TimerAction(period=9.0, actions=[
        Node(package='erc_bringup', executable='gripper_command_clamp.py',
             parameters=[{'use_sim_time': True}],
             output='screen'),
    ])

    # ── Book colour substitution ──
    book_sdf_path = os.path.join(arena_dir, 'models', 'book', 'sdf', 'erc_book.sdf')
    with open(book_sdf_path, 'r') as f:
        book_sdf_template = f.read()

    # One substituted temp file per colour (identical books share SDF)
    tmp_book_paths = {}
    for colour_name, rgba in BOOK_COLOURS.items():
        coloured_sdf = book_sdf_template.replace('BOOK_COLOUR_PLACEHOLDER', rgba)
        tmp_path = f'/tmp/erc_book_{colour_name}.sdf'
        with open(tmp_path, 'w') as f:
            f.write(coloured_sdf)
        tmp_book_paths[colour_name] = tmp_path

    # ── Spawn books ──
    book_spawn_nodes = []
    for col in range(NUM_COLUMNS):
        colours_this_column = list(BOOK_COLOURS.keys())
        random.shuffle(colours_this_column)

        for i, row in enumerate(ACTIVE_ROWS):
            colour_name = colours_this_column[i]
            tmp_path = tmp_book_paths[colour_name]
            book_name = f'book_col_{col + 1}_row_{row + 1}_{colour_name}'

            y = SHELF_Y + COLUMN_Y_OFFSETS[col] + random.uniform(-COLUMN_JITTER_RANGE, COLUMN_JITTER_RANGE)
            z = SHELF_Z + ROW_FLOOR_Z_OFFSETS[row]
            x = SHELF_X - 0.1

            book_spawn_nodes.append(
                Node(package='ros_gz_sim', executable='create',
                    arguments=[
                        '-file', tmp_path,
                        '-name', book_name,
                        '-x', str(x), '-y', str(y), '-z', str(z),
                        '-R', '0', '-P', str(ROTATE_90_DEGREES_RAD), '-Y', '0',
                    ],
                    output='screen')
            )

    spawn_books = TimerAction(period=5.0, actions=book_spawn_nodes)

    # ── Spawn number markers ──
    number_marker_path = os.path.join(
        get_package_share_directory('erc_description'),
        'models', 'number_marker', 'sdf', 'erc_number_marker.sdf'
    )

    with open(number_marker_path, 'r') as f:
        number_str = f.read()

    tmp_number_marker_paths = {}
    for number in range(1, 6):
        sdf = number_str.replace('NUMBER_PLACEHOLDER', str(number))
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False)
        tmp_file.write(sdf)
        tmp_file.close()
        tmp_number_marker_paths[number] = tmp_file.name

    number_marker_spawn_nodes = []
    shelf_number_markers = list(range(1, NUM_COLUMNS + 1))
    random.shuffle(shelf_number_markers)

    for col in range(NUM_COLUMNS):
        number = shelf_number_markers[col]
        y = SHELF_Y + COLUMN_Y_OFFSETS[col]
        z = NUMBER_MARKER_PLATE_Z

        number_marker_spawn_nodes.append(
            Node(package='ros_gz_sim', executable='create',
                arguments=[
                    '-file', tmp_number_marker_paths[number], '-name', f'number_marker_col_{col+1}',
                    '-x', str(NUMBER_MARKER_PLATE_X), '-y', str(y), '-z', str(z),
                    '-R', '0', '-P', '0', '-Y', str(ROTATE_90_DEGREES_RAD),
                ],
                output='screen')
        )

    spawn_number_markers = TimerAction(period=5.0, actions= number_marker_spawn_nodes)

    # ── Depth-camera point cloud (sensors/depth_to_cloud) ──
    # The gz rgbd_camera never fills /depth/points. The `sensors` package
    # rebuilds the cloud with a RealSense-D435 noise model. Launched here
    # with the sim so there's exactly one publisher on the topic.
    # Delayed ~18 s to avoid a CycloneDDS discovery-storm race condition.
    depth_cloud_actions = []
    try:
        sensors_launch = os.path.join(
            get_package_share_directory('sensors'), 'launch', 'depth_to_cloud.launch.py')
        depth_cloud_actions.append(TimerAction(period=18.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensors_launch),
                condition=IfCondition(LaunchConfiguration('depth_cloud')),
            )]))
    except Exception as e:
        print(f'[simulation.launch] depth cloud NOT started — `sensors` package '
              f'unavailable ({e}). Build it: '
              f'colcon build --packages-select sensors --symlink-install')

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('depth_cloud', default_value='true',
                              description='Rebuild the depth point cloud via '
                                          'sensors/depth_to_cloud (owns '
                                          '/head_front_camera/depth/points)'),
        gazebo_gui, gazebo_headless,
        rsp, bridge, head_camera_bridge, contact_bridge, spawn_tiago_pro, controllers, gripper_clamp,
        spawn_books, spawn_number_markers, *depth_cloud_actions,
    ])
