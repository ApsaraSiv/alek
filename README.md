# Team Aleksandria — Emirates Robotics Competition 2026, Phase 1

Autonomous library-assistant solution for the ERC 2026 Simulation Phase: a
TIAGo Pro finds the requested shelf column by reading its overhead number,
identifies the requested book colour, picks the book with **one arm (right)**,
brings it back and places it in the red collection bin.

This repository is based on the official competition environment,
[dfl-rlab/erc_sim_2026](https://github.com/dfl-rlab/erc_sim_2026), and keeps
its Docker image, simulation packages and robot description unchanged. The
team's own work is in the packages listed under
[Packages created by the team](#packages-created-by-the-team).

<img src="docs/assets/erc_3d_env.png" width="300"/> <img src="docs/assets/tiago_pro.png" width="200"/>

---

## Running a challenge trial

The evaluation command from the Phase 1 rules is supported as-is. The base
simulation must be running before the solution is launched.

```bash
# Terminal 1 — inside the container
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```

Wait until the robot, shelves and books have spawned (controllers come up
about 8 s after Gazebo starts), then:

```bash
# Terminal 2 — inside the container (./docker/attach.sh from the host)
source install/setup.bash
ros2 launch erc_solution solution.launch.py shelf_column_number:=2 book_colour:=red
```

| Argument | Values | Meaning |
|---|---|---|
| `shelf_column_number` | `1`–`5` | Number on the overhead marker of the target column. Markers are shuffled on every simulation load, so the column is found by vision, not by position. |
| `book_colour` | `red`, `blue`, `green`, `yellow` | Colour of the target book in that column. |
| `debug_skip_to_bin_after_column` | `false` (default) | Development only: after reaching the column, skip grasping and drive straight to the bin. |

`solution.launch.py` starts every node the solution needs, including MoveIt
(`move_group`) for the right arm.

## What the solution does

`state_machine_node` runs the trial as a fixed sequence of states:

| State | What happens | Main nodes |
|---|---|---|
| `SEEK_COLUMN` | Turns in short steps with the head tilted up until the target number is read on an overhead marker, then turns to centre it in the camera. Backs away if it drifts too close to a wall. | `shelf_number_detector`, `state_machine_node` |
| approach (`/erc/approach_shelf`) | Tucks both arms into PAL's home pose, then drives forward until the front LiDAR reports the shelf standoff distance. | `navigation_node` |
| `SEEK_BOOK` | Detects the target colour in the camera and reports its row (1–4). | `book_color_detector`, `book_detector` |
| `GRASP` (`/erc/grasp_book`) | Back-projects the book through the depth camera, centres the base on it if it is out of reach, raises the torso for high rows, and moves the right gripper straight in with a forward-pointing orientation. Closes slowly, lifts, and pulls the book out. | `book_detector`, `manipulation_node`, `move_group` |
| `NAV_TO_BIN` (`/erc/navigate_to_bin`) | Drives back towards the start zone to a standoff in front of the bin. | `navigation_node` |
| `PLACE` (`/erc/place_in_bin`) | Measures the bin rim and table from the camera, lifts the book above the rim before closing in, releases over the bin, backs off and returns the arm to its home pose. | `manipulation_node`, `move_group` |

If any step fails, the state machine logs `state=FAILED` with the reason and
does not start the remaining steps.

## Competition outputs

| Requirement (Phase 1 rules) | Where it comes from |
|---|---|
| `/erc/shelf_column_identification` — `std_msgs/msg/Int32`, column number | `shelf_number_detector`, once the target number has been centred in two consecutive frames |
| `/erc/shelf_row_identification` — `std_msgs/msg/Int32`, row 1–4 | `book_color_detector` (and `book_detector`) when the target book is visible |
| Annotated, timestamped images of the target column and target book | `shelf_number_detector` (`shelf_*.png`) and `book_color_detector` (`books_*.png`), saved from the live camera during the trial |
| Book placed in the bin | Contact reported on `/bin_contacts` |

## Packages created by the team

| Package | Type | Contents |
|---|---|---|
| [`erc_solution`](src/erc_solution) | `ament_python` | The solution package: `solution.launch.py`, `state_machine_node`, `navigation_node`, `manipulation_node`, `book_detector`. See also [`INTERFACES.md`](src/erc_solution/INTERFACES.md). |
| [`erc_perception`](src/erc_perception) | `ament_python` | Camera-based detectors: `shelf_number_detector`, `book_color_detector`, `bin_detector`. |
| [`erc_interfaces`](src/erc_interfaces) | `ament_cmake` | Service definitions used between the state machine, navigation and manipulation: `ApproachShelf`, `NavigateToBin`, `GraspBook`, `PlaceInBin`. |
| [`tiago_pro_right_arm_moveit_config`](src/tiago_pro_right_arm_moveit_config) | MoveIt config | MoveIt Setup Assistant configuration for the `arm_right` planning group, started by `solution.launch.py`. |
| [`tools/drift_viz`](tools/drift_viz) | Scripts (not a ROS package) | Development tool: overlays planned waypoints, `/odom` and Gazebo ground truth in RViz (`drift_viz.py` + `drift_viz.rviz`), or renders the same view to PNG (`render_drift.py`). |

`src/aleksandria` is an early standalone arm prototype kept for reference; it
is not launched by the solution.

### Nodes

| Node | Package | Subscribes | Publishes / provides |
|---|---|---|---|
| `shelf_number_detector` | `erc_perception` | colour image | `/erc/shelf_column_identification`, `/erc/shelf_column_horizontal_error`, annotated images |
| `book_color_detector` | `erc_perception` | colour image | `/erc/shelf_row_identification`, annotated images |
| `bin_detector` | `erc_perception` | colour image | `/erc/bin_identification` (`std_msgs/Bool`; not currently used by the solution) |
| `book_detector` | `erc_solution` | colour and depth images, camera info, `/scan_front_raw` | `/erc/target_book_point` (`geometry_msgs/PointStamped`), `/erc/shelf_row_identification` |
| `state_machine_node` | `erc_solution` | column identification and horizontal error, row identification, `/scan_front_raw` | `/cmd_vel` and head commands during the column search; calls the four `/erc/*` services |
| `navigation_node` | `erc_solution` | `/odom`, `/scan_front_raw`, `/scan_rear_raw`, `/joint_states` | `/cmd_vel`; arm, torso and head trajectories (arm tuck); services `/erc/approach_shelf`, `/erc/navigate_to_bin` |
| `manipulation_node` | `erc_solution` | `/erc/target_book_point`, camera images, `/joint_states`, `/odom`, `/scan_front_raw` | `move_group` actions/services (`/move_action`, `/compute_cartesian_path`, `/execute_trajectory`); gripper, torso and head trajectories; `/cmd_vel` for base centring; services `/erc/grasp_book`, `/erc/place_in_bin` |

Dependencies of each package are declared in its `package.xml`.

---

## Environment setup

These steps are unchanged from the official
[erc_sim_2026 README](https://github.com/dfl-rlab/erc_sim_2026#quick-start).

### Prerequisites

- x86_64 (amd64) architecture — ARM hosts (e.g. Apple Silicon) are not supported
- Linux host (Ubuntu 22.04/24.04 recommended) with X11
- Docker Engine + Docker Compose v2
- Git
- NVIDIA GPU + nvidia-container-toolkit for GPU-accelerated rendering (works without one, but Gazebo runs slowly)
- ~15 GB free disk space

The default `ROS_DOMAIN_ID` is `23`. If several ROS 2 environments share the
network, make sure the domain IDs don't clash.

### Quick start

All dependencies are vendored in this repository, so no network access is
needed after cloning.

```bash
# — Host terminal, repository root —
./docker/up.sh --build          # build the image and start the container
./docker/attach.sh              # open a shell inside the container

# — Inside the container —
colcon build --symlink-install  # first build takes several minutes
source install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```

For more terminals, run `./docker/attach.sh` again from the repository root.
**Do not run `./docker/up.sh` again** — it restarts the running container.

### Software stack

| Component | Version |
|---|---|
| ROS 2 | Humble |
| Gazebo | Harmonic |
| DDS | CycloneDDS |
| Controllers | ros2_control + gz_ros2_control |
| Motion planning | MoveIt 2 |

The Docker image, OS, ROS 2 distribution, Gazebo version and robot model are
used exactly as provided, as the Phase 1 rules require.

## Robot interfaces used

| Interface | Topic / controller |
|---|---|
| Base velocity | `/cmd_vel` (`geometry_msgs/Twist`, mecanum drive) |
| Odometry | `/odom` |
| LiDARs | `/scan_front_raw`, `/scan_rear_raw` |
| Head camera | `/head_front_camera/head_front_camera/color/image_raw`, `.../depth/image_rect_raw`, `.../color/camera_info`, `.../depth/camera_info` |
| Right arm | `arm_right_controller` (driven through MoveIt) |
| Right gripper | `/gripper_right_controller_raw/joint_trajectory` (`gripper_right_finger_joint`: 0.0 closed, ~0.065 fully open) |
| Left arm | `arm_left_controller` — only used to tuck it out of the way |
| Torso / head | `torso_controller`, `head_controller` |
| Bin contact | `/bin_contacts` |

The full topic, controller and sensor reference is in the
[official README](https://github.com/dfl-rlab/erc_sim_2026#ros-2-topics-and-controllers).
Competition update of 11/09/26: book spine width is **2 cm**; the grasp
parameters in `manipulation_node.py` are set for that width.

## Known limitations

- **Real-time factor.** Without a GPU, Gazebo runs at roughly 0.1–0.4× real
  time. Motion timeouts are generous for this reason, and a full trial takes
  several minutes of wall time.
- **Edge columns.** The column search is less reliable for columns 1 and 5 at
  low real-time factor.
- **Base alignment.** The approach drives straight at the marker, so the base
  can stop at an angle to the shelf; grasp success depends on the book ending
  up within the right arm's reach.
- **Sweeping the shelf.** `navigate_to_bin` can brush books when it turns in
  place close to the shelf.
- **Image folder.** Annotated images are written to `src/erc_images/` (only
  `src/` is mounted into the container), which is listed in `.gitignore`. The
  rules ask for an `/erc_images/` folder in the team repository.
- **Stale entries.** `erc_perception/setup.py` and `erc_solution/setup.py`
  still list console scripts for modules that were removed (`camera_viewer`,
  `shelf_column_detector`), and `erc_perception/package.xml` still depends on
  `tesseract-ocr`, which the template-matching digit reader no longer uses.

## Troubleshooting

These are from the official README.

- **CycloneDDS serialization warnings** (`serdata.cpp`, null-terminated
  strings) are harmless.
- **Robot not spawning:** wait at least 5 s after Gazebo starts. The robot
  spawns after 3 s, books after 5 s and controllers after 8 s.
- **Controllers not activating:** rebuild `erc_bringup` with
  `colcon build --symlink-install --packages-select erc_bringup` and source again.
- **Build errors after mixing build flags:** always use `--symlink-install`;
  recover with `rm -rf build/ install/ log/` and rebuild.
- **After Dockerfile changes:** run `./docker/up.sh --build`.

Issues with the competition environment itself should be reported on the
[erc_sim_2026 issue tracker](https://github.com/dfl-rlab/erc_sim_2026/issues).
