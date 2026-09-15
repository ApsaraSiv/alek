# erc_solution

Team Aleksandria's solution package for ERC 2026 Phase 1. It contains the
launch file used for evaluation and the navigation, manipulation and
orchestration nodes.

```bash
ros2 launch erc_solution solution.launch.py shelf_column_number:=2 book_colour:=red
```

`erc_bringup simulation.launch.py` must already be running. See the
[repository README](../../README.md) for the full run instructions, the state
sequence, and the node and topic overview. See [`INTERFACES.md`](INTERFACES.md)
for the topic and service contract between packages.

## Contents

| Path | Purpose |
|---|---|
| `launch/solution.launch.py` | Starts the `erc_perception` detectors, `book_detector`, `navigation_node`, MoveIt `move_group`, `manipulation_node` and `state_machine_node`. Takes `shelf_column_number` and `book_colour`. |
| `erc_solution/state_machine_node.py` | Runs the trial: column search, then calls `approach_shelf`, `grasp_book`, `navigate_to_bin` and `place_in_bin` in order. |
| `erc_solution/navigation_node.py` | Base motion: arm tuck, LiDAR standoff approach to the shelf, drive to the bin. |
| `erc_solution/manipulation_node.py` | Right-arm grasp and place through MoveIt, gripper and torso control, and bin measurement from the camera. |
| `erc_solution/book_detector.py` | 3-D position of the target book from colour and depth (`/erc/target_book_point`). |

Dependencies are declared in `package.xml`. The package uses the service
definitions in `erc_interfaces` and the MoveIt configuration in
`tiago_pro_right_arm_moveit_config`.
