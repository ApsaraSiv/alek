# Handoff summary — ERC 2026 sim, integration-test branch

## Repo
- Path: `/home/rava/Downloads/ERC/alek`
- Origin: https://github.com/ApsaraSiv/alek
- Current branch: `integration-test`, HEAD commit `3916ae7`
  ("Merge aleksandria-arm-manipulation: adopt upstream's manipulation rewrite")
- `integration-test` is local-only, not pushed to origin.

## What this branch is
Branched from `aleksandria-arm-manipulation`, then merged the latest
`aleksandria-arm-manipulation` back in after the upstream team independently
rewrote manipulation/navigation. Net effect: this branch now tracks
`aleksandria-arm-manipulation`'s manipulation + navigation rewrite, plus a
few fixes on top that upstream hadn't made yet.

## What's been fixed/kept on top of upstream
1. **Gripper topic bug** — `erc_solution/erc_solution/manipulation_node.py`
   was publishing gripper commands to `/gripper_right_controller/joint_trajectory`,
   but the actual spawned controller (see
   `erc_bringup/config/controller_params.yaml`) is
   `/gripper_right_controller_raw/joint_trajectory`. Fixed — the gripper
   should now actually respond to `set_gripper()` calls in grasp/place.
2. **`move_group.launch.py` now included** in
   `erc_solution/launch/solution.launch.py` — `manipulation_node` depends on
   `/move_action` (MoveGroup action server) but nothing else in the bringup
   started it. Included from `tiago_pro_right_arm_moveit_config`.
3. **`skip_manipulation_if_unavailable` defaults to `false`** in
   `state_machine_node.py` — a real grasp/place failure now fails the run
   (`state=FAILED`) instead of being silently skipped, now that manipulation
   is actually wired in.
4. `erc_solution/package.xml` — added `exec_depend` on
   `tiago_pro_right_arm_moveit_config` for the new launch-time reference.

## Known remaining gaps (not yet fixed, called out in code comments)
- `manipulation_node.py`'s `place_x/y/z` (bin drop pose) is a **hardcoded
  guess**, not measured against the real bin model. Needs measuring in
  Gazebo once you can visually check where `navigate_to_bin` parks the robot.
- `approach_offset` (pre-grasp hover offset) is also an untested guess per
  the file's own "STILL TO TUNE" notes.
- `book_detector.py` still publishes a **placeholder** `/erc/target_book_point`,
  not real detection — real perception for the grasp point hasn't landed yet.
- The `aleksandria` package (`src/aleksandria/`) still exists in the repo but
  is now **unreferenced/dead code** — superseded by
  `erc_solution/manipulation_node.py`. Safe to ignore or delete later; not
  currently wired into any launch file.
- End-to-end run (Gazebo + `solution.launch.py`) has **not yet been verified
  live** with these latest merge changes — last thing done was a
  `py_compile` syntax check only.

## Environment
- Runs in Docker: `erc-2026:humble-harmonic` image, container name `erc_sim`.
- No NVIDIA GPU on this host — Gazebo falls back to software rendering
  (slower, but works).
- ROS 2 Humble, Gazebo Harmonic, DART physics.

## How to verify this branch works (next step)
```bash
cd /home/rava/Downloads/ERC/alek
./docker/up.sh              # container already exists, don't rebuild unless Dockerfile changed
./docker/attach.sh
# inside container:
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```
In a second attached terminal:
```bash
source install/setup.bash
ros2 launch erc_solution solution.launch.py shelf_column_number:=1 book_colour:=red
```
Watch for `state=SEEK_COLUMN → SEEK_BOOK → GRASP → NAV_TO_BIN → PLACE → DONE`
in the second terminal's logs. A `state=FAILED` with a reason tells you
where it's actually breaking (this is now meaningful since
`skip_manipulation_if_unavailable=false`).

## Git identity used in this repo (local config, not global)
- user.name: ravacodes06
- user.email: ravalitha.ravi@gmail.com
