# erc_solution

Team solution package for ERC 2026 Phase 1. See `INTERFACES.md` for the
topic/service contract between packages.

## Terminal 1 — base sim

```
./docker/attach.sh
source /opt/erc_ws/install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```

Wait for the robot, shelves, and books to finish spawning before moving on.

## Terminal 2 — the solution

```
./docker/attach.sh
source /opt/erc_ws/install/setup.bash
ros2 launch erc_solution solution.launch.py shelf_column_number:=2 book_colour:=red
```

Swap `shelf_column_number` (1-5) and `book_colour` (red/blue/green/yellow)
for whatever the trial asks for.

## Packages in this repo

- `erc_solution` — navigation, manipulation, state machine, launch file
- `erc_perception` — shelf number + book colour detection
- `erc_interfaces` — custom service types used between navigation and
  manipulation
