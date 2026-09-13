# erc_solution — Interface Contract

This is the agreed contract between subsystems so Perception, Navigation, and
Manipulation can be built and tested independently in parallel, then plugged
into `state_machine_node` at integration time. Edit this file (and open a PR
comment) if you need to change a topic name or message type — don't just
change it silently in code, the whole team depends on this staying accurate.

## Owners
- Perception  (`shelf_column_detector.py`, `book_detector.py`, `bin_detector.py`) — Person A
- Navigation  (`navigation_node.py`) — Person B
- Manipulation (`manipulation_node.py`) — Person C
- Integration (`state_machine_node.py`, this file, launch, report/video) — Person D

## Perception → everyone (also required by the rubric — do not rename)

| Topic | Type | Published by | Notes |
|---|---|---|---|
| `/erc/shelf_column_identification` | `std_msgs/Int32` | shelf_column_detector | column number, once confidently identified |
| `/erc/shelf_row_identification` | `std_msgs/Int32` | book_detector | row (1-4) containing target book |

Annotated, timestamped images saved live during the trial to `/erc_images/`
(repo-root relative) by shelf_column_detector and book_detector — required
for the +2 bonus points each. Do not pre-generate these.

## Perception → Navigation / Manipulation (team-internal, can be renamed if agreed)

| Topic | Type | Published by | Consumed by | Notes |
|---|---|---|---|---|
| `/erc/target_shelf_point` | `geometry_msgs/PointStamped` | shelf_column_detector | navigation_node | 3D point of target column, frame_id=`base_link` (perception does the camera→base_link TF lookup before publishing) |
| `/erc/target_book_point` | `geometry_msgs/PointStamped` | book_detector | manipulation_node | 3D point of target book, frame_id=`base_link` |
| `/erc/collection_bin_point` | `geometry_msgs/PointStamped` | bin_detector | manipulation_node | 3D point of bin opening, frame_id=`base_link` |
| `/erc/perception_status` | `std_msgs/String` | all perception nodes | state_machine_node | one of: `searching`, `found`, `lost` |

## Navigation service/action (Person B implements, Person D calls from the state machine)

| Name | Type | Notes |
|---|---|---|
| `/erc/navigate_to_shelf_column` | custom action or simple `std_srvs/Trigger`-style service taking column index | Blocks until robot is stopped within arm reach of the shelf, facing it |
| `/erc/navigate_to_bin` | same pattern | Blocks until robot is back at Start/End Zone, facing the bin |

While B is still building this, it's fine to stub with hardcoded waypoints —
the state machine only cares that the call blocks and returns success/failure.

## Manipulation service/action (Person C implements, Person D calls)

| Name | Type | Notes |
|---|---|---|
| `/erc/grasp_book` | action/service, input: target row (1-4) + optional 3D offset | Returns success/failure; on success the book is held |
| `/erc/place_in_bin` | action/service, no input | Lowers + releases into bin; aim for "gently placed" not "dropped" |

## State machine (Person D)

`state_machine_node` sequence:
`SEEK_COLUMN → NAV_TO_COLUMN → SEEK_BOOK → GRASP → NAV_TO_BIN → PLACE → DONE`

Each state: subscribe/call the relevant interface above, add a timeout +
bounded retry, transition to `FAILED`/abort-safe on repeated failure (this is
explicitly part of the "Code" rubric — error handling / recovery).

## Collisions

Contact topics already provided by the sim (`/contacts`, `/bin_contacts`) —
don't need to build our own; just be aware navigation speed/approach
tolerances directly affect the -0.5-per-collision penalty.
