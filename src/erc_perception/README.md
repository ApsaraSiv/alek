# erc_perception

Perception package for ERC 2026 Phase 1: identifies the target shelf column
number and the target book's colour/row using the TIAGo Pro head camera.

## Images folder

Annotated images save to `src/erc_images/` - a shared top-level folder
alongside all packages, not nested inside `erc_perception/`. This is as
close to the competition's required repo-root `/erc_images/` folder as the
current Docker setup allows: `docker-compose.yml` only bind-mounts `src/`
into the container, so anything written outside that path inside the
container never reaches the actual git repo on disk. If you want the
folder literally at the repo root instead, you'll need to also bind-mount
the repo root (or just `erc_images/`) in `docker/docker-compose.yml`.

## Nodes

- **`book_color_detector`** (`erc_perception/book_color_detector.py`)
  Detects red/blue/green/yellow book blobs via HSV colour thresholding.
  Highlights the book matching the `target_colour` parameter, estimates its
  shelf row (1-4), publishes to `/erc/shelf_row_identification`, and saves
  an annotated image when the target is visible.

- **`shelf_number_detector`** (`erc_perception/shelf_number_detector.py`)
  Finds the black overhead column-number signs (1-5) using contour
  detection + Tesseract OCR. Highlights the digit matching the
  `target_column` parameter, publishes to `/erc/shelf_column_identification`,
  and saves an annotated image when found.

- **`bin_detector`** (`erc_perception/bin_detector.py`)
  Detects the red collection bin, telling it apart from red books by shape
  (the bin is wide and squat; books are tall and narrow). Highlights the
  bin, publishes to `/erc/bin_identification`, and saves an annotated image
  when found. Perception only - does not handle placing the book in the bin.

- **`camera_viewer`** (`erc_perception/camera_viewer.py`)
  Dev-only utility: shows the raw camera feed in a window. Not part of the
  competition solution; useful for confirming the camera topic is alive.

## Running

```bash
ros2 launch erc_perception perception.launch.py \
    shelf_column_number:=2 book_colour:=red
```

Pass `enable_display:=true` to also pop up live OpenCV preview windows
(requires a display - keep `false`, the default, for headless/evaluation
runs, since `cv2.imshow` will otherwise fail with no display attached).

## Dependencies

Both `tesseract-ocr` (apt) and `pytesseract` (pip) are required by
`shelf_number_detector`. They were installed manually inside the running
dev container - **add both to the Dockerfile/solution package dependencies
before final submission**, since a fresh container build won't have them.

## Known limitations

- HSV colour thresholds and OCR digit-region filters (area, aspect ratio,
  contrast threshold) were tuned against one camera distance/angle in
  simulation. They may need retuning if the robot's approach pose changes.
- Row detection buckets the target book's vertical position into quarters
  of the camera frame, not the shelf's actual physical rows - it only holds
  up while the camera framing stays roughly the same as during tuning.
