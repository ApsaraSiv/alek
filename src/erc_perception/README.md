# erc_perception

Camera-based detectors for ERC 2026 Phase 1. All three subscribe to the head
RGB camera, `/head_front_camera/head_front_camera/color/image_raw`, and are
started by `erc_solution/launch/solution.launch.py`. There is no separate
launch file in this package.

## Nodes

- **`shelf_number_detector`** (`erc_perception/shelf_number_detector.py`)
  Reads the overhead column markers (1–5). Dark glyphs on the bright marker
  plates are cut out, normalised, and matched against digit templates. Only
  glyphs that sit on a plate are accepted, so book spines don't read as
  digits.
  - Parameter: `target_shelf_column_number`.
  - Publishes `/erc/shelf_column_horizontal_error` (`std_msgs/Float32`,
    normalised image offset of the target digit) for the state machine to
    steer on.
  - Publishes `/erc/shelf_column_identification` (`std_msgs/Int32`) once
    the target digit has been centred in two consecutive frames.
  - Saves annotated `shelf_<timestamp>.png` images.

- **`book_color_detector`** (`erc_perception/book_color_detector.py`)
  Finds red, blue, green and yellow books by HSV thresholding and highlights
  the one matching `target_colour`.
  - Estimates its row (1–4) from its vertical position in the frame and
    publishes it on `/erc/shelf_row_identification` (`std_msgs/Int32`).
  - Saves annotated `books_<timestamp>.png` images.
  - `enable_display` (default `false`) opens an OpenCV preview window.

- **`bin_detector`** (`erc_perception/bin_detector.py`)
  Finds the red collection bin. It tells the bin apart from red books by
  shape: the bin is wide and squat, books are tall and narrow.
  - Publishes `/erc/bin_identification` (`std_msgs/Bool`); the current
    solution does not subscribe to it.
  - Saves `bin_<timestamp>.png` images.

The 3-D grasp point for the target book comes from `book_detector` in
`erc_solution`, not from this package.

## Annotated images

Images are saved from the live camera feed during the trial, with a
timestamp drawn on each, to `/opt/erc_ws/src/erc_images` inside the container
(`src/erc_images/` on the host). Only `src/` is bind-mounted into the
container, so that is the only place inside it that reaches the repository.
`src/erc_images/` is currently listed in `.gitignore`.

## Known limitations

- HSV thresholds and glyph filters were tuned in simulation for the distances
  and angles the state machine uses. Very different viewpoints may need
  retuning.
- Row estimation divides the camera frame into quarters rather than using the
  shelf's physical rows, so it depends on the head being framed as during
  tuning.
- `setup.py` still declares a `camera_viewer` entry point, and `package.xml`
  still lists `tesseract-ocr`. Neither is used any more.
