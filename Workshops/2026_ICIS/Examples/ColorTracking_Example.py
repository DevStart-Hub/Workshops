# %% 1. Import libraries

import cv2
import numpy as np
import pandas as pd


# %% 2. Configuration

# ---------- Tablet: pink rectangular border ----------

LOWER_PINK = np.array([145, 52, 64])
UPPER_PINK = np.array([165, 255, 255])

TABLET_DILATE_ITERS = 10
TABLET_MIN_AREA = 2000


# ---------- Coloured objects ----------

COLOR_BANDS = {
    "red": (
        np.array([171, 92, 58]),
        np.array([0, 255, 255]),
    ),
    "yellow": (
        np.array([15, 106, 0]),
        np.array([35, 255, 255]),
    ),
    "blue": (
        np.array([99, 171, 0]),
        np.array([141, 255, 151]),
    ),
    "green": (
        np.array([32, 77, 0]),
        np.array([79, 255, 255]),
    ),
}


DRAW_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "green": (0, 200, 0),
}


BLOB_MIN_AREA = 200
BLOB_MAX_AREA = 40_000

# Removes thin and small colour matches.
BLOB_OPEN_KERNEL = 30

# Makes the displayed rectangle slightly larger.
BLOB_BOX_EXPAND = 1.1

# Maximum number of objects detected for each colour.
MAX_BLOBS_PER_COLOR = 2


# %% 3. Helper functions


def contour_boxes(mask, min_area):
    """
    Find external contours and return them sorted from largest to smallest.

    Returns
    -------
    list
        List of (contour, bounding_rectangle) pairs.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    results = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            bounding_rectangle = cv2.boundingRect(contour)

            results.append(
                (
                    area,
                    contour,
                    bounding_rectangle,
                )
            )

    results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        (contour, bounding_rectangle) for area, contour, bounding_rectangle in results
    ]


def color_mask(hsv, lower, upper):
    """
    Create an HSV mask.

    Supports ranges that cross the hue boundary from 179 back to 0,
    which is necessary for red.
    """
    if lower[0] <= upper[0]:
        return cv2.inRange(
            hsv,
            lower,
            upper,
        )

    # First section: lower hue to 179.
    lower_1 = lower.copy()
    upper_1 = upper.copy()

    lower_1[0] = lower[0]
    upper_1[0] = 179

    # Second section: 0 to upper hue.
    lower_2 = lower.copy()
    upper_2 = upper.copy()

    lower_2[0] = 0
    upper_2[0] = upper[0]

    mask_1 = cv2.inRange(
        hsv,
        lower_1,
        upper_1,
    )

    mask_2 = cv2.inRange(
        hsv,
        lower_2,
        upper_2,
    )

    return mask_1 | mask_2


def expand_rect(rectangle, factor):
    """
    Expand a rotated rectangle around its centre.
    """
    (center_x, center_y), (width, height), angle = rectangle

    return (
        (center_x, center_y),
        (
            width * factor,
            height * factor,
        ),
        angle,
    )


# %% 4. Detection functions


def detect_tablet(hsv, frame):
    """
    Detect the largest pink region and fit a rotated rectangle around it.

    Returns
    -------
    numpy.ndarray or None
        Four tablet corner coordinates.
    """
    tablet_mask = color_mask(
        hsv,
        LOWER_PINK,
        UPPER_PINK,
    )

    # Merge the separate parts of the pink border.
    tablet_mask = cv2.dilate(
        tablet_mask,
        None,
        iterations=TABLET_DILATE_ITERS,
    )

    boxes = contour_boxes(
        tablet_mask,
        TABLET_MIN_AREA,
    )

    if not boxes:
        return None

    # The contours are sorted, so index 0 is the largest.
    tablet_contour, _ = boxes[0]

    tablet_rectangle = cv2.minAreaRect(tablet_contour)

    tablet_box = cv2.boxPoints(tablet_rectangle).astype(np.intp)

    cv2.drawContours(
        frame,
        [tablet_box],
        contourIdx=-1,
        color=(255, 192, 203),
        thickness=2,
    )

    label_x = int(np.min(tablet_box[:, 0]))

    label_y = max(
        25,
        int(np.min(tablet_box[:, 1])) - 10,
    )

    cv2.putText(
        frame,
        "tablet",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 192, 203),
        2,
    )

    return tablet_box


def detect_coloured_objects(
    hsv,
    tablet_box,
    frame,
):
    """
    Detect up to MAX_BLOBS_PER_COLOR objects of each colour.

    Objects are named:

        red1, red2
        yellow1, yellow2
        blue1, blue2
        green1, green2

    The tablet area is excluded.
    """
    detected_objects = {}

    for colour_name, (lower, upper) in COLOR_BANDS.items():
        mask = color_mask(
            hsv,
            lower,
            upper,
        )

        # Exclude everything inside the tablet rectangle.
        if tablet_box is not None:
            cv2.fillPoly(
                mask,
                [tablet_box],
                0,
            )

        # Remove narrow and noisy colour matches.
        if BLOB_OPEN_KERNEL > 1:
            open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    BLOB_OPEN_KERNEL,
                    BLOB_OPEN_KERNEL,
                ),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                open_kernel,
            )

        valid_contours = []

        for contour, _ in contour_boxes(
            mask,
            BLOB_MIN_AREA,
        ):
            area = cv2.contourArea(contour)

            if area > BLOB_MAX_AREA:
                continue

            valid_contours.append(contour)

        # Keep only the largest requested number.
        valid_contours = valid_contours[:MAX_BLOBS_PER_COLOR]

        for blob_number, contour in enumerate(
            valid_contours,
            start=1,
        ):
            object_name = f"{colour_name}{blob_number}"

            rectangle = cv2.minAreaRect(contour)

            rectangle = expand_rect(
                rectangle,
                BLOB_BOX_EXPAND,
            )

            box_points = cv2.boxPoints(rectangle).astype(np.intp)

            draw_colour = DRAW_BGR.get(
                colour_name,
                (0, 255, 0),
            )

            cv2.drawContours(
                frame,
                [box_points],
                contourIdx=-1,
                color=draw_colour,
                thickness=2,
            )

            label_x = int(np.min(box_points[:, 0]))

            label_y = max(
                25,
                int(np.min(box_points[:, 1])) - 10,
            )

            cv2.putText(
                frame,
                object_name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_colour,
                2,
            )

            detected_objects[object_name] = box_points

    return detected_objects


# %% 5. Open the video

VIDEO_PATH = "TestNina1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"The video could not be opened: {VIDEO_PATH}")


WINDOW_NAME = "Puzzle tracker"

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
)


# %% 6. Process the video frame by frame

tracking_data = []

frame_number = 0
quit_requested = False


while True:
    success, frame = cap.read()

    if not success:
        break

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    # Detection order:
    # tablet first, then coloured objects.
    tablet_box = detect_tablet(
        hsv,
        frame,
    )

    detected_objects = detect_coloured_objects(
        hsv,
        tablet_box,
        frame,
    )

    # Build one output row for this frame.
    row = {
        "frame": frame_number,
        "timestamp_ms": cap.get(cv2.CAP_PROP_POS_MSEC),
    }

    # Save tablet corners.
    if tablet_box is not None:
        for corner_number, (x, y) in enumerate(
            tablet_box,
            start=1,
        ):
            row[f"tablet_corner_{corner_number}_x"] = x

            row[f"tablet_corner_{corner_number}_y"] = y

    # Save coloured-object corners.
    #
    # Example column names:
    # red1_corner_1_x
    # red1_corner_1_y
    # yellow2_corner_4_x
    for object_name, corners in detected_objects.items():
        for corner_number, (x, y) in enumerate(
            corners,
            start=1,
        ):
            row[f"{object_name}_corner_{corner_number}_x"] = x

            row[f"{object_name}_corner_{corner_number}_y"] = y

    tracking_data.append(row)

    # Display frame number.
    cv2.putText(
        frame,
        f"frame {frame_number}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Resize the display window based on the first frame.
    if frame_number == 0:
        frame_height, frame_width = frame.shape[:2]

        scale = min(
            DISPLAY_MAX_W / frame_width,
            DISPLAY_MAX_H / frame_height,
            1.0,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            int(frame_width * scale),
            int(frame_height * scale),
        )

    cv2.imshow(
        WINDOW_NAME,
        frame,
    )

    frame_number += 1

    # Controls:
    # Space = pause/resume
    # Q or Escape = quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        while True:
            pause_key = cv2.waitKey(30) & 0xFF

            if pause_key == ord(" "):
                break

            if pause_key in (
                ord("q"),
                27,
            ):
                quit_requested = True
                break

    if key in (
        ord("q"),
        27,
    ):
        quit_requested = True

    if quit_requested:
        break


# %% 7. Build and save the tracking data

tracking_df = pd.DataFrame(tracking_data)

print(tracking_df.head())

print(f"DataFrame shape: {tracking_df.shape}")

tracking_df.to_csv(
    "puzzle_tracking_data.csv",
    index=False,
)


# %% 8. Release resources

cap.release()
cv2.destroyAllWindows()
