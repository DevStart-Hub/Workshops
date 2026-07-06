# %% 1. Import libraries

# MediaPipe provides the pre-trained Hand Landmarker; OpenCV handles the video
# I/O and drawing; pandas/numpy handle the data we extract.
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# %% 2. Prepare the hand-tracking model

# The Hand Landmarker returns 21 landmarks per hand (fingertips, joints, wrist).
# Download the model file once from:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="Models/hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
    # --- ADD THESE LINES ---
    # Lowering these defaults (usually 0.5) helps detect closed/occluded hands.
    min_hand_detection_confidence=0.3, # Helps initial detection of a fist
    min_hand_presence_confidence=0.3,  # Crucial for VIDEO mode to keep tracking
    min_tracking_confidence=0.3,       # Keeps the landmarks stable
)

detector = vision.HandLandmarker.create_from_options(options)


# %% 3. Define the hand connections

# Each pair is two landmark indices that we'll join with a line when drawing.
# This is only for visualisation — it doesn't affect the tracking itself.
HAND_CONNECTIONS = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(5, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(9, 13), (13, 14), (14, 15), (15, 16)],
    "little": [(13, 17), (17, 18), (18, 19), (19, 20)],
    "palm": [(0, 17)],
}


# %% 4. Open the video

# For head-mounted eye-tracking, this is the scene-camera (egocentric) recording
# — the participant's first-person view of their own hands.
cap = cv2.VideoCapture("TestNina1.mp4")

if not cap.isOpened():
    raise FileNotFoundError("The video could not be opened.")

# We need width/height to turn MediaPipe's normalised (0–1) coordinates back
# into pixel coordinates later.
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Average frames per second: {fps:.2f}")
print(f"Total number of frames: {frame_count}")
print(f"Video size: {width} × {height}")


# %% 5. Process the video frame by frame

tracking_data = []
frame_number = 0

while True:

    # Read the next frame; cap.read() returns False when the video ends.
    success, frame = cap.read()
    if not success:
        break

    # MediaPipe's VIDEO mode needs a monotonically increasing timestamp (ms).
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # OpenCV loads frames as BGR, but MediaPipe expects RGB — so convert, then
    # wrap the array in MediaPipe's own image type.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run detection on this frame.
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # One row of the output table = one frame.
    row = {"frame": frame_number, "timestamp_ms": timestamp_ms}

    # result.hand_landmarks is a list with one entry per detected hand.
    for hand_number, hand in enumerate(result.hand_landmarks, start=1):

        points = []

        for point_number, landmark in enumerate(hand):

            # Landmarks come back normalised (0–1), so scale by frame size to
            # get pixel coordinates. z is a *relative* depth (not in mm/cm),
            # roughly 0 at the wrist and negative towards the camera.
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            row[f"hand_{hand_number}_point_{point_number}_x"] = x
            row[f"hand_{hand_number}_point_{point_number}_y"] = y
            row[f"hand_{hand_number}_point_{point_number}_z"] = landmark.z

        # --- Draw the skeleton (purely for the live preview) ---
        for connections in HAND_CONNECTIONS.values():
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # --- Bounding box ---
        # minAreaRect finds the smallest *rotated* rectangle around all 21
        # points. The box gives you a compact hand region — handy later if you
        # want to ask "was the participant's gaze inside the hand?" by checking
        # the gaze coordinate against this rectangle.
        rectangle = cv2.minAreaRect(np.array(points, dtype=np.float32))
        (center_x, center_y), (rectangle_width, rectangle_height), angle = rectangle

        # boxPoints turns the (centre, size, angle) description into 4 corners.
        box = cv2.boxPoints(rectangle).astype(int)

        for corner_number, (x, y) in enumerate(box, start=1):
            row[f"hand_{hand_number}_rectangle_corner_{corner_number}_x"] = x
            row[f"hand_{hand_number}_rectangle_corner_{corner_number}_y"] = y

        cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)

    tracking_data.append(row)

    # Show the annotated frame so you can sanity-check tracking quality.
    cv2.imshow("Hand tracking", frame)

    frame_number += 1

    # --- Playback controls: spacebar pauses/resumes, 'q' quits ---
    key = cv2.waitKey(1) & 0xFF

    # Pause on spacebar; press it again to resume.
    if key == ord(" "):
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):       # resume
                break

    # Quit on q or Esc, during playback OR pause.
    if key in (ord("q"), 27):
        break



# %% 6. Build and save the data table

# Frames where a hand wasn't detected simply lack those columns, so pandas
# fills them with NaN automatically — no need to handle missing hands by hand.
tracking_df = pd.DataFrame(tracking_data)

print(tracking_df.head())
print(f"DataFrame shape: {tracking_df.shape}")

tracking_df.to_csv("Results/hand_tracking_result.csv", index=False)


# %% 7. Release resources

# Always free the video file, the model, and any OpenCV windows at the end.
cap.release()
detector.close()
cv2.destroyAllWindows()