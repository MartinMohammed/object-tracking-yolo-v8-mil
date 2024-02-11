import cv2
import sys
import os
from utils.index import (
    measure_time,
    get_bounding_box,
    get_points_from_bbox,
    get_tracker,
)

(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split(".")

TRACKER_TYPES = [
    "BOOSTING",
    "MIL",
    "KCF",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
    "CSRT",
]

# --------------------- Things that can be changed ---------------------
VIDEO_OF_INTEREST = "drone.mp4"

# 2 (Faster but less accurate), 7
TRACKER_TYPE = TRACKER_TYPES[7]


# ----------------------------------------------------------------------
def main():
    tracker = get_tracker(minor_ver=MINOR_VER, tracker_type=TRACKER_TYPE)
    video = cv2.VideoCapture(os.path.join("assets", "videos", VIDEO_OF_INTEREST))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()

    if not ok:
        print("Cannot read video file")
        sys.exit()

    bbox = get_bounding_box(frame=frame)

    # Initialize tracker with first frame and bounding box
    tracker.init(image=frame, boundingBox=bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()

        # Cannot read frame.
        if not ok:
            break

        # Can be optimised by slicing the video in multiple parts
        frame_copy = measure_time(name="Copy current frame", cb=frame.copy)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = measure_time("Update Tracker", tracker.update, frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1, p2 = get_points_from_bbox(bbox=bbox)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(
                frame,
                "Tracking failure detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        # Display tracker type on frame
        cv2.putText(
            frame,
            TRACKER_TYPE + " Tracker",
            (100, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 170, 50),
            2,
        )

        # Display FPS on frame
        cv2.putText(
            frame,
            "FPS : " + str(int(fps)),
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 170, 50),
            2,
        )
        # Display result
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            # Allow re-selection of bounding box.
            bbox = cv2.selectROI(frame_copy, False)
            tracker = get_tracker(minor_ver=MINOR_VER, tracker_type=TRACKER_TYPE)
            tracker.init(image=frame, boundingBox=bbox)
        # Exit if ESC pressed
        elif k == 27:
            break


if __name__ == "__main__":
    main()
