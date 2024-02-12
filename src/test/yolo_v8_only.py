from ultralytics import YOLO
import cv2
import os
import sys
from constants import VIDEO_OF_INTEREST, ALARM_COLOR
from utils.yolo import get_bounding_box_yolo_v8

def test_yolo_v8_only():
    detector = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    video_capture = cv2.VideoCapture(
        os.path.join("assets", "videos", VIDEO_OF_INTEREST)
    )

    # Exit if video_capture not opened.
    if not video_capture.isOpened():
        print("Could not open video_capture")
        sys.exit()

    # Read first frame.
    ok, frame = video_capture.read()

    if not ok:
        print("Cannot read video_capture file")
        sys.exit()

    while True:
        # Read a new frame
        ok, frame = video_capture.read()

        # Cannot read frame.
        if not ok:
            break

        result = get_bounding_box_yolo_v8(
            frame=frame, detector=detector, xywh_format=False
        )
        if result is not None:
            bbox, _ = result
            cv2.rectangle(
                img=frame,
                pt1=tuple(bbox[:2]),
                pt2=tuple(bbox[2:]),
                color=(0, 255, 0),
                thickness=1,
            )
        else:
            cv2.putText(
                frame,
                "No bounding box was detected",
                (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                ALARM_COLOR,
                2,
            )

        cv2.imshow("Detection YoloV8 Only", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break