import cv2
import sys
import os
from utils.opencv import (
    get_bounding_box_roi,
    get_points_from_bbox,
    get_tracker,
    draw_rectangle_with_label,
)
import time
from utils.yolo import get_bounding_box_yolo_v8, get_name_from_class_id
from test.yolo_v8_only import test_yolo_v8_only
from constants import (
    VIDEO_OF_INTEREST,
    TRACKER_TYPE,
    DEFAULT_COLOR,
    ALARM_COLOR,
    DETECTION_TIME_INTERVAL_MS,
    ONLY_DETECTION,
    MISSED_DETECTIONS_UNTIL_LOST,
    DETECTOR_INTEREST_LABEL,
    REDETECTION_INTERVAL_MS, 
    P
)
from utils.opencv_window import display_default_info_on_frame, display_additional_labels
from ultralytics import YOLO

(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split(".")


# ----------------------------------------------------------------------
def main(
    detection_interval: int,
    redetection_interval_ms: int, 
    missed_detections_until_lost: int,
    p: float,  
    only_detection: bool = False,
):
    if only_detection:
        test_yolo_v8_only()

    detection_interval_orig = detection_interval

    # Load an official or custom model
    detector = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    tracker = get_tracker(minor_ver=MINOR_VER, tracker_type=TRACKER_TYPE)
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

    addit_labels = ["", ""]
    addit_labels_c = [DEFAULT_COLOR, DEFAULT_COLOR]

    # Classification probability (float)
    p = None

    # Classification label (int)
    l = None

    # Keep bbox in the format of (x, y, w, h)
    bbox = None

    result = get_bounding_box_yolo_v8(
        frame=frame, detector=detector
    )  # or get_bounding_box_roi(frame=frame)
    if result is not None:
        bbox_local, mt_local = result
        p_local, l_local, v_local = mt_local
        class_name = get_name_from_class_id(model=detector, class_id=l_local)
        object_match = l_local == DETECTOR_INTEREST_LABEL
        if object_match:
            bbox, _ = bbox_local, mt_local
            p, l, _ = p_local, l_local, v_local
        else:
            # Get user selection.
            bbox = get_bounding_box_roi(frame=frame)

        addit_labels[1] = f"Found {class_name} ({p_local:.2f}): {v_local}s"
        addit_labels_c[1] = DEFAULT_COLOR if object_match else ALARM_COLOR

    else:
        addit_labels[1] = "Inital bbox was not found by Yolov8."
        addit_labels_c[1] = ALARM_COLOR
        bbox = get_bounding_box_roi(frame=frame)

    # Initialize tracker with first frame and bounding box
    tracker.init(image=frame, boundingBox=bbox)

    detection_timer_start = time.time()
    lost_timer = None

    missed_detections_counter = 0
    object_is_lost = False
    
    while True:
        # Read a new frame
        ok, frame = video_capture.read()

        # Cannot read frame.
        if not ok:
            break

        # Can be optimised by slicing the video_capture in multiple parts
        frame_copy = frame.copy()  # measure_time(name="Copy Frame", cb=frame.copy)

        # Start timer
        timer = cv2.getTickCount()
        detection_timer_end = time.time()
         
        time_went_by_ms = 1000 * (detection_timer_end - detection_timer_start)
        if time_went_by_ms >= detection_interval:
            result = get_bounding_box_yolo_v8(frame=frame, detector=detector)
            if result is not None:
                bbox_local, mt_local = result
                p_local, l_local, v_local = mt_local
                class_name = get_name_from_class_id(model=detector, class_id=l_local)
                object_match = l_local == DETECTOR_INTEREST_LABEL and p_local >= P
                if object_match:
                    missed_detections_counter = 0 
                    lost_timer = None
                    object_is_lost = False
                    detection_interval = detection_interval_orig

                    bbox, _ = bbox_local, mt_local
                    p, l, _ = p_local, l_local, v_local
                    # Re-init tracker with new bounding box from detector.
                    tracker.init(image=frame, boundingBox=bbox)
                else:
                    missed_detections_counter += 1
                addit_labels[1] = f"Found {class_name} ({p_local:.2f}): {v_local}s"
                addit_labels_c[1] = DEFAULT_COLOR if object_match else ALARM_COLOR
            else:
                missed_detections_counter += 1
                addit_labels[1] = f"No Object detected ({missed_detections_counter})"
                addit_labels_c[1] = ALARM_COLOR
            detection_timer_start = detection_timer_end
        else:
            addit_labels[
                0
            ] = f"Next detection in {round((detection_interval - time_went_by_ms) / 1000, 2)}s."
            addit_labels_c[0] = DEFAULT_COLOR

        if missed_detections_counter == missed_detections_until_lost:
            object_is_lost = True
            detection_interval = redetection_interval_ms # 200 ms
            lost_timer = time.time()

        if not object_is_lost:
            # Update tracker
            ok, bbox = tracker.update(
                frame
            )  # measure_time("Update Tracker", tracker.update, frame)

            # Draw bounding box - failed to predict the next position of the object.
            if ok:
                # Tracking success
                p1, p2 = get_points_from_bbox(bbox=bbox, xywh_format=True)
                label_text = f"p={p:.2f}, l={l}" if p and l else ""
                draw_rectangle_with_label(frame=frame, p1=p1, p2=p2, label_text=label_text)
            else:
                addit_labels[1] = "Tracking failure detected."
                addit_labels_c[1] = ALARM_COLOR
            
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            display_default_info_on_frame(frame=frame, tracker_type=TRACKER_TYPE, fps=fps)
            display_additional_labels(
                frame=frame,
                addit_labels=addit_labels,
                addit_labels_c=addit_labels_c,
            )
        else:
            cv2.putText(
                frame,
                f"Object is lost ({round(time.time() - lost_timer, 2)}s)",
                (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                ALARM_COLOR,
                2,
            )

        # Display result
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xFF

        # s - manual select
        if k == ord("s"):
            # Allow re-selection of bounding box.
            bbox = get_bounding_box_roi(frame=frame_copy)

            # Reinit-tracker
            tracker.init(image=frame, boundingBox=bbox)
        elif k == ord("d"):
            result = get_bounding_box_yolo_v8(frame=frame_copy, detector=detector)

            # Reset timer:
            detection_timer_start = time.time()
            if result is not None:
                bbox_local, mt_local = result
                p_local, l_local, v_local = mt_local
                class_name = get_name_from_class_id(model=detector, class_id=l_local)
                object_match = l_local == DETECTOR_INTEREST_LABEL and p_local >= P
                if object_match:
                    bbox, _ = bbox_local, mt_local
                    p, l, _ = p_local, l_local, v_local

                    # Reinit tracker
                    tracker.init(image=frame, boundingBox=bbox)
                addit_labels[1] = f"Found {class_name} ({p_local:.2f}): {v_local}s"
                addit_labels_c[1] = DEFAULT_COLOR if object_match else ALARM_COLOR

            else:
                addit_labels[1] = f"No Object detected ({missed_detections_counter})"
                addit_labels_c[1] = ALARM_COLOR

        # Exit if ESC pressed
        elif k == 27:
            break


if __name__ == "__main__":
    main(detection_interval=DETECTION_TIME_INTERVAL_MS, redetection_interval_ms = REDETECTION_INTERVAL_MS,  missed_detections_until_lost = MISSED_DETECTIONS_UNTIL_LOST, p=P, only_detection=ONLY_DETECTION)
