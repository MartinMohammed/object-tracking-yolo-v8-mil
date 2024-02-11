from typing import Tuple, List, Union
import cv2


def get_points_from_bbox(
    bbox: Tuple[float, float, float, float], xywh_format: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Extracts corner points from a bounding box.

    Parameters:
        bbox (Tuple[int]): A tuple representing the bounding box coordinates.
            If xywh_format is False (default), the tuple is assumed to be in (x_min, y_min, x_max, y_max) format.
            If xywh_format is True, the tuple is assumed to be in (x, y, width, height) format,
            and it will be converted to (x_min, y_min, x_max, y_max) format.

        xywh_format (bool): Whether the input bbox is in (x, y, width, height) format.
            If True, the input bbox will be converted to (x_min, y_min, x_max, y_max) format.

    Returns:
        List[Tuple[int]]: A list of tuples representing the corner points of the bounding box.
    """
    if xywh_format:
        bbox = conv_xywh_to_xyxy(bbox=bbox)
    x1, y1, x2, y2 = bbox
    return [(int(x1), int(y1)), (int(x2), int(y2))]


def conv_xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[int]:
    """
    E.g. convert bounding box from tracker in format of rectangle position specification.
    Converts bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

    Parameters:
        bbox (Tuple[float]): A tuple representing the bounding box coordinates in (x, y, width, height) format.

    Returns:
        Tuple[int]: A tuple representing the bounding box coordinates in (x1, y1, x2, y2) format.
    """
    x, y, width, height = bbox
    return (int(x), int(y), int(x + width), int(y + height))


def conv_xyxy_to_xywh(
    bbox: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    """
    E.g. convert bounding box from detector in format of tracker.
    Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Parameters:
        bbox (Tuple[float, float, float, float]): A tuple representing the bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        Tuple[int, int, int, int]: A tuple representing the bounding box coordinates in (x, y, width, height) format.
    """
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def get_bounding_box_roi(frame) -> Tuple[int, int, int, int]:
    """
    Either uses a detector to get the bounding box of the current frame,
    uses a default selection, or asks the user which bounding box to choose from.
    In case of the last option, a new window opens and asks for a region of interest.

    Parameters:
        frame: The current frame.

    Returns:
        Tuple[int, int, int, int]: A tuple representing the selected bounding box coordinates in (x, y, width, height) format.
    """
    # Defaults to ROI selection if no bounding box was discovered.
    # bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    return bbox


def draw_rectangle_with_label(
    frame, p1: Tuple[int, int], p2: Tuple[int, int], label_text
) -> None:
    # Label containing probability of classification (p) and class (label)
    cv2.putText(
        frame,
        label_text,
        (p1[0], p1[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def get_tracker(
    minor_ver: str, tracker_type: str
) -> Union[cv2.TrackerMIL, cv2.TrackerCSRT]:
    """
    Gets an instance of OpenCV tracker based on the provided tracker type.

    Parameters:
        minor_ver (str): The minor version of OpenCV.
        tracker_type (str): The type of tracker to create.

    Returns:
        Union[cv2.TrackerMIL, cv2.TrackerCSRT]: An instance of the specified tracker type.

    Raises:
        ValueError: If the tracker type is not recognized.
    """
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == "BOOSTING":
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == "MIL":
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == "TLD":
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == "MEDIANFLOW":
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == "GOTURN":
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"Tracker {tracker_type} is not known.")
    return tracker
