from typing import Callable, Any, Tuple, List, Union
import time
import cv2
import sys


def measure_time(name: str, cb: Callable[..., Any], *args) -> Any:
    """
    Measures the execution time of a callback function.

    Parameters:
        name (str): A name for the operation being measured.
        cb (Callable[..., Any]): The callback function to be executed.
        *args: Variable length argument list to pass to the callback function.

    Returns:
        Any: The output of the callback function.
    """
    start_time = time.time()
    output = cb(*args)
    end_time = time.time()
    print(f"Time to execute '{name}' took {round(end_time - start_time, 3)} seconds")
    return output


def get_points_from_bbox(bbox: Tuple[int]) -> List[Tuple[int]]:
    """
    Extracts corner points from a bounding box.

    Parameters:
        bbox (Tuple[int]): A tuple representing the bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
        List[Tuple[int]]: A list of tuples representing the corner points of the bounding box.
    """
    x1, y1, width, height = bbox
    return [(int(x1), int(y1)), (int(x1 + width), int(y1 + height))]


def get_bounding_box(frame) -> Tuple[int]:
    """
    Either use a detector to get the bounding box of the current frame,
    use a default selection or ask the user which bounding box to choose from.
    In case of the last option a new window opens and asks for a region of interest.
    """
    # # Define an initial bounding box
    # bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    return bbox


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
