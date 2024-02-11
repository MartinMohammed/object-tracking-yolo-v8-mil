import cv2
from typing import List, Tuple
from constants import DEFAULT_COLOR


def display_default_info_on_frame(frame, tracker_type, fps) -> None:
    """
    Displays default information (tracker type and FPS) on the frame.

    Parameters:
        frame: The frame to display information on.
        tracker_type (str): The type of tracker being used.
        fps (int): The frames per second value.
    """
    # Display tracker type on frame
    cv2.putText(
        frame,
        tracker_type + " Tracker",
        (100, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        DEFAULT_COLOR,
        2,
    )
    # Display FPS on frame
    cv2.putText(
        frame,
        "FPS : " + str(int(fps)),
        (100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        DEFAULT_COLOR,
        2,
    )


def display_additional_labels(
    frame,
    addit_labels: List[str],
    addit_labels_c: List[Tuple[int, int, int]],
) -> None:
    """
    Displays additional labels on the frame with specified colors.

    Parameters:
        frame: The frame to display labels on.
        addit_labels (List[str]): A list of additional labels to display.
        addit_labels_c (List[Tuple[int, int, int]]): A list of RGB tuples representing the colors of the additional labels.

    Raises:
        ValueError: If the size of `addit_labels` does not match the size of `addit_labels_c`.

    Note:
        Each additional label will be displayed vertically below the previous label starting from the position (100, 50).
    """
    if len(addit_labels) != len(addit_labels_c):
        raise ValueError(
            "Additional labels size must equal additional labels color size"
        )
    for i in range(1, len(addit_labels) + 1):
        y = 50 + 30 * (i)
        cv2.putText(
            frame,
            addit_labels[i - 1],
            (100, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            addit_labels_c[i - 1],
            2,
        )
