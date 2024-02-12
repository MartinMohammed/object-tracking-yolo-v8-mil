from typing import Optional, Tuple
from utils.general import measure_time
from utils.opencv import conv_xyxy_to_xywh


def get_bounding_box_yolo_v8(
    frame, detector, xywh_format: bool = True
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, int, float]]]:
    """
    Retrieves the bounding box from the YOLOv8 detector results.

    Parameters:
        frame: The current frame.
        detector: The YOLOv8 detector.
        xywh_format (bool): Whether to return the bounding box coordinates in (x, y, width, height) format.
            If True (default), the coordinates will be converted to (x, y, width, height) format.

    Returns:
        Optional[Tuple[Tuple[int, int, int, int], Tuple[float, int, float]]]:
            A tuple containing the bounding box coordinates (x1, y1, x2, y2)
            and metadata (confidence score, label, detection speed) if a bounding box is detected,
            otherwise returns None.
    """
    # Perform tracking with the model
    results = measure_time(
        "YoloV8 detection", detector, frame, 18
    )  # Tracking with default tracker
    yolov8_results = parse_yolov8_results(results=results)

    if yolov8_results:
        x1, y1, x2, y2, p, label, detection_speed = yolov8_results
        print(
            f"Bounding box was discovered at p1=({x1}, {y1}) & p2=({x2}, {y2}) with {p=}, {label=} and {detection_speed=}s."
        )
        metadata = [p, label, detection_speed]
        bbox = (x1, y1, x2, y2)
        if xywh_format:
            bbox = conv_xyxy_to_xywh(bbox)
        return bbox, tuple(metadata)
    print("No detections from YOLOv8 detector.")
    return None


def parse_yolov8_results(
    results,
) -> Optional[Tuple[int, int, int, int, float, int, float]]:
    """
    Parses the results obtained from YOLOv8 detector and returns the bounding box coordinates.

    Parameters:
        results: The results obtained from the YOLOv8 detector.

    Returns:
        Optional[Tuple[int, int, int, int, float, int, float]]: A tuple containing the bounding box coordinates
        (x1, y1, x2, y2), confidence score (p), label, and detection speed (in seconds).
        Returns None if no bounding box is detected.
    """
    # Iterate over the generator to get each result
    for result in results:
        if len(result.boxes) == 0 or len(result.boxes[0].data) == 0:
            break
        x1, y1, x2, y2, p, label = result.boxes[0].data[0]
        detection_speed = round(sum(result.speed.values()) / 1000, 3)  # in seconds
        print("Total detection speed: ", detection_speed)
        return (
            int(x1.item()),
            int(y1.item()),
            int(x2.item()),
            int(y2.item()),
            p.item(),
            int(label.item()),
            detection_speed,
        )
    return None


def get_name_from_class_id(model, class_id: int) -> str:
    """
    Get the name corresponding to a class ID from the model's list of names.

    Args:
        model: The YOLO model object containing a list of class names.
        class_id (int): The integer class ID for which to retrieve the name.

    Returns:
        str: The name corresponding to the given class ID.

    Raises:
        ValueError: If the class ID is not a valid integer between 0 and 79 inclusive.
    """
    if not isinstance(class_id, int):
        raise ValueError("Class ID must be an integer.")
    if not (0 <= class_id <= 79):
        raise ValueError("Class ID must be between 0 and 79 inclusive.")
    return model.names[class_id]
