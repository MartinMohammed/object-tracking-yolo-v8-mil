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
# 4 = aircraft index.
DETECTOR_INTEREST_LABEL = 4

# --------------------- Things that can be changed ---------------------

ONLY_DETECTION = False

BOUNDING_BOX_COLOR = (255, 0, 0)
ALARM_COLOR = (0, 0, 255)
DEFAULT_COLOR = (50, 170, 50)


VIDEO_OF_INTEREST = "drone.mp4"

# 1 (lower fps on avg 30 but more accurate), 7
TRACKER_TYPE = TRACKER_TYPES[1]


DETECTION_TIME_INTERVAL_MS = 750
# --------------------- Things that can be changed ---------------------
