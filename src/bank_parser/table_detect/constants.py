# --- Configuration Constants ---
# Tolerance for classifying lines as horizontal or vertical
LINE_ANGLE_TOLERANCE = 0.02
HLINES_MIN_LENGTH_RATIO = 0.65
VLINES_MIN_LENGTH_RATIO = 0.05
# Proximity tolerance for clustering parallel lines
PARALLEL_LINE_PROXIMITY = 0.0125
# Intersection tolerance for connecting horizontal and vertical lines
INTERSECTION_TOLERANCE = 0.002
# Minimum number of lines/rectangles to be considered a table
MIN_LINES_FOR_TABLE = 3
MIN_RECTS_FOR_TABLE = 2

# --- CV2-Specific Hough Line Transform Parameters ---
HOUGH_THRESHOLD_RATIO = 0.01
HOUGH_MIN_LINE_LENGTH_RATIO = 0.01
HOUGH_MAX_LINE_GAP_RATIO = 0.01
# --- PDF to Image Conversion DPI ---
PDF_TO_IMAGE_DPI = 300

CLASSIFY_MIN_INTERNAL_H = 2
CLASSIFY_MIN_INTERNAL_V = 2
CLASSIFY_INTERSECTION_RATIO = 0.6
CLASSIFY_COVERAGE_THRESHOLD = 0.6
CLASSIFY_OUTER_MARGIN_TOL = 0.05  # fraction of width/height to treat as outer border
CLASSIFY_INTERSECTION_TOL = 5  # in pixels (reuses same scale as other functions)