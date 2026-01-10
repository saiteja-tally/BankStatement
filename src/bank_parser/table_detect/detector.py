from typing import Tuple, List
import numpy as np
import cv2
from typing import Dict, Optional
import pandas as pd

from .constants import *
from .utils import validate_tables_with_rects, get_lines_inside_table, categorize_lines, cluster_lines, extract_header_blocks_between_edges
from ..config import is_debug_mode

class TableDetector():

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
    
    def get_table(self, plumber_page, recognizer):

        table_xyxy, table_type, table_v_lines, table_h_lines = self.detect_table_cv(plumber_page)

        if not table_xyxy:
            if is_debug_mode():
                print(f"[TableDetector] Falling back to pdfplumber method for page {plumber_page.page_number}.")
            table_xyxy = self.detect_table_altcv(plumber_page, recognizer)

            if not table_xyxy:
                return None, None, None, None

        return table_xyxy, table_type, table_v_lines, table_h_lines

    def detect_table_cv(self, plumber_page) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
        
        # Convert PDF page to image using PDFPlumber's to_image
        img = plumber_page.to_image(resolution=self.dpi)
        # Get the numpy array from the PIL Image
        pil_image = img.original
        img_data = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        self.image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        self.height, self.width, _ = self.image.shape

        # Detect lines
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Create a sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
        # Apply the sharpening filter
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        detected_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(HOUGH_THRESHOLD_RATIO*edges.shape[1]),
                                    minLineLength=int(HOUGH_MIN_LINE_LENGTH_RATIO*edges.shape[1]),
                                    maxLineGap=int(HOUGH_MAX_LINE_GAP_RATIO*edges.shape[1]))

        if detected_lines is None:
            return None, None, None, None

        detected_lines = [[float(line[0][0])/self.width, 
                           float(line[0][1])/self.height, 
                           float(line[0][2])/self.width, 
                           float(line[0][3])/self.height]
                           for line in detected_lines]

        # Detect rectangles
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        detected_rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            norm_x, norm_y = x/self.image.shape[1], y/self.image.shape[0]
            norm_w, norm_h = w/self.image.shape[1], h/self.image.shape[0]
            if norm_w > 0.02 and norm_h > 0.01:
                detected_rects.append([norm_x, norm_y, norm_x+norm_w, norm_y + norm_h])

        # Process detections
        self.horizontal_lines, self.vertical_lines = categorize_lines(detected_lines)
        clusters = cluster_lines(self.horizontal_lines, self.vertical_lines)
        self.detected_tables = validate_tables_with_rects(clusters, detected_rects)

        # Get the largest table as image crop
        largest_table = self._get_largest_table()
        if largest_table:
            self.table_xyxy = self._get_table_xyxy(largest_table)
            
            self.horizontal_lines = get_lines_inside_table(self.horizontal_lines, self.table_xyxy)
            self.vertical_lines = get_lines_inside_table(self.vertical_lines, self.table_xyxy)

            self.horizontal_lines.extend(self._add_border_lines("horizontal"))
            self.vertical_lines.extend(self._add_border_lines("vertical"))
            self._merge_close_lines()
            

            ## Keep only those vertical lines that are larger dthan 50% of table height and stretch their height to match the table height
            filtered_normalized_v = []
            for line in self.vertical_lines:
                line_height = abs(line[3] - line[1])
                if line_height >= 0.5 * (self.table_xyxy[3] - self.table_xyxy[1]):
                    # Stretch line to match table height
                    line[1] = self.table_xyxy[1]
                    line[3] = self.table_xyxy[3]
                    filtered_normalized_v.append(line)
            self.vertical_lines = filtered_normalized_v


            table_type = None
            for vt in self.detected_tables:
                # vt['id'] same format as largest_table['id'] if validated_tables used cluster id
                if vt['id'] == largest_table['id']:
                    table_type = vt.get('type', vt.get('type'))
                    break


        else:
            self.table_xyxy = None
            self.horizontal_lines = []
            self.vertical_lines = []
            table_type = None 

        return self.table_xyxy, self.horizontal_lines, self.vertical_lines, table_type
    
    def detect_table_altcv(self, plumber_page, recognizer) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
        
        header_blocks = extract_header_blocks_between_edges(plumber_page, recognizer)

        if header_blocks:
            best_header = header_blocks[0]
            header_bbox = best_header['bbox']
            table_xyxy = (0, header_bbox[1]/plumber_page.height, 1, 1)
            return table_xyxy
        else:
            return None, None

    def _get_largest_table(self) -> Optional[Dict]:
        """Return the table with the largest area."""
        if not self.detected_tables:
            return None
        return max(self.detected_tables, 
                  key=lambda t: (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1]))
    
    def _get_table_xyxy(self, table: Dict) -> np.ndarray:
        """Convert table cells to a numpy array.

        Previously this returned a binary mask. Now it returns the cropped image
        region (BGR) corresponding to the table bounding box, along with the
        bounding rect (x_min, y_min, x_max, y_max).

        Returns:
            (table_rect, table_image): table_rect is a tuple (x_min, y_min, x_max, y_max)
            and table_image is an ndarray (H, W, C) crop from self.image. If the
            table is invalid or the crop has zero area, returns (None, np.array([])).
        """
        if not table or 'cells' not in table:
            return None, np.array([])

        # Get table boundaries
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for cell in table['cells']:
            x_min = min(x_min, cell[0])
            y_min = min(y_min, cell[1])
            x_max = max(x_max, cell[2])
            y_max = max(y_max, cell[3])

        return (x_min, y_min, x_max, y_max)
    
    def _add_border_lines(self, direction):
        if not direction:
            return
        
        x_min, y_min, x_max, y_max = self.table_xyxy
        
        if direction == "horizontal":
            border_lines = [
                # top: left->right
                [x_min, y_min, x_max, y_min],
                # bottom: left->right
                [x_min, y_max, x_max, y_max],
            ]
        elif direction == "vertical":
            border_lines = [
                # left: top->bottom
                [x_min, y_min, x_min, y_max],
                # right: top->bottom
                [x_max, y_min, x_max, y_max],
            ]

        return border_lines
 
    def _merge_close_lines(self):

        self.horizontal_lines.sort(key=lambda x: x[1])
        self.vertical_lines.sort(key=lambda x: x[0])

        # merge nearby lines (optional, can be implemented if needed)
        merged_h_lines = []
        for line in self.horizontal_lines:
            if not merged_h_lines:
                merged_h_lines.append(line)
                continue
            last_line = merged_h_lines[-1]
            if abs(line[1] - last_line[1]) < PARALLEL_LINE_PROXIMITY:
                merged_h_lines[-1] = [
                    self.table_xyxy[0],
                    (last_line[1] + line[1]) / 2,
                    self.table_xyxy[2],
                    (last_line[3] + line[3]) / 2,
                ]
            else:
                merged_h_lines.append(line)
        
        merged_v_lines = []
        for line in self.vertical_lines:
            if abs(line[1]- line[3]) < 0.5 * (self.table_xyxy[3] - self.table_xyxy[1]):
                continue
            if not merged_v_lines:
                merged_v_lines.append(line)
                continue
            last_line = merged_v_lines[-1]
            if abs(line[0] - last_line[0]) < PARALLEL_LINE_PROXIMITY:
                merged_v_lines[-1] = [
                    (last_line[0] + line[0]) / 2,
                    self.table_xyxy[1],
                    (last_line[2] + line[2]) / 2,
                    self.table_xyxy[3],
                ]
            else:
                merged_v_lines.append(line)

        self.horizontal_lines = merged_h_lines
        self.vertical_lines = merged_v_lines

   