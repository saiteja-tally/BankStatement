from typing import Tuple, List
import numpy as np
import cv2
from typing import Dict, Optional
import pandas as pd

from .constants import *
from .utils import validate_tables_with_rects, get_lines_inside_table, categorize_lines, cluster_lines, chars_to_words, group_words_by_logical_cells
from ..config import is_debug_mode

class TableDetector():

    def __init__(self, recognizer, dpi: int = 300):
        self.dpi = dpi
        self.recognizer = recognizer
    
    def _reset(self):
        self.image = None
        self.plumber_page = None
        self.height = 0
        self.width = 0
        self.detected_tables = []
        self.table_xyxy = None
        self.horizontal_lines = []
        self.vertical_lines = []

    def get_table(self, plumber_page):
        self._reset()

        pil_image = plumber_page.to_image(resolution=self.dpi).original
        cv_image = np.array(pil_image)
        self.image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        self.height, self.width, _ = self.image.shape
        self.plumber_page = plumber_page

        table_xyxy, table_type, table_v_lines, table_h_lines = self.detect_table_cv()

        if not table_xyxy:
            if is_debug_mode():
                print(f"[TableDetector] Falling back to altcv method for page {plumber_page.page_number}.")
            table_xyxy = self.detect_table_altcv()

            if not table_xyxy:
                return None, None, None, None

        return table_xyxy, table_type, table_v_lines, table_h_lines

    def detect_table_cv(self):
        
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
    
    def detect_table_altcv(self):
        
        header_blocks = self._extract_header_blocks_between_edges()

        if header_blocks:
            best_header = header_blocks[0]
            header_top = best_header['top']
            table_xyxy = (0, header_top/self.plumber_page.height, 1, 1)
        else:
            return None

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

    def _extract_header_blocks_between_edges(self, min_phrases_in_row=2, crop_margin=20):
        """Extract header blocks with bounding box and confidence scores"""
        chars = pd.DataFrame(self.plumber_page.chars)
        if chars.empty:
            return []

        hor_edges_ys = self._extract_horizontal_edges()
        if len(hor_edges_ys) < 2:
            return []

        header_blocks = []
        for k in range(len(hor_edges_ys) - 1):
            top, bottom = hor_edges_ys[k], hor_edges_ys[k + 1]

            # Extract chars within the band
            chars_band = chars[
                (chars["top"] >= top) & (chars["bottom"] <= bottom)
            ].sort_values(by=["y0", "x0"], ascending=[False, True])

            if chars_band.empty:
                continue
            
            words = chars_to_words(chars_band, char_tol=2)
            phrases = group_words_by_logical_cells(words, x_gap_threshold=6, y_proximity_threshold=5)

            if len(phrases) < min_phrases_in_row:
                continue

            # Compute confidence
            confidence = self.recognizer.compute_semantic_header_score(phrases)

            # Compute bounding box for this header block
            x0 = min(p["x0"] for p in phrases)
            y0 = min(p["y0"] for p in phrases)
            x1 = max(p["x1"] for p in phrases)
            y1 = max(p["y1"] for p in phrases)

            header_blocks.append({
                "phrases": phrases,
                "confidence": confidence,
                "bbox": (x0, y0, x1, y1),  # full rectangle with margin on top
                "top": top,
                "butttom": bottom
            })

        # Sort by confidence descending
        header_blocks.sort(key=lambda x: x["confidence"], reverse=True)
        return header_blocks

    def _extract_horizontal_edges(self, cluster_tol=1.0):
        """Extract horizontal edges using pdfplumber's metadata and cluster nearby lines"""
        # 1. Gather lines and edges
        raw_objs = self.plumber_page.lines + getattr(self.plumber_page, 'edges', [])
        if not raw_objs:
            return []

        # 2. Filter using pdfplumber's built-in orientation attribute
        # Objects are usually dicts; orientation is 'h' for horizontal or 'v' for vertical
        hor_lines = [obj for obj in raw_objs if obj.get('orientation') == 'h']
        
        if not hor_lines:
            return []

        # 3. Collect and sort unique Y-coordinates
        # We use top (y0) and bottom (y1) to capture the full edge of the line/shape
        all_ys = []
        for l in hor_lines:
            all_ys.extend([float(l['top']), float(l['bottom'])])
        
        edge_ys = np.sort(np.unique(all_ys))

        # 4. Cluster nearby Y-coordinates to handle "noise"
        # If two lines are within 'cluster_tol', they become one single edge
        if len(edge_ys) > 0:
            cleaned_edges = []
            current_cluster = [edge_ys[0]]
            
            for y in edge_ys[1:]:
                if y - current_cluster[-1] <= cluster_tol:
                    current_cluster.append(y)
                else:
                    # Use the average of the cluster as the definitive edge
                    cleaned_edges.append(np.mean(current_cluster))
                    current_cluster = [y]
            
            cleaned_edges.append(np.mean(current_cluster))
            return np.array(cleaned_edges)
        
        return []