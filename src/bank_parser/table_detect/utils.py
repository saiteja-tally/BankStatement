from collections import defaultdict

from .table_classification import classify_cluster_table
from .constants import *

def get_lines_inside_table(lines, table_bbox):
    """
    Normalizes the coordinates of lines relative to a table's bounding box.

    Args:
        lines (list[list[float]]): A list of lines, where each line is [x1, y1, x2, y2].
        table_bbox (list[float]): The bounding box of the table [x_min, y_min, x_max, y_max].

    Returns:
        list[list[float]]: A list of normalized lines with coordinates between 0 and 1.
    """
    x_min, y_min, x_max, y_max = table_bbox

    def _line_intersects_bbox(x1, y1, x2, y2):
        # If either endpoint is inside the bbox -> intersects
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            return True
        if x_min <= x2 <= x_max and y_min <= y2 <= y_max:
            return True

        # Otherwise, check if the bounding boxes overlap (conservative intersection)
        line_min_x, line_max_x = min(x1, x2), max(x1, x2)
        line_min_y, line_max_y = min(y1, y2), max(y1, y2)

        # If one bbox is entirely to the left/right/top/bottom of the other, no intersection
        if line_max_x < x_min or line_min_x > x_max or line_max_y < y_min or line_min_y > y_max:
            return False
        return True

    # Filter lines that are inside or intersect the table bbox
    filtered_lines = [l for l in lines if _line_intersects_bbox(*l)]

    return filtered_lines


def validate_tables_with_rects(clusters, rects):
    """
    Validates line clusters by associating them with nearby rectangles (cells).
    """
    # This function is identical to the pdfplumber version
    # It demonstrates the modularity of the approach
    validated_tables = []
    for cluster in clusters:
        cluster_bbox = cluster['bbox']
        
        candidate_rects = []
        for rect in rects:
            rect_center_x = (rect[0] + rect[2]) / 2
            rect_center_y = (rect[1] + rect[3]) / 2
            if (cluster_bbox[0] <= rect_center_x <= cluster_bbox[2] and
                cluster_bbox[1] <= rect_center_y <= cluster_bbox[3]):
                candidate_rects.append(rect)

        if len(candidate_rects) < MIN_RECTS_FOR_TABLE:
            continue

        rows = defaultdict(list)
        for r in sorted(candidate_rects, key=lambda x: x[1]):
            found_row = any(abs(y_coord - r[1]) < 0.002 for y_coord in rows)
            if not found_row: rows[r[1]].append(r)

        cols = defaultdict(list)
        for r in sorted(candidate_rects, key=lambda x: x[0]):
            found_col = any(abs(x_coord - r[0]) < 0.002 for x_coord in cols)
            if not found_col: cols[r[0]].append(r)

        is_valid = False
        if cluster['type'] == 'full' and len(rows) >= 1 and len(cols) > 1: is_valid = True
        elif cluster['type'] == 'horizontal' and len(rows) > 1: is_valid = True
        elif cluster['type'] == 'vertical' and len(cols) > 1: is_valid = True

        if is_valid:
            table_type_refined, type_score = classify_cluster_table({
                "lines": cluster['lines'],
                "bbox": cluster['bbox']
            }, elements=None)
            validated_tables.append({
                'id': len(validated_tables) + 1,
                'bbox': cluster['bbox'],
                'type': table_type_refined,
                'cells': candidate_rects,
                'type_score': type_score,  
            })
    return validated_tables

def categorize_lines(lines):
    """
    Categorizes lines into horizontal and vertical lists.
    This version adapts the output from OpenCV's HoughLinesP.
    """
    horizontal_lines = []
    vertical_lines = []
    for line_coords in lines:
        x1, y1, x2, y2 = line_coords
        
        if abs(y1 - y2) < LINE_ANGLE_TOLERANCE and abs(x1 - x2) >= HLINES_MIN_LENGTH_RATIO:
            horizontal_lines.append(line_coords)
        elif abs(x1 - x2) < LINE_ANGLE_TOLERANCE and abs(y1 - y2) >= VLINES_MIN_LENGTH_RATIO:
            vertical_lines.append(line_coords)
        else:
            continue  # Ignore diagonal or short lines
    
    return horizontal_lines, vertical_lines


def cluster_lines(horizontal_lines, vertical_lines):
    """
    Clusters lines into potential table grids using a graph-based approach.
    """
    all_lines = [{'id': i, 'xyxy': line, 'type': 'h'} for i, line in enumerate(horizontal_lines)]
    v_offset = len(horizontal_lines)
    all_lines.extend([{'id': i + v_offset, 'xyxy': line, 'type': 'v'} for i, line in enumerate(vertical_lines)])
    
    adj = defaultdict(list)
    for i in range(len(all_lines)):
        for j in range(i + 1, len(all_lines)):
            line1, line2 = all_lines[i], all_lines[j]
            connected = False
            if line1['type'] != line2['type']:
                h_line = line1 if line1['type'] == 'h' else line2
                v_line = line2 if line1['type'] == 'h' else line1
                if lines_intersect(h_line, v_line):
                    connected = True
            if connected:
                adj[line1['id']].append(line2['id'])
                adj[line2['id']].append(line1['id'])

    visited = set()
    clusters = []
    for line_id in range(len(all_lines)):
        if line_id not in visited:
            cluster_lines_ids = set()
            q = [line_id]
            visited.add(line_id)
            while q:
                curr_id = q.pop(0)
                cluster_lines_ids.add(curr_id)
                for neighbor_id in adj[curr_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        q.append(neighbor_id)
            
            if len(cluster_lines_ids) >= MIN_LINES_FOR_TABLE:
                clusters.append([all_lines[i] for i in cluster_lines_ids])

    processed_clusters = []
    for cluster in clusters:
        h_lines = [line for line in cluster if line['type'] == 'h']
        v_lines = [line for line in cluster if line['type'] == 'v']
        
        table_type = 'unknown'
        if h_lines and v_lines: table_type = 'full'
        elif h_lines: table_type = 'horizontal'
        elif v_lines: table_type = 'vertical'
            
        min_x = min(l['xyxy'][0] for l in cluster)
        min_y = min(l['xyxy'][1] for l in cluster)
        max_x = max(l['xyxy'][2] for l in cluster)
        max_y = max(l['xyxy'][3] for l in cluster)
        
        processed_clusters.append({
            'lines': cluster,
            'bbox': (min_x, min_y, max_x, max_y),
            'type': table_type
        })
    return processed_clusters

def lines_intersect(h_line, v_line):
    """
    Checks if a horizontal and a vertical line intersect within a tolerance.
    """
    return (
        min(h_line['xyxy'][0], h_line['xyxy'][2]) - INTERSECTION_TOLERANCE <= v_line['xyxy'][0] <= max(h_line['xyxy'][0], h_line['xyxy'][2]) + INTERSECTION_TOLERANCE and
        min(v_line['xyxy'][1], v_line['xyxy'][3]) - INTERSECTION_TOLERANCE <= h_line['xyxy'][1] <= max(v_line['xyxy'][1], v_line['xyxy'][3]) + INTERSECTION_TOLERANCE
    )

def are_parallel_and_close(line1, line2, is_horizontal):
    """
    Checks if two parallel lines are close enough to be part of the same grid.
    """
    if is_horizontal:
        if abs(line1['xyxy'][1] - line2['xyxy'][1]) < PARALLEL_LINE_PROXIMITY:
            overlap = max(0, min(line1['xyxy'][2], line2['xyxy'][2]) - max(line1['xyxy'][0], line2['xyxy'][0]))
            return overlap > 0
    else: # is_vertical
        if abs(line1['xyxy'][0] - line2['xyxy'][0]) < PARALLEL_LINE_PROXIMITY:
            overlap = max(0, min(line1['xyxy'][3], line2['xyxy'][3]) - max(line1['xyxy'][0], line2['xyxy'][0]))
            return overlap > 0
    return False


def merge_lines(horizontal_lines, vertical_lines):

        # merge nearby lines (optional, can be implemented if needed)
    merged_h_lines = []
    for line in horizontal_lines:
        if not merged_h_lines:
            merged_h_lines.append(line)
            continue
        last_line = merged_h_lines[-1]
        if abs(line[1] - last_line[1]) < PARALLEL_LINE_PROXIMITY:
            merged_h_lines[-1] = [
                min(last_line[0], line[0]),
                (last_line[1] + line[1]) / 2,
                max(last_line[2], line[2]),
                (last_line[3] + line[3]) / 2,
            ]
        else:
            merged_h_lines.append(line)
    
    merged_v_lines = []
    for line in vertical_lines:
        if not merged_v_lines:
            merged_v_lines.append(line)
            continue
        last_line = merged_v_lines[-1]
        if abs(line[0] - last_line[0]) < PARALLEL_LINE_PROXIMITY:
            merged_v_lines[-1] = [
                (last_line[0] + line[0]) / 2,
                min(last_line[1], line[1]),
                (last_line[2] + line[2]) / 2,
                max(last_line[3], line[3]),
            ]
        else:
            merged_v_lines.append(line)

    return merged_h_lines, merged_v_lines

def chars_to_words(chars, char_tol):
    """Convert characters to words, handling horizontal spacing"""
    words = []
    word = []
    prev_x1 = None
    for _, char in chars.iterrows():
        if prev_x1 is not None and abs(char['x0'] - prev_x1) > char_tol:
            if word:
                words.append(word)
            word = []
        word.append(char)
        prev_x1 = char['x1']
    if word:
        words.append(word)
    
    words_out = []
    for word_chars in words:
        if not word_chars:
            continue
        text = "".join([c["text"] for c in word_chars]).strip()
        x0 = word_chars[0]["x0"]
        y0 = min(c["y0"] for c in word_chars)
        x1 = word_chars[-1]["x1"]
        y1 = max(c["y1"] for c in word_chars)
        words_out.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return words_out

def group_words_by_logical_cells(words, x_gap_threshold, y_proximity_threshold):
    """
    Group words into logical cells by:
    1. First identifying words that are on same horizontal line (Y overlap)
    2. Then grouping vertically close text within same column area
    """
    if not words:
        return []
    
    sorted_words = sorted(words, key=lambda w: w['x0'])
    horizontal_groups = []
    
    for word in sorted_words:
        placed = False
        for group in horizontal_groups:
            if any(check_y_overlap(word, existing_word) for existing_word in group):
                group.append(word)
                placed = True
                break
        if not placed:
            horizontal_groups.append([word])
    
    logical_cells = []
    for h_group in horizontal_groups:
        if len(h_group) <= 1:
            logical_cells.extend(h_group)
            continue
            
        h_group_sorted = sorted(h_group, key=lambda w: w['x0'])
        
        columns = []
        current_column = [h_group_sorted[0]]
        
        for i in range(1, len(h_group_sorted)):
            prev_word = h_group_sorted[i-1]
            curr_word = h_group_sorted[i]
            gap = curr_word['x0'] - prev_word['x1']
            
            if gap > x_gap_threshold:
                columns.append(current_column)
                current_column = [curr_word]
            else:
                current_column.append(curr_word)
        
        if current_column:
            columns.append(current_column)
        
        for column in columns:
            if len(column) <= 1:
                logical_cells.extend(column)
            else:
                merged_cell = merge_word_group(column)
                logical_cells.append(merged_cell)
    
    x_grouped = {}
    for cell in logical_cells:
        x_key = round(cell['x0'] / 10) * 10
        if x_key not in x_grouped:
            x_grouped[x_key] = []
        x_grouped[x_key].append(cell)
    
    final_cells = []
    for x_key in sorted(x_grouped.keys()):
        # sort top-to-bottom
        column_cells = sorted(x_grouped[x_key], key=lambda c: -c['y0'])
        
        merged_column = []
        current_group = [column_cells[0]]
        
        for i in range(1, len(column_cells)):
            prev_cell = current_group[-1]
            curr_cell = column_cells[i]
            vertical_gap = curr_cell['y0'] - prev_cell['y1']
            
            if vertical_gap <= y_proximity_threshold:
                current_group.append(curr_cell)
            else:
                if len(current_group) > 1:
                    merged_cell = merge_word_group(current_group)
                    merged_column.append(merged_cell)
                else:
                    merged_column.extend(current_group)
                current_group = [curr_cell]
        
        if len(current_group) > 1:
            merged_cell = merge_word_group(current_group)
            merged_column.append(merged_cell)
        else:
            merged_column.extend(current_group)
        
        final_cells.extend(merged_column)
    
    return final_cells

def check_y_overlap(word1, word2, overlap_threshold=0):
    """Check if two words have overlapping Y coordinates"""
    y1_top, y1_bottom = word1['y0'], word1['y1']
    y2_top, y2_bottom = word2['y0'], word2['y1']
    
    overlap_start = max(y1_top, y2_top)
    overlap_end = min(y1_bottom, y2_bottom)
    overlap = max(0, overlap_end - overlap_start)
    
    min_height = min(y1_bottom - y1_top, y2_bottom - y2_top)
    
    return overlap / min_height >= overlap_threshold if min_height > 0 else False


def merge_word_group(word_group):
    """Merge a group of words into a single word object"""
    if len(word_group) == 1:
        return word_group[0]
    
    sorted_group = sorted(word_group, key=lambda w: (-w['y0'], w['x0']))
    combined_text = " ".join(word['text'] for word in sorted_group)
    
    x0 = min(word['x0'] for word in sorted_group)
    y0 = min(word['y0'] for word in sorted_group)
    x1 = max(word['x1'] for word in sorted_group)
    y1 = max(word['y1'] for word in sorted_group)
    
    return {
        "text": combined_text,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1
    }
