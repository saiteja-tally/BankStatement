from enum import StrEnum

# --- Classification thresholds (tunable) ---
CLASSIFY_MIN_INTERNAL_H_RATIO = 0.001
CLASSIFY_MIN_INTERNAL_V_RATIO = 0.001
CLASSIFY_INTERSECTION_RATIO = 0.6
CLASSIFY_COVERAGE_THRESHOLD = 0.6
CLASSIFY_OUTER_MARGIN_TOL = 0.05  # fraction of width/height to treat as outer border
CLASSIFY_INTERSECTION_TOL = 5  # in pixels (reuses same scale as other functions)
CLASSIFY_LINE_MERGE_TOL = 20  # merge lines within this pixel distance

class TableGridType(StrEnum):
    FULL_GRID = 'full_grid'
    VERTICAL_ONLY = 'vertical_only'
    HORIZONTAL_ONLY = 'horizontal_only'
    BORDER_ONLY = 'border_only'
    UNBORDERED = 'unbordered'

def _table_bbox_from_cluster(cluster):
    """
    cluster['bbox'] is already (min_x, min_y, max_x, max_y)
    Keep it for compatibility.
    """
    return cluster.get("bbox", None)

def _is_outer_line_for_bbox(line, bbox, orientation='h', margin_tol=CLASSIFY_OUTER_MARGIN_TOL):
    """
    line: dict with x0, top, x1, bottom
    bbox: (min_x, min_y, max_x, max_y)
    orientation: 'h' or 'v'
    """
    if not bbox:
        return False
    x_min, y_min, x_max, y_max = bbox
    if orientation == 'h':
        ly = (line['xyxy'][1] + line['xyxy'][3]) / 2.0
        height = max(1.0, y_max - y_min)
        top_dist = abs(ly - y_min) / height
        bottom_dist = abs(y_max - ly) / height
        return top_dist <= margin_tol or bottom_dist <= margin_tol
    else:
        lx = (line['xyxy'][0] + line['xyxy'][2]) / 2.0
        width = max(1.0, x_max - x_min)
        left_dist = abs(lx - x_min) / width
        right_dist = abs(x_max - lx) / width
        return left_dist <= margin_tol or right_dist <= margin_tol

def _line_coverage_ratio_for_bbox(line, bbox, orientation='h'):
    """
    Fraction of table width (for horizontal) or height (for vertical) covered by the line
    """
    if not bbox:
        return 0.0
    x_min, y_min, x_max, y_max = bbox
    if orientation == 'h':
        table_span = max(1.0, x_max - x_min)
        line_span = max(0.0, line['x1'] - line['x0'])
        return line_span / table_span
    else:
        table_span = max(1.0, y_max - y_min)
        line_span = max(0.0, line['bottom'] - line['top'])
        return line_span / table_span

def _merge_close_lines(lines, orientation='h', tolerance=CLASSIFY_LINE_MERGE_TOL):
    """
    Merge lines that are close to each other (within tolerance pixels).
    
    For horizontal lines: merge if their y-coordinates are close and they overlap in x-range.
    For vertical lines: merge if their x-coordinates are close and they overlap in y-range.
    
    Args:
        lines: List of line dicts with x0, top, x1, bottom
        orientation: 'h' for horizontal or 'v' for vertical
        tolerance: Maximum distance in pixels to consider lines as mergeable
        
    Returns:
        List of merged line dicts
    """
    if not lines:
        return []
    
    # Sort lines by their primary coordinate
    if orientation == 'h':
        # Sort by y-coordinate (average of top and bottom)
        lines = sorted(lines, key=lambda l: (l['xyxy'][1] + l['xyxy'][3]) / 2.0)
    else:
        # Sort by x-coordinate (average of x0 and x1)
        lines = sorted(lines, key=lambda l: (l['xyxy'][0] + l['xyxy'][2]) / 2.0)
    
    merged = []
    current_group = [lines[0]]
    
    for line in lines[1:]:
        if orientation == 'h':
            # Check if y-coordinates are close
            curr_y = (current_group[0]['xyxy'][1] + current_group[0]['xyxy'][3]) / 2.0
            line_y = (line['xyxy'][1] + line['xyxy'][3]) / 2.0
            
            if abs(line_y - curr_y) <= tolerance:
                # Check if they overlap or are close in x-range
                curr_x_min = min(l['xyxy'][0] for l in current_group)
                curr_x_max = max(l['xyxy'][2] for l in current_group)
                
                if (line['xyxy'][0] <= curr_x_max + tolerance and line['xyxy'][2] >= curr_x_min - tolerance):
                    current_group.append(line)
                else:
                    # Create merged line from current group
                    merged.append(_create_merged_line(current_group, orientation))
                    current_group = [line]
            else:
                # Create merged line from current group
                merged.append(_create_merged_line(current_group, orientation))
                current_group = [line]
        else:  # vertical
            # Check if x-coordinates are close
            curr_x = (current_group[0]['xyxy'][0] + current_group[0]['xyxy'][2]) / 2.0
            line_x = (line['xyxy'][0] + line['xyxy'][2]) / 2.0
            
            if abs(line_x - curr_x) <= tolerance:
                # Check if they overlap or are close in y-range
                curr_y_min = min(l['xyxy'][1] for l in current_group)
                curr_y_max = max(l['xyxy'][3] for l in current_group)
                
                if (line['xyxy'][1] <= curr_y_max + tolerance and line['xyxy'][3] >= curr_y_min - tolerance):
                    current_group.append(line)
                else:
                    # Create merged line from current group
                    merged.append(_create_merged_line(current_group, orientation))
                    current_group = [line]
            else:
                # Create merged line from current group
                merged.append(_create_merged_line(current_group, orientation))
                current_group = [line]
    
    # Don't forget the last group
    if current_group:
        merged.append(_create_merged_line(current_group, orientation))
    
    return merged

def _create_merged_line(lines, orientation):
    """
    Create a single line from a group of lines to be merged.
    
    Args:
        lines: List of line dicts to merge
        orientation: 'h' for horizontal or 'v' for vertical
        
    Returns:
        A single merged line dict
    """
    if len(lines) == 1:
        return lines[0].copy()
    
    merged = {
        'x0': min(l['xyxy'][0] for l in lines),
        'x1': max(l['xyxy'][2] for l in lines),
        'top': min(l['xyxy'][1] for l in lines),
        'bottom': max(l['xyxy'][3] for l in lines),
        'type': orientation
    }
    
    # Preserve other properties from the first line if they exist
    for key in lines[0]:
        if key not in merged:
            merged[key] = lines[0][key]
    
    return merged

def _count_hv_intersections(hlines, vlines, tol=CLASSIFY_INTERSECTION_TOL):
    """
    Count intersections between hlines and vlines where they geometrically cross (with tol).
    Also return unique count of distinct h and v coordinate lines.
    """
    intersections = 0
    unique_h_ys = set()
    unique_v_xs = set()
    for h in hlines:
        hy = round((h['top'] + h['bottom']) / 2.0, 1)
        unique_h_ys.add(hy)
    for v in vlines:
        vx = round((v['x0'] + v['x1']) / 2.0, 1)
        unique_v_xs.add(vx)

    for h in hlines:
        for v in vlines:
            # check overlap: v.x in h.x-range and h.y in v.y-range (with tol)
            if (v['x0'] - tol) <= h['x1'] and (v['x1'] + tol) >= h['x0'] and (h['top'] >= v['top'] - tol and h['top'] <= v['bottom'] + tol):
                intersections += 1

    return intersections, len(unique_h_ys), len(unique_v_xs)

def classify_cluster_table(cluster, elements=None, cfg=None):
    """
    Returns (classification_str, score_dict) where classification_str is
    one of TableGridType.

    cluster is a processed cluster from cluster_lines() or a validated table dict
    with keys 'lines' (list of line dicts) and 'bbox' (min_x, min_y, max_x, max_y).
    """
    if cfg is None:
        cfg = {}
    min_internal_h = cfg.get("min_internal_h", CLASSIFY_MIN_INTERNAL_H_RATIO)
    min_internal_v = cfg.get("min_internal_v", CLASSIFY_MIN_INTERNAL_V_RATIO)
    inter_ratio_thresh = cfg.get("inter_ratio_thresh", CLASSIFY_INTERSECTION_RATIO)
    coverage_thresh = cfg.get("coverage_thresh", CLASSIFY_COVERAGE_THRESHOLD)
    outer_margin_tol = cfg.get("outer_margin_tol", CLASSIFY_OUTER_MARGIN_TOL)

    bbox = _table_bbox_from_cluster(cluster)
    if bbox is None:
        return TableGridType.UNBORDERED, {"reason": "no_bbox"}

    lines = cluster.get("lines", [])
    hlines = [l for l in lines if l.get("type") == 'h' ]
    vlines = [l for l in lines if l.get("type") == 'v' ]

    # Merge close lines before classification
    merge_tol = cfg.get("line_merge_tol", CLASSIFY_LINE_MERGE_TOL)
    hlines = _merge_close_lines(hlines, orientation='h', tolerance=merge_tol)
    vlines = _merge_close_lines(vlines, orientation='v', tolerance=merge_tol)

    # separate outer vs internal
    internal_h = []
    outer_h = []
    for h in hlines:
        if _is_outer_line_for_bbox(h, bbox, orientation='h', margin_tol=outer_margin_tol):
            outer_h.append(h)
        else:
            internal_h.append(h)

    internal_v = []
    outer_v = []
    for v in vlines:
        if _is_outer_line_for_bbox(v, bbox, orientation='v', margin_tol=outer_margin_tol):
            outer_v.append(v)
        else:
            internal_v.append(v)

    # count long internal lines (cover substantial table width/height)
    long_internal_h = [h for h in internal_h if _line_coverage_ratio_for_bbox(h, bbox, 'h') >= coverage_thresh]
    long_internal_v = [v for v in internal_v if _line_coverage_ratio_for_bbox(v, bbox, 'v') >= coverage_thresh]

    intersections, unique_h_count, unique_v_count = _count_hv_intersections(long_internal_h or internal_h, long_internal_v or internal_v)

    expected = max(1, unique_h_count) * max(1, unique_v_count)
    inter_ratio = intersections / expected if expected > 0 else 0.0

    score = {
        "total_h": len(hlines),
        "total_v": len(vlines),
        "internal_h": len(internal_h),
        "internal_v": len(internal_v),
        "long_internal_h": len(long_internal_h),
        "long_internal_v": len(long_internal_v),
        "outer_h": len(outer_h),
        "outer_v": len(outer_v),
        "intersections": intersections,
        "unique_h": unique_h_count,
        "unique_v": unique_v_count,
        "intersection_ratio": inter_ratio,
    }

    # Decide
    if unique_h_count >= min_internal_h and unique_v_count >= min_internal_v and inter_ratio >= inter_ratio_thresh:
        return TableGridType.FULL_GRID, score

    if len(internal_v) >= max(min_internal_v, 2) and len(internal_h) <= 1:
        return TableGridType.VERTICAL_ONLY, score

    if len(internal_h) >= max(min_internal_h, 2) and len(internal_v) <= 1:
        return TableGridType.HORIZONTAL_ONLY, score

    if len(internal_h) == 0 and len(internal_v) == 0 and (outer_h or outer_v):
        return TableGridType.BORDER_ONLY, score

    # fallback: try to infer from element alignment if provided
    if elements:
        # cluster x-centers and y-centers as a heuristic
        xs, ys = [], []
        for el in elements:
            bx = el.get("bounding_box") or el.get("box")
            if not bx:
                continue
            cx, cy = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0) if not isinstance(bx[0], list) else (sum(p[0] for p in bx)/len(bx), sum(p[1] for p in bx)/len(bx))
            xs.append(cx); ys.append(cy)

        def cluster_count(vals, tol):
            if not vals:
                return 0
            vals = sorted(vals)
            clusters = 1
            last = vals[0]
            for v in vals[1:]:
                if abs(v - last) > tol:
                    clusters += 1
                    last = v
            return clusters

        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        x_tol = max(8, width * 0.05)
        y_tol = max(8, height * 0.03)
        x_clusters = cluster_count(xs, x_tol)
        y_clusters = cluster_count(ys, y_tol)

        if x_clusters >= 3 and y_clusters <= 4:
            return TableGridType.VERTICAL_ONLY, score
        if y_clusters >= 3 and x_clusters <= 4:
            return TableGridType.HORIZONTAL_ONLY, score

    return TableGridType.UNBORDERED, score