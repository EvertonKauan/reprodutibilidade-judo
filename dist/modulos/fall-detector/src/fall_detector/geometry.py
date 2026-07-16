import numpy as np

def euclidean_distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def is_bbox_horizontal(x1, y1, x2, y2, factor: float = 1.25) -> bool:
    return (x2 - x1) > (y2 - y1) * factor

def bbox_intersect(box_a, box_b) -> bool:
    x1, y1, x2, y2 = box_a
    X1, Y1, W, H = box_b
    X2, Y2 = X1 + W, Y1 + H
    return not (x2 < X1 or x1 > X2 or y2 < Y1 or y1 > Y2)

def format_seconds_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
