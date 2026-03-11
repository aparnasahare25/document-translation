import fitz
from typing import List, Tuple, Any, Optional

def poly_to_bbox(poly) -> Optional[Tuple[float, float, float, float]]:
    """Convert DocInt polygon (list or object) to axis-aligned bbox."""
    if not poly: return None
    if isinstance(poly, list) and poly and isinstance(poly[0], (int, float)):
        if len(poly) < 8: return None
        xs, ys = poly[0::2], poly[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    
    xs, ys = [], []
    try:
        if isinstance(poly, list):
            for p in poly:
                if isinstance(p, dict):
                    xs.append(float(p.get("x")))
                    ys.append(float(p.get("y")))
                else:
                    xs.append(float(getattr(p, "x")))
                    ys.append(float(getattr(p, "y")))
        else: # nested object
            for p in poly:
                xs.append(float(getattr(p, "x")))
                ys.append(float(getattr(p, "y")))
        if not xs or not ys: return None
        return (min(xs), min(ys), max(xs), max(ys))
    except (AttributeError, TypeError, ValueError):
        return None

def scale_bbox(bbox: Tuple[float, float, float, float], sx: float, sy: float) -> Tuple[float, float, float, float]:
    """Scale bbox coordinates."""
    x0, y0, x1, y1 = bbox
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

def scale_poly(poly: Any, sx: float, sy: float) -> Optional[List[Tuple[float, float]]]:
    """Scale polygon points."""
    if not poly: return None
    try:
        points = []
        if isinstance(poly, list) and len(poly) > 0 and isinstance(poly[0], (int, float)):
            for i in range(0, len(poly), 2):
                points.append((poly[i]*sx, poly[i+1]*sy))
        else:
            for p in poly:
                if isinstance(p, dict):
                    points.append((p['x']*sx, p['y']*sy))
                else:
                    points.append((getattr(p, 'x', 0)*sx, getattr(p, 'y', 0)*sy))
        return points
    except Exception:
        return None

def union_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Return the union of multiple bboxes."""
    if not bboxes: return (0, 0, 0, 0)
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)

def bbox_overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Compute intersection area of two bboxes."""
    dx = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return dx * dy

def bbox_area(a: Tuple[float, float, float, float]) -> float:
    """Compute area of a bbox."""
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])
