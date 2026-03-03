"""
raster_processor.py — Improved inpainting for raster/hybrid PDF pages.

Key design goals:
  A) mask_regions (word/line OCR polygons) - separate from translation containers.
  B) Glyph-accurate ROI mask via adaptive thresholding (not rectangle fills).
  C) Adaptive per-ROI dilation (not a blind global fixed kernel).
  D) Edge protection: Canny edge map subtracted from mask before inpainting.
"""
from __future__ import annotations

import fitz
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

from services.layout.containers import ContainerRef
from services.layout.geometry import (
    poly_to_bbox as _poly_to_bbox,
    scale_bbox as _scale_bbox,
    scale_poly as _scale_poly,
    union_bbox as _union_bbox,
    bbox_overlap_area as _bbox_overlap_area,
    bbox_area as _bbox_area,
)


# ---------------------------------------------------------------------------
# A) MaskRegion — the OCR primitive used only for mask construction
# ---------------------------------------------------------------------------

@dataclass
class MaskRegion:
    """
    One word- or line-level OCR region used to build the inpaint mask.
    Coordinates are in the same fitz-point space as ContainerRef.bbox
    (i.e. already scaled from DocInt units).

    polygon  – preferred; list of (x, y) corners (can be 4-point or N-point).\n    bbox     – fallback axis-aligned box (x0, y0, x1, y1) in fitz points.
    confidence – DocInt word confidence [0..1]; regions below a threshold can
                 be skipped (defaults to 1.0 = always use).
    """
    bbox:       Tuple[float, float, float, float]
    polygon:    Optional[List[Tuple[float, float]]] = None
    confidence: float = 1.0
    page_index: int = 0


def build_mask_regions_from_analyze_result(
    doc: fitz.Document,
    analyze_result,
    *,
    min_confidence: float = 0.3,
    verbose: bool = False,
) -> dict[int, List[MaskRegion]]:
    """
    Extract word-level MaskRegions from an Azure Document Intelligence
    analyze_result.  Returns a dict keyed by page_index.

    Words give the finest granularity for masking — only the actual glyph
    clusters are targeted, not whole table cells or paragraph bboxes.
    """
    result: dict[int, List[MaskRegion]] = {}

    pages = getattr(analyze_result, "pages", None) or []
    for page_obj in pages:
        pno = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
        if not pno:
            continue
        pi = int(pno) - 1
        if pi < 0 or pi >= doc.page_count:
            continue

        fitz_page = doc[pi]
        fw = float(fitz_page.rect.width)
        fh = float(fitz_page.rect.height)
        dw = float(getattr(page_obj, "width", 0.0) or 0.0)
        dh = float(getattr(page_obj, "height", 0.0) or 0.0)
        sx = fw / dw if dw > 0 else 72.0
        sy = fh / dh if dh > 0 else 72.0

        regions: List[MaskRegion] = []
        words = getattr(page_obj, "words", None) or []
        if verbose:
            print(f"[MASK_REGION] page {pi}: {len(words)} words from DocInt")

        for w in words:
            conf = float(getattr(w, "confidence", 1.0) or 1.0)
            if conf < min_confidence:
                continue
            poly_raw = getattr(w, "polygon", None)
            bb = _poly_to_bbox(poly_raw)
            if not bb:
                continue
            bb_scaled = _scale_bbox(bb, sx, sy)
            poly_scaled = None
            if poly_raw:
                try:
                    if isinstance(poly_raw, list) and poly_raw and isinstance(poly_raw[0], (int, float)):
                        pts = [(poly_raw[i] * sx, poly_raw[i+1] * sy)
                               for i in range(0, len(poly_raw), 2)]
                    else:
                        pts = [(float(getattr(p, "x", 0)) * sx,
                                float(getattr(p, "y", 0)) * sy)
                               for p in poly_raw]
                    poly_scaled = pts
                except Exception:
                    poly_scaled = None

            regions.append(MaskRegion(
                bbox=bb_scaled,
                polygon=poly_scaled,
                confidence=conf,
                page_index=pi,
            ))

        result[pi] = regions

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_page_to_numpy(pix: fitz.Pixmap) -> np.ndarray:
    """fitz.Pixmap (RGB, no alpha) -> HxWx3 uint8 RGB numpy array."""
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        (pix.height, pix.width, 3)
    )
    return arr.copy()  # writable copy


def _numpy_to_pixmap(arr_rgb: np.ndarray) -> fitz.Pixmap:
    """
    HxWx3 uint8 RGB numpy array -> fitz.Pixmap.
    Uses PNG round-trip — safe across ALL PyMuPDF versions.
    """
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode: PNG encode failed")
    return fitz.Pixmap(buf.tobytes())


def _roi_text_mask(
    roi_gray: np.ndarray,
    *,
    block_size: int = 25,
    c_offset: int = 10,
    min_cc_area: int = 4,
) -> np.ndarray:
    """
    B) Glyph-accurate mask inside an ROI via adaptive thresholding.

    Returns a uint8 binary mask (0=background, 255=text-pixel) the same
    size as roi_gray.

    Steps:
      1. Adaptive threshold (Gaussian) to catch varied lighting.
      2. Morphological open (remove specks) + close (fill glyph gaps).
      3. Connected-component filter — remove tiny noise components.
    """
    if roi_gray.size == 0:
        return np.zeros_like(roi_gray, dtype=np.uint8)

    # ensure odd block_size >= 3
    bs = max(3, block_size | 1)

    # 1. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        roi_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        c_offset,
    )

    # 2. Small morphological clean-up
    k_open  = np.ones((2, 2), np.uint8)
    k_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # 3. Remove tiny connected components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for lbl in range(1, num_labels):  # skip background (0)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cc_area:
            clean[labels == lbl] = 255

    return clean


def _adaptive_dilation(
    roi_mask: np.ndarray,
    *,
    base_px: int = 1,
    max_px: int = 4,
) -> np.ndarray:
    """
    C) Per-ROI adaptive dilation.

    Heuristic: if the mask is 'thin' (mean mask density low), use a larger
    kernel to cover anti-aliasing halos; otherwise keep it small.
    Capped at max_px to avoid eating nearby geometry.
    """
    if roi_mask.sum() == 0:
        return roi_mask

    density = roi_mask.sum() / (roi_mask.size * 255.0 + 1e-6)
    # thin mask (<5% of ROI area = likely fine strokes with halos)
    dil_px = max_px if density < 0.05 else base_px
    dil_px = max(1, min(dil_px, max_px))

    k = np.ones((dil_px * 2 + 1, dil_px * 2 + 1), np.uint8)
    return cv2.dilate(roi_mask, k, iterations=1)


def _edge_protected_mask(
    full_mask: np.ndarray,
    img_gray: np.ndarray,
    *,
    canny_lo: int = 30,
    canny_hi: int = 100,
    edge_dil_px: int = 2,
) -> np.ndarray:
    """
    D) Edge protection: subtract a dilated Canny edge map from the mask.
    If subtracting removes >80% of any local region, fall back to a 1-px buffer.
    """
    edges = cv2.Canny(img_gray, canny_lo, canny_hi)
    k = np.ones((edge_dil_px * 2 + 1, edge_dil_px * 2 + 1), np.uint8)
    edge_buf = cv2.dilate(edges, k, iterations=1)

    protected = full_mask.copy()
    protected[edge_buf > 0] = 0

    # Safety: if protection removed too much globally (>70%), use just 1-px buffer
    orig_area  = float(full_mask.sum())
    prot_area  = float(protected.sum())
    if orig_area > 0 and (prot_area / orig_area) < 0.30:
        k1 = np.ones((3, 3), np.uint8)
        edge_buf1 = cv2.dilate(edges, k1, iterations=1)
        protected = full_mask.copy()
        protected[edge_buf1 > 0] = 0

    return protected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_page_to_pixmap(page: fitz.Page, dpi: int = 300) -> fitz.Pixmap:
    """11.1 Render page to raster at specified DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    return page.get_pixmap(matrix=mat, alpha=False)


def inpaint_containers(
    pixmap: fitz.Pixmap,
    containers: List[ContainerRef],
    *,
    sx: float,
    sy: float,
    mask_regions: Optional[List[MaskRegion]] = None,
    verbose: bool = False,
) -> fitz.Pixmap:
    """
    Inpaint raster regions on a page pixmap.

    Parameters
    ----------
    pixmap      : full-page fitz.Pixmap (RGB, no alpha) at render DPI.
    containers  : translation containers (used ONLY for routing decisions).
    sx, sy      : scale factors (pixmap pixels per fitz point).
    mask_regions: word/line-level MaskRegion list from DocInt.
                  If provided, the inpaint mask is built glyph-accurately from
                  these primitives rather than from container bboxes.
                  If None, falls back to container-bbox fill (legacy behaviour).
    verbose     : print diagnostic info.
    """
    if cv2 is None:
        if verbose:
            print("[RASTER][WARN] OpenCV not available; skipping inpainting.")
        return pixmap

    try:
        img_rgb = _render_page_to_numpy(pixmap)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        H, W = img_rgb.shape[:2]
        full_mask = np.zeros((H, W), dtype=np.uint8)

        # ----------------------------------------------------------------
        # Build mask
        # ----------------------------------------------------------------
        if mask_regions:
            # A + B + C: word-level glyph-accurate mask with adaptive dilation
            if verbose:
                print(f"[RASTER] Building glyph-accurate mask from {len(mask_regions)} OCR regions")

            for mr in mask_regions:
                # --- ROI bounds (in pixmap pixels) ---
                bb = mr.bbox
                rx0 = max(0, int(bb[0] * sx) - 2)
                ry0 = max(0, int(bb[1] * sy) - 2)
                rx1 = min(W, int(bb[2] * sx) + 2)
                ry1 = min(H, int(bb[3] * sy) + 2)
                if rx1 <= rx0 or ry1 <= ry0:
                    continue

                roi_gray = img_gray[ry0:ry1, rx0:rx1]

                # B) glyph-accurate text mask inside ROI
                roi_mask = _roi_text_mask(roi_gray)

                # Refine to polygon shape if available
                if mr.polygon and len(mr.polygon) >= 3:
                    poly_px = np.array(
                        [(int(p[0] * sx) - rx0, int(p[1] * sy) - ry0)
                         for p in mr.polygon],
                        dtype=np.int32
                    )
                    poly_canvas = np.zeros_like(roi_mask)
                    cv2.fillPoly(poly_canvas, [poly_px], 255)
                    roi_mask = cv2.bitwise_and(roi_mask, poly_canvas)

                # C) adaptive dilation per ROI
                roi_mask = _adaptive_dilation(roi_mask)

                # Paste back into full mask
                full_mask[ry0:ry1, rx0:rx1] = cv2.bitwise_or(
                    full_mask[ry0:ry1, rx0:rx1], roi_mask
                )

        else:
            # Legacy fallback: rectangle fill from container bboxes
            if verbose:
                print(f"[RASTER] No mask_regions; falling back to container-bbox fill")
            for c in containers:
                if c.original_spans:  # vector container — skip
                    continue
                if c.polygon:
                    pts = np.array(
                        [(int(p[0] * sx), int(p[1] * sy)) for p in c.polygon],
                        dtype=np.int32
                    )
                    cv2.fillPoly(full_mask, [pts], 255)
                else:
                    r = c.bbox
                    x0 = max(0, int(r[0] * sx))
                    y0 = max(0, int(r[1] * sy))
                    x1 = min(W, int(r[2] * sx))
                    y1 = min(H, int(r[3] * sy))
                    if x1 > x0 and y1 > y0:
                        cv2.rectangle(full_mask, (x0, y0), (x1, y1), 255, -1)

            # Legacy: single global dilation (small)
            k = np.ones((3, 3), np.uint8)
            full_mask = cv2.dilate(full_mask, k, iterations=1)

        if full_mask.max() == 0:
            if verbose:
                print("[RASTER] Empty mask after construction — nothing to inpaint.")
            return pixmap

        # ----------------------------------------------------------------
        # D) Edge protection
        # ----------------------------------------------------------------
        if mask_regions:
            full_mask = _edge_protected_mask(full_mask, img_gray)
            if verbose:
                mask_px   = int(full_mask.sum() // 255)
                print(f"[RASTER] Mask after edge protection: {mask_px} pixels")

        # ----------------------------------------------------------------
        # 11.4 Inpaint (Telea)
        # ----------------------------------------------------------------
        res_bgr = cv2.inpaint(img_bgr, full_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

        return _numpy_to_pixmap(res_rgb)

    except Exception as e:
        if verbose:
            print(f"[RASTER][WARN] Inpainting failed: {e!r}. Returning original pixmap.")
        return pixmap


def draw_raster_overlay(doc: fitz.Document, page_index: int, pixmap: fitz.Pixmap) -> None:
    """
    11.5 / 11.6: Place the inpainted raster as a full-page image on the page.
    Translated vector text is overlaid on top of this layer afterwards.
    """
    page = doc[page_index]
    img_rect = fitz.Rect(page.rect)  # defensive cast in case page.rect is a tuple
    page.insert_image(img_rect, pixmap=pixmap)
