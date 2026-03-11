import fitz, re
from typing import List, Dict, Tuple, Any, Optional

from scripts.layout.containers import ContainerRef, ContainerKind, PdfSpanAttrs
from scripts.layout.geometry import (
    poly_to_bbox as _poly_to_bbox,
    scale_bbox as _scale_bbox,
    scale_poly as _scale_poly,
    union_bbox as _union_bbox,
    bbox_overlap_area as _bbox_overlap_area,
    bbox_area as _bbox_area,
)


def _is_bullet(text: str) -> bool:
    return bool(re.match(r'^(\s*[•\-\*>]|(\d+[.)])|([a-z][.)]))', text.strip(), re.IGNORECASE))


def build_containers(doc: fitz.Document, analyze_result, verbose: bool = False) -> List[ContainerRef]:
    """
    Extract one ContainerRef per DocInt line (or table cell).

    KEY DESIGN:
        - Each container gets the EXACT bbox of its original DocInt line/word cluster.
        This ensures translated text is placed back at precisely the right position.
        - Lines that belong to the same paragraph group share a `paragraph_group_id`
        (e.g. "p2_3" = page 2, group 3). The translation stage uses this to build
        a LLM context window for grammar quality, without merging bboxes.
    """
    containers: List[ContainerRef] = []

    if not getattr(analyze_result, "pages", None):
        return containers

    # -------------------------
    # 1. page scale factors (DocInt units -> fitz points)
    # -------------------------
    page_scale: Dict[int, Tuple[float, float]] = {}
    for page_obj in analyze_result.pages:
        pno = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
        if not pno: continue
        page_index = int(pno) - 1
        if page_index < 0 or page_index >= doc.page_count: continue

        fitz_page = doc[page_index]
        fw, fh = float(fitz_page.rect.width), float(fitz_page.rect.height)
        dw = float(getattr(page_obj, "width", 0.0) or 0.0)
        dh = float(getattr(page_obj, "height", 0.0) or 0.0)
        sx = fw / dw if dw > 0 else 72.0
        sy = fh / dh if dh > 0 else 72.0
        page_scale[page_index] = (sx, sy)

    # -------------------------
    # 2. table cells (each cell = one container, no paragraph_group_id needed)
    # -------------------------
    table_bboxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    all_tables = getattr(analyze_result, "tables", None) or []
    for table in all_tables:
        cells = getattr(table, "cells", None) or []
        for cell in cells:
            text = (getattr(cell, "content", "") or "").strip()
            if not text:
                continue

            brs = getattr(cell, "bounding_regions", None) or []
            cell_box_list = []
            cell_page_index = None

            for br in brs:
                pno = getattr(br, "page_number", None) or getattr(br, "pageNumber", None)
                if not pno: continue
                pi = int(pno) - 1
                if pi < 0 or pi >= doc.page_count: continue
                poly = getattr(br, "polygon", None)
                bb = _poly_to_bbox(poly)
                if not bb: continue
                sx, sy = page_scale.get(pi, (72.0, 72.0))
                cell_box_list.append(_scale_bbox(bb, sx, sy))
                cell_page_index = pi

            if cell_page_index is None or not cell_box_list: continue

            bbox = _union_bbox(cell_box_list)
            if _bbox_area(bbox) < 2.0: continue

            # track table geometry per page so body line extraction can skip overlap
            table_bboxes.setdefault(cell_page_index, []).append(bbox)

            style_hints = {
                "row_index": getattr(cell, "row_index", 0),
                "column_index": getattr(cell, "column_index", 0),
            }

            containers.append(ContainerRef(
                page_index=cell_page_index,
                bbox=bbox,
                text=text,
                kind=ContainerKind.TABLE_CELL,
                style_hints=style_hints,
                paragraph_group_id=None, # table cells are self-contained
            ))

    # -------------------------
    # 3. body lines: one ContainerRef per line, grouped for context
    # -------------------------
    global_group_counter = 0 # incremented for every new paragraph group across all pages

    for page_obj in analyze_result.pages:
        pno = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
        if not pno: continue
        page_index = int(pno) - 1
        if page_index < 0 or page_index >= doc.page_count: continue

        sx, sy = page_scale.get(page_index, (72.0, 72.0))
        page_height = float(doc[page_index].rect.height)
        page_width = float(doc[page_index].rect.width)
        tb_boxes = table_bboxes.get(page_index, [])

        lines = getattr(page_obj, "lines", None) or []

        # pre-process lines: scale, skip table overlaps, record bbox+poly
        page_lines = []
        for ln in lines:
            txt = (getattr(ln, "content", "") or "").strip()
            if not txt: continue
            bb = _poly_to_bbox(getattr(ln, "polygon", None))
            if not bb: continue
            bb = _scale_bbox(bb, sx, sy)
            if _bbox_area(bb) < 2.0: continue

            # skip lines that overlap table cells (>40% of line area)
            l_area = _bbox_area(bb)
            overlap = False
            for tb in tb_boxes:
                if _bbox_overlap_area(bb, tb) > 0.4 * l_area:
                    overlap = True
                    break
            if overlap: continue

            page_lines.append({
                "text": txt,
                "bbox": bb,
                "poly": _scale_poly(getattr(ln, "polygon", None), sx, sy),
            })

        # separate headers/footers (top 8% or bottom 8% of page)
        header_footer_lines = []
        body_lines = []
        for ln in page_lines:
            bb = ln["bbox"]
            mid_y = (bb[1] + bb[3]) / 2.0
            if mid_y < page_height * 0.08 or mid_y > page_height * 0.92: header_footer_lines.append(ln)
            else: body_lines.append(ln)

        # emit header/footer lines: each gets its own container, no paragraph group
        for ln in header_footer_lines:
            containers.append(ContainerRef(
                page_index=page_index,
                bbox=ln["bbox"],
                text=ln["text"],
                kind=ContainerKind.HEADER_FOOTER,
                polygon=ln.get("poly"),
                paragraph_group_id=None,
            ))

        # sort body lines: top-to-bottom, left-to-right
        body_lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # -------------------------------------------------------
        # GROUP body lines into paragraph groups for context only.
        # Grouping criteria (same as before):
        #   - vertical gap < 1.5× line height
        #   - meaningful horizontal overlap OR close left edge
        # Each LINE still gets its own bbox — we just share the group ID.
        # -------------------------------------------------------
        groups: List[List[Dict]] = []
        current_group: List[Dict] = []

        for ln in body_lines:
            if not current_group:
                current_group.append(ln)
                continue

            prev = current_group[-1]
            bb, prev_bb = ln["bbox"], prev["bbox"]
            lh = bb[3] - bb[1]
            prev_lh = prev_bb[3] - prev_bb[1]
            y_diff = bb[1] - prev_bb[3]                                     # gap between lines
            x_diff = abs(bb[0] - prev_bb[0])                                # left-edge distance
            x_overlap = min(bb[2], prev_bb[2]) - max(bb[0], prev_bb[0])     # horizontal overlap

            if y_diff < max(lh, prev_lh) * 1.5 and (x_overlap > 0 or x_diff < 15):
                current_group.append(ln)
            else:
                groups.append(current_group)
                current_group = [ln]

        if current_group:
            groups.append(current_group)

        # column detection for reading order (3 vertical bands)
        def get_col_band(x0: float, pw: float) -> int:
            if x0 < pw * 0.35: return 0
            if x0 < pw * 0.65: return 1
            return 2

        # flatten groups back to individual lines while assigning group IDs
        # sort groups by column band then top y for a sane reading order
        groups.sort(key=lambda g: (
            get_col_band(g[0]["bbox"][0], page_width),
            g[0]["bbox"][1],
        ))

        reading_key = 0
        for g in groups:
            global_group_counter += 1
            group_id = f"p{page_index}_{global_group_counter}"

            # determine group-level kind for kind-propagation to each line
            group_text = " ".join(ln["text"] for ln in g)
            group_bbox = _union_bbox([ln["bbox"] for ln in g])

            if len(g) == 1 and len(group_text) < 20 and _bbox_area(group_bbox) < (page_width * page_height * 0.05): group_kind = ContainerKind.LABEL
            elif _is_bullet(group_text): group_kind = ContainerKind.LIST_ITEM
            else: group_kind = ContainerKind.PARAGRAPH

            for ln in g:
                # 10) layout preservation - Line bboxes
                # expand slightly (padding) horizontally to avoid wrapping or unnecessary font shrinkage from tight precision gaps
                bb = list(ln["bbox"])
                bb[0] = max(0, bb[0] - 1.0) # -1pt left
                bb[2] = min(page_width, bb[2] + 1.0) # +1pt right
                final_bbox = tuple(bb)

                containers.append(ContainerRef(
                    page_index=page_index,
                    bbox=final_bbox,
                    text=ln["text"],
                    kind=group_kind,
                    polygon=ln.get("poly"),
                    reading_key=reading_key,
                    paragraph_group_id=group_id,
                ))
                reading_key += 1

    # -------------------------
    # 4. final global sort: page -> header/footer first -> reading order
    # -------------------------
    containers.sort(key=lambda c: (
        c.page_index,
        0 if c.kind == ContainerKind.HEADER_FOOTER else 1,
        c.reading_key,
        c.bbox[1],
        c.bbox[0],
    ))

    # -------------------------
    # 5. span mapping (for precise text removal)
    # -------------------------
    map_spans_to_containers(doc, containers, verbose=verbose)
    return containers


def map_spans_to_containers(doc: fitz.Document, containers: List[ContainerRef], verbose: bool = False):
    """
    Span-targeted removal.
    Maps actual PDF text spans from PyMuPDF back to DocInt containers.
    Because containers now have tight per-line bboxes, span hits are more accurate.
    """
    for pi in range(len(doc)):
        page = doc[pi]
        page_containers = [c for c in containers if c.page_index == pi]
        if not page_containers:
            continue

        raw = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
        for block in raw.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    s_bbox = span["bbox"]
                    s_area = _bbox_area(s_bbox)
                    if s_area < 0.1:
                        continue

                    # assign span to the container with the most overlap
                    best_c = None
                    best_overlap = 0.3 * s_area # Lowered from 0.5 to be more robust
                    for c in page_containers:
                        overlap = _bbox_overlap_area(s_bbox, c.bbox)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_c = c

                    if best_c is not None:
                        best_c.original_spans.append(PdfSpanAttrs(
                            rect=s_bbox,
                            text=span.get("text", ""),
                            font=span.get("font", ""),
                            size=span.get("size", 10.0),
                            color=span.get("color", 0),
                            origin=span.get("origin", (0, 0)),
                            flags=span.get("flags", 0),
                            ascender=span.get("ascender", 0.0),
                            descender=span.get("descender", 0.0),
                        ))
