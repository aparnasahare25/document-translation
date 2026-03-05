import fitz
import re
from typing import List, Optional

from scripts.layout.containers import TranslationPlan, ContainerKind

def _looks_cjk(text: str) -> bool:
    for ch in text:
        if ('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u309f' or 
            '\u30a0' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af'):
            return True
    return False

def _looks_rtl(text: str) -> bool:
    """Detect if text contains RTL characters (Hebrew, Arabic, etc.)"""
    for ch in text:
        if ('\u0590' <= ch <= '\u05FF' or  # Hebrew
            '\u0600' <= ch <= '\u06FF' or  # Arabic
            '\u0750' <= ch <= '\u077F' or  # Arabic Supplement
            '\u08A0' <= ch <= '\u08FF' or  # Arabic Extended-A
            '\uFB50' <= ch <= '\uFDFF' or  # Arabic Presentation Forms-A
            '\uFE70' <= ch <= '\uFEFF'):    # Arabic Presentation Forms-B
            return True
    return False

def _bidirectional_shape(text: str) -> str:
    """
    Placeholder for shaping and BIDI reordering.
    In a full implementation, use 'arabic_reshaper' and 'bidi.algorithm'.
    """
    # return bidi_algorithm.get_display(reshaper.reshape(text))
    return text

def _wrap_text_to_width(text: str, font: fitz.Font, fontsize: float, max_width: float, cjk_mode: bool) -> List[str]:
    lines = []
    def w(s: str) -> float:
        return float(font.text_length(s, fontsize=fontsize))
        
    if not text:
        return []

    # 8.2 RTL scripts: directionality hook
    if _looks_rtl(text):
        text = _bidirectional_shape(text)
        
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
            
        if cjk_mode:
            # 8.2 CJK: character-based wrapping with ASCII-run preservation
            cur = ""
            ascii_buf = ""
            for ch in para:
                # Basic ASCII run preservation (A-Z, a-z, 0-9)
                if ch.isalnum() and ord(ch) < 128:
                    ascii_buf += ch
                    continue
                else:
                    if ascii_buf:
                        cand = cur + ascii_buf
                        if not cur or w(cand) <= max_width:
                            cur = cand
                        else:
                            lines.append(cur.rstrip())
                            cur = ascii_buf
                        ascii_buf = ""
                    
                    if not ch.strip() and not cur: 
                        continue # Skip leading spaces in CJK wrap

                    cand = cur + ch
                    if not cur or w(cand) <= max_width:
                        cur = cand
                    else:
                        lines.append(cur.rstrip())
                        cur = ch
                        
            if ascii_buf:
                cand = cur + ascii_buf
                if not cur or w(cand) <= max_width:
                    cur = cand
                else:
                    lines.append(cur.rstrip())
                    cur = ascii_buf
                    
            if cur:
                lines.append(cur.rstrip())
        else:
            # 8.2 Space-delimited: word wrapping with long-token fallback
            words = re.split(r"(\s+)", para)
            cur = ""
            for tok in words:
                if tok == "": continue
                if not cur and tok.isspace(): continue
                cand = cur + tok
                if not cur or w(cand) <= max_width:
                    cur = cand
                else:
                    # long-token fallback: break word if it exceeds max_width
                    if w(tok) > max_width:
                        if cur:
                            lines.append(cur.rstrip())
                            cur = ""
                        buf = ""
                        for ch in tok:
                            cand2 = buf + ch
                            if not buf or w(cand2) <= max_width:
                                buf = cand2
                            else:
                                lines.append(buf.rstrip())
                                buf = ch
                        cur = buf
                    else:
                        lines.append(cur.rstrip())
                        cur = tok.lstrip() if tok.isspace() else tok
            if cur:
                lines.append(cur.rstrip())
                
    while lines and lines[-1] == "":
        lines.pop()
    return lines

def _truncate_lines_to_height(lines: List[str], font: fitz.Font, fontsize: float, max_width: float, max_lines: int) -> List[str]:
    if max_lines <= 0: return []
    if len(lines) <= max_lines: return lines
    out = lines[:max_lines]
    last = out[-1]
    ell = "…"
    while last and float(font.text_length(last + ell, fontsize=fontsize)) > max_width:
        last = last[:-1]
    out[-1] = (last + ell) if last else ell
    return out

def _int_to_rgb(color_int: int):
    """Convert integer color to (r, g, b) floats."""
    r = ((color_int >> 16) & 255) / 255.0
    g = ((color_int >> 8) & 255) / 255.0
    b = (color_int & 255) / 255.0
    return (r, g, b)


def typeset_and_insert_spans(
    page: fitz.Page,
    plan: TranslationPlan,
    font_map: dict,  # dict of fontname -> fitz.Font for measuring
) -> bool:
    """
    Span-level origin-based text insertion for precise layout fidelity.

    Uses ``original_spans`` to place translated text at the exact original
    baseline origin with the original font size and color.  Falls back to
    progressive font shrinking when the translated text is wider than the
    available space (computed from the container bbox).

    For multi-span containers (e.g. table cells with several visual lines)
    the text is distributed across the span origins so each visual row is
    preserved.
    """
    text = plan.final_rendered_text.strip()
    if not text:
        return False

    spans = plan.container.original_spans
    if not spans:
        return False  # caller should use textbox fallback

    intent = plan.rendering_intent
    fontname = intent.font_name
    mf = font_map.get(fontname)
    if not mf:
        mf = fitz.Font("helv")

    bbox = plan.container.bbox
    available_w = max(1.0, bbox[2] - bbox[0] - 0.4)

    # ── Cluster spans into visual lines by y-proximity ──
    # Spans on the same baseline (within half the font size) belong to
    # the same visual line.
    sorted_spans = sorted(spans, key=lambda s: (s.origin[1], s.origin[0]))
    visual_lines: list = []  # list of lists of PdfSpanAttrs
    cur_line: list = []
    cur_y: Optional[float] = None

    for sp in sorted_spans:
        oy = sp.origin[1]
        if cur_y is None or abs(oy - cur_y) < sp.size * 0.6:
            cur_line.append(sp)
            cur_y = oy
        else:
            if cur_line:
                visual_lines.append(cur_line)
            cur_line = [sp]
            cur_y = oy
    if cur_line:
        visual_lines.append(cur_line)

    num_vis_lines = len(visual_lines)

    # ── Split translated text into chunks, one per visual line ──
    if num_vis_lines == 1:
        line_texts = [text]
    else:
        # Split by newlines first (translator might use them)
        parts = text.split("\n")
        if len(parts) == num_vis_lines:
            line_texts = parts
        else:
            # Proportional split by original span text length
            orig_lengths = []
            for vl in visual_lines:
                orig_lengths.append(sum(len(s.text) for s in vl) or 1)
            total_orig = sum(orig_lengths)
            flat = text.replace("\n", " ")
            line_texts = []
            offset = 0
            for idx, ol in enumerate(orig_lengths):
                proportion = ol / total_orig
                chunk_len = int(round(proportion * len(flat)))
                if idx == len(orig_lengths) - 1:
                    line_texts.append(flat[offset:].strip())
                else:
                    line_texts.append(flat[offset:offset + chunk_len].strip())
                    offset += chunk_len

    # ── Insert each visual line at its origin ──
    shape = page.new_shape()
    cjk_mode = _looks_cjk(text)

    for vl_idx, (vl_spans, lt) in enumerate(zip(visual_lines, line_texts)):
        if not lt:
            continue

        # Use leftmost span origin as insertion point
        first_span = min(vl_spans, key=lambda s: s.origin[0])
        origin = fitz.Point(first_span.origin[0], first_span.origin[1])
        base_fs = max(first_span.size, 4.0)
        color = _int_to_rgb(first_span.color)

        # Progressive font shrinking if text is too wide
        fs = base_fs
        min_fs = max(3.5, base_fs * 0.45)
        text_w = float(mf.text_length(lt, fontsize=fs))

        while text_w > available_w and fs > min_fs:
            fs = max(min_fs, fs - 0.5)
            text_w = float(mf.text_length(lt, fontsize=fs))

        # If still too wide, truncate with ellipsis
        if text_w > available_w:
            ell = "…"
            truncated = lt
            while truncated and float(mf.text_length(truncated + ell, fontsize=fs)) > available_w:
                truncated = truncated[:-1]
            lt = (truncated + ell) if truncated else ell

        # Adjust origin Y if font size changed (keep baseline visually close)
        if abs(fs - base_fs) > 0.1:
            # Shift up slightly so baseline stays roughly in the same place
            origin = fitz.Point(origin.x, origin.y)

        shape.insert_text(
            origin,
            lt,
            fontname=fontname,
            fontsize=fs,
            color=color,
        )

    shape.commit(overlay=True)
    return True


def typeset_and_insert(
    page: fitz.Page, 
    rect: fitz.Rect, 
    plan: TranslationPlan,
    font_map: dict # dict of fontname -> fitz.Font for measuring
) -> bool:
    """Implement kind-aware typesetting policies.
    
    Delegates to span-level origin-based insertion when original_spans
    are available (vector text), otherwise uses bbox-based textbox insertion
    (raster / no-span fallback).
    """
    text = plan.final_rendered_text.strip()
    if not text:
        return False

    # ── Span-based path: precise origin insertion ──
    # Bypass exact-origin placement for table cells so they can reflow inside their full bbox
    if plan.container.original_spans and plan.container.kind != ContainerKind.TABLE_CELL:
        ok = typeset_and_insert_spans(page, plan, font_map)
        if ok:
            return True
        # fall through to textbox if span path fails

    # ── Textbox-based path (raster / fallback) ──
    intent = plan.rendering_intent
    fontname = intent.font_name
    mf = font_map.get(fontname)
    if not mf:
        # fallback measure font
        mf = fitz.Font("helv")
        
    cjk_mode = _looks_cjk(text)
    r_obj = fitz.Rect(rect)
    max_w = max(1.0, r_obj.width - 0.2)
    lineheight_factor = 1.15
    
    # Kind-aware policy bindings
    kind = plan.container.kind
    align = intent.alignment
    fs = max(5.0, intent.font_size_start)
    min_fontsize = 4.0
    
    if kind == ContainerKind.TABLE_CELL:
        # aggressive shrink, conservative wrap, no overflow
        min_fontsize = 3.5
    elif kind == ContainerKind.LABEL:
        # aggressive shrink, minimal lines
        min_fontsize = 4.0
        align = 1 # Force center alignment often for labels
    elif kind == ContainerKind.HEADER_FOOTER:
        # preserve size, minimal shrink
        min_fontsize = max(6.0, fs * 0.8)
    else: # PARAGRAPH / LIST_ITEM
        min_fontsize = 5.0
        
    shape = page.new_shape()
    
    while fs >= min_fontsize - 1e-6:
        raw_lines = _wrap_text_to_width(text, mf, fs, max_w, cjk_mode=cjk_mode)
        max_lines_possible = max(1, int(rect.height / (fs * lineheight_factor)))
        
        if len(raw_lines) > max_lines_possible:
            if kind == ContainerKind.HEADER_FOOTER and fs <= min_fontsize + 0.5:
                # If header/footer can't shrink more, just break and truncate
                break
                
            # Compute new FS aggressively based on overshoot
            fs_est = rect.height / (len(raw_lines) * lineheight_factor)
            new_fs = max(min_fontsize, min(fs - 0.5, fs_est))
            
            if fs - new_fs < 0.2:
                # We've bottomed out the ladder
                break
                
            fs = new_fs
            continue
            
        # Try fitz's native textbox using our computed constraints
        wrapped = "\n".join(raw_lines).strip()
        try:
            rc = shape.insert_textbox(
                rect,
                wrapped,
                fontname=fontname,
                fontsize=fs,
                color=intent.color,
                align=align,
                lineheight=lineheight_factor,
                rotate=intent.rotation,
            )
            if isinstance(rc, float) and rc >= 0:
                shape.commit(overlay=True)
                return True
        except:
            pass
            
        fs -= 0.5
        
    # Final Fallback Component: Hard Truncate
    # If we exit the loop, we could not fit text without overflow at `min_fontsize`. We MUST truncate.
    fs = min_fontsize
    raw_lines = _wrap_text_to_width(text, mf, fs, max_w, cjk_mode=cjk_mode)
    max_lines_possible = max(1, int(rect.height / (fs * lineheight_factor)))
    
    # 8.1 Truncate fallback
    truncated = _truncate_lines_to_height(raw_lines, mf, fs, max_w, max_lines_possible)
    wrapped = "\n".join(truncated).strip()
    
    try:
        rc = shape.insert_textbox(
            rect,
            wrapped,
            fontname=fontname,
            fontsize=fs,
            color=intent.color,
            align=align,
            lineheight=lineheight_factor,
        )
        if isinstance(rc, float) and rc >= 0:
            shape.commit(overlay=True)
            return True
    except:
        pass
        
    # Absolute zero-fallback (just insert at point with no bounds check)
    start_pt = fitz.Point(rect.x0, rect.y0 + fs)
    shape.insert_text(start_pt, "\n".join(truncated), fontname=fontname, fontsize=fs, lineheight=lineheight_factor)
    shape.commit(overlay=True)
    return True
