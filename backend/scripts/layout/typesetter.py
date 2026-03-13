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
            '\uFE70' <= ch <= '\uFEFF'):   # Arabic Presentation Forms-B
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
    """
    Produce explicit line breaks so that each line width <= max_width.
    - For CJK: wrap by characters (with a small ASCII-run exception).
    - For non-CJK: wrap by whitespace with fallback to char-wrap for very long tokens.
    """
    lines = []
    def w(s: str) -> float:
        return float(font.text_length(s, fontsize=fontsize))
        
    if not text:
        return []

    # RTL scripts: directionality hook
    if _looks_rtl(text):
        text = _bidirectional_shape(text)
        
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
            
        if cjk_mode:
            # CJK: character-based wrapping with ASCII-run preservation
            cur = ""
            ascii_buf = ""
            for ch in para:
                # basic ASCII run preservation (A-Z, a-z, 0-9)
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
                        continue # skip leading spaces in CJK wrap

                    cand = cur + ch
                    if not cur or w(cand) <= max_width: cur = cand
                    else:
                        lines.append(cur.rstrip())
                        cur = ch
                        
            if ascii_buf:
                cand = cur + ascii_buf
                if not cur or w(cand) <= max_width: cur = cand
                else:
                    lines.append(cur.rstrip())
                    cur = ascii_buf
                    
            if cur: lines.append(cur.rstrip())
        else:
            # space-delimited: word wrapping with long-token fallback
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
    print(f"Wrapped text to width: max_width={max_width}, original_len={len(text)}, lines={len(lines)}")
    return lines

def _truncate_lines_to_height(lines: List[str], font: fitz.Font, fontsize: float, max_width: float, max_lines: int) -> List[str]:
    """Ensure at most max_lines. If truncated, add ellipsis to last line."""
    if max_lines <= 0: return []
    if len(lines) <= max_lines: return lines
    out = lines[:max_lines]
    last = out[-1]
    ell = "…"
    while last and float(font.text_length(last + ell, fontsize=fontsize)) > max_width:
        last = last[:-1]
    out[-1] = (last + ell) if last else ell
    print(f"Truncating to height: max_lines={max_lines}, original_lines={len(lines)}, truncated_lines={len(out)}")
    return out

def _int_to_rgb(color_int: int):
    """Convert integer color to (r, g, b) floats."""
    r = ((color_int >> 16) & 255) / 255.0
    g = ((color_int >> 8) & 255) / 255.0
    b = (color_int & 255) / 255.0
    return (r, g, b)


def _resolve_textbox_layout(
    plan: TranslationPlan,
    rect: fitz.Rect,
) -> tuple[fitz.Rect, ContainerKind, int, float, float, float]:
    intent = plan.rendering_intent
    r_obj = fitz.Rect(rect)
    kind = plan.container.kind

    if kind in (ContainerKind.PARAGRAPH, ContainerKind.LIST_ITEM):
        r_obj.y1 += min(30.0, r_obj.height * 0.6)

    lineheight_factor = 1.12
    align = intent.alignment
    fs = max(5.0, intent.font_size_start)
    min_fontsize = 4.0

    if kind == ContainerKind.TABLE_CELL:
        min_fontsize = 3.5
    elif kind == ContainerKind.LABEL:
        min_fontsize = 4.0
        align = 1
    elif kind == ContainerKind.HEADER_FOOTER:
        min_fontsize = max(6.0, fs * 0.8)
    else:
        min_fontsize = 5.0

    max_w = max(1.0, r_obj.width - 0.2)
    return r_obj, kind, align, fs, min_fontsize, lineheight_factor


def _fit_text_lines(
    text: str,
    rect: fitz.Rect,
    kind: ContainerKind,
    font: fitz.Font,
    fs: float,
    min_fontsize: float,
    max_w: float,
    lineheight_factor: float,
    *,
    cjk_mode: bool,
) -> tuple[float, List[str]]:
    while fs >= min_fontsize - 1e-6:
        raw_lines = _wrap_text_to_width(text, font, fs, max_w, cjk_mode=cjk_mode)
        max_lines_possible = max(1, int(rect.height / (fs * lineheight_factor)))

        if len(raw_lines) > max_lines_possible:
            if kind == ContainerKind.HEADER_FOOTER and fs <= min_fontsize + 0.5:
                break

            fs_est = rect.height / (len(raw_lines) * lineheight_factor)
            new_fs = max(min_fontsize, min(fs - 0.5, fs_est))
            if fs - new_fs < 0.2:
                break
            fs = new_fs
            continue

        return fs, raw_lines

    fs = min_fontsize
    raw_lines = _wrap_text_to_width(text, font, fs, max_w, cjk_mode=cjk_mode)
    max_lines_possible = max(1, int(rect.height / (fs * lineheight_factor)))
    return fs, _truncate_lines_to_height(raw_lines, font, fs, max_w, max_lines_possible)


def typeset_and_insert_cjk(
    page: fitz.Page,
    rect: fitz.Rect,
    plan: TranslationPlan,
    writer_font: fitz.Font,
) -> bool:
    text = plan.final_rendered_text.strip()
    if not text:
        return False

    intent = plan.rendering_intent
    r_obj, kind, align, fs, min_fontsize, lineheight_factor = _resolve_textbox_layout(plan, rect)
    max_w = max(1.0, r_obj.width - 0.2)
    fs, lines = _fit_text_lines(
        text,
        r_obj,
        kind,
        writer_font,
        fs,
        min_fontsize,
        max_w,
        lineheight_factor,
        cjk_mode=True,
    )
    wrapped = "\n".join(lines).strip()
    if not wrapped:
        return False

    tw = fitz.TextWriter(page.rect)
    try:
        tw.fill_textbox(
            r_obj,
            wrapped,
            font=writer_font,
            fontsize=fs,
            lineheight=lineheight_factor,
            align=align,
            right_to_left=_looks_rtl(text),
        )
        tw.write_text(page, color=intent.color, overlay=True)
        return True
    except Exception:
        return False


def typeset_and_insert_spans(
    page: fitz.Page,
    plan: TranslationPlan,
    font_map: dict,  # dict of fontname -> fitz.Font for measuring
    *,
    writer_font: Optional[fitz.Font] = None,
    prefer_textwriter: bool = False,
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
        return False # caller should use textbox fallback

    intent = plan.rendering_intent
    fontname = intent.font_name
    mf = font_map.get(fontname)
    if not mf:
        mf = fitz.Font("helv")

    bbox = plan.container.bbox
    available_w = max(1.0, bbox[2] - bbox[0] - 0.4)

    # ── cluster spans into visual lines by y-proximity ──
    # spans on the same baseline (within half the font size) belong to the same visual line.
    sorted_spans = sorted(spans, key=lambda s: (s.origin[1], s.origin[0]))
    visual_lines: list = [] # list of lists of PdfSpanAttrs
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

    # ── split translated text into chunks, one per visual line ──
    if num_vis_lines == 1:
        line_texts = [text]
    else:
        # split by newlines first (translator might use them)
        parts = text.split("\n")
        if len(parts) == num_vis_lines:
            line_texts = parts
        else:
            # proportional split by original span text length
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

    # ── insert each visual line at its origin ──
    cjk_mode = _looks_cjk(text)
    shape = page.new_shape()

    for vl_idx, (vl_spans, lt) in enumerate(zip(visual_lines, line_texts)):
        if not lt:
            continue

        # use leftmost span origin as insertion point
        first_span = min(vl_spans, key=lambda s: s.origin[0])
        origin = fitz.Point(first_span.origin[0], first_span.origin[1])
        base_fs = max(first_span.size, 4.0)
        color = _int_to_rgb(first_span.color)

        # progressive font shrinking if text is too wide
        fs = base_fs
        min_fs = max(3.5, base_fs * 0.45)
        text_w = float(mf.text_length(lt, fontsize=fs))

        while text_w > available_w and fs > min_fs:
            fs = max(min_fs, fs - 0.5)
            text_w = float(mf.text_length(lt, fontsize=fs))

        # if still too wide, truncate with ellipsis
        if text_w > available_w:
            ell = "…"
            truncated = lt
            while truncated and float(mf.text_length(truncated + ell, fontsize=fs)) > available_w:
                truncated = truncated[:-1]
            lt = (truncated + ell) if truncated else ell

        # adjust origin Y if font size changed (keep baseline visually close)
        if abs(fs - base_fs) > 0.1:
            # shift up slightly so baseline stays roughly in the same place
            origin = fitz.Point(origin.x, origin.y)

        if prefer_textwriter and writer_font is not None:
            tw = fitz.TextWriter(page.rect)
            try:
                tw.append(
                    origin,
                    lt,
                    font=writer_font,
                    fontsize=fs,
                    right_to_left=_looks_rtl(lt),
                )
                tw.write_text(page, color=color, overlay=True)
            except Exception:
                return False
            continue

        shape.insert_text(
            origin,
            lt,
            fontname=fontname,
            fontsize=fs,
            color=color,
        )

    if not (prefer_textwriter and writer_font is not None):
        shape.commit(overlay=True)
    return True


def typeset_and_insert(
    page: fitz.Page, 
    rect: fitz.Rect, 
    plan: TranslationPlan,
    font_map: dict, # dict of fontname -> fitz.Font for measuring
    *,
    writer_font: Optional[fitz.Font] = None,
    prefer_textwriter: bool = False,
) -> bool:
    """Implement kind-aware typesetting policies.
    
    Delegates to span-level origin-based insertion when original_spans
    are available (vector text), otherwise uses bbox-based textbox insertion
    (raster / no-span fallback).
    """
    text = plan.final_rendered_text.strip()
    if not text:
        return False

    # ── span-based path: precise origin insertion ──
    # bypass exact-origin placement for table cells so they can reflow inside their full bbox.
    # ALSO bypass for singleton paragraphs/list items so they can wrap/reflow if the translation is long.
    use_spans = False
    if plan.container.original_spans:
        kind = plan.container.kind
        if kind == ContainerKind.TABLE_CELL:
            use_spans = False
        elif kind in (ContainerKind.PARAGRAPH, ContainerKind.LIST_ITEM):
            # only use strict spans for prose if it was already multi-line in the original
            use_spans = len(plan.container.original_spans) > 1
        else:
            use_spans = True

    if use_spans:
        ok = typeset_and_insert_spans(
            page,
            plan,
            font_map,
            writer_font=writer_font,
            prefer_textwriter=prefer_textwriter,
        )
        if ok:
            return True
        # fall through to textbox if span path fails

    if prefer_textwriter and writer_font is not None:
        return typeset_and_insert_cjk(page, rect, plan, writer_font)

    # ── textbox-based path (raster / fallback) ──
    intent = plan.rendering_intent
    fontname = intent.font_name
    mf = font_map.get(fontname)
    if not mf:
        # fallback measure font
        mf = fitz.Font("helv")
    cjk_mode = _looks_cjk(text)
    r_obj, kind, align, fs, min_fontsize, lineheight_factor = _resolve_textbox_layout(plan, rect)
    max_w = max(1.0, r_obj.width - 0.2)
    shape = page.new_shape()
    while fs >= min_fontsize - 1e-6:
        raw_lines = _wrap_text_to_width(text, mf, fs, max_w, cjk_mode=cjk_mode)
        max_lines_possible = max(1, int(r_obj.height / (fs * lineheight_factor)))
        if len(raw_lines) > max_lines_possible:
            if kind == ContainerKind.HEADER_FOOTER and fs <= min_fontsize + 0.5:
                break
            fs_est = r_obj.height / (len(raw_lines) * lineheight_factor)
            new_fs = max(min_fontsize, min(fs - 0.5, fs_est))
            if fs - new_fs < 0.2:
                break
            fs = new_fs
            continue
        # try fitz's native textbox using computed constraints
        wrapped = "\n".join(raw_lines).strip()
        try:
            rc = shape.insert_textbox(
                r_obj,
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
        
    # final fallback - hard truncate
    # if we exit the loop, we could not fit text without overflow at `min_fontsize`, so we MUST truncate.
    fs = min_fontsize
    raw_lines = _wrap_text_to_width(text, mf, fs, max_w, cjk_mode=cjk_mode)
    max_lines_possible = max(1, int(r_obj.height / (fs * lineheight_factor)))
    
    # truncate fallback
    truncated = _truncate_lines_to_height(raw_lines, mf, fs, max_w, max_lines_possible)
    wrapped = "\n".join(truncated).strip()
    
    try:
        rc = shape.insert_textbox(
            r_obj,
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
        
    # absolute zero-fallback (just insert at point with no bounds check)
    start_pt = fitz.Point(r_obj.x0, r_obj.y0 + fs)
    shape.insert_text(start_pt, "\n".join(truncated), fontname=fontname, fontsize=fs, lineheight=lineheight_factor)
    shape.commit(overlay=True)
    return True
