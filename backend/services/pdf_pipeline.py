from __future__ import annotations

from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os, fitz, io, re, threading, time, inspect, math # PyMuPDF (used for writing back into the PDF)

# Azure Document Intelligence
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from scripts.layout.extractor import build_containers
from scripts.translation_service_pdf import translate_blocks
from scripts.layout.typesetter import typeset_and_insert
from scripts.layout.classifier import classify_container
from scripts.layout.containers import (
    ContainerRef,
    ContainerKind,
    TranslationPlan,
    RenderingIntent,
    TranslationPolicy,
    ContainerTranslation,
)
from scripts.layout.geometry import (
    poly_to_bbox as _poly_to_bbox,
    scale_bbox as _scale_bbox,
    scale_poly as _scale_poly,
    union_bbox as _union_bbox,
    bbox_overlap_area as _bbox_overlap_area,
    bbox_area as _bbox_area,
)
from scripts.text_normalization.normalizer import apply_normalization_pipeline, restore_protected_tokens
from services.logger import get_logger


# -----------------------------
# Env
# -----------------------------
load_dotenv(override=True)

doc_int_key = "AZURE_DOCUMENT_INTELLIGENCE_KEY"
doc_int_endpoint = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
doc_int_model = "AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID" # default: prebuilt-layout


# -----------------------------
# Verbose logging helpers
# -----------------------------
def _vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def _fmt_bbox(bbox: Tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    return f"({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})"


def _preview_text(text: str, limit: int = 220) -> str:
    s = (text or "").replace("\n", "\\n")
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _log_every_five_extracted(items: List["_ExtractItem"], verbose: bool) -> None:
    if not verbose:
        return
    total = len(items)
    if total == 0 or total % 5 != 0:
        return

    print(f"[EXTRACT][BATCH] Reached {total} extracted blocks. Showing latest 5 Azure DocInt-derived items:")
    for idx, it in enumerate(items[-5:], start=total - 4):
        print(
            f"  #{idx:04d} | source={it.source_kind:<10} | page={it.page_index} | "
            f"bbox={_fmt_bbox(it.bbox)} | min_offset={it.min_offset} | spans={len(it.spans)}"
        )
        print(f"         text={_preview_text(it.text)}")


def _print_skip_summary(verbose: bool, stage_name: str, stats: Dict[str, int]) -> None:
    if not verbose:
        return
    seen = stats.get("seen", 0)
    added = stats.get("added", 0)
    skipped_total = seen - added
    print(f"[EXTRACT][{stage_name}] seen={seen}, added={added}, skipped={skipped_total}")
    for k in sorted(stats.keys()):
        if k in {"seen", "added"}:
            continue
        print(f"  - {k}: {stats[k]}")


def _avg(numer: float, denom: int) -> float:
    return (float(numer)/float(denom)) if denom else 0.0


def _coalesce_number(*vals, default: float = 0.0) -> float:
    for v in vals:
        if v is None: continue
        try: return float(v)
        except Exception: continue
    return float(default)


def _coalesce_int(*vals, default: int = 0) -> int:
    for v in vals:
        if v is None: continue
        try: return int(v)
        except Exception: continue
    return int(default)


def _diag_lookup(diag: Dict[str, Any], *keys, default=None):
    cur: Any = diag
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
        if cur is None: return default
    return cur


def _print_translation_samples(
    blocks: List[ContainerRef],
    translations: List[TranslationPlan],
    diagnostics: Dict[str, Any],
    *,
    verbose: bool,
) -> None:
    if not verbose: return

    llm1_pairs = (
        _diag_lookup(diagnostics, "llm1_pairs")
        or _diag_lookup(diagnostics, "samples", "llm1_pairs")
        or _diag_lookup(diagnostics, "debug", "llm1_pairs")
    )

    if isinstance(llm1_pairs, list) and llm1_pairs:
        print("[TRANSLATE] LLM-1 original vs translated samples (1 in 5):")
        for i, pair in enumerate(llm1_pairs, start=1):
            if i % 5 != 0: continue
            if isinstance(pair, dict):
                src = pair.get("original") or pair.get("source") or pair.get("input") or ""
                dst = pair.get("translated") or pair.get("output") or pair.get("llm1_output") or ""
            elif isinstance(pair, (tuple, list)) and len(pair) >= 2: src, dst = pair[0], pair[1]
            else: continue
            print(f"  [LLM-1 sample #{i}] original  : {_preview_text(str(src), 400)}")
            print(f"  [LLM-1 sample #{i}] translated: {_preview_text(str(dst), 400)}\n")
        return

    # fallback if translation_service doesn't expose LLM-1 diagnostics
    print("[TRANSLATE] LLM-1 samples unavailable from translation_service; showing FINAL original vs translated samples (1 in 5):")
    for i, bt in enumerate(translations, start=1):
        if i % 5 != 0:
            continue
        print(f"  [Final sample #{i}] original  : {_preview_text(bt.container.text, 180)}")
        print(f"  [Final sample #{i}] translated: {_preview_text(bt.final_rendered_text, 180)}")


def _print_timing_report(
    *,
    verbose: bool,
    chunks_processed: int,
    extraction_time: float,
    translation_time: float,
    apply_time: float,
    save_time: float,
    total_time: float,
    diagnostics: Optional[Dict[str, Any]] = None,
    errors: int = 0,
) -> None:
    if not verbose: return

    diagnostics = diagnostics or {}

    cache_hits = _coalesce_int(
        _diag_lookup(diagnostics, "cache_hits"),
        _diag_lookup(diagnostics, "stats", "cache_hits"),
        _diag_lookup(diagnostics, "counts", "cache_hits"),
        default=0,
    )
    azure_mt_total = _coalesce_number(
        _diag_lookup(diagnostics, "azure_mt_total"),
        _diag_lookup(diagnostics, "timings", "azure_mt_total"),
        _diag_lookup(diagnostics, "timings", "azure_mt_seconds"),
        _diag_lookup(diagnostics, "stats", "azure_mt_total"),
        default=0.0,
    )
    llm1_total = _coalesce_number(
        _diag_lookup(diagnostics, "llm1_total"),
        _diag_lookup(diagnostics, "timings", "llm1_total"),
        _diag_lookup(diagnostics, "timings", "llm1_seconds"),
        _diag_lookup(diagnostics, "stats", "llm1_total"),
        default=0.0,
    )
    llm2_total = _coalesce_number(
        _diag_lookup(diagnostics, "llm2_total"),
        _diag_lookup(diagnostics, "timings", "llm2_total"),
        _diag_lookup(diagnostics, "timings", "llm2_seconds"),
        _diag_lookup(diagnostics, "stats", "llm2_total"),
        default=0.0,
    )

    # if diagnostics does not provide per-chunk sum, use translation time as sensible fallback
    end_to_end_sum = _coalesce_number(
        _diag_lookup(diagnostics, "end_to_end_sum"),
        _diag_lookup(diagnostics, "timings", "end_to_end_sum"),
        _diag_lookup(diagnostics, "stats", "end_to_end_sum"),
        default=translation_time,
    )

    print("\nTiming report")
    print("-------------")
    print(f"  Chunks processed     : {chunks_processed}")
    print(f"  Cache hits           : {cache_hits}")
    print(f"  Azure MT total       : {azure_mt_total:.2f} | avg/chunk: {_avg(azure_mt_total, chunks_processed):.2f}")
    print(f"  LLM-1 total          : {llm1_total:.2f} | avg/chunk: {_avg(llm1_total, chunks_processed):.2f}")
    print(f"  LLM-2 total          : {llm2_total:.2f} | avg/chunk: {_avg(llm2_total, chunks_processed):.2f}")
    print(f"  End-to-end (sum)     : {end_to_end_sum:.2f} | avg/chunk: {_avg(end_to_end_sum, chunks_processed):.2f}")
    print(f"  Errors               : {errors}")
    print("  -- pipeline stages --")
    print(f"  Text extraction      : {extraction_time:.2f}")
    print(f"  Translation          : {translation_time:.2f}")
    print(f"  Apply translations   : {apply_time:.2f}")
    print(f"  Save PDF             : {save_time:.2f}")
    print(f"  Total pipeline       : {total_time:.2f}")


# -----------------------------
# Azure Doc Intelligence extraction (read-only)
# -----------------------------
_doc_int_client: Optional[DocumentIntelligenceClient] = None


def _get_doc_int_client(verbose: bool = False) -> DocumentIntelligenceClient:
    global _doc_int_client
    if _doc_int_client is not None:
        _vprint(verbose, "[DOCINT] Reusing cached DocumentIntelligenceClient")
        return _doc_int_client

    endpoint = os.getenv(doc_int_endpoint, "").strip()
    key = os.getenv(doc_int_key, "").strip()
    if not endpoint or not key:
        raise RuntimeError(
            f"Missing Azure Document Intelligence credentials. "
            f"Set {doc_int_endpoint} and {doc_int_key} in your environment."
        )

    _vprint(verbose, f"[DOCINT] Creating DocumentIntelligenceClient for endpoint={endpoint!r}")
    _doc_int_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return _doc_int_client


_PUNCT_FIX_RE = re.compile(r"\s+([,.;:!?])")
def _join_words_preserve_punct(words: List[str]) -> str:
    s = " ".join(w for w in words if w)
    s = _PUNCT_FIX_RE.sub(r"\1", s)
    s = s.replace("( ", "(").replace(" )", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Using shared helpers from geometry.py via imports above.
def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def _min_span_offset(spans) -> Optional[int]:
    if not spans: return None
    offs = []
    for s in spans:
        try: offs.append(int(s.offset))
        except Exception: pass
    return min(offs) if offs else None


def _span_ranges(spans) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not spans: return out
    for s in spans:
        try:
            o = int(s.offset)
            l = int(s.length)
            out.append((o, o + l))
        except Exception: continue
    return out


def _any_overlap(ranges_a: List[Tuple[int, int]], ranges_b: List[Tuple[int, int]]) -> bool:
    for a0, a1 in ranges_a:
        for b0, b1 in ranges_b:
            if _spans_overlap(a0, a1, b0, b1): return True
    return False


@dataclass(frozen=True)
class _ExtractItem:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    min_offset: Optional[int]
    # keep spans only for overlap checks (not exposed outside)
    spans: List[Tuple[int, int]]
    source_kind: str # "table_cell" or "paragraph"


def _extract_items_from_docintel_result(doc: fitz.Document, analyze_result, *, verbose: bool = False) -> List[_ExtractItem]: # line-level extraction
    """
    Extract layout-faithful translation units:
        - table cells
        - line runs (DocInt lines split into left/right runs by large x-gaps)
    Avoid duplication by skipping line runs whose spans overlap table-cell spans.
    """
    items: List[_ExtractItem] = []

    if not getattr(analyze_result, "pages", None):
        _vprint(verbose, "[EXTRACT] analyze_result.pages is empty; nothing to extract")
        return items

    # -------------------------
    # Page scale factors (DI -> fitz points)
    # -------------------------
    page_scale: Dict[int, Tuple[float, float]] = {}
    _vprint(verbose, f"[EXTRACT] Computing page scale factors for {len(analyze_result.pages)} DI pages")

    for page_obj in analyze_result.pages:
        page_number_1 = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
        if not page_number_1:
            _vprint(verbose, "[EXTRACT][PAGE-SCALE] Skipping DI page without page_number")
            continue
        page_index = int(page_number_1) - 1
        if page_index < 0 or page_index >= doc.page_count:
            _vprint(verbose, f"[EXTRACT][PAGE-SCALE] Skipping DI page_number={page_number_1}; out of fitz range")
            continue

        fitz_page = doc[page_index]
        fitz_w = float(fitz_page.rect.width)
        fitz_h = float(fitz_page.rect.height)

        di_w = float(getattr(page_obj, "width", 0.0) or 0.0)
        di_h = float(getattr(page_obj, "height", 0.0) or 0.0)

        if di_w <= 0 or di_h <= 0:
            sx = sy = 72.0
            _vprint(verbose, f"[EXTRACT][PAGE-SCALE] p{page_index}: missing DI dimensions, fallback sx=sy=72.0")
        else:
            sx = fitz_w / di_w
            sy = fitz_h / di_h
            _vprint(
                verbose,
                f"[EXTRACT][PAGE-SCALE] p{page_index}: DI({di_w:.3f}x{di_h:.3f}) -> "
                f"PDF({fitz_w:.3f}x{fitz_h:.3f}) => sx={sx:.6f}, sy={sy:.6f}",
            )
        page_scale[page_index] = (sx, sy)

    # -------------------------
    # 1) TABLE CELLS
    # -------------------------
    table_stats: Dict[str, int] = {
        "seen": 0,
        "added": 0,
        "skipped_empty_text": 0,
        "skipped_no_valid_page_or_bbox": 0,
        "skipped_tiny_area": 0,
    }
    line_stats: Dict[str, int] = {
        "seen": 0,
        "added": 0,
        "skipped_empty_text": 0,
        "skipped_table_span_overlap": 0,
        "skipped_no_words_for_line": 0,
        "skipped_no_valid_page_or_bbox": 0,
        "skipped_tiny_area": 0,
    }

    table_cell_span_ranges: List[Tuple[int, int]] = []
    all_tables = (getattr(analyze_result, "tables", None) or [])
    _vprint(verbose, f"[EXTRACT][TABLE_CELLS] DI tables found: {len(all_tables)}")

    for ti, table in enumerate(all_tables, start=1):
        cells = (getattr(table, "cells", None) or [])
        _vprint(verbose, f"[EXTRACT][TABLE_CELLS] Table #{ti}: cells={len(cells)}")

        for ci, cell in enumerate(cells, start=1):
            table_stats["seen"] += 1

            text = (getattr(cell, "content", "") or "").strip()
            if not text:
                table_stats["skipped_empty_text"] += 1
                continue

            spans = _span_ranges(getattr(cell, "spans", None) or [])
            if spans:
                table_cell_span_ranges.extend(spans)

            brs = getattr(cell, "bounding_regions", None) or []
            cell_bboxes: List[Tuple[float, float, float, float]] = []
            page_index: Optional[int] = None

            for br in brs:
                pno = getattr(br, "page_number", None) or getattr(br, "pageNumber", None)
                if not pno:
                    continue
                pi = int(pno) - 1
                if pi < 0 or pi >= doc.page_count:
                    continue
                poly = getattr(br, "polygon", None)
                bb = _poly_to_bbox(poly)
                if not bb:
                    continue
                sx, sy = page_scale.get(pi, (72.0, 72.0))
                bb = _scale_bbox(bb, sx, sy)
                cell_bboxes.append(bb)
                page_index = pi

            if page_index is None or not cell_bboxes:
                table_stats["skipped_no_valid_page_or_bbox"] += 1
                continue

            bbox = _union_bbox(cell_bboxes)
            if fitz.Rect(*bbox).get_area() < 5:
                table_stats["skipped_tiny_area"] += 1
                continue

            items.append(
                _ExtractItem(
                    page_index=page_index,
                    bbox=bbox,
                    text=text,
                    min_offset=_min_span_offset(getattr(cell, "spans", None) or []),
                    spans=spans,
                    source_kind="table_cell",
                )
            )
            table_stats["added"] += 1
            _log_every_five_extracted(items, verbose)

    _print_skip_summary(verbose, "TABLE_CELLS", table_stats)

    # -------------------------
    # 2) LINE RUNS
    # -------------------------
    _vprint(verbose, f"[EXTRACT][LINE_RUNS] Extracting from DI pages[].lines[]")

    def _word_span(w) -> Optional[Tuple[int, int]]:
        sp = getattr(w, "span", None)
        if sp is None:
            return None
        try:
            o = int(sp.offset)
            l = int(sp.length)
            return (o, o + l)
        except Exception:
            return None

    def _ranges_overlap_any(a: Tuple[int, int], ranges: List[Tuple[int, int]]) -> bool:
        a0, a1 = a
        for b0, b1 in ranges:
            if _spans_overlap(a0, a1, b0, b1):
                return True
        return False

    for page_obj in analyze_result.pages:
        page_number_1 = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
        if not page_number_1:
            continue
        page_index = int(page_number_1) - 1
        if page_index < 0 or page_index >= doc.page_count:
            continue

        sx, sy = page_scale.get(page_index, (72.0, 72.0))

        # build word list (scaled bboxes + span offsets)
        word_items: List[Dict[str, Any]] = []
        for w in (getattr(page_obj, "words", None) or []):
            txt = (getattr(w, "content", "") or "").strip()
            if not txt:
                continue
            sp = _word_span(w)
            if not sp:
                continue
            bb = _poly_to_bbox(getattr(w, "polygon", None))
            if not bb:
                continue
            bb = _scale_bbox(bb, sx, sy)
            word_items.append(
                {
                    "text": txt,
                    "bbox": bb,
                    "span": sp,
                    "h": (bb[3] - bb[1]),
                }
            )

        lines = (getattr(page_obj, "lines", None) or [])
        _vprint(verbose, f"[EXTRACT][LINE_RUNS] page={page_index}: lines={len(lines)} words={len(word_items)}")

        for ln in lines:
            line_stats["seen"] += 1
            spans_ln = _span_ranges(getattr(ln, "spans", None) or [])
            if spans_ln and table_cell_span_ranges and _any_overlap(spans_ln, table_cell_span_ranges):
                line_stats["skipped_table_span_overlap"] += 1
                continue

            # select words whose spans overlap line spans
            ln_words: List[Dict[str, Any]] = []
            if spans_ln:
                for wi in word_items:
                    if _ranges_overlap_any(wi["span"], spans_ln):
                        ln_words.append(wi)
            else:
                # no spans: fallback to line polygon + content as a single run
                txt = (getattr(ln, "content", "") or "").strip()
                if not txt:
                    line_stats["skipped_empty_text"] += 1
                    continue
                bb = _poly_to_bbox(getattr(ln, "polygon", None))
                if not bb:
                    line_stats["skipped_no_valid_page_or_bbox"] += 1
                    continue
                bb = _scale_bbox(bb, sx, sy)
                if fitz.Rect(*bb).get_area() < 5:
                    line_stats["skipped_tiny_area"] += 1
                    continue

                items.append(
                    _ExtractItem(
                        page_index=page_index,
                        bbox=bb,
                        text=txt,
                        min_offset=_min_span_offset(getattr(ln, "spans", None) or []),
                        spans=spans_ln,
                        source_kind="line_run",
                    )
                )
                line_stats["added"] += 1
                _log_every_five_extracted(items, verbose)
                continue

            if not ln_words:
                line_stats["skipped_no_words_for_line"] += 1
                continue

            # sort left-to-right and split by big gaps
            ln_words.sort(key=lambda z: z["bbox"][0])

            heights = [w["h"] for w in ln_words if w["h"] > 0]
            h_med = sorted(heights)[len(heights) // 2] if heights else 10.0
            gap_thr = max(8.0, 2.2 * float(h_med)) # important for separating columns

            run: List[Dict[str, Any]] = []
            prev_x1: Optional[float] = None

            def flush(run_words: List[Dict[str, Any]]) -> None:
                if not run_words:
                    return
                txt = _join_words_preserve_punct([w["text"] for w in run_words])
                if not txt:
                    return
                bb = _union_bbox([w["bbox"] for w in run_words])
                if fitz.Rect(*bb).get_area() < 5:
                    return

                run_spans = [w["span"] for w in run_words if w.get("span")]
                min_off = min((s[0] for s in run_spans), default=None)

                # avoid duplicating table content
                if run_spans and table_cell_span_ranges and _any_overlap(run_spans, table_cell_span_ranges):
                    line_stats["skipped_table_span_overlap"] += 1
                    return

                items.append(
                    _ExtractItem(
                        page_index=page_index,
                        bbox=bb,
                        text=txt,
                        min_offset=min_off,
                        spans=run_spans,
                        source_kind="line_run",
                    )
                )
                line_stats["added"] += 1
                _log_every_five_extracted(items, verbose)

            for wi in ln_words:
                x0, _y0, x1, _y1 = wi["bbox"]
                if prev_x1 is not None and (x0 - prev_x1) > gap_thr:
                    flush(run)
                    run = []
                run.append(wi)
                prev_x1 = x1
            flush(run)

    _print_skip_summary(verbose, "LINE_RUNS", line_stats)

    # -------------------------
    # final reading order sort
    # -------------------------
    _vprint(verbose, f"[EXTRACT] Sorting {len(items)} extracted items into reading order")
    items.sort(
        key=lambda it: (
            it.min_offset if it.min_offset is not None else 10**18,
            it.page_index,
            it.bbox[1],
            it.bbox[0],
        )
    )

    if verbose:
        _vprint(verbose, "[EXTRACT] First 5 items after sort:")
        for i, it in enumerate(items[:5], start=1):
            _vprint(
                verbose,
                f"  #{i}: source={it.source_kind}, page={it.page_index}, bbox={_fmt_bbox(it.bbox)}, "
                f"min_offset={it.min_offset}, text={_preview_text(it.text)}"
            )
        _vprint(verbose, f"[EXTRACT] Total extracted items: {len(items)}")

    return items


def extract_all_blocks(pdf_bytes: bytes, *, verbose: bool = False) -> Tuple[fitz.Document, List[ContainerRef], Dict[str, Any]]:
    """
    Opens the PDF (for writing later) and extracts text blocks using Azure Document Intelligence.
    Returns:
        - open fitz doc (caller will write into it)
        - extracted blocks list as ContainerRef
        - metadata dict (includes 'mask_regions_by_page' for inpainting)
    """
    t0 = time.perf_counter()
    _vprint(verbose, f"[PIPELINE][EXTRACT] Opening PDF bytes (size={len(pdf_bytes)} bytes)")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    _vprint(verbose, f"[PIPELINE][EXTRACT] PDF opened: page_count={doc.page_count}")

    if doc.is_encrypted:
        doc.close()
        raise ValueError("Encrypted PDFs are not supported in this starter example.")

    client = _get_doc_int_client(verbose=verbose)
    model_id = os.getenv(doc_int_model, "prebuilt-layout").strip() or "prebuilt-layout"
    _vprint(verbose, f"[DOCINT] Using model_id={model_id!r}")

    t_di_start = time.perf_counter()
    _vprint(verbose, "[DOCINT] begin_analyze_document(...) starting")
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=io.BytesIO(pdf_bytes), # raw PDF bytes sent as a stream
        content_type="application/pdf",
    )
    _vprint(verbose, "[DOCINT] Poller created; waiting for result()")
    analyze_result = poller.result()
    t_di_end = time.perf_counter()
    _vprint(verbose, f"[DOCINT] analyze_result received in {t_di_end - t_di_start:.2f}s")

    # 7.1 Translate containers, not lines
    blocks = build_containers(doc, analyze_result, verbose=verbose)

    # Build word-level mask regions (used for glyph-accurate inpainting, step 11.3)
    from scripts.layout.raster_processor import build_mask_regions_from_analyze_result
    mask_regions_by_page = build_mask_regions_from_analyze_result(
        doc, analyze_result, verbose=verbose
    )
    _vprint(verbose, f"[PIPELINE][EXTRACT] mask_regions built for {len(mask_regions_by_page)} pages")

    # Metadata for the pipeline (Step 8 routing, etc.)
    metadata = {
        "page_count": doc.page_count,
        "analyze_result": analyze_result,
        "timing_di_seconds": t_di_end - t_di_start,
        # per-page word-level mask primitives (dict: page_index -> List[MaskRegion])
        "mask_regions_by_page": mask_regions_by_page,
    }

    _vprint(verbose, f"[PIPELINE][EXTRACT] Completed container extraction in {time.perf_counter() - t0:.2f}s (blocks={len(blocks)})")
    return doc, blocks, metadata

# -----------------------------
# Apply (write) — keep this single-threaded
# -----------------------------

_FONT_CACHE_LOCK = threading.Lock()
_FONT_BUFFER_CACHE: Dict[str, bytes] = {} # fontfile -> font buffer bytes
_MEASURE_FONT_LOCK = threading.Lock()
_MEASURE_FONT_CACHE: Dict[Tuple[str, Optional[str]], fitz.Font] = {}
# Japanese + CJK detection
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF66-\uFF9F]")
_PAGE_SPAN_CACHE_LOCK = threading.Lock()
_PAGE_SPAN_CACHE: Dict[int, List[Tuple[fitz.Rect, float, float]]] = {} # page_no -> (rect, size, weight)


def _get_page_style_spans(page: fitz.Page, *, verbose: bool = False) -> List[Tuple[fitz.Rect, float, float]]:
    """
    Extract original text spans with font sizes from the PDF (before redaction).
    Cached per page for speed.
    """
    pno = int(getattr(page, "number", 0))
    with _PAGE_SPAN_CACHE_LOCK:
        cached = _PAGE_SPAN_CACHE.get(pno)
        if cached is not None:
            return cached

    d = page.get_text("dict") or {}
    spans_out: List[Tuple[fitz.Rect, float, float]] = []
    for b in d.get("blocks", []) or []:
        for ln in b.get("lines", []) or []:
            for sp in ln.get("spans", []) or []:
                bbox = sp.get("bbox")
                size = sp.get("size")
                txt = sp.get("text", "") or ""
                if not bbox or size is None:
                    continue
                try:
                    r = fitz.Rect(bbox)
                    sz = float(size)
                except Exception:
                    continue
                if r.get_area() <= 0:
                    continue
                # weight: overlap area will dominate; multiply by text length to reduce noise
                w = float(max(1, len(txt.strip())))
                spans_out.append((r, sz, w))

    with _PAGE_SPAN_CACHE_LOCK:
        _PAGE_SPAN_CACHE[pno] = spans_out
    _vprint(verbose, f"[STYLE] Cached {len(spans_out)} font spans for page {pno}")
    return spans_out


def _estimate_fontsize_for_rect(page: fitz.Page, rect: fitz.Rect, *, verbose: bool = False) -> Optional[float]:
    """
    Estimate the original font size inside rect by weighted median of intersecting spans.
    """
    spans = _get_page_style_spans(page, verbose=verbose)
    cand: List[Tuple[float, float]] = [] # (size, weight)

    for r, sz, w in spans:
        inter = r & rect
        if inter.is_empty:
            continue
        ia = inter.get_area()
        if ia < 1.0:
            continue
        # require some meaningful overlap to avoid grabbing neighbors
        if (ia / max(1.0, r.get_area())) < 0.12 and (ia / max(1.0, rect.get_area())) < 0.04:
            continue
        cand.append((sz, ia * w))

    if not cand:
        return None

    cand.sort(key=lambda x: x[0])
    total = sum(w for _, w in cand)
    acc = 0.0
    for sz, w in cand:
        acc += w
        if acc >= total / 2.0:
            return math.floor(float(sz))
    return math.floor(float(cand[-1][0]))


def _get_font_buffer(fontfile: str, *, verbose: bool = False) -> Optional[bytes]:
    """
    Load font file into a MuPDF font buffer (cached).
    """
    if not fontfile:
        _vprint(verbose, "[FONT] _get_font_buffer called with empty fontfile")
        return None
    with _FONT_CACHE_LOCK:
        buf = _FONT_BUFFER_CACHE.get(fontfile)
        if buf is not None:
            _vprint(verbose, f"[FONT] Font buffer cache hit: {fontfile}")
            return buf

        _vprint(verbose, f"[FONT] Loading font buffer from file: {fontfile}")
        # build a MuPDF font and store its buffer (works well with page.insert_font(fontbuffer=...))
        f = fitz.Font(fontfile=fontfile)
        buf = f.buffer # bytes
        _FONT_BUFFER_CACHE[fontfile] = buf
        _vprint(verbose, f"[FONT] Cached font buffer: {fontfile} (bytes={len(buf) if buf else 0})")
        return buf


def _ensure_page_font(page: fitz.Page, fontname: str, fontfile: Optional[str], *, verbose: bool = False) -> None:
    """
    Ensure the given custom font is registered on this page using fontbuffer,
    which improves CID embedding + ToUnicode mapping (copy/paste).
    """
    if not fontfile:
        _vprint(verbose, f"[FONT] Page {page.number}: no custom fontfile for '{fontname}', using built-in font path")
        return
    try:
        buf = _get_font_buffer(fontfile, verbose=verbose)
        if not buf:
            _vprint(verbose, f"[FONT] Page {page.number}: empty font buffer for '{fontname}'")
            return
        # set_simple=False forces non-simple font handling (important for CJK + Unicode)
        page.insert_font(fontname=fontname, fontbuffer=buf, set_simple=False)
        _vprint(verbose, f"[FONT] Registered font '{fontname}' on page {page.number} from file '{fontfile}'")
    except Exception as e:
        # best-effort; fallback path will still try with fontfile directly
        _vprint(verbose, f"[FONT] Failed to register font '{fontname}' on page {page.number}: {e!r}")


def _safe_rect(page: fitz.Page, rect: fitz.Rect, pad: float = 0.5) -> fitz.Rect:
    r = fitz.Rect(rect)
    r.x0 -= pad
    r.y0 -= pad
    r.x1 += pad
    r.y1 += pad
    r.intersect(page.rect)
    return r


def _sanitize_fontname(name: str) -> str:
    # PDF fontnames should be short and ASCII-ish
    base = re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip())
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "font"
    return base[:40]


def _looks_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _pick_font(text: str, default_fontname: str, *, verbose: bool = False) -> Tuple[str, Optional[str]]:
    """
    Choose a font + fontfile. Returns (fontname, fontfile_or_None).

    IMPORTANT:
        - Return a deterministic fontname derived from font file name, not "cjk"/"uni".
        - This allows us to register the font once per page and get correct Unicode mapping.
    """
    # universal font override (best for multi-language)
    uni = (os.getenv("PDF_UNICODE_FONT_FILE", "") or "").strip()
    if uni and os.path.exists(uni):
        chosen = _sanitize_fontname(os.path.splitext(os.path.basename(uni))[0]), uni
        _vprint(verbose, f"[FONT] Using PDF_UNICODE_FONT_FILE override: {chosen[1]}")
        return chosen

    if _looks_cjk(text):
        _vprint(verbose, "[FONT] Detected CJK chars in text")
        cjk = (os.getenv("PDF_CJK_FONT_FILE", "") or "").strip()
        if cjk and os.path.exists(cjk):
            _vprint(verbose, f"[FONT] Using CJK font from env PDF_CJK_FONT_FILE: {cjk}")
            return _sanitize_fontname(os.path.splitext(os.path.basename(cjk))[0]), cjk

        # common Windows JP fonts (best-effort)
        win_candidates = [
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\YuGothR.ttc",
            r"C:\Windows\Fonts\yugothic.ttf",
            r"C:\Windows\Fonts\arialuni.ttf",
        ]
        for p in win_candidates:
            if os.path.exists(p):
                _vprint(verbose, f"[FONT] Using fallback Windows CJK font: {p}")
                return _sanitize_fontname(os.path.splitext(os.path.basename(p))[0]), p

        _vprint(verbose, f"[FONT] CJK text detected but no CJK font found; falling back to '{default_fontname}'")
        return default_fontname, None

    _vprint(verbose, f"[FONT] Non-CJK text; using default font '{default_fontname}'")
    return default_fontname, None


def _get_measure_font(fontname: str, fontfile: Optional[str], *, verbose: bool = False) -> fitz.Font:
    """
    Font used only for measuring text widths.
    For custom fonts, measure using the same font file.
    """
    key = (fontname, fontfile)
    with _MEASURE_FONT_LOCK:
        f = _MEASURE_FONT_CACHE.get(key)
        if f is not None:
            _vprint(verbose, f"[FONT][MEASURE] Cache hit for {key}")
            return f
        
        try:
            if fontfile:
                _vprint(verbose, f"[FONT][MEASURE] Loading measure font from file for {key}")
                f = fitz.Font(fontfile=fontfile)
            else:
                _vprint(verbose, f"[FONT][MEASURE] Loading measure font by name for {key}")
                f = fitz.Font(fontname=fontname) # Base14 like helv
        except Exception:
            # Final fallback: Base14 if everything else fails
            _vprint(verbose, f"[FONT][MEASURE] Fallback to 'helv' for font={fontname}")
            f = fitz.Font("helv")
            
        _MEASURE_FONT_CACHE[key] = f
        return f


def remove_text(page: fitz.Page, rects: List[fitz.Rect], pix: fitz.Pixmap, verbose: bool = False):
    """
    9.3 removal strategy: text-only removal avoiding background nuke.
    Adds redaction annotations filled with the sampled background color of the region.
    When apply_redactions() runs, it removes original intersecting text/graphics
    and leaves behind a solid patch of background color.
    """
    from collections import Counter
    for r in rects:
        # Sample background color from the perimeter (+2px)
        x0 = max(0, int(r.x0))
        y0 = max(0, int(r.y0))
        x1 = min(pix.width - 1, int(r.x1))
        y1 = min(pix.height - 1, int(r.y1))
        
        colors = []
        for x in range(x0, x1 + 1):
            colors.append(pix.pixel(x, y0))
            colors.append(pix.pixel(x, y1))
        for y in range(y0, y1 + 1):
            colors.append(pix.pixel(x0, y))
            colors.append(pix.pixel(x1, y))
            
        if not colors:
            bg_fill = (1.0, 1.0, 1.0)
        else:
            mc = Counter(colors).most_common(1)[0][0]
            if isinstance(mc, int):
                bg_fill = (1.0, 1.0, 1.0)  # Handle grayscale fallback
            else:
                bg_fill = (mc[0]/255.0, mc[1]/255.0, mc[2]/255.0)
                
        page.add_redact_annot(r, fill=bg_fill)


def _int_to_rgb(color_int: int) -> Tuple[float, float, float]:
    r = ((color_int >> 16) & 255) / 255.0
    g = ((color_int >> 8) & 255) / 255.0
    b = (color_int & 255) / 255.0
    return (r, g, b)


def _derive_rotation(poly: Optional[List[Tuple[float, float]]]) -> int:
    """
    10.3 Rotation support: derive 90-degree increments from polygon.
    Polygons from DocInt are [x0,y0, x1,y1, x2,y2, x3,y3].
    """
    if not poly or len(poly) < 2:
        return 0
    # DocInt polygon is often flat list or list of dicts. handle list of Tuples
    p0, p1 = poly[0], poly[1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    angle = math.degrees(math.atan2(dy, dx))
    # Normalize to 0, 90, 180, 270 for PyMuPDF textbox support
    return int((round(angle / 90) * 90) % 360)


def apply_translations(
    doc: fitz.Document,
    translations: List[TranslationPlan],
    *,
    fontname: str = "helv",
    fontsize: float = 11,
    mask_regions_by_page: Optional[Dict[int, Any]] = None,
    verbose: bool = False,
) -> None:
    """
    Overlay mode with precision vector removal (Step 9) and style sampling (Step 10).
    mask_regions_by_page: dict[page_index -> List[MaskRegion]] built from DocInt words.
    If supplied, inpainting uses glyph-accurate masking (A–D improvements).
    """
    by_page: Dict[int, List[TranslationPlan]] = {}
    for t in translations:
        by_page.setdefault(t.container.page_index, []).append(t)

    _vprint(verbose, f"[APPLY] Dispatching {len(translations)} containers across {len(by_page)} pages")

    # Map fontname -> (fitz.Font, Optional[fontfile_path])
    font_resource_cache: Dict[str, Tuple[fitz.Font, Optional[str]]] = {}
    # Track which fonts are registered on which page indices
    page_font_registry: Dict[int, Set[str]] = {}

    for page_index, page_plans in by_page.items():
        page = doc[page_index]
        _vprint(verbose, f"[APPLY][PAGE {page_index}] Processing {len(page_plans)} containers")
        
        if page_index not in page_font_registry:
            page_font_registry[page_index] = set()

        # Step 12: Hybrid/Raster Routing
        raster_plans = [p for p in page_plans if not p.container.original_spans]
        vector_plans = [p for p in page_plans if p.container.original_spans]

        if raster_plans:
            # 11.1 Render to Raster
            from scripts.layout.raster_processor import render_page_to_pixmap, inpaint_containers, draw_raster_overlay
            _vprint(verbose, f"[RASTER][PAGE {page_index}] Inpainting {len(raster_plans)} regions...")

            pix = render_page_to_pixmap(page, dpi=300)
            pref = fitz.Rect(page.rect)
            sx = pix.width / pref.width
            sy = pix.height / pref.height

            # 11.3 & 11.4 Mask & Inpaint (use word-level MaskRegions when available)
            raster_conts = [p.container for p in raster_plans]
            page_mask_regions = (mask_regions_by_page or {}).get(page_index, None)
            inpainted_pix = inpaint_containers(
                pix,
                raster_conts,
                sx=sx,
                sy=sy,
                mask_regions=page_mask_regions,
                verbose=verbose,
            )

            # 11.5/12 Overlay background
            draw_raster_overlay(doc, page_index, inpainted_pix)

        # 1) Precise Text Removal (Step 9 / 12)
        # Even on hybrid pages, we remove vector strokes for mapped containers
        # 1) Precise Text Removal (Step 9 / 12)
        # Even on hybrid pages, we remove vector strokes for mapped containers
        if vector_plans:
            # Create a 72-DPI base pixmap to sample perimeter colors 
            # so we can dynamically color the redaction fill without causing white boxes.
            bg_pix = page.get_pixmap(dpi=72)
            for plan in vector_plans:
                span_rects = [fitz.Rect(s.rect) for s in plan.container.original_spans]
                remove_text(page, span_rects, bg_pix, verbose=verbose)
        
        # If inpainting didn't happen (RASTER containers but no OpenCV), use bbox fallback
        if raster_plans:
            try:
                import cv2
            except ImportError:
                cv2 = None

            if cv2 is None:
                for plan in raster_plans:
                    rect = fitz.Rect(plan.container.bbox)
                    page.add_redact_annot(_safe_rect(page, rect, pad=0.1))

        # This removes the original vector text content
        page.apply_redactions()

        # 2) Typesetting & Style Fidelity (Step 10)
        for plan in page_plans:
            # 10.3 Rotation: preserve polygon angles
            if plan.container.polygon:
                plan.rendering_intent.rotation = _derive_rotation(plan.container.polygon)

            # 10.2 Style fidelity: Sample color/style from original spans
            if plan.container.original_spans:
                s0 = plan.container.original_spans[0]
                plan.rendering_intent.color = _int_to_rgb(s0.color)
                if plan.rendering_intent.font_size_start <= 1e-3:
                    plan.rendering_intent.font_size_start = s0.size
            
            # Fallback: if we still have no font size (raster or unmatched span),
            # estimate from the PDF's text layer at the container's bbox location.
            if plan.rendering_intent.font_size_start <= 1e-3:
                estimated = _estimate_fontsize_for_rect(page, fitz.Rect(plan.container.bbox), verbose=verbose)
                plan.rendering_intent.font_size_start = estimated if estimated else 10.0

            # 10.1 Font Strategy (Unicode Fallback)
            tgt_text = plan.final_rendered_text
            eff_fontname = plan.rendering_intent.font_name or fontname
            
            from scripts.layout.typesetter import _looks_cjk
            if _looks_cjk(tgt_text):
                eff_fontname = "notosans"
            
            plan.rendering_intent.font_name = eff_fontname
            fn = plan.rendering_intent.font_name

            # Cache the font object and file globally for the doc
            if fn not in font_resource_cache:
                eff_name, ff = _pick_font(tgt_text, fn, verbose=verbose)
                mf = _get_measure_font(eff_name, ff, verbose=verbose)
                font_resource_cache[fn] = (mf, ff)

            # REGISTER on the page every time we move to a new page or usage
            mf, ff = font_resource_cache[fn]
            if fn not in page_font_registry[page_index]:
                _ensure_page_font(page, fn, ff, verbose=verbose)
                page_font_registry[page_index].add(fn)

            # Invoke Step 8: Kind-aware Typesetting
            # Pass only the measure fonts part to typeset_and_insert
            measure_only_map = {k: v[0] for k, v in font_resource_cache.items()}
            typeset_and_insert(page, fitz.Rect(plan.container.bbox), plan, measure_only_map)

    _vprint(verbose, "[APPLY] Completed typesetting for all containers.")


# -----------------------------
# Paragraph-aware merge / unmerge
# -----------------------------
_LINE_TAG_RE = re.compile(r"\[L(\d+)\](.*?)\[/L\1\]", re.DOTALL)


@dataclass
class _MergeEntry:
    """Bookkeeping for one item sent into translate_blocks after merging."""
    original_indices: List[int]          # indices into the pre-merge container list
    line_count: int                      # how many lines were merged (1 = passthrough)
    merged_container: ContainerRef       # the (possibly synthetic) container


def _merge_paragraph_groups(
    containers: List[ContainerRef],
    *,
    verbose: bool = False,
) -> Tuple[List[ContainerRef], List[_MergeEntry]]:
    """
    Group containers that share a paragraph_group_id and merge their text
    with [L1]...[/L1][L2]...[/L2] delimiters.

    Singletons (group size 1) and items with paragraph_group_id=None pass
    through unchanged.

    Returns:
        merged_containers – smaller list of ContainerRefs ready for translation
        merge_map         – one _MergeEntry per merged item (for unmerging later)
    """
    # Build ordered groups preserving first-seen order
    group_order: List[Optional[str]] = []          # ordered unique group IDs (None = ungrouped)
    group_items: Dict[Optional[str], List[int]] = {}  # gid -> list of indices
    seen_gids: set = set()

    for idx, c in enumerate(containers):
        gid = c.paragraph_group_id
        if gid is None:
            # Each ungrouped item is its own "group" – use a unique key
            key = f"__ungrouped_{idx}"
            group_order.append(key)
            group_items[key] = [idx]
        else:
            if gid not in seen_gids:
                group_order.append(gid)
                seen_gids.add(gid)
            group_items.setdefault(gid, []).append(idx)

    # Post-process groups: split if there's a large vertical gap between lines
    final_merged_containers: List[ContainerRef] = []
    final_merge_map: List[_MergeEntry] = []
    merged_count = 0

    for gid in group_order:
        indices = group_items[gid]
        if len(indices) <= 1:
            # Singleton – pass through
            c = containers[indices[0]]
            final_merged_containers.append(c)
            final_merge_map.append(_MergeEntry(
                original_indices=indices,
                line_count=1,
                merged_container=c,
            ))
            continue

        # Split group if vertical gaps are too large or line count is massive
        sub_groups: List[List[int]] = []
        current_sub: List[int] = [indices[0]]
        for i in range(1, len(indices)):
            prev = containers[indices[i-1]]
            curr = containers[indices[i]]
            # Vertical gap heuristic: 1.5x line height or 20pt, whichever is smaller context
            # (or if they are on different pages)
            prev_h = prev.bbox[3] - prev.bbox[1]
            gap = curr.bbox[1] - prev.bbox[3]
            
            # Massive gap or page break or too many lines (to keep context manageable)
            if curr.page_index != prev.page_index or gap > max(15.0, prev_h * 1.5) or len(current_sub) >= 20:
                sub_groups.append(current_sub)
                current_sub = [indices[i]]
            else:
                current_sub.append(indices[i])
        sub_groups.append(current_sub)

        for sub_indices in sub_groups:
            if len(sub_indices) == 1:
                # Became singleton
                c = containers[sub_indices[0]]
                final_merged_containers.append(c)
                final_merge_map.append(_MergeEntry(
                    original_indices=sub_indices,
                    line_count=1,
                    merged_container=c,
                ))
            else:
                # Multi-line
                parts: List[str] = []
                bboxes: List[Tuple[float, float, float, float]] = []
                for line_num, ci in enumerate(sub_indices, start=1):
                    parts.append(f"[L{line_num}]{containers[ci].text}[/L{line_num}]")
                    bboxes.append(containers[ci].bbox)

                merged_text = "".join(parts)
                merged_bbox = _union_bbox(bboxes)
                first = containers[sub_indices[0]]
                virtual = ContainerRef(
                    page_index=first.page_index,
                    bbox=merged_bbox,
                    text=merged_text,
                    kind=first.kind,
                    polygon=first.polygon,
                    reading_key=first.reading_key,
                    style_hints=first.style_hints,
                    original_spans=[],
                    paragraph_group_id=first.paragraph_group_id,
                )
                final_merged_containers.append(virtual)
                final_merge_map.append(_MergeEntry(
                    original_indices=sub_indices,
                    line_count=len(sub_indices),
                    merged_container=virtual,
                ))
                merged_count += 1

    _vprint(
        verbose,
        f"[MERGE] {len(containers)} containers -> {len(final_merged_containers)} "
        f"({merged_count} splits/merges created)",
    )
    return final_merged_containers, final_merge_map


def _unmerge_paragraph_translations(
    translations: List[ContainerTranslation],
    merge_map: List[_MergeEntry],
    original_containers: List[ContainerRef],
    *,
    verbose: bool = False,
) -> List[ContainerTranslation]:
    """
    Expand merged translations back to one ContainerTranslation per original
    container using [L1]...[/L1] delimiter parsing.

    Fallback strategy when delimiters are missing/corrupted:
        1) Try regex parse for all N expected line tags
        2) If partial delimiters found, use them for matched lines and
           assign remaining text proportionally
        3) Final fallback: assign the full translated text to every line
    """
    expanded: List[ContainerTranslation] = []

    for entry, trans in zip(merge_map, translations):
        if entry.line_count == 1:
            # Singleton – rewire to original container and pass through
            orig_c = original_containers[entry.original_indices[0]]
            expanded.append(ContainerTranslation(
                container=orig_c,
                translated_text=trans.translated_text,
            ))
            continue

        # Multi-line: parse delimiters
        raw = trans.translated_text
        matches = _LINE_TAG_RE.findall(raw)
        parsed: Dict[int, str] = {}  # 1-indexed line_num -> text
        for num_str, text in matches:
            parsed[int(num_str)] = text.strip()

        n = entry.line_count

        if len(parsed) == n and all(i in parsed for i in range(1, n + 1)):
            # Happy path: all delimiters found
            for line_num, ci in enumerate(entry.original_indices, start=1):
                expanded.append(ContainerTranslation(
                    container=original_containers[ci],
                    translated_text=parsed[line_num],
                ))
            _vprint(verbose, f"[UNMERGE] group gid={entry.merged_container.paragraph_group_id}: "
                    f"parsed {n}/{n} line tags OK")
        else:
            # Fallback: delimiters missing or corrupted
            # Strip ANY remaining tags (even unpaired/malformed ones) before splitting
            # to prevent [L5] etc. from leaking into the final text.
            clean = re.sub(r"\[/?L\d+\]", "", raw).strip()
            if not clean:
                clean = raw  # if stripping removed everything, keep raw
            
            _vprint(verbose, f"[UNMERGE][FALLBACK] group gid={entry.merged_container.paragraph_group_id}: "
                    f"expected {n} tags, found {len(parsed)}. Using proportional split.")

            # Proportional split by source text length
            src_lengths = [len(original_containers[ci].text) for ci in entry.original_indices]
            total_src = sum(src_lengths) or 1
            trans_len = len(clean)

            offset = 0
            for line_idx, ci in enumerate(entry.original_indices):
                proportion = src_lengths[line_idx] / total_src
                chunk_len = int(round(proportion * trans_len))
                # Last line gets the remainder to avoid off-by-one
                if line_idx == len(entry.original_indices) - 1:
                    chunk = clean[offset:]
                else:
                    chunk = clean[offset:offset + chunk_len]
                offset += chunk_len

                expanded.append(ContainerTranslation(
                    container=original_containers[ci],
                    translated_text=chunk.strip(),
                ))

    _vprint(verbose, f"[UNMERGE] Expanded {len(translations)} merged items -> {len(expanded)} translations")
    return expanded


def _normalize_translate_blocks_result(result) -> Tuple[List, Dict[str, Any]]:
    """
    Accept multiple return shapes from services.translation_service.translate_blocks
    """
    diagnostics: Dict[str, Any] = {}

    if isinstance(result, tuple) and len(result) >= 1:
        translations = result[0]
        if len(result) >= 2 and isinstance(result[1], dict):
            diagnostics = result[1]
        return translations, diagnostics

    if isinstance(result, dict):
        translations = result.get("translations", [])
        diagnostics = result.get("diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        return translations, diagnostics

    return result, diagnostics


def _translate_blocks_with_optional_diagnostics(
    blocks: List[ContainerRef],
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[ContainerTranslation], Dict[str, Any]]:
    """
    Calls translation_service.translate_blocks in a backward-compatible way.
    """
    # ... (rest of the logic stays similar but uses ContainerRef/ContainerTranslation)
    # I'll keep the inspect.signature part for robustness
    try:
        sig = inspect.signature(translate_blocks)
        params = sig.parameters
    except Exception:
        params = {}

    kwargs: Dict[str, Any] = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "verbose": verbose,
        "return_diagnostics": True
    }
    
    # Filter kwargs based on what translate_blocks actually accepts
    final_kwargs = {k: v for k, v in kwargs.items() if k in params or not params}

    _vprint(verbose, f"[TRANSLATE] Calling translate_blocks with {len(blocks)} chunks")
    raw = translate_blocks(blocks, **final_kwargs)
    translations, diagnostics = _normalize_translate_blocks_result(raw)
    
    return translations, diagnostics


def translate_pdf_bytes_pipeline(
    pdf_bytes: bytes, 
    *, 
    source_lang: Optional[str] = None, 
    target_lang: Optional[str] = None, 
    verbose: bool = False,
    filename: str = "document.pdf"
) -> bytes:
    """
    Step 7 & 8: High-fidelity container-based translation pipeline.
    """
    total_t0 = time.perf_counter()
    extraction_time = translation_time = apply_time = save_time = 0.0
    diagnostics: Dict[str, Any] = {}
    errors = 0

    _vprint(verbose, f"[PIPELINE] Starting PDF Translation | {source_lang} -> {target_lang}...")

    logger = get_logger()
    logger.start_file_session(filename)

    # Stage 1: Container-first Extraction (Step 7.1)
    t0 = time.perf_counter()
    doc, orig_containers, metadata = extract_all_blocks(pdf_bytes, verbose=verbose)
    extraction_time = time.perf_counter() - t0
    _vprint(verbose, f"[PIPELINE] Extracted {len(orig_containers)} containers in {extraction_time:.2f}s.")

    # Stage 2: Normalization & Policy Attribution (Step 7.3 / 8.3)
    containers_to_translate = []
    translate_idx_map = []
    norm_states = []
    policies = []

    # Track placeholder counters per paragraph group to ensure uniqueness across full context
    group_counters = {} 
    group_all_placeholders = {} # gid -> {ph: token}

    for i, c in enumerate(orig_containers):
        # 7.2 Protected-token integrity (part of normalization)
        gid = c.paragraph_group_id or f"__auto_{i}"
        start_ph = group_counters.get(gid, 0)
        
        norm_text, state, next_ph = apply_normalization_pipeline(c.text, start_counter=start_ph)
        group_counters[gid] = next_ph
        
        # Aggregate placeholders for restoration (so migration between lines within a group is fine)
        if gid not in group_all_placeholders:
            group_all_placeholders[gid] = {}
        group_all_placeholders[gid].update(state.placeholders)
        
        # 8.3 Kind-aware classification
        policy = classify_container(c)
        
        norm_states.append(state)
        policies.append(policy)
        
        if policy != TranslationPolicy.SKIP:
            # Create normalized clone for translation — must preserve paragraph_group_id
            # so translation_service can build the per-paragraph context window.
            c_norm = ContainerRef(
                page_index=c.page_index,
                bbox=c.bbox,
                text=norm_text,
                kind=c.kind,
                polygon=c.polygon,
                style_hints=c.style_hints,
                reading_key=c.reading_key,
                original_spans=c.original_spans,
                paragraph_group_id=c.paragraph_group_id,  # ← crucial for LLM1 context
            )
            containers_to_translate.append(c_norm)
            translate_idx_map.append(i)

    # Stage 3: High-context Translation (Step 7.1 / 7.2)
    try:
        # 3a. Paragraph-aware merge: group lines with shared paragraph_group_id
        #     into [L1]...[/L1][L2]...[/L2] delimited text so the translator
        #     sees full paragraph context.
        merged_containers, merge_map = _merge_paragraph_groups(
            containers_to_translate, verbose=verbose
        )

        t1 = time.perf_counter()
        merged_translations, diagnostics = _translate_blocks_with_optional_diagnostics(
            merged_containers,
            source_lang=source_lang, 
            target_lang=target_lang, 
            verbose=verbose
        )
        translation_time = time.perf_counter() - t1

        # 3b. Unmerge: expand delimited translations back to per-line results
        translations = _unmerge_paragraph_translations(
            merged_translations, merge_map, containers_to_translate, verbose=verbose
        )

        # Stage 4: Restore & Map to Plans (Step 7.2 integrity)
        trans_idx = 0
        plans = []
        from scripts.text_normalization.normalizer import NormalizationState

        for i, c in enumerate(orig_containers):
            policy = policies[i]
            norm_state = norm_states[i]
            gid = c.paragraph_group_id or f"__auto_{i}"
            
            if policy == TranslationPolicy.SKIP:
                trans_text = c.text
            else:
                trans_text = translations[trans_idx].translated_text
                trans_idx += 1
            
            # Step 7.2: Final restoration of placeholders
            # Use aggregated placeholders for the group so placeholders migrated by LLM are still restored
            combined_placeholders = group_all_placeholders.get(gid, norm_state.placeholders)
            # Create a shallow state for restoration that covers the whole paragraph group
            restoration_state = NormalizationState(original_text=norm_state.original_text, placeholders=combined_placeholders)
            
            final_txt = restore_protected_tokens(trans_text, restoration_state)
            
            # Build rendering intent (Step 8 basics)
            intent = RenderingIntent(
                font_name="helv", # Default, typesetter will refine from style spans
                font_size_start=0.0, # typesetter will estimate
                alignment=0 # internal default
            )
            
            plan = TranslationPlan(
                container=c,
                normalized_source_text=containers_to_translate[trans_idx-1].text if policy != TranslationPolicy.SKIP else c.text,
                protected_tokens_map=norm_state.placeholders,
                translated_text=trans_text,
                final_rendered_text=final_txt,
                rendering_intent=intent,
                policy=policy
            )
            plans.append(plan)

        # --- Logging Stage ---
        # Build mapping to merged diagnostics
        # original_idx -> merged_idx
        orig_to_merged = {}
        for midx, entry in enumerate(merge_map):
            for oidx_in_subset in entry.original_indices:
                orig_idx = translate_idx_map[oidx_in_subset]
                orig_to_merged[orig_idx] = midx

        # merged_idx -> debug_info
        llm1_map = {p["index"]: p for p in diagnostics.get("llm1_pairs", [])}
        llm2_map = {p["index"]: p for p in diagnostics.get("llm2_pairs", [])}

        for i, c in enumerate(orig_containers):
            policy = policies[i]
            plan = plans[i]
            
            if policy == TranslationPolicy.SKIP:
                logger.log_entry(
                    source_text=c.text,
                    skipped=True,
                    skip_reason="Policy SKIP (mostly non-textual or layout restriction)"
                )
            else:
                midx = orig_to_merged.get(i)
                merged_src = ""
                mt_text = ""
                llm1_text = ""
                llm2_text = ""
                gloss_hit = ""
                
                if midx is not None:
                    # Delimited versions from diagnostics
                    d1 = llm1_map.get(midx, {})
                    d2 = llm2_map.get(midx, {})
                    merged_src = d1.get("original") or d2.get("original") or ""
                    mt_text = d1.get("mt", "")
                    llm1_text = d1.get("translated", "")
                    llm2_text = d2.get("translated", "")
                    gloss_hit = d2.get("glossary_hit", "")

                logger.log_entry(
                    source_text=c.text,
                    paragraph_group=merged_src,
                    inline_blocks=list(plan.protected_tokens_map.keys()) if plan.protected_tokens_map else None,
                    manual_translation=mt_text,
                    llm1_translation=llm1_text,
                    llm2_translation=llm2_text,
                    glossary_term=gloss_hit,
                    final_text=plan.final_rendered_text
                )

        _print_translation_samples(orig_containers, plans, diagnostics, verbose=verbose)

        # Stage 5: Kind-aware Typesetting (Step 8)
        _vprint(verbose, "[PIPELINE] Applying kind-aware typesetting...")
        t2 = time.perf_counter()
        apply_translations(
            doc,
            plans,
            mask_regions_by_page=metadata.get("mask_regions_by_page"),
            verbose=verbose,
        )
        apply_time = time.perf_counter() - t2

        t3 = time.perf_counter()
        out_bytes = doc.tobytes(deflate=True, garbage=4)
        save_time = time.perf_counter() - t3

        total_time = time.perf_counter() - total_t0
        _print_timing_report(
            verbose=verbose,
            chunks_processed=len(orig_containers),
            extraction_time=extraction_time,
            translation_time=translation_time,
            apply_time=apply_time,
            save_time=save_time,
            total_time=total_time,
            diagnostics=diagnostics,
            errors=errors,
        )

        logger.log_general_insights(
            f"SUMMARY:\n"
            f"  - Total chunks processed: {len(orig_containers)}\n"
            f"  - Extraction time: {extraction_time:.2f}s\n"
            f"  - Translation time: {translation_time:.2f}s\n"
            f"  - Rendering time: {apply_time:.2f}s\n"
            f"  - Total end-to-end time: {total_time:.2f}s\n"
            f"  - Pipeline errors: {errors}\n"
            f"  - Source Language: {source_lang}\n"
            f"  - Target Language: {target_lang}\n"
        )
        return out_bytes

    except Exception as e:
        errors += 1
        _vprint(verbose, f"[PIPELINE][FATAL] {e!r}")
        total_time = time.perf_counter() - total_t0
        _print_timing_report(
            verbose=verbose,
            chunks_processed=len(orig_containers) if 'orig_containers' in locals() else 0,
            extraction_time=extraction_time,
            translation_time=translation_time,
            apply_time=apply_time,
            save_time=save_time,
            total_time=total_time,
            diagnostics=diagnostics if 'diagnostics' in locals() else None,
            errors=errors,
        )
        raise
    finally:
        if 'doc' in locals() and doc:
            try:
                doc.close()
                _vprint(verbose, "[PIPELINE] Pipeline completed! (doc closed)")
            except Exception:
                pass


if __name__ == "__main__":
    path = r"C:\Users\AdityaPathak\Downloads\AdityaPathak_Jan2026 - Copy.pdf"
    with open(path, "rb") as f:
        pdf_bytes = f.read()

    # toggle verbose logs here for local runs
    verbose = True

    doc, blocks = extract_all_blocks(pdf_bytes, verbose=verbose)
    try:
        print(f"Extracted blocks: {len(blocks)}")
        for b in blocks[:80]:
            print(f"[p{b.page_index}] {b.bbox} :: {b.text}")
    finally:
        doc.close()




### DEAD CODE: old paragraph-level extraction (replaced by line-level extraction for better fidelity and control)

# def _extract_items_from_docintel_result(doc: fitz.Document, analyze_result, *, verbose: bool = False) -> List[_ExtractItem]: ### paragraph level extraction
#     """
#     Extract semantic translation units:
#         - table cells (from analyze_result.tables[].cells[])
#         - paragraphs (from analyze_result.paragraphs[])
#     Paragraphs overlapping table-cell spans are skipped to avoid duplication.
#     """
#     items: List[_ExtractItem] = []

#     if not getattr(analyze_result, "pages", None):
#         _vprint(verbose, "[EXTRACT] analyze_result.pages is empty; nothing to extract")
#         return items

#     # precompute per-page scale factors (DI -> fitz)
#     page_scale: Dict[int, Tuple[float, float]] = {}
#     _vprint(verbose, f"[EXTRACT] Computing page scale factors for {len(analyze_result.pages)} DI pages")
#     for page_obj in analyze_result.pages:
#         page_number_1 = getattr(page_obj, "page_number", None) or getattr(page_obj, "pageNumber", None)
#         if not page_number_1:
#             _vprint(verbose, "[EXTRACT][PAGE-SCALE] Skipping DI page without page_number")
#             continue
#         page_index = int(page_number_1) - 1
#         if page_index < 0 or page_index >= doc.page_count:
#             _vprint(verbose, f"[EXTRACT][PAGE-SCALE] Skipping DI page_number={page_number_1}; out of fitz range")
#             continue

#         fitz_page = doc[page_index]
#         fitz_w = float(fitz_page.rect.width)
#         fitz_h = float(fitz_page.rect.height)

#         di_w = float(getattr(page_obj, "width", 0.0) or 0.0)
#         di_h = float(getattr(page_obj, "height", 0.0) or 0.0)

#         if di_w <= 0 or di_h <= 0:
#             sx = sy = 72.0
#             _vprint(verbose, f"[EXTRACT][PAGE-SCALE] p{page_index}: missing DI dimensions, fallback sx=sy=72.0")
#         else:
#             sx = fitz_w / di_w
#             sy = fitz_h / di_h
#             _vprint(
#                 verbose,
#                 f"[EXTRACT][PAGE-SCALE] p{page_index}: DI({di_w:.3f}x{di_h:.3f}) -> "
#                 f"PDF({fitz_w:.3f}x{fitz_h:.3f}) => sx={sx:.6f}, sy={sy:.6f}",
#             )

#         page_scale[page_index] = (sx, sy)

#     table_stats: Dict[str, int] = {
#         "seen": 0,
#         "added": 0,
#         "skipped_empty_text": 0,
#         "skipped_no_valid_page_or_bbox": 0,
#         "skipped_tiny_area": 0,
#     }
#     para_stats: Dict[str, int] = {
#         "seen": 0,
#         "added": 0,
#         "skipped_empty_text": 0,
#         "skipped_table_span_overlap": 0,
#         "skipped_no_valid_page_or_bbox": 0,
#         "skipped_tiny_area": 0,
#     }

#     # --- 1) TABLE CELLS ---
#     table_cell_span_ranges: List[Tuple[int, int]] = []
#     all_tables = (getattr(analyze_result, "tables", None) or [])
#     _vprint(verbose, f"[EXTRACT][TABLE_CELLS] DI tables found: {len(all_tables)}")

#     for ti, table in enumerate(all_tables, start=1):
#         cells = (getattr(table, "cells", None) or [])
#         _vprint(verbose, f"[EXTRACT][TABLE_CELLS] Table #{ti}: cells={len(cells)}")
#         for ci, cell in enumerate(cells, start=1):
#             table_stats["seen"] += 1

#             text = (getattr(cell, "content", "") or "").strip()
#             if not text:
#                 table_stats["skipped_empty_text"] += 1
#                 continue

#             spans = _span_ranges(getattr(cell, "spans", None) or [])
#             if spans:
#                 table_cell_span_ranges.extend(spans)

#             brs = getattr(cell, "bounding_regions", None) or []
#             # A cell can have multiple bounding regions; union them (usually 1)
#             cell_bboxes: List[Tuple[float, float, float, float]] = []
#             page_index: Optional[int] = None

#             for br in brs:
#                 pno = getattr(br, "page_number", None) or getattr(br, "pageNumber", None)
#                 if not pno:
#                     continue
#                 pi = int(pno) - 1
#                 if pi < 0 or pi >= doc.page_count:
#                     continue
#                 poly = getattr(br, "polygon", None)
#                 bb = _poly_to_bbox(poly)
#                 if not bb:
#                     continue
#                 sx, sy = page_scale.get(pi, (72.0, 72.0))
#                 bb = _scale_bbox(bb, sx, sy)
#                 cell_bboxes.append(bb)
#                 page_index = pi

#             if page_index is None or not cell_bboxes:
#                 table_stats["skipped_no_valid_page_or_bbox"] += 1
#                 if verbose and ci % 25 == 0:
#                     _vprint(verbose, f"[EXTRACT][TABLE_CELLS] Table #{ti} cell #{ci}: skipped (no valid page/bbox)")
#                 continue

#             bbox = _union_bbox(cell_bboxes)
#             if fitz.Rect(*bbox).get_area() < 5:
#                 table_stats["skipped_tiny_area"] += 1
#                 continue

#             items.append(
#                 _ExtractItem(
#                     page_index=page_index,
#                     bbox=bbox,
#                     text=text,
#                     min_offset=_min_span_offset(getattr(cell, "spans", None) or []),
#                     spans=spans,
#                     source_kind="table_cell",
#                 )
#             )
#             table_stats["added"] += 1
#             _log_every_five_extracted(items, verbose)

#     _print_skip_summary(verbose, "TABLE_CELLS", table_stats)

#     # --- 2) PARAGRAPHS (skip those that overlap table spans) ---
#     paragraphs = (getattr(analyze_result, "paragraphs", None) or [])
#     _vprint(verbose, f"[EXTRACT][PARAGRAPHS] DI paragraphs found: {len(paragraphs)}")

#     for pi_idx, para in enumerate(paragraphs, start=1):
#         para_stats["seen"] += 1

#         text = (getattr(para, "content", "") or "").strip()
#         if not text:
#             para_stats["skipped_empty_text"] += 1
#             continue

#         spans = _span_ranges(getattr(para, "spans", None) or [])
#         if spans and table_cell_span_ranges and _any_overlap(spans, table_cell_span_ranges):
#             # avoid duplicating table text (cells already extracted)
#             para_stats["skipped_table_span_overlap"] += 1
#             continue

#         brs = getattr(para, "bounding_regions", None) or []
#         para_bboxes: List[Tuple[float, float, float, float]] = []
#         page_index: Optional[int] = None

#         for br in brs:
#             pno = getattr(br, "page_number", None) or getattr(br, "pageNumber", None)
#             if not pno:
#                 continue
#             pi = int(pno) - 1
#             if pi < 0 or pi >= doc.page_count:
#                 continue
#             poly = getattr(br, "polygon", None)
#             bb = _poly_to_bbox(poly)
#             if not bb:
#                 continue
#             sx, sy = page_scale.get(pi, (72.0, 72.0))
#             bb = _scale_bbox(bb, sx, sy)
#             para_bboxes.append(bb)
#             page_index = pi

#         if page_index is None or not para_bboxes:
#             para_stats["skipped_no_valid_page_or_bbox"] += 1
#             if verbose and pi_idx % 50 == 0:
#                 _vprint(verbose, f"[EXTRACT][PARAGRAPHS] Paragraph #{pi_idx}: skipped (no valid page/bbox)")
#             continue

#         bbox = _union_bbox(para_bboxes)
#         if fitz.Rect(*bbox).get_area() < 5:
#             para_stats["skipped_tiny_area"] += 1
#             continue

#         items.append(
#             _ExtractItem(
#                 page_index=page_index,
#                 bbox=bbox,
#                 text=text,
#                 min_offset=_min_span_offset(getattr(para, "spans", None) or []),
#                 spans=spans,
#                 source_kind="paragraph",
#             )
#         )
#         para_stats["added"] += 1
#         _log_every_five_extracted(items, verbose)

#     _print_skip_summary(verbose, "PARAGRAPHS", para_stats)

#     # Sort best-effort reading order:
#     # 1) by min span offset (global reading order)
#     # 2) fallback geometry
#     _vprint(verbose, f"[EXTRACT] Sorting {len(items)} extracted items into reading order")
#     items.sort(
#         key=lambda it: (
#             it.min_offset if it.min_offset is not None else 10**18,
#             it.page_index,
#             it.bbox[1],
#             it.bbox[0],
#         )
#     )

#     if verbose:
#         _vprint(verbose, "[EXTRACT] First 5 items after sort:")
#         for i, it in enumerate(items[:5], start=1):
#             _vprint(
#                 verbose,
#                 f"  #{i}: source={it.source_kind}, page={it.page_index}, bbox={_fmt_bbox(it.bbox)}, "
#                 f"min_offset={it.min_offset}, text={_preview_text(it.text)}"
#             )
#         _vprint(verbose, f"[EXTRACT] Total extracted items: {len(items)}")

#     return items