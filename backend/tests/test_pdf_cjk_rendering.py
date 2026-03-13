from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.layout.containers import (
    ContainerKind,
    ContainerRef,
    PdfSpanAttrs,
    RenderingIntent,
    TranslationPlan,
    TranslationPolicy,
)
from services.pdf_pipeline import apply_translations


def _make_plan(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    page_index: int = 0,
    kind: ContainerKind = ContainerKind.PARAGRAPH,
    with_spans: bool = False,
) -> TranslationPlan:
    spans = []
    if with_spans:
        spans = [
            PdfSpanAttrs(
                rect=bbox,
                text=text,
                font="helv",
                size=18.0,
                color=0,
                origin=(bbox[0], bbox[1] + 18.0),
                flags=0,
                ascender=0.8,
                descender=-0.2,
            )
        ]
    container = ContainerRef(
        page_index=page_index,
        bbox=bbox,
        text=text,
        kind=kind,
        original_spans=spans,
    )
    return TranslationPlan(
        container=container,
        normalized_source_text=text,
        protected_tokens_map={},
        translated_text=text,
        final_rendered_text=text,
        rendering_intent=RenderingIntent(
            font_name="helv",
            font_size_start=18.0,
            alignment=0,
            color=(0.0, 0.0, 0.0),
        ),
        policy=TranslationPolicy.TRANSLATE,
    )


def _render_has_visible_text(page: fitz.Page, bbox: tuple[float, float, float, float]) -> bool:
    pix = page.get_pixmap(dpi=144, alpha=False)
    sx = pix.width / page.rect.width
    sy = pix.height / page.rect.height
    x0, y0, x1, y1 = bbox
    clip = fitz.IRect(
        max(0, int(x0 * sx)),
        max(0, int(y0 * sy)),
        min(pix.width, int(x1 * sx)),
        min(pix.height, int(y1 * sy)),
    )
    if clip.is_empty:
        return False
    roi = pix.samples
    stride = pix.width * pix.n
    for row in range(clip.y0, clip.y1):
        start = row * stride + clip.x0 * pix.n
        end = row * stride + clip.x1 * pix.n
        row_bytes = roi[start:end]
        if any(channel < 245 for channel in row_bytes):
            return True
    return False


def _roundtrip_doc(doc: fitz.Document) -> fitz.Document:
    return fitz.open(stream=doc.tobytes(deflate=True, garbage=4, use_objstms=1), filetype="pdf")


def test_japanese_text_roundtrips_exact_unicode_and_is_visible():
    doc = fitz.open()
    doc.new_page(width=400, height=200)
    text = "検索 資料 翻訳"
    bbox = (40, 40, 320, 100)
    apply_translations(doc, [_make_plan(text, bbox)], target_lang="ja")
    out = _roundtrip_doc(doc)
    page = out[0]

    assert page.get_text("text").strip() == text
    assert _render_has_visible_text(page, bbox)
    assert any(font[1] == "ttf" and font[2] == "Type0" for font in page.get_fonts())


@pytest.mark.parametrize(
    ("text", "target_lang"),
    [
        ("简体中文 搜索 翻译", "zh-cn"),
        ("한국어 검색 번역", "ko"),
    ],
)
def test_cjk_smoke_roundtrips_for_chinese_and_korean(text: str, target_lang: str):
    doc = fitz.open()
    doc.new_page(width=400, height=200)
    bbox = (40, 40, 340, 100)
    apply_translations(doc, [_make_plan(text, bbox)], target_lang=target_lang)
    out = _roundtrip_doc(doc)
    page = out[0]

    assert page.get_text("text").strip() == text
    assert _render_has_visible_text(page, bbox)
    assert any(font[1] == "ttf" and font[2] == "Type0" for font in page.get_fonts())


def test_mixed_latin_and_japanese_across_vector_and_raster_paths():
    doc = fitz.open()
    doc.new_page(width=500, height=250)
    doc.new_page(width=500, height=250)

    plans = [
        _make_plan("Hello translated world", (40, 40, 240, 90), page_index=0, with_spans=True),
        _make_plan("検索 コピー", (260, 40, 460, 100), page_index=0),
        _make_plan("ラスターページ 翻訳", (40, 50, 320, 120), page_index=1),
    ]

    apply_translations(doc, plans, target_lang="ja")
    out = _roundtrip_doc(doc)

    page0 = out[0]
    page1 = out[1]
    extracted0 = page0.get_text("text")
    extracted1 = page1.get_text("text")

    assert "Hello translated world" in extracted0
    assert "検索 コピー" in extracted0
    assert "ラスターページ 翻訳" in extracted1
    assert _render_has_visible_text(page0, (40, 40, 240, 90))
    assert _render_has_visible_text(page0, (260, 40, 460, 100))
    assert _render_has_visible_text(page1, (40, 50, 320, 120))
    assert any(font[1] == "ttf" and font[2] == "Type0" for font in page0.get_fonts())
