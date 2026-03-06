import os
import sys
import time

# Add project root to sys.path so 'scripts' package is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from scripts.translator_service_word import TranslatorService


class DocxFormatter:
    """Translate and format a DOCX file using batch Azure Translator calls."""

    def __init__(self, from_lang: str, to_lang: str):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.translator = TranslatorService()
        self.namespace = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
        }

    # ------------------------------------------------------------------
    # Helper: detect if a run is inside a hyperlink element
    # ------------------------------------------------------------------
    @staticmethod
    def _is_hyperlink_run(run) -> bool:
        """
        Returns True if the run lives inside a <w:hyperlink> element.
        Hyperlink runs contain the visible link text (URL label), which
        should NOT be translated so links remain functional.
        """
        parent = run._element.getparent()
        return parent is not None and parent.tag == qn("w:hyperlink")

    # ------------------------------------------------------------------
    # Helper: collect translatable runs from a paragraph
    # ------------------------------------------------------------------
    def _collect_paragraph_runs(self, paragraph, run_refs: list, texts: list):
        """Append (run, text) pairs from a paragraph to the shared lists.
        Runs that are part of a hyperlink are intentionally skipped.
        """
        for run in paragraph.runs:
            if run.text.strip() and not self._is_hyperlink_run(run):
                run_refs.append(run)
                texts.append(run.text)

    # ------------------------------------------------------------------
    # Helper: apply post-translation formatting to a single paragraph
    # ------------------------------------------------------------------
    @staticmethod
    def _format_paragraph_structure(paragraph, skip_font_enforcement: bool = False):
        """
        Apply structural/visual formatting to a paragraph's runs AFTER text
        has already been translated and written back.

        - Resets character spacing to normal (avoids layout issues with
          CJK / wide-character scripts).
        - Preserves the original paragraph alignment.
        - Font-size enforcement is kept commented out; uncomment if needed.
        """
        original_alignment = paragraph.alignment
        for run in paragraph.runs:
            # Character spacing: set to normal (0 = no extra spacing)
            try:
                if run.font is not None and run.font._element is not None:
                    run.font._element.set("spc", "0")
            except Exception as e:
                print(f"[format] Error adjusting character spacing: {e}")

            # Uncomment the block below to enforce a fixed font size:
            # if run.font is not None and not skip_font_enforcement:
            #     run.font.size = Pt(7)

        # Restore alignment (python-docx sometimes resets it)
        if original_alignment is not None:
            paragraph.alignment = original_alignment

    # ------------------------------------------------------------------
    # Helper: table cell — disable no-wrap so text wraps after translation
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_text_wrap(cell):
        """Remove noWrap flag so translated (wider) text wraps in the cell."""
        tc_pr = cell._tc.get_or_add_tcPr()
        no_wrap = tc_pr.find(qn("w:noWrap"))
        if no_wrap is not None:
            tc_pr.remove(no_wrap)

    # ------------------------------------------------------------------
    # Pass 1 – Collect: gather ALL translatable runs across the document
    # ------------------------------------------------------------------
    def _collect_all_runs(self, doc) -> tuple:
        """
        Walk the entire document and return:
            run_refs – list of live run objects (references into the doc)
            texts    – list of their current text strings (same order)

        Covers: body paragraphs, tables, inline shapes (textboxes/charts),
                section headers & footers.
        """
        run_refs: list = []
        texts:    list = []

        # Body paragraphs
        for paragraph in doc.paragraphs:
            self._collect_paragraph_runs(paragraph, run_refs, texts)

        # Tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        self._collect_paragraph_runs(paragraph, run_refs, texts)

        # Inline shapes: textboxes
        for shape in doc.inline_shapes:
            if hasattr(shape, "text_frame") and shape.text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    self._collect_paragraph_runs(paragraph, run_refs, texts)
            # Charts (InlineShape doesn't have has_chart; check via element type)
            if hasattr(shape, "has_chart") and shape.has_chart:
                chart = shape.chart
                if chart.has_title and chart.chart_title.text_frame:
                    for paragraph in chart.chart_title.text_frame.paragraphs:
                        self._collect_paragraph_runs(paragraph, run_refs, texts)
                for axis in (chart.category_axis, chart.value_axis):
                    if axis and axis.has_title and axis.axis_title.text_frame:
                        for paragraph in axis.axis_title.text_frame.paragraphs:
                            self._collect_paragraph_runs(paragraph, run_refs, texts)

        # Headers and footers
        # We collect from ALL sections (including linked ones) because footer
        # text (e.g. "Page", "Confidential") should be translated even when
        # the footer is shared across sections.  We use a seen-id set to avoid
        # visiting the exact same paragraph object twice.
        seen_paragraph_ids: set = set()

        def _collect_hf_paragraphs(part):
            if part is None:
                return
            for paragraph in part.paragraphs:
                pid = id(paragraph._element)
                if pid not in seen_paragraph_ids:
                    seen_paragraph_ids.add(pid)
                    self._collect_paragraph_runs(paragraph, run_refs, texts)

        for section in doc.sections:
            # Default (odd-page) header & footer
            _collect_hf_paragraphs(section.header)
            _collect_hf_paragraphs(section.footer)
            # First-page header & footer (used when "Different First Page" is enabled)
            _collect_hf_paragraphs(section.first_page_header)
            _collect_hf_paragraphs(section.first_page_footer)
            # Even-page header & footer (used when "Different Odd & Even Pages" is enabled)
            _collect_hf_paragraphs(section.even_page_header)
            _collect_hf_paragraphs(section.even_page_footer)

        return run_refs, texts

    # ------------------------------------------------------------------
    # Pass 3 – Format: apply visual formatting to the whole document
    # ------------------------------------------------------------------
    def _apply_formatting(self, doc):
        """
        Walk the document again and apply structural formatting (spacing,
        alignment, table wrap) AFTER translations have been written back.
        """
        # Body paragraphs
        for paragraph in doc.paragraphs:
            self._format_paragraph_structure(paragraph)

        # Tables: fix cell wrapping + paragraph formatting
        for table in doc.tables:
            table.autofit = False
            table.allow_autofit = False
            for row in table.rows:
                for cell in row.cells:
                    self._ensure_text_wrap(cell)
                    for paragraph in cell.paragraphs:
                        self._format_paragraph_structure(paragraph)

        # Inline shapes
        for shape in doc.inline_shapes:
            if hasattr(shape, "text_frame") and shape.text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    self._format_paragraph_structure(paragraph)

        # Headers and footers (skip font enforcement; avoid double-formatting
        # by tracking seen paragraph element ids)
        seen_fmt_ids: set = set()

        def _fmt_hf_paragraphs(part):
            if part is None:
                return
            for paragraph in part.paragraphs:
                pid = id(paragraph._element)
                if pid not in seen_fmt_ids:
                    seen_fmt_ids.add(pid)
                    self._format_paragraph_structure(paragraph, skip_font_enforcement=True)

        for section in doc.sections:
            # Default (odd-page) header & footer
            _fmt_hf_paragraphs(section.header)
            _fmt_hf_paragraphs(section.footer)
            # First-page header & footer
            _fmt_hf_paragraphs(section.first_page_header)
            _fmt_hf_paragraphs(section.first_page_footer)
            # Even-page header & footer
            _fmt_hf_paragraphs(section.even_page_header)
            _fmt_hf_paragraphs(section.even_page_footer)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_docx(self, input_doc_path: str, output_doc_path: str) -> float:
        """
        Translate and format a DOCX file.

        Strategy
        --------
        1. Load the document.
        2. COLLECT  – walk all paragraphs/tables/shapes/headers/footers and
                      gather (run_ref, text) pairs into flat lists.
        3. TRANSLATE – send all texts in ONE batch call to Azure
                       (Azure handles chunking at 100 texts per request internally).
        4. WRITE BACK – assign each translated string back to its run via the
                        stored reference. Formatting (bold, italic, font, etc.)
                        is untouched because we only modify `run.text`.
        5. FORMAT   – apply structural fixes (character spacing, table wrap,
                      paragraph alignment) now that the correct text is in place.
        6. Save.
        """
        start_time = time.time()
        print("Started DOCX formatting.")

        # Step 1 – Load
        doc = Document(input_doc_path)

        # Step 2 – Collect
        print("Collecting text runs...")
        run_refs, texts = self._collect_all_runs(doc)
        print(f"  → {len(texts)} translatable runs found.")

        # Step 3 – Batch translate: full 3-stage pipeline (MT → LLM1 → RAG+LLM2)
        print(f"Translating '{self.from_lang}' → '{self.to_lang}' via 3-stage pipeline ...")
        # translated_texts = self.translator.batch_translate_with_pipeline(
        #     texts, self.from_lang, self.to_lang
        # )
        translated_texts = self.translator.batch_translate(
            texts, self.to_lang
        )
        print(f"  → Pipeline done ({len(translated_texts)} strings).")

        # Step 4 – Write back
        print("Writing translations back to document runs...")
        success_count = 0
        for run, translated in zip(run_refs, translated_texts):
            if translated is not None and translated != run.text:
                run.text = translated
                success_count += 1
        print(f"  → {success_count} runs updated.")

        # Step 5 – Apply formatting
        print("Applying document formatting...")
        self._apply_formatting(doc)

        # Step 6 – Save
        doc.save(output_doc_path)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nDone! Document saved to: {output_doc_path}")
        print(f"Total time taken: {duration:.2f} seconds")
        return duration


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from_lang = "en"
    to_lang = "ja"
    input_doc_path = r"C:\gen_ai\document-translation\backend\data\word file\Employee_Handbook_NovaTech - Copy.docx"
    file_extension = os.path.splitext(input_doc_path)[1]
    input_doc_name = os.path.splitext(os.path.basename(input_doc_path))[0]
    output_doc_path = f"{input_doc_name}_{from_lang}_{to_lang}{file_extension}"

    formatter = DocxFormatter(from_lang, to_lang)
    formatter.process_docx(input_doc_path, output_doc_path)
