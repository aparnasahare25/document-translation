"""
PDF Translation Workflow Orchestrator

This script routes a PDF through the correct translation pipeline based on its source:

    - Microsoft Word PDF  → PDF→DOCX → Translate DOCX → DOCX→PDF → Output PDF
    - Other PDFs          → Direct PDF translation via pdf_pipeline.py

Usage (standalone):
    python services/pdf_flow.py
    (edit input_pdf_path, output_pdf_path, from_lang, to_lang at the bottom)
"""

import os
import sys
import tempfile

# Ensure the backend root is on sys.path so all service/script imports resolve.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from scripts.document_classifier import PDFMetadataSourceClassifier
from scripts.pdf_to_docx import ExportPDFToDOCX
from scripts.docx_to_pdf import CreatePDFFromDOCX
from services.word_translation_service import DocxFormatter
from services.pdf_pipeline import translate_pdf_bytes_pipeline


def translate_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    from_lang: str,
    to_lang: str,
    verbose: bool = True,
) -> None:
    """
    Translate a PDF file and save the result to output_pdf_path.

    Routing logic:
        - If the classifier identifies the PDF source as 'microsoft_word':
            1. PDF  → DOCX        (Adobe PDF Services)
            2. DOCX → DOCX        (word_translation_service)
            3. DOCX → PDF         (Adobe PDF Services)
            4. Clean up temp files
        - Otherwise:
            1. Translate PDF bytes directly via pdf_pipeline.py
    """

    if not os.path.isfile(input_pdf_path):
        raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")

    # ------------------------------------------------------------------
    # Step 1: Classify the PDF source
    # ------------------------------------------------------------------
    print(f"\n[PDF FLOW] Classifying PDF: {input_pdf_path}")
    classifier = PDFMetadataSourceClassifier()
    result = classifier.classify(input_pdf_path)
    source = result.get("source", "unknown")
    confidence = result.get("confidence", 0.0)
    print(f"[PDF FLOW] Detected source: '{source}' (confidence={confidence})")

    # ------------------------------------------------------------------
    # Step 2: Route based on source
    # ------------------------------------------------------------------
    if source == "microsoft_word":
        _translate_word_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            from_lang=from_lang,
            to_lang=to_lang,
            verbose=verbose,
        )
    else:
        print(f"[PDF FLOW] Source is not Microsoft Word → using direct PDF pipeline.")
        _translate_other_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            from_lang=from_lang,
            to_lang=to_lang,
            verbose=verbose,
        )


# ----------------------------------------------------------------------
# Route A: Microsoft Word → PDF via DOCX round-trip
# ----------------------------------------------------------------------

def _translate_word_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    from_lang: str,
    to_lang: str,
    verbose: bool,
) -> None:
    """
    Full DOCX round-trip translation for Word-origin PDFs:
        PDF → temp_source.docx → temp_translated.docx → output PDF
    """
    tmp_dir = tempfile.mkdtemp(prefix="pdf_flow_")
    temp_source_docx = os.path.join(tmp_dir, "source.docx")
    temp_translated_docx = os.path.join(tmp_dir, "translated.docx")

    try:
        # Step A1: Convert input PDF → DOCX
        print(f"\n[PDF FLOW] Step A1: Converting PDF → DOCX ...")
        print(f"           Input  : {input_pdf_path}")
        print(f"           Output : {temp_source_docx}")
        ExportPDFToDOCX(input_pdf_path, temp_source_docx)
        if not os.path.isfile(temp_source_docx):
            raise RuntimeError(f"PDF→DOCX conversion failed; file not created: {temp_source_docx}")
        print(f"[PDF FLOW] PDF→DOCX conversion complete: ")

        # Step A2: Translate the DOCX
        print(f"\n[PDF FLOW] Step A2: Translating DOCX ({from_lang} → {to_lang}) ...")
        print(f"           Input  : {temp_source_docx}")
        print(f"           Output : {temp_translated_docx}")
        formatter = DocxFormatter(from_lang=from_lang, to_lang=to_lang)
        formatter.process_docx(temp_source_docx, temp_translated_docx)
        if not os.path.isfile(temp_translated_docx):
            raise RuntimeError(f"DOCX translation failed; file not created: {temp_translated_docx}")
        print(f"[PDF FLOW] Translating DOCX complete.")

        # Step A3: Convert translated DOCX → PDF
        print(f"\n[PDF FLOW] Step A3: Converting translated DOCX → PDF ...")
        print(f"           Input  : {temp_translated_docx}")
        print(f"           Output : {output_pdf_path}")
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        CreatePDFFromDOCX(temp_translated_docx, output_pdf_path)
        if not os.path.isfile(output_pdf_path):
            raise RuntimeError(f"DOCX→PDF conversion failed; file not created: {output_pdf_path}")
        print(f"[PDF FLOW] DOCX→PDF conversion complete.")

    except Exception as e:
        print(f"\n[PDF FLOW] ❌ Error during Word-PDF translation: {type(e).__name__}: {e}")

    finally:
        # Step A4: Clean up temp files
        for path in (temp_source_docx, temp_translated_docx):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    print(f"[PDF FLOW] Cleaned up temp file: {path}")
                except Exception as e:
                    print(f"[PDF FLOW][WARN] Could not delete temp file {path}: {e}")
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass  # tmp_dir might not be empty if there are leftover files

    print(f"\n[PDF FLOW] ✅ Word-origin PDF translated successfully → {output_pdf_path}")


# ----------------------------------------------------------------------
# Route B: Non-Word PDF → direct PDF pipeline
# ----------------------------------------------------------------------

def _translate_other_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    from_lang: str,
    to_lang: str,
    verbose: bool,
) -> None:
    """
    Direct PDF translation via translate_pdf_bytes_pipeline().
    """
    print(f"\n[PDF FLOW] Step B1: Reading PDF bytes from: {input_pdf_path}")
    with open(input_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    print(f"[PDF FLOW] Step B1 complete. ({len(pdf_bytes):,} bytes read)")

    print(f"\n[PDF FLOW] Step B2: Running PDF translation pipeline ({from_lang} → {to_lang}) ...")
    translated_bytes = translate_pdf_bytes_pipeline(
        pdf_bytes,
        source_lang=from_lang,
        target_lang=to_lang,
        verbose=verbose,
    )
    print(f"[PDF FLOW] Step B2 complete. ({len(translated_bytes):,} bytes produced)")

    print(f"\n[PDF FLOW] Step B3: Saving translated PDF to: {output_pdf_path}")
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_pdf_path, "wb") as f:
        f.write(translated_bytes)
    print(f"[PDF FLOW] Step B3 complete.")

    print(f"\n[PDF FLOW] ✅ PDF translated successfully → {output_pdf_path}")


# ----------------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    input_pdf_path  = r"C:\gen_ai\document-translation\backend\data\searchable pdf\TA_O Risk Management v1.1.pdf"
    output_pdf_path = r"translated.pdf"
    from_lang       = "en"
    to_lang         = "ja"

    translate_pdf(
        input_pdf_path=input_pdf_path,
        output_pdf_path=output_pdf_path,
        from_lang=from_lang,
        to_lang=to_lang,
        verbose=True,
    )
