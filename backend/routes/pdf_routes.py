"""
Layer 1 HTTP route handlers for PDF translation.

Endpoint:
    POST /api/translate-pdf
        Form data:
            file       : PDF file (multipart)
            src_lang   : source language code (e.g. "en")
            target_lang: target language code (e.g. "de")
        Response:
            Translated PDF file download
"""

import os
import uuid
import tempfile
from flask import Blueprint, request, jsonify, send_file
from services.pdf_translator import PdfTranslator

pdf_bp = Blueprint("pdf", __name__)
translator = PdfTranslator()


@pdf_bp.route("/translate-pdf", methods=["POST"])
def translate_pdf():
    """
    Accepts a PDF upload and returns a translated PDF.
    """
    # Validate file 
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send 'file' as multipart/form-data."}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "File name is empty."}), 400

    if not uploaded_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Language params 
    src_lang = request.form.get("src_lang", "en").strip()
    target_lang = request.form.get("target_lang", "de").strip()

    if not target_lang:
        return jsonify({"error": "target_lang is required."}), 400

    # Save upload to temp file 
    tmp_dir = tempfile.mkdtemp()
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(tmp_dir, f"input_{unique_id}.pdf")
    output_path = os.path.join(tmp_dir, f"translated_{unique_id}.pdf")

    try:
        uploaded_file.save(input_path)

        # Run translation pipeline 
        translator.translate_pdf(
            input_pdf=input_path,
            output_pdf=output_path,
            src_lang=src_lang,
            target_lang=target_lang,
        )

        # Return translated PDF 
        original_name = os.path.splitext(uploaded_file.filename)[0]
        download_name = f"{original_name}_translated_{target_lang}.pdf"

        return send_file(
            output_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
        )

    except Exception as e:
        print(f"[pdf_routes] Error during translation: {e}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    finally:
        # Cleanup input file (output will be cleaned after send_file streams it)
        if os.path.exists(input_path):
            os.remove(input_path)


@pdf_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "pdf-translator"}), 200
