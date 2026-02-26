from __future__ import annotations

import io, os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

from pdf_pipeline import translate_pdf_bytes_pipeline

load_dotenv(override=True)
VERBOSE_FLAG = os.getenv("VERBOSE_FLAG", False).lower() in ("true", "1", "yes")

app = FastAPI(title="PDF Translator (Pipeline Starter)")


@app.post("/translate")
async def translate_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        out_bytes = translate_pdf_bytes_pipeline(pdf_bytes, verbose=VERBOSE_FLAG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    headers = {"Content-Disposition": f'attachment; filename="translated_{file.filename}"'}
    return StreamingResponse(io.BytesIO(out_bytes), media_type="application/pdf", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)