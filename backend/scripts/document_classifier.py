import fitz  
from pypdf import PdfReader
from typing import Dict, Tuple, Optional


class PDFMetadataSourceClassifier:
    """
    PDF Source Classifier using metadata only.
    """

    SOURCE_SIGNATURES = {
        "microsoft_word": ["microsoft word", "word", "microsoft® word", "microsoft® word®"],
        "microsoft_powerpoint": ["microsoft powerpoint", "powerpoint", "microsoft® powerpoint", "microsoft® powerpoint®"],
        "microsoft_excel": ["microsoft excel", "excel", "microsoft® excel", "microsoft® excel®"],
        "adobe_frame_maker": ["framemaker"],
        "adobe_indesign": ["indesign"],
        "adobe_illustrator": ["illustrator"],
        "adobe_acrobat": ["acrobat"],
        "canva": ["canva"],
        "google_docs": ["google docs"],
        "libreoffice_writer": ["libreoffice writer"],
        "libreoffice_impress": ["libreoffice impress"],
        "wps_writer": ["wps writer"],
        "latex": ["latex", "tex"],
        "scanner": ["scan", "scanner", "image capture"]
    }

    def __init__(self):
        pass

    # -----------------------------------------------------
    # Step 1: Extract Metadata
    # -----------------------------------------------------

    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        metadata = {}

        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            metadata.update(meta)
            doc.close()
        except Exception:
            print("Error extracting metadata from PDF using fitz:", e)
            pass

        # Fallback if metadata empty
        if not metadata:
            try:
                reader = PdfReader(pdf_path)
                if reader.metadata:
                    metadata.update(
                        {k.strip("/"): str(v) for k, v in reader.metadata.items()}
                    )
            except Exception:
                print("Error extracting metadata from PDF using pypdf:", e)
                pass

        return metadata

    # -----------------------------------------------------
    # Step 2: Normalize Metadata
    # -----------------------------------------------------

    @staticmethod
    def normalize_metadata(metadata: Dict[str, str]) -> str:
        """
        Cleans and combines only the 'creator' and 'producer' metadata values
        into a single lowercase searchable string.
            1. Iterate metadata items and filter where key is 'creator' or 'producer' (case-insensitive)
            2. Take the VALUE of each matched key
            3. Convert value to lowercase and strip spaces
            4. Join both values into one combined string
        """
        combined = " ".join(
            str(v).lower().strip()
            for k, v in metadata.items()
            if k.lower() in ("creator", "producer") and v
        )
        return combined

    # -----------------------------------------------------
    # Step 3: Score Matching
    # -----------------------------------------------------

    def detect_source(self, normalized_meta: str) -> Tuple[str, float]:
        scores = {}
        print("Normalized Meta: ", normalized_meta)
        for source, keywords in self.SOURCE_SIGNATURES.items():
            score = 0
            for keyword in keywords:
                if keyword in normalized_meta:
                    score += 1
            if score > 0:
                scores[source] = score

        if not scores:
            return "unknown", 0.0

        print("Scores: ", scores)
        best_match = max(scores, key=scores.get)

        # Formula: confidence = matched_keywords / total_keywords_for_that_source
        confidence = scores[best_match] / max(
            len(self.SOURCE_SIGNATURES[best_match]), 1
        )

        return best_match, round(confidence, 2)

    # -----------------------------------------------------
    # Step 4: Public API
    # -----------------------------------------------------

    def classify(self, pdf_path: str) -> Dict[str, Optional[str]]:
        metadata = self.extract_metadata(pdf_path)

        if not metadata:
            return {
                "source": "unknown",
                "confidence": 0.0,
                "reason": "No metadata found"
            }

        normalized_meta = self.normalize_metadata(metadata)

        source, confidence = self.detect_source(normalized_meta)

        return {
            "source": source,
            "confidence": confidence,
            "metadata_detected": metadata
        }


# Example Usage
if __name__ == "__main__":
    pdf_path = r"C:\gen_ai\document-translation\backend\data\searchable pdf\TA_O Risk Management v1.1.pdf"
    classifier = PDFMetadataSourceClassifier()
    result = classifier.classify(pdf_path)

    print("Detected Source:", result["source"])
    print("Confidence:", result["confidence"])
    print("Metadata:", result.get("metadata_detected"))