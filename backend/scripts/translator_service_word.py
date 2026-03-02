import os
import requests
from typing import Union, List, Optional
from dotenv import load_dotenv

load_dotenv()

class TranslatorService:
    """A service class responsible for interacting with Azure Translator API."""

    def __init__(self):
        """Initializes the TranslatorService."""
        self.subscription_key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
        self.region = os.getenv("AZURE_TRANSLATOR_REGION")
        self.api_version = os.getenv("AZURE_TRANSLATOR_API_VERSION")

    def translate(self, text: str, target_language: str = "en") -> Optional[str]:
        """Translate a single text string to the target language."""
        results = self.batch_translate([text], target_language)
        return results[0] if results else None

    def batch_translate(self, texts: List[str], target_language: str = "en",
                        batch_size: int = 100) -> List[Optional[str]]:
        """
        Translate a list of texts in a single (or minimal) API call.

        Azure Translator supports up to 100 texts per request, so this method
        chunks the input into groups of `batch_size` and sends one request per
        chunk — reducing N sequential calls to ceil(N / batch_size) calls.

        Args:
            texts:           List of source strings to translate.
            target_language: BCP-47 target language code, e.g. "ja".
            batch_size:      Max texts per Azure request (Azure cap = 100).

        Returns:
            List of translated strings in the same order as `texts`.
            Items that fail to translate are returned as their original value.
        """
        # Pre-fill with originals as a safe fallback
        results: List[Optional[str]] = list(texts)

        # Only send non-empty strings to the API
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            return results

        url = (
            f"{self.endpoint}translate"
            f"?api-version={self.api_version}"
            f"&to={target_language}"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-Type": "application/json",
        }

        # Process in chunks of batch_size
        for chunk_start in range(0, len(non_empty), batch_size):
            chunk = non_empty[chunk_start: chunk_start + batch_size]
            indices = [i for i, _ in chunk]
            payload = [{"text": t} for _, t in chunk]

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)

                if response.status_code == 200:
                    translated = response.json()
                    for idx, trans in zip(indices, translated):
                        results[idx] = trans["translations"][0]["text"]
                else:
                    print(f"[batch_translate] Failed for chunk at {chunk_start}: "
                          f"{response.status_code}, {response.text}")
                    # fallback: keep originals for this chunk (already pre-filled)

            except Exception as e:
                print(f"[batch_translate] Error for chunk at {chunk_start}: {e}")
                # fallback: keep originals for this chunk

        return results


if __name__ == "__main__":
    t = TranslatorService()
    text = t.translate("2024年12月25日")
    print(text)
