import os
import re
import json
import time
import threading
import requests

from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from scripts.llm1_prompts import build_llm1_refinement_prompt
from scripts.glossary_retrieval.refine_with_glossary import refine_segment_with_glossary

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
AZURE_OPENAI_API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

WORD_LLM_MAX_WORKERS        = int(os.getenv("WORD_LLM_MAX_WORKERS", "6"))
WORD_TOP_K_PARAGRAPHS       = int(os.getenv("WORD_TOP_K_PARAGRAPHS", "5"))

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers  (unchanged-safe: pure functions, no side-effects)
# ─────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")

def _has_alpha(s: str) -> bool:
    return any(ch.isalpha() for ch in s)

def _should_skip(text: str) -> bool:
    """Return True for texts that should bypass both MT and LLM stages."""
    t = (text or "").strip()
    if not t:
        return True
    if not _has_alpha(t):
        return True
    if t.isspace():
        return True
    if re.fullmatch(r"(https?://\S+|www\.\S+)", t, flags=re.IGNORECASE):
        return True
    return False

def _extract_placeholders(s: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(s or "")

def _preserves_placeholders(out: str, src: str) -> bool:
    src_ph = _extract_placeholders(src)
    if not src_ph:
        return True
    return all(p in (out or "") for p in src_ph)

def _preview(s: str, limit: int = 160) -> str:
    t = (s or "").replace("\n", "\\n")
    return t if len(t) <= limit else t[: limit - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Thread-local HTTP sessions
# ─────────────────────────────────────────────────────────────────────────────
_tls = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Azure OpenAI wrapper  (LLM1 calls)
# ─────────────────────────────────────────────────────────────────────────────
class _AzureOpenAIChat:
    """Minimal Azure OpenAI chat client for LLM1 refinement (JSON mode)."""

    def __init__(self):
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT_NAME:
            raise RuntimeError(
                "Missing Azure OpenAI credentials. Set "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME."
            )

    def chat_json(self, system: str, user_payload: dict, *, temperature: float = 0.1, max_tokens: int = 900) -> dict:
        endpoint = AZURE_OPENAI_ENDPOINT.rstrip("/")
        url = f"{endpoint}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions"
        params = {"api-version": AZURE_OPENAI_API_VERSION}
        headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
        body = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }

        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = _get_session().post(url, params=params, headers=headers, data=json.dumps(body), timeout=90)
                if resp.status_code >= 400:
                    raise RuntimeError(f"AOAI HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except Exception:
                    return {"translation": content.strip()}
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (2 ** attempt))
        raise RuntimeError(f"AOAI chat failed after retries: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM1 : single-item refinement  (grammar / fluency fix)
# ─────────────────────────────────────────────────────────────────────────────
def _llm1_refine(
    aoai: _AzureOpenAIChat,
    *,
    source_text: str,
    mt_text: str,
    source_lang: str,
    target_lang: str,
    previous_chunks: Optional[List[dict]] = None,
    is_placeholder: bool = False,
    is_short_mode: bool = False,
) -> str:
    """Ask Azure OpenAI to improve the MT output for one text run."""
    prompt_data = build_llm1_refinement_prompt(
        source_text=source_text,
        mt_text=mt_text,
        previous_chunks=previous_chunks,
        source_lang=source_lang,
        target_lang=target_lang,
        is_short_mode=is_short_mode,
        source_file="word"
    )
    
    system = prompt_data["system"]
    payload = prompt_data["user_payload"]

    cand = mt_text
    max_retries = 3 if is_placeholder else 2

    for attempt in range(max_retries):
        try:
            out = aoai.chat_json(system, payload, temperature=0.1, max_tokens=900)
            curr_cand = (out.get("translation") or "").strip()
            if not curr_cand:
                continue
            if is_placeholder and not _preserves_placeholders(curr_cand, source_text):
                payload["instructions"] = (
                    "FATAL ERROR: You dropped or corrupted mandatory [[...]] placeholders. "
                    "YOU MUST PRESERVE THEM EXACTLY. Re-try now."
                )
                continue
            cand = curr_cand
            break
        except Exception:
            if attempt == max_retries - 1:
                break
            time.sleep(0.5)

    return cand if cand else mt_text


# ─────────────────────────────────────────────────────────────────────────────
# TranslatorService  — original code UNCHANGED below the class definition line
#                      New method `batch_translate_with_pipeline` added at bottom
# ─────────────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Full 3-stage pipeline  (MT → LLM1 → RAG+LLM2)
    # ─────────────────────────────────────────────────────────────────────────
    def batch_translate_with_pipeline(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        *,
        batch_size: int = 100,
        max_workers: int = WORD_LLM_MAX_WORKERS,
        top_k_paragraphs: int = WORD_TOP_K_PARAGRAPHS,
        rolling_context_size: Optional[int] = 3,
    ) -> List[Optional[str]]:
        """
        Full 3-stage translation pipeline for Word (.docx) runs:

            Stage 1  : Azure Translator (MT, batched 100/call, same as batch_translate)
            Stage 2  : LLM1: Azure OpenAI grammar/fluency refinement
                         • Rolling 10-run context window (src + MT)
                         • Short-mode for very short labels (<= 20 chars)
                         • Placeholder protection for [[...]] tokens
            Stage 3  : LLM2+RAG: glossary-grounded final refinement
                         • Dual-query vector search (English + LLM1 translation)
                         • Language-filtered Azure AI Search index
                         • Final Azure OpenAI call with retrieved glossary context

        Args:
            texts         : Original source text strings (all runs from the document).
            source_lang   : BCP-47 source language code, e.g. "en".
            target_lang   : BCP-47 target language code, e.g. "ja".
            batch_size    : MT batch chunk size (default 100, Azure cap).
            max_workers   : Thread pool size for LLM1 and LLM2 stages.
            top_k_paragraphs : Top-K glossary paragraphs to retrieve per run in Stage 3.

        Returns:
            List of final translated strings aligned to input `texts`.
            Falls back gracefully at each stage (LLM1 fail → MT; LLM2 fail → LLM1).
        """
        n = len(texts)
        t_pipeline_start = time.perf_counter()

        # ── pre-compute skip mask (same heuristics as PDF pipeline) ──────────
        skip_mask = [_should_skip(t) for t in texts]
        skip_count = sum(skip_mask)
        print(f"[WORD-PIPELINE] Total runs: {n}, skipping {skip_count} (empty/no-alpha/URL)")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1 — MT  (existing batch_translate, unchanged)
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[WORD-PIPELINE][STAGE 1 - MT] Translating {n} runs (batch_size={batch_size}) ...")
        t0 = time.perf_counter()
        mt_out: List[Optional[str]] = self.batch_translate(texts, target_lang, batch_size=batch_size)
        # Ensure skipped texts and identity/empty MT results fall back to source
        for i in range(n):
            if skip_mask[i]:
                mt_out[i] = texts[i]
            elif not mt_out[i] or (mt_out[i] or "").strip() == "" or mt_out[i] == texts[i]:
                mt_out[i] = texts[i]
        print(f"[WORD-PIPELINE][STAGE 1 - MT] Done in {time.perf_counter() - t0:.2f}s")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2 — LLM1 (grammar / fluency refinement, parallel)
        # ─────────────────────────────────────────────────────────────────────
        aoai = _AzureOpenAIChat()
        pass1: List[str] = list(mt_out)  # type: ignore[arg-type]

        # Only refine runs where MT actually changed something
        llm1_indices = [
            i for i in range(n)
            if not skip_mask[i] and mt_out[i] != texts[i]
        ]
        print(
            f"\n[WORD-PIPELINE][STAGE 2 - LLM1] Candidates: {len(llm1_indices)} "
            f"(skipped heuristic={skip_count}, no-MT-change={n - skip_count - len(llm1_indices)})"
        )
        stats_lock = threading.Lock()
        llm1_failures = 0

        def _llm1_task(i: int) -> Tuple[int, str, float]:
            t_start = time.perf_counter()
            src = (texts[i] or "").strip()
            mt  = (mt_out[i] or "").strip()

            ctx: Optional[List[dict]] = None
            if rolling_context_size is not None and rolling_context_size > 0:
                ctx = [
                    {"src": texts[j], "mt": mt_out[j]}
                    for j in range(max(0, i - rolling_context_size), i)
                ]

            is_placeholder = bool(_extract_placeholders(src))
            # Short mode: very short text (likely a label/caption)
            is_short_mode = len(src) <= 20

            try:
                refined = _llm1_refine(
                    aoai,
                    source_text=src,
                    mt_text=mt,
                    previous_chunks=ctx,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    is_placeholder=is_placeholder,
                    is_short_mode=is_short_mode,
                )
                # Placeholder safety net
                if is_placeholder and not _preserves_placeholders(refined, src):
                    refined = mt if _preserves_placeholders(mt, src) else src
                return i, refined, time.perf_counter() - t_start
            except Exception as e:
                nonlocal llm1_failures
                with stats_lock:
                    llm1_failures += 1
                print(f"[LLM1] run i={i} failed: {e} → fallback to MT")
                return i, mt_out[i], time.perf_counter() - t_start  # type: ignore[return-value]

        t0 = time.perf_counter()
        if llm1_indices:
            print(f"[LLM1] Refining {len(llm1_indices)} runs with {max_workers} workers ...")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_llm1_task, i) for i in llm1_indices]
                for done_idx, fut in enumerate(as_completed(futs), start=1):
                    i, out_t, elapsed = fut.result()
                    pass1[i] = out_t
                    if done_idx % 10 == 0:
                        print(f"  [LLM1] progress {done_idx}/{len(llm1_indices)} | last i={i} elapsed={elapsed:.2f}s")
        print(
            f"[WORD-PIPELINE][STAGE 2 - LLM1] Done in {time.perf_counter() - t0:.2f}s "
            f"(failures={llm1_failures})"
        )

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3 — LLM2 + RAG (glossary-grounded refinement, parallel)
        # ─────────────────────────────────────────────────────────────────────
        final_out: List[str] = list(pass1)

        # Only refine runs where LLM1 output differs from source (translation happened)
        llm2_indices = [
            i for i in range(n)
            if not skip_mask[i] and pass1[i] != texts[i]
        ]
        print(
            f"\n[WORD-PIPELINE][STAGE 3 - LLM2+RAG] Candidates: {len(llm2_indices)} "
            f"(skipped heuristic={skip_count})"
        )
        llm2_failures = 0

        def _llm2_task(i: int) -> Tuple[int, str, float]:
            t_start = time.perf_counter()
            src = (texts[i] or "").strip()
            cur = (pass1[i] or "").strip()
            is_placeholder = bool(_extract_placeholders(src))
            is_short_mode = len(src) <= 20

            ctx_llm2: Optional[List[dict]] = None
            if rolling_context_size is not None and rolling_context_size > 0:
                ctx_llm2 = [
                    {"src": texts[j], "mt": pass1[j]}
                    for j in range(max(0, i - rolling_context_size), i)
                ]

            try:
                refined = refine_segment_with_glossary(
                    source_chunk=src,
                    current_translation=cur,
                    is_placeholder=is_placeholder,
                    verbose=False,
                    top_k_paragraphs=top_k_paragraphs,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    is_short_mode=is_short_mode,
                    previous_chunks=ctx_llm2,
                )
                # Placeholder safety net
                if is_placeholder and not _preserves_placeholders(refined, src):
                    refined = cur
                return i, refined if refined else cur, time.perf_counter() - t_start
            except Exception as e:
                nonlocal llm2_failures
                with stats_lock:
                    llm2_failures += 1
                print(f"[LLM2] run i={i} failed: {e} → keep LLM1")
                return i, cur, time.perf_counter() - t_start

        t0 = time.perf_counter()
        if llm2_indices:
            print(f"[LLM2] Refining {len(llm2_indices)} runs with {max_workers} workers ...")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_llm2_task, i) for i in llm2_indices]
                for done_idx, fut in enumerate(as_completed(futs), start=1):
                    i, out_t, elapsed = fut.result()
                    final_out[i] = out_t
                    if done_idx % 10 == 0:
                        print(f"  [LLM2] progress {done_idx}/{len(llm2_indices)} | last i={i} elapsed={elapsed:.2f}s")
        print(
            f"[WORD-PIPELINE][STAGE 3 - LLM2+RAG] Done in {time.perf_counter() - t0:.2f}s "
            f"(failures={llm2_failures})"
        )

        total_wall = time.perf_counter() - t_pipeline_start
        print(f"\n[WORD-PIPELINE] Complete. Total wall time: {total_wall:.2f}s")

        return final_out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = TranslatorService()
    text = t.translate("HI")
    print(text, "ja")
