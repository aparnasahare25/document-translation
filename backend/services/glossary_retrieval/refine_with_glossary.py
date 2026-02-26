"""
Translation refinement with glossary context.

Uses:
- Azure OpenAI for embeddings (via direct REST)
- Azure AI Search for vector search over glossary paragraphs
    - FILTER:
        * lang (e.g. "ja", "de")
    - Missing/empty lang -> unfiltered search (broadens retrieval)
- Azure OpenAI for LLM refinement with retrieved context

Returns refined target-language translation segments.
"""

from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import os, json, time, requests, re, threading, random
from typing import List, Dict, Optional, Callable, Any

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from services.glossary_retrieval.prompts import get_placeholder_sys_prompt, get_no_placeholder_sys_prompt


# -------------------------
# logging utilities
# -------------------------
def _now() -> str:
    return time.strftime("%H:%M:%S")

def _divider(title: str = "", ch: str = "─", width: int = 80) -> str:
    if not title:
        return ch * width
    pad = max(0, width - len(title) - 2)
    return f"{title}\n{ch * pad}"

def _preview(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")

def _print_kv(k: str, v: str, indent: int = 2):
    print(" " * indent + f"{k}: {v}")

def _step(title: str):
    print()
    print(_divider(f"[{_now()}] {title}", "═"))


# -----------------------------------------------------------------------------
# CONFIG (from environment)
# -----------------------------------------------------------------------------
load_dotenv(override=True)

# Azure OpenAI LLM (LLM2 refine)
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# Azure OpenAI Embedding endpoint (direct REST)
AZURE_AI_EMBEDDING_URI = os.environ["AZURE_AI_EMBEDDING_URI"]
AZURE_AI_EMBEDDING_KEY = os.environ["AZURE_AI_EMBEDDING_KEY"]

# Azure AI Search
AZURE_AI_SEARCH_URI = os.environ["AZURE_AI_SEARCH_URI"]
AZURE_AI_SEARCH_API_KEY = os.environ["AZURE_AI_SEARCH_API_KEY"]
AZURE_AI_SEARCH_INDEX_NAME = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

# Default retrieval language (optional). If empty -> unfiltered
RAG_LANG_DEFAULT = os.environ.get("RAG_LANG", "").strip()  # e.g. "de"

# Embedding dimension (keep if your index expects this; not strictly required here)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "3072"))

# Optional: prevent noisy prints on import
_IMPORT_LOGS = os.environ.get("RAG_IMPORT_LOGS", "0").strip().lower() in ("1", "true", "yes", "on")
if _IMPORT_LOGS:
    _step("RAG MODULE CONFIG")
    _print_kv("Search index", AZURE_AI_SEARCH_INDEX_NAME)
    _print_kv("RAG_LANG_DEFAULT", RAG_LANG_DEFAULT or "(none)")


# -----------------------------------------------------------------------------
# RETRY / BACKOFF (429/503)
# -----------------------------------------------------------------------------
_RETRYABLE_STATUS = {429, 503, 502, 504}

_RETRY_MAX_ATTEMPTS = int(os.environ.get("TRANSLATION_RETRY_MAX_ATTEMPTS", "10"))
_RETRY_BASE_DELAY_S = float(os.environ.get("TRANSLATION_RETRY_BASE_DELAY_S", "0.25"))
_RETRY_MAX_DELAY_S = float(os.environ.get("TRANSLATION_RETRY_MAX_DELAY_S", "25"))
_RETRY_JITTER_FRAC = float(os.environ.get("TRANSLATION_RETRY_JITTER_FRAC", "0.1"))

RETRY_COUNT = {"retry_counter": 0, "retry_time_wasted": 0.0}


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """
    Best-effort status code extraction across:
        - requests HTTPError (exc.response.status_code)
        - OpenAI SDK errors (often exc.status_code)
        - Azure SDK errors (often exc.status_code)
    """
    for attr in ("status_code", "http_status", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v

    resp = getattr(exc, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
        v = getattr(resp, "status", None)
        if isinstance(v, int):
            return v

    msg = str(exc)
    for sc in (429, 503, 502, 504):
        if str(sc) in msg:
            return sc
    return None


def _retry_after_seconds_from_exc(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers and hasattr(headers, "get"):
        val = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return float(val) if val is not None else None
        except Exception:
            return None
    return None


def _sleep_backoff(attempt_idx: int, retry_after_s: Optional[float] = None) -> float:
    if retry_after_s is not None and retry_after_s >= 0:
        delay = min(_RETRY_MAX_DELAY_S, retry_after_s)
    else:
        base = min(_RETRY_MAX_DELAY_S, _RETRY_BASE_DELAY_S * (2 ** attempt_idx))
        jitter = base * _RETRY_JITTER_FRAC * random.random()
        delay = min(_RETRY_MAX_DELAY_S, base + jitter)
    time.sleep(delay)
    return delay


def _call_with_retry(fn: Callable[[], Any], *, verbose: bool, what: str) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            status = _extract_status_code(exc)
            if status in _RETRYABLE_STATUS and attempt < (_RETRY_MAX_ATTEMPTS - 1):
                retry_after = _retry_after_seconds_from_exc(exc)
                delay = _sleep_backoff(attempt, retry_after)
                if verbose:
                    _print_kv("RAG Retry", f"{what} status={status} attempt={attempt+1}/{_RETRY_MAX_ATTEMPTS} sleep={delay:.2f}s")
                RETRY_COUNT["retry_counter"] += 1
                RETRY_COUNT["retry_time_wasted"] += delay
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{what} failed (unknown error)")


# -----------------------------------------------------------------------------
# THREAD-LOCAL CLIENTS (safe for multithreading)
# -----------------------------------------------------------------------------
_TLS = threading.local()

def _make_embed_session() -> requests.Session:
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def _get_embed_session() -> requests.Session:
    sess = getattr(_TLS, "embed_session", None)
    if sess is None:
        sess = _make_embed_session()
        _TLS.embed_session = sess
    return sess

def _get_llm_client() -> AzureOpenAI:
    client = getattr(_TLS, "llm_client", None)
    if client is None:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        _TLS.llm_client = client
    return client

def _get_search_client() -> SearchClient:
    client = getattr(_TLS, "search_client", None)
    if client is None:
        client = SearchClient(
            endpoint=AZURE_AI_SEARCH_URI,
            index_name=AZURE_AI_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY),
        )
        _TLS.search_client = client
    return client


# -----------------------------------------------------------------------------
# EMBEDDING (thread-local session + batch)
# -----------------------------------------------------------------------------
_EMBED_HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_AI_EMBEDDING_KEY,
}

def embed_many(texts: List[str], verbose: bool) -> List[List[float]]:
    """
    Get embeddings for multiple texts in ONE HTTP call.
    Returns vectors aligned to the same order as `texts`.
    """
    norm_texts = [(t or "").strip() for t in (texts or [])]
    if not norm_texts:
        return []

    if verbose:
        _step("EMBEDDINGS (BATCH)")
        _print_kv("Batch Size", str(len(norm_texts)))

    payload = {"input": norm_texts}
    session = _get_embed_session()

    def _do_post() -> requests.Response:
        t0 = time.perf_counter()
        resp = session.post(
            AZURE_AI_EMBEDDING_URI,
            headers=_EMBED_HEADERS,
            json=payload,
            timeout=60,
        )
        dt = time.perf_counter() - t0
        if verbose:
            _print_kv("Embedding latency (s)", f"{dt:.2f}")
        resp.raise_for_status()
        return resp

    resp = _call_with_retry(_do_post, verbose=verbose, what="Embeddings Request")
    data = resp.json()

    items = data.get("data", [])
    if len(items) != len(norm_texts):
        raise RuntimeError(f"Embedding response size mismatch: got {len(items)} for {len(norm_texts)} inputs")

    return [it["embedding"] for it in items]


def embed(text: str, verbose: bool) -> List[float]:
    vecs = embed_many([text], verbose)
    return vecs[0] if vecs else []


# -----------------------------------------------------------------------------
# GLOSSARY (optional) - safe fallback if missing
# -----------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}

utils_path = Path(__file__).parent.parent.absolute() / "utils"
glossary_path = utils_path / "glossary" / "german.json"
json_glossary = _load_json(glossary_path)


# -----------------------------------------------------------------------------
# LANG NORMALIZATION + FILTER BUILDING (Azure AI Search)
# -----------------------------------------------------------------------------
def _normalize_lang(lang: Optional[str]) -> str:
    t = (lang or "").strip().lower()
    if not t:
        return ""
    mapping = {
        "english": "en", "japanese": "ja", "german": "de",
        "french": "fr", "spanish": "es", "italian": "it",
        "portuguese": "pt", "dutch": "nl", "russian": "ru",
        "chinese": "zh", "korean": "ko", "arabic": "ar"
    }
    t = mapping.get(t, t)
    t = t.split("-", 1)[0]
    return re.sub(r"\s+", "", t)

def _escape_odata_string(value: str) -> str:
    return (value or "").replace("'", "''")

def _build_search_filter(*, lang: Optional[str], verbose: bool) -> Optional[str]:
    """
    Only filter supported: lang
        - if lang provided -> lang eq '<lang>'
        - else -> None (unfiltered)
    """
    lang_n = _normalize_lang(lang)
    flt = f"lang eq '{_escape_odata_string(lang_n)}'" if lang_n else None

    if verbose:
        _print_kv("RAG scope.lang", lang_n or "(none)")
        _print_kv("Azure Search filter", flt or "(none - unfiltered)")
    return flt


# -----------------------------------------------------------------------------
# VECTOR SEARCH OVER GLOSSARY PARAGRAPHS
# -----------------------------------------------------------------------------
def search_paragraphs_by_vector(
    vec: List[float],
    verbose: bool,
    top_k: int = 5,
    *,
    lang: Optional[str] = None,
) -> List[Dict]:
    """
    Vector search over glossary paragraphs.
        - Applies OData filter ONLY on lang
        - Missing/empty lang -> unfiltered search
    """
    if not vec:
        return []

    vector_query = VectorizedQuery(
        vector=vec,
        k_nearest_neighbors=top_k,
        fields="vector", # name of vector field in index
    )

    search_client = _get_search_client()
    search_filter = _build_search_filter(lang=lang, verbose=verbose)

    def _do_search() -> List[Any]:
        return list(
            search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["id", "content", "lang"],
                filter=search_filter,
                top=top_k,
            )
        )

    t0 = time.perf_counter()
    results_list = _call_with_retry(_do_search, verbose=verbose, what="Azure Search Vector Query")
    dt = time.perf_counter() - t0

    hits: List[Dict] = []
    for r in results_list:
        hits.append(
            {
                "id": r["id"],
                "content": r.get("content", ""),
                "score": r["@search.score"],
                "lang": r.get("lang", ""),
            }
        )

    if verbose:
        _print_kv("Paragraphs returned", str(len(hits)))
        _print_kv("Search latency (s)", f"{dt:.2f}")
        if hits:
            for hit in hits:
                print(f"\n    Hit: Score={hit['score']:.4f} | lang={hit.get('lang','')}\n    Content: {hit['content'][:250]}...\n")

    return hits


def retrieve_glossary_paragraphs(
    english_chunk: str,
    current_translation: str,
    verbose: bool,
    top_k: int = 5,
    *,
    lang: Optional[str] = None,
) -> List[Dict]:
    """
    Dual retrieval (embeds EN + current translation together in ONE call):
        - Query 1: English original chunk
        - Query 2: current target-language translation

    Scoped only by lang:
        - if lang omitted -> uses env RAG_LANG_DEFAULT
        - if still empty -> unfiltered
    """
    if verbose:
        _step("RETRIEVING GLOSSARY PARAGRAPHS")

    en_q = (english_chunk or "").strip()
    tl_q = (current_translation or "").strip()

    lang_eff = lang if (lang is not None) else (RAG_LANG_DEFAULT or None)

    if en_q and tl_q:
        if verbose:
            print("\n[INFO] Embedding both EN + target queries in one batch call.\n")
        vec_en, vec_tl = embed_many([en_q, tl_q], verbose)
        hits_en = search_paragraphs_by_vector(vec_en, verbose, top_k=top_k, lang=lang_eff)
        hits_tl = search_paragraphs_by_vector(vec_tl, verbose, top_k=top_k, lang=lang_eff)
    elif en_q:
        if verbose:
            print("\n[INFO] Embedding only EN query.\n")
        vec_en = embed_many([en_q], verbose)[0]
        hits_en = search_paragraphs_by_vector(vec_en, verbose, top_k=top_k, lang=lang_eff)
        hits_tl = []
    elif tl_q:
        if verbose:
            print("\n[INFO] Embedding only target query.\n")
        vec_tl = embed_many([tl_q], verbose)[0]
        hits_en = []
        hits_tl = search_paragraphs_by_vector(vec_tl, verbose, top_k=top_k, lang=lang_eff)
    else:
        if verbose:
            print("\n[INFO] No valid EN or target query provided; skipping retrieval.\n")
        hits_en, hits_tl = [], []

    merged: Dict[str, Dict] = {}
    for hit in hits_en + hits_tl:
        cur = merged.get(hit["id"])
        if cur is None or hit["score"] > cur["score"]:
            merged[hit["id"]] = hit

    all_hits = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    if verbose:
        _print_kv("Merged unique hits", str(len(all_hits)))

    return all_hits


# -----------------------------------------------------------------------------
# LLM REFINEMENT (JSON contract)
# -----------------------------------------------------------------------------
def call_llm_json(system_msg: str, user_payload: dict, verbose: bool) -> dict:
    """
    Call Azure OpenAI chat deployment and parse JSON response.
    """
    if verbose:
        _step("LLM CALL (JSON MODE)")
        _print_kv("Model", AZURE_OPENAI_DEPLOYMENT_NAME)

    llm_client = _get_llm_client()

    def _do_call():
        return llm_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    t0 = time.perf_counter()
    resp = _call_with_retry(_do_call, verbose=verbose, what="LLM Refinement Chat Completion")
    dt = time.perf_counter() - t0

    content = resp.choices[0].message.content
    if verbose:
        _print_kv("LLM latency (s)", f"{dt:.2f}")

    try:
        return json.loads(content)
    except Exception as e:
        print("  ERROR: Failed to parse LLM JSON:", e)
        raise


def build_refinement_prompt(
    english_chunk: str,
    current_translation: str,
    glossary_paragraphs: List[Dict],
    is_placeholder: bool,
    glossary: Dict[str, Any],
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    """
    LLM sees:
        - English source
        - current target-language translation
        - retrieved target-language glossary/context paragraphs
        - glossary
    Returns JSON with: final_translation
    """
    if verbose:
        _step("BUILDING REFINEMENT PROMPT")

    if is_placeholder:
        system_msg = get_placeholder_sys_prompt(source_lang, target_lang).strip()
    else:
        system_msg = get_no_placeholder_sys_prompt(source_lang, target_lang).strip()

    user_payload = {
        "source_en": english_chunk,
        "current_translation": current_translation,
        "glossary_context_paragraphs": [
            {"id": p.get("id", ""), "content": p.get("content", "")} for p in glossary_paragraphs
        ],
        "glossary": glossary,
        "output_contract": {"final_translation": "string"},
    }

    if verbose:
        _print_kv("Glossary paragraphs", str(len(glossary_paragraphs)))
        if glossary_paragraphs:
            _print_kv("Top glossary preview", _preview(glossary_paragraphs[0].get("content", ""), 220))

    return {"system": system_msg, "user_payload": user_payload}


# -----------------------------------------------------------------------------
# MAIN API
# -----------------------------------------------------------------------------
def refine_segment_with_glossary(
    english_chunk: str,
    current_translation: str,
    is_placeholder: bool,
    verbose: bool,
    top_k_paragraphs: int = 5,
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> str:
    """
    Main API:

    Inputs:
        - english_chunk       : original English text
        - current_translation : existing target-language translation (MT + LLM1 refined)

    Output:
        - final ultra-refined target-language paragraph string

    OPTIONAL KWARGS:
        - target_lang: used to filter 'lang' in glossary index (e.g. "de", "fr")

    If not provided, falls back to env default:
        - RAG_LANG (or none -> no filter)
    """
    if verbose:
        _step("REFINING SINGLE SEGMENT")
        _print_kv("English Preview", _preview(english_chunk, 250))
        _print_kv("Current Translation Preview", _preview(current_translation, 250))

    # Effective lang scope:
    # - if target_lang passed -> normalize "de-DE" -> "de"
    # - else use env RAG_LANG_DEFAULT
    # - else None => unfiltered
    lang_eff = None
    if target_lang is not None:
        tl = (target_lang or "").strip()
        lang_eff = (tl.split("-", 1)[0] if tl else None)
    else:
        lang_eff = (RAG_LANG_DEFAULT or None)

    if verbose:
        _step("ACTIVE RETRIEVAL FILTERS")
        _print_kv("target_lang (requested)", str(target_lang or "(none)"))

    glossary_paragraphs = retrieve_glossary_paragraphs(
        english_chunk=english_chunk,
        current_translation=current_translation,
        verbose=verbose,
        top_k=top_k_paragraphs,
        lang=lang_eff,
    )

    prompt = build_refinement_prompt(
        english_chunk,
        current_translation,
        glossary_paragraphs,
        is_placeholder,
        json_glossary,
        verbose=verbose,
        source_lang=source_lang or "en",
        target_lang=target_lang or "ja",
    )

    llm_output = call_llm_json(prompt["system"], prompt["user_payload"], verbose=verbose)

    final_tl = (llm_output.get("final_translation") or "").strip()
    if verbose:
        _print_kv("Final Translated Segment", final_tl)

    if RETRY_COUNT["retry_counter"] != 0 and RETRY_COUNT["retry_counter"] % 5 == 0:
        print(f"[RETRY-WASTE] {RETRY_COUNT['retry_counter']} retries - wasted {RETRY_COUNT['retry_time_wasted']} seconds.")
    return final_tl


def refine_document_with_glossary(
    english_chunks: List[str],
    current_translations: List[str],
    verbose: bool,
    top_k_paragraphs: int = 5,
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> List[str]:
    """
    Batch version.
    Returns list of refined target-language segments.
    """
    assert len(english_chunks) == len(current_translations), "EN and target lists must have same length"

    if verbose:
        _step("REFINING DOCUMENT (BATCH)")
        _print_kv("Segments", str(len(english_chunks)))
        _print_kv("Top K per seg", str(top_k_paragraphs))
        _print_kv("Scope.target_lang", str(target_lang or "(none)"))

    out: List[str] = []
    for idx, (en_seg, tl_seg) in enumerate(zip(english_chunks, current_translations), start=1):
        if verbose:
            print(_divider(f"[{_now()}] SEGMENT {idx}", "─"))

        is_placeholder = ("[[BLOCK" in (en_seg or "")) or ("[[INLINE" in (en_seg or ""))
        result = refine_segment_with_glossary(
            english_chunk=en_seg,
            current_translation=tl_seg,
            is_placeholder=is_placeholder,
            verbose=verbose,
            top_k_paragraphs=top_k_paragraphs,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        out.append(result)

    if RETRY_COUNT["retry_counter"] % 20 == 0 and RETRY_COUNT["retry_counter"] != 0:
        print(f"[RETRY-WASTE] {RETRY_COUNT['retry_counter']} retries - wasted {RETRY_COUNT['retry_time_wasted']} seconds.")
    return out