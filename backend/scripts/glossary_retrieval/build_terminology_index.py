"""
Run as standalone script to build an Azure AI Search terminology index with vector search.
What this script does:
1) Extract paragraphs from a terminology file (PDF or plain text).
2) Assign a domain to each paragraph from the remain_func_group env variable.
3) Compute embeddings for each paragraph using Azure OpenAI embedding deployment.
4) Create an Azure AI Search index (if it doesn't exist) with:
    - id      : key
    - content : searchable
    - vector  : vector field
    - lang    : filterable (e.g., de, fr, ja, ar, es, en)
    - domain  : filterable (ENGINE/COOLING/... or NONE)
5) Upload docs into the index.

Required/optional env vars:
- AZURE_AI_EMBEDDING_URI
- AZURE_AI_EMBEDDING_KEY
- AZURE_AI_SEARCH_URI
- AZURE_AI_SEARCH_API_KEY
- AZURE_AI_SEARCH_INDEX_NAME
- TERMINOLOGY_FILE_PATH
- TERMINOLOGY_LANG (required; e.g. "de", "en", "ja")
- remain_func_group (domain to assign to all paragraphs; e.g. "ENGINE", "COOLING")
- EMBED_MAX_WORKERS
"""

import os, re, uuid, pymupdf, requests, time
from typing import List, Dict, Tuple, Optional, Iterable
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

from concurrent.futures import ThreadPoolExecutor, as_completed


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

load_dotenv(override=True)

# Azure OpenAI Embedding endpoint
AZURE_AI_EMBEDDING_URI = os.environ["AZURE_AI_EMBEDDING_URI"]
AZURE_AI_EMBEDDING_KEY = os.environ["AZURE_AI_EMBEDDING_KEY"]

# Azure AI Search
AZURE_AI_SEARCH_URI = os.environ["AZURE_AI_SEARCH_URI"]  # e.g. https://<name>.search.windows.net
AZURE_AI_SEARCH_API_KEY = os.environ["AZURE_AI_SEARCH_API_KEY"]
AZURE_AI_SEARCH_INDEX_NAME = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]  # e.g. terminology-de

# Terminology metadata
TERMINOLOGY_LANG = (os.environ.get("TERMINOLOGY_LANG") or "").strip().lower()
DOMAIN_DEFAULT = "NONE"
REMAIN_FUNC_GROUP = (os.environ.get("remain_func_group") or "").strip()

# Embedding dimension for text-embedding-3-large
EMBEDDING_DIM = 3072
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "6"))


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# -----------------------------------------------------------------------------
# CLIENTS
# -----------------------------------------------------------------------------

search_index_client = SearchIndexClient(endpoint=AZURE_AI_SEARCH_URI, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY),)

search_client = SearchClient(endpoint=AZURE_AI_SEARCH_URI, index_name=AZURE_AI_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY),)


# -----------------------------------------------------------------------------
# NORMALIZATION (domains/lang)
# -----------------------------------------------------------------------------

def normalize_domain(raw: Optional[str]) -> str:
    """
    Normalize domain so it's consistent across ingestion:
        - trim
        - collapse whitespace
        - replace spaces with underscore
        - uppercase
        - fallback to 'NONE'
    """
    d = (raw or "").strip()
    if not d:
        return DOMAIN_DEFAULT
    d = re.sub(r"\s+", " ", d).replace(" ", "_").upper()
    return d or DOMAIN_DEFAULT


def normalize_lang(raw: str) -> str:
    """
    Normalize language code for filtering.
    Example: 'de-DE' -> 'de-de' (we keep as-is lowercased; you can enforce 2-letter if you want).
    """
    return (raw or "").strip().lower()




# -----------------------------------------------------------------------------
# CONCURRENT EMBEDDING HELPERS
# -----------------------------------------------------------------------------

def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    """
    Exponential backoff with jitter-free cap.
    attempt: 1,2,3...
    """
    delay = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(delay)


def embed_batch_with_retries(
    texts: List[str],
    *,
    max_retries: int = 6,
    timeout_s: int = 60,
) -> List[List[float]]:
    """
    Thread-safe wrapper around embed_batch() with retry/backoff for transient failures:
        - HTTP 429 (rate limit)
        - HTTP 408/5xx
        - network timeouts / request exceptions

    NOTE: Uses the same endpoint + API key as embed_batch().
    """
    if not texts:
        return []

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Content-Type": "application/json", "api-key": AZURE_AI_EMBEDDING_KEY}
            payload = {"input": texts}

            resp = requests.post(AZURE_AI_EMBEDDING_URI, headers=headers, json=payload, timeout=timeout_s)

            # Retry-worthy statuses
            if resp.status_code in (408, 429) or 500 <= resp.status_code <= 599:
                last_err = RuntimeError(f"Embedding HTTP {resp.status_code}: {resp.text[:300]}")
                _log(f"[EMBED][RETRY] attempt={attempt}/{max_retries} status={resp.status_code} batch_size={len(texts)} -> backoff")
                _sleep_backoff(attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            return [row["embedding"] for row in data["data"]]

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                _log(f"[EMBED][RETRY] attempt={attempt}/{max_retries} exception={type(e).__name__}: {str(e)[:200]} -> backoff")
                _sleep_backoff(attempt)
                continue
            break

    raise RuntimeError(f"Embedding batch failed after {max_retries} retries: {last_err}")


def _batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """
    Yield (start, end) slices for n items.
    """
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield start, end


def embed_docs_concurrently(
    docs: List[Dict],
    *,
    embed_batch_size: int,
    max_workers: int,
) -> List[Dict]:
    """
    Compute embeddings for docs concurrently (thread-safe).

    - Splits docs into batches of embed_batch_size.
    - Runs embed_batch_with_retries() concurrently for batches.
    - Preserves original order by stitching results via batch index.
    - Returns docs_with_vec (each doc has 'vector').

    Raises on batch failure (so you don't partially index silently).
    """
    if not docs:
        return []

    n = len(docs)
    batches = list(_batched_indices(n, embed_batch_size))
    total_batches = len(batches)

    _log(f"[EMBED] Concurrent embedding: docs={n} batch_size={embed_batch_size} "
        f"batches={total_batches} workers={max_workers}")

    # Will store embeddings per doc index
    vectors: List[Optional[List[float]]] = [None] * n

    t0 = time.perf_counter()

    def _work(batch_no: int, start: int, end: int) -> Tuple[int, int, int, List[List[float]]]:
        texts = [docs[i]["content"] for i in range(start, end)]
        embeds = embed_batch_with_retries(texts)
        if len(embeds) != (end - start):
            raise RuntimeError(f"Embedding count mismatch in batch {batch_no}: got {len(embeds)} expected {end-start}")
        return batch_no, start, end, embeds

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for bidx, (start, end) in enumerate(batches, start=1):
            futs.append(ex.submit(_work, bidx, start, end))

        for fut in as_completed(futs):
            batch_no, start, end, embeds = fut.result()
            # Stitch into correct slots
            for i, vec in enumerate(embeds):
                vectors[start + i] = vec

            completed += 1

            # Progress logging
            if completed == 1 or completed % 5 == 0 or completed == total_batches:
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                _log(f"[EMBED] done {completed}/{total_batches} batches "
                    f"| elapsed={elapsed:.1f}s | ~{rate:.2f} batches/s")

    # Ensure all vectors filled
    missing = sum(1 for v in vectors if v is None)
    if missing:
        raise RuntimeError(f"[EMBED] Missing vectors for {missing} docs (unexpected).")

    out: List[Dict] = []
    for i, d in enumerate(docs):
        dd = dict(d)
        dd["vector"] = vectors[i]  # type: ignore[assignment]
        out.append(dd)
    return out


# -----------------------------------------------------------------------------
# INDEX CREATION
# -----------------------------------------------------------------------------

def create_index_if_not_exists():
    """
    Create the Azure AI Search index for terminology if it doesn't exist.

    Schema:
        - id      : key
        - content : searchable text (terminology context)
        - vector  : vector embedding field (3072 dims)
        - lang    : filterable (e.g. de, fr, ja, ar, es, en)
        - domain  : filterable (ENGINE/COOLING/... or NONE)
    """
    try:
        search_index_client.get_index(AZURE_AI_SEARCH_INDEX_NAME)
        _log(f"Index '{AZURE_AI_SEARCH_INDEX_NAME}' already exists, skipping creation.\n")
        return
    except ResourceNotFoundError:
        pass

    algorithm_name = "hnsw-terms"
    profile_name = "hnsw-terms-profile"

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=algorithm_name, kind="hnsw")],
        profiles=[VectorSearchProfile(name=profile_name, algorithm_configuration_name=algorithm_name)],
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),

        # filterable metadata
        SimpleField(name="lang", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True),

        # searchable text
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),

        # vector field
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,  # must be searchable for vector search
            vector_search_dimensions=EMBEDDING_DIM,
            vector_search_profile_name=profile_name,
        ),
    ]

    index = SearchIndex(
        name=AZURE_AI_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    search_index_client.create_or_update_index(index)
    _log(f"Created index '{AZURE_AI_SEARCH_INDEX_NAME}' with filterable fields: lang, domain.\n")


# -----------------------------------------------------------------------------
# PARAGRAPH EXTRACTION HELPERS
# -----------------------------------------------------------------------------

def is_noise_paragraph(text: str) -> bool:
    """
    Heuristics to drop non-content junk from PDFs:
        - empty or extremely short
        - pure numbers / codes / page numbers
        - almost no letters
    """
    t = text.strip()
    if not t:
        return True

    # very short stuff like "11-1", "W0017", "11-3"
    if len(t) <= 4:
        return True

    # mostly non-letters → likely IDs or page labels
    alpha = sum(c.isalpha() for c in t)
    if alpha <= 1:
        return True

    # single token (no spaces) and short → most likely a code (P11237E, J5090aE, etc.)
    if " " not in t and len(t) <= 10:
        return True

    # pure "section number" like 11-3, 11-300, etc.
    if re.fullmatch(r"\d+(-\d+)*", t):
        return True

    return False


def _clean_block_text(raw: str) -> str:
    """Collapse internal newlines and extra spaces inside a block."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    paragraph = " ".join(lines)
    # normalize multiple spaces
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph


# -----------------------------------------------------------------------------
# DOMAIN EXTRACTION FROM PDF TOC (top-level only)
# -----------------------------------------------------------------------------

def extract_top_level_domains_from_pdf_toc(doc: pymupdf.Document) -> List[Tuple[int, str]]:
    """
    Read PDF Table of Contents (TOC) and return top-level entries only.

    Returns: list of (start_page_0based, domain_name_normalized)

    Notes:
    - We only take TOC entries where level == 1 (top-level domains)
    - We do NOT extract sub-domains (levels > 1)
    - If TOC missing or no level-1 entries, returns [] (caller falls back to NONE)
    """
    toc = doc.get_toc(simple=True)  # list of [level, title, page]
    top: List[Tuple[int, str]] = []

    for entry in toc or []:
        # entry shape: [lvl, title, page]
        if not entry or len(entry) < 3:
            continue
        lvl, title, page_1based = entry[0], entry[1], entry[2]
        if lvl != 1:
            continue
        title_norm = normalize_domain(title)
        # PyMuPDF TOC pages are generally 1-based
        start_page_0based = max(int(page_1based) - 1, 0)
        top.append((start_page_0based, title_norm))

    # sort by start page
    top.sort(key=lambda x: x[0])
    return top


def domain_for_page(top_domains: List[Tuple[int, str]], page_0based: int) -> str:
    """
    Given top-level domains as (start_page, domain), choose the domain for a page:
    - the last domain whose start_page <= page
    - if none apply, 'NONE'
    """
    if not top_domains:
        return DOMAIN_DEFAULT

    chosen = DOMAIN_DEFAULT
    for start_page, dom in top_domains:
        if start_page <= page_0based:
            chosen = dom
        else:
            break
    return chosen or DOMAIN_DEFAULT


# -----------------------------------------------------------------------------
# PDF extraction returning (paragraph, domain)
# -----------------------------------------------------------------------------

def extract_paragraphs_from_pdf_with_domain(path: str, domain: str) -> List[Tuple[str, str]]:
    """
    Extract paragraphs from a PDF using PyMuPDF (layout-aware blocks),
    and assign each paragraph the provided domain.

    Returns: [(paragraph_text, domain), ...]
    """
    doc = pymupdf.open(path)
    out: List[Tuple[str, str]] = []

    try:
        _log(f"[PDF] Opened '{path}' with {len(doc)} pages.")
        _log(f"[DOMAIN] Using domain from env variable: '{domain}'\n")

        total_blocks = 0
        kept = 0
        dropped = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

            if (page_num + 1) % 25 == 0 or page_num == 0:
                _log(f"[PDF] Processing page {page_num + 1}/{len(doc)} | blocks={len(blocks)} | domain='{domain}'\n")

            for (_x0, _y0, _x1, _y1, text, *_rest) in blocks:
                total_blocks += 1
                cleaned = _clean_block_text(text)
                if not cleaned:
                    continue
                if is_noise_paragraph(cleaned):
                    if dropped % 10 == 0: _log(f"Discarding noise paragraph: {cleaned[:100]}...")
                    dropped += 1
                    continue
                out.append((cleaned, domain))
                kept += 1

        _log(f"\n-------------------------\n[PDF] Extraction done. blocks_seen={total_blocks} kept={kept} dropped={dropped}\n-------------------------")
        return out

    finally:
        doc.close()


def _extract_paragraphs_from_plain_text(text: str) -> List[str]:
    """
    Fallback for .txt files – similar heuristics as for PDF blocks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n\s*\n", text)

    paragraphs: List[str] = []
    for block in raw_blocks:
        cleaned = _clean_block_text(block)
        if not cleaned:
            continue
        if is_noise_paragraph(cleaned):
            continue
        print(f"Extracted paragraph (txt): {cleaned[:100]}...")
        paragraphs.append(cleaned)
    return paragraphs


def extract_paragraphs_from_file_with_metadata(path: str, domain: str) -> List[Tuple[str, str]]:
    """
    Entry point for extraction:
        - PDF -> returns [(paragraph, domain), ...] using provided domain
        - else -> returns [(paragraph, domain), ...] using provided domain
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_paragraphs_from_pdf_with_domain(path, domain)
    with open(path, "r", encoding="utf-8") as f:
        paragraphs = _extract_paragraphs_from_plain_text(f.read())
    return [(p, domain) for p in paragraphs]


def build_paragraph_docs_from_file(path: str, *, lang: str, domain: str) -> List[Dict]:
    """
    Convert extracted paragraphs into docs ready for indexing.

    Each doc:
        - id      : random UUID
        - content : paragraph text
        - lang    : filterable language code
        - domain  : filterable content domain (from remain_func_group env variable), default 'NONE'
    """
    lang_norm = normalize_lang(lang)
    pairs = extract_paragraphs_from_file_with_metadata(path, domain)

    docs: List[Dict] = []
    for (p, dom) in pairs:
        docs.append(
            {
                "id": str(uuid.uuid4()),
                "content": p,
                "lang": lang_norm,
                "domain": normalize_domain(dom),
            }
        )

    _log(f"[DOCS] Prepared {len(docs)} docs for indexing from '{path}'. lang='{lang_norm}'\n")
    return docs


# -----------------------------------------------------------------------------
# INDEXING
# -----------------------------------------------------------------------------

def index_paragraph_docs(
    docs: List[Dict],
    embed_batch_size: int = 32,
    upload_batch_size: int = 100,
    embed_max_workers: Optional[int] = None,
):
    """
    Index docs into Azure AI Search, adding embeddings.

    Changes vs old version:
    - Embeddings are computed concurrently in multiple threads (batch-parallel).
    - Retries + exponential backoff for 429/5xx/timeout are included.
    - Upload remains batched and sequential to avoid piling up throttles.

    Tuning knobs:
    - EMBED_MAX_WORKERS env var (default 6)
    - embed_batch_size (default 32)
    """
    if not docs:
        _log("[INDEX] No docs provided. Nothing to index.")
        return

    workers = int(embed_max_workers or EMBED_MAX_WORKERS)
    workers = max(1, workers)

    _log(f"[INDEX] Starting embedding for {len(docs)} docs "
        f"(batch_size={embed_batch_size}, workers={workers})")

    # 1) concurrent embedding
    docs_with_vec = embed_docs_concurrently(
        docs,
        embed_batch_size=embed_batch_size,
        max_workers=workers,
    )

    _log(f"[INDEX] Embedding complete. Now uploading {len(docs_with_vec)} docs "
        f"(upload_batch_size={upload_batch_size})")

    # 2) uploading in batches (sequential)
    total_failed = 0
    t0 = time.perf_counter()

    for i in range(0, len(docs_with_vec), upload_batch_size):
        batch = docs_with_vec[i : i + upload_batch_size]
        _log(f"[UPLOAD] Batch {i//upload_batch_size + 1} | docs {i + 1}-{i + len(batch)} / {len(docs_with_vec)}")

        res = search_client.upload_documents(documents=batch)
        failed = [r for r in res if not r.succeeded]
        if failed:
            total_failed += len(failed)
            _log(f"[UPLOAD][WARN] Failed to index {len(failed)} docs in this batch (starting idx={i}).")
        else:
            _log("[UPLOAD] ✅ batch succeeded")

        # occasional pacing log
        if (i // upload_batch_size) % 10 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            _log(f"[UPLOAD] progress: uploaded {i + len(batch)} / {len(docs_with_vec)} | elapsed={elapsed:.1f}s")

    if total_failed:
        _log(f"[UPLOAD] Completed with failures. total_failed={total_failed}")
    else:
        _log("[UPLOAD] Completed successfully with no failures.")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

    terminology_file_path = os.environ.get("TERMINOLOGY_FILE_PATH")
    if not terminology_file_path:
        raise RuntimeError("Missing env var: TERMINOLOGY_FILE_PATH")

    if not TERMINOLOGY_LANG:
        if os.getenv("ENVIRON") == "local":
            TERMINOLOGY_LANG = input("Enter TERMINOLOGY_LANG (e.g. de, en, fr, ja, ar, es): ").strip().lower()
            if len(TERMINOLOGY_LANG) == 0: raise RuntimeError("TERMINOLOGY_LANG cannot be empty.")
        else: raise RuntimeError("Missing env var: TERMINOLOGY_LANG (e.g. de, en, fr, ja, ar, es)")

    _log(f"[SETUP] Using terminology file: {terminology_file_path}")
    _log(f"[SETUP] Index name: {AZURE_AI_SEARCH_INDEX_NAME}")
    _log(f"[SETUP] TERMINOLOGY_LANG='{TERMINOLOGY_LANG}' (filterable field 'lang')\n")

    # 1) create index if it doesn't exist
    _log(f"[SETUP] File: {terminology_file_path}")
    _log(f"[SETUP] Index: {AZURE_AI_SEARCH_INDEX_NAME}")
    _log(f"[SETUP] lang='{TERMINOLOGY_LANG}'")

    create_index_if_not_exists()

    # 2) extract paragraphs + build docs with metadata (lang, domain)
    _log("[STEP] Extracting paragraphs + domains, then building docs...")
    domain_norm = normalize_domain(REMAIN_FUNC_GROUP)
    _log(f"[SETUP] Using domain from remain_func_group: '{domain_norm}'\n")
    paragraph_docs = build_paragraph_docs_from_file(terminology_file_path, lang=TERMINOLOGY_LANG, domain=domain_norm)

    # quick stats
    domains = {}
    for d in paragraph_docs:
        domains[d["domain"]] = domains.get(d["domain"], 0) + 1
    top_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:12]
    _log("[STATS] Top domains by doc count: " + ", ".join([f"{k}={v}" for k, v in top_domains]))

    # 3) index them
    _log("[STEP] Indexing docs (embed -> upload)...")
    index_paragraph_docs(paragraph_docs)

    end = time.perf_counter()
    mins = (end - start) / 60.0
    _log(f"[DONE] Finished indexing into '{AZURE_AI_SEARCH_INDEX_NAME}' in {mins:.2f} minutes.")
