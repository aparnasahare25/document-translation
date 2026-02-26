from __future__ import annotations

from dataclasses import dataclass
import json, os, re, threading, time, requests
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.glossary_retrieval.refine_with_glossary import refine_segment_with_glossary


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class BlockRef:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str


@dataclass(frozen=True)
class BlockTranslation:
    block: BlockRef
    translated_text: str


# -----------------------------
# Env
# -----------------------------
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "").strip()
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "").strip()
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "").strip()
AZURE_TRANSLATOR_API_VERSION = os.getenv("AZURE_TRANSLATOR_API_VERSION", "3.0").strip()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

SOURCE_LANG_DEFAULT = os.getenv("SOURCE_LANG", "en").strip() or "en"
TARGET_LANG_DEFAULT = os.getenv("TARGET_LANG", "ja").strip() or "ja"

PDF_TRANSLATION_BATCH_SIZE = int(os.getenv("PDF_TRANSLATION_BATCH_SIZE", "40"))
PDF_TRANSLATION_MAX_WORKERS = int(os.getenv("PDF_TRANSLATION_MAX_WORKERS", "6"))
VERBOSE = os.getenv("PDF_TRANSLATION_VERBOSE", "1").strip().lower() in ("1", "true", "yes", "on")


# -----------------------------
# Small helpers
# -----------------------------
_PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")

def _norm_lang(lang: str) -> str:
    t = (lang or "").strip().lower()
    mapping = {
        "english": "en", "japanese": "ja", "german": "de",
        "french": "fr", "spanish": "es", "italian": "it",
        "portuguese": "pt", "dutch": "nl", "russian": "ru",
        "chinese": "zh", "korean": "ko", "arabic": "ar"
    }
    t = mapping.get(t, t)
    return t.split("-", 1)[0] if t else t

def _has_alpha(s: str) -> bool:
    return any(ch.isalpha() for ch in s)

def _should_skip_translation(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # skip “mostly non-linguistic” fragments
    if not _has_alpha(t):
        return True
    # common junk: pure URLs / file paths (leave as-is)
    if re.fullmatch(r"(https?://\S+|www\.\S+)", t, flags=re.IGNORECASE):
        return True
    return False

def _extract_placeholders(s: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(s or "")

def _preserves_placeholders(out: str, src: str) -> bool:
    src_ph = _extract_placeholders(src)
    if not src_ph:
        return True
    out_s = out or ""
    return all(p in out_s for p in src_ph)

def _safe_print(msg: str, *, verbose: Optional[bool] = None) -> None:
    if VERBOSE if verbose is None else bool(verbose):
        print(msg)

def _preview(s: str, limit: int = 180) -> str:
    t = (s or "").replace("\n", "\\n")
    return t if len(t) <= limit else t[: limit - 1] + "…"


# -----------------------------
# Thread-local HTTP sessions (thread-safe/faster)
# -----------------------------
_tls = threading.local()

def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        setattr(_tls, "session", s)
    return s


# -----------------------------
# Azure Translator (MT)
# -----------------------------
class AzureTranslator:
    def __init__(self) -> None:
        if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_ENDPOINT:
            raise RuntimeError(
                "Missing Azure Translator credentials. "
                "Set AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT."
            )

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts:
            return []
        src = _norm_lang(source_lang)
        tgt = _norm_lang(target_lang)

        endpoint = AZURE_TRANSLATOR_ENDPOINT.rstrip("/")
        url = f"{endpoint}/translate"
        params = {
            "api-version": AZURE_TRANSLATOR_API_VERSION or "3.0",
            "from": src,
            "to": tgt,
        }
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
            "Content-Type": "application/json",
        }
        if AZURE_TRANSLATOR_REGION:
            headers["Ocp-Apim-Subscription-Region"] = AZURE_TRANSLATOR_REGION

        body = [{"text": t} for t in texts]

        # simple retries
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = _session().post(url, params=params, headers=headers, data=json.dumps(body), timeout=60)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Translator HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                out: List[str] = []
                # response aligns with input order
                for i, item in enumerate(data):
                    try:
                        out.append(item["translations"][0]["text"])
                    except Exception:
                        out.append(texts[i])
                return out
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (2**attempt))
        # fallback to originals on failure
        _safe_print(f"[MT] batch failed, falling back to source. err={last_err}")
        return list(texts)


# -----------------------------
# Azure OpenAI Chat (LLM1)
# -----------------------------
class AzureOpenAIChat:
    def __init__(self) -> None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT_NAME:
            raise RuntimeError(
                "Missing Azure OpenAI credentials. Set "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME."
            )

    def chat_json(self, system: str, user_payload: Dict[str, Any], *, temperature: float = 0.2, max_tokens: int = 800) -> Dict[str, Any]:
        endpoint = AZURE_OPENAI_ENDPOINT.rstrip("/")
        dep = AZURE_OPENAI_DEPLOYMENT_NAME
        url = f"{endpoint}/openai/deployments/{dep}/chat/completions"
        params = {"api-version": AZURE_OPENAI_API_VERSION}
        headers = {
            "api-key": AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }
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
                resp = _session().post(url, params=params, headers=headers, data=json.dumps(body), timeout=90)
                if resp.status_code >= 400:
                    raise RuntimeError(f"AOAI HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except Exception:
                    # sometimes model returns JSON-ish text; try to salvage
                    return {"translation": content.strip()}
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (2**attempt))
        raise RuntimeError(f"AOAI chat failed after retries: {last_err}")


def _llm1_refine(
    aoai: AzureOpenAIChat,
    *,
    source_text: str,
    mt_text: str,
    context_prev_10: List[Dict[str, str]],
    source_lang: str,
    target_lang: str,
) -> str:
    system = (
        f"You are a professional technical translator and editor ({source_lang.upper()} -> {target_lang.upper()}).\n"
        "Task: Improve the provided machine translation so it is grammatically correct, fluent, and faithful.\n"
        "Rules:\n"
        " - Preserve meaning exactly.\n"
        " - Preserve numbers, units, codes, part numbers, and tokens like [[...]] exactly.\n"
        " - Do not add extra commentary.\n"
        "Output JSON with key: translation\n"
    )

    payload = {
        "source_lang": _norm_lang(source_lang),
        "target_lang": _norm_lang(target_lang),
        "previous_context": context_prev_10, # list of {"src":..., "mt":...}
        "source": source_text,
        "machine_translation": mt_text,
        "instructions": "Return only JSON.",
    }

    out = aoai.chat_json(system, payload, temperature=0.2, max_tokens=900)
    cand = (out.get("translation") or "").strip()
    return cand if cand else mt_text


# -----------------------------
# Public API: translate_blocks
# -----------------------------
def translate_blocks(
    blocks: List[BlockRef],
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_workers: Optional[int] = None,
    top_k_paragraphs: int = 5,
    verbose: Optional[bool] = None,
    return_diagnostics: bool = False,
    with_diagnostics: bool = False,
    debug: Optional[bool] = None,
    log_every_n: int = 5,
) -> Union[List[BlockTranslation], Tuple[List[BlockTranslation], Dict[str, Any]]]:
    """
    Three-step pipeline (no cache):
        1) Azure Translator (baseline MT)
        2) LLM1 (grammar / structure fix) — uses prev 10 chunks (src + mt) as context
        3) LLM2 (RAG refine) — uses refine_segment_with_glossary(), language-only filters

    Multithreading:
        - runs in 3 stages with ThreadPoolExecutor
        - preserves deterministic order + context windows

    Backward-compatible extras:
        - verbose / debug
        - return_diagnostics / with_diagnostics
    """
    t_pipeline_start = time.perf_counter()

    if source_lang is None:
        source_lang = SOURCE_LANG_DEFAULT
    if target_lang is None:
        target_lang = TARGET_LANG_DEFAULT
    if max_workers is None:
        max_workers = max(1, PDF_TRANSLATION_MAX_WORKERS)

    eff_verbose = VERBOSE if verbose is None else bool(verbose)
    if debug is not None:
        eff_verbose = bool(debug)

    want_diagnostics = bool(return_diagnostics or with_diagnostics)
    log_every_n = max(1, int(log_every_n or 5))

    # keep deterministic order as provided
    n = len(blocks)
    src_texts = [b.text for b in blocks]

    # precompute which to skip
    skip_mask = [_should_skip_translation(t) for t in src_texts]
    skip_count = sum(skip_mask)

    # granular skip reasons (heuristics stage)
    skip_reasons = {
        "empty_or_whitespace": 0,
        "no_alpha": 0,
        "url_or_www_only": 0,
        "other": 0,
    }
    for t in src_texts:
        s = (t or "").strip()
        if not s:
            skip_reasons["empty_or_whitespace"] += 1
        elif not _has_alpha(s):
            skip_reasons["no_alpha"] += 1
        elif re.fullmatch(r"(https?://\S+|www\.\S+)", s, flags=re.IGNORECASE):
            skip_reasons["url_or_www_only"] += 1
        else:
            # not skipped by heuristic
            pass

    _safe_print(
        f"[TRANSLATE] Total blocks: {n}, skipping {skip_count} based on heuristics "
        f"(workers={max_workers}, batch_size={PDF_TRANSLATION_BATCH_SIZE}, src={source_lang}, tgt={target_lang})",
        verbose=eff_verbose,
    )
    _safe_print(f"[TRANSLATE][SKIP][HEURISTIC] reasons={skip_reasons}", verbose=eff_verbose)

    translator = AzureTranslator()
    aoai = AzureOpenAIChat()

    # Diagnostics / timing
    stats_lock = threading.Lock()
    timings: Dict[str, float] = {
        "azure_mt_total": 0.0, # wall time sum of MT batches
        "llm1_total": 0.0, # sum of per-call LLM1 elapsed
        "llm2_total": 0.0, # sum of per-call LLM2 elapsed
        "end_to_end_sum": 0.0, # sum of per-chunk approx latencies across stages
        "stage_mt_wall": 0.0,
        "stage_llm1_wall": 0.0,
        "stage_llm2_wall": 0.0,
        "translate_blocks_wall": 0.0,
    }
    counts: Dict[str, int] = {
        "chunks_total": n,
        "cache_hits": 0, # no cache in this implementation
        "skip_heuristic": skip_count,
        "mt_todo": 0,
        "mt_batches": 0,
        "mt_failed_batches": 0,
        "mt_identity_or_empty_fallbacks": 0,
        "llm1_candidates": 0,
        "llm1_failures": 0,
        "llm2_candidates": 0,
        "llm2_failures": 0,
        "errors": 0,
    }

    # per-chunk timing accumulator (approx; MT batch time distributed equally across batch)
    per_chunk_total: List[float] = [0.0] * n

    # -------------------------
    # Stage 1: MT (batched + threaded)
    # -------------------------
    mt_out: List[str] = list(src_texts)

    # create batches of indices we actually translate
    todo_indices = [i for i in range(n) if not skip_mask[i]]
    batches: List[List[int]] = [todo_indices[i : i + PDF_TRANSLATION_BATCH_SIZE] for i in range(0, len(todo_indices), PDF_TRANSLATION_BATCH_SIZE)]
    counts["mt_todo"] = len(todo_indices)
    counts["mt_batches"] = len(batches)

    _safe_print(
        f"[TRANSLATE][STAGE1-MT] todo={len(todo_indices)}, skipped={skip_count}, batches={len(batches)}",
        verbose=eff_verbose,
    )

    def mt_worker(idxs: List[int]) -> Tuple[List[int], List[str], float]:
        t0 = time.perf_counter()
        texts = [src_texts[i] for i in idxs]
        outs = translator.translate_batch(texts, source_lang, target_lang)
        elapsed = time.perf_counter() - t0
        return idxs, outs, elapsed

    t_stage_mt_start = time.perf_counter()
    if batches:
        _safe_print(f"[MT] translating {len(todo_indices)} chunks in {len(batches)} batch(es) with {max_workers} workers", verbose=eff_verbose)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(mt_worker, b) for b in batches]
            for fut_idx, fut in enumerate(as_completed(futs), start=1):
                try:
                    idxs, outs, elapsed = fut.result()
                except Exception as e:
                    # defensive fallback (translate_batch already handles failures, but keeping this for safety)
                    with stats_lock:
                        counts["mt_failed_batches"] += 1
                        counts["errors"] += 1
                    _safe_print(f"[MT] worker future failed: {e!r}", verbose=eff_verbose)
                    continue

                for i, out_t in zip(idxs, outs):
                    mt_out[i] = out_t if out_t is not None else src_texts[i]

                # timing bookkeeping (distribute batch elapsed across items for per-chunk end-to-end)
                with stats_lock:
                    timings["azure_mt_total"] += elapsed
                    if idxs:
                        share = elapsed / len(idxs)
                        for i in idxs:
                            per_chunk_total[i] += share

                if eff_verbose:
                    sample_i = idxs[0] if idxs else None
                    sample_src = _preview(src_texts[sample_i], 110) if sample_i is not None else ""
                    sample_mt = _preview(mt_out[sample_i], 110) if sample_i is not None else ""
                    _safe_print(
                        f"[MT][BATCH {fut_idx}/{len(batches)}] size={len(idxs)} elapsed={elapsed:.2f}s "
                        f"sample_i={sample_i} src='{sample_src}' -> mt='{sample_mt}'",
                        verbose=eff_verbose,
                    )
    timings["stage_mt_wall"] = time.perf_counter() - t_stage_mt_start

    # if MT outputs identical to source or empty, keep source (no-op)
    mt_identity_or_empty = 0
    for i in range(n):
        if skip_mask[i]:
            mt_out[i] = src_texts[i]
        else:
            if not mt_out[i] or mt_out[i].strip() == "" or mt_out[i] == src_texts[i]:
                mt_out[i] = src_texts[i]
                mt_identity_or_empty += 1
    counts["mt_identity_or_empty_fallbacks"] = mt_identity_or_empty

    _safe_print(
        f"[TRANSLATE][STAGE1-MT][SUMMARY] processed={len(todo_indices)}, skipped={skip_count}, "
        f"identity_or_empty_fallbacks={mt_identity_or_empty}, wall={timings['stage_mt_wall']:.2f}s",
        verbose=eff_verbose,
    )

    # -------------------------
    # Stage 2: LLM1 (parallel; context is previous 10 src+mt)
    # -------------------------
    pass1: List[str] = list(mt_out)

    # store per-index info to create deterministic "1 in 5" samples later.
    llm1_debug_by_index: Dict[int, Dict[str, str]] = {}
    llm1_debug_lock = threading.Lock()

    def llm1_task(i: int) -> Tuple[int, str, float]:
        t0 = time.perf_counter()
        if skip_mask[i]:
            return i, src_texts[i], (time.perf_counter() - t0)

        ctx: List[Dict[str, str]] = []
        start = max(0, i - 10)
        for j in range(start, i):
            ctx.append({"src": src_texts[j], "mt": mt_out[j]})

        src = src_texts[i].strip()
        mt = mt_out[i].strip()
        if not src or _should_skip_translation(src):
            return i, src_texts[i], (time.perf_counter() - t0)

        # placeholder safety
        is_placeholder = ("[[BLOCK" in src) or ("[[INLINE" in src) or bool(_extract_placeholders(src))

        try:
            refined = _llm1_refine(
                aoai,
                source_text=src,
                mt_text=mt,
                context_prev_10=ctx,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if is_placeholder and not _preserves_placeholders(refined, src):
                # fallback to MT or source
                if _preserves_placeholders(mt, src):
                    refined = mt
                else:
                    refined = src
            return i, refined, (time.perf_counter() - t0)
        except Exception as e:
            with stats_lock:
                counts["llm1_failures"] += 1
                counts["errors"] += 1
            _safe_print(f"[LLM1] failed at i={i}: {e} -> fallback to MT", verbose=eff_verbose)
            return i, mt_out[i], (time.perf_counter() - t0)

    llm1_indices = [i for i in range(n) if not skip_mask[i] and mt_out[i] != src_texts[i]]
    counts["llm1_candidates"] = len(llm1_indices)

    _safe_print(
        f"[TRANSLATE][STAGE2-LLM1] candidates={len(llm1_indices)}, skipped_heuristic={skip_count}, "
        f"skipped_no_mt_change={len(todo_indices) - len(llm1_indices)}",
        verbose=eff_verbose,
    )

    t_stage_llm1_start = time.perf_counter()
    if llm1_indices:
        _safe_print(f"[LLM1] refining {len(llm1_indices)} chunk(s) with {max_workers} workers", verbose=eff_verbose)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(llm1_task, i) for i in llm1_indices]
            for done_idx, fut in enumerate(as_completed(futs), start=1):
                i, out_t, elapsed = fut.result()
                pass1[i] = out_t

                with stats_lock:
                    timings["llm1_total"] += elapsed
                    per_chunk_total[i] += elapsed

                # keep deterministic sample data by source index
                with llm1_debug_lock:
                    llm1_debug_by_index[i] = {
                        "original": src_texts[i],
                        "translated": out_t,
                        "mt": mt_out[i],
                    }

                if eff_verbose and (done_idx % log_every_n == 0):
                    _safe_print(
                        f"[LLM1][PROGRESS] done={done_idx}/{len(llm1_indices)} | i={i} | elapsed={elapsed:.2f}s",
                        verbose=eff_verbose,
                    )
                    _safe_print(f"  [LLM1 sample] original  : {_preview(src_texts[i], 180)}", verbose=eff_verbose)
                    _safe_print(f"  [LLM1 sample] translated: {_preview(out_t, 180)}", verbose=eff_verbose)
    timings["stage_llm1_wall"] = time.perf_counter() - t_stage_llm1_start

    _safe_print(
        f"[TRANSLATE][STAGE2-LLM1][SUMMARY] candidates={len(llm1_indices)}, "
        f"failures={counts['llm1_failures']}, wall={timings['stage_llm1_wall']:.2f}s",
        verbose=eff_verbose,
    )

    # -------------------------
    # Stage 3: LLM2 (RAG refine; parallel; language-only)
    # -------------------------
    final_out: List[str] = list(pass1)

    def llm2_task(i: int) -> Tuple[int, str, float]:
        t0 = time.perf_counter()
        if skip_mask[i]:
            return i, src_texts[i], (time.perf_counter() - t0)

        src = src_texts[i].strip()
        cur = pass1[i].strip()

        if not src or _should_skip_translation(src):
            return i, src_texts[i], (time.perf_counter() - t0)

        is_placeholder = ("[[BLOCK" in src) or ("[[INLINE" in src) or bool(_extract_placeholders(src))

        # OPTIONAL: can use prev 10 pass1 chunks as extra context inside your refine function
        try:
            refined = refine_segment_with_glossary(
                english_chunk=src,
                current_translation=cur,
                is_placeholder=is_placeholder,
                verbose=eff_verbose, # use effective verbose, not env-only VERBOSE
                top_k_paragraphs=top_k_paragraphs,
                source_lang=source_lang,
                target_lang=target_lang
            )
            if is_placeholder and not _preserves_placeholders(refined, src):
                refined = cur
            return i, refined if refined else cur, (time.perf_counter() - t0)
        except Exception as e:
            with stats_lock:
                counts["llm2_failures"] += 1
                counts["errors"] += 1
            _safe_print(f"[LLM2] failed at i={i}: {e} -> keep LLM1", verbose=eff_verbose)
            return i, cur, (time.perf_counter() - t0)

    llm2_indices = [i for i in range(n) if not skip_mask[i] and pass1[i] != src_texts[i]]
    counts["llm2_candidates"] = len(llm2_indices)

    _safe_print(
        f"[TRANSLATE][STAGE3-LLM2] candidates={len(llm2_indices)}, skipped_heuristic={skip_count}, "
        f"skipped_no_llm1_change={len(todo_indices) - len(llm2_indices)}",
        verbose=eff_verbose,
    )

    t_stage_llm2_start = time.perf_counter()
    if llm2_indices:
        _safe_print(f"[LLM2] refining {len(llm2_indices)} chunk(s) with {max_workers} workers", verbose=eff_verbose)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(llm2_task, i) for i in llm2_indices]
            for done_idx, fut in enumerate(as_completed(futs), start=1):
                i, out_t, elapsed = fut.result()
                final_out[i] = out_t

                with stats_lock:
                    timings["llm2_total"] += elapsed
                    per_chunk_total[i] += elapsed

                if eff_verbose and (done_idx % log_every_n == 0):
                    _safe_print(
                        f"[LLM2][PROGRESS] done={done_idx}/{len(llm2_indices)} | i={i} | elapsed={elapsed:.2f}s",
                        verbose=eff_verbose,
                    )
    timings["stage_llm2_wall"] = time.perf_counter() - t_stage_llm2_start

    _safe_print(
        f"[TRANSLATE][STAGE3-LLM2][SUMMARY] candidates={len(llm2_indices)}, "
        f"failures={counts['llm2_failures']}, wall={timings['stage_llm2_wall']:.2f}s",
        verbose=eff_verbose,
    )

    # finalize end-to-end sum (sum of per-chunk approx translation latency across stages)
    timings["end_to_end_sum"] = float(sum(per_chunk_total))
    timings["translate_blocks_wall"] = time.perf_counter() - t_pipeline_start

    # -------------------------
    # Return BlockTranslation list in original order
    # -------------------------
    out: List[BlockTranslation] = []
    for i, b in enumerate(blocks):
        out.append(BlockTranslation(block=b, translated_text=final_out[i]))

    # deterministic LLM1 sample list in original order (wrapper will print 1 in 5)
    llm1_pairs: List[Dict[str, Any]] = []
    for i in sorted(llm1_debug_by_index.keys()):
        row = llm1_debug_by_index[i]
        llm1_pairs.append({
            "index": i,
            "original": row.get("original", ""),
            "translated": row.get("translated", ""),
            "mt": row.get("mt", ""),
        })

    diagnostics: Dict[str, Any] = {
        # top-level keys (pdf pipeline logger looks for these)
        "cache_hits": counts["cache_hits"],
        "azure_mt_total": timings["azure_mt_total"],
        "llm1_total": timings["llm1_total"],
        "llm2_total": timings["llm2_total"],
        "end_to_end_sum": timings["end_to_end_sum"],
        "errors": counts["errors"],

        # structured details
        "counts": counts,
        "timings": timings,
        "skip_summary": {
            "heuristic_total": skip_count,
            "heuristic_reasons": skip_reasons,
            "stage1_mt": {
                "todo": counts["mt_todo"],
                "skipped_heuristic": skip_count,
                "batches": counts["mt_batches"],
                "identity_or_empty_fallbacks": counts["mt_identity_or_empty_fallbacks"],
                "failed_batches": counts["mt_failed_batches"],
            },
            "stage2_llm1": {
                "candidates": counts["llm1_candidates"],
                "skipped_heuristic": skip_count,
                "skipped_no_mt_change": len(todo_indices) - len(llm1_indices),
                "failures": counts["llm1_failures"],
            },
            "stage3_llm2": {
                "candidates": counts["llm2_candidates"],
                "skipped_heuristic": skip_count,
                "skipped_no_llm1_change": len(todo_indices) - len(llm2_indices),
                "failures": counts["llm2_failures"],
            },
        },
        # samples for pdf pipeline log printer
        "llm1_pairs": llm1_pairs,
        "samples": {
            "llm1_pairs": llm1_pairs,
        },
    }

    if eff_verbose:
        _safe_print("[TRANSLATE][TIMING] Summary", verbose=eff_verbose)
        _safe_print(f"  chunks_total={counts['chunks_total']}", verbose=eff_verbose)
        _safe_print(f"  cache_hits={counts['cache_hits']}", verbose=eff_verbose)
        _safe_print(f"  azure_mt_total={timings['azure_mt_total']:.2f}s", verbose=eff_verbose)
        _safe_print(f"  llm1_total={timings['llm1_total']:.2f}s", verbose=eff_verbose)
        _safe_print(f"  llm2_total={timings['llm2_total']:.2f}s", verbose=eff_verbose)
        _safe_print(f"  end_to_end_sum={timings['end_to_end_sum']:.2f}s", verbose=eff_verbose)
        _safe_print(f"  translate_blocks_wall={timings['translate_blocks_wall']:.2f}s", verbose=eff_verbose)
        _safe_print(f"  errors={counts['errors']}", verbose=eff_verbose)

    if want_diagnostics:
        return out, diagnostics
    return out