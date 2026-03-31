from typing import Dict, Any, List, Optional

def preview_string(s: str, limit: int = 180) -> str:
    """Helper to preview strings safely in logs/prompts."""
    t = (s or "").replace("\n", "\\n")
    return t if len(t) <= limit else t[: limit - 1] + "…"

def build_llm1_refinement_prompt(
    source_text: str,
    mt_text: str,
    previous_chunks: Optional[List[Dict[str, str]]],
    source_lang: str,
    target_lang: str,
    is_short_mode: bool = False,
    kind: Optional[str] = None,
    paragraph_context: Optional[List[str]] = None,
    source_file: Optional[str] = "pdf"
) -> Dict[str, Any]:
    """
    Builds the system instructions and user payload for the LLM1 (Grammar/Fluency) refinement stage.
    """
    system_lines = [
        f"You are an EXPERIENCED LANGUAGE TRANSLATOR AND EDITOR for ({source_lang.upper()} -> {target_lang.upper()}).",
        # f"Task: IMPROVE the provided machine translation so it is GRAMMATICALLY CORRECT, NATURAL, and FAITHFUL to the original content (also provided to you). The result must read as if translated by a skilled native professional in the {target_lang.upper()} language.",
        f"### TASK:\nUsing the provided source text in {source_lang.upper()} language, and it's machine translation as context, provide a REVISED TRANSLATION in {target_lang.upper()} that is GRAMMATICALLY CORRECT, NATURAL, and most importantly, FAITHFUL to the original content. The result must read as if translated by a skilled native professional in the {target_lang.upper()} language. Do not consider this a free-form creative writing task. Your goal is to produce the most accurate, natural, and fluent translation possible that fully preserves the meaning, nuance, tone, and formality of the original text. You should use the machine translation as a base reference, but you are not obligated to preserve its wording or structure if it can be improved for naturalness and fluency while still being faithful to the source meaning.",
        "### STRICT RULES TO FOLLOW:",
        "   - BE FAITHFUL TO SOURCE MEANING, NOT TO SOURCE WORDING.",
        f"  - ALWAYS PRESERVE the source meaning, intent, tone, and level of formality EXACTLY. However, DO NOT blindly use source-language ({source_lang.upper()}) wording, word order, or sentence structure when a MORE NATURAL target-language ({target_lang.upper()}) expression conveys the SAME MEANING.",
        f"  - AVOID word-for-word translation and translationese (overly-literal translations from {source_lang.upper()} to {target_lang.upper()}). Your output should be NATURAL and FLUENT while also MAINTAINING THE MEANING AND NUANCE OF THE CONTENT.",
        f"  - PROPER NOUNS: For names, labels, and fixed expressions, prefer the form that sounds STANDARD and NATURAL in the target-language ({target_lang.upper()}) context. Do not mechanically transliterate or literally translate if a more established or contextually appropriate form is clearly indicated. Do NOT leave them in source-language ({source_lang.upper()}) if they have a target-language ({target_lang.upper()}) equivalent.",
        "   - ALWAYS PRESERVE numbers, units, codes, part numbers, and tokens like '[[INLINE#]]' and '[[BLOCK#]]' EXACTLY.",
        f"  - PUNCTUATION: You must also use punctuations where required for naturalness and grammaticality in the target language ({target_lang.upper()}), even if they are not present in the source. However, do NOT add punctuation if it would change the meaning or tone of the original text. Also, keep in mind the differences in punctuation norms and special characters between the source ({source_lang.upper()}) and target ({target_lang.upper()}) languages. Convert punctuation to the appropriate form for the target language ({target_lang.upper()}) when necessary, but STRICTLY DO NOT add punctuation that changes meaning or tone.",
        "   - ALWAYS PRESERVE tokens like '[L#]' and '[/L#]' which indicate line breaks in the original PDF. Do not add or remove them, and keep them in the same position relative to the text.",
        "   - CRITICAL: DO NOT merge content between different '[L#]' tags. Every [L#] tag provided in the input MUST exist in the output with its corresponding line's content. Do not combine L5 and L6 into one block.",
        "   - NEVER add extra commentary, chat, or information. This includes any auto-completion of facts, prefatory remarks, explanations, apologies, or notes about the translation process. Provide ONLY the translation text as output.",
    ]
    # file-type specific instructions
    if source_file == "word":
        print("Applying WORD file instructions")
        system_lines.append("   - CRITICAL: STRICTLY PRESERVE ALL WHITESPACE EXACTLY as in the input. Do NOT remove, add, or shift spaces between words, even if the spacing appears unnatural.\n   - NEVER merge words together. If two words are separated by a space in the input, they MUST remain separated in the output.")

    # kind-aware specific instructions
    if kind == "TABLE_CELL":
        print(f"Applying TABLE_CELL instructions for source: '{preview_string(source_text)}'")
        system_lines.append("   - CONTEXT: This is a table cell. Keep it professional and avoid leading/trailing punctuation if not present in source.")
    elif kind == "HEADER_FOOTER":
        print(f"Applying HEADER_FOOTER instructions for source: '{preview_string(source_text)}'")
        system_lines.append("   - CONTEXT: This is a header or footer. Keep it brief and formal.")
    elif kind == "LABEL":
        print(f"Applying LABEL instructions for source: '{preview_string(source_text)}'")
        system_lines.append("   - CONTEXT: This is a short label inside a diagram or UI. Conciseness is CRITICAL.")
    if is_short_mode:
        print(f"Applying SHORT MODE instructions for source: '{preview_string(source_text)}'")
        system_lines.append("   - SHORT MODE ACTIVE: Prefer the shortest natural and professionally acceptable translation that fits tight space. Do not omit information that changes meaning, tone, role, or terminology. Avoid unnatural compression.")
    
    # language-specific instructions
    if source_lang.lower() in ("ja", "jp", "japanese"):
        print(f"Source is Japanese, applying extra instructions for text: '{preview_string(source_text)}'")
        system_lines.append(f"""    - STRICTLY ALWAYS change Japanese punctuation marks like 【】 brackets, ' apostrophies to the appropriate form for the target language ({target_lang.upper()}).""")
    system_lines.append("STRICTLY ALWAYS PROVIDE OUTPUT JSON with key: translation")
    
    system = "\n".join(system_lines)

    user_payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "previous_context": previous_chunks,
        "source": source_text,
        "machine_translation": mt_text,
        "instructions": "Return only JSON.",
    }

    if kind: 
        user_payload["container_kind"] = kind

    if previous_chunks: print(f"\n\n\nPREVIOUS CONTEXT ({len(previous_chunks)} chunks):\n{previous_chunks}\n\n\n")

    # paragraph context: sibling lines that belong to the same paragraph group; gives the LLM full-sentence context for grammar quality even though each line is translated and placed individually.
    if paragraph_context:
        pass
        # user_payload["paragraph_context"] = paragraph_context

    return {
        "system": system,
        "user_payload": user_payload
    }
