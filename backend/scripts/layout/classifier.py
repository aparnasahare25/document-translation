import re
from typing import Dict, Any

from scripts.layout.containers import ContainerRef, ContainerKind, TranslationPolicy

def classify_container(container: ContainerRef, verbose: bool = False) -> TranslationPolicy:
    text = container.text.strip()
    
    # 1. Skip Empty or Tiny Symbols
    if not text or len(text) <= 1:
        return TranslationPolicy.SKIP
        
    # 2. Skip Pure Numbers, Punctuation, and Currencies
    # e.g., "123.45", "(1,000)", "$45.00"
    if re.fullmatch(r'[\d\s.,\-+*/=()$%€£¥]+', text):
        return TranslationPolicy.SKIP
        
    # 3. Skip Weird Artifacts / Hyphen-only / Dots-only
    if re.fullmatch(r'[\-%_.]+', text):
        return TranslationPolicy.SKIP
        
    # 4. Skip Serial numbers, Hex codes, strict alphanumeric codes
    # Must contain at least one digit and be composed of letters, numbers, and dashes
    code_match = re.fullmatch(r'[A-Za-z0-9\-_]+', text)
    if code_match and any(c.isdigit() for c in text):
        if len(text.split()) == 1:
            return TranslationPolicy.SKIP
            
    # 5. Check for Partial Translate: mixed prose + dense codes
    # Heuristic: Count "code-like" words (uppercase acronyms, alphanumeric strings) vs normal words
    words = text.split()
    if len(words) > 2:
        code_words = [w for w in words if (any(c.isdigit() for c in w) or w.isupper()) and len(w) > 1]
        ratio = len(code_words) / len(words)
        
        # If it's heavily coded but has prose around it, flag as partial
        if ratio >= 0.3 and ratio < 0.9:
            return TranslationPolicy.PARTIAL_TRANSLATE
            
    # 6. Tiny label containers often require restricted expansions (keep tight)
    if container.kind == ContainerKind.LABEL:
        return TranslationPolicy.RESTRICT_EXPANSION
        
    # 7. Otherwise, standard translation
    return TranslationPolicy.TRANSLATE
