import re
from typing import Dict, Any

from scripts.layout.containers import ContainerRef, ContainerKind, TranslationPolicy


def classify_container(container: ContainerRef, verbose: bool = False) -> TranslationPolicy:
    text = container.text.strip()
    
    # 1. skip empty or tiny symbols
    if not text or len(text) <= 1:
        print(f"Classify: SKIP empty/tiny text: '{text}'") if verbose else None
        return TranslationPolicy.SKIP
        
    # 2. skip pure numbers, punctuation, and currencies
    # e.g., "123.45", "(1,000)", "$45.00"
    if re.fullmatch(r'[\d\s.,\-+*/=()$%€£¥]+', text):
        print(f"Classify: SKIP numeric/punct text: '{text}'") if verbose else None
        return TranslationPolicy.SKIP
        
    # 3. skip weird artifacts / hyphen-only / dots-only
    if re.fullmatch(r'[\-%_.]+', text):
        print(f"Classify: SKIP artifact text: '{text}'") if verbose else None
        return TranslationPolicy.SKIP
        
    # 4. skip serial numbers, hex codes, strict alphanumeric codes
    # must contain at least one digit and be composed of letters, numbers, and dashes
    code_match = re.fullmatch(r'[A-Za-z0-9\-_]+', text)
    if code_match and any(c.isdigit() for c in text):
        if len(text.split()) == 1:
            print(f"Classify: SKIP code-like text: '{text}'") if verbose else None
            return TranslationPolicy.SKIP
            
    # 5. check for partial translate: mixed prose + dense codes
    # heuristic: count "code-like" words (uppercase acronyms, alphanumeric strings) vs normal words
    words = text.split()
    if len(words) > 2:
        code_words = [w for w in words if (any(c.isdigit() for c in w) or w.isupper()) and len(w) > 1]
        ratio = len(code_words) / len(words)
        
        # if heavily coded but has prose around it, flag as partial
        if ratio >= 0.3 and ratio < 0.9:
            print(f"Classify: PARTIAL_TRANSLATE mixed text: '{text}' (code ratio: {ratio:.2f})") if verbose else None
            return TranslationPolicy.PARTIAL_TRANSLATE
            
    # 6. tiny label containers often require restricted expansions (keep tight)
    if container.kind == ContainerKind.LABEL:
        print(f"Classify: RESTRICT_EXPANSION for LABEL container: '{text}'") if verbose else None
        return TranslationPolicy.RESTRICT_EXPANSION
        
    # 7. otherwise, standard translation
    print(f"Classify: TRANSLATE standard text: '{text}'") if verbose else None
    return TranslationPolicy.TRANSLATE
