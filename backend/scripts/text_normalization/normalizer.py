import re
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class NormalizationState:
    original_text: str
    placeholders: Dict[str, str]

# Common artifacts list (configurable)
OCR_ARTIFACTS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Script ranges for de-spacing (unnatural spaces between characters)
CJK_RANGE = r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]'

# Protected tokens
URL_EMAIL_RE = r'\b(?:https?://|www\.)\S+\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
FILE_PATH_RE = r'(?:/[a-zA-Z0-9_.-]+)+/?|[A-Z]:\\[a-zA-Z0-9_.-\\]+'
PART_NUMBER_RE = r'\b[A-Z0-9]+-(?:[A-Z0-9]+-)*[A-Z0-9]+\b|\b[A-Z]+\d+[A-Z0-9]*\b|\b\d+[A-Z]+[A-Z0-9]*\b|\b\d{5,12}\b'
COMMON_WORDS = "THE|AND|FOR|NOT|BUT|ARE|YOU|CAN|HAS|HIS|HER|WAS|ALL|ANY|OUT|HOW|WHO|THIS|THAT|WITH|FROM"
ACRONYM_RE = rf'\b(?!(?:{COMMON_WORDS})\b)[A-Z]{{3,7}}\b'
MEASUREMENT_RE = r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|kg|g|mg|l|ml|V|A|Hz|W|kW|MPa|psi|°C|°F|%)\b'
UI_MARKER_RE = r'[▶▸■◆●➔➞►▼▲☑☐☒]'

PROTECTED_COMBINED_RE = re.compile(
    f'({URL_EMAIL_RE})|({FILE_PATH_RE})|({PART_NUMBER_RE})|({MEASUREMENT_RE})|({ACRONYM_RE})|({UI_MARKER_RE})'
)

def normalize_whitespace(text: str) -> str:
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse repeated spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove weird OCR artifacts
    text = OCR_ARTIFACTS_RE.sub('', text)
    return text.strip()

def script_aware_despace(text: str) -> str:
    # Remove unnatural spaces between non-Latin script characters (e.g. CJK)
    pattern = f'({CJK_RANGE})[ \t]+({CJK_RANGE})'
    # run twice to catch overlapping matches
    text = re.sub(pattern, r'\1\2', text)
    text = re.sub(pattern, r'\1\2', text)
    return text

def extract_protected_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    placeholders = {}
    counter = 0

    def repl(match):
        nonlocal counter
        token = match.group(0)
        # Avoid replacing purely whitespace or empty
        if not token.strip():
            return token
        ph = f"[[INLINE{counter}]]"
        placeholders[ph] = token
        counter += 1
        return ph

    new_text = PROTECTED_COMBINED_RE.sub(repl, text)
    return new_text, placeholders

def apply_normalization_pipeline(text: str) -> Tuple[str, NormalizationState]:
    norm_text = normalize_whitespace(text)
    norm_text = script_aware_despace(norm_text)
    ph_text, placeholders = extract_protected_tokens(norm_text)
    
    return ph_text, NormalizationState(original_text=text, placeholders=placeholders)

def restore_protected_tokens(translated_text: str, state: NormalizationState) -> str:
    restored = translated_text
    # Restore placedholders
    for ph, token in state.placeholders.items():
        restored = restored.replace(ph, token)
        
    return restored
