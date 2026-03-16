from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


class ContainerKind(str, Enum):
    TABLE_CELL = "TABLE_CELL"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    HEADER_FOOTER = "HEADER_FOOTER"
    LABEL = "LABEL"

@dataclass
class PdfSpanAttrs:
    rect: Tuple[float, float, float, float] # fitz coords
    text: str
    font: str
    size: float
    color: int # integer color from fitz
    origin: Tuple[float, float]
    flags: int
    ascender: float
    descender: float


@dataclass
class ContainerRef:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    kind: ContainerKind
    polygon: Optional[List[Tuple[float, float]]] = None
    reading_key: int = 0
    style_hints: Dict[str, Any] = field(default_factory=dict)
    original_spans: List[PdfSpanAttrs] = field(default_factory=list) # for targeted removal
    source_layer: str = "docint"
    # Layout grouping: lines in the same paragraph group share an ID (e.g. "p2_3").
    # none for table cells and standalone elements; used to form context windows in LLM1
    paragraph_group_id: Optional[str] = None

@dataclass
class ContainerTranslation:
    container: ContainerRef
    translated_text: str

class TranslationPolicy(str, Enum):
    SKIP = "SKIP"
    PARTIAL_TRANSLATE = "PARTIAL_TRANSLATE"
    TRANSLATE = "TRANSLATE"
    RESTRICT_EXPANSION = "RESTRICT_EXPANSION"

@dataclass
class RenderingIntent:
    font_name: str = "helv"
    font_size_start: float = 11.0
    alignment: int = 0 # 0: left, 1: center, 2: right, 3: justify (fitz default aligns)
    rotation: int = 0
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class TranslationPlan:
    container: ContainerRef
    normalized_source_text: str
    protected_tokens_map: Dict[str, str]
    translated_text: str
    final_rendered_text: str
    rendering_intent: RenderingIntent
    policy: TranslationPolicy
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
