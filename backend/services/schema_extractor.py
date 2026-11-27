"""Shared prompt & class-list schema extraction for detection/segmentation engines."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA_KEY_TOKENS = frozenset(
    {
        "bbox_2d",
        "bbox",
        "box",
        "boxes",
        "polygon",
        "polygons",
        "segmentation",
        "label",
        "labels",
        "confidence",
        "score",
        "scores",
        "object",
        "objects",
        "json",
        "detections",
        "x1",
        "y1",
        "x2",
        "y2",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
)

COLOR_TOKENS = frozenset(
    {
        "red",
        "rose",
        "blue",
        "sky",
        "cyan",
        "teal",
        "green",
        "emerald",
        "lime",
        "yellow",
        "amber",
        "orange",
        "purple",
        "violet",
        "magenta",
        "pink",
        "white",
        "black",
        "gray",
        "grey",
    }
)

GENERIC_PLACEHOLDER_NOUNS = frozenset(
    {
        "object",
        "objects",
        "target",
        "targets",
        "target class",
        "target classes",
        "class",
        "classes",
        "label",
        "labels",
        "instance",
        "instances",
        "item",
        "items",
        "thing",
        "things",
        "entity",
        "entities",
        "everything",
        "anything",
        "something",
        "nothing",
        "none",
        "etc",
        "coordinate",
        "coordinates",
        "coord",
        "coords",
        "box",
        "boxes",
        "bonding box",
        "bonding boxes",
        "bounding box",
        "bounding boxes",
        "2d box",
        "2d boxes",
        "2d bounding box",
        "2d bounding boxes",
        "surrounding box",
        "surrounding boxes",
        "bbox",
        "bboxes",
        "bbox_2d",
        "polygon",
        "polygons",
        "mask",
        "masks",
        "json",
        "json format",
        "format",
        "list",
        "image",
        "images",
        "photo",
        "picture",
        "frame",
        "scene",
        "annotation",
        "annotations",
        "value",
        "values",
        "key",
        "keys",
    }
)

_IRREGULAR_SINGULARS: Dict[str, str] = {
    "people": "person",
    "persons": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "machineries": "machinery",
}

_PLURAL_KEEP_AS_IS = frozenset(
    {
        "glass",
        "grass",
        "bus",
        "canvas",
        "status",
        "series",
        "species",
        "scissors",
        "clothes",
        "pants",
        "shorts",
        "binoculars",
        "lens",
    }
)

_EXCLUSION_SYNONYMS: Dict[str, frozenset[str]] = {
    "person": frozenset(
        {
            "person",
            "people",
            "man",
            "woman",
            "boy",
            "girl",
            "human",
            "pedestrian",
            "worker",
        }
    ),
    "luggage": frozenset({"luggage", "luggage bag", "suitcase", "baggage"}),
    "luggage bag": frozenset({"luggage", "luggage bag", "suitcase", "baggage"}),
    "suitcase": frozenset({"luggage", "luggage bag", "suitcase", "baggage"}),
    "helmet": frozenset({"helmet", "hard hat", "safety helmet"}),
    "safety helmet": frozenset({"helmet", "hard hat", "safety helmet"}),
    "hard hat": frozenset({"helmet", "hard hat", "safety helmet"}),
    "plastic table": frozenset({"plastic table", "table", "dining table"}),
    "table": frozenset({"plastic table", "table", "dining table"}),
    "dining table": frozenset({"plastic table", "table", "dining table"}),
}

_NEGATIVE_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"ignore|ignoring|exclude|excluding|without|skip|skipping|avoid|avoiding|"
    r"not\s+including|"
    r"(?:do\s+not|don't)\s+(?:annotate|detect|box|include|label|find|mark)"
    r")\b",
    flags=re.IGNORECASE,
)

_NEGATIVE_STOP_RE = re.compile(
    r"(?:[.;\n]"
    r"|(?:,\s*)?\b(?:but|however|while|whereas|instead|otherwise)\b"
    r"|(?:,\s*)?\band\s+only\s+(?:return|output|detect|keep|include|annotate)\b"
    r"|\bexcept(?:\s+for)?\b"
    r")",
    flags=re.IGNORECASE,
)

_EXCEPT_STANDALONE_RE = re.compile(
    r"\bexcept(?:\s+for)?\s+([^.;\n]+)",
    flags=re.IGNORECASE,
)

_LEADING_NOISE_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "all",
        "every",
        "each",
        "any",
        "other",
        "visible",
        "detected",
        "present",
        "individual",
        "distinct",
        "separate",
        "proper",
        "tight",
        "unattended",
        "unworn",
        "loose",
        "empty",
        "remaining",
        "only",
    }
)


@dataclass(frozen=True)
class ExtractedPromptSchema:
    targets: List[str]
    excludes: List[str]


@dataclass(frozen=True)
class ResolvedSchemaPlan:
    targets: List[str]
    excludes: List[str]
    source: str  # "explicit_classes" | "prompt_targets" | "unfiltered"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def canonicalize_term(text: str) -> str:
    cleaned = re.sub(r"[_]+", " ", normalize_whitespace(text)).lower()
    return normalize_whitespace(cleaned)


def singularize_word(word: str) -> str:
    w = word.lower().strip()
    if not w:
        return w
    if w in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[w]
    if w in _PLURAL_KEEP_AS_IS or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4 and w[-4] not in "aeiou":
        return f"{w[:-3]}y"
    if w.endswith(("ches", "shes", "xes", "zes", "sses")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        return w[:-1]
    return w


def normalize_label_singular(phrase: str) -> str:
    norm = canonicalize_term(phrase)
    if not norm:
        return ""
    tokens = norm.split()
    tokens[-1] = singularize_word(tokens[-1])
    return " ".join(tokens)


def is_placeholder_classes(classes: Optional[Sequence[str]]) -> bool:
    cleaned = [normalize_whitespace(str(c)) for c in (classes or ()) if normalize_whitespace(str(c))]
    if not cleaned:
        return True
    return len(cleaned) == 1 and cleaned[0].lower() == "object"


def dedupe_terms(terms: Sequence[str], *, singularize: bool = True) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in terms:
        val = normalize_label_singular(raw) if singularize else normalize_whitespace(raw)
        key = canonicalize_term(val)
        if not val or not key or key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _clean_noun_chunk(chunk: str, *, singularize: bool = True) -> Optional[str]:
    text = normalize_whitespace(chunk)
    text = re.sub(r"^[\"'`(\[]+|[\"'`),.!?:;\]]+$", "", text)
    text = re.sub(r"\s*(?:\(|\[|:)\s*#?[a-zA-Z0-9]{2,16}\s*[\)\]]?$", "", text)
    text = re.sub(r"\s+in\s+(?:" + "|".join(sorted(COLOR_TOKENS)) + r")\b$", "", text, flags=re.I)

    # Trim spatial / boilerplate suffixes inside chunks
    text = re.sub(
        r"\b(?:visible\s+)?(?:detected\s+)?(?:present\s+)?"
        r"(?:in|on|from|inside)\s+(?:this|that|these|those|the)\s+"
        r"(?:image|images|photo|picture|frame|scene)s?\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:wearing|holding|carrying)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:that\s+(?:is|are)|which\s+(?:is|are)|etc\.?)\b.*$", "", text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    if not text:
        return None

    # Remove leading filler modifiers
    words = text.split()
    while words and words[0].lower() in _LEADING_NOISE_WORDS:
        words.pop(0)
    while words and words[-1].lower() in {"etc", "detected", "visible", "present"}:
        words.pop()
    if not words:
        return None

    # Reject long prose sentences that aren't noun phrases
    if len(words) > 4:
        return None

    candidate = " ".join(words)
    cand_low = canonicalize_term(candidate)
    if (
        not cand_low
        or len(cand_low) < 2
        or cand_low in GENERIC_PLACEHOLDER_NOUNS
        or cand_low in SCHEMA_KEY_TOKENS
        or cand_low.isdigit()
    ):
        return None

    return normalize_label_singular(candidate) if singularize else normalize_whitespace(candidate)


def _split_list_chunks(text: str) -> List[str]:
    cleaned = text.replace("/", ",")
    cleaned = re.sub(r"\band\s*/\s*or\b|\band\b|\bor\b", ",", cleaned, flags=re.IGNORECASE)
    return [p.strip() for p in cleaned.split(",") if p.strip()]


def _extract_negative_spans(prompt: str) -> Tuple[List[str], str]:
    spans: List[Tuple[int, int]] = []
    neg_texts: List[str] = []
    pos = 0
    length = len(prompt)

    while pos < length:
        m = _NEGATIVE_TRIGGER_RE.search(prompt, pos)
        if not m:
            break
        start = m.start()
        body_start = m.end()
        stop_m = _NEGATIVE_STOP_RE.search(prompt, body_start)
        end = stop_m.start() if stop_m else length
        spans.append((start, end))
        body = prompt[body_start:end].strip()
        if body:
            neg_texts.append(body)
        pos = max(end, m.end())

    # Build positive text by replacing negative spans with punctuation boundary
    if not spans:
        positive = prompt
    else:
        pieces: List[str] = []
        cursor = 0
        for s, e in spans:
            if s > cursor:
                pieces.append(prompt[cursor:s])
            pieces.append(". ")
            cursor = e
        if cursor < length:
            pieces.append(prompt[cursor:])
        positive = "".join(pieces)

    # Also capture simple standalone "except dogs/cars" only if it names a short noun
    # rather than a contrastive instruction like "except detect people wearing helmets"
    for m in _EXCEPT_STANDALONE_RE.finditer(positive):
        after = m.group(1).strip()
        if re.match(r"^(?:detect|locate|find|annotate|segment|include|keep|people\s+wearing)\b", after, re.I):
            continue
        if len(after.split()) <= 6:
            neg_texts.append(after)
            positive = positive.replace(m.group(0), ". ")

    excludes: List[str] = []
    for raw_span in neg_texts:
        # Remove empty/JSON boilerplate that may trail an unpunctuated clause
        span_clean = re.sub(
            r"\b(?:only\s+)?return\b.*|\boutput\b.*|\bin\s+json\b.*",
            "",
            raw_span,
            flags=re.IGNORECASE,
        )
        for chunk in _split_list_chunks(span_clean):
            cleaned = _clean_noun_chunk(chunk, singularize=True)
            if cleaned:
                excludes.append(cleaned)

    return dedupe_terms(excludes, singularize=True), positive


def _strip_schema_boilerplate(text: str) -> str:
    cleaned = text
    # Drop parenthetical alternatives/examples like "(whether there is a single person or multiple people)"
    cleaned = re.sub(r"\([^)]{0,160}\)", " ", cleaned)
    # Remove empty-result rules ("If no person is present, return [].")
    cleaned = re.sub(
        r"\b(?:if|when)\s+(?:no|nothing)\b[^.;\n]*?\b(?:return|output)\b[^.;\n]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove grouping instructions ("never merge ...", "create proper separate ...")
    cleaned = re.sub(
        r"\b(?:never|do\s+not|don't)\s+merge\b[^.;\n]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bcreate\s+(?:proper\s+)?separate\b[^.;\n]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:it\s+might\s+have|there\s+(?:is|are|might\s+be))\b[^.;\n]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove JSON schema formatting boilerplate
    cleaned = re.sub(
        r"\b(?:return|output)\s+(?:only\s+)?(?:valid\s+)?(?:normalized\s+)?"
        r"(?:\[?\s*0\s*\.\.\s*1000\s*\]?\s*)?json\b[^.;\n]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:with\s+keys|as\s+a\s+list\s+of\s+objects|in\s+json\s+format|"
        r"normalized\s+\[?0\s*\.\.\s*1000\]?|bounding\s+box(?:es)?\s+coordinates|"
        r"bonding\s+box(?:es)?|surrounding\s+box(?:es)?|tight\s+2d\s+bounding\s+box(?:es)?)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[\s*\]|\[\s*0\s*\.\.\s*1000\s*\]", " ", cleaned)
    return normalize_whitespace(cleaned)


def _extract_explicit_schema_targets(positive_text: str) -> List[str]:
    targets: List[str] = []

    # 1) Explicit "classes: a, b" or "target classes: a, b"
    for m in re.finditer(
        r"\b(?:target\s+)?(?:classes|labels)\s*:\s*([^.;\n]+)",
        positive_text,
        flags=re.IGNORECASE,
    ):
        for chunk in _split_list_chunks(m.group(1)):
            cleaned = _clean_noun_chunk(chunk, singularize=True)
            if cleaned:
                targets.append(cleaned)

    # 2) Explicit JSON label schema value: "label": "person" or "label" ("person")
    for m in re.finditer(
        r"""["']?label["']?\s*(?::|=|is|\()\s*([^)\n,;]+)""",
        positive_text,
        flags=re.IGNORECASE,
    ):
        for qm in re.finditer(r"""["']([a-zA-Z0-9 _-]{2,40})["']""", m.group(1)):
            val = normalize_label_singular(qm.group(1))
            if val and val not in SCHEMA_KEY_TOKENS and val not in GENERIC_PLACEHOLDER_NOUNS:
                targets.append(val)

    # 3) UI Color hints: "person (blue)", "helmet: #4ADE80", "forklift in yellow"
    color_alt = "|".join(sorted(COLOR_TOKENS))
    for m in re.finditer(
        rf"\b([a-zA-Z][a-zA-Z0-9 _-]{{1,30}}?)\s*(?:\(|\[|:|-)\s*(?:#[0-9a-fA-F]{{3,6}}|{color_alt})\b\s*[\)\]]?",
        positive_text,
        flags=re.IGNORECASE,
    ):
        cand = re.sub(
            r"^(?:detect|locate|find|segment|annotate|identify|label|track)\s+",
            "",
            m.group(1).strip(),
            flags=re.IGNORECASE,
        )
        cleaned = _clean_noun_chunk(cand, singularize=True)
        if cleaned:
            targets.append(cleaned)

    for m in re.finditer(
        rf"\b([a-zA-Z][a-zA-Z0-9 _-]{{1,26}}?)\s+in\s+(?:{color_alt})\b",
        positive_text,
        flags=re.IGNORECASE,
    ):
        cand = re.sub(
            r"^(?:detect|locate|find|segment|annotate|identify|label|track)\s+",
            "",
            m.group(1).strip(),
            flags=re.IGNORECASE,
        )
        cleaned = _clean_noun_chunk(cand, singularize=True)
        if cleaned:
            targets.append(cleaned)

    if targets:
        return dedupe_terms(targets, singularize=True)

    # 4) Quoted class terms outside JSON key names
    for m in re.finditer(r"""["']([a-zA-Z0-9 _-]{2,40})["']""", positive_text):
        raw = normalize_whitespace(m.group(1))
        low = canonicalize_term(raw)
        if low in SCHEMA_KEY_TOKENS or low in GENERIC_PLACEHOLDER_NOUNS:
            continue
        cleaned = _clean_noun_chunk(raw, singularize=True)
        if cleaned:
            targets.append(cleaned)

    return dedupe_terms(targets, singularize=True)


def extract_prompt_schema(prompt: str) -> ExtractedPromptSchema:
    """
    Parse target classes and excluded terms dynamically from any free-text or
    structured detection prompt.
    """
    raw_prompt = (prompt or "").strip()
    if not raw_prompt:
        return ExtractedPromptSchema(targets=[], excludes=[])

    excludes, positive_raw = _extract_negative_spans(raw_prompt)
    explicit_targets = _extract_explicit_schema_targets(positive_raw)

    normalized_positive = re.sub(
        r"\b(?:(?:and\s+)?only\s+return|output|return)\s+(?:only\s+)?(?:the\s+)?"
        r"(?:bounding\s+box(?:es)?\s+coordinates\s+(?:of|for)|coordinates\s+(?:of|for)|"
        r"bounding\s+box(?:es)?\s+(?:of|for)|boxes\s+(?:of|for))\b",
        ". detect ",
        positive_raw,
        flags=re.IGNORECASE,
    )
    clean_positive = _strip_schema_boilerplate(normalized_positive)
    imperative_targets: List[str] = []

    if not explicit_targets and clean_positive:
        # Remove secondary qualifiers like ", including people wearing safety helmets"
        primary_positive = re.sub(
            r",?\s*\bincluding\b[^.;\n]*",
            " ",
            clean_positive,
            flags=re.IGNORECASE,
        )

        verb_re = re.compile(
            r"\b(?:detect|locate|find|annotate|segment|identify|label|track|mark)\s+([^.;\n]+)",
            flags=re.IGNORECASE,
        )
        matches = list(verb_re.finditer(primary_positive))
        if matches:
            for vm in matches:
                clause = vm.group(1)
                clause = re.sub(
                    r"\band\s+(?:output|return|draw|create|give)\b.*$",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                )
                for chunk in _split_list_chunks(clause):
                    cleaned = _clean_noun_chunk(chunk, singularize=True)
                    if cleaned:
                        imperative_targets.append(cleaned)
        else:
            # Fallback for direct noun-list prompts like "person, safety helmet"
            for sent in re.split(r"[.;\n]+", primary_positive):
                for chunk in _split_list_chunks(sent):
                    cleaned = _clean_noun_chunk(chunk, singularize=True)
                    if cleaned:
                        imperative_targets.append(cleaned)

    targets = explicit_targets if explicit_targets else dedupe_terms(imperative_targets, singularize=True)

    # Subtract excluded terms if targets came purely from natural-language fallback
    if targets and excludes and not explicit_targets:
        excl_keys = {canonicalize_term(e) for e in excludes}
        targets = [t for t in targets if canonicalize_term(t) not in excl_keys]

    return ExtractedPromptSchema(
        targets=dedupe_terms(targets, singularize=True),
        excludes=excludes,
    )


def label_matches_exclusion(label: str, excludes: Sequence[str]) -> bool:
    if not excludes:
        return False
    norm = normalize_label_singular(label)
    raw_low = canonicalize_term(label)
    if not norm and not raw_low:
        return False

    for ex in excludes:
        ex_norm = normalize_label_singular(ex)
        if not ex_norm:
            continue
        if norm == ex_norm or raw_low == ex_norm:
            return True
        syn_set = _EXCLUSION_SYNONYMS.get(ex_norm)
        if syn_set and (norm in syn_set or raw_low in syn_set):
            return True
    return False


def resolve_detection_schema(
    prompt: str,
    explicit_classes: Optional[Sequence[str]] = None,
) -> ResolvedSchemaPlan:
    """
    Resolve active target classes + excluded classes across explicit UI chips
    and free-text prompts cleanly:
      1. Non-placeholder explicit class chips keep exact user casing & order.
      2. Single placeholder ["object"] or empty classes dynamically fall back to
         targets parsed from the prompt (or unfiltered <OD> if none).
      3. Negative exclusions from prompt are preserved in all modes.
    """
    parsed = extract_prompt_schema(prompt)
    cleaned_explicit: List[str] = []
    seen = set()
    for c in explicit_classes or ():
        val = normalize_whitespace(str(c))
        key = canonicalize_term(val)
        if val and key not in seen:
            seen.add(key)
            cleaned_explicit.append(val)

    if cleaned_explicit and not is_placeholder_classes(cleaned_explicit):
        return ResolvedSchemaPlan(
            targets=cleaned_explicit,
            excludes=parsed.excludes,
            source="explicit_classes",
        )

    if parsed.targets:
        return ResolvedSchemaPlan(
            targets=parsed.targets,
            excludes=parsed.excludes,
            source="prompt_targets",
        )

    return ResolvedSchemaPlan(
        targets=[],
        excludes=parsed.excludes,
        source="unfiltered",
    )
