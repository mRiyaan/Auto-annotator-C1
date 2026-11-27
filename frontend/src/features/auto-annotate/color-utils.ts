import type { ClassColorEntry } from "./types";

export const DEFAULT_PALETTE = [
  "#38BDF8", // Sky blue
  "#4ADE80", // Emerald green
  "#F2853F", // Control One orange
  "#E879F9", // Magenta
  "#FACC15", // Amber yellow
  "#FB7185", // Rose coral
  "#A78BFA", // Violet
  "#2DD4BF", // Teal
  "#60A5FA", // Cobalt blue
  "#F97316", // Orange
];

const NAMED_COLORS: Record<string, string> = {
  red: "#F43F5E",
  rose: "#FB7185",
  blue: "#38BDF8",
  sky: "#38BDF8",
  cyan: "#22D3EE",
  teal: "#2DD4BF",
  green: "#4ADE80",
  emerald: "#34D399",
  lime: "#A3E635",
  yellow: "#FACC15",
  amber: "#FBBF24",
  orange: "#F2853F",
  purple: "#A78BFA",
  violet: "#A78BFA",
  magenta: "#E879F9",
  pink: "#F472B6",
  white: "#F4F4F5",
};

const SCHEMA_KEY_WORDS = new Set([
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
]);

const GENERIC_PLACEHOLDERS = new Set([
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
  "everything",
  "anything",
  "nothing",
  "none",
  "etc",
  "coordinate",
  "coordinates",
  "box",
  "boxes",
  "bounding box",
  "bounding boxes",
  "bonding box",
  "bonding boxes",
  "2d box",
  "2d boxes",
  "2d bounding box",
  "2d bounding boxes",
  "surrounding box",
  "surrounding boxes",
  "bbox",
  "bbox_2d",
  "polygon",
  "polygons",
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
]);

const LEADING_FILLER = new Set([
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
]);

const IRREGULAR_SINGULAR: Record<string, string> = {
  people: "person",
  persons: "person",
  men: "man",
  women: "woman",
  children: "child",
  machineries: "machinery",
};

const KEEP_PLURAL = new Set([
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
]);

function singularizePhrase(phrase: string): string {
  const words = phrase.trim().replace(/\s+/g, " ").split(" ");
  if (!words.length || !words[0]) return "";
  const last = words[words.length - 1].toLowerCase();
  if (IRREGULAR_SINGULAR[last]) {
    words[words.length - 1] = IRREGULAR_SINGULAR[last];
  } else if (!KEEP_PLURAL.has(last) && last.length > 3) {
    if (last.endsWith("ies") && last.length > 4 && !/[aeiou]/.test(last[last.length - 4])) {
      words[words.length - 1] = `${last.slice(0, -3)}y`;
    } else if (/(ches|shes|xes|zes|sses)$/.test(last) && last.length > 4) {
      words[words.length - 1] = last.slice(0, -2);
    } else if (last.endsWith("s") && !/(ss|us|is|as)$/.test(last)) {
      words[words.length - 1] = last.slice(0, -1);
    }
  }
  return words.join(" ");
}

function cleanNounChunk(chunk: string): string | null {
  let text = chunk
    .trim()
    .replace(/\s+/g, " ")
    .replace(/^["'`(\[]+|["'`)\],.!?:;\]]+$/g, "")
    .replace(/\s*(?:\(|\[|:)\s*#?[a-zA-Z0-9]{2,16}\s*[\)\]]?$/g, "")
    .replace(
      /\s+in\s+(?:red|rose|blue|sky|cyan|teal|green|emerald|lime|yellow|amber|orange|purple|violet|magenta|pink|white|black|gray|grey)\b$/i,
      ""
    )
    .replace(
      /\b(?:visible\s+)?(?:detected\s+)?(?:present\s+)?(?:in|on|from|inside)\s+(?:this|that|these|those|the)\s+(?:image|images|photo|picture|frame|scene)s?\b.*$/i,
      ""
    )
    .replace(/\b(?:wearing|holding|carrying)\b.*$/i, "")
    .replace(/\b(?:that\s+(?:is|are)|which\s+(?:is|are)|etc\.?)\b.*$/i, "")
    .trim();

  if (!text) return null;
  const words = text.split(/\s+/);
  while (words.length && LEADING_FILLER.has(words[0].toLowerCase())) {
    words.shift();
  }
  while (
    words.length &&
    ["etc", "detected", "visible", "present"].includes(
      words[words.length - 1].toLowerCase()
    )
  ) {
    words.pop();
  }
  if (!words.length || words.length > 4) return null;

  const candidate = words.join(" ");
  const low = candidate.toLowerCase().replace(/_/g, " ");
  if (
    low.length < 2 ||
    GENERIC_PLACEHOLDERS.has(low) ||
    SCHEMA_KEY_WORDS.has(low) ||
    /^\d+$/.test(low)
  ) {
    return null;
  }
  return singularizePhrase(candidate);
}

function splitChunks(clause: string): string[] {
  return clause
    .replace(/\//g, ",")
    .replace(/\band\s*\/\s*or\b|\band\b|\bor\b/gi, ",")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function stripNegativeSpans(prompt: string): {
  positive: string;
  excludes: Set<string>;
} {
  const triggerRe =
    /\b(?:ignore|ignoring|exclude|excluding|without|skip|skipping|avoid|avoiding|not\s+including|(?:do\s+not|don't)\s+(?:annotate|detect|box|include|label|find|mark))\b/gi;
  const stopRe =
    /(?:[.;\n]|(?:,\s*)?\b(?:but|however|while|whereas|instead|otherwise)\b|(?:,\s*)?\band\s+only\s+(?:return|output|detect|keep|include|annotate)\b|\bexcept(?:\s+for)?\b)/i;

  const excludes = new Set<string>();
  const pieces: string[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = triggerRe.exec(prompt)) !== null) {
    if (match.index > cursor) {
      pieces.push(prompt.slice(cursor, match.index));
    }
    const bodyStart = match.index + match[0].length;
    const rest = prompt.slice(bodyStart);
    const stopHit = stopRe.exec(rest);
    const bodyEnd = stopHit ? bodyStart + stopHit.index : prompt.length;
    const spanBody = prompt.slice(bodyStart, bodyEnd).replace(
      /\b(?:only\s+)?return\b.*|\boutput\b.*|\bin\s+json\b.*/i,
      ""
    );
    for (const chunk of splitChunks(spanBody)) {
      const cleaned = cleanNounChunk(chunk);
      if (cleaned) excludes.add(cleaned.toLowerCase());
    }
    pieces.push(". ");
    cursor = bodyEnd;
    triggerRe.lastIndex = bodyEnd;
  }
  pieces.push(prompt.slice(cursor));
  return { positive: pieces.join(""), excludes };
}

export function isPlaceholderClassList(entries: ClassColorEntry[]): boolean {
  const cleaned = entries.map((e) => e.name.trim()).filter(Boolean);
  return cleaned.length === 0 || (cleaned.length === 1 && cleaned[0].toLowerCase() === "object");
}

export function extractPromptTargetClasses(prompt: string): string[] {
  const raw = (prompt || "").trim();
  if (!raw) return [];
  const { positive, excludes } = stripNegativeSpans(raw);

  const explicit: string[] = [];
  let m: RegExpExecArray | null;

  // Explicit "classes: a, b" or "target classes: ..."
  const classesPrefixRe = /\b(?:target\s+)?(?:classes|labels)\s*:\s*([^.;\n]+)/gi;
  while ((m = classesPrefixRe.exec(positive)) !== null) {
    for (const chunk of splitChunks(m[1])) {
      const cleaned = cleanNounChunk(chunk);
      if (cleaned) explicit.push(cleaned);
    }
  }

  // Explicit JSON schema `"label" ("person")` / `"label": "person"`
  const jsonLabelRe = /["']?label["']?\s*(?::|=|is|\()\s*([^)\n,;]+)/gi;
  while ((m = jsonLabelRe.exec(positive)) !== null) {
    const qRe = /["']([a-zA-Z0-9 _-]{2,40})["']/g;
    let qm: RegExpExecArray | null;
    while ((qm = qRe.exec(m[1])) !== null) {
      const val = singularizePhrase(qm[1]);
      if (
        val &&
        !SCHEMA_KEY_WORDS.has(val.toLowerCase()) &&
        !GENERIC_PLACEHOLDERS.has(val.toLowerCase())
      ) {
        explicit.push(val);
      }
    }
  }

  if (explicit.length > 0) {
    return Array.from(new Set(explicit));
  }

  const normalizedPos = positive
    .replace(
      /\b(?:(?:and\s+)?only\s+return|output|return)\s+(?:only\s+)?(?:the\s+)?(?:bounding\s+box(?:es)?\s+coordinates\s+(?:of|for)|coordinates\s+(?:of|for)|bounding\s+box(?:es)?\s+(?:of|for)|boxes\s+(?:of|for))\b/gi,
      ". detect "
    )
    .replace(/\([^)]{0,160}\)/g, " ")
    .replace(/\b(?:if|when)\s+(?:no|nothing)\b[^.;\n]*?\b(?:return|output)\b[^.;\n]*/gi, " ")
    .replace(/\b(?:never|do\s+not|don't)\s+merge\b[^.;\n]*/gi, " ")
    .replace(/\bcreate\s+(?:proper\s+)?separate\b[^.;\n]*/gi, " ")
    .replace(/\b(?:it\s+might\s+have|there\s+(?:is|are|might\s+be))\b[^.;\n]*/gi, " ")
    .replace(/\b(?:return|output)\s+(?:only\s+)?(?:valid\s+)?(?:normalized\s+)?(?:\[?\s*0\s*\.\.\s*1000\s*\]?\s*)?json\b[^.;\n]*/gi, " ")
    .replace(/\b(?:with\s+keys|as\s+a\s+list\s+of\s+objects|in\s+json\s+format|normalized\s+\[?0\s*\.\.\s*1000\]?|bounding\s+box(?:es)?\s+coordinates|bonding\s+box(?:es)?|surrounding\s+box(?:es)?|tight\s+2d\s+bounding\s+box(?:es)?)\b/gi, " ")
    .replace(/,?\s*\bincluding\b[^.;\n]*/gi, " ");

  const extracted: string[] = [];
  const verbRe =
    /\b(?:detect|locate|find|annotate|segment|identify|label|track|mark)\s+([^.;\n]+)/gi;
  const verbMatches = Array.from(normalizedPos.matchAll(verbRe));
  if (verbMatches.length > 0) {
    for (const vm of verbMatches) {
      const clause = vm[1].replace(/\band\s+(?:output|return|draw|create|give)\b.*$/i, "");
      for (const chunk of splitChunks(clause)) {
        const cleaned = cleanNounChunk(chunk);
        if (cleaned && !excludes.has(cleaned.toLowerCase())) {
          extracted.push(cleaned);
        }
      }
    }
  } else {
    for (const sent of normalizedPos.split(/[.;\n]+/)) {
      for (const chunk of splitChunks(sent)) {
        const cleaned = cleanNounChunk(chunk);
        if (cleaned && !excludes.has(cleaned.toLowerCase())) {
          extracted.push(cleaned);
        }
      }
    }
  }

  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const t of extracted) {
    const k = t.toLowerCase();
    if (!seen.has(k)) {
      seen.add(k);
      deduped.push(t);
    }
  }
  return deduped;
}

function resolveColorToken(token: string): string | null {
  const clean = token.trim().toLowerCase();
  if (/^#[0-9a-f]{6}$/i.test(clean)) return clean.toUpperCase();
  if (/^#[0-9a-f]{3}$/i.test(clean)) {
    const r = clean[1];
    const g = clean[2];
    const b = clean[3];
    return `#${r}${r}${g}${g}${b}${b}`.toUpperCase();
  }
  return NAMED_COLORS[clean] ?? null;
}

/**
 * Parse class names and optional color hints from both structured color prompts
 * ("person (blue), helmet: #4ADE80") and natural-language detection prompts
 * ("Detect every person ... Ignore dogs, bags.").
 */
export function syncClassColorsFromPrompt(
  prompt: string,
  currentEntries: ClassColorEntry[]
): ClassColorEntry[] {
  const onlyPlaceholderBefore = isPlaceholderClassList(currentEntries);
  const firstPlaceholderColor =
    currentEntries[0]?.color ?? DEFAULT_PALETTE[0];

  const mapByLower = new Map<string, ClassColorEntry>();
  const explicitKeys = new Set<string>();
  for (const entry of currentEntries) {
    const name = entry.name.trim();
    if (name) {
      mapByLower.set(name.toLowerCase(), { name, color: entry.color });
    }
  }

  // Pattern 1: "className (color)" or "className [color]" or "className: color"
  const parenRegex =
    /\b([a-zA-Z][a-zA-Z0-9 _-]{1,30}?)\s*(?:\(|\[|:|-)\s*(#[0-9a-fA-F]{3,6}|red|rose|blue|sky|cyan|teal|green|emerald|lime|yellow|amber|orange|purple|violet|magenta|pink|white)\b\s*[\)\]]?/gi;
  let m: RegExpExecArray | null;
  while ((m = parenRegex.exec(prompt)) !== null) {
    const rawClass = m[1]
      .trim()
      .replace(/^(and|or|the|a|an|for|detect|segment|find|label|locate|annotate)\s+/i, "")
      .trim();
    const resolved = resolveColorToken(m[2]);
    if (!rawClass || !resolved || rawClass.length < 2) continue;
    const key = rawClass.toLowerCase();
    explicitKeys.add(key);
    const existing = mapByLower.get(key);
    mapByLower.set(key, {
      name: existing ? existing.name : rawClass,
      color: resolved,
    });
  }

  // Pattern 2: "className in <color>"
  const inColorRegex =
    /\b([a-zA-Z][a-zA-Z0-9 _-]{1,26}?)\s+in\s+(red|rose|blue|sky|cyan|teal|green|emerald|lime|yellow|amber|orange|purple|violet|magenta|pink|white)\b/gi;
  while ((m = inColorRegex.exec(prompt)) !== null) {
    const rawClass = m[1]
      .trim()
      .replace(/^(and|or|the|a|an|for|detect|segment|find|label|locate|annotate)\s+/i, "")
      .trim();
    const resolved = resolveColorToken(m[2]);
    if (!rawClass || !resolved || rawClass.length < 2) continue;
    const key = rawClass.toLowerCase();
    explicitKeys.add(key);
    const existing = mapByLower.get(key);
    mapByLower.set(key, {
      name: existing ? existing.name : rawClass,
      color: resolved,
    });
  }

  // Pattern 3: Dynamically extract non-color target nouns from plain prompt text
  const promptTargets = extractPromptTargetClasses(prompt);
  for (const target of promptTargets) {
    const key = target.toLowerCase();
    if (!mapByLower.has(key)) {
      const idx = mapByLower.size;
      mapByLower.set(key, {
        name: target,
        color: DEFAULT_PALETTE[idx % DEFAULT_PALETTE.length],
      });
    }
  }

  // If the only pre-existing chip was the untouched `"object"` placeholder and real
  // classes were discovered from the prompt, replace `"object"` cleanly.
  if (onlyPlaceholderBefore && mapByLower.size > 1 && mapByLower.has("object")) {
    mapByLower.delete("object");
    const entries = Array.from(mapByLower.values());
    if (entries.length > 0 && !explicitKeys.has(entries[0].name.toLowerCase())) {
      entries[0] = { ...entries[0], color: firstPlaceholderColor };
    }
    return entries;
  }

  // Fallback: check if an unmapped existing class appears near a named color word
  const colorListPattern = Object.keys(NAMED_COLORS).join("|");
  for (const [key, entry] of mapByLower.entries()) {
    if (explicitKeys.has(key)) continue;
    const escaped = entry.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const nearRegex = new RegExp(
      `(?:\\b(${colorListPattern})\\b[^.\\n,;()\\[\\]]{0,16}\\b${escaped}\\b|\\b${escaped}\\b[^.\\n,;()\\[\\]]{0,16}\\b(${colorListPattern})\\b)`,
      "i"
    );
    const hit = nearRegex.exec(prompt);
    if (hit) {
      const tok = hit[1] || hit[2];
      const hex = tok ? resolveColorToken(tok) : null;
      if (hex) {
        mapByLower.set(key, { name: entry.name, color: hex });
      }
    }
  }

  return Array.from(mapByLower.values());
}

export function buildDefaultClassColors(
  classes: string[],
  existing: ClassColorEntry[] = []
): ClassColorEntry[] {
  const existingMap = new Map(
    existing.map((e) => [e.name.trim().toLowerCase(), e.color])
  );
  const out: ClassColorEntry[] = [];
  const seen = new Set<string>();
  classes.forEach((raw, idx) => {
    const name = raw.trim();
    if (!name) return;
    const key = name.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push({
      name,
      color:
        existingMap.get(key) ?? DEFAULT_PALETTE[idx % DEFAULT_PALETTE.length],
    });
  });
  return out;
}

export function getClassColor(
  label: string,
  classColors: ClassColorEntry[]
): string {
  const key = (label || "object").trim().toLowerCase();
  const match = classColors.find((c) => c.name.trim().toLowerCase() === key);
  if (match) return match.color;
  let hash = 0;
  for (let i = 0; i < key.length; i += 1) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  return DEFAULT_PALETTE[hash % DEFAULT_PALETTE.length];
}

export function hexToRgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return `rgba(56, 189, 248, ${alpha})`;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
