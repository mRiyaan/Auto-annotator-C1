"""Robust parsing and normalization for Qwen3-VL 2D bounding boxes, polygons, and video events."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from schemas import DetectionItem

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FENCE_START_RE = re.compile(r"^```(?:json)?\s*", flags=re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```$")
_FALLBACK_BOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def strip_thinking_and_fences(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = _THINK_RE.sub("", raw_text.strip()).strip()
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    cleaned = _FENCE_START_RE.sub("", cleaned)
    cleaned = _FENCE_END_RE.sub("", cleaned).strip()
    return cleaned


def _extract_json_value(raw_text: str) -> Optional[Any]:
    cleaned = strip_thinking_and_fences(raw_text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first balanced JSON array or object
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return None


def _clamp_1000(val: float) -> int:
    return int(round(min(max(float(val), 0.0), 1000.0)))


def _normalize_bbox(raw_box: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [_clamp_1000(v) for v in raw_box]
    except (TypeError, ValueError):
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (x1, y1, x2, y2)


def _normalize_polygon(raw_poly: Any) -> Optional[List[Tuple[int, int]]]:
    if not isinstance(raw_poly, (list, tuple)) or not raw_poly:
        return None
    points: List[Tuple[int, int]] = []
    if all(isinstance(v, (int, float)) for v in raw_poly) and len(raw_poly) >= 6:
        for i in range(0, len(raw_poly) - 1, 2):
            points.append((_clamp_1000(raw_poly[i]), _clamp_1000(raw_poly[i + 1])))
    else:
        for pt in raw_poly:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    points.append((_clamp_1000(pt[0]), _clamp_1000(pt[1])))
                except (TypeError, ValueError):
                    continue
    return points if len(points) >= 3 else None


def parse_detections(
    raw_text: str,
    *,
    output_mode: str = "both",
    id_prefix: str = "det",
    event_id: Optional[int] = None,
) -> Tuple[List[DetectionItem], bool]:
    """
    Parse Qwen3-VL output into normalized 0..1000 DetectionItem objects.
    Supports list of dicts, {"detections": [...]}, single dict, or regex fallback.
    """
    parsed = _extract_json_value(raw_text)
    items_raw: List[Any] = []
    json_valid = parsed is not None

    if isinstance(parsed, list):
        items_raw = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("detections"), list):
            items_raw = parsed["detections"]
        elif isinstance(parsed.get("objects"), list):
            items_raw = parsed["objects"]
        elif "bbox_2d" in parsed or "polygon" in parsed or "bbox" in parsed:
            items_raw = [parsed]

    detections: List[DetectionItem] = []
    for idx, item in enumerate(items_raw, start=1):
        if not isinstance(item, dict):
            continue
        if event_id is not None and item.get("event_id") is not None:
            try:
                if int(item["event_id"]) != int(event_id):
                    continue
            except (TypeError, ValueError):
                pass

        label = str(item.get("label") or "object").strip() or "object"
        try:
            conf = float(item.get("confidence", 0.95) or 0.95)
        except (TypeError, ValueError):
            conf = 0.95
        conf = min(max(conf, 0.05), 1.0)

        polygon = _normalize_polygon(item.get("polygon") or item.get("segmentation"))
        raw_box = item.get("bbox_2d") or item.get("bbox") or item.get("box")

        # Handle nested list of boxes: {"bbox_2d": [[...], [...]]}
        if (
            isinstance(raw_box, list)
            and raw_box
            and all(isinstance(sub, (list, tuple)) and len(sub) == 4 for sub in raw_box)
        ):
            for sub_idx, sub_box in enumerate(raw_box, start=1):
                bbox = _normalize_bbox(sub_box)
                if bbox:
                    detections.append(
                        DetectionItem(
                            id=f"{id_prefix}-{idx}-{sub_idx}",
                            label=label,
                            confidence=round(conf, 3),
                            bbox_2d=bbox,
                            polygon=None,
                        )
                    )
            continue

        bbox = _normalize_bbox(raw_box)
        if not bbox and polygon:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])

        if not bbox:
            continue

        if output_mode == "bbox":
            polygon = None

        detections.append(
            DetectionItem(
                id=f"{id_prefix}-{len(detections) + 1}",
                label=label,
                confidence=round(conf, 3),
                bbox_2d=bbox,
                polygon=polygon,
            )
        )

    if not detections and not json_valid:
        cleaned = strip_thinking_and_fences(raw_text)
        for match in _FALLBACK_BOX_RE.finditer(cleaned):
            bbox = _normalize_bbox([float(match.group(i)) for i in range(1, 5)])
            if bbox:
                detections.append(
                    DetectionItem(
                        id=f"{id_prefix}-{len(detections) + 1}",
                        label="target",
                        confidence=0.85,
                        bbox_2d=bbox,
                        polygon=None,
                    )
                )

    return detections, json_valid


def norm_1000_to_pixel_xyxy(
    bbox_2d: Tuple[int, int, int, int], width: int, height: int
) -> List[int]:
    x1, y1, x2, y2 = bbox_2d
    px1 = min(max(int(round((x1 / 1000.0) * width)), 0), width - 1)
    py1 = min(max(int(round((y1 / 1000.0) * height)), 0), height - 1)
    px2 = min(max(int(round((x2 / 1000.0) * width)), 0), width - 1)
    py2 = min(max(int(round((y2 / 1000.0) * height)), 0), height - 1)
    if px2 <= px1:
        px2 = min(width - 1, px1 + 1)
    if py2 <= py1:
        py2 = min(height - 1, py1 + 1)
    return [px1, py1, px2, py2]


def parse_stage1_events(
    raw_text: str, keyframe_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Parse Stage 1 video temporal multi-event JSON output."""
    parsed = _extract_json_value(raw_text)
    n = len(keyframe_info)
    if not isinstance(parsed, dict):
        return {
            "event_detected": False,
            "overall_event_type": "None",
            "overall_summary": strip_thinking_and_fences(raw_text)[:400]
            or "No structured event output returned.",
            "events": [],
        }

    events_out: List[Dict[str, Any]] = []
    raw_events = parsed.get("events") if isinstance(parsed.get("events"), list) else []

    for idx, ev in enumerate(raw_events, start=1):
        if not isinstance(ev, dict):
            continue
        try:
            s_img = int(ev.get("event_start_image_number") or 1)
        except (TypeError, ValueError):
            s_img = 1
        s_img = min(max(s_img, 1), max(1, n))

        try:
            e_img = int(ev.get("event_end_image_number") or s_img)
        except (TypeError, ValueError):
            e_img = s_img
        e_img = min(max(e_img, s_img), max(1, n))

        s_ts = float(
            keyframe_info[s_img - 1]["timestamp_sec"]
            if keyframe_info
            else ev.get("event_start_timestamp_sec") or 0.0
        )
        e_ts = float(
            keyframe_info[e_img - 1]["timestamp_sec"]
            if keyframe_info
            else ev.get("event_end_timestamp_sec") or s_ts
        )
        if e_ts < s_ts:
            e_ts = s_ts

        events_out.append(
            {
                "event_id": int(ev.get("event_id") or idx),
                "event_type": str(ev.get("event_type") or "Event").strip(),
                "event_summary": str(ev.get("event_summary") or "").strip(),
                "trigger_subject": str(ev.get("trigger_subject") or "target").strip(),
                "short_label": str(ev.get("short_label") or f"event {idx}")[:24].strip(),
                "event_start_image_number": s_img,
                "event_start_timestamp_sec": round(s_ts, 2),
                "event_end_image_number": e_img,
                "event_end_timestamp_sec": round(e_ts, 2),
            }
        )

    event_detected = bool(parsed.get("event_detected")) and len(events_out) > 0
    return {
        "event_detected": event_detected,
        "overall_event_type": str(parsed.get("overall_event_type") or "Event"),
        "overall_summary": str(parsed.get("overall_summary") or "").strip()
        or (events_out[0]["event_summary"] if events_out else "No matching event detected."),
        "events": events_out,
    }
