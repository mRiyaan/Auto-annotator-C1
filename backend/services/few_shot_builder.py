"""Zero-shot & few-shot multimodal prompt assembly and PIL image scaling."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

import config


def prepare_pil_image(path: Path | str, max_side: int = 0) -> Image.Image:
    """Load an image as RGB PIL Image, keeping original size when max_side <= 0."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        longest = max(w, h)
        if max_side > 0 and longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        return rgb.copy()


def _output_schema_instruction(output_mode: str) -> str:
    if output_mode == "polygon":
        return (
            'Return ONLY valid JSON as a list of objects with keys '
            '"label" (string), "confidence" (0..1), and '
            '"polygon" (list of [x, y] vertices normalized 0..1000). '
            "If nothing matches, return []."
        )
    if output_mode == "both":
        return (
            'Return ONLY valid JSON as a list of objects with keys '
            '"label" (string), "confidence" (0..1), '
            '"bbox_2d" ([xmin, ymin, xmax, ymax] normalized 0..1000), and '
            '"polygon" (list of [x, y] contour vertices normalized 0..1000). '
            "If nothing matches, return []."
        )
    return (
        'Return ONLY valid JSON as a list of objects with keys '
        '"bbox_2d" ([xmin, ymin, xmax, ymax] normalized 0..1000), '
        '"label" (string), and "confidence" (0..1). '
        "If nothing matches, return []."
    )


def build_image_prompt(
    user_prompt: str,
    *,
    output_mode: str,
    shot_mode: str,
    ref_notes: Sequence[str] = (),
    style_note: str = "",
    has_lora_adapter: bool = False,
) -> str:
    clean_prompt = user_prompt.strip()

    # Keep exact SFT instruction untouched for zero-shot person_lora bbox runs
    if (
        has_lora_adapter
        and shot_mode == "zero-shot"
        and output_mode == "bbox"
        and clean_prompt == config.PERSON_LORA_EXACT_PROMPT
    ):
        return clean_prompt

    schema_hint = _output_schema_instruction(output_mode)
    needs_schema = (
        "bbox_2d" not in clean_prompt
        and "polygon" not in clean_prompt
        and "json" not in clean_prompt.lower()
    )

    if shot_mode != "few-shot" or not ref_notes:
        return f"{clean_prompt}\n\n{schema_hint}" if needs_schema else clean_prompt

    ref_lines = []
    for idx, note in enumerate(ref_notes, start=1):
        suffix = f" — {note.strip()}" if note and note.strip() else ""
        ref_lines.append(
            f"- Image {idx}: ANNOTATED REFERENCE EXAMPLE showing expected annotation style{suffix}"
        )
    target_idx = len(ref_notes) + 1
    ref_block = "\n".join(ref_lines)
    style_block = f"\nAnnotation Style Guidance: {style_note.strip()}" if style_note.strip() else ""

    return (
        f"You are given {target_idx} images.\n"
        f"{ref_block}\n"
        f"- Image {target_idx}: TARGET IMAGE to annotate.\n"
        f"{style_block}\n\n"
        f"Instructions:\n"
        f"1. Learn the object classes, tightness, and box/polygon style from Reference Images 1..{len(ref_notes)}.\n"
        f"2. Annotate ONLY Image {target_idx} (the final TARGET IMAGE). NEVER copy coordinates from reference images.\n"
        f"3. Emit one separate annotation for each distinct target instance — do not merge separate nearby instances into one annotation unless the task explicitly asks for a group.\n"
        f"4. Task: {clean_prompt}\n"
        f"5. {schema_hint}"
    )


def build_interleaved_image_messages(
    user_prompt: str,
    *,
    output_mode: str,
    shot_mode: str,
    ref_notes: Sequence[str] = (),
    style_note: str = "",
    has_lora_adapter: bool = False,
) -> Any:
    """Build interleaved image/text chat messages for MLX few-shot inference."""
    if shot_mode != "few-shot" or not ref_notes:
        return build_image_prompt(
            user_prompt,
            output_mode=output_mode,
            shot_mode=shot_mode,
            ref_notes=ref_notes,
            style_note=style_note,
            has_lora_adapter=has_lora_adapter,
        )

    clean_prompt = user_prompt.strip()
    schema_hint = _output_schema_instruction(output_mode)
    parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"You are given {len(ref_notes)} annotated reference example(s) followed by 1 TARGET IMAGE.",
        }
    ]
    for idx, note in enumerate(ref_notes, start=1):
        suffix = f" — {note.strip()}" if note and note.strip() else ""
        parts.append(
            {
                "type": "text",
                "text": f"Reference Example {idx} (ANNOTATED STYLE GUIDE — do NOT annotate this image){suffix}:",
            }
        )
        parts.append({"type": "image"})

    parts.append(
        {
            "type": "text",
            "text": "TARGET IMAGE (annotate ONLY this image — NEVER copy coordinates from reference examples):",
        }
    )
    parts.append({"type": "image"})

    style_block = f"Annotation Style Guidance: {style_note.strip()}\n\n" if style_note.strip() else ""
    instructions = (
        f"{style_block}"
        f"Instructions:\n"
        f"1. Learn the object class, tightness, and boundary style from Reference Examples 1..{len(ref_notes)}.\n"
        f"2. Annotate ONLY the TARGET IMAGE above.\n"
        f"3. Emit one separate annotation for each distinct target instance — do not merge separate nearby instances into one annotation unless the task explicitly asks for a group.\n"
        f"4. Task: {clean_prompt}\n"
        f"5. {schema_hint}"
    )
    parts.append({"type": "text", "text": instructions})
    return [{"role": "user", "content": parts}]


def build_video_stage1_prompt(
    user_prompt: str,
    keyframe_info: List[Dict[str, Any]],
    *,
    ref_notes: Sequence[str] = (),
    style_note: str = "",
) -> str:
    ref_count = len(ref_notes)
    lines = []
    if ref_count > 0:
        for i, note in enumerate(ref_notes, start=1):
            suffix = f" ({note.strip()})" if note and note.strip() else ""
            lines.append(f"- Image {i}: ANNOTATED REFERENCE EXAMPLE{suffix}")
    for i, kf in enumerate(keyframe_info, start=1):
        lines.append(
            f"- Image {ref_count + i}: TARGET VIDEO timestamp {kf['timestamp_sec']}s (frame {kf['frame_idx']})"
        )
    ts_block = "\n".join(lines)
    style_block = f"\nReference Style Note: {style_note.strip()}\n" if style_note.strip() else ""

    return (
        f"You are analyzing {len(keyframe_info)} chronological frames sampled from a target video"
        + (f" preceded by {ref_count} annotated reference image(s)" if ref_count else "")
        + f":\n{ts_block}\n{style_block}\n"
        f'User Prompt: "{user_prompt.strip()}"\n\n'
        "Return ONLY valid JSON matching:\n"
        "{\n"
        '  "event_detected": boolean,\n'
        '  "overall_event_type": string,\n'
        '  "overall_summary": string,\n'
        '  "events": [\n'
        "    {\n"
        '      "event_id": integer,\n'
        '      "event_type": string,\n'
        '      "event_summary": string,\n'
        '      "trigger_subject": string,\n'
        '      "short_label": string,\n'
        '      "event_start_image_number": integer (1..N relative to target frames),\n'
        '      "event_start_timestamp_sec": number,\n'
        '      "event_end_image_number": integer (1..N relative to target frames),\n'
        '      "event_end_timestamp_sec": number\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Emit one entry in `events` for EACH distinct incident or subject. "
        "Keep `trigger_subject` pose-neutral (clothing/headwear/screen area). "
        "If no matching event occurs, set `event_detected` to false and `events` to []."
    )


def build_video_stage2_prompt(
    user_prompt: str,
    *,
    output_mode: str,
    active_events: List[Dict[str, Any]],
    ref_notes: Sequence[str] = (),
    style_note: str = "",
) -> str:
    targets_desc = "\n".join(
        f'- event_id={ev["event_id"]}: "{ev["trigger_subject"]}" (action: {ev["short_label"]})'
        for ev in active_events
    )
    ref_count = len(ref_notes)
    prefix = ""
    if ref_count > 0:
        prefix = (
            f"Images 1..{ref_count} are annotated reference examples showing expected tightness. "
            f"Image {ref_count + 1} is the TARGET video frame to annotate.\n"
        )
        if style_note.strip():
            prefix += f"Style note: {style_note.strip()}\n"

    extra_poly = (
        ' and optional "polygon" ([[x, y], ...])'
        if output_mode in ("polygon", "both")
        else ""
    )

    return (
        f"{prefix}"
        f"Context prompt: {user_prompt.strip()}\n"
        f"Locate ONLY the specific triggering target(s) listed below on the target video frame:\n"
        f"{targets_desc}\n\n"
        "Output ONLY valid JSON matching:\n"
        '{"detections": [{"event_id": integer, "bbox_2d": [xmin, ymin, xmax, ymax], "label": string, "confidence": number'
        f"{extra_poly}"
        "}]}\n"
        "Coordinates must be integers normalized 0..1000. Exclude uninvolved background bystanders."
    )
