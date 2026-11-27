"""Apple Silicon MPS/CPU inference engine for native Transformers Florence-2 models."""
from __future__ import annotations

import gc
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil
from PIL import Image

import config
from engines.mlx_lora_engine import mlx_engine
from services.schema_extractor import (
    canonicalize_term,
    label_matches_exclusion,
    resolve_detection_schema,
)

_FLORENCE_LOCK = threading.Lock()

# Canonical person labels emitted by FLD-5B <OD>
_PERSON_ALIASES = frozenset(
    {
        "person",
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "human",
        "pedestrian",
    }
)

_GENERIC_PERSON_TARGETS = frozenset(
    {
        "person",
        "people",
        "human",
        "humans",
        "pedestrian",
        "pedestrians",
    }
)

# Common <OD> (COCO / Objects365 / OpenImages) nouns reliably predicted by <OD>
_OD_VOCABULARY = frozenset(
    {
        *_PERSON_ALIASES,
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "bear",
        "elephant",
        "zebra",
        "giraffe",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "train",
        "boat",
        "airplane",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "orange",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
    }
)

_EXPLICIT_TASK_TOKENS = (
    "<OPEN_VOCABULARY_DETECTION>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REGION_TO_SEGMENTATION>",
    "<DENSE_REGION_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<DETAILED_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR_WITH_REGION>",
    "<CAPTION>",
    "<OCR>",
    "<OD>",
)

_TASK_WITHOUT_INPUT = frozenset(
    {
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<OCR>",
        "<OCR_WITH_REGION>",
    }
)

def _extract_target_classes(
    prompt: str, explicit_classes: Optional[Sequence[str]] = None
) -> List[str]:
    """Resolve active detection target classes using the shared schema extractor."""
    return resolve_detection_schema(prompt, explicit_classes).targets


def _clamp_int(val: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(val))))


def _pixel_box_to_1000(
    box_xyxy: Sequence[float], width: int, height: int
) -> Optional[List[int]]:
    if len(box_xyxy) != 4 or width <= 0 or height <= 0:
        return None
    px1, py1, px2, py2 = [float(v) for v in box_xyxy]
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1
    nx1 = _clamp_int((px1 / width) * 1000.0, 0, 1000)
    ny1 = _clamp_int((py1 / height) * 1000.0, 0, 1000)
    nx2 = _clamp_int((px2 / width) * 1000.0, 0, 1000)
    ny2 = _clamp_int((py2 / height) * 1000.0, 0, 1000)
    if (nx2 - nx1) < 2 or (ny2 - ny1) < 2:
        return None
    return [nx1, ny1, nx2, ny2]


def _pixel_poly_to_1000(
    contours: Sequence[Any], width: int, height: int
) -> Optional[List[List[int]]]:
    if not contours or width <= 0 or height <= 0:
        return None

    # `polygons` from Florence2PostProcessor can be [[x1, y1, ...], ...]
    best_flat: Optional[List[float]] = None
    if isinstance(contours[0], (int, float)):
        best_flat = [float(v) for v in contours]
    else:
        for candidate in contours:
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 6:
                if best_flat is None or len(candidate) > len(best_flat):
                    best_flat = [float(v) for v in candidate]

    if not best_flat or len(best_flat) < 6:
        return None

    pts: List[List[int]] = []
    for i in range(0, len(best_flat) - 1, 2):
        nx = _clamp_int((best_flat[i] / width) * 1000.0, 0, 1000)
        ny = _clamp_int((best_flat[i + 1] / height) * 1000.0, 0, 1000)
        if not pts or pts[-1] != [nx, ny]:
            pts.append([nx, ny])

    # Downsample overly dense vertices while keeping uniform perimeter resolution
    if len(pts) > 48:
        step = max(1, math.ceil(len(pts) / 48))
        pts = pts[::step]

    return pts if len(pts) >= 3 else None


def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
    area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    return inter / max(1e-6, area_a + area_b - inter)


def _deduplicate_detections(
    items: List[Dict[str, Any]], iou_threshold: float = 0.65
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    # Prefer canonical exact-label detections first, then larger clear boxes
    ordered = sorted(
        items,
        key=lambda d: (
            1 if d.pop("_exact_label", True) else 0,
            d.get("confidence", 0.9),
            (d["bbox_2d"][2] - d["bbox_2d"][0]) * (d["bbox_2d"][3] - d["bbox_2d"][1]),
        ),
        reverse=True,
    )
    for cand in ordered:
        c_label = str(cand["label"]).strip().lower()
        c_box = cand["bbox_2d"]
        duplicate = False
        for k in kept:
            if str(k["label"]).strip().lower() != c_label:
                continue
            if _box_iou(c_box, k["bbox_2d"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


class FlorenceEngine:
    """Manages native Transformers Florence-2 loading, caching, and MPS inference."""

    def __init__(self) -> None:
        self._loaded_id: Optional[str] = None
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None

    @property
    def loaded_model_id(self) -> Optional[str]:
        return self._loaded_id

    @staticmethod
    def is_hf_model_cached(model_path: str) -> bool:
        return mlx_engine.is_hf_model_cached(model_path)

    @staticmethod
    def _preferred_device_and_dtype() -> Tuple[str, Any]:
        import torch

        dtype_pref = os.getenv("FLORENCE_DTYPE", "float16").strip().lower()
        dtype = torch.float32 if dtype_pref == "float32" else torch.float16
        if torch.cuda.is_available():
            return "cuda", dtype
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", dtype
        return "cpu", torch.float32

    def unload(self) -> None:
        with _FLORENCE_LOCK:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        if self._model is not None or self._processor is not None:
            self._model = None
            self._processor = None
            self._loaded_id = None
            self._dtype = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    def _ensure_loaded_unlocked(self, model_id: str) -> None:
        if self._loaded_id == model_id and self._model is not None:
            return

        entry = config.get_model_entry(model_id)
        if entry["engine"] != "florence2":
            raise ValueError(f"Model '{model_id}' is not a Florence-2 engine model.")

        model_path = entry["model_path"]
        self._unload_unlocked()
        mlx_engine.unload()

        from transformers import Florence2ForConditionalGeneration, Florence2Processor

        device, dtype = self._preferred_device_and_dtype()
        processor = Florence2Processor.from_pretrained(model_path)
        model = Florence2ForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True
        )
        model = model.to(device=device, dtype=dtype)
        model.eval()

        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype
        self._loaded_id = model_id

    def _run_task_unlocked(
        self,
        *,
        image: Image.Image,
        task_token: str,
        prompt_text: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> Tuple[Dict[str, Any], int, int]:
        import torch

        inputs = self._processor(text=prompt_text, images=image, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        def _move(target_dev: str, target_dtype: Any) -> Dict[str, Any]:
            moved: Dict[str, Any] = {}
            for k, v in inputs.items():
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    moved[k] = v.to(device=target_dev, dtype=target_dtype)
                else:
                    moved[k] = v.to(device=target_dev)
            return moved

        try:
            model_inputs = _move(self._device, self._dtype)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )
        except Exception:
            if self._device not in ("cuda", "mps"):
                raise
            # Graceful fallback to CPU float32 if GPU backend encounters an unsupported op
            self._device = "cpu"
            self._dtype = torch.float32
            self._model = self._model.to(device="cpu", dtype=torch.float32)
            model_inputs = _move("cpu", torch.float32)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )

        gen_tokens = int(generated_ids.shape[-1])
        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            decoded, task=task_token, image_size=image.size
        )
        return parsed.get(task_token, {}), prompt_tokens, gen_tokens

    def _segment_regions_unlocked(
        self,
        *,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        max_regions: Optional[int] = None,
        num_beams: int = 1,
    ) -> Tuple[int, int]:
        total_p = 0
        total_g = 0
        w, h = image.size
        limit = (
            max_regions
            if max_regions is not None
            else max(1, int(os.getenv("FLORENCE_MAX_REGIONS", "48")))
        )
        prioritized = sorted(
            detections,
            key=lambda d: (
                float(d.get("confidence", 0.9)),
                (d["bbox_2d"][2] - d["bbox_2d"][0]) * (d["bbox_2d"][3] - d["bbox_2d"][1]),
            ),
            reverse=True,
        )
        for det in prioritized[:limit]:
            nx1, ny1, nx2, ny2 = det["bbox_2d"]
            q1 = _clamp_int((nx1 / 1000.0) * 999.0, 0, 999)
            q2 = _clamp_int((ny1 / 1000.0) * 999.0, 0, 999)
            q3 = _clamp_int((nx2 / 1000.0) * 999.0, 0, 999)
            q4 = _clamp_int((ny2 / 1000.0) * 999.0, 0, 999)
            task = "<REGION_TO_SEGMENTATION>"
            region_prompt = f"{task}<loc_{q1}><loc_{q2}><loc_{q3}><loc_{q4}>"
            parsed, pt, gt = self._run_task_unlocked(
                image=image,
                task_token=task,
                prompt_text=region_prompt,
                max_new_tokens=512,
                num_beams=num_beams,
            )
            total_p += pt
            total_g += gt
            polys = parsed.get("polygons") or []
            if polys:
                norm_poly = _pixel_poly_to_1000(polys[0], w, h)
                if norm_poly:
                    det["polygon"] = norm_poly
        return total_p, total_g

    def _current_memory_gb(self) -> float:
        try:
            import torch

            if self._device == "cuda" and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()
                if alloc > 0:
                    return round(alloc / (1024**3), 3)
            elif self._device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                alloc = torch.mps.current_allocated_memory()
                if alloc > 0:
                    return round(alloc / (1024**3), 3)
        except Exception:
            pass
        try:
            rss = psutil.Process().memory_info().rss
            return round(rss / (1024**3), 3)
        except Exception:
            return 0.0

    def predict_detections(
        self,
        *,
        model_id: str,
        image: Image.Image,
        prompt: str = "",
        classes: Optional[Sequence[str]] = None,
        output_mode: str = "bbox",
        max_tokens: int = 512,
        generate_polygons: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """
        Run Florence-2 on a single RGB image and return normalized [0..1000] detections
        plus optional pure-text output (for caption/OCR tasks) and telemetry.
        """
        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        w, h = rgb_image.size
        num_beams = max(1, int(os.getenv("FLORENCE_NUM_BEAMS", "1")))
        max_new_tokens = max(128, min(int(max_tokens or 512), 1024))

        with _FLORENCE_LOCK:
            t_load0 = time.perf_counter()
            was_loaded = self._loaded_id == model_id and self._model is not None
            self._ensure_loaded_unlocked(model_id)
            load_s = 0.0 if was_loaded else round(time.perf_counter() - t_load0, 3)

            t0 = time.perf_counter()
            total_p_tok = 0
            total_g_tok = 0
            raw_text_answer: Optional[str] = None
            collected: List[Dict[str, Any]] = []

            stripped = (prompt or "").strip()
            explicit_task = next(
                (t for t in _EXPLICIT_TASK_TOKENS if stripped.startswith(t)), None
            )
            schema_info: Dict[str, Any] = {
                "source": "explicit_task",
                "targets": [],
                "excludes": [],
            }

            if explicit_task is not None:
                task_prompt = (
                    explicit_task
                    if explicit_task in _TASK_WITHOUT_INPUT
                    else stripped
                )
                parsed, pt, gt = self._run_task_unlocked(
                    image=rgb_image,
                    task_token=explicit_task,
                    prompt_text=task_prompt,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                total_p_tok += pt
                total_g_tok += gt

                if isinstance(parsed, str):
                    raw_text_answer = parsed
                elif isinstance(parsed, dict):
                    bboxes = parsed.get("bboxes") or []
                    labels = (
                        parsed.get("labels")
                        or parsed.get("bboxes_labels")
                        or ["object"] * len(bboxes)
                    )
                    for idx, box in enumerate(bboxes):
                        norm_box = _pixel_box_to_1000(box, w, h)
                        if not norm_box:
                            continue
                        lbl = str(labels[idx] if idx < len(labels) else "object").strip() or "object"
                        collected.append(
                            {
                                "label": lbl,
                                "confidence": 0.9,
                                "bbox_2d": norm_box,
                                "_exact_label": True,
                            }
                        )
                    polys = parsed.get("polygons") or []
                    poly_labels = (
                        parsed.get("polygons_labels")
                        or parsed.get("labels")
                        or ["object"] * len(polys)
                    )
                    for idx, poly in enumerate(polys):
                        norm_poly = _pixel_poly_to_1000(poly, w, h)
                        if not norm_poly:
                            continue
                        xs = [p[0] for p in norm_poly]
                        ys = [p[1] for p in norm_poly]
                        box = [min(xs), min(ys), max(xs), max(ys)]
                        lbl = (
                            str(poly_labels[idx] if idx < len(poly_labels) else "object").strip()
                            or "object"
                        )
                        collected.append(
                            {
                                "label": lbl,
                                "confidence": 0.9,
                                "bbox_2d": box,
                                "polygon": norm_poly,
                                "_exact_label": True,
                            }
                        )
            else:
                schema_plan = resolve_detection_schema(prompt, classes)
                targets = schema_plan.targets
                excludes = schema_plan.excludes
                schema_info = {
                    "source": schema_plan.source,
                    "targets": list(targets),
                    "excludes": list(excludes),
                }

                # Preserve distinct person subtypes when requested explicitly (e.g. "worker", "man")
                person_specific = [
                    t
                    for t in targets
                    if canonicalize_term(t) in (_PERSON_ALIASES - _GENERIC_PERSON_TARGETS)
                ]
                od_targets = [
                    t for t in targets if canonicalize_term(t) in _OD_VOCABULARY
                ]
                open_targets = [
                    t for t in targets if canonicalize_term(t) not in _OD_VOCABULARY
                ]
                excluded_boxes: List[Tuple[str, List[int]]] = []

                # Run <OD> once when targets include generic COCO/OD classes or unfiltered
                if od_targets or not targets:
                    parsed, pt, gt = self._run_task_unlocked(
                        image=rgb_image,
                        task_token="<OD>",
                        prompt_text="<OD>",
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                    total_p_tok += pt
                    total_g_tok += gt

                    bboxes = parsed.get("bboxes") or [] if isinstance(parsed, dict) else []
                    labels = parsed.get("labels") or [] if isinstance(parsed, dict) else []
                    target_lower_map = {canonicalize_term(t): t for t in od_targets}
                    generic_person_targets = [
                        t
                        for t in od_targets
                        if canonicalize_term(t) in _GENERIC_PERSON_TARGETS
                    ]
                    allow_generic_person_alias = (
                        bool(generic_person_targets) and not person_specific
                    )
                    person_canonical = (
                        generic_person_targets[0] if generic_person_targets else "person"
                    )

                    for idx, box in enumerate(bboxes):
                        raw_lbl = (
                            str(labels[idx] if idx < len(labels) else "").strip()
                        )
                        low_lbl = canonicalize_term(raw_lbl)
                        if not low_lbl:
                            continue

                        if excludes and label_matches_exclusion(raw_lbl, excludes):
                            ex_box = _pixel_box_to_1000(box, w, h)
                            if ex_box:
                                excluded_boxes.append((low_lbl, ex_box))
                            continue

                        matched_label: Optional[str] = None
                        exact = True
                        if not targets:
                            matched_label = raw_lbl
                        elif low_lbl in target_lower_map:
                            matched_label = target_lower_map[low_lbl]
                        elif allow_generic_person_alias and low_lbl in _PERSON_ALIASES:
                            matched_label = person_canonical
                            exact = low_lbl == "person"

                        if matched_label is None:
                            continue

                        norm_box = _pixel_box_to_1000(box, w, h)
                        if norm_box:
                            collected.append(
                                {
                                    "label": matched_label,
                                    "confidence": 0.92 if exact else 0.86,
                                    "bbox_2d": norm_box,
                                    "_exact_label": exact,
                                }
                            )

                # Run <OPEN_VOCABULARY_DETECTION> for all custom / non-COCO schema classes
                max_open = max(1, int(os.getenv("FLORENCE_MAX_OPEN_CLASSES", "32")))
                for phrase in open_targets[:max_open]:
                    task = "<OPEN_VOCABULARY_DETECTION>"
                    parsed, pt, gt = self._run_task_unlocked(
                        image=rgb_image,
                        task_token=task,
                        prompt_text=f"{task}{phrase}",
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                    total_p_tok += pt
                    total_g_tok += gt
                    if not isinstance(parsed, dict):
                        continue
                    phrase_is_person = canonicalize_term(phrase) in _PERSON_ALIASES
                    for idx, box in enumerate(parsed.get("bboxes") or []):
                        norm_box = _pixel_box_to_1000(box, w, h)
                        if not norm_box:
                            continue
                        if excluded_boxes and not phrase_is_person:
                            if any(
                                _box_iou(norm_box, ex_box) >= 0.80
                                for _, ex_box in excluded_boxes
                            ):
                                continue
                        collected.append(
                            {
                                "label": phrase,
                                "confidence": 0.88,
                                "bbox_2d": norm_box,
                                "_exact_label": True,
                            }
                        )
                    for idx, poly in enumerate(parsed.get("polygons") or []):
                        norm_poly = _pixel_poly_to_1000(poly, w, h)
                        if not norm_poly:
                            continue
                        xs = [p[0] for p in norm_poly]
                        ys = [p[1] for p in norm_poly]
                        poly_box = [min(xs), min(ys), max(xs), max(ys)]
                        if excluded_boxes and not phrase_is_person:
                            if any(
                                _box_iou(poly_box, ex_box) >= 0.80
                                for _, ex_box in excluded_boxes
                            ):
                                continue
                        collected.append(
                            {
                                "label": phrase,
                                "confidence": 0.88,
                                "bbox_2d": poly_box,
                                "polygon": norm_poly,
                                "_exact_label": True,
                            }
                        )

            detections = _deduplicate_detections(collected)

            # Generate region polygons natively if requested without external SAM refiner
            if (
                generate_polygons
                and output_mode in ("polygon", "both")
                and detections
            ):
                missing_poly = [d for d in detections if not d.get("polygon")]
                if missing_poly:
                    pt_s, gt_s = self._segment_regions_unlocked(
                        image=rgb_image,
                        detections=missing_poly,
                        num_beams=num_beams,
                    )
                    total_p_tok += pt_s
                    total_g_tok += gt_s

            if output_mode == "bbox":
                for d in detections:
                    d.pop("polygon", None)

            latency_s = round(time.perf_counter() - t0, 3)
            tps = (
                round(total_g_tok / max(latency_s, 1e-3), 1)
                if total_g_tok > 0
                else 0.0
            )
            telemetry = {
                "load_s": load_s,
                "latency_s": latency_s,
                "prompt_tokens": total_p_tok,
                "generation_tokens": total_g_tok,
                "generation_tps": tps,
                "peak_memory_gb": self._current_memory_gb(),
                "device": self._device,
                "schema": schema_info,
            }
            return detections, raw_text_answer, telemetry

    def generate_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images: List[Any],
        output_mode: str = "bbox",
        max_tokens: int = 512,
    ) -> Tuple[str, Dict[str, Any]]:
        """Compatibility wrapper matching VLM engines for /analyze image execution."""
        import json

        if not images:
            raise ValueError("Florence-2 requires a target image.")
        if len(images) > 1 and "event_start_image_number" in (prompt_text or ""):
            raise RuntimeError(
                "Florence-2 is a single-image detection/segmentation model. "
                "Use Qwen3-VL MLX/Ollama for multi-frame temporal video event tracking, "
                "or run Florence-2 on Image mode / Auto-Annotate."
            )

        target_img = images[-1]
        if not isinstance(target_img, Image.Image):
            target_img = Image.open(target_img)

        detections, pure_text, tel = self.predict_detections(
            model_id=model_id,
            image=target_img,
            prompt=prompt_text,
            output_mode=output_mode,
            max_tokens=max_tokens,
            generate_polygons=(output_mode in ("polygon", "both")),
        )

        if pure_text is not None and not detections:
            return pure_text, tel

        payload: List[Dict[str, Any]] = []
        for det in detections:
            entry: Dict[str, Any] = {
                "label": det["label"],
                "confidence": det["confidence"],
                "bbox_2d": det["bbox_2d"],
            }
            if output_mode in ("polygon", "both") and det.get("polygon"):
                entry["polygon"] = det["polygon"]
            payload.append(entry)
        return json.dumps(payload, indent=2), tel


florence_engine = FlorenceEngine()
