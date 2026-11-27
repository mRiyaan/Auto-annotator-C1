"""PyTorch MPS/CPU inference engine for dataset-agnostic YOLOE-26s-seg & SAM 2.1 / MobileSAM."""
from __future__ import annotations

import gc
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from auto_annotate import config as aa_config
from auto_annotate.schemas import AutoDetection
from engines.mlx_lora_engine import mlx_engine

_PT_LOCK = threading.Lock()


class PyTorchVisionEngine:
    """Manages YOLOE open-vocabulary detection/seg and SAM box-to-polygon refinement."""

    def __init__(self) -> None:
        self._yoloe_model: Any = None
        self._yoloe_path: Optional[str] = None
        self._sam_models: Dict[str, Any] = {}
        self._sam_predictors: Dict[str, Any] = {}
        self._sam_predictor_devices: Dict[str, str] = {}
        self._sam_encoded_key: Dict[str, Tuple[str, int, int]] = {}
        self._owl_processor: Any = None
        self._owl_model: Any = None
        self._owl_device: str = "cpu"
        self._owl_model_id: str = aa_config.DEFAULT_OWLV2_MODEL_ID
        self._owl_box_cache: Dict[
            Tuple[str, str, Tuple[str, ...], float], List[Dict[str, Any]]
        ] = {}
        self._owl_feat_cache: Dict[str, Dict[str, Any]] = {}
        self._owl_visual_queries: Dict[str, Dict[str, Any]] = {}
        self._last_top_objectness: Optional[float] = None
        self._settings_configured = False

    @staticmethod
    def is_hf_model_cached(model_path: str) -> bool:
        return mlx_engine.is_hf_model_cached(model_path)

    @staticmethod
    def _owl_short_tag(owl_model_id: Optional[str], *, embed: bool = False) -> str:
        resolved = aa_config.resolve_owlv2_model_id(owl_model_id)
        if "large-patch14" in resolved:
            return "owlv2-large-p14-embed" if embed else "owlv2-large-p14"
        return "owlv2-embed" if embed else "owlv2"

    def _configure_ultralytics(self) -> None:
        if self._settings_configured:
            return
        from ultralytics import settings

        settings.update({"weights_dir": str(aa_config.WEIGHTS_DIR)})
        self._settings_configured = True

    @staticmethod
    def _preferred_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _empty_device_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def unload_pytorch(self) -> None:
        with _PT_LOCK:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        changed = (
            self._yoloe_model is not None
            or bool(self._sam_models)
            or bool(self._sam_predictors)
            or self._owl_model is not None
        )
        self._yoloe_model = None
        self._yoloe_path = None
        self._sam_models.clear()
        self._sam_predictors.clear()
        self._sam_predictor_devices.clear()
        self._sam_encoded_key.clear()
        self._owl_processor = None
        self._owl_model = None
        self._owl_device = "cpu"
        self._owl_box_cache.clear()
        self._owl_feat_cache.clear()
        self._owl_visual_queries.clear()
        self._last_top_objectness = None
        if changed:
            gc.collect()
            self._empty_device_cache()

    def _get_yoloe(self, weight_path: Path) -> Any:
        if not weight_path.is_file():
            raise FileNotFoundError(f"YOLOE weight file missing: {weight_path}")
        self._configure_ultralytics()
        if self._yoloe_model is not None and self._yoloe_path == str(weight_path):
            return self._yoloe_model
        from ultralytics import YOLOE

        self._yoloe_model = YOLOE(str(weight_path))
        self._yoloe_path = str(weight_path)
        return self._yoloe_model

    def _get_sam(self, refiner_id: str) -> Any:
        entry = aa_config.SAM_REFINERS.get(refiner_id)
        if not entry or not entry.get("weight_path"):
            raise ValueError(f"Invalid SAM refiner: {refiner_id}")
        weight_path = Path(entry["weight_path"])
        if not weight_path.is_file():
            raise FileNotFoundError(f"SAM weight file missing: {weight_path}")
        self._configure_ultralytics()
        if refiner_id in self._sam_models:
            return self._sam_models[refiner_id]
        from ultralytics import SAM

        model = SAM(str(weight_path))
        self._sam_models[refiner_id] = model
        return model

    def _build_sam_predictor_unlocked(self, refiner_id: str, device: str) -> Any:
        entry = aa_config.SAM_REFINERS.get(refiner_id)
        if not entry or not entry.get("weight_path"):
            raise ValueError(f"Invalid SAM refiner: {refiner_id}")
        weight_path = Path(entry["weight_path"])
        if not weight_path.is_file():
            raise FileNotFoundError(f"SAM weight file missing: {weight_path}")
        self._configure_ultralytics()
        from ultralytics.models.sam import Predictor as SAMPredictor, SAM2Predictor

        pred_cls = SAM2Predictor if "sam2" in refiner_id else SAMPredictor
        predictor = pred_cls(
            overrides=dict(
                conf=0.25,
                task="segment",
                mode="predict",
                imgsz=1024,
                model=str(weight_path),
                device=device,
                verbose=False,
                save=False,
            )
        )
        self._sam_predictors[refiner_id] = predictor
        self._sam_predictor_devices[refiner_id] = device
        return predictor

    def _ensure_sam_image_encoded_unlocked(
        self, refiner_id: str, image_path: Path
    ) -> Tuple[Any, int, int, bool, str]:
        cached_key = self._sam_encoded_key.get(refiner_id)
        if (
            refiner_id in self._sam_predictors
            and cached_key is not None
            and cached_key[0] == str(image_path)
        ):
            device = self._sam_predictor_devices.get(refiner_id, "cpu")
            return (
                self._sam_predictors[refiner_id],
                cached_key[1],
                cached_key[2],
                True,
                device,
            )

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None or img_bgr.size == 0:
            with Image.open(image_path) as im_raw:
                img_rgb = np.array(im_raw.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        device = self._sam_predictor_devices.get(refiner_id) or self._preferred_device()
        predictor = self._sam_predictors.get(refiner_id)
        if predictor is None:
            try:
                predictor = self._build_sam_predictor_unlocked(refiner_id, device)
            except Exception:
                device = "cpu"
                predictor = self._build_sam_predictor_unlocked(refiner_id, "cpu")

        try:
            predictor.reset_image()
        except Exception:
            pass

        try:
            predictor.set_image(img_bgr)
        except Exception:
            if device != "cpu":
                device = "cpu"
                predictor = self._build_sam_predictor_unlocked(refiner_id, "cpu")
                predictor.set_image(img_bgr)
            else:
                raise

        self._sam_encoded_key[refiner_id] = (str(image_path), int(w), int(h))
        return predictor, int(w), int(h), False, device

    @staticmethod
    def _simplify_norm_polygon(
        pts_norm_01: np.ndarray,
    ) -> Optional[List[List[int]]]:
        if pts_norm_01 is None or len(pts_norm_01) < 3:
            return None
        pts_1000 = np.clip(np.round(pts_norm_01 * 1000.0), 0, 1000).astype(np.float32)
        if len(pts_1000) > 36:
            contour = pts_1000.reshape(-1, 1, 2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, max(1.5, 0.004 * peri), True)
            if len(approx) >= 3:
                pts_1000 = approx.reshape(-1, 2)
        cleaned = [
            [int(min(max(round(x), 0), 1000)), int(min(max(round(y), 0), 1000))]
            for x, y in pts_1000.tolist()
        ]
        return cleaned if len(cleaned) >= 3 else None

    @staticmethod
    def _mask_to_norm_polygon(
        mask_2d: np.ndarray, complexity: float = 0.5
    ) -> Optional[List[List[int]]]:
        if mask_2d is None or mask_2d.size == 0:
            return None
        mh, mw = mask_2d.shape[:2]
        binary = (mask_2d > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 4.0:
            return None
        peri = cv2.arcLength(largest, True)
        c = max(0.0, min(float(complexity), 1.0))
        eps_ratio = 0.014 * ((0.0008 / 0.014) ** c)
        approx = cv2.approxPolyDP(largest, max(0.5, eps_ratio * peri), True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            return None
        norm_pts: List[List[int]] = []
        for px, py in pts.tolist():
            nx = int(min(max(round((px / float(max(1, mw))) * 1000.0), 0), 1000))
            ny = int(min(max(round((py / float(max(1, mh))) * 1000.0), 0), 1000))
            norm_pts.append([nx, ny])
        return norm_pts if len(norm_pts) >= 3 else None

    def predict_yoloe(
        self,
        *,
        image_path: Path,
        class_names: List[str],
        output_mode: str,
        conf_threshold: float = 0.25,
        id_prefix: str = "det",
    ) -> Tuple[List[AutoDetection], Dict[str, Any]]:
        clean_classes = [c.strip() for c in class_names if c.strip()]
        if not clean_classes:
            clean_classes = ["object"]

        entry = aa_config.PYTORCH_DETECTORS["yoloe-26s-seg"]
        weight_path = Path(entry["weight_path"])

        # Free Apple Silicon MLX Metal cache before loading PyTorch MPS weights
        mlx_engine.unload()

        with _PT_LOCK:
            model = self._get_yoloe(weight_path)
            t0 = time.perf_counter()

            # Configure open-vocabulary text prompts
            try:
                model.to("cpu")
            except Exception:
                pass
            pe = model.get_text_pe(clean_classes)
            model.set_classes(clean_classes, pe)

            device = self._preferred_device()
            try:
                results = model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    verbose=False,
                    device=device,
                )
            except Exception:
                device = "cpu"
                results = model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    verbose=False,
                    device="cpu",
                )

            latency_s = round(time.perf_counter() - t0, 3)

            detections: List[AutoDetection] = []
            if results:
                res = results[0]
                boxes = res.boxes
                masks = res.masks
                if boxes is not None and len(boxes) > 0:
                    xyxyn = boxes.xyxyn.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    xyn_masks = (
                        masks.xyn
                        if (masks is not None and output_mode in ("polygon", "both"))
                        else None
                    )
                    for idx in range(len(xyxyn)):
                        x1, y1, x2, y2 = [
                            int(min(max(round(v * 1000.0), 0), 1000))
                            for v in xyxyn[idx].tolist()
                        ]
                        if (x2 - x1) < 2 or (y2 - y1) < 2:
                            continue
                        cls_idx = int(clss[idx])
                        label = (
                            clean_classes[cls_idx]
                            if 0 <= cls_idx < len(clean_classes)
                            else str(cls_idx)
                        )
                        conf = round(float(confs[idx]), 3)
                        poly = None
                        if xyn_masks is not None and idx < len(xyn_masks):
                            poly = self._simplify_norm_polygon(xyn_masks[idx])

                        detections.append(
                            AutoDetection(
                                id=f"{id_prefix}-{len(detections) + 1}",
                                label=label,
                                confidence=conf,
                                bbox_2d=[x1, y1, x2, y2],
                                polygon=poly if output_mode in ("polygon", "both") else None,
                            )
                        )

            return detections, {"latency_s": latency_s, "device": device}

    def refine_boxes_with_sam(
        self,
        *,
        image_path: Path,
        detections: List[AutoDetection],
        sam_refiner_id: str,
    ) -> Tuple[List[AutoDetection], float]:
        if not detections or not sam_refiner_id or sam_refiner_id == "none":
            return detections, 0.0

        with Image.open(image_path) as im:
            w, h = im.size

        pixel_boxes: List[List[float]] = []
        valid_indices: List[int] = []
        for idx, det in enumerate(detections):
            if len(det.bbox_2d) != 4:
                continue
            x1, y1, x2, y2 = det.bbox_2d
            px1 = max(0.0, (x1 / 1000.0) * w)
            py1 = max(0.0, (y1 / 1000.0) * h)
            px2 = min(float(w), (x2 / 1000.0) * w)
            py2 = min(float(h), (y2 / 1000.0) * h)
            if px2 > px1 + 1.0 and py2 > py1 + 1.0:
                pixel_boxes.append([px1, py1, px2, py2])
                valid_indices.append(idx)

        if not pixel_boxes:
            return detections, 0.0

        mlx_engine.unload()

        with _PT_LOCK:
            sam_model = self._get_sam(sam_refiner_id)
            t0 = time.perf_counter()
            device = self._preferred_device()
            try:
                results = sam_model.predict(
                    source=str(image_path),
                    bboxes=pixel_boxes,
                    verbose=False,
                    device=device,
                )
            except Exception:
                results = sam_model.predict(
                    source=str(image_path),
                    bboxes=pixel_boxes,
                    verbose=False,
                    device="cpu",
                )
            latency_s = round(time.perf_counter() - t0, 3)

            if results and results[0].masks is not None and results[0].masks.data is not None:
                mask_tensor = results[0].masks.data.cpu().numpy()
                for m_idx, det_idx in enumerate(valid_indices):
                    if m_idx >= len(mask_tensor):
                        break
                    poly = self._mask_to_norm_polygon(mask_tensor[m_idx])
                    if poly:
                        detections[det_idx].polygon = poly

            return detections, latency_s

    def _select_owl_model_unlocked(self, owl_model_id: Optional[str]) -> str:
        target_id = (
            aa_config.resolve_owlv2_model_id(owl_model_id)
            if owl_model_id
            else aa_config.resolve_owlv2_model_id(self._owl_model_id)
        )
        if self._owl_model_id != target_id:
            if self._owl_model is not None or self._owl_processor is not None:
                self._owl_processor = None
                self._owl_model = None
                self._owl_device = "cpu"
                gc.collect()
                self._empty_device_cache()
            self._owl_model_id = target_id
        return target_id

    def _get_owlv2(self, model_id: Optional[str] = None) -> Tuple[Any, Any, str]:
        target_id = self._select_owl_model_unlocked(model_id)
        if self._owl_processor is not None and self._owl_model is not None:
            return self._owl_processor, self._owl_model, self._owl_device
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        try:
            processor = Owlv2Processor.from_pretrained(
                target_id, local_files_only=True
            )
            model = Owlv2ForObjectDetection.from_pretrained(
                target_id, torch_dtype=torch.float32, local_files_only=True
            )
        except Exception:
            processor = Owlv2Processor.from_pretrained(target_id)
            model = Owlv2ForObjectDetection.from_pretrained(
                target_id, torch_dtype=torch.float32
            )

        device = self._preferred_device()
        try:
            model = model.to(device).eval()
        except Exception:
            device = "cpu"
            model = model.to("cpu").eval()

        self._owl_processor = processor
        self._owl_model = model
        self._owl_device = device
        return processor, model, device

    @staticmethod
    def _mask_to_norm_bbox(mask_2d: np.ndarray) -> Optional[List[int]]:
        if mask_2d is None or mask_2d.size == 0:
            return None
        mh, mw = mask_2d.shape[:2]
        ys, xs = np.where(mask_2d > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            return None
        nx1 = int(min(max(round((float(xs.min()) / float(max(1, mw))) * 1000.0), 0), 999))
        ny1 = int(min(max(round((float(ys.min()) / float(max(1, mh))) * 1000.0), 0), 999))
        nx2 = int(min(max(round((float(xs.max() + 1) / float(max(1, mw))) * 1000.0), nx1 + 1), 1000))
        ny2 = int(min(max(round((float(ys.max() + 1) / float(max(1, mh))) * 1000.0), ny1 + 1), 1000))
        return [nx1, ny1, nx2, ny2]

    def _run_owlv2_boxes_unlocked(
        self,
        *,
        image_path: Path,
        class_names: List[str],
        conf_threshold: float,
        owl_model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        target_owl_id = self._select_owl_model_unlocked(owl_model_id)
        clean_classes = [c.strip() for c in class_names if c.strip()] or ["object"]
        cache_key = (
            target_owl_id,
            str(image_path),
            tuple(clean_classes),
            round(float(conf_threshold), 3),
        )
        if cache_key in self._owl_box_cache:
            return self._owl_box_cache[cache_key]

        import torch
        from torchvision.ops import nms

        processor, model, device = self._get_owlv2(target_owl_id)
        with Image.open(image_path) as im_raw:
            im = im_raw.convert("RGB")
            w, h = im.size

        inputs = processor(text=[clean_classes], images=im, return_tensors="pt")
        try:
            dev_inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**dev_inputs)
        except Exception:
            device = "cpu"
            model = model.to("cpu").eval()
            self._owl_model = model
            self._owl_device = "cpu"
            dev_inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**dev_inputs)

        post = processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(conf_threshold),
            target_sizes=[(h, w)],
            text_labels=[clean_classes],
        )[0]

        boxes_t = post["boxes"].detach().cpu()
        scores_t = post["scores"].detach().cpu()
        raw_labels = post.get("text_labels")
        if raw_labels is None:
            raw_labels = post.get("labels")
        labels_list = list(raw_labels) if raw_labels is not None else []

        candidates: List[Dict[str, Any]] = []
        if len(boxes_t) > 0:
            # Per-class NMS to clean overlapping proposals cleanly
            label_strs = [str(lbl) for lbl in labels_list]
            unique_lbls = list(dict.fromkeys(label_strs))
            keep_indices: List[int] = []
            for lbl in unique_lbls:
                idxs = [i for i, l in enumerate(label_strs) if l == lbl]
                if not idxs:
                    continue
                kept = nms(boxes_t[idxs], scores_t[idxs], iou_threshold=0.50).tolist()
                keep_indices.extend(idxs[k] for k in kept)

            for i in keep_indices:
                bx1, by1, bx2, by2 = [float(v) for v in boxes_t[i].tolist()]
                bx1 = max(0.0, min(bx1, float(w - 1)))
                by1 = max(0.0, min(by1, float(h - 1)))
                bx2 = max(bx1 + 1.0, min(bx2, float(w)))
                by2 = max(by1 + 1.0, min(by2, float(h)))
                nx1 = int(min(max(round((bx1 / float(max(1, w))) * 1000.0), 0), 999))
                ny1 = int(min(max(round((by1 / float(max(1, h))) * 1000.0), 0), 999))
                nx2 = int(min(max(round((bx2 / float(max(1, w))) * 1000.0), nx1 + 1), 1000))
                ny2 = int(min(max(round((by2 / float(max(1, h))) * 1000.0), ny1 + 1), 1000))
                candidates.append(
                    {
                        "bbox_px": [bx1, by1, bx2, by2],
                        "bbox_1000": [nx1, ny1, nx2, ny2],
                        "score": float(scores_t[i].item()),
                        "label": label_strs[i] if i < len(label_strs) else clean_classes[0],
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        if len(self._owl_box_cache) >= 6:
            self._owl_box_cache.pop(next(iter(self._owl_box_cache)))
        self._owl_box_cache[cache_key] = candidates
        return candidates

    def prepare_sam_image(
        self, *, image_path: Path, sam_refiner_id: str = "sam2.1-s"
    ) -> Dict[str, Any]:
        eff_sam_id = (
            sam_refiner_id
            if (sam_refiner_id in aa_config.SAM_REFINERS and sam_refiner_id != "none")
            else "sam2.1-s"
        )
        cached_key = self._sam_encoded_key.get(eff_sam_id)
        if cached_key is None or cached_key[0] != str(image_path):
            mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            _, _, _, cache_hit, device = self._ensure_sam_image_encoded_unlocked(
                eff_sam_id, image_path
            )
            latency_s = round(time.perf_counter() - t0, 3)
            return {
                "ready": True,
                "cached": cache_hit,
                "latency_s": latency_s,
                "engineUsed": eff_sam_id,
                "device": device,
            }

    def predict_sam_interactive(
        self,
        *,
        image_path: Path,
        points: List[List[float]],
        point_labels: List[int],
        bbox_prompt: Optional[List[float]] = None,
        sam_refiner_id: str = "sam2.1-s",
        label: str = "object",
        output_mode: str = "both",
        use_owlv2_assist: bool = False,
        candidate_classes: Optional[List[str]] = None,
        owl_model_id: Optional[str] = None,
        owl_threshold: float = 0.12,
        complexity: float = 0.5,
        preview: bool = False,
    ) -> Tuple[Optional[AutoDetection], Dict[str, Any]]:
        eff_sam_id = (
            sam_refiner_id
            if (sam_refiner_id in aa_config.SAM_REFINERS and sam_refiner_id != "none")
            else "sam2.1-s"
        )
        cached_key = self._sam_encoded_key.get(eff_sam_id)
        needs_encode = (
            eff_sam_id not in self._sam_predictors
            or cached_key is None
            or cached_key[0] != str(image_path)
        )
        if needs_encode:
            mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            predictor, w, h, cache_hit, device = self._ensure_sam_image_encoded_unlocked(
                eff_sam_id, image_path
            )

            pixel_points: List[List[float]] = []
            clean_labels: List[int] = []
            for idx, pt in enumerate(points):
                if len(pt) < 2:
                    continue
                px = max(0.0, min((float(pt[0]) / 1000.0) * w, float(w)))
                py = max(0.0, min((float(pt[1]) / 1000.0) * h, float(h)))
                pixel_points.append([px, py])
                lbl_val = 1 if (idx >= len(point_labels) or int(point_labels[idx]) != 0) else 0
                clean_labels.append(lbl_val)

            pixel_bbox: Optional[List[float]] = None
            if bbox_prompt and len(bbox_prompt) == 4:
                bx1 = max(0.0, (float(bbox_prompt[0]) / 1000.0) * w)
                by1 = max(0.0, (float(bbox_prompt[1]) / 1000.0) * h)
                bx2 = min(float(w), (float(bbox_prompt[2]) / 1000.0) * w)
                by2 = min(float(h), (float(bbox_prompt[3]) / 1000.0) * h)
                if bx2 > bx1 + 2.0 and by2 > by1 + 2.0:
                    pixel_bbox = [bx1, by1, bx2, by2]

            if not pixel_points and pixel_bbox is None:
                return None, {
                    "latency_s": 0.0,
                    "engineUsed": eff_sam_id,
                    "owlMatched": False,
                    "cached": cache_hit,
                }

            owl_matched = False
            resolved_label = (label or "object").strip() or "object"
            resolved_conf = 0.98

            if use_owlv2_assist and not preview and pixel_bbox is None:
                pos_idx = next(
                    (i for i, v in enumerate(clean_labels) if v == 1), None
                )
                if pos_idx is not None:
                    qx, qy = pixel_points[pos_idx]
                    owl_classes = [
                        c.strip() for c in (candidate_classes or []) if c.strip()
                    ] or [resolved_label]
                    owl_boxes = self._run_owlv2_boxes_unlocked(
                        image_path=image_path,
                        class_names=owl_classes,
                        conf_threshold=owl_threshold,
                        owl_model_id=owl_model_id,
                    )
                    containing = [
                        b
                        for b in owl_boxes
                        if (b["bbox_px"][0] - 12.0) <= qx <= (b["bbox_px"][2] + 12.0)
                        and (b["bbox_px"][1] - 12.0) <= qy <= (b["bbox_px"][3] + 12.0)
                    ]
                    if containing:
                        containing.sort(
                            key=lambda b: (
                                -b["score"],
                                (b["bbox_px"][2] - b["bbox_px"][0])
                                * (b["bbox_px"][3] - b["bbox_px"][1]),
                            )
                        )
                        chosen = containing[0]
                        pixel_bbox = list(chosen["bbox_px"])
                        resolved_conf = round(float(chosen["score"]), 3)
                        if len(owl_classes) > 1 or resolved_label.lower() == "object":
                            resolved_label = chosen["label"]
                        owl_matched = True

            pred_kwargs: Dict[str, Any] = {}
            if pixel_bbox is not None:
                pred_kwargs["bboxes"] = [pixel_bbox]
            if pixel_points:
                pred_kwargs["points"] = [pixel_points]
                pred_kwargs["labels"] = [clean_labels]

            try:
                results = predictor(**pred_kwargs)
            except Exception:
                if device != "cpu":
                    device = "cpu"
                    self._sam_encoded_key.pop(eff_sam_id, None)
                    predictor, _, _, _, device = self._ensure_sam_image_encoded_unlocked(
                        eff_sam_id, image_path
                    )
                    results = predictor(**pred_kwargs)
                else:
                    raise

            # SAM2Predictor box-only quirk fallback: add box center positive point if empty
            if (
                pixel_bbox is not None
                and not pixel_points
                and (
                    not results
                    or results[0].masks is None
                    or results[0].masks.data is None
                    or len(results[0].masks.data) == 0
                )
            ):
                cx = (pixel_bbox[0] + pixel_bbox[2]) * 0.5
                cy = (pixel_bbox[1] + pixel_bbox[3]) * 0.5
                results = predictor(
                    bboxes=[pixel_bbox],
                    points=[[[cx, cy]]],
                    labels=[[1]],
                )

            latency_s = round(time.perf_counter() - t0, 3)

            owl_tag = self._owl_short_tag(owl_model_id or self._owl_model_id, embed=False)
            if (
                not results
                or results[0].masks is None
                or results[0].masks.data is None
                or len(results[0].masks.data) == 0
            ):
                return None, {
                    "latency_s": latency_s,
                    "engineUsed": f"{owl_tag}+{eff_sam_id}" if owl_matched else eff_sam_id,
                    "owlMatched": owl_matched,
                    "cached": cache_hit,
                }

            mask_2d = results[0].masks.data[0].cpu().numpy()
            poly = self._mask_to_norm_polygon(mask_2d, complexity=complexity)
            bbox_1000 = self._mask_to_norm_bbox(mask_2d)
            if bbox_1000 is None and poly:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox_1000 = [min(xs), min(ys), max(xs), max(ys)]
            if bbox_1000 is None:
                return None, {
                    "latency_s": latency_s,
                    "engineUsed": eff_sam_id,
                    "owlMatched": owl_matched,
                    "cached": cache_hit,
                }

            det = AutoDetection(
                id=f"sam-{int(time.time() * 1000) % 10000000}",
                label=resolved_label,
                confidence=resolved_conf,
                bbox_2d=bbox_1000,
                polygon=poly if (preview or output_mode in ("polygon", "both")) else None,
            )
            return det, {
                "latency_s": latency_s,
                "engineUsed": f"{owl_tag}+{eff_sam_id}" if owl_matched else eff_sam_id,
                "owlMatched": owl_matched,
                "cached": cache_hit,
            }

    def predict_owlv2_find_objects(
        self,
        *,
        image_path: Path,
        class_names: List[str],
        conf_threshold: float = 0.20,
        output_mode: str = "bbox",
        sam_refiner_id: str = "sam2.1-s",
        owl_model_id: Optional[str] = None,
        id_prefix: str = "owl",
    ) -> Tuple[List[AutoDetection], Dict[str, Any]]:
        mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            target_owl_id = self._select_owl_model_unlocked(owl_model_id)
            boxes = self._run_owlv2_boxes_unlocked(
                image_path=image_path,
                class_names=class_names,
                conf_threshold=conf_threshold,
                owl_model_id=target_owl_id,
            )
            owl_latency = time.perf_counter() - t0
            device = self._owl_device

        detections = [
            AutoDetection(
                id=f"{id_prefix}-{int(time.time() * 1000) % 100000}-{idx + 1}",
                label=b["label"],
                confidence=round(float(b["score"]), 3),
                bbox_2d=list(b["bbox_1000"]),
                polygon=None,
            )
            for idx, b in enumerate(boxes)
        ]

        sam_latency = 0.0
        engine_used = self._owl_short_tag(target_owl_id, embed=False)
        if detections and output_mode in ("polygon", "both") and sam_refiner_id != "none":
            eff_sam = (
                sam_refiner_id
                if sam_refiner_id in aa_config.SAM_REFINERS
                else "sam2.1-s"
            )
            detections, sam_latency = self.refine_boxes_with_sam(
                image_path=image_path,
                detections=detections,
                sam_refiner_id=eff_sam,
            )
            for det in detections:
                if det.polygon and len(det.polygon) >= 3:
                    xs = [p[0] for p in det.polygon]
                    ys = [p[1] for p in det.polygon]
                    det.bbox_2d = [min(xs), min(ys), max(xs), max(ys)]
            engine_used = f"{engine_used}+{eff_sam}"

        return detections, {
            "latency_s": round(owl_latency + sam_latency, 3),
            "engineUsed": engine_used,
            "owlModelId": target_owl_id,
            "device": device,
        }

    @staticmethod
    def _cxcywh_to_xyxy(b: Any) -> Any:
        import torch

        cx, cy, bw, bh = b.unbind(-1)
        return torch.stack(
            [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], dim=-1
        )

    @staticmethod
    def _box_iou_1000(a: List[int], b: List[int]) -> float:
        if len(a) != 4 or len(b) != 4:
            return 0.0
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

    @staticmethod
    def _new_visual_query_session(owl_model_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "owlModelId": aa_config.resolve_owlv2_model_id(owl_model_id),
            "classes": {},
            "label": "object",
            "refItemIds": set(),
            "refSpecs": [],
            "drawnStack": [],
            "drawnRedoStack": [],
        }

    @staticmethod
    def _normalize_target_classes(
        target_classes: Optional[List[str]],
    ) -> Optional[Set[str]]:
        if target_classes is None:
            return None
        return {str(c).strip().lower() for c in target_classes if str(c).strip()}

    @classmethod
    def _filter_class_buckets(
        cls,
        class_buckets: Dict[str, Dict[str, Any]],
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        allowed = cls._normalize_target_classes(target_classes)
        if allowed is None:
            return class_buckets
        if not allowed:
            return {}
        return {
            ck: bucket
            for ck, bucket in class_buckets.items()
            if (str(bucket.get("label") or ck).strip().lower() in allowed)
            or (ck.strip().lower() in allowed)
        }

    @classmethod
    def _session_counts_unlocked(
        cls,
        session: Dict[str, Any],
        target_classes: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        classes_map = cls._filter_class_buckets(
            session.get("classes") or {}, target_classes
        )
        pos_cnt = sum(len(b.get("positives") or []) for b in classes_map.values())
        neg_cnt = sum(len(b.get("negatives") or []) for b in classes_map.values())
        return pos_cnt, neg_cnt

    @classmethod
    def _session_class_summaries_unlocked(
        cls,
        session: Dict[str, Any],
        target_classes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        classes_map = cls._filter_class_buckets(
            session.get("classes") or {}, target_classes
        )
        out: List[Dict[str, Any]] = []
        for b in sorted(
            classes_map.values(), key=lambda entry: str(entry.get("label", "")).lower()
        ):
            p_cnt = len(b.get("positives") or [])
            n_cnt = len(b.get("negatives") or [])
            if p_cnt == 0 and n_cnt == 0:
                continue
            out.append(
                {
                    "label": str(b.get("label") or "object"),
                    "exemplarCount": p_cnt,
                    "negativeCount": n_cnt,
                    "imageCount": len(b.get("imagePaths") or ()),
                }
            )
        return out

    def reset_visual_query(self, query_id: Optional[str] = None) -> None:
        with _PT_LOCK:
            if query_id:
                self._owl_visual_queries.pop(query_id, None)
            else:
                self._owl_visual_queries.clear()

    def has_visual_query(
        self, query_id: str, target_classes: Optional[List[str]] = None
    ) -> bool:
        with _PT_LOCK:
            session = self._owl_visual_queries.get(query_id)
            if not session:
                return False
            classes_map = self._filter_class_buckets(
                session.get("classes") or {}, target_classes
            )
            return any(bool(b.get("positives")) for b in classes_map.values())

    def get_visual_query_classes(
        self, query_id: str, target_classes: Optional[List[str]] = None
    ) -> List[str]:
        with _PT_LOCK:
            session = self._owl_visual_queries.get((query_id or "").strip())
            if not session:
                return []
            classes_map = self._filter_class_buckets(
                session.get("classes") or {}, target_classes
            )
            return [
                str(b.get("label") or "object")
                for b in classes_map.values()
                if b.get("positives")
            ]

    def _extract_owlv2_features_unlocked(
        self, image_path: Path
    ) -> Tuple[Dict[str, Any], bool]:
        target_id = aa_config.resolve_owlv2_model_id(self._owl_model_id)
        cache_key = f"{target_id}::{image_path}"
        if cache_key in self._owl_feat_cache:
            return self._owl_feat_cache[cache_key], True

        import torch
        import torch.nn.functional as F

        processor, model, device = self._get_owlv2(target_id)
        with Image.open(image_path) as im_raw:
            im = im_raw.convert("RGB")
            w, h = im.size

        pixel_values = processor(images=im, return_tensors="pt").pixel_values
        try:
            dev_pv = pixel_values.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                feature_map = model.image_embedder(pixel_values=dev_pv)[0]
                b, fh, fw, d = feature_map.shape
                feats = feature_map.reshape(b, fh * fw, d)
                _, class_embeds = model.class_predictor(feats)
                boxes = model.box_predictor(feats, feature_map)
                objectness = model.objectness_predictor(feats).sigmoid()
        except Exception:
            device = "cpu"
            model = model.to("cpu").eval()
            self._owl_model = model
            self._owl_device = "cpu"
            dev_pv = pixel_values.to(device="cpu", dtype=torch.float32)
            with torch.no_grad():
                feature_map = model.image_embedder(pixel_values=dev_pv)[0]
                b, fh, fw, d = feature_map.shape
                feats = feature_map.reshape(b, fh * fw, d)
                _, class_embeds = model.class_predictor(feats)
                boxes = model.box_predictor(feats, feature_map)
                objectness = model.objectness_predictor(feats).sigmoid()

        norm_embeds = F.normalize(class_embeds.float(), dim=-1)[0].detach().cpu()
        pred_boxes = self._cxcywh_to_xyxy(boxes[0].float()).detach().cpu()
        obj_scores = objectness[0].float().detach().cpu()

        pack: Dict[str, Any] = {
            "class_embeds": norm_embeds,
            "pred_boxes": pred_boxes,
            "objectness": obj_scores,
            "width": int(w),
            "height": int(h),
        }
        if len(self._owl_feat_cache) >= 64:
            self._owl_feat_cache.pop(next(iter(self._owl_feat_cache)))
        self._owl_feat_cache[cache_key] = pack
        return pack, False

    def _embed_region_from_features(
        self, feat_pack: Dict[str, Any], box_1000: List[float]
    ) -> Tuple[np.ndarray, float, List[int]]:
        import torch
        from torchvision.ops import box_iou

        w = int(feat_pack["width"])
        h = int(feat_pack["height"])
        scale = float(max(1, w, h))

        x1_raw = float(min(box_1000[0], box_1000[2]))
        y1_raw = float(min(box_1000[1], box_1000[3]))
        x2_raw = float(max(box_1000[0], box_1000[2]))
        y2_raw = float(max(box_1000[1], box_1000[3]))
        px1 = max(0.0, (x1_raw / 1000.0) * w)
        py1 = max(0.0, (y1_raw / 1000.0) * h)
        px2 = min(float(w), (x2_raw / 1000.0) * w)
        py2 = min(float(h), (y2_raw / 1000.0) * h)
        if px2 <= px1 + 1.0:
            px2 = min(float(w), px1 + 2.0)
        if py2 <= py1 + 1.0:
            py2 = min(float(h), py1 + 2.0)

        query = torch.tensor([[px1, py1, px2, py2]], dtype=torch.float32) / scale
        pred_boxes = feat_pack["pred_boxes"]
        class_embeds = feat_pack["class_embeds"]
        objectness = feat_pack["objectness"]

        ious = box_iou(query, pred_boxes.float())[0]
        max_iou = float(ious.max().item()) if len(ious) > 0 else 0.0
        if max_iou > 1e-6:
            candidates = (ious >= max_iou * 0.9).nonzero().squeeze(1)
            best = int(candidates[objectness[candidates].argmax()].item())
        else:
            q_cx = float((query[0, 0] + query[0, 2]) * 0.5)
            q_cy = float((query[0, 1] + query[0, 3]) * 0.5)
            p_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
            p_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
            dists = (p_cx - q_cx) ** 2 + (p_cy - q_cy) ** 2
            best = int(dists.argmin().item())

        matched_px = (pred_boxes[best].float() * scale).tolist()
        mx1 = max(0.0, min(float(w - 1), float(matched_px[0])))
        my1 = max(0.0, min(float(h - 1), float(matched_px[1])))
        mx2 = max(mx1 + 1.0, min(float(w), float(matched_px[2])))
        my2 = max(my1 + 1.0, min(float(h), float(matched_px[3])))
        matched_1000 = [
            int(min(max(round((mx1 / float(max(1, w))) * 1000.0), 0), 999)),
            int(min(max(round((my1 / float(max(1, h))) * 1000.0), 0), 999)),
            int(min(max(round((mx2 / float(max(1, w))) * 1000.0), 1), 1000)),
            int(min(max(round((my2 / float(max(1, h))) * 1000.0), 1), 1000)),
        ]
        matched_1000[2] = max(matched_1000[0] + 1, matched_1000[2])
        matched_1000[3] = max(matched_1000[1] + 1, matched_1000[3])
        return class_embeds[best].numpy().astype(np.float32), max_iou if max_iou > 1e-6 else float(ious[best].item()), matched_1000

    def _search_features_unlocked(
        self,
        *,
        feat_pack: Dict[str, Any],
        positives: List[np.ndarray],
        negatives: List[np.ndarray],
        threshold: float,
        min_objectness: float,
        nms_iou: float,
        label: str,
        id_prefix: str = "owlv",
    ) -> Tuple[List[AutoDetection], float]:
        import torch
        import torch.nn.functional as F
        from torchvision.ops import nms

        self._last_top_objectness = None
        if not positives:
            return [], 0.0

        class_embeds = feat_pack["class_embeds"]
        pred_boxes = feat_pack["pred_boxes"]
        objectness = feat_pack["objectness"]
        w = int(feat_pack["width"])
        h = int(feat_pack["height"])
        scale = float(max(1, w, h))

        q = F.normalize(torch.from_numpy(np.stack(positives, axis=0)).float(), dim=-1)
        scores = class_embeds @ q.T
        best_score, _ = scores.max(dim=-1)
        if len(best_score) > 0:
            top_idx = int(best_score.argmax().item())
            top_score = float(best_score[top_idx].item())
            self._last_top_objectness = round(float(objectness[top_idx].item()), 4)
        else:
            top_score = 0.0

        keep = (best_score >= float(threshold)) & (objectness >= float(min_objectness))
        if negatives:
            n = F.normalize(
                torch.from_numpy(np.stack(negatives, axis=0)).float(), dim=-1
            )
            neg_max = (class_embeds @ n.T).max(dim=-1).values
            keep &= best_score > neg_max

        kept_boxes = pred_boxes[keep].float()
        kept_scores = best_score[keep]
        if len(kept_boxes) == 0:
            return [], round(top_score, 4)

        keep_idxs = nms(kept_boxes, kept_scores, float(nms_iou)).tolist()
        stamp = int(time.time() * 1000) % 100000
        detections: List[AutoDetection] = []
        for idx in keep_idxs:
            raw_px = (kept_boxes[idx] * scale).tolist()
            bx1 = max(0.0, float(raw_px[0]))
            by1 = max(0.0, float(raw_px[1]))
            bx2 = min(float(w), float(raw_px[2]))
            by2 = min(float(h), float(raw_px[3]))
            if (bx2 - bx1) < 2.0 or (by2 - by1) < 2.0:
                continue
            nx1 = int(min(max(round((bx1 / float(max(1, w))) * 1000.0), 0), 999))
            ny1 = int(min(max(round((by1 / float(max(1, h))) * 1000.0), 0), 999))
            nx2 = int(min(max(round((bx2 / float(max(1, w))) * 1000.0), nx1 + 1), 1000))
            ny2 = int(min(max(round((by2 / float(max(1, h))) * 1000.0), ny1 + 1), 1000))
            detections.append(
                AutoDetection(
                    id=f"{id_prefix}-{stamp}-{len(detections) + 1}",
                    label=label,
                    confidence=round(float(kept_scores[idx].item()), 4),
                    bbox_2d=[nx1, ny1, nx2, ny2],
                    polygon=None,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections, round(top_score, 4)

    def _search_multiclass_features_unlocked(
        self,
        *,
        feat_pack: Dict[str, Any],
        class_buckets: Dict[str, Dict[str, Any]],
        threshold: float,
        min_objectness: float,
        nms_iou: float,
        label_override: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        id_prefix: str = "owlv",
    ) -> Tuple[List[AutoDetection], float]:
        self._last_top_objectness = None
        filtered_buckets = self._filter_class_buckets(class_buckets, target_classes)
        active_buckets = [
            b for b in filtered_buckets.values() if b.get("positives")
        ]
        if not active_buckets:
            return [], 0.0

        if len(active_buckets) == 1:
            only = active_buckets[0]
            eff_label = (label_override or "").strip() or str(only.get("label") or "object")
            return self._search_features_unlocked(
                feat_pack=feat_pack,
                positives=only["positives"],
                negatives=only.get("negatives") or [],
                threshold=threshold,
                min_objectness=min_objectness,
                nms_iou=nms_iou,
                label=eff_label,
                id_prefix=id_prefix,
            )

        import torch
        import torch.nn.functional as F
        from torchvision.ops import nms

        class_embeds = feat_pack["class_embeds"]
        pred_boxes = feat_pack["pred_boxes"]
        objectness = feat_pack["objectness"]
        w = int(feat_pack["width"])
        h = int(feat_pack["height"])
        scale = float(max(1, w, h))

        per_class_masked: List[torch.Tensor] = []
        raw_max_score = 0.0
        raw_max_obj: Optional[float] = None
        for bucket in active_buckets:
            pos_list = bucket["positives"]
            neg_list = bucket.get("negatives") or []
            q = F.normalize(
                torch.from_numpy(np.stack(pos_list, axis=0)).float(), dim=-1
            )
            pos_scores = (class_embeds @ q.T).max(dim=-1).values
            if len(pos_scores) > 0:
                p_idx = int(pos_scores.argmax().item())
                p_val = float(pos_scores[p_idx].item())
                if p_val > raw_max_score:
                    raw_max_score = p_val
                    raw_max_obj = round(float(objectness[p_idx].item()), 4)
            if neg_list:
                n = F.normalize(
                    torch.from_numpy(np.stack(neg_list, axis=0)).float(), dim=-1
                )
                neg_max = (class_embeds @ n.T).max(dim=-1).values
                masked = torch.where(
                    pos_scores > neg_max,
                    pos_scores,
                    torch.full_like(pos_scores, -1.0),
                )
            else:
                masked = pos_scores
            per_class_masked.append(masked)

        stacked = torch.stack(per_class_masked, dim=-1)
        best_scores, best_cls_idxs = stacked.max(dim=-1)
        if len(best_scores) > 0:
            top_idx = int(best_scores.argmax().item())
            top_score = float(best_scores[top_idx].item())
            if top_score < 0.0:
                top_score = raw_max_score
                self._last_top_objectness = raw_max_obj
            else:
                self._last_top_objectness = round(float(objectness[top_idx].item()), 4)
        else:
            top_score = 0.0

        stamp = int(time.time() * 1000) % 100000
        candidates: List[AutoDetection] = []
        obj_ok = objectness >= float(min_objectness)
        thresh_ok = best_scores >= float(threshold)

        for c_idx, bucket in enumerate(active_buckets):
            cls_mask = (best_cls_idxs == c_idx) & thresh_ok & obj_ok
            cls_boxes = pred_boxes[cls_mask].float()
            cls_scores = best_scores[cls_mask]
            if len(cls_boxes) == 0:
                continue
            keep_idxs = nms(cls_boxes, cls_scores, float(nms_iou)).tolist()
            cls_label = str(bucket.get("label") or "object")
            for idx in keep_idxs:
                raw_px = (cls_boxes[idx] * scale).tolist()
                bx1 = max(0.0, float(raw_px[0]))
                by1 = max(0.0, float(raw_px[1]))
                bx2 = min(float(w), float(raw_px[2]))
                by2 = min(float(h), float(raw_px[3]))
                if (bx2 - bx1) < 2.0 or (by2 - by1) < 2.0:
                    continue
                nx1 = int(min(max(round((bx1 / float(max(1, w))) * 1000.0), 0), 999))
                ny1 = int(min(max(round((by1 / float(max(1, h))) * 1000.0), 0), 999))
                nx2 = int(min(max(round((bx2 / float(max(1, w))) * 1000.0), nx1 + 1), 1000))
                ny2 = int(min(max(round((by2 / float(max(1, h))) * 1000.0), ny1 + 1), 1000))
                candidates.append(
                    AutoDetection(
                        id=f"{id_prefix}-{stamp}-{len(candidates) + 1}",
                        label=cls_label,
                        confidence=round(float(cls_scores[idx].item()), 4),
                        bbox_2d=[nx1, ny1, nx2, ny2],
                        polygon=None,
                    )
                )

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        accepted: List[AutoDetection] = []
        for cand in candidates:
            if any(
                self._box_iou_1000(cand.bbox_2d, prev.bbox_2d) >= 0.70
                for prev in accepted
            ):
                continue
            accepted.append(cand)

        return accepted, round(top_score, 4)

    def _refine_dets_with_cached_sam_unlocked(
        self,
        *,
        image_path: Path,
        detections: List[AutoDetection],
        sam_refiner_id: str,
        complexity: float,
        base_engine_tag: str = "owlv2-embed",
    ) -> Tuple[List[AutoDetection], str]:
        if not detections or not sam_refiner_id or sam_refiner_id == "none":
            return detections, base_engine_tag

        eff_sam_id = (
            sam_refiner_id
            if (sam_refiner_id in aa_config.SAM_REFINERS and sam_refiner_id != "none")
            else "sam2.1-s"
        )
        predictor, w, h, _, device = self._ensure_sam_image_encoded_unlocked(
            eff_sam_id, image_path
        )

        for det in detections:
            if len(det.bbox_2d) != 4:
                continue
            px1 = max(0.0, (det.bbox_2d[0] / 1000.0) * w)
            py1 = max(0.0, (det.bbox_2d[1] / 1000.0) * h)
            px2 = min(float(w), (det.bbox_2d[2] / 1000.0) * w)
            py2 = min(float(h), (det.bbox_2d[3] / 1000.0) * h)
            if px2 <= px1 + 2.0 or py2 <= py1 + 2.0:
                continue
            pixel_bbox = [px1, py1, px2, py2]
            try:
                results = predictor(bboxes=[pixel_bbox])
            except Exception:
                if device != "cpu":
                    device = "cpu"
                    self._sam_encoded_key.pop(eff_sam_id, None)
                    predictor, _, _, _, device = self._ensure_sam_image_encoded_unlocked(
                        eff_sam_id, image_path
                    )
                    results = predictor(bboxes=[pixel_bbox])
                else:
                    raise

            if (
                not results
                or results[0].masks is None
                or results[0].masks.data is None
                or len(results[0].masks.data) == 0
            ):
                cx = (px1 + px2) * 0.5
                cy = (py1 + py2) * 0.5
                results = predictor(
                    bboxes=[pixel_bbox],
                    points=[[[cx, cy]]],
                    labels=[[1]],
                )

            if (
                results
                and results[0].masks is not None
                and results[0].masks.data is not None
                and len(results[0].masks.data) > 0
            ):
                mask_2d = results[0].masks.data[0].cpu().numpy()
                poly = self._mask_to_norm_polygon(mask_2d, complexity=complexity)
                tight_box = self._mask_to_norm_bbox(mask_2d)
                if poly:
                    det.polygon = poly
                if tight_box:
                    det.bbox_2d = tight_box

        return detections, f"{base_engine_tag}+{eff_sam_id}"

    def _ensure_session_owl_model_unlocked(
        self, session: Dict[str, Any], target_owl_id: str
    ) -> None:
        self._select_owl_model_unlocked(target_owl_id)
        if session.get("owlModelId") == target_owl_id:
            return

        had_pos, had_neg = self._session_counts_unlocked(session)
        had_redo = len(session.get("drawnRedoStack") or []) > 0

        ref_specs = list(session.get("refSpecs") or [])
        drawn_stack = list(session.get("drawnStack") or [])
        drawn_redo = list(session.get("drawnRedoStack") or [])

        classes_map: Dict[str, Dict[str, Any]] = {}
        ref_items_set: set[str] = set()

        for spec in ref_specs:
            img_path = Path(spec.get("imagePath") or "")
            if not img_path.is_file():
                continue
            boxes = spec.get("boxes") or []
            if not boxes:
                continue
            feat_pack, _ = self._extract_owlv2_features_unlocked(img_path)
            item_id = str(spec.get("itemId") or "").strip()
            item_added = False
            for box_1000, clean_lbl in boxes:
                cls_key = clean_lbl.lower()
                emb, _, _ = self._embed_region_from_features(feat_pack, list(box_1000))
                bucket = classes_map.setdefault(
                    cls_key,
                    {
                        "label": clean_lbl,
                        "positives": [],
                        "negatives": [],
                        "imagePaths": set(),
                    },
                )
                bucket["positives"].append(emb)
                bucket["imagePaths"].add(str(img_path))
                session["label"] = clean_lbl
                item_added = True
            if item_added and item_id:
                ref_items_set.add(item_id)

        new_drawn_stack: List[Dict[str, Any]] = []
        for entry in drawn_stack:
            img_path = Path(entry.get("imagePath") or "")
            box_1000 = entry.get("bbox1000")
            if not img_path.is_file() or not box_1000 or len(box_1000) != 4:
                continue
            feat_pack, _ = self._extract_owlv2_features_unlocked(img_path)
            emb, _, _ = self._embed_region_from_features(feat_pack, list(box_1000))
            clean_lbl = str(entry.get("label") or "object")
            polarity = str(entry.get("polarity") or "positive")
            target_cls_keys: List[str] = []
            if polarity == "negative":
                cls_key = clean_lbl.lower()
                if cls_key in classes_map and classes_map[cls_key].get("positives"):
                    classes_map[cls_key]["negatives"].append(emb)
                    target_cls_keys.append(cls_key)
                elif len(classes_map) == 1:
                    only_key = next(iter(classes_map.keys()))
                    classes_map[only_key]["negatives"].append(emb)
                    target_cls_keys.append(only_key)
                else:
                    for ck, b in classes_map.items():
                        if b.get("positives"):
                            b["negatives"].append(emb)
                            target_cls_keys.append(ck)
            else:
                cls_key = clean_lbl.lower()
                bucket = classes_map.setdefault(
                    cls_key,
                    {
                        "label": clean_lbl,
                        "positives": [],
                        "negatives": [],
                        "imagePaths": set(),
                    },
                )
                bucket["positives"].append(emb)
                bucket["imagePaths"].add(str(img_path))
                session["label"] = clean_lbl
                target_cls_keys.append(cls_key)
            updated_entry = dict(entry)
            updated_entry["embedding"] = emb
            updated_entry["clsKeys"] = target_cls_keys or list(entry.get("clsKeys") or [])
            new_drawn_stack.append(updated_entry)

        new_drawn_redo: List[Dict[str, Any]] = []
        for entry in drawn_redo:
            img_path = Path(entry.get("imagePath") or "")
            box_1000 = entry.get("bbox1000")
            if not img_path.is_file() or not box_1000 or len(box_1000) != 4:
                continue
            feat_pack, _ = self._extract_owlv2_features_unlocked(img_path)
            emb, _, _ = self._embed_region_from_features(feat_pack, list(box_1000))
            updated_entry = dict(entry)
            updated_entry["embedding"] = emb
            new_drawn_redo.append(updated_entry)

        session["classes"] = classes_map
        session["refItemIds"] = ref_items_set
        session["drawnStack"] = new_drawn_stack
        session["drawnRedoStack"] = new_drawn_redo
        session["owlModelId"] = target_owl_id

        new_pos, new_neg = self._session_counts_unlocked(session)
        if (had_pos or had_neg or had_redo) and new_pos == 0 and new_neg == 0 and not new_drawn_redo:
            raise KeyError(
                "Visual query model changed; please re-index references or draw exemplars again."
            )

    def build_visual_ref_bank(
        self,
        *,
        query_id: Optional[str] = None,
        replace: bool = True,
        owl_model_id: Optional[str] = None,
        ref_specs: List[Dict[str, Any]],
        target_classes: Optional[List[str]] = None,
        search_image_path: Optional[Path] = None,
        threshold: float = 0.65,
        min_objectness: float = 0.01,
        nms_iou: float = 0.30,
        use_sam_masks: bool = False,
        sam_refiner_id: str = "sam2.1-s",
        complexity: float = 0.5,
    ) -> Dict[str, Any]:
        allowed = self._normalize_target_classes(target_classes)
        if allowed is not None and not allowed:
            raise ValueError("Select at least one target class to index.")

        mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            qid = (query_id or "").strip()
            existing = self._owl_visual_queries.get(qid) if qid else None
            target_owl_id = self._select_owl_model_unlocked(
                owl_model_id or (existing or {}).get("owlModelId")
            )
            if not qid or qid not in self._owl_visual_queries:
                if len(self._owl_visual_queries) >= 16:
                    self._owl_visual_queries.pop(next(iter(self._owl_visual_queries)))
                qid = f"vq-{uuid.uuid4().hex[:10]}"
                self._owl_visual_queries[qid] = self._new_visual_query_session(target_owl_id)
            elif replace:
                self._owl_visual_queries[qid] = self._new_visual_query_session(target_owl_id)
            else:
                self._ensure_session_owl_model_unlocked(
                    self._owl_visual_queries[qid], target_owl_id
                )

            session = self._owl_visual_queries[qid]
            classes_map = session["classes"]
            ref_items_set = session.setdefault("refItemIds", set())
            stored_specs = session.setdefault("refSpecs", [])

            for spec in ref_specs:
                img_path = Path(spec["imagePath"])
                if not img_path.is_file():
                    continue
                boxes = spec.get("boxes") or []
                valid_boxes = [
                    (list(box), (lbl or "").strip() or "object")
                    for box, lbl in boxes
                    if isinstance(box, (list, tuple))
                    and len(box) == 4
                    and (
                        allowed is None
                        or ((lbl or "").strip() or "object").lower() in allowed
                    )
                ]
                if not valid_boxes:
                    continue

                feat_pack, _ = self._extract_owlv2_features_unlocked(img_path)
                item_id = str(spec.get("itemId") or "").strip()
                item_added = False
                for box_1000, clean_lbl in valid_boxes:
                    cls_key = clean_lbl.lower()
                    emb, _, _ = self._embed_region_from_features(feat_pack, list(box_1000))
                    bucket = classes_map.setdefault(
                        cls_key,
                        {
                            "label": clean_lbl,
                            "positives": [],
                            "negatives": [],
                            "imagePaths": set(),
                        },
                    )
                    bucket["positives"].append(emb)
                    bucket["imagePaths"].add(str(img_path))
                    session["label"] = clean_lbl
                    item_added = True
                if item_added:
                    stored_specs.append(
                        {
                            "itemId": item_id,
                            "imagePath": str(img_path),
                            "boxes": valid_boxes,
                        }
                    )
                    if item_id:
                        ref_items_set.add(item_id)

            pos_cnt, neg_cnt = self._session_counts_unlocked(session, target_classes)
            if pos_cnt == 0:
                if allowed is not None:
                    raise ValueError(
                        "No bounding boxes found matching the selected target classes."
                    )
                raise ValueError(
                    "No valid labeled bounding boxes found in selected ViT references."
                )

            dets: List[AutoDetection] = []
            top_score = 0.0
            top_obj: Optional[float] = None
            engine_used = self._owl_short_tag(target_owl_id, embed=True)
            if search_image_path and Path(search_image_path).is_file():
                s_path = Path(search_image_path)
                s_pack, _ = self._extract_owlv2_features_unlocked(s_path)
                dets, top_score = self._search_multiclass_features_unlocked(
                    feat_pack=s_pack,
                    class_buckets=classes_map,
                    threshold=threshold,
                    min_objectness=min_objectness,
                    nms_iou=nms_iou,
                    label_override=None,
                    target_classes=target_classes,
                    id_prefix="owlv",
                )
                top_obj = self._last_top_objectness
                if use_sam_masks and dets:
                    dets, engine_used = self._refine_dets_with_cached_sam_unlocked(
                        image_path=s_path,
                        detections=dets,
                        sam_refiner_id=sam_refiner_id,
                        complexity=complexity,
                        base_engine_tag=engine_used,
                    )

            summaries = self._session_class_summaries_unlocked(session, target_classes)
            latency_s = round(time.perf_counter() - t0, 3)
            return {
                "queryId": qid,
                "owlModelId": target_owl_id,
                "totalExemplars": pos_cnt,
                "totalNegatives": neg_cnt,
                "totalRefImages": len(ref_items_set),
                "refItemIds": sorted(ref_items_set),
                "classes": summaries,
                "detections": dets,
                "topScore": top_score,
                "topObjectness": top_obj,
                "latencySec": latency_s,
                "engineUsed": engine_used,
            }

    def add_visual_embed_exemplar(
        self,
        *,
        query_id: Optional[str],
        image_path: Path,
        bbox_1000: Optional[List[float]] = None,
        point_1000: Optional[List[float]] = None,
        polarity: str = "positive",
        label: str = "object",
        target_classes: Optional[List[str]] = None,
        owl_model_id: Optional[str] = None,
        threshold: float = 0.65,
        min_objectness: float = 0.01,
        nms_iou: float = 0.30,
        use_sam_masks: bool = False,
        sam_refiner_id: str = "sam2.1-s",
        complexity: float = 0.5,
    ) -> Dict[str, Any]:
        resolved_box = list(bbox_1000) if (bbox_1000 and len(bbox_1000) == 4) else None
        if resolved_box is None and point_1000 and len(point_1000) >= 2:
            sam_det, _ = self.predict_sam_interactive(
                image_path=image_path,
                points=[[float(point_1000[0]), float(point_1000[1])]],
                point_labels=[1],
                bbox_prompt=None,
                sam_refiner_id=sam_refiner_id,
                label=label,
                output_mode="bbox",
                use_owlv2_assist=False,
                complexity=complexity,
                preview=False,
            )
            if sam_det is None or len(sam_det.bbox_2d) != 4:
                raise ValueError("Could not segment an object at the clicked point to extract an example box.")
            resolved_box = [float(v) for v in sam_det.bbox_2d]

        if resolved_box is None or len(resolved_box) != 4:
            raise ValueError("Draw a bounding box (or click an object) to extract its OWLv2 visual embedding.")

        mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            qid = (query_id or "").strip()
            existing_session = self._owl_visual_queries.get(qid) if qid else None
            target_owl_id = self._select_owl_model_unlocked(
                owl_model_id or (existing_session or {}).get("owlModelId")
            )
            if existing_session is not None:
                self._ensure_session_owl_model_unlocked(existing_session, target_owl_id)

            feat_pack, feat_cached = self._extract_owlv2_features_unlocked(image_path)
            emb, matched_iou, matched_box = self._embed_region_from_features(
                feat_pack, resolved_box
            )

            if not qid or qid not in self._owl_visual_queries:
                if polarity == "negative":
                    raise ValueError(
                        "Add at least one positive example box before adding a negative box."
                    )
                if len(self._owl_visual_queries) >= 16:
                    self._owl_visual_queries.pop(next(iter(self._owl_visual_queries)))
                qid = f"vq-{uuid.uuid4().hex[:10]}"
                self._owl_visual_queries[qid] = self._new_visual_query_session(target_owl_id)

            session = self._owl_visual_queries[qid]
            classes_map = session["classes"]
            clean_label = (label or "").strip() or session.get("label") or "object"
            cls_key = clean_label.lower()
            ex_id = f"ex-{uuid.uuid4().hex[:8]}"
            target_cls_keys: List[str] = []

            if polarity == "negative":
                pos_cnt_now, _ = self._session_counts_unlocked(session)
                if pos_cnt_now == 0:
                    raise ValueError(
                        "Add at least one positive example box before adding a negative box."
                    )
                if cls_key in classes_map and classes_map[cls_key].get("positives"):
                    classes_map[cls_key]["negatives"].append(emb)
                    target_cls_keys.append(cls_key)
                elif len(classes_map) == 1:
                    only_key = next(iter(classes_map.keys()))
                    classes_map[only_key]["negatives"].append(emb)
                    target_cls_keys.append(only_key)
                else:
                    for ck, b in classes_map.items():
                        if b.get("positives"):
                            b["negatives"].append(emb)
                            target_cls_keys.append(ck)
            else:
                bucket = classes_map.setdefault(
                    cls_key,
                    {
                        "label": clean_label,
                        "positives": [],
                        "negatives": [],
                        "imagePaths": set(),
                    },
                )
                bucket["positives"].append(emb)
                bucket["imagePaths"].add(str(image_path))
                session["label"] = clean_label
                target_cls_keys.append(cls_key)

            drawn_stack = session.setdefault("drawnStack", [])
            drawn_redo = session.setdefault("drawnRedoStack", [])
            drawn_stack.append(
                {
                    "exemplarId": ex_id,
                    "clsKeys": target_cls_keys,
                    "label": clean_label,
                    "polarity": polarity,
                    "embedding": emb,
                    "imagePath": str(image_path),
                    "bbox1000": [float(v) for v in resolved_box],
                }
            )
            drawn_redo.clear()

            pos_cnt, neg_cnt = self._session_counts_unlocked(session, target_classes)
            dets, top_score = self._search_multiclass_features_unlocked(
                feat_pack=feat_pack,
                class_buckets=classes_map,
                threshold=threshold,
                min_objectness=min_objectness,
                nms_iou=nms_iou,
                label_override=clean_label
                if (len(classes_map) == 1 and target_classes is None)
                else None,
                target_classes=target_classes,
                id_prefix="owlv",
            )
            top_obj = self._last_top_objectness
            engine_used = self._owl_short_tag(target_owl_id, embed=True)
            if use_sam_masks and dets:
                dets, engine_used = self._refine_dets_with_cached_sam_unlocked(
                    image_path=image_path,
                    detections=dets,
                    sam_refiner_id=sam_refiner_id,
                    complexity=complexity,
                    base_engine_tag=engine_used,
                )

            latency_s = round(time.perf_counter() - t0, 3)
            return {
                "queryId": qid,
                "exemplarId": ex_id,
                "owlModelId": target_owl_id,
                "canUndo": len(drawn_stack) > 0,
                "canRedo": False,
                "detections": dets,
                "topScore": top_score,
                "topObjectness": top_obj,
                "matchedIou": round(float(matched_iou), 4),
                "matchedBox": matched_box,
                "positivesCount": pos_cnt,
                "negativesCount": neg_cnt,
                "classes": self._session_class_summaries_unlocked(session, target_classes),
                "latencySec": latency_s,
                "cached": feat_cached,
                "engineUsed": engine_used,
            }

    def undo_redo_visual_embed_exemplar(
        self,
        *,
        query_id: str,
        action: str,
        search_image_path: Optional[Path] = None,
        label: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        owl_model_id: Optional[str] = None,
        threshold: float = 0.65,
        min_objectness: float = 0.01,
        nms_iou: float = 0.30,
        use_sam_masks: bool = False,
        sam_refiner_id: str = "sam2.1-s",
        complexity: float = 0.5,
    ) -> Dict[str, Any]:
        if self._owl_model is None:
            mlx_engine.unload()

        with _PT_LOCK:
            t0 = time.perf_counter()
            qid = (query_id or "").strip()
            session = self._owl_visual_queries.get(qid)
            if not session:
                raise KeyError(
                    "Visual query expired or reset; draw a box or pin ViT references first."
                )

            target_owl_id = self._select_owl_model_unlocked(
                owl_model_id or session.get("owlModelId")
            )
            self._ensure_session_owl_model_unlocked(session, target_owl_id)

            classes_map = session.setdefault("classes", {})
            drawn_stack = session.setdefault("drawnStack", [])
            drawn_redo = session.setdefault("drawnRedoStack", [])
            acted_id: Optional[str] = None

            if action == "undo" and drawn_stack:
                entry = drawn_stack.pop()
                acted_id = entry.get("exemplarId")
                target_field = (
                    "positives" if entry.get("polarity") == "positive" else "negatives"
                )
                emb = entry["embedding"]
                for ck in entry.get("clsKeys") or []:
                    bucket = classes_map.get(ck)
                    if not bucket:
                        continue
                    lst = bucket.get(target_field) or []
                    for idx in range(len(lst) - 1, -1, -1):
                        if np.array_equal(lst[idx], emb):
                            lst.pop(idx)
                            break
                    if not bucket.get("positives") and not bucket.get("negatives"):
                        classes_map.pop(ck, None)
                drawn_redo.append(entry)
            elif action == "redo" and drawn_redo:
                entry = drawn_redo.pop()
                acted_id = entry.get("exemplarId")
                target_field = (
                    "positives" if entry.get("polarity") == "positive" else "negatives"
                )
                emb = entry["embedding"]
                clean_lbl = str(entry.get("label") or "object")
                for ck in entry.get("clsKeys") or [clean_lbl.lower()]:
                    bucket = classes_map.setdefault(
                        ck,
                        {
                            "label": clean_lbl,
                            "positives": [],
                            "negatives": [],
                            "imagePaths": set(),
                        },
                    )
                    bucket[target_field].append(emb)
                    if target_field == "positives" and entry.get("imagePath"):
                        bucket["imagePaths"].add(str(entry["imagePath"]))
                        session["label"] = clean_lbl
                drawn_stack.append(entry)

            pos_cnt, neg_cnt = self._session_counts_unlocked(session, target_classes)
            filtered_buckets = self._filter_class_buckets(classes_map, target_classes)
            active_buckets = [b for b in filtered_buckets.values() if b.get("positives")]
            dets: List[AutoDetection] = []
            top_score = 0.0
            top_obj: Optional[float] = None
            feat_cached = False
            engine_used = self._owl_short_tag(target_owl_id, embed=True)

            if (
                active_buckets
                and search_image_path is not None
                and Path(search_image_path).is_file()
            ):
                s_path = Path(search_image_path)
                label_override = (
                    (label or "").strip() or None
                    if (len(active_buckets) == 1 and target_classes is None)
                    else None
                )
                feat_pack, feat_cached = self._extract_owlv2_features_unlocked(s_path)
                dets, top_score = self._search_multiclass_features_unlocked(
                    feat_pack=feat_pack,
                    class_buckets=classes_map,
                    threshold=threshold,
                    min_objectness=min_objectness,
                    nms_iou=nms_iou,
                    label_override=label_override,
                    target_classes=target_classes,
                    id_prefix="owlv",
                )
                top_obj = self._last_top_objectness
                if use_sam_masks and dets:
                    dets, engine_used = self._refine_dets_with_cached_sam_unlocked(
                        image_path=s_path,
                        detections=dets,
                        sam_refiner_id=sam_refiner_id,
                        complexity=complexity,
                        base_engine_tag=engine_used,
                    )

            latency_s = round(time.perf_counter() - t0, 3)
            return {
                "queryId": qid,
                "exemplarId": acted_id,
                "owlModelId": target_owl_id,
                "canUndo": len(drawn_stack) > 0,
                "canRedo": len(drawn_redo) > 0,
                "detections": dets,
                "topScore": top_score,
                "topObjectness": top_obj,
                "matchedIou": None,
                "matchedBox": None,
                "positivesCount": pos_cnt,
                "negativesCount": neg_cnt,
                "classes": self._session_class_summaries_unlocked(session, target_classes),
                "latencySec": latency_s,
                "cached": feat_cached,
                "engineUsed": engine_used,
            }

    def search_visual_embed_image(
        self,
        *,
        query_id: str,
        image_path: Path,
        label: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        owl_model_id: Optional[str] = None,
        threshold: float = 0.65,
        min_objectness: float = 0.01,
        nms_iou: float = 0.30,
        use_sam_masks: bool = False,
        sam_refiner_id: str = "sam2.1-s",
        complexity: float = 0.5,
    ) -> Dict[str, Any]:
        allowed = self._normalize_target_classes(target_classes)
        if allowed is not None and not allowed:
            raise ValueError("Select at least one target class to search.")

        if self._owl_model is None:
            mlx_engine.unload()

        with _PT_LOCK:
            qid = (query_id or "").strip()
            session = self._owl_visual_queries.get(qid)
            raw_buckets = (session or {}).get("classes") or {}
            if not session or not any(b.get("positives") for b in raw_buckets.values()):
                raise KeyError(
                    "Visual query expired or reset; draw a box to start a new query."
                )

            target_owl_id = self._select_owl_model_unlocked(
                owl_model_id or session.get("owlModelId")
            )
            self._ensure_session_owl_model_unlocked(session, target_owl_id)
            classes_map = session.get("classes") or {}
            filtered_buckets = self._filter_class_buckets(classes_map, target_classes)
            active_buckets = [b for b in filtered_buckets.values() if b.get("positives")]
            if allowed is not None and not active_buckets:
                raise ValueError(
                    "No indexed visual references match the selected classes."
                )

            label_override = (
                (label or "").strip() or None
                if (len(active_buckets) == 1 and target_classes is None)
                else None
            )
            t0 = time.perf_counter()
            feat_pack, feat_cached = self._extract_owlv2_features_unlocked(image_path)
            dets, top_score = self._search_multiclass_features_unlocked(
                feat_pack=feat_pack,
                class_buckets=classes_map,
                threshold=threshold,
                min_objectness=min_objectness,
                nms_iou=nms_iou,
                label_override=label_override,
                target_classes=target_classes,
                id_prefix="owlv",
            )
            top_obj = self._last_top_objectness
            engine_used = self._owl_short_tag(target_owl_id, embed=True)
            if use_sam_masks and dets:
                dets, engine_used = self._refine_dets_with_cached_sam_unlocked(
                    image_path=image_path,
                    detections=dets,
                    sam_refiner_id=sam_refiner_id,
                    complexity=complexity,
                    base_engine_tag=engine_used,
                )

            pos_cnt, neg_cnt = self._session_counts_unlocked(session, target_classes)
            latency_s = round(time.perf_counter() - t0, 3)
            return {
                "queryId": qid,
                "exemplarId": None,
                "owlModelId": target_owl_id,
                "canUndo": len(session.get("drawnStack") or []) > 0,
                "canRedo": len(session.get("drawnRedoStack") or []) > 0,
                "detections": dets,
                "topScore": top_score,
                "topObjectness": top_obj,
                "matchedIou": None,
                "matchedBox": None,
                "positivesCount": pos_cnt,
                "negativesCount": neg_cnt,
                "classes": self._session_class_summaries_unlocked(session, target_classes),
                "latencySec": latency_s,
                "cached": feat_cached,
                "engineUsed": engine_used,
            }


pytorch_engine = PyTorchVisionEngine()
