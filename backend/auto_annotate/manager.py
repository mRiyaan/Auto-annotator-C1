"""Isolated batch worker queue, pause/cancel handling, and SSE streaming for Auto-Annotation."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as base_config
from auto_annotate import config as aa_config
from auto_annotate import db as aa_db
from auto_annotate.pytorch_engine import pytorch_engine
from auto_annotate.schemas import AutoDetection, AutoItemRecord, AutoJobRecord
from engines.florence_engine import florence_engine
from engines.mlx_lora_engine import mlx_engine
from engines.ollama_engine import ollama_engine
from services.annotation_parser import parse_detections
from services.few_shot_builder import (
    build_image_prompt,
    build_interleaved_image_messages,
    prepare_pil_image,
)


class AutoAnnotateManager:
    """Runs Roboflow batch inference sequentially across images in 50-image batches."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, List[queue.Queue[Dict[str, Any]]]] = {}
        self._pause_flags: Dict[str, threading.Event] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def clear_memory_cache(self) -> None:
        with self._lock:
            self._progress.clear()
            self._pause_flags.clear()
            self._cancel_flags.clear()

    def _ensure_flags(self, job_id: str) -> None:
        with self._lock:
            self._pause_flags.setdefault(job_id, threading.Event())
            self._cancel_flags.setdefault(job_id, threading.Event())

    def get_progress(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id in self._progress:
                return dict(self._progress[job_id])
        job = aa_db.get_job(job_id)
        if job:
            pct = (
                100
                if job.status in ("completed", "cancelled", "failed")
                else (
                    int((job.annotatedCount / max(1, job.totalImages)) * 100)
                    if job.totalImages > 0
                    else 0
                )
            )
            return {
                "jobId": job_id,
                "status": job.status,
                "progressPct": pct,
                "stage": f"{job.annotatedCount} / {job.totalImages} images annotated",
                "subtext": f"{job.projectSlug} · {job.modelLabel}",
                "job": job.model_dump(),
            }
        return {
            "jobId": job_id,
            "status": "failed",
            "progressPct": 0,
            "stage": "Not Found",
            "subtext": f"Job {job_id} does not exist.",
        }

    def subscribe(self, job_id: str) -> queue.Queue[Dict[str, Any]]:
        sub: queue.Queue[Dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(sub)
            snap = self._progress.get(job_id)
        if snap:
            sub.put(dict(snap))
        return sub

    def unsubscribe(self, job_id: str, sub: queue.Queue[Dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if sub in subs:
                subs.remove(sub)

    def emit(
        self,
        job_id: str,
        *,
        status: str,
        progress_pct: int,
        stage: str,
        subtext: str,
        job: Optional[AutoJobRecord] = None,
        latest_item: Optional[AutoItemRecord] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "jobId": job_id,
            "status": status,
            "progressPct": progress_pct,
            "stage": stage,
            "subtext": subtext,
        }
        if job is not None:
            payload["job"] = job.model_dump()
        if latest_item is not None:
            payload["latestItem"] = latest_item.model_dump()
        with self._lock:
            self._progress[job_id] = payload
            subs = list(self._subscribers.get(job_id, []))
        for s in subs:
            s.put(payload)

    def enqueue(self, job_id: str) -> None:
        self._ensure_flags(job_id)
        self._pause_flags[job_id].clear()
        self._cancel_flags[job_id].clear()
        job = aa_db.get_job(job_id)
        if job:
            job.status = "queued"
            job.errorMessage = None
            aa_db.upsert_job(job)
            self.emit(
                job_id,
                status="queued",
                progress_pct=0,
                stage="Queued on Local M3 Pro Worker",
                subtext=f"{job.totalImages} images across {len(job.batches)} batch(es)",
                job=job,
            )
        self._queue.put(job_id)

    def pause(self, job_id: str) -> Optional[AutoJobRecord]:
        self._ensure_flags(job_id)
        self._pause_flags[job_id].set()
        job = aa_db.get_job(job_id)
        if job and job.status in ("queued", "running"):
            job.status = "paused"
            aa_db.upsert_job(job)
            self.emit(
                job_id,
                status="paused",
                progress_pct=int((job.annotatedCount / max(1, job.totalImages)) * 100),
                stage=f"Paused at {job.annotatedCount} / {job.totalImages}",
                subtext="Click Resume to continue remaining images.",
                job=job,
            )
            return aa_db.get_job(job_id)
        return job

    def resume(self, job_id: str) -> Optional[AutoJobRecord]:
        self._ensure_flags(job_id)
        self._pause_flags[job_id].clear()
        job = aa_db.get_job(job_id)
        if job and job.totalImages == 0 and len(aa_db.list_items(job_id)) == 0:
            raise ValueError(
                "Cannot resume a job with 0 fetched images. Fix the Roboflow connection and start a new fetch."
            )
        if job and job.status in ("paused", "cancelled", "failed"):
            self.enqueue(job_id)
            return aa_db.get_job(job_id)
        return job

    def cancel(self, job_id: str) -> Optional[AutoJobRecord]:
        self._ensure_flags(job_id)
        self._cancel_flags[job_id].set()
        job = aa_db.get_job(job_id)
        if job and job.status in ("fetching", "queued", "running", "paused"):
            job.status = "cancelled"
            aa_db.upsert_job(job)
            self.emit(
                job_id,
                status="cancelled",
                progress_pct=100,
                stage=f"Cancelled ({job.annotatedCount} / {job.totalImages} completed)",
                subtext="Completed images remain available in Verify Auto-Annotation.",
                job=job,
            )
            return aa_db.get_job(job_id)
        return job

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute_job(job_id)
            except Exception as exc:
                job = aa_db.get_job(job_id)
                if job:
                    job.status = "failed"
                    job.errorMessage = str(exc)
                    aa_db.upsert_job(job)
                    self.emit(
                        job_id,
                        status="failed",
                        progress_pct=100,
                        stage="Batch Failed",
                        subtext=str(exc),
                        job=job,
                    )
            finally:
                self._queue.task_done()

    def _infer_single_image(
        self,
        *,
        job: AutoJobRecord,
        item_path: Path,
        item_id: str,
        ref_pils: List[Any],
        ref_notes: List[str],
    ) -> tuple[List[AutoDetection], float]:
        detector = aa_config.get_detector_entry(job.modelId)
        engine = detector["engine"]
        needs_poly = job.outputMode in ("polygon", "both")
        use_sam = needs_poly and bool(job.samRefinerId and job.samRefinerId != "none")
        class_names = list(job.classes) or [c.name for c in job.classColors]

        # When a SAM refiner is chosen with VLM/Florence-2, get bbox_2d first so SAM draws tight contours
        vlm_output_mode = (
            "bbox" if (use_sam and engine in ("mlx", "ollama", "florence2")) else job.outputMode
        )

        t0 = time.perf_counter()
        if engine == "pytorch":
            florence_engine.unload()
            detections, _ = pytorch_engine.predict_yoloe(
                image_path=item_path,
                class_names=class_names,
                output_mode=job.outputMode,
                conf_threshold=job.confidenceThreshold,
                id_prefix=f"{item_id}-det",
            )
            if use_sam and detections:
                detections, _ = pytorch_engine.refine_boxes_with_sam(
                    image_path=item_path,
                    detections=detections,
                    sam_refiner_id=job.samRefinerId,
                )
            elif job.outputMode == "bbox":
                for d in detections:
                    d.polygon = None
            return detections, round(time.perf_counter() - t0, 3)

        if engine == "owlv2":
            florence_engine.unload()
            eff_sam_id = job.samRefinerId if use_sam else "none"
            detections, _ = pytorch_engine.predict_owlv2_find_objects(
                image_path=item_path,
                class_names=class_names,
                conf_threshold=job.confidenceThreshold,
                output_mode=job.outputMode,
                sam_refiner_id=eff_sam_id,
                owl_model_id=detector.get("model_path"),
                id_prefix=f"{item_id}-det",
            )
            if job.outputMode == "bbox":
                for d in detections:
                    d.polygon = None
            return detections, round(time.perf_counter() - t0, 3)

        if engine == "florence2":
            if not use_sam:
                pytorch_engine.unload_pytorch()
            target_pil = prepare_pil_image(item_path, max_side=0)
            raw_dets, _, _ = florence_engine.predict_detections(
                model_id=job.modelId,
                image=target_pil,
                prompt=job.prompt,
                classes=class_names,
                output_mode=vlm_output_mode,
                max_tokens=job.maxTokens,
                generate_polygons=(needs_poly and not use_sam),
            )
            detections = [
                AutoDetection(
                    id=f"{item_id}-det-{idx}",
                    label=str(d["label"]),
                    confidence=float(d.get("confidence", 0.9)),
                    bbox_2d=list(d["bbox_2d"]),
                    polygon=d.get("polygon") if (needs_poly and not use_sam) else None,
                )
                for idx, d in enumerate(raw_dets, start=1)
                if float(d.get("confidence", 0.9)) >= job.confidenceThreshold
            ]
            if use_sam and detections:
                mlx_engine.unload()
                detections, _ = pytorch_engine.refine_boxes_with_sam(
                    image_path=item_path,
                    detections=detections,
                    sam_refiner_id=job.samRefinerId,
                )
            elif job.outputMode == "bbox":
                for d in detections:
                    d.polygon = None
            return detections, round(time.perf_counter() - t0, 3)

        # Free PyTorch MPS & Florence-2 before running MLX or Ollama on 18 GB M3 Pro RAM
        pytorch_engine.unload_pytorch()
        florence_engine.unload()

        target_pil = prepare_pil_image(item_path, max_side=0)
        all_images = [*ref_pils, target_pil]
        has_lora = bool(detector.get("adapter_path"))

        prompt_str = build_image_prompt(
            job.prompt,
            output_mode=vlm_output_mode,
            shot_mode=job.shotMode,
            ref_notes=ref_notes,
            style_note=job.styleNote or "",
            has_lora_adapter=has_lora,
        )

        if engine == "mlx":
            mlx_prompt = build_interleaved_image_messages(
                job.prompt,
                output_mode=vlm_output_mode,
                shot_mode=job.shotMode,
                ref_notes=ref_notes,
                style_note=job.styleNote or "",
                has_lora_adapter=has_lora,
            )
            raw_text, _ = mlx_engine.generate_multimodal(
                model_id=job.modelId,
                prompt_text=mlx_prompt,
                images=all_images,
                max_tokens=job.maxTokens,
                temperature=job.temperature,
            )
        else:
            mlx_engine.unload()
            reachable, available_models = ollama_engine.check_health()
            if not reachable:
                raise RuntimeError(
                    f"Ollama is not reachable at {base_config.OLLAMA_BASE_URL}. Run `ollama serve` first."
                )
            target_tag = detector["model_path"]
            if not any(target_tag in m for m in available_models):
                raise RuntimeError(
                    f"Ollama model '{target_tag}' not found locally. Available: {available_models}"
                )
            raw_text, _ = ollama_engine.generate_multimodal(
                model_name=target_tag,
                prompt_text=prompt_str,
                images=all_images,
                max_tokens=job.maxTokens,
                temperature=job.temperature,
            )

        parsed, _ = parse_detections(
            raw_text, output_mode=vlm_output_mode, id_prefix=f"{item_id}-det"
        )
        detections = [
            AutoDetection(
                id=d.id,
                label=d.label,
                confidence=d.confidence,
                bbox_2d=list(d.bbox_2d),
                polygon=[list(p) for p in d.polygon] if d.polygon else None,
            )
            for d in parsed
            if d.confidence >= job.confidenceThreshold
        ]

        if use_sam and detections:
            # Unload MLX temporarily if both VLM & SAM would strain RAM, or let SAM load on MPS cleanly
            mlx_engine.unload()
            detections, _ = pytorch_engine.refine_boxes_with_sam(
                image_path=item_path,
                detections=detections,
                sam_refiner_id=job.samRefinerId,
            )
        elif job.outputMode == "bbox":
            for d in detections:
                d.polygon = None

        return detections, round(time.perf_counter() - t0, 3)

    def _execute_job(self, job_id: str) -> None:
        self._ensure_flags(job_id)
        pause_ev = self._pause_flags[job_id]
        cancel_ev = self._cancel_flags[job_id]

        job = aa_db.get_job(job_id)
        if not job or cancel_ev.is_set() or job.status == "cancelled":
            return

        job.status = "running"
        job.errorMessage = None
        aa_db.upsert_job(job)

        meta = aa_db.get_job_internal_meta(job_id)
        ref_paths = [Path(p) for p in meta.get("ref_disk_paths", []) if Path(p).is_file()]
        ref_pils = (
            [prepare_pil_image(p, max_side=0) for p in ref_paths]
            if job.shotMode == "few-shot"
            else []
        )
        ref_notes = meta.get("ref_notes", [])[: len(ref_pils)]

        all_entries = aa_db.get_all_items_with_paths(job_id)
        total = len(all_entries)

        for idx, entry in enumerate(all_entries, start=1):
            if cancel_ev.is_set():
                return
            if pause_ev.is_set():
                return

            item: AutoItemRecord = entry["item"]
            if item.inferenceStatus == "completed":
                continue

            item_path = Path(entry["localPath"])
            pct = int(((idx - 1) / max(1, total)) * 100)
            self.emit(
                job_id,
                status="running",
                progress_pct=pct,
                stage=f"Annotating {item.batchId} · Image {idx} / {total}",
                subtext=f"{item.filename} · {job.modelLabel}",
            )

            try:
                detections, dt = self._infer_single_image(
                    job=job,
                    item_path=item_path,
                    item_id=item.id,
                    ref_pils=ref_pils,
                    ref_notes=ref_notes,
                )
                aa_db.update_item_inference(
                    item.id,
                    inference_status="completed",
                    detections=detections,
                    latency_sec=dt,
                )
            except Exception as exc:
                aa_db.update_item_inference(
                    item.id,
                    inference_status="failed",
                    detections=[],
                    latency_sec=0.0,
                    error_message=str(exc),
                )

            refreshed_job = aa_db.get_job(job_id)
            refreshed_item_wrap = aa_db.get_item_with_path(job_id, item.id)
            done_count = refreshed_job.annotatedCount if refreshed_job else idx
            done_pct = int((done_count / max(1, total)) * 100)
            self.emit(
                job_id,
                status="running",
                progress_pct=done_pct,
                stage=f"Annotated {done_count} / {total} images ({item.batchId})",
                subtext=f"Latest: {item.filename} ({len(refreshed_item_wrap['item'].detections) if refreshed_item_wrap else 0} detections)",
                job=refreshed_job,
                latest_item=refreshed_item_wrap["item"] if refreshed_item_wrap else None,
            )

        if cancel_ev.is_set() or pause_ev.is_set():
            return

        final_job = aa_db.get_job(job_id)
        if final_job:
            final_job.status = "completed"
            aa_db.upsert_job(final_job)
            hydrated = aa_db.get_job(job_id)
            self.emit(
                job_id,
                status="completed",
                progress_pct=100,
                stage=f"Completed {hydrated.annotatedCount if hydrated else total} / {total} images",
                subtext="Ready for batch-by-batch review on Verify Auto-Annotation.",
                job=hydrated,
            )


auto_annotate_manager = AutoAnnotateManager()
