"""Single-concurrency local GPU worker queue + live SSE progress tracking."""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import db
from engines.florence_engine import florence_engine
from engines.mlx_lora_engine import mlx_engine
from engines.ollama_engine import ollama_engine
from schemas import (
    DetectionItem,
    ReferencePreview,
    RunRecord,
    StageLogItem,
    VideoEventItem,
    VideoKeyframeItem,
)
from services.annotation_parser import (
    norm_1000_to_pixel_xyxy,
    parse_detections,
    parse_stage1_events,
    strip_thinking_and_fences,
)
from services.few_shot_builder import (
    build_image_prompt,
    build_interleaved_image_messages,
    build_video_stage1_prompt,
    build_video_stage2_prompt,
    prepare_pil_image,
)
from services.media_service import extract_video_keyframes, render_video_output


class JobManager:
    """Ensures at most 1 heavy VLM job runs at a time on M3 Pro 18 GB Unified Memory."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, List[queue.Queue[Dict[str, Any]]]] = {}
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def get_progress(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            if run_id in self._progress:
                return dict(self._progress[run_id])
        rec = db.get_run(run_id)
        if rec:
            return {
                "runId": run_id,
                "status": rec.status,
                "progressPct": 100 if rec.status in ("completed", "failed") else 0,
                "stage": "Completed" if rec.status == "completed" else (rec.errorMessage or rec.status),
                "subtext": rec.summaryText,
                "record": rec.model_dump(),
            }
        return {
            "runId": run_id,
            "status": "failed",
            "progressPct": 0,
            "stage": "Not Found",
            "subtext": f"Run {run_id} does not exist.",
        }

    def subscribe(self, run_id: str) -> queue.Queue[Dict[str, Any]]:
        sub: queue.Queue[Dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(sub)
            snap = self._progress.get(run_id)
        if snap:
            sub.put(dict(snap))
        return sub

    def unsubscribe(self, run_id: str, sub: queue.Queue[Dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if sub in subs:
                subs.remove(sub)

    def _emit_progress(
        self,
        run_id: str,
        *,
        status: str,
        progress_pct: int,
        stage: str,
        subtext: str,
        record: Optional[RunRecord] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "runId": run_id,
            "status": status,
            "progressPct": progress_pct,
            "stage": stage,
            "subtext": subtext,
        }
        if record is not None:
            payload["record"] = record.model_dump()

        with self._lock:
            self._progress[run_id] = payload
            subs = list(self._subscribers.get(run_id, []))
        for sub in subs:
            sub.put(payload)

    def submit_job(
        self,
        *,
        record: RunRecord,
        target_disk_path: str,
        ref_disk_paths: List[str],
    ) -> None:
        db.upsert_run(
            record,
            internal_meta={
                "target_disk_path": target_disk_path,
                "ref_disk_paths": ref_disk_paths,
            },
        )
        self._emit_progress(
            record.id,
            status="queued",
            progress_pct=5,
            stage="Queued on Local M3 Pro Worker",
            subtext=f"{record.modelLabel} · {record.target.filename}",
        )
        self._queue.put(
            {
                "run_id": record.id,
                "target_disk_path": target_disk_path,
                "ref_disk_paths": ref_disk_paths,
            }
        )

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._execute_job(job)
            except Exception as exc:
                run_id = job["run_id"]
                rec = db.get_run(run_id)
                if rec:
                    rec.status = "failed"
                    rec.errorMessage = str(exc)
                    rec.summaryText = f"Run failed: {exc}"
                    db.upsert_run(rec)
                    self._emit_progress(
                        run_id,
                        status="failed",
                        progress_pct=100,
                        stage="Execution Failed",
                        subtext=str(exc),
                        record=rec,
                    )
            finally:
                self._queue.task_done()

    def _run_model(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images: List[Any],
        max_tokens: int,
        temperature: float,
        prompt_input: Optional[Any] = None,
        output_mode: str = "bbox",
        raw_prompt: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        entry = config.get_model_entry(model_id)
        if entry["engine"] == "mlx":
            florence_engine.unload()
            return mlx_engine.generate_multimodal(
                model_id=model_id,
                prompt_text=prompt_input if prompt_input is not None else prompt_text,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if entry["engine"] == "florence2":
            mlx_engine.unload()
            return florence_engine.generate_multimodal(
                model_id=model_id,
                prompt_text=raw_prompt if raw_prompt is not None else prompt_text,
                images=images,
                output_mode=output_mode,
                max_tokens=max_tokens,
            )

        # Unload local GPU models before calling Ollama 8B so 18 GB unified RAM stays clear
        mlx_engine.unload()
        florence_engine.unload()
        reachable, available_models = ollama_engine.check_health()
        if not reachable:
            raise RuntimeError(
                f"Ollama is not reachable at {config.OLLAMA_BASE_URL}. Start Ollama (`ollama serve`) to run {entry['model_path']}."
            )
        target_tag = entry["model_path"]
        if not any(target_tag in m for m in available_models):
            raise RuntimeError(
                f"Ollama model '{target_tag}' not found locally. Available: {available_models}"
            )
        return ollama_engine.generate_multimodal(
            model_name=target_tag,
            prompt_text=prompt_text,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _execute_job(self, job: Dict[str, Any]) -> None:
        run_id = job["run_id"]
        rec = db.get_run(run_id)
        if not rec:
            return

        rec.status = "running"
        db.upsert_run(rec)

        target_path = Path(job["target_disk_path"])
        ref_paths = (
            [Path(p) for p in job.get("ref_disk_paths", [])]
            if rec.shotMode == "few-shot"
            else []
        )

        if rec.target.mediaType == "image":
            self._execute_image_job(rec, target_path, ref_paths)
        else:
            self._execute_video_job(rec, target_path, ref_paths)

    def _execute_image_job(
        self,
        rec: RunRecord,
        target_path: Path,
        ref_paths: List[Path],
    ) -> None:
        entry = config.get_model_entry(rec.modelId)
        has_lora = bool(entry.get("adapter_path"))

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=20,
            stage=f"Loading {rec.modelLabel}",
            subtext="Preparing target image & visual context...",
        )

        ref_pils = [prepare_pil_image(p, max_side=0) for p in ref_paths]
        target_pil = prepare_pil_image(target_path, max_side=0)
        all_images = [*ref_pils, target_pil]

        ref_notes = [r.note or r.filename for r in rec.references] if ref_pils else []
        prompt_text = build_image_prompt(
            rec.prompt,
            output_mode=rec.outputMode,
            shot_mode=rec.shotMode,
            ref_notes=ref_notes,
            style_note=rec.referenceStyleNote or "",
            has_lora_adapter=has_lora,
        )
        mlx_prompt = build_interleaved_image_messages(
            rec.prompt,
            output_mode=rec.outputMode,
            shot_mode=rec.shotMode,
            ref_notes=ref_notes,
            style_note=rec.referenceStyleNote or "",
            has_lora_adapter=has_lora,
        )

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=52,
            stage=f"Running {entry['short_label']}",
            subtext=f"Inferencing {len(all_images)} visual input(s) on local GPU...",
        )

        raw_text, tel = self._run_model(
            model_id=rec.modelId,
            prompt_text=prompt_text,
            prompt_input=mlx_prompt,
            images=all_images,
            max_tokens=rec.maxTokens,
            temperature=rec.temperature,
            output_mode=rec.outputMode,
            raw_prompt=rec.prompt,
        )

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=88,
            stage="Parsing Normalized Coordinates",
            subtext="Validating 0..1000 bounding boxes & polygons...",
        )

        detections, json_valid = parse_detections(
            raw_text, output_mode=rec.outputMode, id_prefix=f"{rec.id}-det"
        )

        total_latency = round(tel["load_s"] + tel["latency_s"], 3)
        rec.status = "completed"
        rec.latencySec = total_latency
        rec.promptTokens = tel["prompt_tokens"]
        rec.generatedTokens = tel["generation_tokens"]
        rec.peakMemoryGb = tel["peak_memory_gb"]
        rec.detections = detections
        rec.summaryText = (
            f"Detected {len(detections)} object(s) using {entry['short_label']} "
            f"in {total_latency:.2f}s ({'valid JSON' if json_valid else 'fallback parsed'})."
        )
        rec.stageLogs = [
            StageLogItem(
                stage=f"{entry['engine'].upper()} Inference ({rec.shotMode})",
                target=rec.target.filename,
                wall_time_sec=total_latency,
                prompt_tokens=tel["prompt_tokens"],
                generated_tokens=tel["generation_tokens"],
                tokens_per_sec=tel["generation_tps"],
                summary=(
                    f"{len(detections)} detection(s) parsed · target {target_pil.width}x{target_pil.height} · "
                    f"{len(ref_pils)} ref(s) native res · load {tel['load_s']}s · peak {tel['peak_memory_gb']} GB"
                ),
            )
        ]
        if detections:
            rec.rawJson = json.dumps(
                [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "bbox_2d": list(d.bbox_2d),
                        **({"polygon": [list(p) for p in d.polygon]} if d.polygon else {}),
                    }
                    for d in detections
                ],
                indent=2,
            )
        else:
            rec.rawJson = strip_thinking_and_fences(raw_text) or "[]"

        db.upsert_run(rec)
        self._emit_progress(
            rec.id,
            status="completed",
            progress_pct=100,
            stage="Completed",
            subtext=rec.summaryText,
            record=rec,
        )

    def _execute_video_job(
        self,
        rec: RunRecord,
        target_path: Path,
        ref_paths: List[Path],
    ) -> None:
        entry = config.get_model_entry(rec.modelId)
        sampling_fps = rec.samplingFps or 1.0
        output_fps_mode = rec.outputFpsMode or "original"
        lost_buf = rec.byteTrackMemorySec or 3.0

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=12,
            stage="Stage 1/3 · Extracting Video Keyframes",
            subtext=f"Sampling {rec.target.filename} at {sampling_fps} FPS...",
        )

        video_meta, keyframes = extract_video_keyframes(
            target_path, rec.id, sampling_fps=sampling_fps, max_frames=14
        )
        rec.target.width = video_meta["width"]
        rec.target.height = video_meta["height"]
        rec.target.fps = video_meta["fps"]
        rec.target.durationSec = video_meta["duration_sec"]
        rec.target.previewUrl = keyframes[0]["image_url"]

        ref_pils = [prepare_pil_image(p, max_side=640) for p in ref_paths[:4]]
        ref_notes = [r.note or r.filename for r in rec.references[: len(ref_pils)]]

        s1_frame_pils = [prepare_pil_image(kf["stage1_path"], max_side=640) for kf in keyframes]
        s1_prompt = build_video_stage1_prompt(
            rec.prompt,
            keyframes,
            ref_notes=ref_notes,
            style_note=rec.referenceStyleNote or "",
        )

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=28,
            stage=f"Stage 1/3 · Temporal Event Scan ({entry['short_label']})",
            subtext=f"Scanning {len(keyframes)} frame(s) across {video_meta['duration_sec']}s...",
        )

        s1_text, s1_tel = self._run_model(
            model_id=rec.modelId,
            prompt_text=s1_prompt,
            images=[*ref_pils, *s1_frame_pils],
            max_tokens=min(max(rec.maxTokens, 512), 1024),
            temperature=rec.temperature,
        )
        s1_parsed = parse_stage1_events(s1_text, keyframes)
        events_list = s1_parsed["events"]

        stage_logs: List[StageLogItem] = [
            StageLogItem(
                stage="Stage 1: Temporal Event Scan",
                target=f"{len(keyframes)} frames (640x360)",
                wall_time_sec=round(s1_tel["load_s"] + s1_tel["latency_s"], 2),
                prompt_tokens=s1_tel["prompt_tokens"],
                generated_tokens=s1_tel["generation_tokens"],
                tokens_per_sec=s1_tel["generation_tps"],
                summary=f"events={len(events_list)} · {s1_parsed['overall_summary'][:90]}",
            )
        ]

        total_p_tok = s1_tel["prompt_tokens"]
        total_g_tok = s1_tel["generation_tokens"]
        peak_mem = s1_tel["peak_memory_gb"]
        total_vlm_s = s1_tel["load_s"] + s1_tel["latency_s"]

        # Map active events onto keyframes for Stage 2 grounding
        active_kf_map: Dict[int, List[Dict[str, Any]]] = {}
        for ev in events_list:
            for k_num in range(ev["event_start_image_number"], ev["event_end_image_number"] + 1):
                active_kf_map.setdefault(k_num, []).append(ev)

        keyframe_detections_for_render: List[Dict[str, Any]] = []
        dets_by_sample: Dict[int, List[DetectionItem]] = {}
        all_detections_summary: List[DetectionItem] = []

        active_samples = sorted(active_kf_map.keys())
        for idx_s, sample_num in enumerate(active_samples, start=1):
            kf = keyframes[sample_num - 1]
            active_evs = active_kf_map[sample_num]
            pct = 45 + int((idx_s / max(1, len(active_samples))) * 35)
            self._emit_progress(
                rec.id,
                status="running",
                progress_pct=pct,
                stage=f"Stage 2/3 · Grounding Frame #{sample_num} ({idx_s}/{len(active_samples)})",
                subtext=f"Timestamp {kf['timestamp_sec']}s · {len(active_evs)} active event(s)",
            )

            s2_target_pil = prepare_pil_image(kf["stage2_path"], max_side=960)
            s2_prompt = build_video_stage2_prompt(
                rec.prompt,
                output_mode=rec.outputMode,
                active_events=active_evs,
                ref_notes=ref_notes,
                style_note=rec.referenceStyleNote or "",
            )
            s2_text, s2_tel = self._run_model(
                model_id=rec.modelId,
                prompt_text=s2_prompt,
                images=[*ref_pils, s2_target_pil],
                max_tokens=min(rec.maxTokens, 512),
                temperature=rec.temperature,
            )
            total_p_tok += s2_tel["prompt_tokens"]
            total_g_tok += s2_tel["generation_tokens"]
            peak_mem = max(peak_mem, s2_tel["peak_memory_gb"])
            total_vlm_s += s2_tel["latency_s"]

            frame_dets: List[DetectionItem] = []
            for ev in active_evs:
                ev_dets, _ = parse_detections(
                    s2_text,
                    output_mode=rec.outputMode,
                    id_prefix=f"kf{sample_num}-e{ev['event_id']}",
                    event_id=ev["event_id"],
                )
                if not ev_dets and len(active_evs) == 1:
                    ev_dets, _ = parse_detections(
                        s2_text,
                        output_mode=rec.outputMode,
                        id_prefix=f"kf{sample_num}-e{ev['event_id']}",
                    )
                for d in ev_dets:
                    d_labeled = DetectionItem(
                        id=d.id,
                        label=f"{ev['short_label']} ({d.label})",
                        confidence=d.confidence,
                        bbox_2d=d.bbox_2d,
                        polygon=d.polygon,
                    )
                    frame_dets.append(d_labeled)
                    all_detections_summary.append(d_labeled)
                    keyframe_detections_for_render.append(
                        {
                            "sample_number": sample_num,
                            "video_frame_index": kf["frame_idx"],
                            "timestamp_sec": kf["timestamp_sec"],
                            "event_id": ev["event_id"],
                            "label": ev["short_label"],
                            "confidence": d.confidence,
                            "pixel_box": norm_1000_to_pixel_xyxy(
                                d.bbox_2d, video_meta["width"], video_meta["height"]
                            ),
                        }
                    )

            dets_by_sample[sample_num] = frame_dets
            stage_logs.append(
                StageLogItem(
                    stage=f"Stage 2: Grounding #{idx_s}/{len(active_samples)}",
                    target=f"Frame #{sample_num} @ {kf['timestamp_sec']}s",
                    wall_time_sec=round(s2_tel["latency_s"], 2),
                    prompt_tokens=s2_tel["prompt_tokens"],
                    generated_tokens=s2_tel["generation_tokens"],
                    tokens_per_sec=s2_tel["generation_tps"],
                    summary=f"{len(frame_dets)} box(es) grounded",
                )
            )

        self._emit_progress(
            rec.id,
            status="running",
            progress_pct=86,
            stage="Stage 3/3 · ByteTrack + FFmpeg H.264 Encoding",
            subtext=f"Rendering {output_fps_mode} FPS annotated MP4...",
        )

        t_render0 = time.perf_counter()
        sampled_indices = [kf["frame_idx"] for kf in keyframes]
        render_info = render_video_output(
            input_video_path=str(target_path),
            run_id=rec.id,
            filename=rec.target.filename,
            keyframe_detections=keyframe_detections_for_render,
            events=events_list,
            lost_buffer_seconds=lost_buf,
            output_fps_mode=output_fps_mode,
            target_output_fps=sampling_fps if output_fps_mode == "sampled" else None,
            sampled_frame_indices=sampled_indices,
        )
        render_s = round(time.perf_counter() - t_render0, 2)

        stage_logs.append(
            StageLogItem(
                stage="Stage 3: ByteTrack + FFmpeg H.264",
                target=f"{render_info['frames_rendered']} frames ({render_info['output_fps']} FPS)",
                wall_time_sec=render_s,
                prompt_tokens=0,
                generated_tokens=0,
                tokens_per_sec=0.0,
                summary=f"{render_info['total_unique_tracks']} track(s), {render_info['annotated_frames_count']} annotated frames",
            )
        )

        rec.status = "completed"
        rec.latencySec = round(total_vlm_s + render_s, 2)
        rec.promptTokens = total_p_tok
        rec.generatedTokens = total_g_tok
        rec.peakMemoryGb = round(peak_mem, 3)
        rec.detections = all_detections_summary
        rec.annotatedVideoUrl = render_info["annotated_video_url"]
        rec.videoEvents = [
            VideoEventItem(
                event_id=ev["event_id"],
                event_type=ev["event_type"],
                short_label=ev["short_label"],
                trigger_subject=ev["trigger_subject"],
                event_summary=ev["event_summary"],
                event_start_timestamp_sec=ev["event_start_timestamp_sec"],
                event_end_timestamp_sec=ev["event_end_timestamp_sec"],
            )
            for ev in events_list
        ]
        rec.videoKeyframes = [
            VideoKeyframeItem(
                sample_number=kf["sample_number"],
                frame_idx=kf["frame_idx"],
                timestamp_sec=kf["timestamp_sec"],
                imageUrl=kf["image_url"],
                detections=dets_by_sample.get(kf["sample_number"], []),
            )
            for kf in keyframes
        ]
        rec.summaryText = s1_parsed["overall_summary"]
        rec.stageLogs = stage_logs
        rec.rawJson = json.dumps(
            {
                "event_detected": s1_parsed["event_detected"],
                "overall_summary": s1_parsed["overall_summary"],
                "events": events_list,
                "keyframe_detections": keyframe_detections_for_render,
            },
            indent=2,
        )

        db.upsert_run(
            rec,
            internal_meta={
                "target_disk_path": str(target_path),
                "ref_disk_paths": [str(p) for p in ref_paths],
                "keyframe_detections": keyframe_detections_for_render,
                "events": events_list,
                "sampled_frame_indices": sampled_indices,
            },
        )
        self._emit_progress(
            rec.id,
            status="completed",
            progress_pct=100,
            stage="Completed",
            subtext=rec.summaryText,
            record=rec,
        )


job_manager = JobManager()
