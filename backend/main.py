"""Control One Local FastAPI Server for MLX Qwen3-VL + LoRA & Optional Ollama 8B."""
from __future__ import annotations

import asyncio
import json
import platform
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from auto_annotate.manager import auto_annotate_manager
from auto_annotate.pytorch_engine import pytorch_engine
from auto_annotate.router import router as auto_annotate_router

import config
import db
from engines.florence_engine import florence_engine
from engines.mlx_lora_engine import mlx_engine
from engines.ollama_engine import ollama_engine
from schemas import (
    HealthResponse,
    ModelOptionId,
    ModelStatusItem,
    OllamaControlResponse,
    OutputMode,
    ReencodeRequest,
    ReferencePreview,
    RunRecord,
    RunSubmitResponse,
    ShotMode,
    TargetMediaMeta,
    MediaType,
)
from services.job_manager import job_manager
from services.media_service import (
    cleanup_run_artifacts,
    inspect_image_file,
    render_video_output,
)


SERVER_BOOT_ID = uuid.uuid4().hex


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    try:
        yield
    finally:
        auto_annotate_manager.clear_memory_cache()
        pytorch_engine.reset_visual_query()


app = FastAPI(
    title="CONTROL ONE — Qwen3-VL Local Workbench API",
    version="0.2.0",
    description="Local Apple Silicon MLX-VLM (+ LoRA) and optional Ollama 8B evaluation API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auto_annotate_router)


def _safe_filename(name: Optional[str], fallback: str) -> str:
    base = Path(name or fallback).name.strip()
    return base or fallback


def _runtime_device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    if mlx_engine.is_mlx_available():
        return f"Apple Silicon MLX ({platform.machine()})"
    return f"CPU ({platform.system()} {platform.machine()})"


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok, ollama_models = ollama_engine.check_health()
    loaded_id = mlx_engine.loaded_model_id or florence_engine.loaded_model_id
    return HealthResponse(
        status="ok",
        bootId=SERVER_BOOT_ID,
        device=_runtime_device_label(),
        loadedModelId=loaded_id,
        queueDepth=job_manager.queue_depth,
        ollamaReachable=ollama_ok,
        ollamaModels=ollama_models,
        adapterBestExists=(config.ADAPTER_BEST_DIR / "adapters.safetensors").is_file(),
    )


@app.get("/api/v1/models", response_model=List[ModelStatusItem])
def list_models() -> List[ModelStatusItem]:
    ollama_ok, ollama_models = ollama_engine.check_health()
    mlx_ok = mlx_engine.is_mlx_available()
    items: List[ModelStatusItem] = []
    for mid, entry in config.MODEL_REGISTRY.items():
        if entry["engine"] == "mlx":
            cached = mlx_engine.is_hf_model_cached(entry["model_path"])
            adapter_ok = config.is_adapter_present(entry)
            available = mlx_ok and adapter_ok
            if not mlx_ok:
                note = "Apple Silicon MLX only (unavailable on this host)"
            elif not adapter_ok:
                note = "Adapter missing (adapters.safetensors not found)"
            elif cached:
                note = "Cached locally"
            else:
                note = "Downloads from HF Hub on first run"
            engine_label = "MLX-VLM"
            loaded = mlx_engine.loaded_model_id == mid
        elif entry["engine"] == "florence2":
            cached = florence_engine.is_hf_model_cached(entry["model_path"])
            available = True
            note = (
                "Cached locally"
                if cached
                else f"Downloads HF Hub weights on first run ({entry['model_path']})"
            )
            engine_label = "Florence-2 MPS"
            loaded = florence_engine.loaded_model_id == mid
        else:
            available = ollama_ok and any(entry["model_path"] in m for m in ollama_models)
            note = (
                "Ollama tag ready"
                if available
                else ("Ollama unreachable" if not ollama_ok else f"Pull {entry['model_path']} first")
            )
            engine_label = "Ollama"
            loaded = False
        items.append(
            ModelStatusItem(
                id=mid,
                label=entry["label"],
                shortLabel=entry["short_label"],
                engine=engine_label,
                quantization=entry["quantization"],
                adapterLabel=entry.get("adapter_label"),
                available=available,
                loaded=loaded,
                note=note,
            )
        )
    return items


@app.post("/api/v1/ollama/start", response_model=OllamaControlResponse)
def start_ollama() -> OllamaControlResponse:
    ok, models, msg = ollama_engine.start_server()
    reachable, fresh_models = ollama_engine.check_health()
    return OllamaControlResponse(
        ok=ok,
        ollamaReachable=reachable,
        ollamaModels=fresh_models or models,
        message=msg,
    )


@app.post("/api/v1/ollama/stop", response_model=OllamaControlResponse)
def stop_ollama() -> OllamaControlResponse:
    ok, msg = ollama_engine.stop_server()
    reachable, fresh_models = ollama_engine.check_health()
    return OllamaControlResponse(
        ok=ok and not reachable,
        ollamaReachable=reachable,
        ollamaModels=fresh_models,
        message=msg,
    )


@app.post("/api/v1/runs", response_model=RunSubmitResponse)
async def create_run(
    prompt: str = Form(...),
    mediaType: MediaType = Form(...),
    shotMode: ShotMode = Form(...),
    outputMode: OutputMode = Form(...),
    modelId: ModelOptionId = Form(...),
    temperature: float = Form(0.0),
    maxTokens: int = Form(1024),
    samplingFps: float = Form(1.0),
    outputFpsMode: str = Form("original"),
    byteTrackMemorySec: float = Form(3.0),
    referenceStyleNote: str = Form(""),
    refNotesJson: str = Form("[]"),
    target_file: UploadFile = File(...),
    ref_files: List[UploadFile] = File(default=[]),
) -> RunSubmitResponse:
    db.init_db()
    try:
        entry = config.get_model_entry(modelId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if entry.get("engine") == "ollama" and not ollama_engine.check_health()[0]:
        raise HTTPException(
            status_code=400,
            detail="Ollama server is turned off (unreachable). Turn Ollama ON in the sidebar first.",
        )

    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    run_id = f"run-{local_now.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"

    target_name = _safe_filename(
        target_file.filename, "target.mp4" if mediaType == "video" else "target.jpg"
    )
    target_save_name = f"{run_id}_{target_name}"
    target_disk_path = config.TARGETS_DIR / target_save_name
    target_bytes = await target_file.read()
    if not target_bytes:
        raise HTTPException(status_code=400, detail="Uploaded target_file is empty.")
    target_disk_path.write_bytes(target_bytes)

    if mediaType == "image":
        try:
            img_info = inspect_image_file(target_disk_path)
        except Exception as exc:
            target_disk_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc
        target_meta = TargetMediaMeta(
            filename=target_name,
            mediaType="image",
            width=img_info["width"],
            height=img_info["height"],
            sizeBytes=img_info["size_bytes"],
            previewUrl=f"/api/v1/files/targets/{target_save_name}",
        )
    else:
        import cv2

        cap = cv2.VideoCapture(str(target_disk_path))
        if not cap.isOpened():
            target_disk_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="OpenCV could not open uploaded video.")
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        vfps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0) or 24.0
        vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        target_meta = TargetMediaMeta(
            filename=target_name,
            mediaType="video",
            width=vw,
            height=vh,
            sizeBytes=len(target_bytes),
            fps=round(vfps, 2),
            durationSec=round(vframes / vfps, 2) if vframes > 0 else 0.0,
            previewUrl=f"/api/v1/files/targets/{target_save_name}",
        )

    # Strictly enforce Zero-Shot isolation on the backend too
    ref_disk_paths: List[str] = []
    ref_previews: List[ReferencePreview] = []
    if shotMode == "few-shot" and ref_files:
        try:
            parsed_notes = json.loads(refNotesJson)
            if not isinstance(parsed_notes, list):
                parsed_notes = []
        except json.JSONDecodeError:
            parsed_notes = []

        run_ref_dir = config.REFERENCES_DIR / run_id
        run_ref_dir.mkdir(parents=True, exist_ok=True)

        for idx, rf in enumerate(ref_files):
            rf_bytes = await rf.read()
            if not rf_bytes:
                continue
            rf_name = _safe_filename(rf.filename, f"ref_{idx + 1}.jpg")
            rf_save = f"{idx + 1:02d}_{rf_name}"
            rf_path = run_ref_dir / rf_save
            rf_path.write_bytes(rf_bytes)
            ref_disk_paths.append(str(rf_path))

            note_str = str(parsed_notes[idx]).strip() if idx < len(parsed_notes) else ""
            is_vid_frame = rf_name.startswith("vid_frame_")
            ref_previews.append(
                ReferencePreview(
                    id=f"{run_id}-ref-{idx + 1}",
                    kind="video_frame" if is_vid_frame else "image",
                    filename=rf_name,
                    imageUrl=f"/api/v1/files/references/{run_id}/{rf_save}",
                    note=note_str or None,
                )
            )

    record = RunRecord(
        id=run_id,
        createdAtIso=now.isoformat(),
        createdAtDisplay=local_now.strftime("%b %d, %H:%M"),
        status="queued",
        isMockPreview=False,
        target=target_meta,
        shotMode=shotMode,
        outputMode=outputMode,
        modelId=modelId,
        modelLabel=entry["label"],
        adapterLabel=entry.get("adapter_label"),
        prompt=prompt.strip(),
        temperature=float(temperature),
        maxTokens=int(maxTokens),
        samplingFps=float(samplingFps) if mediaType == "video" else None,
        outputFpsMode="sampled" if outputFpsMode == "sampled" else ("original" if mediaType == "video" else None),
        byteTrackMemorySec=float(byteTrackMemorySec) if mediaType == "video" else None,
        references=ref_previews,
        referenceStyleNote=referenceStyleNote.strip() if shotMode == "few-shot" and referenceStyleNote.strip() else None,
        summaryText="Queued for local execution...",
    )

    job_manager.submit_job(
        record=record,
        target_disk_path=str(target_disk_path),
        ref_disk_paths=ref_disk_paths,
    )
    return RunSubmitResponse(runId=run_id, status="queued", record=record)


@app.get("/api/v1/runs", response_model=List[RunRecord])
def get_runs() -> List[RunRecord]:
    db.init_db()
    return db.list_runs()


@app.get("/api/v1/runs/{run_id}", response_model=RunRecord)
def get_single_run(run_id: str) -> RunRecord:
    db.init_db()
    rec = db.get_run(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return rec


@app.get("/api/v1/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    db.init_db()
    if not db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        sub = job_manager.subscribe(run_id)
        try:
            snap = job_manager.get_progress(run_id)
            yield f"data: {json.dumps(snap)}\n\n"
            if snap.get("status") in ("completed", "failed"):
                return
            while True:
                try:
                    msg = await asyncio.to_thread(sub.get, True, 15.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("status") in ("completed", "failed"):
                        break
                except Exception:
                    yield ": keep-alive\n\n"
        finally:
            job_manager.unsubscribe(run_id, sub)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/v1/runs/{run_id}")
def delete_single_run(run_id: str) -> dict:
    db.init_db()
    rec = db.get_run(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    cleanup_run_artifacts(run_id, rec.target.previewUrl, rec.annotatedVideoUrl)
    db.delete_run(run_id)
    return {"deleted": True, "runId": run_id}


@app.post("/api/v1/runs/{run_id}/reencode", response_model=RunRecord)
def reencode_run(run_id: str, body: ReencodeRequest) -> RunRecord:
    db.init_db()
    rec = db.get_run(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if rec.target.mediaType != "video":
        raise HTTPException(status_code=400, detail="Re-encoding applies only to video runs.")

    meta = db.get_internal_meta(run_id)
    target_path = meta.get("target_disk_path")
    if not target_path or not Path(target_path).is_file():
        raise HTTPException(status_code=404, detail="Original target video file missing on disk.")

    lost_buf = body.byteTrackMemorySec if body.byteTrackMemorySec is not None else (rec.byteTrackMemorySec or 3.0)
    sampling_fps = rec.samplingFps or 1.0

    render_info = render_video_output(
        input_video_path=target_path,
        run_id=rec.id,
        filename=rec.target.filename,
        keyframe_detections=meta.get("keyframe_detections", []),
        events=meta.get("events", []),
        lost_buffer_seconds=lost_buf,
        output_fps_mode=body.outputFpsMode,
        target_output_fps=sampling_fps if body.outputFpsMode == "sampled" else None,
        sampled_frame_indices=meta.get("sampled_frame_indices", []),
    )

    rec.outputFpsMode = body.outputFpsMode
    rec.byteTrackMemorySec = lost_buf
    rec.annotatedVideoUrl = render_info["annotated_video_url"]
    db.upsert_run(rec)
    return rec


@app.get("/api/v1/files/{category}/{filepath:path}")
def serve_file(category: str, filepath: str) -> FileResponse:
    category_map = {
        "targets": config.TARGETS_DIR,
        "references": config.REFERENCES_DIR,
        "outputs": config.OUTPUTS_DIR,
    }
    base_dir = category_map.get(category)
    if base_dir is None:
        raise HTTPException(status_code=404, detail="Unknown file category.")

    base_resolved = base_dir.resolve()
    target = (base_resolved / filepath).resolve()
    if not target.is_relative_to(base_resolved):
        raise HTTPException(status_code=403, detail="Invalid file path.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)
