"""Self-contained FastAPI APIRouter for Roboflow Batch Auto-Annotation & Verification."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import config as base_config
from auto_annotate import config as aa_config
from auto_annotate import db as aa_db
from auto_annotate import roboflow_service as rf_service
from auto_annotate.manager import auto_annotate_manager
from auto_annotate.pytorch_engine import pytorch_engine
from auto_annotate.schemas import (
    AnnotationStatusFilter,
    ApproveBatchRequest,
    AutoDetection,
    AutoItemRecord,
    AutoJobDetailResponse,
    AutoJobRecord,
    AutoModelCatalogResponse,
    ClassColorEntry,
    FetchScope,
    ManualFetchJobRequest,
    ManualFindObjectsRequest,
    ManualFindObjectsResponse,
    ManualSmartPrepareRequest,
    ManualSmartPrepareResponse,
    ManualSmartSelectRequest,
    ManualSmartSelectResponse,
    OutputMode,
    PushAcceptedRequest,
    PushAcceptedResponse,
    ReviewItemPatchRequest,
    RoboflowInspectRequest,
    RoboflowInspectResponse,
    ShotMode,
    VisualEmbedAddRequest,
    VisualEmbedBatchSearchRequest,
    VisualEmbedBuildRefsRequest,
    VisualEmbedBuildRefsResponse,
    VisualEmbedResetRequest,
    VisualEmbedResponse,
    VisualEmbedSearchRequest,
    VisualEmbedUndoRedoRequest,
)
from engines.florence_engine import florence_engine
from engines.mlx_lora_engine import mlx_engine
from engines.ollama_engine import ollama_engine
from services.schema_extractor import extract_prompt_schema, is_placeholder_classes

aa_db.init_auto_db()

router = APIRouter(prefix="/api/v1/auto-annotate", tags=["auto-annotate"])

_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID.")


@router.get("/models", response_model=AutoModelCatalogResponse)
def list_auto_models() -> AutoModelCatalogResponse:
    ollama_ok, ollama_models = ollama_engine.check_health()
    mlx_ok = mlx_engine.is_mlx_available()
    detectors: List[Dict[str, Any]] = []

    for mid, entry in base_config.MODEL_REGISTRY.items():
        if entry["engine"] == "mlx":
            cached = mlx_engine.is_hf_model_cached(entry["model_path"])
            adapter_ok = base_config.is_adapter_present(entry)
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
            supports_few_shot = True
        elif entry["engine"] == "florence2":
            cached = florence_engine.is_hf_model_cached(entry["model_path"])
            available = True
            note = (
                "Cached locally"
                if cached
                else f"Downloads HF Hub weights on first run ({entry['model_path']})"
            )
            engine_label = "Florence-2 MPS"
            supports_few_shot = False
        else:
            available = ollama_ok and any(
                entry["model_path"] in m for m in ollama_models
            )
            note = (
                "Ollama tag ready"
                if available
                else (
                    "Ollama unreachable"
                    if not ollama_ok
                    else f"Pull {entry['model_path']} first"
                )
            )
            engine_label = "Ollama"
            supports_few_shot = True

        detectors.append(
            {
                "id": mid,
                "label": entry["label"],
                "shortLabel": entry["short_label"],
                "engine": engine_label,
                "quantization": entry["quantization"],
                "available": available,
                "supportsPolygon": True,
                "supportsFewShot": supports_few_shot,
                "note": note,
            }
        )

    for mid, entry in aa_config.PYTORCH_DETECTORS.items():
        if entry.get("engine") == "owlv2":
            cached = pytorch_engine.is_hf_model_cached(entry["model_path"])
            detectors.append(
                {
                    "id": mid,
                    "label": entry["label"],
                    "shortLabel": entry["short_label"],
                    "engine": "OWLv2 PyTorch MPS",
                    "quantization": entry["quantization"],
                    "available": True,
                    "supportsPolygon": bool(entry.get("supports_polygon", True)),
                    "supportsFewShot": bool(entry.get("supports_few_shot", False)),
                    "note": (
                        "Cached locally"
                        if cached
                        else f"Downloads HF Hub weights ({entry.get('approx_size', 'HF')})"
                    ),
                }
            )
            continue

        w_path = Path(entry["weight_path"])
        available = w_path.is_file()
        detectors.append(
            {
                "id": mid,
                "label": entry["label"],
                "shortLabel": entry["short_label"],
                "engine": "PyTorch MPS",
                "quantization": entry["quantization"],
                "available": available,
                "supportsPolygon": bool(entry.get("supports_polygon", True)),
                "supportsFewShot": bool(entry.get("supports_few_shot", False)),
                "note": "Ready (.pt local weight)" if available else f"Missing {w_path.name}",
            }
        )

    owlv2_models: List[Dict[str, Any]] = []
    for hf_id, oentry in aa_config.OWLV2_MODELS.items():
        cached = pytorch_engine.is_hf_model_cached(hf_id)
        owlv2_models.append(
            {
                "id": oentry["id"],
                "hfModelId": hf_id,
                "label": oentry["label"],
                "shortLabel": oentry["short_label"],
                "quantization": oentry["quantization"],
                "approxSize": oentry["approx_size"],
                "cached": cached,
                "available": True,
                "note": (
                    "Cached locally"
                    if cached
                    else f"Downloads HF Hub ({oentry['approx_size']})"
                ),
                "description": oentry["description"],
            }
        )

    sam_refiners: List[Dict[str, Any]] = []
    for rid, entry in aa_config.SAM_REFINERS.items():
        w_path = entry.get("weight_path")
        available = True if w_path is None else Path(w_path).is_file()
        sam_refiners.append(
            {
                "id": rid,
                "label": entry["label"],
                "available": available,
                "description": entry["description"],
            }
        )

    return AutoModelCatalogResponse(
        detectors=detectors,
        samRefiners=sam_refiners,
        owlv2Models=owlv2_models,
    )


@router.post("/roboflow/inspect", response_model=RoboflowInspectResponse)
def inspect_roboflow_project(body: RoboflowInspectRequest) -> RoboflowInspectResponse:
    if not body.apiKey.strip() or not body.projectSlug.strip():
        raise HTTPException(
            status_code=400, detail="Roboflow API key and project slug are required."
        )
    try:
        return rf_service.inspect_project(
            api_key=body.apiKey,
            project_slug=body.projectSlug,
            workspace_slug=body.workspaceSlug,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Roboflow connect failed: {exc}") from exc


@router.post("/jobs", response_model=AutoJobRecord)
async def create_auto_job(
    apiKey: str = Form(""),
    projectSlug: str = Form(""),
    workspaceSlug: str = Form(""),
    fetchScope: FetchScope = Form("unannotated"),
    annotationStatusFilter: AnnotationStatusFilter = Form("all"),
    sourceBatchId: str = Form(""),
    sourceAnnotationJobId: str = Form(""),
    assigneeEmail: str = Form(""),
    sourceJobId: str = Form(""),
    sourceItemIdsJson: str = Form("[]"),
    maxImages: int = Form(100),
    batchSize: int = Form(aa_config.DEFAULT_BATCH_SIZE),
    modelId: str = Form(...),
    samRefinerId: str = Form("none"),
    shotMode: ShotMode = Form("zero-shot"),
    outputMode: OutputMode = Form("bbox"),
    prompt: str = Form(...),
    styleNote: str = Form(""),
    classesJson: str = Form("[]"),
    classColorsJson: str = Form("[]"),
    confidenceThreshold: float = Form(0.25),
    temperature: float = Form(0.0),
    maxTokens: int = Form(1024),
    refNotesJson: str = Form("[]"),
    ref_files: List[UploadFile] = File(default=[]),
) -> AutoJobRecord:
    aa_db.init_auto_db()
    clean_source_job_id = sourceJobId.strip()
    is_retry = bool(clean_source_job_id)

    if not is_retry and (not apiKey.strip() or not projectSlug.strip()):
        raise HTTPException(
            status_code=400, detail="Roboflow API key and project slug are required."
        )
    try:
        detector = aa_config.get_detector_entry(modelId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if detector.get("engine") == "ollama" and not ollama_engine.check_health()[0]:
        raise HTTPException(
            status_code=400,
            detail="Ollama server is turned off (unreachable). Turn Ollama ON in the sidebar first.",
        )

    if samRefinerId not in aa_config.SAM_REFINERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown SAM refiner '{samRefinerId}'."
        )

    try:
        custom_classes = [
            str(c).strip() for c in json.loads(classesJson or "[]") if str(c).strip()
        ]
    except Exception:
        custom_classes = []

    try:
        raw_cc = json.loads(classColorsJson or "[]")
        custom_colors = [
            ClassColorEntry.model_validate(c) for c in raw_cc if isinstance(c, dict)
        ]
    except Exception:
        custom_colors = []

    try:
        parsed_ref_notes = json.loads(refNotesJson or "[]")
        if not isinstance(parsed_ref_notes, list):
            parsed_ref_notes = []
    except Exception:
        parsed_ref_notes = []

    try:
        selected_retry_ids = [
            str(i).strip() for i in json.loads(sourceItemIdsJson or "[]") if str(i).strip()
        ]
    except Exception:
        selected_retry_ids = []

    src_job: Optional[AutoJobRecord] = None
    if is_retry:
        _validate_job_id(clean_source_job_id)
        src_job = aa_db.get_job(clean_source_job_id)
        if not src_job:
            raise HTTPException(
                status_code=404,
                detail=f"Source job '{clean_source_job_id}' not found.",
            )

    had_placeholder_only = is_placeholder_classes(custom_classes)
    prompt_schema = extract_prompt_schema(prompt.strip())
    prompt_classes_applied = False
    if had_placeholder_only and prompt_schema.targets:
        custom_classes = list(prompt_schema.targets)
        prompt_classes_applied = True
        first_color = (
            custom_colors[0].color
            if custom_colors
            else aa_config.DEFAULT_CLASS_COLORS[0]
        )
        custom_colors = [
            ClassColorEntry(
                name=cname,
                color=(
                    first_color
                    if idx == 0
                    else aa_config.DEFAULT_CLASS_COLORS[
                        idx % len(aa_config.DEFAULT_CLASS_COLORS)
                    ]
                ),
            )
            for idx, cname in enumerate(custom_classes)
        ]

    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    job_id = f"aa-{local_now.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    job_dir = aa_config.AUTO_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = job_dir / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    # Save optional few-shot reference images locally
    ref_disk_paths: List[str] = []
    ref_notes: List[str] = []
    if shotMode == "few-shot" and ref_files and detector.get("supports_few_shot", True):
        ref_dir = job_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for idx, rf_up in enumerate(ref_files):
            raw_bytes = await rf_up.read()
            if not raw_bytes:
                continue
            rname = Path(rf_up.filename or f"ref_{idx + 1}.jpg").name
            rpath = ref_dir / f"{idx + 1:02d}_{rname}"
            rpath.write_bytes(raw_bytes)
            ref_disk_paths.append(str(rpath))
            note = str(parsed_ref_notes[idx]).strip() if idx < len(parsed_ref_notes) else ""
            ref_notes.append(note or rname)

    def _merge_job_classes(
        job_rec: AutoJobRecord,
        pulled_classes: List[str],
        pulled_colors: List[ClassColorEntry],
        pulled_project_type: str,
    ) -> None:
        if had_placeholder_only and not prompt_classes_applied and pulled_classes:
            merged_classes = list(pulled_classes)
        else:
            merged_classes = (
                list(custom_classes) if custom_classes else list(pulled_classes)
            )
        merged_lower = {c.lower() for c in merged_classes}

        color_by_lower: Dict[str, ClassColorEntry] = {
            c.name.lower(): c
            for c in custom_colors
            if c.name.strip()
            and not (
                c.name.strip().lower() == "object"
                and "object" not in merged_lower
                and merged_lower
            )
        }
        for pc in pulled_colors:
            key = pc.name.lower()
            if key not in color_by_lower and (not merged_lower or key in merged_lower):
                color_by_lower[key] = pc
        palette = aa_config.DEFAULT_CLASS_COLORS
        for idx, cname in enumerate(merged_classes):
            if cname.lower() not in color_by_lower:
                color_by_lower[cname.lower()] = ClassColorEntry(
                    name=cname, color=palette[idx % len(palette)]
                )

        if not custom_classes or had_placeholder_only:
            job_rec.classes = merged_classes
        job_rec.classColors = list(color_by_lower.values())
        job_rec.projectType = (
            "instance-segmentation"
            if outputMode in ("polygon", "both")
            else pulled_project_type
        )

    safe_batch_size = max(1, min(int(batchSize or 50), 200))

    if is_retry and src_job is not None:
        src_with_paths = aa_db.get_all_items_with_paths(src_job.id)
        try:
            cloned = rf_service.clone_items_for_retry_job(
                source_job=src_job,
                source_items_with_paths=src_with_paths,
                target_job_id=job_id,
                target_job_dir=job_dir,
                selected_item_ids=selected_retry_ids or None,
                batch_size=safe_batch_size,
            )
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        record = AutoJobRecord(
            id=job_id,
            createdAtIso=now.isoformat(),
            createdAtDisplay=local_now.strftime("%b %d, %H:%M"),
            workspaceSlug=src_job.workspaceSlug,
            projectSlug=src_job.projectSlug,
            projectName=src_job.projectName,
            projectType="instance-segmentation"
            if outputMode in ("polygon", "both")
            else src_job.projectType,
            classes=custom_classes,
            classColors=custom_colors,
            fetchScope="rejected_retry",
            parentJobId=src_job.id,
            sourceBatchId=src_job.sourceBatchId,
            sourceAnnotationJobId=src_job.sourceAnnotationJobId,
            assigneeEmail=src_job.assigneeEmail,
            maxImages=len(cloned["items"]),
            batchSize=safe_batch_size,
            modelId=modelId,
            modelLabel=detector["label"],
            samRefinerId=samRefinerId,
            shotMode=shotMode if detector.get("supports_few_shot", True) else "zero-shot",
            outputMode=outputMode,
            prompt=prompt.strip(),
            styleNote=styleNote.strip() or None,
            confidenceThreshold=float(confidenceThreshold),
            temperature=float(temperature),
            maxTokens=int(maxTokens),
            status="queued",
            localAcceptedDir=str(accepted_dir),
        )
        _merge_job_classes(
            record,
            cloned["classes"],
            cloned["classColors"],
            cloned["projectType"],
        )
        aa_db.upsert_job(
            record,
            internal_meta={
                "ref_disk_paths": ref_disk_paths,
                "ref_notes": ref_notes,
            },
        )
        aa_db.insert_batches(job_id, cloned["batches"])
        aa_db.insert_items(cloned["items"])
        auto_annotate_manager.enqueue(job_id)
        return aa_db.get_job(job_id) or record

    record = AutoJobRecord(
        id=job_id,
        createdAtIso=now.isoformat(),
        createdAtDisplay=local_now.strftime("%b %d, %H:%M"),
        workspaceSlug=workspaceSlug.strip(),
        projectSlug=projectSlug.strip(),
        projectName=projectSlug.strip(),
        projectType="instance-segmentation"
        if outputMode in ("polygon", "both")
        else "object-detection",
        classes=custom_classes,
        classColors=custom_colors,
        fetchScope=fetchScope,
        annotationStatusFilter=annotationStatusFilter,
        sourceBatchId=sourceBatchId.strip() or None,
        sourceAnnotationJobId=sourceAnnotationJobId.strip() or None,
        assigneeEmail=assigneeEmail.strip() or None,
        maxImages=max(0, int(maxImages)),
        batchSize=safe_batch_size,
        modelId=modelId,
        modelLabel=detector["label"],
        samRefinerId=samRefinerId,
        shotMode=shotMode if detector.get("supports_few_shot", True) else "zero-shot",
        outputMode=outputMode,
        prompt=prompt.strip(),
        styleNote=styleNote.strip() or None,
        confidenceThreshold=float(confidenceThreshold),
        temperature=float(temperature),
        maxTokens=int(maxTokens),
        status="fetching",
        localAcceptedDir=str(accepted_dir),
    )

    aa_db.upsert_job(
        record,
        internal_meta={
            "ref_disk_paths": ref_disk_paths,
            "ref_notes": ref_notes,
        },
    )
    auto_annotate_manager.emit(
        job_id,
        status="fetching",
        progress_pct=0,
        stage="Connecting to Roboflow...",
        subtext=f"Fetching images from {projectSlug.strip()}",
        job=record,
    )

    # Keep API key strictly in memory during this background fetch thread (never saved to DB)
    raw_api_key = apiKey.strip()

    def _fetch_then_queue() -> None:
        try:
            def _on_dl(done: int, total: int, fname: str) -> None:
                pct = int((done / max(1, total)) * 100)
                auto_annotate_manager.emit(
                    job_id,
                    status="fetching",
                    progress_pct=pct,
                    stage=f"Fetching Roboflow Images ({done} / {total})",
                    subtext=fname,
                )

            pulled = rf_service.fetch_project_images(
                api_key=raw_api_key,
                workspace_slug=workspaceSlug.strip() or None,
                project_slug=projectSlug.strip(),
                job_id=job_id,
                job_dir=job_dir,
                fetch_scope=fetchScope,
                annotation_status_filter=annotationStatusFilter,
                source_batch_id=sourceBatchId.strip() or None,
                source_annotation_job_id=sourceAnnotationJobId.strip() or None,
                assignee_email=assigneeEmail.strip() or None,
                max_images=max(0, int(maxImages)),
                batch_size=safe_batch_size,
                progress_cb=_on_dl,
            )

            job_now = aa_db.get_job(job_id)
            if not job_now or job_now.status == "cancelled":
                return

            job_now.workspaceSlug = pulled["workspaceSlug"]
            job_now.projectName = pulled["projectName"]
            _merge_job_classes(
                job_now,
                pulled["classes"],
                pulled["classColors"],
                pulled["projectType"],
            )
            aa_db.upsert_job(job_now)
            aa_db.insert_batches(job_id, pulled["batches"])
            aa_db.insert_items(pulled["items"])
            auto_annotate_manager.enqueue(job_id)
        except Exception as exc:
            print(f"[auto-annotate] fetch failed for {job_id}: {exc}", flush=True)
            err_job = aa_db.get_job(job_id)
            if err_job:
                err_job.status = "failed"
                err_job.errorMessage = str(exc)
                aa_db.upsert_job(err_job)
                auto_annotate_manager.emit(
                    job_id,
                    status="failed",
                    progress_pct=100,
                    stage="Roboflow Fetch Failed",
                    subtext=str(exc),
                    job=err_job,
                )

    threading.Thread(target=_fetch_then_queue, daemon=True).start()
    return record


@router.post("/jobs/fetch-only", response_model=AutoJobRecord)
def create_manual_fetch_job(body: ManualFetchJobRequest) -> AutoJobRecord:
    """Fetch images from Roboflow directly into a local manual-ready job without VLM auto-inference."""
    aa_db.init_auto_db()
    clean_key = body.apiKey.strip()
    clean_proj = body.projectSlug.strip()
    if not clean_key or not clean_proj:
        raise HTTPException(
            status_code=400, detail="Roboflow API key and project slug are required."
        )

    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    job_id = f"aa-{local_now.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    job_dir = aa_config.AUTO_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = job_dir / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    safe_batch_size = max(1, min(int(body.batchSize or aa_config.DEFAULT_BATCH_SIZE), 200))
    safe_max = max(0, int(body.maxImages))
    clean_ws = body.workspaceSlug.strip()

    record = AutoJobRecord(
        id=job_id,
        createdAtIso=now.isoformat(),
        createdAtDisplay=local_now.strftime("%b %d, %H:%M"),
        workspaceSlug=clean_ws,
        projectSlug=clean_proj,
        projectName=clean_proj,
        projectType="object-detection",
        classes=[],
        classColors=[],
        fetchScope=body.fetchScope,
        annotationStatusFilter=body.annotationStatusFilter,
        sourceBatchId=body.sourceBatchId.strip() or None,
        sourceAnnotationJobId=body.sourceAnnotationJobId.strip() or None,
        assigneeEmail=body.assigneeEmail.strip() or None,
        maxImages=safe_max,
        batchSize=safe_batch_size,
        modelId="manual-only",
        modelLabel="Manual / Smart Select",
        samRefinerId="sam2.1-s",
        shotMode="zero-shot",
        outputMode="bbox",
        prompt="Manual & Smart Select dataset fetch",
        styleNote=None,
        confidenceThreshold=0.25,
        temperature=0.0,
        maxTokens=1024,
        status="fetching",
        localAcceptedDir=str(accepted_dir),
    )

    aa_db.upsert_job(record, internal_meta={})
    auto_annotate_manager.emit(
        job_id,
        status="fetching",
        progress_pct=0,
        stage="Connecting to Roboflow...",
        subtext=f"Fetching images from {clean_proj}",
        job=record,
    )

    def _fetch_only_worker() -> None:
        try:
            def _on_dl(done: int, total: int, fname: str) -> None:
                pct = int((done / max(1, total)) * 100)
                auto_annotate_manager.emit(
                    job_id,
                    status="fetching",
                    progress_pct=pct,
                    stage=f"Fetching Roboflow Images ({done} / {total})",
                    subtext=fname,
                )

            pulled = rf_service.fetch_project_images(
                api_key=clean_key,
                workspace_slug=clean_ws or None,
                project_slug=clean_proj,
                job_id=job_id,
                job_dir=job_dir,
                fetch_scope=body.fetchScope,
                annotation_status_filter=body.annotationStatusFilter,
                source_batch_id=body.sourceBatchId.strip() or None,
                source_annotation_job_id=body.sourceAnnotationJobId.strip() or None,
                assignee_email=body.assigneeEmail.strip() or None,
                max_images=safe_max,
                batch_size=safe_batch_size,
                progress_cb=_on_dl,
            )

            job_now = aa_db.get_job(job_id)
            if not job_now or job_now.status == "cancelled":
                return

            job_now.workspaceSlug = pulled["workspaceSlug"]
            job_now.projectSlug = pulled["projectSlug"]
            job_now.projectName = pulled["projectName"]
            job_now.projectType = pulled["projectType"]
            job_now.classes = list(pulled.get("classes") or [])
            job_now.classColors = list(pulled.get("classColors") or [])
            job_now.outputMode = (
                "both"
                if pulled.get("projectType") == "instance-segmentation"
                else "bbox"
            )
            job_now.status = "completed"
            job_now.errorMessage = None
            aa_db.upsert_job(job_now)
            aa_db.insert_batches(job_id, pulled["batches"])
            aa_db.insert_items(pulled["items"])
            hydrated = aa_db.get_job(job_id) or job_now
            auto_annotate_manager.emit(
                job_id,
                status="completed",
                progress_pct=100,
                stage=f"Fetched {hydrated.totalImages} images",
                subtext="Ready in Manual Annotator",
                job=hydrated,
            )
        except Exception as exc:
            print(f"[auto-annotate] manual fetch failed for {job_id}: {exc}", flush=True)
            err_job = aa_db.get_job(job_id)
            if err_job:
                err_job.status = "failed"
                err_job.errorMessage = str(exc)
                aa_db.upsert_job(err_job)
                auto_annotate_manager.emit(
                    job_id,
                    status="failed",
                    progress_pct=100,
                    stage="Roboflow Fetch Failed",
                    subtext=str(exc),
                    job=err_job,
                )

    threading.Thread(target=_fetch_only_worker, daemon=True).start()
    return record


@router.get("/jobs", response_model=List[AutoJobRecord])
def list_auto_jobs() -> List[AutoJobRecord]:
    aa_db.init_auto_db()
    return aa_db.list_jobs()


@router.get("/jobs/{job_id}", response_model=AutoJobDetailResponse)
def get_auto_job_detail(
    job_id: str,
    batch_id: Optional[str] = None,
    review_status: Optional[str] = None,
    reject_reason: Optional[str] = None,
    detection_filter: Optional[str] = None,
) -> AutoJobDetailResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    items = aa_db.list_items(
        job_id,
        batch_id=batch_id,
        review_status=review_status,
        reject_reason=reject_reason,
        detection_filter=detection_filter,
    )
    return AutoJobDetailResponse(job=job, items=items)


@router.get("/jobs/{job_id}/items", response_model=List[AutoItemRecord])
def get_auto_job_items(
    job_id: str,
    batch_id: Optional[str] = None,
    review_status: Optional[str] = None,
    reject_reason: Optional[str] = None,
    detection_filter: Optional[str] = None,
) -> List[AutoItemRecord]:
    _validate_job_id(job_id)
    if not aa_db.get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return aa_db.list_items(
        job_id,
        batch_id=batch_id,
        review_status=review_status,
        reject_reason=reject_reason,
        detection_filter=detection_filter,
    )


@router.post("/jobs/{job_id}/start", response_model=AutoJobRecord)
def start_or_resume_job(job_id: str) -> AutoJobRecord:
    _validate_job_id(job_id)
    try:
        job = auto_annotate_manager.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/jobs/{job_id}/pause", response_model=AutoJobRecord)
def pause_job(job_id: str) -> AutoJobRecord:
    _validate_job_id(job_id)
    job = auto_annotate_manager.pause(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=AutoJobRecord)
def cancel_job(job_id: str) -> AutoJobRecord:
    _validate_job_id(job_id)
    job = auto_annotate_manager.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("/jobs/{job_id}/stream")
async def stream_auto_job(job_id: str) -> StreamingResponse:
    _validate_job_id(job_id)
    if not aa_db.get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        sub = auto_annotate_manager.subscribe(job_id)
        try:
            snap = auto_annotate_manager.get_progress(job_id)
            yield f"data: {json.dumps(snap)}\n\n"
            if snap.get("status") in ("completed", "cancelled", "failed"):
                return
            while True:
                try:
                    msg = await asyncio.to_thread(sub.get, True, 15.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("status") in ("completed", "cancelled", "failed"):
                        break
                except Exception:
                    yield ": keep-alive\n\n"
        finally:
            auto_annotate_manager.unsubscribe(job_id, sub)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _append_detection_classes_to_job(
    job: AutoJobRecord, detections: Optional[List[AutoDetection]]
) -> None:
    if not detections:
        return
    known_lower = {c.strip().lower() for c in job.classes if c.strip()}
    color_lower = {c.name.strip().lower() for c in job.classColors if c.name.strip()}
    palette = aa_config.DEFAULT_CLASS_COLORS
    for det in detections:
        cname = det.label.strip()
        if not cname:
            continue
        key = cname.lower()
        if key not in known_lower:
            job.classes.append(cname)
            known_lower.add(key)
        if key not in color_lower:
            job.classColors.append(
                ClassColorEntry(
                    name=cname,
                    color=palette[len(job.classColors) % len(palette)],
                )
            )
            color_lower.add(key)


@router.patch("/jobs/{job_id}/items/{item_id}")
def review_auto_item(
    job_id: str, item_id: str, body: ReviewItemPatchRequest
) -> Dict[str, Any]:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    updated = aa_db.update_item_review(
        job_id,
        item_id,
        review_status=body.reviewStatus,
        reject_reason=body.rejectReason,
        detections=body.detections,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    hydrated_job = aa_db.get_job(job_id)
    if hydrated_job:
        _append_detection_classes_to_job(hydrated_job, body.detections)
        all_items = aa_db.list_items(job_id)
        accepted_dir = rf_service.sync_single_item_to_disk(
            job=hydrated_job,
            item=updated["item"],
            local_path=updated["localPath"],
            all_items=all_items,
        )
        hydrated_job.localAcceptedDir = accepted_dir
        aa_db.upsert_job(hydrated_job)
        hydrated_job = aa_db.get_job(job_id)

    return {"job": hydrated_job, "item": updated["item"]}


@router.post(
    "/jobs/{job_id}/items/{item_id}/smart-select/prepare",
    response_model=ManualSmartPrepareResponse,
)
def prepare_manual_smart_select(
    job_id: str, item_id: str, body: ManualSmartPrepareRequest
) -> ManualSmartPrepareResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = aa_db.get_item_with_path(job_id, item_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    img_path = Path(entry["localPath"])
    if not img_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Local image file missing for item '{item_id}'."
        )

    meta = pytorch_engine.prepare_sam_image(
        image_path=img_path,
        sam_refiner_id=body.samRefinerId,
    )
    return ManualSmartPrepareResponse(
        ready=bool(meta.get("ready", True)),
        cached=bool(meta.get("cached", False)),
        latencySec=float(meta.get("latency_s", 0.0)),
        engineUsed=str(meta.get("engineUsed", body.samRefinerId)),
    )


@router.post(
    "/jobs/{job_id}/items/{item_id}/smart-select",
    response_model=ManualSmartSelectResponse,
)
def manual_smart_select(
    job_id: str, item_id: str, body: ManualSmartSelectRequest
) -> ManualSmartSelectResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = aa_db.get_item_with_path(job_id, item_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    img_path = Path(entry["localPath"])
    if not img_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Local image file missing for item '{item_id}'."
        )

    has_points = len(body.points) > 0
    has_bbox = body.bboxPrompt is not None and len(body.bboxPrompt) == 4
    if not has_points and not has_bbox:
        raise HTTPException(
            status_code=400,
            detail="At least one click point or a 4-value bboxPrompt is required.",
        )

    use_owl = body.engine == "owlv2_sam"
    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if use_owl or body.owlModelId
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cand_classes = [c.strip() for c in body.candidateClasses if c.strip()] or [
        body.label.strip() or "object"
    ]
    det, meta = pytorch_engine.predict_sam_interactive(
        image_path=img_path,
        points=[[p.x, p.y] for p in body.points],
        point_labels=[int(p.label) for p in body.points],
        bbox_prompt=body.bboxPrompt,
        sam_refiner_id=body.samRefinerId,
        label=body.label.strip() or cand_classes[0],
        output_mode=body.outputMode,
        use_owlv2_assist=use_owl,
        candidate_classes=cand_classes,
        owl_model_id=owl_model_id,
        owl_threshold=body.owlThreshold,
        complexity=body.complexity,
        preview=body.preview,
    )
    msg = (
        None
        if det is not None
        else "No clear mask returned at the selected point; try another point or box prompt."
    )
    return ManualSmartSelectResponse(
        detection=det,
        latencySec=float(meta.get("latency_s", 0.0)),
        engineUsed=str(meta.get("engineUsed", body.samRefinerId)),
        owlMatched=bool(meta.get("owlMatched", False)),
        cached=bool(meta.get("cached", False)),
        message=msg,
    )


@router.post(
    "/jobs/{job_id}/items/{item_id}/find-objects",
    response_model=ManualFindObjectsResponse,
)
def manual_find_objects(
    job_id: str, item_id: str, body: ManualFindObjectsRequest
) -> ManualFindObjectsResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = aa_db.get_item_with_path(job_id, item_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    img_path = Path(entry["localPath"])
    if not img_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Local image file missing for item '{item_id}'."
        )

    classes = [c.strip() for c in body.classes if c.strip()] or list(job.classes) or ["object"]

    if body.detector == "yoloe":
        dets, meta = pytorch_engine.predict_yoloe(
            image_path=img_path,
            class_names=classes,
            output_mode=body.outputMode,
            conf_threshold=body.confThreshold,
            id_prefix="yoloe",
        )
        total_lat = float(meta.get("latency_s", 0.0))
        engine_used = "yoloe-26s-seg"
        if (
            dets
            and body.outputMode in ("polygon", "both")
            and body.samRefinerId
            and body.samRefinerId != "none"
        ):
            dets, sam_lat = pytorch_engine.refine_boxes_with_sam(
                image_path=img_path,
                detections=dets,
                sam_refiner_id=body.samRefinerId,
            )
            total_lat = round(total_lat + sam_lat, 3)
            engine_used = f"yoloe+{body.samRefinerId}"
        return ManualFindObjectsResponse(
            detections=dets,
            latencySec=total_lat,
            engineUsed=engine_used,
        )

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dets, meta = pytorch_engine.predict_owlv2_find_objects(
        image_path=img_path,
        class_names=classes,
        conf_threshold=body.confThreshold,
        output_mode=body.outputMode,
        sam_refiner_id=body.samRefinerId,
        owl_model_id=owl_model_id,
    )
    return ManualFindObjectsResponse(
        detections=dets,
        latencySec=float(meta.get("latency_s", 0.0)),
        engineUsed=str(meta.get("engineUsed", "owlv2")),
    )


@router.post(
    "/jobs/{job_id}/items/{item_id}/visual-embed/add",
    response_model=VisualEmbedResponse,
)
def visual_embed_add_exemplar(
    job_id: str, item_id: str, body: VisualEmbedAddRequest
) -> VisualEmbedResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = aa_db.get_item_with_path(job_id, item_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    img_path = Path(entry["localPath"])
    if not img_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Local image file missing for item '{item_id}'."
        )

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
        out = pytorch_engine.add_visual_embed_exemplar(
            query_id=body.queryId,
            image_path=img_path,
            bbox_1000=body.bbox,
            point_1000=body.point,
            polarity=body.polarity,
            label=body.label.strip() or "object",
            target_classes=body.targetClasses,
            owl_model_id=owl_model_id,
            threshold=body.threshold,
            min_objectness=body.minObjectness,
            nms_iou=body.nmsIou,
            use_sam_masks=body.useSamMasks,
            sam_refiner_id=body.samRefinerId,
            complexity=body.complexity,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VisualEmbedResponse(**out)


@router.post(
    "/jobs/{job_id}/items/{item_id}/visual-embed/search",
    response_model=VisualEmbedResponse,
)
def visual_embed_search_item(
    job_id: str, item_id: str, body: VisualEmbedSearchRequest
) -> VisualEmbedResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = aa_db.get_item_with_path(job_id, item_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    img_path = Path(entry["localPath"])
    if not img_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Local image file missing for item '{item_id}'."
        )

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        out = pytorch_engine.search_visual_embed_image(
            query_id=body.queryId,
            image_path=img_path,
            label=body.label,
            target_classes=body.targetClasses,
            owl_model_id=owl_model_id,
            threshold=body.threshold,
            min_objectness=body.minObjectness,
            nms_iou=body.nmsIou,
            use_sam_masks=body.useSamMasks,
            sam_refiner_id=body.samRefinerId,
            complexity=body.complexity,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Visual query expired or reset; draw a box to start a new query.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VisualEmbedResponse(**out)


@router.post("/jobs/{job_id}/visual-embed/reset")
def visual_embed_reset_query(
    job_id: str, body: Optional[VisualEmbedResetRequest] = None
) -> Dict[str, Any]:
    _validate_job_id(job_id)
    pytorch_engine.reset_visual_query(body.queryId if body else None)
    return {"ok": True}


@router.post(
    "/jobs/{job_id}/visual-embed/build-references",
    response_model=VisualEmbedBuildRefsResponse,
)
async def build_visual_references(
    job_id: str, body: VisualEmbedBuildRefsRequest
) -> VisualEmbedBuildRefsResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    all_entries = aa_db.get_all_items_with_paths(job_id)
    entry_by_id = {e["item"].id: e for e in all_entries}

    allowed_classes = pytorch_engine._normalize_target_classes(body.targetClasses)
    if allowed_classes is not None and not allowed_classes:
        raise HTTPException(
            status_code=400,
            detail="Select at least one target class to index into the ViT Reference Bank.",
        )

    ref_specs: List[Dict[str, Any]] = []
    if body.autoFromAnnotated:
        for entry in all_entries:
            item = entry["item"]
            if len(item.detections) == 0:
                continue
            if (
                body.autoScope == "current_batch"
                and body.batchId
                and body.batchId != "all"
                and item.batchId != body.batchId
            ):
                continue
            if body.autoScope == "accepted_only" and item.reviewStatus != "accepted":
                continue
            img_path = Path(entry["localPath"])
            if not img_path.is_file():
                continue
            boxes = [
                (list(det.bbox_2d), det.label.strip() or "object")
                for det in item.detections
                if len(det.bbox_2d) == 4
                and (
                    allowed_classes is None
                    or (det.label.strip() or "object").lower() in allowed_classes
                )
            ]
            if boxes:
                ref_specs.append(
                    {"itemId": item.id, "imagePath": img_path, "boxes": boxes}
                )
    else:
        for spec in body.references:
            entry = entry_by_id.get(spec.itemId)
            if not entry:
                continue
            item = entry["item"]
            img_path = Path(entry["localPath"])
            if not img_path.is_file():
                continue
            source_dets = (
                list(spec.inlineDetections)
                if spec.inlineDetections
                else list(item.detections)
            )
            wanted_ids = {did for did in (spec.detectionIds or []) if did}
            if wanted_ids:
                source_dets = [d for d in source_dets if d.id in wanted_ids]
            boxes = [
                (list(det.bbox_2d), det.label.strip() or "object")
                for det in source_dets
                if len(det.bbox_2d) == 4
                and (
                    allowed_classes is None
                    or (det.label.strip() or "object").lower() in allowed_classes
                )
            ]
            if boxes:
                ref_specs.append(
                    {"itemId": item.id, "imagePath": img_path, "boxes": boxes}
                )

    if not ref_specs:
        detail_msg = (
            "No annotated boxes match the selected target classes."
            if allowed_classes is not None
            else "No valid labeled bounding boxes found in selected ViT references."
        )
        raise HTTPException(status_code=400, detail=detail_msg)

    search_path: Optional[Path] = None
    if body.searchItemId:
        s_entry = entry_by_id.get(body.searchItemId)
        if s_entry and Path(s_entry["localPath"]).is_file():
            search_path = Path(s_entry["localPath"])

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
        out = await asyncio.to_thread(
            pytorch_engine.build_visual_ref_bank,
            query_id=body.queryId,
            replace=body.replace,
            owl_model_id=owl_model_id,
            ref_specs=ref_specs,
            target_classes=body.targetClasses,
            search_image_path=search_path,
            threshold=body.threshold,
            min_objectness=body.minObjectness,
            nms_iou=body.nmsIou,
            use_sam_masks=body.useSamMasks,
            sam_refiner_id=body.samRefinerId,
            complexity=body.complexity,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VisualEmbedBuildRefsResponse(**out)


@router.post("/jobs/{job_id}/visual-embed/undo-redo", response_model=VisualEmbedResponse)
async def visual_embed_undo_redo_exemplar(
    job_id: str, body: VisualEmbedUndoRedoRequest
) -> VisualEmbedResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    search_path: Optional[Path] = None
    if body.searchItemId:
        entry = aa_db.get_item_with_path(job_id, body.searchItemId)
        if entry and Path(entry["localPath"]).is_file():
            search_path = Path(entry["localPath"])

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        out = await asyncio.to_thread(
            pytorch_engine.undo_redo_visual_embed_exemplar,
            query_id=body.queryId,
            action=body.action,
            search_image_path=search_path,
            label=body.label,
            target_classes=body.targetClasses,
            owl_model_id=owl_model_id,
            threshold=body.threshold,
            min_objectness=body.minObjectness,
            nms_iou=body.nmsIou,
            use_sam_masks=body.useSamMasks,
            sam_refiner_id=body.samRefinerId,
            complexity=body.complexity,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Visual query expired or reset; draw a box or pin ViT references first.",
        ) from exc

    return VisualEmbedResponse(**out)


@router.post("/jobs/{job_id}/visual-embed/search-batch")
async def visual_embed_search_batch(
    job_id: str, body: VisualEmbedBatchSearchRequest, request: Request
) -> StreamingResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not pytorch_engine.has_visual_query(body.queryId):
        raise HTTPException(
            status_code=404,
            detail="Visual query expired or reset; draw a box or pin ViT references first.",
        )
    allowed_classes = pytorch_engine._normalize_target_classes(body.targetClasses)
    if allowed_classes is not None and not allowed_classes:
        raise HTTPException(
            status_code=400,
            detail="Select at least one target class to search.",
        )
    if body.targetClasses is not None and not pytorch_engine.has_visual_query(
        body.queryId, target_classes=body.targetClasses
    ):
        raise HTTPException(
            status_code=400,
            detail="No indexed visual references match the selected classes. Click Index Now to index references for these classes.",
        )

    try:
        owl_model_id = (
            aa_config.resolve_owlv2_model_id(body.owlModelId)
            if body.owlModelId
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    all_entries = aa_db.get_all_items_with_paths(job_id)
    if body.scope == "current_batch" and body.batchId and body.batchId != "all":
        target_entries = [
            e for e in all_entries if e["item"].batchId == body.batchId
        ]
    elif body.scope == "item_ids":
        wanted = set(body.itemIds)
        target_entries = [e for e in all_entries if e["item"].id in wanted]
    else:
        target_entries = all_entries

    excluded_ids = set(body.excludeItemIds or [])
    if excluded_ids or body.skipAnnotated:
        target_entries = [
            e
            for e in target_entries
            if e["item"].id not in excluded_ids
            and (not body.skipAnnotated or len(e["item"].detections) == 0)
        ]

    target_label = (body.label or "object").strip() or "object"
    target_label_low = target_label.lower()
    use_bucket_labels = body.multiClass or (body.targetClasses is not None)
    active_cls_lows = (
        {
            c.strip().lower()
            for c in pytorch_engine.get_visual_query_classes(
                body.queryId, target_classes=body.targetClasses
            )
            if c.strip()
        }
        if use_bucket_labels
        else {target_label_low}
    )

    async def _ndjson_stream() -> AsyncGenerator[str, None]:
        total = len(target_entries)
        total_added = 0
        items_with_matches = 0

        for idx, entry in enumerate(target_entries, start=1):
            if await request.is_disconnected():
                break

            item: AutoItemRecord = entry["item"]
            img_path = Path(entry["localPath"])
            if not img_path.is_file():
                yield json.dumps(
                    {
                        "type": "progress",
                        "done": idx,
                        "total": total,
                        "itemId": item.id,
                        "filename": item.filename,
                        "matchedCount": 0,
                        "addedCount": 0,
                        "totalAdded": total_added,
                        "itemsWithMatches": items_with_matches,
                        "detections": [],
                        "item": None if body.previewOnly else item.model_dump(),
                    }
                ) + "\n"
                continue

            res = await asyncio.to_thread(
                pytorch_engine.search_visual_embed_image,
                query_id=body.queryId,
                image_path=img_path,
                label=None if use_bucket_labels else target_label,
                target_classes=body.targetClasses,
                owl_model_id=owl_model_id,
                threshold=body.threshold,
                min_objectness=body.minObjectness,
                nms_iou=body.nmsIou,
                use_sam_masks=body.useSamMasks,
                sam_refiner_id=body.samRefinerId,
                complexity=body.complexity,
            )
            matches: List[AutoDetection] = list(res.get("detections") or [])

            if body.previewOnly:
                if body.mergeMode == "replace_class":
                    staged_matches = list(matches)
                else:
                    staged_matches = []
                    for cand in matches:
                        cand_low = cand.label.strip().lower()
                        is_dup = any(
                            ex.label.strip().lower() == cand_low
                            and pytorch_engine._box_iou_1000(cand.bbox_2d, ex.bbox_2d)
                            >= body.dedupeIou
                            for ex in item.detections
                        )
                        if not is_dup:
                            staged_matches.append(cand)
                if staged_matches:
                    items_with_matches += 1
                    total_added += len(staged_matches)
                yield json.dumps(
                    {
                        "type": "progress",
                        "done": idx,
                        "total": total,
                        "itemId": item.id,
                        "filename": item.filename,
                        "matchedCount": len(matches),
                        "addedCount": len(staged_matches),
                        "totalAdded": total_added,
                        "itemsWithMatches": items_with_matches,
                        "detections": [d.model_dump() for d in staged_matches],
                        "item": None,
                    }
                ) + "\n"
                continue

            if matches:
                items_with_matches += 1

            if body.mergeMode == "replace_class":
                replace_set = active_cls_lows or {
                    m.label.strip().lower() for m in matches if m.label.strip()
                } or {target_label_low}
                kept_existing = [
                    d
                    for d in item.detections
                    if d.label.strip().lower() not in replace_set
                ]
                removed_cnt = len(item.detections) - len(kept_existing)
                next_dets = [*kept_existing, *matches]
                added_cnt = len(matches)
                changed = added_cnt > 0 or removed_cnt > 0
            else:
                next_dets = list(item.detections)
                added_cnt = 0
                for cand in matches:
                    cand_low = cand.label.strip().lower()
                    is_dup = any(
                        ex.label.strip().lower() == cand_low
                        and pytorch_engine._box_iou_1000(cand.bbox_2d, ex.bbox_2d)
                        >= body.dedupeIou
                        for ex in next_dets
                    )
                    if not is_dup:
                        next_dets.append(cand)
                        added_cnt += 1
                changed = added_cnt > 0

            latest_item = item
            if changed:
                total_added += added_cnt
                updated = aa_db.update_item_review(
                    job_id,
                    item.id,
                    review_status=item.reviewStatus,
                    reject_reason=item.rejectReason,
                    detections=next_dets,
                )
                if updated:
                    latest_item = updated["item"]
                    hydrated = aa_db.get_job(job_id)
                    if hydrated:
                        _append_detection_classes_to_job(hydrated, matches)
                        if latest_item.reviewStatus == "accepted":
                            all_items_now = aa_db.list_items(job_id)
                            accepted_dir = rf_service.sync_single_item_to_disk(
                                job=hydrated,
                                item=latest_item,
                                local_path=updated["localPath"],
                                all_items=all_items_now,
                            )
                            hydrated.localAcceptedDir = accepted_dir
                        aa_db.upsert_job(hydrated)

            yield json.dumps(
                {
                    "type": "progress",
                    "done": idx,
                    "total": total,
                    "itemId": latest_item.id,
                    "filename": latest_item.filename,
                    "matchedCount": len(matches),
                    "addedCount": added_cnt,
                    "totalAdded": total_added,
                    "itemsWithMatches": items_with_matches,
                    "item": latest_item.model_dump(),
                }
            ) + "\n"

        final_job = None if body.previewOnly else aa_db.get_job(job_id)
        yield json.dumps(
            {
                "type": "done",
                "done": total,
                "total": total,
                "totalAdded": total_added,
                "itemsWithMatches": items_with_matches,
                "job": final_job.model_dump() if final_job else None,
            }
        ) + "\n"

    return StreamingResponse(
        _ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/jobs/{job_id}/batches/{batch_id}/approve")
def approve_batch(
    job_id: str, batch_id: str, body: ApproveBatchRequest
) -> Dict[str, Any]:
    """
    Approve a 50-image batch:
    - Marks ONLY pending items in `batch_id` as accepted (if acceptPending=True).
    - Keeps explicitly rejected items in `batch_id` as rejected.
    - Marks the batch `checked = True` and persists accepted items locally.
    """
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    batch_items = aa_db.bulk_approve_batch_pending(
        job_id, batch_id, accept_pending=body.acceptPending
    )
    hydrated_job = aa_db.get_job(job_id)
    if hydrated_job:
        all_with_paths = aa_db.get_all_items_with_paths(job_id)
        accepted_dir = rf_service.sync_accepted_items_to_disk(
            job=hydrated_job, items_with_paths=all_with_paths
        )
        hydrated_job.localAcceptedDir = accepted_dir
        aa_db.upsert_job(hydrated_job)
        hydrated_job = aa_db.get_job(job_id)

    return {
        "job": hydrated_job,
        "batchItems": [entry["item"] for entry in batch_items],
    }


@router.post("/jobs/{job_id}/push", response_model=PushAcceptedResponse)
def push_accepted_to_roboflow(
    job_id: str, body: PushAcceptedRequest
) -> PushAcceptedResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not body.apiKey.strip():
        raise HTTPException(
            status_code=400, detail="Roboflow API key is required."
        )

    dest = body.pushDestination
    if dest == "new_project":
        if not (body.targetProjectName or "").strip():
            raise HTTPException(
                status_code=400,
                detail="Target Roboflow project name is required when creating a new project.",
            )
        zip_slug_target = body.targetProjectName.strip()
    else:
        zip_slug_target = job.projectSlug.strip()

    all_with_paths = aa_db.get_all_items_with_paths(job_id)
    rf_service.sync_accepted_items_to_disk(job=job, items_with_paths=all_with_paths)

    (
        eligible_entries,
        included_items,
        class_lower_to_id,
        ordered_names,
        null_count,
        skipped_empty,
        excluded_rejected,
        excluded_pending,
    ) = rf_service._prepare_accepted_dataset(
        job=job,
        items_with_paths=all_with_paths,
        batch_id_filter=body.batchId,
        include_empty=body.includeEmpty,
        only_unpushed=body.onlyUnpushed,
    )

    if not included_items:
        raise HTTPException(
            status_code=400,
            detail=(
                "No pushable Accepted images found "
                f"(excluded {excluded_rejected} rejected, {excluded_pending} pending, "
                f"{skipped_empty} empty detections)."
            ),
        )

    zip_path, _, _, _, _ = rf_service.build_accepted_zip(
        job=job,
        items_with_paths=all_with_paths,
        target_project_name=zip_slug_target,
        batch_id_filter=body.batchId,
        include_empty=body.includeEmpty,
    )

    item_ids = [it.id for it in included_items]
    aa_db.mark_items_push_status(item_ids, push_status="pushing")

    try:
        rf_res = rf_service.push_accepted_to_roboflow(
            api_key=body.apiKey,
            workspace_slug=body.workspaceSlug or job.workspaceSlug,
            job=job,
            eligible_entries=eligible_entries,
            class_lower_to_id=class_lower_to_id,
            ordered_names=ordered_names,
            push_destination=dest,
            target_project_name=body.targetProjectName,
            target_folder_name=body.targetFolderName,
            target_annotation_job_id=body.targetAnnotationJobId,
            add_to_dataset=body.addToDataset,
            project_license=body.projectLicense,
            zip_path=zip_path,
            batch_name=body.batchTag or f"{job.id}-{body.batchId or 'verified'}",
            tags=["auto-annotated", "verified", job.projectSlug],
        )
    except Exception as exc:
        aa_db.mark_items_push_status(
            item_ids, push_status="failed", push_error=str(exc)
        )
        raise HTTPException(
            status_code=502, detail=f"Roboflow push failed: {exc}"
        ) from exc

    succeeded_ids: List[str] = list(rf_res.get("succeededItemIds") or [])
    failed_items: List[tuple[str, str]] = list(rf_res.get("failedItems") or [])

    if succeeded_ids:
        aa_db.mark_items_push_status(succeeded_ids, push_status="pushed")
    for fi_id, fi_err in failed_items:
        aa_db.mark_items_push_status([fi_id], push_status="failed", push_error=fi_err)

    if not succeeded_ids:
        first_err = failed_items[0][1] if failed_items else "No items succeeded."
        raise HTTPException(
            status_code=502, detail=f"Roboflow push failed: {first_err}"
        )

    hydrated_job = aa_db.get_job(job_id)
    upload_status = rf_res.get("uploadStatus") or {}
    task_id = (
        str(upload_status.get("taskId") or upload_status.get("task_id") or "")
        or None
    )
    target_proj = rf_res["targetProjectSlug"]
    target_folder = rf_res.get("targetFolderName")
    eff_ann_job = rf_res.get("targetAnnotationJobId")
    added_to_dataset = bool(rf_res.get("addedToDataset", True))

    succeeded_set = set(succeeded_ids)
    pushed_null_count = sum(
        1
        for it in included_items
        if it.id in succeeded_set and len(it.detections) == 0
    )
    null_note = f" (including {pushed_null_count} Null background frame(s))" if pushed_null_count > 0 else ""

    if dest == "new_project":
        action_desc = f"Pushed {len(succeeded_ids)} accepted image(s){null_note} to new project '{target_proj}'"
    elif dest == "same_project_dataset":
        action_desc = f"Saved {len(succeeded_ids)} accepted frame(s){null_note} to Dataset in '{target_proj}'"
    elif dest == "same_project_annotate":
        job_note = f" / job '{eff_ann_job}'" if eff_ann_job else ""
        action_desc = (
            f"Staged {len(succeeded_ids)} image(s){null_note} in Annotate ({target_proj}{job_note}, folder '{target_folder or 'default'}')"
        )
    else:
        loc = "Dataset" if added_to_dataset else "Annotate"
        action_desc = (
            f"Saved {len(succeeded_ids)} accepted frame(s){null_note} to '{target_proj}' ({loc}, folder '{target_folder or 'default'}')"
        )

    advisory_notes: List[str] = []
    if isinstance(upload_status.get("folderMove"), dict) and upload_status["folderMove"].get("note"):
        advisory_notes.append(str(upload_status["folderMove"]["note"]))
    if isinstance(upload_status.get("annotationJobStage"), dict) and upload_status["annotationJobStage"].get("note"):
        advisory_notes.append(str(upload_status["annotationJobStage"]["note"]))
    if failed_items:
        advisory_notes.append(f"{len(failed_items)} item(s) failed.")

    msg = (
        f"{action_desc}. Excluded {excluded_rejected} rejected & {excluded_pending} pending."
        + (f" Note: {' '.join(advisory_notes)}" if advisory_notes else "")
    )
    if hydrated_job:
        hydrated_job.localZipPath = str(zip_path)
        hydrated_job.lastPushDestination = dest
        hydrated_job.lastPushProject = target_proj
        hydrated_job.lastPushTaskId = task_id
        hydrated_job.lastPushStatus = "pushed"
        hydrated_job.lastPushMessage = msg
        rf_service.sync_accepted_items_to_disk(
            job=hydrated_job, items_with_paths=aa_db.get_all_items_with_paths(job_id)
        )
        aa_db.upsert_job(hydrated_job)

    return PushAcceptedResponse(
        jobId=job_id,
        pushDestination=dest,
        targetProjectName=target_proj,
        targetFolderName=target_folder,
        targetAnnotationJobId=eff_ann_job,
        addedToDataset=added_to_dataset,
        pushedImagesCount=len(succeeded_ids),
        nullImagesCount=pushed_null_count,
        failedImagesCount=len(failed_items),
        skippedEmptyCount=skipped_empty,
        excludedRejectedCount=excluded_rejected,
        excludedPendingCount=excluded_pending,
        localZipPath=str(zip_path),
        taskId=task_id,
        roboflowStatus=upload_status,
        message=msg,
    )


@router.get("/jobs/{job_id}/download-zip")
def download_accepted_zip(
    job_id: str,
    batch_id: Optional[str] = None,
    include_empty: bool = True,
) -> FileResponse:
    _validate_job_id(job_id)
    job = aa_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    all_with_paths = aa_db.get_all_items_with_paths(job_id)
    rf_service.sync_accepted_items_to_disk(job=job, items_with_paths=all_with_paths)
    zip_path, included, _, _, _ = rf_service.build_accepted_zip(
        job=job,
        items_with_paths=all_with_paths,
        target_project_name=f"{job.projectSlug}-auto-annotated",
        batch_id_filter=batch_id,
        include_empty=include_empty,
    )
    if not included or not zip_path.is_file():
        raise HTTPException(
            status_code=400, detail="No accepted images available to package yet."
        )
    return FileResponse(
        zip_path, media_type="application/zip", filename=zip_path.name
    )


@router.delete("/jobs/{job_id}")
def delete_auto_job(job_id: str) -> Dict[str, Any]:
    _validate_job_id(job_id)
    if not aa_db.get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    auto_annotate_manager.cancel(job_id)
    aa_db.delete_job(job_id)
    job_dir = (aa_config.AUTO_JOBS_DIR / job_id).resolve()
    base_resolved = aa_config.AUTO_JOBS_DIR.resolve()
    if job_dir.is_relative_to(base_resolved) and job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": True, "jobId": job_id}


@router.get("/files/{job_id}/{filepath:path}")
def serve_auto_file(job_id: str, filepath: str) -> FileResponse:
    _validate_job_id(job_id)
    base_dir = (aa_config.AUTO_JOBS_DIR / job_id).resolve()
    target = (base_dir / filepath).resolve()
    if not target.is_relative_to(base_dir):
        raise HTTPException(status_code=403, detail="Invalid file path.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        target,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
