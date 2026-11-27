"""Roboflow project inspection, native image fetching, local YOLO/manifest persistence, and bulk ZIP push."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
import yaml
from PIL import Image
from roboflow import Roboflow
from roboflow.adapters import rfapi

from auto_annotate import config as aa_config
from auto_annotate.schemas import (
    AutoBatchSummary,
    AutoDetection,
    AutoItemRecord,
    AutoJobRecord,
    ClassColorEntry,
    PushAcceptedResponse,
    RoboflowAnnotationJobInfo,
    RoboflowBatchInfo,
    RoboflowInspectResponse,
)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_stem_filename(name: str) -> str:
    base = Path(name).name.strip() or "image.jpg"
    cleaned = _SAFE_NAME_RE.sub("_", base)
    return cleaned or "image.jpg"


def _normalize_project_slug(name_or_slug: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", name_or_slug.strip().lower()).strip("-")
    return slug or "auto-annotated-dataset"


def _extract_project_classes_and_colors(
    project: Any,
) -> Tuple[List[str], List[ClassColorEntry]]:
    raw_classes = getattr(project, "classes", None)
    if isinstance(raw_classes, dict):
        # Could be {class_name: count} or {"0": "class_name"}
        keys = [str(k) for k in raw_classes.keys()]
        if keys and all(k.isdigit() for k in keys):
            classes = [str(raw_classes[k]) for k in sorted(keys, key=lambda x: int(x))]
        else:
            classes = keys
    elif isinstance(raw_classes, list):
        classes = [str(c) for c in raw_classes if str(c).strip()]
    else:
        classes = []

    raw_colors = getattr(project, "colors", None)
    color_map: Dict[str, str] = {}
    if isinstance(raw_colors, dict):
        for k, v in raw_colors.items():
            if isinstance(v, str) and v.strip():
                color_map[str(k).strip().lower()] = v.strip()

    palette = aa_config.DEFAULT_CLASS_COLORS
    entries: List[ClassColorEntry] = []
    for idx, cls_name in enumerate(classes):
        hex_color = color_map.get(cls_name.lower(), palette[idx % len(palette)])
        entries.append(ClassColorEntry(name=cls_name, color=hex_color))
    return classes, entries


def _extract_email_hint(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        clean = raw.strip()
        return clean or None
    if isinstance(raw, dict):
        for k in ("email", "labelerEmail", "assigneeEmail", "username", "name", "displayName", "id"):
            val = raw.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(raw, list):
        parts = [_extract_email_hint(v) for v in raw]
        joined = ", ".join(p for p in parts if p)
        return joined or None
    return None


def _parse_annotation_job_entry(
    raw: Dict[str, Any], idx: int
) -> Optional[RoboflowAnnotationJobInfo]:
    jid = str(raw.get("id") or raw.get("jobId") or raw.get("job_id") or "").strip()
    if not jid:
        return None
    name = str(raw.get("name") or raw.get("jobName") or f"Job {idx}").strip()
    status = str(raw.get("status") or raw.get("stage") or "").strip()
    num_imgs = int(
        raw.get("numImages")
        or raw.get("imagesCount")
        or raw.get("imageCount")
        or raw.get("totalImages")
        or raw.get("images")
        if isinstance(raw.get("images"), (int, float))
        else (
            raw.get("numImages")
            or raw.get("imagesCount")
            or raw.get("imageCount")
            or raw.get("totalImages")
            or (len(raw["images"]) if isinstance(raw.get("images"), list) else 0)
        )
    )
    ann_cnt = int(
        raw.get("annotated")
        or raw.get("annotatedCount")
        or raw.get("numAnnotated")
        or 0
    )
    if num_imgs == 0 and ann_cnt > 0:
        num_imgs = ann_cnt
    labeler = None
    for k in (
        "labelerEmail",
        "labeler",
        "assigneeEmail",
        "assignee",
        "assignees",
        "assignedTo",
        "assigned_to",
        "user",
        "owner",
    ):
        if raw.get(k) is not None:
            labeler = _extract_email_hint(raw.get(k))
            if labeler:
                break
    reviewer = None
    for k in ("reviewerEmail", "reviewer", "reviewers"):
        if raw.get(k) is not None:
            reviewer = _extract_email_hint(raw.get(k))
            if reviewer:
                break
    return RoboflowAnnotationJobInfo(
        id=jid,
        name=name,
        status=status,
        imagesCount=num_imgs,
        annotatedCount=ann_cnt,
        labelerEmail=labeler,
        reviewerEmail=reviewer,
    )


def _list_annotation_jobs(project: Any) -> List[RoboflowAnnotationJobInfo]:
    by_id: Dict[str, RoboflowAnnotationJobInfo] = {}

    def _absorb(payload: Any) -> None:
        if isinstance(payload, dict):
            rows = (
                payload.get("jobs")
                or payload.get("annotationJobs")
                or payload.get("annotation_jobs")
                or payload.get("data")
                or payload.get("results")
                or []
            )
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        for idx, item in enumerate(rows, start=1):
            if not isinstance(item, dict):
                continue
            parsed = _parse_annotation_job_entry(item, idx)
            if not parsed:
                continue
            existing = by_id.get(parsed.id)
            if existing is None:
                by_id[parsed.id] = parsed
            else:
                if not existing.labelerEmail and parsed.labelerEmail:
                    existing.labelerEmail = parsed.labelerEmail
                if not existing.reviewerEmail and parsed.reviewerEmail:
                    existing.reviewerEmail = parsed.reviewerEmail
                if not existing.status and parsed.status:
                    existing.status = parsed.status
                if existing.imagesCount == 0 and parsed.imagesCount > 0:
                    existing.imagesCount = parsed.imagesCount
                if existing.annotatedCount == 0 and parsed.annotatedCount > 0:
                    existing.annotatedCount = parsed.annotatedCount

    try:
        if hasattr(project, "get_annotation_jobs"):
            _absorb(project.get_annotation_jobs())
    except Exception as exc:
        print(f"[roboflow] get_annotation_jobs failed: {exc}", flush=True)

    try:
        if hasattr(project, "get_annotation_jobs_admin"):
            cursor: Optional[str] = None
            for _ in range(10):
                res = project.get_annotation_jobs_admin(
                    limit=50, after=cursor, show_empty=True
                )
                _absorb(res)
                next_cur = (
                    (res.get("nextCursor") or res.get("next_cursor") or res.get("after"))
                    if isinstance(res, dict)
                    else None
                )
                if not next_cur or next_cur == cursor:
                    break
                cursor = str(next_cur)
    except Exception as exc:
        print(f"[roboflow] get_annotation_jobs_admin failed: {exc}", flush=True)

    # Enrich missing assignee metadata if there is a small set of jobs
    if len(by_id) <= 20 and hasattr(project, "get_annotation_job"):
        for jid, info in list(by_id.items()):
            if info.labelerEmail and info.imagesCount > 0:
                continue
            try:
                detail = project.get_annotation_job(jid)
                raw = detail.get("job", detail) if isinstance(detail, dict) else {}
                if isinstance(raw, dict):
                    parsed = _parse_annotation_job_entry(raw, 1)
                    if parsed:
                        if not info.labelerEmail and parsed.labelerEmail:
                            info.labelerEmail = parsed.labelerEmail
                        if not info.reviewerEmail and parsed.reviewerEmail:
                            info.reviewerEmail = parsed.reviewerEmail
                        if not info.status and parsed.status:
                            info.status = parsed.status
                        if info.imagesCount == 0 and parsed.imagesCount > 0:
                            info.imagesCount = parsed.imagesCount
                        if info.annotatedCount == 0 and parsed.annotatedCount > 0:
                            info.annotatedCount = parsed.annotatedCount
            except Exception as exc:
                print(f"[roboflow] get_annotation_job({jid}) failed: {exc}", flush=True)

    return list(by_id.values())


def _parse_roboflow_target(
    project_slug: str,
    workspace_slug: Optional[str] = None,
) -> Tuple[str, str]:
    """Extract (workspace_slug, project_slug) from raw slugs, `ws/proj`, or Roboflow URLs."""
    raw_proj = (project_slug or "").strip()
    raw_ws = (workspace_slug or "").strip()

    if raw_ws.startswith("http://") or raw_ws.startswith("https://"):
        parts = [p for p in urlparse(raw_ws).path.split("/") if p]
        raw_ws = parts[0] if parts else ""
    else:
        raw_ws = raw_ws.strip("/")

    if raw_proj.startswith("http://") or raw_proj.startswith("https://"):
        parts = [p for p in urlparse(raw_proj).path.split("/") if p]
        if len(parts) >= 2:
            if not raw_ws:
                raw_ws = parts[0]
            raw_proj = parts[1]
        elif len(parts) == 1:
            raw_proj = parts[0]
    else:
        raw_proj = raw_proj.strip("/")
        if "/" in raw_proj:
            parts = [p for p in raw_proj.split("/") if p]
            if len(parts) >= 2:
                if not raw_ws:
                    raw_ws = parts[-2]
                raw_proj = parts[-1]

    return raw_ws, raw_proj


def _extract_ws_project_entries(ws: Any) -> List[Tuple[str, str]]:
    """Return list of (slug, display_name) from `ws.project_list`."""
    entries: List[Tuple[str, str]] = []
    raw_list = getattr(ws, "project_list", None) or []
    for item in raw_list:
        if isinstance(item, str):
            s = item.strip().strip("/")
            slug = s.rsplit("/", 1)[-1] if "/" in s else s
            if slug:
                entries.append((slug, slug))
        elif isinstance(item, dict):
            pid = str(item.get("id") or "").strip().strip("/")
            url_val = str(item.get("url") or "").strip().strip("/")
            name_val = str(item.get("name") or "").strip()
            slug = pid.rsplit("/", 1)[-1] if "/" in pid else (url_val or pid)
            if slug:
                entries.append((slug, name_val or slug))
    return entries


def _match_project_in_workspace(
    ws: Any, candidate: str
) -> Tuple[Optional[str], List[str]]:
    """Match a user-supplied project name/slug against `ws.project_list`."""
    entries = _extract_ws_project_entries(ws)
    all_slugs = [slug for slug, _ in entries]
    if not entries or not candidate:
        return None, all_slugs

    cand_low = candidate.strip().lower()
    norm_cand = _normalize_project_slug(candidate)

    # 1) Exact slug match (case-insensitive)
    for slug, _ in entries:
        if slug.lower() == cand_low:
            return slug, all_slugs

    # 2) Exact display name match (case-insensitive)
    for slug, name in entries:
        if name.lower() == cand_low:
            return slug, all_slugs

    # 3) Normalized slug match against slug or display name
    for slug, name in entries:
        if (
            _normalize_project_slug(slug) == norm_cand
            or _normalize_project_slug(name) == norm_cand
        ):
            return slug, all_slugs

    # 4) Unique Roboflow auto-suffixed slug match (e.g. `tapping-4cam` -> `tapping-4cam-abc12`)
    prefix_matches = [
        slug for slug, _ in entries if slug.lower().startswith(f"{norm_cand}-")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0], all_slugs

    return None, all_slugs


def _resolve_workspace_and_project(
    api_key: str,
    project_slug: str,
    workspace_slug: Optional[str] = None,
) -> Tuple[Any, Any, Any, str, str, str]:
    clean_key = api_key.strip()
    explicit_ws, clean_proj = _parse_roboflow_target(project_slug, workspace_slug)

    rf = Roboflow(api_key=clean_key)
    key_ws = str(getattr(rf, "current_workspace", "") or "").strip()
    ws = rf.workspace(explicit_ws) if explicit_ws else rf.workspace()
    ws_url = str(getattr(ws, "url", "") or explicit_ws or key_ws).strip()

    matched_slug, ws_slugs = _match_project_in_workspace(ws, clean_proj)
    target_slug = matched_slug or clean_proj

    try:
        project = ws.project(target_slug)
    except Exception as proj_exc:
        if ws_slugs:
            preview = ", ".join(ws_slugs[:10])
            raise RuntimeError(
                f"Project '{target_slug}' not found or inaccessible in workspace '{ws_url}'. "
                f"Available projects in '{ws_url}': {preview}. ({proj_exc})"
            ) from proj_exc
        raise

    proj_id = str(getattr(project, "id", "") or "").strip()
    if "/" in proj_id:
        proj_ws, canonical_slug = proj_id.rsplit("/", 1)
    else:
        proj_ws = ws_url
        canonical_slug = target_slug

    in_ws_list = (
        canonical_slug.lower() in {s.lower() for s in ws_slugs}
        if ws_slugs
        else None
    )
    print(
        f"[roboflow] resolved key_ws={key_ws or '?'} target_ws={proj_ws} project={canonical_slug} "
        f"in_ws_list={in_ws_list} ws_projects_count={len(ws_slugs)} public={getattr(project, 'public', None)}",
        flush=True,
    )
    return rf, ws, project, key_ws, proj_ws, canonical_slug


def inspect_project(
    api_key: str,
    project_slug: str,
    workspace_slug: Optional[str] = None,
) -> RoboflowInspectResponse:
    _, _, project, key_ws, proj_ws, canonical_slug = _resolve_workspace_and_project(
        api_key=api_key,
        project_slug=project_slug,
        workspace_slug=workspace_slug,
    )

    classes, _ = _extract_project_classes_and_colors(project)
    batches_out: List[RoboflowBatchInfo] = []
    seen_batch_ids: set[str] = set()
    try:
        raw_batches = project.get_batches()
        raw_list = (
            raw_batches.get("batches", [])
            if isinstance(raw_batches, dict)
            else (raw_batches if isinstance(raw_batches, list) else [])
        )
        for idx, b in enumerate(raw_list, start=1):
            if not isinstance(b, dict):
                continue
            bid = str(b.get("id") or b.get("batch_id") or f"rf-batch-{idx}")
            if bid in seen_batch_ids:
                continue
            seen_batch_ids.add(bid)
            bname = str(b.get("name") or b.get("batchName") or bid)
            bcount = int(b.get("images") or b.get("numImages") or b.get("count") or 0)
            batches_out.append(
                RoboflowBatchInfo(id=bid, name=bname, imagesCount=bcount)
            )
    except Exception as exc:
        print(f"[roboflow] get_batches failed for {proj_ws}/{canonical_slug}: {exc}", flush=True)
        batches_out = []

    if not batches_out and hasattr(project, "get_annotation_batches"):
        try:
            raw_ab = project.get_annotation_batches(limit=50, show_empty=True)
            rows = (
                raw_ab.get("batches")
                or raw_ab.get("annotationBatches")
                or raw_ab.get("data")
                or []
                if isinstance(raw_ab, dict)
                else (raw_ab if isinstance(raw_ab, list) else [])
            )
            for idx, b in enumerate(rows, start=1):
                if not isinstance(b, dict):
                    continue
                bid = str(b.get("id") or b.get("batchId") or f"rf-ab-{idx}")
                if bid in seen_batch_ids:
                    continue
                seen_batch_ids.add(bid)
                bname = str(b.get("name") or b.get("batchName") or bid)
                bcount = int(b.get("numImages") or b.get("imagesCount") or b.get("images") or 0)
                batches_out.append(
                    RoboflowBatchInfo(id=bid, name=bname, imagesCount=bcount)
                )
        except Exception as exc:
            print(f"[roboflow] get_annotation_batches failed for {proj_ws}/{canonical_slug}: {exc}", flush=True)

    annotation_jobs = _list_annotation_jobs(project)

    versions_count = 0
    if hasattr(project, "get_version_information"):
        try:
            v_info = project.get_version_information()
            if isinstance(v_info, list):
                versions_count = len(v_info)
        except Exception as exc:
            print(f"[roboflow] get_version_information failed for {proj_ws}/{canonical_slug}: {exc}", flush=True)

    print(
        f"[roboflow] inspect metadata for {proj_ws}/{canonical_slug}: "
        f"batches={len(batches_out)} ann_jobs={len(annotation_jobs)} versions={versions_count}",
        flush=True,
    )

    warning: Optional[str] = None
    try:
        if hasattr(project, "search"):
            project.search(limit=1, fields=["id"])
    except Exception as search_exc:
        print(
            f"[roboflow] inspect search probe failed for {proj_ws}/{canonical_slug}: {search_exc}",
            flush=True,
        )
        if key_ws and proj_ws and key_ws.lower() != proj_ws.lower():
            warning = (
                f"API key belongs to workspace '{key_ws}', not '{proj_ws}'. "
                f"Paste the Private API Key for workspace '{proj_ws}' (Roboflow Settings -> API Keys) "
                f"so images can be fetched."
            )
        elif api_key.strip().startswith("rf_"):
            warning = (
                "This key starts with 'rf_', which looks like a Roboflow Publishable Key. "
                "Publishable keys cannot query Search or dataset endpoints — paste your Private API Key "
                "from Roboflow Settings -> API Keys."
            )
        else:
            avail_parts: List[str] = []
            if versions_count > 0:
                avail_parts.append(f"{versions_count} dataset version(s)")
            if batches_out:
                avail_parts.append(f"{len(batches_out)} batch(es)")
            if annotation_jobs:
                avail_parts.append(f"{len(annotation_jobs)} annotation job(s)")
            if avail_parts:
                warning = (
                    f"Roboflow Search API returned an error ({search_exc}). "
                    f"Fallback endpoints are available ({', '.join(avail_parts)}) and will be used automatically when fetching."
                )
            else:
                warning = (
                    f"Roboflow Search API returned an error ({search_exc}), and no batches, annotation jobs, "
                    f"or dataset versions were returned. Verify you are using a Private API Key (not Publishable) "
                    f"with access to this project/folder, or generate a Dataset Version in Roboflow."
                )

    return RoboflowInspectResponse(
        workspaceSlug=proj_ws,
        projectSlug=canonical_slug,
        projectName=str(getattr(project, "name", "") or canonical_slug),
        projectType=str(getattr(project, "type", "") or "object-detection"),
        classes=classes,
        totalImages=int(getattr(project, "images", 0) or 0),
        unannotatedImages=int(getattr(project, "unannotated", 0) or 0),
        keyWorkspaceSlug=key_ws or None,
        warning=warning,
        batches=batches_out,
        annotationJobs=annotation_jobs,
    )


def _download_image_with_retry(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise last_exc


def _entry_annotation_count(entry: Dict[str, Any]) -> Optional[int]:
    """Return known annotation count for a Roboflow search/image entry, or None if unknown."""
    anns = entry.get("annotations")
    if isinstance(anns, list):
        return len(anns)
    if isinstance(anns, dict):
        boxes = anns.get("boxes") or anns.get("polygons") or anns.get("annotations")
        if isinstance(boxes, list):
            return len(boxes)
        count = anns.get("count")
        if isinstance(count, (int, float)):
            return int(count)
    labels = entry.get("labels")
    if isinstance(labels, list):
        return len(labels)
    for key in ("annotationCount", "annotationsCount", "numAnnotations"):
        val = entry.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    if "annotated" in entry and isinstance(entry.get("annotated"), bool):
        return 1 if entry["annotated"] else 0
    cached = entry.get("_cached_info")
    if isinstance(cached, dict):
        ann_obj = cached.get("annotation") or {}
        if isinstance(ann_obj, dict):
            boxes = ann_obj.get("boxes") or ann_obj.get("polygons") or []
            if isinstance(boxes, list):
                return len(boxes)
        if isinstance(cached.get("annotated"), bool):
            return 1 if cached["annotated"] else 0
        return 0
    return None


def _merge_entry_metadata(
    existing: Dict[str, Any], incoming: Dict[str, Any]
) -> Dict[str, Any]:
    merged = dict(existing)
    for k, v in incoming.items():
        if v is None:
            continue
        if k not in merged or merged[k] in (None, "", []):
            merged[k] = v
    return merged


def _collect_job_entries(
    project: Any,
    job_id: str,
    *,
    max_images: int,
    seen_ids: set[str],
    annotation_status_filter: str = "all",
) -> List[Dict[str, Any]]:
    """Fetch ALL images belonging to a Roboflow Annotate job (unannotated, review, and dataset/annotated)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _record(entry: Dict[str, Any]) -> None:
        rf_id = str(entry.get("id") or "").strip()
        if not rf_id or rf_id in seen_ids:
            return
        if rf_id in by_id:
            by_id[rf_id] = _merge_entry_metadata(by_id[rf_id], entry)
        else:
            by_id[rf_id] = dict(entry)
            order.append(rf_id)

    search_variants: List[Dict[str, Any]] = [
        {
            "limit": 250,
            "fields": ["id", "name", "split", "annotations", "labels"],
            "annotation_job": True,
            "annotation_job_id": job_id,
        },
        {
            "limit": 250,
            "fields": ["id", "name", "split", "annotations", "labels"],
            "annotation_job_id": job_id,
            "in_dataset": True,
        },
        {
            "limit": 250,
            "fields": ["id", "name", "split", "annotations", "labels"],
            "annotation_job_id": job_id,
            "in_dataset": False,
        },
    ]

    for search_kwargs in search_variants:
        try:
            for page in project.search_all(**search_kwargs):
                for entry in page:
                    if isinstance(entry, dict) and entry.get("id"):
                        _record(entry)
        except Exception:
            continue

    # Always also paginate `/annotation-jobs/{job_id}/images` so annotated/completed images are never missed
    job_images_found = False
    if hasattr(project, "get_annotation_job_images"):
        cursor: Optional[str] = None
        for _ in range(100):
            try:
                res = project.get_annotation_job_images(job_id, limit=50, after=cursor)
            except Exception as exc:
                print(f"[roboflow] get_annotation_job_images({job_id}) failed: {exc}", flush=True)
                break
            if isinstance(res, dict):
                rows = (
                    res.get("images")
                    or res.get("imageIds")
                    or res.get("results")
                    or res.get("data")
                    or []
                )
                next_cur = res.get("nextCursor") or res.get("next_cursor") or res.get("after")
            elif isinstance(res, list):
                rows = res
                next_cur = None
            else:
                break

            if not rows:
                break

            job_images_found = True
            for row in rows:
                if isinstance(row, str):
                    rf_id = row.strip()
                    entry: Dict[str, Any] = {"id": rf_id}
                elif isinstance(row, dict):
                    rf_id = str(row.get("id") or row.get("imageId") or row.get("image_id") or "").strip()
                    entry = {**row, "id": rf_id}
                else:
                    continue
                if rf_id:
                    _record(entry)

            if not next_cur or str(next_cur) == cursor:
                break
            cursor = str(next_cur)

    if not job_images_found and hasattr(project, "get_annotation_job"):
        try:
            detail = project.get_annotation_job(job_id)
            raw_job = detail.get("job", detail) if isinstance(detail, dict) else {}
            if isinstance(raw_job, dict):
                inline_rows = raw_job.get("images") or raw_job.get("imageIds") or []
                if isinstance(inline_rows, list):
                    for row in inline_rows:
                        if isinstance(row, str) and row.strip():
                            _record({"id": row.strip()})
                        elif isinstance(row, dict):
                            rf_id = str(row.get("id") or row.get("imageId") or row.get("image_id") or "").strip()
                            if rf_id:
                                _record({**row, "id": rf_id})
        except Exception as exc:
            print(f"[roboflow] get_annotation_job({job_id}) inline images fallback failed: {exc}", flush=True)

    candidates = [by_id[rf_id] for rf_id in order]

    # Hydrate annotation metadata only for entries whose status is unknown when filtering annotated/unannotated
    if annotation_status_filter in ("annotated", "unannotated"):
        unknown_entries = [
            e for e in candidates if _entry_annotation_count(e) is None
        ]
        if unknown_entries:
            def _hydrate_status(ent: Dict[str, Any]) -> None:
                try:
                    ent["_cached_info"] = project.image(str(ent["id"]))
                except Exception:
                    ent["_cached_info"] = {}

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(_hydrate_status, unknown_entries))

        filtered: List[Dict[str, Any]] = []
        for ent in candidates:
            cnt = _entry_annotation_count(ent) or 0
            if annotation_status_filter == "annotated" and cnt <= 0:
                continue
            if annotation_status_filter == "unannotated" and cnt > 0:
                continue
            filtered.append(ent)
        candidates = filtered

    collected: List[Dict[str, Any]] = []
    for ent in candidates:
        rf_id = str(ent["id"])
        seen_ids.add(rf_id)
        collected.append(ent)
        if max_images > 0 and len(collected) >= max_images:
            break

    return collected


def _collect_entries_via_fallbacks(
    *,
    api_key: str,
    project: Any,
    key_ws: str,
    proj_ws: str,
    proj_slug: str,
    fetch_scope: str,
    annotation_status_filter: str,
    source_batch_id: Optional[str],
    max_images: int,
    orig_err: Exception,
    ws: Any = None,
) -> List[Dict[str, Any]]:
    """Fallback image collection when project.search_all fails (workspace mismatch or /search restriction)."""
    if key_ws and proj_ws and key_ws.lower() != proj_ws.lower():
        raise RuntimeError(
            f"API key belongs to workspace '{key_ws}', not '{proj_ws}'. "
            f"Paste the Private API Key for workspace '{proj_ws}' (Roboflow Settings -> API Keys) "
            f"to fetch images from '{proj_slug}'. (Roboflow error: {orig_err})"
        ) from orig_err

    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    fallback_notes: List[str] = []

    def _record(entry: Dict[str, Any]) -> None:
        rf_id = str(entry.get("id") or "").strip()
        if not rf_id:
            return
        if rf_id in by_id:
            by_id[rf_id] = _merge_entry_metadata(by_id[rf_id], entry)
        else:
            by_id[rf_id] = dict(entry)
            order.append(rf_id)

    # Tier 1: workspace-level RoboQL search (`/search/v1`)
    ws_slugs = [slug for slug, _ in _extract_ws_project_entries(ws)] if ws is not None else []
    ws_queries: List[Tuple[str, bool]] = []
    for prefix in (f"project:{proj_slug}", f"dataset:{proj_slug}"):
        q_parts = [prefix]
        if fetch_scope == "dataset":
            q_parts.append("in-dataset:true")
        elif fetch_scope == "batch" and source_batch_id:
            q_parts.append(f"batch:{source_batch_id}")
        ws_queries.append((" ".join(q_parts), False))

    if len(ws_slugs) <= 1:
        unfiltered_q = "in-dataset:true" if fetch_scope == "dataset" else ""
        ws_queries.append((unfiltered_q, True))

    for query, is_unfiltered in ws_queries:
        if order:
            break
        try:
            token: Optional[str] = None
            for _ in range(50):
                res = rfapi.workspace_search(
                    api_key.strip(),
                    proj_ws,
                    query=query,
                    page_size=50,
                    fields=["id", "name", "split", "annotations", "labels"],
                    continuation_token=token,
                )
                results = res.get("results") or [] if isinstance(res, dict) else []
                for ent in results:
                    if not isinstance(ent, dict) or not ent.get("id"):
                        continue
                    if is_unfiltered and len(ws_slugs) > 1:
                        ent_proj = str(ent.get("project") or "").strip().lower()
                        ent_projs = [str(p).strip().lower() for p in (ent.get("projects") or [])] if isinstance(ent.get("projects"), list) else []
                        if ent_proj and ent_proj != proj_slug.lower() and proj_slug.lower() not in ent_projs:
                            continue
                    _record(ent)
                token = res.get("continuationToken") if isinstance(res, dict) else None
                if not token or not results:
                    break
        except Exception as ws_exc:
            msg = f"workspace_search({query!r}) failed: {ws_exc}"
            print(f"[roboflow] {msg}", flush=True)
            fallback_notes.append(msg)

    # Tier 2a: annotation batches endpoints (`/annotation-batches`)
    if not order and hasattr(project, "get_annotation_batches") and hasattr(
        project, "get_annotation_batch_images"
    ):
        batch_ids: List[str] = []
        if fetch_scope == "batch" and source_batch_id:
            batch_ids = [source_batch_id]
        else:
            try:
                raw_ab = project.get_annotation_batches(limit=50, show_empty=True)
                rows = (
                    raw_ab.get("batches")
                    or raw_ab.get("annotationBatches")
                    or raw_ab.get("data")
                    or []
                    if isinstance(raw_ab, dict)
                    else (raw_ab if isinstance(raw_ab, list) else [])
                )
                batch_ids = [
                    str(b.get("id") or b.get("batchId") or "").strip()
                    for b in rows
                    if isinstance(b, dict) and (b.get("id") or b.get("batchId"))
                ]
            except Exception as ab_exc:
                msg = f"get_annotation_batches failed: {ab_exc}"
                print(f"[roboflow] {msg}", flush=True)
                fallback_notes.append(msg)
                batch_ids = []

        for bid in batch_ids:
            cursor: Optional[str] = None
            for _ in range(100):
                try:
                    res = project.get_annotation_batch_images(
                        bid, limit=50, after=cursor
                    )
                except Exception as abi_exc:
                    msg = f"get_annotation_batch_images({bid}) failed: {abi_exc}"
                    print(f"[roboflow] {msg}", flush=True)
                    fallback_notes.append(msg)
                    break
                rows = (
                    res.get("images")
                    or res.get("imageIds")
                    or res.get("results")
                    or res.get("data")
                    or []
                    if isinstance(res, dict)
                    else (res if isinstance(res, list) else [])
                )
                next_cur = (
                    res.get("nextCursor")
                    or res.get("next_cursor")
                    or res.get("after")
                    if isinstance(res, dict)
                    else None
                )
                if not rows:
                    break
                for row in rows:
                    if isinstance(row, str):
                        _record({"id": row.strip()})
                    elif isinstance(row, dict):
                        rid = str(
                            row.get("id")
                            or row.get("imageId")
                            or row.get("image_id")
                            or ""
                        ).strip()
                        if rid:
                            _record({**row, "id": rid})
                if not next_cur or str(next_cur) == cursor:
                    break
                cursor = str(next_cur)

    # Tier 2b: project batches endpoints (`/batches` -> `/batches/{batch_id}`)
    if not order and hasattr(project, "get_batch"):
        legacy_batch_ids: List[str] = []
        if fetch_scope == "batch" and source_batch_id:
            legacy_batch_ids = [source_batch_id]
        elif hasattr(project, "get_batches"):
            try:
                raw_batches = project.get_batches()
                rows = (
                    raw_batches.get("batches", [])
                    if isinstance(raw_batches, dict)
                    else (raw_batches if isinstance(raw_batches, list) else [])
                )
                legacy_batch_ids = [
                    str(b.get("id") or b.get("batch_id") or "").strip()
                    for b in rows
                    if isinstance(b, dict) and (b.get("id") or b.get("batch_id"))
                ]
            except Exception as gb_exc:
                msg = f"get_batches failed: {gb_exc}"
                print(f"[roboflow] {msg}", flush=True)
                fallback_notes.append(msg)

        nested_job_ids: List[str] = []
        for bid in legacy_batch_ids:
            try:
                detail = project.get_batch(bid)
                raw_b = detail.get("batch", detail) if isinstance(detail, dict) else {}
                if isinstance(raw_b, dict):
                    img_rows = raw_b.get("images") or raw_b.get("imageIds") or raw_b.get("items") or []
                    if isinstance(img_rows, list):
                        for row in img_rows:
                            if isinstance(row, str) and row.strip():
                                _record({"id": row.strip()})
                            elif isinstance(row, dict):
                                rid = str(row.get("id") or row.get("imageId") or row.get("image_id") or "").strip()
                                if rid:
                                    _record({**row, "id": rid})
                    b_jobs = raw_b.get("jobs") or []
                    if isinstance(b_jobs, list):
                        for bj in b_jobs:
                            if isinstance(bj, str) and bj.strip():
                                nested_job_ids.append(bj.strip())
                            elif isinstance(bj, dict):
                                jid = str(bj.get("id") or bj.get("jobId") or "").strip()
                                if jid:
                                    nested_job_ids.append(jid)
            except Exception as gbi_exc:
                msg = f"get_batch({bid}) failed: {gbi_exc}"
                print(f"[roboflow] {msg}", flush=True)
                fallback_notes.append(msg)

        for jid in nested_job_ids:
            for ent in _collect_job_entries(
                project,
                jid,
                max_images=0,
                seen_ids=set(),
                annotation_status_filter="all",
            ):
                _record(ent)

    # Tier 3: project annotation jobs
    if not order and fetch_scope != "batch":
        for aj in _list_annotation_jobs(project):
            for ent in _collect_job_entries(
                project,
                aj.id,
                max_images=0,
                seen_ids=set(),
                annotation_status_filter="all",
            ):
                _record(ent)

    if not order:
        notes_suffix = f" (Fallbacks tried: {'; '.join(fallback_notes[:3])})" if fallback_notes else ""
        raise RuntimeError(
            f"Roboflow search failed for '{proj_ws}/{proj_slug}' (API key workspace: '{key_ws or '?'}') "
            f"and no images were returned by fallback endpoints: {orig_err}{notes_suffix}"
        ) from orig_err

    candidates = [by_id[rf_id] for rf_id in order]
    need_ann_filter = fetch_scope in ("unannotated", "annotated") or (
        fetch_scope == "dataset" and annotation_status_filter in ("annotated", "unannotated")
    )
    if need_ann_filter:
        unknown_entries = [
            e for e in candidates if _entry_annotation_count(e) is None
        ]
        if unknown_entries:
            def _hydrate(ent: Dict[str, Any]) -> None:
                try:
                    ent["_cached_info"] = project.image(str(ent["id"]))
                except Exception:
                    ent["_cached_info"] = {}

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(_hydrate, unknown_entries))

        filtered: List[Dict[str, Any]] = []
        for ent in candidates:
            cnt = _entry_annotation_count(ent) or 0
            if fetch_scope == "unannotated" and cnt > 0:
                continue
            if fetch_scope == "annotated" and cnt <= 0:
                continue
            if fetch_scope == "dataset":
                if annotation_status_filter == "annotated" and cnt <= 0:
                    continue
                if annotation_status_filter == "unannotated" and cnt > 0:
                    continue
            filtered.append(ent)
        candidates = filtered

    return candidates[:max_images] if max_images > 0 else candidates


def _fetch_items_from_version_export(
    *,
    project: Any,
    ws_slug: str,
    canonical_slug: str,
    classes: List[str],
    class_colors: List[ClassColorEntry],
    job_id: str,
    job_dir: Path,
    fetch_scope: str,
    annotation_status_filter: str,
    max_images: int,
    batch_size: int,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Tier 4 fallback: download the latest generated Dataset Version ZIP when /search is blocked."""
    if not hasattr(project, "versions"):
        return None
    try:
        versions = list(project.versions() or [])
    except Exception as exc:
        print(f"[roboflow] project.versions() failed for {ws_slug}/{canonical_slug}: {exc}", flush=True)
        return None
    if not versions:
        print(f"[roboflow] no generated dataset versions available for {ws_slug}/{canonical_slug}", flush=True)
        return None

    def _ver_num(v: Any) -> int:
        try:
            return int(str(getattr(v, "version", 0) or 0))
        except Exception:
            return 0

    latest_ver = sorted(versions, key=_ver_num, reverse=True)[0]
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    with tempfile.TemporaryDirectory(prefix="rf_export_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        export_fmt = "yolov8"
        try:
            latest_ver.download("yolov8", location=str(tmp_path), overwrite=True)
        except Exception as y_exc:
            print(f"[roboflow] version.download('yolov8') failed ({y_exc}); trying 'coco'", flush=True)
            export_fmt = "coco"
            try:
                latest_ver.download("coco", location=str(tmp_path), overwrite=True)
            except Exception as c_exc:
                print(f"[roboflow] version.download('coco') failed: {c_exc}", flush=True)
                return None

        discovered: List[Tuple[Path, str, int]] = []
        if export_fmt == "yolov8":
            for split_name in ("train", "valid", "test"):
                img_dir = tmp_path / split_name / "images"
                lbl_dir = tmp_path / split_name / "labels"
                if not img_dir.is_dir():
                    continue
                for img_file in sorted(img_dir.iterdir()):
                    if not img_file.is_file() or img_file.suffix.lower() not in img_exts:
                        continue
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    ann_count = 0
                    if lbl_file.is_file():
                        ann_count = sum(
                            1
                            for ln in lbl_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                            if ln.strip()
                        )
                    discovered.append((img_file, split_name, ann_count))
        else:
            for split_name in ("train", "valid", "test"):
                split_dir = tmp_path / split_name
                if not split_dir.is_dir():
                    continue
                ann_by_fname: Dict[str, int] = {}
                coco_json = split_dir / "_annotations.coco.json"
                if coco_json.is_file():
                    try:
                        coco_data = json.loads(coco_json.read_text(encoding="utf-8"))
                        id_to_fname = {
                            im.get("id"): str(im.get("file_name") or "")
                            for im in (coco_data.get("images") or [])
                            if isinstance(im, dict)
                        }
                        for ann in (coco_data.get("annotations") or []):
                            if isinstance(ann, dict):
                                fn = id_to_fname.get(ann.get("image_id"), "")
                                if fn:
                                    ann_by_fname[fn] = ann_by_fname.get(fn, 0) + 1
                    except Exception:
                        pass
                for img_file in sorted(split_dir.iterdir()):
                    if not img_file.is_file() or img_file.suffix.lower() not in img_exts:
                        continue
                    discovered.append((img_file, split_name, ann_by_fname.get(img_file.name, 0)))

        filtered: List[Tuple[Path, str, int]] = []
        for img_file, split_name, ann_count in discovered:
            if fetch_scope == "unannotated" and ann_count > 0:
                continue
            if fetch_scope == "annotated" and ann_count <= 0:
                continue
            if fetch_scope == "dataset":
                if annotation_status_filter == "annotated" and ann_count <= 0:
                    continue
                if annotation_status_filter == "unannotated" and ann_count > 0:
                    continue
            filtered.append((img_file, split_name, ann_count))

        if max_images > 0:
            filtered = filtered[:max_images]
        if not filtered:
            return None

        raw_dir = job_dir / "raw_images"
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_batch_size = max(1, int(batch_size or aa_config.DEFAULT_BATCH_SIZE))
        total = len(filtered)
        taken_names: set[str] = set()
        items_out: List[Dict[str, Any]] = []

        for idx, (src_img, split_name, ann_count) in enumerate(filtered):
            seq_num = idx + 1
            batch_num = (idx // safe_batch_size) + 1
            batch_id = f"batch-{batch_num:02d}"
            orig_name = _safe_stem_filename(src_img.name)
            if orig_name in taken_names:
                orig_name = f"{seq_num:04d}_{orig_name}"
            taken_names.add(orig_name)
            save_name = f"{seq_num:04d}_{orig_name}"
            dest_path = raw_dir / save_name
            shutil.copy2(src_img, dest_path)

            with Image.open(dest_path) as im:
                w, h = im.size

            items_out.append(
                {
                    "id": f"{job_id}-item-{seq_num:04d}",
                    "jobId": job_id,
                    "batchId": batch_id,
                    "batchIndex": batch_num,
                    "seqIndex": seq_num,
                    "roboflowImageId": src_img.stem,
                    "filename": orig_name,
                    "localPath": str(dest_path),
                    "imageUrl": f"/api/v1/auto-annotate/files/{job_id}/raw_images/{save_name}",
                    "width": w,
                    "height": h,
                    "split": split_name,
                    "sourceBatchId": None,
                    "existingAnnotationsCount": ann_count,
                    "inferenceStatus": "pending",
                    "detections": [],
                    "latencySec": 0.0,
                    "reviewStatus": "pending",
                    "rejectReason": None,
                    "pushStatus": "not_pushed",
                }
            )
            if progress_cb:
                progress_cb(seq_num, total, orig_name)

        return {
            "workspaceSlug": ws_slug,
            "projectSlug": canonical_slug,
            "projectName": str(getattr(project, "name", "") or canonical_slug),
            "projectType": str(getattr(project, "type", "") or "object-detection"),
            "classes": classes,
            "classColors": class_colors,
            "items": items_out,
            "batches": _build_batch_summaries(len(items_out), safe_batch_size),
        }


def fetch_project_images(
    *,
    api_key: str,
    workspace_slug: Optional[str],
    project_slug: str,
    job_id: str,
    job_dir: Path,
    fetch_scope: str,
    annotation_status_filter: str = "all",
    source_batch_id: Optional[str],
    source_annotation_job_id: Optional[str] = None,
    assignee_email: Optional[str] = None,
    max_images: int,
    batch_size: int = aa_config.DEFAULT_BATCH_SIZE,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Fetch images from Roboflow read-only, save natively on disk, and chunk into batches of `batch_size`."""
    _, ws, project, key_ws, ws_slug, canonical_slug = _resolve_workspace_and_project(
        api_key=api_key,
        project_slug=project_slug,
        workspace_slug=workspace_slug,
    )
    classes, class_colors = _extract_project_classes_and_colors(project)

    raw_items: List[Dict[str, Any]] = []
    seen_rf_ids: set[str] = set()

    if fetch_scope == "annotation_job":
        target_job_ids: List[str] = []
        clean_job_id = (source_annotation_job_id or "").strip()
        clean_email = (assignee_email or "").strip().lower()
        if clean_job_id:
            target_job_ids = [clean_job_id]
        elif clean_email:
            all_ann_jobs = _list_annotation_jobs(project)
            for aj in all_ann_jobs:
                lab = (aj.labelerEmail or "").strip().lower()
                rev = (aj.reviewerEmail or "").strip().lower()
                if clean_email in lab or clean_email in rev:
                    target_job_ids.append(aj.id)
        if not target_job_ids:
            raise RuntimeError(
                "No Roboflow annotation job matched. Pick an Annotation Job from the list or check the Assignee Email ID."
            )
        for rf_ann_jid in target_job_ids:
            remaining = max(0, max_images - len(raw_items)) if max_images > 0 else 0
            entries = _collect_job_entries(
                project,
                rf_ann_jid,
                max_images=remaining,
                seen_ids=seen_rf_ids,
                annotation_status_filter=annotation_status_filter,
            )
            raw_items.extend(entries)
            if max_images > 0 and len(raw_items) >= max_images:
                raw_items = raw_items[:max_images]
                break
    else:
        search_kwargs: Dict[str, Any] = {
            "limit": 250,
            "fields": ["id", "name", "split", "annotations", "labels"],
        }
        if fetch_scope == "batch" and source_batch_id:
            search_kwargs["batch"] = True
            search_kwargs["batch_id"] = source_batch_id
        elif fetch_scope == "dataset":
            search_kwargs["in_dataset"] = True

        try:
            for page in project.search_all(**search_kwargs):
                for entry in page:
                    if not isinstance(entry, dict) or not entry.get("id"):
                        continue
                    ann_count = _entry_annotation_count(entry) or 0
                    if fetch_scope == "unannotated" and ann_count > 0:
                        continue
                    if fetch_scope == "annotated" and ann_count <= 0:
                        continue
                    if fetch_scope == "dataset":
                        if annotation_status_filter == "annotated" and ann_count <= 0:
                            continue
                        if annotation_status_filter == "unannotated" and ann_count > 0:
                            continue
                    raw_items.append(entry)
                    if max_images > 0 and len(raw_items) >= max_images:
                        break
                if max_images > 0 and len(raw_items) >= max_images:
                    break
        except Exception as search_exc:
            print(
                f"[roboflow] project.search_all failed for {ws_slug}/{canonical_slug}: {search_exc}; trying fallbacks",
                flush=True,
            )
            try:
                raw_items = _collect_entries_via_fallbacks(
                    api_key=api_key,
                    project=project,
                    key_ws=key_ws,
                    proj_ws=ws_slug,
                    proj_slug=canonical_slug,
                    fetch_scope=fetch_scope,
                    annotation_status_filter=annotation_status_filter,
                    source_batch_id=source_batch_id,
                    max_images=max_images,
                    orig_err=search_exc,
                    ws=ws,
                )
            except RuntimeError as fb_err:
                is_ws_mismatch = bool(key_ws and ws_slug and key_ws.lower() != ws_slug.lower())
                if (
                    not is_ws_mismatch
                    and fetch_scope in ("dataset", "annotated", "all", "unannotated")
                ):
                    exported = _fetch_items_from_version_export(
                        project=project,
                        ws_slug=ws_slug,
                        canonical_slug=canonical_slug,
                        classes=classes,
                        class_colors=class_colors,
                        job_id=job_id,
                        job_dir=job_dir,
                        fetch_scope=fetch_scope,
                        annotation_status_filter=annotation_status_filter,
                        max_images=max_images,
                        batch_size=batch_size,
                        progress_cb=progress_cb,
                    )
                    if exported and exported.get("items"):
                        return exported
                hint = ""
                if api_key.strip().startswith("rf_"):
                    hint = (
                        " Hint: keys starting with 'rf_' are usually Roboflow Publishable Keys, "
                        "which cannot access Search or Dataset APIs — use your Private API Key from Roboflow Settings -> API Keys."
                    )
                elif not is_ws_mismatch:
                    hint = (
                        " Hint: verify your Private API Key has access to this project/folder, "
                        "or generate a Dataset Version in Roboflow so Dataset Version export can be used."
                    )
                raise RuntimeError(f"{fb_err}{hint}") from fb_err

    if not raw_items:
        raise RuntimeError(
            "No matching images found in Roboflow for the selected scope."
        )

    raw_dir = job_dir / "raw_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_batch_size = max(1, int(batch_size or aa_config.DEFAULT_BATCH_SIZE))

    # Resolve original URLs and download into raw_images/
    total = len(raw_items)
    taken_lock = threading.Lock()
    taken_names: Dict[str, str] = {}
    prepared: List[Optional[Dict[str, Any]]] = [None] * total
    completed = 0

    def _fetch_one(idx: int, item: Dict[str, Any]) -> None:
        nonlocal completed
        rf_id = str(item["id"])
        seq_num = idx + 1
        batch_num = (idx // safe_batch_size) + 1
        batch_id = f"batch-{batch_num:02d}"

        url = item.get("url") or (item.get("urls") or {}).get("original")
        raw_name = item.get("name")
        info: Optional[Dict[str, Any]] = (
            item.get("_cached_info") if isinstance(item.get("_cached_info"), dict) else None
        )
        if not url or not raw_name:
            if not info:
                info = project.image(rf_id)
            if not url:
                url = (info.get("urls") or {}).get("original") or info.get("url")
            if not raw_name and info.get("name"):
                raw_name = info.get("name")
        if not url:
            raise RuntimeError(f"Roboflow image {rf_id} has no downloadable URL.")

        orig_name = _safe_stem_filename(str(raw_name or f"{rf_id}.jpg"))
        with taken_lock:
            if taken_names.get(orig_name, rf_id) != rf_id:
                orig_name = f"{rf_id}_{orig_name}"
            taken_names[orig_name] = rf_id

        save_name = f"{seq_num:04d}_{orig_name}"
        dest_path = raw_dir / save_name

        _download_image_with_retry(str(url), dest_path)

        with Image.open(dest_path) as im:
            w, h = im.size

        split_source = item.get("split") or (info.get("split") if info else None) or "train"
        raw_split = str(split_source).strip().lower()
        split = raw_split if raw_split in ("train", "valid", "test") else "train"

        ann_src = item.get("annotations")
        if ann_src is None and info is not None:
            ann_src = (info.get("annotation") or {}).get("boxes") or []

        prepared[idx] = {
            "id": f"{job_id}-item-{seq_num:04d}",
            "jobId": job_id,
            "batchId": batch_id,
            "batchIndex": batch_num,
            "seqIndex": seq_num,
            "roboflowImageId": rf_id,
            "filename": orig_name,
            "localPath": str(dest_path),
            "imageUrl": f"/api/v1/auto-annotate/files/{job_id}/raw_images/{save_name}",
            "width": w,
            "height": h,
            "split": split,
            "sourceBatchId": (
                source_annotation_job_id
                if fetch_scope == "annotation_job"
                else (source_batch_id if fetch_scope == "batch" else None)
            ),
            "existingAnnotationsCount": len(ann_src or []),
            "inferenceStatus": "pending",
            "detections": [],
            "latencySec": 0.0,
            "reviewStatus": "pending",
            "rejectReason": None,
            "pushStatus": "not_pushed",
        }
        completed += 1
        if progress_cb:
            progress_cb(completed, total, orig_name)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(_fetch_one, idx, item) for idx, item in enumerate(raw_items)
        ]
        for fut in futures:
            fut.result()

    items_out = [p for p in prepared if p is not None]
    batches_out = _build_batch_summaries(len(items_out), safe_batch_size)

    return {
        "workspaceSlug": ws_slug,
        "projectSlug": canonical_slug,
        "projectName": str(getattr(project, "name", "") or canonical_slug),
        "projectType": str(getattr(project, "type", "") or "object-detection"),
        "classes": classes,
        "classColors": class_colors,
        "items": items_out,
        "batches": batches_out,
    }


def _build_batch_summaries(total_items: int, batch_size: int) -> List[Dict[str, Any]]:
    safe_batch_size = max(1, int(batch_size or aa_config.DEFAULT_BATCH_SIZE))
    num_batches = (total_items + safe_batch_size - 1) // safe_batch_size
    batches_out: List[Dict[str, Any]] = []
    for b_num in range(1, num_batches + 1):
        start_i = (b_num - 1) * safe_batch_size + 1
        end_i = min(b_num * safe_batch_size, total_items)
        batches_out.append(
            {
                "batchId": f"batch-{b_num:02d}",
                "batchIndex": b_num,
                "label": f"Batch {b_num:02d} · {start_i}–{end_i}",
                "startIndex": start_i,
                "endIndex": end_i,
                "checked": False,
                "checkedAtIso": None,
            }
        )
    return batches_out


def clone_items_for_retry_job(
    *,
    source_job: AutoJobRecord,
    source_items_with_paths: Sequence[Dict[str, Any]],
    target_job_id: str,
    target_job_dir: Path,
    selected_item_ids: Optional[Sequence[str]] = None,
    batch_size: int = aa_config.DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Create a standalone local retry batch by cloning selected (default: rejected)
    already-downloaded frames from `source_job` without re-fetching from Roboflow.
    Preserves each frame's `roboflowImageId`, `split`, and `sourceBatchId` so accepted
    results can still be pushed right back into the same Roboflow project/job/dataset.
    """
    id_set = {str(iid).strip() for iid in (selected_item_ids or ()) if str(iid).strip()}
    chosen: List[Dict[str, Any]] = []
    for entry in source_items_with_paths:
        it: AutoItemRecord = entry["item"]
        if id_set:
            if it.id in id_set:
                chosen.append(entry)
        elif it.reviewStatus == "rejected":
            chosen.append(entry)

    if not chosen:
        raise RuntimeError("No rejected or selected images found in the source job to re-annotate.")

    raw_dir = target_job_dir / "raw_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_batch_size = max(1, int(batch_size or aa_config.DEFAULT_BATCH_SIZE))
    items_out: List[Dict[str, Any]] = []

    for idx, entry in enumerate(chosen):
        src_item: AutoItemRecord = entry["item"]
        src_path = Path(entry["localPath"])
        if not src_path.is_file():
            raise FileNotFoundError(
                f"Source local image missing on disk for '{src_item.filename}': {src_path}"
            )
        seq_num = idx + 1
        batch_num = (idx // safe_batch_size) + 1
        batch_id = f"batch-{batch_num:02d}"
        orig_name = _safe_stem_filename(src_item.filename or src_path.name)
        save_name = f"{seq_num:04d}_{orig_name}"
        dest_path = raw_dir / save_name
        shutil.copy2(src_path, dest_path)

        items_out.append(
            {
                "id": f"{target_job_id}-item-{seq_num:04d}",
                "jobId": target_job_id,
                "batchId": batch_id,
                "batchIndex": batch_num,
                "seqIndex": seq_num,
                "roboflowImageId": src_item.roboflowImageId,
                "filename": orig_name,
                "localPath": str(dest_path),
                "imageUrl": f"/api/v1/auto-annotate/files/{target_job_id}/raw_images/{save_name}",
                "width": int(src_item.width),
                "height": int(src_item.height),
                "split": src_item.split,
                "sourceBatchId": src_item.sourceBatchId,
                "existingAnnotationsCount": int(src_item.existingAnnotationsCount or 0),
                "inferenceStatus": "pending",
                "detections": [],
                "latencySec": 0.0,
                "reviewStatus": "pending",
                "rejectReason": None,
                "pushStatus": "not_pushed",
            }
        )

    return {
        "workspaceSlug": source_job.workspaceSlug,
        "projectSlug": source_job.projectSlug,
        "projectName": source_job.projectName,
        "projectType": source_job.projectType,
        "classes": list(source_job.classes),
        "classColors": list(source_job.classColors),
        "items": items_out,
        "batches": _build_batch_summaries(len(items_out), safe_batch_size),
    }


def _accepted_disk_stem(item: AutoItemRecord) -> str:
    stem = Path(_safe_stem_filename(item.filename)).stem
    return f"{item.batchId}__{item.seqIndex:04d}__{stem}"


def _ordered_class_map(
    job: AutoJobRecord, items: Sequence[AutoItemRecord]
) -> Dict[str, int]:
    ordered: List[str] = []
    seen_lower: Dict[str, int] = {}
    for cls_name in job.classes:
        clean = cls_name.strip()
        if clean and clean.lower() not in seen_lower:
            seen_lower[clean.lower()] = len(ordered)
            ordered.append(clean)
    for it in items:
        if it.reviewStatus != "accepted":
            continue
        for det in it.detections:
            clean = det.label.strip() or "object"
            if clean.lower() not in seen_lower:
                seen_lower[clean.lower()] = len(ordered)
                ordered.append(clean)
    return seen_lower


def detections_to_yolo_lines(
    detections: Sequence[AutoDetection],
    class_lower_to_id: Dict[str, int],
    *,
    output_mode: str,
    project_type: str,
) -> List[str]:
    """Convert normalized 0..1000 bbox_2d / polygon detections into YOLO / YOLO-seg .txt lines."""
    use_polygon = (
        output_mode == "polygon"
        or project_type == "instance-segmentation"
        or (output_mode == "both" and any(d.polygon and len(d.polygon) >= 3 for d in detections))
    )
    lines: List[str] = []
    for det in detections:
        cls_key = (det.label.strip() or "object").lower()
        cls_id = class_lower_to_id.get(cls_key, 0)
        if use_polygon:
            poly = det.polygon
            if not poly or len(poly) < 3:
                if len(det.bbox_2d) != 4:
                    continue
                x1, y1, x2, y2 = det.bbox_2d
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            coords = " ".join(
                f"{min(max(px / 1000.0, 0.0), 1.0):.6f} {min(max(py / 1000.0, 0.0), 1.0):.6f}"
                for px, py in poly
            )
            lines.append(f"{cls_id} {coords}")
        else:
            if len(det.bbox_2d) != 4:
                continue
            x1, y1, x2, y2 = [min(max(v / 1000.0, 0.0), 1.0) for v in det.bbox_2d]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(0.001, x2 - x1)
            bh = max(0.001, y2 - y1)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _resolve_ordered_class_map(
    job: AutoJobRecord, items: Sequence[AutoItemRecord]
) -> Tuple[Dict[str, int], List[str]]:
    class_lower_to_id = _ordered_class_map(job, items)
    ordered_names = [""] * len(class_lower_to_id)
    orig_case_by_lower: Dict[str, str] = {}
    for c in job.classes:
        clean = c.strip()
        if clean and clean.lower() not in orig_case_by_lower:
            orig_case_by_lower[clean.lower()] = clean
    for it in items:
        for d in it.detections:
            clean = d.label.strip() or "object"
            if clean.lower() not in orig_case_by_lower:
                orig_case_by_lower[clean.lower()] = clean
    for lower_k, cid in class_lower_to_id.items():
        ordered_names[cid] = orig_case_by_lower.get(lower_k, lower_k)
    return class_lower_to_id, ordered_names


def _write_accepted_metadata(
    accepted_dir: Path,
    job: AutoJobRecord,
    all_items: Sequence[AutoItemRecord],
    ordered_names: List[str],
) -> None:
    manifest_items: List[Dict[str, Any]] = []
    for item in all_items:
        if item.reviewStatus != "accepted":
            continue
        stem = _accepted_disk_stem(item)
        ext = Path(_safe_stem_filename(item.filename)).suffix or ".jpg"
        manifest_items.append(
            {
                "itemId": item.id,
                "batchId": item.batchId,
                "batchIndex": item.batchIndex,
                "seqIndex": item.seqIndex,
                "roboflowImageId": item.roboflowImageId,
                "filename": item.filename,
                "acceptedImageFile": f"images/{stem}{ext}",
                "acceptedLabelFile": f"labels/{stem}.txt",
                "split": item.split,
                "width": item.width,
                "height": item.height,
                "pushStatus": item.pushStatus,
                "detections": [d.model_dump() for d in item.detections],
            }
        )

    yaml_payload = {
        "path": ".",
        "train": "./images",
        "val": "./images",
        "nc": len(ordered_names),
        "names": {i: name for i, name in enumerate(ordered_names)},
    }
    (accepted_dir / "data.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8"
    )

    manifest_payload = {
        "jobId": job.id,
        "workspaceSlug": job.workspaceSlug,
        "sourceProject": job.projectSlug,
        "projectType": job.projectType,
        "outputMode": job.outputMode,
        "batchSize": job.batchSize,
        "acceptedCount": len(manifest_items),
        "classes": ordered_names,
        "batches": [b.model_dump() for b in job.batches],
        "acceptedItems": manifest_items,
    }
    (accepted_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )


def sync_single_item_to_disk(
    *,
    job: AutoJobRecord,
    item: AutoItemRecord,
    local_path: str,
    all_items: Sequence[AutoItemRecord],
) -> str:
    """Incrementally sync one reviewed item into `<job_dir>/accepted/` + refresh metadata."""
    accepted_dir = aa_config.AUTO_JOBS_DIR / job.id / "accepted"
    img_dir = accepted_dir / "images"
    lbl_dir = accepted_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_lower_to_id, ordered_names = _resolve_ordered_class_map(job, all_items)
    src_path = Path(local_path)
    stem = _accepted_disk_stem(item)
    ext = src_path.suffix or Path(_safe_stem_filename(item.filename)).suffix or ".jpg"
    out_img = img_dir / f"{stem}{ext}"
    out_lbl = lbl_dir / f"{stem}.txt"

    if item.reviewStatus == "accepted":
        if src_path.is_file() and not out_img.exists():
            shutil.copy2(src_path, out_img)
        lines = detections_to_yolo_lines(
            item.detections,
            class_lower_to_id,
            output_mode=job.outputMode,
            project_type=job.projectType,
        )
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
    else:
        out_img.unlink(missing_ok=True)
        out_lbl.unlink(missing_ok=True)

    _write_accepted_metadata(accepted_dir, job, all_items, ordered_names)
    return str(accepted_dir)


def sync_accepted_items_to_disk(
    *,
    job: AutoJobRecord,
    items_with_paths: Sequence[Dict[str, Any]],
) -> str:
    """
    Keep `<job_dir>/accepted/` in exact sync with database `review_status`:
    - Accepted items exist under `accepted/images/` & `accepted/labels/` with `<batch_id>__<seq>__<name>`
    - Rejected or Pending items are strictly absent from `accepted/`
    - `data.yaml` and `manifest.json` are always persisted cleanly
    """
    accepted_dir = aa_config.AUTO_JOBS_DIR / job.id / "accepted"
    img_dir = accepted_dir / "images"
    lbl_dir = accepted_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    all_items = [entry["item"] for entry in items_with_paths]
    class_lower_to_id, ordered_names = _resolve_ordered_class_map(job, all_items)

    expected_img_names: set[str] = set()
    expected_lbl_names: set[str] = set()

    for entry in items_with_paths:
        item: AutoItemRecord = entry["item"]
        src_path = Path(entry["localPath"])
        stem = _accepted_disk_stem(item)
        ext = src_path.suffix or ".jpg"
        out_img = img_dir / f"{stem}{ext}"
        out_lbl = lbl_dir / f"{stem}.txt"

        if item.reviewStatus == "accepted":
            expected_img_names.add(out_img.name)
            expected_lbl_names.add(out_lbl.name)
            if src_path.is_file() and not out_img.exists():
                shutil.copy2(src_path, out_img)
            lines = detections_to_yolo_lines(
                item.detections,
                class_lower_to_id,
                output_mode=job.outputMode,
                project_type=job.projectType,
            )
            out_lbl.write_text("\n".join(lines), encoding="utf-8")
        else:
            out_img.unlink(missing_ok=True)
            out_lbl.unlink(missing_ok=True)

    # Prune any stale files not in expected sets
    for existing in img_dir.iterdir():
        if existing.is_file() and existing.name not in expected_img_names:
            existing.unlink(missing_ok=True)
    for existing in lbl_dir.iterdir():
        if existing.is_file() and existing.name not in expected_lbl_names:
            existing.unlink(missing_ok=True)

    _write_accepted_metadata(accepted_dir, job, all_items, ordered_names)
    return str(accepted_dir)


def _prepare_accepted_dataset(
    *,
    job: AutoJobRecord,
    items_with_paths: Sequence[Dict[str, Any]],
    batch_id_filter: Optional[str] = None,
    include_empty: bool = True,
    only_unpushed: bool = False,
) -> Tuple[
    List[Dict[str, Any]],
    List[AutoItemRecord],
    Dict[str, int],
    List[str],
    int,
    int,
    int,
    int,
]:
    """Filter strictly Accepted items & resolve deterministic YOLO class map."""
    excluded_rejected = 0
    excluded_pending = 0
    skipped_empty = 0
    eligible_entries: List[Dict[str, Any]] = []

    for entry in items_with_paths:
        item: AutoItemRecord = entry["item"]
        if batch_id_filter and batch_id_filter != "all" and item.batchId != batch_id_filter:
            continue
        if item.reviewStatus == "rejected":
            excluded_rejected += 1
            continue
        if item.reviewStatus != "accepted":
            excluded_pending += 1
            continue
        if not include_empty and len(item.detections) == 0:
            skipped_empty += 1
            continue
        if only_unpushed and item.pushStatus == "pushed":
            continue
        eligible_entries.append(entry)

    included_items = [e["item"] for e in eligible_entries]
    null_count = sum(1 for it in included_items if len(it.detections) == 0)
    class_lower_to_id = _ordered_class_map(job, included_items)
    orig_case_by_lower: Dict[str, str] = {}
    for c in job.classes:
        clean = c.strip()
        if clean and clean.lower() not in orig_case_by_lower:
            orig_case_by_lower[clean.lower()] = clean
    for it in included_items:
        for d in it.detections:
            clean = d.label.strip() or "object"
            if clean.lower() not in orig_case_by_lower:
                orig_case_by_lower[clean.lower()] = clean
    ordered_names = [""] * len(class_lower_to_id)
    for lk, cid in class_lower_to_id.items():
        ordered_names[cid] = orig_case_by_lower.get(lk, lk)

    return (
        eligible_entries,
        included_items,
        class_lower_to_id,
        ordered_names,
        null_count,
        skipped_empty,
        excluded_rejected,
        excluded_pending,
    )


def build_accepted_zip(
    *,
    job: AutoJobRecord,
    items_with_paths: Sequence[Dict[str, Any]],
    target_project_name: str,
    batch_id_filter: Optional[str] = None,
    include_empty: bool = True,
) -> Tuple[Path, List[AutoItemRecord], int, int, int]:
    """
    Package strictly `reviewStatus == 'accepted'` items (annotated + null frames)
    into a single YOLOv8/YOLO-seg `.zip` persisted under `<job_dir>/accepted/`.
    Returns (zip_path, included_items, skipped_empty_count, excluded_rejected_count, excluded_pending_count).
    """
    accepted_dir = aa_config.AUTO_JOBS_DIR / job.id / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    (
        eligible_entries,
        included_items,
        class_lower_to_id,
        ordered_names,
        _null_count,
        skipped_empty,
        excluded_rejected,
        excluded_pending,
    ) = _prepare_accepted_dataset(
        job=job,
        items_with_paths=items_with_paths,
        batch_id_filter=batch_id_filter,
        include_empty=include_empty,
    )

    slug = _normalize_project_slug(target_project_name)
    suffix = f"_{batch_id_filter}" if batch_id_filter and batch_id_filter != "all" else ""
    zip_path = accepted_dir / f"{slug}{suffix}.zip"

    splits_present = {e["item"].split for e in eligible_entries}
    zip_yaml = {
        "train": "train/images" if "train" in splits_present or not splits_present else f"{next(iter(splits_present))}/images",
        "val": "valid/images" if "valid" in splits_present else ("train/images" if "train" in splits_present or not splits_present else f"{next(iter(splits_present))}/images"),
        **({"test": "test/images"} if "test" in splits_present else {}),
        "nc": len(ordered_names),
        "names": ordered_names,
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", yaml.safe_dump(zip_yaml, sort_keys=False))
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "jobId": job.id,
                    "sourceProject": job.projectSlug,
                    "targetProject": slug,
                    "batchFilter": batch_id_filter or "all",
                    "includedCount": len(included_items),
                    "excludedRejectedCount": excluded_rejected,
                    "excludedPendingCount": excluded_pending,
                    "items": [
                        {
                            "itemId": it.id,
                            "batchId": it.batchId,
                            "seqIndex": it.seqIndex,
                            "roboflowImageId": it.roboflowImageId,
                            "filename": it.filename,
                            "split": it.split,
                        }
                        for it in included_items
                    ],
                },
                indent=2,
            ),
        )
        for entry in eligible_entries:
            item = entry["item"]
            src_path = Path(entry["localPath"])
            if not src_path.is_file():
                continue
            stem = _accepted_disk_stem(item)
            ext = src_path.suffix or ".jpg"
            split = item.split if item.split in ("train", "valid", "test") else "train"
            zf.write(src_path, arcname=f"{split}/images/{stem}{ext}")
            lines = detections_to_yolo_lines(
                item.detections,
                class_lower_to_id,
                output_mode=job.outputMode,
                project_type=job.projectType,
            )
            zf.writestr(f"{split}/labels/{stem}.txt", "\n".join(lines))

    return (
        zip_path,
        included_items,
        skipped_empty,
        excluded_rejected,
        excluded_pending,
    )


def _resolve_source_batch_id(
    project: Any,
    job: AutoJobRecord,
    rf_ids: Sequence[str],
) -> Optional[str]:
    if job.sourceBatchId and job.sourceBatchId.strip():
        return job.sourceBatchId.strip()
    if job.sourceAnnotationJobId and hasattr(project, "get_annotation_job"):
        for getter_name in ("get_annotation_job", "get_annotation_job_admin"):
            getter = getattr(project, getter_name, None)
            if not getter:
                continue
            try:
                detail = getter(job.sourceAnnotationJobId)
                raw = detail.get("job", detail) if isinstance(detail, dict) else {}
                if isinstance(raw, dict):
                    bid = raw.get("batchId") or raw.get("sourceBatchId") or raw.get("batch")
                    if isinstance(bid, dict):
                        bid = bid.get("id")
                    if bid:
                        return str(bid).strip()
            except Exception:
                pass
    if rf_ids and hasattr(project, "image"):
        try:
            info = project.image(rf_ids[0])
            if isinstance(info, dict):
                bid = info.get("batch") or info.get("batchId")
                if not bid and isinstance(info.get("batches"), list) and info["batches"]:
                    first = info["batches"][0]
                    bid = first.get("id") if isinstance(first, dict) else first
                if bid:
                    return str(bid).strip()
        except Exception:
            pass
    try:
        raw_batches = project.get_batches()
        rows = (
            raw_batches.get("batches", [])
            if isinstance(raw_batches, dict)
            else (raw_batches if isinstance(raw_batches, list) else [])
        )
        if len(rows) == 1 and isinstance(rows[0], dict) and rows[0].get("id"):
            return str(rows[0]["id"]).strip()
    except Exception:
        pass
    return None


def _build_null_coco_json(
    *,
    item: AutoItemRecord,
    file_name: str,
    ordered_names: Sequence[str],
) -> str:
    """
    Build a Roboflow-compatible COCO Null payload matching `roboflow.util.folderparser`:
    includes `info`, `licenses`, `categories`, `images`, and the non-matching dummy
    annotation record so Roboflow's Annotations.js parser recognizes the file as COCO
    with 0 annotations for image `id=1` (registering the frame as Null).
    """
    cats = [
        {"id": i + 1, "name": cls_name, "supercategory": "none"}
        for i, cls_name in enumerate(ordered_names or ["object"])
    ]
    payload = {
        "info": {"description": "Verified Null (background) image"},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": file_name,
                "width": max(1, int(item.width or 1)),
                "height": max(1, int(item.height or 1)),
            }
        ],
        "categories": cats,
        "annotations": [
            {
                "id": 999999999,
                "image_id": 999999999,
                "category_id": 0,
                "area": 1,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
    }
    return json.dumps(payload)


def _build_null_voc_xml(*, item: AutoItemRecord, file_name: str) -> str:
    safe_name = (
        (file_name or "image.jpg")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    w = max(1, int(item.width or 1))
    h = max(1, int(item.height or 1))
    return (
        "<annotation>\n"
        "  <folder>dataset</folder>\n"
        f"  <filename>{safe_name}</filename>\n"
        f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>\n"
        "  <segmented>0</segmented>\n"
        "</annotation>"
    )


def _build_null_createml_json(*, file_name: str) -> str:
    return json.dumps([{"image": file_name, "annotations": []}])


def _push_entries_per_item(
    *,
    project: Any,
    job: AutoJobRecord,
    eligible_entries: Sequence[Dict[str, Any]],
    class_lower_to_id: Dict[str, int],
    ordered_names: List[str],
    upload_new_images: bool,
    batch_name: str,
    tags: List[str],
    add_to_dataset: Optional[bool],
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    Push verified annotations (and images when uploading to a new project) concurrently.
    Accepted frames with 0 detections are pushed as explicit Null annotations (COCO JSON
    with empty annotations array, falling back to empty YOLO .txt if needed).
    Returns (succeeded_item_ids, failed_item_id_and_err, succeeded_rf_image_ids).
    """
    safe_names = ordered_names or (list(job.classes) if job.classes else ["object"])
    labelmap = {int(i): name for i, name in enumerate(safe_names)}
    succeeded_ids: List[str] = []
    failed_items: List[Tuple[str, str]] = []
    succeeded_rf_ids: List[str] = []
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(prefix="rf_push_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        def _push_one(idx: int, entry: Dict[str, Any]) -> None:
            item: AutoItemRecord = entry["item"]
            src_path = Path(entry["localPath"])
            stem = _accepted_disk_stem(item)
            txt_path = tmp_path / f"{idx:04d}_{stem}.txt"
            lines = detections_to_yolo_lines(
                item.detections,
                class_lower_to_id,
                output_mode=job.outputMode,
                project_type=job.projectType,
            )
            is_null = len(lines) == 0
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            split = item.split if item.split in ("train", "valid", "test") else "train"

            try:
                rf_id = (item.roboflowImageId or "").strip()
                if not upload_new_images and not rf_id:
                    raise ValueError(
                        f"Missing original Roboflow image ID for '{item.filename}'; "
                        "refusing to upload a duplicate copy with a new ID."
                    )
                if is_null:
                    target_fname = (
                        src_path.name
                        if upload_new_images
                        else (item.filename or src_path.name)
                    )
                    null_candidates = [
                        {
                            "name": "annotation.coco.json",
                            "rawText": _build_null_coco_json(
                                item=item,
                                file_name=target_fname,
                                ordered_names=safe_names,
                            ),
                        },
                        {
                            "name": f"{stem}.xml",
                            "rawText": _build_null_voc_xml(
                                item=item,
                                file_name=target_fname,
                            ),
                        },
                        {
                            "name": "annotation.createml.json",
                            "rawText": _build_null_createml_json(
                                file_name=target_fname,
                            ),
                        },
                    ]
                    first_err: Optional[Exception] = None
                    if not upload_new_images:
                        for cand in null_candidates:
                            try:
                                project.save_annotation(
                                    annotation_path=cand,
                                    annotation_labelmap=None,
                                    image_id=rf_id,
                                    job_name=batch_name,
                                    is_prediction=False,
                                    annotation_overwrite=True,
                                    num_retry_uploads=1,
                                    add_to_dataset=add_to_dataset,
                                )
                                first_err = None
                                break
                            except Exception as exc:
                                if first_err is None:
                                    first_err = exc
                        if first_err is not None:
                            raise first_err
                        resolved_rf_id = rf_id
                    else:
                        if not src_path.is_file():
                            raise FileNotFoundError(f"Missing local image: {src_path}")
                        res = None
                        for cand in null_candidates:
                            try:
                                res = project.single_upload(
                                    image_path=str(src_path),
                                    annotation_path=cand,
                                    annotation_labelmap=None,
                                    split=split,
                                    num_retry_uploads=1,
                                    batch_name=batch_name,
                                    tag_names=tags,
                                    is_prediction=False,
                                    annotation_overwrite=True,
                                    add_to_dataset=add_to_dataset,
                                )
                                first_err = None
                                break
                            except Exception as exc:
                                if first_err is None:
                                    first_err = exc
                        if first_err is not None:
                            raise first_err
                        resolved_rf_id = str(
                            ((res or {}).get("image") or {}).get("id") or rf_id or ""
                        ).strip()
                elif not upload_new_images:
                    project.save_annotation(
                        annotation_path=str(txt_path),
                        annotation_labelmap=labelmap,
                        image_id=rf_id,
                        job_name=batch_name,
                        is_prediction=False,
                        annotation_overwrite=True,
                        num_retry_uploads=2,
                        add_to_dataset=add_to_dataset,
                    )
                    resolved_rf_id = rf_id
                else:
                    if not src_path.is_file():
                        raise FileNotFoundError(f"Missing local image: {src_path}")
                    res = project.single_upload(
                        image_path=str(src_path),
                        annotation_path=str(txt_path),
                        annotation_labelmap=labelmap,
                        split=split,
                        num_retry_uploads=2,
                        batch_name=batch_name,
                        tag_names=tags,
                        is_prediction=False,
                        annotation_overwrite=True,
                        add_to_dataset=add_to_dataset,
                    )
                    resolved_rf_id = str(
                        ((res or {}).get("image") or {}).get("id") or rf_id or ""
                    ).strip()
                with lock:
                    succeeded_ids.append(item.id)
                    if resolved_rf_id:
                        succeeded_rf_ids.append(resolved_rf_id)
            except Exception as exc:
                with lock:
                    failed_items.append((item.id, str(exc)))

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [
                pool.submit(_push_one, idx, entry)
                for idx, entry in enumerate(eligible_entries)
            ]
            for fut in futs:
                fut.result()

    return succeeded_ids, failed_items, succeeded_rf_ids


def push_accepted_to_roboflow(
    *,
    api_key: str,
    workspace_slug: Optional[str],
    job: AutoJobRecord,
    eligible_entries: Sequence[Dict[str, Any]],
    class_lower_to_id: Dict[str, int],
    ordered_names: List[str],
    push_destination: str,
    target_project_name: Optional[str],
    target_folder_name: Optional[str],
    target_annotation_job_id: Optional[str],
    add_to_dataset: bool,
    project_license: str,
    zip_path: Path,
    batch_name: str,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Push verified accepted items to Roboflow under one of four destination modes:
      - new_project: create/open new project & upload images + annotations (ZIP with automatic single_upload fallback)
      - same_project_dataset: save annotations directly onto fetched images in source project & add to Dataset
      - same_project_annotate: save annotations in-place on fetched images keeping them staged in Annotate (addToDataset=False)
      - same_project_new_folder: save annotations on fetched images & move/organize into a new Roboflow folder/batch
    """
    rf = Roboflow(api_key=api_key.strip())
    ws = (
        rf.workspace(workspace_slug.strip())
        if workspace_slug and workspace_slug.strip()
        else rf.workspace()
    )
    rf_project_type = (
        "instance-segmentation"
        if job.projectType == "instance-segmentation"
        else "object-detection"
    )
    license_val = (project_license or "MIT").strip() or "MIT"
    tag_list = tags or ["auto-annotated", "verified"]

    if push_destination == "new_project":
        slug = _normalize_project_slug(target_project_name or f"{job.projectSlug}-auto-annotated")
        try:
            project = ws.project(slug)
        except Exception:
            project = ws.create_project(
                project_name=slug,
                project_type=rf_project_type,
                project_license=license_val,
                annotation=slug,
            )

        folder_label = (target_folder_name or batch_name).strip() or batch_name
        zip_fallback_reason: Optional[str] = None
        try:
            status = ws.upload_dataset(
                dataset_path=str(zip_path),
                project_name=slug,
                project_license=license_val,
                project_type=rf_project_type,
                batch_name=folder_label,
                tags=tag_list,
                use_zip_upload=True,
                annotation_overwrite=True,
                wait=True,
                poll_interval=3.0,
                poll_timeout=600.0,
            )
            all_ids = [e["item"].id for e in eligible_entries]
            return {
                "pushDestination": "new_project",
                "targetProjectSlug": slug,
                "targetFolderName": folder_label,
                "targetAnnotationJobId": None,
                "addedToDataset": True,
                "succeededItemIds": all_ids,
                "failedItems": [],
                "uploadStatus": (
                    status
                    if isinstance(status, dict)
                    else {"result": status, "method": "zip_upload"}
                ),
            }
        except Exception as exc:
            zip_fallback_reason = str(exc)

        succeeded_ids, failed_items, _ = _push_entries_per_item(
            project=project,
            job=job,
            eligible_entries=eligible_entries,
            class_lower_to_id=class_lower_to_id,
            ordered_names=ordered_names,
            upload_new_images=True,
            batch_name=folder_label,
            tags=tag_list,
            add_to_dataset=True,
        )
        return {
            "pushDestination": "new_project",
            "targetProjectSlug": slug,
            "targetFolderName": folder_label,
            "targetAnnotationJobId": None,
            "addedToDataset": True,
            "succeededItemIds": succeeded_ids,
            "failedItems": failed_items,
            "uploadStatus": {
                "method": "single_upload_fallback",
                "zipFallbackNote": zip_fallback_reason,
                "pushed": len(succeeded_ids),
                "failed": len(failed_items),
            },
        }

    # All `same_project_*` modes target the original Roboflow source project
    slug = job.projectSlug.strip()
    project = ws.project(slug)
    effective_add_to_dataset = (
        True
        if push_destination == "same_project_dataset"
        else (False if push_destination == "same_project_annotate" else bool(add_to_dataset))
    )
    default_folder = (
        job.sourceBatchId or job.sourceAnnotationJobId or batch_name
        if push_destination in ("same_project_dataset", "same_project_annotate")
        else f"{job.id}-verified"
    )
    folder_label = (target_folder_name or default_folder).strip() or batch_name
    eff_ann_job_id = (
        (target_annotation_job_id or job.sourceAnnotationJobId or "").strip() or None
        if push_destination == "same_project_annotate"
        else None
    )

    succeeded_ids, failed_items, pushed_rf_ids = _push_entries_per_item(
        project=project,
        job=job,
        eligible_entries=eligible_entries,
        class_lower_to_id=class_lower_to_id,
        ordered_names=ordered_names,
        upload_new_images=False,
        batch_name=folder_label,
        tags=tag_list,
        add_to_dataset=effective_add_to_dataset,
    )

    extra_status: Dict[str, Any] = {
        "method": "save_annotation",
        "addToDataset": effective_add_to_dataset,
        "pushed": len(succeeded_ids),
        "failed": len(failed_items),
    }

    if push_destination == "same_project_annotate" and eff_ann_job_id and pushed_rf_ids:
        # Move into target annotation job if images didn't already originate from it
        if job.fetchScope != "annotation_job" or eff_ann_job_id != job.sourceAnnotationJobId:
            if hasattr(project, "add_annotation_job_images"):
                try:
                    add_res = project.add_annotation_job_images(eff_ann_job_id, pushed_rf_ids)
                    extra_status["annotationJobStage"] = {"ok": True, "response": add_res}
                except Exception as exc:
                    extra_status["annotationJobStage"] = {
                        "ok": False,
                        "note": f"Annotations saved in Annotate; moving images to job '{eff_ann_job_id}' skipped: {exc}",
                    }

    if push_destination == "same_project_new_folder" and pushed_rf_ids:
        src_batch_id = _resolve_source_batch_id(project, job, pushed_rf_ids)
        if src_batch_id and hasattr(project, "create_annotation_batch"):
            try:
                batch_res = project.create_annotation_batch(
                    source_batch_id=src_batch_id,
                    image_ids=pushed_rf_ids,
                    name=folder_label,
                )
                extra_status["folderMove"] = {
                    "ok": True,
                    "sourceBatchId": src_batch_id,
                    "folderName": folder_label,
                    "response": batch_res,
                }
            except Exception as exc:
                extra_status["folderMove"] = {
                    "ok": False,
                    "sourceBatchId": src_batch_id,
                    "folderName": folder_label,
                    "note": f"Annotations saved; creating new Roboflow folder batch skipped: {exc}",
                }
        else:
            extra_status["folderMove"] = {
                "ok": False,
                "folderName": folder_label,
                "note": "Annotations saved; source batch ID unavailable to move existing project images into a separate batch folder.",
            }

    return {
        "pushDestination": push_destination,
        "targetProjectSlug": slug,
        "targetFolderName": folder_label,
        "targetAnnotationJobId": eff_ann_job_id,
        "addedToDataset": effective_add_to_dataset,
        "succeededItemIds": succeeded_ids,
        "failedItems": failed_items,
        "uploadStatus": extra_status,
    }
