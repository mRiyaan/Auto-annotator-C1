"""Media inspection, video keyframe extraction, and native-resolution tracked video rendering."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config

# Reuse root video_processor.py + tracker.py without duplicating ByteTrack or FFmpeg logic
if str(config.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(config.ROOT_DIR))

from video_processor import VideoProcessor  # noqa: E402


def inspect_image_file(image_path: Path) -> Dict[str, int]:
    with Image.open(image_path) as im:
        width, height = im.size
    return {
        "width": int(width),
        "height": int(height),
        "size_bytes": int(image_path.stat().st_size),
    }


def extract_video_keyframes(
    video_path: Path,
    run_id: str,
    sampling_fps: float = 1.0,
    max_frames: int = 14,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Inspect a video file and extract evenly spaced 640x360 (Stage 1) and
    960x540/native (Stage 2) keyframes into backend/storage/outputs/keyframes/{run_id}/.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video '{video_path.name}'.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 24.0
    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Video '{video_path.name}' has invalid dimensions or 0 frames.")

    duration_sec = total_frames / native_fps
    requested = max(1, int(round(duration_sec * max(sampling_fps, 0.25))))
    num_samples = min(requested, max_frames)

    if num_samples == 1:
        indices = [0]
    else:
        raw_indices = np.linspace(0, total_frames - 1, num=num_samples)
        indices = sorted(list(dict.fromkeys(int(round(x)) for x in raw_indices)))

    kf_dir = config.KEYFRAMES_DIR / run_id
    kf_dir.mkdir(parents=True, exist_ok=True)

    keyframes: List[Dict[str, Any]] = []
    for sample_num, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        ts = round(frame_idx / native_fps, 2)

        # Stage 1 compact frame (640x360) for multi-frame temporal scan
        s1_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        s1_name = f"kf_{sample_num:03d}_s1.jpg"
        s1_path = kf_dir / s1_name
        cv2.imwrite(str(s1_path), s1_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # Stage 2 sharper frame (up to 960px wide) for spatial grounding & UI inspection
        s2_w = min(width, 960)
        s2_h = max(1, int(round((height / float(width)) * s2_w)))
        s2_frame = (
            cv2.resize(frame, (s2_w, s2_h), interpolation=cv2.INTER_AREA)
            if s2_w < width
            else frame
        )
        s2_name = f"kf_{sample_num:03d}.jpg"
        s2_path = kf_dir / s2_name
        cv2.imwrite(str(s2_path), s2_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        keyframes.append(
            {
                "sample_number": sample_num,
                "frame_idx": frame_idx,
                "timestamp_sec": ts,
                "stage1_path": str(s1_path),
                "stage2_path": str(s2_path),
                "image_url": f"/api/v1/files/outputs/keyframes/{run_id}/{s2_name}",
            }
        )

    cap.release()

    if not keyframes:
        raise ValueError(f"Could not extract any readable frames from '{video_path.name}'.")

    meta = {
        "width": width,
        "height": height,
        "fps": round(native_fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "size_bytes": int(video_path.stat().st_size),
    }
    return meta, keyframes


def render_video_output(
    *,
    input_video_path: str,
    run_id: str,
    filename: str,
    keyframe_detections: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    lost_buffer_seconds: float,
    output_fps_mode: str,
    target_output_fps: Optional[float],
    sampled_frame_indices: List[int],
) -> Dict[str, Any]:
    event_detected = len(events) > 0
    first_start = (
        min(float(ev["event_start_timestamp_sec"]) for ev in events)
        if events
        else None
    )
    last_end = (
        max(float(ev["event_end_timestamp_sec"]) for ev in events)
        if events
        else None
    )
    render_res = VideoProcessor.render_tracked_video(
        input_video_path=input_video_path,
        output_dir=str(config.OUTPUTS_DIR),
        filename=filename,
        keyframe_detections=keyframe_detections,
        event_detected=event_detected,
        event_start_timestamp_sec=first_start,
        event_end_timestamp_sec=last_end,
        event_type=events[0]["event_type"] if events else "",
        event_summary=events[0]["event_summary"] if events else "",
        events=events,
        lost_buffer_seconds=lost_buffer_seconds,
        tag_suffix=run_id,
        output_fps_mode=output_fps_mode,
        target_output_fps=target_output_fps,
        sampled_frame_indices=sampled_frame_indices,
    )
    out_name = render_res["output_video_filename"]
    render_res["annotated_video_url"] = f"/api/v1/files/outputs/{out_name}"
    return render_res


def cleanup_run_artifacts(run_id: str, target_url: str, annotated_video_url: Optional[str]) -> None:
    """Remove per-run disk files on delete if they live inside storage/."""
    kf_dir = config.KEYFRAMES_DIR / run_id
    if kf_dir.is_dir():
        shutil.rmtree(kf_dir, ignore_errors=True)

    for rel_prefix, base_dir in (
        ("/api/v1/files/targets/", config.TARGETS_DIR),
        ("/api/v1/files/outputs/", config.OUTPUTS_DIR),
    ):
        for url in (target_url, annotated_video_url):
            if url and url.startswith(rel_prefix):
                rel = url[len(rel_prefix) :]
                candidate = (base_dir / rel).resolve()
                if candidate.is_relative_to(base_dir.resolve()) and candidate.is_file():
                    candidate.unlink(missing_ok=True)
    ref_dir = config.REFERENCES_DIR / run_id
    if ref_dir.is_dir():
        shutil.rmtree(ref_dir, ignore_errors=True)
