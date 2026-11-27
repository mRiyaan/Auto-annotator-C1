import base64
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker import ByteTracker, calculate_iou_matrix


class VideoProcessor:
    """
    Dynamic video inspection, frame separation, 640x360 resolution/size reduction for Qwen3-VL,
    multi-event post-event dense interpolation, and native-resolution ByteTrack video rendering.
    """

    @staticmethod
    def extract_keyframes_from_bytes(
        file_bytes: bytes,
        filename: str,
        sampling_fps: float = 1.0,
        target_size: Tuple[int, int] = (640, 360),
        jpeg_quality: int = 75,
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any], str]:
        """
        1. Dynamically inspects the uploaded video's native resolution, actual FPS, frame count, and duration.
        2. Separates frames at `sampling_fps` based on the detected native FPS.
        3. Reduces each separated frame to `640x360` and compresses JPEG byte size for Qwen3-VL ingestion.
        4. Preserves the original video file on disk for native-resolution output encoding.
        """
        suffix = os.path.splitext(filename)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(f"OpenCV could not open uploaded video '{filename}'.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if native_fps <= 0 or np.isnan(native_fps):
            cap.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(f"Could not detect a valid FPS from video '{filename}'.")
        if total_frames <= 0 or orig_width <= 0 or orig_height <= 0:
            cap.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(
                f"Video '{filename}' has invalid dimensions or 0 readable frames."
            )

        duration_sec = total_frames / native_fps
        orig_size_kb = round(len(file_bytes) / 1024.0, 2)

        num_target = max(1, int(round(duration_sec * sampling_fps)))
        indices: List[int] = []
        for i in range(num_target):
            idx = min(total_frames - 1, int(round((i / sampling_fps) * native_fps)))
            if not indices or idx > indices[-1]:
                indices.append(idx)

        target_w, target_h = target_size
        extracted_b64: List[str] = []
        keyframe_info: List[Dict[str, Any]] = []
        total_jpeg_bytes = 0

        base_name = os.path.splitext(os.path.basename(filename))[0]
        ts_folder = time.strftime("%Y%m%d_%H%M%S")
        frames_output_dir = os.path.abspath(
            os.path.join("outputs", "separated_frames", f"{ts_folder}_{base_name}")
        )
        os.makedirs(frames_output_dir, exist_ok=True)

        for sample_num, idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            ts = round(idx / native_fps, 2)

            # Reduce resolution strictly to 640x360 and compress JPEG byte size for VLM
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            success, buffer = cv2.imencode(
                ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if not success:
                continue

            raw_bytes = buffer.tobytes()
            frame_kb = round(len(raw_bytes) / 1024.0, 2)
            total_jpeg_bytes += len(raw_bytes)
            b64_str = base64.b64encode(raw_bytes).decode("utf-8")

            frame_filename = (
                f"frame_{sample_num:03d}_idx{idx:04d}_{ts:.2f}s_640x360.jpg"
            )
            frame_disk_path = os.path.join(frames_output_dir, frame_filename)
            with open(frame_disk_path, "wb") as f_out:
                f_out.write(raw_bytes)

            extracted_b64.append(b64_str)
            keyframe_info.append(
                {
                    "sample_number": sample_num,
                    "frame_idx": idx,
                    "timestamp_sec": ts,
                    "downscaled_width": target_w,
                    "downscaled_height": target_h,
                    "frame_size_kb": frame_kb,
                    "frame_path": frame_disk_path,
                    "b64_jpeg": b64_str,
                }
            )

        cap.release()

        avg_kb = (
            round((total_jpeg_bytes / len(extracted_b64)) / 1024.0, 2)
            if extracted_b64
            else 0.0
        )
        total_downscaled_kb = round(total_jpeg_bytes / 1024.0, 2)

        metadata = {
            "filename": filename,
            "original_file_size_kb": orig_size_kb,
            "total_video_frames": total_frames,
            "fps": round(native_fps, 3),
            "duration_seconds": round(duration_sec, 2),
            "width": orig_width,
            "height": orig_height,
            "original_resolution": f"{orig_width}x{orig_height}",
            "model_input_resolution": f"{target_w}x{target_h}",
            "avg_model_frame_size_kb": avg_kb,
            "total_model_frames_size_kb": total_downscaled_kb,
            "separated_frames_dir": frames_output_dir,
            "sampling_fps_used": sampling_fps,
            "sampled_frames_count": len(extracted_b64),
            "keyframe_timestamps": [k["timestamp_sec"] for k in keyframe_info],
        }

        return extracted_b64, keyframe_info, metadata, tmp_path

    @staticmethod
    def _match_detections_between_keyframes(
        dets_a: List[Dict[str, Any]],
        dets_b: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Associates detections between consecutive keyframes within the same event."""
        if not dets_a or not dets_b:
            return []
        if len(dets_a) == 1 and len(dets_b) == 1:
            return [(dets_a[0], dets_b[0])]

        boxes_a = np.array([d["bbox"] for d in dets_a], dtype=np.float32)
        boxes_b = np.array([d["bbox"] for d in dets_b], dtype=np.float32)
        iou = calculate_iou_matrix(boxes_a, boxes_b)

        centers_a = np.column_stack(
            ((boxes_a[:, 0] + boxes_a[:, 2]) / 2.0, (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0)
        )
        centers_b = np.column_stack(
            ((boxes_b[:, 0] + boxes_b[:, 2]) / 2.0, (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0)
        )
        dist = np.linalg.norm(centers_a[:, None, :] - centers_b[None, :, :], axis=-1)
        scale = max(float(np.max(boxes_a[:, 2] - boxes_a[:, 0])), 150.0)
        cost = (1.0 - iou) + 0.35 * np.clip(dist / scale, 0.0, 2.0)
        rows, cols = linear_sum_assignment(cost)
        return [(dets_a[r], dets_b[c]) for r, c in zip(rows, cols)]

    @staticmethod
    def _normalize_event_windows(
        events: Optional[List[Dict[str, Any]]],
        event_detected: bool,
        event_start_timestamp_sec: Optional[float],
        event_end_timestamp_sec: Optional[float],
        event_type: str,
        fps: float,
        total_frames: int,
        lost_buffer_seconds: float,
        keyframe_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalizes single or multi-event windows into frame-indexed bounds."""
        if not event_detected:
            return []

        raw_events = list(events or [])
        if not raw_events and event_start_timestamp_sec is not None:
            raw_events = [
                {
                    "event_id": 1,
                    "event_type": event_type or "Detected Event",
                    "short_label": event_type or "target",
                    "event_start_timestamp_sec": event_start_timestamp_sec,
                    "event_end_timestamp_sec": event_end_timestamp_sec,
                }
            ]

        windows: List[Dict[str, Any]] = []
        for idx, ev in enumerate(raw_events, start=1):
            s_ts = ev.get("event_start_timestamp_sec")
            if s_ts is None:
                continue
            eid = int(ev.get("event_id", idx) or idx)
            e_ts = ev.get("event_end_timestamp_sec")
            start_f = max(0, int(round(float(s_ts) * fps)))
            if e_ts is not None and float(e_ts) >= float(s_ts):
                end_f = min(total_frames - 1, int(round(float(e_ts) * fps)))
            else:
                end_f = total_frames - 1

            ev_dets = [
                int(d["video_frame_index"])
                for d in keyframe_detections
                if int(d.get("event_id", 1) or 1) == eid
                and start_f <= int(d["video_frame_index"]) <= end_f
            ]
            last_det_f = max(ev_dets) if ev_dets else end_f

            dur_sec = max(0.0, (end_f - start_f) / max(fps, 1.0))
            if dur_sec < 1.5:
                tail_sec = lost_buffer_seconds
                draw_end_f = min(
                    total_frames - 1,
                    max(end_f, last_det_f) + int(round(fps * tail_sec)),
                )
            else:
                tail_sec = min(0.75, lost_buffer_seconds * 0.3)
                # Stop coasting shortly after actual last grounded frame once event completed.
                draw_end_f = min(
                    total_frames - 1,
                    max(end_f, last_det_f + int(round(fps * tail_sec))),
                    last_det_f + int(round(fps * min(1.25, max(tail_sec, 0.5)))),
                )

            windows.append(
                {
                    "event_id": eid,
                    "event_type": str(ev.get("event_type") or event_type or "Detected Event"),
                    "short_label": str(ev.get("short_label") or f"event {eid}"),
                    "start_ts": float(s_ts),
                    "end_ts": float(e_ts) if e_ts is not None else float(total_frames / fps),
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "draw_end_frame": draw_end_f,
                }
            )
        return windows

    @staticmethod
    def build_dense_post_event_detections(
        keyframe_detections: List[Dict[str, Any]],
        event_start_frame_idx: Optional[int],
        event_end_frame_idx: Optional[int],
        total_frames: int,
        events: Optional[List[Dict[str, Any]]] = None,
        fps: float = 24.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Builds per-frame detections strictly at and after each event's start frame.
        Interpolates independently per `event_id` across keyframes up to 2.5s apart
        (scaled by native FPS), preserving multiple simultaneous subjects without ever
        morphing boxes between separate events.
        """
        if not keyframe_detections:
            return {}

        bounds_by_eid: Dict[int, Tuple[int, int]] = {}
        if events:
            for idx, ev in enumerate(events, start=1):
                s_ts = ev.get("event_start_timestamp_sec")
                if s_ts is None:
                    continue
                eid = int(ev.get("event_id", idx) or idx)
                e_ts = ev.get("event_end_timestamp_sec")
                s_f = max(0, int(round(float(s_ts) * fps)))
                e_f = (
                    min(total_frames - 1, int(round(float(e_ts) * fps)))
                    if e_ts is not None and float(e_ts) >= float(s_ts)
                    else total_frames - 1
                )
                bounds_by_eid[eid] = (s_f, e_f)
        elif event_start_frame_idx is not None:
            e_f = (
                event_end_frame_idx
                if event_end_frame_idx is not None
                else (total_frames - 1)
            )
            bounds_by_eid[1] = (event_start_frame_idx, e_f)
        else:
            return {}

        grouped_by_eid: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        for d in keyframe_detections:
            eid = int(d.get("event_id", 1) or 1)
            if eid not in bounds_by_eid:
                # Match scalar single-window callers or unmatched fallback detections
                s_f, e_f = (
                    bounds_by_eid[1]
                    if 1 in bounds_by_eid
                    else (0, total_frames - 1)
                )
            else:
                s_f, e_f = bounds_by_eid[eid]

            f_idx = int(d["video_frame_index"])
            if s_f <= f_idx <= e_f:
                grouped_by_eid.setdefault(eid, {}).setdefault(f_idx, []).append(
                    {
                        "event_id": eid,
                        "bbox": d["pixel_box"],
                        "score": d.get("confidence", 0.92),
                        "label": d.get("label", "event_subject"),
                    }
                )

        max_gap_frames = max(2, int(round(max(fps, 1.0) * 2.5)))
        dense_dets: Dict[int, List[Dict[str, Any]]] = {}

        for eid, by_kf in grouped_by_eid.items():
            sorted_kfs = sorted(by_kf.keys())
            for i, f_a in enumerate(sorted_kfs):
                dense_dets.setdefault(f_a, []).extend(by_kf[f_a])
                if i + 1 < len(sorted_kfs):
                    f_b = sorted_kfs[i + 1]
                    gap = f_b - f_a
                    if 1 < gap <= max_gap_frames:
                        pairs = VideoProcessor._match_detections_between_keyframes(
                            by_kf[f_a], by_kf[f_b]
                        )
                        for det_a, det_b in pairs:
                            box_a = np.array(det_a["bbox"], dtype=np.float32)
                            box_b = np.array(det_b["bbox"], dtype=np.float32)
                            for f_mid in range(f_a + 1, f_b):
                                alpha = (f_mid - f_a) / float(gap)
                                interp_box = (1.0 - alpha) * box_a + alpha * box_b
                                dense_dets.setdefault(f_mid, []).append(
                                    {
                                        "event_id": eid,
                                        "bbox": [round(float(v), 1) for v in interp_box],
                                        "score": round(
                                            (1.0 - alpha) * det_a["score"]
                                            + alpha * det_b["score"],
                                            2,
                                        ),
                                        "label": det_a["label"],
                                    }
                                )

        return dense_dets

    @staticmethod
    def _draw_annotations_and_hud(
        frame: np.ndarray,
        active_tracks: List[Dict[str, Any]],
        palette: List[Tuple[int, int, int]],
        width: int,
        height: int,
        effective_fps: float,
        display_idx: int,
        display_total: int,
        time_curr: float,
        active_windows: List[Dict[str, Any]],
        has_prior_triggered_event: bool,
        mode_label: str,
    ) -> None:
        """Draws fast top HUD status banner and color-coded post-event bounding boxes."""
        hud_bg_height = 44
        hud_roi = frame[0:hud_bg_height, :]

        if active_windows and active_tracks:
            bg_color = np.array([90, 25, 15], dtype=np.float32)
            if len(active_windows) == 1:
                w0 = active_windows[0]
                status_str = (
                    f"EVENT ACTIVE (E{w0['event_id']}: {w0['event_type']}) "
                    f"| Triggered @ {w0['start_ts']:.2f}s"
                )
            else:
                tags = ", ".join(
                    f"E{w['event_id']} @ {w['start_ts']:.1f}s" for w in active_windows
                )
                status_str = f"{len(active_windows)} EVENTS ACTIVE ({tags})"
            status_color = (80, 200, 255)
        elif active_windows:
            bg_color = np.array([70, 35, 15], dtype=np.float32)
            w0 = active_windows[0]
            status_str = (
                f"EVENT WINDOW (E{w0['event_id']}: {w0['event_type']}) "
                f"| @ {w0['start_ts']:.2f}s"
            )
            status_color = (120, 210, 255)
        else:
            bg_color = np.array([20, 20, 20], dtype=np.float32)
            status_str = (
                "Monitoring Video (Watching for Next Event...)"
                if has_prior_triggered_event
                else "Monitoring Video (No Event Triggered Yet)"
            )
            status_color = (180, 220, 180)

        blended = (0.22 * hud_roi.astype(np.float32) + 0.78 * bg_color).astype(np.uint8)
        frame[0:hud_bg_height, :] = blended

        hud_left = (
            f"Frame: {display_idx + 1}/{display_total} | Time: {time_curr:.2f}s | "
            f"{width}x{height} @ {effective_fps:.1f}fps ({mode_label})"
        )
        cv2.putText(
            frame,
            hud_left,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        (sw, _), _ = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            frame,
            status_str,
            (max(14, width - sw - 14), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            1,
            cv2.LINE_AA,
        )

        for trk in active_tracks:
            b = trk["bbox"]
            x1 = max(0, int(b[0]))
            y1 = max(0, int(b[1]))
            x2 = min(width - 1, int(b[2]))
            y2 = min(height - 1, int(b[3]))

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            trk_id = trk["track_id"]
            color_idx = (trk.get("event_id") or trk_id) - 1
            color = palette[color_idx % len(palette)]
            is_lost = trk.get("is_lost", False)

            if is_lost:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                label_text = (
                    f"ID:{trk_id} {trk['label']} "
                    f"(Track Mem {trk['time_lost_frames'] / effective_fps:.1f}s)"
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)
                label_text = f"ID:{trk_id} {trk['label']} ({trk['score']:.2f})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.56
            thickness = 2
            (lw, lh), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            badge_y1 = max(hud_bg_height + 2, y1 - lh - 12)
            badge_y2 = min(height - 2, badge_y1 + lh + 12)
            badge_x2 = min(width - 1, x1 + lw + 12)
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 6, badge_y2 - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def render_tracked_video(
        input_video_path: str,
        output_dir: str,
        filename: str,
        keyframe_detections: List[Dict[str, Any]],
        event_detected: bool = False,
        event_start_timestamp_sec: Optional[float] = None,
        event_end_timestamp_sec: Optional[float] = None,
        event_type: str = "",
        event_summary: str = "",
        events: Optional[List[Dict[str, Any]]] = None,
        lost_buffer_seconds: float = 3.0,
        tag_suffix: str = "",
        output_fps_mode: str = "original",
        target_output_fps: Optional[float] = None,
        sampled_frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Renders the output video at native spatial resolution and either the original video FPS
        (`output_fps_mode="original"`) or the user-selected sampling FPS (`output_fps_mode="sampled"`).
        Supports multiple sequential or overlapping events independently without drawing pre-event boxes.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video for rendering.")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        windows = VideoProcessor._normalize_event_windows(
            events=events,
            event_detected=event_detected,
            event_start_timestamp_sec=event_start_timestamp_sec,
            event_end_timestamp_sec=event_end_timestamp_sec,
            event_type=event_type,
            fps=fps,
            total_frames=total_frames,
            lost_buffer_seconds=lost_buffer_seconds,
            keyframe_detections=keyframe_detections,
        )

        event_start_frame_idx: Optional[int] = (
            min(w["start_frame"] for w in windows) if windows else None
        )
        event_end_frame_idx: Optional[int] = (
            max(w["end_frame"] for w in windows) if windows else None
        )

        use_sampled = (
            output_fps_mode == "sampled"
            and target_output_fps is not None
            and target_output_fps > 0
        )
        out_fps = float(target_output_fps) if use_sampled else fps
        mode_label = "Sampled" if use_sampled else "Original"

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suffix_part = f"_{tag_suffix}" if tag_suffix else ""
        raw_output_path = os.path.join(
            output_dir, f"temp_{timestamp_str}{suffix_part}_{base_name}.mp4"
        )
        final_output_filename = (
            f"annotated_{timestamp_str}{suffix_part}_{base_name}.mp4"
        )
        final_output_path = os.path.join(output_dir, final_output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, out_fps, (width, height))

        tracker = ByteTracker(
            fps=out_fps,
            lost_buffer_seconds=lost_buffer_seconds,
            match_thresh=0.15 if use_sampled else 0.2,
        )

        unique_track_ids: Set[int] = set()
        rendered_count = 0
        annotated_frames_count = 0

        palette = [
            (0, 80, 255),    # Event 1: High-visibility Red-Orange (BGR)
            (60, 215, 70),   # Event 2: Vivid Emerald Green
            (255, 150, 30),  # Event 3: Deep Sky Blue
            (0, 195, 255),   # Event 4: Amber Gold
            (215, 70, 235),  # Event 5: Magenta
        ]

        if use_sampled:
            if sampled_frame_indices:
                frame_indices_to_render = sorted(
                    list(dict.fromkeys(sampled_frame_indices))
                )
            else:
                duration_sec = total_frames / fps if fps > 0 else 0.0
                num_target = max(1, int(round(duration_sec * out_fps)))
                frame_indices_to_render = []
                for i in range(num_target):
                    idx = min(total_frames - 1, int(round((i / out_fps) * fps)))
                    if not frame_indices_to_render or idx > frame_indices_to_render[-1]:
                        frame_indices_to_render.append(idx)

            bounds_by_eid = {w["event_id"]: (w["start_frame"], w["end_frame"]) for w in windows}
            sampled_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
            for d in keyframe_detections:
                f_idx = int(d["video_frame_index"])
                eid = int(d.get("event_id", 1) or 1)
                if eid in bounds_by_eid:
                    s_f, e_f = bounds_by_eid[eid]
                    if s_f <= f_idx <= e_f:
                        sampled_dets_by_frame.setdefault(f_idx, []).append(
                            {
                                "event_id": eid,
                                "bbox": d["pixel_box"],
                                "score": d.get("confidence", 0.95),
                                "label": d.get("label", "target"),
                            }
                        )

            total_display = len(frame_indices_to_render)
            for out_pos, src_frame_idx in enumerate(frame_indices_to_render):
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                time_curr = src_frame_idx / fps
                active_wins = [
                    w
                    for w in windows
                    if w["start_frame"] <= src_frame_idx <= w["draw_end_frame"]
                ]
                has_prior = any(src_frame_idx >= w["start_frame"] for w in windows)
                active_eids = {w["event_id"] for w in active_wins}

                active_tracks: List[Dict[str, Any]] = []
                if active_wins or tracker.tracked_stracks:
                    current_dets = sampled_dets_by_frame.get(src_frame_idx, None)
                    if current_dets:
                        current_dets = [
                            d for d in current_dets if d["event_id"] in active_eids
                        ] or None
                    active_tracks = tracker.step(
                        frame, current_dets, active_event_ids=active_eids
                    )

                if active_tracks:
                    annotated_frames_count += 1
                    for trk in active_tracks:
                        unique_track_ids.add(trk["track_id"])

                VideoProcessor._draw_annotations_and_hud(
                    frame=frame,
                    active_tracks=active_tracks,
                    palette=palette,
                    width=width,
                    height=height,
                    effective_fps=out_fps,
                    display_idx=out_pos,
                    display_total=total_display,
                    time_curr=time_curr,
                    active_windows=active_wins,
                    has_prior_triggered_event=has_prior,
                    mode_label=mode_label,
                )
                writer.write(frame)
                rendered_count += 1
        else:
            dense_detections = VideoProcessor.build_dense_post_event_detections(
                keyframe_detections=keyframe_detections,
                event_start_frame_idx=event_start_frame_idx,
                event_end_frame_idx=event_end_frame_idx,
                total_frames=total_frames,
                events=events,
                fps=fps,
            )

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                time_curr = frame_idx / fps
                active_wins = [
                    w
                    for w in windows
                    if w["start_frame"] <= frame_idx <= w["draw_end_frame"]
                ]
                has_prior = any(frame_idx >= w["start_frame"] for w in windows)
                active_eids = {w["event_id"] for w in active_wins}

                active_tracks = []
                if active_wins or tracker.tracked_stracks:
                    current_dets = dense_detections.get(frame_idx, None)
                    if current_dets:
                        current_dets = [
                            d for d in current_dets if d["event_id"] in active_eids
                        ] or None
                    active_tracks = tracker.step(
                        frame, current_dets, active_event_ids=active_eids
                    )

                if active_tracks:
                    annotated_frames_count += 1
                    for trk in active_tracks:
                        unique_track_ids.add(trk["track_id"])

                VideoProcessor._draw_annotations_and_hud(
                    frame=frame,
                    active_tracks=active_tracks,
                    palette=palette,
                    width=width,
                    height=height,
                    effective_fps=out_fps,
                    display_idx=frame_idx,
                    display_total=total_frames,
                    time_curr=time_curr,
                    active_windows=active_wins,
                    has_prior_triggered_event=has_prior,
                    mode_label=mode_label,
                )
                writer.write(frame)
                frame_idx += 1
                rendered_count += 1

        cap.release()
        writer.release()

        ffmpeg_bin = (
            os.getenv("FFMPEG_BIN")
            or shutil.which("ffmpeg")
            or next(
                (
                    p
                    for p in (
                        "/opt/homebrew/bin/ffmpeg",
                        "/usr/local/bin/ffmpeg",
                        "/usr/bin/ffmpeg",
                    )
                    if os.path.isfile(p)
                ),
                "ffmpeg",
            )
        )
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            raw_output_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            final_output_path,
        ]
        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)
        except Exception:
            if os.path.exists(raw_output_path):
                os.rename(raw_output_path, final_output_path)

        return {
            "output_video_path": os.path.abspath(final_output_path),
            "output_video_filename": final_output_filename,
            "output_resolution": f"{width}x{height}",
            "native_input_fps": round(fps, 3),
            "output_fps": round(out_fps, 3),
            "output_fps_mode": output_fps_mode,
            "frames_rendered": rendered_count,
            "annotated_frames_count": annotated_frames_count,
            "event_start_frame_idx": event_start_frame_idx,
            "event_end_frame_idx": event_end_frame_idx,
            "events_count": len(windows),
            "total_unique_tracks": len(unique_track_ids),
            "lost_buffer_seconds": lost_buffer_seconds,
        }
