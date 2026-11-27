# 02 — VLM Evaluation Workbench (`/analyze`, `/logs`, `/results`)

The evaluation workbench lets you benchmark multimodal models side by side on single images or videos before running batch auto-annotation.

---

## 1. Analyze Studio (`/analyze`)

### Target Input & Mode
- **Image Mode**: Upload any JPG/PNG/WebP image.
- **Video Mode**: Upload a short clip (MP4/MOV/AVI). Additional controls appear:
  - **Sampling FPS**: Frame sampling rate sent to Stage 1 event analysis (up to 14 keyframes extracted into 640x360 Stage 1 & <=960px Stage 2).
  - **Output FPS Mode**: `original` (native FPS output) or `sampled`.
  - **ByteTrack Memory (sec)**: `lost_track_buffer` duration (default `3.0s`) keeping smooth Kalman filter tracks across brief occlusions after an event starts.

### Shot Mode & Output Mode
- **Zero-Shot**: Evaluates strictly from the text prompt (no reference images).
- **Few-Shot**: Upload annotated reference frames + optional per-reference notes and overall `Reference Style Guidance` so the VLM learns tightness and class conventions before predicting on the target image/video.
- **Output Mode**:
  - `bbox`: 2D bounding boxes (`[xmin, ymin, xmax, ymax]` normalized `0..1000`).
  - `polygon`: Contour vertices (`[[x, y], ...]`).
  - `both`: Bounding boxes + contour polygons.

### Supported Models
| Model ID | Engine | Hardware Notes |
|---|---|---|
| `florence2-large` | Transformers `Florence-2-large-ft` | **NVIDIA CUDA**, Apple MPS, CPU (single-image & region polygon) |
| `florence2-base` | Transformers `Florence-2-base-ft` | **NVIDIA CUDA**, Apple MPS, CPU (faster throughput) |
| `ollama-8b-zero-shot` | Ollama (`qwen3-vl:8b`) | **NVIDIA CUDA**, Apple Silicon, CPU (supports image + video) |
| `mlx-4b-8bit-lora-best` / `final` / `base` / `4bit` | Apple `mlx_vlm` | Apple Silicon macOS **only** (`available: false` on NVIDIA/Linux/Windows) |

### Video Two-Stage Pipeline
1. **Stage 1 (Temporal Event Scan)**: Samples `640x360` chronological frames to identify `event_detected`, `trigger_subject`, `short_label`, and event start/end frame numbers & timestamps.
2. **Stage 2 (Post-Event Spatial Grounding)**: Runs spatial grounding strictly on keyframes belonging to active event intervals (`no boxes prior to event start`).
3. **Stage 3 (ByteTrack + FFmpeg Rendering)**: Runs Kalman Filter ByteTrack (`tracker.py`) and encodes an H.264 `yuv420p` MP4 (`video_processor.py`) at native resolution.

---

## 2. Test Logs (`/logs`) & Results (`/results`)

- **`/logs`**: Lists all SQLite-backed runs (`studio.db`) with status, model used, wall-time latency, prompt/generation tokens, and peak VRAM/RAM.
- **`/results`**: Inspect rendered image overlays (`AnnotationCanvas.tsx`), raw JSON output, per-stage latency telemetry, video event timelines (`VideoResultView.tsx`), and trigger **Re-encode Video** with different `byteTrackMemorySec` or `outputFpsMode` without re-running VLM inference.
- **Sidebar Ollama Control**: Start/stop local `ollama serve` directly from the sidebar (`POST /api/v1/ollama/start` & `/stop`).
