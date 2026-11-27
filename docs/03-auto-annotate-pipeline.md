# 03 — Auto-Annotate & Verify Pipeline (`/auto-annotate`, `/verify`)

This pipeline automates dataset ingestion from Roboflow, batch inference across large image batches, human QA verification, local dataset export, and cloud upload.

---

## 1. Auto-Annotate (`/auto-annotate`)

### Step A — Roboflow Source Configuration (`RoboflowSourceCard`)
1. Paste your **Roboflow Private API Key** (held in session memory / `sessionStorage` for the browser session, wiped automatically if backend restarts).
2. Enter **Project Name / Slug** (optional **Workspace Slug**, auto-detected from your key if omitted).
3. Click **Connect & Inspect Project** (`POST /api/v1/auto-annotate/roboflow/inspect`) to retrieve batches, annotation jobs, versions, project type, and classes.
4. Configure Fetch Scope:
   - **Fetch Filter**: `unannotated`, `all`, `annotated`, `batch`, `annotation_job`, or `dataset`.
   - **Annotation Status Filter** (optional): `all`, `annotated`, `unannotated`.
   - **Assignee Email** / **Source Batch** / **Source Annotation Job** (when project metadata provides them).
   - **Max Images to Fetch** & **Verify Batch Size** (default `50` images per batch).

### Step B — Detector, Mask Refiner & Schema
- **Detectors**:
  - **NVIDIA & Mac**: `yoloe-26s-seg`, `owlv2-base-patch16-ensemble`, `owlv2-large-patch14-ensemble`, `florence2-large`, `florence2-base`, `ollama-8b-zero-shot`.
  - **Mac Only**: `mlx-4b-*` Qwen3-VL checkpoints.
- **SAM Polygon Refiner**: Pair any box detector with **`sam2.1-s`**, **`sam2.1-t`**, **`mobile-sam`**, or **`none`** to automatically convert 2D boxes into tight segmentation polygons when `Output Mode` is `polygon` or `both`.
- **Target Classes & Color Palette**: Define expected class names and per-class colors.
- **Prompt & Confidence Threshold** (+ optional **Few-Shot Reference Images** for VLMs).

### Step C — Execution & Live Controls
- Submitting creates a job (`POST /api/v1/auto-annotate/jobs`) organized into structured batches (`batch-01`, `batch-02`, ...).
- Progress streams live over SSE (`GET /api/v1/auto-annotate/jobs/{jobId}/stream`) with **Pause**, **Resume**, and **Cancel** buttons.

---

## 2. Verify Queue (`/verify`)

1. Select a job and filter by **Batch** (`batch-01`, ...), **Review Status** (`pending`, `accepted`, `rejected`), or **Reject Reason** (`missed_object`, `false_detection`, `bad_box`, `double_box`).
2. Inspect frames in the grid or open **Verify Lightbox Modal** for full-size review:
   - **Accept**: Marks frame accepted and syncs disk export cache.
   - **Reject**: Prompts/records one of the 4 structured failure reasons.
   - **Edit Manually**: Jumps directly to `/annotate` focused on that exact job & frame.
3. **Batch Operations**:
   - **Approve Batch** (`POST /jobs/{jobId}/batches/{batchId}/approve`): Accepts remaining pending frames in the active batch.
   - **Retry Rejected Batch** (`?retryJob=...`): Sends rejected items back to `/auto-annotate` with tuned prompt/model/confidence.
4. **Export & Push**:
   - **Download Accepted ZIP** (`GET /jobs/{jobId}/download-zip`): Packages accepted images + YOLO/COCO annotations.
   - **Push to Roboflow** (`POST /jobs/{jobId}/push`): Supports `new_project`, `same_project_dataset`, `same_project_new_folder`, or `same_project_annotate` (`includeEmpty` / `onlyUnpushed` options + optional `batchTag`).
