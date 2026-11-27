# Project Context — Control One Auto-Annotator & VLM Workbench

This document captures the full codebase architecture, module boundaries, state/caching contracts, coordinate system, and cross-platform behavior so any engineer or coding agent can immediately pick up development on macOS (Apple Silicon) or NVIDIA CUDA workstations.

---

## 1. Repository Layout

```text
.
├── README.md                              # Project overview & quick start
├── context.md                             # Architecture & engineering context (this file)
├── MLX_dependencies.md                    # MLX raw tensor audit & NVIDIA CUDA migration map
├── .gitignore                             # Ignores runtime storage, >=100 MB weights, and MLX adapters
├── requirements.txt                       # Points to backend/requirements.txt
├── run.sh                                 # Unified dev launcher (FastAPI backend + Next.js frontend)
├── tracker.py                             # Linear Kalman Filter ByteTracker for post-event tracking
├── video_processor.py                     # OpenCV video inspector & FFmpeg H.264 MP4 renderer
├── docs/                                  # Feature & deployment markdown documentation (01..06)
├── backend/
│   ├── __init__.py
│   ├── .env.example                       # Configurable environment variables (HOST, PORT, paths, URLs)
│   ├── requirements.txt                   # Runtime Python dependencies (conditional PEP 508 MLX markers)
│   ├── requirements-nvidia.txt            # CUDA 12.x PyTorch + bitsandbytes + core requirements
│   ├── run_backend.sh                     # Uvicorn entrypoint on configurable HOST/PORT (default 127.0.0.1:8000)
│   ├── main.py                            # FastAPI app setup, /health, /models, /ollama/*, /runs/*, /files/*
│   ├── config.py                          # Relative/env-overridable storage & adapter paths + MODEL_REGISTRY
│   ├── db.py                              # SQLite persistence for studio.db (runs + metadata)
│   ├── schemas.py                         # Strict Pydantic v2 models matching frontend/src/lib/types.ts
│   ├── engines/
│   │   ├── mlx_lora_engine.py             # ONLY file calling raw mlx.core tensors / mlx_vlm (macOS arm64 only)
│   │   ├── florence_engine.py             # Transformers Florence-2 Base/Large engine (CUDA -> MPS -> CPU)
│   │   └── ollama_engine.py               # Cross-platform Ollama REST & process manager
│   ├── services/
│   │   ├── job_manager.py                 # Single-concurrency queue worker & SSE streaming for /runs
│   │   ├── media_service.py               # Image inspection & 2-stage video keyframe + MP4 rendering bridge
│   │   ├── annotation_parser.py           # Robust JSON extractor for VLM coordinates -> normalized 0..1000
│   │   ├── few_shot_builder.py            # Prompt builders (zero-shot, few-shot, video stage 1/2)
│   │   └── schema_extractor.py            # Dynamic class/exclude parsing from natural language prompts
│   ├── auto_annotate/
│   │   ├── __init__.py
│   │   ├── config.py                      # OWLv2_MODELS, PYTORCH_DETECTORS, SAM_REFINERS, AUTO_STORAGE_DIR
│   │   ├── db.py                          # SQLite persistence for auto_annotate.db (jobs + items)
│   │   ├── schemas.py                     # Request/response schemas for inspect, jobs, smart-select, visual-embed
│   │   ├── roboflow_service.py            # Roboflow workspace/project discovery, batch fetch, ZIP & cloud push
│   │   ├── pytorch_engine.py              # YOLOE-26s-seg, SAM 2.1/MobileSAM, OWLv2 Base/Large & ViT Ref Bank
│   │   ├── manager.py                     # Auto-Annotate batch job queue worker with pause/cancel/resume
│   │   ├── router.py                      # FastAPI APIRouter mounted at /api/v1/auto-annotate
│   │   └── weights/                       # <100 MB .pt weights tracked in Git; >=100 MB (mobileclip2_b.ts) ignored
│   ├── adapters/                          # Ignored in Git (6x 126.1 MiB MLX-only Qwen3-VL LoRA safetensors)
│   └── storage/                           # Ignored runtime data (studio.db, auto_annotate.db, raw images, exports)
└── frontend/
    ├── .env.example                       # NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
    ├── package.json                       # Next.js 15, React 19, Tailwind CSS v4, lucide-react, TypeScript 5.8
    ├── next.config.ts
    └── src/
        ├── app/
        │   ├── layout.tsx, page.tsx, globals.css
        │   ├── analyze/page.tsx           # Workbench single-image & video evaluation UI
        │   ├── logs/page.tsx              # Historical evaluation run telemetry table
        │   ├── results/page.tsx           # Detailed inspection of detections, stage logs, video re-encode
        │   └── (auto-annotate)/           # Grouped layout providing AutoAnnotateProvider context
        │       ├── layout.tsx
        │       ├── auto-annotate/page.tsx
        │       ├── verify/page.tsx
        │       └── annotate/page.tsx
        ├── components/
        │   ├── AppShell.tsx               # Sidebar navigation, Local API status, Ollama toggle
        │   ├── AnnotationCanvas.tsx       # Evaluation result bbox/polygon overlay viewer
        │   └── VideoResultView.tsx        # Keyframe timeline + ByteTrack video output + re-encode form
        ├── lib/
        │   ├── types.ts, api.ts, mock-data.ts
        │   └── workbench-store.tsx        # Global backend health polling, SERVER_BOOT_ID tracking, Ollama state
        └── features/auto-annotate/
            ├── types.ts, api.ts, color-utils.ts
            ├── cache.ts                   # window.sessionStorage manager (`c1_aa_*` prefix)
            ├── store.tsx                  # AutoAnnotateProvider (Roboflow credentials/RAM + job list + cache lifecycle)
            ├── RoboflowSourceCard.tsx     # Reusable Roboflow API Key / Project / Fetch Filter configuration card
            ├── AutoAnnotateView.tsx       # Batch auto-annotation setup & live progress stream
            ├── VerifyQueueView.tsx        # Batch QA grid, filters, approve batch, Roboflow push & ZIP export
            ├── VerifyLightboxModal.tsx    # Full-resolution frame inspection modal in Verify Queue
            ├── ManualAnnotatorView.tsx    # Full annotation studio, layer manager, Smart Select & ViT Ref Bank logic
            ├── ManualAnnotatorCanvas.tsx  # Interactive zoom/pan/edit/draw canvas with staged green overlay preview
            └── FindObjectsAiPanel.tsx     # Left-rail "Find Objects with AI" panel (ViT Ref / OWL Text / YOLOE)
```

---

## 2. Core Technical Contracts

### Normalized Coordinate System (`0..1000`)
- All bounding boxes (`bbox_2d`) across backend databases, VLM parsers, PyTorch detectors, and canvas overlays use **integer `[xmin, ymin, xmax, ymax]` normalized to `[0, 1000]`**.
- All polygons use `[[x, y], ...]` normalized to `[0, 1000]`.
- Scaling back to image pixels happens strictly at render/export time (`yolo`, `coco`, OpenCV MP4 renderer, or canvas pixel projection).

### Concurrency & Memory Management
- Two single-worker background queues exist:
  1. `services.job_manager.JobManager` (for `/analyze` single runs)
  2. `auto_annotate.manager.AutoAnnotateManager` (for batch Roboflow jobs)
- Heavy model engines guard execution with module locks (`_ENGINE_LOCK`, `_FLORENCE_LOCK`, `_PT_LOCK`) and explicitly unload sibling engines (`mlx_engine.unload()`, `pytorch_engine.unload_pytorch()`, `florence_engine.unload()`) before loading weights so unified/GPU VRAM never OOMs.
- Device resolution on PyTorch (`pytorch_engine.py` & `florence_engine.py`) follows priority: **`cuda` (NVIDIA) -> `mps` (Apple Silicon) -> `cpu`**.

### Persistent Caching vs Session Invalidation
- Backend API emits a fresh UUID `bootId` in `GET /api/v1/health` on every server startup.
- Frontend `cache.ts` persists state in `window.sessionStorage` under keys prefixed `c1_aa_` (`draft`, `connection`, `jobs`, `last_verify_job`, `owl_model`, `verify_ui_v1:*`, `job_detail_v1:*`, `vit_refs_v1:*`, `vit_classes_v1:*`).
- Switching routes (`/auto-annotate` <-> `/verify` <-> `/annotate`) preserves all settings, selected batches, pinned ViT references, and target class selections during the browser session.
- If the backend restarts (`bootId` changes) or goes offline cleanly, `AutoAnnotateProvider` calls `clearAllAutoCaches()` so stale in-memory backend `queryId`s or offline artifacts are reset.

### ViT Reference Bank & Target Class Filtering
- Selecting specific **Target Classes** in `FindObjectsAiPanel.tsx` filters:
  1. Which frames and bounding boxes are pinned/displayed in the frontend (`activeVitRefsByItem`, `Use Frame`, `Use Annotated`).
  2. Which boxes and frames are sent to `POST /api/v1/auto-annotate/jobs/{jobId}/visual-embed/build-references` (`targetClasses`).
  3. Backend `build_visual_ref_bank()` filters `valid_boxes` **before** calling `_extract_owlv2_features_unlocked(img_path)`, completely skipping ViT feature extraction for images containing only excluded classes.
  4. Single-image `/visual-embed/search` and across-batch `/visual-embed/search-batch` evaluate only the matching class embedding buckets and scope `replace_class` merges strictly to the active classes.

---

## 3. Model Weights Strategy

- **Tracked directly in Git (`< 100 MB` each, 4 files / ~231 MiB total)**:
  - `yoloe-26s-seg.pt`, `mobile_sam.pt`, `sam2.1_t.pt`, `sam2.1_s.pt` inside `backend/auto_annotate/weights/`.
- **Excluded via `.gitignore` (15 total unpushed models/checkpoints)**:
  - `backend/auto_annotate/weights/mobileclip2_b.ts` (`242 MiB` — needed on NVIDIA for `YOLOE-26s-seg`).
  - `backend/adapters/qwen3_vl_4b_8bit_person_lora/` (6 MLX LoRA `.safetensors` files — Apple Silicon MLX only).
  - 6 Hugging Face Hub cached repos under `~/.cache/huggingface/hub/` (4 NVIDIA-compatible OWLv2/Florence-2 + 2 MLX-only Qwen3-VL).
  - 2 Ollama models (`qwen3-vl:8b`, `qwen3-vl:4b`).
- See **[`docs/06-model-weights-and-transfer.md`](./docs/06-model-weights-and-transfer.md)** for the full table and Google Cloud compression/transfer commands.
