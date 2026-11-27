# Control One — Auto-Annotator & VLM Evaluation Workbench

A full-stack computer vision evaluation, batch auto-annotation, verification, and interactive visual-grounding annotation workbench built with **Next.js 15**, **FastAPI**, **PyTorch (CUDA / MPS / CPU)**, **SAM 2.1**, **OWLv2**, **Florence-2**, **YOLOE-26s-seg**, **ByteTrack**, **Roboflow SDK**, and **Qwen3-VL (Ollama / MLX-VLM)**.

---

## Core Capabilities

1. **VLM Evaluation Workbench (`/analyze`, `/logs`, `/results`)**
   - Single-image and two-stage video event evaluation across **Qwen3-VL**, **Florence-2**, and **Ollama**.
   - Supports **Zero-Shot** and **Few-Shot** visual prompting, normalized `0..1000` bounding boxes, polygons, ByteTrack post-event video tracking, and FFmpeg H.264 video rendering.
2. **Roboflow Auto-Annotate Pipeline (`/auto-annotate`, `/verify`)**
   - Connects directly to Roboflow projects via Private API Key and fetches unannotated, annotated, batch-scoped, annotation-job, or assignee-filtered datasets in structured batches (default 50 images/batch).
   - Runs **OWLv2**, **YOLOE-26s-seg**, **Florence-2**, or **Qwen3-VL** paired with optional **SAM 2.1 / MobileSAM** polygon refinement and streams live progress over SSE.
   - Provides a batch **Verify Queue** with Lightbox inspection, batch approval (`Accept All`, `Null`, `Reject` with reason tracking), rejected-batch retry, local YOLO/COCO/JSON ZIP export, and direct push back to Roboflow.
3. **Manual Annotator, Smart Select & ViT Reference Bank (`/annotate`)**
   - Built-in Roboflow fetch panel + persistent per-session caching (`sessionStorage` with boot-ID invalidation).
   - Canvas editor supporting `BBox`, `Polygon`, `Drag`, interactive **SAM 2.1 Smart Select** (hover preview + multi-point), Undo/Redo, and per-class palette controls.
   - **Find Objects with AI**:
     - **ViT Ref**: Pin annotated boxes/frames or draw canvas positive/negative exemplars to search via cached **OWLv2 Base P16** (`768-d`) or **OWLv2 Large P14** (`1024-d`) patch embeddings, with **Target Classes** filtering (single-class `Only` or multi-class indexing & batch scan), non-destructive green **Overlay Preview** acceptance (`Enter` / `Esc`), and `Append` / `Replace Matched Classes` merge modes.
     - **OWL Text** & **YOLOE**: Open-vocabulary single-image detection with clickable **Target Classes** pills synced to prompt input.

---

## Architecture Overview

```text
+-------------------------------------------------------------------+
| Next.js 15 Frontend (http://127.0.0.1:3000)                       |
|  /analyze  /logs  /results  /auto-annotate  /verify  /annotate    |
+---------------------------------+---------------------------------+
                                  | HTTP + SSE + NDJSON Streams
+---------------------------------v---------------------------------+
| FastAPI Backend (http://127.0.0.1:8000)                           |
|  - Core Evaluation Router      : backend/main.py                  |
|  - Auto-Annotate & Studio API  : backend/auto_annotate/router.py  |
|  - PyTorch CUDA/MPS/CPU Engine : pytorch_engine.py / florence.py  |
|  - Ollama Local Client         : backend/engines/ollama_engine.py |
|  - Apple Silicon MLX Engine    : backend/engines/mlx_lora_engine  |
|  - SQLite Persistence          : studio.db & auto_annotate.db     |
+-------------------------------------------------------------------+
```

---

## Quick Start

### 1. Clone Repository & Configure Environment

```bash
git clone https://github.com/mRiyaan/Auto-annotator-C1.git
cd Auto-annotator-C1
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### 2. Backend Setup

#### Option A — NVIDIA GPU (Linux / Windows with CUDA 12.x)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-nvidia.txt
```

#### Option B — Apple Silicon Mac (MPS + MLX-VLM)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run Both Services

```bash
# From repository root (macOS / Linux):
./run.sh

# Or separately:
./backend/run_backend.sh
npm --prefix frontend run dev
```

Open **http://127.0.0.1:3000** in your browser. System prerequisites: **Python 3.11–3.13**, **Node.js 20+**, **FFmpeg** on `PATH` (or `FFMPEG_BIN`), and optional **Ollama** (`ollama pull qwen3-vl:8b`).

---

## Documentation Index

- **[`context.md`](./context.md)**: Technical architecture, API surface, storage schema, caching model, and coordinate conventions.
- **[`MLX_dependencies.md`](./MLX_dependencies.md)**: Exact audit of raw MLX tensor usage, directory tree, and NVIDIA CUDA migration guide.
- **[`docs/README.md`](./docs/README.md)**: User guides for every route, feature, keyboard shortcut, and workflow:
  - [01 — Setup & NVIDIA Deployment](./docs/01-setup-and-nvidia-deployment.md)
  - [02 — Evaluation Workbench (`/analyze`, `/logs`, `/results`)](./docs/02-evaluation-workbench.md)
  - [03 — Auto-Annotate & Verify Queue (`/auto-annotate`, `/verify`)](./docs/03-auto-annotate-pipeline.md)
  - [04 — Manual Annotator & Smart Select (`/annotate`)](./docs/04-manual-annotator-and-smart-select.md)
  - [05 — ViT Reference Bank, Target Classes & AI Find](./docs/05-vit-reference-bank-and-ai-find.md)
  - [06 — Model Weights Inventory & Google Cloud Transfer](./docs/06-model-weights-and-transfer.md)
