# 01 — Setup & NVIDIA Deployment Guide

## 1. Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| **Python** | 3.11 – 3.13 | Virtualenv recommended at `backend/.venv` |
| **Node.js & npm** | Node 20+ | Required for the Next.js 15 frontend |
| **FFmpeg** | Any recent build (`libx264`) | Must be in `PATH` or specified via `FFMPEG_BIN` |
| **Ollama** *(optional)* | Latest | Needed only if running `ollama-8b-zero-shot` / local Qwen3-VL models |
| **NVIDIA Driver + CUDA** *(NVIDIA PC)* | CUDA 12.1 / 12.4 / 12.6 | Verify with `nvidia-smi` |

---

## 2. Environment Configuration

Copy the tracked `.env.example` templates before starting the servers:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### Important Environment Variables (`backend/.env`)

- `HOST` / `PORT` — FastAPI listen host/port (defaults: `127.0.0.1` / `8000`).
- `CORS_ORIGINS` — Allowed browser origins (`http://127.0.0.1:3000,http://localhost:3000`). If accessing over LAN on an NVIDIA server, add its IP/hostname here and update `frontend/.env.local` (`NEXT_PUBLIC_API_BASE_URL=http://<server-ip>:8000`).
- `OLLAMA_BASE_URL` / `OLLAMA_BIN` — Ollama API URL (`http://127.0.0.1:11434`) and optional custom executable path.
- `FFMPEG_BIN` — Optional custom FFmpeg executable path if `ffmpeg` is not on `PATH`.
- `STORAGE_DIR` — Optional override for SQLite DBs, downloaded Roboflow batches, and rendered outputs (default: `./storage` under `backend/`).
- `AUTO_WEIGHTS_DIR` — Optional override for `.pt` / `.ts` checkpoints (default: `./auto_annotate/weights`).
- `FLORENCE_DTYPE` — `float16` (default on CUDA/MPS) or `float32`.

---

## 3. Installation

### A. NVIDIA GPU Workstation (Linux / Windows)

1. Create and activate the backend environment:

   ```bash
   cd backend
   python3 -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell:
   # .venv\Scripts\Activate.ps1

   pip install --upgrade pip
   pip install -r requirements-nvidia.txt
   ```

   > `backend/requirements.txt` uses PEP 508 platform markers (`sys_platform == "darwin" and platform_machine == "arm64"`) for `mlx` and `mlx-vlm`, so they are automatically skipped on Linux and Windows.

2. Verify PyTorch sees your NVIDIA GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

3. Restore large/external weights (see [06 — Model Weights & Transfer](./06-model-weights-and-transfer.md)):
   - Place `mobileclip2_b.ts` inside `backend/auto_annotate/weights/` if using `YOLOE-26s-seg`.
   - Restore `~/.cache/huggingface/hub/` models (OWLv2 Base/Large, Florence-2 Base/Large) or allow Hugging Face Hub to download on first run.
   - Optionally install Ollama and run `ollama pull qwen3-vl:8b`.

4. Install frontend dependencies:

   ```bash
   cd ../frontend
   npm install
   ```

### B. Apple Silicon Mac (macOS arm64)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd ../frontend
npm install
```

---

## 4. Starting & Verifying

### One-Command Startup (macOS / Linux)

```bash
./run.sh
```

### Manual Two-Terminal Startup (Linux / Windows / macOS)

Terminal 1 (Backend):
```bash
cd backend
# macOS / Linux:
./run_backend.sh
# Windows:
# .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Open **http://127.0.0.1:3000**. In the left sidebar:
- **Status dot** turns green (`Local API Online`).
- **Device banner** confirms **`NVIDIA CUDA (<GPU>)`** on NVIDIA PCs or **`Apple Silicon MLX (arm64)`** on Mac.
- Check **[`../MLX_dependencies.md`](../MLX_dependencies.md)** for which models are Apple-only vs CUDA-native.
