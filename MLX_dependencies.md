# MLX Dependencies & Apple Silicon Audit (NVIDIA Migration Reference)

This document maps **every file and directory** in the repository that uses **Apple MLX (`mlx.core` / `mlx_vlm`)**, Apple Silicon Metal (`mps`) fallbacks, or MLX-only quantized LoRA checkpoints, so you can cleanly port or replace those components on an **NVIDIA CUDA** workstation.

---

## 1. Exact Directory Structure of MLX-Related Files

```text
<repo-root>/
├── backend/
│   ├── requirements.txt                           # Conditional PEP 508 markers: mlx / mlx-vlm (macOS arm64 only)
│   ├── requirements-nvidia.txt                    # NVIDIA CUDA 12.x requirements (excludes mlx / mlx-vlm)
│   ├── .env.example                               # MLX_8BIT_MODEL_PATH & MLX_4BIT_MODEL_PATH vars
│   ├── config.py                                  # Registers "mlx-*" model IDs in MODEL_REGISTRY & ADAPTERS_DIR
│   ├── schemas.py                                 # Literal ModelOptionId includes "mlx-4b-*" options
│   ├── main.py                                    # Health & /models availability gating via is_mlx_available()
│   ├── adapters/                                  # [GIT-IGNORED · MLX-ONLY] 6 affine-quantized LoRA .safetensors
│   │   └── qwen3_vl_4b_8bit_person_lora/
│   │       ├── adapter_config.json                # 8-bit group_size=64 affine LoRA config (MLX format)
│   │       ├── adapters.safetensors               # 126.1 MiB (iter 360)
│   │       ├── 0000090_adapters.safetensors       # 126.1 MiB
│   │       ├── 0000180_adapters.safetensors       # 126.1 MiB
│   │       ├── 0000270_adapters.safetensors       # 126.1 MiB
│   │       ├── 0000360_adapters.safetensors       # 126.1 MiB
│   │       └── best/
│   │           └── adapters.safetensors           # 126.1 MiB (iter 180 best val checkpoint)
│   ├── engines/
│   │   ├── mlx_lora_engine.py                     # *** ONLY SCRIPT EXECUTING RAW MLX TENSORS (`mlx.core`) ***
│   │   └── florence_engine.py                     # PyTorch Transformers (CUDA -> MPS -> CPU); calls mlx_engine.unload()
│   ├── services/
│   │   ├── job_manager.py                         # Dispatches "engine == 'mlx'" runs & unloads MLX before Ollama/Florence
│   │   └── few_shot_builder.py                    # Formats interleaved chat messages consumed by mlx_lora_engine
│   └── auto_annotate/
│       ├── pytorch_engine.py                      # PyTorch engine (CUDA -> MPS -> CPU); calls mlx_engine.unload()
│       ├── manager.py                             # Batch auto-annotate dispatcher ("mlx" vs "florence2" vs "pytorch"/"owlv2"/"ollama")
│       └── router.py                              # /api/v1/auto-annotate/models checks is_mlx_available()
└── frontend/
    └── src/
        ├── app/
        │   ├── layout.tsx                         # App metadata description mentioning MLX LoRA
        │   └── analyze/page.tsx                   # Default modelId + UI display copy mentioning MLX-VLM
        ├── features/auto-annotate/
        │   └── AutoAnnotateView.tsx               # Initial modelId state ("mlx-4b-8bit-base")
        └── lib/
            ├── types.ts                           # ModelOptionId union ("mlx-4b-*") + engine label "MLX-VLM"
            └── mock-data.ts                       # Model card metadata descriptions
```

---

## 2. Script Using Raw MLX Tensors (Exact Audit)

**Only ONE Python file in the entire repository imports `mlx` or operates on raw MLX tensors / Metal unified cache:**

### `backend/engines/mlx_lora_engine.py`

Located at: `<repo-root>/backend/engines/mlx_lora_engine.py`

| Item | Lines / Symbols | What It Uses | Behavior on Non-Mac / NVIDIA PC |
|---|---|---|---|
| **Availability Probe** | `MLXLoRAEngine.is_mlx_available()` | `importlib.util.find_spec("mlx")` and `find_spec("mlx_vlm")` | Returns `False` on Linux / Windows; prevents import errors at server startup. |
| **Raw MLX Tensor & Cache API** | `_unload_unlocked()` (lazy import inside method) | `import mlx.core as mx` -> `mx.clear_cache()` | Only called if `_model is not None`; wrapped in `try/except`. |
| **Model & Affine Adapter Loader** | `_ensure_loaded_unlocked()` | `from mlx_vlm import load`, `from mlx_vlm.utils import load_config` | Guarded by `if not self.is_mlx_available(): raise RuntimeError(...)`. |
| **Multimodal MLX Inference** | `generate_multimodal()` | `import mlx.core as mx` -> `mx.reset_peak_memory()`, `mx.get_cache_memory()`, `mx.clear_cache()`; `from mlx_vlm import generate`; `from mlx_vlm.prompt_utils import apply_chat_template` | Guarded at entry by `is_mlx_available()`; raises descriptive `RuntimeError` pointing user to CUDA PyTorch / Ollama models. |
| **Hugging Face Cache Helper** | `is_hf_model_cached()` | Pure `pathlib` check under `~/.cache/huggingface/hub/` (zero MLX import) | Works identically on Linux / Windows / macOS (reused by `florence_engine.py` and `pytorch_engine.py`). |

### Why Raw MLX Weights Cannot Run Directly on NVIDIA
- The `mlx-community/Qwen3-VL-4B-Instruct-8bit` / `4bit` models and `backend/adapters/qwen3_vl_4b_8bit_person_lora/*.safetensors` use **MLX affine quantization** (`bits=8, group_size=64, mode="affine"` weights + scales + biases layout).
- NVIDIA PyTorch/CUDA uses **Hugging Face Transformers + PEFT / bitsandbytes / AWQ / GPTQ / GGUF** instead of MLX affine tensors.

---

## 3. Indirect Backend & Config References (Zero Raw MLX Tensors)

These files do **not** import `mlx` directly; they only reference `mlx_engine` for lifecycle orchestration or list `mlx-*` IDs in registries:

1. **`backend/config.py`**:
   - Defines `"mlx-4b-8bit-lora-best"`, `"mlx-4b-8bit-lora-final"`, `"mlx-4b-8bit-base"`, `"mlx-4b-4bit-base"` inside `MODEL_REGISTRY` and helper `is_adapter_present()`.
2. **`backend/schemas.py`**:
   - `ModelOptionId = Literal["mlx-4b-8bit-lora-best", ..., "florence2-base"]`.
3. **`backend/main.py`**:
   - Calls `mlx_engine.is_mlx_available()` and `config.is_adapter_present(entry)` in `/api/v1/models` so MLX models automatically report `available: false` (`"Apple Silicon MLX only (unavailable on this host)"`) on NVIDIA/Linux/Windows instead of crashing.
   - Reports `"NVIDIA CUDA (<GPU Name>)"` in `/api/v1/health` when `torch.cuda.is_available()`.
4. **`backend/services/job_manager.py`**:
   - Dispatches `entry["engine"] == "mlx"` jobs to `mlx_engine.generate_multimodal()`, and calls `mlx_engine.unload()` (no-op when unloaded) before running Florence-2 or Ollama.
5. **`backend/services/few_shot_builder.py`**:
   - Function `build_interleaved_image_messages()` constructs plain Python dicts (`[{"role": "user", "content": ...}]`) — no MLX library imports.
6. **`backend/engines/florence_engine.py`**:
   - Native PyTorch Transformers engine; prioritizes **`cuda` -> `mps` -> `cpu`**. Only calls `mlx_engine.is_hf_model_cached()` and `mlx_engine.unload()` (both safe no-ops on NVIDIA).
7. **`backend/auto_annotate/pytorch_engine.py`**:
   - Native PyTorch engine (**YOLOE-26s-seg**, **SAM 2.1 / MobileSAM**, **OWLv2 Base/Large**, **ViT Reference Bank**); prioritizes **`cuda` -> `mps` -> `cpu`**. Only calls `mlx_engine.is_hf_model_cached()` and `mlx_engine.unload()` (no raw MLX tensors).
8. **`backend/auto_annotate/manager.py`** & **`backend/auto_annotate/router.py`**:
   - Calls `mlx_engine.is_mlx_available()`, `mlx_engine.unload()`, or dispatches to `mlx_engine.generate_multimodal()` only when the user selects an `mlx-*` model ID.
9. **`backend/requirements.txt`**:
   - `mlx>=0.32.4` and `mlx-vlm>=0.7.12` use PEP 508 environment markers (`; sys_platform == "darwin" and platform_machine == "arm64"`), so `pip install` automatically skips them on Linux/Windows/NVIDIA.

---

## 4. NVIDIA CUDA Runtime Status & Replacement Options

### What Already Runs Natively on NVIDIA CUDA Right Now
- **Manual Annotator (`/annotate`)**:
  - **ViT Reference Bank + Single/Multi-Class Indexing & Batch Overlay**: **OWLv2 Base P16** and **OWLv2 Large P14** -> runs on `cuda` automatically via `PyTorchVisionEngine`.
  - **Smart Select & Polygon Refinement**: **SAM 2.1 Small**, **SAM 2.1 Tiny**, **MobileSAM** -> runs on `cuda` automatically.
  - **Open-Vocab `YOLOE` & `OWL Text` Detection**: **YOLOE-26s-seg** & **OWLv2** -> runs on `cuda` automatically.
- **Auto-Annotate (`/auto-annotate` & `/verify`)**:
  - **Florence-2 Large FT** & **Florence-2 Base FT** -> runs on `cuda` automatically (`float16` or `float32`).
  - **YOLOE-26s-seg**, **OWLv2 Base/Large** (+ SAM 2.1 / MobileSAM refiners) -> runs on `cuda` automatically.
  - **Ollama Qwen3-VL 8B (`qwen3-vl:8b`)** -> runs on NVIDIA GPU automatically via Ollama CUDA runtime.
- **Evaluation Workbench (`/analyze`, `/logs`, `/results`)**:
  - **Florence-2 Large / Base**, **Ollama Qwen3-VL 8B**, **ByteTrack**, **OpenCV + FFmpeg H.264**.

### Replacing `mlx_lora_engine.py` on NVIDIA (If Qwen3-VL 4B Transformers/CUDA LoRA is Wanted)
1. Use Ollama directly (`ollama pull qwen3-vl:4b` or `ollama pull qwen3-vl:8b`) — zero Python VLM code changes needed.
2. Or add a PyTorch CUDA engine (e.g., `backend/engines/hf_vlm_engine.py`) using `Qwen3VLForConditionalGeneration` + `AutoProcessor` (`Qwen/Qwen3-VL-4B-Instruct` or AWQ/bitsandbytes 4-bit/8-bit) and standard **PEFT LoRA** retrained on CUDA (note: existing `backend/adapters/qwen3_vl_4b_8bit_person_lora/` files are MLX affine checkpoints and cannot be loaded directly by `peft`).
3. Update `AutoAnnotateView.tsx` (line 94) / `analyze/page.tsx` default `modelId` from `"mlx-4b-8bit-base"` to `"ollama-8b-zero-shot"` or `"florence2-large"` on NVIDIA-only deployments if you want a CUDA default pre-selected in the UI.
