"""Central configuration and model registry for the Control One local FastAPI backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

ADAPTERS_DIR = Path(
    os.getenv("ADAPTERS_DIR", str(BACKEND_DIR / "adapters"))
).expanduser()
ADAPTER_BEST_DIR = (
    ADAPTERS_DIR / "qwen3_vl_4b_8bit_person_lora" / "best"
)
ADAPTER_FINAL_DIR = (
    ADAPTERS_DIR / "qwen3_vl_4b_8bit_person_lora"
)

STORAGE_DIR = Path(
    os.getenv("STORAGE_DIR", str(BACKEND_DIR / "storage"))
).expanduser()
TARGETS_DIR = STORAGE_DIR / "targets"
REFERENCES_DIR = STORAGE_DIR / "references"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
KEYFRAMES_DIR = OUTPUTS_DIR / "keyframes"
DB_PATH = STORAGE_DIR / "studio.db"

for _d in (STORAGE_DIR, TARGETS_DIR, REFERENCES_DIR, OUTPUTS_DIR, KEYFRAMES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"
    ).split(",")
    if o.strip()
]

PERSON_LORA_EXACT_PROMPT = (
    "Locate every person in the image and output tight 2D bounding boxes in JSON "
    'format as a list of objects with keys "bbox_2d" ([x1, y1, x2, y2]) and '
    '"label" ("person"). If no person is present, return [].'
)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlx-4b-8bit-lora-best": {
        "id": "mlx-4b-8bit-lora-best",
        "label": "MLX Qwen3-VL 4B (8-bit) + LoRA (best · iter 180)",
        "short_label": "MLX 4B-8bit + LoRA (best)",
        "engine": "mlx",
        "model_path": os.getenv(
            "MLX_8BIT_MODEL_PATH", "mlx-community/Qwen3-VL-4B-Instruct-8bit"
        ),
        "adapter_path": str(ADAPTER_BEST_DIR),
        "adapter_label": "qwen3_vl_4b_8bit_person_lora/best",
        "quantization": "8-bit affine",
    },
    "mlx-4b-8bit-lora-final": {
        "id": "mlx-4b-8bit-lora-final",
        "label": "MLX Qwen3-VL 4B (8-bit) + LoRA (final · iter 360)",
        "short_label": "MLX 4B-8bit + LoRA (final)",
        "engine": "mlx",
        "model_path": os.getenv(
            "MLX_8BIT_MODEL_PATH", "mlx-community/Qwen3-VL-4B-Instruct-8bit"
        ),
        "adapter_path": str(ADAPTER_FINAL_DIR),
        "adapter_label": "qwen3_vl_4b_8bit_person_lora",
        "quantization": "8-bit affine",
    },
    "mlx-4b-8bit-base": {
        "id": "mlx-4b-8bit-base",
        "label": "MLX Qwen3-VL 4B (8-bit) — Base Only",
        "short_label": "MLX 4B-8bit Base",
        "engine": "mlx",
        "model_path": os.getenv(
            "MLX_8BIT_MODEL_PATH", "mlx-community/Qwen3-VL-4B-Instruct-8bit"
        ),
        "adapter_path": None,
        "adapter_label": None,
        "quantization": "8-bit affine",
    },
    "mlx-4b-4bit-base": {
        "id": "mlx-4b-4bit-base",
        "label": "MLX Qwen3-VL 4B (4-bit) — Base Only",
        "short_label": "MLX 4B-4bit Base",
        "engine": "mlx",
        "model_path": os.getenv(
            "MLX_4BIT_MODEL_PATH", "mlx-community/Qwen3-VL-4B-Instruct-4bit"
        ),
        "adapter_path": None,
        "adapter_label": None,
        "quantization": "4-bit affine",
    },
    "ollama-8b-zero-shot": {
        "id": "ollama-8b-zero-shot",
        "label": "Ollama Qwen3-VL 8B (Q4_K_M) — Zero-Shot",
        "short_label": "Ollama 8B Zero-Shot",
        "engine": "ollama",
        "model_path": os.getenv("OLLAMA_8B_MODEL", "qwen3-vl:8b"),
        "adapter_path": None,
        "adapter_label": None,
        "quantization": "Q4_K_M GGUF",
    },
    "florence2-large": {
        "id": "florence2-large",
        "label": "Florence-2 Large FT (0.77B · PyTorch MPS)",
        "short_label": "Florence-2 Large",
        "engine": "florence2",
        "model_path": os.getenv(
            "FLORENCE2_LARGE_MODEL_PATH", "florence-community/Florence-2-large-ft"
        ),
        "adapter_path": None,
        "adapter_label": None,
        "quantization": "FP16 MPS",
    },
    "florence2-base": {
        "id": "florence2-base",
        "label": "Florence-2 Base FT (0.23B · PyTorch MPS)",
        "short_label": "Florence-2 Base",
        "engine": "florence2",
        "model_path": os.getenv(
            "FLORENCE2_BASE_MODEL_PATH", "florence-community/Florence-2-base-ft"
        ),
        "adapter_path": None,
        "adapter_label": None,
        "quantization": "FP16 MPS",
    },
}


def get_model_entry(model_id: str) -> Dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'.")
    return MODEL_REGISTRY[model_id]


def is_adapter_present(entry: Dict[str, Any]) -> bool:
    path = entry.get("adapter_path")
    if not path:
        return True
    adapter_dir = Path(path)
    return adapter_dir.is_dir() and (adapter_dir / "adapters.safetensors").is_file()


def resolve_adapter_path(entry: Dict[str, Any]) -> Optional[str]:
    path = entry.get("adapter_path")
    if not path:
        return None
    adapter_dir = Path(path)
    if not adapter_dir.is_dir() or not (adapter_dir / "adapters.safetensors").is_file():
        raise FileNotFoundError(
            f"LoRA adapter directory or weights (adapters.safetensors) not found: {path}"
        )
    return str(path)
