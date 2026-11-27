"""Isolated configuration for Roboflow Auto-Annotation, PyTorch MPS weights, and local verified datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as base_config

DEFAULT_OWLV2_MODEL_ID = "google/owlv2-base-patch16-ensemble"

OWLV2_MODELS: Dict[str, Dict[str, Any]] = {
    "google/owlv2-base-patch16-ensemble": {
        "id": "owlv2-base-patch16-ensemble",
        "hf_model_id": "google/owlv2-base-patch16-ensemble",
        "label": "OWLv2 Base Patch16 Ensemble (Fast · ~155M)",
        "short_label": "OWLv2 Base P16",
        "quantization": "FP32 CUDA/MPS/CPU",
        "approx_size": "~600 MB",
        "description": "Default Google OWLv2 Base (ViT-B/16) — fast open-vocab & 768-d visual reference search.",
    },
    "google/owlv2-large-patch14-ensemble": {
        "id": "owlv2-large-patch14-ensemble",
        "hf_model_id": "google/owlv2-large-patch14-ensemble",
        "label": "OWLv2 Large Patch14 Ensemble (High Precision · ~430M)",
        "short_label": "OWLv2 Large P14",
        "quantization": "FP32 CUDA/MPS/CPU",
        "approx_size": "~1.7 GB",
        "description": "Larger Google OWLv2 Large (ViT-L/14) — finer 14x14 patches & richer 1024-d visual embeddings.",
    },
}

OWLV2_ALIAS_MAP: Dict[str, str] = {
    "owlv2-base-patch16-ensemble": "google/owlv2-base-patch16-ensemble",
    "google/owlv2-base-patch16-ensemble": "google/owlv2-base-patch16-ensemble",
    "owlv2-large-patch14-ensemble": "google/owlv2-large-patch14-ensemble",
    "google/owlv2-large-patch14-ensemble": "google/owlv2-large-patch14-ensemble",
}


def resolve_owlv2_model_id(raw: Optional[str]) -> str:
    clean = (raw or "").strip()
    if not clean:
        return DEFAULT_OWLV2_MODEL_ID
    resolved = OWLV2_ALIAS_MAP.get(clean)
    if resolved:
        return resolved
    raise ValueError(
        f"Unsupported OWLv2 model '{clean}'. Supported: {', '.join(OWLV2_MODELS.keys())}."
    )

AUTO_ANNOTATE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(
    os.getenv("AUTO_WEIGHTS_DIR", str(AUTO_ANNOTATE_DIR / "weights"))
).expanduser()

AUTO_STORAGE_DIR = base_config.STORAGE_DIR / "auto_annotate"
AUTO_JOBS_DIR = AUTO_STORAGE_DIR / "jobs"
AUTO_DB_PATH = AUTO_STORAGE_DIR / "auto_annotate.db"

for _d in (AUTO_STORAGE_DIR, AUTO_JOBS_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_BATCH_SIZE = 50

DEFAULT_CLASS_COLORS: List[str] = [
    "#38BDF8",  # Sky cyan
    "#4ADE80",  # Emerald green
    "#F2853F",  # Control One orange
    "#E879F9",  # Magenta
    "#FACC15",  # Amber yellow
    "#FB7185",  # Coral rose
    "#A78BFA",  # Violet
    "#2DD4BF",  # Teal
    "#60A5FA",  # Royal blue
    "#F97316",  # Bright tangerine
]

PYTORCH_DETECTORS: Dict[str, Dict[str, Any]] = {
    "yoloe-26s-seg": {
        "id": "yoloe-26s-seg",
        "label": "YOLOE-26s-seg (PyTorch · Open-Vocab)",
        "short_label": "YOLOE-26s-seg",
        "engine": "pytorch",
        "quantization": "FP16/FP32 CUDA/MPS",
        "weight_path": WEIGHTS_DIR / "yoloe-26s-seg.pt",
        "supports_polygon": True,
        "supports_few_shot": False,
        "description": "Dataset-agnostic open-vocabulary YOLOE segmentation/detection checkpoint.",
    },
    "owlv2-base-patch16-ensemble": {
        "id": "owlv2-base-patch16-ensemble",
        "label": "OWLv2 Base Patch16 Ensemble (PyTorch · Open-Vocab)",
        "short_label": "OWLv2 Base P16",
        "engine": "owlv2",
        "quantization": "FP32 CUDA/MPS/CPU",
        "weight_path": None,
        "model_path": "google/owlv2-base-patch16-ensemble",
        "approx_size": "~600 MB",
        "supports_polygon": True,
        "supports_few_shot": False,
        "description": "Google OWLv2 Base (ViT-B/16) zero-shot detector; combine with SAM 2.1 refiner for polygons.",
    },
    "owlv2-large-patch14-ensemble": {
        "id": "owlv2-large-patch14-ensemble",
        "label": "OWLv2 Large Patch14 Ensemble (PyTorch · High Precision)",
        "short_label": "OWLv2 Large P14",
        "engine": "owlv2",
        "quantization": "FP32 CUDA/MPS/CPU",
        "weight_path": None,
        "model_path": "google/owlv2-large-patch14-ensemble",
        "approx_size": "~1.7 GB",
        "supports_polygon": True,
        "supports_few_shot": False,
        "description": "Google OWLv2 Large (ViT-L/14) zero-shot detector with finer 14x14 patches; combine with SAM 2.1 for polygons.",
    },
}

SAM_REFINERS: Dict[str, Dict[str, Any]] = {
    "none": {
        "id": "none",
        "label": "None (Use Detector Output)",
        "weight_path": None,
        "description": "Keep native bounding boxes or polygons from the selected detector.",
    },
    "sam2.1-t": {
        "id": "sam2.1-t",
        "label": "SAM 2.1 Tiny (Fast Polygon Refiner)",
        "weight_path": WEIGHTS_DIR / "sam2.1_t.pt",
        "description": "Converts detected 2D boxes into tight segmentation polygons quickly.",
    },
    "sam2.1-s": {
        "id": "sam2.1-s",
        "label": "SAM 2.1 Small (Higher Precision Refiner)",
        "weight_path": WEIGHTS_DIR / "sam2.1_s.pt",
        "description": "Higher-accuracy SAM 2.1 Small mask refinement from detected boxes.",
    },
    "mobile-sam": {
        "id": "mobile-sam",
        "label": "MobileSAM (Lightweight Polygon Refiner)",
        "weight_path": WEIGHTS_DIR / "mobile_sam.pt",
        "description": "Ultra-light MobileSAM box-to-polygon refiner for large batches.",
    },
}


def get_detector_entry(model_id: str) -> Dict[str, Any]:
    if model_id in PYTORCH_DETECTORS:
        return PYTORCH_DETECTORS[model_id]
    return base_config.get_model_entry(model_id)
