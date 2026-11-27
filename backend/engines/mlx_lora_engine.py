"""Apple Silicon native MLX-VLM engine (lazy Metal imports + single-model memory cache)."""
from __future__ import annotations

import gc
import importlib.util
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

import config

_ENGINE_LOCK = threading.Lock()


class MLXLoRAEngine:
    """Manages loading, caching, and unloading MLX-VLM models + LoRA adapters on 18 GB M3 Pro."""

    def __init__(self) -> None:
        self._loaded_id: Optional[str] = None
        self._model: Any = None
        self._processor: Any = None
        self._model_config: Any = None

    @property
    def loaded_model_id(self) -> Optional[str]:
        return self._loaded_id

    @staticmethod
    def is_mlx_available() -> bool:
        return (
            importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx_vlm") is not None
        )

    @staticmethod
    def is_hf_model_cached(model_path: str) -> bool:
        if Path(model_path).exists():
            return True
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_name = f"models--{model_path.replace('/', '--')}"
        snap_dir = hub_dir / cache_name / "snapshots"
        return snap_dir.is_dir() and any(snap_dir.iterdir())

    def unload(self) -> None:
        with _ENGINE_LOCK:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        if self._model is not None or self._processor is not None:
            self._model = None
            self._processor = None
            self._model_config = None
            self._loaded_id = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass

    def _ensure_loaded_unlocked(self, model_id: str) -> None:
        if self._loaded_id == model_id and self._model is not None:
            return

        entry = config.get_model_entry(model_id)
        if entry["engine"] != "mlx":
            raise ValueError(f"Model '{model_id}' is not an MLX engine model.")
        if not self.is_mlx_available():
            raise RuntimeError(
                "MLX-VLM is only available on Apple Silicon macOS. Use Florence-2, OWLv2, YOLOE, or Ollama on NVIDIA/Linux/Windows."
            )

        model_path = entry["model_path"]
        adapter_path = config.resolve_adapter_path(entry)

        self._unload_unlocked()

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(model_path, adapter_path=adapter_path)
        model_config = load_config(model_path)

        self._model = model
        self._processor = processor
        self._model_config = model_config
        self._loaded_id = model_id

    def generate_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: Any,
        images: List[Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cache_limit_bytes: int = int(1.5 * 1024**3),
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.is_mlx_available():
            raise RuntimeError(
                "MLX-VLM is only available on Apple Silicon macOS. Use Florence-2, OWLv2, YOLOE, or Ollama on NVIDIA/Linux/Windows."
            )

        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        with _ENGINE_LOCK:
            t_load0 = time.perf_counter()
            was_loaded = self._loaded_id == model_id and self._model is not None
            self._ensure_loaded_unlocked(model_id)
            load_s = 0.0 if was_loaded else round(time.perf_counter() - t_load0, 3)

            formatted_prompt = apply_chat_template(
                self._processor,
                self._model_config,
                prompt_text,
                num_images=len(images),
                enable_thinking=False,
            )

            mx.reset_peak_memory()
            t0 = time.perf_counter()
            res = generate(
                self._model,
                self._processor,
                formatted_prompt,
                images,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=False,
                verbose=False,
            )
            latency_s = round(time.perf_counter() - t0, 3)
            peak_gb = round(float(res.peak_memory or 0.0), 3)

            if cache_limit_bytes and mx.get_cache_memory() > cache_limit_bytes:
                mx.clear_cache()

            rss_gb = round(psutil.Process().memory_info().rss / 1e9, 3)
            telemetry = {
                "load_s": load_s,
                "latency_s": latency_s,
                "prompt_tokens": int(res.prompt_tokens or 0),
                "generation_tokens": int(res.generation_tokens or 0),
                "prompt_tps": round(float(res.prompt_tps or 0.0), 2),
                "generation_tps": round(float(res.generation_tps or 0.0), 2),
                "peak_memory_gb": peak_gb if peak_gb > 0 else rss_gb,
                "finish_reason": str(res.finish_reason or "stop"),
            }
            return str(res.text or ""), telemetry


mlx_engine = MLXLoRAEngine()
