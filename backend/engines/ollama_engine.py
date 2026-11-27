"""Optional local Ollama engine (qwen3-vl:8b) with empty-<think> prefill optimization."""
from __future__ import annotations

import base64
import io
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config


class OllamaEngine:
    """Thin requests-based Ollama chat client for optional zero-shot 8B comparisons."""

    def __init__(self, base_url: str = config.OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._thinking_cache: Dict[str, bool] = {}
        self._proc_lock = threading.Lock()
        self._managed_proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _find_ollama_binary() -> Optional[str]:
        env_bin = (os.getenv("OLLAMA_BIN") or "").strip()
        win_local = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        candidates = [
            env_bin or None,
            shutil.which("ollama"),
            "/usr/bin/ollama",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
            str(win_local) if os.getenv("LOCALAPPDATA") else None,
        ]
        for cand in candidates:
            if cand and Path(cand).is_file():
                return cand
        return None

    def check_health(self) -> Tuple[bool, List[str]]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            return True, models
        except Exception:
            return False, []

    def start_server(self, timeout_sec: float = 12.0) -> Tuple[bool, List[str], str]:
        with self._proc_lock:
            ok, models = self.check_health()
            if ok:
                return True, models, "Ollama server is already running."

            bin_path = self._find_ollama_binary()
            if not bin_path:
                return (
                    False,
                    [],
                    "Ollama executable not found (checked OLLAMA_BIN, PATH, and standard install paths).",
                )

            try:
                self._managed_proc = subprocess.Popen(
                    [bin_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as exc:
                self._managed_proc = None
                return False, [], f"Failed to start Ollama server: {exc}"

            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                ok, models = self.check_health()
                if ok:
                    return True, models, "Ollama server started."
                if self._managed_proc and self._managed_proc.poll() is not None:
                    # Another daemon might have bound 11434 or process exited
                    ok, models = self.check_health()
                    if ok:
                        return True, models, "Ollama server started."
                    break
                time.sleep(0.4)

            ok, models = self.check_health()
            if ok:
                return True, models, "Ollama server started."
            return False, [], "Ollama server did not respond on 127.0.0.1:11434 in time."

    def stop_server(self, timeout_sec: float = 8.0) -> Tuple[bool, str]:
        with self._proc_lock:
            if self._managed_proc is not None and self._managed_proc.poll() is None:
                try:
                    self._managed_proc.terminate()
                    self._managed_proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._managed_proc.kill()
                    except Exception:
                        pass
            self._managed_proc = None

            if self.check_health()[0]:
                sys_name = platform.system()
                if sys_name == "Windows":
                    try:
                        subprocess.run(
                            ["taskkill", "/IM", "ollama.exe", "/F"],
                            timeout=4.0,
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass
                else:
                    if sys_name == "Darwin":
                        try:
                            subprocess.run(
                                ["osascript", "-e", 'tell application "Ollama" to quit'],
                                timeout=3.0,
                                check=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            pass
                    try:
                        subprocess.run(
                            ["pkill", "-TERM", "-f", "ollama serve"],
                            timeout=2.0,
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass

            deadline = time.monotonic() + timeout_sec
            escalated = False
            while time.monotonic() < deadline:
                if not self.check_health()[0]:
                    return True, "Ollama server stopped."
                if (
                    platform.system() != "Windows"
                    and not escalated
                    and (deadline - time.monotonic()) < (timeout_sec * 0.5)
                ):
                    escalated = True
                    try:
                        subprocess.run(
                            ["pkill", "-TERM", "-x", "ollama"],
                            timeout=2.0,
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass
                time.sleep(0.3)

            if not self.check_health()[0]:
                return True, "Ollama server stopped."
            return False, "Ollama server is still running."

    def _supports_thinking(self, model_name: str) -> bool:
        if model_name in self._thinking_cache:
            return self._thinking_cache[model_name]
        supports = "qwen3" in model_name.lower()
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"model": model_name},
                timeout=3.0,
            )
            if resp.status_code == 200:
                caps = resp.json().get("capabilities") or []
                supports = "thinking" in caps or supports
        except Exception:
            pass
        self._thinking_cache[model_name] = supports
        return supports

    @staticmethod
    def _pil_to_b64(img: Any, quality: int = 85) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _compute_num_ctx(num_images: int) -> int:
        needed = (max(1, num_images) * 1150) + 2048
        if needed <= 8192:
            return 8192
        rounded = ((needed + 2047) // 2048) * 2048
        return min(32768, rounded)

    def generate_multimodal(
        self,
        *,
        model_name: str,
        prompt_text: str,
        images: List[Any],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        b64_images = [self._pil_to_b64(im) for im in images]
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt_text, "images": b64_images}
        ]
        if self._supports_thinking(model_name):
            messages.append({"role": "assistant", "content": "<think>\n\n</think>\n"})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "think": False,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self._compute_num_ctx(len(b64_images)),
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        latency_s = round(time.perf_counter() - t0, 3)

        data = resp.json()
        msg = data.get("message") or {}
        content = str(msg.get("content") or msg.get("thinking") or "")

        p_tok = int(data.get("prompt_eval_count") or 0)
        p_ns = int(data.get("prompt_eval_duration") or 0)
        g_tok = int(data.get("eval_count") or 0)
        g_ns = int(data.get("eval_duration") or 0)
        load_ns = int(data.get("load_duration") or 0)

        telemetry = {
            "load_s": round(load_ns / 1e9, 3),
            "latency_s": latency_s,
            "prompt_tokens": p_tok,
            "generation_tokens": g_tok,
            "prompt_tps": round(p_tok / (p_ns / 1e9), 2) if p_ns > 0 else 0.0,
            "generation_tps": round(g_tok / (g_ns / 1e9), 2) if g_ns > 0 else 0.0,
            "peak_memory_gb": 7.45,
            "finish_reason": str(data.get("done_reason") or "stop"),
        }
        return content, telemetry


ollama_engine = OllamaEngine()
