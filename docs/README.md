# Control One Documentation Hub

Welcome to the documentation for **Control One — Auto-Annotator & VLM Evaluation Workbench**.

---

## Feature & Deployment Guides

| Guide | Topics Covered |
|---|---|
| **[01 — Setup & NVIDIA Deployment](./01-setup-and-nvidia-deployment.md)** | System prerequisites, macOS Apple Silicon & NVIDIA CUDA installation, environment variables, starting frontend/backend. |
| **[02 — VLM Evaluation Workbench](./02-evaluation-workbench.md)** | `/analyze`, `/logs`, `/results` — zero-shot/few-shot image & video analysis, two-stage temporal detection, ByteTrack, re-encoding. |
| **[03 — Auto-Annotate & Verify Pipeline](./03-auto-annotate-pipeline.md)** | `/auto-annotate`, `/verify` — Roboflow dataset sync, fetch scopes, batch detectors + SAM 2.1 refiners, QA review, cloud push & ZIP export. |
| **[04 — Manual Annotator & Smart Select](./04-manual-annotator-and-smart-select.md)** | `/annotate` — dataset fetch inside annotator, BBox/Polygon drawing, keyboard shortcuts, hover-preview SAM 2.1 Smart Select, session caching. |
| **[05 — ViT Reference Bank & Target Class AI Find](./05-vit-reference-bank-and-ai-find.md)** | **Find Objects with AI** (`ViT Ref`, `OWL Text`, `YOLOE`), OWLv2 Base vs Large backbones, **Target Classes** 1-or-multi-class indexing & search, staged overlay preview/accept/reject workflow. |
| **[06 — Model Weights & Cloud Transfer](./06-model-weights-and-transfer.md)** | Exact inventory of the **4 weights pushed to GitHub (`< 100 MB`)** vs **15 unpushed weights**, NVIDIA compatibility matrix, and Google Cloud Storage (`gcloud storage`) zip/transfer commands. |

---

## Architecture & Migration References

- **[`../context.md`](../context.md)** — Repository layout, API architecture, normalized `0..1000` coordinate contract, SQLite tables, and frontend `sessionStorage` cache behavior.
- **[`../MLX_dependencies.md`](../MLX_dependencies.md)** — Complete file tree and line-level audit of raw `mlx.core` tensor usage (`backend/engines/mlx_lora_engine.py`) vs cross-platform PyTorch CUDA code for NVIDIA deployment.
- **[`../MISSING_WEIGHTS.md`](../MISSING_WEIGHTS.md)** — Exact log of missing non-MLX weights (`7` total), target directories, `missing_nvidia_weights_bundle.zip` layout, and restore/verification commands.
