# 06 — Model Weights Inventory & Google Cloud Transfer

GitHub enforces a strict **100 MiB per-file limit**. To keep the repository clean while shipping ready-to-run Ultralytics segmentation checkpoints directly in Git, all checkpoints **under 100 MB** are pushed in Git, while **15 larger/external or MLX-only weights** are excluded via `.gitignore`.

---

## 1. Weights Pushed to GitHub (`< 100 MB`) — **4 Files (~231.0 MiB)**

Committed on branch `feat/model-weights-under-100mb` and merged into `main`:

| # | Workspace Path | Size (Bytes) | Human Size | Runtime Use | NVIDIA CUDA Compatible? |
|---|---|---:|---:|---|---|
| 1 | `backend/auto_annotate/weights/yoloe-26s-seg.pt` | `31,072,171` | 29.6 MiB | YOLOE-26s-seg open-vocab detector/seg *(requires `mobileclip2_b.ts` for text prompts)* | Yes |
| 2 | `backend/auto_annotate/weights/mobile_sam.pt` | `40,728,226` | 38.8 MiB | Lightweight Smart Select & polygon refiner | Yes |
| 3 | `backend/auto_annotate/weights/sam2.1_t.pt` | `78,105,722` | 74.5 MiB | Fast SAM 2.1 Tiny polygon refiner | Yes |
| 4 | `backend/auto_annotate/weights/sam2.1_s.pt` | `92,319,866` | 88.0 MiB | High-precision SAM 2.1 Small polygon refiner | Yes |

---

## 2. Complete Count & List of Unpushed Weights — **15 Items**

### Summary Count
- **Total NOT pushed to GitHub**: **15** items (`7` NVIDIA-compatible + `8` Apple Silicon MLX-only)
  - Workspace weights `>= 100 MB` (NVIDIA-compatible): **1** file (`242.0 MiB`)
  - Workspace MLX LoRA adapter `.safetensors` (Mac MLX-only): **6** files (`756.6 MiB`)
  - Hugging Face Hub cached repos (`~/.cache/huggingface/hub/`): **6** repos (`4` NVIDIA-compatible = **4.04 GB**, `2` MLX-only = **7.7 GB**)
  - Local Ollama models (`ollama list`): **2** models (`9.5 GB` total; NVIDIA-compatible via Ollama)

### Detailed Inventory

| # | Weight Name / Identifier | Location on Current Mac | Exact Size | Git Status | Transfer to NVIDIA PC? |
|---|---|---|---:|---|---|
| 1 | `mobileclip2_b.ts` | `backend/auto_annotate/weights/mobileclip2_b.ts` | `253,794,476` B (~242.0 MiB) | Ignored (`>= 100 MB`) | **YES** (required by `yoloe-26s-seg.pt`) |
| 2 | `adapters.safetensors` *(best · iter 180)* | `backend/adapters/qwen3_vl_4b_8bit_person_lora/best/` | `132,183,313` B (~126.1 MiB) | Ignored (`adapters/` & `*.safetensors`) | **NO** (Apple MLX 8-bit affine only) |
| 3 | `adapters.safetensors` *(final · iter 360)* | `backend/adapters/qwen3_vl_4b_8bit_person_lora/` | `132,183,313` B (~126.1 MiB) | Ignored | **NO** (Apple MLX 8-bit affine only) |
| 4 | `0000090_adapters.safetensors` | `backend/adapters/qwen3_vl_4b_8bit_person_lora/` | `132,183,313` B (~126.1 MiB) | Ignored | **NO** (Apple MLX checkpoint) |
| 5 | `0000180_adapters.safetensors` | `backend/adapters/qwen3_vl_4b_8bit_person_lora/` | `132,183,313` B (~126.1 MiB) | Ignored | **NO** (Apple MLX checkpoint) |
| 6 | `0000270_adapters.safetensors` | `backend/adapters/qwen3_vl_4b_8bit_person_lora/` | `132,183,313` B (~126.1 MiB) | Ignored | **NO** (Apple MLX checkpoint) |
| 7 | `0000360_adapters.safetensors` | `backend/adapters/qwen3_vl_4b_8bit_person_lora/` | `132,183,313` B (~126.1 MiB) | Ignored | **NO** (Apple MLX checkpoint) |
| 8 | `google/owlv2-base-patch16-ensemble` | `~/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble` | ~593 MB | External HF cache | **YES** *(or auto-downloads on first run)* |
| 9 | `google/owlv2-large-patch14-ensemble` | `~/.cache/huggingface/hub/models--google--owlv2-large-patch14-ensemble` | ~1.6 GB | External HF cache | **YES** *(or auto-downloads on first run)* |
| 10 | `florence-community/Florence-2-base-ft` | `~/.cache/huggingface/hub/models--florence-community--Florence-2-base-ft` | ~447 MB | External HF cache | **YES** *(or auto-downloads on first run)* |
| 11 | `florence-community/Florence-2-large-ft` | `~/.cache/huggingface/hub/models--florence-community--Florence-2-large-ft` | ~1.4 GB | External HF cache | **YES** *(or auto-downloads on first run)* |
| 12 | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-4B-Instruct-4bit` | ~2.9 GB | External HF cache | **NO** (Apple MLX affine only) |
| 13 | `mlx-community/Qwen3-VL-4B-Instruct-8bit` | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-4B-Instruct-8bit` | ~4.8 GB | External HF cache | **NO** (Apple MLX affine only) |
| 14 | `qwen3-vl:8b` | Ollama local blob store | ~6.2 GB | External Ollama store | **YES** *(recommended: `ollama pull qwen3-vl:8b` on target)* |
| 15 | `qwen3-vl:4b` | Ollama local blob store | ~3.3 GB | External Ollama store | **YES** *(recommended: `ollama pull qwen3-vl:4b` on target)* |

> *(Note: `~/.cache/huggingface/hub/models--mosaic-laboratory--normwear` ~739 MB may also exist in local cache from prior unrelated work and is NOT used by this repository.)*

---

## 3. Compressing & Transferring Unpushed Weights via Google Cloud

Because Hugging Face Hub caches use relative symlinks from `snapshots/` to `blobs/`, always pass `-h` (dereference symlinks) or archive the entire `models--*` folder cleanly without breaking relative links.

### A. On Source Mac — Create Archive of NVIDIA-Compatible Weights

```bash
# 1) Archive ONLY the 5 NVIDIA-compatible weights needed (~4.28 GB raw):
#    - backend/auto_annotate/weights/mobileclip2_b.ts
#    - 4 Hugging Face models (OWLv2 Base/Large + Florence-2 Base/Large)
mkdir -p ~/c1_weight_bundle/hf_hub
cp backend/auto_annotate/weights/mobileclip2_b.ts ~/c1_weight_bundle/
cp -R ~/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble ~/c1_weight_bundle/hf_hub/
cp -R ~/.cache/huggingface/hub/models--google--owlv2-large-patch14-ensemble ~/c1_weight_bundle/hf_hub/
cp -R ~/.cache/huggingface/hub/models--florence-community--Florence-2-base-ft ~/c1_weight_bundle/hf_hub/
cp -R ~/.cache/huggingface/hub/models--florence-community--Florence-2-large-ft ~/c1_weight_bundle/hf_hub/

# Compress (use -C so archive unpacks cleanly on the target PC)
tar -czf ~/control_one_nvidia_weights.tar.gz -C ~/c1_weight_bundle .
rm -rf ~/c1_weight_bundle
```

*(If you also want to archive the Apple MLX LoRA adapters separately for backup on another Mac: `tar -czf ~/control_one_mlx_adapters.tar.gz -C backend adapters`.)*

### B. Upload to Google Cloud Storage (`gcloud` CLI)

```bash
gcloud storage cp ~/control_one_nvidia_weights.tar.gz gs://<YOUR-BUCKET-NAME>/control-one/control_one_nvidia_weights.tar.gz
```

*(Alternatively, `gsutil cp ...` or uploading via Google Drive / Cloud Console works identically.)*

### C. Restore on Target NVIDIA PC

```bash
# 1) Clone repo from GitHub (already includes the 4 <100 MB .pt files!)
git clone https://github.com/mRiyaan/Auto-annotator-C1.git
cd Auto-annotator-C1

# 2) Download & extract the weight bundle
gcloud storage cp gs://<YOUR-BUCKET-NAME>/control-one/control_one_nvidia_weights.tar.gz .
mkdir -p /tmp/c1_weights ~/.cache/huggingface/hub
tar -xzf control_one_nvidia_weights.tar.gz -C /tmp/c1_weights

# 3) Move mobileclip2_b.ts and HF caches into place
mv /tmp/c1_weights/mobileclip2_b.ts backend/auto_annotate/weights/
cp -R /tmp/c1_weights/hf_hub/models--* ~/.cache/huggingface/hub/
rm -rf /tmp/c1_weights control_one_nvidia_weights.tar.gz

# 4) Pull Ollama Qwen3-VL models natively on the target PC (if using Ollama)
ollama pull qwen3-vl:8b
```
