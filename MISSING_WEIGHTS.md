# Missing Non-MLX Model Weights & Exact Target Directories (NVIDIA PC Deployment)

This log lists **every model weight NOT pushed to GitHub (excluding Apple Silicon MLX weights)** that is needed for full functionality on an **NVIDIA CUDA workstation** (Linux or Windows), along with the **exact target directories**, **unzip/restore instructions** for `missing_nvidia_weights_bundle.zip`, and an **offline verification check**.

> Looking for Apple Silicon MLX LoRA adapters (`backend/adapters/`) and `mlx-community/*` checkpoints? Those use MLX affine quantization and cannot run natively on NVIDIA CUDA — see [`MLX_dependencies.md`](./MLX_dependencies.md) for the exact MLX file inventory and CUDA conversion path.

---

## 1. Complete Count of Missing Non-MLX Weights — **7 Total**

### A. Packaged in `missing_nvidia_weights_bundle.zip` — **5 Model Weights (~4.34 GiB raw)**

All 5 items below are packaged directly into `<repo-root>/missing_nvidia_weights_bundle.zip` (dereferenced real files, zero symlinks, ZIP64 format) so you can upload one single ZIP to Google Cloud and extract it on the target NVIDIA PC:

| # | Model / Weight | Required For | Size | Exact Target Destination on NVIDIA PC |
|---|---|---|---|---|
| 1 | `mobileclip2_b.ts` | Text/visual CLIP encoder required by `yoloe-26s-seg.pt` (`YOLOE` mode & `/auto-annotate`) | `253,794,476` B (~242.0 MiB) | **`<repo-root>/backend/auto_annotate/weights/mobileclip2_b.ts`** |
| 2 | `google/owlv2-base-patch16-ensemble` *(rev `cfd3195ba4ea9592eec887ded089f4c08eff231d`)* | Fast ViT Reference Bank (`768-d`), `OWL Text` detection, `/auto-annotate` (`model.safetensors`: `619,918,824` B + 7 config/tokenizer files) | ~593 MB | **Linux/macOS**: `~/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/`<br>**Windows**: `%USERPROFILE%\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\` |
| 3 | `google/owlv2-large-patch14-ensemble` *(rev `95e26936e865f87db1742128404b3c035d47d89d`)* | High-Precision ViT Reference Bank (`1024-d`), `OWL Text` (`model.safetensors`: `1,750,520,144` B + 7 config/tokenizer files) | ~1.63 GiB | **Linux/macOS**: `~/.cache/huggingface/hub/models--google--owlv2-large-patch14-ensemble/`<br>**Windows**: `%USERPROFILE%\.cache\huggingface\hub\models--google--owlv2-large-patch14-ensemble\` |
| 4 | `florence-community/Florence-2-base-ft` *(rev `0b03b6f15a4a211370fb204aee4e7dd48887ea37`)* | Fast Florence-2 CUDA open-vocab & region polygon model (`model.safetensors`: `463,178,864` B + 10 config/tokenizer files) | ~447 MB | **Linux/macOS**: `~/.cache/huggingface/hub/models--florence-community--Florence-2-base-ft/`<br>**Windows**: `%USERPROFILE%\.cache\huggingface\hub\models--florence-community--Florence-2-base-ft\` |
| 5 | `florence-community/Florence-2-large-ft` *(rev `26b734a54fdfbf9c398351eedfabb7f27fc470b7`)* | High-accuracy Florence-2 CUDA open-vocab & region polygon model (`model.safetensors`: `1,540,958,064` B + 10 config/tokenizer files) | ~1.44 GiB | **Linux/macOS**: `~/.cache/huggingface/hub/models--florence-community--Florence-2-large-ft/`<br>**Windows**: `%USERPROFILE%\.cache\huggingface\hub\models--florence-community--Florence-2-large-ft\` |

> *(Note: If you have custom `HF_HOME` or `HUGGINGFACE_HUB_CACHE` set on the target PC, place the `models--*` directories under `$HF_HOME/hub/` or `$HUGGINGFACE_HUB_CACHE/` instead of `~/.cache/huggingface/hub/`. Note also that if online internet access exists on the NVIDIA PC, items #2–#5 auto-download from Hugging Face Hub on first run if omitted.)*

### B. Native CLI Pull on Target PC (Not in ZIP) — **2 Ollama Models**

Ollama manages its own deduplicated content-addressed blob store (`~/.ollama/models` or `%USERPROFILE%\.ollama\models`). Pull these directly after installing Ollama on the NVIDIA PC:

| # | Ollama Tag | Required For | Approx Size | Command on Target NVIDIA PC |
|---|---|---|---|---|
| 6 | `qwen3-vl:8b` | `ollama-8b-zero-shot` in `/analyze` & `/auto-annotate` | ~6.2 GB | `ollama pull qwen3-vl:8b` |
| 7 | `qwen3-vl:4b` *(optional)* | Lightweight local Ollama 4B runs | ~3.3 GB | `ollama pull qwen3-vl:4b` |

---

## 2. Already Bundled in GitHub (No Extra Download Needed — **4 Files `< 100 MB`**)

Cloning `https://github.com/mRiyaan/Auto-annotator-C1.git` already includes these **4** PyTorch weights under `<repo-root>/backend/auto_annotate/weights/`:

1. `yoloe-26s-seg.pt` (`31,072,171` B · 29.6 MiB)
2. `mobile_sam.pt` (`40,728,226` B · 38.8 MiB)
3. `sam2.1_t.pt` (`78,105,722` B · 74.5 MiB)
4. `sam2.1_s.pt` (`92,319,866` B · 88.0 MiB)

---

## 3. Exact Folder Structure Inside `missing_nvidia_weights_bundle.zip`

All files inside the archive are real files (symlinks resolved to actual bytes) so extraction works identically on Linux, macOS, and Windows:

```text
missing_nvidia_weights_bundle.zip
├── RESTORE_INSTRUCTIONS.md
├── repo_weights/
│   └── backend/
│       └── auto_annotate/
│           └── weights/
│               └── mobileclip2_b.ts                          # -> <repo-root>/backend/auto_annotate/weights/mobileclip2_b.ts
└── huggingface_hub_cache/                                    # -> ~/.cache/huggingface/hub/  (or %USERPROFILE%\.cache\huggingface\hub\)
    ├── models--google--owlv2-base-patch16-ensemble/
    │   ├── refs/
    │   │   └── main                                          # cfd3195ba4ea9592eec887ded089f4c08eff231d
    │   └── snapshots/
    │       └── cfd3195ba4ea9592eec887ded089f4c08eff231d/
    │           ├── added_tokens.json
    │           ├── config.json
    │           ├── merges.txt
    │           ├── model.safetensors
    │           ├── preprocessor_config.json
    │           ├── special_tokens_map.json
    │           ├── tokenizer_config.json
    │           └── vocab.json
    ├── models--google--owlv2-large-patch14-ensemble/
    │   ├── refs/
    │   │   └── main                                          # 95e26936e865f87db1742128404b3c035d47d89d
    │   └── snapshots/
    │       └── 95e26936e865f87db1742128404b3c035d47d89d/
    │           ├── added_tokens.json
    │           ├── config.json
    │           ├── merges.txt
    │           ├── model.safetensors
    │           ├── preprocessor_config.json
    │           ├── special_tokens_map.json
    │           ├── tokenizer_config.json
    │           └── vocab.json
    ├── models--florence-community--Florence-2-base-ft/
    │   ├── refs/
    │   │   └── main                                          # 0b03b6f15a4a211370fb204aee4e7dd48887ea37
    │   └── snapshots/
    │       └── 0b03b6f15a4a211370fb204aee4e7dd48887ea37/
    │           ├── added_tokens.json
    │           ├── config.json
    │           ├── generation_config.json
    │           ├── merges.txt
    │           ├── model.safetensors
    │           ├── preprocessor_config.json
    │           ├── processor_config.json
    │           ├── special_tokens_map.json
    │           ├── tokenizer.json
    │           ├── tokenizer_config.json
    │           └── vocab.json
    └── models--florence-community--Florence-2-large-ft/
        ├── refs/
        │   └── main                                          # 26b734a54fdfbf9c398351eedfabb7f27fc470b7
        └── snapshots/
            └── 26b734a54fdfbf9c398351eedfabb7f27fc470b7/
                ├── added_tokens.json
                ├── config.json
                ├── generation_config.json
                ├── merges.txt
                ├── model.safetensors
                ├── preprocessor_config.json
                ├── processor_config.json
                ├── special_tokens_map.json
                ├── tokenizer.json
                ├── tokenizer_config.json
                └── vocab.json
```

---

## 4. Step-by-Step Transfer & Placement Commands

### Step A — Upload `missing_nvidia_weights_bundle.zip` from Source Machine via Google Cloud

```bash
gcloud storage cp missing_nvidia_weights_bundle.zip gs://<YOUR-BUCKET-NAME>/control-one/missing_nvidia_weights_bundle.zip
```

### Step B — Extract & Place on Target NVIDIA PC

#### Option 1 — Linux NVIDIA PC (Bash)

```bash
# Inside your cloned <repo-root> (Auto-annotator-C1/):
gcloud storage cp gs://<YOUR-BUCKET-NAME>/control-one/missing_nvidia_weights_bundle.zip .

mkdir -p /tmp/c1_weights_unpack ~/.cache/huggingface/hub backend/auto_annotate/weights
unzip -q missing_nvidia_weights_bundle.zip -d /tmp/c1_weights_unpack

# 1) Place mobileclip2_b.ts into backend/auto_annotate/weights/
cp /tmp/c1_weights_unpack/repo_weights/backend/auto_annotate/weights/mobileclip2_b.ts backend/auto_annotate/weights/

# 2) Place all 4 Hugging Face models into ~/.cache/huggingface/hub/
cp -R /tmp/c1_weights_unpack/huggingface_hub_cache/models--* ~/.cache/huggingface/hub/

# 3) Clean up temp extraction
rm -rf /tmp/c1_weights_unpack

# 4) Pull Ollama models natively on NVIDIA PC (optional, for Qwen3-VL via Ollama)
ollama pull qwen3-vl:8b
```

#### Option 2 — Windows NVIDIA PC (PowerShell)

```powershell
# Inside your cloned <repo-root> (Auto-annotator-C1\):
$unpack = "$env:TEMP\c1_weights_unpack"
$hfHub = "$env:USERPROFILE\.cache\huggingface\hub"
New-Item -ItemType Directory -Force -Path $unpack, $hfHub, "backend\auto_annotate\weights" | Out-Null

Expand-Archive -Path "missing_nvidia_weights_bundle.zip" -DestinationPath $unpack -Force

# 1) Place mobileclip2_b.ts
Copy-Item "$unpack\repo_weights\backend\auto_annotate\weights\mobileclip2_b.ts" "backend\auto_annotate\weights\" -Force

# 2) Place all 4 Hugging Face models
Copy-Item "$unpack\huggingface_hub_cache\models--*" $hfHub -Recurse -Force

# 3) Clean up temp extraction
Remove-Item $unpack -Recurse -Force

# 4) Pull Ollama models natively
ollama pull qwen3-vl:8b
```

#### Option 3 — Ready-to-Use Prompt for Claude Code on the Target NVIDIA PC

```text
In this cloned Auto-annotator-C1 repository, I have placed `missing_nvidia_weights_bundle.zip` in the repo root. Read `MISSING_WEIGHTS.md`, extract `missing_nvidia_weights_bundle.zip` using Python/unzip without breaking anything, move `repo_weights/backend/auto_annotate/weights/mobileclip2_b.ts` into `backend/auto_annotate/weights/`, copy `huggingface_hub_cache/models--*` into the active Hugging Face cache directory (`~/.cache/huggingface/hub` or `$HF_HOME/hub`), remove the temporary unpack folder, and run the offline verification command in Section 5 of `MISSING_WEIGHTS.md`.
```

---

## 5. Offline Verification Script (Run on NVIDIA PC After Unzipping)

Run inside `<repo-root>/backend` with the virtualenv Python active:

```bash
python -c '
from pathlib import Path
from huggingface_hub import try_to_load_from_cache

repo_weights = [
    "auto_annotate/weights/yoloe-26s-seg.pt",
    "auto_annotate/weights/mobile_sam.pt",
    "auto_annotate/weights/sam2.1_t.pt",
    "auto_annotate/weights/sam2.1_s.pt",
    "auto_annotate/weights/mobileclip2_b.ts",
]
for rel in repo_weights:
    p = Path(rel)
    assert p.is_file() and p.stat().st_size > 1_000_000, f"Missing or corrupt repo weight: {p}"
    print(f"[OK] {rel} ({p.stat().st_size:,} bytes)")

hf_repos = [
    "google/owlv2-base-patch16-ensemble",
    "google/owlv2-large-patch14-ensemble",
    "florence-community/Florence-2-base-ft",
    "florence-community/Florence-2-large-ft",
]
for repo_id in hf_repos:
    cfg = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    wts = try_to_load_from_cache(repo_id=repo_id, filename="model.safetensors")
    assert cfg and Path(cfg).is_file(), f"Missing config.json in HF cache for {repo_id}"
    assert wts and Path(wts).is_file(), f"Missing model.safetensors in HF cache for {repo_id}"
    print(f"[OK] HF cache {repo_id} -> {Path(wts).stat().st_size:,} bytes")

print("ALL_WEIGHTS_VERIFIED_OK")
'
```

---

## 6. Excluded MLX-Only Weights Reference

See [`MLX_dependencies.md`](./MLX_dependencies.md) for the **8 Apple Silicon MLX-only items** intentionally omitted from `missing_nvidia_weights_bundle.zip`:
- `backend/adapters/qwen3_vl_4b_8bit_person_lora/` (6 MLX affine `.safetensors` files)
- `mlx-community/Qwen3-VL-4B-Instruct-4bit` (~2.9 GB)
- `mlx-community/Qwen3-VL-4B-Instruct-8bit` (~4.8 GB)
