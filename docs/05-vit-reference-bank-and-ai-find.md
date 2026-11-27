# 05 — ViT Reference Bank, Target Classes & Find Objects with AI

Inside **`/annotate`**, the collapsible **Find Objects with AI** panel provides three detection engines: **`ViT Ref`** (visual exemplar grounding), **`OWL Text`** (zero-shot text prompt), and **`YOLOE`** (open-vocab checkpoint).

---

## 1. OWLv2 Backbone Selection (`ViT Ref` & `OWL Text`)

Choose the visual/text feature extractor directly inside the panel:

| Backbone Option | Model ID | Characteristics |
|---|---|---|
| **OWLv2 Base Patch16 Ensemble** *(Default)* | `google/owlv2-base-patch16-ensemble` | ~155M params (`~593 MB`), `768-d` patch embeddings, ~3–4x faster indexing & batch scan. |
| **OWLv2 Large Patch14 Ensemble** | `google/owlv2-large-patch14-ensemble` | ~430M params (`~1.6 GB`), `1024-d` finer 14x14 patches for small/dense objects, slower per-image ViT forward pass. |

Switching checkpoints automatically re-indexes active pinned ViT references for the new embedding dimension. Once an image's feature map is extracted once, subsequent searches on that frame hit the in-memory LRU cache (`_owl_feat_cache`, up to 64 frames per model) in milliseconds.

---

## 2. Target Classes Filter (Single Class or Multiple Classes)

Right at the top of **`ViT Ref`** (and as quick-toggle pills above **`Prompt Classes`** in **`OWL Text`** / **`YOLOE`**), the **Target Classes** panel lets you scope indexing and searching cleanly:

- **Per-Class Checkboxes + Counts**: Every class displays its color dot, frame count (`Xf`), and box count (`Yb`), e.g. `hole (1f · 4b)`, `cylinder (32f · 32b)`.
- **1-Click `Only`**: Click **`Only`** next to any class (e.g. `hole`) to deselect all other classes in a single click.
- **Multi-Class Toggling**: Check/uncheck any subset (e.g. 2 out of 3 classes).
- **`All` / `None`**: Select all dataset classes or clear selection (when 0 classes are selected, indexing and detection buttons safely disable).
- **Performance Guarantee**:
  - Clicking **`Use Annotated (N)`** or **`Use Frame (+N)`** only counts and pins frames/boxes matching the selected classes.
  - On the backend (`build_visual_ref_bank`), frames containing **only excluded classes are skipped before OWLv2 ViT feature extraction runs** — dramatically cutting indexing latency when you only need 1 or 2 classes out of a multi-class batch.
  - Single-image and batch search score **only** the selected class buckets, preserving exact matched class labels and scoping `Replace Matched Classes` so existing boxes of unselected classes remain untouched.
  - Your class selection is remembered per job in `sessionStorage` (`c1_aa_vit_classes_v1:<jobId>`).

---

## 3. Building the ViT Reference Bank

You have two ways (combinable!) to supply visual grounding examples:

1. **Pin Existing Annotations**:
   - **`Use Frame (+N)`**: Pins matching boxes on the current frame.
   - **`Use Annotated (N)`**: Pins every annotated frame matching your selected Target Classes and immediately indexes their OWLv2 embeddings.
   - **Per-Box Bookmark Icon (Right Layers Panel)**: Pin/unpin individual boxes (automatically activates that box's class if disabled).
   - **Status Banner**: Shows amber `Selected · Not indexed yet` (`Index Now` button) when dirty, or green `Active & Indexed · X imgs · Y boxes` once ready. Expand **`View Selected Frames`** to jump to or unpin specific reference frames.
2. **Draw Ad-Hoc Canvas Exemplars**:
   - **`+ Pos Box` / `+ Pos Click`** & **`- Neg Box` / `- Neg Click`**: Drag boxes or click objects directly on the current image (`- Neg` suppresses false-positive regions).
   - Includes **Undo** / **Redo** buttons specifically for drawn canvas exemplars.

### Tuning Parameters
- **Similarity Threshold**: Cosine similarity threshold against reference box embeddings (`0.30`–`0.95`, default `0.65`).
- **Min Objectness**: OWLv2 foreground objectness floor (`0.00`–`0.50`, default `0.01`).
- **Use SAM 2.1 Masks**: Converts matched ViT boxes into SAM segmentation polygons automatically.

---

## 4. Safe Overlay Acceptance Workflow (Never Overwrites Blindly)

AI detections never overwrite your ground-truth annotations immediately:

1. **Single Image (`Detect on This Image`)** or **Across Batch (`Find & Add Overlay`)**:
   - Scans `Current Batch`, `All Batches`, or `Filtered Queue`.
   - With **`Skip annotated & ref frames`** checked (default), frames that already have boxes or are pinned in the ViT Bank are skipped during scanning.
   - Results appear purely as **staged green dashed overlays** on matching frames (`+N AI` green badges on the right filmstrip).
2. **Inspect & Decide Frame-by-Frame**:
   - Use **`Jump to Next Overlay`** to step straight to frames with matches.
   - Press **`Enter`** (or click **`Accept N Overlay Matches`**) to commit them to the frame.
   - Press **`Esc`** (or click **`Reject Overlay`**) to permanently dismiss staged overlays on that frame without re-triggering on navigation.
   - Use **`When Accepting Overlay`**:
     - `Append (skip duplicates)` — adds non-overlapping matches (deduped by IoU >= 0.65).
     - `Replace "<class>"` / `Replace Matched Classes` — removes existing boxes of the matched/selected target class(es) cleanly and replaces them with the accepted matches (while leaving unrelated classes intact).
