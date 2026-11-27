# 04 — Manual Annotator & Smart Select (`/annotate`)

The **Manual Annotator** (`/annotate`) is a keyboard-driven canvas editor with built-in Roboflow ingestion, pure **SAM 2.1 Smart Select**, layer controls, and the **Find Objects with AI** visual grounding suite.

---

## 1. Loading or Pulling a Dataset Inside `/annotate`

- **From an existing job**: Pick any Auto-Annotate or Manual job in the top job selector, or arrive directly from `/verify` ("Edit Manually").
- **Pull directly without Auto-Annotate**: Click **Pull from Roboflow** in `/annotate` to open the same **`RoboflowSourceCard`** modal. It calls `POST /api/v1/auto-annotate/jobs/fetch-only` — downloading the requested batch cleanly into local storage as a completed manual job ready for immediate annotation without running batch VLM inference.

---

## 2. Canvas Tools & Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `D`, `V`, `H` | Switch to **Drag / Edit** tool (select, move, resize handles, drag polygon vertices) |
| `B` | Switch to **Bounding Box** draw tool |
| `P` | Switch to **Polygon** click-to-plot tool (`Enter` commits >= 3 vertices; `Backspace`/`Delete` removes last vertex) |
| `S` | Switch to **Smart Select** (interactive SAM 2.1 segmentation) |
| `Space` (hold) + Drag | Temporary **Pan** across zoomed image |
| `Mouse Wheel` / `+` `-` | Zoom in / out smoothly around cursor |
| `U` / `Ctrl+Z` / `Cmd+Z` | **Undo** annotation change (`Ctrl+Y` / `Cmd+Shift+Z` / `R` = **Redo**) |
| `O` | Toggle overlay visibility |
| `L` | Toggle class label badges on boxes |
| `Backspace` / `Delete` | Delete selected annotation (or undo last point in Polygon / Multi-Point Smart Select) |
| `Enter` | Commit polygon, commit Smart Select preview, OR **Accept staged ViT AI overlay** |
| `Esc` | Cancel current draw, clear Smart Select points, OR **Reject staged ViT AI overlay on this frame** |
| `N` | **Mark Null** (clears frame boxes and sets frame status to `accepted` Null) |
| `Ctrl+S` / `Cmd+S` | **Save & Accept** current frame |
| `ArrowLeft` / `[` and `ArrowRight` / `]` | Previous / Next frame in active queue filter |

---

## 3. Pure SAM 2.1 Smart Select (`S`)

Smart Select uses cached SAM 2.1 / MobileSAM image embeddings (`POST /jobs/{jobId}/items/{itemId}/smart-select/prepare`) so every subsequent hover/click responds interactively:

1. **Instant Hover Preview (Single-Point Mode)**: Moving the cursor across the image shows a live translucent polygon mask under your pointer; **left-click** commits the mask immediately!
2. **Multi-Point Refinement Mode**: Toggle `Multi-Point` to add **Left-Click (Foreground `+`)** and **Right-Click (Background `-`)** points or drag a box prompt around tightly packed objects, then press `Enter` or click **Commit**.
3. **Output & Quality Controls**:
   - **Output Shape**: `polygon`, `bbox`, or `both`.
   - **SAM Refiner**: `sam2.1-s` (higher precision), `sam2.1-t` (fast), or `mobile-sam`.
   - **Polygon Complexity Slider**: Adjusts contour simplification granularity from coarse vertices to dense organic boundaries.

---

## 4. Session Caching Across Pages

- Navigating between `/auto-annotate`, `/verify`, and `/annotate` preserves your Roboflow API key, project slug, fetch filters, selected OWLv2 model checkpoint, pinned ViT reference frames, and per-job **Target Classes** in `sessionStorage` (`c1_aa_*`).
- If the FastAPI backend restarts (detected automatically via `SERVER_BOOT_ID` mismatch) or is turned off, ephemeral session caches reset cleanly to prevent stale `queryId` errors.
