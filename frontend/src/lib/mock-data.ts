import type {
  DetectionItem,
  DraftAnalyzeConfig,
  ModelOption,
  ModelOptionId,
  RunRecord,
} from "./types";

export const MODEL_CATALOG: ModelOption[] = [
  {
    id: "mlx-4b-8bit-lora-best",
    label: "MLX Qwen3-VL 4B (8-bit) + LoRA (best · iter 180)",
    shortLabel: "MLX 4B-8bit + LoRA (best)",
    engine: "MLX-VLM",
    quantization: "8-bit affine",
    adapterName: "qwen3_vl_4b_8bit_person_lora/best",
    description:
      "Primary Apple Silicon Metal model with fine-tuned person LoRA (val_loss 0.3438, IoU@0.50 F1 0.8075).",
    recommendedFor: "Primary person detection & few-shot factory evaluation",
  },
  {
    id: "mlx-4b-8bit-lora-final",
    label: "MLX Qwen3-VL 4B (8-bit) + LoRA (final · iter 360)",
    shortLabel: "MLX 4B-8bit + LoRA (final)",
    engine: "MLX-VLM",
    quantization: "8-bit affine",
    adapterName: "qwen3_vl_4b_8bit_person_lora",
    description:
      "End-of-epoch 4 checkpoint (iter 360) for comparing overfit vs. best validation adapter.",
    recommendedFor: "Checkpoint A/B comparison",
  },
  {
    id: "mlx-4b-8bit-base",
    label: "MLX Qwen3-VL 4B (8-bit) — Base Only",
    shortLabel: "MLX 4B-8bit Base",
    engine: "MLX-VLM",
    quantization: "8-bit affine",
    adapterName: null,
    description:
      "Unadapted Qwen3-VL-4B-Instruct-8bit base weights for zero-shot / few-shot baseline benchmarking.",
    recommendedFor: "LoRA vs. Base ablation testing",
  },
  {
    id: "mlx-4b-4bit-base",
    label: "MLX Qwen3-VL 4B (4-bit) — Base Only",
    shortLabel: "MLX 4B-4bit Base",
    engine: "MLX-VLM",
    quantization: "4-bit affine",
    adapterName: null,
    description:
      "4-bit quantized Qwen3-VL-4B-Instruct base weights (mlx-community/Qwen3-VL-4B-Instruct-4bit) — lower RAM (~3.5–4.0 GB) & faster decode for 4-bit vs. 8-bit comparison.",
    recommendedFor: "4-bit vs. 8-bit speed, RAM & accuracy benchmarking",
  },
  {
    id: "ollama-8b-zero-shot",
    label: "Ollama Qwen3-VL 8B (Q4_K_M) — Zero-Shot",
    shortLabel: "Ollama 8B Zero-Shot",
    engine: "Ollama",
    quantization: "Q4_K_M GGUF",
    adapterName: null,
    description:
      "Optional 8B parameter fallback via local Ollama for broader zero-shot event & open-vocabulary checks.",
    recommendedFor: "Zero-shot multi-class or event reasoning",
  },
  {
    id: "florence2-large",
    label: "Florence-2 Large FT (0.77B · PyTorch MPS)",
    shortLabel: "Florence-2 Large",
    engine: "Florence-2 MPS",
    quantization: "FP16 MPS",
    adapterName: null,
    description:
      "Microsoft Florence-2-large-ft native Transformers checkpoint (florence-community/Florence-2-large-ft) on Apple Silicon MPS — strong open-vocab/OD bounding boxes & region masks.",
    recommendedFor: "High-recall zero-shot object detection & occluded CCTV person boxing",
  },
  {
    id: "florence2-base",
    label: "Florence-2 Base FT (0.23B · PyTorch MPS)",
    shortLabel: "Florence-2 Base",
    engine: "Florence-2 MPS",
    quantization: "FP16 MPS",
    adapterName: null,
    description:
      "Lightweight Microsoft Florence-2-base-ft checkpoint (florence-community/Florence-2-base-ft) on Apple Silicon MPS — ~2-3x faster throughput for large 4,000-image batches.",
    recommendedFor: "Fast batch auto-annotation under tight 2-hour ETA targets",
  },
];

export function getModelOption(id: ModelOptionId): ModelOption {
  return MODEL_CATALOG.find((m) => m.id === id) ?? MODEL_CATALOG[0];
}

export interface PromptPreset {
  id: string;
  label: string;
  outputMode: "bbox" | "polygon" | "both";
  prompt: string;
}

export const PROMPT_PRESETS: PromptPreset[] = [
  {
    id: "person-bbox",
    label: "Person Detection",
    outputMode: "bbox",
    prompt:
      'Locate every person in the image and output tight 2D bounding boxes in JSON format as a list of objects with keys "bbox_2d" ([x1, y1, x2, y2]) and "label" ("person"). If no person is present, return [].',
  },
  {
    id: "polygon-seg",
    label: "Polygon Segmentation",
    outputMode: "both",
    prompt:
      'Following the provided annotated reference images, detect every target object and return normalized [0..1000] JSON objects containing both "bbox_2d" ([x1, y1, x2, y2]), "polygon" ([[x1, y1], ...]), and "label". Match the tight contour boundary style of the reference masks.',
  },
  {
    id: "safety-fall",
    label: "Safety / Fall Event",
    outputMode: "bbox",
    prompt:
      "Detect if any person slips, falls, or enters a hazardous zone in this factory clip. Explain what happened, identify which person triggered the event by clothing/PPE, and return start/end timestamps plus tight post-event bounding boxes.",
  },
  {
    id: "crane-load",
    label: "Overhead Crane & Load",
    outputMode: "both",
    prompt:
      'Segment and box the overhead EOT crane hook, suspended pallet/blade load, and any personnel beneath the crane path. Return JSON objects with "label", "bbox_2d", and "polygon" normalized to 0..1000.',
  },
  {
    id: "hand-machine-reach",
    label: "Hand Detection",
    outputMode: "bbox",
    prompt:
      'Detect every human hand reaching toward, touching, or positioned near the operating zone, die, blade, or bed of the factory machine. Output tight wrist-to-fingertip 2D bounding boxes only (do NOT include the forearm, elbow, or torso; no segmentation mask) as a JSON list of objects with keys "bbox_2d" ([x1, y1, x2, y2] normalized 0..1000) and "label" ("hand"). Ignore hands kept safely away from the machine. If no hand is near the machine, return [].',
  },
];

export const DEFAULT_DRAFT_CONFIG: DraftAnalyzeConfig = {
  mediaType: "image",
  shotMode: "few-shot",
  outputMode: "both",
  modelId: "mlx-4b-8bit-lora-best",
  prompt: PROMPT_PRESETS[1].prompt,
  temperature: 0.0,
  maxTokens: 1024,
  samplingFps: 1.0,
  outputFpsMode: "original",
  byteTrackMemorySec: 3.0,
  referenceStyleNote:
    "Follow the neon-green tight contour polygons and cyan outer boxes from the reference frames. Include partially occluded workers near the gantry bay.",
};

function svgToDataUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/**
 * Generates clean 1280x720 overhead factory camera SVG illustrations aligned with
 * normalized 0..1000 detection coordinates so mock previews look cohesive offline.
 */
function makeFactorySceneSvg(options: {
  title: string;
  camId: string;
  workers: Array<{
    cx: number; // 0..1000
    cy: number; // 0..1000
    w: number;
    h: number;
    vestColor: string;
    helmetColor: string;
    fallen?: boolean;
  }>;
  showCraneBeam?: boolean;
  showReferenceOverlay?: boolean;
}): string {
  const W = 1280;
  const H = 720;
  const workerEls = options.workers
    .map((wk) => {
      const x = ((wk.cx - wk.w / 2) / 1000) * W;
      const y = ((wk.cy - wk.h / 2) / 1000) * H;
      const pw = (wk.w / 1000) * W;
      const ph = (wk.h / 1000) * H;
      const midX = x + pw / 2;
      const midY = y + ph / 2;

      if (wk.fallen) {
        return `
          <g>
            <ellipse cx="${midX}" cy="${midY + ph * 0.2}" rx="${pw * 0.45}" ry="${ph * 0.25}" fill="#050507" opacity="0.55"/>
            <rect x="${x + pw * 0.12}" y="${y + ph * 0.34}" width="${pw * 0.65}" height="${ph * 0.34}" rx="12" fill="${wk.vestColor}" />
            <circle cx="${x + pw * 0.82}" cy="${y + ph * 0.46}" r="${Math.min(pw, ph) * 0.17}" fill="${wk.helmetColor}" />
            <rect x="${x + pw * 0.04}" y="${y + ph * 0.45}" width="${pw * 0.2}" height="${ph * 0.22}" rx="6" fill="#2b3444" />
          </g>`;
      }

      return `
        <g>
          <ellipse cx="${midX}" cy="${y + ph * 0.92}" rx="${pw * 0.36}" ry="${ph * 0.08}" fill="#050507" opacity="0.55"/>
          <rect x="${x + pw * 0.28}" y="${y + ph * 0.56}" width="${pw * 0.2}" height="${ph * 0.36}" rx="6" fill="#232c3b"/>
          <rect x="${x + pw * 0.52}" y="${y + ph * 0.56}" width="${pw * 0.2}" height="${ph * 0.36}" rx="6" fill="#1e2634"/>
          <rect x="${x + pw * 0.2}" y="${y + ph * 0.26}" width="${pw * 0.6}" height="${ph * 0.35}" rx="10" fill="${wk.vestColor}"/>
          <circle cx="${midX}" cy="${y + ph * 0.14}" r="${Math.min(pw, ph) * 0.16}" fill="${wk.helmetColor}"/>
        </g>`;
    })
    .join("");

  const refOverlayEls = options.showReferenceOverlay
    ? options.workers
        .map((wk) => {
          const x = ((wk.cx - wk.w / 2) / 1000) * W;
          const y = ((wk.cy - wk.h / 2) / 1000) * H;
          const pw = (wk.w / 1000) * W;
          const ph = (wk.h / 1000) * H;
          return `
            <g>
              <rect x="${x}" y="${y}" width="${pw}" height="${ph}" fill="rgba(74, 222, 128, 0.22)" stroke="#4ADE80" stroke-width="2.5" rx="4"/>
              <rect x="${x - 3}" y="${y - 3}" width="${pw + 6}" height="${ph + 6}" fill="none" stroke="#22D3EE" stroke-width="1.5" stroke-dasharray="6 4"/>
            </g>`;
        })
        .join("")
    : "";

  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
    <defs>
      <pattern id="grid" width="80" height="80" patternUnits="userSpaceOnUse">
        <path d="M 80 0 L 0 0 0 80" fill="none" stroke="#1f232b" stroke-width="1"/>
      </pattern>
      <linearGradient id="floor" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#14171d"/>
        <stop offset="100%" stop-color="#0d1015"/>
      </linearGradient>
    </defs>
    <rect width="${W}" height="${H}" fill="url(#floor)"/>
    <rect width="${W}" height="${H}" fill="url(#grid)"/>
    <!-- Safety walkway lanes -->
    <rect x="70" y="90" width="1140" height="540" fill="none" stroke="#2a313d" stroke-width="3"/>
    <line x1="70" y1="440" x2="1210" y2="440" stroke="#eab308" stroke-width="2" stroke-dasharray="18 14" opacity="0.45"/>
    <!-- Industrial pallets / equipment bays -->
    <rect x="115" y="130" width="170" height="110" rx="6" fill="#232730" stroke="#363c49" stroke-width="2"/>
    <rect x="130" y="145" width="62" height="80" rx="4" fill="#3a2c22"/>
    <rect x="205" y="145" width="64" height="80" rx="4" fill="#3f3126"/>
    <rect x="920" y="135" width="230" height="125" rx="6" fill="#20252e" stroke="#333946" stroke-width="2"/>
    <rect x="945" y="485" width="190" height="105" rx="6" fill="#1e232c" stroke="#323845" stroke-width="2"/>
    ${
      options.showCraneBeam
        ? `<g opacity="0.85">
             <rect x="0" y="265" width="${W}" height="32" fill="#3b2616" stroke="#f2853f" stroke-width="2"/>
             <rect x="550" y="250" width="110" height="62" rx="6" fill="#27272d" stroke="#f2853f" stroke-width="2"/>
             <line x1="605" y1="312" x2="605" y2="395" stroke="#a1a1aa" stroke-width="4"/>
             <rect x="545" y="395" width="120" height="44" rx="5" fill="#334155" stroke="#64748b" stroke-width="2"/>
           </g>`
        : ""
    }
    ${workerEls}
    ${refOverlayEls}
    <!-- Camera HUD -->
    <rect x="22" y="18" width="335" height="34" rx="6" fill="rgba(10,10,12,0.78)" stroke="#2a2a32"/>
    <circle cx="42" cy="35" r="5" fill="#f2853f"/>
    <text x="56" y="40" fill="#e4e4e7" font-family="monospace" font-size="14" letter-spacing="1">${options.camId} • ${options.title}</text>
  </svg>`;
  return svgToDataUri(svg);
}

const SCENE_RUN1_TARGET = makeFactorySceneSvg({
  title: "AEROLIFT BAY 04",
  camId: "CAM0",
  showCraneBeam: true,
  workers: [
    { cx: 290, cy: 520, w: 68, h: 148, vestColor: "#f2853f", helmetColor: "#facc15" },
    { cx: 525, cy: 495, w: 72, h: 155, vestColor: "#eab308", helmetColor: "#ffffff" },
    { cx: 705, cy: 540, w: 64, h: 140, vestColor: "#f2853f", helmetColor: "#facc15" },
    { cx: 840, cy: 360, w: 58, h: 125, vestColor: "#38bdf8", helmetColor: "#ffffff" },
  ],
});

const SCENE_REF_1 = makeFactorySceneSvg({
  title: "REF SAMPLE A (ANNOTATED)",
  camId: "GT_00139",
  showCraneBeam: true,
  showReferenceOverlay: true,
  workers: [
    { cx: 340, cy: 500, w: 70, h: 150, vestColor: "#f2853f", helmetColor: "#facc15" },
    { cx: 640, cy: 520, w: 68, h: 148, vestColor: "#eab308", helmetColor: "#ffffff" },
  ],
});

const SCENE_REF_2 = makeFactorySceneSvg({
  title: "REF SAMPLE B (ANNOTATED)",
  camId: "GT_00366",
  showCraneBeam: false,
  showReferenceOverlay: true,
  workers: [
    { cx: 460, cy: 470, w: 68, h: 144, vestColor: "#eab308", helmetColor: "#facc15" },
    { cx: 790, cy: 510, w: 74, h: 154, vestColor: "#f2853f", helmetColor: "#ffffff" },
  ],
});

const SCENE_RUN2_TARGET = makeFactorySceneSvg({
  title: "BIAX LAMINATION ZONE",
  camId: "CAM1",
  showCraneBeam: false,
  workers: [
    { cx: 380, cy: 480, w: 68, h: 146, vestColor: "#eab308", helmetColor: "#facc15" },
    { cx: 615, cy: 420, w: 64, h: 138, vestColor: "#f2853f", helmetColor: "#ffffff" },
    { cx: 785, cy: 530, w: 70, h: 150, vestColor: "#eab308", helmetColor: "#facc15" },
  ],
});

const SCENE_VIDEO_KF_1 = makeFactorySceneSvg({
  title: "FRAME #3 @ 2.0s (EVENT 1)",
  camId: "BAY_FALL",
  showCraneBeam: false,
  workers: [
    { cx: 515, cy: 520, w: 150, h: 95, vestColor: "#eab308", helmetColor: "#facc15", fallen: true },
    { cx: 260, cy: 470, w: 68, h: 145, vestColor: "#38bdf8", helmetColor: "#ffffff" },
  ],
});

const SCENE_VIDEO_KF_2 = makeFactorySceneSvg({
  title: "FRAME #5 @ 4.0s (EVENT 1)",
  camId: "BAY_FALL",
  showCraneBeam: false,
  workers: [
    { cx: 540, cy: 515, w: 140, h: 98, vestColor: "#eab308", helmetColor: "#facc15", fallen: true },
    { cx: 390, cy: 495, w: 68, h: 145, vestColor: "#38bdf8", helmetColor: "#ffffff" },
  ],
});

const SCENE_VIDEO_KF_3 = makeFactorySceneSvg({
  title: "FRAME #9 @ 8.0s (EVENT 2)",
  camId: "BAY_FALL",
  showCraneBeam: false,
  workers: [
    { cx: 530, cy: 490, w: 66, h: 142, vestColor: "#eab308", helmetColor: "#facc15" },
    { cx: 745, cy: 540, w: 148, h: 94, vestColor: "#38bdf8", helmetColor: "#38bdf8", fallen: true },
  ],
});

function boxToOctagonPolygon(box: [number, number, number, number]): [number, number][] {
  const [x1, y1, x2, y2] = box;
  const w = x2 - x1;
  const h = y2 - y1;
  return [
    [Math.round(x1 + w * 0.34), Math.round(y1 + h * 0.03)],
    [Math.round(x1 + w * 0.66), Math.round(y1 + h * 0.03)],
    [Math.round(x1 + w * 0.88), Math.round(y1 + h * 0.25)],
    [Math.round(x1 + w * 0.92), Math.round(y1 + h * 0.65)],
    [Math.round(x1 + w * 0.8), Math.round(y1 + h * 0.96)],
    [Math.round(x1 + w * 0.2), Math.round(y1 + h * 0.96)],
    [Math.round(x1 + w * 0.08), Math.round(y1 + h * 0.65)],
    [Math.round(x1 + w * 0.12), Math.round(y1 + h * 0.25)],
  ];
}

const RUN1_DETECTIONS: DetectionItem[] = [
  {
    id: "det-1",
    label: "person",
    confidence: 0.96,
    bbox_2d: [256, 446, 324, 594],
    polygon: boxToOctagonPolygon([256, 446, 324, 594]),
  },
  {
    id: "det-2",
    label: "person",
    confidence: 0.94,
    bbox_2d: [489, 417, 561, 573],
    polygon: boxToOctagonPolygon([489, 417, 561, 573]),
  },
  {
    id: "det-3",
    label: "person",
    confidence: 0.93,
    bbox_2d: [673, 470, 737, 610],
    polygon: boxToOctagonPolygon([673, 470, 737, 610]),
  },
  {
    id: "det-4",
    label: "person",
    confidence: 0.89,
    bbox_2d: [811, 297, 869, 423],
    polygon: boxToOctagonPolygon([811, 297, 869, 423]),
  },
];

const RUN2_DETECTIONS: DetectionItem[] = [
  {
    id: "det-r2-1",
    label: "person",
    confidence: 0.97,
    bbox_2d: [346, 407, 414, 553],
  },
  {
    id: "det-r2-2",
    label: "person",
    confidence: 0.95,
    bbox_2d: [583, 351, 647, 489],
  },
  {
    id: "det-r2-3",
    label: "person",
    confidence: 0.92,
    bbox_2d: [750, 455, 820, 605],
  },
];

export const INITIAL_MOCK_RUNS: RunRecord[] = [
  {
    id: "run-0104",
    createdAtIso: "2026-11-21T20:42:18.000Z",
    createdAtDisplay: "Nov 21, 20:42",
    status: "preview-mock",
    isMockPreview: true,
    target: {
      filename: "areolift_bay04_cam0_00412.jpg",
      mediaType: "image",
      width: 1280,
      height: 720,
      sizeBytes: 418920,
      previewUrl: SCENE_RUN1_TARGET,
    },
    shotMode: "few-shot",
    outputMode: "both",
    modelId: "mlx-4b-8bit-lora-best",
    modelLabel: "MLX Qwen3-VL 4B (8-bit) + LoRA (best · iter 180)",
    adapterLabel: "qwen3_vl_4b_8bit_person_lora/best",
    prompt: PROMPT_PRESETS[1].prompt,
    temperature: 0.0,
    maxTokens: 1024,
    references: [
      {
        id: "ref-1",
        kind: "image",
        filename: "areolift_1__cam0_00139_mapped.jpg",
        imageUrl: SCENE_REF_1,
        note: "Tight polygon contours + outer cyan bbox around PPE workers.",
      },
      {
        id: "ref-2",
        kind: "image",
        filename: "areolift__cam0_00366_mapped.jpg",
        imageUrl: SCENE_REF_2,
        note: "Include partially occluded workers near pallets.",
      },
    ],
    referenceStyleNote: DEFAULT_DRAFT_CONFIG.referenceStyleNote,
    summaryText:
      "Detected 4 workers across the Aerolift bay floor with tight 2D bounding boxes and 8-point contour polygons conditioned on 2 annotated reference samples.",
    latencySec: 8.34,
    promptTokens: 3290,
    generatedTokens: 214,
    peakMemoryGb: 6.58,
    detections: RUN1_DETECTIONS,
    stageLogs: [
      {
        stage: "Few-Shot Context Prep",
        target: "2 annotated refs + 1 target image",
        wall_time_sec: 0.14,
        prompt_tokens: 3290,
        generated_tokens: 0,
        tokens_per_sec: 0,
        summary: "Formatted 3 vision inputs within 16,384 token budget",
      },
      {
        stage: "MLX-VLM Grounding + Polygon",
        target: "areolift_bay04_cam0_00412.jpg",
        wall_time_sec: 8.2,
        prompt_tokens: 3290,
        generated_tokens: 214,
        tokens_per_sec: 26.1,
        summary: "4 person instances parsed (100% valid JSON)",
      },
    ],
    rawJson: JSON.stringify(
      RUN1_DETECTIONS.map(({ label, confidence, bbox_2d, polygon }) => ({
        label,
        confidence,
        bbox_2d,
        polygon,
      })),
      null,
      2
    ),
  },
  {
    id: "run-0103",
    createdAtIso: "2026-11-21T20:15:04.000Z",
    createdAtDisplay: "Nov 21, 20:15",
    status: "preview-mock",
    isMockPreview: true,
    target: {
      filename: "biax_lamination_cam1_00324.jpg",
      mediaType: "image",
      width: 1280,
      height: 720,
      sizeBytes: 392110,
      previewUrl: SCENE_RUN2_TARGET,
    },
    shotMode: "zero-shot",
    outputMode: "bbox",
    modelId: "mlx-4b-8bit-lora-best",
    modelLabel: "MLX Qwen3-VL 4B (8-bit) + LoRA (best · iter 180)",
    adapterLabel: "qwen3_vl_4b_8bit_person_lora/best",
    prompt: PROMPT_PRESETS[0].prompt,
    temperature: 0.0,
    maxTokens: 1024,
    references: [],
    summaryText:
      "Zero-shot LoRA run located 3 workers in the biaxial lamination zone with zero false positives.",
    latencySec: 6.42,
    promptTokens: 1120,
    generatedTokens: 96,
    peakMemoryGb: 6.41,
    detections: RUN2_DETECTIONS,
    stageLogs: [
      {
        stage: "MLX-VLM Zero-Shot BBox",
        target: "biax_lamination_cam1_00324.jpg",
        wall_time_sec: 6.42,
        prompt_tokens: 1120,
        generated_tokens: 96,
        tokens_per_sec: 28.4,
        summary: "3 person boxes parsed (valid JSON)",
      },
    ],
    rawJson: JSON.stringify(
      RUN2_DETECTIONS.map(({ label, confidence, bbox_2d }) => ({
        label,
        confidence,
        bbox_2d,
      })),
      null,
      2
    ),
  },
  {
    id: "run-0102",
    createdAtIso: "2026-11-21T19:40:11.000Z",
    createdAtDisplay: "Nov 21, 19:40",
    status: "preview-mock",
    isMockPreview: true,
    target: {
      filename: "Task1PersonFalling1.mp4",
      mediaType: "video",
      width: 1280,
      height: 720,
      sizeBytes: 3939266,
      fps: 24.0,
      durationSec: 10.0,
      previewUrl: SCENE_VIDEO_KF_1,
    },
    shotMode: "zero-shot",
    outputMode: "bbox",
    modelId: "ollama-8b-zero-shot",
    modelLabel: "Ollama Qwen3-VL 8B (Q4_K_M) — Zero-Shot",
    adapterLabel: null,
    prompt: PROMPT_PRESETS[2].prompt,
    temperature: 0.1,
    maxTokens: 1024,
    samplingFps: 1.0,
    outputFpsMode: "original",
    byteTrackMemorySec: 3.0,
    references: [],
    summaryText:
      "Detected 2 separate fall events: Event 1 (worker in yellow vest, 2.0s -> 5.0s) and Event 2 (worker in blue uniform, 8.0s -> 9.0s). Tracked cleanly with event-isolated ByteTrack.",
    latencySec: 87.71,
    promptTokens: 18809,
    generatedTokens: 617,
    peakMemoryGb: 7.45,
    detections: [
      {
        id: "vid-d1",
        label: "Worker falls (E1 @ 2.0s)",
        confidence: 0.95,
        bbox_2d: [440, 472, 590, 568],
      },
      {
        id: "vid-d2",
        label: "Worker falls (E2 @ 8.0s)",
        confidence: 0.93,
        bbox_2d: [671, 493, 819, 587],
      },
    ],
    videoEvents: [
      {
        event_id: 1,
        event_type: "Fall",
        short_label: "Worker falls (E1)",
        trigger_subject: "Worker in yellow vest & yellow helmet",
        event_summary:
          "Worker slips and falls on factory floor, then receives assistance from nearby colleague.",
        event_start_timestamp_sec: 2.0,
        event_end_timestamp_sec: 5.0,
      },
      {
        event_id: 2,
        event_type: "Fall",
        short_label: "Worker falls (E2)",
        trigger_subject: "Worker in blue uniform & blue helmet",
        event_summary:
          "Second worker loses balance near right walkway and falls to the floor.",
        event_start_timestamp_sec: 8.0,
        event_end_timestamp_sec: 9.0,
      },
    ],
    videoKeyframes: [
      {
        sample_number: 3,
        frame_idx: 48,
        timestamp_sec: 2.0,
        imageUrl: SCENE_VIDEO_KF_1,
        detections: [
          {
            id: "kf3-d1",
            label: "Worker falls (E1)",
            confidence: 0.95,
            bbox_2d: [440, 472, 590, 568],
          },
        ],
      },
      {
        sample_number: 5,
        frame_idx: 96,
        timestamp_sec: 4.0,
        imageUrl: SCENE_VIDEO_KF_2,
        detections: [
          {
            id: "kf5-d1",
            label: "Worker falls (E1)",
            confidence: 0.94,
            bbox_2d: [470, 466, 610, 564],
          },
        ],
      },
      {
        sample_number: 9,
        frame_idx: 192,
        timestamp_sec: 8.0,
        imageUrl: SCENE_VIDEO_KF_3,
        detections: [
          {
            id: "kf9-d1",
            label: "Worker falls (E2)",
            confidence: 0.93,
            bbox_2d: [671, 493, 819, 587],
          },
        ],
      },
    ],
    stageLogs: [
      {
        stage: "Stage 1: Temporal Multi-Event Scan",
        target: "10 frames @ 1.0 FPS (640x360)",
        wall_time_sec: 49.6,
        prompt_tokens: 11237,
        generated_tokens: 287,
        tokens_per_sec: 24.8,
        summary: "Detected E1 (2.0s-5.0s) and E2 (8.0s-9.0s)",
      },
      {
        stage: "Stage 2: Post-Event Spatial Grounding",
        target: "6 keyframes (2.0s..5.0s, 8.0s..9.0s)",
        wall_time_sec: 33.9,
        prompt_tokens: 7572,
        generated_tokens: 330,
        tokens_per_sec: 36.2,
        summary: "Grounded tight boxes across 6 post-event keyframes",
      },
      {
        stage: "Stage 3: ByteTrack + FFmpeg H.264",
        target: "240 frames @ 24.0 FPS (1280x720)",
        wall_time_sec: 4.21,
        prompt_tokens: 0,
        generated_tokens: 0,
        tokens_per_sec: 0,
        summary: "2 isolated tracks rendered (139/240 annotated frames)",
      },
    ],
    rawJson: JSON.stringify(
      {
        event_detected: true,
        events: [
          {
            event_id: 1,
            short_label: "Worker falls (E1)",
            trigger_subject: "Worker in yellow vest & yellow helmet",
            start_sec: 2.0,
            end_sec: 5.0,
          },
          {
            event_id: 2,
            short_label: "Worker falls (E2)",
            trigger_subject: "Worker in blue uniform & blue helmet",
            start_sec: 8.0,
            end_sec: 9.0,
          },
        ],
      },
      null,
      2
    ),
  },
];

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const idx = Math.min(
    units.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024))
  );
  const val = bytes / Math.pow(1024, idx);
  return `${val >= 100 || idx === 0 ? val.toFixed(0) : val.toFixed(1)} ${units[idx]}`;
}

export function buildYoloSegTxt(detections: DetectionItem[]): string {
  if (!detections.length) return "";
  const labelMap = new Map<string, number>();
  return detections
    .map((det) => {
      const key = det.label.trim().toLowerCase() || "object";
      if (!labelMap.has(key)) {
        labelMap.set(key, labelMap.size);
      }
      const clsId = labelMap.get(key) ?? 0;
      const pts: [number, number][] =
        det.polygon && det.polygon.length >= 3
          ? det.polygon
          : [
              [det.bbox_2d[0], det.bbox_2d[1]],
              [det.bbox_2d[2], det.bbox_2d[1]],
              [det.bbox_2d[2], det.bbox_2d[3]],
              [det.bbox_2d[0], det.bbox_2d[3]],
            ];
      const coords = pts
        .map(
          ([x, y]) =>
            `${Math.min(Math.max(x / 1000, 0), 1).toFixed(6)} ${Math.min(
              Math.max(y / 1000, 0),
              1
            ).toFixed(6)}`
        )
        .join(" ");
      return `${clsId} ${coords}`;
    })
    .join("\n");
}
