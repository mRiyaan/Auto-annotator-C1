export type MediaType = "image" | "video";

export type ShotMode = "zero-shot" | "few-shot";

export type OutputMode = "bbox" | "polygon" | "both";

export type ModelOptionId =
  | "mlx-4b-8bit-lora-best"
  | "mlx-4b-8bit-lora-final"
  | "mlx-4b-8bit-base"
  | "mlx-4b-4bit-base"
  | "ollama-8b-zero-shot"
  | "florence2-large"
  | "florence2-base";

export interface ModelOption {
  id: ModelOptionId;
  label: string;
  shortLabel: string;
  engine: "MLX-VLM" | "Ollama" | "Florence-2 MPS";
  quantization: string;
  adapterName: string | null;
  description: string;
  recommendedFor: string;
}

export interface DetectionItem {
  id: string;
  label: string;
  confidence: number;
  /** Normalized 0..1000 [xmin, ymin, xmax, ymax] */
  bbox_2d: [number, number, number, number];
  /** Normalized 0..1000 polygon vertices [[x, y], ...] */
  polygon?: [number, number][];
}

export interface VideoEventItem {
  event_id: number;
  event_type: string;
  short_label: string;
  trigger_subject: string;
  event_summary: string;
  event_start_timestamp_sec: number;
  event_end_timestamp_sec: number;
}

export interface VideoKeyframeItem {
  sample_number: number;
  frame_idx: number;
  timestamp_sec: number;
  imageUrl: string;
  detections: DetectionItem[];
}

export interface StageLogItem {
  stage: string;
  target: string;
  wall_time_sec: number;
  prompt_tokens: number;
  generated_tokens: number;
  tokens_per_sec: number;
  summary: string;
}

export interface ReferencePreview {
  id: string;
  kind: "image" | "video_frame";
  filename: string;
  imageUrl: string;
  note?: string;
  timestampSec?: number;
}

export interface TargetMediaMeta {
  filename: string;
  mediaType: MediaType;
  width: number;
  height: number;
  sizeBytes: number;
  fps?: number;
  durationSec?: number;
  previewUrl: string;
}

export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "preview-mock"
  | "failed";

export interface RunRecord {
  id: string;
  createdAtIso: string;
  createdAtDisplay: string;
  status: RunStatus;
  isMockPreview: boolean;
  target: TargetMediaMeta;
  shotMode: ShotMode;
  outputMode: OutputMode;
  modelId: ModelOptionId;
  modelLabel: string;
  adapterLabel: string | null;
  prompt: string;
  temperature: number;
  maxTokens: number;
  samplingFps?: number;
  outputFpsMode?: "original" | "sampled";
  byteTrackMemorySec?: number;
  references: ReferencePreview[];
  referenceStyleNote?: string;
  summaryText: string;
  latencySec: number;
  promptTokens: number;
  generatedTokens: number;
  peakMemoryGb: number;
  detections: DetectionItem[];
  videoEvents?: VideoEventItem[];
  videoKeyframes?: VideoKeyframeItem[];
  annotatedVideoUrl?: string;
  errorMessage?: string;
  stageLogs: StageLogItem[];
  rawJson: string;
}

export interface BackendHealthInfo {
  status: "ok";
  bootId?: string;
  device: string;
  loadedModelId: string | null;
  queueDepth: number;
  ollamaReachable: boolean;
  ollamaModels: string[];
  adapterBestExists: boolean;
}

export interface BackendModelStatus {
  id: string;
  label: string;
  shortLabel: string;
  engine: string;
  quantization: string;
  adapterLabel?: string | null;
  available: boolean;
  loaded: boolean;
  note: string;
}

export interface OllamaControlResponse {
  ok: boolean;
  ollamaReachable: boolean;
  ollamaModels: string[];
  message: string;
}

export interface ActiveEngineDisplay {
  id: string;
  shortLabel: string;
  engine: string;
  quantization: string;
}

export interface RunProgressEvent {
  runId: string;
  status: RunStatus;
  progressPct: number;
  stage: string;
  subtext: string;
  record?: RunRecord;
}

export interface DraftAnalyzeConfig {
  mediaType: MediaType;
  shotMode: ShotMode;
  outputMode: OutputMode;
  modelId: ModelOptionId;
  prompt: string;
  temperature: number;
  maxTokens: number;
  samplingFps: number;
  outputFpsMode: "original" | "sampled";
  byteTrackMemorySec: number;
  referenceStyleNote: string;
}
