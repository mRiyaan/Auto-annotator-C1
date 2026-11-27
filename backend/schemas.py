"""Strict Pydantic v2 schemas aligned 1:1 with frontend/src/lib/types.ts."""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

MediaType = Literal["image", "video"]
ShotMode = Literal["zero-shot", "few-shot"]
OutputMode = Literal["bbox", "polygon", "both"]
ModelOptionId = Literal[
    "mlx-4b-8bit-lora-best",
    "mlx-4b-8bit-lora-final",
    "mlx-4b-8bit-base",
    "mlx-4b-4bit-base",
    "ollama-8b-zero-shot",
    "florence2-large",
    "florence2-base",
]
RunStatus = Literal["queued", "running", "completed", "failed"]


class DetectionItem(BaseModel):
    id: str
    label: str
    confidence: float = 0.95
    bbox_2d: Tuple[int, int, int, int]
    polygon: Optional[List[Tuple[int, int]]] = None


class VideoEventItem(BaseModel):
    event_id: int
    event_type: str
    short_label: str
    trigger_subject: str
    event_summary: str
    event_start_timestamp_sec: float
    event_end_timestamp_sec: float


class VideoKeyframeItem(BaseModel):
    sample_number: int
    frame_idx: int
    timestamp_sec: float
    imageUrl: str
    detections: List[DetectionItem] = Field(default_factory=list)


class StageLogItem(BaseModel):
    stage: str
    target: str
    wall_time_sec: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_sec: float
    summary: str


class ReferencePreview(BaseModel):
    id: str
    kind: Literal["image", "video_frame"]
    filename: str
    imageUrl: str
    note: Optional[str] = None
    timestampSec: Optional[float] = None


class TargetMediaMeta(BaseModel):
    filename: str
    mediaType: MediaType
    width: int
    height: int
    sizeBytes: int
    fps: Optional[float] = None
    durationSec: Optional[float] = None
    previewUrl: str


class RunRecord(BaseModel):
    id: str
    createdAtIso: str
    createdAtDisplay: str
    status: RunStatus
    isMockPreview: bool = False
    target: TargetMediaMeta
    shotMode: ShotMode
    outputMode: OutputMode
    modelId: ModelOptionId
    modelLabel: str
    adapterLabel: Optional[str] = None
    prompt: str
    temperature: float
    maxTokens: int
    samplingFps: Optional[float] = None
    outputFpsMode: Optional[Literal["original", "sampled"]] = None
    byteTrackMemorySec: Optional[float] = None
    references: List[ReferencePreview] = Field(default_factory=list)
    referenceStyleNote: Optional[str] = None
    summaryText: str = ""
    latencySec: float = 0.0
    promptTokens: int = 0
    generatedTokens: int = 0
    peakMemoryGb: float = 0.0
    detections: List[DetectionItem] = Field(default_factory=list)
    videoEvents: Optional[List[VideoEventItem]] = None
    videoKeyframes: Optional[List[VideoKeyframeItem]] = None
    annotatedVideoUrl: Optional[str] = None
    errorMessage: Optional[str] = None
    stageLogs: List[StageLogItem] = Field(default_factory=list)
    rawJson: str = "[]"


class RunSubmitResponse(BaseModel):
    runId: str
    status: RunStatus
    record: RunRecord


class ModelStatusItem(BaseModel):
    id: str
    label: str
    shortLabel: str
    engine: str
    quantization: str
    adapterLabel: Optional[str] = None
    available: bool
    loaded: bool
    note: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    bootId: str = ""
    device: str
    loadedModelId: Optional[str] = None
    queueDepth: int
    ollamaReachable: bool
    ollamaModels: List[str] = Field(default_factory=list)
    adapterBestExists: bool


class ReencodeRequest(BaseModel):
    outputFpsMode: Literal["original", "sampled"] = "original"
    byteTrackMemorySec: Optional[float] = None


class OllamaControlResponse(BaseModel):
    ok: bool
    ollamaReachable: bool
    ollamaModels: List[str] = Field(default_factory=list)
    message: str
