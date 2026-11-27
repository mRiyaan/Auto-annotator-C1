"""Pydantic schemas for the isolated Roboflow Auto-Annotation & Batch Verification feature."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

OutputMode = Literal["bbox", "polygon", "both"]
ShotMode = Literal["zero-shot", "few-shot"]
FetchScope = Literal[
    "unannotated", "annotated", "dataset", "batch", "annotation_job", "all", "rejected_retry"
]
AnnotationStatusFilter = Literal["all", "unannotated", "annotated"]
ReviewStatus = Literal["pending", "accepted", "rejected"]
RejectReason = Literal["missed_object", "false_detection", "bad_box", "double_box"]
PushStatus = Literal["not_pushed", "pushing", "pushed", "failed"]
PushDestination = Literal[
    "new_project",
    "same_project_dataset",
    "same_project_annotate",
    "same_project_new_folder",
]
JobStatus = Literal[
    "fetching",
    "queued",
    "running",
    "paused",
    "completed",
    "cancelled",
    "failed",
]


class ClassColorEntry(BaseModel):
    name: str
    color: str


class AutoDetection(BaseModel):
    id: str
    label: str
    confidence: float = 0.95
    bbox_2d: List[int]
    polygon: Optional[List[List[int]]] = None


class AutoBatchSummary(BaseModel):
    batchId: str
    batchIndex: int
    label: str
    startIndex: int
    endIndex: int
    total: int
    annotated: int
    accepted: int
    rejected: int
    pending: int
    pushed: int
    checked: bool = False
    checkedAtIso: Optional[str] = None


class AutoItemRecord(BaseModel):
    id: str
    jobId: str
    batchId: str
    batchIndex: int
    seqIndex: int
    roboflowImageId: str
    filename: str
    imageUrl: str
    width: int
    height: int
    split: str = "train"
    sourceBatchId: Optional[str] = None
    existingAnnotationsCount: int = 0
    inferenceStatus: Literal["pending", "running", "completed", "failed"] = "pending"
    detections: List[AutoDetection] = Field(default_factory=list)
    latencySec: float = 0.0
    errorMessage: Optional[str] = None
    reviewStatus: ReviewStatus = "pending"
    rejectReason: Optional[RejectReason] = None
    pushStatus: PushStatus = "not_pushed"
    pushError: Optional[str] = None
    pushedAtIso: Optional[str] = None


class AutoJobRecord(BaseModel):
    id: str
    createdAtIso: str
    createdAtDisplay: str
    workspaceSlug: str
    projectSlug: str
    projectName: str
    projectType: str
    classes: List[str] = Field(default_factory=list)
    classColors: List[ClassColorEntry] = Field(default_factory=list)
    fetchScope: FetchScope = "unannotated"
    annotationStatusFilter: AnnotationStatusFilter = "all"
    parentJobId: Optional[str] = None
    sourceBatchId: Optional[str] = None
    sourceAnnotationJobId: Optional[str] = None
    assigneeEmail: Optional[str] = None
    maxImages: int = 100
    batchSize: int = 50
    modelId: str
    modelLabel: str
    samRefinerId: str = "none"
    shotMode: ShotMode = "zero-shot"
    outputMode: OutputMode = "bbox"
    prompt: str
    styleNote: Optional[str] = None
    confidenceThreshold: float = 0.25
    temperature: float = 0.0
    maxTokens: int = 1024
    status: JobStatus = "fetching"
    errorMessage: Optional[str] = None
    totalImages: int = 0
    annotatedCount: int = 0
    acceptedCount: int = 0
    rejectedCount: int = 0
    pendingCount: int = 0
    pushedCount: int = 0
    checkedBatchesCount: int = 0
    batches: List[AutoBatchSummary] = Field(default_factory=list)
    rejectReasonCounts: Dict[str, int] = Field(default_factory=dict)
    localAcceptedDir: str = ""
    localZipPath: Optional[str] = None
    lastPushDestination: Optional[PushDestination] = None
    lastPushProject: Optional[str] = None
    lastPushTaskId: Optional[str] = None
    lastPushStatus: Optional[str] = None
    lastPushMessage: Optional[str] = None


class AutoJobDetailResponse(BaseModel):
    job: AutoJobRecord
    items: List[AutoItemRecord]


class RoboflowInspectRequest(BaseModel):
    apiKey: str
    projectSlug: str
    workspaceSlug: Optional[str] = None


class RoboflowBatchInfo(BaseModel):
    id: str
    name: str
    imagesCount: int = 0


class RoboflowAnnotationJobInfo(BaseModel):
    id: str
    name: str
    status: str = ""
    imagesCount: int = 0
    annotatedCount: int = 0
    labelerEmail: Optional[str] = None
    reviewerEmail: Optional[str] = None


class RoboflowInspectResponse(BaseModel):
    workspaceSlug: str
    projectSlug: str
    projectName: str
    projectType: str
    classes: List[str] = Field(default_factory=list)
    totalImages: int = 0
    unannotatedImages: int = 0
    keyWorkspaceSlug: Optional[str] = None
    warning: Optional[str] = None
    batches: List[RoboflowBatchInfo] = Field(default_factory=list)
    annotationJobs: List[RoboflowAnnotationJobInfo] = Field(default_factory=list)


class AutoModelCatalogResponse(BaseModel):
    detectors: List[Dict[str, Any]]
    samRefiners: List[Dict[str, Any]]
    owlv2Models: List[Dict[str, Any]] = Field(default_factory=list)


class ReviewItemPatchRequest(BaseModel):
    reviewStatus: ReviewStatus
    rejectReason: Optional[RejectReason] = None
    detections: Optional[List[AutoDetection]] = None


class SmartSelectPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1] = 1


class ManualSmartPrepareRequest(BaseModel):
    samRefinerId: str = "sam2.1-s"


class ManualSmartPrepareResponse(BaseModel):
    ready: bool = True
    cached: bool = False
    latencySec: float = 0.0
    engineUsed: str = "sam2.1-s"


class ManualSmartSelectRequest(BaseModel):
    points: List[SmartSelectPoint] = Field(default_factory=list)
    bboxPrompt: Optional[List[float]] = None
    label: str = "object"
    candidateClasses: List[str] = Field(default_factory=list)
    outputMode: OutputMode = "both"
    samRefinerId: str = "sam2.1-s"
    engine: Literal["sam", "owlv2_sam"] = "sam"
    owlModelId: Optional[str] = None
    owlThreshold: float = 0.12
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    preview: bool = False


class ManualSmartSelectResponse(BaseModel):
    detection: Optional[AutoDetection] = None
    latencySec: float = 0.0
    engineUsed: str = "sam2.1-s"
    owlMatched: bool = False
    cached: bool = False
    message: Optional[str] = None


class ManualFindObjectsRequest(BaseModel):
    classes: List[str] = Field(default_factory=list)
    confThreshold: float = 0.20
    outputMode: OutputMode = "bbox"
    detector: Literal["owlv2", "yoloe"] = "owlv2"
    owlModelId: Optional[str] = None
    samRefinerId: str = "sam2.1-s"


class ManualFindObjectsResponse(BaseModel):
    detections: List[AutoDetection] = Field(default_factory=list)
    latencySec: float = 0.0
    engineUsed: str = "owlv2"


class VisualEmbedAddRequest(BaseModel):
    queryId: Optional[str] = None
    bbox: Optional[List[float]] = None
    point: Optional[List[float]] = None
    polarity: Literal["positive", "negative"] = "positive"
    label: str = "object"
    targetClasses: Optional[List[str]] = None
    owlModelId: Optional[str] = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    minObjectness: float = Field(default=0.01, ge=0.0, le=1.0)
    nmsIou: float = Field(default=0.30, ge=0.05, le=0.95)
    useSamMasks: bool = False
    samRefinerId: str = "sam2.1-s"
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualEmbedSearchRequest(BaseModel):
    queryId: str
    label: Optional[str] = None
    targetClasses: Optional[List[str]] = None
    owlModelId: Optional[str] = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    minObjectness: float = Field(default=0.01, ge=0.0, le=1.0)
    nmsIou: float = Field(default=0.30, ge=0.05, le=0.95)
    useSamMasks: bool = False
    samRefinerId: str = "sam2.1-s"
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualEmbedClassSummary(BaseModel):
    label: str
    exemplarCount: int
    negativeCount: int = 0
    imageCount: int = 0


class VisualEmbedResponse(BaseModel):
    queryId: str
    exemplarId: Optional[str] = None
    owlModelId: str = "google/owlv2-base-patch16-ensemble"
    canUndo: bool = False
    canRedo: bool = False
    detections: List[AutoDetection] = Field(default_factory=list)
    topScore: float = 0.0
    topObjectness: Optional[float] = None
    matchedIou: Optional[float] = None
    matchedBox: Optional[List[int]] = None
    positivesCount: int = 0
    negativesCount: int = 0
    classes: List[VisualEmbedClassSummary] = Field(default_factory=list)
    latencySec: float = 0.0
    cached: bool = False
    engineUsed: str = "owlv2-embed"


class VisualEmbedBatchSearchRequest(BaseModel):
    queryId: str
    label: str = "object"
    targetClasses: Optional[List[str]] = None
    owlModelId: Optional[str] = None
    scope: Literal["current_batch", "all_batches", "item_ids"] = "current_batch"
    batchId: Optional[str] = None
    itemIds: List[str] = Field(default_factory=list)
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    minObjectness: float = Field(default=0.01, ge=0.0, le=1.0)
    nmsIou: float = Field(default=0.30, ge=0.05, le=0.95)
    useSamMasks: bool = False
    samRefinerId: str = "sam2.1-s"
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    mergeMode: Literal["append", "replace_class"] = "append"
    dedupeIou: float = Field(default=0.65, ge=0.1, le=1.0)
    skipAnnotated: bool = False
    excludeItemIds: List[str] = Field(default_factory=list)
    multiClass: bool = False
    previewOnly: bool = False


class VisualEmbedResetRequest(BaseModel):
    queryId: Optional[str] = None


class VisualEmbedUndoRedoRequest(BaseModel):
    queryId: str
    action: Literal["undo", "redo"]
    searchItemId: Optional[str] = None
    label: Optional[str] = None
    targetClasses: Optional[List[str]] = None
    owlModelId: Optional[str] = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    minObjectness: float = Field(default=0.01, ge=0.0, le=1.0)
    nmsIou: float = Field(default=0.30, ge=0.05, le=0.95)
    useSamMasks: bool = False
    samRefinerId: str = "sam2.1-s"
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualEmbedRefSpec(BaseModel):
    itemId: str
    detectionIds: List[str] = Field(default_factory=list)
    inlineDetections: List[AutoDetection] = Field(default_factory=list)


class VisualEmbedBuildRefsRequest(BaseModel):
    queryId: Optional[str] = None
    replace: bool = True
    owlModelId: Optional[str] = None
    targetClasses: Optional[List[str]] = None
    references: List[VisualEmbedRefSpec] = Field(default_factory=list)
    autoFromAnnotated: bool = False
    autoScope: Literal["current_batch", "all_batches", "accepted_only"] = "all_batches"
    batchId: Optional[str] = None
    searchItemId: Optional[str] = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    minObjectness: float = Field(default=0.01, ge=0.0, le=1.0)
    nmsIou: float = Field(default=0.30, ge=0.05, le=0.95)
    useSamMasks: bool = False
    samRefinerId: str = "sam2.1-s"
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualEmbedBuildRefsResponse(BaseModel):
    queryId: str
    owlModelId: str = "google/owlv2-base-patch16-ensemble"
    totalExemplars: int = 0
    totalNegatives: int = 0
    totalRefImages: int = 0
    refItemIds: List[str] = Field(default_factory=list)
    classes: List[VisualEmbedClassSummary] = Field(default_factory=list)
    detections: List[AutoDetection] = Field(default_factory=list)
    topScore: float = 0.0
    topObjectness: Optional[float] = None
    latencySec: float = 0.0
    engineUsed: str = "owlv2-embed"


class ManualFetchJobRequest(BaseModel):
    apiKey: str
    projectSlug: str
    workspaceSlug: str = ""
    fetchScope: FetchScope = "unannotated"
    annotationStatusFilter: AnnotationStatusFilter = "all"
    sourceBatchId: str = ""
    sourceAnnotationJobId: str = ""
    assigneeEmail: str = ""
    maxImages: int = 50
    batchSize: int = 50


class ApproveBatchRequest(BaseModel):
    acceptPending: bool = True


class PushAcceptedRequest(BaseModel):
    apiKey: str
    workspaceSlug: Optional[str] = None
    pushDestination: PushDestination = "new_project"
    targetProjectName: Optional[str] = None
    targetFolderName: Optional[str] = None
    targetAnnotationJobId: Optional[str] = None
    addToDataset: bool = True
    projectLicense: str = "MIT"
    batchId: Optional[str] = None
    includeEmpty: bool = True
    onlyUnpushed: bool = False
    batchTag: Optional[str] = None


class PushAcceptedResponse(BaseModel):
    jobId: str
    pushDestination: PushDestination = "new_project"
    targetProjectName: str
    targetFolderName: Optional[str] = None
    targetAnnotationJobId: Optional[str] = None
    addedToDataset: bool = True
    pushedImagesCount: int
    nullImagesCount: int = 0
    failedImagesCount: int = 0
    skippedEmptyCount: int
    excludedRejectedCount: int
    excludedPendingCount: int
    localZipPath: str
    taskId: Optional[str] = None
    roboflowStatus: Optional[Dict[str, Any]] = None
    message: str
