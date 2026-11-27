export type AutoOutputMode = "bbox" | "polygon" | "both";
export type AutoShotMode = "zero-shot" | "few-shot";
export type AutoFetchScope =
  | "unannotated"
  | "annotated"
  | "dataset"
  | "batch"
  | "annotation_job"
  | "all"
  | "rejected_retry";
export type AutoAnnotationStatusFilter = "all" | "unannotated" | "annotated";
export type AutoReviewStatus = "pending" | "accepted" | "rejected";
export type AutoRejectReason =
  | "missed_object"
  | "false_detection"
  | "bad_box"
  | "double_box";
export type AutoPushStatus = "not_pushed" | "pushing" | "pushed" | "failed";
export type PushDestination =
  | "new_project"
  | "same_project_dataset"
  | "same_project_annotate"
  | "same_project_new_folder";
export type AutoJobStatus =
  | "fetching"
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "cancelled"
  | "failed";

export interface ClassColorEntry {
  name: string;
  color: string;
}

export interface AutoDetection {
  id: string;
  label: string;
  confidence: number;
  bbox_2d: number[];
  polygon?: number[][] | null;
}

export interface AutoBatchSummary {
  batchId: string;
  batchIndex: number;
  label: string;
  startIndex: number;
  endIndex: number;
  total: number;
  annotated: number;
  accepted: number;
  rejected: number;
  pending: number;
  pushed: number;
  checked: boolean;
  checkedAtIso?: string | null;
}

export interface AutoItemRecord {
  id: string;
  jobId: string;
  batchId: string;
  batchIndex: number;
  seqIndex: number;
  roboflowImageId: string;
  filename: string;
  imageUrl: string;
  width: number;
  height: number;
  split: string;
  sourceBatchId?: string | null;
  existingAnnotationsCount: number;
  inferenceStatus: "pending" | "running" | "completed" | "failed";
  detections: AutoDetection[];
  latencySec: number;
  errorMessage?: string | null;
  reviewStatus: AutoReviewStatus;
  rejectReason?: AutoRejectReason | null;
  pushStatus: AutoPushStatus;
  pushError?: string | null;
  pushedAtIso?: string | null;
}

export interface AutoJobRecord {
  id: string;
  createdAtIso: string;
  createdAtDisplay: string;
  workspaceSlug: string;
  projectSlug: string;
  projectName: string;
  projectType: string;
  classes: string[];
  classColors: ClassColorEntry[];
  fetchScope: AutoFetchScope;
  annotationStatusFilter?: AutoAnnotationStatusFilter;
  parentJobId?: string | null;
  sourceBatchId?: string | null;
  sourceAnnotationJobId?: string | null;
  assigneeEmail?: string | null;
  maxImages: number;
  batchSize: number;
  modelId: string;
  modelLabel: string;
  samRefinerId: string;
  shotMode: AutoShotMode;
  outputMode: AutoOutputMode;
  prompt: string;
  styleNote?: string | null;
  confidenceThreshold: number;
  temperature: number;
  maxTokens: number;
  status: AutoJobStatus;
  errorMessage?: string | null;
  totalImages: number;
  annotatedCount: number;
  acceptedCount: number;
  rejectedCount: number;
  pendingCount: number;
  pushedCount: number;
  checkedBatchesCount: number;
  batches: AutoBatchSummary[];
  rejectReasonCounts: Record<string, number>;
  localAcceptedDir: string;
  localZipPath?: string | null;
  lastPushDestination?: PushDestination | null;
  lastPushProject?: string | null;
  lastPushTaskId?: string | null;
  lastPushStatus?: string | null;
  lastPushMessage?: string | null;
}

export interface AutoJobDetailResponse {
  job: AutoJobRecord;
  items: AutoItemRecord[];
}

export interface RoboflowBatchInfo {
  id: string;
  name: string;
  imagesCount: number;
}

export interface RoboflowAnnotationJobInfo {
  id: string;
  name: string;
  status: string;
  imagesCount: number;
  annotatedCount: number;
  labelerEmail?: string | null;
  reviewerEmail?: string | null;
}

export interface RoboflowInspectResponse {
  workspaceSlug: string;
  projectSlug: string;
  projectName: string;
  projectType: string;
  classes: string[];
  totalImages: number;
  unannotatedImages: number;
  keyWorkspaceSlug?: string | null;
  warning?: string | null;
  batches: RoboflowBatchInfo[];
  annotationJobs?: RoboflowAnnotationJobInfo[];
}

export interface AutoDetectorModel {
  id: string;
  label: string;
  shortLabel: string;
  engine: string;
  quantization: string;
  available: boolean;
  supportsPolygon: boolean;
  supportsFewShot: boolean;
  note: string;
}

export interface AutoSamRefiner {
  id: string;
  label: string;
  available: boolean;
  description: string;
}

export type OwlModelId =
  | "google/owlv2-base-patch16-ensemble"
  | "google/owlv2-large-patch14-ensemble";

export interface OwlModelOption {
  id: string;
  hfModelId: OwlModelId;
  label: string;
  shortLabel: string;
  quantization: string;
  approxSize: string;
  cached: boolean;
  available: boolean;
  note: string;
  description: string;
}

export interface AutoModelCatalogResponse {
  detectors: AutoDetectorModel[];
  samRefiners: AutoSamRefiner[];
  owlv2Models?: OwlModelOption[];
}

export interface AutoJobProgressEvent {
  jobId: string;
  status: AutoJobStatus;
  progressPct: number;
  stage: string;
  subtext: string;
  job?: AutoJobRecord;
  latestItem?: AutoItemRecord;
}

export interface PushAcceptedResponse {
  jobId: string;
  pushDestination: PushDestination;
  targetProjectName: string;
  targetFolderName?: string | null;
  targetAnnotationJobId?: string | null;
  addedToDataset: boolean;
  pushedImagesCount: number;
  nullImagesCount?: number;
  failedImagesCount?: number;
  skippedEmptyCount: number;
  excludedRejectedCount: number;
  excludedPendingCount: number;
  localZipPath: string;
  taskId?: string | null;
  roboflowStatus?: Record<string, unknown> | null;
  message: string;
}

export type ManualTool = "drag" | "bbox" | "polygon" | "smart_select";

export interface SmartSelectPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface ManualSmartPrepareResponse {
  ready: boolean;
  cached: boolean;
  latencySec: number;
  engineUsed: string;
}

export interface ManualSmartSelectRequest {
  points: SmartSelectPoint[];
  bboxPrompt?: number[] | null;
  label: string;
  candidateClasses?: string[];
  outputMode: AutoOutputMode;
  samRefinerId: string;
  engine: "sam" | "owlv2_sam";
  owlModelId?: string;
  owlThreshold?: number;
  complexity?: number;
  preview?: boolean;
}

export interface ManualSmartSelectResponse {
  detection?: AutoDetection | null;
  latencySec: number;
  engineUsed: string;
  owlMatched: boolean;
  cached?: boolean;
  message?: string | null;
}

export interface ManualFindObjectsRequest {
  classes: string[];
  confThreshold: number;
  outputMode: AutoOutputMode;
  detector: "owlv2" | "yoloe";
  owlModelId?: string;
  samRefinerId: string;
}

export interface ManualFindObjectsResponse {
  detections: AutoDetection[];
  latencySec: number;
  engineUsed: string;
}

export type VisualEmbedScope = "same_image" | "across_batch";
export type FindAiMode = "vit_ref" | "owl_text" | "yoloe";
export type VisualDrawMode = null | "positive" | "negative";
export type VisualEmbedBatchTarget =
  | "current_batch"
  | "all_batches"
  | "filtered_queue";
export type VisualEmbedMergeMode = "append" | "replace_class";

export interface VisualEmbedClassSummary {
  label: string;
  exemplarCount: number;
  negativeCount: number;
  imageCount: number;
}

export interface VisualEmbedExemplarBox {
  id: string;
  itemId: string;
  bbox_2d: number[];
  matchedBox?: number[] | null;
  polarity: "positive" | "negative";
}

export interface TargetClassOption {
  name: string;
  color: string;
  frameCount: number;
  boxCount: number;
}

export interface VisualEmbedAddRequest {
  queryId?: string | null;
  bbox?: number[] | null;
  point?: number[] | null;
  polarity: "positive" | "negative";
  label: string;
  targetClasses?: string[];
  owlModelId?: string;
  threshold: number;
  minObjectness: number;
  nmsIou?: number;
  useSamMasks: boolean;
  samRefinerId: string;
  complexity?: number;
}

export interface VisualEmbedSearchRequest {
  queryId: string;
  label?: string;
  targetClasses?: string[];
  owlModelId?: string;
  threshold: number;
  minObjectness: number;
  nmsIou?: number;
  useSamMasks: boolean;
  samRefinerId: string;
  complexity?: number;
}

export interface VisualEmbedResponse {
  queryId: string;
  exemplarId?: string | null;
  owlModelId?: string;
  canUndo?: boolean;
  canRedo?: boolean;
  detections: AutoDetection[];
  topScore: number;
  topObjectness?: number | null;
  matchedIou?: number | null;
  matchedBox?: number[] | null;
  positivesCount: number;
  negativesCount: number;
  classes?: VisualEmbedClassSummary[];
  latencySec: number;
  cached?: boolean;
  engineUsed: string;
}

export interface PinnedVitFrameSummary {
  itemId: string;
  seqIndex: number;
  filename: string;
  boxCount: number;
  classes: string[];
}

export interface VisualEmbedUndoRedoRequest {
  queryId: string;
  action: "undo" | "redo";
  searchItemId?: string | null;
  label?: string;
  targetClasses?: string[];
  owlModelId?: string;
  threshold: number;
  minObjectness: number;
  nmsIou?: number;
  useSamMasks: boolean;
  samRefinerId: string;
  complexity?: number;
}

export interface VisualEmbedRefSpec {
  itemId: string;
  detectionIds?: string[];
  inlineDetections?: AutoDetection[];
}

export interface VisualEmbedBuildRefsRequest {
  queryId?: string | null;
  replace?: boolean;
  owlModelId?: string;
  targetClasses?: string[];
  references?: VisualEmbedRefSpec[];
  autoFromAnnotated?: boolean;
  autoScope?: "current_batch" | "all_batches" | "accepted_only";
  batchId?: string | null;
  searchItemId?: string | null;
  threshold: number;
  minObjectness: number;
  nmsIou?: number;
  useSamMasks: boolean;
  samRefinerId: string;
  complexity?: number;
}

export interface VisualEmbedBuildRefsResponse {
  queryId: string;
  owlModelId?: string;
  totalExemplars: number;
  totalNegatives: number;
  totalRefImages: number;
  refItemIds: string[];
  classes: VisualEmbedClassSummary[];
  detections: AutoDetection[];
  topScore: number;
  topObjectness?: number | null;
  latencySec: number;
  engineUsed: string;
}

export interface VisualEmbedBatchSearchRequest {
  queryId: string;
  label: string;
  targetClasses?: string[];
  owlModelId?: string;
  scope: "current_batch" | "all_batches" | "item_ids";
  batchId?: string | null;
  itemIds?: string[];
  threshold: number;
  minObjectness: number;
  nmsIou?: number;
  useSamMasks: boolean;
  samRefinerId: string;
  complexity?: number;
  mergeMode: VisualEmbedMergeMode;
  dedupeIou?: number;
  skipAnnotated?: boolean;
  excludeItemIds?: string[];
  multiClass?: boolean;
  previewOnly?: boolean;
}

export interface VisualEmbedBatchEvent {
  type: "progress" | "done";
  done: number;
  total: number;
  itemId?: string;
  filename?: string;
  matchedCount?: number;
  addedCount?: number;
  totalAdded: number;
  itemsWithMatches: number;
  detections?: AutoDetection[];
  item?: AutoItemRecord | null;
  job?: AutoJobRecord | null;
}
