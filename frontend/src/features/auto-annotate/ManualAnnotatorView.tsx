"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  Ban,
  BookmarkCheck,
  BookmarkPlus,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  Eye,
  EyeOff,
  FolderGit2,
  Loader2,
  Pin,
  Plus,
  Sparkles,
  Trash2,
  X,
  XCircle,
} from "lucide-react";
import { useWorkbench } from "@/lib/workbench-store";
import {
  AutoAnnotateApiError,
  fetchAutoJobDetail,
  findObjectsItemApi,
  patchAutoItemReview,
  prepareSmartSelectItemApi,
  smartSelectItemApi,
  streamAutoJobProgress,
  submitManualFetchJob,
  visualEmbedAddApi,
  visualEmbedBuildRefsApi,
  visualEmbedResetApi,
  visualEmbedSearchApi,
  visualEmbedSearchBatchStreamApi,
  visualEmbedUndoRedoApi,
} from "./api";
import {
  loadSelectedOwlModelId,
  loadVitRefsForJob,
  loadVitTargetClassesForJob,
  saveCachedJobDetail,
  saveSelectedOwlModelId,
  saveVitRefsForJob,
  saveVitTargetClassesForJob,
} from "./cache";
import { DEFAULT_PALETTE, getClassColor } from "./color-utils";
import { FindObjectsAiPanel } from "./FindObjectsAiPanel";
import { ManualAnnotatorCanvas } from "./ManualAnnotatorCanvas";
import { RoboflowSourceCard } from "./RoboflowSourceCard";
import { useAutoAnnotate } from "./store";
import type {
  AutoDetection,
  AutoItemRecord,
  AutoJobProgressEvent,
  AutoJobRecord,
  AutoOutputMode,
  AutoReviewStatus,
  ClassColorEntry,
  FindAiMode,
  ManualTool,
  OwlModelId,
  OwlModelOption,
  PinnedVitFrameSummary,
  SmartSelectPoint,
  TargetClassOption,
  VisualDrawMode,
  VisualEmbedBatchTarget,
  VisualEmbedClassSummary,
  VisualEmbedExemplarBox,
  VisualEmbedMergeMode,
  VisualEmbedScope,
} from "./types";

const DEFAULT_OWL_MODELS: OwlModelOption[] = [
  {
    id: "owlv2-base-patch16-ensemble",
    hfModelId: "google/owlv2-base-patch16-ensemble",
    label: "OWLv2 Base Patch16 Ensemble (Fast · ~155M)",
    shortLabel: "OWLv2 Base P16",
    quantization: "FP32 MPS/CPU",
    approxSize: "~600 MB",
    cached: true,
    available: true,
    note: "Cached locally",
    description:
      "Default Google OWLv2 Base (ViT-B/16) — fast open-vocab & 768-d visual reference search.",
  },
  {
    id: "owlv2-large-patch14-ensemble",
    hfModelId: "google/owlv2-large-patch14-ensemble",
    label: "OWLv2 Large Patch14 Ensemble (High Precision · ~430M)",
    shortLabel: "OWLv2 Large P14",
    quantization: "FP32 MPS/CPU",
    approxSize: "~1.7 GB",
    cached: false,
    available: true,
    note: "Downloads HF Hub (~1.7 GB)",
    description:
      "Larger Google OWLv2 Large (ViT-L/14) — finer 14x14 patches & richer 1024-d visual embeddings.",
  },
];

type QueueFilter = "all" | "rejected" | "pending" | "accepted";

function normalizeFilterParam(raw: string | null): QueueFilter {
  if (raw === "rejected" || raw === "pending" || raw === "accepted") return raw;
  return "all";
}

function cloneDetections(dets: AutoDetection[]): AutoDetection[] {
  return dets.map((d) => ({
    ...d,
    bbox_2d: [...d.bbox_2d],
    polygon: d.polygon ? d.polygon.map((p) => [...p]) : null,
  }));
}

function boxIou1000(a: number[], b: number[]): number {
  if (a.length !== 4 || b.length !== 4) return 0;
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
  return inter / Math.max(1e-6, areaA + areaB - inter);
}

export function ManualAnnotatorView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryJobId = searchParams.get("job");
  const queryFilter = searchParams.get("filter");
  const queryBatch = searchParams.get("batch");
  const queryReason = searchParams.get("reason");

  const { backendUrl, backendStatus } = useWorkbench();
  const {
    apiKey,
    workspaceSlug,
    projectSlug,
    fetchScope,
    annotationStatusFilter,
    sourceBatchId,
    sourceAnnotationJobId,
    assigneeEmail,
    maxImages,
    batchSize,
    models,
    jobs,
    setJobs,
    refreshJobs,
    sessionEpoch,
  } = useAutoAnnotate();

  const owlModelOptions = useMemo<OwlModelOption[]>(
    () =>
      models?.owlv2Models?.length
        ? (models.owlv2Models as OwlModelOption[])
        : DEFAULT_OWL_MODELS,
    [models?.owlv2Models]
  );

  // Pull Roboflow dataset batches directly inside Manual Annotator
  const [fetchModalOpen, setFetchModalOpen] = useState(false);
  const [manualFetchBusy, setManualFetchBusy] = useState(false);
  const [manualFetchError, setManualFetchError] = useState<string | null>(null);
  const [manualFetchProgress, setManualFetchProgress] = useState<{
    progressPct: number;
    stage: string;
    subtext: string;
  } | null>(null);
  const manualFetchSseRef = useRef<(() => void) | null>(null);

  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [jobDetail, setJobDetail] = useState<AutoJobRecord | null>(null);
  const [allItems, setAllItems] = useState<AutoItemRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Filters
  const [queueFilter, setQueueFilter] = useState<QueueFilter>(() =>
    normalizeFilterParam(queryFilter)
  );
  const [batchFilter, setBatchFilter] = useState<string>(queryBatch || "all");
  const [reasonFilter, setReasonFilter] = useState<string>(queryReason || "all");

  // Keep current working queue pinned across status updates until user changes filter/job/batch
  const [pinnedKey, setPinnedKey] = useState<string>("");
  const [pinnedItemIds, setPinnedItemIds] = useState<string[]>([]);
  const [activeItemId, setActiveItemId] = useState<string>("");

  // Canvas & annotation editing state
  const [tool, setTool] = useState<ManualTool>("bbox");
  const [leftTab, setLeftTab] = useState<"classes" | "layers">("classes");
  const [customClasses, setCustomClasses] = useState<ClassColorEntry[]>([]);
  const [newClassInput, setNewClassInput] = useState("");
  const [activeClass, setActiveClass] = useState<string>("object");

  const [draftDetections, setDraftDetections] = useState<AutoDetection[]>([]);
  const [undoStack, setUndoStack] = useState<AutoDetection[][]>([]);
  const [redoStack, setRedoStack] = useState<AutoDetection[][]>([]);
  const [dirty, setDirty] = useState(false);

  const [selectedDetId, setSelectedDetId] = useState<string | null>(null);
  const [hiddenDetIds, setHiddenDetIds] = useState<Set<string>>(new Set());

  // Save state
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">(
    "idle"
  );
  const [saveError, setSaveError] = useState<string | null>(null);
  const [copiedFilename, setCopiedFilename] = useState(false);

  // Smart Select controls (pure SAM 2.1 interactive segmentation)
  const [smartOutputMode, setSmartOutputMode] = useState<AutoOutputMode>("polygon");
  const [smartSamRefinerId, setSmartSamRefinerId] = useState<string>("sam2.1-s");
  const [smartMultiPoint, setSmartMultiPoint] = useState<boolean>(false);
  const [smartComplexity, setSmartComplexity] = useState<number>(0.5);
  const [smartPreparing, setSmartPreparing] = useState<boolean>(false);
  const [smartPreparedKey, setSmartPreparedKey] = useState<string>("");
  const [smartBusy, setSmartBusy] = useState<boolean>(false);
  const [smartStatusMsg, setSmartStatusMsg] = useState<string | null>(null);

  // Find Objects with AI: ViT Reference Bank + OWLv2 Visual Embed + Text Prompts
  const [findOpen, setFindOpen] = useState<boolean>(false);
  const [findMode, setFindMode] = useState<FindAiMode>("vit_ref");
  const [owlModelId, setOwlModelId] = useState<OwlModelId>(
    () => loadSelectedOwlModelId() ?? "google/owlv2-base-patch16-ensemble"
  );
  const [visualDrawMode, setVisualDrawMode] = useState<VisualDrawMode>(null);

  useEffect(() => {
    saveSelectedOwlModelId(owlModelId);
  }, [owlModelId]);
  const [vitRefsByItem, setVitRefsByItem] = useState<Record<string, string[]>>({});
  const [vitRefsLoadedJobId, setVitRefsLoadedJobId] = useState<string>("");
  const [vitBankDirty, setVitBankDirty] = useState<boolean>(false);
  const [vitIndexing, setVitIndexing] = useState<boolean>(false);
  const [vitClassSummaries, setVitClassSummaries] = useState<
    VisualEmbedClassSummary[]
  >([]);
  const [vitSkipAnnotated, setVitSkipAnnotated] = useState<boolean>(true);
  const [selectedAiClasses, setSelectedAiClasses] = useState<string[]>([]);
  const [selectedAiClassesLoadedJobId, setSelectedAiClassesLoadedJobId] =
    useState<string>("");

  const [visualQueryId, setVisualQueryId] = useState<string>("");
  const [visualScope, setVisualScope] = useState<VisualEmbedScope>("same_image");
  const [visualBatchTarget, setVisualBatchTarget] =
    useState<VisualEmbedBatchTarget>("current_batch");
  const [visualMergeMode, setVisualMergeMode] =
    useState<VisualEmbedMergeMode>("append");
  const [visualThreshold, setVisualThreshold] = useState<number>(0.65);
  const [visualMinObjectness, setVisualMinObjectness] = useState<number>(0.01);
  const [visualUseSamMasks, setVisualUseSamMasks] = useState<boolean>(false);
  const [visualAutoSearchOnSwitch, setVisualAutoSearchOnSwitch] =
    useState<boolean>(false);
  const [stagedVisualHitsByItem, setStagedVisualHitsByItem] = useState<
    Record<string, AutoDetection[]>
  >({});
  const [showVisualOverlay, setShowVisualOverlay] = useState<boolean>(true);
  const [dismissedVisualItemIds, setDismissedVisualItemIds] = useState<
    Set<string>
  >(() => new Set());
  const [visualExemplars, setVisualExemplars] = useState<VisualEmbedExemplarBox[]>(
    []
  );
  const [visualExemplarsRedo, setVisualExemplarsRedo] = useState<
    VisualEmbedExemplarBox[]
  >([]);
  const [canUndoVisualEx, setCanUndoVisualEx] = useState<boolean>(false);
  const [canRedoVisualEx, setCanRedoVisualEx] = useState<boolean>(false);
  const [visualTopScore, setVisualTopScore] = useState<number | null>(null);
  const [visualTopObjectness, setVisualTopObjectness] = useState<number | null>(
    null
  );
  const [vitBankFeedback, setVitBankFeedback] = useState<string | null>(null);
  const [visualPosCount, setVisualPosCount] = useState<number>(0);
  const [visualNegCount, setVisualNegCount] = useState<number>(0);
  const [visualLastLatency, setVisualLastLatency] = useState<number | null>(null);
  const [visualMatchedIou, setVisualMatchedIou] = useState<number | null>(null);
  const [visualBusy, setVisualBusy] = useState<boolean>(false);
  const [visualError, setVisualError] = useState<string | null>(null);
  const [visualBatchProgress, setVisualBatchProgress] = useState<{
    running: boolean;
    done: number;
    total: number;
    totalAdded: number;
    itemsWithMatches: number;
    currentFilename: string;
  } | null>(null);
  const visualBatchAbortRef = useRef<AbortController | null>(null);
  const visualSearchAbortRef = useRef<AbortController | null>(null);
  const skipAutoSearchItemRef = useRef<string>("");

  // Text prompt detection parameters (OWL Text / YOLOE-26s)
  const [findClassesText, setFindClassesText] = useState<string>("");
  const [findOutputMode, setFindOutputMode] = useState<AutoOutputMode>("both");
  const [findConf, setFindConf] = useState<number>(0.2);
  const [findBusy, setFindBusy] = useState<boolean>(false);
  const [findMessage, setFindMessage] = useState<string | null>(null);

  // Sync URL params cleanly on incoming navigation
  useEffect(() => {
    if (queryFilter) setQueueFilter(normalizeFilterParam(queryFilter));
    if (queryBatch) setBatchFilter(queryBatch);
    if (queryReason) setReasonFilter(queryReason);
  }, [queryBatch, queryFilter, queryReason]);

  useEffect(() => {
    if (queryJobId) {
      setSelectedJobId(queryJobId);
      return;
    }
    if (!selectedJobId && jobs.length > 0) {
      setSelectedJobId(jobs[0].id);
    } else if (jobs.length > 0 && !jobs.some((j) => j.id === selectedJobId)) {
      setSelectedJobId(jobs[0].id);
    }
  }, [jobs, queryJobId, selectedJobId]);

  const loadJob = useCallback(
    async (jobId: string) => {
      if (!jobId || backendStatus !== "online") return;
      setLoading(true);
      setLoadError(null);
      try {
        const res = await fetchAutoJobDetail(backendUrl, jobId);
        setJobDetail(res.job);
        setAllItems(res.items);
        saveCachedJobDetail(res.job.id, res);
        setJobs((prev) => {
          const exists = prev.some((j) => j.id === res.job.id);
          return exists
            ? prev.map((j) => (j.id === res.job.id ? res.job : j))
            : [res.job, ...prev];
        });
        if (res.job.classes.length > 0) {
          setActiveClass((prev) =>
            res.job.classes.some((c) => c.toLowerCase() === prev.toLowerCase())
              ? prev
              : res.job.classes[0]
          );
          setFindClassesText((prev) =>
            prev.trim() ? prev : res.job.classes.join(", ")
          );
        }
        if (
          res.job.outputMode === "bbox" ||
          res.job.outputMode === "polygon" ||
          res.job.outputMode === "both"
        ) {
          setSmartOutputMode(res.job.outputMode);
        }
      } catch (err) {
        if (err instanceof AutoAnnotateApiError && err.status === 404) {
          setJobDetail(null);
          setAllItems([]);
          void refreshJobs();
          setLoadError(`Job '${jobId}' was not found on the backend.`);
        } else {
          setLoadError(
            err instanceof Error ? err.message : "Failed to load annotation job."
          );
        }
      } finally {
        setLoading(false);
      }
    },
    [backendStatus, backendUrl, refreshJobs, setJobs]
  );

  useEffect(() => {
    if (selectedJobId) {
      void loadJob(selectedJobId);
    }
  }, [loadJob, selectedJobId]);

  const currentFilterKey = `${jobDetail?.id || ""}::${queueFilter}::${batchFilter}::${reasonFilter}`;

  useEffect(() => {
    if (!jobDetail) return;
    if (pinnedKey === currentFilterKey && pinnedItemIds.length > 0) return;

    const matching = allItems.filter((it) => {
      if (queueFilter !== "all" && it.reviewStatus !== queueFilter) return false;
      if (batchFilter !== "all" && it.batchId !== batchFilter) return false;
      if (
        queueFilter === "rejected" &&
        reasonFilter !== "all" &&
        it.rejectReason !== reasonFilter
      ) {
        return false;
      }
      return true;
    });

    const ids = matching.map((it) => it.id);
    setPinnedKey(currentFilterKey);
    setPinnedItemIds(ids);
    if (ids.length > 0 && !ids.includes(activeItemId)) {
      setActiveItemId(ids[0]);
    } else if (ids.length === 0) {
      setActiveItemId("");
    }
  }, [
    activeItemId,
    allItems,
    batchFilter,
    currentFilterKey,
    jobDetail,
    pinnedItemIds.length,
    pinnedKey,
    queueFilter,
    reasonFilter,
  ]);

  const itemsById = useMemo(() => {
    const m = new Map<string, AutoItemRecord>();
    for (const it of allItems) m.set(it.id, it);
    return m;
  }, [allItems]);

  const queueItems = useMemo(
    () =>
      pinnedItemIds
        .map((id) => itemsById.get(id))
        .filter((it): it is AutoItemRecord => Boolean(it)),
    [itemsById, pinnedItemIds]
  );

  const currentIndex = useMemo(
    () => queueItems.findIndex((it) => it.id === activeItemId),
    [activeItemId, queueItems]
  );

  const currentItem = useMemo(
    () => (currentIndex >= 0 ? queueItems[currentIndex] : null),
    [currentIndex, queueItems]
  );

  const activeBatchId = useMemo(
    () => (batchFilter !== "all" ? batchFilter : currentItem?.batchId || "all"),
    [batchFilter, currentItem?.batchId]
  );

  const currentBatchCount = useMemo(
    () =>
      activeBatchId !== "all"
        ? allItems.filter((it) => it.batchId === activeBatchId).length
        : allItems.length,
    [activeBatchId, allItems]
  );

  // Sync draft detections when switching active item
  const loadedItemIdRef = useRef<string>("");
  useEffect(() => {
    if (!currentItem) {
      loadedItemIdRef.current = "";
      setDraftDetections([]);
      setUndoStack([]);
      setRedoStack([]);
      setDirty(false);
      setSelectedDetId(null);
      return;
    }
    if (loadedItemIdRef.current !== currentItem.id) {
      loadedItemIdRef.current = currentItem.id;
      setDraftDetections(cloneDetections(currentItem.detections));
      setUndoStack([]);
      setRedoStack([]);
      setDirty(false);
      setSelectedDetId(null);
      setHiddenDetIds(new Set());
      setSaveState("idle");
      setSaveError(null);
      setSmartStatusMsg(null);
      setFindMessage(null);
    }
  }, [currentItem]);

  const mergedClassColors = useMemo<ClassColorEntry[]>(() => {
    const base = [...(jobDetail?.classColors ?? []), ...customClasses];
    const seen = new Set<string>();
    const out: ClassColorEntry[] = [];
    for (const entry of base) {
      const k = entry.name.trim().toLowerCase();
      if (!k || seen.has(k)) continue;
      seen.add(k);
      out.push(entry);
    }
    for (const c of jobDetail?.classes ?? []) {
      const clean = c.trim();
      if (!clean || seen.has(clean.toLowerCase())) continue;
      seen.add(clean.toLowerCase());
      out.push({
        name: clean,
        color: DEFAULT_PALETTE[out.length % DEFAULT_PALETTE.length],
      });
    }
    for (const det of draftDetections) {
      const clean = det.label.trim();
      if (!clean || seen.has(clean.toLowerCase())) continue;
      seen.add(clean.toLowerCase());
      out.push({
        name: clean,
        color: DEFAULT_PALETTE[out.length % DEFAULT_PALETTE.length],
      });
    }
    if (out.length === 0) {
      out.push({ name: "object", color: DEFAULT_PALETTE[0] });
    }
    return out;
  }, [customClasses, draftDetections, jobDetail?.classColors, jobDetail?.classes]);

  const classCountsOnItem = useMemo(() => {
    const counts = new Map<string, number>();
    for (const d of draftDetections) {
      const k = d.label.trim().toLowerCase() || "object";
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    return counts;
  }, [draftDetections]);

  const presentClassColors = useMemo(
    () =>
      mergedClassColors.filter(
        (row) => (classCountsOnItem.get(row.name.trim().toLowerCase()) || 0) > 0
      ),
    [classCountsOnItem, mergedClassColors]
  );

  const availableClassOptions = useMemo<TargetClassOption[]>(() => {
    const orderedKeys: string[] = [];
    const byKey = new Map<
      string,
      { name: string; color: string; frames: Set<string>; boxCount: number }
    >();
    const ensureEntry = (rawName: string) => {
      const clean = rawName.trim() || "object";
      const key = clean.toLowerCase();
      let entry = byKey.get(key);
      if (!entry) {
        entry = {
          name: clean,
          color: getClassColor(clean, mergedClassColors),
          frames: new Set<string>(),
          boxCount: 0,
        };
        byKey.set(key, entry);
        orderedKeys.push(key);
      }
      return entry;
    };
    for (const cc of mergedClassColors) {
      ensureEntry(cc.name);
    }
    for (const it of allItems) {
      const dets =
        currentItem && it.id === currentItem.id
          ? draftDetections
          : it.detections;
      for (const det of dets) {
        const entry = ensureEntry(det.label);
        entry.frames.add(it.id);
        entry.boxCount += 1;
      }
    }
    return orderedKeys.map((k) => {
      const e = byKey.get(k)!;
      return {
        name: e.name,
        color: e.color,
        frameCount: e.frames.size,
        boxCount: e.boxCount,
      };
    });
  }, [allItems, currentItem, draftDetections, mergedClassColors]);

  type QueuedSave = {
    jobId: string;
    itemId: string;
    detections: AutoDetection[];
    reviewStatus: AutoReviewStatus;
    rejectReason: AutoItemRecord["rejectReason"];
    resolveWaiters: Array<(ok: boolean) => void>;
  };

  const activeItemIdRef = useRef(activeItemId);
  activeItemIdRef.current = activeItemId;
  const saveInFlightRef = useRef(false);
  const queuedSavesRef = useRef<QueuedSave[]>([]);

  const drainSaveQueue = useCallback(async () => {
    if (saveInFlightRef.current) return;
    saveInFlightRef.current = true;
    try {
      while (queuedSavesRef.current.length > 0) {
        const task = queuedSavesRef.current.shift()!;
        setSaveState("saving");
        setSaveError(null);
        let ok = false;
        try {
          const res = await patchAutoItemReview(
            backendUrl,
            task.jobId,
            task.itemId,
            {
              reviewStatus: task.reviewStatus,
              rejectReason:
                task.reviewStatus === "rejected" ? task.rejectReason ?? null : null,
              detections: task.detections,
            }
          );
          ok = true;
          setJobDetail(res.job);
          setJobs((prev) => prev.map((j) => (j.id === res.job.id ? res.job : j)));
          setAllItems((prev) => {
            const updated = prev.map((it) => (it.id === res.item.id ? res.item : it));
            saveCachedJobDetail(res.job.id, { job: res.job, items: updated });
            return updated;
          });
          const hasMoreForSameItem = queuedSavesRef.current.some(
            (q) => q.itemId === task.itemId
          );
          if (activeItemIdRef.current === task.itemId && !hasMoreForSameItem) {
            setDirty(false);
            setSaveState("saved");
          }
        } catch (err) {
          ok = false;
          setSaveState("error");
          setSaveError(
            err instanceof Error
              ? err.message
              : "Failed to save manual annotations."
          );
        } finally {
          for (const resolve of task.resolveWaiters) {
            resolve(ok);
          }
        }
      }
    } finally {
      saveInFlightRef.current = false;
      if (queuedSavesRef.current.length > 0) {
        void drainSaveQueue();
      }
    }
  }, [backendUrl, setJobs]);

  const persistItem = useCallback(
    (
      targetItem: AutoItemRecord,
      nextDetections: AutoDetection[],
      nextStatus: AutoReviewStatus,
      nextReason: AutoItemRecord["rejectReason"] = null
    ): Promise<boolean> => {
      if (!jobDetail) return Promise.resolve(false);
      return new Promise<boolean>((resolve) => {
        const snapshot = cloneDetections(nextDetections);
        const existingIdx = queuedSavesRef.current.findIndex(
          (q) => q.jobId === jobDetail.id && q.itemId === targetItem.id
        );
        if (existingIdx >= 0) {
          const existing = queuedSavesRef.current[existingIdx];
          queuedSavesRef.current[existingIdx] = {
            ...existing,
            detections: snapshot,
            reviewStatus: nextStatus,
            rejectReason: nextStatus === "rejected" ? nextReason ?? null : null,
            resolveWaiters: [...existing.resolveWaiters, resolve],
          };
        } else {
          queuedSavesRef.current.push({
            jobId: jobDetail.id,
            itemId: targetItem.id,
            detections: snapshot,
            reviewStatus: nextStatus,
            rejectReason: nextStatus === "rejected" ? nextReason ?? null : null,
            resolveWaiters: [resolve],
          });
        }
        void drainSaveQueue();
      });
    },
    [drainSaveQueue, jobDetail]
  );

  const enqueueAutoSave = useCallback(
    (targetItem: AutoItemRecord | null | undefined, nextDetections: AutoDetection[]) => {
      if (!targetItem) return;
      const nextStatus: AutoReviewStatus =
        nextDetections.length > 0 &&
        (targetItem.reviewStatus === "rejected" ||
          targetItem.reviewStatus === "pending")
          ? "accepted"
          : targetItem.reviewStatus;
      const nextReason =
        nextStatus === "rejected" ? targetItem.rejectReason ?? null : null;
      void persistItem(targetItem, nextDetections, nextStatus, nextReason);
    },
    [persistItem]
  );

  const handleDetectionsChange = useCallback(
    (next: AutoDetection[], recordHistory = true) => {
      if (recordHistory) {
        setUndoStack((prev) => [...prev.slice(-49), cloneDetections(draftDetections)]);
        setRedoStack([]);
      }
      if (currentItem) {
        const pinnedIds = vitRefsByItem[currentItem.id];
        if (pinnedIds && pinnedIds.length > 0) {
          const nextById = new Map(next.map((d) => [d.id, d]));
          const prevById = new Map(draftDetections.map((d) => [d.id, d]));
          const surviving = pinnedIds.filter((id) => nextById.has(id));
          if (surviving.length !== pinnedIds.length) {
            setVitRefsByItem((prev) => {
              const copy = { ...prev };
              if (surviving.length > 0) copy[currentItem.id] = surviving;
              else delete copy[currentItem.id];
              return copy;
            });
            setVitBankDirty(true);
          } else {
            const mutatedPinned = surviving.some((id) => {
              const prevDet = prevById.get(id);
              const nextDet = nextById.get(id);
              if (!prevDet || !nextDet) return true;
              return (
                prevDet.label !== nextDet.label ||
                JSON.stringify(prevDet.bbox_2d) !== JSON.stringify(nextDet.bbox_2d)
              );
            });
            if (mutatedPinned) {
              setVitBankDirty(true);
            }
          }
        }
      }
      setDraftDetections(next);
      const changed =
        JSON.stringify(next) !== JSON.stringify(currentItem?.detections ?? []);
      setDirty(changed);
      setSaveState("idle");
      if (recordHistory && currentItem) {
        enqueueAutoSave(currentItem, next);
      }
    },
    [currentItem, draftDetections, enqueueAutoSave, vitRefsByItem]
  );

  const handleCommitDragEdit = useCallback(
    (latestDetections: AutoDetection[]) => {
      if (!currentItem) return;
      if ((vitRefsByItem[currentItem.id] || []).length > 0) {
        setVitBankDirty(true);
      }
      enqueueAutoSave(currentItem, cloneDetections(latestDetections));
    },
    [currentItem, enqueueAutoSave, vitRefsByItem]
  );

  const handlePushHistorySnapshot = useCallback((snapshot: AutoDetection[]) => {
    setUndoStack((prev) => [...prev.slice(-49), cloneDetections(snapshot)]);
    setRedoStack([]);
  }, []);

  const handleUndo = useCallback(() => {
    if (undoStack.length === 0) return;
    const last = cloneDetections(undoStack[undoStack.length - 1]);
    setUndoStack((prev) => prev.slice(0, -1));
    setRedoStack((prev) => [...prev.slice(-49), cloneDetections(draftDetections)]);
    setDraftDetections(last);
    const changed =
      JSON.stringify(last) !== JSON.stringify(currentItem?.detections ?? []);
    setDirty(changed);
    setSaveState("idle");
    if (currentItem) {
      enqueueAutoSave(currentItem, last);
    }
  }, [currentItem, draftDetections, enqueueAutoSave, undoStack]);

  const handleRedo = useCallback(() => {
    if (redoStack.length === 0) return;
    const next = cloneDetections(redoStack[redoStack.length - 1]);
    setRedoStack((prev) => prev.slice(0, -1));
    setUndoStack((prev) => [...prev.slice(-49), cloneDetections(draftDetections)]);
    setDraftDetections(next);
    const changed =
      JSON.stringify(next) !== JSON.stringify(currentItem?.detections ?? []);
    setDirty(changed);
    setSaveState("idle");
    if (currentItem) {
      enqueueAutoSave(currentItem, next);
    }
  }, [currentItem, draftDetections, enqueueAutoSave, redoStack]);

  const navigateToItem = useCallback(
    async (targetId: string) => {
      if (!targetId || targetId === activeItemId) return;
      if (currentItem && dirty) {
        await persistItem(
          currentItem,
          draftDetections,
          currentItem.reviewStatus,
          currentItem.rejectReason ?? null
        );
      }
      setActiveItemId(targetId);
    },
    [activeItemId, currentItem, dirty, draftDetections, persistItem]
  );

  const handleSaveAndAccept = useCallback(async () => {
    if (!currentItem) return;
    await persistItem(currentItem, draftDetections, "accepted", null);
  }, [currentItem, draftDetections, persistItem]);

  const handleAcceptAndNext = useCallback(async () => {
    if (!currentItem) return;
    const ok = await persistItem(currentItem, draftDetections, "accepted", null);
    if (ok && currentIndex >= 0 && currentIndex + 1 < queueItems.length) {
      setActiveItemId(queueItems[currentIndex + 1].id);
    }
  }, [currentIndex, currentItem, draftDetections, persistItem, queueItems]);

  const handleMarkNull = useCallback(async () => {
    if (!currentItem) return;
    if (draftDetections.length > 0) {
      handlePushHistorySnapshot(draftDetections);
    }
    if ((vitRefsByItem[currentItem.id] || []).length > 0) {
      setVitRefsByItem((prev) => {
        const copy = { ...prev };
        delete copy[currentItem.id];
        return copy;
      });
      setVitBankDirty(true);
    }
    setDraftDetections([]);
    setSelectedDetId(null);
    await persistItem(currentItem, [], "accepted", null);
  }, [
    currentItem,
    draftDetections,
    handlePushHistorySnapshot,
    persistItem,
    vitRefsByItem,
  ]);

  const handleDeleteSelected = useCallback(() => {
    if (!selectedDetId) return;
    handleDetectionsChange(
      draftDetections.filter((d) => d.id !== selectedDetId),
      true
    );
    setSelectedDetId(null);
  }, [draftDetections, handleDetectionsChange, selectedDetId]);

  // Pre-encode active image for instant hover & 1-click Smart Select / ViT SAM masks
  const currentPrepareKey =
    currentItem && jobDetail
      ? `${jobDetail.id}:${currentItem.id}:${smartSamRefinerId}`
      : "";
  const smartReady =
    Boolean(currentPrepareKey) && smartPreparedKey === currentPrepareKey;

  useEffect(() => {
    const needsSamPrep =
      tool === "smart_select" ||
      (findOpen && findMode === "vit_ref" && visualUseSamMasks);
    if (
      !needsSamPrep ||
      !jobDetail ||
      !currentItem ||
      backendStatus !== "online" ||
      !currentPrepareKey
    ) {
      setSmartPreparing(false);
      return;
    }
    if (smartPreparedKey === currentPrepareKey) {
      return;
    }

    const controller = new AbortController();
    setSmartPreparing(true);
    setSmartStatusMsg("Encoding image...");
    void prepareSmartSelectItemApi(
      backendUrl,
      jobDetail.id,
      currentItem.id,
      smartSamRefinerId,
      controller.signal
    )
      .then((res) => {
        if (controller.signal.aborted) return;
        setSmartPreparedKey(currentPrepareKey);
        setSmartStatusMsg(
          res.cached
            ? `${res.engineUsed} · Ready`
            : `${res.engineUsed} · ${res.latencySec.toFixed(2)}s`
        );
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setSmartStatusMsg(
          err instanceof Error ? err.message : "Failed to prepare Smart Select."
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setSmartPreparing(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [
    backendStatus,
    backendUrl,
    currentItem,
    currentPrepareKey,
    findMode,
    findOpen,
    jobDetail,
    smartPreparedKey,
    smartSamRefinerId,
    tool,
    visualUseSamMasks,
  ]);

  // Load & save session-scoped pinned ViT Reference boxes per job
  useEffect(() => {
    const jid = jobDetail?.id || "";
    if (!jid) {
      setVitRefsByItem({});
      setVitRefsLoadedJobId("");
      setVitBankDirty(false);
      setVitClassSummaries([]);
      setVitBankFeedback(null);
      return;
    }
    if (vitRefsLoadedJobId !== jid) {
      const restored = loadVitRefsForJob(jid);
      setVitRefsByItem(restored);
      setVitRefsLoadedJobId(jid);
      setVitBankDirty(Object.keys(restored).length > 0);
      setVitClassSummaries([]);
      setVitBankFeedback(null);
    }
  }, [jobDetail?.id, vitRefsLoadedJobId]);

  useEffect(() => {
    setVisualTopScore(null);
    setVisualTopObjectness(null);
  }, [currentItem?.id]);

  useEffect(() => {
    if (!vitRefsLoadedJobId || vitRefsLoadedJobId !== jobDetail?.id) return;
    saveVitRefsForJob(vitRefsLoadedJobId, vitRefsByItem);
  }, [jobDetail?.id, vitRefsByItem, vitRefsLoadedJobId]);

  useEffect(() => {
    const jid = jobDetail?.id || "";
    if (!jid) {
      setSelectedAiClasses([]);
      setSelectedAiClassesLoadedJobId("");
      return;
    }
    if (selectedAiClassesLoadedJobId === jid) return;
    const saved = loadVitTargetClassesForJob(jid);
    if (saved !== null) {
      const canonByLow = new Map(
        availableClassOptions.map((o) => [o.name.trim().toLowerCase(), o.name])
      );
      const next = Array.from(
        new Set(
          saved
            .map((c) => canonByLow.get(c.trim().toLowerCase()) || c.trim())
            .filter(Boolean)
        )
      );
      setSelectedAiClasses(next);
    } else {
      setSelectedAiClasses(availableClassOptions.map((o) => o.name));
    }
    setSelectedAiClassesLoadedJobId(jid);
  }, [availableClassOptions, jobDetail?.id, selectedAiClassesLoadedJobId]);

  useEffect(() => {
    if (
      !selectedAiClassesLoadedJobId ||
      selectedAiClassesLoadedJobId !== jobDetail?.id
    ) {
      return;
    }
    saveVitTargetClassesForJob(selectedAiClassesLoadedJobId, selectedAiClasses);
  }, [jobDetail?.id, selectedAiClasses, selectedAiClassesLoadedJobId]);

  const selectedAiClassLowSet = useMemo(
    () =>
      new Set(
        selectedAiClasses.map((c) => c.trim().toLowerCase()).filter(Boolean)
      ),
    [selectedAiClasses]
  );

  const activeVitRefsByItem = useMemo(() => {
    const filtered: Record<string, string[]> = {};
    if (selectedAiClassLowSet.size === 0) return filtered;
    for (const [iid, detIds] of Object.entries(vitRefsByItem)) {
      if (!detIds || detIds.length === 0) continue;
      const detList =
        currentItem && currentItem.id === iid
          ? draftDetections
          : itemsById.get(iid)?.detections ?? [];
      const wanted = new Set(detIds);
      const matchingIds = detList
        .filter(
          (d) =>
            wanted.has(d.id) &&
            selectedAiClassLowSet.has((d.label.trim() || "object").toLowerCase())
        )
        .map((d) => d.id);
      if (matchingIds.length > 0) {
        filtered[iid] = matchingIds;
      }
    }
    return filtered;
  }, [
    currentItem,
    draftDetections,
    itemsById,
    selectedAiClassLowSet,
    vitRefsByItem,
  ]);

  const vitRefDetIdsForCurrent = useMemo(
    () => new Set(currentItem ? activeVitRefsByItem[currentItem.id] || [] : []),
    [activeVitRefsByItem, currentItem]
  );

  const currentMatchingDetIds = useMemo(
    () =>
      draftDetections
        .filter((d) =>
          selectedAiClassLowSet.has((d.label.trim() || "object").toLowerCase())
        )
        .map((d) => d.id),
    [draftDetections, selectedAiClassLowSet]
  );

  const annotatedItemsCount = useMemo(
    () =>
      allItems.filter((it) => {
        const dets =
          currentItem && it.id === currentItem.id
            ? draftDetections
            : it.detections;
        if (selectedAiClassLowSet.size === 0) return false;
        return dets.some((d) =>
          selectedAiClassLowSet.has((d.label.trim() || "object").toLowerCase())
        );
      }).length,
    [allItems, currentItem, draftDetections, selectedAiClassLowSet]
  );

  const { refImageCount, refBoxCount, vitClassBadges } = useMemo(() => {
    const clsStats = new Map<
      string,
      { label: string; count: number; items: Set<string> }
    >();
    let imgs = 0;
    let boxes = 0;
    for (const [iid, detIds] of Object.entries(activeVitRefsByItem)) {
      if (!detIds || detIds.length === 0) continue;
      const detList =
        currentItem && currentItem.id === iid
          ? draftDetections
          : itemsById.get(iid)?.detections ?? [];
      const wanted = new Set(detIds);
      const matched = detList.filter((d) => wanted.has(d.id));
      if (matched.length === 0) continue;
      imgs += 1;
      boxes += matched.length;
      for (const d of matched) {
        const rawLbl = d.label.trim() || "object";
        const k = rawLbl.toLowerCase();
        let entry = clsStats.get(k);
        if (!entry) {
          entry = { label: rawLbl, count: 0, items: new Set() };
          clsStats.set(k, entry);
        }
        entry.count += 1;
        entry.items.add(iid);
      }
    }
    const badges = Array.from(clsStats.values())
      .map((e) => ({ label: e.label, count: e.count, imageCount: e.items.size }))
      .sort((a, b) => a.label.localeCompare(b.label));
    return { refImageCount: imgs, refBoxCount: boxes, vitClassBadges: badges };
  }, [activeVitRefsByItem, currentItem, draftDetections, itemsById]);

  const isCurrentFramePinned = useMemo(() => {
    if (!currentItem || currentMatchingDetIds.length === 0) return false;
    const pinned = activeVitRefsByItem[currentItem.id];
    if (!pinned || pinned.length === 0) return false;
    const set = new Set(pinned);
    return currentMatchingDetIds.every((id) => set.has(id));
  }, [activeVitRefsByItem, currentItem, currentMatchingDetIds]);

  const pinnedVitFrames = useMemo<PinnedVitFrameSummary[]>(() => {
    const out: PinnedVitFrameSummary[] = [];
    for (const [iid, detIds] of Object.entries(activeVitRefsByItem)) {
      if (!detIds || detIds.length === 0) continue;
      const it = itemsById.get(iid);
      const isCur = currentItem && currentItem.id === iid;
      const detList = isCur ? draftDetections : it?.detections ?? [];
      const wanted = new Set(detIds);
      const matched = detList.filter((d) => wanted.has(d.id));
      if (matched.length === 0) continue;
      const clsSet = Array.from(
        new Set(matched.map((d) => d.label.trim() || "object"))
      );
      out.push({
        itemId: iid,
        seqIndex: it?.seqIndex ?? 0,
        filename: it?.filename ?? iid,
        boxCount: matched.length,
        classes: clsSet,
      });
    }
    return out.sort((a, b) => a.seqIndex - b.seqIndex);
  }, [activeVitRefsByItem, currentItem, draftDetections, itemsById]);

  const allAnnotatedPinned = useMemo(() => {
    if (annotatedItemsCount === 0 || selectedAiClassLowSet.size === 0) {
      return false;
    }
    for (const it of allItems) {
      const dets =
        currentItem && it.id === currentItem.id
          ? draftDetections
          : it.detections;
      const matching = dets.filter((d) =>
        selectedAiClassLowSet.has((d.label.trim() || "object").toLowerCase())
      );
      if (matching.length === 0) continue;
      const pinnedIds = activeVitRefsByItem[it.id];
      if (!pinnedIds || pinnedIds.length === 0) return false;
      const pinnedSet = new Set(pinnedIds);
      if (!matching.every((d) => pinnedSet.has(d.id))) return false;
    }
    return true;
  }, [
    activeVitRefsByItem,
    allItems,
    annotatedItemsCount,
    currentItem,
    draftDetections,
    selectedAiClassLowSet,
  ]);

  const visualHits = useMemo(
    () => (currentItem ? stagedVisualHitsByItem[currentItem.id] ?? [] : []),
    [currentItem, stagedVisualHitsByItem]
  );

  const { totalStagedOverlayItems, totalStagedOverlayBoxes } = useMemo(() => {
    let items = 0;
    let boxes = 0;
    for (const dets of Object.values(stagedVisualHitsByItem)) {
      if (dets.length > 0) {
        items += 1;
        boxes += dets.length;
      }
    }
    return { totalStagedOverlayItems: items, totalStagedOverlayBoxes: boxes };
  }, [stagedVisualHitsByItem]);

  const nextStagedOverlayItemId = useMemo(() => {
    if (totalStagedOverlayItems === 0 || queueItems.length === 0) return null;
    const curIdx = queueItems.findIndex((it) => it.id === activeItemId);
    const start = curIdx >= 0 ? curIdx + 1 : 0;
    for (let offset = 0; offset < queueItems.length; offset += 1) {
      const cand = queueItems[(start + offset) % queueItems.length];
      if (
        cand.id !== activeItemId &&
        (stagedVisualHitsByItem[cand.id]?.length ?? 0) > 0
      ) {
        return cand.id;
      }
    }
    return null;
  }, [activeItemId, queueItems, stagedVisualHitsByItem, totalStagedOverlayItems]);

  const clearVisualSessionLocal = useCallback(() => {
    visualBatchAbortRef.current?.abort();
    visualBatchAbortRef.current = null;
    visualSearchAbortRef.current?.abort();
    visualSearchAbortRef.current = null;
    skipAutoSearchItemRef.current = "";
    setVisualDrawMode(null);
    setVisualQueryId("");
    setStagedVisualHitsByItem({});
    setDismissedVisualItemIds(new Set());
    setVisualExemplars([]);
    setVisualExemplarsRedo([]);
    setCanUndoVisualEx(false);
    setCanRedoVisualEx(false);
    setVitClassSummaries([]);
    setVisualTopScore(null);
    setVisualTopObjectness(null);
    setVisualPosCount(0);
    setVisualNegCount(0);
    setVisualLastLatency(null);
    setVisualMatchedIou(null);
    setVisualError(null);
    setVisualBatchProgress(null);
  }, []);

  // Reset visual query & fetch stream cleanly on backend restart / offline or unmount
  const prevEpochRef = useRef(sessionEpoch);
  useEffect(() => {
    if (sessionEpoch === 0 || prevEpochRef.current === sessionEpoch) return;
    prevEpochRef.current = sessionEpoch;
    manualFetchSseRef.current?.();
    manualFetchSseRef.current = null;
    setManualFetchBusy(false);
    setManualFetchProgress(null);
    setManualFetchError(null);
    setFetchModalOpen(false);
    clearVisualSessionLocal();
  }, [clearVisualSessionLocal, sessionEpoch]);

  useEffect(() => {
    return () => {
      manualFetchSseRef.current?.();
      visualBatchAbortRef.current?.abort();
      visualSearchAbortRef.current?.abort();
    };
  }, []);

  const buildVitRefsFromMap = useCallback(
    async (
      refsMap: Record<string, string[]>,
      searchCurrent: boolean,
      owlModelOverride?: OwlModelId,
      targetClassesOverride?: string[]
    ): Promise<string | null> => {
      if (!jobDetail || backendStatus !== "online") return null;
      const effOwlModelId = owlModelOverride ?? owlModelId;
      const effClasses = targetClassesOverride ?? selectedAiClasses;
      const effClassLowSet = new Set(
        effClasses.map((c) => c.trim().toLowerCase()).filter(Boolean)
      );
      if (availableClassOptions.length > 0 && effClassLowSet.size === 0) {
        setVisualError("Select at least one target class to index or search.");
        return null;
      }

      const references: {
        itemId: string;
        detectionIds: string[];
        inlineDetections?: AutoDetection[];
      }[] = [];

      for (const [iid, detIds] of Object.entries(refsMap)) {
        if (!detIds || detIds.length === 0) continue;
        if (currentItem && iid === currentItem.id) {
          const validIds = new Set(
            draftDetections
              .filter(
                (d) =>
                  effClassLowSet.size === 0 ||
                  effClassLowSet.has((d.label.trim() || "object").toLowerCase())
              )
              .map((d) => d.id)
          );
          const filtered = detIds.filter((id) => validIds.has(id));
          if (filtered.length > 0) {
            references.push({
              itemId: iid,
              detectionIds: filtered,
              inlineDetections: draftDetections,
            });
          }
        } else {
          const it = itemsById.get(iid);
          if (!it) continue;
          const validIds = new Set(
            it.detections
              .filter(
                (d) =>
                  effClassLowSet.size === 0 ||
                  effClassLowSet.has((d.label.trim() || "object").toLowerCase())
              )
              .map((d) => d.id)
          );
          const filtered = detIds.filter((id) => validIds.has(id));
          if (filtered.length > 0) {
            references.push({ itemId: iid, detectionIds: filtered });
          }
        }
      }

      if (references.length === 0) {
        setVisualError(
          "No pinned boxes match the selected target classes. Pin matching boxes or select more classes."
        );
        return null;
      }

      setVitIndexing(true);
      setVisualError(null);
      try {
        if (currentItem && dirty && (refsMap[currentItem.id] || []).length > 0) {
          await persistItem(
            currentItem,
            draftDetections,
            currentItem.reviewStatus,
            currentItem.rejectReason ?? null
          );
        }
        const res = await visualEmbedBuildRefsApi(backendUrl, jobDetail.id, {
          queryId: visualQueryId || null,
          replace: true,
          owlModelId: effOwlModelId,
          targetClasses:
            availableClassOptions.length > 0 || effClasses.length > 0
              ? effClasses
              : undefined,
          references,
          searchItemId: searchCurrent && currentItem ? currentItem.id : null,
          threshold: visualThreshold,
          minObjectness: visualMinObjectness,
          nmsIou: 0.3,
          useSamMasks: visualUseSamMasks,
          samRefinerId: smartSamRefinerId,
          complexity: smartComplexity,
        });
        skipAutoSearchItemRef.current = currentItem?.id || "";
        setVisualQueryId(res.queryId);
        setVitClassSummaries(res.classes);
        setVisualPosCount(res.totalExemplars);
        setVisualNegCount(res.totalNegatives);
        setVitBankDirty(false);
        setVitBankFeedback(
          `Active & Indexed: ${res.totalExemplars} box(es) across ${res.totalRefImages} frame(s) (${res.latencySec.toFixed(2)}s)`
        );
        setVisualLastLatency(res.latencySec);
        setVisualExemplarsRedo([]);
        setCanUndoVisualEx(false);
        setCanRedoVisualEx(false);
        if (searchCurrent && currentItem) {
          setShowVisualOverlay(true);
          setDismissedVisualItemIds((prev) => {
            if (!prev.has(currentItem.id)) return prev;
            const next = new Set(prev);
            next.delete(currentItem.id);
            return next;
          });
          setStagedVisualHitsByItem((prev) => {
            const copy = { ...prev };
            if (res.detections.length > 0) copy[currentItem.id] = res.detections;
            else delete copy[currentItem.id];
            return copy;
          });
          setVisualTopScore(res.topScore);
          setVisualTopObjectness(res.topObjectness ?? null);
          setSmartStatusMsg(
            `${res.engineUsed} · ${res.detections.length} match(es) from ${res.totalExemplars} ViT ref(s) in ${res.latencySec.toFixed(2)}s`
          );
        } else {
          setSmartStatusMsg(
            `Indexed ${res.totalExemplars} ViT ref(s) across ${res.totalRefImages} frame(s)`
          );
        }
        return res.queryId;
      } catch (err) {
        setVitBankFeedback(null);
        setVisualError(
          err instanceof Error ? err.message : "Failed to index ViT Reference embeddings."
        );
        return null;
      } finally {
        setVitIndexing(false);
      }
    },
    [
      availableClassOptions.length,
      backendStatus,
      backendUrl,
      currentItem,
      dirty,
      draftDetections,
      jobDetail,
      itemsById,
      owlModelId,
      persistItem,
      selectedAiClasses,
      smartComplexity,
      smartSamRefinerId,
      visualMinObjectness,
      visualQueryId,
      visualThreshold,
      visualUseSamMasks,
    ]
  );

  const markAiClassesUpdated = useCallback(
    (nextClasses: string[], feedbackMsg: string) => {
      setSelectedAiClasses(nextClasses);
      setStagedVisualHitsByItem({});
      setVisualTopScore(null);
      setVisualTopObjectness(null);
      setVisualError(null);
      if (Object.keys(vitRefsByItem).length > 0 || Boolean(visualQueryId)) {
        setVitBankDirty(true);
        setVitBankFeedback(feedbackMsg);
      }
    },
    [visualQueryId, vitRefsByItem]
  );

  const handleToggleAiClass = useCallback(
    (className: string) => {
      const clean = className.trim();
      if (!clean) return;
      const low = clean.toLowerCase();
      const exists = selectedAiClasses.some((c) => c.toLowerCase() === low);
      const next = exists
        ? selectedAiClasses.filter((c) => c.toLowerCase() !== low)
        : [...selectedAiClasses, clean];
      markAiClassesUpdated(
        next,
        next.length === 0
          ? "No target classes selected — select 1+ class above"
          : `Target classes updated (${next.length} selected) — click Index Now or Detect`
      );
    },
    [markAiClassesUpdated, selectedAiClasses]
  );

  const handleSelectOnlyAiClass = useCallback(
    (className: string) => {
      const clean = className.trim();
      if (!clean) return;
      markAiClassesUpdated(
        [clean],
        `Target class set to "${clean}" only — click Index Now or Detect`
      );
    },
    [markAiClassesUpdated]
  );

  const handleSelectAllAiClasses = useCallback(() => {
    const allNames = availableClassOptions.map((o) => o.name);
    markAiClassesUpdated(
      allNames,
      `All ${allNames.length} classes selected — click Index Now or Detect`
    );
  }, [availableClassOptions, markAiClassesUpdated]);

  const handleClearAiClasses = useCallback(() => {
    markAiClassesUpdated(
      [],
      "No target classes selected — select at least 1 class to index or search"
    );
  }, [markAiClassesUpdated]);

  const ensureVitRefBankSynced = useCallback(
    (searchCurrent: boolean, owlModelOverride?: OwlModelId) =>
      buildVitRefsFromMap(activeVitRefsByItem, searchCurrent, owlModelOverride),
    [activeVitRefsByItem, buildVitRefsFromMap]
  );

  const handleToggleVitRefDet = useCallback(
    (detId: string) => {
      if (!currentItem || !detId) return;
      const targetDet = draftDetections.find((d) => d.id === detId);
      const targetLabel = (targetDet?.label || "object").trim() || "object";
      let added = false;
      setVitRefsByItem((prev) => {
        const cur = prev[currentItem.id] || [];
        const exists = cur.includes(detId);
        const nextList = exists
          ? cur.filter((id) => id !== detId)
          : [...cur, detId];
        added = !exists;
        const copy = { ...prev };
        if (nextList.length > 0) copy[currentItem.id] = nextList;
        else delete copy[currentItem.id];
        return copy;
      });
      if (
        added &&
        !selectedAiClasses.some(
          (c) => c.toLowerCase() === targetLabel.toLowerCase()
        )
      ) {
        setSelectedAiClasses((prev) => [...prev, targetLabel]);
      }
      setVitBankDirty(true);
      setVitBankFeedback(
        added ? "Pinned box — click Index Now or Detect" : "Unpinned box — re-index needed"
      );
      if (added) {
        setFindOpen(true);
        setFindMode("vit_ref");
      }
    },
    [currentItem, draftDetections, selectedAiClasses]
  );

  const handleTogglePinCurrentFrame = useCallback(() => {
    if (!currentItem) return;
    if (isCurrentFramePinned) {
      const removeSet = new Set(currentMatchingDetIds);
      const nextMap: Record<string, string[]> = {};
      for (const [iid, ids] of Object.entries(vitRefsByItem)) {
        if (iid === currentItem.id) {
          const kept = ids.filter((id) => !removeSet.has(id));
          if (kept.length > 0) nextMap[iid] = kept;
        } else if (ids.length > 0) {
          nextMap[iid] = ids;
        }
      }
      const remaining = Object.keys(nextMap).filter(
        (k) => (nextMap[k] || []).length > 0
      );
      if (remaining.length === 0) {
        setVitRefsByItem({});
        setVitBankDirty(false);
        setVitClassSummaries([]);
        setVitBankFeedback("Unpinned frame — ViT Reference Bank cleared");
        const prevQid = visualQueryId;
        clearVisualSessionLocal();
        if (jobDetail?.id && prevQid) {
          void visualEmbedResetApi(backendUrl, jobDetail.id, prevQid).catch(
            () => {}
          );
        }
        return;
      }
      setVitRefsByItem(nextMap);
      setVitBankDirty(true);
      setVitBankFeedback("Frame unpinned — re-indexing remaining refs...");
      if (backendStatus === "online") {
        void buildVitRefsFromMap(nextMap, false);
      }
      return;
    }
    if (currentMatchingDetIds.length === 0) return;
    const mergedForCur = Array.from(
      new Set([...(vitRefsByItem[currentItem.id] || []), ...currentMatchingDetIds])
    );
    const nextMap: Record<string, string[]> = {
      ...vitRefsByItem,
      [currentItem.id]: mergedForCur,
    };
    setVitRefsByItem(nextMap);
    setVitBankDirty(true);
    setFindOpen(true);
    setFindMode("vit_ref");
    setVitBankFeedback(
      `Pinned frame #${currentItem.seqIndex} (+${currentMatchingDetIds.length} matching box) — indexing...`
    );
    if (backendStatus === "online") {
      void buildVitRefsFromMap(nextMap, false);
    }
  }, [
    backendStatus,
    backendUrl,
    buildVitRefsFromMap,
    clearVisualSessionLocal,
    currentItem,
    currentMatchingDetIds,
    isCurrentFramePinned,
    jobDetail?.id,
    visualQueryId,
    vitRefsByItem,
  ]);

  const handleUnpinVitRefItem = useCallback(
    (targetItemId: string) => {
      const nextMap = { ...vitRefsByItem };
      delete nextMap[targetItemId];
      const remaining = Object.keys(nextMap).filter(
        (k) => (nextMap[k] || []).length > 0
      );
      if (remaining.length === 0) {
        setVitRefsByItem({});
        setVitBankDirty(false);
        setVitClassSummaries([]);
        setVitBankFeedback("Cleared all pinned ViT references");
        const prevQid = visualQueryId;
        clearVisualSessionLocal();
        if (jobDetail?.id && prevQid) {
          void visualEmbedResetApi(backendUrl, jobDetail.id, prevQid).catch(
            () => {}
          );
        }
        return;
      }
      setVitRefsByItem(nextMap);
      setVitBankDirty(true);
      setVitBankFeedback("Removed frame — re-indexing remaining refs...");
      if (backendStatus === "online") {
        void buildVitRefsFromMap(nextMap, false);
      }
    },
    [
      backendStatus,
      backendUrl,
      buildVitRefsFromMap,
      clearVisualSessionLocal,
      jobDetail?.id,
      visualQueryId,
      vitRefsByItem,
    ]
  );

  const handleUseAllAnnotatedRefs = useCallback(() => {
    if (!allItems.length || selectedAiClassLowSet.size === 0) return;
    const nextMap: Record<string, string[]> = {};
    for (const it of allItems) {
      const dets =
        currentItem && it.id === currentItem.id ? draftDetections : it.detections;
      const matchedIds = dets
        .filter((d) =>
          selectedAiClassLowSet.has((d.label.trim() || "object").toLowerCase())
        )
        .map((d) => d.id);
      if (matchedIds.length > 0) {
        nextMap[it.id] = matchedIds;
      }
    }
    if (Object.keys(nextMap).length === 0) return;
    setVitRefsByItem(nextMap);
    setVitBankDirty(true);
    setFindOpen(true);
    setFindMode("vit_ref");
    const clsSummary =
      selectedAiClasses.length === availableClassOptions.length
        ? "all annotated frames"
        : `${selectedAiClasses.join(", ")} refs`;
    setVitBankFeedback(`Indexing ${clsSummary}...`);
    void buildVitRefsFromMap(nextMap, visualScope === "same_image");
  }, [
    allItems,
    availableClassOptions.length,
    buildVitRefsFromMap,
    currentItem,
    draftDetections,
    selectedAiClassLowSet,
    selectedAiClasses,
    visualScope,
  ]);

  const handleIndexVitRefsNow = useCallback(() => {
    void ensureVitRefBankSynced(false);
  }, [ensureVitRefBankSynced]);

  const handleJumpToNextStagedOverlay = useCallback(() => {
    if (nextStagedOverlayItemId) {
      void navigateToItem(nextStagedOverlayItemId);
    }
  }, [navigateToItem, nextStagedOverlayItemId]);

  const runVisualSearchCurrentItem = useCallback(
    async (qidOverride?: string, owlModelOverride?: OwlModelId) => {
      const qid = qidOverride ?? visualQueryId;
      const effOwlModelId = owlModelOverride ?? owlModelId;
      if (!jobDetail || !currentItem || !qid) return;
      if (availableClassOptions.length > 0 && selectedAiClasses.length === 0) {
        setVisualError("Select at least one target class first.");
        return;
      }
      visualSearchAbortRef.current?.abort();
      const ctrl = new AbortController();
      visualSearchAbortRef.current = ctrl;
      setVisualBusy(true);
      setVisualError(null);
      try {
        const isMulti =
          vitClassBadges.length > 1 || vitClassSummaries.length > 1;
        const hasClassFilter =
          availableClassOptions.length > 0 || selectedAiClasses.length > 0;
        const res = await visualEmbedSearchApi(
          backendUrl,
          jobDetail.id,
          currentItem.id,
          {
            queryId: qid,
            label:
              refBoxCount > 0 || isMulti || hasClassFilter
                ? undefined
                : activeClass.trim() || undefined,
            targetClasses: hasClassFilter ? selectedAiClasses : undefined,
            owlModelId: effOwlModelId,
            threshold: visualThreshold,
            minObjectness: visualMinObjectness,
            nmsIou: 0.3,
            useSamMasks: visualUseSamMasks,
            samRefinerId: smartSamRefinerId,
            complexity: smartComplexity,
          },
          ctrl.signal
        );
        if (ctrl.signal.aborted) return;
        setShowVisualOverlay(true);
        setDismissedVisualItemIds((prev) => {
          if (!prev.has(currentItem.id)) return prev;
          const next = new Set(prev);
          next.delete(currentItem.id);
          return next;
        });
        setStagedVisualHitsByItem((prev) => {
          const copy = { ...prev };
          if (res.detections.length > 0) copy[currentItem.id] = res.detections;
          else delete copy[currentItem.id];
          return copy;
        });
        setVisualTopScore(res.topScore);
        setVisualTopObjectness(res.topObjectness ?? null);
        setVisualPosCount(res.positivesCount);
        setVisualNegCount(res.negativesCount);
        if (typeof res.canUndo === "boolean") setCanUndoVisualEx(res.canUndo);
        if (typeof res.canRedo === "boolean") setCanRedoVisualEx(res.canRedo);
        if (res.classes) setVitClassSummaries(res.classes);
        setVisualLastLatency(res.latencySec);
        setSmartStatusMsg(
          `${res.engineUsed} · ${res.detections.length} hit(s) in ${res.latencySec.toFixed(2)}s`
        );
      } catch (err) {
        if (ctrl.signal.aborted) return;
        if (err instanceof AutoAnnotateApiError && err.status === 404) {
          setVisualQueryId("");
          setStagedVisualHitsByItem({});
          setVisualExemplars([]);
          setVisualExemplarsRedo([]);
          setCanUndoVisualEx(false);
          setCanRedoVisualEx(false);
          setVitClassSummaries([]);
          setVisualPosCount(0);
          setVisualNegCount(0);
          if (refBoxCount > 0) setVitBankDirty(true);
        }
        setVisualError(
          err instanceof Error ? err.message : "Visual embed search failed."
        );
      } finally {
        if (!ctrl.signal.aborted) {
          setVisualBusy(false);
        }
      }
    },
    [
      activeClass,
      backendUrl,
      availableClassOptions.length,
      currentItem,
      jobDetail,
      owlModelId,
      refBoxCount,
      selectedAiClasses,
      smartComplexity,
      smartSamRefinerId,
      visualMinObjectness,
      visualQueryId,
      visualThreshold,
      visualUseSamMasks,
      vitClassBadges.length,
      vitClassSummaries.length,
    ]
  );

  // Debounced live re-search when slider thresholds / SAM mask toggle change
  useEffect(() => {
    if (
      !findOpen ||
      findMode !== "vit_ref" ||
      !visualQueryId ||
      vitBankDirty ||
      !showVisualOverlay ||
      !jobDetail ||
      !currentItem ||
      dismissedVisualItemIds.has(currentItem.id) ||
      visualBatchProgress?.running
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void runVisualSearchCurrentItem(visualQueryId);
    }, 90);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    smartComplexity,
    smartSamRefinerId,
    visualMinObjectness,
    visualThreshold,
    visualUseSamMasks,
  ]);

  // Auto-search when switching images ONLY when enabled and frame was not dismissed/staged
  useEffect(() => {
    if (
      !findOpen ||
      findMode !== "vit_ref" ||
      !visualAutoSearchOnSwitch ||
      !showVisualOverlay ||
      !jobDetail ||
      !currentItem ||
      visualBatchProgress?.running
    ) {
      return;
    }
    if (skipAutoSearchItemRef.current === currentItem.id) {
      skipAutoSearchItemRef.current = "";
      return;
    }
    if (
      dismissedVisualItemIds.has(currentItem.id) ||
      (stagedVisualHitsByItem[currentItem.id]?.length ?? 0) > 0
    ) {
      return;
    }
    if (refBoxCount > 0 && (vitBankDirty || !visualQueryId)) {
      void ensureVitRefBankSynced(true);
    } else if (visualQueryId) {
      void runVisualSearchCurrentItem(visualQueryId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentItem?.id, findMode, findOpen]);

  const handleResetVisualQuery = useCallback(
    async (preserveDirty = true) => {
      const prevQid = visualQueryId;
      const jid = jobDetail?.id;
      clearVisualSessionLocal();
      if (preserveDirty && refBoxCount > 0) setVitBankDirty(true);
      setVitBankFeedback(null);
      setSmartStatusMsg("Visual query reset");
      if (jid && prevQid) {
        try {
          await visualEmbedResetApi(backendUrl, jid, prevQid);
        } catch {
          // Ignore reset failures if server already restarted
        }
      }
    },
    [
      backendUrl,
      clearVisualSessionLocal,
      jobDetail?.id,
      refBoxCount,
      visualQueryId,
    ]
  );

  const handleClearVitRefs = useCallback(() => {
    setVitRefsByItem({});
    setVitBankDirty(false);
    setVitClassSummaries([]);
    setVitBankFeedback("Cleared all pinned ViT references");
    const prevQid = visualQueryId;
    clearVisualSessionLocal();
    setSmartStatusMsg("Cleared all pinned ViT references");
    if (jobDetail?.id && prevQid) {
      void visualEmbedResetApi(backendUrl, jobDetail.id, prevQid).catch(
        () => {}
      );
    }
  }, [backendUrl, clearVisualSessionLocal, jobDetail?.id, visualQueryId]);

  const handleDetectCurrentImage = useCallback(async () => {
    if (!jobDetail || !currentItem) return;
    setShowVisualOverlay(true);
    setDismissedVisualItemIds((prev) => {
      if (!prev.has(currentItem.id)) return prev;
      const next = new Set(prev);
      next.delete(currentItem.id);
      return next;
    });
    if (refBoxCount > 0 && (vitBankDirty || !visualQueryId)) {
      await ensureVitRefBankSynced(true);
      return;
    }
    if (visualQueryId) {
      await runVisualSearchCurrentItem(visualQueryId);
    }
  }, [
    currentItem,
    ensureVitRefBankSynced,
    jobDetail,
    refBoxCount,
    runVisualSearchCurrentItem,
    visualQueryId,
    vitBankDirty,
  ]);

  const handleVisualEmbedAdd = useCallback(
    async (params: {
      bbox?: [number, number, number, number];
      point?: [number, number];
      polarity: "positive" | "negative";
    }) => {
      if (!jobDetail || !currentItem || visualBatchProgress?.running) return;
      visualSearchAbortRef.current?.abort();
      setVisualBusy(true);
      setVisualError(null);
      try {
        const boxLabel = activeClass.trim() || "object";
        let effTargetClasses = selectedAiClasses;
        if (
          params.polarity === "positive" &&
          !selectedAiClasses.some(
            (c) => c.toLowerCase() === boxLabel.toLowerCase()
          )
        ) {
          effTargetClasses = [...selectedAiClasses, boxLabel];
          setSelectedAiClasses(effTargetClasses);
        }
        let activeQid = visualQueryId || null;
        if (refBoxCount > 0 && (vitBankDirty || !activeQid)) {
          activeQid = await buildVitRefsFromMap(
            activeVitRefsByItem,
            false,
            undefined,
            effTargetClasses
          );
        }
        const res = await visualEmbedAddApi(
          backendUrl,
          jobDetail.id,
          currentItem.id,
          {
            queryId: activeQid,
            bbox: params.bbox ?? null,
            point: params.point ?? null,
            polarity: params.polarity,
            label: boxLabel,
            targetClasses:
              availableClassOptions.length > 0 || effTargetClasses.length > 0
                ? effTargetClasses
                : undefined,
            owlModelId,
            threshold: visualThreshold,
            minObjectness: visualMinObjectness,
            nmsIou: 0.3,
            useSamMasks: visualUseSamMasks,
            samRefinerId: smartSamRefinerId,
            complexity: smartComplexity,
          }
        );
        skipAutoSearchItemRef.current = currentItem.id;
        setShowVisualOverlay(true);
        setDismissedVisualItemIds((prev) => {
          if (!prev.has(currentItem.id)) return prev;
          const next = new Set(prev);
          next.delete(currentItem.id);
          return next;
        });
        setVisualQueryId(res.queryId);
        setStagedVisualHitsByItem((prev) => {
          const copy = { ...prev };
          if (res.detections.length > 0) copy[currentItem.id] = res.detections;
          else delete copy[currentItem.id];
          return copy;
        });
        setVisualTopScore(res.topScore);
        setVisualTopObjectness(res.topObjectness ?? null);
        setVisualPosCount(res.positivesCount);
        setVisualNegCount(res.negativesCount);
        setVisualExemplarsRedo([]);
        setCanUndoVisualEx(res.canUndo ?? true);
        setCanRedoVisualEx(Boolean(res.canRedo));
        if (res.classes) setVitClassSummaries(res.classes);
        setVisualLastLatency(res.latencySec);
        setVisualMatchedIou(res.matchedIou ?? null);
        const exBox = params.bbox ?? res.matchedBox;
        if (exBox && exBox.length === 4) {
          setVisualExemplars((prev) => [
            ...prev,
            {
              id:
                res.exemplarId ||
                `ex-${Date.now().toString(36)}-${Math.random()
                  .toString(36)
                  .slice(2, 6)}`,
              itemId: currentItem.id,
              bbox_2d: [...exBox],
              matchedBox: res.matchedBox ?? null,
              polarity: params.polarity,
            },
          ]);
        }
        setSmartStatusMsg(
          `${res.engineUsed} · ${res.detections.length} hit(s) · IoU ${(
            res.matchedIou ?? 0
          ).toFixed(2)}`
        );
      } catch (err) {
        setVisualError(
          err instanceof Error ? err.message : "Failed to extract visual embedding."
        );
      } finally {
        setVisualBusy(false);
      }
    },
    [
      activeClass,
      activeVitRefsByItem,
      availableClassOptions.length,
      backendUrl,
      buildVitRefsFromMap,
      currentItem,
      jobDetail,
      owlModelId,
      refBoxCount,
      selectedAiClasses,
      smartComplexity,
      smartSamRefinerId,
      visualBatchProgress?.running,
      visualMinObjectness,
      visualQueryId,
      visualThreshold,
      visualUseSamMasks,
      vitBankDirty,
    ]
  );

  const handleUndoRedoVisualExemplar = useCallback(
    async (action: "undo" | "redo") => {
      if (!jobDetail || !visualQueryId || visualBatchProgress?.running) return;
      visualSearchAbortRef.current?.abort();
      setVisualBusy(true);
      setVisualError(null);
      try {
        const isMulti =
          vitClassBadges.length > 1 || vitClassSummaries.length > 1;
        const hasClassFilter =
          availableClassOptions.length > 0 || selectedAiClasses.length > 0;
        const res = await visualEmbedUndoRedoApi(backendUrl, jobDetail.id, {
          queryId: visualQueryId,
          action,
          searchItemId: currentItem?.id ?? null,
          label:
            refBoxCount > 0 || isMulti || hasClassFilter
              ? undefined
              : activeClass.trim() || undefined,
          targetClasses: hasClassFilter ? selectedAiClasses : undefined,
          owlModelId,
          threshold: visualThreshold,
          minObjectness: visualMinObjectness,
          nmsIou: 0.3,
          useSamMasks: visualUseSamMasks,
          samRefinerId: smartSamRefinerId,
          complexity: smartComplexity,
        });
        if (action === "undo") {
          setVisualExemplars((prev) => {
            if (prev.length === 0) return prev;
            const popped = prev[prev.length - 1];
            setVisualExemplarsRedo((r) => [...r, popped]);
            return prev.slice(0, -1);
          });
        } else {
          setVisualExemplarsRedo((prev) => {
            if (prev.length === 0) return prev;
            const popped = prev[prev.length - 1];
            setVisualExemplars((ex) => [...ex, popped]);
            return prev.slice(0, -1);
          });
        }
        setCanUndoVisualEx(Boolean(res.canUndo));
        setCanRedoVisualEx(Boolean(res.canRedo));
        setVisualPosCount(res.positivesCount);
        setVisualNegCount(res.negativesCount);
        setVisualTopScore(res.topScore);
        setVisualTopObjectness(res.topObjectness ?? null);
        setVisualLastLatency(res.latencySec);
        if (res.classes) setVitClassSummaries(res.classes);
        if (currentItem) {
          setShowVisualOverlay(true);
          setDismissedVisualItemIds((prev) => {
            if (!prev.has(currentItem.id)) return prev;
            const next = new Set(prev);
            next.delete(currentItem.id);
            return next;
          });
          setStagedVisualHitsByItem((prev) => {
            const copy = { ...prev };
            if (res.detections.length > 0) copy[currentItem.id] = res.detections;
            else delete copy[currentItem.id];
            return copy;
          });
        }
        setSmartStatusMsg(
          `${action === "undo" ? "Undid" : "Redid"} drawn exemplar · ${res.detections.length} match(es)`
        );
      } catch (err) {
        setVisualError(
          err instanceof Error ? err.message : `Failed to ${action} drawn exemplar.`
        );
      } finally {
        setVisualBusy(false);
      }
    },
    [
      activeClass,
      backendUrl,
      availableClassOptions.length,
      currentItem,
      jobDetail,
      owlModelId,
      refBoxCount,
      selectedAiClasses,
      smartComplexity,
      smartSamRefinerId,
      visualBatchProgress?.running,
      visualMinObjectness,
      visualQueryId,
      visualThreshold,
      visualUseSamMasks,
      vitClassBadges.length,
      vitClassSummaries.length,
    ]
  );

  const handleVisualEmbedDragBox = useCallback(
    (bbox: [number, number, number, number], polarity: "positive" | "negative") => {
      void handleVisualEmbedAdd({ bbox, polarity });
    },
    [handleVisualEmbedAdd]
  );

  const handleVisualEmbedClickPoint = useCallback(
    (point: [number, number], polarity: "positive" | "negative") => {
      void handleVisualEmbedAdd({ point, polarity });
    },
    [handleVisualEmbedAdd]
  );

  const handleCommitVisualHits = useCallback(() => {
    if (!currentItem || visualHits.length === 0) return;
    const fallbackLabel = activeClass.trim() || "object";
    const matchedClassLows = new Set(
      visualHits.map((h) => (h.label?.trim() || fallbackLabel).toLowerCase())
    );
    const nextList =
      visualMergeMode === "replace_class"
        ? draftDetections.filter(
            (d) => !matchedClassLows.has(d.label.trim().toLowerCase())
          )
        : [...draftDetections];

    let added = 0;
    for (const hit of visualHits) {
      const hitLabel = hit.label?.trim() || fallbackLabel;
      const hitLow = hitLabel.toLowerCase();
      const normalized: AutoDetection = {
        ...hit,
        id: `owlv-${Date.now().toString(36)}-${Math.random()
          .toString(36)
          .slice(2, 6)}-${added + 1}`,
        label: hitLabel,
      };
      const isDup =
        visualMergeMode !== "replace_class" &&
        nextList.some(
          (ex) =>
            ex.label.trim().toLowerCase() === hitLow &&
            boxIou1000(normalized.bbox_2d, ex.bbox_2d) >= 0.65
        );
      if (!isDup) {
        nextList.push(normalized);
        added += 1;
      }
    }
    if (added > 0 || visualMergeMode === "replace_class") {
      handleDetectionsChange(nextList, true);
      if (nextList.length > 0) {
        setSelectedDetId(nextList[nextList.length - 1].id);
      }
      setSmartStatusMsg(`Accepted ${added} overlay match(es)`);
    } else {
      setSmartStatusMsg("All matched boxes already present on frame");
    }
    setStagedVisualHitsByItem((prev) => {
      const copy = { ...prev };
      delete copy[currentItem.id];
      return copy;
    });
    setDismissedVisualItemIds((prev) => {
      const next = new Set(prev);
      next.add(currentItem.id);
      return next;
    });
  }, [
    activeClass,
    currentItem,
    draftDetections,
    handleDetectionsChange,
    visualHits,
    visualMergeMode,
  ]);

  const handleClearVisualHits = useCallback(() => {
    if (!currentItem) return;
    setStagedVisualHitsByItem((prev) => {
      if (!(currentItem.id in prev)) return prev;
      const copy = { ...prev };
      delete copy[currentItem.id];
      return copy;
    });
    setDismissedVisualItemIds((prev) => {
      const next = new Set(prev);
      next.add(currentItem.id);
      return next;
    });
  }, [currentItem]);

  const handleDismissSingleVisualHit = useCallback(
    (hitId: string) => {
      if (!currentItem) return;
      setStagedVisualHitsByItem((prev) => {
        const cur = prev[currentItem.id];
        if (!cur) return prev;
        const remaining = cur.filter((d) => d.id !== hitId);
        const copy = { ...prev };
        if (remaining.length > 0) {
          copy[currentItem.id] = remaining;
        } else {
          delete copy[currentItem.id];
          setDismissedVisualItemIds((p) => {
            const next = new Set(p);
            next.add(currentItem.id);
            return next;
          });
        }
        return copy;
      });
    },
    [currentItem]
  );

  const handleClearAllStagedOverlays = useCallback(() => {
    setStagedVisualHitsByItem({});
  }, []);

  const handleRunVisualBatch = useCallback(async () => {
    if (!jobDetail) return;
    if (visualBatchProgress?.running) {
      visualBatchAbortRef.current?.abort();
      visualBatchAbortRef.current = null;
      setVisualBatchProgress((prev) =>
        prev ? { ...prev, running: false } : null
      );
      return;
    }

    if (availableClassOptions.length > 0 && selectedAiClasses.length === 0) {
      setVisualError("Select at least one target class to search across batch.");
      return;
    }

    let effQueryId = visualQueryId;
    if (refBoxCount > 0 && (vitBankDirty || !effQueryId)) {
      const synced = await ensureVitRefBankSynced(false);
      if (!synced) return;
      effQueryId = synced;
    }
    if (!effQueryId) return;

    const ctrl = new AbortController();
    visualBatchAbortRef.current = ctrl;
    const targetLabel = activeClass.trim() || "object";
    const resolvedScope: "current_batch" | "all_batches" | "item_ids" =
      visualBatchTarget === "filtered_queue"
        ? "item_ids"
        : visualBatchTarget === "current_batch" && activeBatchId !== "all"
          ? "current_batch"
          : "all_batches";

    const excludeItemIds = vitSkipAnnotated
      ? Object.keys(activeVitRefsByItem).filter(
          (k) => (activeVitRefsByItem[k] || []).length > 0
        )
      : [];
    const excludeSet = new Set(excludeItemIds);
    const scopePool =
      resolvedScope === "item_ids"
        ? queueItems
        : resolvedScope === "current_batch"
          ? allItems.filter((it) => it.batchId === activeBatchId)
          : allItems;
    const filteredPool = scopePool.filter(
      (it) =>
        !excludeSet.has(it.id) &&
        (!vitSkipAnnotated || it.detections.length === 0)
    );
    const expectedTotal = Math.max(1, filteredPool.length);
    const hasClassFilter =
      availableClassOptions.length > 0 || selectedAiClasses.length > 0;
    const isMultiClass =
      refBoxCount > 0 || vitClassSummaries.length > 1 || hasClassFilter;

    setShowVisualOverlay(true);
    setVisualError(null);
    setVisualBatchProgress({
      running: true,
      done: 0,
      total: expectedTotal,
      totalAdded: 0,
      itemsWithMatches: 0,
      currentFilename: "Starting overlay scan...",
    });

    try {
      await visualEmbedSearchBatchStreamApi(
        backendUrl,
        jobDetail.id,
        {
          queryId: effQueryId,
          label: targetLabel,
          targetClasses: hasClassFilter ? selectedAiClasses : undefined,
          owlModelId,
          scope: resolvedScope,
          batchId: resolvedScope === "current_batch" ? activeBatchId : null,
          itemIds:
            resolvedScope === "item_ids" ? queueItems.map((it) => it.id) : [],
          threshold: visualThreshold,
          minObjectness: visualMinObjectness,
          nmsIou: 0.3,
          useSamMasks: visualUseSamMasks,
          samRefinerId: smartSamRefinerId,
          complexity: smartComplexity,
          mergeMode: visualMergeMode,
          dedupeIou: 0.65,
          skipAnnotated: vitSkipAnnotated,
          excludeItemIds,
          multiClass: isMultiClass,
          previewOnly: true,
        },
        (evt) => {
          if (evt.type === "progress") {
            const targetId = evt.itemId;
            if (targetId && Array.isArray(evt.detections)) {
              const staged = evt.detections;
              setStagedVisualHitsByItem((prev) => {
                const copy = { ...prev };
                if (staged.length > 0) copy[targetId] = staged;
                else delete copy[targetId];
                return copy;
              });
              if (staged.length > 0) {
                setDismissedVisualItemIds((prev) => {
                  if (!prev.has(targetId)) return prev;
                  const next = new Set(prev);
                  next.delete(targetId);
                  return next;
                });
              }
            }
            setVisualBatchProgress({
              running: true,
              done: evt.done,
              total: evt.total,
              totalAdded: evt.totalAdded,
              itemsWithMatches: evt.itemsWithMatches,
              currentFilename: evt.filename || "",
            });
          } else if (evt.type === "done") {
            setVisualBatchProgress({
              running: false,
              done: evt.done,
              total: evt.total,
              totalAdded: evt.totalAdded,
              itemsWithMatches: evt.itemsWithMatches,
              currentFilename: "Overlay scan complete",
            });
            setSmartStatusMsg(
              `Staged ${evt.totalAdded} overlay match(es) across ${evt.itemsWithMatches}/${evt.total} image(s) for acceptance`
            );
          }
        },
        ctrl.signal
      );
    } catch (err) {
      if (ctrl.signal.aborted) {
        setVisualBatchProgress((prev) =>
          prev
            ? { ...prev, running: false, currentFilename: "Stopped by user" }
            : null
        );
        return;
      }
      if (err instanceof AutoAnnotateApiError && err.status === 404) {
        setVisualQueryId("");
        setStagedVisualHitsByItem({});
        setVisualExemplars([]);
        setVisualExemplarsRedo([]);
        setCanUndoVisualEx(false);
        setCanRedoVisualEx(false);
        setVitClassSummaries([]);
        if (refBoxCount > 0) setVitBankDirty(true);
      }
      setVisualError(
        err instanceof Error ? err.message : "Batch overlay scan failed."
      );
      setVisualBatchProgress((prev) =>
        prev ? { ...prev, running: false } : null
      );
    } finally {
      if (visualBatchAbortRef.current === ctrl) {
        visualBatchAbortRef.current = null;
      }
    }
  }, [
    activeBatchId,
    activeClass,
    activeVitRefsByItem,
    allItems,
    availableClassOptions.length,
    backendUrl,
    ensureVitRefBankSynced,
    jobDetail,
    owlModelId,
    queueItems,
    refBoxCount,
    selectedAiClasses,
    smartComplexity,
    smartSamRefinerId,
    visualBatchProgress?.running,
    visualBatchTarget,
    visualMergeMode,
    visualMinObjectness,
    visualQueryId,
    visualThreshold,
    visualUseSamMasks,
    vitBankDirty,
    vitClassBadges.length,
    vitClassSummaries.length,
    vitSkipAnnotated,
  ]);

  const handleSmartHoverPreview = useCallback(
    async ({
      point,
      signal,
    }: {
      point: [number, number];
      signal?: AbortSignal;
    }): Promise<AutoDetection | null> => {
      if (!jobDetail || !currentItem) return null;
      try {
        const res = await smartSelectItemApi(
          backendUrl,
          jobDetail.id,
          currentItem.id,
          {
            points: [{ x: point[0], y: point[1], label: 1 }],
            bboxPrompt: null,
            label: activeClass.trim() || "object",
            candidateClasses: mergedClassColors.map((c) => c.name),
            outputMode: smartOutputMode,
            samRefinerId: smartSamRefinerId,
            engine: "sam",
            complexity: smartComplexity,
            preview: true,
          },
          signal
        );
        return res.detection ?? null;
      } catch {
        return null;
      }
    },
    [
      activeClass,
      backendUrl,
      currentItem,
      jobDetail,
      mergedClassColors,
      smartComplexity,
      smartOutputMode,
      smartSamRefinerId,
    ]
  );

  const handleCommitSmartDetection = useCallback(
    (rawDet: AutoDetection) => {
      const finalDet: AutoDetection = {
        ...rawDet,
        id: `sam-${Date.now().toString(36)}-${Math.random()
          .toString(36)
          .slice(2, 6)}`,
        label: activeClass.trim() || rawDet.label || "object",
        polygon:
          smartOutputMode === "bbox"
            ? null
            : rawDet.polygon
              ? rawDet.polygon.map((p) => [...p])
              : null,
      };
      handleDetectionsChange([...draftDetections, finalDet], true);
      setSelectedDetId(finalDet.id);
      setSmartStatusMsg("Added");
    },
    [activeClass, draftDetections, handleDetectionsChange, smartOutputMode]
  );

  const handleRunSmartSelect = useCallback(
    async ({
      points,
      bboxPrompt,
      commitImmediately,
    }: {
      points: SmartSelectPoint[];
      bboxPrompt?: number[] | null;
      commitImmediately: boolean;
    }): Promise<AutoDetection | null> => {
      if (!jobDetail || !currentItem) return null;
      setSmartBusy(true);
      setSmartStatusMsg(null);
      try {
        const res = await smartSelectItemApi(
          backendUrl,
          jobDetail.id,
          currentItem.id,
          {
            points,
            bboxPrompt: bboxPrompt ?? null,
            label: activeClass.trim() || "object",
            candidateClasses: mergedClassColors.map((c) => c.name),
            outputMode: commitImmediately ? smartOutputMode : "both",
            samRefinerId: smartSamRefinerId,
            engine: "sam",
            owlModelId,
            owlThreshold: 0.12,
            complexity: smartComplexity,
            preview: false,
          }
        );
        if (!res.detection) {
          setSmartStatusMsg(res.message || "No mask detected at point.");
          return null;
        }
        if (currentPrepareKey) {
          setSmartPreparedKey(currentPrepareKey);
        }
        setSmartStatusMsg(`${res.engineUsed} · ${res.latencySec.toFixed(2)}s`);
        if (commitImmediately) {
          const det: AutoDetection = {
            ...res.detection,
            label: activeClass.trim() || res.detection.label,
            polygon:
              smartOutputMode === "bbox" ? null : (res.detection.polygon ?? null),
          };
          handleDetectionsChange([...draftDetections, det], true);
          setSelectedDetId(det.id);
        }
        return res.detection;
      } catch (err) {
        setSmartStatusMsg(
          err instanceof Error ? err.message : "Smart Select failed."
        );
        return null;
      } finally {
        setSmartBusy(false);
      }
    },
    [
      activeClass,
      backendUrl,
      currentItem,
      currentPrepareKey,
      draftDetections,
      handleDetectionsChange,
      jobDetail,
      mergedClassColors,
      owlModelId,
      smartComplexity,
      smartOutputMode,
      smartSamRefinerId,
    ]
  );

  const handleOwlModelChange = useCallback(
    (nextModelId: OwlModelId) => {
      if (nextModelId === owlModelId) return;
      visualSearchAbortRef.current?.abort();
      setOwlModelId(nextModelId);
      setStagedVisualHitsByItem({});
      setVisualTopScore(null);
      setVisualTopObjectness(null);
      setVisualError(null);
      setFindMessage(null);
      const pickedLabel =
        owlModelOptions.find((o) => o.hfModelId === nextModelId)?.shortLabel ??
        nextModelId;
      if (refBoxCount > 0 && jobDetail) {
        setVitBankDirty(true);
        setVitBankFeedback(`Re-indexing ViT refs with ${pickedLabel}...`);
        void buildVitRefsFromMap(
          activeVitRefsByItem,
          visualScope === "same_image",
          nextModelId
        );
      } else if (visualQueryId && currentItem && jobDetail) {
        setSmartStatusMsg(`Re-indexing query with ${pickedLabel}...`);
        void runVisualSearchCurrentItem(visualQueryId, nextModelId);
      } else {
        setSmartStatusMsg(`OWLv2 backbone set to ${pickedLabel}`);
      }
    },
    [
      activeVitRefsByItem,
      buildVitRefsFromMap,
      currentItem,
      jobDetail,
      owlModelId,
      owlModelOptions,
      refBoxCount,
      runVisualSearchCurrentItem,
      visualQueryId,
      visualScope,
    ]
  );

  const handleRunFindObjects = async () => {
    if (!jobDetail || !currentItem || findBusy) return;
    setFindBusy(true);
    setFindMessage(null);
    const classesList = findClassesText
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const resolvedClasses =
      classesList.length > 0 ? classesList : mergedClassColors.map((c) => c.name);
    const detectorEngine: "owlv2" | "yoloe" =
      findMode === "yoloe" ? "yoloe" : "owlv2";
    try {
      const res = await findObjectsItemApi(backendUrl, jobDetail.id, currentItem.id, {
        classes: resolvedClasses,
        confThreshold: findConf,
        outputMode: findOutputMode,
        detector: detectorEngine,
        owlModelId: detectorEngine === "owlv2" ? owlModelId : undefined,
        samRefinerId: smartSamRefinerId,
      });
      if (res.detections.length > 0) {
        handleDetectionsChange([...draftDetections, ...res.detections], true);
        setSelectedDetId(res.detections[0].id);
      }
      setFindMessage(
        `Added ${res.detections.length} object(s) in ${res.latencySec.toFixed(2)}s (${res.engineUsed}).`
      );
    } catch (err) {
      setFindMessage(
        err instanceof Error ? err.message : "Find Objects AI request failed."
      );
    } finally {
      setFindBusy(false);
    }
  };

  const thumbRefs = useRef<Map<string, HTMLButtonElement | null>>(new Map());

  useEffect(() => {
    if (!activeItemId) return;
    const btn = thumbRefs.current.get(activeItemId);
    btn?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
      behavior: "smooth",
    });
  }, [activeItemId]);

  const addClassIfMissing = useCallback(
    (name: string): string => {
      const clean = name.trim();
      if (!clean) return activeClass;
      const existing = mergedClassColors.find(
        (c) => c.name.trim().toLowerCase() === clean.toLowerCase()
      );
      if (existing) {
        return existing.name;
      }
      setCustomClasses((prev) => [
        ...prev,
        {
          name: clean,
          color:
            DEFAULT_PALETTE[
              (mergedClassColors.length + prev.length) % DEFAULT_PALETTE.length
            ],
        },
      ]);
      return clean;
    },
    [activeClass, mergedClassColors]
  );

  const handleAddClass = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = newClassInput.trim();
    if (!clean) return;
    const canonical = addClassIfMissing(clean);
    setActiveClass(canonical);
    setNewClassInput("");
  };

  const handleUpdateDetLabel = useCallback(
    (detId: string, newLabel: string) => {
      const clean = newLabel.trim();
      if (!clean) return;
      handleDetectionsChange(
        draftDetections.map((d) =>
          d.id === detId ? { ...d, label: clean } : d
        ),
        true
      );
      setActiveClass(clean);
    },
    [draftDetections, handleDetectionsChange]
  );

  const handleAddClassAndAssign = useCallback(
    (detId: string, newClassName: string) => {
      const clean = newClassName.trim();
      if (!clean) return;
      const canonical = addClassIfMissing(clean);
      handleDetectionsChange(
        draftDetections.map((d) =>
          d.id === detId ? { ...d, label: canonical } : d
        ),
        true
      );
      setActiveClass(canonical);
    },
    [addClassIfMissing, draftDetections, handleDetectionsChange]
  );

  // Global Studio Keyboard Shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        Boolean(target?.isContentEditable);

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        handleRedo();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void handleSaveAndAccept();
        return;
      }
      if (inInput || e.metaKey || e.ctrlKey || e.altKey) return;

      const k = e.key.toLowerCase();
      if (k === "d" || k === "v" || k === "h") {
        e.preventDefault();
        setVisualDrawMode(null);
        setTool("drag");
      } else if (k === "b") {
        e.preventDefault();
        setVisualDrawMode(null);
        setTool("bbox");
      } else if (k === "p") {
        e.preventDefault();
        setVisualDrawMode(null);
        setTool("polygon");
      } else if (k === "s") {
        e.preventDefault();
        setVisualDrawMode(null);
        setTool("smart_select");
      } else if (k === "u") {
        e.preventDefault();
        handleUndo();
      } else if (k === "r") {
        e.preventDefault();
        handleRedo();
      } else if (k === "n") {
        e.preventDefault();
        void handleMarkNull();
      } else if (e.key === "ArrowLeft" || e.key === "[") {
        if (currentIndex > 0) {
          e.preventDefault();
          void navigateToItem(queueItems[currentIndex - 1].id);
        }
      } else if (e.key === "ArrowRight" || e.key === "]") {
        if (currentIndex >= 0 && currentIndex + 1 < queueItems.length) {
          e.preventDefault();
          void navigateToItem(queueItems[currentIndex + 1].id);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    currentIndex,
    handleMarkNull,
    handleRedo,
    handleSaveAndAccept,
    handleUndo,
    navigateToItem,
    queueItems,
  ]);

  const handleManualFetchDataset = async () => {
    if (backendStatus !== "online") {
      setManualFetchError("Backend server is offline. Start it first.");
      return;
    }
    if (!apiKey.trim() || !projectSlug.trim()) {
      setManualFetchError("Enter Roboflow Private API Key and Project Slug first.");
      return;
    }
    setManualFetchBusy(true);
    setManualFetchError(null);
    setManualFetchProgress({
      progressPct: 5,
      stage: "Connecting to Roboflow...",
      subtext: `Preparing ${maxImages === 0 ? "all" : maxImages} image(s) in ${batchSize}-image batches`,
    });
    manualFetchSseRef.current?.();
    manualFetchSseRef.current = null;

    const finishLoadedJob = async (loadedJob: AutoJobRecord) => {
      clearVisualSessionLocal();
      setSelectedJobId(loadedJob.id);
      setQueueFilter("all");
      setBatchFilter("all");
      setReasonFilter("all");
      setPinnedKey("");
      setPinnedItemIds([]);
      setActiveItemId("");
      setCustomClasses([]);
      setFindClassesText("");
      setFetchModalOpen(false);
      setManualFetchBusy(false);
      setManualFetchProgress(null);
      router.replace(`/annotate?job=${encodeURIComponent(loadedJob.id)}&filter=all`);
      await refreshJobs();
      await loadJob(loadedJob.id);
    };

    try {
      const job = await submitManualFetchJob(backendUrl, {
        apiKey: apiKey.trim(),
        projectSlug: projectSlug.trim(),
        workspaceSlug: workspaceSlug.trim(),
        fetchScope,
        annotationStatusFilter,
        sourceBatchId: fetchScope === "batch" ? sourceBatchId.trim() : "",
        sourceAnnotationJobId:
          fetchScope === "annotation_job" ? sourceAnnotationJobId.trim() : "",
        assigneeEmail: fetchScope === "annotation_job" ? assigneeEmail.trim() : "",
        maxImages,
        batchSize,
      });

      setJobs((prev) => [job, ...prev.filter((j) => j.id !== job.id)]);

      if (job.status === "completed") {
        await finishLoadedJob(job);
        return;
      }

      if (job.status === "failed") {
        setManualFetchBusy(false);
        setManualFetchProgress(null);
        setManualFetchError(job.errorMessage || "Roboflow fetch failed.");
        return;
      }

      manualFetchSseRef.current = streamAutoJobProgress(
        backendUrl,
        job.id,
        (evt: AutoJobProgressEvent) => {
          setManualFetchProgress({
            progressPct: evt.progressPct ?? 10,
            stage: evt.stage || "Fetching Roboflow dataset...",
            subtext: evt.subtext || "",
          });
          if (evt.job) {
            const updatedJob = evt.job;
            setJobs((prev) => [updatedJob, ...prev.filter((j) => j.id !== updatedJob.id)]);
          }
          if (evt.status === "completed") {
            manualFetchSseRef.current?.();
            manualFetchSseRef.current = null;
            void finishLoadedJob(evt.job ?? job);
          } else if (evt.status === "failed" || evt.status === "cancelled") {
            manualFetchSseRef.current?.();
            manualFetchSseRef.current = null;
            setManualFetchBusy(false);
            setManualFetchProgress(null);
            setManualFetchError(
              evt.subtext || evt.job?.errorMessage || "Roboflow fetch failed."
            );
          }
        },
        (errMsg) => {
          manualFetchSseRef.current?.();
          manualFetchSseRef.current = null;
          setManualFetchBusy(false);
          setManualFetchProgress(null);
          setManualFetchError(errMsg || "Progress stream disconnected while fetching dataset.");
        }
      );
    } catch (err) {
      setManualFetchBusy(false);
      setManualFetchProgress(null);
      setManualFetchError(
        err instanceof Error ? err.message : "Failed to fetch dataset from Roboflow."
      );
    }
  };

  const pullDatasetFooter = (
    <div className="mt-4 space-y-3 border-t border-[#222226] pt-4">
      <p className="text-[11px] leading-relaxed text-[#A1A1AA]">
        Downloads images directly in{" "}
        <span className="font-mono-tech text-white">{batchSize}-image batches</span>{" "}
        without running VLM auto-inference, so you can annotate right away using BBox,
        Polygon, SAM 2.1, or OWL-ViT Visual Embed.
      </p>

      {manualFetchProgress && (
        <div className="rounded-lg border border-[#303035] bg-[#0A0A0C] px-3 py-2.5">
          <div className="flex items-center justify-between gap-2 text-[11px] text-[#E4E4E7]">
            <span className="truncate font-medium">{manualFetchProgress.stage}</span>
            <span className="font-mono-tech text-[#F2853F]">
              {manualFetchProgress.progressPct}%
            </span>
          </div>
          {manualFetchProgress.subtext && (
            <p className="font-mono-tech mt-0.5 truncate text-[10px] text-[#8A8A8E]">
              {manualFetchProgress.subtext}
            </p>
          )}
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#222226]">
            <div
              className="h-full rounded-full bg-[#F2853F] transition-all"
              style={{
                width: `${Math.max(4, Math.min(100, manualFetchProgress.progressPct))}%`,
              }}
            />
          </div>
        </div>
      )}

      {manualFetchError && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/15 px-3 py-2 text-xs text-rose-200">
          {manualFetchError}
        </div>
      )}

      <button
        type="button"
        disabled={
          manualFetchBusy ||
          backendStatus !== "online" ||
          !apiKey.trim() ||
          !projectSlug.trim()
        }
        onClick={() => void handleManualFetchDataset()}
        className="flex w-full items-center justify-center gap-2 rounded-full bg-[#F2853F] px-5 py-2.5 text-xs font-semibold text-black transition-colors hover:bg-[#FF9652] disabled:cursor-not-allowed disabled:opacity-45"
      >
        {manualFetchBusy ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Fetching Dataset Batches...
          </>
        ) : (
          <>
            <FolderGit2 className="h-3.5 w-3.5" />
            {`Fetch ${maxImages === 0 ? "All" : maxImages} Images (${batchSize}/batch) for Manual / Smart Select`}
          </>
        )}
      </button>
    </div>
  );

  const handleJobDropdownChange = (newJobId: string) => {
    clearVisualSessionLocal();
    setSelectedJobId(newJobId);
    setPinnedKey("");
    setPinnedItemIds([]);
    setActiveItemId("");
    setCustomClasses([]);
    setFindClassesText("");
    router.replace(`/annotate?job=${encodeURIComponent(newJobId)}`);
  };

  if (!jobs.length && !jobDetail) {
    return (
      <div className="mx-auto max-w-3xl space-y-6 p-6 md:p-8">
        <div className="c1-card flex flex-wrap items-center justify-between gap-4 p-6">
          <div>
            <p className="c1-kicker">06 / MANUAL ANNOTATOR &amp; SMART SELECT</p>
            <h2 className="mt-1.5 text-lg font-semibold text-white">
              No Roboflow Dataset Loaded Yet
            </h2>
            <p className="mt-1 text-xs text-[#A1A1AA]">
              Pull a project or batch below to start manual BBox, Polygon, SAM 2.1, or
              OWL-ViT Visual Embed annotation — or open Auto-Annotate for AI pre-labeling.
            </p>
          </div>
          <Link
            href="/auto-annotate"
            className="inline-flex items-center gap-2 rounded-full border border-[#303035] bg-[#141416] px-4 py-2 text-xs font-medium text-[#E4E4E7] hover:border-[#F2853F] hover:text-white"
          >
            <Sparkles className="h-3.5 w-3.5 text-[#F2853F]" />
            Open Auto-Annotate
          </Link>
        </div>

        <RoboflowSourceCard
          kicker="01 / ROBOFLOW DATASET SOURCE"
          title="Connect Project & Pull Batches for Manual / Smart Select"
          footer={pullDatasetFooter}
        />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-[#0A0A0C] lg:overflow-hidden">
      {fetchModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
          onClick={() => {
            if (!manualFetchBusy) setFetchModalOpen(false);
          }}
        >
          <div
            className="max-h-[90vh] w-full max-w-2xl overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-2 flex items-center justify-between px-1">
              <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#A1A1AA]">
                Pull Roboflow Batch Into Manual Annotator
              </span>
              <button
                type="button"
                onClick={() => setFetchModalOpen(false)}
                aria-label="Close Pull Dataset modal"
                className="rounded-lg border border-[#303035] bg-[#141416] p-1.5 text-[#A1A1AA] hover:border-white/30 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <RoboflowSourceCard
              kicker="01 / ROBOFLOW DATASET SOURCE"
              title="Connect Project & Pull Batches for Manual / Smart Select"
              footer={pullDatasetFooter}
            />
          </div>
        </div>
      )}

      {/* Top Studio Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-[#222226] bg-[#101013] px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <Link
            href={jobDetail ? `/verify?job=${encodeURIComponent(jobDetail.id)}` : "/verify"}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#303035] bg-[#141416] px-2.5 py-1.5 text-xs font-medium text-[#A1A1AA] hover:border-white/30 hover:text-white"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Verify
          </Link>

          <select
            value={selectedJobId}
            onChange={(e) => handleJobDropdownChange(e.target.value)}
            aria-label="Select annotation job"
            className="font-mono-tech rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            {jobs.map((j) => (
              <option key={j.id} value={j.id}>
                {j.projectSlug} · {j.id} ({j.rejectedCount} rejected)
              </option>
            ))}
          </select>

          <select
            value={queueFilter}
            onChange={(e) => {
              setQueueFilter(e.target.value as QueueFilter);
              setPinnedKey("");
            }}
            aria-label="Filter frames by review status"
            className="rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="rejected">Rejected Frames</option>
            <option value="all">All Frames ({allItems.length})</option>
            <option value="pending">Pending Review</option>
            <option value="accepted">Accepted Frames</option>
          </select>

          {jobDetail && jobDetail.batches.length > 0 && (
            <select
              value={batchFilter}
              onChange={(e) => {
                setBatchFilter(e.target.value);
                setPinnedKey("");
              }}
              aria-label="Filter by batch"
              className="font-mono-tech rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
            >
              <option value="all">All Batches</option>
              {jobDetail.batches.map((b) => (
                <option key={b.batchId} value={b.batchId}>
                  {b.label}
                </option>
              ))}
            </select>
          )}

          <button
            type="button"
            onClick={() => {
              setManualFetchError(null);
              setFetchModalOpen(true);
            }}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#F2853F]/50 bg-[#F2853F]/10 px-2.5 py-1.5 text-xs font-medium text-[#F2853F] hover:border-[#F2853F] hover:bg-[#F2853F]/20"
          >
            <FolderGit2 className="h-3.5 w-3.5" />
            Pull Dataset
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {saveState === "saving" && (
            <span className="font-mono-tech inline-flex items-center gap-1 rounded-full border border-[#F2853F]/40 bg-[#F2853F]/15 px-2.5 py-1 text-[11px] text-[#F2853F]">
              <Loader2 className="h-3 w-3 animate-spin" />
              Saving...
            </span>
          )}
          {saveState === "saved" && !dirty && (
            <span className="font-mono-tech inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-1 text-[11px] text-emerald-300">
              <Check className="h-3 w-3" />
              Auto-saved
            </span>
          )}
          {dirty && saveState !== "saving" && (
            <span className="font-mono-tech rounded-full border border-amber-500/40 bg-amber-500/15 px-2.5 py-1 text-[11px] text-amber-200">
              Unsaved edits
            </span>
          )}

          <button
            type="button"
            disabled={!currentItem || saveState === "saving"}
            onClick={() => void handleSaveAndAccept()}
            className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#1A1A1E] px-3.5 py-1.5 text-xs font-medium text-white hover:border-[#F2853F] disabled:opacity-40"
          >
            <Check className="h-3.5 w-3.5 text-emerald-400" />
            Save & Accept (⌘S)
          </button>

          <button
            type="button"
            disabled={!currentItem || saveState === "saving"}
            onClick={() => void handleAcceptAndNext()}
            className="inline-flex items-center gap-1.5 rounded-full bg-[#F2853F] px-4 py-1.5 text-xs font-semibold text-black hover:bg-[#FF9652] disabled:opacity-40"
          >
            Accept & Next
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {loadError && (
        <div className="flex items-center gap-2 border-b border-rose-500/40 bg-rose-500/15 px-4 py-2 text-xs text-rose-200">
          <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
          <span>{loadError}</span>
        </div>
      )}

      {saveError && (
        <div className="flex items-center gap-2 border-b border-rose-500/40 bg-rose-500/15 px-4 py-2 text-xs text-rose-200">
          <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
          <span>{saveError}</span>
        </div>
      )}

      {/* Main 3-Pane Studio Workspace (Left Classes/Layers/AI | Center Canvas | Right Filmstrip) */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* LEFT PANE: Annotations, Classes/Layers, Find Objects with AI, Frame Status */}
        <aside className="flex w-full shrink-0 flex-col border-b border-[#222226] bg-[#101013] lg:w-[276px] lg:border-b-0 lg:border-r">
          {/* Filename + Copy + Group */}
          <div className="border-b border-[#222226] p-3.5">
            <div className="flex items-center justify-between gap-2">
              <p
                title={currentItem?.filename || "No image selected"}
                className="font-mono-tech truncate text-xs font-medium text-white"
              >
                {currentItem?.filename || "No matching frames"}
              </p>
              {currentItem && (
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard?.writeText(currentItem.filename);
                    setCopiedFilename(true);
                    setTimeout(() => setCopiedFilename(false), 1200);
                  }}
                  title="Copy filename"
                  aria-label="Copy filename"
                  className="rounded p-1 text-[#8A8A8E] hover:bg-[#1E1E24] hover:text-white"
                >
                  {copiedFilename ? (
                    <Check className="h-3.5 w-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="h-3.5 w-3.5" />
                  )}
                </button>
              )}
            </div>

            <div className="mt-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-white">
                  Annotations
                </span>
                <span className="font-mono-tech rounded-full border border-[#303035] bg-[#1A1A1E] px-2 py-0.5 text-[10px] text-[#A1A1AA]">
                  {draftDetections.length}
                </span>
              </div>
              {currentItem?.roboflowImageId && (
                <span
                  title={`Preserves original Roboflow Image ID: ${currentItem.roboflowImageId}`}
                  className="font-mono-tech max-w-[120px] truncate text-[10px] text-[#8A8A8E]"
                >
                  ID: {currentItem.roboflowImageId}
                </span>
              )}
            </div>

            {jobDetail && (
              <p className="mt-1 truncate text-[11px] text-[#8A8A8E]">
                Group:{" "}
                <span className="font-mono-tech text-[#D4D4D8]">
                  {jobDetail.projectName}
                </span>
              </p>
            )}
          </div>

          {/* Classes / Layers Tabs (matching screenshot) */}
          <div className="grid grid-cols-2 border-b border-[#222226] text-xs font-medium">
            <button
              type="button"
              onClick={() => setLeftTab("classes")}
              className={`border-b-2 py-2.5 transition-colors ${
                leftTab === "classes"
                  ? "border-[#F2853F] text-white"
                  : "border-transparent text-[#8A8A8E] hover:text-white"
              }`}
            >
              Classes ({presentClassColors.length})
            </button>
            <button
              type="button"
              onClick={() => setLeftTab("layers")}
              className={`border-b-2 py-2.5 transition-colors ${
                leftTab === "layers"
                  ? "border-[#F2853F] text-white"
                  : "border-transparent text-[#8A8A8E] hover:text-white"
              }`}
            >
              Layers ({draftDetections.length})
            </button>
          </div>

          {/* Tab Body */}
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {leftTab === "classes" ? (
              <div className="space-y-2">
                {presentClassColors.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-[#2C2C32] p-3 text-center text-[11px] text-[#8A8A8E]">
                    No classes present on this image yet.
                  </div>
                ) : (
                  presentClassColors.map((row) => {
                    const isActive =
                      row.name.toLowerCase() === activeClass.toLowerCase();
                    const count =
                      classCountsOnItem.get(row.name.trim().toLowerCase()) || 0;
                    const selectedDet = draftDetections.find(
                      (d) => d.id === selectedDetId
                    );
                    const canAssignToSelected =
                      Boolean(selectedDet) &&
                      selectedDet!.label.toLowerCase() !== row.name.toLowerCase();

                    return (
                      <div
                        key={row.name}
                        className={`flex items-center justify-between gap-2 rounded-xl border px-2.5 py-2 text-xs transition-colors ${
                          isActive
                            ? "border-[#F2853F] bg-[#F2853F]/12 text-white"
                            : "border-[#26262A] bg-[#141416] text-[#D4D4D8] hover:border-white/25"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => setActiveClass(row.name)}
                          className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        >
                          <span
                            className="h-3 w-3 shrink-0 rounded-full border border-white/25"
                            style={{ backgroundColor: row.color }}
                          />
                          <span className="font-mono-tech truncate font-medium">
                            {row.name}
                          </span>
                        </button>

                        <div className="flex items-center gap-1.5">
                          {canAssignToSelected && (
                            <button
                              type="button"
                              onClick={() => {
                                setActiveClass(row.name);
                                handleDetectionsChange(
                                  draftDetections.map((d) =>
                                    d.id === selectedDetId
                                      ? { ...d, label: row.name }
                                      : d
                                  ),
                                  true
                                );
                              }}
                              className="rounded border border-[#F2853F]/50 px-1.5 py-0.5 text-[10px] text-[#F2853F] hover:bg-[#F2853F] hover:text-black"
                              title={`Change selected annotation label to '${row.name}'`}
                            >
                              Apply
                            </button>
                          )}
                          <span className="font-mono-tech rounded-full border border-[#303035] bg-[#0A0A0C] px-2 py-0.5 text-[10px] text-[#A1A1AA]">
                            {count}
                          </span>
                        </div>
                      </div>
                    );
                  })
                )}

                <form onSubmit={handleAddClass} className="flex gap-1.5 pt-1">
                  <input
                    type="text"
                    value={newClassInput}
                    onChange={(e) => setNewClassInput(e.target.value)}
                    placeholder="Add class..."
                    className="font-mono-tech min-w-0 flex-1 rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white placeholder:text-[#5A5A60] focus:border-[#F2853F] focus:outline-none"
                  />
                  <button
                    type="submit"
                    aria-label="Add class"
                    className="inline-flex items-center justify-center rounded-lg border border-[#303035] bg-[#141416] px-2.5 py-1.5 text-xs text-white hover:border-[#F2853F]"
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                </form>

                {activeClass.trim() &&
                  !presentClassColors.some(
                    (c) =>
                      c.name.trim().toLowerCase() ===
                      activeClass.trim().toLowerCase()
                  ) && (
                    <div className="flex items-center gap-1.5 px-1 pt-0.5 text-[11px] text-[#8A8A8E]">
                      <span>Next draw class:</span>
                      <span
                        className="h-2 w-2 shrink-0 rounded-full border border-white/25"
                        style={{
                          backgroundColor: getClassColor(
                            activeClass,
                            mergedClassColors
                          ),
                        }}
                      />
                      <span className="font-mono-tech truncate text-[#D4D4D8]">
                        {activeClass}
                      </span>
                    </div>
                  )}
              </div>
            ) : (
              <div className="space-y-2">
                {draftDetections.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-[#2C2C32] p-4 text-center text-xs text-[#8A8A8E]">
                    No annotations on this frame. Draw with Box (`B`), Polygon
                    (`P`), Smart Select (`S`), or mark Null (`N`).
                  </div>
                ) : (
                  draftDetections.map((det, idx) => {
                    const color = getClassColor(det.label, mergedClassColors);
                    const isSelected = selectedDetId === det.id;
                    const isHidden = hiddenDetIds.has(det.id);
                    return (
                      <div
                        key={det.id}
                        onClick={() => setSelectedDetId(det.id)}
                        className={`flex items-center justify-between gap-2 rounded-xl border px-2.5 py-2 text-xs transition-colors ${
                          isSelected
                            ? "border-[#F2853F] bg-[#F2853F]/12"
                            : "border-[#26262A] bg-[#141416] hover:border-white/25"
                        }`}
                      >
                        <div className="flex min-w-0 flex-1 items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: color }}
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1">
                              <span className="font-mono-tech text-[10px] text-[#8A8A8E]">
                                #{idx + 1}
                              </span>
                              <select
                                value={det.label}
                                onClick={(e) => e.stopPropagation()}
                                onChange={(e) => {
                                  const nextLabel = e.target.value;
                                  handleDetectionsChange(
                                    draftDetections.map((d) =>
                                      d.id === det.id
                                        ? { ...d, label: nextLabel }
                                        : d
                                    ),
                                    true
                                  );
                                }}
                                aria-label={`Class for annotation ${idx + 1}`}
                                className="font-mono-tech min-w-0 flex-1 truncate rounded bg-[#0A0A0C] px-1 py-0.5 text-[11px] font-medium text-white focus:outline-none"
                              >
                                {mergedClassColors.map((cc) => (
                                  <option key={cc.name} value={cc.name}>
                                    {cc.name}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <p className="font-mono-tech mt-0.5 text-[10px] text-[#8A8A8E]">
                              {(det.confidence * 100).toFixed(0)}% ·{" "}
                              {det.polygon && det.polygon.length >= 3
                                ? `${det.polygon.length}v poly`
                                : "bbox"}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleToggleVitRefDet(det.id);
                            }}
                            title={
                              vitRefDetIdsForCurrent.has(det.id)
                                ? "Unpin ViT Reference box"
                                : "Pin as ViT Reference box"
                            }
                            aria-label={`Toggle ViT reference for layer ${idx + 1}`}
                            className={`rounded p-1 transition-colors ${
                              vitRefDetIdsForCurrent.has(det.id)
                                ? "bg-[#F2853F]/20 text-[#F2853F]"
                                : "text-[#8A8A8E] hover:bg-[#1E1E24] hover:text-white"
                            }`}
                          >
                            {vitRefDetIdsForCurrent.has(det.id) ? (
                              <BookmarkCheck className="h-3.5 w-3.5" />
                            ) : (
                              <BookmarkPlus className="h-3.5 w-3.5" />
                            )}
                          </button>

                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setHiddenDetIds((prev) => {
                                const next = new Set(prev);
                                if (next.has(det.id)) next.delete(det.id);
                                else next.add(det.id);
                                return next;
                              });
                            }}
                            title={isHidden ? "Show layer" : "Hide layer"}
                            aria-label="Toggle layer visibility"
                            className="rounded p-1 text-[#8A8A8E] hover:bg-[#1E1E24] hover:text-white"
                          >
                            {isHidden ? (
                              <EyeOff className="h-3.5 w-3.5" />
                            ) : (
                              <Eye className="h-3.5 w-3.5" />
                            )}
                          </button>

                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDetectionsChange(
                                draftDetections.filter((d) => d.id !== det.id),
                                true
                              );
                              if (selectedDetId === det.id) setSelectedDetId(null);
                            }}
                            title="Delete annotation"
                            aria-label={`Delete layer ${idx + 1}`}
                            className="rounded p-1 text-[#8A8A8E] hover:bg-[#1E1E24] hover:text-rose-400"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {/* Find Objects with AI (ViT Ref / OWL Text / YOLOE) */}
            <FindObjectsAiPanel
              disabled={!currentItem}
              findOpen={findOpen}
              onToggleOpen={() => setFindOpen((v) => !v)}
              findMode={findMode}
              onFindModeChange={(m) => {
                setFindMode(m);
                if (m !== "vit_ref") setVisualDrawMode(null);
              }}
              owlModelId={owlModelId}
              owlModelOptions={owlModelOptions}
              onOwlModelChange={handleOwlModelChange}
              classColors={mergedClassColors}
              activeClass={activeClass}
              targetClassOptions={availableClassOptions}
              selectedAiClasses={selectedAiClasses}
              onToggleAiClass={handleToggleAiClass}
              onSelectOnlyAiClass={handleSelectOnlyAiClass}
              onSelectAllAiClasses={handleSelectAllAiClasses}
              onClearAiClasses={handleClearAiClasses}
              refImageCount={refImageCount}
              refBoxCount={refBoxCount}
              vitBankDirty={vitBankDirty}
              vitIndexing={vitIndexing}
              vitClassBadges={vitClassBadges}
              vitClassSummaries={vitClassSummaries}
              currentItemDetCount={draftDetections.length}
              currentMatchingDetCount={currentMatchingDetIds.length}
              isCurrentFramePinned={isCurrentFramePinned}
              allAnnotatedPinned={allAnnotatedPinned}
              pinnedVitFrames={pinnedVitFrames}
              vitBankFeedback={vitBankFeedback}
              annotatedItemsCount={annotatedItemsCount}
              onTogglePinCurrentFrame={handleTogglePinCurrentFrame}
              onUseAllAnnotatedRefs={handleUseAllAnnotatedRefs}
              onIndexVitRefs={handleIndexVitRefsNow}
              onSelectPinnedItem={(id) => void navigateToItem(id)}
              onUnpinItem={handleUnpinVitRefItem}
              onClearVitRefs={handleClearVitRefs}
              visualQueryId={visualQueryId}
              visualScope={visualScope}
              onVisualScopeChange={setVisualScope}
              visualDrawMode={visualDrawMode}
              onVisualDrawModeChange={setVisualDrawMode}
              visualBatchTarget={visualBatchTarget}
              onVisualBatchTargetChange={setVisualBatchTarget}
              visualMergeMode={visualMergeMode}
              onVisualMergeModeChange={setVisualMergeMode}
              vitSkipAnnotated={vitSkipAnnotated}
              onVitSkipAnnotatedChange={setVitSkipAnnotated}
              visualThreshold={visualThreshold}
              onVisualThresholdChange={setVisualThreshold}
              visualMinObjectness={visualMinObjectness}
              onVisualMinObjectnessChange={setVisualMinObjectness}
              visualUseSamMasks={visualUseSamMasks}
              onVisualUseSamMasksChange={setVisualUseSamMasks}
              smartSamRefinerId={smartSamRefinerId}
              onSmartSamRefinerIdChange={setSmartSamRefinerId}
              visualAutoSearchOnSwitch={visualAutoSearchOnSwitch}
              onVisualAutoSearchOnSwitchChange={setVisualAutoSearchOnSwitch}
              visualHits={visualHits}
              showVisualOverlay={showVisualOverlay}
              onShowVisualOverlayChange={setShowVisualOverlay}
              totalStagedOverlayItems={totalStagedOverlayItems}
              totalStagedOverlayBoxes={totalStagedOverlayBoxes}
              nextStagedOverlayItemId={nextStagedOverlayItemId}
              onJumpToNextStagedOverlay={handleJumpToNextStagedOverlay}
              visualExemplars={visualExemplars}
              canUndoVisualEx={canUndoVisualEx}
              canRedoVisualEx={canRedoVisualEx}
              visualRedoCount={visualExemplarsRedo.length}
              visualTopScore={visualTopScore}
              visualTopObjectness={visualTopObjectness}
              visualPosCount={visualPosCount}
              visualNegCount={visualNegCount}
              visualLastLatency={visualLastLatency}
              visualMatchedIou={visualMatchedIou}
              visualBusy={visualBusy}
              visualError={visualError}
              visualBatchProgress={visualBatchProgress}
              activeBatchId={activeBatchId}
              currentBatchCount={currentBatchCount}
              allItemsCount={allItems.length}
              queueItemsCount={queueItems.length}
              onDetectCurrentImage={() => void handleDetectCurrentImage()}
              onCommitVisualHits={handleCommitVisualHits}
              onRejectCurrentVisualOverlay={handleClearVisualHits}
              onClearAllStagedOverlays={handleClearAllStagedOverlays}
              onDismissVisualHit={handleDismissSingleVisualHit}
              onUndoVisualExemplar={() => void handleUndoRedoVisualExemplar("undo")}
              onRedoVisualExemplar={() => void handleUndoRedoVisualExemplar("redo")}
              onResetVisualQuery={() => void handleResetVisualQuery()}
              onRunVisualBatch={() => void handleRunVisualBatch()}
              findClassesText={findClassesText}
              onFindClassesTextChange={setFindClassesText}
              findOutputMode={findOutputMode}
              onFindOutputModeChange={setFindOutputMode}
              findConf={findConf}
              onFindConfChange={setFindConf}
              findBusy={findBusy}
              findMessage={findMessage}
              onRunFindObjects={() => void handleRunFindObjects()}
            />
          </div>

          {/* Bottom Frame Status & Quick Actions (bottom-left section) */}
          {currentItem && (
            <div className="border-t border-[#222226] bg-[#0B0B0D] p-3">
              <div className="flex items-center justify-between text-[11px]">
                <span className="c1-kicker">FRAME STATUS</span>
                <span
                  className={`font-mono-tech font-semibold ${
                    currentItem.reviewStatus === "accepted"
                      ? "text-emerald-400"
                      : currentItem.reviewStatus === "rejected"
                        ? "text-rose-400"
                        : "text-[#A1A1AA]"
                  }`}
                >
                  {currentItem.reviewStatus === "accepted" &&
                  draftDetections.length === 0
                    ? "NULL (ACCEPTED)"
                    : currentItem.reviewStatus.toUpperCase()}
                </span>
              </div>

              {currentItem.rejectReason && (
                <p className="font-mono-tech mt-0.5 text-[10px] text-rose-300/80">
                  Reject reason: {currentItem.rejectReason.replace("_", " ")}
                </p>
              )}

              <div className="mt-2 grid grid-cols-3 gap-1.5">
                <button
                  type="button"
                  disabled={saveState === "saving"}
                  onClick={() => void handleSaveAndAccept()}
                  className="inline-flex items-center justify-center gap-1 rounded-lg border border-emerald-500/40 bg-emerald-500/12 py-1.5 text-[11px] font-medium text-emerald-300 hover:bg-emerald-500/22"
                >
                  <CheckCircle2 className="h-3 w-3" />
                  Accept
                </button>
                <button
                  type="button"
                  disabled={saveState === "saving"}
                  onClick={() => void handleMarkNull()}
                  className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#141416] py-1.5 text-[11px] font-medium text-[#D4D4D8] hover:border-[#F2853F] hover:text-white"
                  title="Clear annotations & mark Accepted as Null frame (N)"
                >
                  <Ban className="h-3 w-3 text-[#F2853F]" />
                  Null
                </button>
                <button
                  type="button"
                  disabled={saveState === "saving"}
                  onClick={() =>
                    void persistItem(
                      currentItem,
                      draftDetections,
                      "rejected",
                      currentItem.rejectReason ?? "bad_box"
                    )
                  }
                  className="inline-flex items-center justify-center gap-1 rounded-lg border border-rose-500/40 bg-rose-500/12 py-1.5 text-[11px] font-medium text-rose-300 hover:bg-rose-500/22"
                >
                  <XCircle className="h-3 w-3" />
                  Reject
                </button>
              </div>
            </div>
          )}
        </aside>

        {/* CENTER PANE: Interactive Canvas */}
        {currentItem ? (
          <ManualAnnotatorCanvas
            item={currentItem}
            detections={draftDetections}
            classColors={mergedClassColors}
            activeClass={activeClass}
            tool={tool}
            onToolChange={(nextTool) => {
              setVisualDrawMode(null);
              setTool(nextTool);
            }}
            selectedDetId={selectedDetId}
            onSelectDet={setSelectedDetId}
            hiddenDetIds={hiddenDetIds}
            onDetectionsChange={handleDetectionsChange}
            onPushHistorySnapshot={handlePushHistorySnapshot}
            canUndo={undoStack.length > 0}
            canRedo={redoStack.length > 0}
            onUndo={handleUndo}
            onRedo={handleRedo}
            onMarkNull={() => void handleMarkNull()}
            onDeleteSelected={handleDeleteSelected}
            onUpdateDetLabel={handleUpdateDetLabel}
            onAddClassAndAssign={handleAddClassAndAssign}
            onCommitDragEdit={handleCommitDragEdit}
            smartOutputMode={smartOutputMode}
            onSmartOutputModeChange={setSmartOutputMode}
            smartSamRefinerId={smartSamRefinerId}
            onSmartSamRefinerIdChange={setSmartSamRefinerId}
            smartMultiPoint={smartMultiPoint}
            onSmartMultiPointChange={setSmartMultiPoint}
            smartComplexity={smartComplexity}
            onSmartComplexityChange={setSmartComplexity}
            smartPreparing={smartPreparing}
            smartReady={smartReady}
            smartBusy={smartBusy}
            smartStatusMsg={smartStatusMsg}
            onSmartHoverPreview={handleSmartHoverPreview}
            onCommitSmartDetection={handleCommitSmartDetection}
            onRunSmartSelect={handleRunSmartSelect}
            onExitSmartSelect={() => setTool("drag")}
            visualHits={visualHits}
            showVisualOverlay={showVisualOverlay}
            visualExemplars={visualExemplars}
            visualBusy={visualBusy || vitIndexing}
            visualDrawMode={visualDrawMode}
            onExitVisualDrawMode={() => setVisualDrawMode(null)}
            showVisualExemplars={findOpen && findMode === "vit_ref"}
            vitRefDetIds={vitRefDetIdsForCurrent}
            onToggleVitRefDet={handleToggleVitRefDet}
            onVisualEmbedDragBox={handleVisualEmbedDragBox}
            onVisualEmbedClickPoint={handleVisualEmbedClickPoint}
            onCommitVisualHits={handleCommitVisualHits}
            onClearVisualHits={handleClearVisualHits}
            onDismissVisualHit={handleDismissSingleVisualHit}
          />
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center bg-[#060608] p-8 text-center">
            {loading ? (
              <span className="font-mono-tech inline-flex items-center gap-2 text-xs text-[#A1A1AA]">
                <Loader2 className="h-4 w-4 animate-spin text-[#F2853F]" />
                Loading job frames...
              </span>
            ) : (
              <>
                <p className="text-sm font-medium text-white">
                  No frames match the current filter.
                </p>
                <p className="mt-1 text-xs text-[#8A8A8E]">
                  Switch the filter above to All Frames or select another batch.
                </p>
                {queueFilter !== "all" && (
                  <button
                    type="button"
                    onClick={() => {
                      setQueueFilter("all");
                      setPinnedKey("");
                    }}
                    className="mt-4 rounded-full bg-[#F2853F] px-4 py-1.5 text-xs font-semibold text-black hover:bg-[#FF9652]"
                  >
                    Show All Frames
                  </button>
                )}
              </>
            )}
          </div>
        )}

        {/* RIGHT PANE: Images Filmstrip (matching Roboflow right panel) */}
        <aside className="flex w-full shrink-0 flex-col border-t border-[#222226] bg-[#101013] lg:w-[172px] lg:border-l lg:border-t-0">
          <div className="border-b border-[#222226] p-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white">Images</span>
              <span className="font-mono-tech text-[10px] text-[#8A8A8E]">
                {queueItems.length > 0 && currentIndex >= 0
                  ? `${currentIndex + 1} / ${queueItems.length}`
                  : `0 / ${queueItems.length}`}
              </span>
            </div>
            <div className="mt-2 flex items-center justify-between gap-1.5">
              <button
                type="button"
                disabled={currentIndex <= 0}
                onClick={() => {
                  if (currentIndex > 0) {
                    void navigateToItem(queueItems[currentIndex - 1].id);
                  }
                }}
                aria-label="Previous image"
                className="flex flex-1 items-center justify-center rounded-lg border border-[#303035] bg-[#141416] py-1 text-xs text-[#A1A1AA] hover:border-white/30 hover:text-white disabled:opacity-35"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                disabled={
                  currentIndex < 0 || currentIndex + 1 >= queueItems.length
                }
                onClick={() => {
                  if (currentIndex + 1 < queueItems.length) {
                    void navigateToItem(queueItems[currentIndex + 1].id);
                  }
                }}
                aria-label="Next image"
                className="flex flex-1 items-center justify-center rounded-lg border border-[#303035] bg-[#141416] py-1 text-xs text-[#A1A1AA] hover:border-white/30 hover:text-white disabled:opacity-35"
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-row gap-2 overflow-x-auto p-2.5 lg:flex-col lg:overflow-y-auto lg:overflow-x-hidden">
            {queueItems.map((it) => {
              const isActive = it.id === activeItemId;
              const detCount = isActive
                ? draftDetections.length
                : it.detections.length;
              const isNull = it.reviewStatus === "accepted" && detCount === 0;
              const vitPinCount = (activeVitRefsByItem[it.id] || []).length;
              const stagedCnt = (stagedVisualHitsByItem[it.id] || []).length;
              return (
                <button
                  key={it.id}
                  ref={(el) => {
                    if (el) thumbRefs.current.set(it.id, el);
                    else thumbRefs.current.delete(it.id);
                  }}
                  type="button"
                  onClick={() => void navigateToItem(it.id)}
                  className={`group flex w-36 shrink-0 flex-col overflow-hidden rounded-xl border text-left transition-colors lg:w-full ${
                    isActive
                      ? "border-[#F2853F] bg-[#1B1715]"
                      : "border-[#26262A] bg-[#141416] hover:border-white/30"
                  }`}
                >
                  <div className="relative aspect-[4/3] w-full overflow-hidden bg-black">
                    <img
                      src={it.imageUrl}
                      alt={it.filename}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                    {vitPinCount > 0 && (
                      <span className="font-mono-tech absolute left-1 top-1 inline-flex items-center gap-0.5 rounded bg-[#F2853F] px-1 py-0.5 text-[9px] font-bold text-black shadow">
                        <Pin className="h-2.5 w-2.5" />
                        ViT
                      </span>
                    )}
                    <span className="font-mono-tech absolute bottom-1 left-1 rounded bg-black/75 px-1.5 py-0.5 text-[9px] text-white">
                      {detCount === 0 && it.reviewStatus === "accepted"
                        ? "NULL"
                        : `${detCount} box`}
                    </span>
                    {stagedCnt > 0 && (
                      <span className="font-mono-tech absolute bottom-1 right-1 rounded bg-emerald-500/90 px-1.5 py-0.5 text-[9px] font-bold text-black shadow">
                        +{stagedCnt} AI
                      </span>
                    )}
                    <span
                      className={`absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full border text-[9px] ${
                        it.reviewStatus === "accepted"
                          ? isNull
                            ? "border-emerald-400/60 bg-emerald-500/80 text-black"
                            : "border-emerald-400 bg-emerald-500 text-black"
                          : it.reviewStatus === "rejected"
                            ? "border-rose-400 bg-rose-500 text-white"
                            : "border-[#505058] bg-[#141416]/85 text-[#A1A1AA]"
                      }`}
                    >
                      {it.reviewStatus === "accepted" ? (
                        <Check className="h-2.5 w-2.5" />
                      ) : it.reviewStatus === "rejected" ? (
                        "×"
                      ) : (
                        "·"
                      )}
                    </span>
                  </div>
                  <p className="font-mono-tech truncate px-2 py-1 text-[10px] text-[#A1A1AA] group-hover:text-white">
                    {it.filename}
                  </p>
                </button>
              );
            })}
          </div>
        </aside>
      </div>
    </div>
  );
}
