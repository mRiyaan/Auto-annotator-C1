"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Check,
  CheckCheck,
  ChevronLeft,
  ChevronRight,
  CloudUpload,
  Download,
  Loader2,
  PenTool,
  RotateCcw,
  Sparkles,
  X,
} from "lucide-react";
import { useWorkbench } from "@/lib/workbench-store";
import {
  approveAutoBatch,
  AutoAnnotateApiError,
  fetchAutoJobDetail,
  getDownloadAcceptedZipUrl,
  patchAutoItemReview,
  pushAcceptedZipApi,
} from "./api";
import {
  clearCachedJobDetail,
  loadCachedJobDetail,
  loadLastVerifyJobId,
  loadVerifyUiState,
  saveCachedJobDetail,
  saveLastVerifyJobId,
  saveVerifyUiState,
} from "./cache";
import { getClassColor, hexToRgba } from "./color-utils";
import { useAutoAnnotate } from "./store";
import type {
  AutoItemRecord,
  AutoJobRecord,
  AutoRejectReason,
  AutoReviewStatus,
  PushAcceptedResponse,
  PushDestination,
} from "./types";
import { VerifyLightboxModal } from "./VerifyLightboxModal";

const REASON_LABELS: Record<AutoRejectReason, string> = {
  missed_object: "Missed object",
  false_detection: "False detection",
  bad_box: "Bad box",
  double_box: "Double box",
};

function recomputeJobCounts(
  prevJob: AutoJobRecord,
  items: AutoItemRecord[]
): AutoJobRecord {
  const reasonCounts: Record<string, number> = {
    missed_object: 0,
    false_detection: 0,
    bad_box: 0,
    double_box: 0,
  };
  const byBatch = new Map<
    string,
    { accepted: number; rejected: number; pending: number }
  >();
  let accepted = 0;
  let rejected = 0;
  let pending = 0;

  for (const it of items) {
    const st = byBatch.get(it.batchId) ?? {
      accepted: 0,
      rejected: 0,
      pending: 0,
    };
    if (it.reviewStatus === "accepted") {
      accepted += 1;
      st.accepted += 1;
    } else if (it.reviewStatus === "rejected") {
      rejected += 1;
      st.rejected += 1;
      if (it.rejectReason && it.rejectReason in reasonCounts) {
        reasonCounts[it.rejectReason] += 1;
      }
    } else {
      pending += 1;
      st.pending += 1;
    }
    byBatch.set(it.batchId, st);
  }

  return {
    ...prevJob,
    acceptedCount: accepted,
    rejectedCount: rejected,
    pendingCount: pending,
    rejectReasonCounts: reasonCounts,
    batches: prevJob.batches.map((b) => {
      const st = byBatch.get(b.batchId);
      return st
        ? {
            ...b,
            accepted: st.accepted,
            rejected: st.rejected,
            pending: st.pending,
          }
        : b;
    }),
  };
}

function inferSameProjectMode(job: AutoJobRecord): PushDestination | null {
  if (job.lastPushDestination) return job.lastPushDestination;
  if (job.lastPushProject && job.lastPushProject === job.projectSlug) {
    if (job.lastPushMessage?.startsWith("Staged ")) {
      return "same_project_annotate";
    }
    if (job.lastPushMessage?.includes("folder '")) {
      return "same_project_new_folder";
    }
    return "same_project_dataset";
  }
  return null;
}

function resolveDefaultPushDestination(
  job: AutoJobRecord,
  allJobs: AutoJobRecord[]
): PushDestination {
  if (job.lastPushDestination) {
    return job.lastPushDestination;
  }
  if (job.fetchScope === "rejected_retry") {
    const parent = job.parentJobId
      ? allJobs.find((j) => j.id === job.parentJobId)
      : undefined;
    if (parent) {
      const parentDest = inferSameProjectMode(parent);
      if (parentDest && parentDest !== "new_project") {
        return parentDest;
      }
    }
    return "same_project_dataset";
  }
  const inferred = inferSameProjectMode(job);
  if (inferred) return inferred;
  if (
    job.fetchScope === "annotation_job" ||
    job.fetchScope === "dataset" ||
    job.fetchScope === "batch" ||
    job.fetchScope === "all" ||
    job.fetchScope === "unannotated" ||
    job.fetchScope === "annotated"
  ) {
    return "same_project_dataset";
  }
  return "new_project";
}

export function VerifyQueueView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryJobId = searchParams.get("job");

  const { backendUrl, backendStatus } = useWorkbench();
  const { apiKey, setApiKey, jobs, setJobs, refreshJobs, sessionEpoch } =
    useAutoAnnotate();

  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [jobDetail, setJobDetail] = useState<AutoJobRecord | null>(null);
  const [allItems, setAllItems] = useState<AutoItemRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Batch-wise & status filters matching screenshot UX (persisted per job)
  const [selectedBatchId, setSelectedBatchId] = useState<string>("batch-01");
  const [frameFilter, setFrameFilter] = useState<string>("all");
  const [batchStateFilter, setBatchStateFilter] = useState<string>("all");
  const [reasonFilter, setReasonFilter] = useState<string>("all");
  const [autoAdvance, setAutoAdvance] = useState<boolean>(false);

  // Double-click Lightbox state & save status feedback
  const [lightboxItemId, setLightboxItemId] = useState<string | null>(null);
  const [reviewSaveState, setReviewSaveState] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [reviewSaveError, setReviewSaveError] = useState<string | null>(null);

  // Bulk Roboflow Push modal state
  const [pushModalOpen, setPushModalOpen] = useState(false);
  const [pushDestination, setPushDestination] = useState<PushDestination>(
    "same_project_dataset"
  );
  const prevDestJobIdRef = useRef<string | null>(null);
  const userPickedDestRef = useRef<boolean>(false);
  const [targetProjectName, setTargetProjectName] = useState("");
  const [targetFolderName, setTargetFolderName] = useState("");
  const [targetAnnotationJobId, setTargetAnnotationJobId] = useState("");
  const [newFolderAddToDataset, setNewFolderAddToDataset] = useState(true);
  const [pushScopeMode, setPushScopeMode] = useState<"all" | "batch" | "unpushed">(
    "all"
  );
  const [includeEmpty, setIncludeEmpty] = useState(true);
  const [projectLicense, setProjectLicense] = useState<"MIT" | "private">(
    "MIT"
  );
  const [pushKey, setPushKey] = useState(apiKey);
  const [pushBusy, setPushBusy] = useState(false);
  const [pushResult, setPushResult] = useState<PushAcceptedResponse | null>(
    null
  );
  const [pushError, setPushError] = useState<string | null>(null);

  useEffect(() => {
    if (apiKey && !pushKey) setPushKey(apiKey);
  }, [apiKey, pushKey]);

  const prevEpochRef = useRef(sessionEpoch);
  useEffect(() => {
    if (sessionEpoch === 0 || prevEpochRef.current === sessionEpoch) return;
    prevEpochRef.current = sessionEpoch;
    setSelectedJobId("");
    setJobDetail(null);
    setAllItems([]);
    setPushKey("");
    setPushResult(null);
    setPushError(null);
    setLightboxItemId(null);
  }, [sessionEpoch]);

  // Default Push Destination to parent/source same-project mode unless the user explicitly overrides it
  useEffect(() => {
    if (!jobDetail) return;
    if (prevDestJobIdRef.current !== jobDetail.id) {
      prevDestJobIdRef.current = jobDetail.id;
      userPickedDestRef.current = false;
    }
    if (!userPickedDestRef.current) {
      setPushDestination(resolveDefaultPushDestination(jobDetail, jobs));
    }
  }, [jobDetail, jobs]);

  // Select job from query param, persisted last job, or most recent active job
  useEffect(() => {
    if (queryJobId) {
      setSelectedJobId(queryJobId);
      return;
    }
    if (!selectedJobId) {
      const lastId = loadLastVerifyJobId();
      if (lastId && (!jobs.length || jobs.some((j) => j.id === lastId))) {
        setSelectedJobId(lastId);
        return;
      }
      if (jobs.length > 0) {
        setSelectedJobId(jobs[0].id);
      }
    } else if (jobs.length > 0 && !jobs.some((j) => j.id === selectedJobId)) {
      // Selected job no longer exists on server — fall back cleanly
      clearCachedJobDetail(selectedJobId);
      setSelectedJobId(jobs[0].id);
    }
  }, [queryJobId, jobs, selectedJobId]);

  // Hydrate persisted filter state + cached job items immediately when switching jobs
  useEffect(() => {
    if (!selectedJobId) return;
    saveLastVerifyJobId(selectedJobId);
    const ui = loadVerifyUiState(selectedJobId);
    setSelectedBatchId(ui?.selectedBatchId || "batch-01");
    setFrameFilter(ui?.frameFilter || "all");
    setBatchStateFilter(ui?.batchStateFilter || "all");
    setReasonFilter(ui?.reasonFilter || "all");
    setAutoAdvance(Boolean(ui?.autoAdvance));

    const cached = loadCachedJobDetail(selectedJobId);
    if (cached && cached.job?.id === selectedJobId) {
      setJobDetail(cached.job);
      setAllItems(cached.items);
      setTargetProjectName((prev) =>
        prev ? prev : `${cached.job.projectSlug}-auto-annotated`
      );
      setTargetAnnotationJobId(cached.job.sourceAnnotationJobId || "");
    }
  }, [selectedJobId]);

  // Save filter/UI preferences persistently for the selected job
  useEffect(() => {
    if (!selectedJobId) return;
    saveVerifyUiState(selectedJobId, {
      selectedBatchId,
      frameFilter,
      batchStateFilter,
      reasonFilter,
      autoAdvance,
    });
  }, [
    selectedJobId,
    selectedBatchId,
    frameFilter,
    batchStateFilter,
    reasonFilter,
    autoAdvance,
  ]);

  const loadJob = useCallback(
    async (jobId: string) => {
      if (!jobId || backendStatus !== "online") return;
      setLoading(true);
      setActionError(null);
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
        setTargetProjectName((prev) =>
          prev ? prev : `${res.job.projectSlug}-auto-annotated`
        );
        setTargetAnnotationJobId((prev) =>
          prev ? prev : res.job.sourceAnnotationJobId || ""
        );
        if (
          res.job.batches.length > 0 &&
          !res.job.batches.some((b) => b.batchId === selectedBatchId) &&
          selectedBatchId !== "all"
        ) {
          setSelectedBatchId(res.job.batches[0].batchId);
        }
      } catch (err) {
        if (err instanceof AutoAnnotateApiError && err.status === 404) {
          clearCachedJobDetail(jobId);
          setJobDetail(null);
          setAllItems([]);
          setLightboxItemId(null);
          void refreshJobs();
          if (
            typeof window !== "undefined" &&
            window.location.search.includes(encodeURIComponent(jobId))
          ) {
            window.history.replaceState(null, "", "/verify");
          }
          setActionError(
            `Verification job '${jobId}' was deleted or no longer exists on the server. Select an active job or create a new one on Auto-Annotate.`
          );
        } else {
          setActionError(
            err instanceof Error
              ? err.message
              : "Failed to load verification job."
          );
        }
      } finally {
        setLoading(false);
      }
    },
    [backendStatus, backendUrl, refreshJobs, selectedBatchId, setJobs]
  );

  useEffect(() => {
    if (selectedJobId) {
      void loadJob(selectedJobId);
    }
  }, [selectedJobId, loadJob]);

  const handleJobSwitch = (newJobId: string) => {
    setSelectedJobId(newJobId);
    prevDestJobIdRef.current = null;
    userPickedDestRef.current = false;
    setTargetProjectName("");
    setTargetFolderName("");
    setTargetAnnotationJobId("");
    setPushScopeMode("all");
    setPushResult(null);
    setPushError(null);
    setLightboxItemId(null);
    setReviewSaveState("idle");
    setReviewSaveError(null);
    if (typeof window !== "undefined") {
      window.history.replaceState(
        null,
        "",
        `/verify?job=${encodeURIComponent(newJobId)}`
      );
    }
  };

  // Batches matching batchStateFilter
  const visibleBatches = useMemo(() => {
    if (!jobDetail) return [];
    return jobDetail.batches.filter((b) => {
      if (batchStateFilter === "checked") return b.checked;
      if (batchStateFilter === "unchecked") return !b.checked;
      return true;
    });
  }, [jobDetail, batchStateFilter]);

  // Apply all filters cleanly to items
  const filteredItems = useMemo(() => {
    const allowedBatchIds = new Set(visibleBatches.map((b) => b.batchId));
    return allItems.filter((it) => {
      if (batchStateFilter !== "all" && !allowedBatchIds.has(it.batchId)) {
        return false;
      }
      if (selectedBatchId !== "all" && it.batchId !== selectedBatchId) {
        return false;
      }
      if (reasonFilter !== "all") {
        if (it.reviewStatus !== "rejected" || it.rejectReason !== reasonFilter) {
          return false;
        }
      }
      if (frameFilter === "pending" && it.reviewStatus !== "pending") {
        return false;
      }
      if (frameFilter === "accepted" && it.reviewStatus !== "accepted") {
        return false;
      }
      if (frameFilter === "rejected" && it.reviewStatus !== "rejected") {
        return false;
      }
      if (frameFilter === "with_detections" && it.detections.length === 0) {
        return false;
      }
      if (frameFilter === "no_detections" && it.detections.length > 0) {
        return false;
      }
      return true;
    });
  }, [
    allItems,
    visibleBatches,
    batchStateFilter,
    selectedBatchId,
    reasonFilter,
    frameFilter,
  ]);

  const activeBatchSummary = useMemo(
    () =>
      jobDetail?.batches.find((b) => b.batchId === selectedBatchId) ?? null,
    [jobDetail, selectedBatchId]
  );

  const pushStats = useMemo(() => {
    const scopeItems =
      pushScopeMode === "batch" && selectedBatchId !== "all"
        ? allItems.filter((it) => it.batchId === selectedBatchId)
        : allItems;
    const acceptedItems = scopeItems.filter((it) => it.reviewStatus === "accepted");
    const annotatedCount = acceptedItems.filter((it) => it.detections.length > 0).length;
    const nullCount = acceptedItems.length - annotatedCount;
    const rejectedCount = scopeItems.filter((it) => it.reviewStatus === "rejected").length;
    const pendingCount = scopeItems.filter((it) => it.reviewStatus === "pending").length;
    const eligibleAccepted = includeEmpty
      ? acceptedItems
      : acceptedItems.filter((it) => it.detections.length > 0);
    const unpushedEligible = eligibleAccepted.filter(
      (it) => it.pushStatus !== "pushed"
    );
    const pushNowCount =
      pushScopeMode === "unpushed"
        ? unpushedEligible.length
        : eligibleAccepted.length;
    return {
      acceptedTotal: acceptedItems.length,
      annotatedCount,
      nullCount,
      rejectedCount,
      pendingCount,
      unpushedCount: unpushedEligible.length,
      pushNowCount,
    };
  }, [allItems, includeEmpty, pushScopeMode, selectedBatchId]);

  const handleReviewItem = useCallback(
    async (
      itemId: string,
      reviewStatus: AutoReviewStatus,
      rejectReason?: AutoRejectReason | null,
      options?: { fromLightbox?: boolean }
    ) => {
      if (!jobDetail) return;
      setActionError(null);
      setReviewSaveError(null);
      setReviewSaveState("saving");

      const currentIdx = filteredItems.findIndex((it) => it.id === itemId);
      const nextAfterId =
        options?.fromLightbox &&
        autoAdvance &&
        currentIdx >= 0 &&
        currentIdx < filteredItems.length - 1
          ? filteredItems[currentIdx + 1].id
          : null;

      const snapshotJob = jobDetail;
      const snapshotItems = allItems;
      const cleanReason =
        reviewStatus === "rejected" ? rejectReason ?? null : null;

      const optimisticItems = allItems.map((it) =>
        it.id === itemId
          ? { ...it, reviewStatus, rejectReason: cleanReason }
          : it
      );
      const optimisticJob = recomputeJobCounts(jobDetail, optimisticItems);
      setAllItems(optimisticItems);
      setJobDetail(optimisticJob);
      setJobs((prev) =>
        prev.map((j) => (j.id === optimisticJob.id ? optimisticJob : j))
      );

      if (nextAfterId) {
        setLightboxItemId(nextAfterId);
      }

      try {
        const res = await patchAutoItemReview(backendUrl, jobDetail.id, itemId, {
          reviewStatus,
          rejectReason: cleanReason,
        });
        const mergedItems = optimisticItems.map((it) =>
          it.id === res.item.id ? res.item : it
        );
        setJobDetail(res.job);
        setAllItems(mergedItems);
        setJobs((prev) =>
          prev.map((j) => (j.id === res.job.id ? res.job : j))
        );
        saveCachedJobDetail(res.job.id, { job: res.job, items: mergedItems });
        setReviewSaveState("saved");
      } catch (err) {
        // Roll back optimistic state on error
        setJobDetail(snapshotJob);
        setAllItems(snapshotItems);
        setJobs((prev) =>
          prev.map((j) => (j.id === snapshotJob.id ? snapshotJob : j))
        );
        if (nextAfterId) {
          setLightboxItemId(itemId);
        }

        const msg =
          err instanceof AutoAnnotateApiError && err.status === 404
            ? `Job '${jobDetail.id}' no longer exists on the backend (stale URL or deleted). Refresh / select an active batch.`
            : err instanceof Error
              ? err.message
              : "Failed to save review state.";
        if (err instanceof AutoAnnotateApiError && err.status === 404) {
          clearCachedJobDetail(jobDetail.id);
          void refreshJobs();
        }
        setActionError(msg);
        setReviewSaveError(msg);
        setReviewSaveState("error");
      }
    },
    [allItems, autoAdvance, backendUrl, filteredItems, jobDetail, refreshJobs, setJobs]
  );

  const handleUpdateItemDetections = useCallback(
    async (updatedItem: AutoItemRecord) => {
      if (!jobDetail) return;
      setActionError(null);
      setReviewSaveError(null);
      setReviewSaveState("saving");
      const snapshotItems = allItems;
      const snapshotJob = jobDetail;

      const optimisticItems = allItems.map((it) =>
        it.id === updatedItem.id ? updatedItem : it
      );
      setAllItems(optimisticItems);

      try {
        const res = await patchAutoItemReview(
          backendUrl,
          jobDetail.id,
          updatedItem.id,
          {
            reviewStatus: updatedItem.reviewStatus,
            rejectReason: updatedItem.rejectReason ?? null,
            detections: updatedItem.detections,
          }
        );
        const mergedItems = optimisticItems.map((it) =>
          it.id === res.item.id ? res.item : it
        );
        setJobDetail(res.job);
        setAllItems(mergedItems);
        saveCachedJobDetail(res.job.id, { job: res.job, items: mergedItems });
        setReviewSaveState("saved");
      } catch (err) {
        setAllItems(snapshotItems);
        setJobDetail(snapshotJob);
        const msg =
          err instanceof Error ? err.message : "Failed to update annotations.";
        setActionError(msg);
        setReviewSaveError(msg);
        setReviewSaveState("error");
      }
    },
    [allItems, backendUrl, jobDetail]
  );

  const handleApproveBatch = async () => {
    if (!jobDetail || selectedBatchId === "all") return;
    setActionError(null);
    setReviewSaveState("saving");
    try {
      const res = await approveAutoBatch(
        backendUrl,
        jobDetail.id,
        selectedBatchId,
        true
      );
      const updatedMap = new Map(res.batchItems.map((it) => [it.id, it]));
      const nextItems = allItems.map((it) => updatedMap.get(it.id) ?? it);
      setJobDetail(res.job);
      setAllItems(nextItems);
      setJobs((prev) => prev.map((j) => (j.id === res.job.id ? res.job : j)));
      saveCachedJobDetail(res.job.id, { job: res.job, items: nextItems });
      setReviewSaveState("saved");
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to approve batch.";
      setActionError(msg);
      setReviewSaveState("error");
    }
  };

  const handleStepBatch = (dir: -1 | 1) => {
    if (!jobDetail || jobDetail.batches.length === 0) return;
    if (selectedBatchId === "all") {
      setSelectedBatchId(jobDetail.batches[0].batchId);
      return;
    }
    const idx = jobDetail.batches.findIndex(
      (b) => b.batchId === selectedBatchId
    );
    const nextIdx = Math.min(
      Math.max(idx + dir, 0),
      jobDetail.batches.length - 1
    );
    setSelectedBatchId(jobDetail.batches[nextIdx].batchId);
  };

  const handlePushAccepted = async () => {
    if (!jobDetail) return;
    setPushError(null);
    setPushResult(null);
    if (!pushKey.trim()) {
      setPushError("Roboflow Private API Key is required.");
      return;
    }
    if (pushDestination === "new_project" && !targetProjectName.trim()) {
      setPushError(
        "Target Project Name is required when pushing to a new project."
      );
      return;
    }
    setApiKey(pushKey.trim());
    setPushBusy(true);
    try {
      const res = await pushAcceptedZipApi(backendUrl, jobDetail.id, {
        apiKey: pushKey.trim(),
        workspaceSlug: jobDetail.workspaceSlug || undefined,
        pushDestination,
        targetProjectName:
          pushDestination === "new_project"
            ? targetProjectName.trim()
            : jobDetail.projectSlug,
        targetFolderName: targetFolderName.trim() || undefined,
        targetAnnotationJobId:
          pushDestination === "same_project_annotate"
            ? targetAnnotationJobId.trim() || undefined
            : undefined,
        addToDataset:
          pushDestination === "same_project_new_folder"
            ? newFolderAddToDataset
            : pushDestination !== "same_project_annotate",
        projectLicense,
        batchId:
          pushScopeMode === "batch" && selectedBatchId !== "all"
            ? selectedBatchId
            : undefined,
        includeEmpty,
        onlyUnpushed: pushScopeMode === "unpushed",
      });
      setPushResult(res);
      await loadJob(jobDetail.id);
      await refreshJobs();
    } catch (err) {
      setPushError(
        err instanceof Error ? err.message : "Failed to push verified images."
      );
    } finally {
      setPushBusy(false);
    }
  };

  const lightboxIndex = useMemo(
    () =>
      lightboxItemId
        ? filteredItems.findIndex((it) => it.id === lightboxItemId)
        : -1,
    [filteredItems, lightboxItemId]
  );
  // Fall back to allItems so reviewing a frame while filtered never abruptly vanishes the open modal
  const lightboxItem = useMemo(() => {
    if (!lightboxItemId) return null;
    if (lightboxIndex >= 0) return filteredItems[lightboxIndex];
    return allItems.find((it) => it.id === lightboxItemId) ?? null;
  }, [allItems, filteredItems, lightboxIndex, lightboxItemId]);

  if (!jobs.length && !jobDetail) {
    return (
      <div className="c1-card flex flex-col items-center justify-center p-12 text-center">
        <Sparkles className="h-8 w-8 text-[#F2853F]" />
        <h1 className="mt-3 text-xl font-semibold text-white">
          No Auto-Annotation Batches Yet
        </h1>
        <p className="mt-1.5 max-w-md text-sm text-[#A1A1AA]">
          Start a Roboflow batch job on Auto-Annotate first. Once images finish
          processing, review every 50-image batch here before pushing.
        </p>
        <Link
          href="/auto-annotate"
          className="mt-5 inline-flex items-center gap-2 rounded-full bg-[#F2853F] px-5 py-2.5 text-xs font-semibold text-black hover:bg-[#FF9652]"
        >
          Go to Auto-Annotate
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Compact Header Summary matching Screenshot 1 UX + Control One UI */}
      <div className="c1-card p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2.5">
              <p className="c1-kicker">05 / VERIFY AUTO-ANNOTATION</p>
              {jobs.length > 0 && (
                <select
                  value={selectedJobId}
                  onChange={(e) => handleJobSwitch(e.target.value)}
                  aria-label="Select Auto-Annotation Job"
                  className="font-mono-tech rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                >
                  {jobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.projectSlug} · {j.id} ({j.totalImages} imgs)
                    </option>
                  ))}
                </select>
              )}
            </div>

            {jobDetail && (
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                <span className="font-semibold text-white">
                  {jobDetail.projectName}
                </span>
                <span className="font-mono-tech text-xs text-emerald-400">
                  {jobDetail.acceptedCount} accepted
                </span>
                <span className="font-mono-tech text-xs text-rose-400">
                  {jobDetail.rejectedCount} rejected
                </span>
                <span className="font-mono-tech text-xs text-[#A1A1AA]">
                  {jobDetail.pendingCount} pending / {jobDetail.totalImages}{" "}
                  frames
                </span>
                <span className="font-mono-tech text-xs text-[#F2853F]">
                  {jobDetail.checkedBatchesCount} of {jobDetail.batches.length}{" "}
                  batches checked
                </span>
                {jobDetail.pushedCount > 0 && (
                  <span className="font-mono-tech text-xs text-sky-400">
                    {jobDetail.pushedCount} pushed to Roboflow
                  </span>
                )}
              </div>
            )}
          </div>

          {jobDetail && (
            <div className="flex flex-wrap items-center gap-2">
              {jobDetail.rejectedCount > 0 && (() => {
                const navigateRejected = (
                  route: string,
                  keys: { batchKey: string; reasonKey: string; fixed?: Record<string, string> }
                ) => {
                  const rejectedAll = allItems.filter(
                    (it) => it.reviewStatus === "rejected"
                  );
                  const batchMatched =
                    selectedBatchId !== "all"
                      ? rejectedAll.filter(
                          (it) => it.batchId === selectedBatchId
                        )
                      : rejectedAll;
                  const reasonMatched =
                    reasonFilter !== "all"
                      ? batchMatched.filter(
                          (it) => it.rejectReason === reasonFilter
                        )
                      : batchMatched;
                  const params = new URLSearchParams(keys.fixed);
                  if (selectedBatchId !== "all" && batchMatched.length > 0) {
                    params.set(keys.batchKey, selectedBatchId);
                  }
                  if (reasonFilter !== "all" && reasonMatched.length > 0) {
                    params.set(keys.reasonKey, reasonFilter);
                  }
                  router.push(`${route}?${params.toString()}`);
                };

                return (
                  <>
                    <button
                      type="button"
                      onClick={() =>
                        navigateRejected("/auto-annotate", {
                          batchKey: "retryBatch",
                          reasonKey: "retryReason",
                          fixed: { retryJob: jobDetail.id },
                        })
                      }
                      title="Send rejected images back to Auto-Annotate as a separate batch to re-run with another model & threshold"
                      className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/45 bg-rose-500/12 px-3.5 py-2 text-xs font-medium text-rose-200 transition-colors hover:border-[#F2853F]/70 hover:bg-[#F2853F]/15 hover:text-white"
                    >
                      <RotateCcw className="h-3.5 w-3.5 text-[#F2853F]" />
                      Re-Annotate Rejected ({jobDetail.rejectedCount})
                    </button>

                    <button
                      type="button"
                      onClick={() =>
                        navigateRejected("/annotate", {
                          batchKey: "batch",
                          reasonKey: "reason",
                          fixed: { job: jobDetail.id, filter: "rejected" },
                        })
                      }
                      title="Open rejected images inside the interactive Manual Annotator (Draw Boxes, Polygons, SAM 2.1 & OWL-ViT + SAM)"
                      className="inline-flex items-center gap-1.5 rounded-full border border-[#F2853F]/55 bg-[#F2853F]/15 px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-[#F2853F]/25"
                    >
                      <PenTool className="h-3.5 w-3.5 text-[#F2853F]" />
                      Annotate Manually ({jobDetail.rejectedCount})
                    </button>
                  </>
                );
              })()}

              {jobDetail.rejectedCount === 0 && (
                <button
                  type="button"
                  onClick={() => {
                    const params = new URLSearchParams({ job: jobDetail.id });
                    if (selectedBatchId !== "all") {
                      params.set("batch", selectedBatchId);
                    }
                    router.push(`/annotate?${params.toString()}`);
                  }}
                  title="Open Manual Annotator editor for this job"
                  className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#1A1A1E] px-3.5 py-2 text-xs font-medium text-white transition-colors hover:border-[#F2853F]"
                >
                  <PenTool className="h-3.5 w-3.5 text-[#F2853F]" />
                  Annotate Manually
                </button>
              )}

              <a
                href={getDownloadAcceptedZipUrl(
                  backendUrl,
                  jobDetail.id,
                  selectedBatchId
                )}
                className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#1A1A1E] px-3.5 py-2 text-xs font-medium text-white hover:border-white/35"
              >
                <Download className="h-3.5 w-3.5 text-[#F2853F]" />
                Saved Local ZIP
              </a>

              <button
                type="button"
                disabled={jobDetail.acceptedCount === 0}
                onClick={() => {
                  setPushResult(null);
                  setPushError(null);
                  if (!userPickedDestRef.current) {
                    setPushDestination(
                      resolveDefaultPushDestination(jobDetail, jobs)
                    );
                  }
                  if (
                    pushScopeMode === "unpushed" &&
                    pushStats.unpushedCount === 0
                  ) {
                    setPushScopeMode("all");
                  }
                  if (!targetProjectName) {
                    setTargetProjectName(`${jobDetail.projectSlug}-auto-annotated`);
                  }
                  if (!targetFolderName) {
                    setTargetFolderName(
                      jobDetail.sourceBatchId ||
                        jobDetail.sourceAnnotationJobId ||
                        `${jobDetail.id}-verified`
                    );
                  }
                  if (!targetAnnotationJobId && jobDetail.sourceAnnotationJobId) {
                    setTargetAnnotationJobId(jobDetail.sourceAnnotationJobId);
                  }
                  setPushModalOpen(true);
                }}
                className="inline-flex items-center gap-1.5 rounded-full bg-[#F2853F] px-4 py-2 text-xs font-semibold text-black transition-colors hover:bg-[#FF9652] disabled:cursor-not-allowed disabled:opacity-40"
              >
                <CloudUpload className="h-3.5 w-3.5" />
                Push Accepted ({jobDetail.acceptedCount})
              </button>
            </div>
          )}
        </div>

        {/* 3 Filter Dropdowns + Batch Stepper + Approve Batch Button (Screenshot 1 & 3 UX) */}
        {jobDetail && (
          <div className="mt-4 flex flex-wrap items-center gap-2.5 border-t border-[#222226] pt-3.5">
            {/* Batch Dropdown + Prev/Next Batch */}
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => handleStepBatch(-1)}
                aria-label="Previous batch"
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:border-white/35 hover:text-white"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <select
                value={selectedBatchId}
                onChange={(e) => setSelectedBatchId(e.target.value)}
                aria-label="Filter by 50-image batch"
                className="font-mono-tech h-8 min-w-[240px] rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
              >
                <option value="all">
                  All batches ({jobDetail.totalImages} frames)
                </option>
                {jobDetail.batches.map((b) => (
                  <option key={b.batchId} value={b.batchId}>
                    {b.checked ? "✓ " : ""}
                    {b.label} ({b.total}) · {b.accepted} acc / {b.rejected} rej
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => handleStepBatch(1)}
                aria-label="Next batch"
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:border-white/35 hover:text-white"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            {/* Frame Status Dropdown */}
            <select
              value={frameFilter}
              onChange={(e) => setFrameFilter(e.target.value)}
              aria-label="Filter frames"
              className="h-8 rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
            >
              <option value="all">All frames</option>
              <option value="pending">Pending review</option>
              <option value="accepted">Accepted only</option>
              <option value="rejected">Rejected only</option>
              <option value="with_detections">With detections</option>
              <option value="no_detections">No detections</option>
            </select>

            {/* Batch Checked Dropdown */}
            <select
              value={batchStateFilter}
              onChange={(e) => setBatchStateFilter(e.target.value)}
              aria-label="Filter batch approval state"
              className="h-8 rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
            >
              <option value="all">All batches</option>
              <option value="unchecked">Unchecked batches</option>
              <option value="checked">Checked batches</option>
            </select>

            {selectedBatchId !== "all" && (
              <button
                type="button"
                onClick={handleApproveBatch}
                className={`ml-auto inline-flex h-8 items-center gap-1.5 rounded-lg border px-3 text-xs font-medium transition-colors ${
                  activeBatchSummary?.checked
                    ? "border-emerald-500/45 bg-emerald-500/15 text-emerald-300"
                    : "border-emerald-500/50 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20"
                }`}
              >
                <CheckCheck className="h-3.5 w-3.5" />
                {activeBatchSummary?.checked
                  ? `Batch ${activeBatchSummary.batchId} Checked · Accept New Pending`
                  : `Approve Batch (Accept Unrejected in ${selectedBatchId})`}
              </button>
            )}
          </div>
        )}

        {/* Guidance line + Generic Reject-Reason Counter Chips + Class Color Legend */}
        {jobDetail && (
          <div className="mt-3 space-y-2.5">
            <p className="text-[11px] leading-relaxed text-[#8A8A8E]">
              Reject wrong frames on cards (`✕`) or double-click any image for
              full view + class color column (`A` Accept, `R`/`Space` Reject,
              keys `1–4` reject reasons, `←`/`→` move, `Esc` close). Approving a
              batch accepts remaining unrejected frames while keeping rejected
              frames excluded.
            </p>

            <div className="flex flex-wrap items-center justify-between gap-2">
              {/* Generic Reject Reason Chips */}
              <div className="flex flex-wrap items-center gap-1.5">
                {(Object.keys(REASON_LABELS) as AutoRejectReason[]).map(
                  (rk) => {
                    const cnt = jobDetail.rejectReasonCounts?.[rk] ?? 0;
                    const active = reasonFilter === rk;
                    return (
                      <button
                        key={rk}
                        type="button"
                        onClick={() =>
                          setReasonFilter((prev) => (prev === rk ? "all" : rk))
                        }
                        className={`font-mono-tech rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                          active
                            ? "border-[#F2853F] bg-[#F2853F]/15 text-[#F2853F]"
                            : "border-[#2A2A30] bg-[#0E0E11] text-[#A1A1AA] hover:border-white/30 hover:text-white"
                        }`}
                      >
                        {REASON_LABELS[rk]} · {cnt}
                      </button>
                    );
                  }
                )}
              </div>

              {/* Compact Class Color Legend Strip */}
              <div className="flex flex-wrap items-center gap-2">
                {jobDetail.classColors.map((cc) => (
                  <span
                    key={cc.name}
                    className="font-mono-tech inline-flex items-center gap-1.5 rounded-full border border-[#2C2C32] bg-[#0D0D10] px-2.5 py-0.5 text-[11px] text-[#D4D4D8]"
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: cc.color }}
                    />
                    {cc.name}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {actionError && (
        <p className="rounded-xl border border-rose-500/35 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
          {actionError}
        </p>
      )}

      {/* Contact-Sheet Grid (50 images/batch UX) */}
      {loading && !allItems.length ? (
        <div className="c1-card flex items-center justify-center p-16 text-sm text-[#A1A1AA]">
          <Loader2 className="mr-2 h-4 w-4 animate-spin text-[#F2853F]" />
          Loading batch images...
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="c1-card p-12 text-center text-sm text-[#8A8A8E]">
          No frames match the current batch / status filters.
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
          {filteredItems.map((item) => {
            const isAccepted = item.reviewStatus === "accepted";
            const isRejected = item.reviewStatus === "rejected";
            const borderClass = isAccepted
              ? "border-emerald-500/80 bg-[#0E1813]"
              : isRejected
                ? "border-rose-500/85 bg-[#190F12]"
                : "border-[#26262A] bg-[#141416] hover:border-[#42424A]";

            const badgeText = isAccepted
              ? "ACCEPTED"
              : isRejected
                ? item.rejectReason
                  ? REASON_LABELS[item.rejectReason]
                  : "REJECTED"
                : null;

            return (
              <div
                key={item.id}
                onDoubleClick={() => {
                  setReviewSaveError(null);
                  setReviewSaveState("idle");
                  setLightboxItemId(item.id);
                }}
                title="Double-click to inspect full image & class color legend"
                className={`group relative flex cursor-pointer flex-col overflow-hidden rounded-xl border transition-colors ${borderClass}`}
              >
                {/* Thumbnail Stage with Label-Free Color-Coded Overlays */}
                <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden bg-[#070709]">
                  <div
                    className="relative max-h-full max-w-full"
                    style={{
                      aspectRatio: `${Math.max(1, item.width)} / ${Math.max(1, item.height)}`,
                    }}
                  >
                    <img
                      src={item.imageUrl}
                      alt={item.filename}
                      loading="lazy"
                      className="block h-full w-full select-none object-contain"
                    />

                    {item.detections.length > 0 && (
                      <svg
                        viewBox="0 0 1000 1000"
                        preserveAspectRatio="none"
                        className="pointer-events-none absolute inset-0 h-full w-full"
                      >
                        {item.detections.map((det) => {
                          const color = getClassColor(
                            det.label,
                            jobDetail?.classColors ?? []
                          );
                          const showPoly =
                            jobDetail?.outputMode !== "bbox" &&
                            det.polygon &&
                            det.polygon.length >= 3;
                          const showBox =
                            jobDetail?.outputMode === "bbox" ||
                            jobDetail?.outputMode === "both" ||
                            !showPoly;
                          const [x1, y1, x2, y2] = det.bbox_2d;

                          return (
                            <g key={det.id}>
                              {showPoly && det.polygon && (
                                <polygon
                                  points={det.polygon
                                    .map((p) => `${p[0]},${p[1]}`)
                                    .join(" ")}
                                  fill={hexToRgba(color, 0.22)}
                                  stroke={color}
                                  strokeWidth={2.2}
                                  vectorEffect="non-scaling-stroke"
                                />
                              )}
                              {showBox && det.bbox_2d.length === 4 && (
                                <rect
                                  x={x1}
                                  y={y1}
                                  width={Math.max(1, x2 - x1)}
                                  height={Math.max(1, y2 - y1)}
                                  fill={
                                    showPoly
                                      ? "transparent"
                                      : hexToRgba(color, 0.1)
                                  }
                                  stroke={color}
                                  strokeWidth={2.2}
                                  vectorEffect="non-scaling-stroke"
                                />
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>

                  {/* Top-right status pill only (no user avatars) */}
                  {badgeText && (
                    <span
                      className={`font-mono-tech absolute right-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide shadow ${
                        isAccepted
                          ? "bg-emerald-500 text-black"
                          : "bg-rose-500 text-white"
                      }`}
                    >
                      {badgeText}
                    </span>
                  )}

                  {item.pushStatus === "pushed" && (
                    <span className="font-mono-tech absolute bottom-1.5 right-1.5 rounded bg-sky-500/90 px-1.5 py-0.5 text-[9px] font-semibold text-black">
                      PUSHED
                    </span>
                  )}
                </div>

                {/* Compact Footer with Filename + Explicit Accept / Reject Buttons */}
                <div className="flex items-center justify-between gap-1.5 border-t border-[#222226] px-2 py-1.5">
                  <div className="min-w-0">
                    <div className="truncate font-mono-tech text-[10px] text-[#D4D4D8]">
                      #{item.seqIndex} · {item.filename}
                    </div>
                    <div className="font-mono-tech text-[10px] text-[#8A8A8E]">
                      {item.inferenceStatus !== "completed"
                        ? item.inferenceStatus
                        : item.detections.length > 0
                          ? `${item.detections.length} det(s)`
                          : "nothing"}
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      type="button"
                      onDoubleClick={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleReviewItem(
                          item.id,
                          isAccepted ? "pending" : "accepted",
                          null
                        );
                      }}
                      aria-label={`Accept ${item.filename}`}
                      title={isAccepted ? "Set Pending" : "Accept frame"}
                      className={`inline-flex h-6 w-6 items-center justify-center rounded-md border transition-colors ${
                        isAccepted
                          ? "border-emerald-400 bg-emerald-500 text-black"
                          : "border-[#303035] bg-[#1C1C21] text-[#A1A1AA] hover:border-emerald-500/60 hover:text-emerald-300"
                      }`}
                    >
                      <Check className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onDoubleClick={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleReviewItem(
                          item.id,
                          isRejected ? "pending" : "rejected",
                          null
                        );
                      }}
                      aria-label={`Reject ${item.filename}`}
                      title={isRejected ? "Unreject frame" : "Reject frame"}
                      className={`inline-flex h-6 w-6 items-center justify-center rounded-md border transition-colors ${
                        isRejected
                          ? "border-rose-400 bg-rose-500 text-white"
                          : "border-[#303035] bg-[#1C1C21] text-[#A1A1AA] hover:border-rose-500/60 hover:text-rose-300"
                      }`}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Double-Click Full-Resolution Lightbox Modal */}
      {jobDetail && lightboxItem && (
        <VerifyLightboxModal
          job={jobDetail}
          item={lightboxItem}
          itemIndex={lightboxIndex}
          totalFiltered={filteredItems.length}
          classColors={jobDetail.classColors}
          saveState={reviewSaveState}
          saveError={reviewSaveError}
          autoAdvance={autoAdvance}
          onToggleAutoAdvance={() => setAutoAdvance((v) => !v)}
          onClose={() => setLightboxItemId(null)}
          onPrev={() => {
            if (lightboxIndex > 0) {
              setReviewSaveState("idle");
              setReviewSaveError(null);
              setLightboxItemId(filteredItems[lightboxIndex - 1].id);
            } else if (lightboxIndex === -1 && filteredItems.length > 0) {
              setLightboxItemId(filteredItems[0].id);
            }
          }}
          onNext={() => {
            if (lightboxIndex >= 0 && lightboxIndex < filteredItems.length - 1) {
              setReviewSaveState("idle");
              setReviewSaveError(null);
              setLightboxItemId(filteredItems[lightboxIndex + 1].id);
            } else if (lightboxIndex === -1 && filteredItems.length > 0) {
              setLightboxItemId(filteredItems[0].id);
            }
          }}
          onReview={handleReviewItem}
          onUpdateDetections={handleUpdateItemDetections}
        />
      )}

      {/* Destination-Aware Roboflow Push Modal */}
      {jobDetail && pushModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-xs"
          role="dialog"
          aria-modal="true"
          aria-label="Push Verified Annotations to Roboflow"
        >
          <div className="c1-card max-h-[92vh] w-full max-w-xl overflow-y-auto p-6 shadow-2xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="c1-kicker">ROBOFLOW PUSH DESTINATION</p>
                <h2 className="mt-1 text-lg font-semibold text-white">
                  Push Accepted Annotations to Roboflow
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setPushModalOpen(false)}
                aria-label="Close push modal"
                className="rounded-lg p-1 text-[#8A8A8E] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="c1-inset mt-4 grid grid-cols-3 gap-2 p-3 text-center">
              <div>
                <p className="font-mono-tech text-lg font-semibold text-emerald-400">
                  {pushStats.pushNowCount}
                </p>
                <p className="text-[11px] text-[#A1A1AA]">Accepted (Push)</p>
                {pushStats.nullCount > 0 && (
                  <p className="font-mono-tech mt-0.5 text-[10px] text-[#8A8A8E]">
                    {pushStats.annotatedCount} w/ boxes · {pushStats.nullCount} null
                  </p>
                )}
              </div>
              <div>
                <p className="font-mono-tech text-lg font-semibold text-rose-400">
                  {pushStats.rejectedCount}
                </p>
                <p className="text-[11px] text-[#A1A1AA]">Rejected (Excluded)</p>
              </div>
              <div>
                <p className="font-mono-tech text-lg font-semibold text-[#8A8A8E]">
                  {pushStats.pendingCount}
                </p>
                <p className="text-[11px] text-[#A1A1AA]">Pending (Excluded)</p>
              </div>
            </div>

            {jobDetail.lastPushProject &&
              jobDetail.lastPushProject !== jobDetail.projectSlug && (
                <div className="mt-3 rounded-xl border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-[11px] leading-snug text-amber-200">
                  Last push uploaded copies to separate project{" "}
                  <span className="font-mono-tech font-semibold text-white">
                    {jobDetail.lastPushProject}
                  </span>
                  . Original{" "}
                  <span className="font-mono-tech font-semibold text-white">
                    {jobDetail.projectSlug}
                  </span>{" "}
                  image IDs are still preserved — push with a{" "}
                  <span className="font-semibold text-white">Same Project</span>{" "}
                  destination and{" "}
                  <span className="font-semibold text-white">All Accepted</span>{" "}
                  below to update them in place.
                </div>
              )}

            <div className="mt-4">
              <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                Choose Push Destination *
              </span>
              <div className="mt-1.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {(
                  [
                    {
                      id: "same_project_dataset",
                      title: "Same Project · Save to Dataset",
                      desc: `Update '${jobDetail.projectSlug}' in-place keeping original image IDs & save to Dataset.`,
                    },
                    {
                      id: "same_project_annotate",
                      title: "Same Project · Stage in Annotate",
                      desc: "Update in-place keeping original image IDs & keep in fetched Job ID/folder.",
                    },
                    {
                      id: "same_project_new_folder",
                      title: "Same Project · New Folder",
                      desc: "Update in-place keeping original image IDs & move into a new folder/batch.",
                    },
                    {
                      id: "new_project",
                      title: "New Project (Copies)",
                      desc: "Upload new image copies into a separate project (assigns new image IDs).",
                    },
                  ] as const
                ).map((opt) => {
                  const active = pushDestination === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => {
                        userPickedDestRef.current = true;
                        setPushDestination(opt.id);
                        setPushError(null);
                        if (opt.id === "same_project_annotate") {
                          setTargetFolderName(
                            jobDetail.sourceBatchId ||
                              jobDetail.sourceAnnotationJobId ||
                              `${jobDetail.id}-annotate`
                          );
                          setTargetAnnotationJobId(
                            jobDetail.sourceAnnotationJobId || ""
                          );
                        } else if (
                          opt.id === "same_project_new_folder" ||
                          opt.id === "new_project"
                        ) {
                          setTargetFolderName(`${jobDetail.id}-verified`);
                        }
                      }}
                      className={`rounded-xl border p-2.5 text-left transition-colors ${
                        active
                          ? "border-[#F2853F] bg-[#F2853F]/12 text-white"
                          : "border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:border-white/30 hover:text-white"
                      }`}
                    >
                      <p className="text-xs font-semibold text-white">
                        {opt.title}
                      </p>
                      <p className="mt-0.5 text-[11px] leading-snug text-[#8A8A8E]">
                        {opt.desc}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {pushDestination !== "new_project" && (
              <div className="mt-3 rounded-xl border border-[#2C2C32] bg-[#0B0B0E] px-3 py-2 font-mono-tech text-[11px] text-[#A1A1AA]">
                <div>
                  Source Project:{" "}
                  <span className="text-white">{jobDetail.projectSlug}</span>
                </div>
                {(jobDetail.sourceAnnotationJobId || jobDetail.sourceBatchId) && (
                  <div className="mt-0.5 flex flex-wrap gap-x-3 text-[10px] text-[#8A8A8E]">
                    {jobDetail.sourceAnnotationJobId && (
                      <span>
                        Fetched Job ID:{" "}
                        <span className="text-[#F2853F]">
                          {jobDetail.sourceAnnotationJobId}
                        </span>
                      </span>
                    )}
                    {jobDetail.sourceBatchId && (
                      <span>
                        Fetched Folder/Batch:{" "}
                        <span className="text-white">
                          {jobDetail.sourceBatchId}
                        </span>
                      </span>
                    )}
                  </div>
                )}
                <p className="mt-1 font-sans text-[10px] text-emerald-300/85">
                  Updates annotations in-place using the exact Roboflow image IDs
                  originally fetched (no duplicate uploads).
                </p>
              </div>
            )}

            <div className="mt-3.5 space-y-3">
              {pushDestination === "new_project" && (
                <>
                  <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-[11px] leading-snug text-amber-200">
                    Uploads new image copies into{" "}
                    <span className="font-mono-tech font-semibold text-white">
                      {targetProjectName || `${jobDetail.projectSlug}-auto-annotated`}
                    </span>{" "}
                    with new Roboflow image IDs. Existing images in{" "}
                    <span className="font-mono-tech font-semibold text-white">
                      {jobDetail.projectSlug}
                    </span>{" "}
                    will NOT be updated.
                  </div>
                  <label className="block">
                    <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                      New Roboflow Target Project Name *
                    </span>
                    <input
                      type="text"
                      value={targetProjectName}
                      onChange={(e) => setTargetProjectName(e.target.value)}
                      className="font-mono-tech mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                    />
                  </label>
                </>
              )}

              {(pushDestination === "same_project_new_folder" ||
                pushDestination === "same_project_annotate" ||
                pushDestination === "new_project") && (
                <label className="block">
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                    {pushDestination === "same_project_new_folder"
                      ? "New Folder / Batch Name *"
                      : pushDestination === "same_project_annotate"
                        ? "Annotate Folder / Tag Name"
                        : "Target Batch / Folder Name"}
                  </span>
                  <input
                    type="text"
                    value={targetFolderName}
                    onChange={(e) => setTargetFolderName(e.target.value)}
                    placeholder={`${jobDetail.id}-verified`}
                    className="font-mono-tech mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                  />
                </label>
              )}

              {pushDestination === "same_project_annotate" && (
                <label className="block">
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                    Roboflow Annotation Job ID (same job or target job)
                  </span>
                  <input
                    type="text"
                    value={targetAnnotationJobId}
                    onChange={(e) => setTargetAnnotationJobId(e.target.value)}
                    placeholder={
                      jobDetail.sourceAnnotationJobId ||
                      "Optional annotation job ID..."
                    }
                    className="font-mono-tech mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                  />
                </label>
              )}

              {pushDestination === "same_project_new_folder" && (
                <div>
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                    Folder Placement
                  </span>
                  <div className="mt-1 grid grid-cols-2 rounded-xl border border-[#303035] bg-[#0A0A0C] p-1">
                    <button
                      type="button"
                      onClick={() => setNewFolderAddToDataset(true)}
                      className={`rounded-lg py-1.5 text-xs font-medium transition-colors ${
                        newFolderAddToDataset
                          ? "bg-[#F2853F] text-black"
                          : "text-[#A1A1AA] hover:text-white"
                      }`}
                    >
                      Save to Dataset
                    </button>
                    <button
                      type="button"
                      onClick={() => setNewFolderAddToDataset(false)}
                      className={`rounded-lg py-1.5 text-xs font-medium transition-colors ${
                        !newFolderAddToDataset
                          ? "bg-[#F2853F] text-black"
                          : "text-[#A1A1AA] hover:text-white"
                      }`}
                    >
                      Stage in Unassigned
                    </button>
                  </div>
                </div>
              )}

              <label className="block">
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Roboflow Private API Key *
                </span>
                <input
                  type="password"
                  value={pushKey}
                  onChange={(e) => setPushKey(e.target.value)}
                  placeholder="rf_xxxxxxxxxxxxxxxx"
                  className="font-mono-tech mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                />
              </label>

              <div
                className={`grid gap-3 ${
                  pushDestination === "new_project"
                    ? "grid-cols-2"
                    : "grid-cols-1"
                }`}
              >
                <label className="block">
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                    Push Scope
                  </span>
                  <select
                    value={pushScopeMode}
                    onChange={(e) =>
                      setPushScopeMode(
                        e.target.value as "all" | "batch" | "unpushed"
                      )
                    }
                    className="mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                  >
                    <option value="all">
                      All Accepted ({includeEmpty ? pushStats.acceptedTotal : pushStats.annotatedCount})
                    </option>
                    {jobDetail.pushedCount > 0 && pushStats.unpushedCount > 0 && (
                      <option value="unpushed">
                        Unpushed Accepted Only ({pushStats.unpushedCount} remaining)
                      </option>
                    )}
                    {selectedBatchId !== "all" && (
                      <option value="batch">
                        Current Batch Only ({selectedBatchId})
                      </option>
                    )}
                  </select>
                </label>

                {pushDestination === "new_project" && (
                  <label className="block">
                    <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                      Project License
                    </span>
                    <select
                      value={projectLicense}
                      onChange={(e) =>
                        setProjectLicense(e.target.value as "MIT" | "private")
                      }
                      className="mt-1 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                    >
                      <option value="MIT">MIT (Standard / Free tier)</option>
                      <option value="private">Private (Paid workspace)</option>
                    </select>
                  </label>
                )}
              </div>

              <label className="flex cursor-pointer items-start gap-2.5 rounded-xl border border-[#2C2C32] bg-[#0B0B0E] px-3 py-2.5">
                <input
                  type="checkbox"
                  checked={includeEmpty}
                  onChange={(e) => setIncludeEmpty(e.target.checked)}
                  className="mt-0.5 h-3.5 w-3.5 accent-[#F2853F]"
                />
                <span className="text-xs leading-snug text-[#D4D4D8]">
                  Include empty accepted images as{" "}
                  <span className="font-semibold text-white">Null</span>{" "}
                  annotations{" "}
                  {pushStats.nullCount > 0 && (
                    <span className="font-mono-tech text-[#F2853F]">
                      ({pushStats.nullCount} empty accepted)
                    </span>
                  )}
                  <span className="mt-0.5 block text-[11px] text-[#8A8A8E]">
                    Pushes accepted 0-detection background images as explicit Null
                    samples for balanced negative training.
                  </span>
                </span>
              </label>
            </div>

            {pushError && (
              <p className="mt-3 rounded-xl border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
                {pushError}
              </p>
            )}

            {pushResult && (
              <div className="mt-3 rounded-xl border border-emerald-500/35 bg-emerald-500/10 p-3 text-xs text-emerald-200">
                <p className="font-semibold">{pushResult.message}</p>
                {pushResult.taskId && (
                  <p className="font-mono-tech mt-1 text-[11px]">
                    Roboflow ZIP Task ID: {pushResult.taskId}
                  </p>
                )}
                <p className="font-mono-tech mt-1 truncate text-[11px] text-emerald-300/80">
                  Local backup ZIP: {pushResult.localZipPath}
                </p>
              </div>
            )}

            <div className="mt-5 flex items-center justify-end gap-2.5">
              <button
                type="button"
                onClick={() => setPushModalOpen(false)}
                className="rounded-full border border-[#303035] px-4 py-2 text-xs text-[#A1A1AA] hover:text-white"
              >
                Close
              </button>
              <button
                type="button"
                disabled={pushBusy || pushStats.pushNowCount === 0}
                onClick={handlePushAccepted}
                className="inline-flex items-center gap-2 rounded-full bg-[#F2853F] px-5 py-2 text-xs font-semibold text-black hover:bg-[#FF9652] disabled:opacity-50"
              >
                {pushBusy ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <CloudUpload className="h-3.5 w-3.5" />
                )}
                {pushBusy
                  ? `Pushing ${pushStats.pushNowCount} Frame(s)...`
                  : pushDestination === "new_project"
                    ? `Push ${pushStats.pushNowCount} to New Project`
                    : pushDestination === "same_project_dataset"
                      ? `Save ${pushStats.pushNowCount} to Dataset`
                      : pushDestination === "same_project_annotate"
                        ? `Stage ${pushStats.pushNowCount} in Annotate`
                        : `Save ${pushStats.pushNowCount} to New Folder`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
