"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowRight,
  Check,
  Layers,
  Loader2,
  Palette,
  Pause,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { useWorkbench } from "@/lib/workbench-store";
import {
  controlAutoJob,
  deleteAutoJobApi,
  fetchAutoJobDetail,
  streamAutoJobProgress,
  submitAutoJob,
} from "./api";
import {
  buildDefaultClassColors,
  DEFAULT_PALETTE,
  isPlaceholderClassList,
  syncClassColorsFromPrompt,
} from "./color-utils";
import {
  clearCachedJobDetail,
  loadAutoDraft,
  loadCachedJobDetail,
  saveAutoDraft,
} from "./cache";
import { RoboflowSourceCard } from "./RoboflowSourceCard";
import { useAutoAnnotate } from "./store";
import type {
  AutoItemRecord,
  AutoJobProgressEvent,
  AutoJobRecord,
  AutoOutputMode,
  AutoShotMode,
  ClassColorEntry,
  RoboflowInspectResponse,
} from "./types";

interface LocalRefItem {
  id: string;
  file: File;
  previewUrl: string;
  note: string;
}

const DEFAULT_AA_PROMPT =
  'Detect all instances of the target classes in the image and output normalized [0..1000] JSON annotations.';

export function AutoAnnotateView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const retryJobParam = searchParams.get("retryJob") || "";
  const retryBatchParam = searchParams.get("retryBatch") || "";
  const retryReasonParam = searchParams.get("retryReason") || "";

  const { backendUrl, backendStatus, setActiveEngineDisplay } = useWorkbench();
  const {
    apiKey,
    workspaceSlug,
    setWorkspaceSlug,
    projectSlug,
    setProjectSlug,
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
    refreshModels,
    sessionEpoch,
  } = useAutoAnnotate();

  const [outputMode, setOutputMode] = useState<AutoOutputMode>("both");
  const [modelId, setModelId] = useState<string>("mlx-4b-8bit-base");
  const [samRefinerId, setSamRefinerId] = useState<string>("sam2.1-t");
  const [shotMode, setShotMode] = useState<AutoShotMode>("zero-shot");

  const [prompt, setPrompt] = useState<string>(DEFAULT_AA_PROMPT);
  const [styleNote, setStyleNote] = useState<string>("");
  const [classColors, setClassColors] = useState<ClassColorEntry[]>([
    { name: "object", color: DEFAULT_PALETTE[0] },
  ]);
  const [newClassName, setNewClassName] = useState("");

  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [maxTokens, setMaxTokens] = useState<number>(1024);

  const [refImages, setRefImages] = useState<LocalRefItem[]>([]);
  const refInputRef = useRef<HTMLInputElement | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<AutoJobRecord | null>(null);
  const [liveProgress, setLiveProgress] =
    useState<AutoJobProgressEvent | null>(null);
  const sseRef = useRef<(() => void) | null>(null);
  const [draftReady, setDraftReady] = useState(false);

  // Rejected-batch retry mode state (when sent back from /verify)
  const [retrySourceJob, setRetrySourceJob] = useState<AutoJobRecord | null>(null);
  const [retryRejectedItems, setRetryRejectedItems] = useState<AutoItemRecord[]>([]);
  const [selectedRetryIds, setSelectedRetryIds] = useState<string[]>([]);
  const [retryLoading, setRetryLoading] = useState(false);
  const [retryLoadError, setRetryLoadError] = useState<string | null>(null);
  const loadedRetryKeyRef = useRef<string>("");

  // Restore non-workspace inference draft settings cleanly on mount
  useEffect(() => {
    const saved = loadAutoDraft();
    if (saved) {
      if (saved.modelId) setModelId(saved.modelId);
      if (saved.samRefinerId) setSamRefinerId(saved.samRefinerId);
      if (saved.shotMode) setShotMode(saved.shotMode);
      if (saved.outputMode) setOutputMode(saved.outputMode);
      if (typeof saved.prompt === "string" && saved.prompt) setPrompt(saved.prompt);
      if (typeof saved.styleNote === "string") setStyleNote(saved.styleNote);
      if (Array.isArray(saved.classColors) && saved.classColors.length > 0) {
        setClassColors(saved.classColors);
      }
      if (typeof saved.confidenceThreshold === "number") {
        setConfidenceThreshold(saved.confidenceThreshold);
      }
      if (typeof saved.temperature === "number") setTemperature(saved.temperature);
      if (typeof saved.maxTokens === "number") setMaxTokens(saved.maxTokens);
    }
    setDraftReady(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Wipe local job stream & UI errors when backend restarts or goes offline
  const prevEpochRef = useRef(sessionEpoch);
  useEffect(() => {
    if (sessionEpoch === 0 || prevEpochRef.current === sessionEpoch) return;
    prevEpochRef.current = sessionEpoch;
    sseRef.current?.();
    setActiveJob(null);
    setLiveProgress(null);
    setSubmitError(null);
  }, [sessionEpoch]);

  // Hydrate rejected batch when navigated from Verify (`?retryJob=<id>`)
  useEffect(() => {
    if (!draftReady || !retryJobParam) return;
    const key = `${retryJobParam}|${retryBatchParam}|${retryReasonParam}|${backendStatus}`;
    if (loadedRetryKeyRef.current === key) return;

    const applyRetrySource = (srcJob: AutoJobRecord, srcItems: AutoItemRecord[]) => {
      const allRejected = srcItems.filter((it) => it.reviewStatus === "rejected");
      let preselected = allRejected;
      if (retryBatchParam) {
        const bm = preselected.filter((it) => it.batchId === retryBatchParam);
        if (bm.length > 0) preselected = bm;
      }
      if (retryReasonParam) {
        const rm = preselected.filter((it) => it.rejectReason === retryReasonParam);
        if (rm.length > 0) preselected = rm;
      }

      setRetrySourceJob(srcJob);
      setRetryRejectedItems(allRejected);
      setSelectedRetryIds(preselected.map((it) => it.id));
      setSubmitError(null);
      setRetryLoadError(null);

      if (srcJob.workspaceSlug) setWorkspaceSlug(srcJob.workspaceSlug);
      if (srcJob.projectSlug) setProjectSlug(srcJob.projectSlug);
      setOutputMode(srcJob.outputMode);
      if (srcJob.prompt) setPrompt(srcJob.prompt);
      setStyleNote(srcJob.styleNote || "");
      if (srcJob.classColors?.length) {
        setClassColors(srcJob.classColors);
      } else if (srcJob.classes?.length) {
        setClassColors(buildDefaultClassColors(srcJob.classes));
      }
      setConfidenceThreshold(srcJob.confidenceThreshold);
      setTemperature(srcJob.temperature);
      setMaxTokens(srcJob.maxTokens);
      if (srcJob.samRefinerId) setSamRefinerId(srcJob.samRefinerId);
      if (srcJob.modelId) setModelId(srcJob.modelId);
    };

    const cached = loadCachedJobDetail(retryJobParam);
    if (cached && cached.job?.id === retryJobParam) {
      applyRetrySource(cached.job, cached.items);
    }

    if (backendStatus !== "online") return;
    loadedRetryKeyRef.current = key;
    let cancelled = false;
    setRetryLoading(true);
    setRetryLoadError(null);
    void fetchAutoJobDetail(backendUrl, retryJobParam)
      .then((res) => {
        if (cancelled) return;
        applyRetrySource(res.job, res.items);
      })
      .catch((err) => {
        if (cancelled) return;
        setRetryLoadError(
          err instanceof Error
            ? err.message
            : `Failed to load rejected images from ${retryJobParam}.`
        );
      })
      .finally(() => {
        if (!cancelled) setRetryLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftReady, retryJobParam, retryBatchParam, retryReasonParam, backendStatus, backendUrl]);

  const clearRetryMode = () => {
    setRetrySourceJob(null);
    setRetryRejectedItems([]);
    setSelectedRetryIds([]);
    setRetryLoadError(null);
    loadedRetryKeyRef.current = "";
    if (retryJobParam) {
      router.replace("/auto-annotate");
    }
  };

  // Persist non-workspace inference draft settings in sessionStorage
  useEffect(() => {
    if (!draftReady) return;
    saveAutoDraft({
      modelId,
      samRefinerId,
      shotMode,
      outputMode,
      prompt,
      styleNote,
      classColors,
      confidenceThreshold,
      temperature,
      maxTokens,
    });
  }, [
    draftReady,
    modelId,
    samRefinerId,
    shotMode,
    outputMode,
    prompt,
    styleNote,
    classColors,
    confidenceThreshold,
    temperature,
    maxTokens,
  ]);

  // Restore any active / running job & reconnect SSE after reload
  useEffect(() => {
    if (activeJob || !jobs.length) return;
    const runningOrRecent =
      jobs.find(
        (j) =>
          j.status === "fetching" ||
          j.status === "queued" ||
          j.status === "running" ||
          j.status === "paused"
      ) ?? jobs[0];
    if (runningOrRecent) {
      setActiveJob(runningOrRecent);
      if (
        runningOrRecent.status === "fetching" ||
        runningOrRecent.status === "queued" ||
        runningOrRecent.status === "running"
      ) {
        attachJobStream(runningOrRecent.id);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs]);

  useEffect(() => {
    return () => {
      sseRef.current?.();
    };
  }, []);

  const detectors = useMemo(() => {
    if (models?.detectors?.length) return models.detectors;
    return [
      {
        id: "mlx-4b-8bit-base",
        label: "MLX Qwen3-VL 4B (8-bit) — Base Only",
        shortLabel: "MLX 4B-8bit Base",
        engine: "MLX-VLM",
        quantization: "8-bit affine",
        available: true,
        supportsPolygon: true,
        supportsFewShot: true,
        note: "Preview",
      },
      {
        id: "yoloe-26s-seg",
        label: "YOLOE-26s-seg (PyTorch · Open-Vocab)",
        shortLabel: "YOLOE-26s-seg",
        engine: "PyTorch MPS",
        quantization: "FP16/FP32 MPS",
        available: true,
        supportsPolygon: true,
        supportsFewShot: false,
        note: "Preview",
      },
      {
        id: "owlv2-base-patch16-ensemble",
        label: "OWLv2 Base Patch16 Ensemble (PyTorch · Open-Vocab)",
        shortLabel: "OWLv2 Base P16",
        engine: "OWLv2 PyTorch MPS",
        quantization: "FP32 MPS/CPU",
        available: true,
        supportsPolygon: true,
        supportsFewShot: false,
        note: "Cached locally",
      },
      {
        id: "owlv2-large-patch14-ensemble",
        label: "OWLv2 Large Patch14 Ensemble (PyTorch · High Precision)",
        shortLabel: "OWLv2 Large P14",
        engine: "OWLv2 PyTorch MPS",
        quantization: "FP32 MPS/CPU",
        available: true,
        supportsPolygon: true,
        supportsFewShot: false,
        note: "Downloads HF Hub weights (~1.7 GB)",
      },
    ];
  }, [models]);

  const samRefiners = useMemo(() => {
    if (models?.samRefiners?.length) return models.samRefiners;
    return [
      {
        id: "none",
        label: "None (Use Detector Output)",
        available: true,
        description: "Native output only",
      },
      {
        id: "sam2.1-t",
        label: "SAM 2.1 Tiny (Fast Polygon Refiner)",
        available: true,
        description: "Fast box-to-polygon refiner",
      },
    ];
  }, [models]);

  const selectedDetector = useMemo(
    () => detectors.find((d) => d.id === modelId) ?? detectors[0],
    [detectors, modelId]
  );

  useEffect(() => {
    if (!selectedDetector) return;
    setActiveEngineDisplay({
      id: selectedDetector.id,
      shortLabel: selectedDetector.shortLabel,
      engine: selectedDetector.engine,
      quantization: selectedDetector.quantization,
    });
  }, [selectedDetector, setActiveEngineDisplay]);

  useEffect(() => {
    if (!models?.detectors?.length || !selectedDetector) return;
    if (!selectedDetector.available) {
      const fallback = models.detectors.find((d) => d.available);
      if (fallback && fallback.id !== modelId) {
        setModelId(fallback.id);
      }
    }
  }, [modelId, models?.detectors, selectedDetector]);

  const isFlorenceDetector = useMemo(() => {
    const idLow = (selectedDetector?.id || "").toLowerCase();
    const engLow = (selectedDetector?.engine || "").toLowerCase();
    return idLow.includes("florence") || engLow.includes("florence");
  }, [selectedDetector]);

  const isOwlv2Detector = useMemo(() => {
    const idLow = (selectedDetector?.id || "").toLowerCase();
    const engLow = (selectedDetector?.engine || "").toLowerCase();
    return idLow.includes("owlv2") || engLow.includes("owlv2");
  }, [selectedDetector]);

  const isYoloeDetector = useMemo(() => {
    const idLow = (selectedDetector?.id || "").toLowerCase();
    return idLow.includes("yoloe");
  }, [selectedDetector]);

  // Enforce zero-shot when selected detector uses open-vocab class prompts (e.g. YOLOE)
  useEffect(() => {
    if (selectedDetector && !selectedDetector.supportsFewShot && shotMode === "few-shot") {
      setShotMode("zero-shot");
    }
  }, [selectedDetector, shotMode]);

  const handleInspectSuccess = (info: RoboflowInspectResponse) => {
    if (info.classes.length > 0) {
      setClassColors((prev) => {
        const synced = syncClassColorsFromPrompt(
          prompt,
          buildDefaultClassColors(info.classes, prev)
        );
        return synced.length ? synced : buildDefaultClassColors(info.classes);
      });
    }
    if (info.projectType === "object-detection" && outputMode === "both") {
      setOutputMode("bbox");
    } else if (info.projectType === "instance-segmentation") {
      setOutputMode("both");
    }
  };

  const handleSyncColorsFromPrompt = () => {
    setClassColors((prev) => syncClassColorsFromPrompt(prompt, prev));
  };

  const handleAddClass = () => {
    const clean = newClassName.trim();
    if (!clean) return;
    if (
      classColors.some((c) => c.name.toLowerCase() === clean.toLowerCase())
    ) {
      setNewClassName("");
      return;
    }
    const nextColor = DEFAULT_PALETTE[classColors.length % DEFAULT_PALETTE.length];
    setClassColors((prev) => [...prev, { name: clean, color: nextColor }]);
    setNewClassName("");
  };

  const handleRefFilesPicked = (files: FileList | null) => {
    if (!files || !files.length) return;
    const incoming: LocalRefItem[] = [];
    Array.from(files)
      .slice(0, Math.max(0, 6 - refImages.length))
      .forEach((file) => {
        if (!file.type.startsWith("image/")) return;
        incoming.push({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          file,
          previewUrl: URL.createObjectURL(file),
          note: "",
        });
      });
    setRefImages((prev) => [...prev, ...incoming]);
  };

  const attachJobStream = (jobId: string) => {
    sseRef.current?.();
    sseRef.current = streamAutoJobProgress(
      backendUrl,
      jobId,
      (evt) => {
        setLiveProgress(evt);
        if (evt.job) {
          setActiveJob(evt.job);
          setJobs((prev) => {
            const exists = prev.some((j) => j.id === evt.job!.id);
            if (!exists) return [evt.job!, ...prev];
            return prev.map((j) => (j.id === evt.job!.id ? evt.job! : j));
          });
        }
      },
      (errMsg) => {
        setSubmitError(errMsg);
      }
    );
  };

  const handleStartBatch = async () => {
    setSubmitError(null);
    if (backendStatus !== "online") {
      setSubmitError("Local FastAPI server is offline. Start backend on port 8000 first.");
      return;
    }
    const isRetryMode = Boolean(retrySourceJob);
    if (isRetryMode) {
      if (!selectedRetryIds.length) {
        setSubmitError("Select at least one rejected image to re-annotate.");
        return;
      }
    } else if (!apiKey.trim() || !projectSlug.trim()) {
      setSubmitError("Roboflow API Key and Project Slug are required.");
      return;
    }

    const effectiveClassColors = isPlaceholderClassList(classColors)
      ? syncClassColorsFromPrompt(prompt, classColors)
      : classColors;
    if (effectiveClassColors !== classColors && !isPlaceholderClassList(effectiveClassColors)) {
      setClassColors(effectiveClassColors);
    }

    const activeClasses = effectiveClassColors
      .map((c) => c.name.trim())
      .filter(Boolean);
    if (!activeClasses.length) {
      setSubmitError("Add at least one target class name before starting.");
      return;
    }

    setSubmitting(true);
    try {
      const fd = new FormData();
      fd.append("apiKey", apiKey.trim());
      fd.append(
        "projectSlug",
        isRetryMode && retrySourceJob ? retrySourceJob.projectSlug : projectSlug.trim()
      );
      fd.append(
        "workspaceSlug",
        isRetryMode && retrySourceJob
          ? retrySourceJob.workspaceSlug
          : workspaceSlug.trim()
      );
      fd.append("fetchScope", isRetryMode ? "rejected_retry" : fetchScope);
      fd.append(
        "annotationStatusFilter",
        !isRetryMode &&
          (fetchScope === "annotation_job" || fetchScope === "dataset")
          ? annotationStatusFilter
          : "all"
      );
      fd.append(
        "sourceBatchId",
        isRetryMode
          ? retrySourceJob?.sourceBatchId || ""
          : fetchScope === "batch"
            ? sourceBatchId
            : ""
      );
      fd.append(
        "sourceAnnotationJobId",
        isRetryMode
          ? retrySourceJob?.sourceAnnotationJobId || ""
          : fetchScope === "annotation_job"
            ? sourceAnnotationJobId
            : ""
      );
      fd.append(
        "assigneeEmail",
        isRetryMode
          ? retrySourceJob?.assigneeEmail || ""
          : fetchScope === "annotation_job"
            ? assigneeEmail.trim()
            : ""
      );
      fd.append("sourceJobId", isRetryMode && retrySourceJob ? retrySourceJob.id : "");
      fd.append(
        "sourceItemIdsJson",
        isRetryMode ? JSON.stringify(selectedRetryIds) : "[]"
      );
      fd.append(
        "maxImages",
        String(isRetryMode ? selectedRetryIds.length : maxImages)
      );
      fd.append("batchSize", String(batchSize));
      fd.append("modelId", modelId);
      fd.append(
        "samRefinerId",
        outputMode === "bbox" ? "none" : samRefinerId
      );
      fd.append(
        "shotMode",
        selectedDetector?.supportsFewShot ? shotMode : "zero-shot"
      );
      fd.append("outputMode", outputMode);
      fd.append("prompt", prompt.trim());
      fd.append("styleNote", styleNote.trim());
      fd.append("classesJson", JSON.stringify(activeClasses));
      fd.append("classColorsJson", JSON.stringify(effectiveClassColors));
      fd.append("confidenceThreshold", String(confidenceThreshold));
      fd.append("temperature", String(temperature));
      fd.append("maxTokens", String(maxTokens));

      if (shotMode === "few-shot" && selectedDetector?.supportsFewShot) {
        fd.append(
          "refNotesJson",
          JSON.stringify(refImages.map((r) => r.note))
        );
        refImages.forEach((r) => {
          fd.append("ref_files", r.file, r.file.name);
        });
      } else {
        fd.append("refNotesJson", "[]");
      }

      const job = await submitAutoJob(backendUrl, fd);
      setActiveJob(job);
      setJobs((prev) => [job, ...prev.filter((j) => j.id !== job.id)]);
      attachJobStream(job.id);
      if (isRetryMode) {
        clearRetryMode();
      }
    } catch (err) {
      setSubmitError(
        err instanceof Error ? err.message : "Failed to start auto-annotation job."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleControl = async (
    jobId: string,
    action: "start" | "pause" | "cancel"
  ) => {
    try {
      const updated = await controlAutoJob(backendUrl, jobId, action);
      setActiveJob(updated);
      setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)));
      if (action === "start") attachJobStream(jobId);
    } catch (err) {
      setSubmitError(
        err instanceof Error ? err.message : "Control command failed."
      );
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await deleteAutoJobApi(backendUrl, jobId);
      clearCachedJobDetail(jobId);
      setJobs((prev) => prev.filter((j) => j.id !== jobId));
      if (activeJob?.id === jobId) {
        setActiveJob(null);
        setLiveProgress(null);
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Delete failed.");
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[#1E1E22] pb-5">
        <div>
          <p className="c1-kicker">04 / ROBOFLOW AUTO-ANNOTATION PIPELINE</p>
          <h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-white">
            Dataset-Agnostic Local Auto-Annotate
          </h1>
          <p className="mt-1 text-sm text-[#A1A1AA]">
            Fetch images from any Roboflow project, auto-annotate in 50-image batches
            locally with Qwen3-VL or YOLOE + SAM 2.1, then review Accept / Reject
            before bulk-pushing to a clean Roboflow project.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => {
              void refreshModels();
              void refreshJobs();
            }}
            className="inline-flex items-center gap-2 rounded-full border border-[#303035] bg-[#141416] px-4 py-2 text-xs font-medium text-[#A1A1AA] transition-colors hover:border-white/35 hover:text-white"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh Status
          </button>
          <Link
            href="/verify"
            className="inline-flex items-center gap-2 rounded-full border border-[#F2853F]/50 bg-[#F2853F]/10 px-4 py-2 text-xs font-medium text-[#F2853F] transition-colors hover:bg-[#F2853F]/20"
          >
            Open Verify Auto-Annotation
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        {/* Left Column: Roboflow Connection + Model & Class Colors */}
        <div className="space-y-6 xl:col-span-7">
          {/* Card 01: Roboflow Source OR Rejected Retry Batch */}
          {retrySourceJob || retryLoading || retryLoadError ? (
            <section className="c1-card border-[#F2853F]/45 p-5 sm:p-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="c1-kicker text-[#F2853F]">
                    01 / REJECTED IMAGES RETRY BATCH
                  </p>
                  <h2 className="mt-1 flex items-center gap-2 text-base font-medium text-white">
                    <RotateCcw className="h-4 w-4 text-[#F2853F]" />
                    Re-Annotate Rejected Frames as New Batch
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={clearRetryMode}
                  className="rounded-full border border-[#303035] bg-[#141416] px-3 py-1.5 text-xs text-[#A1A1AA] hover:border-white/35 hover:text-white"
                >
                  Exit Retry Mode
                </button>
              </div>

              {retryLoading && !retrySourceJob && (
                <div className="mt-4 flex items-center gap-2 rounded-xl border border-[#2C2C32] bg-[#0B0B0E] p-4 text-xs text-[#A1A1AA]">
                  <Loader2 className="h-4 w-4 animate-spin text-[#F2853F]" />
                  Loading rejected images from {retryJobParam}...
                </div>
              )}

              {retryLoadError && (
                <p className="mt-3 rounded-xl border border-rose-500/35 bg-rose-500/10 px-3.5 py-2 text-xs text-rose-300">
                  {retryLoadError}
                </p>
              )}

              {retrySourceJob && (
                <>
                  <div className="c1-inset mt-4 flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 text-xs">
                    <div className="space-y-0.5">
                      <div className="font-mono-tech text-white">
                        Source:{" "}
                        <span className="text-[#F2853F]">
                          {retrySourceJob.projectSlug}
                        </span>{" "}
                        · {retrySourceJob.id}
                      </div>
                      <div className="font-mono-tech text-[11px] text-[#8A8A8E]">
                        Previous run: {retrySourceJob.modelLabel} @ Conf{" "}
                        {retrySourceJob.confidenceThreshold.toFixed(2)} — pick a
                        new model &amp; threshold below
                      </div>
                    </div>
                    <span className="font-mono-tech rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] text-emerald-300">
                      Local clone · No API key needed
                    </span>
                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
                      Selected{" "}
                      <span className="font-semibold text-white">
                        {selectedRetryIds.length}
                      </span>{" "}
                      of {retryRejectedItems.length} rejected frame(s)
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          setSelectedRetryIds(
                            retryRejectedItems.map((it) => it.id)
                          )
                        }
                        className="rounded-lg border border-[#303035] bg-[#141416] px-2.5 py-1 text-[11px] text-white hover:border-[#F2853F]/50"
                      >
                        Select All ({retryRejectedItems.length})
                      </button>
                      <button
                        type="button"
                        onClick={() => setSelectedRetryIds([])}
                        className="rounded-lg border border-[#303035] bg-[#141416] px-2.5 py-1 text-[11px] text-[#A1A1AA] hover:text-white"
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  {retryRejectedItems.length === 0 ? (
                    <p className="mt-3 rounded-xl border border-dashed border-[#303035] p-5 text-center text-xs text-[#8A8A8E]">
                      No rejected images found in {retrySourceJob.id}.
                    </p>
                  ) : (
                    <div className="mt-3 grid max-h-64 grid-cols-2 gap-2.5 overflow-y-auto pr-1 sm:grid-cols-3 md:grid-cols-4">
                      {retryRejectedItems.map((item) => {
                        const checked = selectedRetryIds.includes(item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            onClick={() =>
                              setSelectedRetryIds((prev) =>
                                prev.includes(item.id)
                                  ? prev.filter((id) => id !== item.id)
                                  : [...prev, item.id]
                              )
                            }
                            className={`group relative flex flex-col overflow-hidden rounded-xl border text-left transition-colors ${
                              checked
                                ? "border-[#F2853F] bg-[#1C140F]"
                                : "border-[#2A2A30] bg-[#0B0B0D] opacity-60 hover:opacity-100"
                            }`}
                          >
                            <div className="relative aspect-[4/3] w-full overflow-hidden bg-black">
                              <img
                                src={item.imageUrl}
                                alt={item.filename}
                                className="h-full w-full object-cover"
                              />
                              <span
                                className={`absolute right-1.5 top-1.5 inline-flex h-5 w-5 items-center justify-center rounded-md border ${
                                  checked
                                    ? "border-[#F2853F] bg-[#F2853F] text-black"
                                    : "border-white/30 bg-black/70 text-transparent"
                                }`}
                              >
                                <Check className="h-3 w-3" />
                              </span>
                              {item.rejectReason && (
                                <span className="font-mono-tech absolute bottom-1 left-1 rounded bg-rose-500/90 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-white">
                                  {item.rejectReason.replace("_", " ")}
                                </span>
                              )}
                            </div>
                            <div className="truncate px-2 py-1 font-mono-tech text-[10px] text-[#D4D4D8]">
                              #{item.seqIndex} · {item.filename}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            </section>
          ) : (
            <RoboflowSourceCard onInspectSuccess={handleInspectSuccess} />
          )}

          {/* Card 02: Output Schema, Models & Class Colors */}
          <section className="c1-card p-5 sm:p-6">
            <p className="c1-kicker">
              02 / ANNOTATION SCHEMA, MODELS &amp; CLASS COLORS
            </p>
            <h2 className="mt-1 text-base font-medium text-white">
              Choose 2D BBox / Polygons, Detector &amp; Optional SAM Refiner
            </h2>

            {/* Output Type + Mode Toggles */}
            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <span className="c1-kicker">ANNOTATION OUTPUT *</span>
                <div className="mt-1.5 grid grid-cols-3 rounded-xl border border-[#303035] bg-[#0A0A0C] p-1">
                  {(
                    [
                      { id: "bbox", label: "2D BBox" },
                      { id: "polygon", label: "Polygon" },
                      { id: "both", label: "Both" },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setOutputMode(opt.id)}
                      className={`rounded-lg py-1.5 text-xs font-medium transition-colors ${
                        outputMode === opt.id
                          ? "bg-[#F2853F] text-black"
                          : "text-[#A1A1AA] hover:text-white"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <span className="c1-kicker">EVALUATION MODE</span>
                <div className="mt-1.5 grid grid-cols-2 rounded-xl border border-[#303035] bg-[#0A0A0C] p-1">
                  {(
                    [
                      { id: "zero-shot", label: "Zero-Shot" },
                      { id: "few-shot", label: "Few-Shot" },
                    ] as const
                  ).map((opt) => {
                    const disabled =
                      opt.id === "few-shot" &&
                      selectedDetector &&
                      !selectedDetector.supportsFewShot;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        disabled={Boolean(disabled)}
                        onClick={() => setShotMode(opt.id)}
                        className={`rounded-lg py-1.5 text-xs font-medium transition-colors ${
                          shotMode === opt.id
                            ? "bg-[#F2853F] text-black"
                            : "text-[#A1A1AA] hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Detector + SAM Refiner Selectors */}
            <div className="mt-4 grid grid-cols-1 gap-3.5 sm:grid-cols-2">
              <label className="block">
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Local Detector / VLM Model
                </span>
                {retrySourceJob && (
                  <p className="mt-1 font-mono-tech text-[10px] text-[#F2853F]">
                    Prev: {retrySourceJob.modelLabel} · pick another model or threshold
                  </p>
                )}
                <select
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                >
                  {detectors.map((d) => (
                    <option key={d.id} value={d.id} disabled={!d.available}>
                      {d.label} {!d.available ? `(${d.note})` : ""}
                    </option>
                  ))}
                </select>
                {selectedDetector && (
                  <p className="font-mono-tech mt-1 text-[11px] text-[#8A8A8E]">
                    {selectedDetector.engine} · {selectedDetector.quantization} ·{" "}
                    {selectedDetector.note}
                  </p>
                )}
                {isFlorenceDetector && (
                  <p className="mt-1.5 rounded-lg border border-[#F2853F]/25 bg-[#181411] px-2.5 py-1.5 text-[11px] leading-snug text-[#E5B89B]">
                    Schema ingestion: parses Class &amp; Color Notation chips + target
                    &amp; &ldquo;Ignore ...&rdquo; classes dynamically from prompt (or raw{" "}
                    <code className="text-[#F2853F]">&lt;TASK&gt;</code> tokens). JSON
                    prose &amp; Temperature are ignored.
                  </p>
                )}
                {isYoloeDetector && (
                  <p className="mt-1.5 rounded-lg border border-[#38BDF8]/25 bg-[#10161B] px-2.5 py-1.5 text-[11px] leading-snug text-[#9DD9F3]">
                    Schema ingestion: encodes Class &amp; Color Notation chips directly via
                    open-vocab text embeddings (click Sync to extract classes from prompt).
                  </p>
                )}
                {isOwlv2Detector && (
                  <p className="mt-1.5 rounded-lg border border-[#38BDF8]/25 bg-[#10161B] px-2.5 py-1.5 text-[11px] leading-snug text-[#9DD9F3]">
                    Schema ingestion: queries Class &amp; Color Notation chips directly with{" "}
                    {selectedDetector?.shortLabel || "OWLv2"} (pair with SAM 2.1 refiner for
                    polygons).
                  </p>
                )}
              </label>

              <label className="block">
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Polygon Mask Refiner (SAM)
                </span>
                <select
                  value={outputMode === "bbox" ? "none" : samRefinerId}
                  onChange={(e) => setSamRefinerId(e.target.value)}
                  disabled={outputMode === "bbox"}
                  className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2.5 text-xs text-white focus:border-[#F2853F] focus:outline-none disabled:opacity-45"
                >
                  {samRefiners.map((r) => (
                    <option key={r.id} value={r.id} disabled={!r.available}>
                      {r.label}
                    </option>
                  ))}
                </select>
                <p className="font-mono-tech mt-1 text-[11px] text-[#8A8A8E]">
                  {outputMode === "bbox"
                    ? "Select Polygon or Both to enable SAM mask refinement."
                    : "Converts detected 2D boxes into pixel-tight segmentation polygons."}
                </p>
              </label>
            </div>

            {/* Prompt */}
            <div className="mt-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Detection Prompt &amp; Color Hints
                </span>
                <button
                  type="button"
                  onClick={handleSyncColorsFromPrompt}
                  className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#1B1B1F] px-3 py-1 text-[11px] text-[#F2853F] hover:border-[#F2853F]/50"
                >
                  <Palette className="h-3 w-3" />
                  Sync Classes &amp; Colors from Prompt
                </button>
              </div>
              <textarea
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe target classes and optional colors, e.g. Detect person (blue), helmet (green), forklift (#F2853F)..."
                className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] p-3 text-xs leading-relaxed text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
              />
            </div>

            {/* Class & Color Legend Table */}
            <div className="c1-inset mt-4 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="c1-kicker">CLASS &amp; COLOR NOTATION</p>
                  <p className="mt-0.5 text-[11px] text-[#A1A1AA]">
                    Boxes render cleanly in these colors with no text labels on the box.
                    Verify lightbox highlights each class in a separate legend column.
                  </p>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap gap-2">
                {classColors.map((entry, idx) => (
                  <div
                    key={`${entry.name}-${idx}`}
                    className="flex items-center gap-2 rounded-xl border border-[#303035] bg-[#121215] px-2.5 py-1.5"
                  >
                    <input
                      type="color"
                      value={entry.color}
                      onChange={(e) =>
                        setClassColors((prev) =>
                          prev.map((c, i) =>
                            i === idx ? { ...c, color: e.target.value } : c
                          )
                        )
                      }
                      aria-label={`Color for ${entry.name}`}
                      className="h-5 w-5 cursor-pointer rounded border-0 bg-transparent p-0"
                    />
                    <input
                      type="text"
                      value={entry.name}
                      onChange={(e) =>
                        setClassColors((prev) =>
                          prev.map((c, i) =>
                            i === idx ? { ...c, name: e.target.value } : c
                          )
                        )
                      }
                      className="font-mono-tech w-28 bg-transparent text-xs text-white focus:outline-none"
                    />
                    <span className="font-mono-tech text-[10px] text-[#8A8A8E]">
                      {entry.color}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        setClassColors((prev) =>
                          prev.filter((_, i) => i !== idx)
                        )
                      }
                      aria-label={`Remove class ${entry.name}`}
                      className="text-xs text-[#8A8A8E] hover:text-rose-400"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>

              <div className="mt-3 flex items-center gap-2">
                <input
                  type="text"
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddClass();
                    }
                  }}
                  placeholder="Add target class name..."
                  className="font-mono-tech flex-1 rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-1.5 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleAddClass}
                  className="inline-flex items-center gap-1 rounded-xl border border-[#303035] bg-[#18181C] px-3 py-1.5 text-xs font-medium text-white hover:border-[#F2853F]/60"
                >
                  <Plus className="h-3.5 w-3.5 text-[#F2853F]" />
                  Add Class
                </button>
              </div>
            </div>

            {/* Optional Few-Shot Section (locked completely unless VLM + few-shot) */}
            {shotMode === "few-shot" && selectedDetector?.supportsFewShot && (
              <div className="mt-4 rounded-xl border border-[#F2853F]/30 bg-[#181512] p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="c1-kicker text-[#F2853F]">
                      FEW-SHOT VISUAL REFERENCES
                    </p>
                    <p className="mt-0.5 text-xs text-[#A1A1AA]">
                      Upload 1–6 annotated example images at native resolution.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => refInputRef.current?.click()}
                    className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#141416] px-3 py-1.5 text-xs text-white hover:border-[#F2853F]"
                  >
                    <Upload className="h-3.5 w-3.5 text-[#F2853F]" />
                    Add Ref Images
                  </button>
                  <input
                    ref={refInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={(e) => handleRefFilesPicked(e.target.files)}
                  />
                </div>

                {refImages.length > 0 && (
                  <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {refImages.map((ref) => (
                      <div
                        key={ref.id}
                        className="overflow-hidden rounded-xl border border-[#303035] bg-[#0A0A0C] p-2"
                      >
                        <div className="relative aspect-video overflow-hidden rounded-lg bg-black">
                          <img
                            src={ref.previewUrl}
                            alt={ref.file.name}
                            className="h-full w-full object-cover"
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setRefImages((prev) =>
                                prev.filter((r) => r.id !== ref.id)
                              )
                            }
                            aria-label="Remove reference"
                            className="absolute right-1.5 top-1.5 rounded-full bg-black/75 p-1 text-white hover:text-rose-400"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                        <input
                          type="text"
                          value={ref.note}
                          onChange={(e) =>
                            setRefImages((prev) =>
                              prev.map((r) =>
                                r.id === ref.id
                                  ? { ...r, note: e.target.value }
                                  : r
                              )
                            )
                          }
                          placeholder="Optional note..."
                          className="mt-2 w-full rounded-lg border border-[#26262A] bg-[#141416] px-2 py-1 text-[11px] text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                        />
                      </div>
                    ))}
                  </div>
                )}

                <textarea
                  rows={2}
                  value={styleNote}
                  onChange={(e) => setStyleNote(e.target.value)}
                  placeholder="Optional shared annotation style note across reference frames..."
                  className="mt-3 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] p-2.5 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                />
              </div>
            )}

            {/* Inference Parameters */}
            <div className="c1-inset mt-4 grid grid-cols-1 gap-4 p-4 sm:grid-cols-3">
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#A1A1AA]">Confidence Min</span>
                  <span className="font-mono-tech text-white">
                    {confidenceThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={confidenceThreshold}
                  onChange={(e) =>
                    setConfidenceThreshold(Number(e.target.value))
                  }
                  className="mt-2 w-full accent-[#F2853F]"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#A1A1AA]">VLM Temperature</span>
                  <span className="font-mono-tech text-white">
                    {temperature.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.8}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="mt-2 w-full accent-[#F2853F]"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#A1A1AA]">Max Tokens</span>
                  <span className="font-mono-tech text-white">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min={256}
                  max={4096}
                  step={128}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="mt-2 w-full accent-[#F2853F]"
                />
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Batch Runner & Jobs History */}
        <div className="space-y-6 xl:col-span-5">
          {/* Card 03: Launch & Live Progress */}
          <section className="c1-card p-5 sm:p-6">
            <p className="c1-kicker">03 / LOCAL BATCH EXECUTION</p>
            <h2 className="mt-1 text-base font-medium text-white">
              {retrySourceJob
                ? `Re-Annotate ${selectedRetryIds.length} Rejected Image(s)`
                : `Fetch & Auto-Annotate in ${batchSize}-Image Batches`}
            </h2>
            <p className="mt-1 text-xs text-[#A1A1AA]">
              {retrySourceJob
                ? `Creates a standalone retry batch linked to ${retrySourceJob.id} using your newly selected detector and confidence threshold.`
                : `Images are downloaded natively to local storage and chunked into ${batchSize}-image batches. Nothing is pushed to Roboflow until you Accept in Verify Auto-Annotation.`}
            </p>

            {submitError && (
              <p className="mt-3 rounded-xl border border-rose-500/35 bg-rose-500/10 px-3.5 py-2 text-xs text-rose-300">
                {submitError}
              </p>
            )}

            <button
              type="button"
              onClick={handleStartBatch}
              disabled={submitting || (Boolean(retrySourceJob) && selectedRetryIds.length === 0)}
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-full bg-[#F2853F] px-5 py-3 text-sm font-semibold text-black transition-colors hover:bg-[#FF9652] disabled:opacity-50"
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : retrySourceJob ? (
                <RotateCcw className="h-4 w-4" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              {submitting
                ? "Submitting Batch Job..."
                : retrySourceJob
                  ? `Re-Run Auto-Annotation on ${selectedRetryIds.length} Rejected Image(s)`
                  : "Fetch from Roboflow & Start Auto-Annotation"}
            </button>

            {activeJob && (
              <div className="c1-inset mt-5 p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono-tech text-xs font-semibold text-white">
                    {activeJob.id}
                  </span>
                  <span className="font-mono-tech rounded-full border border-[#F2853F]/40 bg-[#F2853F]/15 px-2.5 py-0.5 text-[11px] uppercase text-[#F2853F]">
                    {liveProgress?.status ?? activeJob.status}
                  </span>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#0A0A0C]">
                  <div
                    className="h-full bg-[#F2853F] transition-all duration-300"
                    style={{
                      width: `${
                        liveProgress?.progressPct ??
                        (activeJob.totalImages > 0
                          ? Math.round(
                              (activeJob.annotatedCount /
                                activeJob.totalImages) *
                                100
                            )
                          : 0)
                      }%`,
                    }}
                  />
                </div>

                <p className="mt-2.5 text-xs font-medium text-white">
                  {liveProgress?.stage ||
                    `${activeJob.annotatedCount} / ${activeJob.totalImages} annotated`}
                </p>
                <p className="mt-0.5 text-[11px] text-[#A1A1AA]">
                  {liveProgress?.subtext || activeJob.modelLabel}
                </p>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {(activeJob.status === "running" ||
                    activeJob.status === "queued") && (
                    <button
                      type="button"
                      onClick={() => handleControl(activeJob.id, "pause")}
                      className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#141416] px-3 py-1 text-xs text-white hover:border-white/40"
                    >
                      <Pause className="h-3 w-3" />
                      Pause
                    </button>
                  )}
                  {activeJob.totalImages > 0 &&
                    (activeJob.status === "paused" ||
                      activeJob.status === "cancelled" ||
                      activeJob.status === "failed") && (
                    <button
                      type="button"
                      onClick={() => handleControl(activeJob.id, "start")}
                      className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300 hover:bg-emerald-500/20"
                    >
                      <Play className="h-3 w-3" />
                      Resume
                    </button>
                  )}
                  {(activeJob.status === "running" ||
                    activeJob.status === "queued" ||
                    activeJob.status === "fetching") && (
                    <button
                      type="button"
                      onClick={() => handleControl(activeJob.id, "cancel")}
                      className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/35 bg-rose-500/10 px-3 py-1 text-xs text-rose-300 hover:bg-rose-500/20"
                    >
                      <Square className="h-3 w-3" />
                      Cancel
                    </button>
                  )}

                  <Link
                    href={`/verify?job=${encodeURIComponent(activeJob.id)}`}
                    className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-[#F2853F]/50 bg-[#F2853F]/15 px-3.5 py-1 text-xs font-medium text-[#F2853F] hover:bg-[#F2853F]/25"
                  >
                    Verify Batches
                    <ArrowRight className="h-3 w-3" />
                  </Link>
                </div>
              </div>
            )}
          </section>

          {/* Recent Batch Jobs */}
          <section className="c1-card p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="c1-kicker">RECENT AUTO-ANNOTATION JOBS</p>
                <h2 className="mt-1 text-base font-medium text-white">
                  Local Batch Datasets ({jobs.length})
                </h2>
              </div>
              <Layers className="h-4 w-4 text-[#F2853F]" />
            </div>

            {jobs.length === 0 ? (
              <p className="mt-4 rounded-xl border border-dashed border-[#303035] p-6 text-center text-xs text-[#8A8A8E]">
                No auto-annotation jobs yet. Connect a Roboflow project above to
                pull your first 50-image batch.
              </p>
            ) : (
              <div className="mt-4 space-y-3">
                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className="c1-inset p-3.5 transition-colors hover:border-[#3F3F46]"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="truncate text-sm font-medium text-white">
                            {job.projectSlug}
                          </span>
                          {job.fetchScope === "rejected_retry" && (
                            <span className="font-mono-tech rounded-full border border-[#F2853F]/45 bg-[#F2853F]/15 px-2 py-0.5 text-[10px] uppercase text-[#F2853F]">
                              Retry Batch{job.parentJobId ? ` · ${job.parentJobId}` : ""}
                            </span>
                          )}
                          <span className="font-mono-tech rounded-full border border-[#303035] bg-[#141416] px-2 py-0.5 text-[10px] uppercase text-[#A1A1AA]">
                            {job.status}
                          </span>
                        </div>
                        <p className="font-mono-tech mt-1 truncate text-[11px] text-[#8A8A8E]">
                          {job.modelLabel} · {job.outputMode} · {job.createdAtDisplay}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteJob(job.id)}
                        aria-label={`Delete ${job.id}`}
                        className="rounded-lg p-1.5 text-[#8A8A8E] hover:bg-[#141416] hover:text-rose-400"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>

                    <div className="font-mono-tech mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
                      <span className="text-white">
                        {job.annotatedCount}/{job.totalImages} annotated
                      </span>
                      <span className="text-emerald-400">
                        {job.acceptedCount} accepted
                      </span>
                      <span className="text-rose-400">
                        {job.rejectedCount} rejected
                      </span>
                      <span className="text-[#A1A1AA]">
                        {job.checkedBatchesCount}/{job.batches.length} batches checked
                      </span>
                      {job.pushedCount > 0 && (
                        <span className="text-sky-400">
                          {job.pushedCount} pushed
                        </span>
                      )}
                    </div>

                    <div className="mt-3 flex items-center justify-between gap-2 border-t border-[#2D2D33] pt-2.5">
                      <span className="font-mono-tech truncate text-[10px] text-[#8A8A8E]">
                        {job.id}
                      </span>
                      <Link
                        href={`/verify?job=${encodeURIComponent(job.id)}`}
                        className="inline-flex items-center gap-1.5 rounded-full border border-[#F2853F]/45 bg-[#F2853F]/10 px-3 py-1 text-xs font-medium text-[#F2853F] hover:bg-[#F2853F]/20"
                      >
                        Open Verify
                        <ArrowRight className="h-3 w-3" />
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
