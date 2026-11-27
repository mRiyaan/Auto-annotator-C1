"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Check,
  Film,
  Image as ImageIcon,
  Layers,
  Lock,
  Play,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  Video,
} from "lucide-react";
import {
  streamRunProgress,
  submitRun,
} from "@/lib/api";
import {
  INITIAL_MOCK_RUNS,
  MODEL_CATALOG,
  PROMPT_PRESETS,
  formatBytes,
  getModelOption,
} from "@/lib/mock-data";
import { useWorkbench } from "@/lib/workbench-store";
import type {
  DetectionItem,
  OutputMode,
  ReferencePreview,
  RunRecord,
  ShotMode,
  TargetMediaMeta,
} from "@/lib/types";

interface LocalRefImage {
  id: string;
  file?: File;
  filename: string;
  imageUrl: string;
  note: string;
}

async function dataUrlToImageBlob(
  dataUrl: string,
  mime = "image/png"
): Promise<Blob> {
  if (!dataUrl.startsWith("data:image/svg+xml")) {
    const res = await fetch(dataUrl);
    return await res.blob();
  }
  return await new Promise<Blob>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || 1280;
      canvas.height = img.naturalHeight || 720;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not create canvas context"));
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to rasterize sample image"));
        },
        mime,
        0.92
      );
    };
    img.onerror = () => reject(new Error("Failed to load preview image"));
    img.src = dataUrl;
  });
}

interface LocalRefVideoFrame {
  id: string;
  sampleIdx: number;
  timestampSec: number;
  imageUrl: string;
  selected: boolean;
}

interface LocalRefVideo {
  id: string;
  filename: string;
  sizeBytes: number;
  durationSec: number;
  frames: LocalRefVideoFrame[];
  sampling: boolean;
  error?: string;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

function inspectImageDimensions(
  src: string
): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () =>
      resolve({
        width: img.naturalWidth || 1280,
        height: img.naturalHeight || 720,
      });
    img.onerror = () => resolve({ width: 1280, height: 720 });
    img.src = src;
  });
}

function inspectVideoMetaAndSampleFrames(
  objectUrl: string,
  sampleCount = 4
): Promise<{
  width: number;
  height: number;
  durationSec: number;
  frames: { timestampSec: number; imageUrl: string }[];
}> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = objectUrl;

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
    };

    video.onerror = () => {
      cleanup();
      reject(new Error("Browser could not decode this video codec for sampling."));
    };

    video.onloadedmetadata = async () => {
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      const durationSec =
        Number.isFinite(video.duration) && video.duration > 0
          ? video.duration
          : 4.0;

      const ratios = [0.15, 0.4, 0.65, 0.85].slice(0, sampleCount);
      const timestamps = ratios.map((r) =>
        Math.min(Math.max(0.1, durationSec * r), Math.max(0.1, durationSec - 0.1))
      );

      const canvas = document.createElement("canvas");
      const targetW = 640;
      const targetH = Math.max(240, Math.round((height / width) * targetW));
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext("2d");

      const frames: { timestampSec: number; imageUrl: string }[] = [];

      const seekTo = (sec: number) =>
        new Promise<void>((res, rej) => {
          const onSeeked = () => {
            video.removeEventListener("seeked", onSeeked);
            res();
          };
          const onErr = () => {
            video.removeEventListener("error", onErr);
            rej(new Error("Seek failed"));
          };
          video.addEventListener("seeked", onSeeked, { once: true });
          video.addEventListener("error", onErr, { once: true });
          video.currentTime = sec;
        });

      try {
        for (const ts of timestamps) {
          await seekTo(ts);
          if (ctx) {
            ctx.drawImage(video, 0, 0, targetW, targetH);
            frames.push({
              timestampSec: Number(ts.toFixed(2)),
              imageUrl: canvas.toDataURL("image/jpeg", 0.82),
            });
          }
        }
        cleanup();
        resolve({ width, height, durationSec, frames });
      } catch (err) {
        cleanup();
        reject(err instanceof Error ? err : new Error("Frame extraction failed"));
      }
    };
  });
}

export default function AnalyzePage() {
  const {
    draftConfig,
    setDraftConfig,
    addRun,
    refreshRuns,
    backendStatus,
    backendUrl,
    backendHealth,
    backendModels,
    setActiveEngineDisplay,
  } = useWorkbench();

  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [targetMeta, setTargetMeta] = useState<TargetMediaMeta | null>(null);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [refImages, setRefImages] = useState<LocalRefImage[]>([]);
  const [refVideo, setRefVideo] = useState<LocalRefVideo | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressStage, setProgressStage] = useState("Idle — Ready for evaluation");
  const [progressSubtext, setProgressSubtext] = useState(
    "Upload a target image/video or run with the sample preview scene."
  );
  const [lastCreatedRun, setLastCreatedRun] = useState<RunRecord | null>(null);

  const targetInputRef = useRef<HTMLInputElement | null>(null);
  const refImgInputRef = useRef<HTMLInputElement | null>(null);
  const refVideoInputRef = useRef<HTMLInputElement | null>(null);
  const timersRef = useRef<number[]>([]);
  const sseCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      timersRef.current.forEach((id) => window.clearTimeout(id));
      sseCleanupRef.current?.();
    };
  }, []);

  const selectedModel = useMemo(
    () => getModelOption(draftConfig.modelId),
    [draftConfig.modelId]
  );

  useEffect(() => {
    setActiveEngineDisplay({
      id: selectedModel.id,
      shortLabel: selectedModel.shortLabel,
      engine: selectedModel.engine,
      quantization: selectedModel.quantization,
    });
  }, [selectedModel, setActiveEngineDisplay]);

  useEffect(() => {
    if (selectedModel.engine !== "Ollama") return;
    const serverStatus = backendModels.find((m) => m.id === selectedModel.id);
    const unavailable =
      backendStatus !== "online" ||
      !backendHealth?.ollamaReachable ||
      (serverStatus ? !serverStatus.available : false);
    if (unavailable) {
      setDraftConfig((prev) => ({
        ...prev,
        modelId: "mlx-4b-8bit-lora-best",
      }));
    }
  }, [
    backendHealth?.ollamaReachable,
    backendModels,
    backendStatus,
    selectedModel,
    setDraftConfig,
  ]);

  const selectedRefVideoFramesCount = useMemo(
    () => (refVideo ? refVideo.frames.filter((f) => f.selected).length : 0),
    [refVideo]
  );

  const estimatedTargetFrames =
    draftConfig.mediaType === "video"
      ? Math.max(
          1,
          Math.round(
            (targetMeta?.durationSec ?? 10) * draftConfig.samplingFps
          )
        )
      : 1;

  const activeReferenceCount =
    draftConfig.shotMode === "few-shot"
      ? refImages.length + selectedRefVideoFramesCount
      : 0;

  const estimatedVisionTokens =
    (estimatedTargetFrames + activeReferenceCount) * 1100 + 250;
  const contextBudget = 16384;
  const contextUsagePct = Math.min(
    100,
    Math.round((estimatedVisionTokens / contextBudget) * 100)
  );

  const handleTargetFile = async (file: File | undefined) => {
    if (!file) return;
    setTargetError(null);
    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");

    if (!isVideo && !isImage) {
      setTargetError("Unsupported file type. Please choose an image or video.");
      return;
    }

    if (isImage) {
      setTargetFile(file);
      setDraftConfig((prev) => ({ ...prev, mediaType: "image" }));
      const dataUrl = await readFileAsDataUrl(file);
      const dims = await inspectImageDimensions(dataUrl);
      setTargetMeta({
        filename: file.name,
        mediaType: "image",
        width: dims.width,
        height: dims.height,
        sizeBytes: file.size,
        previewUrl: dataUrl,
      });
    } else {
      setTargetFile(file);
      setDraftConfig((prev) => ({ ...prev, mediaType: "video" }));
      const blobUrl = URL.createObjectURL(file);
      const videoEl = document.createElement("video");
      videoEl.preload = "metadata";
      videoEl.src = blobUrl;
      videoEl.onloadedmetadata = () => {
        setTargetMeta({
          filename: file.name,
          mediaType: "video",
          width: videoEl.videoWidth || 1280,
          height: videoEl.videoHeight || 720,
          sizeBytes: file.size,
          fps: 24.0,
          durationSec:
            Number.isFinite(videoEl.duration) && videoEl.duration > 0
              ? Number(videoEl.duration.toFixed(1))
              : 10.0,
          previewUrl: blobUrl,
        });
      };
      videoEl.onerror = () => {
        setTargetMeta({
          filename: file.name,
          mediaType: "video",
          width: 1280,
          height: 720,
          sizeBytes: file.size,
          fps: 24.0,
          durationSec: 10.0,
          previewUrl: blobUrl,
        });
      };
    }
  };

  const handleRefImagesUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (draftConfig.shotMode !== "few-shot") {
      setDraftConfig((prev) => ({ ...prev, shotMode: "few-shot" }));
    }
    const items: LocalRefImage[] = [];
    for (const file of Array.from(files)) {
      if (!file.type.startsWith("image/")) continue;
      const dataUrl = await readFileAsDataUrl(file);
      items.push({
        id: `ref-img-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        file,
        filename: file.name,
        imageUrl: dataUrl,
        note: "",
      });
    }
    if (items.length) {
      setRefImages((prev) => [...prev, ...items]);
    }
  };

  const handleRefVideoUpload = async (file: File | undefined) => {
    if (!file) return;
    if (draftConfig.shotMode !== "few-shot") {
      setDraftConfig((prev) => ({ ...prev, shotMode: "few-shot" }));
    }
    const vidId = `ref-vid-${Date.now()}`;
    setRefVideo({
      id: vidId,
      filename: file.name,
      sizeBytes: file.size,
      durationSec: 0,
      frames: [],
      sampling: true,
    });

    const blobUrl = URL.createObjectURL(file);
    try {
      const sampled = await inspectVideoMetaAndSampleFrames(blobUrl, 4);
      setRefVideo({
        id: vidId,
        filename: file.name,
        sizeBytes: file.size,
        durationSec: Number(sampled.durationSec.toFixed(1)),
        sampling: false,
        frames: sampled.frames.map((fr, idx) => ({
          id: `${vidId}-f${idx + 1}`,
          sampleIdx: idx + 1,
          timestampSec: fr.timestampSec,
          imageUrl: fr.imageUrl,
          selected: idx < 3,
        })),
      });
    } catch (err) {
      setRefVideo({
        id: vidId,
        filename: file.name,
        sizeBytes: file.size,
        durationSec: 0,
        sampling: false,
        frames: [],
        error:
          err instanceof Error
            ? err.message
            : "Could not sample frames in browser.",
      });
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  };

  const handleLoadSampleTarget = () => {
    const sample =
      draftConfig.mediaType === "video"
        ? INITIAL_MOCK_RUNS[2].target
        : INITIAL_MOCK_RUNS[0].target;
    setTargetFile(null);
    setTargetMeta({ ...sample });
    setTargetError(null);
  };

  const handleRunAnalysis = async () => {
    if (isRunning) return;
    timersRef.current.forEach((id) => window.clearTimeout(id));
    timersRef.current = [];
    sseCleanupRef.current?.();
    sseCleanupRef.current = null;
    setRunError(null);

    if (backendStatus === "online") {
      if (draftConfig.mediaType === "video" && !targetFile) {
        setRunError(
          "Please upload a real video file (.mp4 / .mov) under Target Media to run local backend video analysis."
        );
        return;
      }

      setIsRunning(true);
      setLastCreatedRun(null);
      setProgressPct(6);
      setProgressStage("Uploading Media & Dispatching to Local Backend...");
      setProgressSubtext(
        `${selectedModel.shortLabel} · ${draftConfig.shotMode} · ${draftConfig.outputMode}`
      );

      try {
        const fd = new FormData();
        if (targetFile) {
          fd.append("target_file", targetFile, targetFile.name);
        } else {
          const sampleUrl =
            targetMeta?.previewUrl ?? INITIAL_MOCK_RUNS[0].target.previewUrl;
          const pngBlob = await dataUrlToImageBlob(sampleUrl, "image/png");
          fd.append("target_file", pngBlob, "sample_factory_scene.png");
        }

        fd.append("prompt", draftConfig.prompt);
        fd.append("mediaType", draftConfig.mediaType);
        fd.append("shotMode", draftConfig.shotMode);
        fd.append("outputMode", draftConfig.outputMode);
        fd.append("modelId", draftConfig.modelId);
        fd.append("temperature", String(draftConfig.temperature));
        fd.append("maxTokens", String(draftConfig.maxTokens));
        fd.append("samplingFps", String(draftConfig.samplingFps));
        fd.append("outputFpsMode", draftConfig.outputFpsMode);
        fd.append("byteTrackMemorySec", String(draftConfig.byteTrackMemorySec));
        fd.append("referenceStyleNote", draftConfig.referenceStyleNote);

        const refNotes: string[] = [];
        if (draftConfig.shotMode === "few-shot") {
          for (const item of refImages) {
            if (item.file) {
              fd.append("ref_files", item.file, item.filename);
            } else {
              const b = await dataUrlToImageBlob(item.imageUrl, "image/jpeg");
              fd.append("ref_files", b, item.filename);
            }
            refNotes.push(item.note || "");
          }
          if (refVideo) {
            for (const fr of refVideo.frames.filter((f) => f.selected)) {
              const b = await dataUrlToImageBlob(fr.imageUrl, "image/jpeg");
              fd.append(
                "ref_files",
                b,
                `vid_frame_${fr.sampleIdx}_${fr.timestampSec}s.jpg`
              );
              refNotes.push(
                `Reference frame @ ${fr.timestampSec}s from ${refVideo.filename}`
              );
            }
          }
        }
        fd.append("refNotesJson", JSON.stringify(refNotes));

        const submitted = await submitRun(backendUrl, fd);
        addRun(submitted.record);
        setProgressPct(10);
        setProgressStage("Queued on Local Apple Silicon Worker");
        setProgressSubtext(`Run ID: ${submitted.runId}`);

        sseCleanupRef.current = streamRunProgress(
          backendUrl,
          submitted.runId,
          (evt) => {
            setProgressPct(evt.progressPct);
            setProgressStage(evt.stage);
            setProgressSubtext(evt.subtext);
            if (evt.record) {
              addRun(evt.record);
              if (evt.status === "completed") {
                setLastCreatedRun(evt.record);
                setIsRunning(false);
                void refreshRuns();
              } else if (evt.status === "failed") {
                setRunError(
                  evt.record.errorMessage || evt.subtext || "Local run failed."
                );
                setIsRunning(false);
                void refreshRuns();
              }
            }
          },
          (errMsg) => {
            setRunError(errMsg);
            setIsRunning(false);
          }
        );
      } catch (err) {
        setRunError(
          err instanceof Error ? err.message : "Failed to submit local run."
        );
        setProgressPct(0);
        setProgressStage("Submission Failed");
        setIsRunning(false);
      }
      return;
    }

    // Fallback Phase 1 UI preview simulation when backend server is offline
    const effectiveTarget: TargetMediaMeta =
      targetMeta ??
      (draftConfig.mediaType === "video"
        ? { ...INITIAL_MOCK_RUNS[2].target }
        : { ...INITIAL_MOCK_RUNS[0].target });

    setIsRunning(true);
    setLastCreatedRun(null);
    setProgressPct(16);
    setProgressStage("Stage 1/3 · Preparing Target & Reference Context");
    setProgressSubtext(
      `${effectiveTarget.filename} (${effectiveTarget.width}×${effectiveTarget.height}) · ${activeReferenceCount} reference frame(s)`
    );

    const t1 = window.setTimeout(() => {
      setProgressPct(54);
      setProgressStage(`Stage 2/3 · Simulating ${selectedModel.shortLabel}`);
      setProgressSubtext(
        `UI preview run (${draftConfig.shotMode}, output: ${draftConfig.outputMode}) — start FastAPI server for live MLX execution`
      );
    }, 520);

    const t2 = window.setTimeout(() => {
      setProgressPct(86);
      setProgressStage("Stage 3/3 · Parsing Coordinates & Formatting Masks");
      setProgressSubtext("Validating normalized [0..1000] JSON geometry...");
    }, 1050);

    const t3 = window.setTimeout(() => {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, "0");
      const runId = `run-${String(now.getHours())}${pad(now.getMinutes())}${pad(
        now.getSeconds()
      )}`;

      const refs: ReferencePreview[] =
        draftConfig.shotMode === "few-shot"
          ? [
              ...refImages.map((img) => ({
                id: img.id,
                kind: "image" as const,
                filename: img.filename,
                imageUrl: img.imageUrl,
                note: img.note || undefined,
              })),
              ...(refVideo
                ? refVideo.frames
                    .filter((f) => f.selected)
                    .map((f) => ({
                      id: f.id,
                      kind: "video_frame" as const,
                      filename: `${refVideo.filename} @ ${f.timestampSec}s`,
                      imageUrl: f.imageUrl,
                      timestampSec: f.timestampSec,
                    }))
                : []),
            ]
          : [];

      const baseDetections: DetectionItem[] =
        effectiveTarget.mediaType === "video"
          ? INITIAL_MOCK_RUNS[2].detections
          : INITIAL_MOCK_RUNS[0].detections;

      const detectionsForOutput = baseDetections.map((d) => ({
        ...d,
        polygon: draftConfig.outputMode === "bbox" ? undefined : d.polygon,
      }));

      const newRun: RunRecord = {
        id: runId,
        createdAtIso: now.toISOString(),
        createdAtDisplay: `Today, ${pad(now.getHours())}:${pad(
          now.getMinutes()
        )}`,
        status: "preview-mock",
        isMockPreview: true,
        target: effectiveTarget,
        shotMode: draftConfig.shotMode,
        outputMode: draftConfig.outputMode,
        modelId: draftConfig.modelId,
        modelLabel: selectedModel.label,
        adapterLabel: selectedModel.adapterName,
        prompt: draftConfig.prompt,
        temperature: draftConfig.temperature,
        maxTokens: draftConfig.maxTokens,
        samplingFps:
          effectiveTarget.mediaType === "video"
            ? draftConfig.samplingFps
            : undefined,
        outputFpsMode:
          effectiveTarget.mediaType === "video"
            ? draftConfig.outputFpsMode
            : undefined,
        byteTrackMemorySec:
          effectiveTarget.mediaType === "video"
            ? draftConfig.byteTrackMemorySec
            : undefined,
        references: refs,
        referenceStyleNote:
          draftConfig.shotMode === "few-shot"
            ? draftConfig.referenceStyleNote
            : undefined,
        summaryText: `UI preview run captured for ${effectiveTarget.filename} using ${selectedModel.shortLabel} (${draftConfig.shotMode}, ${draftConfig.outputMode}). Live MLX/Ollama execution connects in Phase 2.`,
        latencySec: effectiveTarget.mediaType === "video" ? 18.4 : 7.2,
        promptTokens: estimatedVisionTokens,
        generatedTokens: draftConfig.outputMode === "bbox" ? 108 : 224,
        peakMemoryGb:
          draftConfig.modelId === "mlx-4b-4bit-base"
            ? 3.85
            : draftConfig.modelId === "florence2-base"
              ? 0.95
              : draftConfig.modelId === "florence2-large"
                ? 1.85
                : selectedModel.engine === "MLX-VLM"
                  ? 6.56
                  : 7.42,
        detections: detectionsForOutput,
        videoEvents:
          effectiveTarget.mediaType === "video"
            ? INITIAL_MOCK_RUNS[2].videoEvents
            : undefined,
        videoKeyframes:
          effectiveTarget.mediaType === "video"
            ? INITIAL_MOCK_RUNS[2].videoKeyframes
            : undefined,
        stageLogs: [
          {
            stage: "UI Preview Simulation",
            target: effectiveTarget.filename,
            wall_time_sec: effectiveTarget.mediaType === "video" ? 18.4 : 7.2,
            prompt_tokens: estimatedVisionTokens,
            generated_tokens: draftConfig.outputMode === "bbox" ? 108 : 224,
            tokens_per_sec: 26.8,
            summary: `Simulated ${detectionsForOutput.length} detections (mock preview)`,
          },
        ],
        rawJson: JSON.stringify(
          detectionsForOutput.map(({ label, confidence, bbox_2d, polygon }) => ({
            label,
            confidence,
            bbox_2d,
            ...(polygon ? { polygon } : {}),
          })),
          null,
          2
        ),
      };

      addRun(newRun);
      setLastCreatedRun(newRun);
      setProgressPct(100);
      setProgressStage("Preview Run Saved to Test Logs & Results");
      setProgressSubtext(
        `Run ${runId} · ${detectionsForOutput.length} simulated detections · ${refs.length} ref(s)`
      );
      setIsRunning(false);
    }, 1550);

    timersRef.current = [t1, t2, t3];
  };

  return (
    <div className="space-y-6">
      {/* Page Hero */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="c1-kicker">MODEL EVALUATION WORKBENCH</p>
          <h1 className="mt-1.5 text-3xl font-bold tracking-[-0.03em] text-white sm:text-4xl">
            Analyze.
          </h1>
          <p className="mt-1.5 max-w-2xl text-sm text-[#A1A1AA]">
            Test zero-shot and few-shot prompts on local 4-bit / 8-bit MLX
            Qwen3-VL 4B (+ LoRA) or optional Ollama 8B using target images,
            videos, and annotated visual references.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`font-mono-tech rounded-full border px-3 py-1 text-[11px] uppercase tracking-[0.1em] ${
              backendStatus === "online"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                : "border-[#F2853F]/40 bg-[#F2853F]/10 text-[#F2853F]"
            }`}
          >
            {backendStatus === "online"
              ? "Local MLX Backend Online"
              : "Offline · UI Preview"}
          </span>
        </div>
      </div>

      {/* Main 2-Column Studio Grid */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        {/* LEFT COLUMN: 01 Target Media + 02 Few-Shot References */}
        <div className="space-y-6 xl:col-span-6">
          {/* 01 / TARGET MEDIA */}
          <section className="c1-card p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="c1-kicker">01 / TARGET MEDIA</p>
                <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">
                  Upload Image or Video to Test
                </h2>
              </div>

              <div
                role="group"
                aria-label="Target media type"
                className="inline-flex rounded-full border border-[#303035] bg-[#0A0A0C] p-1 text-xs"
              >
                <button
                  type="button"
                  onClick={() =>
                    setDraftConfig((p) => ({ ...p, mediaType: "image" }))
                  }
                  className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 font-medium transition-colors ${
                    draftConfig.mediaType === "image"
                      ? "bg-[#F2853F] text-[#0A0A0C]"
                      : "text-[#A1A1AA] hover:text-white"
                  }`}
                >
                  <ImageIcon className="h-3.5 w-3.5" />
                  Image
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setDraftConfig((p) => ({ ...p, mediaType: "video" }))
                  }
                  className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 font-medium transition-colors ${
                    draftConfig.mediaType === "video"
                      ? "bg-[#F2853F] text-[#0A0A0C]"
                      : "text-[#A1A1AA] hover:text-white"
                  }`}
                >
                  <Video className="h-3.5 w-3.5" />
                  Video
                </button>
              </div>
            </div>

            <input
              ref={targetInputRef}
              type="file"
              accept={
                draftConfig.mediaType === "image" ? "image/*" : "video/*"
              }
              className="hidden"
              onChange={(e) => {
                void handleTargetFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />

            {!targetMeta ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  void handleTargetFile(e.dataTransfer.files?.[0]);
                }}
                className="c1-inset mt-4 flex flex-col items-center justify-center border-dashed border-[#383840] px-5 py-9 text-center"
              >
                <div className="flex h-11 w-11 items-center justify-center rounded-full border border-[#303035] bg-[#141416] text-[#F2853F]">
                  <Upload className="h-5 w-5" />
                </div>
                <p className="mt-3 text-sm font-medium text-white">
                  Drop a target{" "}
                  {draftConfig.mediaType === "image"
                    ? "image (.jpg, .png, .webp)"
                    : "video (.mp4, .mov, .avi)"}
                </p>
                <p className="mt-1 text-xs text-[#8A8A8E]">
                  Local browser preview only — zero files leave your machine.
                </p>
                <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5">
                  <button
                    type="button"
                    onClick={() => targetInputRef.current?.click()}
                    className="rounded-full bg-[#F2853F] px-4 py-1.5 text-xs font-semibold text-[#0A0A0C] transition-colors hover:bg-[#FF9652]"
                  >
                    Choose{" "}
                    {draftConfig.mediaType === "image" ? "Image" : "Video"}
                  </button>
                  <button
                    type="button"
                    onClick={handleLoadSampleTarget}
                    className="rounded-full border border-white/25 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:border-white/55 hover:bg-white/5"
                  >
                    Load Sample Scene
                  </button>
                </div>
              </div>
            ) : (
              <div className="c1-inset mt-4 p-3.5">
                <div className="overflow-hidden rounded-xl border border-[#303035] bg-[#0A0A0C]">
                  {targetMeta.mediaType === "image" ? (
                    <img
                      src={targetMeta.previewUrl}
                      alt={targetMeta.filename}
                      className="max-h-64 w-full object-contain"
                    />
                  ) : targetMeta.previewUrl.startsWith("blob:") ? (
                    <video
                      src={targetMeta.previewUrl}
                      controls
                      className="max-h-64 w-full object-contain"
                    />
                  ) : (
                    <img
                      src={targetMeta.previewUrl}
                      alt={targetMeta.filename}
                      className="max-h-64 w-full object-cover"
                    />
                  )}
                </div>

                <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-white">
                      {targetMeta.filename}
                    </p>
                    <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
                      {targetMeta.width}×{targetMeta.height} px ·{" "}
                      {formatBytes(targetMeta.sizeBytes)}
                      {targetMeta.mediaType === "video" &&
                      targetMeta.durationSec
                        ? ` · ${targetMeta.durationSec}s @ ${
                            targetMeta.fps ?? 24
                          } FPS`
                        : ""}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => targetInputRef.current?.click()}
                      className="rounded-full border border-white/25 px-3 py-1 text-xs text-white hover:border-white/55"
                    >
                      Replace
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setTargetFile(null);
                        setTargetMeta(null);
                      }}
                      aria-label="Remove target media"
                      className="inline-flex items-center gap-1 rounded-full border border-[#303035] px-3 py-1 text-xs text-[#A1A1AA] hover:border-red-500/50 hover:text-red-400"
                    >
                      <Trash2 className="h-3 w-3" />
                      Clear
                    </button>
                  </div>
                </div>
              </div>
            )}

            {targetError && (
              <p className="mt-2.5 text-xs text-red-400">{targetError}</p>
            )}
          </section>

          {/* 02 / FEW-SHOT REFERENCES */}
          <section className="c1-card p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="c1-kicker">02 / FEW-SHOT REFERENCES</p>
                <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">
                  Annotated Examples for Model Guidance
                </h2>
              </div>
              {draftConfig.shotMode === "few-shot" ? (
                <span className="font-mono-tech rounded-full border border-[#F2853F]/50 bg-[#F2853F]/15 px-3 py-1 text-[11px] uppercase tracking-[0.1em] text-[#F2853F]">
                  Active · Few-Shot
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() =>
                    setDraftConfig((p) => ({ ...p, shotMode: "few-shot" }))
                  }
                  className="rounded-full border border-white/25 bg-transparent px-3.5 py-1 text-xs font-medium text-white transition-colors hover:border-white/55 hover:bg-white/5"
                >
                  Switch to Few-Shot
                </button>
              )}
            </div>

            {draftConfig.shotMode === "zero-shot" ? (
              <div className="c1-inset mt-4 flex items-start gap-3 p-4">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#303035] bg-[#141416] text-[#8A8A8E]">
                  <Lock className="h-4 w-4" />
                </span>
                <div>
                  <p className="text-xs font-medium text-white">
                    Disabled in Zero-Shot Mode
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-[#A1A1AA]">
                    Few-shot references are disabled in Zero-Shot mode. Switch
                    to Few-Shot to attach annotated images or reference videos.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <p className="mt-1.5 text-xs leading-relaxed text-[#A1A1AA]">
                  Provide pre-annotated images or reference videos showing how
                  bounding boxes or segmentation masks should look on the
                  target.
                </p>

                <div className="mt-4 space-y-4">
                  {/* Subsection A: Annotated Reference Images */}
                  <div className="c1-inset p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <span className="c1-kicker">
                          A · ANNOTATED REFERENCE IMAGES
                        </span>
                        <p className="mt-0.5 text-xs text-[#A1A1AA]">
                          Upload 1–6 already-annotated example images.
                        </p>
                      </div>
                      <input
                        ref={refImgInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden"
                        onChange={(e) => {
                          void handleRefImagesUpload(e.target.files);
                          e.target.value = "";
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => refImgInputRef.current?.click()}
                        className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3 py-1 text-xs font-medium text-white hover:border-white/55 hover:bg-white/5"
                      >
                        <Plus className="h-3.5 w-3.5 text-[#F2853F]" />
                        Add Ref Images
                      </button>
                    </div>

                    {refImages.length > 0 ? (
                      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                        {refImages.map((item) => (
                          <div
                            key={item.id}
                            className="rounded-xl border border-[#303035] bg-[#141416] p-2.5"
                          >
                            <div className="relative overflow-hidden rounded-lg bg-[#0A0A0C]">
                              <img
                                src={item.imageUrl}
                                alt={item.filename}
                                className="aspect-video w-full object-cover"
                              />
                              <button
                                type="button"
                                onClick={() =>
                                  setRefImages((prev) =>
                                    prev.filter((r) => r.id !== item.id)
                                  )
                                }
                                aria-label={`Remove ${item.filename}`}
                                className="absolute top-1.5 right-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-black/80 text-zinc-300 hover:text-red-400"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                            <p className="mt-2 truncate text-xs font-medium text-white">
                              {item.filename}
                            </p>
                            <input
                              type="text"
                              value={item.note}
                              onChange={(e) =>
                                setRefImages((prev) =>
                                  prev.map((r) =>
                                    r.id === item.id
                                      ? { ...r, note: e.target.value }
                                      : r
                                  )
                                )
                              }
                              placeholder="Optional note (e.g., tight helmet+vest box)..."
                              className="mt-1.5 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1 text-[11px] text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                            />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="font-mono-tech mt-3 rounded-lg border border-dashed border-[#303035] bg-[#141416]/60 px-3 py-3 text-center text-[11px] text-[#8A8A8E]">
                        No annotated reference images attached yet.
                      </p>
                    )}

                    <label className="mt-3 block">
                      <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                        Shared Annotation Style Note
                      </span>
                      <textarea
                        rows={2}
                        value={draftConfig.referenceStyleNote}
                        onChange={(e) =>
                          setDraftConfig((p) => ({
                            ...p,
                            referenceStyleNote: e.target.value,
                          }))
                        }
                        placeholder="Explain how boxes or segmentation polygons should mirror your reference examples..."
                        className="mt-1 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                      />
                    </label>
                  </div>

                  {/* Subsection B: Separate Reference Videos Section */}
                  <div className="c1-inset p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <span className="c1-kicker">
                          B · REFERENCE VIDEOS (SEPARATE)
                        </span>
                        <p className="mt-0.5 text-xs text-[#A1A1AA]">
                          Upload an annotated reference video — samples
                          candidate frames for you to select.
                        </p>
                      </div>
                      <input
                        ref={refVideoInputRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => {
                          void handleRefVideoUpload(e.target.files?.[0]);
                          e.target.value = "";
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => refVideoInputRef.current?.click()}
                        className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3 py-1 text-xs font-medium text-white hover:border-white/55 hover:bg-white/5"
                      >
                        <Film className="h-3.5 w-3.5 text-[#F2853F]" />
                        {refVideo ? "Replace Ref Video" : "Upload Ref Video"}
                      </button>
                    </div>

                    {!refVideo ? (
                      <p className="font-mono-tech mt-3 rounded-lg border border-dashed border-[#303035] bg-[#141416]/60 px-3 py-3 text-center text-[11px] text-[#8A8A8E]">
                        Dedicated section for annotated reference clips (keeps
                        target & reference videos strictly separate).
                      </p>
                    ) : (
                      <div className="mt-3 space-y-3 rounded-xl border border-[#303035] bg-[#141416] p-3">
                        <div className="flex items-center justify-between gap-2">
                          <div className="min-w-0">
                            <p className="truncate text-xs font-medium text-white">
                              {refVideo.filename}
                            </p>
                            <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
                              {formatBytes(refVideo.sizeBytes)}
                              {refVideo.durationSec
                                ? ` · ${refVideo.durationSec}s`
                                : ""}
                              {" · "}
                              {selectedRefVideoFramesCount} frame(s) selected
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => setRefVideo(null)}
                            aria-label="Remove reference video"
                            className="rounded-full border border-[#303035] px-2.5 py-1 text-[11px] text-[#A1A1AA] hover:text-red-400"
                          >
                            Remove
                          </button>
                        </div>

                        {refVideo.sampling && (
                          <p className="font-mono-tech text-[11px] text-[#F2853F]">
                            Extracting candidate reference keyframes...
                          </p>
                        )}

                        {refVideo.error && (
                          <p className="text-xs text-amber-400">
                            {refVideo.error} (Frame extraction for non-browser
                            codecs runs natively via OpenCV in Phase 2.)
                          </p>
                        )}

                        {refVideo.frames.length > 0 && (
                          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                            {refVideo.frames.map((frame) => (
                              <button
                                key={frame.id}
                                type="button"
                                onClick={() =>
                                  setRefVideo((prev) =>
                                    prev
                                      ? {
                                          ...prev,
                                          frames: prev.frames.map((f) =>
                                            f.id === frame.id
                                              ? { ...f, selected: !f.selected }
                                              : f
                                          ),
                                        }
                                      : null
                                  )
                                }
                                className={`overflow-hidden rounded-lg border text-left transition-colors ${
                                  frame.selected
                                    ? "border-[#F2853F] bg-[#1D1612]"
                                    : "border-[#303035] bg-[#0A0A0C] opacity-65 hover:opacity-100"
                                }`}
                              >
                                <div className="relative">
                                  <img
                                    src={frame.imageUrl}
                                    alt={`Reference frame ${frame.sampleIdx}`}
                                    className="aspect-video w-full object-cover"
                                  />
                                  <span
                                    className={`absolute top-1.5 right-1.5 flex h-4 w-4 items-center justify-center rounded border ${
                                      frame.selected
                                        ? "border-[#F2853F] bg-[#F2853F] text-[#0A0A0C]"
                                        : "border-white/50 bg-black/70 text-transparent"
                                    }`}
                                  >
                                    <Check className="h-3 w-3" />
                                  </span>
                                </div>
                                <div className="font-mono-tech px-2 py-1 text-[10px] text-[#A1A1AA]">
                                  F{frame.sampleIdx} · {frame.timestampSec}s
                                </div>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </section>
        </div>

        {/* RIGHT COLUMN: 03 Prompting & Model Controls + 04 Live Run Status */}
        <div className="space-y-6 xl:col-span-6">
          {/* 03 / PROMPTING & MODEL CONTROLS */}
          <section className="c1-card p-5 sm:p-6">
            <p className="c1-kicker">03 / PROMPTING & MODEL CONTROLS</p>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">
              Configure Prompt, Weights & Output Schema
            </h2>

            {/* Shot Mode + Output Format */}
            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Evaluation Mode
                </span>
                <div className="mt-1.5 grid grid-cols-2 gap-1.5 rounded-xl border border-[#303035] bg-[#0A0A0C] p-1">
                  {(["zero-shot", "few-shot"] as ShotMode[]).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() =>
                        setDraftConfig((p) => ({ ...p, shotMode: mode }))
                      }
                      className={`rounded-lg px-3 py-1.5 text-xs font-medium capitalize transition-colors ${
                        draftConfig.shotMode === mode
                          ? "bg-[#F2853F] text-[#0A0A0C]"
                          : "text-[#A1A1AA] hover:text-white"
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Annotation Output
                </span>
                <div className="mt-1.5 grid grid-cols-3 gap-1 rounded-xl border border-[#303035] bg-[#0A0A0C] p-1">
                  {(
                    [
                      { id: "bbox", label: "2D BBox" },
                      { id: "polygon", label: "Polygon" },
                      { id: "both", label: "Both" },
                    ] as { id: OutputMode; label: string }[]
                  ).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() =>
                        setDraftConfig((p) => ({ ...p, outputMode: opt.id }))
                      }
                      className={`rounded-lg px-2 py-1.5 text-xs font-medium transition-colors ${
                        draftConfig.outputMode === opt.id
                          ? "bg-[#F2853F] text-[#0A0A0C]"
                          : "text-[#A1A1AA] hover:text-white"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Model & LoRA Selector */}
            <div className="mt-4">
              <label className="block">
                <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Local Model & LoRA Weights
                </span>
                <select
                  value={draftConfig.modelId}
                  onChange={(e) =>
                    setDraftConfig((p) => ({
                      ...p,
                      modelId: e.target.value as typeof p.modelId,
                    }))
                  }
                  className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2.5 text-sm text-white focus:border-[#F2853F] focus:outline-none"
                >
                  {MODEL_CATALOG.map((model) => {
                    const isOllama = model.engine === "Ollama";
                    const serverStatus = backendModels.find(
                      (m) => m.id === model.id
                    );
                    const disabled = isOllama
                      ? backendStatus !== "online" ||
                        !backendHealth?.ollamaReachable ||
                        (serverStatus ? !serverStatus.available : false)
                      : false;
                    const note = disabled
                      ? !backendHealth?.ollamaReachable
                        ? "Ollama unreachable"
                        : serverStatus?.note || "Ollama unavailable"
                      : "";
                    return (
                      <option
                        key={model.id}
                        value={model.id}
                        disabled={disabled}
                      >
                        {model.label}
                        {note ? ` (${note})` : ""}
                      </option>
                    );
                  })}
                </select>
              </label>
              <p className="mt-1.5 text-xs text-[#A1A1AA]">
                {selectedModel.description}
              </p>
            </div>

            {/* Prompt Presets & Editor */}
            <div className="mt-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Prompt Presets
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {PROMPT_PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() =>
                        setDraftConfig((p) => ({
                          ...p,
                          prompt: preset.prompt,
                          outputMode: preset.outputMode,
                        }))
                      }
                      className="rounded-full border border-[#303035] bg-[#222225] px-2.5 py-1 text-[11px] text-[#A1A1AA] transition-colors hover:border-[#F2853F]/60 hover:text-white"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              <label className="mt-2.5 block">
                <span className="sr-only">Model evaluation prompt</span>
                <textarea
                  rows={4}
                  value={draftConfig.prompt}
                  onChange={(e) =>
                    setDraftConfig((p) => ({ ...p, prompt: e.target.value }))
                  }
                  placeholder="Write your zero-shot or few-shot instruction prompt..."
                  className="w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3.5 py-2.5 text-sm leading-relaxed text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                />
              </label>
            </div>

            {/* Inference Settings Inset Card */}
            <div className="c1-inset mt-4 p-4">
              <div className="flex items-center justify-between">
                <span className="c1-kicker">INFERENCE PARAMETERS</span>
                <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
                  M3 Pro 18 GB Budget
                </span>
              </div>

              <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <label className="block">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A1A1AA]">Temperature</span>
                    <span className="font-mono-tech text-white">
                      {draftConfig.temperature.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={draftConfig.temperature}
                    onChange={(e) =>
                      setDraftConfig((p) => ({
                        ...p,
                        temperature: Number(e.target.value),
                      }))
                    }
                    className="mt-1.5 w-full accent-[#F2853F]"
                  />
                </label>

                <label className="block">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A1A1AA]">Max Tokens</span>
                    <span className="font-mono-tech text-white">
                      {draftConfig.maxTokens}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={256}
                    max={2048}
                    step={128}
                    value={draftConfig.maxTokens}
                    onChange={(e) =>
                      setDraftConfig((p) => ({
                        ...p,
                        maxTokens: Number(e.target.value),
                      }))
                    }
                    className="mt-1.5 w-full accent-[#F2853F]"
                  />
                </label>

                {draftConfig.mediaType === "video" && (
                  <>
                    <label className="block">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#A1A1AA]">Sampling FPS</span>
                        <span className="font-mono-tech text-white">
                          {draftConfig.samplingFps.toFixed(1)} FPS
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.5}
                        max={2.0}
                        step={0.5}
                        value={draftConfig.samplingFps}
                        onChange={(e) =>
                          setDraftConfig((p) => ({
                            ...p,
                            samplingFps: Number(e.target.value),
                          }))
                        }
                        className="mt-1.5 w-full accent-[#F2853F]"
                      />
                    </label>

                    <label className="block">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#A1A1AA]">
                          ByteTrack Memory
                        </span>
                        <span className="font-mono-tech text-white">
                          {draftConfig.byteTrackMemorySec.toFixed(1)}s
                        </span>
                      </div>
                      <input
                        type="range"
                        min={1.0}
                        max={5.0}
                        step={0.5}
                        value={draftConfig.byteTrackMemorySec}
                        onChange={(e) =>
                          setDraftConfig((p) => ({
                            ...p,
                            byteTrackMemorySec: Number(e.target.value),
                          }))
                        }
                        className="mt-1.5 w-full accent-[#F2853F]"
                      />
                    </label>
                  </>
                )}
              </div>

              {/* Vision Context Token Estimate */}
              <div className="mt-4 border-t border-[#303035] pt-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.1em] text-[#8A8A8E]">
                    Est. Vision Context
                  </span>
                  <span className="font-mono-tech text-white">
                    {estimatedVisionTokens.toLocaleString()} /{" "}
                    {contextBudget.toLocaleString()} tok
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#3F3F46]">
                  <div
                    style={{ width: `${contextUsagePct}%` }}
                    className={`h-full rounded-full transition-all ${
                      contextUsagePct > 85 ? "bg-amber-400" : "bg-white"
                    }`}
                  />
                </div>
              </div>
            </div>

            {/* Primary CTA */}
            {runError && (
              <div className="mt-4 rounded-xl border border-red-500/40 bg-red-500/10 px-3.5 py-2.5 text-xs text-red-300">
                {runError}
              </div>
            )}

            <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs text-[#8A8A8E]">
                {backendStatus === "online"
                  ? "Executes locally on your MacBook via MLX-VLM / Ollama & saves to SQLite."
                  : "Backend offline — simulates preview run in browser."}
              </span>
              <button
                type="button"
                onClick={() => void handleRunAnalysis()}
                disabled={isRunning || !draftConfig.prompt.trim()}
                className="inline-flex items-center gap-2 rounded-full bg-[#F2853F] px-6 py-2.5 text-sm font-semibold text-[#0A0A0C] transition-colors hover:bg-[#FF9652] disabled:opacity-50"
              >
                <Play className="h-4 w-4 fill-current" />
                {isRunning
                  ? backendStatus === "online"
                    ? "Running on Local GPU..."
                    : "Running Preview..."
                  : backendStatus === "online"
                    ? "Run Local Analysis"
                    : "Run Preview Analysis"}
              </button>
            </div>
          </section>

          {/* 04 / LIVE RUN STATUS (Control One Progress Card Style) */}
          <section className="c1-card p-5 sm:p-6">
            <div className="flex items-center justify-between gap-2">
              <p className="c1-kicker">04 / LIVE EVALUATION STATUS</p>
              <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
                {progressPct}%
              </span>
            </div>

            <div className="c1-inset mt-3 p-4">
              {/* Crisp White Progress Bar over #3F3F46 track */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-[#3F3F46]">
                <div
                  style={{ width: `${progressPct}%` }}
                  className="h-full rounded-full bg-white transition-all duration-300"
                />
              </div>

              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold text-white">
                  {progressStage}
                </p>
                {lastCreatedRun && (
                  <Link
                    href={`/results?run=${encodeURIComponent(lastCreatedRun.id)}`}
                    className="inline-flex items-center gap-1.5 rounded-full bg-[#F2853F] px-3.5 py-1 text-xs font-semibold text-[#0A0A0C] hover:bg-[#FF9652]"
                  >
                    Open in Results
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                )}
              </div>
              <p className="mt-1 text-xs text-[#A1A1AA]">{progressSubtext}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 border-t border-[#303035] pt-3 text-xs">
                <div>
                  <span className="c1-kicker block">SHOT / OUTPUT</span>
                  <span className="mt-0.5 block font-medium capitalize text-white">
                    {draftConfig.shotMode} · {draftConfig.outputMode}
                  </span>
                </div>
                <div>
                  <span className="c1-kicker block">WEIGHTS</span>
                  <span className="mt-0.5 block truncate font-medium text-white">
                    {selectedModel.shortLabel}
                  </span>
                </div>
                <div>
                  <span className="c1-kicker block">REFERENCES ATTACHED</span>
                  <span className="mt-0.5 block font-medium text-white">
                    {draftConfig.shotMode === "zero-shot"
                      ? "Disabled (Zero-Shot)"
                      : `${refImages.length} img · ${selectedRefVideoFramesCount} vid frame(s)`}
                  </span>
                </div>
                <div>
                  <span className="c1-kicker block">TARGET</span>
                  <span className="mt-0.5 block truncate font-medium text-white">
                    {targetMeta?.filename ?? "Sample scene (on run)"}
                  </span>
                </div>
              </div>
            </div>

            {lastCreatedRun && (
              <div className="mt-3 flex items-center justify-between rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3.5 py-2.5 text-xs text-emerald-300">
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Saved preview run <strong>{lastCreatedRun.id}</strong> to
                  Test Logs & Results.
                </span>
                <Link
                  href="/logs"
                  className="inline-flex items-center gap-1 font-medium text-white underline-offset-4 hover:underline"
                >
                  <Layers className="h-3.5 w-3.5" />
                  View Logs
                </Link>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
