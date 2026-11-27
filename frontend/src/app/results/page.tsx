"use client";

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Copy,
  CopyCheck,
  Download,
  Eye,
  EyeOff,
  RotateCcw,
  Tag,
} from "lucide-react";
import { AnnotationCanvas } from "@/components/AnnotationCanvas";
import { VideoResultView } from "@/components/VideoResultView";
import { buildYoloSegTxt, formatBytes } from "@/lib/mock-data";
import { useWorkbench } from "@/lib/workbench-store";

type ResultTab = "reasoning" | "references" | "coordinates" | "json";

function triggerTextDownload(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function ResultsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { runs, cloneRunToAnalyze } = useWorkbench();

  const queryRunId = searchParams.get("run");
  const prevQueryRunIdRef = useRef<string | null>(queryRunId);
  const [selectedRunId, setSelectedRunId] = useState<string>(
    queryRunId || runs[0]?.id || ""
  );

  useEffect(() => {
    if (queryRunId !== prevQueryRunIdRef.current) {
      prevQueryRunIdRef.current = queryRunId;
      if (queryRunId && runs.some((r) => r.id === queryRunId)) {
        setSelectedRunId(queryRunId);
      }
    }
  }, [queryRunId, runs]);

  useEffect(() => {
    if (!runs.some((r) => r.id === selectedRunId) && runs[0]) {
      setSelectedRunId(runs[0].id);
    }
  }, [runs, selectedRunId]);

  const handleSelectRun = (nextRunId: string) => {
    setSelectedRunId(nextRunId);
    if (typeof window !== "undefined") {
      window.history.replaceState(
        null,
        "",
        `/results?run=${encodeURIComponent(nextRunId)}`
      );
    }
  };

  const activeRun = useMemo(
    () => runs.find((r) => r.id === selectedRunId) ?? runs[0] ?? null,
    [runs, selectedRunId]
  );

  const [showBoxes, setShowBoxes] = useState(true);
  const [showPolygons, setShowPolygons] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.24);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>("reasoning");
  const [copiedJson, setCopiedJson] = useState(false);

  const exportCanvasRef = useRef<HTMLCanvasElement | null>(null);

  if (!activeRun) {
    return (
      <div className="c1-card p-10 text-center">
        <p className="c1-kicker">NO RESULTS YET</p>
        <h1 className="mt-2 text-2xl font-bold text-white">Results.</h1>
        <p className="mt-2 text-sm text-[#A1A1AA]">
          Run an evaluation on the Analyze page first to inspect annotations.
        </p>
        <Link
          href="/analyze"
          className="mt-5 inline-flex rounded-full bg-[#F2853F] px-5 py-2 text-xs font-semibold text-[#0A0A0C]"
        >
          Go to Analyze
        </Link>
      </div>
    );
  }

  const hasPolygons = activeRun.detections.some(
    (d) => d.polygon && d.polygon.length >= 3
  );

  const handleDownloadPng = () => {
    const canvas = exportCanvasRef.current;
    if (!canvas) return;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeRun.id}_annotated.png`;
    a.click();
  };

  const handleCopyJson = async () => {
    try {
      await navigator.clipboard.writeText(activeRun.rawJson);
      setCopiedJson(true);
      window.setTimeout(() => setCopiedJson(false), 1500);
    } catch {
      // Ignore clipboard permission errors
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Hero + Run Selector */}
      <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div>
          <p className="c1-kicker">VISUAL INSPECTION & EXPORT</p>
          <h1 className="mt-1.5 text-3xl font-bold tracking-[-0.03em] text-white sm:text-4xl">
            Results.
          </h1>
          <p className="mt-1.5 text-sm text-[#A1A1AA]">
            Compare original media against 2D bounding boxes and polygon
            segmentation overlays, inspect references, and export coordinates.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            aria-label="Select run to inspect"
            value={activeRun.id}
            onChange={(e) => handleSelectRun(e.target.value)}
            className="rounded-full border border-[#303035] bg-[#141416] px-4 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {r.target.filename} ({r.shotMode})
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => {
              cloneRunToAnalyze(activeRun);
              router.push("/analyze");
            }}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-4 py-2 text-xs font-medium text-white hover:border-white/55 hover:bg-white/5"
          >
            <RotateCcw className="h-3.5 w-3.5 text-[#F2853F]" />
            Clone to Analyze
          </button>
        </div>
      </div>

      {/* KPI Summary Strip */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <div className="c1-card p-4">
          <span className="c1-kicker">DETECTIONS</span>
          <p className="mt-1 text-xl font-bold text-white">
            {activeRun.detections.length}
          </p>
          <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
            {activeRun.videoEvents
              ? `${activeRun.videoEvents.length} video event(s)`
              : `${activeRun.outputMode} output`}
          </p>
        </div>
        <div className="c1-card p-4">
          <span className="c1-kicker">SHOT & REFS</span>
          <p className="mt-1 text-xl font-bold capitalize text-white">
            {activeRun.shotMode}
          </p>
          <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
            {activeRun.references.length} reference(s)
          </p>
        </div>
        <div className="c1-card p-4">
          <span className="c1-kicker">MODEL</span>
          <p className="mt-1 truncate text-sm font-semibold text-white">
            {activeRun.modelLabel}
          </p>
          <p className="font-mono-tech truncate text-[11px] text-[#F2853F]">
            {activeRun.adapterLabel ?? "Zero-Shot Base"}
          </p>
        </div>
        <div className="c1-card p-4">
          <span className="c1-kicker">WALL LATENCY</span>
          <p className="mt-1 text-xl font-bold text-white">
            {activeRun.latencySec.toFixed(2)}s
          </p>
          <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
            Peak {activeRun.peakMemoryGb.toFixed(2)} GB RAM
          </p>
        </div>
        <div className="c1-card p-4">
          <span className="c1-kicker">TOKENS</span>
          <p className="mt-1 text-xl font-bold text-white">
            {(
              activeRun.promptTokens + activeRun.generatedTokens
            ).toLocaleString()}
          </p>
          <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
            {activeRun.promptTokens} in / {activeRun.generatedTokens} out
          </p>
        </div>
        <div className="c1-card p-4">
          <span className="c1-kicker">NATIVE MEDIA</span>
          <p className="mt-1 text-base font-bold text-white">
            {activeRun.target.width}×{activeRun.target.height}
          </p>
          <p className="font-mono-tech text-[11px] text-[#8A8A8E]">
            {formatBytes(activeRun.target.sizeBytes)}
            {activeRun.target.fps ? ` · ${activeRun.target.fps} FPS` : ""}
          </p>
        </div>
      </section>

      {/* Main Comparison Card */}
      <section className="c1-card p-5 sm:p-6">
        {/* Overlay Toolbar & Export Actions */}
        <div className="mb-5 flex flex-wrap items-center justify-between gap-4 border-b border-[#26262A] pb-4 select-none">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              aria-pressed={showBoxes}
              onClick={() => setShowBoxes((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
                showBoxes
                  ? "border-cyan-400/50 bg-cyan-400/10 text-cyan-300"
                  : "border-[#303035] bg-[#222225] text-[#8A8A8E]"
              }`}
            >
              {showBoxes ? (
                <Eye className="h-3.5 w-3.5" />
              ) : (
                <EyeOff className="h-3.5 w-3.5" />
              )}
              2D Boxes
            </button>

            <button
              type="button"
              aria-pressed={showPolygons}
              onClick={() => setShowPolygons((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
                showPolygons
                  ? "border-emerald-400/50 bg-emerald-400/10 text-emerald-300"
                  : "border-[#303035] bg-[#222225] text-[#8A8A8E]"
              }`}
            >
              {showPolygons ? (
                <Eye className="h-3.5 w-3.5" />
              ) : (
                <EyeOff className="h-3.5 w-3.5" />
              )}
              Polygon Masks {hasPolygons ? "" : "(None in Run)"}
            </button>

            <button
              type="button"
              aria-pressed={showLabels}
              onClick={() => setShowLabels((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
                showLabels
                  ? "border-[#F2853F]/50 bg-[#F2853F]/10 text-[#F2853F]"
                  : "border-[#303035] bg-[#222225] text-[#8A8A8E]"
              }`}
            >
              <Tag className="h-3.5 w-3.5" />
              Labels
            </button>

            <label className="ml-1 flex items-center gap-2 rounded-full border border-[#303035] bg-[#222225] px-3.5 py-1 text-xs text-[#A1A1AA]">
              <span>Mask Opacity</span>
              <input
                type="range"
                min={0}
                max={0.7}
                step={0.05}
                value={maskOpacity}
                onChange={(e) => setMaskOpacity(Number(e.target.value))}
                className="w-20 accent-[#F2853F]"
              />
              <span className="font-mono-tech w-8 text-right text-white">
                {Math.round(maskOpacity * 100)}%
              </span>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() =>
                triggerTextDownload(
                  `${activeRun.id}_detections.json`,
                  activeRun.rawJson,
                  "application/json"
                )
              }
              className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3.5 py-1.5 text-xs font-medium text-white hover:border-white/55"
            >
              <Download className="h-3.5 w-3.5 text-[#F2853F]" />
              Download JSON
            </button>

            <button
              type="button"
              onClick={() =>
                triggerTextDownload(
                  `${activeRun.id}_yoloseg.txt`,
                  buildYoloSegTxt(activeRun.detections),
                  "text/plain"
                )
              }
              className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3.5 py-1.5 text-xs font-medium text-white hover:border-white/55"
            >
              <Download className="h-3.5 w-3.5 text-emerald-400" />
              Export YOLO-seg (.txt)
            </button>

            {activeRun.target.mediaType === "image" && (
              <button
                type="button"
                onClick={handleDownloadPng}
                className="inline-flex items-center gap-1.5 rounded-full bg-[#F2853F] px-4 py-1.5 text-xs font-semibold text-[#0A0A0C] hover:bg-[#FF9652]"
              >
                <Download className="h-3.5 w-3.5" />
                Annotated PNG
              </button>
            )}
          </div>
        </div>

        {/* Image vs Video Side-by-Side Viewer */}
        {activeRun.target.mediaType === "image" ? (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="c1-inset p-3.5">
              <div className="mb-2.5 flex items-center justify-between">
                <span className="c1-kicker">ORIGINAL TARGET</span>
                <span className="font-mono-tech text-[11px] text-[#8A8A8E]">
                  {activeRun.target.filename}
                </span>
              </div>
              <div className="overflow-hidden rounded-xl border border-[#303035] bg-[#0A0A0C]">
                <img
                  src={activeRun.target.previewUrl}
                  alt={`Original ${activeRun.target.filename}`}
                  className="block w-full h-auto"
                />
              </div>
            </div>

            <div className="c1-inset p-3.5">
              <div className="mb-2.5 flex items-center justify-between">
                <span className="c1-kicker">ANNOTATED OVERLAY OUTPUT</span>
                <span className="font-mono-tech text-[11px] text-[#F2853F]">
                  Hover boxes or table rows to highlight
                </span>
              </div>
              <AnnotationCanvas
                canvasRef={exportCanvasRef}
                imageUrl={activeRun.target.previewUrl}
                filename={activeRun.target.filename}
                detections={activeRun.detections}
                showBoxes={showBoxes}
                showPolygons={showPolygons}
                showLabels={showLabels}
                maskOpacity={maskOpacity}
                hoveredIndex={hoveredIdx}
                onHoverIndexChange={setHoveredIdx}
              />
            </div>
          </div>
        ) : (
          <VideoResultView
            run={activeRun}
            showBoxes={showBoxes}
            showPolygons={showPolygons}
            showLabels={showLabels}
            maskOpacity={maskOpacity}
          />
        )}
      </section>

      {/* Bottom Inspection Tabs */}
      <section className="c1-card p-5 sm:p-6">
        <div
          role="tablist"
          aria-label="Inspection details"
          className="flex flex-wrap gap-2 border-b border-[#26262A] pb-4"
        >
          {(
            [
              { id: "reasoning", label: "Events & Reasoning" },
              {
                id: "references",
                label: `References Used (${activeRun.references.length})`,
              },
              {
                id: "coordinates",
                label: `Coordinates Table (${activeRun.detections.length})`,
              },
              { id: "json", label: "Raw JSON" },
            ] as { id: ResultTab; label: string }[]
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-full px-4 py-1.5 text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-[#F2853F] font-semibold text-[#0A0A0C]"
                  : "border border-[#303035] bg-[#222225] text-[#A1A1AA] hover:text-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {activeTab === "reasoning" && (
            <div className="space-y-4">
              <div className="c1-inset p-4">
                <span className="c1-kicker">MODEL SUMMARY</span>
                <p className="mt-1.5 text-sm leading-relaxed text-white">
                  {activeRun.summaryText}
                </p>
              </div>

              <div className="c1-inset p-4">
                <span className="c1-kicker">EVALUATION PROMPT</span>
                <p className="mt-1.5 text-xs leading-relaxed text-[#A1A1AA]">
                  {activeRun.prompt}
                </p>
              </div>

              {activeRun.videoEvents && activeRun.videoEvents.length > 0 && (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  {activeRun.videoEvents.map((ev) => (
                    <div key={ev.event_id} className="c1-inset p-4">
                      <div className="flex items-center justify-between">
                        <span className="font-mono-tech text-xs font-semibold text-[#F2853F]">
                          EVENT #{ev.event_id} · {ev.short_label}
                        </span>
                        <span className="font-mono-tech text-[11px] text-white">
                          {ev.event_start_timestamp_sec.toFixed(1)}s →{" "}
                          {ev.event_end_timestamp_sec.toFixed(1)}s
                        </span>
                      </div>
                      <p className="mt-1.5 text-xs font-medium text-white">
                        Trigger: {ev.trigger_subject}
                      </p>
                      <p className="mt-1 text-xs text-[#A1A1AA]">
                        {ev.event_summary}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === "references" && (
            <div>
              {activeRun.references.length === 0 ? (
                <div className="c1-inset p-8 text-center text-xs text-[#8A8A8E]">
                  This run was executed in <strong>Zero-Shot</strong> mode with
                  no reference images or reference video frames attached.
                </div>
              ) : (
                <div className="space-y-4">
                  {activeRun.referenceStyleNote && (
                    <div className="c1-inset p-3.5 text-xs text-[#A1A1AA]">
                      <span className="c1-kicker mr-2">STYLE NOTE:</span>
                      {activeRun.referenceStyleNote}
                    </div>
                  )}
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {activeRun.references.map((ref) => (
                      <div key={ref.id} className="c1-inset overflow-hidden p-3">
                        <img
                          src={ref.imageUrl}
                          alt={ref.filename}
                          className="aspect-video w-full rounded-lg border border-[#303035] object-cover"
                        />
                        <div className="mt-2.5 flex items-center justify-between gap-2">
                          <span className="truncate text-xs font-medium text-white">
                            {ref.filename}
                          </span>
                          <span className="font-mono-tech shrink-0 rounded-full border border-[#303035] px-2 py-0.5 text-[10px] uppercase text-[#F2853F]">
                            {ref.kind === "video_frame" ? "Video Ref" : "Image Ref"}
                          </span>
                        </div>
                        {ref.note && (
                          <p className="mt-1 text-[11px] text-[#A1A1AA]">
                            {ref.note}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "coordinates" && (
            <div className="overflow-x-auto rounded-xl border border-[#303035]">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-[#303035] bg-[#101013]">
                    <th className="c1-kicker px-4 py-2.5">#</th>
                    <th className="c1-kicker px-4 py-2.5">Label</th>
                    <th className="c1-kicker px-4 py-2.5">Confidence</th>
                    <th className="c1-kicker px-4 py-2.5">
                      bbox_2d [0..1000]
                    </th>
                    <th className="c1-kicker px-4 py-2.5">Polygon Vertices</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#26262A]">
                  {activeRun.detections.map((det, idx) => (
                    <tr
                      key={det.id}
                      onMouseEnter={() => setHoveredIdx(idx)}
                      onMouseLeave={() => setHoveredIdx(null)}
                      className={`transition-colors ${
                        hoveredIdx === idx
                          ? "bg-[#1F1713]"
                          : "hover:bg-[#19191D]"
                      }`}
                    >
                      <td className="font-mono-tech px-4 py-2.5 text-[#8A8A8E]">
                        {idx + 1}
                      </td>
                      <td className="px-4 py-2.5 font-medium text-white">
                        {det.label}
                      </td>
                      <td className="font-mono-tech px-4 py-2.5 text-emerald-400">
                        {Math.round(det.confidence * 100)}%
                      </td>
                      <td className="font-mono-tech px-4 py-2.5 text-cyan-300">
                        [{det.bbox_2d.join(", ")}]
                      </td>
                      <td className="font-mono-tech px-4 py-2.5 text-[#A1A1AA]">
                        {det.polygon
                          ? `${det.polygon.length} points`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "json" && (
            <div className="c1-inset p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="c1-kicker">STRUCTURED OUTPUT PAYLOAD</span>
                <button
                  type="button"
                  onClick={() => void handleCopyJson()}
                  className="inline-flex items-center gap-1 rounded-full border border-white/25 px-3 py-1 text-xs text-white hover:border-white/55"
                >
                  {copiedJson ? (
                    <>
                      <CopyCheck className="h-3.5 w-3.5 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" />
                      Copy JSON
                    </>
                  )}
                </button>
              </div>
              <pre className="font-mono-tech max-h-80 overflow-auto rounded-lg bg-[#0A0A0C] p-4 text-xs leading-relaxed text-zinc-200">
                {activeRun.rawJson}
              </pre>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="c1-card p-8 text-xs text-[#8A8A8E]">
          Loading evaluation results...
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}
