"use client";

import React, { useMemo, useRef, useState } from "react";
import { Film, Play, RefreshCw, Sparkles } from "lucide-react";
import { reencodeRunApi } from "@/lib/api";
import type { RunRecord } from "@/lib/types";
import { useWorkbench } from "@/lib/workbench-store";
import { AnnotationCanvas } from "./AnnotationCanvas";

interface VideoResultViewProps {
  run: RunRecord;
  showBoxes: boolean;
  showPolygons: boolean;
  showLabels: boolean;
  maskOpacity: number;
}

export function VideoResultView({
  run,
  showBoxes,
  showPolygons,
  showLabels,
  maskOpacity,
}: VideoResultViewProps) {
  const { backendStatus, backendUrl, updateRunInState } = useWorkbench();
  const keyframes = run.videoKeyframes ?? [];
  const events = run.videoEvents ?? [];
  const duration = Math.max(run.target.durationSec ?? 10.0, 1);

  const [selectedKfIdx, setSelectedKfIdx] = useState(0);
  const [rightPaneMode, setRightPaneMode] = useState<"video" | "keyframe">(
    run.annotatedVideoUrl ? "video" : "keyframe"
  );
  const [reencodeFpsMode, setReencodeFpsMode] = useState<"original" | "sampled">(
    run.outputFpsMode ?? "original"
  );
  const [isReencoding, setIsReencoding] = useState(false);
  const [reencodeNotice, setReencodeNotice] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const annotatedVideoRef = useRef<HTMLVideoElement | null>(null);

  const activeKeyframe = keyframes[selectedKfIdx] ?? null;
  const hasRealVideoSrc =
    run.target.previewUrl.startsWith("blob:") ||
    run.target.previewUrl.startsWith("http://") ||
    run.target.previewUrl.startsWith("https://");

  const activeBoxes = useMemo(() => {
    if (activeKeyframe) return activeKeyframe.detections;
    return run.detections;
  }, [activeKeyframe, run.detections]);

  const jumpToTimestamp = (sec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
    }
    if (annotatedVideoRef.current) {
      annotatedVideoRef.current.currentTime = sec;
    }
    if (keyframes.length) {
      let bestIdx = 0;
      let bestDiff = Infinity;
      keyframes.forEach((kf, idx) => {
        const diff = Math.abs(kf.timestamp_sec - sec);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = idx;
        }
      });
      setSelectedKfIdx(bestIdx);
    }
  };

  const handleReencode = async () => {
    if (backendStatus !== "online" || run.isMockPreview) {
      setReencodeNotice(
        `Queued ${reencodeFpsMode === "original" ? "Original FPS" : "Sampled FPS"} re-encode preview (connect local backend to run FFmpeg).`
      );
      return;
    }
    setIsReencoding(true);
    setReencodeNotice(
      `Re-encoding ${run.target.filename} at ${reencodeFpsMode} FPS using cached VLM detections...`
    );
    try {
      const updated = await reencodeRunApi(
        backendUrl,
        run.id,
        reencodeFpsMode,
        run.byteTrackMemorySec
      );
      updateRunInState(updated);
      setRightPaneMode("video");
      setReencodeNotice(
        `Re-encoded cleanly at ${reencodeFpsMode === "original" ? "Original FPS" : "Sampled FPS"} without re-running VLM inference.`
      );
    } catch (err) {
      setReencodeNotice(
        err instanceof Error ? `Re-encode failed: ${err.message}` : "Re-encode failed."
      );
    } finally {
      setIsReencoding(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Side-by-Side Original vs Annotated Preview */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Left: Original Video */}
        <div className="c1-inset p-3.5">
          <div className="mb-2.5 flex items-center justify-between">
            <span className="c1-kicker">ORIGINAL TARGET VIDEO</span>
            <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
              {run.target.width}×{run.target.height} ·{" "}
              {(run.target.fps ?? 24).toFixed(1)} FPS · {duration.toFixed(1)}s
            </span>
          </div>
          {hasRealVideoSrc ? (
            <video
              ref={videoRef}
              src={run.target.previewUrl}
              controls
              className="aspect-video w-full rounded-xl border border-[#303035] bg-black object-contain"
            />
          ) : (
            <div className="relative overflow-hidden rounded-xl border border-[#303035] bg-[#0A0A0C]">
              <img
                src={run.target.previewUrl}
                alt={`Original frame preview for ${run.target.filename}`}
                className="aspect-video w-full object-cover"
              />
              <div className="absolute inset-x-3 bottom-3 flex items-center justify-between rounded-lg border border-white/15 bg-black/75 px-3 py-1.5 backdrop-blur-xs">
                <div className="flex items-center gap-2 text-xs text-white">
                  <Film className="h-3.5 w-3.5 text-[#F2853F]" />
                  <span>{run.target.filename}</span>
                </div>
                <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
                  SAMPLE CLIP SHELL
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Annotated Post-Event Keyframe / Render Output */}
        <div className="c1-inset p-3.5">
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
            <span className="c1-kicker">
              {rightPaneMode === "video" && run.annotatedVideoUrl
                ? "ANNOTATED H.264 VIDEO"
                : `ANNOTATED OUTPUT ${
                    activeKeyframe
                      ? `(KEYFRAME @ ${activeKeyframe.timestamp_sec.toFixed(1)}s)`
                      : ""
                  }`}
            </span>
            <div className="flex items-center gap-2">
              {run.annotatedVideoUrl && (
                <div className="inline-flex rounded-full border border-[#303035] bg-[#141416] p-0.5 text-[10px]">
                  <button
                    type="button"
                    onClick={() => setRightPaneMode("video")}
                    className={`rounded-full px-2.5 py-0.5 ${
                      rightPaneMode === "video"
                        ? "bg-[#F2853F] font-semibold text-[#0A0A0C]"
                        : "text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    Video
                  </button>
                  <button
                    type="button"
                    onClick={() => setRightPaneMode("keyframe")}
                    className={`rounded-full px-2.5 py-0.5 ${
                      rightPaneMode === "keyframe"
                        ? "bg-[#F2853F] font-semibold text-[#0A0A0C]"
                        : "text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    Keyframes
                  </button>
                </div>
              )}
              <span className="font-mono-tech text-[11px] text-[#F2853F]">
                ByteTrack {run.byteTrackMemorySec ?? 3.0}s Mem
              </span>
            </div>
          </div>

          {rightPaneMode === "video" && run.annotatedVideoUrl ? (
            <video
              ref={annotatedVideoRef}
              key={run.annotatedVideoUrl}
              src={run.annotatedVideoUrl}
              controls
              className="aspect-video w-full rounded-xl border border-[#303035] bg-black object-contain"
            />
          ) : (
            <AnnotationCanvas
              imageUrl={activeKeyframe?.imageUrl ?? run.target.previewUrl}
              filename={run.target.filename}
              detections={activeBoxes}
              showBoxes={showBoxes}
              showPolygons={showPolygons}
              showLabels={showLabels}
              maskOpacity={maskOpacity}
              hoveredIndex={null}
            />
          )}
          <p className="mt-2 text-[11px] text-[#A1A1AA]">
            {run.annotatedVideoUrl
              ? `Native H.264 output rendered with ByteTrack (${run.outputFpsMode ?? "original"} FPS).`
              : "UI preview shows post-event grounded keyframes."}
          </p>
        </div>
      </div>

      {/* Multi-Event Timeline Scrubber */}
      <div className="c1-inset p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="c1-kicker">MULTI-EVENT TIMELINE</span>
            <p className="mt-0.5 text-xs text-[#A1A1AA]">
              Click any detected event segment to jump to its nearest grounded
              keyframe.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-full border border-[#303035] bg-[#141416] p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setReencodeFpsMode("original")}
                className={`rounded-full px-3 py-1 transition-colors ${
                  reencodeFpsMode === "original"
                    ? "bg-[#F2853F] font-medium text-[#0A0A0C]"
                    : "text-[#A1A1AA] hover:text-white"
                }`}
              >
                Original FPS
              </button>
              <button
                type="button"
                onClick={() => setReencodeFpsMode("sampled")}
                className={`rounded-full px-3 py-1 transition-colors ${
                  reencodeFpsMode === "sampled"
                    ? "bg-[#F2853F] font-medium text-[#0A0A0C]"
                    : "text-[#A1A1AA] hover:text-white"
                }`}
              >
                Sampled FPS
              </button>
            </div>
            <button
              type="button"
              onClick={() => void handleReencode()}
              disabled={isReencoding}
              className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3.5 py-1.5 text-xs font-medium text-white transition-colors hover:border-white/55 hover:bg-white/5 disabled:opacity-50"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 text-[#F2853F] ${
                  isReencoding ? "animate-spin" : ""
                }`}
              />
              {isReencoding ? "Re-encoding..." : "Re-encode FPS"}
            </button>
          </div>
        </div>

        {reencodeNotice && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-[#F2853F]/35 bg-[#F2853F]/10 px-3 py-2 text-xs text-[#F2853F]">
            <Sparkles className="h-3.5 w-3.5 shrink-0" />
            <span>{reencodeNotice}</span>
          </div>
        )}

        {/* Timeline Track */}
        <div className="mt-4">
          <div className="relative h-9 w-full overflow-hidden rounded-lg border border-[#303035] bg-[#0A0A0C]">
            {events.map((ev, idx) => {
              const leftPct = Math.max(
                0,
                Math.min(100, (ev.event_start_timestamp_sec / duration) * 100)
              );
              const widthPct = Math.max(
                4,
                Math.min(
                  100 - leftPct,
                  ((ev.event_end_timestamp_sec - ev.event_start_timestamp_sec) /
                    duration) *
                    100
                )
              );
              const barColor =
                idx % 2 === 0
                  ? "bg-[#F2853F]/80 hover:bg-[#F2853F]"
                  : "bg-cyan-400/80 hover:bg-cyan-400";
              const barLabel =
                widthPct >= 20
                  ? `E${ev.event_id} (${ev.event_start_timestamp_sec.toFixed(1)}s–${ev.event_end_timestamp_sec.toFixed(1)}s)`
                  : `E${ev.event_id}`;
              return (
                <button
                  key={ev.event_id}
                  type="button"
                  onClick={() => jumpToTimestamp(ev.event_start_timestamp_sec)}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  title={`${ev.short_label} (${ev.event_start_timestamp_sec}s - ${ev.event_end_timestamp_sec}s)`}
                  className={`font-mono-tech absolute top-1 bottom-1 flex items-center justify-center overflow-hidden whitespace-nowrap rounded-md px-1.5 text-[11px] font-semibold text-[#0A0A0C] transition-colors ${barColor}`}
                >
                  {barLabel}
                </button>
              );
            })}
          </div>
          <div className="font-mono-tech mt-1.5 flex justify-between text-[10px] text-[#8A8A8E]">
            <span>0.0s</span>
            <span>{(duration * 0.25).toFixed(1)}s</span>
            <span>{(duration * 0.5).toFixed(1)}s</span>
            <span>{(duration * 0.75).toFixed(1)}s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>
        </div>
      </div>

      {/* Separated 640x360 Keyframes Strip */}
      {keyframes.length > 0 && (
        <div className="c1-inset p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="c1-kicker">
              SEPARATED 640×360 POST-EVENT KEYFRAMES ({keyframes.length})
            </span>
            <span className="font-mono-tech text-[11px] text-[#8A8A8E]">
              Click a frame to inspect bounding boxes
            </span>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {keyframes.map((kf, idx) => {
              const selected = idx === selectedKfIdx;
              return (
                <button
                  key={`${kf.sample_number}-${kf.frame_idx}`}
                  type="button"
                  onClick={() => {
                    setSelectedKfIdx(idx);
                    if (videoRef.current) {
                      videoRef.current.currentTime = kf.timestamp_sec;
                    }
                  }}
                  className={`overflow-hidden rounded-xl border text-left transition-colors ${
                    selected
                      ? "border-[#F2853F] bg-[#181411]"
                      : "border-[#303035] bg-[#141416] hover:border-[#4B4B55]"
                  }`}
                >
                  <div className="relative">
                    <img
                      src={kf.imageUrl}
                      alt={`Keyframe ${kf.sample_number} at ${kf.timestamp_sec}s`}
                      className="aspect-video w-full object-cover"
                    />
                    <span className="font-mono-tech absolute top-2 left-2 inline-flex items-center gap-1 rounded-md bg-black/80 px-2 py-0.5 text-[10px] text-white">
                      <Play className="h-2.5 w-2.5 text-[#F2853F]" />
                      {kf.timestamp_sec.toFixed(1)}s
                    </span>
                  </div>
                  <div className="flex items-center justify-between px-3 py-2">
                    <span className="font-mono-tech text-[11px] text-white">
                      Frame #{kf.sample_number}
                    </span>
                    <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
                      {kf.detections.length} box
                      {kf.detections.length === 1 ? "" : "es"}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
