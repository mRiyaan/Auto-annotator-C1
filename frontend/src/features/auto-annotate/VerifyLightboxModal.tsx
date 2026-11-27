"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Loader2,
  Trash2,
  X,
} from "lucide-react";
import { getClassColor, hexToRgba } from "./color-utils";
import type {
  AutoItemRecord,
  AutoJobRecord,
  AutoRejectReason,
  AutoReviewStatus,
  ClassColorEntry,
} from "./types";

interface VerifyLightboxModalProps {
  job: AutoJobRecord;
  item: AutoItemRecord;
  itemIndex: number;
  totalFiltered: number;
  classColors: ClassColorEntry[];
  saveState: "idle" | "saving" | "saved" | "error";
  saveError: string | null;
  autoAdvance: boolean;
  onToggleAutoAdvance: () => void;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  onReview: (
    itemId: string,
    reviewStatus: AutoReviewStatus,
    rejectReason?: AutoRejectReason | null,
    options?: { fromLightbox?: boolean }
  ) => Promise<void>;
  onUpdateDetections: (item: AutoItemRecord) => Promise<void>;
}

const REJECT_SHORTCUTS: {
  key: string;
  reason: AutoRejectReason;
  label: string;
}[] = [
  { key: "1", reason: "missed_object", label: "1 · Missed object" },
  { key: "2", reason: "false_detection", label: "2 · False detection" },
  { key: "3", reason: "bad_box", label: "3 · Bad box" },
  { key: "4", reason: "double_box", label: "4 · Double / merged" },
];

export function VerifyLightboxModal({
  job,
  item,
  itemIndex,
  totalFiltered,
  classColors,
  saveState,
  saveError,
  autoAdvance,
  onToggleAutoAdvance,
  onClose,
  onPrev,
  onNext,
  onReview,
  onUpdateDetections,
}: VerifyLightboxModalProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [highlightedClass, setHighlightedClass] = useState<string | null>(null);
  const [hoveredDetId, setHoveredDetId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleSetReview = useCallback(
    async (
      status: AutoReviewStatus,
      reason?: AutoRejectReason | null
    ) => {
      if (busy) return;
      setBusy(true);
      try {
        await onReview(item.id, status, reason ?? null, { fromLightbox: true });
      } finally {
        setBusy(false);
      }
    },
    [busy, item.id, onReview]
  );

  // Keyboard shortcuts matching screenshot 2 UX + explicit Accept (A)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        onPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onNext();
      } else if (e.key.toLowerCase() === "a") {
        e.preventDefault();
        void handleSetReview("accepted", null);
      } else if (e.key.toLowerCase() === "r" || e.key === " ") {
        e.preventDefault();
        if (item.reviewStatus === "rejected" && !item.rejectReason) {
          void handleSetReview("pending", null);
        } else {
          void handleSetReview("rejected", null);
        }
      } else if (e.key === "1") {
        e.preventDefault();
        void handleSetReview("rejected", "missed_object");
      } else if (e.key === "2") {
        e.preventDefault();
        void handleSetReview("rejected", "false_detection");
      } else if (e.key === "3") {
        e.preventDefault();
        void handleSetReview("rejected", "bad_box");
      } else if (e.key === "4") {
        e.preventDefault();
        void handleSetReview("rejected", "double_box");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleSetReview, item.rejectReason, item.reviewStatus, onClose, onNext, onPrev]);

  const handleDeleteSingleDet = async (detId: string) => {
    if (busy) return;
    setBusy(true);
    try {
      const remaining = item.detections.filter((d) => d.id !== detId);
      await onUpdateDetections({ ...item, detections: remaining });
    } finally {
      setBusy(false);
    }
  };

  // Distinct class legend summarizing colors & counts for this frame + job palette
  const classLegend = useMemo(() => {
    const counts = new Map<string, number>();
    item.detections.forEach((d) => {
      const k = d.label.trim() || "object";
      counts.set(k, (counts.get(k) || 0) + 1);
    });

    const seen = new Set<string>();
    const rows: { name: string; color: string; count: number }[] = [];

    classColors.forEach((cc) => {
      const key = cc.name.trim();
      if (!key || seen.has(key.toLowerCase())) return;
      seen.add(key.toLowerCase());
      let matchedCount = 0;
      for (const [detCls, c] of counts.entries()) {
        if (detCls.toLowerCase() === key.toLowerCase()) matchedCount += c;
      }
      rows.push({ name: key, color: cc.color, count: matchedCount });
    });

    for (const [detCls, count] of counts.entries()) {
      if (!seen.has(detCls.toLowerCase())) {
        seen.add(detCls.toLowerCase());
        rows.push({
          name: detCls,
          color: getClassColor(detCls, classColors),
          count,
        });
      }
    }
    return rows;
  }, [item.detections, classColors]);

  const statusText =
    item.reviewStatus === "accepted"
      ? "ACCEPTED"
      : item.reviewStatus === "rejected"
        ? `REJECTED${item.rejectReason ? ` (${item.rejectReason.replace("_", " ")})` : ""}`
        : "PENDING";

  const displayNum = itemIndex >= 0 ? itemIndex + 1 : item.seqIndex;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/88 p-3 backdrop-blur-xs sm:p-5"
      role="dialog"
      aria-modal="true"
      aria-label={`Inspect ${item.filename}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="flex max-h-[95vh] w-full max-w-[1380px] flex-col overflow-hidden rounded-2xl border border-[#2A2A30] bg-[#101013] shadow-2xl">
        {/* Top Main Area: Image Canvas (Left) + Separate Class Color & Detection Column (Right) */}
        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-12">
          {/* Left Stage: Full Image with CLEAN label-free colored BBox / Polygon overlays */}
          <div className="relative flex min-h-[360px] items-center justify-center bg-[#060608] p-3 md:col-span-8 lg:col-span-9">
            <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowOverlay((v) => !v)}
                className="inline-flex items-center gap-1.5 rounded-full border border-white/20 bg-black/75 px-3 py-1 text-xs font-medium text-white hover:border-[#F2853F]"
              >
                {showOverlay ? (
                  <Eye className="h-3.5 w-3.5 text-[#F2853F]" />
                ) : (
                  <EyeOff className="h-3.5 w-3.5 text-[#8A8A8E]" />
                )}
                {showOverlay ? "Overlay On" : "Raw Image"}
              </button>
              <span className="font-mono-tech rounded-full border border-white/15 bg-black/75 px-2.5 py-1 text-[11px] text-[#A1A1AA]">
                {job.outputMode === "bbox"
                  ? "2D BBox (label-free)"
                  : job.outputMode === "polygon"
                    ? "Polygon Mask (label-free)"
                    : "BBox + Polygon (label-free)"}
              </span>
            </div>

            <div
              className="relative max-h-[72vh] w-auto overflow-hidden rounded-lg border border-[#222226]"
              style={{
                aspectRatio: `${Math.max(1, item.width)} / ${Math.max(1, item.height)}`,
              }}
            >
              <img
                src={item.imageUrl}
                alt={item.filename}
                className="block h-full w-full select-none object-contain"
              />

              {showOverlay && item.detections.length > 0 && (
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  className="pointer-events-none absolute inset-0 h-full w-full"
                >
                  {item.detections.map((det) => {
                    const color = getClassColor(det.label, classColors);
                    const clsLower = det.label.trim().toLowerCase();
                    const dimmed =
                      (highlightedClass &&
                        highlightedClass.toLowerCase() !== clsLower) ||
                      (hoveredDetId && hoveredDetId !== det.id);
                    const strokeOpacity = dimmed ? 0.25 : 0.96;
                    const fillOpacity = dimmed ? 0.06 : 0.24;
                    const strokeWidth =
                      hoveredDetId === det.id ||
                      highlightedClass?.toLowerCase() === clsLower
                        ? 4.5
                        : 2.6;

                    const showPoly =
                      job.outputMode !== "bbox" &&
                      det.polygon &&
                      det.polygon.length >= 3;
                    const showBox =
                      job.outputMode === "bbox" ||
                      job.outputMode === "both" ||
                      !showPoly;

                    const [x1, y1, x2, y2] = det.bbox_2d;

                    return (
                      <g key={det.id} style={{ opacity: strokeOpacity }}>
                        {showPoly && det.polygon && (
                          <polygon
                            points={det.polygon
                              .map((p) => `${p[0]},${p[1]}`)
                              .join(" ")}
                            fill={hexToRgba(color, fillOpacity)}
                            stroke={color}
                            strokeWidth={strokeWidth}
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
                              showPoly ? "transparent" : hexToRgba(color, 0.12)
                            }
                            stroke={color}
                            strokeWidth={strokeWidth}
                            vectorEffect="non-scaling-stroke"
                          />
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>

          {/* Right Column: Separate Class Color Legend + Individual Detection Instances */}
          <aside className="flex max-h-[76vh] flex-col border-t border-[#222226] bg-[#141416] p-4 md:col-span-4 md:border-l md:border-t-0 lg:col-span-3">
            <div>
              <p className="c1-kicker">CLASS COLOR NOTATION</p>
              <p className="mt-0.5 text-[11px] text-[#8A8A8E]">
                Hover or click a class to isolate its boxes on the frame.
              </p>

              <div className="mt-3 space-y-1.5">
                {classLegend.map((row) => {
                  const isActive =
                    highlightedClass?.toLowerCase() === row.name.toLowerCase();
                  return (
                    <button
                      key={row.name}
                      type="button"
                      onClick={() =>
                        setHighlightedClass((prev) =>
                          prev?.toLowerCase() === row.name.toLowerCase()
                            ? null
                            : row.name
                        )
                      }
                      onMouseEnter={() => setHighlightedClass(row.name)}
                      onMouseLeave={() => setHighlightedClass(null)}
                      className={`flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left text-xs transition-colors ${
                        isActive
                          ? "border-white/40 bg-[#222227]"
                          : "border-[#26262A] bg-[#0F0F12] hover:border-[#3A3A42]"
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className="h-3.5 w-3.5 shrink-0 rounded-sm border border-white/25"
                          style={{ backgroundColor: row.color }}
                        />
                        <span className="font-mono-tech font-medium text-white">
                          {row.name}
                        </span>
                      </div>
                      <span className="font-mono-tech rounded-full border border-[#303035] bg-[#1A1A1E] px-2 py-0.5 text-[11px] text-[#A1A1AA]">
                        {row.count}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-5 flex min-h-0 flex-1 flex-col border-t border-[#242429] pt-4">
              <div className="flex items-center justify-between">
                <p className="c1-kicker">
                  DETECTED INSTANCES ({item.detections.length})
                </p>
              </div>

              <div className="mt-2.5 flex-1 space-y-2 overflow-y-auto pr-1">
                {item.detections.length === 0 ? (
                  <p className="rounded-xl border border-dashed border-[#2C2C32] p-4 text-center text-xs text-[#8A8A8E]">
                    No detections on this frame.
                  </p>
                ) : (
                  item.detections.map((det, idx) => {
                    const color = getClassColor(det.label, classColors);
                    return (
                      <div
                        key={det.id}
                        onMouseEnter={() => setHoveredDetId(det.id)}
                        onMouseLeave={() => setHoveredDetId(null)}
                        className="flex items-center justify-between gap-2 rounded-xl border border-[#26262A] bg-[#0E0E11] px-3 py-2 text-xs hover:border-white/30"
                      >
                        <div className="flex min-w-0 items-center gap-2.5">
                          <span
                            className="h-3 w-3 shrink-0 rounded-full"
                            style={{ backgroundColor: color }}
                          />
                          <div className="min-w-0">
                            <div className="truncate font-mono-tech font-medium text-white">
                              #{idx + 1} · {det.label}
                            </div>
                            <div className="font-mono-tech text-[10px] text-[#8A8A8E]">
                              {(det.confidence * 100).toFixed(0)}% ·{" "}
                              {det.polygon && det.polygon.length >= 3
                                ? `${det.polygon.length}v poly`
                                : "bbox"}
                            </div>
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => handleDeleteSingleDet(det.id)}
                          disabled={busy}
                          aria-label={`Delete detection ${idx + 1}`}
                          title="Remove false-positive annotation"
                          className="rounded-lg p-1 text-[#8A8A8E] hover:bg-[#1E1E24] hover:text-rose-400"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </aside>
        </div>

        {/* Bottom Metadata + Keyboard Action Bar */}
        <div className="border-t border-[#242429] bg-[#0B0B0D] px-4 py-3 text-center">
          {saveError && (
            <div className="mx-auto mb-2.5 flex max-w-2xl items-center justify-center gap-2 rounded-xl border border-rose-500/50 bg-rose-500/15 px-3 py-1.5 text-xs text-rose-200">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
              <span>{saveError}</span>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-[#A1A1AA]">
            <span className="font-mono-tech">
              {item.filename} · {item.batchId} (#{displayNum} of{" "}
              {Math.max(totalFiltered, 1)}) · {item.detections.length}{" "}
              detection(s) ·{" "}
              <span
                className={
                  item.reviewStatus === "accepted"
                    ? "font-semibold text-emerald-400"
                    : item.reviewStatus === "rejected"
                      ? "font-semibold text-rose-400"
                      : "text-white"
                }
              >
                {statusText}
              </span>
            </span>

            {saveState === "saving" && (
              <span className="font-mono-tech inline-flex items-center gap-1 rounded-full border border-[#F2853F]/40 bg-[#F2853F]/15 px-2 py-0.5 text-[10px] text-[#F2853F]">
                <Loader2 className="h-2.5 w-2.5 animate-spin" />
                Saving...
              </span>
            )}
            {saveState === "saved" && (
              <span className="font-mono-tech inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-[10px] text-emerald-300">
                <Check className="h-2.5 w-2.5" />
                Saved locally
              </span>
            )}
          </div>

          <div className="mt-2.5 flex flex-wrap items-center justify-center gap-2">
            <button
              type="button"
              onClick={onPrev}
              className="inline-flex items-center gap-1 rounded-lg border border-[#303035] bg-[#16161A] px-3 py-1.5 text-xs text-[#A1A1AA] hover:border-white/35 hover:text-white"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Prev (←)
            </button>

            <button
              type="button"
              disabled={busy}
              onClick={() => void handleSetReview("accepted", null)}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors ${
                item.reviewStatus === "accepted"
                  ? "border border-emerald-400 bg-emerald-500 text-black"
                  : "border border-emerald-500/50 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/30"
              }`}
            >
              <Check className="h-3.5 w-3.5" />
              Accept (A)
            </button>

            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void handleSetReview(
                  item.reviewStatus === "rejected" && !item.rejectReason
                    ? "pending"
                    : "rejected",
                  null
                )
              }
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors ${
                item.reviewStatus === "rejected" && !item.rejectReason
                  ? "border border-rose-400 bg-rose-500 text-white"
                  : "border border-rose-500/50 bg-rose-500/15 text-rose-300 hover:bg-rose-500/30"
              }`}
            >
              <X className="h-3.5 w-3.5" />
              {item.reviewStatus === "rejected" && !item.rejectReason
                ? "Unreject (Space)"
                : "Reject (R / Space)"}
            </button>

            {REJECT_SHORTCUTS.map((sc) => {
              const active =
                item.reviewStatus === "rejected" &&
                item.rejectReason === sc.reason;
              return (
                <button
                  key={sc.reason}
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    void handleSetReview(
                      active ? "pending" : "rejected",
                      active ? null : sc.reason
                    )
                  }
                  className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    active
                      ? "border-[#F2853F] bg-[#F2853F] text-black"
                      : "border-[#303035] bg-[#16161A] text-[#D4D4D8] hover:border-white/35 hover:text-white"
                  }`}
                >
                  {sc.label}
                </button>
              );
            })}

            <button
              type="button"
              onClick={onNext}
              className="inline-flex items-center gap-1 rounded-lg border border-[#303035] bg-[#16161A] px-3 py-1.5 text-xs text-[#A1A1AA] hover:border-white/35 hover:text-white"
            >
              Next (→)
              <ChevronRight className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={onToggleAutoAdvance}
              className={`rounded-lg border px-2.5 py-1.5 text-xs transition-colors ${
                autoAdvance
                  ? "border-[#F2853F]/60 bg-[#F2853F]/15 text-[#F2853F]"
                  : "border-[#303035] bg-[#16161A] text-[#8A8A8E] hover:text-white"
              }`}
              title="Automatically jump to next frame after Accept or Reject"
            >
              Auto-next: {autoAdvance ? "On" : "Off"}
            </button>

            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-[#303035] bg-[#16161A] px-3 py-1.5 text-xs text-[#D4D4D8] hover:border-white/35 hover:text-white"
            >
              Close (Esc)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
