"use client";

import React from "react";
import {
  ArrowRight,
  BookmarkCheck,
  BookmarkPlus,
  Check,
  ChevronDown,
  ChevronRight,
  Loader2,
  Pin,
  Redo2,
  RotateCcw,
  Sparkles,
  Square,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { getClassColor } from "./color-utils";
import type {
  AutoDetection,
  AutoOutputMode,
  ClassColorEntry,
  FindAiMode,
  OwlModelId,
  OwlModelOption,
  PinnedVitFrameSummary,
  TargetClassOption,
  VisualDrawMode,
  VisualEmbedBatchTarget,
  VisualEmbedClassSummary,
  VisualEmbedExemplarBox,
  VisualEmbedMergeMode,
  VisualEmbedScope,
} from "./types";

interface VitClassBadge {
  label: string;
  count: number;
  imageCount: number;
}

interface FindObjectsAiPanelProps {
  disabled: boolean;
  findOpen: boolean;
  onToggleOpen: () => void;
  findMode: FindAiMode;
  onFindModeChange: (mode: FindAiMode) => void;
  owlModelId: OwlModelId;
  owlModelOptions: OwlModelOption[];
  onOwlModelChange: (modelId: OwlModelId) => void;
  classColors: ClassColorEntry[];
  activeClass: string;
  targetClassOptions: TargetClassOption[];
  selectedAiClasses: string[];
  onToggleAiClass: (className: string) => void;
  onSelectOnlyAiClass: (className: string) => void;
  onSelectAllAiClasses: () => void;
  onClearAiClasses: () => void;

  // ViT Ref (OWLv2 + SAM visual reference & embedding engine)
  refImageCount: number;
  refBoxCount: number;
  vitBankDirty: boolean;
  vitIndexing: boolean;
  vitClassBadges: VitClassBadge[];
  vitClassSummaries: VisualEmbedClassSummary[];
  currentItemDetCount: number;
  currentMatchingDetCount: number;
  isCurrentFramePinned: boolean;
  allAnnotatedPinned: boolean;
  pinnedVitFrames: PinnedVitFrameSummary[];
  vitBankFeedback: string | null;
  annotatedItemsCount: number;
  onTogglePinCurrentFrame: () => void;
  onUseAllAnnotatedRefs: () => void;
  onIndexVitRefs: () => void;
  onSelectPinnedItem: (itemId: string) => void;
  onUnpinItem: (itemId: string) => void;
  onClearVitRefs: () => void;

  visualQueryId: string;
  visualScope: VisualEmbedScope;
  onVisualScopeChange: (scope: VisualEmbedScope) => void;
  visualDrawMode: VisualDrawMode;
  onVisualDrawModeChange: (mode: VisualDrawMode) => void;
  visualBatchTarget: VisualEmbedBatchTarget;
  onVisualBatchTargetChange: (target: VisualEmbedBatchTarget) => void;
  visualMergeMode: VisualEmbedMergeMode;
  onVisualMergeModeChange: (mode: VisualEmbedMergeMode) => void;
  vitSkipAnnotated: boolean;
  onVitSkipAnnotatedChange: (skip: boolean) => void;

  visualThreshold: number;
  onVisualThresholdChange: (val: number) => void;
  visualMinObjectness: number;
  onVisualMinObjectnessChange: (val: number) => void;
  visualUseSamMasks: boolean;
  onVisualUseSamMasksChange: (val: boolean) => void;
  smartSamRefinerId: string;
  onSmartSamRefinerIdChange: (id: string) => void;
  visualAutoSearchOnSwitch: boolean;
  onVisualAutoSearchOnSwitchChange: (val: boolean) => void;

  visualHits: AutoDetection[];
  showVisualOverlay: boolean;
  onShowVisualOverlayChange: (show: boolean) => void;
  totalStagedOverlayItems: number;
  totalStagedOverlayBoxes: number;
  nextStagedOverlayItemId: string | null;
  onJumpToNextStagedOverlay: () => void;
  visualExemplars: VisualEmbedExemplarBox[];
  canUndoVisualEx: boolean;
  canRedoVisualEx: boolean;
  visualRedoCount: number;
  onUndoVisualExemplar: () => void;
  onRedoVisualExemplar: () => void;
  visualTopScore: number | null;
  visualTopObjectness: number | null;
  visualPosCount: number;
  visualNegCount: number;
  visualLastLatency: number | null;
  visualMatchedIou: number | null;
  visualBusy: boolean;
  visualError: string | null;
  visualBatchProgress: {
    running: boolean;
    done: number;
    total: number;
    totalAdded: number;
    itemsWithMatches: number;
    currentFilename: string;
  } | null;
  activeBatchId: string;
  currentBatchCount: number;
  allItemsCount: number;
  queueItemsCount: number;

  onDetectCurrentImage: () => void;
  onCommitVisualHits: () => void;
  onRejectCurrentVisualOverlay: () => void;
  onClearAllStagedOverlays: () => void;
  onDismissVisualHit: (hitId: string) => void;
  onResetVisualQuery: () => void;
  onRunVisualBatch: () => void;

  // Text prompt modes (OWL Text / YOLOE-26s)
  findClassesText: string;
  onFindClassesTextChange: (val: string) => void;
  findOutputMode: AutoOutputMode;
  onFindOutputModeChange: (val: AutoOutputMode) => void;
  findConf: number;
  onFindConfChange: (val: number) => void;
  findBusy: boolean;
  findMessage: string | null;
  onRunFindObjects: () => void;
}

export function FindObjectsAiPanel({
  disabled,
  findOpen,
  onToggleOpen,
  findMode,
  onFindModeChange,
  owlModelId,
  owlModelOptions,
  onOwlModelChange,
  classColors,
  activeClass,
  targetClassOptions,
  selectedAiClasses,
  onToggleAiClass,
  onSelectOnlyAiClass,
  onSelectAllAiClasses,
  onClearAiClasses,
  refImageCount,
  refBoxCount,
  vitBankDirty,
  vitIndexing,
  vitClassBadges,
  vitClassSummaries,
  currentItemDetCount,
  currentMatchingDetCount,
  isCurrentFramePinned,
  allAnnotatedPinned,
  pinnedVitFrames,
  vitBankFeedback,
  annotatedItemsCount,
  onTogglePinCurrentFrame,
  onUseAllAnnotatedRefs,
  onIndexVitRefs,
  onSelectPinnedItem,
  onUnpinItem,
  onClearVitRefs,
  visualQueryId,
  visualScope,
  onVisualScopeChange,
  visualDrawMode,
  onVisualDrawModeChange,
  visualBatchTarget,
  onVisualBatchTargetChange,
  visualMergeMode,
  onVisualMergeModeChange,
  vitSkipAnnotated,
  onVitSkipAnnotatedChange,
  visualThreshold,
  onVisualThresholdChange,
  visualMinObjectness,
  onVisualMinObjectnessChange,
  visualUseSamMasks,
  onVisualUseSamMasksChange,
  smartSamRefinerId,
  onSmartSamRefinerIdChange,
  visualAutoSearchOnSwitch,
  onVisualAutoSearchOnSwitchChange,
  visualHits,
  showVisualOverlay,
  onShowVisualOverlayChange,
  totalStagedOverlayItems,
  totalStagedOverlayBoxes,
  nextStagedOverlayItemId,
  onJumpToNextStagedOverlay,
  visualExemplars,
  canUndoVisualEx,
  canRedoVisualEx,
  visualRedoCount,
  onUndoVisualExemplar,
  onRedoVisualExemplar,
  visualTopScore,
  visualTopObjectness,
  visualPosCount,
  visualNegCount,
  visualLastLatency,
  visualMatchedIou,
  visualBusy,
  visualError,
  visualBatchProgress,
  activeBatchId,
  currentBatchCount,
  allItemsCount,
  queueItemsCount,
  onDetectCurrentImage,
  onCommitVisualHits,
  onRejectCurrentVisualOverlay,
  onClearAllStagedOverlays,
  onDismissVisualHit,
  onResetVisualQuery,
  onRunVisualBatch,
  findClassesText,
  onFindClassesTextChange,
  findOutputMode,
  onFindOutputModeChange,
  findConf,
  onFindConfChange,
  findBusy,
  findMessage,
  onRunFindObjects,
}: FindObjectsAiPanelProps) {
  const [pinnedListOpen, setPinnedListOpen] = React.useState(false);

  const selectedAiClassSet = React.useMemo(
    () =>
      new Set(
        selectedAiClasses.map((c) => c.trim().toLowerCase()).filter(Boolean)
      ),
    [selectedAiClasses]
  );
  const hasAnySelectedClass =
    targetClassOptions.length === 0 || selectedAiClasses.length > 0;
  const allAiClassesSelected =
    targetClassOptions.length > 0 &&
    targetClassOptions.every((opt) =>
      selectedAiClassSet.has(opt.name.trim().toLowerCase())
    );

  const promptClassList = React.useMemo(
    () =>
      findClassesText
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    [findClassesText]
  );
  const promptClassSet = React.useMemo(
    () => new Set(promptClassList.map((s) => s.toLowerCase())),
    [promptClassList]
  );
  const handleTogglePromptClass = React.useCallback(
    (clsName: string) => {
      const low = clsName.trim().toLowerCase();
      if (!low) return;
      const exists = promptClassList.some((c) => c.toLowerCase() === low);
      const next = exists
        ? promptClassList.filter((c) => c.toLowerCase() !== low)
        : [...promptClassList, clsName];
      onFindClassesTextChange(next.join(", "));
    },
    [onFindClassesTextChange, promptClassList]
  );

  const displayClassChips =
    vitClassSummaries.length > 0 && !vitBankDirty
      ? vitClassSummaries.map((s) => ({
          label: s.label,
          count: s.exemplarCount,
          imageCount: s.imageCount,
        }))
      : vitClassBadges;

  const hasQueryOrRefs = refBoxCount > 0 || Boolean(visualQueryId);

  return (
    <div className="mt-4 border-t border-[#222226] pt-3">
      <button
        type="button"
        disabled={disabled}
        onClick={onToggleOpen}
        className={`flex w-full items-center justify-between gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition-colors disabled:opacity-40 ${
          findOpen
            ? "border-[#F2853F] bg-[#F2853F]/15 text-white"
            : "border-[#F2853F]/45 bg-[#141416] text-white hover:border-[#F2853F]"
        }`}
      >
        <span className="inline-flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-[#F2853F]" />
          Find Objects with AI
        </span>
        {refBoxCount > 0 && (
          <span className="font-mono-tech rounded-full border border-[#F2853F]/50 bg-[#F2853F]/15 px-1.5 py-0.5 text-[10px] font-semibold text-[#F2853F]">
            {refBoxCount} ref{refBoxCount === 1 ? "" : "s"}
          </span>
        )}
      </button>

      {findOpen && (
        <div className="c1-inset mt-2.5 space-y-2.5 p-2.5">
          {/* 3 AI Engines: ViT Ref | OWL Text | YOLOE */}
          <div className="grid grid-cols-3 rounded-lg border border-[#303035] bg-[#0A0A0C] p-0.5 text-[10px]">
            {(
              [
                { id: "vit_ref", label: "ViT Ref" },
                { id: "owl_text", label: "OWL Text" },
                { id: "yoloe", label: "YOLOE" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => onFindModeChange(tab.id)}
                className={`rounded-md py-1 font-medium transition-colors ${
                  findMode === tab.id
                    ? "bg-[#F2853F] text-black"
                    : "text-[#A1A1AA] hover:text-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {findMode !== "yoloe" && owlModelOptions.length > 0 && (() => {
            const activeOwl =
              owlModelOptions.find((o) => o.hfModelId === owlModelId) ??
              owlModelOptions[0];
            return (
              <div className="rounded-lg border border-[#26262A] bg-[#0B0B0D] p-2 space-y-1">
                <div className="flex items-center justify-between gap-1">
                  <span className="font-mono-tech text-[9px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                    OWLv2 Backbone
                  </span>
                  <span
                    className={`font-mono-tech text-[9px] ${
                      activeOwl.cached ? "text-emerald-300" : "text-amber-300"
                    }`}
                  >
                    {activeOwl.cached ? "Cached" : activeOwl.approxSize}
                  </span>
                </div>
                <select
                  value={owlModelId}
                  disabled={
                    vitIndexing ||
                    visualBusy ||
                    findBusy ||
                    Boolean(visualBatchProgress?.running)
                  }
                  onChange={(e) => onOwlModelChange(e.target.value as OwlModelId)}
                  aria-label="Select OWLv2 model checkpoint"
                  className="w-full rounded-md border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[11px] text-white focus:border-[#F2853F] focus:outline-none disabled:opacity-45"
                >
                  {owlModelOptions.map((opt) => (
                    <option key={opt.hfModelId} value={opt.hfModelId}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="text-[9px] leading-snug text-[#8A8A8E]">
                  {activeOwl.description}
                </p>
              </div>
            );
          })()}

          {findMode === "vit_ref" ? (
            <div className="space-y-2.5">
              {/* Target Classes Filter (1 or Multiple Classes) */}
              {targetClassOptions.length > 0 && (
                <div className="rounded-lg border border-[#26262A] bg-[#0B0B0D] p-2 space-y-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono-tech text-[10px] font-semibold uppercase tracking-[0.1em] text-white">
                      Target Classes ({selectedAiClasses.length}/
                      {targetClassOptions.length})
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        disabled={
                          vitIndexing ||
                          Boolean(visualBatchProgress?.running) ||
                          allAiClassesSelected
                        }
                        onClick={onSelectAllAiClasses}
                        title="Select all classes"
                        className="rounded border border-[#303035] bg-[#141416] px-1.5 py-0.5 font-mono-tech text-[9px] text-[#D4D4D8] hover:border-[#F2853F] hover:text-white disabled:opacity-40"
                      >
                        All
                      </button>
                      <button
                        type="button"
                        disabled={
                          vitIndexing ||
                          Boolean(visualBatchProgress?.running) ||
                          selectedAiClasses.length === 0
                        }
                        onClick={onClearAiClasses}
                        title="Unselect all classes"
                        className="rounded border border-[#303035] bg-[#141416] px-1.5 py-0.5 font-mono-tech text-[9px] text-[#8A8A8E] hover:border-white/30 hover:text-white disabled:opacity-40"
                      >
                        None
                      </button>
                    </div>
                  </div>

                  <div className="max-h-28 space-y-1 overflow-y-auto pr-0.5">
                    {targetClassOptions.map((opt) => {
                      const low = opt.name.trim().toLowerCase();
                      const checked = selectedAiClassSet.has(low);
                      const isOnly =
                        checked && selectedAiClasses.length === 1;
                      return (
                        <div
                          key={opt.name}
                          className={`flex items-center gap-1 rounded-md border px-1.5 py-1 transition-colors ${
                            checked
                              ? "border-[#F2853F]/50 bg-[#F2853F]/10"
                              : "border-[#232328] bg-[#121215] opacity-75"
                          }`}
                        >
                          <button
                            type="button"
                            disabled={
                              vitIndexing ||
                              Boolean(visualBatchProgress?.running)
                            }
                            onClick={() => onToggleAiClass(opt.name)}
                            aria-pressed={checked}
                            className="flex min-w-0 flex-1 items-center gap-1.5 text-left font-mono-tech text-[10px] disabled:opacity-45"
                          >
                            <span
                              className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border ${
                                checked
                                  ? "border-[#F2853F] bg-[#F2853F] text-black"
                                  : "border-[#4A4A52] bg-[#0A0A0C] text-transparent"
                              }`}
                            >
                              <Check className="h-2.5 w-2.5 stroke-[3]" />
                            </span>
                            <span
                              className="h-2 w-2 shrink-0 rounded-full"
                              style={{ backgroundColor: opt.color }}
                            />
                            <span
                              className={`truncate ${
                                checked ? "text-white font-medium" : "text-[#A1A1AA]"
                              }`}
                            >
                              {opt.name}
                            </span>
                            <span className="ml-auto shrink-0 text-[9px] text-[#8A8A8E]">
                              {opt.frameCount}f · {opt.boxCount}b
                            </span>
                          </button>
                          <button
                            type="button"
                            disabled={
                              vitIndexing ||
                              Boolean(visualBatchProgress?.running) ||
                              isOnly
                            }
                            onClick={() => onSelectOnlyAiClass(opt.name)}
                            title={`Index & search only "${opt.name}"`}
                            className={`shrink-0 rounded border px-1 py-0.5 font-mono-tech text-[9px] transition-colors disabled:opacity-40 ${
                              isOnly
                                ? "border-[#F2853F]/60 bg-[#F2853F]/20 text-[#F2853F]"
                                : "border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:border-[#F2853F] hover:text-white"
                            }`}
                          >
                            Only
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  {!hasAnySelectedClass ? (
                    <p className="font-mono-tech text-[9px] leading-snug text-amber-300">
                      Select at least 1 class above to index or search across frames.
                    </p>
                  ) : !allAiClassesSelected ? (
                    <p className="font-mono-tech text-[9px] leading-snug text-[#F2853F]/90">
                      Filtering indexing &amp; batch search to {selectedAiClasses.length} selected class
                      {selectedAiClasses.length === 1 ? "" : "es"}.
                    </p>
                  ) : null}
                </div>
              )}

              {/* ViT Reference Grounding Bank */}
              <div className="rounded-lg border border-[#26262A] bg-[#0B0B0D] p-2 space-y-2">
                <div className="flex items-center justify-between gap-1">
                  <span className="font-mono-tech inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white">
                    <Pin className="h-3 w-3 text-[#F2853F]" />
                    ViT Reference Bank
                  </span>
                  <span className="font-mono-tech text-[10px] text-[#A1A1AA]">
                    {refImageCount} imgs · {refBoxCount} boxes
                  </span>
                </div>

                {/* Live Active / Indexed Status Indicator */}
                {vitIndexing ? (
                  <div className="flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 font-mono-tech text-[10px] text-amber-200">
                    <Loader2 className="h-3 w-3 shrink-0 animate-spin text-amber-300" />
                    <span>Indexing OWLv2 embeddings...</span>
                  </div>
                ) : refBoxCount > 0 &&
                  !vitBankDirty &&
                  Boolean(visualQueryId) &&
                  hasAnySelectedClass ? (
                  <div className="flex items-center justify-between rounded-md border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 font-mono-tech text-[10px] text-emerald-200">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      Active &amp; Indexed · {refImageCount} img{refImageCount === 1 ? "" : "s"} · {refBoxCount} box{refBoxCount === 1 ? "" : "es"}
                    </span>
                  </div>
                ) : refBoxCount > 0 ? (
                  <div className="flex items-center justify-between gap-1 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 font-mono-tech text-[10px] text-amber-200">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                      Selected · Not indexed yet
                    </span>
                    <button
                      type="button"
                      disabled={
                        Boolean(visualBatchProgress?.running) ||
                        vitIndexing ||
                        !hasAnySelectedClass
                      }
                      onClick={onIndexVitRefs}
                      className="rounded border border-amber-400/50 bg-amber-400/20 px-1.5 py-0.5 text-[9px] font-semibold text-amber-100 hover:bg-amber-400/30 disabled:opacity-40"
                    >
                      Index Now
                    </button>
                  </div>
                ) : null}

                {displayClassChips.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {displayClassChips.map((chip) => {
                      const color = getClassColor(chip.label, classColors);
                      return (
                        <span
                          key={chip.label}
                          className="font-mono-tech inline-flex items-center gap-1 rounded-full border border-[#303035] bg-[#141416] px-2 py-0.5 text-[10px] text-[#E4E4E7]"
                        >
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: color }}
                          />
                          <span className="max-w-[96px] truncate">{chip.label}</span>
                          <span className="text-[#8A8A8E]">×{chip.count}</span>
                        </span>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-[10px] leading-snug text-[#8A8A8E]">
                    Pin annotated boxes/frames as OWLv2 grounding references, or draw an example box directly on the canvas.
                  </p>
                )}

                {vitBankFeedback && (
                  <p className="font-mono-tech text-[9px] leading-snug text-emerald-300/90">
                    {vitBankFeedback}
                  </p>
                )}

                <div className="grid grid-cols-2 gap-1">
                  <button
                    type="button"
                    disabled={
                      vitIndexing ||
                      Boolean(visualBatchProgress?.running) ||
                      !hasAnySelectedClass ||
                      (!isCurrentFramePinned && currentMatchingDetCount === 0)
                    }
                    onClick={onTogglePinCurrentFrame}
                    title={
                      !hasAnySelectedClass
                        ? "Select at least one target class above first"
                        : isCurrentFramePinned
                          ? "Click to unpin the current frame from the ViT Reference Bank"
                          : currentMatchingDetCount > 0
                            ? `Pin this frame's ${currentMatchingDetCount} matching box(es) into the ViT Reference Bank`
                            : currentItemDetCount > 0
                              ? "Current frame has 0 boxes matching selected target classes"
                              : "Current image has 0 boxes — draw a box or switch to an annotated frame to pin it"
                    }
                    className={`inline-flex items-center justify-center gap-1 rounded-lg border px-1.5 py-1 text-[10px] font-medium transition-colors disabled:opacity-40 ${
                      isCurrentFramePinned
                        ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-200"
                        : "border-[#303035] bg-[#141416] text-[#D4D4D8] hover:border-[#F2853F] hover:text-white"
                    }`}
                  >
                    {isCurrentFramePinned ? (
                      <>
                        <BookmarkCheck className="h-3 w-3 shrink-0" />
                        <span>Frame Pinned ({currentMatchingDetCount})</span>
                      </>
                    ) : currentMatchingDetCount > 0 ? (
                      <>
                        <BookmarkPlus className="h-3 w-3 shrink-0 text-[#F2853F]" />
                        <span>Use Frame (+{currentMatchingDetCount})</span>
                      </>
                    ) : (
                      <>
                        <BookmarkPlus className="h-3 w-3 shrink-0 text-[#8A8A8E]" />
                        <span>
                          {currentItemDetCount > 0
                            ? "Frame (0 match)"
                            : "Frame (0 boxes)"}
                        </span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    disabled={
                      vitIndexing ||
                      Boolean(visualBatchProgress?.running) ||
                      !hasAnySelectedClass ||
                      annotatedItemsCount === 0
                    }
                    onClick={onUseAllAnnotatedRefs}
                    title={
                      !hasAnySelectedClass
                        ? "Select at least one target class above first"
                        : allAnnotatedPinned && !vitBankDirty && Boolean(visualQueryId)
                          ? "All matching annotated frames are active & indexed in the ViT Reference Bank (click to re-sync)"
                          : "Pin matching annotated frames & index OWLv2 embeddings"
                    }
                    className={`inline-flex items-center justify-center gap-1 rounded-lg border px-1.5 py-1 text-[10px] font-medium transition-colors disabled:opacity-40 ${
                      allAnnotatedPinned && !vitBankDirty && Boolean(visualQueryId)
                        ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-200"
                        : allAnnotatedPinned
                          ? "border-amber-500/55 bg-amber-500/15 text-amber-200"
                          : "border-[#303035] bg-[#141416] text-[#D4D4D8] hover:border-[#F2853F] hover:text-white"
                    }`}
                  >
                    {vitIndexing ? (
                      <Loader2 className="h-3 w-3 shrink-0 animate-spin text-[#F2853F]" />
                    ) : allAnnotatedPinned && !vitBankDirty && Boolean(visualQueryId) ? (
                      <Check className="h-3 w-3 shrink-0 text-emerald-300" />
                    ) : (
                      <Pin className="h-3 w-3 shrink-0 text-[#F2853F]" />
                    )}
                    <span>
                      {allAnnotatedPinned && !vitBankDirty && Boolean(visualQueryId)
                        ? `Annotated Active (${annotatedItemsCount})`
                        : allAnnotatedPinned
                          ? `Index Annotated (${annotatedItemsCount})`
                          : `Use Annotated (${annotatedItemsCount})`}
                    </span>
                  </button>
                </div>

                {!isCurrentFramePinned &&
                  currentMatchingDetCount === 0 &&
                  annotatedItemsCount > 0 && (
                    <p className="text-[10px] leading-snug text-[#8A8A8E]">
                      {currentItemDetCount > 0
                        ? "Current frame has no boxes matching the selected classes (use Annotated bank or toggle classes above)."
                        : "This frame has 0 boxes (draw a box to pin it, or use the active Annotated bank)."}
                    </p>
                  )}

                {pinnedVitFrames.length > 0 && (
                  <div className="border-t border-[#202024] pt-1.5">
                    <button
                      type="button"
                      onClick={() => setPinnedListOpen((v) => !v)}
                      className="flex w-full items-center justify-between font-mono-tech text-[10px] text-[#A1A1AA] hover:text-white"
                    >
                      <span className="inline-flex items-center gap-1">
                        {pinnedListOpen ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                        View Selected Frames ({pinnedVitFrames.length})
                      </span>
                      <span className="text-[9px] text-[#8A8A8E]">
                        {refBoxCount} box{refBoxCount === 1 ? "" : "es"}
                      </span>
                    </button>
                    {pinnedListOpen && (
                      <div className="mt-1.5 max-h-32 space-y-1 overflow-y-auto pr-0.5">
                        {pinnedVitFrames.map((f) => (
                          <div
                            key={f.itemId}
                            className="flex items-center justify-between gap-1 rounded border border-[#232328] bg-[#121215] px-1.5 py-1 font-mono-tech text-[10px]"
                          >
                            <button
                              type="button"
                              onClick={() => onSelectPinnedItem(f.itemId)}
                              title={`Jump to #${f.seqIndex} ${f.filename} (${f.classes.join(", ")})`}
                              className="min-w-0 flex-1 truncate text-left text-[#D4D4D8] hover:text-[#F2853F]"
                            >
                              #{f.seqIndex} {f.filename} · {f.boxCount} box
                              {f.boxCount === 1 ? "" : "es"}
                            </button>
                            <button
                              type="button"
                              disabled={vitIndexing || Boolean(visualBatchProgress?.running)}
                              onClick={() => onUnpinItem(f.itemId)}
                              title="Remove frame from ViT reference bank"
                              aria-label={`Unpin ${f.filename}`}
                              className="shrink-0 rounded p-0.5 text-[#8A8A8E] hover:bg-rose-500/15 hover:text-rose-300 disabled:opacity-40"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {refBoxCount > 0 && (
                  <button
                    type="button"
                    disabled={vitIndexing || Boolean(visualBatchProgress?.running)}
                    onClick={onClearVitRefs}
                    className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-[#2C2C32] bg-[#0A0A0C] py-1 text-[10px] text-[#A1A1AA] hover:border-rose-500/40 hover:text-rose-300 disabled:opacity-40"
                  >
                    <Trash2 className="h-3 w-3" />
                    Clear Pinned Refs
                  </button>
                )}
              </div>

              {/* Ad-hoc Canvas Draw Exemplar Controls */}
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="font-mono-tech uppercase text-[#8A8A8E]">
                    Canvas Draw
                  </span>
                  <span className="font-mono-tech truncate text-[#D4D4D8]">
                    Class:{" "}
                    <span className="font-semibold text-white">
                      {activeClass}
                    </span>
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <button
                    type="button"
                    disabled={visualBatchProgress?.running}
                    onClick={() =>
                      onVisualDrawModeChange(
                        visualDrawMode === "positive" ? null : "positive"
                      )
                    }
                    className={`rounded-lg border px-2 py-1 text-[11px] font-medium transition-colors disabled:opacity-40 ${
                      visualDrawMode === "positive"
                        ? "border-sky-400 bg-sky-500/25 text-sky-200"
                        : "border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    + Draw Example
                  </button>
                  <button
                    type="button"
                    disabled={
                      visualBatchProgress?.running ||
                      (visualPosCount === 0 && refBoxCount === 0)
                    }
                    onClick={() =>
                      onVisualDrawModeChange(
                        visualDrawMode === "negative" ? null : "negative"
                      )
                    }
                    title="Or hold Alt + Drag while in draw mode"
                    className={`rounded-lg border px-2 py-1 text-[11px] font-medium transition-colors disabled:opacity-40 ${
                      visualDrawMode === "negative"
                        ? "border-rose-400 bg-rose-500/25 text-rose-200"
                        : "border-[#303035] bg-[#0A0A0C] text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    − Negative
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <button
                    type="button"
                    disabled={
                      visualBusy ||
                      vitIndexing ||
                      Boolean(visualBatchProgress?.running) ||
                      !canUndoVisualEx
                    }
                    onClick={onUndoVisualExemplar}
                    title="Undo last drawn example / negative box"
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[10px] text-[#D4D4D8] hover:border-white/30 hover:text-white disabled:opacity-35"
                  >
                    <Undo2 className="h-3 w-3" />
                    Undo Draw
                  </button>
                  <button
                    type="button"
                    disabled={
                      visualBusy ||
                      vitIndexing ||
                      Boolean(visualBatchProgress?.running) ||
                      !canRedoVisualEx
                    }
                    onClick={onRedoVisualExemplar}
                    title="Redo drawn example / negative box"
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[10px] text-[#D4D4D8] hover:border-white/30 hover:text-white disabled:opacity-35"
                  >
                    <Redo2 className="h-3 w-3" />
                    Redo{visualRedoCount > 0 ? ` (${visualRedoCount})` : ""}
                  </button>
                </div>

                <div className="flex items-center justify-between pt-0.5 font-mono-tech text-[9px] text-[#8A8A8E]">
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-xs bg-emerald-400" />
                    Match
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-xs bg-sky-400" />
                    Example
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-xs bg-rose-400" />
                    Negative
                  </span>
                </div>
              </div>

              {/* Search Scope: Same Image vs Across Batch */}
              <div>
                <span className="font-mono-tech block text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Search Scope
                </span>
                <div className="mt-1 grid grid-cols-2 rounded-lg border border-[#303035] bg-[#0A0A0C] p-0.5 text-[11px]">
                  <button
                    type="button"
                    onClick={() => onVisualScopeChange("same_image")}
                    className={`rounded-md py-1 font-medium transition-colors ${
                      visualScope === "same_image"
                        ? "bg-[#F2853F] text-black"
                        : "text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    Same Image
                  </button>
                  <button
                    type="button"
                    onClick={() => onVisualScopeChange("across_batch")}
                    className={`rounded-md py-1 font-medium transition-colors ${
                      visualScope === "across_batch"
                        ? "bg-[#F2853F] text-black"
                        : "text-[#A1A1AA] hover:text-white"
                    }`}
                  >
                    Across Batch
                  </button>
                </div>
              </div>

              {/* Sliders */}
              <label className="block text-[11px] text-[#A1A1AA]">
                <span className="flex items-center justify-between">
                  <span>Threshold</span>
                  <span className="font-mono-tech font-semibold text-white">
                    {visualThreshold.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0.3}
                  max={0.98}
                  step={0.01}
                  value={visualThreshold}
                  aria-label="Visual Embed Threshold"
                  onChange={(e) => onVisualThresholdChange(Number(e.target.value))}
                  className="mt-1 w-full accent-[#F2853F]"
                />
              </label>

              <label className="block text-[11px] text-[#A1A1AA]">
                <span className="flex items-center justify-between">
                  <span>Min Objectness</span>
                  <span className="font-mono-tech font-semibold text-white">
                    {visualMinObjectness.toFixed(3)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0.005}
                  max={0.5}
                  step={0.005}
                  value={visualMinObjectness}
                  aria-label="Visual Embed Min Objectness"
                  onChange={(e) =>
                    onVisualMinObjectnessChange(Number(e.target.value))
                  }
                  className="mt-1 w-full accent-[#F2853F]"
                />
              </label>

              {/* Overlay Visibility, SAM Mask & Auto-Search Toggles */}
              <div className="space-y-1.5 text-[11px] text-[#D4D4D8]">
                <label className="flex cursor-pointer items-center justify-between gap-2">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showVisualOverlay}
                      onChange={(e) => onShowVisualOverlayChange(e.target.checked)}
                      className="accent-[#F2853F]"
                    />
                    <span>Show AI Match Overlay</span>
                  </span>
                  {totalStagedOverlayItems > 0 && (
                    <span className="font-mono-tech text-[10px] text-emerald-300">
                      {totalStagedOverlayBoxes}/{totalStagedOverlayItems} imgs
                    </span>
                  )}
                </label>

                <div className="flex items-center justify-between gap-2">
                  <label className="flex cursor-pointer items-center gap-2">
                    <input
                      type="checkbox"
                      checked={visualUseSamMasks}
                      onChange={(e) => onVisualUseSamMasksChange(e.target.checked)}
                      className="accent-[#F2853F]"
                    />
                    <span>Use SAM 2.1 masks</span>
                  </label>
                  {visualUseSamMasks && (
                    <select
                      value={smartSamRefinerId}
                      onChange={(e) => onSmartSamRefinerIdChange(e.target.value)}
                      aria-label="SAM model for ViT mask refinement"
                      className="font-mono-tech rounded border border-[#303035] bg-[#0A0A0C] px-1.5 py-0.5 text-[10px] text-white"
                    >
                      <option value="sam2.1-s">SAM2.1-S</option>
                      <option value="sam2.1-t">SAM2.1-T</option>
                      <option value="mobile-sam">MobileSAM</option>
                    </select>
                  )}
                </div>

                {visualScope === "same_image" && (
                  <label className="flex cursor-pointer items-center gap-2">
                    <input
                      type="checkbox"
                      checked={visualAutoSearchOnSwitch}
                      onChange={(e) =>
                        onVisualAutoSearchOnSwitchChange(e.target.checked)
                      }
                      className="accent-[#F2853F]"
                    />
                    <span>Auto-search on image switch</span>
                  </label>
                )}
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-5 gap-1 rounded-lg border border-[#26262A] bg-[#0B0B0D] px-2 py-1.5 text-center font-mono-tech">
                <div>
                  <div className="text-[9px] text-[#8A8A8E]">Hits</div>
                  <div className="text-[11px] font-semibold text-emerald-300">
                    {visualHits.length}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-[#8A8A8E]">Top</div>
                  <div className="text-[11px] font-semibold text-white">
                    {visualTopScore !== null ? visualTopScore.toFixed(2) : "—"}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-[#8A8A8E]">Ex+</div>
                  <div className="text-[11px] font-semibold text-sky-300">
                    {vitBankDirty && refBoxCount > 0 ? refBoxCount : visualPosCount}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-[#8A8A8E]">Neg−</div>
                  <div className="text-[11px] font-semibold text-rose-300">
                    {visualNegCount}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-[#8A8A8E]">Sec</div>
                  <div className="text-[11px] font-semibold text-[#D4D4D8]">
                    {visualLastLatency !== null
                      ? visualLastLatency.toFixed(2)
                      : "—"}
                  </div>
                </div>
              </div>

              {!visualBusy &&
                !vitIndexing &&
                visualHits.length === 0 &&
                visualTopScore !== null &&
                visualTopScore >= visualThreshold && (
                  <div className="space-y-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2 text-[10px] leading-snug text-amber-200">
                    <p>
                      Top similarity ({visualTopScore.toFixed(2)}) beat
                      threshold ({visualThreshold.toFixed(2)}), but 0 boxes
                      passed{" "}
                      <span className="font-semibold text-white">
                        Min Objectness ({visualMinObjectness.toFixed(3)})
                      </span>
                      {visualTopObjectness !== null
                        ? ` (top obj: ${visualTopObjectness.toFixed(3)})`
                        : ""}
                      {visualNegCount > 0 ? " or negative filter" : ""}.
                    </p>
                    {visualMinObjectness > 0.005 && (
                      <button
                        type="button"
                        onClick={() =>
                          onVisualMinObjectnessChange(
                            visualMinObjectness > 0.01 ? 0.01 : 0.005
                          )
                        }
                        className="inline-flex w-full items-center justify-center rounded border border-amber-400/50 bg-amber-400/20 px-2 py-1 font-mono-tech text-[10px] font-semibold text-amber-100 hover:bg-amber-400/30"
                      >
                        Set Min Objectness to{" "}
                        {visualMinObjectness > 0.01 ? "0.010" : "0.005"}
                      </button>
                    )}
                  </div>
                )}

              {!visualBusy &&
                !vitIndexing &&
                visualHits.length === 0 &&
                visualTopScore !== null &&
                visualTopScore > 0 &&
                visualTopScore < visualThreshold && (
                  <p className="font-mono-tech text-[10px] leading-snug text-[#8A8A8E]">
                    Best match on this frame was {visualTopScore.toFixed(2)}{" "}
                    (below threshold {visualThreshold.toFixed(2)}).
                  </p>
                )}

              {visualError && (
                <p className="rounded-lg border border-rose-500/40 bg-rose-500/12 px-2 py-1.5 text-[10px] text-rose-200">
                  {visualError}
                </p>
              )}

              {/* Staged Overlay Accept / Reject Bar (shown in both Same Image & Across Batch) */}
              {visualHits.length > 0 && (
                <div className="space-y-1.5 rounded-lg border border-emerald-500/35 bg-emerald-500/10 p-2">
                  <div className="flex items-center justify-between font-mono-tech text-[10px] text-emerald-200">
                    <span>Staged Overlay on Frame</span>
                    <span className="font-semibold">
                      {visualHits.length} box{visualHits.length === 1 ? "" : "es"}
                    </span>
                  </div>
                  <button
                    type="button"
                    disabled={visualBusy || vitIndexing}
                    onClick={onCommitVisualHits}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-[#F2853F] px-3 py-1.5 text-xs font-semibold text-black hover:bg-[#FF9652] disabled:opacity-40"
                  >
                    <Check className="h-3.5 w-3.5" />
                    Accept {visualHits.length} Overlay Match
                    {visualHits.length === 1 ? "" : "es"} (Enter)
                  </button>
                  <button
                    type="button"
                    disabled={visualBusy || vitIndexing}
                    onClick={onRejectCurrentVisualOverlay}
                    className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-rose-500/40 bg-rose-500/12 py-1 text-[11px] font-medium text-rose-200 hover:bg-rose-500/22 disabled:opacity-40"
                  >
                    <X className="h-3 w-3" />
                    Reject Overlay (Esc)
                  </button>
                </div>
              )}

              {/* Scope-Specific Actions */}
              {visualScope === "same_image" ? (
                <div className="space-y-1.5">
                  <button
                    type="button"
                    disabled={
                      visualBusy ||
                      vitIndexing ||
                      !hasQueryOrRefs ||
                      !hasAnySelectedClass
                    }
                    onClick={onDetectCurrentImage}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-[#F2853F]/60 bg-[#F2853F]/15 px-3 py-1.5 text-xs font-medium text-white hover:bg-[#F2853F]/25 disabled:opacity-40"
                  >
                    {visualBusy || vitIndexing ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-[#F2853F]" />
                    ) : (
                      <Sparkles className="h-3.5 w-3.5 text-[#F2853F]" />
                    )}
                    {vitIndexing
                      ? "Indexing ViT Refs..."
                      : visualBusy
                        ? "Detecting..."
                        : "Detect on This Image"}
                  </button>

                  <button
                    type="button"
                    disabled={
                      visualBusy ||
                      vitIndexing ||
                      (!visualQueryId &&
                        visualExemplars.length === 0 &&
                        visualRedoCount === 0 &&
                        totalStagedOverlayItems === 0)
                    }
                    onClick={onResetVisualQuery}
                    className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] py-1 text-[11px] text-[#A1A1AA] hover:text-white disabled:opacity-40"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset Query
                  </button>
                </div>
              ) : (
                <div className="space-y-2 border-t border-[#26262A] pt-2">
                  <label className="block">
                    <span className="font-mono-tech text-[9px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                      Batch Target
                    </span>
                    <select
                      value={visualBatchTarget}
                      onChange={(e) =>
                        onVisualBatchTargetChange(
                          e.target.value as VisualEmbedBatchTarget
                        )
                      }
                      aria-label="Batch visual search target"
                      className="mt-1 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[11px] text-white focus:border-[#F2853F] focus:outline-none"
                    >
                      <option value="current_batch">
                        Current Batch (
                        {activeBatchId === "all"
                          ? `All Batches · ${allItemsCount} imgs`
                          : `${activeBatchId} · ${currentBatchCount} imgs`}
                        )
                      </option>
                      <option value="all_batches">
                        All Batches ({allItemsCount} imgs)
                      </option>
                      <option value="filtered_queue">
                        Filtered Queue ({queueItemsCount} imgs)
                      </option>
                    </select>
                  </label>

                  <label className="flex cursor-pointer items-center gap-2 text-[11px] text-[#D4D4D8]">
                    <input
                      type="checkbox"
                      checked={vitSkipAnnotated}
                      onChange={(e) => onVitSkipAnnotatedChange(e.target.checked)}
                      className="accent-[#F2853F]"
                    />
                    <span>Skip annotated &amp; ref frames</span>
                  </label>

                  <label className="block">
                    <span className="font-mono-tech text-[9px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                      When Accepting Overlay
                    </span>
                    <select
                      value={visualMergeMode}
                      onChange={(e) =>
                        onVisualMergeModeChange(
                          e.target.value as VisualEmbedMergeMode
                        )
                      }
                      aria-label="When accepting overlay merge mode"
                      className="mt-1 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[11px] text-white focus:border-[#F2853F] focus:outline-none"
                    >
                      <option value="append">Append (skip duplicates)</option>
                      <option value="replace_class">
                        {selectedAiClasses.length > 1 ||
                        (refBoxCount > 0 && displayClassChips.length > 1)
                          ? "Replace Matched Classes"
                          : `Replace "${
                              displayClassChips[0]?.label ||
                              selectedAiClasses[0] ||
                              activeClass
                            }"`}
                      </option>
                    </select>
                    {vitSkipAnnotated && (
                      <p className="mt-1 text-[10px] leading-snug text-[#8A8A8E]">
                        Annotated &amp; ref frames are skipped in scan; applies when you accept overlays onto a frame with existing boxes.
                      </p>
                    )}
                  </label>

                  <button
                    type="button"
                    disabled={
                      (!hasQueryOrRefs && !visualBatchProgress?.running) ||
                      vitIndexing ||
                      (visualBusy && !visualBatchProgress?.running) ||
                      (!visualBatchProgress?.running && !hasAnySelectedClass)
                    }
                    onClick={onRunVisualBatch}
                    className={`inline-flex w-full items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40 ${
                      visualBatchProgress?.running
                        ? "bg-rose-500 text-white hover:bg-rose-400"
                        : "bg-[#F2853F] text-black hover:bg-[#FF9652]"
                    }`}
                  >
                    {visualBatchProgress?.running ? (
                      <>
                        <Square className="h-3 w-3 fill-current" />
                        Stop Scan ({visualBatchProgress.done}/
                        {visualBatchProgress.total})
                      </>
                    ) : vitIndexing ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Indexing ViT Refs...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-3.5 w-3.5" />
                        Find &amp; Add Overlay
                      </>
                    )}
                  </button>
                  <p className="text-[10px] leading-snug text-[#8A8A8E]">
                    Preview overlay only — press Enter on each frame to accept &amp; save.
                  </p>

                  {visualBatchProgress && (
                    <div className="space-y-1 rounded-lg border border-[#26262A] bg-[#0B0B0D] p-2">
                      <div className="flex items-center justify-between font-mono-tech text-[10px]">
                        <span className="truncate text-[#D4D4D8]">
                          {visualBatchProgress.currentFilename}
                        </span>
                        <span className="shrink-0 text-emerald-300">
                          Staged +{visualBatchProgress.totalAdded}
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#1E1E22]">
                        <div
                          style={{
                            width: `${Math.min(
                              100,
                              Math.round(
                                (visualBatchProgress.done /
                                  Math.max(1, visualBatchProgress.total)) *
                                  100
                              )
                            )}%`,
                          }}
                          className="h-full bg-[#F2853F] transition-all"
                        />
                      </div>
                      <div className="flex justify-between font-mono-tech text-[9px] text-[#8A8A8E]">
                        <span>
                          {visualBatchProgress.done}/{visualBatchProgress.total}{" "}
                          imgs
                        </span>
                        <span>
                          {visualBatchProgress.itemsWithMatches} w/ overlay
                        </span>
                      </div>
                    </div>
                  )}

                  {totalStagedOverlayItems > 0 &&
                    visualHits.length === 0 &&
                    nextStagedOverlayItemId && (
                      <button
                        type="button"
                        disabled={Boolean(visualBatchProgress?.running) || vitIndexing}
                        onClick={onJumpToNextStagedOverlay}
                        className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-emerald-500/45 bg-emerald-500/15 py-1.5 text-[11px] font-medium text-emerald-200 hover:bg-emerald-500/25 disabled:opacity-40"
                      >
                        <ArrowRight className="h-3.5 w-3.5" />
                        Review Next Staged Overlay ({totalStagedOverlayItems}{" "}
                        imgs)
                      </button>
                    )}

                  {totalStagedOverlayItems > 0 && (
                    <button
                      type="button"
                      disabled={Boolean(visualBatchProgress?.running) || vitIndexing}
                      onClick={onClearAllStagedOverlays}
                      className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-rose-500/35 bg-rose-500/10 py-1 text-[11px] text-rose-200 hover:bg-rose-500/20 disabled:opacity-40"
                    >
                      <X className="h-3 w-3" />
                      Clear All Staged Overlays ({totalStagedOverlayItems} imgs)
                    </button>
                  )}

                  <button
                    type="button"
                    disabled={
                      Boolean(visualBatchProgress?.running) ||
                      vitIndexing ||
                      (!visualQueryId &&
                        visualExemplars.length === 0 &&
                        visualRedoCount === 0 &&
                        totalStagedOverlayItems === 0)
                    }
                    onClick={onResetVisualQuery}
                    className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] py-1 text-[11px] text-[#A1A1AA] hover:text-white disabled:opacity-40"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset Query
                  </button>
                </div>
              )}

              {/* Ranked Visual Detections List */}
              {visualHits.length > 0 && (
                <div className="space-y-1 border-t border-[#26262A] pt-2">
                  <div className="flex items-center justify-between font-mono-tech text-[9px] uppercase text-[#8A8A8E]">
                    <span>Ranked Matches ({visualHits.length})</span>
                    {visualMatchedIou !== null && (
                      <span>Box IoU {visualMatchedIou.toFixed(2)}</span>
                    )}
                  </div>
                  <div className="max-h-28 space-y-1 overflow-y-auto pr-0.5">
                    {visualHits.map((hit, idx) => (
                      <div
                        key={hit.id}
                        className="flex items-center justify-between gap-1 rounded border border-[#26262A] bg-[#0A0A0C] px-2 py-1 font-mono-tech text-[10px]"
                      >
                        <span className="truncate text-[#D4D4D8]">
                          #{idx + 1} {hit.label}
                        </span>
                        <div className="flex shrink-0 items-center gap-1.5">
                          <span className="font-semibold text-emerald-300">
                            {hit.confidence.toFixed(3)}
                          </span>
                          <button
                            type="button"
                            onClick={() => onDismissVisualHit(hit.id)}
                            title="Dismiss match"
                            aria-label={`Dismiss match ${idx + 1}`}
                            className="text-[#8A8A8E] hover:text-rose-400"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-2.5">
              {targetClassOptions.length > 0 && (
                <div className="rounded-lg border border-[#26262A] bg-[#0B0B0D] p-2 space-y-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono-tech text-[9px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                      Target Classes ({promptClassList.length || "All"}/
                      {targetClassOptions.length})
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() =>
                          onFindClassesTextChange(
                            targetClassOptions.map((o) => o.name).join(", ")
                          )
                        }
                        className="rounded border border-[#303035] bg-[#141416] px-1.5 py-0.5 font-mono-tech text-[9px] text-[#D4D4D8] hover:border-[#F2853F] hover:text-white"
                      >
                        All
                      </button>
                      <button
                        type="button"
                        onClick={() => onFindClassesTextChange("")}
                        className="rounded border border-[#303035] bg-[#141416] px-1.5 py-0.5 font-mono-tech text-[9px] text-[#8A8A8E] hover:border-white/30 hover:text-white"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {targetClassOptions.map((opt) => {
                      const active = promptClassSet.has(opt.name.toLowerCase());
                      return (
                        <button
                          key={opt.name}
                          type="button"
                          onClick={() => handleTogglePromptClass(opt.name)}
                          title={
                            active
                              ? `Exclude "${opt.name}" from prompt`
                              : `Include "${opt.name}" in prompt`
                          }
                          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono-tech text-[10px] transition-colors ${
                            active
                              ? "border-[#F2853F] bg-[#F2853F]/20 text-white"
                              : "border-[#303035] bg-[#141416] text-[#8A8A8E] hover:border-white/30 hover:text-white"
                          }`}
                        >
                          <span
                            className="h-1.5 w-1.5 shrink-0 rounded-full"
                            style={{ backgroundColor: opt.color }}
                          />
                          <span className="max-w-[96px] truncate">{opt.name}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              <label className="block">
                <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                  Prompt Classes
                </span>
                <input
                  type="text"
                  value={findClassesText}
                  onChange={(e) => onFindClassesTextChange(e.target.value)}
                  placeholder="person, helmet, crate"
                  className="font-mono-tech mt-1 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                />
                {promptClassList.length === 0 && targetClassOptions.length > 0 && (
                  <span className="mt-1 block font-mono-tech text-[9px] text-[#8A8A8E]">
                    Empty runs all {targetClassOptions.length} class
                    {targetClassOptions.length === 1 ? "" : "es"}. Click pills above to pick specific classes.
                  </span>
                )}
              </label>

              <div className="flex items-center justify-between gap-2">
                <span className="font-mono-tech text-[10px] uppercase text-[#8A8A8E]">
                  Shape
                </span>
                <div className="flex rounded-lg border border-[#303035] bg-[#0A0A0C] p-0.5 text-[10px]">
                  {(["bbox", "polygon", "both"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => onFindOutputModeChange(m)}
                      className={`font-mono-tech rounded px-1.5 py-0.5 uppercase ${
                        findOutputMode === m
                          ? "bg-white/15 text-white"
                          : "text-[#8A8A8E]"
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              <label className="block text-[11px] text-[#A1A1AA]">
                <span className="flex justify-between">
                  <span>Confidence</span>
                  <span className="font-mono-tech text-white">
                    {findConf.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0.05}
                  max={0.85}
                  step={0.05}
                  value={findConf}
                  onChange={(e) => onFindConfChange(Number(e.target.value))}
                  className="mt-1 w-full accent-[#F2853F]"
                />
              </label>

              <button
                type="button"
                disabled={findBusy || disabled}
                onClick={onRunFindObjects}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-[#F2853F] px-3 py-1.5 text-xs font-semibold text-black hover:bg-[#FF9652] disabled:opacity-50"
              >
                {findBusy ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Sparkles className="h-3.5 w-3.5" />
                )}
                {findBusy ? "Detecting..." : "Detect & Add Objects"}
              </button>

              {findMessage && (
                <p className="font-mono-tech text-[10px] leading-snug text-[#D4D4D8]">
                  {findMessage}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
