"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Ban,
  BookmarkCheck,
  BookmarkPlus,
  Check,
  Eye,
  EyeOff,
  GripVertical,
  Hand,
  Loader2,
  Pentagon,
  Redo2,
  RotateCcw,
  Square,
  Sun,
  Tag,
  Trash2,
  Undo2,
  Wand2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getClassColor, hexToRgba } from "./color-utils";
import type {
  AutoDetection,
  AutoItemRecord,
  AutoOutputMode,
  ClassColorEntry,
  ManualTool,
  SmartSelectPoint,
  VisualDrawMode,
  VisualEmbedExemplarBox,
} from "./types";

interface ManualAnnotatorCanvasProps {
  item: AutoItemRecord;
  detections: AutoDetection[];
  classColors: ClassColorEntry[];
  activeClass: string;
  tool: ManualTool;
  onToolChange: (tool: ManualTool) => void;
  selectedDetId: string | null;
  onSelectDet: (detId: string | null) => void;
  hiddenDetIds: Set<string>;
  onDetectionsChange: (next: AutoDetection[], recordHistory?: boolean) => void;
  onPushHistorySnapshot: (snapshot: AutoDetection[]) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onMarkNull: () => void;
  onDeleteSelected: () => void;
  onUpdateDetLabel: (detId: string, newLabel: string) => void;
  onAddClassAndAssign: (detId: string, newClassName: string) => void;
  onCommitDragEdit?: (latestDetections: AutoDetection[]) => void;
  smartOutputMode: AutoOutputMode;
  onSmartOutputModeChange: (mode: AutoOutputMode) => void;
  smartSamRefinerId: string;
  onSmartSamRefinerIdChange: (id: string) => void;
  smartMultiPoint: boolean;
  onSmartMultiPointChange: (multi: boolean) => void;
  smartComplexity: number;
  onSmartComplexityChange: (complexity: number) => void;
  smartPreparing: boolean;
  smartReady: boolean;
  smartBusy: boolean;
  smartStatusMsg: string | null;
  onSmartHoverPreview: (params: {
    point: [number, number];
    signal?: AbortSignal;
  }) => Promise<AutoDetection | null>;
  onCommitSmartDetection: (det: AutoDetection) => void;
  onRunSmartSelect: (params: {
    points: SmartSelectPoint[];
    bboxPrompt?: number[] | null;
    commitImmediately: boolean;
  }) => Promise<AutoDetection | null>;
  onExitSmartSelect: () => void;
  visualHits: AutoDetection[];
  showVisualOverlay: boolean;
  visualExemplars: VisualEmbedExemplarBox[];
  visualBusy: boolean;
  visualDrawMode: VisualDrawMode;
  onExitVisualDrawMode: () => void;
  showVisualExemplars: boolean;
  vitRefDetIds: Set<string>;
  onToggleVitRefDet: (detId: string) => void;
  onVisualEmbedDragBox: (
    bbox: [number, number, number, number],
    polarity: "positive" | "negative"
  ) => void;
  onVisualEmbedClickPoint: (
    point: [number, number],
    polarity: "positive" | "negative"
  ) => void;
  onCommitVisualHits: () => void;
  onClearVisualHits: () => void;
  onDismissVisualHit: (hitId: string) => void;
}

type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

type DragAction =
  | { kind: "pan"; startX: number; startY: number; origPanX: number; origPanY: number }
  | {
      kind: "move_box";
      detId: string;
      startNorm: [number, number];
      origBbox: [number, number, number, number];
      origPolygon?: number[][] | null;
      snapshot: AutoDetection[];
      moved: boolean;
    }
  | {
      kind: "resize_box";
      detId: string;
      handle: ResizeHandle;
      origBbox: [number, number, number, number];
      origPolygon?: number[][] | null;
      snapshot: AutoDetection[];
      moved: boolean;
    }
  | {
      kind: "move_vertex";
      detId: string;
      vertexIdx: number;
      snapshot: AutoDetection[];
      moved: boolean;
    };

function clamp1000(v: number): number {
  return Math.round(Math.min(Math.max(v, 0), 1000));
}

export function bboxFromPolygon(pts: number[][]): [number, number, number, number] {
  let minX = 1000;
  let minY = 1000;
  let maxX = 0;
  let maxY = 0;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return [
    clamp1000(minX),
    clamp1000(minY),
    Math.max(clamp1000(minX) + 1, clamp1000(maxX)),
    Math.max(clamp1000(minY) + 1, clamp1000(maxY)),
  ];
}

function makeManualId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 6)}`;
}

function isPointInPolygon(pt: [number, number], poly: number[][]): boolean {
  const [px, py] = pt;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect =
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi || 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function isPointInDetection(pt: [number, number], det: AutoDetection): boolean {
  if (det.polygon && det.polygon.length >= 3) {
    return isPointInPolygon(pt, det.polygon);
  }
  if (det.bbox_2d.length === 4) {
    const [x1, y1, x2, y2] = det.bbox_2d;
    return pt[0] >= x1 && pt[0] <= x2 && pt[1] >= y1 && pt[1] <= y2;
  }
  return false;
}

export function ManualAnnotatorCanvas({
  item,
  detections,
  classColors,
  activeClass,
  tool,
  onToolChange,
  selectedDetId,
  onSelectDet,
  hiddenDetIds,
  onDetectionsChange,
  onPushHistorySnapshot,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onMarkNull,
  onDeleteSelected,
  onUpdateDetLabel,
  onAddClassAndAssign,
  onCommitDragEdit,
  smartOutputMode,
  onSmartOutputModeChange,
  smartSamRefinerId,
  onSmartSamRefinerIdChange,
  smartMultiPoint,
  onSmartMultiPointChange,
  smartComplexity,
  onSmartComplexityChange,
  smartPreparing,
  smartReady,
  smartBusy,
  smartStatusMsg,
  onSmartHoverPreview,
  onCommitSmartDetection,
  onRunSmartSelect,
  onExitSmartSelect,
  visualHits,
  showVisualOverlay,
  visualExemplars,
  visualBusy,
  visualDrawMode,
  onExitVisualDrawMode,
  showVisualExemplars,
  vitRefDetIds,
  onToggleVitRefDet,
  onVisualEmbedDragBox,
  onVisualEmbedClickPoint,
  onCommitVisualHits,
  onClearVisualHits,
  onDismissVisualHit,
}: ManualAnnotatorCanvasProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);

  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [spaceDown, setSpaceDown] = useState(false);

  const [showOverlay, setShowOverlay] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [displayOpen, setDisplayOpen] = useState(false);

  const [cursorNorm, setCursorNorm] = useState<[number, number] | null>(null);
  const [hoveredDetId, setHoveredDetId] = useState<string | null>(null);

  // BBox draft
  const [draftBoxStart, setDraftBoxStart] = useState<[number, number] | null>(null);
  const [draftBoxEnd, setDraftBoxEnd] = useState<[number, number] | null>(null);

  // Polygon draft
  const [draftPoly, setDraftPoly] = useState<number[][]>([]);

  // Smart Select live hover, multi-point refine & box-drag draft
  const [hoverPreview, setHoverPreview] = useState<AutoDetection | null>(null);
  const hoverPreviewPointRef = useRef<[number, number] | null>(null);
  const hoverInFlightRef = useRef<boolean>(false);
  const hoverPendingPointRef = useRef<[number, number] | null>(null);
  const hoverLastSentAtRef = useRef<number>(0);
  const hoverAbortRef = useRef<AbortController | null>(null);
  const hoverTimerRef = useRef<number | null>(null);

  const [smartPoints, setSmartPoints] = useState<SmartSelectPoint[]>([]);
  const [smartRedoPoints, setSmartRedoPoints] = useState<SmartSelectPoint[]>([]);
  const [smartPreview, setSmartPreview] = useState<AutoDetection | null>(null);
  const [smartDragStart, setSmartDragStart] = useState<{
    clientX: number;
    clientY: number;
    norm: [number, number];
  } | null>(null);
  const [smartDragEndNorm, setSmartDragEndNorm] = useState<[number, number] | null>(
    null
  );

  // Floating draggable Smart Select panel
  const smartCardRef = useRef<HTMLDivElement | null>(null);
  const smartCardDragRef = useRef<{
    startX: number;
    startY: number;
    origLeft: number;
    origTop: number;
  } | null>(null);
  const [smartCardPos, setSmartCardPos] = useState<{ left: number; top: number }>({
    left: 14,
    top: 14,
  });

  // Latest detections ref for clean pointer-move callbacks
  const detectionsRef = useRef(detections);
  detectionsRef.current = detections;
  const dragRef = useRef<DragAction | null>(null);
  const [isDraggingDet, setIsDraggingDet] = useState(false);

  // Floating Annotation Editor pop-up (Drag tool selection)
  const [editorLabel, setEditorLabel] = useState("");
  const editorInputRef = useRef<HTMLInputElement | null>(null);
  const editorCardRef = useRef<HTMLDivElement | null>(null);
  const commitEditorLabelRef = useRef<() => void>(() => {});
  const prevSelectedEditorIdRef = useRef<string | null>(null);
  const editorDraggedRef = useRef<boolean>(false);
  const editorDragRef = useRef<{
    startX: number;
    startY: number;
    origLeft: number;
    origTop: number;
  } | null>(null);
  const [editorPos, setEditorPos] = useState<{ left: number; top: number }>({
    left: 16,
    top: 16,
  });

  const clearHoverPreview = useCallback(() => {
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    hoverAbortRef.current?.abort();
    hoverAbortRef.current = null;
    hoverPendingPointRef.current = null;
    hoverPreviewPointRef.current = null;
    setHoverPreview(null);
  }, []);

  // Reset per-image view drafts cleanly when switching items
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setDraftBoxStart(null);
    setDraftBoxEnd(null);
    setDraftPoly([]);
    setSmartPoints([]);
    setSmartRedoPoints([]);
    setSmartPreview(null);
    setSmartDragStart(null);
    setSmartDragEndNorm(null);
    clearHoverPreview();
    dragRef.current = null;
    editorDraggedRef.current = false;
    editorDragRef.current = null;
    smartCardDragRef.current = null;
  }, [clearHoverPreview, item.id]);

  // Clear in-progress polygon / smart draft if user changes tools
  useEffect(() => {
    if (tool !== "polygon") setDraftPoly([]);
    if (tool !== "bbox") {
      setDraftBoxStart(null);
      setDraftBoxEnd(null);
    }
    if (tool !== "smart_select") {
      setSmartPoints([]);
      setSmartRedoPoints([]);
      setSmartPreview(null);
      setSmartDragStart(null);
      setSmartDragEndNorm(null);
      clearHoverPreview();
    }
  }, [clearHoverPreview, tool]);

  // Clear stale hover mask whenever complexity or model changes
  useEffect(() => {
    hoverPreviewPointRef.current = null;
    setHoverPreview(null);
  }, [smartComplexity, smartOutputMode, smartSamRefinerId]);

  // Live complexity re-decode when adjusting slider during Refine mode
  useEffect(() => {
    if (
      tool !== "smart_select" ||
      !smartMultiPoint ||
      smartPoints.length === 0
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void onRunSmartSelect({
        points: smartPoints,
        commitImmediately: false,
      }).then((det) => {
        if (det) setSmartPreview(det);
      });
    }, 120);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smartComplexity]);

  const dispatchHoverAt = useCallback(
    (pt: [number, number]) => {
      if (
        tool !== "smart_select" ||
        Boolean(visualDrawMode) ||
        smartPreparing ||
        smartBusy ||
        spaceDown ||
        (smartMultiPoint && smartPoints.length > 0)
      ) {
        return;
      }
      if (hoverTimerRef.current !== null) {
        window.clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      if (hoverInFlightRef.current) {
        hoverPendingPointRef.current = pt;
        return;
      }
      const now = Date.now();
      const waitMs = 110 - (now - hoverLastSentAtRef.current);
      if (waitMs > 0) {
        hoverTimerRef.current = window.setTimeout(() => {
          hoverTimerRef.current = null;
          dispatchHoverAt(pt);
        }, waitMs);
        return;
      }
      if (
        hoverPreview !== null &&
        hoverPreviewPointRef.current &&
        Math.hypot(
          pt[0] - hoverPreviewPointRef.current[0],
          pt[1] - hoverPreviewPointRef.current[1]
        ) < 6
      ) {
        return;
      }

      hoverInFlightRef.current = true;
      hoverLastSentAtRef.current = now;
      hoverPendingPointRef.current = null;
      const ctrl = new AbortController();
      hoverAbortRef.current = ctrl;

      void onSmartHoverPreview({ point: pt, signal: ctrl.signal }).then((det) => {
        hoverInFlightRef.current = false;
        if (hoverAbortRef.current === ctrl) {
          hoverAbortRef.current = null;
        }
        if (ctrl.signal.aborted) return;
        if (det) {
          setHoverPreview(det);
          hoverPreviewPointRef.current = pt;
        }
        const pending = hoverPendingPointRef.current;
        hoverPendingPointRef.current = null;
        if (pending) {
          dispatchHoverAt(pending);
        }
      });
    },
    [
      hoverPreview,
      onSmartHoverPreview,
      smartBusy,
      smartMultiPoint,
      smartPoints.length,
      smartPreparing,
      spaceDown,
      tool,
      visualDrawMode,
    ]
  );

  const runRefineWithPoints = useCallback(
    (nextPts: SmartSelectPoint[]) => {
      setSmartPoints(nextPts);
      if (nextPts.length === 0) {
        setSmartPreview(null);
        return;
      }
      void onRunSmartSelect({
        points: nextPts,
        commitImmediately: false,
      }).then((det) => {
        if (det) setSmartPreview(det);
      });
    },
    [onRunSmartSelect]
  );

  const handleSmartPointUndo = useCallback(() => {
    if (smartPoints.length === 0) return;
    const popped = smartPoints[smartPoints.length - 1];
    setSmartRedoPoints((prev) => [...prev, popped]);
    runRefineWithPoints(smartPoints.slice(0, -1));
  }, [runRefineWithPoints, smartPoints]);

  const handleSmartPointRedo = useCallback(() => {
    if (smartRedoPoints.length === 0) return;
    const restored = smartRedoPoints[smartRedoPoints.length - 1];
    setSmartRedoPoints((prev) => prev.slice(0, -1));
    runRefineWithPoints([...smartPoints, restored]);
  }, [runRefineWithPoints, smartPoints, smartRedoPoints]);

  const commitPolygon = useCallback(
    (pts: number[][]) => {
      if (pts.length < 3) return;
      const cleaned = pts.map(([x, y]) => [clamp1000(x), clamp1000(y)]);
      const bbox = bboxFromPolygon(cleaned);
      const newDet: AutoDetection = {
        id: makeManualId("poly"),
        label: activeClass.trim() || "object",
        confidence: 1.0,
        bbox_2d: bbox,
        polygon: cleaned,
      };
      onDetectionsChange([...detectionsRef.current, newDet], true);
      onSelectDet(newDet.id);
      setDraftPoly([]);
    },
    [activeClass, onDetectionsChange, onSelectDet]
  );

  const commitSmartPreview = useCallback(() => {
    if (!smartPreview) return;
    onCommitSmartDetection(smartPreview);
    setSmartPoints([]);
    setSmartRedoPoints([]);
    setSmartPreview(null);
  }, [onCommitSmartDetection, smartPreview]);

  const activeVisualHits = useMemo(
    () => (showVisualOverlay ? visualHits : []),
    [showVisualOverlay, visualHits]
  );

  // Spacebar pan + canvas tool keyboard helpers
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        setSpaceDown(true);
      } else if (e.key === "Escape") {
        if (draftPoly.length > 0) {
          e.preventDefault();
          setDraftPoly([]);
        } else if (activeVisualHits.length > 0) {
          e.preventDefault();
          onClearVisualHits();
        } else if (visualDrawMode) {
          e.preventDefault();
          setSmartDragStart(null);
          setSmartDragEndNorm(null);
          onExitVisualDrawMode();
        } else if (smartPoints.length > 0 || smartPreview || hoverPreview) {
          e.preventDefault();
          setSmartPoints([]);
          setSmartRedoPoints([]);
          setSmartPreview(null);
          clearHoverPreview();
        } else if (draftBoxStart) {
          e.preventDefault();
          setDraftBoxStart(null);
          setDraftBoxEnd(null);
        } else {
          onSelectDet(null);
        }
      } else if (e.key === "Enter") {
        if (tool === "polygon" && draftPoly.length >= 3) {
          e.preventDefault();
          commitPolygon(draftPoly);
        } else if (activeVisualHits.length > 0) {
          e.preventDefault();
          onCommitVisualHits();
        } else if (tool === "smart_select" && smartPreview) {
          e.preventDefault();
          commitSmartPreview();
        } else if (tool === "drag" && selectedDetId) {
          e.preventDefault();
          commitEditorLabelRef.current();
        }
      } else if (e.key === "Backspace" || e.key === "Delete") {
        if (tool === "polygon" && draftPoly.length > 0) {
          e.preventDefault();
          setDraftPoly((prev) => prev.slice(0, -1));
        } else if (
          tool === "smart_select" &&
          smartMultiPoint &&
          smartPoints.length > 0
        ) {
          e.preventDefault();
          handleSmartPointUndo();
        } else if (selectedDetId) {
          e.preventDefault();
          onDeleteSelected();
        }
      } else if (e.key.toLowerCase() === "o") {
        e.preventDefault();
        setShowOverlay((v) => !v);
      } else if (e.key.toLowerCase() === "l") {
        e.preventDefault();
        setShowLabels((v) => !v);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceDown(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [
    activeVisualHits.length,
    clearHoverPreview,
    commitPolygon,
    commitSmartPreview,
    draftBoxStart,
    draftPoly,
    handleSmartPointUndo,
    hoverPreview,
    onClearVisualHits,
    onCommitVisualHits,
    onDeleteSelected,
    onExitVisualDrawMode,
    onSelectDet,
    selectedDetId,
    smartMultiPoint,
    smartPoints.length,
    smartPreview,
    tool,
    visualDrawMode,
  ]);

  const getNormFromClient = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const stage = stageRef.current;
      if (!stage) return null;
      const rect = stage.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) return null;
      const nx = ((clientX - rect.left) / rect.width) * 1000;
      const ny = ((clientY - rect.top) / rect.height) * 1000;
      return [clamp1000(nx), clamp1000(ny)];
    },
    []
  );

  // Smooth, lower-sensitivity native non-passive wheel/trackpad zoom
  useEffect(() => {
    const vp = viewportRef.current;
    if (!vp) return;
    const onNativeWheel = (e: WheelEvent) => {
      if (
        editorCardRef.current?.contains(e.target as Node) ||
        smartCardRef.current?.contains(e.target as Node)
      ) {
        return;
      }
      let deltaPx = e.deltaY;
      if (e.deltaMode === 1) deltaPx *= 16;
      else if (e.deltaMode === 2) deltaPx *= 120;
      if (deltaPx === 0) return;

      e.preventDefault();
      const clampedDelta = Math.max(-50, Math.min(50, deltaPx));
      const sensitivity = e.ctrlKey || e.metaKey ? 0.0025 : 0.0014;
      const factor = Math.exp(-clampedDelta * sensitivity);
      setZoom((z) => Number(Math.min(Math.max(z * factor, 0.35), 6.0).toFixed(4)));
    };
    vp.addEventListener("wheel", onNativeWheel, { passive: false });
    return () => {
      vp.removeEventListener("wheel", onNativeWheel);
    };
  }, []);

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleStagePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Middle click or Space+Left drags canvas pan
    if (e.button === 1 || (e.button === 0 && spaceDown)) {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = {
        kind: "pan",
        startX: e.clientX,
        startY: e.clientY,
        origPanX: pan.x,
        origPanY: pan.y,
      };
      return;
    }

    const norm = getNormFromClient(e.clientX, e.clientY);
    if (!norm) return;

    if (visualDrawMode) {
      if (e.button === 2 || e.button !== 0 || visualBusy) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      setSmartDragStart({
        clientX: e.clientX,
        clientY: e.clientY,
        norm,
      });
      setSmartDragEndNorm(norm);
      return;
    }

    if (tool === "smart_select") {
      if (e.button === 2) {
        // Negative refine click
        e.preventDefault();
        clearHoverPreview();
        if (!smartMultiPoint) {
          onSmartMultiPointChange(true);
        }
        const nextPts: SmartSelectPoint[] = [
          ...smartPoints,
          { x: norm[0], y: norm[1], label: 0 },
        ];
        setSmartRedoPoints([]);
        runRefineWithPoints(nextPts);
        return;
      }
      if (e.button !== 0 || smartBusy || visualBusy) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      setSmartDragStart({
        clientX: e.clientX,
        clientY: e.clientY,
        norm,
      });
      setSmartDragEndNorm(norm);
      return;
    }

    if (e.button !== 0) return;

    if (tool === "drag") {
      // Empty stage pan if clicked directly on stage/img/svg background
      onSelectDet(null);
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = {
        kind: "pan",
        startX: e.clientX,
        startY: e.clientY,
        origPanX: pan.x,
        origPanY: pan.y,
      };
      return;
    }

    if (tool === "bbox") {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraftBoxStart(norm);
      setDraftBoxEnd(norm);
      return;
    }

    if (tool === "polygon") {
      e.preventDefault();
      // Snap close if clicking near first vertex with >= 3 vertices
      if (draftPoly.length >= 3) {
        const first = draftPoly[0];
        const stageRect = stageRef.current?.getBoundingClientRect();
        const distPx = stageRect
          ? Math.hypot(
              ((norm[0] - first[0]) / 1000) * stageRect.width,
              ((norm[1] - first[1]) / 1000) * stageRect.height
            )
          : Math.hypot(norm[0] - first[0], norm[1] - first[1]);
        if (distPx <= 14) {
          commitPolygon(draftPoly);
          return;
        }
      }
      // Avoid accidental double-point at same coordinate (double-click end)
      if (draftPoly.length > 0) {
        const last = draftPoly[draftPoly.length - 1];
        if (Math.hypot(norm[0] - last[0], norm[1] - last[1]) < 2) {
          return;
        }
      }
      setDraftPoly((prev) => [...prev, norm]);
    }
  };

  const handleStagePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const norm = getNormFromClient(e.clientX, e.clientY);
    if (norm) setCursorNorm(norm);

    if (smartDragStart && norm) {
      setSmartDragEndNorm(norm);
      return;
    }

    if (
      !visualDrawMode &&
      tool === "smart_select" &&
      norm &&
      !spaceDown &&
      (!smartMultiPoint || smartPoints.length === 0)
    ) {
      dispatchHoverAt(norm);
    }

    if (tool === "bbox" && draftBoxStart && norm) {
      setDraftBoxEnd(norm);
      return;
    }

    const drag = dragRef.current;
    if (!drag) return;

    if (drag.kind === "pan") {
      setPan({
        x: drag.origPanX + (e.clientX - drag.startX),
        y: drag.origPanY + (e.clientY - drag.startY),
      });
      return;
    }

    if (!norm) return;

    if (drag.kind === "move_box") {
      const dxRaw = norm[0] - drag.startNorm[0];
      const dyRaw = norm[1] - drag.startNorm[1];
      const [ox1, oy1, ox2, oy2] = drag.origBbox;
      const dx = Math.min(Math.max(dxRaw, -ox1), 1000 - ox2);
      const dy = Math.min(Math.max(dyRaw, -oy1), 1000 - oy2);
      if (dx === 0 && dy === 0) return;

      if (!drag.moved) {
        onPushHistorySnapshot(drag.snapshot);
        drag.moved = true;
        setIsDraggingDet(true);
      }

      const nextBbox: [number, number, number, number] = [
        clamp1000(ox1 + dx),
        clamp1000(oy1 + dy),
        clamp1000(ox2 + dx),
        clamp1000(oy2 + dy),
      ];
      const nextPoly = drag.origPolygon
        ? drag.origPolygon.map(([px, py]) => [clamp1000(px + dx), clamp1000(py + dy)])
        : undefined;

      onDetectionsChange(
        detectionsRef.current.map((d) =>
          d.id === drag.detId
            ? { ...d, bbox_2d: nextBbox, ...(nextPoly ? { polygon: nextPoly } : {}) }
            : d
        ),
        false
      );
      return;
    }

    if (drag.kind === "resize_box") {
      let [x1, y1, x2, y2] = drag.origBbox;
      const minSize = 5;
      if (drag.handle.includes("w")) x1 = Math.min(norm[0], x2 - minSize);
      if (drag.handle.includes("e")) x2 = Math.max(norm[0], x1 + minSize);
      if (drag.handle.includes("n")) y1 = Math.min(norm[1], y2 - minSize);
      if (drag.handle.includes("s")) y2 = Math.max(norm[1], y1 + minSize);

      const nextBbox: [number, number, number, number] = [
        clamp1000(x1),
        clamp1000(y1),
        clamp1000(x2),
        clamp1000(y2),
      ];
      const origW = Math.max(1, drag.origBbox[2] - drag.origBbox[0]);
      const origH = Math.max(1, drag.origBbox[3] - drag.origBbox[1]);
      const nextPoly = drag.origPolygon
        ? drag.origPolygon.map(([px, py]) => [
            clamp1000(
              nextBbox[0] +
                ((px - drag.origBbox[0]) / origW) * (nextBbox[2] - nextBbox[0])
            ),
            clamp1000(
              nextBbox[1] +
                ((py - drag.origBbox[1]) / origH) * (nextBbox[3] - nextBbox[1])
            ),
          ])
        : undefined;

      if (!drag.moved) {
        onPushHistorySnapshot(drag.snapshot);
        drag.moved = true;
        setIsDraggingDet(true);
      }

      onDetectionsChange(
        detectionsRef.current.map((d) =>
          d.id === drag.detId
            ? { ...d, bbox_2d: nextBbox, ...(nextPoly ? { polygon: nextPoly } : {}) }
            : d
        ),
        false
      );
      return;
    }

    if (drag.kind === "move_vertex") {
      if (!drag.moved) {
        onPushHistorySnapshot(drag.snapshot);
        drag.moved = true;
        setIsDraggingDet(true);
      }
      onDetectionsChange(
        detectionsRef.current.map((d) => {
          if (d.id !== drag.detId || !d.polygon) return d;
          const nextPoly = d.polygon.map((pt, idx) =>
            idx === drag.vertexIdx ? [norm[0], norm[1]] : pt
          );
          return {
            ...d,
            polygon: nextPoly,
            bbox_2d: bboxFromPolygon(nextPoly),
          };
        }),
        false
      );
    }
  };

  const handleStagePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (smartDragStart) {
      const start = smartDragStart;
      const endNorm = getNormFromClient(e.clientX, e.clientY) ?? start.norm;
      const dragDistPx = Math.hypot(
        e.clientX - start.clientX,
        e.clientY - start.clientY
      );
      setSmartDragStart(null);
      setSmartDragEndNorm(null);

      const cachedHover = hoverPreview;
      const cachedHoverPt = hoverPreviewPointRef.current;

      if (visualDrawMode) {
        clearHoverPreview();
        const effPolarity: "positive" | "negative" = e.altKey
          ? "negative"
          : visualDrawMode;
        if (dragDistPx > 12) {
          const bx1 = Math.min(start.norm[0], endNorm[0]);
          const by1 = Math.min(start.norm[1], endNorm[1]);
          const bx2 = Math.max(start.norm[0], endNorm[0]);
          const by2 = Math.max(start.norm[1], endNorm[1]);
          onVisualEmbedDragBox([bx1, by1, bx2, by2], effPolarity);
        } else {
          onVisualEmbedClickPoint(endNorm, effPolarity);
        }
      } else if (dragDistPx > 12) {
        clearHoverPreview();
        const bx1 = Math.min(start.norm[0], endNorm[0]);
        const by1 = Math.min(start.norm[1], endNorm[1]);
        const bx2 = Math.max(start.norm[0], endNorm[0]);
        const by2 = Math.max(start.norm[1], endNorm[1]);
        void onRunSmartSelect({
          points: [],
          bboxPrompt: [bx1, by1, bx2, by2],
          commitImmediately: !smartMultiPoint,
        }).then((det) => {
          if (det && smartMultiPoint) setSmartPreview(det);
        });
      } else {
        const matchesCached = Boolean(
          cachedHover &&
            cachedHoverPt &&
            !e.shiftKey &&
            Math.hypot(
              endNorm[0] - cachedHoverPt[0],
              endNorm[1] - cachedHoverPt[1]
            ) <= 16
        );

        if (!smartMultiPoint) {
          if (matchesCached && cachedHover) {
            onCommitSmartDetection(cachedHover);
            clearHoverPreview();
          } else {
            clearHoverPreview();
            const clickLabel: 0 | 1 = e.shiftKey ? 0 : 1;
            void onRunSmartSelect({
              points: [{ x: endNorm[0], y: endNorm[1], label: clickLabel }],
              commitImmediately: true,
            });
          }
        } else if (smartPoints.length === 0) {
          const firstLabel: 0 | 1 = e.shiftKey ? 0 : 1;
          const firstPt: SmartSelectPoint = {
            x: endNorm[0],
            y: endNorm[1],
            label: firstLabel,
          };
          setSmartRedoPoints([]);
          if (matchesCached && cachedHover) {
            setSmartPoints([firstPt]);
            setSmartPreview(cachedHover);
            clearHoverPreview();
          } else {
            clearHoverPreview();
            runRefineWithPoints([firstPt]);
          }
        } else {
          // Roboflow Refine UX: clicking inside active mask removes (-), outside adds (+)
          const autoLabel: 0 | 1 = e.shiftKey
            ? 0
            : smartPreview && isPointInDetection(endNorm, smartPreview)
              ? 0
              : 1;
          const nextPt: SmartSelectPoint = {
            x: endNorm[0],
            y: endNorm[1],
            label: autoLabel,
          };
          setSmartRedoPoints([]);
          runRefineWithPoints([...smartPoints, nextPt]);
        }
      }
    }

    if (tool === "bbox" && draftBoxStart) {
      const endNorm = getNormFromClient(e.clientX, e.clientY) ?? draftBoxEnd ?? draftBoxStart;
      const x1 = Math.min(draftBoxStart[0], endNorm[0]);
      const y1 = Math.min(draftBoxStart[1], endNorm[1]);
      const x2 = Math.max(draftBoxStart[0], endNorm[0]);
      const y2 = Math.max(draftBoxStart[1], endNorm[1]);
      setDraftBoxStart(null);
      setDraftBoxEnd(null);
      if (x2 - x1 >= 4 && y2 - y1 >= 4) {
        const newDet: AutoDetection = {
          id: makeManualId("box"),
          label: activeClass.trim() || "object",
          confidence: 1.0,
          bbox_2d: [x1, y1, x2, y2],
          polygon: null,
        };
        onDetectionsChange([...detectionsRef.current, newDet], true);
        onSelectDet(newDet.id);
      }
    }

    const finishedDrag = dragRef.current;
    dragRef.current = null;
    setIsDraggingDet(false);
    if (
      finishedDrag &&
      finishedDrag.kind !== "pan" &&
      finishedDrag.moved
    ) {
      onCommitDragEdit?.(detectionsRef.current);
    }
    if (
      finishedDrag &&
      finishedDrag.kind === "move_box" &&
      !finishedDrag.moved
    ) {
      requestAnimationFrame(() => {
        editorInputRef.current?.focus();
        editorInputRef.current?.select();
      });
    }
  };

  const selectedDet = useMemo(
    () => detections.find((d) => d.id === selectedDetId) ?? null,
    [detections, selectedDetId]
  );

  // Sync floating Annotation Editor input & anchored position on Drag selection
  useEffect(() => {
    if (tool !== "drag" || !selectedDet) {
      prevSelectedEditorIdRef.current = null;
      editorDraggedRef.current = false;
      editorDragRef.current = null;
      return;
    }
    setEditorLabel(selectedDet.label);
    if (prevSelectedEditorIdRef.current !== selectedDet.id) {
      prevSelectedEditorIdRef.current = selectedDet.id;
      requestAnimationFrame(() => {
        editorInputRef.current?.focus();
        editorInputRef.current?.select();
      });
    }
  }, [selectedDet, tool]);

  useEffect(() => {
    if (tool !== "drag" || !selectedDet || isDraggingDet || editorDraggedRef.current) {
      return;
    }
    const vp = viewportRef.current;
    const st = stageRef.current;
    if (!vp || !st || selectedDet.bbox_2d.length !== 4) {
      setEditorPos({ left: 16, top: 16 });
      return;
    }
    const vpRect = vp.getBoundingClientRect();
    const stRect = st.getBoundingClientRect();
    const boxLeft =
      stRect.left - vpRect.left + (selectedDet.bbox_2d[0] / 1000) * stRect.width;
    const boxTop =
      stRect.top - vpRect.top + (selectedDet.bbox_2d[1] / 1000) * stRect.height;
    const boxRight =
      stRect.left - vpRect.left + (selectedDet.bbox_2d[2] / 1000) * stRect.width;

    const panelW = 240;
    const panelH = 235;
    let rawLeft = boxLeft - panelW - 14;
    if (rawLeft < 12) {
      rawLeft = boxRight + 14;
    }
    if (rawLeft + panelW > vpRect.width - 56) {
      rawLeft = 16;
    }
    const clampedLeft = Math.round(
      Math.min(Math.max(rawLeft, 12), Math.max(12, vpRect.width - panelW - 56))
    );
    const clampedTop = Math.round(
      Math.min(Math.max(boxTop, 12), Math.max(12, vpRect.height - panelH - 16))
    );
    setEditorPos({ left: clampedLeft, top: clampedTop });
  }, [isDraggingDet, item.id, pan, selectedDet, tool, zoom]);

  const commitEditorLabel = useCallback(() => {
    if (!selectedDet) return;
    const clean = editorLabel.trim();
    if (clean) {
      const canonical = classColors.find(
        (c) => c.name.trim().toLowerCase() === clean.toLowerCase()
      );
      const targetLabel = canonical ? canonical.name : clean;
      if (targetLabel !== selectedDet.label) {
        if (canonical) {
          onUpdateDetLabel(selectedDet.id, canonical.name);
        } else {
          onAddClassAndAssign(selectedDet.id, clean);
        }
      }
    }
    viewportRef.current?.focus({ preventScroll: true });
    onSelectDet(null);
  }, [
    classColors,
    editorLabel,
    onAddClassAndAssign,
    onSelectDet,
    onUpdateDetLabel,
    selectedDet,
  ]);
  commitEditorLabelRef.current = commitEditorLabel;

  const visibleDetections = useMemo(
    () => detections.filter((d) => !hiddenDetIds.has(d.id)),
    [detections, hiddenDetIds]
  );

  const stageCursor = useMemo(() => {
    if (spaceDown) return "cursor-grab";
    if (visualDrawMode) return visualBusy ? "cursor-wait" : "cursor-crosshair";
    if (tool === "drag") return "cursor-default";
    if (tool === "smart_select") {
      return smartBusy ? "cursor-wait" : "cursor-crosshair";
    }
    return "cursor-crosshair";
  }, [smartBusy, spaceDown, tool, visualBusy, visualDrawMode]);

  const activeColor = getClassColor(activeClass, classColors);

  const isNullFrame = detections.length === 0 && item.reviewStatus === "accepted";

  const invScale = 1 / Math.max(zoom, 0.35);

  const bboxHandles: { id: ResizeHandle; x: number; y: number; cursor: string }[] =
    selectedDet && selectedDet.bbox_2d.length === 4
      ? [
          { id: "nw", x: selectedDet.bbox_2d[0], y: selectedDet.bbox_2d[1], cursor: "nwse-resize" },
          {
            id: "n",
            x: (selectedDet.bbox_2d[0] + selectedDet.bbox_2d[2]) / 2,
            y: selectedDet.bbox_2d[1],
            cursor: "ns-resize",
          },
          { id: "ne", x: selectedDet.bbox_2d[2], y: selectedDet.bbox_2d[1], cursor: "nesw-resize" },
          {
            id: "e",
            x: selectedDet.bbox_2d[2],
            y: (selectedDet.bbox_2d[1] + selectedDet.bbox_2d[3]) / 2,
            cursor: "ew-resize",
          },
          { id: "se", x: selectedDet.bbox_2d[2], y: selectedDet.bbox_2d[3], cursor: "nwse-resize" },
          {
            id: "s",
            x: (selectedDet.bbox_2d[0] + selectedDet.bbox_2d[2]) / 2,
            y: selectedDet.bbox_2d[3],
            cursor: "ns-resize",
          },
          { id: "sw", x: selectedDet.bbox_2d[0], y: selectedDet.bbox_2d[3], cursor: "nesw-resize" },
          {
            id: "w",
            x: selectedDet.bbox_2d[0],
            y: (selectedDet.bbox_2d[1] + selectedDet.bbox_2d[3]) / 2,
            cursor: "ew-resize",
          },
        ]
      : [];

  return (
    <div
      ref={viewportRef}
      tabIndex={-1}
      onPointerLeave={() => {
        setCursorNorm(null);
        clearHoverPreview();
      }}
      onContextMenu={(e) => {
        if (tool === "smart_select") e.preventDefault();
      }}
      className="relative flex min-h-[460px] flex-1 items-center justify-center overflow-hidden bg-[#060608] select-none focus:outline-none"
    >
      {/* Top-Center Preparing / Matching Pill */}
      {(visualBusy || (tool === "smart_select" && smartPreparing)) && (
        <div className="pointer-events-none absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-[#F2853F]/45 bg-[#141416]/95 px-3 py-1 text-[11px] text-[#F2853F] shadow-xl backdrop-blur-md">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span className="font-mono-tech">
            {visualBusy ? "Matching OWLv2 Embeddings..." : "Preparing Smart Select..."}
          </span>
        </div>
      )}

      {/* Top-Center Visual Embed Hit Count Badge */}
      {!visualBusy &&
        !(tool === "smart_select" && smartPreparing) &&
        activeVisualHits.length > 0 && (
          <div className="pointer-events-none absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-2 rounded-lg border border-emerald-500/60 bg-emerald-500/20 px-3 py-1 font-mono-tech text-xs font-semibold text-emerald-300 shadow-xl backdrop-blur-md">
            <span>Count: {activeVisualHits.length}</span>
            <span className="text-[10px] font-normal text-emerald-100/85">
              Enter to accept · Esc to reject
            </span>
          </div>
        )}

      {/* Top-Center Canvas Exemplar Draw Mode Indicator */}
      {!visualBusy && visualDrawMode !== null && (
        <div
          className={`absolute left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full border px-3 py-1 font-mono-tech text-[11px] font-medium shadow-xl backdrop-blur-md ${
            activeVisualHits.length > 0 ? "top-12" : "top-3"
          } ${
            visualDrawMode === "negative"
              ? "border-rose-400/70 bg-rose-500/20 text-rose-200"
              : "border-sky-400/70 bg-sky-500/20 text-sky-200"
          }`}
        >
          <span>
            {visualDrawMode === "negative"
              ? "Draw − Negative box (Esc to exit)"
              : "Draw + Example box · Alt+Drag for − (Esc to exit)"}
          </span>
          <button
            type="button"
            onClick={onExitVisualDrawMode}
            aria-label="Exit ViT example draw mode"
            className="rounded-full p-0.5 hover:bg-black/30"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* Floating Draggable Roboflow-Style Smart Select Control Card (SAM 2.1) */}
      {tool === "smart_select" && (
        <div
          ref={smartCardRef}
          onPointerDown={(e) => e.stopPropagation()}
          onPointerEnter={clearHoverPreview}
          onWheel={(e) => e.stopPropagation()}
          style={{ left: `${smartCardPos.left}px`, top: `${smartCardPos.top}px` }}
          className="absolute z-30 w-[258px] overflow-hidden rounded-2xl border border-[#F2853F]/50 bg-[#141416]/95 shadow-2xl backdrop-blur-md"
          role="dialog"
          aria-label="Smart Select Controls"
        >
          <div
            onPointerDown={(e) => {
              if ((e.target as HTMLElement).closest("button")) return;
              if (e.button !== 0) return;
              e.preventDefault();
              e.stopPropagation();
              e.currentTarget.setPointerCapture(e.pointerId);
              smartCardDragRef.current = {
                startX: e.clientX,
                startY: e.clientY,
                origLeft: smartCardPos.left,
                origTop: smartCardPos.top,
              };
            }}
            onPointerMove={(e) => {
              const dr = smartCardDragRef.current;
              if (!dr) return;
              const vp = viewportRef.current;
              const card = smartCardRef.current;
              const vpW = vp ? vp.clientWidth : 900;
              const vpH = vp ? vp.clientHeight : 600;
              const cardW = card ? card.offsetWidth : 258;
              const cardH = card ? card.offsetHeight : 320;
              const nextLeft = Math.round(
                Math.min(
                  Math.max(dr.origLeft + (e.clientX - dr.startX), 8),
                  Math.max(8, vpW - cardW - 8)
                )
              );
              const nextTop = Math.round(
                Math.min(
                  Math.max(dr.origTop + (e.clientY - dr.startY), 8),
                  Math.max(8, vpH - cardH - 8)
                )
              );
              setSmartCardPos({ left: nextLeft, top: nextTop });
            }}
            onPointerUp={(e) => {
              if (smartCardDragRef.current) {
                try {
                  e.currentTarget.releasePointerCapture(e.pointerId);
                } catch {
                  // ignore
                }
                smartCardDragRef.current = null;
              }
            }}
            onPointerCancel={() => {
              smartCardDragRef.current = null;
            }}
            title="Drag header to move Smart Select panel"
            className="flex cursor-grab select-none items-center justify-between border-b border-[#26262A] bg-[#1A1A1E] px-3 py-2 active:cursor-grabbing"
          >
            <div className="flex items-center gap-1.5">
              <GripVertical className="h-3.5 w-3.5 text-[#8A8A8E]" />
              <Wand2 className="h-3.5 w-3.5 text-[#F2853F]" />
              <span className="text-xs font-semibold text-white">
                Smart Select
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              {smartPreparing || smartBusy ? (
                <span className="font-mono-tech inline-flex items-center gap-1 rounded border border-[#F2853F]/40 bg-[#F2853F]/10 px-1.5 py-0.5 text-[9px] text-[#F2853F]">
                  <Loader2 className="h-2.5 w-2.5 animate-spin" />
                  {smartPreparing ? "Prep" : "Run"}
                </span>
              ) : smartReady ? (
                <span className="font-mono-tech rounded border border-emerald-500/40 bg-emerald-500/12 px-1.5 py-0.5 text-[9px] text-emerald-300">
                  Ready
                </span>
              ) : null}
              <button
                type="button"
                onClick={onExitSmartSelect}
                aria-label="Close Smart Select"
                className="rounded p-0.5 text-[#8A8A8E] hover:text-white"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <div className="space-y-2.5 p-3">
            {/* SAM Model Checkpoint Selector */}
            <select
              value={smartSamRefinerId}
              onChange={(e) => onSmartSamRefinerIdChange(e.target.value)}
              aria-label="SAM model checkpoint"
              className="font-mono-tech w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2 py-1 text-[11px] text-white focus:border-[#F2853F] focus:outline-none"
            >
              <option value="sam2.1-s">SAM 2.1 Small</option>
              <option value="sam2.1-t">SAM 2.1 Tiny</option>
              <option value="mobile-sam">MobileSAM</option>
            </select>

            {/* Output Mode: BBox / Polygon / Both */}
            <div className="grid grid-cols-3 rounded-lg border border-[#303035] bg-[#0A0A0C] p-0.5 text-[11px]">
              {(
                [
                  { id: "bbox", label: "BBox" },
                  { id: "polygon", label: "Polygon" },
                  { id: "both", label: "Both" },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => onSmartOutputModeChange(m.id)}
                  className={`rounded-md py-1 font-medium transition-colors ${
                    smartOutputMode === m.id
                      ? "bg-[#F2853F] text-black"
                      : "text-[#A1A1AA] hover:text-white"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {/* Click Mode: 1-Click Instant vs Refine (+ / -) for SAM */}
            <div className="grid grid-cols-2 rounded-lg border border-[#303035] bg-[#0A0A0C] p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => {
                  onSmartMultiPointChange(false);
                  setSmartPoints([]);
                  setSmartRedoPoints([]);
                  setSmartPreview(null);
                }}
                className={`rounded-md py-1 font-medium transition-colors ${
                  !smartMultiPoint
                    ? "bg-white/15 text-white"
                    : "text-[#8A8A8E] hover:text-white"
                }`}
              >
                1-Click
              </button>
              <button
                type="button"
                onClick={() => {
                  onSmartMultiPointChange(true);
                  clearHoverPreview();
                }}
                className={`rounded-md py-1 font-medium transition-colors ${
                  smartMultiPoint
                    ? "bg-white/15 text-white"
                    : "text-[#8A8A8E] hover:text-white"
                }`}
              >
                Refine (+/−)
              </button>
            </div>

            {/* Contextual Hint */}
            <div className="rounded-xl border border-[#26262A] bg-[#0E0E11] px-2.5 py-2 text-[11px] leading-snug text-[#A1A1AA]">
              {!smartMultiPoint ? (
                <span>
                  Hover to preview boundary, then{" "}
                  <span className="text-white">click once</span> (or drag a box)
                  to add instantaneously.
                </span>
              ) : smartPoints.length === 0 ? (
                <span>
                  Hover &amp; click an object to start, then refine before
                  finishing.
                </span>
              ) : (
                <span>
                  Click <span className="text-emerald-300">outside</span> to add
                  to area, or <span className="text-rose-300">inside</span> to
                  remove from it.
                </span>
              )}
            </div>

            {/* Polygon Complexity Slider (Simple <-> Complex) */}
            {smartOutputMode !== "bbox" && (
              <div className="space-y-1">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={smartComplexity}
                  aria-label="Polygon Complexity"
                  onChange={(e) => onSmartComplexityChange(Number(e.target.value))}
                  className="w-full accent-[#F2853F]"
                />
                <div className="flex justify-between font-mono-tech text-[10px] text-[#8A8A8E]">
                  <span>Simple</span>
                  <span>Complex</span>
                </div>
              </div>
            )}

            {/* Refine Action Controls */}
            {smartMultiPoint && (smartPoints.length > 0 || smartPreview) && (
              <div className="space-y-2 border-t border-[#26262A] pt-2">
                <div className="grid grid-cols-3 gap-1.5">
                  <button
                    type="button"
                    disabled={smartPoints.length === 0 || smartBusy}
                    onClick={handleSmartPointUndo}
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] py-1 text-[11px] text-[#D4D4D8] hover:border-white/30 disabled:opacity-35"
                  >
                    <Undo2 className="h-3 w-3" />
                    Undo
                  </button>
                  <button
                    type="button"
                    disabled={smartRedoPoints.length === 0 || smartBusy}
                    onClick={handleSmartPointRedo}
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] py-1 text-[11px] text-[#D4D4D8] hover:border-white/30 disabled:opacity-35"
                  >
                    <Redo2 className="h-3 w-3" />
                    Redo
                  </button>
                  <button
                    type="button"
                    disabled={smartBusy}
                    onClick={() => {
                      setSmartPoints([]);
                      setSmartRedoPoints([]);
                      setSmartPreview(null);
                    }}
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#303035] bg-[#0A0A0C] py-1 text-[11px] text-[#A1A1AA] hover:text-white disabled:opacity-35"
                  >
                    <X className="h-3 w-3" />
                    Clear
                  </button>
                </div>

                {smartPreview && (
                  <button
                    type="button"
                    onClick={commitSmartPreview}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-[#F2853F] py-1.5 text-xs font-semibold text-black transition-colors hover:bg-[#FF9652]"
                  >
                    <Check className="h-3.5 w-3.5" />
                    Finish (Enter)
                  </button>
                )}
              </div>
            )}

            {smartStatusMsg && (
              <p className="font-mono-tech truncate text-[10px] text-[#8A8A8E]">
                {smartStatusMsg}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Polygon Draft Action Chip */}
      {tool === "polygon" && draftPoly.length > 0 && (
        <div className="absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full border border-[#2C2C32] bg-[#141416]/95 px-3.5 py-1.5 text-xs shadow-xl backdrop-blur-md">
          <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
            Polygon: {draftPoly.length} pt(s)
          </span>
          {draftPoly.length >= 3 && (
            <button
              type="button"
              onClick={() => commitPolygon(draftPoly)}
              className="rounded-full bg-[#F2853F] px-2.5 py-0.5 text-[11px] font-semibold text-black hover:bg-[#FF9652]"
            >
              Finish (Enter)
            </button>
          )}
          <button
            type="button"
            onClick={() => setDraftPoly([])}
            className="rounded-full border border-[#303035] px-2.5 py-0.5 text-[11px] text-[#A1A1AA] hover:text-white"
          >
            Cancel (Esc)
          </button>
        </div>
      )}

      {/* Interactive Image + Vector Stage */}
      <div
        ref={stageRef}
        onPointerDown={handleStagePointerDown}
        onPointerMove={handleStagePointerMove}
        onPointerUp={handleStagePointerUp}
        onDoubleClick={() => {
          if (tool === "polygon" && draftPoly.length >= 3) {
            commitPolygon(draftPoly);
          }
        }}
        style={{
          aspectRatio: `${Math.max(1, item.width)} / ${Math.max(1, item.height)}`,
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: "center center",
        }}
        className={`relative max-h-[calc(100%-3.75rem)] max-w-[calc(100%-5.5rem)] border border-[#26262A] bg-black shadow-2xl ${stageCursor}`}
      >
        <img
          src={item.imageUrl}
          alt={item.filename}
          draggable={false}
          style={{
            filter: `brightness(${brightness}%) contrast(${contrast}%)`,
          }}
          className="pointer-events-none block h-full w-full select-none object-contain"
        />

        {/* SVG Shape Layer */}
        <svg
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          className="absolute inset-0 h-full w-full"
        >
          {showOverlay &&
            visibleDetections.map((det) => {
              const color = getClassColor(det.label, classColors);
              const isSelected = selectedDetId === det.id;
              const isHovered = hoveredDetId === det.id;
              const hasPoly = Boolean(det.polygon && det.polygon.length >= 3);
              const [x1, y1, x2, y2] = det.bbox_2d;

              return (
                <g
                  key={det.id}
                  onPointerEnter={() => setHoveredDetId(det.id)}
                  onPointerLeave={() => setHoveredDetId(null)}
                  onPointerDown={(e) => {
                    if (tool !== "drag" || e.button !== 0 || spaceDown) return;
                    e.stopPropagation();
                    e.preventDefault();
                    onSelectDet(det.id);
                    const norm = getNormFromClient(e.clientX, e.clientY);
                    if (!norm) return;
                    e.currentTarget.setPointerCapture?.(e.pointerId);
                    dragRef.current = {
                      kind: "move_box",
                      detId: det.id,
                      startNorm: norm,
                      origBbox: [x1, y1, x2, y2],
                      origPolygon: det.polygon ? det.polygon.map((p) => [...p]) : null,
                      snapshot: detectionsRef.current,
                      moved: false,
                    };
                  }}
                  style={{ cursor: tool === "drag" ? "move" : undefined }}
                >
                  {hasPoly && det.polygon && (
                    <polygon
                      points={det.polygon.map((p) => `${p[0]},${p[1]}`).join(" ")}
                      fill={hexToRgba(color, isSelected ? 0.32 : isHovered ? 0.28 : 0.2)}
                      stroke={isSelected ? "#FFFFFF" : color}
                      strokeWidth={isSelected ? 3 : 2.2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  {det.bbox_2d.length === 4 && (
                    <rect
                      x={x1}
                      y={y1}
                      width={Math.max(1, x2 - x1)}
                      height={Math.max(1, y2 - y1)}
                      fill={
                        hasPoly
                          ? "transparent"
                          : hexToRgba(color, isSelected ? 0.22 : isHovered ? 0.18 : 0.11)
                      }
                      stroke={color}
                      strokeDasharray={hasPoly ? "4 3" : undefined}
                      strokeWidth={hasPoly ? 1.3 : isSelected ? 3.2 : 2.4}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </g>
              );
            })}

          {/* Draft Bounding Box */}
          {tool === "bbox" && draftBoxStart && draftBoxEnd && (
            <rect
              x={Math.min(draftBoxStart[0], draftBoxEnd[0])}
              y={Math.min(draftBoxStart[1], draftBoxEnd[1])}
              width={Math.max(1, Math.abs(draftBoxEnd[0] - draftBoxStart[0]))}
              height={Math.max(1, Math.abs(draftBoxEnd[1] - draftBoxStart[1]))}
              fill={hexToRgba(activeColor, 0.2)}
              stroke={activeColor}
              strokeWidth={2.5}
              strokeDasharray="6 3"
              vectorEffect="non-scaling-stroke"
              className="pointer-events-none"
            />
          )}

          {/* Draft Polygon */}
          {tool === "polygon" && draftPoly.length > 0 && (
            <g className="pointer-events-none">
              <polyline
                points={[
                  ...draftPoly,
                  ...(cursorNorm ? [cursorNorm] : []),
                ]
                  .map((p) => `${p[0]},${p[1]}`)
                  .join(" ")}
                fill={
                  draftPoly.length >= 3 ? hexToRgba(activeColor, 0.16) : "none"
                }
                stroke={activeColor}
                strokeWidth={2.4}
                strokeDasharray="5 3"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}

          {/* Smart Select Live Hover Preview */}
          {tool === "smart_select" &&
            hoverPreview &&
            !smartPreview &&
            smartPoints.length === 0 &&
            !smartBusy && (
              <g className="pointer-events-none">
                {hoverPreview.polygon && hoverPreview.polygon.length >= 3 && (
                  <polygon
                    points={hoverPreview.polygon
                      .map((p) => `${p[0]},${p[1]}`)
                      .join(" ")}
                    fill={hexToRgba(
                      "#F2853F",
                      smartOutputMode === "bbox" ? 0.16 : 0.3
                    )}
                    stroke="#F2853F"
                    strokeWidth={smartOutputMode === "bbox" ? 1.4 : 2.6}
                    strokeDasharray={smartOutputMode === "bbox" ? "4 3" : undefined}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {hoverPreview.bbox_2d.length === 4 &&
                  (smartOutputMode === "bbox" || smartOutputMode === "both") && (
                    <rect
                      x={hoverPreview.bbox_2d[0]}
                      y={hoverPreview.bbox_2d[1]}
                      width={Math.max(
                        1,
                        hoverPreview.bbox_2d[2] - hoverPreview.bbox_2d[0]
                      )}
                      height={Math.max(
                        1,
                        hoverPreview.bbox_2d[3] - hoverPreview.bbox_2d[1]
                      )}
                      fill={
                        smartOutputMode === "bbox" && !hoverPreview.polygon
                          ? hexToRgba("#F2853F", 0.16)
                          : "transparent"
                      }
                      stroke={smartOutputMode === "bbox" ? "#F2853F" : "#4ADE80"}
                      strokeWidth={smartOutputMode === "bbox" ? 2.6 : 1.8}
                      strokeDasharray={
                        smartOutputMode === "both" ? "5 3" : undefined
                      }
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
              </g>
            )}

          {/* Smart Select Multi-Point Live Refine Preview */}
          {tool === "smart_select" && smartPreview && (
            <g className="pointer-events-none">
              {smartPreview.polygon && smartPreview.polygon.length >= 3 && (
                <polygon
                  points={smartPreview.polygon.map((p) => `${p[0]},${p[1]}`).join(" ")}
                  fill={hexToRgba(
                    "#38BDF8",
                    smartOutputMode === "bbox" ? 0.18 : 0.28
                  )}
                  stroke="#38BDF8"
                  strokeWidth={2.6}
                  vectorEffect="non-scaling-stroke"
                />
              )}
              {smartPreview.bbox_2d.length === 4 &&
                (smartOutputMode === "bbox" || smartOutputMode === "both") && (
                  <rect
                    x={smartPreview.bbox_2d[0]}
                    y={smartPreview.bbox_2d[1]}
                    width={Math.max(
                      1,
                      smartPreview.bbox_2d[2] - smartPreview.bbox_2d[0]
                    )}
                    height={Math.max(
                      1,
                      smartPreview.bbox_2d[3] - smartPreview.bbox_2d[1]
                    )}
                    fill="transparent"
                    stroke={smartOutputMode === "bbox" ? "#F2853F" : "#4ADE80"}
                    strokeWidth={smartOutputMode === "bbox" ? 2.4 : 1.8}
                    strokeDasharray="4 3"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
            </g>
          )}

          {/* Visual Embed Exemplars (Example + / Negative -) & Live Matches */}
          {(showVisualExemplars || visualDrawMode !== null) &&
            visualExemplars
              .filter((ex) => ex.itemId === item.id && ex.bbox_2d.length === 4)
              .map((ex) => {
                const isNeg = ex.polarity === "negative";
                const col = isNeg ? "#FB7185" : "#38BDF8";
                return (
                  <rect
                    key={ex.id}
                    x={ex.bbox_2d[0]}
                    y={ex.bbox_2d[1]}
                    width={Math.max(1, ex.bbox_2d[2] - ex.bbox_2d[0])}
                    height={Math.max(1, ex.bbox_2d[3] - ex.bbox_2d[1])}
                    fill={hexToRgba(col, 0.14)}
                    stroke={col}
                    strokeWidth={2.4}
                    strokeDasharray="6 3"
                    vectorEffect="non-scaling-stroke"
                    className="pointer-events-none"
                  />
                );
              })}

          {activeVisualHits.map((det) => {
            const hasPoly = Boolean(det.polygon && det.polygon.length >= 3);
            const [x1, y1, x2, y2] = det.bbox_2d;
            return (
              <g key={`vh-${det.id}`} className="pointer-events-none">
                {hasPoly && det.polygon && (
                  <polygon
                    points={det.polygon.map((p) => `${p[0]},${p[1]}`).join(" ")}
                    fill={hexToRgba("#4ADE80", 0.3)}
                    stroke="#4ADE80"
                    strokeWidth={2.4}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {det.bbox_2d.length === 4 && (
                  <rect
                    x={x1}
                    y={y1}
                    width={Math.max(1, x2 - x1)}
                    height={Math.max(1, y2 - y1)}
                    fill={hasPoly ? "transparent" : hexToRgba("#4ADE80", 0.24)}
                    stroke="#4ADE80"
                    strokeDasharray={hasPoly ? "4 3" : undefined}
                    strokeWidth={hasPoly ? 1.5 : 2.5}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </g>
            );
          })}

          {/* Smart Select / Visual Embed Box-Drag Prompt */}
          {(visualDrawMode !== null || tool === "smart_select") &&
            smartDragStart &&
            smartDragEndNorm &&
            (() => {
              const dragCol =
                visualDrawMode === "negative"
                  ? "#FB7185"
                  : visualDrawMode === "positive"
                    ? "#38BDF8"
                    : "#F2853F";
              return (
                <rect
                  x={Math.min(smartDragStart.norm[0], smartDragEndNorm[0])}
                  y={Math.min(smartDragStart.norm[1], smartDragEndNorm[1])}
                  width={Math.max(
                    1,
                    Math.abs(smartDragEndNorm[0] - smartDragStart.norm[0])
                  )}
                  height={Math.max(
                    1,
                    Math.abs(smartDragEndNorm[1] - smartDragStart.norm[1])
                  )}
                  fill={hexToRgba(dragCol, 0.16)}
                  stroke={dragCol}
                  strokeWidth={2.2}
                  strokeDasharray="4 3"
                  vectorEffect="non-scaling-stroke"
                  className="pointer-events-none"
                />
              );
            })()}

          {/* Crosshair Guides in Draw, ViT Exemplar Draw & Smart Select Hover */}
          {(visualDrawMode !== null ||
            tool === "bbox" ||
            tool === "polygon" ||
            (tool === "smart_select" && smartPoints.length === 0)) &&
            cursorNorm &&
            !spaceDown && (
            <g className="pointer-events-none">
              <line
                x1={cursorNorm[0]}
                y1={0}
                x2={cursorNorm[0]}
                y2={1000}
                stroke="rgba(242,133,63,0.48)"
                strokeWidth={1}
                strokeDasharray="4 4"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={0}
                y1={cursorNorm[1]}
                x2={1000}
                y2={cursorNorm[1]}
                stroke="rgba(242,133,63,0.48)"
                strokeWidth={1}
                strokeDasharray="4 4"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}
        </svg>

        {/* Class Labels Overlay (constant screen scale regardless of zoom) */}
        {showOverlay &&
          showLabels &&
          visibleDetections.map((det) => {
            if (det.bbox_2d.length !== 4) return null;
            const color = getClassColor(det.label, classColors);
            const isVitRef = vitRefDetIds.has(det.id);
            return (
              <div
                key={`lbl-${det.id}`}
                style={{
                  left: `${det.bbox_2d[0] / 10}%`,
                  top: `${det.bbox_2d[1] / 10}%`,
                  transform: `translateY(-100%) scale(${invScale})`,
                  transformOrigin: "bottom left",
                  backgroundColor: color,
                }}
                className="pointer-events-none absolute z-10 flex items-center gap-1 rounded-t px-1.5 py-0.5 font-mono-tech text-[10px] font-semibold leading-none text-black shadow"
              >
                <span>{det.label}</span>
                {isVitRef && (
                  <span className="rounded bg-black/25 px-1 py-0.2 text-[8px] font-bold">
                    ViT
                  </span>
                )}
              </div>
            );
          })}

        {/* Visual Embed Match Score Pills */}
        {activeVisualHits.map((det) => {
          if (det.bbox_2d.length !== 4) return null;
          return (
            <div
              key={`vhlbl-${det.id}`}
              style={{
                left: `${det.bbox_2d[0] / 10}%`,
                top: `${det.bbox_2d[1] / 10}%`,
                transform: `translateY(-100%) scale(${invScale})`,
                transformOrigin: "bottom left",
              }}
              className="pointer-events-auto absolute z-15 flex items-center gap-1 rounded-t bg-emerald-400 px-1.5 py-0.5 font-mono-tech text-[10px] font-bold leading-none text-black shadow"
            >
              <span>
                {det.label} · {det.confidence.toFixed(2)}
              </span>
              {onDismissVisualHit && (
                <button
                  type="button"
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDismissVisualHit(det.id);
                  }}
                  title="Remove this match overlay"
                  aria-label="Remove match overlay"
                  className="rounded p-0.5 text-black/75 transition hover:bg-black/20 hover:text-black"
                >
                  <X className="h-2.5 w-2.5" />
                </button>
              )}
            </div>
          );
        })}

        {/* Selected BBox Resize Handles */}
        {tool === "drag" &&
          selectedDet &&
          !hiddenDetIds.has(selectedDet.id) &&
          bboxHandles.map((h) => (
            <div
              key={h.id}
              onPointerDown={(e) => {
                if (e.button !== 0 || spaceDown) return;
                e.stopPropagation();
                e.currentTarget.setPointerCapture(e.pointerId);
                dragRef.current = {
                  kind: "resize_box",
                  detId: selectedDet.id,
                  handle: h.id,
                  origBbox: [...selectedDet.bbox_2d] as [
                    number,
                    number,
                    number,
                    number,
                  ],
                  origPolygon: selectedDet.polygon
                    ? selectedDet.polygon.map((p) => [...p])
                    : null,
                  snapshot: detectionsRef.current,
                  moved: false,
                };
              }}
              style={{
                left: `${h.x / 10}%`,
                top: `${h.y / 10}%`,
                cursor: h.cursor,
                transform: `translate(-50%, -50%) scale(${invScale})`,
              }}
              className="absolute z-20 h-2.5 w-2.5 rounded-xs border border-black bg-white shadow"
            />
          ))}

        {/* Selected Polygon Vertex Handles & Midpoint Insert Dots */}
        {tool === "drag" &&
          selectedDet &&
          !hiddenDetIds.has(selectedDet.id) &&
          selectedDet.polygon &&
          selectedDet.polygon.length >= 3 && (
            <>
              {selectedDet.polygon.map((pt, idx) => (
                <div
                  key={`vtx-${idx}`}
                  title="Drag vertex · Double-click to remove vertex"
                  onPointerDown={(e) => {
                    if (e.button !== 0 || spaceDown) return;
                    e.stopPropagation();
                    e.currentTarget.setPointerCapture(e.pointerId);
                    dragRef.current = {
                      kind: "move_vertex",
                      detId: selectedDet.id,
                      vertexIdx: idx,
                      snapshot: detectionsRef.current,
                      moved: false,
                    };
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    if (!selectedDet.polygon || selectedDet.polygon.length <= 3) {
                      return;
                    }
                    const nextPoly = selectedDet.polygon.filter((_, i) => i !== idx);
                    onDetectionsChange(
                      detectionsRef.current.map((d) =>
                        d.id === selectedDet.id
                          ? {
                              ...d,
                              polygon: nextPoly,
                              bbox_2d: bboxFromPolygon(nextPoly),
                            }
                          : d
                      ),
                      true
                    );
                  }}
                  style={{
                    left: `${pt[0] / 10}%`,
                    top: `${pt[1] / 10}%`,
                    transform: `translate(-50%, -50%) scale(${invScale})`,
                  }}
                  className="absolute z-20 h-3 w-3 cursor-pointer rounded-full border-2 border-[#0A0A0C] bg-[#F2853F] shadow"
                />
              ))}
              {selectedDet.polygon.map((pt, idx) => {
                const nextPt =
                  selectedDet.polygon![(idx + 1) % selectedDet.polygon!.length];
                const mx = Math.round((pt[0] + nextPt[0]) / 2);
                const my = Math.round((pt[1] + nextPt[1]) / 2);
                return (
                  <div
                    key={`mid-${idx}`}
                    title="Click to insert vertex"
                    onPointerDown={(e) => {
                      if (e.button !== 0 || spaceDown || !selectedDet.polygon) return;
                      e.stopPropagation();
                      const insertAt = idx + 1;
                      const nextPoly = [
                        ...selectedDet.polygon.slice(0, insertAt),
                        [mx, my],
                        ...selectedDet.polygon.slice(insertAt),
                      ];
                      onDetectionsChange(
                        detectionsRef.current.map((d) =>
                          d.id === selectedDet.id
                            ? {
                                ...d,
                                polygon: nextPoly,
                                bbox_2d: bboxFromPolygon(nextPoly),
                              }
                            : d
                        ),
                        true
                      );
                    }}
                    style={{
                      left: `${mx / 10}%`,
                      top: `${my / 10}%`,
                      transform: `translate(-50%, -50%) scale(${invScale})`,
                    }}
                    className="absolute z-10 h-2 w-2 cursor-copy rounded-full border border-white/80 bg-[#141416] opacity-75 hover:opacity-100"
                  />
                );
              })}
            </>
          )}

        {/* Draft Polygon Vertices (click vertex 0 to close cleanly) */}
        {tool === "polygon" &&
          draftPoly.map((pt, idx) => (
            <div
              key={`draft-vtx-${idx}`}
              style={{
                left: `${pt[0] / 10}%`,
                top: `${pt[1] / 10}%`,
                transform: `translate(-50%, -50%) scale(${invScale})`,
                borderColor: activeColor,
              }}
              className={`pointer-events-none absolute z-20 rounded-full border-2 bg-white ${
                idx === 0 && draftPoly.length >= 3 ? "h-3.5 w-3.5 ring-2 ring-[#F2853F]" : "h-2.5 w-2.5"
              }`}
            />
          ))}

        {/* Smart Select Click Markers & Refine Cursor Action Badge */}
        {tool === "smart_select" &&
          smartPoints.map((pt, idx) => (
            <div
              key={`sp-${idx}`}
              style={{
                left: `${pt.x / 10}%`,
                top: `${pt.y / 10}%`,
                transform: `translate(-50%, -50%) scale(${invScale})`,
              }}
              className={`pointer-events-none absolute z-20 flex h-4 w-4 items-center justify-center rounded-full border border-white text-[10px] font-bold text-black shadow ${
                pt.label === 1 ? "bg-emerald-400" : "bg-rose-400"
              }`}
            >
              {pt.label === 1 ? "+" : "−"}
            </div>
          ))}

        {tool === "smart_select" &&
          smartMultiPoint &&
          smartPreview &&
          cursorNorm &&
          !spaceDown &&
          !smartBusy && (() => {
            const inside = isPointInDetection(cursorNorm, smartPreview);
            return (
              <div
                style={{
                  left: `${cursorNorm[0] / 10}%`,
                  top: `${cursorNorm[1] / 10}%`,
                  transform: `translate(10px, 10px) scale(${invScale})`,
                }}
                className={`pointer-events-none absolute z-20 flex h-4 w-4 items-center justify-center rounded-full border border-white text-[10px] font-bold text-black shadow ${
                  inside ? "bg-rose-400" : "bg-emerald-400"
                }`}
              >
                {inside ? "−" : "+"}
              </div>
            );
          })()}
      </div>

      {/* Floating Draggable Annotation Editor Pop-Up on Drag Selection (matching Roboflow UX) */}
      {tool === "drag" && selectedDet && !isDraggingDet && (
        <div
          ref={editorCardRef}
          onPointerDown={(e) => e.stopPropagation()}
          onWheel={(e) => e.stopPropagation()}
          style={{ left: `${editorPos.left}px`, top: `${editorPos.top}px` }}
          className="absolute z-30 w-[238px] overflow-hidden rounded-2xl border border-[#F2853F]/55 bg-[#141416]/95 shadow-2xl backdrop-blur-md"
          role="dialog"
          aria-label="Annotation Editor"
        >
          <div
            onPointerDown={(e) => {
              if ((e.target as HTMLElement).closest("button")) return;
              if (e.button !== 0) return;
              e.preventDefault();
              e.stopPropagation();
              e.currentTarget.setPointerCapture(e.pointerId);
              editorDragRef.current = {
                startX: e.clientX,
                startY: e.clientY,
                origLeft: editorPos.left,
                origTop: editorPos.top,
              };
            }}
            onPointerMove={(e) => {
              const dr = editorDragRef.current;
              if (!dr) return;
              const vp = viewportRef.current;
              const card = editorCardRef.current;
              const vpW = vp ? vp.clientWidth : 900;
              const vpH = vp ? vp.clientHeight : 600;
              const cardW = card ? card.offsetWidth : 238;
              const cardH = card ? card.offsetHeight : 235;
              const nextLeft = Math.round(
                Math.min(
                  Math.max(dr.origLeft + (e.clientX - dr.startX), 8),
                  Math.max(8, vpW - cardW - 8)
                )
              );
              const nextTop = Math.round(
                Math.min(
                  Math.max(dr.origTop + (e.clientY - dr.startY), 8),
                  Math.max(8, vpH - cardH - 8)
                )
              );
              editorDraggedRef.current = true;
              setEditorPos({ left: nextLeft, top: nextTop });
            }}
            onPointerUp={(e) => {
              if (editorDragRef.current) {
                try {
                  e.currentTarget.releasePointerCapture(e.pointerId);
                } catch {
                  // ignore
                }
                editorDragRef.current = null;
              }
            }}
            onPointerCancel={() => {
              editorDragRef.current = null;
            }}
            title="Drag header to move Annotation Editor"
            className="flex cursor-grab select-none items-center justify-between border-b border-[#26262A] bg-[#1A1A1E] px-3 py-2 active:cursor-grabbing"
          >
            <div className="flex items-center gap-1.5">
              <GripVertical className="h-3.5 w-3.5 text-[#8A8A8E]" />
              <span className="text-xs font-semibold text-[#F2853F]">
                Annotation Editor
              </span>
              <span className="font-mono-tech rounded border border-[#303035] bg-[#0A0A0C] px-1.5 py-0.5 text-[9px] text-[#A1A1AA]">
                {selectedDet.polygon && selectedDet.polygon.length >= 3
                  ? `${selectedDet.polygon.length}v`
                  : "box"}
              </span>
            </div>
            <button
              type="button"
              onClick={() => onSelectDet(null)}
              aria-label="Close Annotation Editor"
              className="rounded p-0.5 text-[#8A8A8E] hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="p-3">
            <input
              ref={editorInputRef}
              type="text"
              value={editorLabel}
              onChange={(e) => setEditorLabel(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitEditorLabel();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  viewportRef.current?.focus({ preventScroll: true });
                  onSelectDet(null);
                } else if (
                  (e.key === "Backspace" || e.key === "Delete") &&
                  !e.repeat &&
                  e.currentTarget.value.length === 0
                ) {
                  e.preventDefault();
                  viewportRef.current?.focus({ preventScroll: true });
                  onDeleteSelected();
                }
              }}
              placeholder="Class name..."
              aria-label="Edit annotation class"
              className="font-mono-tech w-full rounded-xl border border-[#F2853F]/70 bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white focus:border-[#F2853F] focus:outline-none"
            />

            <div className="mt-2.5 flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => {
                  viewportRef.current?.focus({ preventScroll: true });
                  onDeleteSelected();
                }}
                title="Delete annotation (or press Delete twice from class input)"
                className="rounded-lg border border-rose-500/45 bg-rose-500/10 px-2.5 py-1 text-[11px] font-medium text-rose-300 transition-colors hover:bg-rose-500/20"
              >
                Delete
              </button>
              <button
                type="button"
                onClick={commitEditorLabel}
                className="rounded-lg bg-[#F2853F] px-3 py-1 text-[11px] font-semibold text-black transition-colors hover:bg-[#FF9652]"
              >
                Save (Enter)
              </button>
            </div>

            <button
              type="button"
              onClick={() => onToggleVitRefDet(selectedDet.id)}
              className={`mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                vitRefDetIds.has(selectedDet.id)
                  ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-200"
                  : "border-[#303035] bg-[#0A0A0C] text-[#D4D4D8] hover:border-[#F2853F] hover:text-white"
              }`}
            >
              {vitRefDetIds.has(selectedDet.id) ? (
                <>
                  <BookmarkCheck className="h-3.5 w-3.5" />
                  Pinned as ViT Ref
                </>
              ) : (
                <>
                  <BookmarkPlus className="h-3.5 w-3.5 text-[#F2853F]" />
                  Set as ViT Ref
                </>
              )}
            </button>
          </div>

          <div className="max-h-36 overflow-y-auto border-t border-[#26262A] bg-[#101013]">
            {editorLabel.trim() &&
              !classColors.some(
                (c) =>
                  c.name.trim().toLowerCase() ===
                  editorLabel.trim().toLowerCase()
              ) && (
                <button
                  type="button"
                  onClick={() => {
                    onAddClassAndAssign(selectedDet.id, editorLabel.trim());
                    viewportRef.current?.focus({ preventScroll: true });
                    onSelectDet(null);
                  }}
                  className="flex w-full items-center justify-between border-b border-[#222226] px-3 py-1.5 text-left text-[11px] text-[#F2853F] hover:bg-[#F2853F]/12"
                >
                  <span className="truncate font-medium">
                    + Add &amp; assign &ldquo;{editorLabel.trim()}&rdquo;
                  </span>
                </button>
              )}

            {classColors.map((cc, idx) => {
              const isCurrent =
                cc.name.trim().toLowerCase() ===
                selectedDet.label.trim().toLowerCase();
              return (
                <button
                  key={cc.name}
                  type="button"
                  onClick={() => {
                    setEditorLabel(cc.name);
                    if (cc.name !== selectedDet.label) {
                      onUpdateDetLabel(selectedDet.id, cc.name);
                    }
                    viewportRef.current?.focus({ preventScroll: true });
                    onSelectDet(null);
                  }}
                  className={`flex w-full items-center justify-between px-3 py-1.5 text-left text-xs transition-colors ${
                    isCurrent
                      ? "bg-[#F2853F] font-semibold text-black"
                      : "text-[#D4D4D8] hover:bg-[#1E1E24]"
                  }`}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className={`font-mono-tech inline-flex h-4 min-w-4 items-center justify-center rounded px-1 text-[10px] ${
                        isCurrent
                          ? "bg-black/20 text-black"
                          : "border border-[#303035] bg-[#141416] text-[#A1A1AA]"
                      }`}
                    >
                      {idx + 1}
                    </span>
                    <span className="font-mono-tech truncate">{cc.name}</span>
                  </div>
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full border border-black/30"
                    style={{ backgroundColor: cc.color }}
                  />
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Count Chip matching Roboflow indicator (top-center in Drag mode so zero bottom-bar collision) */}
      {tool === "drag" && selectedDet && !isDraggingDet && (
        <div className="pointer-events-none absolute top-3 left-1/2 z-20 -translate-x-1/2 rounded-full border border-[#303035] bg-[#141416]/95 px-3 py-1 font-mono-tech text-[11px] text-[#D4D4D8] shadow">
          1 annotation selected
        </div>
      )}

      {/* Floating Vertical Right Toolbar (Screenshot UX in Control One theme) */}
      <div
        role="toolbar"
        aria-label="Manual Annotation Tools"
        aria-orientation="vertical"
        className="absolute right-3 top-1/2 z-20 flex -translate-y-1/2 flex-col items-center gap-1 rounded-2xl border border-[#2C2C32] bg-[#141416]/95 p-1.5 shadow-2xl backdrop-blur-md"
      >
        {(
          [
            { id: "drag", label: "Drag / Edit (D)", icon: Hand },
            { id: "bbox", label: "Bounding Box (B)", icon: Square },
            { id: "polygon", label: "Polygon (P)", icon: Pentagon },
            { id: "smart_select", label: "Smart Select · SAM 2.1 (S)", icon: Wand2 },
          ] as const
        ).map((t) => {
          const Icon = t.icon;
          const active = tool === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => onToolChange(t.id)}
              title={t.label}
              aria-label={t.label}
              aria-pressed={active}
              className={`flex h-9 w-9 items-center justify-center rounded-xl transition-colors ${
                active
                  ? "bg-[#F2853F] text-black shadow"
                  : "text-[#A1A1AA] hover:bg-[#222226] hover:text-white"
              }`}
            >
              <Icon className="h-4 w-4" />
            </button>
          );
        })}

        <div className="my-1 h-px w-6 bg-[#2C2C32]" />

        <button
          type="button"
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (U · Cmd/Ctrl+Z)"
          aria-label="Undo"
          className="flex h-9 w-9 items-center justify-center rounded-xl text-[#A1A1AA] transition-colors hover:bg-[#222226] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
        >
          <Undo2 className="h-4 w-4" />
        </button>

        <button
          type="button"
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (R · Cmd/Ctrl+Shift+Z)"
          aria-label="Redo"
          className="flex h-9 w-9 items-center justify-center rounded-xl text-[#A1A1AA] transition-colors hover:bg-[#222226] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
        >
          <Redo2 className="h-4 w-4" />
        </button>

        <div className="my-1 h-px w-6 bg-[#2C2C32]" />

        <button
          type="button"
          onClick={onMarkNull}
          title="Mark Null / 0 Objects & Accept (N)"
          aria-label="Mark Null"
          className={`flex h-9 w-9 items-center justify-center rounded-xl transition-colors ${
            isNullFrame
              ? "border border-emerald-500/60 bg-emerald-500/20 text-emerald-300"
              : "text-[#A1A1AA] hover:bg-rose-500/15 hover:text-rose-300"
          }`}
        >
          <Ban className="h-4 w-4" />
        </button>

        <button
          type="button"
          onClick={onDeleteSelected}
          disabled={!selectedDetId}
          title="Delete Selected Annotation (Del)"
          aria-label="Delete Selected Annotation"
          className="flex h-9 w-9 items-center justify-center rounded-xl text-[#A1A1AA] transition-colors hover:bg-rose-500/15 hover:text-rose-400 disabled:cursor-not-allowed disabled:opacity-30"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      {/* Bottom-Left Zoom / Brightness / Overlay Controls (matching Roboflow screenshot) */}
      <div className="absolute bottom-3 left-3 z-20 flex max-w-[calc(100%-4.5rem)] flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 rounded-xl border border-[#2C2C32] bg-[#141416]/95 px-2 py-1 text-xs shadow-xl backdrop-blur-md">
          <button
            type="button"
            onClick={() =>
              setZoom((z) => Number(Math.max(0.35, z - 0.1).toFixed(2)))
            }
            aria-label="Zoom out"
            className="rounded-lg p-1 text-[#A1A1AA] hover:bg-[#222226] hover:text-white"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <span className="font-mono-tech min-w-[44px] text-center text-[11px] text-white">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() =>
              setZoom((z) => Number(Math.min(6.0, z + 0.1).toFixed(2)))
            }
            aria-label="Zoom in"
            className="rounded-lg p-1 text-[#A1A1AA] hover:bg-[#222226] hover:text-white"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={resetView}
            className="font-mono-tech ml-1 rounded-lg border border-[#303035] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#A1A1AA] hover:border-[#F2853F] hover:text-white"
          >
            RESET
          </button>
        </div>

        <div className="relative">
          <button
            type="button"
            onClick={() => setDisplayOpen((v) => !v)}
            title="Image Brightness & Contrast"
            aria-label="Display settings"
            className={`flex h-8 w-8 items-center justify-center rounded-xl border transition-colors ${
              brightness !== 100 || contrast !== 100
                ? "border-[#F2853F] bg-[#F2853F]/15 text-[#F2853F]"
                : "border-[#2C2C32] bg-[#141416]/95 text-[#A1A1AA] hover:text-white"
            }`}
          >
            <Sun className="h-3.5 w-3.5" />
          </button>

          {displayOpen && (
            <div className="c1-card absolute bottom-10 left-0 z-30 w-56 space-y-3 p-3 shadow-2xl">
              <div className="flex items-center justify-between">
                <span className="c1-kicker">DISPLAY</span>
                <button
                  type="button"
                  onClick={() => {
                    setBrightness(100);
                    setContrast(100);
                  }}
                  className="inline-flex items-center gap-1 text-[10px] text-[#F2853F] hover:underline"
                >
                  <RotateCcw className="h-2.5 w-2.5" />
                  Reset
                </button>
              </div>
              <label className="block text-[11px] text-[#A1A1AA]">
                <span className="flex justify-between">
                  <span>Brightness</span>
                  <span className="font-mono-tech text-white">{brightness}%</span>
                </span>
                <input
                  type="range"
                  min={50}
                  max={180}
                  value={brightness}
                  onChange={(e) => setBrightness(Number(e.target.value))}
                  className="mt-1 w-full accent-[#F2853F]"
                />
              </label>
              <label className="block text-[11px] text-[#A1A1AA]">
                <span className="flex justify-between">
                  <span>Contrast</span>
                  <span className="font-mono-tech text-white">{contrast}%</span>
                </span>
                <input
                  type="range"
                  min={50}
                  max={180}
                  value={contrast}
                  onChange={(e) => setContrast(Number(e.target.value))}
                  className="mt-1 w-full accent-[#F2853F]"
                />
              </label>
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={() => setShowOverlay((v) => !v)}
          title="Toggle Annotations Overlay (O)"
          aria-label="Toggle Annotations Overlay"
          className={`flex h-8 items-center gap-1.5 rounded-xl border px-2.5 text-[11px] font-medium transition-colors ${
            showOverlay
              ? "border-[#F2853F]/60 bg-[#F2853F]/15 text-white"
              : "border-[#2C2C32] bg-[#141416]/95 text-[#8A8A8E] hover:text-white"
          }`}
        >
          {showOverlay ? (
            <Eye className="h-3.5 w-3.5 text-[#F2853F]" />
          ) : (
            <EyeOff className="h-3.5 w-3.5" />
          )}
          Overlay
        </button>

        <button
          type="button"
          onClick={() => setShowLabels((v) => !v)}
          title="Toggle Class Labels (L)"
          aria-label="Toggle Class Labels"
          className={`flex h-8 items-center gap-1 rounded-xl border px-2.5 text-[11px] font-medium transition-colors ${
            showLabels
              ? "border-[#303035] bg-[#141416]/95 text-white"
              : "border-[#2C2C32] bg-[#141416]/95 text-[#8A8A8E] hover:text-white"
          }`}
        >
          <Tag className="h-3 w-3 text-[#F2853F]" />
          Labels
        </button>
      </div>
    </div>
  );
}
