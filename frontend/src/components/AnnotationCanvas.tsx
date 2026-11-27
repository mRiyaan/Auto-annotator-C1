"use client";

import React, { useEffect, useRef, useState } from "react";
import type { DetectionItem } from "@/lib/types";

const POLYGON_PALETTE = [
  { fill: [74, 222, 128], stroke: "#4ADE80" }, // Neon green
  { fill: [242, 133, 63], stroke: "#F2853F" }, // Control One orange
  { fill: [232, 121, 249], stroke: "#E879F9" }, // Magenta
];

interface AnnotationCanvasProps {
  imageUrl: string;
  filename: string;
  detections: DetectionItem[];
  showBoxes: boolean;
  showPolygons: boolean;
  showLabels: boolean;
  maskOpacity: number; // 0..1
  hoveredIndex: number | null;
  onHoverIndexChange?: (idx: number | null) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
}

export function AnnotationCanvas({
  imageUrl,
  filename,
  detections,
  showBoxes,
  showPolygons,
  showLabels,
  maskOpacity,
  hoveredIndex,
  onHoverIndexChange,
  canvasRef: externalCanvasRef,
}: AnnotationCanvasProps) {
  const internalRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = externalCanvasRef ?? internalRef;
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number }>({
    w: 1280,
    h: 720,
  });

  useEffect(() => {
    let active = true;
    setImageReady(false);
    const img = new Image();
    img.crossOrigin = "anonymous";
    const finalize = () => {
      if (!active) return;
      loadedImageRef.current = img;
      setNaturalSize({
        w: img.naturalWidth || 1280,
        h: img.naturalHeight || 720,
      });
      setImageReady(true);
    };
    img.onload = () => {
      if (typeof img.decode === "function") {
        void img.decode().then(finalize, finalize);
      } else {
        finalize();
      }
    };
    img.onerror = () => {
      if (!active) return;
      loadedImageRef.current = null;
      setImageReady(false);
    };
    img.src = imageUrl;
    return () => {
      active = false;
    };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = loadedImageRef.current;
    if (!canvas || !img || !imageReady) return;

    const w = naturalSize.w;
    const h = naturalSize.h;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);

    const scale = Math.max(1, w / 960);

    detections.forEach((det, idx) => {
      const isHovered = hoveredIndex === idx;
      const palette = POLYGON_PALETTE[idx % POLYGON_PALETTE.length];

      // 1. Polygon Segmentation Overlay
      if (showPolygons && det.polygon && det.polygon.length >= 3) {
        const [r, g, b] = palette.fill;
        const alpha = isHovered
          ? Math.min(0.65, maskOpacity + 0.2)
          : maskOpacity;
        ctx.save();
        ctx.beginPath();
        det.polygon.forEach(([nx, ny], pIdx) => {
          const px = (nx / 1000) * w;
          const py = (ny / 1000) * h;
          if (pIdx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.fill();
        ctx.lineWidth = (isHovered ? 3.5 : 2) * scale;
        ctx.strokeStyle = palette.stroke;
        ctx.stroke();
        ctx.restore();
      }

      // 2. 2D Bounding Box Overlay (Cyan #22D3EE)
      const [nx1, ny1, nx2, ny2] = det.bbox_2d;
      const x1 = (nx1 / 1000) * w;
      const y1 = (ny1 / 1000) * h;
      const bw = ((nx2 - nx1) / 1000) * w;
      const bh = ((ny2 - ny1) / 1000) * h;

      if (showBoxes) {
        ctx.save();
        ctx.lineWidth = (isHovered ? 3.5 : 2.2) * scale;
        ctx.strokeStyle = isHovered ? "#F2853F" : "#22D3EE";
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.restore();
      }

      // 3. Label Pill
      if (showLabels && (showBoxes || showPolygons)) {
        const text = `${det.label} ${Math.round(det.confidence * 100)}%`;
        const fontSize = Math.round(13 * scale);
        ctx.save();
        ctx.font = `600 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        const textW = ctx.measureText(text).width;
        const padX = 7 * scale;
        const badgeH = 20 * scale;
        const badgeX = Math.max(0, Math.min(w - textW - padX * 2, x1));
        const badgeY = Math.max(0, y1 - badgeH - 2 * scale);

        ctx.fillStyle = isHovered ? "#F2853F" : "rgba(10, 10, 12, 0.88)";
        ctx.strokeStyle = isHovered ? "#F2853F" : "#22D3EE";
        ctx.lineWidth = 1.2 * scale;
        ctx.beginPath();
        ctx.roundRect(badgeX, badgeY, textW + padX * 2, badgeH, 5 * scale);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isHovered ? "#0A0A0C" : "#FFFFFF";
        ctx.textBaseline = "middle";
        ctx.fillText(text, badgeX + padX, badgeY + badgeH / 2);
        ctx.restore();
      }
    });
  }, [
    canvasRef,
    detections,
    hoveredIndex,
    imageReady,
    maskOpacity,
    naturalSize,
    showBoxes,
    showLabels,
    showPolygons,
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onHoverIndexChange || !canvasRef.current || !detections.length) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 1000;
    const ny = ((e.clientY - rect.top) / rect.height) * 1000;

    let matchIdx: number | null = null;
    for (let i = 0; i < detections.length; i++) {
      const [x1, y1, x2, y2] = detections[i].bbox_2d;
      if (nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2) {
        matchIdx = i;
        break;
      }
    }
    onHoverIndexChange(matchIdx);
  };

  return (
    <div className="relative overflow-hidden rounded-xl border border-[#303035] bg-[#0A0A0C]">
      {!imageReady && (
        <div className="flex aspect-video items-center justify-center text-xs text-[#8A8A8E]">
          Loading {filename}...
        </div>
      )}
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => onHoverIndexChange?.(null)}
        aria-label={`Annotated output canvas for ${filename}`}
        className={`block w-full h-auto ${imageReady ? "opacity-100" : "hidden"}`}
      />
    </div>
  );
}
