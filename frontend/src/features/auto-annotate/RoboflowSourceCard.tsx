"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Eye, EyeOff, FolderGit2, Loader2 } from "lucide-react";
import { useWorkbench } from "@/lib/workbench-store";
import { inspectRoboflowProject } from "./api";
import { useAutoAnnotate } from "./store";
import type {
  AutoAnnotationStatusFilter,
  AutoFetchScope,
  RoboflowInspectResponse,
} from "./types";

interface RoboflowSourceCardProps {
  kicker?: string;
  title?: string;
  badgeText?: string;
  onInspectSuccess?: (info: RoboflowInspectResponse) => void;
  footer?: React.ReactNode;
  className?: string;
}

export function RoboflowSourceCard({
  kicker = "01 / ROBOFLOW DATASET SOURCE",
  title = "Connect Project & Fetch Scope",
  badgeText = "Cached for active window session",
  onInspectSuccess,
  footer,
  className = "",
}: RoboflowSourceCardProps) {
  const { backendUrl } = useWorkbench();
  const {
    apiKey,
    setApiKey,
    workspaceSlug,
    setWorkspaceSlug,
    projectSlug,
    setProjectSlug,
    inspectInfo,
    setInspectInfo,
    fetchScope,
    setFetchScope,
    annotationStatusFilter,
    setAnnotationStatusFilter,
    sourceBatchId,
    setSourceBatchId,
    sourceAnnotationJobId,
    setSourceAnnotationJobId,
    assigneeEmail,
    setAssigneeEmail,
    maxImages,
    setMaxImages,
    batchSize,
    setBatchSize,
    sessionEpoch,
    clearConnection,
  } = useAutoAnnotate();

  const [showKey, setShowKey] = useState(false);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);

  const prevEpochRef = useRef(sessionEpoch);
  useEffect(() => {
    if (sessionEpoch === 0 || prevEpochRef.current === sessionEpoch) return;
    prevEpochRef.current = sessionEpoch;
    setInspectError(null);
  }, [sessionEpoch]);

  const filteredAnnotationJobs = useMemo(() => {
    const allJobs = inspectInfo?.annotationJobs ?? [];
    const q = assigneeEmail.trim().toLowerCase();
    if (!q) return allJobs;
    return allJobs.filter((j) => {
      const lab = (j.labelerEmail || "").toLowerCase();
      const rev = (j.reviewerEmail || "").toLowerCase();
      const nm = (j.name || "").toLowerCase();
      return lab.includes(q) || rev.includes(q) || nm.includes(q);
    });
  }, [inspectInfo?.annotationJobs, assigneeEmail]);

  useEffect(() => {
    if (fetchScope !== "annotation_job" || !filteredAnnotationJobs.length) return;
    if (assigneeEmail.trim() && sourceAnnotationJobId === "") return;
    if (!filteredAnnotationJobs.some((j) => j.id === sourceAnnotationJobId)) {
      setSourceAnnotationJobId(filteredAnnotationJobs[0].id);
    }
  }, [
    assigneeEmail,
    fetchScope,
    filteredAnnotationJobs,
    setSourceAnnotationJobId,
    sourceAnnotationJobId,
  ]);

  const handleInspect = async () => {
    setInspectError(null);
    if (!apiKey.trim() || !projectSlug.trim()) {
      setInspectError("Enter both Roboflow API Key and Project Slug first.");
      return;
    }
    setInspectLoading(true);
    try {
      const info = await inspectRoboflowProject(backendUrl, {
        apiKey: apiKey.trim(),
        projectSlug: projectSlug.trim(),
        workspaceSlug: workspaceSlug.trim() || undefined,
      });
      setInspectInfo(info);
      if (info.workspaceSlug) {
        setWorkspaceSlug(info.workspaceSlug);
      }
      if (info.projectSlug) {
        setProjectSlug(info.projectSlug);
      }
      setSourceBatchId(info.batches[0]?.id || "");
      if (info.annotationJobs && info.annotationJobs.length > 0) {
        const emailQ = assigneeEmail.trim().toLowerCase();
        const emailMatch = emailQ
          ? info.annotationJobs.find(
              (j) =>
                (j.labelerEmail || "").toLowerCase().includes(emailQ) ||
                (j.reviewerEmail || "").toLowerCase().includes(emailQ)
            )
          : info.annotationJobs.find(
              (j) =>
                j.status.toLowerCase().includes("annotat") ||
                j.status.toLowerCase().includes("assign")
            ) ?? info.annotationJobs[0];
        setSourceAnnotationJobId(emailMatch ? emailMatch.id : "");
      } else {
        setSourceAnnotationJobId("");
      }
      onInspectSuccess?.(info);
    } catch (err) {
      setInspectError(
        err instanceof Error ? err.message : "Failed to inspect Roboflow project."
      );
    } finally {
      setInspectLoading(false);
    }
  };

  return (
    <section className={`c1-card p-5 sm:p-6 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="c1-kicker">{kicker}</p>
          <h2 className="mt-1 text-base font-medium text-white">{title}</h2>
        </div>
        <span className="font-mono-tech rounded-full border border-[#303035] bg-[#1C1C20] px-2.5 py-1 text-[11px] text-[#A1A1AA]">
          {badgeText}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3.5 sm:grid-cols-2">
        <label className="block sm:col-span-2">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Roboflow Private API Key *
          </span>
          <div className="relative mt-1.5">
            <input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setInspectInfo(null);
                setInspectError(null);
                setSourceBatchId("");
                setSourceAnnotationJobId("");
              }}
              placeholder="Paste Roboflow Private API Key (not rf_ publishable key)"
              autoComplete="off"
              className="font-mono-tech w-full rounded-xl border border-[#303035] bg-[#0A0A0C] py-2.5 pl-3.5 pr-10 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              aria-label={showKey ? "Hide API key" : "Show API key"}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-[#8A8A8E] hover:text-white"
            >
              {showKey ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          {apiKey.trim().startsWith("rf_") && (
            <p className="mt-1.5 text-[11px] text-amber-300">
              Keys starting with <code className="font-mono-tech">rf_</code> are
              usually Roboflow Publishable Keys, which cannot query datasets or
              Search. Use your Private API Key from Roboflow Settings -&gt; API
              Keys.
            </p>
          )}
        </label>

        <label className="block">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Project Name / Slug *
          </span>
          <input
            type="text"
            value={projectSlug}
            onChange={(e) => {
              setProjectSlug(e.target.value);
              setInspectInfo(null);
              setInspectError(null);
              setSourceBatchId("");
              setSourceAnnotationJobId("");
            }}
            placeholder="e.g. warehouse-safety-v2"
            className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
          />
        </label>

        <label className="block">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Workspace Slug (Optional)
          </span>
          <input
            type="text"
            value={workspaceSlug}
            onChange={(e) => {
              setWorkspaceSlug(e.target.value);
              setInspectInfo(null);
              setInspectError(null);
              setSourceBatchId("");
              setSourceAnnotationJobId("");
            }}
            placeholder="Auto-detect from key"
            className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleInspect}
          disabled={inspectLoading}
          className="inline-flex items-center gap-2 rounded-full border border-white/25 bg-[#1B1B1F] px-4 py-2 text-xs font-medium text-white transition-colors hover:border-[#F2853F]/70 hover:bg-[#242429] disabled:opacity-50"
        >
          {inspectLoading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-[#F2853F]" />
          ) : (
            <FolderGit2 className="h-3.5 w-3.5 text-[#F2853F]" />
          )}
          {inspectLoading
            ? "Connecting to Roboflow..."
            : "Connect & Inspect Project"}
        </button>

        {(apiKey || workspaceSlug || projectSlug || inspectInfo) && (
          <button
            type="button"
            onClick={() => {
              clearConnection();
              setInspectError(null);
            }}
            className="inline-flex items-center gap-1.5 rounded-full border border-[#303035] bg-[#141416] px-3.5 py-2 text-xs text-[#A1A1AA] transition-colors hover:border-white/35 hover:text-white"
          >
            Clear Connection
          </button>
        )}

        {inspectInfo && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="font-mono-tech rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2.5 py-1 text-[11px] text-emerald-300">
              {inspectInfo.workspaceSlug}/{inspectInfo.projectSlug}
            </span>
            <span className="font-mono-tech rounded-full border border-[#303035] bg-[#1D1D22] px-2.5 py-1 text-[11px] text-[#A1A1AA]">
              {inspectInfo.projectType}
            </span>
            <span className="font-mono-tech rounded-full border border-[#303035] bg-[#1D1D22] px-2.5 py-1 text-[11px] text-white">
              {inspectInfo.totalImages} total · {inspectInfo.unannotatedImages}{" "}
              unannotated
            </span>
            {inspectInfo.annotationJobs &&
              inspectInfo.annotationJobs.length > 0 && (
                <span className="font-mono-tech rounded-full border border-[#F2853F]/40 bg-[#F2853F]/10 px-2.5 py-1 text-[11px] text-[#F2853F]">
                  {inspectInfo.annotationJobs.length} annotation job
                  {inspectInfo.annotationJobs.length === 1 ? "" : "s"}
                </span>
              )}
          </div>
        )}
      </div>

      {inspectError && (
        <p className="mt-3 rounded-xl border border-rose-500/35 bg-rose-500/10 px-3.5 py-2 text-xs text-rose-300">
          {inspectError}
        </p>
      )}

      {inspectInfo?.warning && (
        <p className="mt-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3.5 py-2 text-xs text-amber-200">
          {inspectInfo.warning}
        </p>
      )}

      <div className="mt-5 grid grid-cols-1 gap-3.5 border-t border-[#222226] pt-4 sm:grid-cols-3">
        <label className="block">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Fetch Filter
          </span>
          <select
            value={fetchScope}
            onChange={(e) => setFetchScope(e.target.value as AutoFetchScope)}
            className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="unannotated">Unannotated Images Only</option>
            <option value="annotated">Annotated Images Only</option>
            <option value="dataset">
              Dataset Images (Direct from Dataset)
            </option>
            <option value="annotation_job">
              Assigned Annotation Job (Annotate Tab)
            </option>
            <option value="batch">Specific Roboflow Batch</option>
            <option value="all">All Project Images</option>
          </select>
        </label>

        <label className="block">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Max Images to Fetch
          </span>
          <select
            value={maxImages}
            onChange={(e) => setMaxImages(Number(e.target.value))}
            className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value={50}>50 images (1 batch)</option>
            <option value={100}>100 images (2 batches)</option>
            <option value={250}>250 images (5 batches)</option>
            <option value={500}>500 images (10 batches)</option>
            <option value={0}>All matching images</option>
          </select>
        </label>

        <label className="block">
          <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Verify Batch Size
          </span>
          <input
            type="number"
            min={10}
            max={200}
            value={batchSize}
            onChange={(e) =>
              setBatchSize(Math.max(1, Number(e.target.value) || 50))
            }
            className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          />
        </label>

        {fetchScope === "annotation_job" && (
          <div className="grid grid-cols-1 gap-3 sm:col-span-3 sm:grid-cols-3">
            <label className="block">
              <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                Assignee Email ID (Optional Filter)
              </span>
              <input
                type="text"
                value={assigneeEmail}
                onChange={(e) => setAssigneeEmail(e.target.value)}
                placeholder="e.g. your.email@company.com"
                className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                Roboflow Annotate Job
              </span>
              {filteredAnnotationJobs.length > 0 ? (
                <select
                  value={sourceAnnotationJobId}
                  onChange={(e) => setSourceAnnotationJobId(e.target.value)}
                  className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
                >
                  {assigneeEmail.trim() && (
                    <option value="">
                      All jobs for {assigneeEmail.trim()} (
                      {filteredAnnotationJobs.length})
                    </option>
                  )}
                  {filteredAnnotationJobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.name} ({j.annotatedCount}/{j.imagesCount} annotated
                      {j.labelerEmail ? ` · ${j.labelerEmail}` : ""}
                      {j.status ? ` · ${j.status}` : ""})
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={sourceAnnotationJobId}
                  onChange={(e) => setSourceAnnotationJobId(e.target.value)}
                  placeholder={
                    inspectInfo
                      ? "No matching job found — leave blank if filtering by email or paste Job ID"
                      : "Connect & Inspect Project first or paste Job ID"
                  }
                  className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
                />
              )}
            </label>

            <label className="block">
              <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
                Job Image Status
              </span>
              <select
                value={annotationStatusFilter}
                onChange={(e) =>
                  setAnnotationStatusFilter(
                    e.target.value as AutoAnnotationStatusFilter
                  )
                }
                className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
              >
                <option value="all">
                  All Job Images (Unannotated + Annotated)
                </option>
                <option value="annotated">Annotated Images Only</option>
                <option value="unannotated">Unannotated Images Only</option>
              </select>
            </label>
          </div>
        )}

        {fetchScope === "dataset" && (
          <label className="block sm:col-span-3">
            <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
              Dataset Image Status
            </span>
            <select
              value={annotationStatusFilter}
              onChange={(e) =>
                setAnnotationStatusFilter(
                  e.target.value as AutoAnnotationStatusFilter
                )
              }
              className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
            >
              <option value="all">All Dataset Images (Annotated + Null)</option>
              <option value="annotated">Annotated Dataset Images Only</option>
              <option value="unannotated">
                Null / Unannotated Dataset Images Only
              </option>
            </select>
          </label>
        )}

        {fetchScope === "batch" && (
          <label className="block sm:col-span-3">
            <span className="font-mono-tech text-[11px] uppercase tracking-[0.12em] text-[#8A8A8E]">
              Source Roboflow Batch
            </span>
            {inspectInfo && inspectInfo.batches.length > 0 ? (
              <select
                value={sourceBatchId}
                onChange={(e) => setSourceBatchId(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
              >
                {inspectInfo.batches.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} ({b.imagesCount} imgs · {b.id})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={sourceBatchId}
                onChange={(e) => setSourceBatchId(e.target.value)}
                placeholder="Paste Roboflow batch ID or inspect project first"
                className="font-mono-tech mt-1.5 w-full rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
              />
            )}
          </label>
        )}
      </div>

      {footer}
    </section>
  );
}
