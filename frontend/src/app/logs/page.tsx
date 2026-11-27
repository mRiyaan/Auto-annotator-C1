"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  Copy,
  CopyCheck,
  RotateCcw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { MODEL_CATALOG } from "@/lib/mock-data";
import { useWorkbench } from "@/lib/workbench-store";
import type { MediaType, OutputMode, RunRecord, ShotMode } from "@/lib/types";

export default function TestLogsPage() {
  const router = useRouter();
  const { runs, deleteRun, cloneRunToAnalyze } = useWorkbench();

  const [query, setQuery] = useState("");
  const [mediaFilter, setMediaFilter] = useState<"all" | MediaType>("all");
  const [shotFilter, setShotFilter] = useState<"all" | ShotMode>("all");
  const [outputFilter, setOutputFilter] = useState<"all" | OutputMode>("all");
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    runs[0]?.id ?? null
  );
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredRuns = useMemo(() => {
    const q = query.trim().toLowerCase();
    return runs.filter((run) => {
      if (mediaFilter !== "all" && run.target.mediaType !== mediaFilter)
        return false;
      if (shotFilter !== "all" && run.shotMode !== shotFilter) return false;
      if (outputFilter !== "all" && run.outputMode !== outputFilter)
        return false;
      if (modelFilter !== "all" && run.modelId !== modelFilter) return false;
      if (q) {
        const hay = `${run.id} ${run.target.filename} ${run.prompt} ${run.modelLabel}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [runs, query, mediaFilter, shotFilter, outputFilter, modelFilter]);

  const selectedRun: RunRecord | null = useMemo(() => {
    if (!selectedRunId) return null;
    return runs.find((r) => r.id === selectedRunId) ?? null;
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (!selectedRun) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedRunId(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedRun]);

  const handleCopyJson = async (run: RunRecord) => {
    try {
      await navigator.clipboard.writeText(run.rawJson);
      setCopiedId(run.id);
      window.setTimeout(() => setCopiedId(null), 1500);
    } catch {
      // Clipboard fallback ignored
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="c1-kicker">RUN HISTORY & TELEMETRY</p>
          <h1 className="mt-1.5 text-3xl font-bold tracking-[-0.03em] text-white sm:text-4xl">
            Test logs.
          </h1>
          <p className="mt-1.5 text-sm text-[#A1A1AA]">
            Inspect prompts, attached zero/few-shot references, per-stage
            timings, and raw JSON across local test runs.
          </p>
        </div>

        <Link
          href="/analyze"
          className="inline-flex items-center justify-center gap-2 self-start rounded-full bg-[#F2853F] px-5 py-2 text-xs font-semibold text-[#0A0A0C] hover:bg-[#FF9652]"
        >
          New Evaluation Run
        </Link>
      </div>

      {/* Filters Card */}
      <section className="c1-card p-4 sm:p-5">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-6">
          <label className="relative block md:col-span-2">
            <span className="sr-only">Search runs</span>
            <Search className="pointer-events-none absolute top-1/2 left-3.5 h-4 w-4 -translate-y-1/2 text-[#8A8A8E]" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by run ID, file name, prompt, or model..."
              className="w-full rounded-xl border border-[#303035] bg-[#0A0A0C] py-2 pr-3 pl-10 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
            />
          </label>

          <select
            aria-label="Filter by media type"
            value={mediaFilter}
            onChange={(e) =>
              setMediaFilter(e.target.value as "all" | MediaType)
            }
            className="rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="all">All Media Types</option>
            <option value="image">Image Only</option>
            <option value="video">Video Only</option>
          </select>

          <select
            aria-label="Filter by shot mode"
            value={shotFilter}
            onChange={(e) =>
              setShotFilter(e.target.value as "all" | ShotMode)
            }
            className="rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="all">All Shot Modes</option>
            <option value="zero-shot">Zero-Shot</option>
            <option value="few-shot">Few-Shot</option>
          </select>

          <select
            aria-label="Filter by output format"
            value={outputFilter}
            onChange={(e) =>
              setOutputFilter(e.target.value as "all" | OutputMode)
            }
            className="rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="all">All Outputs</option>
            <option value="bbox">2D BBox</option>
            <option value="polygon">Polygon</option>
            <option value="both">Both</option>
          </select>

          <select
            aria-label="Filter by model"
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="rounded-xl border border-[#303035] bg-[#0A0A0C] px-3 py-2 text-xs text-white focus:border-[#F2853F] focus:outline-none"
          >
            <option value="all">All Models</option>
            {MODEL_CATALOG.map((m) => (
              <option key={m.id} value={m.id}>
                {m.shortLabel}
              </option>
            ))}
          </select>
        </div>
      </section>

      {/* Main Table + Detail Split */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        {/* Runs Table */}
        <section
          className={`c1-card overflow-hidden ${
            selectedRun ? "xl:col-span-7" : "xl:col-span-12"
          }`}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-[#26262A] bg-[#101013]">
                  <th className="c1-kicker px-2.5 py-3 font-medium whitespace-nowrap">
                    Run ID
                  </th>
                  <th className="c1-kicker px-2.5 py-3 font-medium">Target</th>
                  <th className="c1-kicker px-2.5 py-3 font-medium whitespace-nowrap">
                    Mode
                  </th>
                  {!selectedRun && (
                    <th className="c1-kicker px-2.5 py-3 font-medium">
                      Model / Weights
                    </th>
                  )}
                  <th className="c1-kicker px-2.5 py-3 font-medium whitespace-nowrap">
                    Latency
                  </th>
                  <th className="c1-kicker px-2.5 py-3 font-medium whitespace-nowrap">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#222225]">
                {filteredRuns.length === 0 ? (
                  <tr>
                    <td
                      colSpan={selectedRun ? 5 : 6}
                      className="px-4 py-10 text-center text-xs text-[#8A8A8E]"
                    >
                      No runs match the current filters.
                    </td>
                  </tr>
                ) : (
                  filteredRuns.map((run) => {
                    const active = run.id === selectedRun?.id;
                    return (
                      <tr
                        key={run.id}
                        onClick={() => setSelectedRunId(run.id)}
                        className={`cursor-pointer transition-colors ${
                          active
                            ? "bg-[#1C1613]"
                            : "hover:bg-[#19191D]"
                        }`}
                      >
                        <td className="font-mono-tech px-2.5 py-3 whitespace-nowrap text-white">
                          <div className="font-semibold">{run.id}</div>
                          <div className="text-[10px] text-[#8A8A8E]">
                            {run.createdAtDisplay}
                          </div>
                        </td>
                        <td className="px-2.5 py-3">
                          <div className="max-w-[130px] truncate font-medium text-white">
                            {run.target.filename}
                          </div>
                          <div className="font-mono-tech text-[10px] uppercase text-[#8A8A8E]">
                            {run.target.mediaType} · {run.target.width}×
                            {run.target.height}
                          </div>
                          {selectedRun && (
                            <div className="mt-0.5 max-w-[130px] truncate text-[10px] text-[#A1A1AA]">
                              {run.modelLabel}
                            </div>
                          )}
                        </td>
                        <td className="px-2.5 py-3 whitespace-nowrap">
                          <div className="capitalize text-white">
                            {run.shotMode}
                          </div>
                          <div className="font-mono-tech text-[10px] text-[#8A8A8E]">
                            {run.outputMode} · {run.references.length} ref
                          </div>
                        </td>
                        {!selectedRun && (
                          <td className="px-2.5 py-3">
                            <div className="max-w-[200px] truncate text-white">
                              {run.modelLabel}
                            </div>
                            <div className="font-mono-tech text-[10px] text-[#8A8A8E]">
                              {run.detections.length} detection(s)
                            </div>
                          </td>
                        )}
                        <td className="font-mono-tech px-2.5 py-3 whitespace-nowrap text-white">
                          <div>{run.latencySec.toFixed(2)}s</div>
                          <div className="text-[10px] text-[#8A8A8E]">
                            {run.promptTokens.toLocaleString()} /{" "}
                            {run.generatedTokens}
                          </div>
                        </td>
                        <td className="px-2.5 py-3 whitespace-nowrap">
                          <span
                            className={`font-mono-tech inline-flex rounded-full border px-2 py-0.5 text-[10px] uppercase ${
                              run.status === "failed"
                                ? "border-red-500/40 bg-red-500/10 text-red-400"
                                : run.status === "running" || run.status === "queued"
                                  ? "border-amber-400/40 bg-amber-400/10 text-amber-300"
                                  : run.isMockPreview
                                    ? "border-[#F2853F]/40 bg-[#F2853F]/10 text-[#F2853F]"
                                    : "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                            }`}
                          >
                            {run.isMockPreview
                              ? "Preview"
                              : run.status === "completed"
                                ? "Done"
                                : run.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Expandable Detail Drawer */}
        {selectedRun && (
          <aside
            aria-label="Run log details"
            className="c1-card p-5 xl:col-span-5"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="c1-kicker">LOG DETAIL · {selectedRun.id}</p>
                <h2 className="mt-1 text-base font-semibold text-white">
                  {selectedRun.target.filename}
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setSelectedRunId(null)}
                aria-label="Close detail panel"
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[#303035] bg-[#222225] text-[#A1A1AA] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Quick Actions */}
            <div className="mt-4 flex flex-wrap gap-2">
              <Link
                href={`/results?run=${encodeURIComponent(selectedRun.id)}`}
                className="inline-flex items-center gap-1.5 rounded-full bg-[#F2853F] px-4 py-1.5 text-xs font-semibold text-[#0A0A0C] hover:bg-[#FF9652]"
              >
                Open in Results
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
              <button
                type="button"
                onClick={() => {
                  cloneRunToAnalyze(selectedRun);
                  router.push("/analyze");
                }}
                className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3.5 py-1.5 text-xs font-medium text-white hover:border-white/55"
              >
                <RotateCcw className="h-3.5 w-3.5 text-[#F2853F]" />
                Clone to Analyze
              </button>
              <button
                type="button"
                onClick={() => {
                  deleteRun(selectedRun.id);
                  setSelectedRunId(null);
                }}
                className="inline-flex items-center gap-1 rounded-full border border-[#303035] px-3 py-1.5 text-xs text-[#A1A1AA] hover:border-red-500/50 hover:text-red-400"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </button>
            </div>

            {/* Prompt */}
            <div className="c1-inset mt-4 p-3.5">
              <span className="c1-kicker">PROMPT</span>
              <p className="mt-1.5 text-xs leading-relaxed text-zinc-200">
                {selectedRun.prompt}
              </p>
            </div>

            {/* Attached References */}
            {selectedRun.references.length > 0 && (
              <div className="c1-inset mt-3 p-3.5">
                <span className="c1-kicker">
                  ATTACHED REFERENCES ({selectedRun.references.length})
                </span>
                <div className="mt-2 grid grid-cols-2 gap-2.5">
                  {selectedRun.references.map((ref) => (
                    <div
                      key={ref.id}
                      className="overflow-hidden rounded-lg border border-[#303035] bg-[#141416]"
                    >
                      <img
                        src={ref.imageUrl}
                        alt={ref.filename}
                        className="aspect-video w-full object-cover"
                      />
                      <p className="truncate px-2 py-1 text-[10px] text-[#A1A1AA]">
                        {ref.filename}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Per-Stage Metrics */}
            <div className="c1-inset mt-3 p-3.5">
              <span className="c1-kicker">PER-STAGE EXECUTION BREAKDOWN</span>
              <div className="mt-2 space-y-2">
                {selectedRun.stageLogs.map((st, idx) => (
                  <div
                    key={`${st.stage}-${idx}`}
                    className="rounded-lg border border-[#303035] bg-[#141416] p-2.5 text-xs"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-white">{st.stage}</span>
                      <span className="font-mono-tech text-[#F2853F]">
                        {st.wall_time_sec.toFixed(2)}s
                      </span>
                    </div>
                    <p className="mt-0.5 text-[11px] text-[#A1A1AA]">
                      {st.summary}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Raw JSON */}
            <div className="c1-inset mt-3 p-3.5">
              <div className="flex items-center justify-between">
                <span className="c1-kicker">RAW MODEL JSON</span>
                <button
                  type="button"
                  onClick={() => void handleCopyJson(selectedRun)}
                  className="inline-flex items-center gap-1 rounded-full border border-white/20 px-2.5 py-0.5 text-[11px] text-white hover:border-white/50"
                >
                  {copiedId === selectedRun.id ? (
                    <>
                      <CopyCheck className="h-3 w-3 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      Copy JSON
                    </>
                  )}
                </button>
              </div>
              <pre className="font-mono-tech mt-2 max-h-48 overflow-auto rounded-lg bg-[#0A0A0C] p-3 text-[11px] leading-relaxed text-zinc-300">
                {selectedRun.rawJson}
              </pre>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
