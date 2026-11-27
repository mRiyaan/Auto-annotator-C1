"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  CheckSquare,
  Cpu,
  Layers,
  Loader2,
  Menu,
  PenTool,
  ScanSearch,
  ScrollText,
  Sparkles,
  X,
} from "lucide-react";
import { useWorkbench } from "@/lib/workbench-store";

interface NavItem {
  href: string;
  label: string;
  kicker: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  {
    href: "/analyze",
    label: "Analyze",
    kicker: "01 / STUDIO",
    icon: ScanSearch,
  },
  {
    href: "/logs",
    label: "Test Logs",
    kicker: "02 / HISTORY",
    icon: ScrollText,
  },
  {
    href: "/results",
    label: "Results",
    kicker: "03 / INSPECT",
    icon: Layers,
  },
  {
    href: "/auto-annotate",
    label: "Auto-Annotate",
    kicker: "04 / ROBOFLOW",
    icon: Sparkles,
  },
  {
    href: "/verify",
    label: "Verify Auto-Annotation",
    kicker: "05 / REVIEW",
    icon: CheckSquare,
  },
  {
    href: "/annotate",
    label: "Annotate Manually",
    kicker: "06 / EDITOR",
    icon: PenTool,
  },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAnnotateRoute =
    pathname === "/annotate" || pathname.startsWith("/annotate/");
  const {
    runs,
    backendUrl,
    setBackendUrl,
    backendStatus,
    backendMessage,
    backendHealth,
    checkBackendHealth,
    activeEngineDisplay,
    ollamaBusy,
    ollamaMessage,
    toggleOllama,
  } = useWorkbench();

  const [desktopSidebarOpen, setDesktopSidebarOpen] = useState(
    () => !isAnnotateRoute
  );
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [urlInput, setUrlInput] = useState(backendUrl);
  const prevIsAnnotateRef = useRef(isAnnotateRoute);

  useEffect(() => {
    if (isAnnotateRoute) {
      setDesktopSidebarOpen(false);
      setMobileDrawerOpen(false);
    } else if (prevIsAnnotateRef.current) {
      setDesktopSidebarOpen(true);
    }
    prevIsAnnotateRef.current = isAnnotateRoute;
  }, [isAnnotateRoute]);

  useEffect(() => {
    setUrlInput(backendUrl);
  }, [backendUrl]);

  useEffect(() => {
    if (!mobileDrawerOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMobileDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileDrawerOpen]);

  const handleHamburgerClick = () => {
    if (typeof window !== "undefined" && window.innerWidth >= 1024) {
      setDesktopSidebarOpen((prev) => !prev);
    } else {
      setMobileDrawerOpen((prev) => !prev);
    }
  };

  const ollamaOn = Boolean(backendHealth?.ollamaReachable);

  const statusDotClass =
    backendStatus === "online"
      ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.75)]"
      : backendStatus === "checking"
        ? "bg-[#F2853F] animate-pulse"
        : backendStatus === "offline"
          ? "bg-amber-400"
          : "bg-zinc-400";

  const statusText =
    backendStatus === "online"
      ? "Local API Online"
      : backendStatus === "checking"
        ? "Checking..."
        : backendStatus === "offline"
          ? "UI Preview Mode"
          : "Local Preview";

  const sidebarContent = (
    <div className="flex h-full w-[264px] flex-col justify-between p-4">
      <div className="space-y-6">
        <div className="px-2 pt-1">
          <p className="c1-kicker mb-2">NAVIGATION</p>
          <nav aria-label="Main Navigation" className="space-y-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive =
                pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(`${item.href}/`));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileDrawerOpen(false)}
                  className={`group flex items-center justify-between rounded-xl px-3 py-2.5 transition-colors ${
                    isActive
                      ? "border border-[#F2853F]/40 bg-[#1B1715] text-white"
                      : "border border-transparent text-[#A1A1AA] hover:border-[#26262A] hover:bg-[#141416] hover:text-white"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`flex h-8 w-8 items-center justify-center rounded-lg border ${
                        isActive
                          ? "border-[#F2853F]/40 bg-[#F2853F]/15 text-[#F2853F]"
                          : "border-[#26262A] bg-[#141416] text-[#8A8A8E] group-hover:text-white"
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <div>
                      <div className="text-sm font-medium leading-tight">
                        {item.label}
                      </div>
                      <div className="font-mono-tech text-[10px] tracking-[0.12em] text-[#8A8A8E]">
                        {item.kicker}
                      </div>
                    </div>
                  </div>
                  {item.href === "/logs" && (
                    <span className="font-mono-tech rounded-full border border-[#303035] bg-[#222225] px-2 py-0.5 text-[11px] text-[#A1A1AA]">
                      {runs.length}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="c1-inset p-3.5">
          <div className="flex items-center justify-between">
            <span className="c1-kicker">ACTIVE ENGINE</span>
            <Cpu className="h-3.5 w-3.5 text-[#F2853F]" />
          </div>
          <p className="mt-1.5 text-xs font-medium text-white">
            {activeEngineDisplay.shortLabel}
          </p>
          <p className="font-mono-tech mt-1 text-[11px] text-[#8A8A8E]">
            {activeEngineDisplay.engine} · {activeEngineDisplay.quantization}
          </p>
          {backendHealth && (
            <p className="font-mono-tech mt-1 text-[10px] text-emerald-400">
              Queue: {backendHealth.queueDepth} · Ollama:{" "}
              {ollamaOn ? "Ready" : "Off"}
            </p>
          )}

          <div className="mt-2.5 flex items-center justify-between border-t border-[#2E2E34] pt-2.5">
            <div className="min-w-0">
              <span className="font-mono-tech flex items-center gap-1.5 text-[10px] uppercase tracking-[0.1em] text-[#A1A1AA]">
                Ollama Server
                {ollamaBusy && (
                  <Loader2 className="h-2.5 w-2.5 animate-spin text-[#F2853F]" />
                )}
              </span>
              <p
                className={`font-mono-tech text-[10px] ${
                  ollamaOn ? "text-emerald-400" : "text-[#8A8A8E]"
                }`}
              >
                {ollamaBusy
                  ? ollamaOn
                    ? "Stopping..."
                    : "Starting..."
                  : ollamaOn
                    ? "On · Reachable"
                    : "Off · Unreachable"}
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={ollamaOn}
              aria-label="Toggle Ollama Server"
              disabled={backendStatus !== "online" || ollamaBusy}
              onClick={() => void toggleOllama(ollamaOn ? "stop" : "start")}
              title={
                backendStatus !== "online"
                  ? "Local FastAPI backend must be online to start or stop Ollama"
                  : ollamaOn
                    ? "Stop local Ollama server"
                    : "Start local Ollama server (`ollama serve`)"
              }
              className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
                ollamaOn
                  ? "border-emerald-400/60 bg-emerald-500/30"
                  : "border-[#3A3A44] bg-[#141416]"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 rounded-full transition-transform ${
                  ollamaBusy
                    ? "translate-x-2.5 animate-pulse bg-[#F2853F]"
                    : ollamaOn
                      ? "translate-x-[18px] bg-emerald-400"
                      : "translate-x-0.5 bg-[#8A8A8E]"
                }`}
              />
            </button>
          </div>
          {ollamaMessage && (
            <p className="mt-1.5 text-[10px] leading-snug text-amber-300">
              {ollamaMessage}
            </p>
          )}
        </div>
      </div>

      {/* Bottom Local Laptop Server Status Card */}
      <div className="c1-card p-4">
        <div className="flex items-center justify-between">
          <span className="c1-kicker">LAPTOP SERVER</span>
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-[#A1A1AA]">
            <span className={`h-2 w-2 rounded-full ${statusDotClass}`} />
            {statusText}
          </span>
        </div>

        <label className="mt-3 block">
          <span className="font-mono-tech text-[10px] uppercase tracking-[0.12em] text-[#8A8A8E]">
            Local Backend URL
          </span>
          <input
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onBlur={() => setBackendUrl(urlInput)}
            placeholder="http://127.0.0.1:8000"
            className="font-mono-tech mt-1 w-full rounded-lg border border-[#303035] bg-[#0A0A0C] px-2.5 py-1.5 text-xs text-white placeholder:text-zinc-600 focus:border-[#F2853F] focus:outline-none"
          />
        </label>

        <p className="mt-2 text-[11px] leading-relaxed text-[#A1A1AA]">
          {backendMessage}
        </p>

        <button
          type="button"
          onClick={() => {
            setBackendUrl(urlInput);
            void checkBackendHealth(urlInput);
          }}
          disabled={backendStatus === "checking"}
          className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-full border border-white/25 bg-transparent px-4 py-1.5 text-xs font-medium text-white transition-colors hover:border-white/55 hover:bg-white/5 disabled:opacity-50"
        >
          <Activity className="h-3.5 w-3.5 text-[#F2853F]" />
          {backendStatus === "checking" ? "Checking..." : "Check status"}
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      {/* Top Sticky Header */}
      <header className="z-30 flex h-16 shrink-0 items-center justify-between border-b border-[#1E1E22] bg-[#0A0A0C]/90 px-4 backdrop-blur-md lg:px-6">
        <div className="flex items-center gap-3.5">
          <button
            type="button"
            onClick={handleHamburgerClick}
            aria-label="Toggle navigation sidebar"
            aria-expanded={desktopSidebarOpen || mobileDrawerOpen}
            aria-controls="workbench-sidebar"
            className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-[#26262A] bg-[#141416] text-white transition-colors hover:border-[#F2853F]/60 hover:text-[#F2853F]"
          >
            <Menu className="h-4.5 w-4.5" />
          </button>

          <Link href="/analyze" className="flex items-center gap-3">
            <img
              src="/control-one-logo.webp"
              alt="CONTROL ONE"
              className="h-6 w-auto object-contain"
            />
            <span className="hidden h-4 w-px bg-[#26262A] sm:inline-block" />
            <span className="c1-kicker hidden sm:inline-block">
              VLM WORKBENCH
            </span>
          </Link>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="hidden items-center gap-2 rounded-full border border-[#26262A] bg-[#141416] px-3 py-1.5 md:flex">
            <span className="h-2 w-2 rounded-full bg-[#F2853F]" />
            <span className="font-mono-tech text-[11px] text-[#A1A1AA]">
              {activeEngineDisplay.shortLabel}
            </span>
          </div>
          <span className="font-mono-tech rounded-full border border-white/20 bg-[#141416] px-3 py-1 text-[11px] uppercase tracking-[0.12em] text-white">
            LOCAL MODE
          </span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Desktop Collapsible Independent-Scroll Sidebar */}
        <aside
          id="workbench-sidebar"
          aria-hidden={!desktopSidebarOpen}
          className={`hidden h-full min-w-0 shrink-0 overflow-hidden border-r bg-[#101013] transition-[width,opacity,border-color] duration-150 lg:block ${
            desktopSidebarOpen
              ? "w-[264px] border-[#222226] opacity-100"
              : "pointer-events-none w-0 border-transparent opacity-0"
          }`}
        >
          <div className="h-full overflow-y-auto overscroll-contain">
            {sidebarContent}
          </div>
        </aside>

        {/* Mobile / Tablet Slide-Over Drawer */}
        {mobileDrawerOpen && (
          <div
            className="fixed inset-0 z-40 flex lg:hidden"
            role="dialog"
            aria-modal="true"
            aria-label="Navigation menu"
          >
            <div
              className="fixed inset-0 bg-black/75 backdrop-blur-xs"
              onClick={() => setMobileDrawerOpen(false)}
            />
            <div className="relative z-50 flex h-full w-[264px] flex-col border-r border-[#222226] bg-[#101013] shadow-2xl">
              <div className="flex h-16 items-center justify-between border-b border-[#1E1E22] px-4">
                <img
                  src="/control-one-logo.webp"
                  alt="CONTROL ONE"
                  className="h-5 w-auto object-contain"
                />
                <button
                  type="button"
                  onClick={() => setMobileDrawerOpen(false)}
                  aria-label="Close navigation menu"
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[#26262A] bg-[#141416] text-[#A1A1AA] hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto">{sidebarContent}</div>
            </div>
          </div>
        )}

        {/* Main Content (Independently Scrollable) */}
        {isAnnotateRoute ? (
          <main className="h-full min-w-0 flex-1 overflow-hidden p-0">
            {children}
          </main>
        ) : (
          <main className="h-full min-w-0 flex-1 overflow-y-auto overscroll-contain px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
            <div className="mx-auto max-w-[1440px]">{children}</div>
          </main>
        )}
      </div>
    </div>
  );
}
