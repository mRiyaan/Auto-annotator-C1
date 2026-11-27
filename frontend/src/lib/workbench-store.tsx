"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  controlOllamaServer,
  deleteRunApi,
  fetchBackendModels,
  fetchHealth,
  fetchRuns,
} from "./api";
import {
  DEFAULT_DRAFT_CONFIG,
  INITIAL_MOCK_RUNS,
  getModelOption,
} from "./mock-data";
import type {
  ActiveEngineDisplay,
  BackendHealthInfo,
  BackendModelStatus,
  DraftAnalyzeConfig,
  RunRecord,
} from "./types";

export type BackendStatus = "idle" | "checking" | "online" | "offline";

const STORAGE_KEY_BACKEND_URL = "control_one_backend_url";
const DEFAULT_BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

interface WorkbenchContextValue {
  runs: RunRecord[];
  addRun: (run: RunRecord) => void;
  updateRunInState: (run: RunRecord) => void;
  deleteRun: (runId: string) => void;
  refreshRuns: () => Promise<void>;
  draftConfig: DraftAnalyzeConfig;
  setDraftConfig: React.Dispatch<React.SetStateAction<DraftAnalyzeConfig>>;
  cloneRunToAnalyze: (run: RunRecord) => void;
  backendUrl: string;
  setBackendUrl: (url: string) => void;
  backendStatus: BackendStatus;
  backendMessage: string;
  backendHealth: BackendHealthInfo | null;
  backendModels: BackendModelStatus[];
  checkBackendHealth: (overrideUrl?: string) => Promise<void>;
  ollamaBusy: boolean;
  ollamaMessage: string | null;
  toggleOllama: (action: "start" | "stop") => Promise<void>;
  activeEngineDisplay: ActiveEngineDisplay;
  setActiveEngineDisplay: React.Dispatch<
    React.SetStateAction<ActiveEngineDisplay>
  >;
}

const WorkbenchContext = createContext<WorkbenchContextValue | null>(null);

export function WorkbenchProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<RunRecord[]>(INITIAL_MOCK_RUNS);
  const [draftConfig, setDraftConfig] =
    useState<DraftAnalyzeConfig>(DEFAULT_DRAFT_CONFIG);
  const [backendUrl, setBackendUrlState] = useState<string>(DEFAULT_BACKEND_URL);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("idle");
  const [backendHealth, setBackendHealth] = useState<BackendHealthInfo | null>(
    null
  );
  const [backendModels, setBackendModels] = useState<BackendModelStatus[]>([]);
  const [ollamaBusy, setOllamaBusy] = useState(false);
  const [ollamaMessage, setOllamaMessage] = useState<string | null>(null);
  const [activeEngineDisplay, setActiveEngineDisplay] =
    useState<ActiveEngineDisplay>(() => {
      const m = getModelOption(DEFAULT_DRAFT_CONFIG.modelId);
      return {
        id: m.id,
        shortLabel: m.shortLabel,
        engine: m.engine,
        quantization: m.quantization,
      };
    });
  const [backendMessage, setBackendMessage] = useState<string>(
    "Checking local FastAPI server..."
  );

  const setBackendUrl = useCallback((url: string) => {
    const cleaned = url.trim().replace(/\/+$/, "") || DEFAULT_BACKEND_URL;
    setBackendUrlState(cleaned);
    try {
      window.localStorage.setItem(STORAGE_KEY_BACKEND_URL, cleaned);
    } catch {
      // Ignore storage errors
    }
  }, []);

  const refreshRunsFromUrl = useCallback(async (targetUrl: string) => {
    try {
      const serverRuns = await fetchRuns(targetUrl);
      setRuns(serverRuns);
    } catch {
      // Keep current list on transient fetch errors
    }
  }, []);

  const refreshRuns = useCallback(async () => {
    if (backendStatus !== "online") return;
    await refreshRunsFromUrl(backendUrl);
  }, [backendStatus, backendUrl, refreshRunsFromUrl]);

  const checkBackendHealth = useCallback(
    async (overrideUrl?: string) => {
      const target = (overrideUrl ?? backendUrl).trim().replace(/\/+$/, "");
      if (!target) return;
      setBackendStatus("checking");
      setBackendMessage(`Checking ${target}/api/v1/health...`);

      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 3000);
      try {
        const healthInfo = await fetchHealth(target, controller.signal);
        setBackendHealth(healthInfo);
        setBackendStatus("online");
        setBackendMessage(
          `${healthInfo.device}${
            healthInfo.loadedModelId
              ? ` · Loaded: ${healthInfo.loadedModelId}`
              : " · Idle"
          }`
        );
        try {
          const modelsList = await fetchBackendModels(target);
          setBackendModels(modelsList);
        } catch {
          // Non-fatal
        }
        await refreshRunsFromUrl(target);
      } catch {
        setBackendHealth(null);
        setBackendModels([]);
        setBackendStatus("offline");
        setBackendMessage(
          "Local FastAPI server not reachable (UI preview mode active)"
        );
        setRuns((prev) =>
          prev.length === 0 ? INITIAL_MOCK_RUNS : prev
        );
      } finally {
        window.clearTimeout(timer);
      }
    },
    [backendUrl, refreshRunsFromUrl]
  );

  const toggleOllama = useCallback(
    async (action: "start" | "stop") => {
      if (ollamaBusy || backendStatus !== "online") return;
      setOllamaBusy(true);
      setOllamaMessage(null);
      try {
        const res = await controlOllamaServer(backendUrl, action);
        setBackendHealth((prev) =>
          prev
            ? {
                ...prev,
                ollamaReachable: res.ollamaReachable,
                ollamaModels: res.ollamaModels,
              }
            : prev
        );
        try {
          const modelsList = await fetchBackendModels(backendUrl);
          setBackendModels(modelsList);
        } catch {
          // Non-fatal
        }
        setOllamaMessage(res.ok ? null : res.message);
      } catch (err) {
        setOllamaMessage(
          err instanceof Error ? err.message : `Failed to ${action} Ollama.`
        );
      } finally {
        setOllamaBusy(false);
      }
    },
    [backendStatus, backendUrl, ollamaBusy]
  );

  useEffect(() => {
    let targetUrl = DEFAULT_BACKEND_URL;
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY_BACKEND_URL);
      if (saved && saved.trim()) {
        targetUrl = saved.trim();
        setBackendUrlState(targetUrl);
      }
    } catch {
      // Ignore storage errors
    }
    void checkBackendHealth(targetUrl);
    // Run once on initial mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastBootIdRef = useRef<string | undefined>(undefined);
  const lastStatusRef = useRef<BackendStatus>("idle");
  useEffect(() => {
    lastBootIdRef.current = backendHealth?.bootId;
    lastStatusRef.current = backendStatus;
  }, [backendHealth?.bootId, backendStatus]);

  // Silent background heartbeat so server shutdown / port kill / restart is detected promptly
  useEffect(() => {
    const target = backendUrl.trim().replace(/\/+$/, "");
    if (!target) return;

    let consecutiveFailures = 0;

    const timer = window.setInterval(async () => {
      const controller = new AbortController();
      const abortTimer = window.setTimeout(() => controller.abort(), 6000);
      try {
        const healthInfo = await fetchHealth(target, controller.signal);
        consecutiveFailures = 0;
        const prevBoot = lastBootIdRef.current;
        const wasOffline = lastStatusRef.current !== "online";
        setBackendHealth(healthInfo);
        setBackendStatus("online");
        setBackendMessage(
          `${healthInfo.device}${
            healthInfo.loadedModelId
              ? ` · Loaded: ${healthInfo.loadedModelId}`
              : " · Idle"
          }`
        );
        if (wasOffline || (prevBoot && healthInfo.bootId && prevBoot !== healthInfo.bootId)) {
          try {
            const modelsList = await fetchBackendModels(target);
            setBackendModels(modelsList);
          } catch {
            // Non-fatal
          }
          await refreshRunsFromUrl(target);
        }
      } catch {
        consecutiveFailures += 1;
        if (consecutiveFailures >= 2) {
          setBackendHealth(null);
          setBackendModels([]);
          setBackendStatus("offline");
          setBackendMessage(
            "Local FastAPI server not reachable (UI preview mode active)"
          );
        }
      } finally {
        window.clearTimeout(abortTimer);
      }
    }, 10000);

    return () => window.clearInterval(timer);
  }, [backendUrl, refreshRunsFromUrl]);

  const addRun = useCallback((run: RunRecord) => {
    setRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)]);
  }, []);

  const updateRunInState = useCallback((updated: RunRecord) => {
    setRuns((prev) =>
      prev.map((r) => (r.id === updated.id ? updated : r))
    );
  }, []);

  const deleteRun = useCallback(
    (runId: string) => {
      setRuns((prev) => prev.filter((r) => r.id !== runId));
      if (backendStatus === "online") {
        void deleteRunApi(backendUrl, runId).catch(() => {
          void refreshRunsFromUrl(backendUrl);
        });
      }
    },
    [backendStatus, backendUrl, refreshRunsFromUrl]
  );

  const cloneRunToAnalyze = useCallback((run: RunRecord) => {
    setDraftConfig({
      mediaType: run.target.mediaType,
      shotMode: run.shotMode,
      outputMode: run.outputMode,
      modelId: run.modelId,
      prompt: run.prompt,
      temperature: run.temperature,
      maxTokens: run.maxTokens,
      samplingFps: run.samplingFps ?? 1.0,
      outputFpsMode: run.outputFpsMode ?? "original",
      byteTrackMemorySec: run.byteTrackMemorySec ?? 3.0,
      referenceStyleNote: run.referenceStyleNote ?? "",
    });
  }, []);

  return (
    <WorkbenchContext.Provider
      value={{
        runs,
        addRun,
        updateRunInState,
        deleteRun,
        refreshRuns,
        draftConfig,
        setDraftConfig,
        cloneRunToAnalyze,
        backendUrl,
        setBackendUrl,
        backendStatus,
        backendMessage,
        backendHealth,
        backendModels,
        checkBackendHealth,
        ollamaBusy,
        ollamaMessage,
        toggleOllama,
        activeEngineDisplay,
        setActiveEngineDisplay,
      }}
    >
      {children}
    </WorkbenchContext.Provider>
  );
}

export function useWorkbench(): WorkbenchContextValue {
  const ctx = useContext(WorkbenchContext);
  if (!ctx) {
    throw new Error("useWorkbench must be used inside WorkbenchProvider");
  }
  return ctx;
}
