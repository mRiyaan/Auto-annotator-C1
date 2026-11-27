"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { useWorkbench } from "@/lib/workbench-store";
import { fetchAutoJobs, fetchAutoModels } from "./api";
import {
  clearAllAutoCaches,
  clearConnectionSession,
  clearLegacyLocalAutoCaches,
  loadCachedBootId,
  loadCachedJobs,
  loadConnectionSession,
  saveCachedBootId,
  saveCachedJobs,
  saveConnectionSession,
} from "./cache";
import type {
  AutoAnnotationStatusFilter,
  AutoFetchScope,
  AutoJobRecord,
  AutoModelCatalogResponse,
  RoboflowInspectResponse,
} from "./types";

interface AutoAnnotateContextValue {
  apiKey: string;
  setApiKey: (val: string) => void;
  workspaceSlug: string;
  setWorkspaceSlug: (val: string) => void;
  projectSlug: string;
  setProjectSlug: (val: string) => void;
  inspectInfo: RoboflowInspectResponse | null;
  setInspectInfo: React.Dispatch<
    React.SetStateAction<RoboflowInspectResponse | null>
  >;
  fetchScope: AutoFetchScope;
  setFetchScope: (val: AutoFetchScope) => void;
  annotationStatusFilter: AutoAnnotationStatusFilter;
  setAnnotationStatusFilter: (val: AutoAnnotationStatusFilter) => void;
  sourceBatchId: string;
  setSourceBatchId: (val: string) => void;
  sourceAnnotationJobId: string;
  setSourceAnnotationJobId: (val: string) => void;
  assigneeEmail: string;
  setAssigneeEmail: (val: string) => void;
  maxImages: number;
  setMaxImages: (val: number) => void;
  batchSize: number;
  setBatchSize: (val: number) => void;
  models: AutoModelCatalogResponse | null;
  jobs: AutoJobRecord[];
  setJobs: React.Dispatch<React.SetStateAction<AutoJobRecord[]>>;
  refreshJobs: () => Promise<void>;
  refreshModels: () => Promise<void>;
  sessionEpoch: number;
  clearConnection: () => void;
}

const AutoAnnotateContext = createContext<AutoAnnotateContextValue | null>(
  null
);

export function AutoAnnotateProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { backendUrl, backendStatus, backendHealth } = useWorkbench();

  // Credentials, project details, and fetch filters are cached in window sessionStorage
  // (persists across page switches & soft reloads while window is open; cleared on server down/restart)
  const [apiKey, setApiKey] = useState("");
  const [workspaceSlug, setWorkspaceSlug] = useState("");
  const [projectSlug, setProjectSlug] = useState("");
  const [inspectInfo, setInspectInfo] =
    useState<RoboflowInspectResponse | null>(null);
  const [fetchScope, setFetchScope] = useState<AutoFetchScope>("unannotated");
  const [annotationStatusFilter, setAnnotationStatusFilter] =
    useState<AutoAnnotationStatusFilter>("all");
  const [sourceBatchId, setSourceBatchId] = useState("");
  const [sourceAnnotationJobId, setSourceAnnotationJobId] = useState("");
  const [assigneeEmail, setAssigneeEmail] = useState("");
  const [maxImages, setMaxImages] = useState<number>(100);
  const [batchSize, setBatchSize] = useState<number>(50);

  const [models, setModels] = useState<AutoModelCatalogResponse | null>(null);
  const [jobs, setJobs] = useState<AutoJobRecord[]>([]);
  const [cacheHydrated, setCacheHydrated] = useState(false);
  const [sessionEpoch, setSessionEpoch] = useState(0);

  const clearConnection = useCallback(() => {
    setApiKey("");
    setWorkspaceSlug("");
    setProjectSlug("");
    setInspectInfo(null);
    setSourceBatchId("");
    setSourceAnnotationJobId("");
    setAssigneeEmail("");
    clearConnectionSession();
  }, []);

  // Purge any legacy localStorage caches on mount; hydrate session-scoped connection & job list
  useEffect(() => {
    clearLegacyLocalAutoCaches();
    const cachedJobs = loadCachedJobs();
    if (cachedJobs.length > 0) {
      setJobs((prev) => (prev.length > 0 ? prev : cachedJobs));
    }
    const savedConn = loadConnectionSession();
    if (savedConn) {
      if (typeof savedConn.apiKey === "string") setApiKey(savedConn.apiKey);
      if (typeof savedConn.workspaceSlug === "string") {
        setWorkspaceSlug(savedConn.workspaceSlug);
      }
      if (typeof savedConn.projectSlug === "string") {
        setProjectSlug(savedConn.projectSlug);
      }
      if (savedConn.inspectInfo) setInspectInfo(savedConn.inspectInfo);
      if (savedConn.fetchScope) setFetchScope(savedConn.fetchScope);
      if (savedConn.annotationStatusFilter) {
        setAnnotationStatusFilter(savedConn.annotationStatusFilter);
      }
      if (typeof savedConn.sourceBatchId === "string") {
        setSourceBatchId(savedConn.sourceBatchId);
      }
      if (typeof savedConn.sourceAnnotationJobId === "string") {
        setSourceAnnotationJobId(savedConn.sourceAnnotationJobId);
      }
      if (typeof savedConn.assigneeEmail === "string") {
        setAssigneeEmail(savedConn.assigneeEmail);
      }
      if (typeof savedConn.maxImages === "number") {
        setMaxImages(savedConn.maxImages);
      }
      if (typeof savedConn.batchSize === "number") {
        setBatchSize(savedConn.batchSize);
      }
    }
    setCacheHydrated(true);
  }, []);

  useEffect(() => {
    if (!cacheHydrated || backendStatus === "offline") return;
    saveCachedJobs(jobs);
  }, [cacheHydrated, jobs, backendStatus]);

  useEffect(() => {
    if (!cacheHydrated || backendStatus === "offline") return;
    saveConnectionSession({
      apiKey,
      workspaceSlug,
      projectSlug,
      inspectInfo,
      fetchScope,
      annotationStatusFilter,
      sourceBatchId,
      sourceAnnotationJobId,
      assigneeEmail,
      maxImages,
      batchSize,
    });
  }, [
    annotationStatusFilter,
    apiKey,
    assigneeEmail,
    backendStatus,
    batchSize,
    cacheHydrated,
    fetchScope,
    inspectInfo,
    maxImages,
    projectSlug,
    sourceAnnotationJobId,
    sourceBatchId,
    workspaceSlug,
  ]);

  const refreshModels = useCallback(async () => {
    try {
      const cat = await fetchAutoModels(backendUrl);
      setModels(cat);
    } catch {
      // ignore if backend offline
    }
  }, [backendUrl]);

  const refreshJobs = useCallback(async () => {
    try {
      const list = await fetchAutoJobs(backendUrl);
      setJobs(list);
    } catch {
      // keep current list on transient error
    }
  }, [backendUrl]);

  // Keep cache while server is live; wipe automatically only when backend goes offline or restarts
  useEffect(() => {
    if (!cacheHydrated) return;

    if (backendStatus === "offline") {
      clearAllAutoCaches();
      clearConnection();
      setJobs([]);
      setModels(null);
      setSessionEpoch((n) => n + 1);
      return;
    }

    if (backendStatus === "online") {
      const liveBootId = backendHealth?.bootId || "";
      if (liveBootId) {
        const cachedBootId = loadCachedBootId();
        if (!cachedBootId) {
          saveCachedBootId(liveBootId);
        } else if (cachedBootId !== liveBootId) {
          clearAllAutoCaches();
          saveCachedBootId(liveBootId);
          clearConnection();
          setSessionEpoch((n) => n + 1);
        }
      }
      void refreshModels();
      void refreshJobs();
    }
  }, [
    backendHealth?.bootId,
    backendStatus,
    cacheHydrated,
    clearConnection,
    refreshJobs,
    refreshModels,
  ]);

  useEffect(() => {
    if (backendStatus === "online") {
      void refreshModels();
    }
  }, [
    backendHealth?.ollamaReachable,
    backendHealth?.ollamaModels.length,
    backendStatus,
    refreshModels,
  ]);

  return (
    <AutoAnnotateContext.Provider
      value={{
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
        models,
        jobs,
        setJobs,
        refreshJobs,
        refreshModels,
        sessionEpoch,
        clearConnection,
      }}
    >
      {children}
    </AutoAnnotateContext.Provider>
  );
}

export function useAutoAnnotate() {
  const ctx = useContext(AutoAnnotateContext);
  if (!ctx) {
    throw new Error(
      "useAutoAnnotate must be used within an AutoAnnotateProvider."
    );
  }
  return ctx;
}
