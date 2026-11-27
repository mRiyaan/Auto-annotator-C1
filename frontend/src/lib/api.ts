import type {
  BackendHealthInfo,
  BackendModelStatus,
  OllamaControlResponse,
  RunProgressEvent,
  RunRecord,
} from "./types";

export function resolveBackendFileUrl(
  baseUrl: string,
  maybeRelative: string | undefined
): string | undefined {
  if (!maybeRelative) return maybeRelative;
  if (
    maybeRelative.startsWith("http://") ||
    maybeRelative.startsWith("https://") ||
    maybeRelative.startsWith("data:") ||
    maybeRelative.startsWith("blob:")
  ) {
    return maybeRelative;
  }
  const cleanBase = baseUrl.trim().replace(/\/+$/, "");
  const cleanPath = maybeRelative.startsWith("/")
    ? maybeRelative
    : `/${maybeRelative}`;
  return `${cleanBase}${cleanPath}`;
}

export function normalizeRunRecordUrls(
  run: RunRecord,
  baseUrl: string
): RunRecord {
  return {
    ...run,
    target: {
      ...run.target,
      previewUrl:
        resolveBackendFileUrl(baseUrl, run.target.previewUrl) ??
        run.target.previewUrl,
    },
    references: run.references.map((r) => ({
      ...r,
      imageUrl: resolveBackendFileUrl(baseUrl, r.imageUrl) ?? r.imageUrl,
    })),
    videoKeyframes: run.videoKeyframes?.map((kf) => ({
      ...kf,
      imageUrl: resolveBackendFileUrl(baseUrl, kf.imageUrl) ?? kf.imageUrl,
    })),
    annotatedVideoUrl: resolveBackendFileUrl(baseUrl, run.annotatedVideoUrl),
  };
}

export async function fetchHealth(
  baseUrl: string,
  signal?: AbortSignal
): Promise<BackendHealthInfo> {
  const res = await fetch(`${baseUrl}/api/v1/health`, {
    method: "GET",
    signal,
  });
  if (!res.ok) {
    throw new Error(`Health check failed (HTTP ${res.status})`);
  }
  return (await res.json()) as BackendHealthInfo;
}

export async function fetchBackendModels(
  baseUrl: string,
  signal?: AbortSignal
): Promise<BackendModelStatus[]> {
  const res = await fetch(`${baseUrl}/api/v1/models`, {
    method: "GET",
    signal,
  });
  if (!res.ok) {
    throw new Error(`Model status fetch failed (HTTP ${res.status})`);
  }
  return (await res.json()) as BackendModelStatus[];
}

export async function controlOllamaServer(
  baseUrl: string,
  action: "start" | "stop"
): Promise<OllamaControlResponse> {
  const res = await fetch(`${baseUrl}/api/v1/ollama/${action}`, {
    method: "POST",
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await res.json()) as OllamaControlResponse;
}

export async function fetchRuns(baseUrl: string): Promise<RunRecord[]> {
  const res = await fetch(`${baseUrl}/api/v1/runs`, { method: "GET" });
  if (!res.ok) {
    throw new Error(`Failed to fetch runs (HTTP ${res.status})`);
  }
  const data = (await res.json()) as RunRecord[];
  return data.map((r) => normalizeRunRecordUrls(r, baseUrl));
}

export async function fetchRun(
  baseUrl: string,
  runId: string
): Promise<RunRecord> {
  const res = await fetch(
    `${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}`,
    { method: "GET" }
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch run ${runId} (HTTP ${res.status})`);
  }
  const data = (await res.json()) as RunRecord;
  return normalizeRunRecordUrls(data, baseUrl);
}

export async function submitRun(
  baseUrl: string,
  formData: FormData
): Promise<{ runId: string; status: string; record: RunRecord }> {
  const res = await fetch(`${baseUrl}/api/v1/runs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      // Ignore parse failure
    }
    throw new Error(detail);
  }
  const payload = (await res.json()) as {
    runId: string;
    status: string;
    record: RunRecord;
  };
  return {
    ...payload,
    record: normalizeRunRecordUrls(payload.record, baseUrl),
  };
}

export async function deleteRunApi(
  baseUrl: string,
  runId: string
): Promise<void> {
  const res = await fetch(
    `${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE" }
  );
  if (!res.ok && res.status !== 404) {
    throw new Error(`Failed to delete run ${runId} (HTTP ${res.status})`);
  }
}

export async function reencodeRunApi(
  baseUrl: string,
  runId: string,
  outputFpsMode: "original" | "sampled",
  byteTrackMemorySec?: number
): Promise<RunRecord> {
  const res = await fetch(
    `${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}/reencode`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        outputFpsMode,
        ...(byteTrackMemorySec !== undefined ? { byteTrackMemorySec } : {}),
      }),
    }
  );
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      // Ignore
    }
    throw new Error(detail);
  }
  const data = (await res.json()) as RunRecord;
  return normalizeRunRecordUrls(data, baseUrl);
}

export function streamRunProgress(
  baseUrl: string,
  runId: string,
  onEvent: (evt: RunProgressEvent) => void,
  onStreamError: (errMsg: string) => void
): () => void {
  let closed = false;
  const es = new EventSource(
    `${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}/stream`
  );

  es.onmessage = (message) => {
    if (closed) return;
    try {
      const parsed = JSON.parse(message.data) as RunProgressEvent;
      if (parsed.record) {
        parsed.record = normalizeRunRecordUrls(parsed.record, baseUrl);
      }
      onEvent(parsed);
      if (parsed.status === "completed" || parsed.status === "failed") {
        closed = true;
        es.close();
      }
    } catch {
      // Ignore malformed keep-alive comments
    }
  };

  es.onerror = async () => {
    if (closed) return;
    es.close();
    // Fallback check in case SSE disconnected right around completion
    try {
      const latest = await fetchRun(baseUrl, runId);
      if (latest.status === "completed" || latest.status === "failed") {
        closed = true;
        onEvent({
          runId,
          status: latest.status,
          progressPct: 100,
          stage: latest.status === "completed" ? "Completed" : "Execution Failed",
          subtext: latest.errorMessage || latest.summaryText,
          record: latest,
        });
        return;
      }
    } catch {
      // Fall through to error handler
    }
    if (!closed) {
      closed = true;
      onStreamError("Lost connection to live run stream.");
    }
  };

  return () => {
    closed = true;
    es.close();
  };
}
