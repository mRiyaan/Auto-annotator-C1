import type {
  AutoDetection,
  AutoItemRecord,
  AutoJobDetailResponse,
  AutoJobProgressEvent,
  AutoJobRecord,
  AutoModelCatalogResponse,
  AutoRejectReason,
  AutoReviewStatus,
  ManualFindObjectsRequest,
  ManualFindObjectsResponse,
  ManualSmartPrepareResponse,
  ManualSmartSelectRequest,
  ManualSmartSelectResponse,
  PushAcceptedResponse,
  PushDestination,
  RoboflowInspectResponse,
  VisualEmbedAddRequest,
  VisualEmbedBatchEvent,
  VisualEmbedBatchSearchRequest,
  VisualEmbedBuildRefsRequest,
  VisualEmbedBuildRefsResponse,
  VisualEmbedResponse,
  VisualEmbedSearchRequest,
  VisualEmbedUndoRedoRequest,
} from "./types";

function resolveAutoUrl(baseUrl: string, maybeRelative: string): string {
  if (
    !maybeRelative ||
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

function normalizeItemUrls(
  item: AutoItemRecord,
  baseUrl: string
): AutoItemRecord {
  return {
    ...item,
    imageUrl: resolveAutoUrl(baseUrl, item.imageUrl),
  };
}

export class AutoAnnotateApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "AutoAnnotateApiError";
    this.status = status;
  }
}

async function throwResponseError(res: Response): Promise<never> {
  let detail = `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as { detail?: string };
    if (body?.detail) detail = body.detail;
  } catch {
    // ignore JSON error
  }
  throw new AutoAnnotateApiError(detail, res.status);
}

export async function fetchAutoModels(
  baseUrl: string
): Promise<AutoModelCatalogResponse> {
  const res = await fetch(`${baseUrl}/api/v1/auto-annotate/models`, {
    method: "GET",
  });
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as AutoModelCatalogResponse;
}

export async function inspectRoboflowProject(
  baseUrl: string,
  payload: { apiKey: string; projectSlug: string; workspaceSlug?: string }
): Promise<RoboflowInspectResponse> {
  const res = await fetch(`${baseUrl}/api/v1/auto-annotate/roboflow/inspect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as RoboflowInspectResponse;
}

export async function submitAutoJob(
  baseUrl: string,
  formData: FormData
): Promise<AutoJobRecord> {
  const res = await fetch(`${baseUrl}/api/v1/auto-annotate/jobs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as AutoJobRecord;
}

export async function submitManualFetchJob(
  baseUrl: string,
  payload: {
    apiKey: string;
    projectSlug: string;
    workspaceSlug?: string;
    fetchScope: string;
    annotationStatusFilter?: string;
    sourceBatchId?: string;
    sourceAnnotationJobId?: string;
    assigneeEmail?: string;
    maxImages: number;
    batchSize: number;
  }
): Promise<AutoJobRecord> {
  const res = await fetch(`${baseUrl}/api/v1/auto-annotate/jobs/fetch-only`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as AutoJobRecord;
}

export async function fetchAutoJobs(baseUrl: string): Promise<AutoJobRecord[]> {
  const res = await fetch(`${baseUrl}/api/v1/auto-annotate/jobs`, {
    method: "GET",
  });
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as AutoJobRecord[];
}

export async function fetchAutoJobDetail(
  baseUrl: string,
  jobId: string,
  filters?: {
    batchId?: string;
    reviewStatus?: string;
    rejectReason?: string;
    detectionFilter?: string;
  }
): Promise<AutoJobDetailResponse> {
  const params = new URLSearchParams();
  if (filters?.batchId && filters.batchId !== "all") {
    params.set("batch_id", filters.batchId);
  }
  if (filters?.reviewStatus && filters.reviewStatus !== "all") {
    params.set("review_status", filters.reviewStatus);
  }
  if (filters?.rejectReason && filters.rejectReason !== "all") {
    params.set("reject_reason", filters.rejectReason);
  }
  if (filters?.detectionFilter && filters.detectionFilter !== "all") {
    params.set("detection_filter", filters.detectionFilter);
  }
  const qs = params.toString();
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(jobId)}${
      qs ? `?${qs}` : ""
    }`,
    { method: "GET" }
  );
  if (!res.ok) await throwResponseError(res);
  const data = (await res.json()) as AutoJobDetailResponse;
  return {
    job: data.job,
    items: data.items.map((it) => normalizeItemUrls(it, baseUrl)),
  };
}

export async function controlAutoJob(
  baseUrl: string,
  jobId: string,
  action: "start" | "pause" | "cancel"
): Promise<AutoJobRecord> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(jobId)}/${action}`,
    { method: "POST" }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as AutoJobRecord;
}

export async function deleteAutoJobApi(
  baseUrl: string,
  jobId: string
): Promise<void> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" }
  );
  if (!res.ok && res.status !== 404) await throwResponseError(res);
}

export async function patchAutoItemReview(
  baseUrl: string,
  jobId: string,
  itemId: string,
  payload: {
    reviewStatus: AutoReviewStatus;
    rejectReason?: AutoRejectReason | null;
    detections?: AutoDetection[];
  }
): Promise<{ job: AutoJobRecord; item: AutoItemRecord }> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!res.ok) await throwResponseError(res);
  const data = (await res.json()) as {
    job: AutoJobRecord;
    item: AutoItemRecord;
  };
  return {
    job: data.job,
    item: normalizeItemUrls(data.item, baseUrl),
  };
}

export async function approveAutoBatch(
  baseUrl: string,
  jobId: string,
  batchId: string,
  acceptPending = true
): Promise<{ job: AutoJobRecord; batchItems: AutoItemRecord[] }> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/batches/${encodeURIComponent(batchId)}/approve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ acceptPending }),
    }
  );
  if (!res.ok) await throwResponseError(res);
  const data = (await res.json()) as {
    job: AutoJobRecord;
    batchItems: AutoItemRecord[];
  };
  return {
    job: data.job,
    batchItems: data.batchItems.map((it) => normalizeItemUrls(it, baseUrl)),
  };
}

export async function pushAcceptedZipApi(
  baseUrl: string,
  jobId: string,
  payload: {
    apiKey: string;
    workspaceSlug?: string;
    pushDestination: PushDestination;
    targetProjectName?: string;
    targetFolderName?: string;
    targetAnnotationJobId?: string;
    addToDataset?: boolean;
    projectLicense: string;
    batchId?: string;
    includeEmpty?: boolean;
    onlyUnpushed?: boolean;
    batchTag?: string;
  }
): Promise<PushAcceptedResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(jobId)}/push`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as PushAcceptedResponse;
}

export async function prepareSmartSelectItemApi(
  baseUrl: string,
  jobId: string,
  itemId: string,
  samRefinerId: string,
  signal?: AbortSignal
): Promise<ManualSmartPrepareResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}/smart-select/prepare`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samRefinerId }),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as ManualSmartPrepareResponse;
}

export async function smartSelectItemApi(
  baseUrl: string,
  jobId: string,
  itemId: string,
  payload: ManualSmartSelectRequest,
  signal?: AbortSignal
): Promise<ManualSmartSelectResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}/smart-select`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as ManualSmartSelectResponse;
}

export async function findObjectsItemApi(
  baseUrl: string,
  jobId: string,
  itemId: string,
  payload: ManualFindObjectsRequest
): Promise<ManualFindObjectsResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}/find-objects`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as ManualFindObjectsResponse;
}

export async function visualEmbedAddApi(
  baseUrl: string,
  jobId: string,
  itemId: string,
  payload: VisualEmbedAddRequest,
  signal?: AbortSignal
): Promise<VisualEmbedResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}/visual-embed/add`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as VisualEmbedResponse;
}

export async function visualEmbedSearchApi(
  baseUrl: string,
  jobId: string,
  itemId: string,
  payload: VisualEmbedSearchRequest,
  signal?: AbortSignal
): Promise<VisualEmbedResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/items/${encodeURIComponent(itemId)}/visual-embed/search`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as VisualEmbedResponse;
}

export async function visualEmbedResetApi(
  baseUrl: string,
  jobId: string,
  queryId?: string | null
): Promise<void> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/visual-embed/reset`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(queryId ? { queryId } : {}),
    }
  );
  if (!res.ok && res.status !== 404) await throwResponseError(res);
}

export async function visualEmbedBuildRefsApi(
  baseUrl: string,
  jobId: string,
  payload: VisualEmbedBuildRefsRequest,
  signal?: AbortSignal
): Promise<VisualEmbedBuildRefsResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/visual-embed/build-references`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as VisualEmbedBuildRefsResponse;
}

export async function visualEmbedUndoRedoApi(
  baseUrl: string,
  jobId: string,
  payload: VisualEmbedUndoRedoRequest,
  signal?: AbortSignal
): Promise<VisualEmbedResponse> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/visual-embed/undo-redo`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  return (await res.json()) as VisualEmbedResponse;
}

export async function visualEmbedSearchBatchStreamApi(
  baseUrl: string,
  jobId: string,
  payload: VisualEmbedBatchSearchRequest,
  onEvent: (evt: VisualEmbedBatchEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
      jobId
    )}/visual-embed/search-batch`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }
  );
  if (!res.ok) await throwResponseError(res);
  if (!res.body) {
    throw new Error("Empty streaming response from batch visual search.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nlIdx = buffer.indexOf("\n");
    while (nlIdx !== -1) {
      const line = buffer.slice(0, nlIdx).trim();
      buffer = buffer.slice(nlIdx + 1);
      if (line) {
        const parsed = JSON.parse(line) as VisualEmbedBatchEvent;
        if (parsed.item) {
          parsed.item = normalizeItemUrls(parsed.item, baseUrl);
        }
        onEvent(parsed);
      }
      nlIdx = buffer.indexOf("\n");
    }
  }

  const rem = buffer.trim();
  if (rem) {
    const parsed = JSON.parse(rem) as VisualEmbedBatchEvent;
    if (parsed.item) {
      parsed.item = normalizeItemUrls(parsed.item, baseUrl);
    }
    onEvent(parsed);
  }
}

export function getDownloadAcceptedZipUrl(
  baseUrl: string,
  jobId: string,
  batchId?: string,
  includeEmpty = true
): string {
  const params = new URLSearchParams();
  if (batchId && batchId !== "all") params.set("batch_id", batchId);
  if (!includeEmpty) params.set("include_empty", "false");
  const qs = params.toString();
  return `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(
    jobId
  )}/download-zip${qs ? `?${qs}` : ""}`;
}

export function streamAutoJobProgress(
  baseUrl: string,
  jobId: string,
  onEvent: (evt: AutoJobProgressEvent) => void,
  onError: (errMsg: string) => void
): () => void {
  let closed = false;
  const es = new EventSource(
    `${baseUrl}/api/v1/auto-annotate/jobs/${encodeURIComponent(jobId)}/stream`
  );

  es.onmessage = (msg) => {
    if (closed) return;
    try {
      const parsed = JSON.parse(msg.data) as AutoJobProgressEvent;
      if (parsed.latestItem) {
        parsed.latestItem = normalizeItemUrls(parsed.latestItem, baseUrl);
      }
      onEvent(parsed);
      if (
        parsed.status === "completed" ||
        parsed.status === "cancelled" ||
        parsed.status === "failed"
      ) {
        closed = true;
        es.close();
      }
    } catch {
      // ignore keep-alive
    }
  };

  es.onerror = async () => {
    if (closed) return;
    es.close();
    try {
      const detail = await fetchAutoJobDetail(baseUrl, jobId);
      if (
        detail.job.status === "completed" ||
        detail.job.status === "cancelled" ||
        detail.job.status === "failed"
      ) {
        closed = true;
        onEvent({
          jobId,
          status: detail.job.status,
          progressPct: 100,
          stage: detail.job.status,
          subtext: detail.job.errorMessage || "Updated",
          job: detail.job,
        });
        return;
      }
    } catch {
      // ignore fallback error
    }
    if (!closed) {
      closed = true;
      onError("Lost connection to auto-annotate progress stream.");
    }
  };

  return () => {
    closed = true;
    es.close();
  };
}
