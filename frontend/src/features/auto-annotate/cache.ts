import type {
  AutoAnnotationStatusFilter,
  AutoFetchScope,
  AutoJobDetailResponse,
  AutoJobRecord,
  AutoOutputMode,
  AutoShotMode,
  ClassColorEntry,
  OwlModelId,
  RoboflowInspectResponse,
} from "./types";

const PREFIX_AUTO_CACHE = "c1_aa_";
const KEY_DRAFT = "c1_aa_draft_v1";
const KEY_CONNECTION = "c1_aa_connection_v1";
const KEY_JOBS = "c1_aa_jobs_cache_v1";
const KEY_LAST_VERIFY_JOB = "c1_aa_last_verify_job_v1";
const KEY_BOOT_ID = "c1_aa_boot_id_v1";
const KEY_OWL_MODEL = "c1_aa_owl_model_v1";
const PREFIX_VERIFY_UI = "c1_aa_verify_ui_v1:";
const PREFIX_JOB_DETAIL = "c1_aa_job_detail_v1:";
const PREFIX_VIT_REFS = "c1_aa_vit_refs_v1:";
const PREFIX_VIT_CLASSES = "c1_aa_vit_classes_v1:";

export interface AutoConnectionSessionCache {
  apiKey: string;
  workspaceSlug: string;
  projectSlug: string;
  inspectInfo: RoboflowInspectResponse | null;
  fetchScope: AutoFetchScope;
  annotationStatusFilter: AutoAnnotationStatusFilter;
  sourceBatchId: string;
  sourceAnnotationJobId: string;
  assigneeEmail: string;
  maxImages: number;
  batchSize: number;
}

export interface AutoDraftCache {
  modelId: string;
  samRefinerId: string;
  shotMode: AutoShotMode;
  outputMode: AutoOutputMode;
  prompt: string;
  styleNote: string;
  classColors: ClassColorEntry[];
  confidenceThreshold: number;
  temperature: number;
  maxTokens: number;
}

export interface VerifyUiCache {
  selectedBatchId: string;
  frameFilter: string;
  batchStateFilter: string;
  reasonFilter: string;
  autoAdvance: boolean;
}

function getSessionStore(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage ?? null;
  } catch {
    return null;
  }
}

function getLocalStore(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

function wipePrefixFromStorage(storage: Storage | null, prefix: string): void {
  if (!storage) return;
  try {
    const keys: string[] = [];
    for (let i = 0; i < storage.length; i += 1) {
      const k = storage.key(i);
      if (k && k.startsWith(prefix)) keys.push(k);
    }
    keys.forEach((k) => storage.removeItem(k));
  } catch {
    // Ignore storage errors
  }
}

export function clearLegacyLocalAutoCaches(): void {
  wipePrefixFromStorage(getLocalStore(), PREFIX_AUTO_CACHE);
}

export function clearAllAutoCaches(): void {
  wipePrefixFromStorage(getLocalStore(), PREFIX_AUTO_CACHE);
  wipePrefixFromStorage(getSessionStore(), PREFIX_AUTO_CACHE);
}

function safeReadJson<T>(key: string): T | null {
  const store = getSessionStore();
  if (!store) return null;
  try {
    const raw = store.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function safeWriteJson(key: string, value: unknown): void {
  const store = getSessionStore();
  if (!store) return;
  const serialized = JSON.stringify(value);
  try {
    store.setItem(key, serialized);
  } catch {
    // Prune older cached job details on storage quota pressure, then retry once
    try {
      const detailKeys: string[] = [];
      for (let i = 0; i < store.length; i += 1) {
        const k = store.key(i);
        if (k && k.startsWith(PREFIX_JOB_DETAIL) && k !== key) {
          detailKeys.push(k);
        }
      }
      detailKeys.forEach((k) => store.removeItem(k));
      store.setItem(key, serialized);
    } catch {
      // Ignore unrecoverable quota failures
    }
  }
}

export function loadAutoDraft(): Partial<AutoDraftCache> | null {
  return safeReadJson<Partial<AutoDraftCache>>(KEY_DRAFT);
}

export function saveAutoDraft(draft: AutoDraftCache): void {
  safeWriteJson(KEY_DRAFT, draft);
}

export function clearAutoDraft(): void {
  const store = getSessionStore();
  if (!store) return;
  try {
    store.removeItem(KEY_DRAFT);
  } catch {
    // Ignore
  }
}

export function loadConnectionSession(): Partial<AutoConnectionSessionCache> | null {
  return safeReadJson<Partial<AutoConnectionSessionCache>>(KEY_CONNECTION);
}

export function saveConnectionSession(state: AutoConnectionSessionCache): void {
  safeWriteJson(KEY_CONNECTION, state);
}

export function clearConnectionSession(): void {
  const store = getSessionStore();
  if (!store) return;
  try {
    store.removeItem(KEY_CONNECTION);
  } catch {
    // Ignore
  }
}

export function loadCachedBootId(): string {
  const store = getSessionStore();
  if (!store) return "";
  try {
    return store.getItem(KEY_BOOT_ID) || "";
  } catch {
    return "";
  }
}

export function loadSelectedOwlModelId(): OwlModelId | null {
  const store = getSessionStore();
  if (!store) return null;
  try {
    const val = store.getItem(KEY_OWL_MODEL);
    if (
      val === "google/owlv2-base-patch16-ensemble" ||
      val === "google/owlv2-large-patch14-ensemble"
    ) {
      return val;
    }
    return null;
  } catch {
    return null;
  }
}

export function saveSelectedOwlModelId(modelId: OwlModelId): void {
  const store = getSessionStore();
  if (!store) return;
  try {
    store.setItem(KEY_OWL_MODEL, modelId);
  } catch {
    // Ignore storage errors
  }
}

export function saveCachedBootId(bootId: string): void {
  const store = getSessionStore();
  if (!store) return;
  try {
    if (bootId) store.setItem(KEY_BOOT_ID, bootId);
    else store.removeItem(KEY_BOOT_ID);
  } catch {
    // Ignore
  }
}

export function loadCachedJobs(): AutoJobRecord[] {
  const list = safeReadJson<AutoJobRecord[]>(KEY_JOBS);
  return Array.isArray(list) ? list : [];
}

export function saveCachedJobs(jobs: AutoJobRecord[]): void {
  safeWriteJson(KEY_JOBS, jobs);
}

export function loadLastVerifyJobId(): string {
  const store = getSessionStore();
  if (!store) return "";
  try {
    return store.getItem(KEY_LAST_VERIFY_JOB) || "";
  } catch {
    return "";
  }
}

export function saveLastVerifyJobId(jobId: string): void {
  const store = getSessionStore();
  if (!store) return;
  try {
    if (jobId) store.setItem(KEY_LAST_VERIFY_JOB, jobId);
    else store.removeItem(KEY_LAST_VERIFY_JOB);
  } catch {
    // Ignore storage errors
  }
}

export function loadVerifyUiState(jobId: string): Partial<VerifyUiCache> | null {
  if (!jobId) return null;
  return safeReadJson<Partial<VerifyUiCache>>(`${PREFIX_VERIFY_UI}${jobId}`);
}

export function saveVerifyUiState(jobId: string, state: VerifyUiCache): void {
  if (!jobId) return;
  safeWriteJson(`${PREFIX_VERIFY_UI}${jobId}`, state);
}

export function loadCachedJobDetail(
  jobId: string
): AutoJobDetailResponse | null {
  if (!jobId) return null;
  return safeReadJson<AutoJobDetailResponse>(`${PREFIX_JOB_DETAIL}${jobId}`);
}

export function saveCachedJobDetail(
  jobId: string,
  detail: AutoJobDetailResponse
): void {
  if (!jobId) return;
  safeWriteJson(`${PREFIX_JOB_DETAIL}${jobId}`, detail);
}

export function loadVitRefsForJob(jobId: string): Record<string, string[]> {
  if (!jobId) return {};
  const data = safeReadJson<Record<string, unknown>>(`${PREFIX_VIT_REFS}${jobId}`);
  if (!data || typeof data !== "object" || Array.isArray(data)) return {};
  const cleaned: Record<string, string[]> = {};
  for (const [k, v] of Object.entries(data)) {
    if (Array.isArray(v)) {
      const ids = v.filter((x): x is string => typeof x === "string" && x.length > 0);
      if (ids.length > 0) cleaned[k] = ids;
    }
  }
  return cleaned;
}

export function saveVitRefsForJob(
  jobId: string,
  refs: Record<string, string[]>
): void {
  if (!jobId) return;
  const store = getSessionStore();
  if (!store) return;
  const nonEmpty: Record<string, string[]> = {};
  for (const [k, v] of Object.entries(refs)) {
    if (Array.isArray(v) && v.length > 0) nonEmpty[k] = v;
  }
  if (Object.keys(nonEmpty).length === 0) {
    try {
      store.removeItem(`${PREFIX_VIT_REFS}${jobId}`);
    } catch {
      // Ignore
    }
    return;
  }
  safeWriteJson(`${PREFIX_VIT_REFS}${jobId}`, nonEmpty);
}

export function loadVitTargetClassesForJob(jobId: string): string[] | null {
  if (!jobId) return null;
  const data = safeReadJson<unknown>(`${PREFIX_VIT_CLASSES}${jobId}`);
  if (!Array.isArray(data)) return null;
  return data.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
}

export function saveVitTargetClassesForJob(
  jobId: string,
  classes: string[]
): void {
  if (!jobId) return;
  safeWriteJson(`${PREFIX_VIT_CLASSES}${jobId}`, classes);
}

export function clearCachedJobDetail(jobId: string): void {
  const store = getSessionStore();
  if (!store || !jobId) return;
  try {
    store.removeItem(`${PREFIX_JOB_DETAIL}${jobId}`);
    store.removeItem(`${PREFIX_VERIFY_UI}${jobId}`);
    store.removeItem(`${PREFIX_VIT_REFS}${jobId}`);
    store.removeItem(`${PREFIX_VIT_CLASSES}${jobId}`);
    if (loadLastVerifyJobId() === jobId) {
      saveLastVerifyJobId("");
    }
  } catch {
    // Ignore
  }
}
