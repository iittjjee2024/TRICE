/**
 * Typed fetch layer for the TRICE workbench API.
 *
 * Requests go to a relative /api path so the Vite dev proxy keeps the browser
 * same-origin; in a production build the same paths work behind any reverse proxy.
 */

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
      if (Array.isArray(detail)) detail = JSON.stringify(detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, String(detail));
  }
  return (await res.json()) as T;
}

export const get = <T>(path: string) => request<T>(path);
export const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

/* ------------------------------------------------------------------ types ---- */

export interface Health {
  status: string;
  version: string;
  python: string;
  platform: string;
  capabilities: Record<string, boolean>;
  paths: Record<string, string>;
  state: {
    store_built: boolean;
    n_runs: number;
    latest_run: string | null;
    output_written: boolean;
  };
}

export interface RunHeadline {
  val_macro_f05: number | null;
  val_macro_precision: number | null;
  val_macro_recall: number | null;
  mean_k: number | null;
  blocking_recall: number | null;
  stage1_auc: number | null;
  stage2_auc: number | null;
  best_threshold_rule: string | null;
  best_threshold_f05: number | null;
}

export interface RunSummary {
  run_id: string;
  created: string | null;
  countries: string[];
  n_entities: number | null;
  n_val_entities: number | null;
  n_pairs: number | null;
  model_kind: string | null;
  use_stage2: boolean | null;
  headline: RunHeadline;
  has_inference: boolean;
  has_val_pairs: boolean;
}

export interface BlockingChannel {
  channel: string;
  vocab: number;
  nnz: number;
  index_gb: number;
  pairs: number;
  seconds: Record<string, number>;
}

export interface BlockingCountry {
  country: string;
  macro_recall: number;
  micro_recall: number;
  n_true_links: number;
  n_found_links: number;
  entities_full_recall: number;
  entities_zero_recall: number;
  channels: BlockingChannel[];
}

export interface BlockingResponse {
  overall_macro_recall: number | null;
  per_country: BlockingCountry[];
  achieved_precision: number;
  f05_ceiling_curve: { recall: number; f05_ceiling: number }[];
  sweeps: { file: string; data: unknown }[];
}

export interface Importance {
  0: string;
  1: number;
}

export interface FeaturesResponse {
  model: {
    kind: string | null;
    stage1: { auc: number; average_precision: number } | null;
    stage2: { auc: number; average_precision: number } | null;
    use_stage2: boolean | null;
    stage1_parameters: number | null;
    stage2_parameters: number | null;
  };
  importances_stage1: [string, number][];
  importances_stage2: [string, number][];
  separation: {
    feature: string;
    pos_mean: number;
    neg_mean: number;
    auc: number | null;
  }[];
}

export interface CalibrationGroup {
  group: string;
  is_overall: boolean;
  n: number;
  ece: number;
  brier: number;
  has_own_curve: boolean;
  points: { p_pred: number; p_true: number; n: number }[];
}

export interface SimulateRequest {
  rule: "expected_f" | "global_threshold" | "top1" | "topk";
  beta: number;
  prune_epsilon: number;
  max_emit: number;
  global_threshold: number;
  fixed_k: number;
  probability_power: number;
  use_missing_mass: boolean;
  disjointness_repair: boolean;
  use_stage1_probabilities: boolean;
  sample_entities?: number;
}

export interface SimulateResponse {
  n_entities: number;
  sampled: boolean;
  score: {
    macro_f05: number;
    macro_precision: number;
    macro_recall: number;
    mean_k: number;
    predicted_pairs: number;
  };
  by_country: {
    country: string;
    n: number;
    macro_f05: number;
    macro_precision: number;
    macro_recall: number;
    mean_k: number;
  }[];
  verdicts: Record<string, number>;
  k_histogram: Record<string, number>;
  truth_size_histogram: Record<string, number>;
  ev_calibration: {
    bin: number;
    predicted_ef: number;
    realised_f: number;
    n: number;
  }[];
}

export interface EntityRow {
  index: number;
  entity_id: string;
  country: string;
  n_candidates: number;
  k_star: number;
  truth_size: number;
  tp: number;
  expected_f: number;
  realised_f: number;
  verdict: string;
  predicted: string[];
}

export interface EntityListResponse {
  total: number;
  page: number;
  page_size: number;
  rows: EntityRow[];
  filters: string[];
  countries: string[];
}

export interface EntityDetail {
  index: number;
  entity_id: string;
  country: string;
  truth_size: number;
  n_candidates: number;
  k_star: number;
  realised_f: number;
  expected_f: number;
  verdict: string;
  missing_mass: number;
  candidates: {
    entity_id: string;
    source: number;
    p_calibrated: number;
    p_stage1: number;
    block_score: number;
    selected: boolean;
    is_true: boolean;
  }[];
  ev_curve: { k: number; expected_f: number }[];
  ev_curve_k_star: number;
  note: string;
}

export interface ValidationReport {
  ok: boolean;
  issues: string[];
  warnings: string[];
  checks: Record<string, boolean>;
  matching: {
    rows: number;
    nonempty: number;
    empty: number;
    total_ids: number;
    mean_ids_per_nonempty: number;
  } | null;
  candidates: { rows: number; nonempty: number; total_ids: number } | null;
  id_existence_checked?: boolean;
}

export interface OutputStatus {
  files: {
    name: string;
    exists: boolean;
    bytes?: number;
    mb?: number;
    head?: string[];
  }[];
  output_dir: string;
}

export interface AblationRow {
  config: string;
  macro_f05: number | null;
  delta_vs_expected_f: number | null;
  note: string;
}

/* ------------------------------------------------------------------ calls ---- */

export const api = {
  health: () => get<Health>("/health"),
  runs: () => get<{ runs: RunSummary[] }>("/runs"),
  run: (id: string) => get<Record<string, any>>(`/runs/${id}`),
  blocking: (id: string) => get<BlockingResponse>(`/runs/${id}/blocking`),
  features: (id: string) => get<FeaturesResponse>(`/runs/${id}/features`),
  calibration: (id: string) =>
    get<{ groups: CalibrationGroup[] }>(`/runs/${id}/calibration`),
  ablation: (id: string) =>
    get<{ rows: AblationRow[]; all_rules: Record<string, any> }>(
      `/runs/${id}/ablation`,
    ),
  decision: (id: string) => get<Record<string, any>>(`/runs/${id}/decision`),
  simulate: (id: string, body: SimulateRequest) =>
    post<SimulateResponse>(`/runs/${id}/decision/simulate`, body),
  entities: (id: string, params: Record<string, string | number | boolean>) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    );
    return get<EntityListResponse>(`/runs/${id}/entities?${qs}`);
  },
  entity: (id: string, index: number) =>
    get<EntityDetail>(`/runs/${id}/entities/${index}`),
  datasets: () => get<Record<string, any>>("/datasets"),
  datasetCountries: () => get<Record<string, any>>("/datasets/countries"),
  preview: (params: Record<string, string | number>) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    );
    return get<Record<string, any>>(`/datasets/preview?${qs}`);
  },
  normalize: (body: { name: string; address: string }) =>
    post<Record<string, any>>("/datasets/normalize", body),
  variants: () => get<Record<string, any>>("/datasets/variants"),
  output: () => get<OutputStatus>("/output"),
  validateOutput: (checkIds: boolean) =>
    get<ValidationReport>(`/output/validate?check_ids=${checkIds}`),
  inference: (id: string) => get<Record<string, any>>(`/runs/${id}/inference`),
  docsList: () =>
    get<{ docs: { name: string; title: string; bytes: number }[] }>("/docs-md"),
  doc: (name: string) => get<{ name: string; markdown: string }>(`/docs-md/${name}`),
};
