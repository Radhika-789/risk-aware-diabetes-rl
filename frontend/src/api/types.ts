// Shapes returned by backend/app.py. Field names mirror the Python side on purpose.

export type Split = 'train' | 'val' | 'test' | 'unknown'
export type SessionStatus = 'idle' | 'running' | 'paused' | 'completed' | 'terminated' | 'error'

export interface Meta {
  disclaimer: string
  patients: { train: string[]; val: string[]; test: string[] }
  baselines: string[]
  default_settings: Settings
  checkpoint_locked_fields: string[]
  action_space: { basal_mults: number[]; bolus_fracs_of_tdi: number[] }
  reward: Record<string, number | string>
  shield_defaults: Record<string, number | boolean>
  thresholds: { severe_low: number; low: number; high: number; very_high: number }
  sample_min: number
}

export interface Settings {
  patient: string
  seed: number
  episode_days: number
  action_repeat: number
  history_len: number
  announce_meals: boolean
  reward_mode: 'risk' | 'tir'
  shield: boolean
  suspend_below_cgm: number
  iob_cap_frac_of_tdi: number
}

export interface Trace {
  sample_min: number
  t_min: number[]
  cgm: number[]
  bg_sim_only: number[]
  insulin_u: number[]
  meal_g: number[]
}

export interface Decision {
  i: number
  t_min: number
  sim_time: string
  cgm: number
  bg_sim_only: number
  iob_u: number
  action: number
  req_basal_mult: number
  req_bolus_u: number
  basal_mult: number
  bolus_u: number
  shield_suspend: boolean
  shield_bolus_blocked: boolean
  reward: number
  meal_g: number
  q_values: number[] | null
  wall_ms: number
}

export interface SimEvent {
  i: number
  t_min: number
  kind: 'suspend' | 'bolus_blocked' | 'meal' | 'terminated'
  detail: string
}

export interface EpisodeMetrics {
  tir_70_180: number
  tbr_lt70: number
  tbr_lt54: number
  tar_gt180: number
  tar_gt250: number
  mean_bg: number
  sd_bg: number
  cv_bg: number
  min_bg: number
  max_bg: number
  lbgi: number
  hbgi: number
  terminated_early: number
  n_samples: number
  total_insulin_u?: number
  total_carbs_g?: number
  survival_fraction?: number
  episode_return?: number
  scope?: string
}

export interface SessionState {
  id: string
  status: SessionStatus
  error: string | null
  version: number
  policy: string
  settings: Settings
  locked_fields: string[]
  patient_split: Split
  patient_info: { CR: number; CF: number; TDI: number; BW: number; Age: number; basal_U_per_min: number }
  speed: number
  clock: {
    start: string; now: string; elapsed_min: number; total_min: number
    decision: number; max_decisions: number; decision_min: number
  }
  current: { cgm: number; bg_sim_only: number; iob_u: number; iob_cap_u: number }
  last_decision: Decision | null
  shield_counts: { suspends: number; bolus_blocks: number }
  events: SimEvent[]
  metrics_so_far: Partial<EpisodeMetrics>
  action_space: { n: number; basal_mults: number[]; bolus_fracs_of_tdi: number[]; tdi_u: number; basal_u_per_min: number }
  decisions: Decision[] | null
  trace?: Trace
}

export interface BranchRollout {
  policy: string
  error?: string
  future_return?: number
  decisions?: { t_min: number; action: number; basal_mult: number; bolus_u: number; shield_suspend: boolean; shield_bolus_blocked: boolean }[]
  trace?: Trace
  segment_metrics?: Partial<EpisodeMetrics>
  full_episode_metrics?: EpisodeMetrics
}

export interface BranchResult {
  kind: 'counterfactual_branch'
  branch_decision: number
  branch_t_min: number
  rollouts: BranchRollout[]
}

export interface PairedRollout {
  policy: string
  error?: string
  metrics?: EpisodeMetrics & { patient: string; seed: number; shield_suspend_decisions: number; shield_bolus_blocks: number }
  trace?: Trace
  shield?: { suspends: number; bolus_blocks: number }
}

export interface Job<T = unknown> {
  id: string
  kind: string
  params: Record<string, unknown>
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  message: string
  error: string | null
  created: number
  finished: number | null
  result?: T
}

export interface PairedResult {
  kind: 'counterfactual_full_episode'
  settings: Settings
  locked_fields: string[]
  rollouts: PairedRollout[]
}

export interface Checkpoint {
  id: string
  file: string
  path: string
  obs_dim: number
  n_actions: number
  hidden: number
  n_params: number
  env_config: Partial<Settings> & Record<string, unknown>
  layers: string[]
  size_bytes: number
  modified: number
}

export interface TrainRow {
  episode: number; env_steps: number; patient: string; epsilon: number; return: number
  tir: number; tbr_lt70: number; tbr_lt54: number; tar_gt180: number; mean_bg: number
  terminated: number; mean_loss: number | null; shield_suspends: number; elapsed_s: number
}
export interface ValRow {
  episode: number; val_return: number; val_tir: number; val_tbr_lt70: number; val_tbr_lt54: number
  val_terminated: number; is_best: number
}

export interface Run {
  name: string
  dir: string
  config: { args?: Record<string, unknown>; env_config?: Partial<Settings>; train_patients?: string[]; val_patients?: string[] }
  episodes_planned: number | null
  episodes_done: number
  complete: boolean
  last_train: TrainRow | null
  n_val_evals: number
  selected_checkpoint: ValRow | null
  checkpoints: Checkpoint[]
  demo: boolean
}

export interface SummaryRow {
  policy: string
  n_episodes: number
  [k: string]: number | string
}

export interface EpisodeRow {
  policy: string; patient: string; seed: number
  [k: string]: number | string
}

export interface ResultSet {
  name: string
  dir: string
  split: Split
  meta: Record<string, unknown>
  policies: string[]
  summary: SummaryRow[]
  episodes: EpisodeRow[]
  modified: number
}
