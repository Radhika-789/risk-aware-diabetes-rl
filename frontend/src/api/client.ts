import type {
  BranchResult, Job, Meta, PairedResult, ResultSet, Run, SessionState, Settings, TrainRow, ValRow,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init })
  } catch {
    throw new ApiError(0, 'Backend unreachable. Start it with: uvicorn backend.app:app --port 8000')
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const api = {
  meta: () => req<Meta>('/api/meta'),
  health: () => req<{ ok: boolean }>('/api/health'),
  runs: () => req<{ runs_dir: string; runs: Run[] }>('/api/runs'),
  runLogs: (name: string) => req<{ name: string; train: TrainRow[]; val: ValRow[] }>(`/api/runs/${encodeURIComponent(name)}/logs`),
  results: () => req<{ results: ResultSet[] }>('/api/results'),

  sessions: () => req<SessionState[]>('/api/sessions'),
  createSession: (policy: string, settings: Partial<Settings>) =>
    req<SessionState>('/api/sessions', json({ policy, settings })),
  session: (id: string, trace = true) => req<SessionState>(`/api/sessions/${id}?trace=${trace}`),
  step: (id: string, n = 1) => req<SessionState>(`/api/sessions/${id}/step?n=${n}`, { method: 'POST' }),
  run: (id: string, speed?: number) => req<SessionState>(`/api/sessions/${id}/run`, json({ speed })),
  pause: (id: string) => req<SessionState>(`/api/sessions/${id}/pause`, { method: 'POST' }),
  reset: (id: string, seed?: number) => req<SessionState>(`/api/sessions/${id}/reset`, json({ seed })),
  deleteSession: (id: string) => req<{ ok: boolean }>(`/api/sessions/${id}`, { method: 'DELETE' }),
  branch: (id: string, policies: string[]) => req<BranchResult>(`/api/sessions/${id}/branch`, json({ policies })),

  pairedRollouts: (policies: string[], settings: Partial<Settings>) =>
    req<Job<PairedResult>>('/api/rollouts', json({ policies, settings })),
  evaluate: (body: { checkpoints: string[]; split: string; episodes_per_patient: number; out: string; episode_days?: number | null; no_shield?: boolean }) =>
    req<Job>('/api/evaluations', json(body)),
  jobs: () => req<Job[]>('/api/jobs'),
  job: <T = unknown>(id: string) => req<Job<T>>(`/api/jobs/${id}`),
}

/** Poll a job until it finishes. Resolves with the finished job (status done or error). */
export async function waitForJob<T>(id: string, onProgress?: (j: Job<T>) => void, intervalMs = 500): Promise<Job<T>> {
  for (;;) {
    const j = await api.job<T>(id)
    onProgress?.(j)
    if (j.status === 'done' || j.status === 'error') return j
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}
