/**
 * Live-session store shared across pages.
 *  - draft (policy + editable settings) persists in localStorage and is what "New session" sends.
 *  - sessionId persists so a reload (or switching pages) re-attaches to the backend session.
 *  - state polling: fast while running, slow otherwise. The UI only ever displays what the backend returned;
 *    no interpolation between polls.
 */
import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ApiError } from '../api/client'
import type { SessionState, Settings } from '../api/types'

interface Draft { policy: string; settings: Partial<Settings> }

const LS_DRAFT = 'gg.draft'
const LS_SID = 'gg.sessionId'

let draft: Draft = (() => {
  try { return JSON.parse(localStorage.getItem(LS_DRAFT) ?? '') as Draft } catch { return { policy: 'cgm_rule', settings: {} } }
})()
let sessionId: string | null = localStorage.getItem(LS_SID)
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())

export const store = {
  getDraft: () => draft,
  setDraft: (patch: Partial<Draft>) => {
    draft = { ...draft, ...patch, settings: { ...draft.settings, ...(patch.settings ?? {}) } }
    localStorage.setItem(LS_DRAFT, JSON.stringify(draft))
    emit()
  },
  getSessionId: () => sessionId,
  setSessionId: (id: string | null) => {
    sessionId = id
    if (id) localStorage.setItem(LS_SID, id); else localStorage.removeItem(LS_SID)
    emit()
  },
  subscribe: (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } },
}

export function useDraft() {
  const d = useSyncExternalStore(store.subscribe, store.getDraft)
  return [d, store.setDraft] as const
}

export function useSessionId() {
  return useSyncExternalStore(store.subscribe, store.getSessionId)
}

export function useSession() {
  const id = useSessionId()
  const qc = useQueryClient()

  const q = useQuery<SessionState, ApiError>({
    queryKey: ['session', id],
    queryFn: () => api.session(id!),
    enabled: !!id,
    retry: 0,
    staleTime: 0,
    refetchInterval: (query) => {
      const s = query.state.data
      if (!s) return 2000
      return s.status === 'running' ? 250 : 2500
    },
  })

  // A stale id (server restarted) -> drop it so the page shows the "no session" state.
  useEffect(() => {
    if (q.error && q.error.status === 404 && id) store.setSessionId(null)
  }, [q.error, id])

  const setState = useCallback((s: SessionState) => qc.setQueryData(['session', s.id], s), [qc])

  const create = useMutation({
    mutationFn: async (d: Draft) => {
      if (sessionId) { try { await api.deleteSession(sessionId) } catch { /* ignore */ } }
      return api.createSession(d.policy, d.settings)
    },
    onSuccess: (s) => { store.setSessionId(s.id); setState(s) },
  })
  const step = useMutation({ mutationFn: (n: number) => api.step(id!, n), onSuccess: setState })
  const run = useMutation({ mutationFn: (speed?: number) => api.run(id!, speed), onSuccess: () => q.refetch() })
  const pause = useMutation({ mutationFn: () => api.pause(id!), onSuccess: () => q.refetch() })
  const reset = useMutation({ mutationFn: (seed?: number) => api.reset(id!, seed), onSuccess: setState })

  return { id, state: q.data, query: q, create, step, run, pause, reset }
}
