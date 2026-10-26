import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { policyLabel } from '../lib/format'

export const useMeta = () => useQuery({ queryKey: ['meta'], queryFn: api.meta, staleTime: Infinity })
export const useRuns = () => useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 15_000 })
export const useResults = () => useQuery({ queryKey: ['results'], queryFn: api.results, refetchInterval: 15_000 })

export interface PolicyOption { id: string; label: string; group: 'Baselines' | 'DQN checkpoints'; demo?: boolean }

/** Baselines + every checkpoint on disk, for pickers. */
export function usePolicyOptions(): { options: PolicyOption[]; isLoading: boolean } {
  const meta = useMeta()
  const runs = useRuns()
  const options: PolicyOption[] = []
  for (const b of meta.data?.baselines ?? []) options.push({ id: b, label: policyLabel(b), group: 'Baselines' })
  for (const r of runs.data?.runs ?? []) for (const c of r.checkpoints) options.push({ id: c.id, label: policyLabel(c.id), group: 'DQN checkpoints', demo: r.demo })
  return { options, isLoading: meta.isLoading || runs.isLoading }
}
