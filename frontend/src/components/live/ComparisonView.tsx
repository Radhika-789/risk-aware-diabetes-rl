/**
 * Baseline vs candidate rollouts.
 *  - "Branch from now": deep-copies the live simulator at its current state and runs each policy forward. The live
 *    history is drawn solid; the branch futures are dashed and labelled counterfactual.
 *  - "Paired full episode": every policy runs the identical episode (same patient, seed, meals, sensor noise) from t=0.
 * Nothing drawn here is predicted by the UI: every point came back from SimGlucose.
 */
import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, waitForJob } from '../../api/client'
import type { BranchResult, PairedResult, SessionState } from '../../api/types'
import { fmt, policyLabel, seriesColor } from '../../lib/format'
import { GlucoseTimeline, type TimelineSeries } from '../charts/GlucoseTimeline'
import { RangeBars } from '../charts/RangeBars'
import { Empty, ErrorState, Loading } from '../ui'
import { PolicySelect } from '../PolicySelect'

type Mode = 'branch' | 'paired'

export function ComparisonView({ s }: { s: SessionState }) {
  const [mode, setMode] = useState<Mode>('branch')
  const [baseline, setBaseline] = useState('cgm_rule')
  const [candidate, setCandidate] = useState(s.policy)
  const [showBG, setShowBG] = useState(false)
  const running = s.status === 'running'

  const branch = useMutation({ mutationFn: () => api.branch(s.id, [baseline, candidate]) })
  const paired = useMutation({
    mutationFn: async () => {
      const j = await api.pairedRollouts([baseline, candidate], s.settings)
      const done = await waitForJob<PairedResult>(j.id)
      if (done.status === 'error') throw new Error(done.error ?? 'rollout failed')
      return done.result!
    },
  })
  const m = mode === 'branch' ? branch : paired
  const stale = mode === 'branch' && branch.data && branch.data.branch_decision !== s.clock.decision

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex rounded-md border border-line-2 overflow-hidden" role="radiogroup" aria-label="Comparison mode">
          {(['branch', 'paired'] as Mode[]).map((x) => (
            <button key={x} role="radio" aria-checked={mode === x} onClick={() => setMode(x)}
              className={`px-3 h-[34px] text-[12.5px] font-medium transition-colors ${mode === x ? 'bg-surface-3 text-fg' : 'bg-bg-1 text-fg-3 hover:text-fg-2'}`}>
              {x === 'branch' ? 'Branch from now' : 'Paired full episode'}
            </button>
          ))}
        </div>
        <label className="min-w-[180px]"><span className="label">Baseline</span><PolicySelect value={baseline} onChange={setBaseline} /></label>
        <label className="min-w-[180px]"><span className="label">Candidate</span><PolicySelect value={candidate} onChange={setCandidate} /></label>
        <button className="btn btn-primary" onClick={() => m.mutate()} disabled={m.isPending || running || baseline === candidate}
          title={running ? 'Pause the live simulation first' : undefined}>
          {m.isPending ? 'Simulating…' : mode === 'branch' ? `Branch at ${fmt.clock(s.clock.elapsed_min)}` : 'Run paired episode'}
        </button>
        <label className="inline-flex items-center gap-1.5 text-[12px] text-fg-3 ml-auto">
          <input type="checkbox" checked={showBG} onChange={(e) => setShowBG(e.target.checked)} /> true BG (sim-only)
        </label>
      </div>
      <p className="text-[11.5px] text-fg-3 -mt-1">
        {mode === 'branch'
          ? 'Copies the simulator at the current state and runs each policy to the end of the episode. Dashed lines are counterfactual futures; the solid line is what actually happened in the live session.'
          : 'Each policy runs the identical episode from t = 0: same patient, seed, meals, sensor noise and initial state. These are counterfactual full-episode rollouts, independent of the live session.'}
        {running && <span className="text-warn"> Pause the live simulation to compare.</span>}
      </p>

      {m.isPending && <Loading label={mode === 'branch' ? 'Running branch rollouts in SimGlucose' : 'Running paired episodes in SimGlucose'} rows={4} />}
      {m.isError && <ErrorState title="Rollout failed" detail={(m.error as Error).message} retry={() => m.mutate()} />}
      {!m.isPending && !m.isError && !m.data && (
        <Empty title="No comparison yet" hint="Choose a baseline and a candidate, then run. Results come back only after the simulator finishes every step." />
      )}
      {mode === 'branch' && branch.data && !branch.isPending && (
        <BranchResultView s={s} r={branch.data} showBG={showBG} stale={!!stale} />
      )}
      {mode === 'paired' && paired.data && !paired.isPending && (
        <PairedResultView r={paired.data} showBG={showBG} />
      )}
    </div>
  )
}

function BranchResultView({ s, r, showBG, stale }: { s: SessionState; r: BranchResult; showBG: boolean; stale: boolean }) {
  const series = useMemo<TimelineSeries[]>(() => {
    const out: TimelineSeries[] = []
    const tr = s.trace
    if (tr) {
      out.push({ id: 'live', label: `live (${policyLabel(s.policy)}) CGM`, color: 'var(--color-fg-2)', t_min: tr.t_min, values: tr.cgm, kind: 'live', width: 2.2 })
      if (showBG) out.push({ id: 'live_bg', label: 'live true BG', color: 'var(--color-fg-4)', t_min: tr.t_min, values: tr.bg_sim_only, kind: 'live', width: 1 })
    }
    r.rollouts.forEach((ro, i) => {
      if (!ro.trace) return
      out.push({ id: `cf_${i}`, label: policyLabel(ro.policy), color: seriesColor(i), t_min: ro.trace.t_min, values: showBG ? ro.trace.bg_sim_only : ro.trace.cgm, kind: 'counterfactual' })
    })
    return out
  }, [s.trace, s.policy, r, showBG])

  return (
    <div className="space-y-4 fade-in">
      {stale && <div className="text-[12px] text-warn">The live session has advanced since this branch was taken at decision {r.branch_decision}. Re-run to branch from the current state.</div>}
      <GlucoseTimeline series={series} totalMin={s.clock.total_min} branchAt={r.branch_t_min} height={300} />
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-fg-3 mb-2">Counterfactual future from {fmt.clock(r.branch_t_min)} (range distribution)</div>
          <RangeBars compact rows={r.rollouts.map((ro, i) => ({ label: policyLabel(ro.policy), color: seriesColor(i), m: ro.segment_metrics ?? {}, sub: ro.error }))} />
        </div>
        <div className="overflow-auto"><table className="table">
          <thead><tr><th>Policy</th><th className="num">Future return</th><th className="num">Mean BG</th><th className="num">LBGI</th><th className="num">HBGI</th><th className="num">Shield</th></tr></thead>
          <tbody>
            {r.rollouts.map((ro, i) => (
              <tr key={ro.policy}>
                <td><span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />{policyLabel(ro.policy)}</span>{ro.error && <div className="text-[11px] text-danger">{ro.error}</div>}</td>
                <td className="num">{fmt.n(ro.future_return, 1)}</td>
                <td className="num">{fmt.n(ro.segment_metrics?.mean_bg, 0)}</td>
                <td className="num">{fmt.n(ro.segment_metrics?.lbgi, 2)}</td>
                <td className="num">{fmt.n(ro.segment_metrics?.hbgi, 2)}</td>
                <td className="num">{ro.decisions ? `${ro.decisions.filter((d) => d.shield_suspend).length}s / ${ro.decisions.filter((d) => d.shield_bolus_blocked).length}b` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </div>
    </div>
  )
}

function PairedResultView({ r, showBG }: { r: PairedResult; showBG: boolean }) {
  const total = r.settings.episode_days * 1440
  const series = useMemo<TimelineSeries[]>(() => r.rollouts.flatMap((ro, i) => ro.trace
    ? [{ id: `p_${i}`, label: policyLabel(ro.policy), color: seriesColor(i), t_min: ro.trace.t_min, values: showBG ? ro.trace.bg_sim_only : ro.trace.cgm, kind: 'historical' as const }]
    : []), [r, showBG])
  const meals = useMemo(() => {
    const tr = r.rollouts.find((x) => x.trace)?.trace
    if (!tr) return []
    return tr.meal_g.flatMap((g, i) => (g > 0 ? [{ t_min: tr.t_min[i], kind: 'meal' as const, label: `${g.toFixed(0)} g` }] : []))
  }, [r])
  return (
    <div className="space-y-4 fade-in">
      <div className="flex flex-wrap gap-2 text-[11.5px] text-fg-3">
        <span className="badge">{r.settings.patient}</span><span className="badge mono">seed {r.settings.seed}</span><span className="badge">{r.settings.episode_days} d</span>
        <span className="badge badge-warn">counterfactual full-episode rollouts</span>
      </div>
      <GlucoseTimeline series={series} totalMin={total} markers={meals} height={300} />
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-fg-3 mb-2">Range distribution (full episode)</div>
          <RangeBars compact rows={r.rollouts.map((ro, i) => ({ label: policyLabel(ro.policy), color: seriesColor(i), m: ro.metrics ?? {}, sub: ro.error }))} />
        </div>
        <div className="overflow-auto"><table className="table">
          <thead><tr><th>Policy</th><th className="num">Return</th><th className="num">Mean BG</th><th className="num">CV</th><th className="num">LBGI</th><th className="num">HBGI</th><th className="num">Insulin</th><th className="num">Shield</th></tr></thead>
          <tbody>
            {r.rollouts.map((ro, i) => (
              <tr key={ro.policy}>
                <td><span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />{policyLabel(ro.policy)}</span></td>
                <td className="num">{fmt.n(ro.metrics?.episode_return, 1)}</td>
                <td className="num">{fmt.n(ro.metrics?.mean_bg, 0)}</td>
                <td className="num">{fmt.pct(ro.metrics?.cv_bg, 0)}</td>
                <td className="num">{fmt.n(ro.metrics?.lbgi, 2)}</td>
                <td className="num">{fmt.n(ro.metrics?.hbgi, 2)}</td>
                <td className="num">{fmt.n(ro.metrics?.total_insulin_u, 1)} U</td>
                <td className="num">{ro.shield ? `${ro.shield.suspends}s / ${ro.shield.bolus_blocks}b` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </div>
    </div>
  )
}
