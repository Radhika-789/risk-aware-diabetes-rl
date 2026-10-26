import { useCallback, useMemo, useState } from 'react'
import { useDraft, useSession, store } from '../hooks/useSession'
import { fmt, policyLabel } from '../lib/format'
import { Card, ErrorState, Loading, SplitBadge, StatusBadge, Tabs } from '../components/ui'
import { GlucoseTimeline, type Marker, type TimelineSeries } from '../components/charts/GlucoseTimeline'
import { SetupPanel } from '../components/live/SetupPanel'
import { SimulatorPanel } from '../components/live/SimulatorPanel'
import { Controls } from '../components/live/Controls'
import { ComparisonView } from '../components/live/ComparisonView'
import { DecisionsTable } from '../components/live/DecisionsTable'
import { SidePanel } from '../components/live/SidePanel'

type Tab = 'timeline' | 'comparison' | 'decisions'

export function LiveSimulation() {
  const { id, state: s, query, create, step, run, pause, reset } = useSession()
  const [draft] = useDraft()
  const [tab, setTab] = useState<Tab>('timeline')
  const [showBG, setShowBG] = useState(true)
  const [showPanel, setShowPanel] = useState(true)

  const busy = create.isPending || step.isPending || run.isPending || pause.isPending || reset.isPending
  const onRun = useCallback((speed: number) => run.mutate(speed), [run])
  const onPause = useCallback(() => pause.mutate(), [pause])
  const onStep = useCallback((n: number) => step.mutate(n), [step])
  const onReset = useCallback((seed?: number) => reset.mutate(seed), [reset])
  const onNew = useCallback(() => store.setSessionId(null), [])

  const series = useMemo<TimelineSeries[]>(() => {
    if (!s?.trace) return []
    const out: TimelineSeries[] = [{ id: 'cgm', label: 'CGM (observed)', color: 'var(--color-accent)', t_min: s.trace.t_min, values: s.trace.cgm, kind: 'live', width: 2.2 }]
    if (showBG) out.push({ id: 'bg', label: 'true BG (simulator-only)', color: 'var(--color-fg-3)', t_min: s.trace.t_min, values: s.trace.bg_sim_only, kind: 'live', width: 1.2 })
    return out
  }, [s?.trace, showBG])
  const markers = useMemo<Marker[]>(() => (s?.events ?? []).map((e) => ({ t_min: e.t_min, kind: e.kind, label: e.detail })), [s?.events])

  if (!id) {
    return (
      <div className="p-4 md:p-6 max-w-6xl mx-auto fade-in">
        <SetupPanel onStart={() => create.mutate(draft)} busy={create.isPending} error={create.error?.message} />
      </div>
    )
  }
  if (query.isLoading || !s) return <div className="p-6"><Card><Loading label="Attaching to session" /></Card></div>
  if (query.isError) return <div className="p-6"><ErrorState title="Could not load the session" detail={query.error.message} retry={() => query.refetch()} /></div>

  const actionErr = (step.error ?? run.error ?? pause.error ?? reset.error) as Error | null

  return (
    <div className="p-4 md:p-6 space-y-4 max-w-[1600px] mx-auto">
      {/* header strip */}
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <StatusBadge status={s.status} />
        <span className="badge badge-accent" title="Policy driving this session">{policyLabel(s.policy)}</span>
        <span className="badge mono">{s.settings.patient}</span>
        <SplitBadge split={s.patient_split} />
        <span className="badge mono">seed {s.settings.seed}</span>
        <span className="badge">{s.settings.episode_days} d · {s.clock.decision_min}-min decisions</span>
        {!s.settings.shield && <span className="badge badge-warn">shield off</span>}
        <span className="ml-auto text-fg-3 hidden md:inline">
          <kbd className="kbd">Space</kbd> run/pause · <kbd className="kbd">N</kbd> step · <kbd className="kbd">R</kbd> reset · <kbd className="kbd">[</kbd> sidebar
        </span>
        <button className="btn btn-ghost btn-sm" onClick={() => setShowPanel((v) => !v)} aria-pressed={showPanel}>{showPanel ? 'Hide details' : 'Show details'}</button>
      </div>

      {s.status === 'error' && <ErrorState title="Simulator error" detail={s.error ?? undefined} retry={() => onReset()} />}
      {actionErr && <div role="alert" className="text-[12px] text-danger border border-danger/30 rounded-md px-3 py-2 bg-danger/5">{actionErr.message}</div>}

      <div className={`grid gap-4 ${showPanel ? 'xl:grid-cols-[minmax(0,1fr)_340px]' : ''}`}>
        <div className="space-y-4 min-w-0">
          <Card title="SimGlucose virtual patient" subtitle="Every value below was produced by a completed simulator step. The display never extrapolates."
            actions={s.status === 'running' && <span className="text-[11px] text-fg-3 inline-flex items-center gap-1.5"><span className="live-dot" />{s.speed}×/s</span>}>
            <SimulatorPanel s={s} />
            <div className="mt-5 pt-4 border-t border-line">
              <Controls s={s} busy={busy} onRun={onRun} onPause={onPause} onStep={onStep} onReset={onReset} onNew={onNew} />
              {(s.status === 'completed' || s.status === 'terminated') && (
                <div className={`mt-3 text-[12px] rounded-md px-3 py-2 border ${s.status === 'terminated' ? 'border-danger/30 bg-danger/5 text-fg-2' : 'border-ok/30 bg-ok/5 text-fg-2'}`}>
                  {s.status === 'terminated'
                    ? 'The simulator ended this episode early (true BG left its physiological bounds). Remaining time counts as out of range in the metrics. Reset to rerun the same seed, or start a new session.'
                    : `Episode complete: ${fmt.hours(s.clock.total_min)} simulated in ${s.clock.decision} decisions. Open the Comparison tab for paired rollouts, or Analytics Lab for the full chart workspace.`}
                </div>
              )}
            </div>
          </Card>

          <Card title="Glucose timeline" subtitle="3-minute samples. Bands mark 54, 70–180 and 250 mg/dL."
            actions={
              <label className="inline-flex items-center gap-1.5 text-[12px] text-fg-3"><input type="checkbox" checked={showBG} onChange={(e) => setShowBG(e.target.checked)} /> true BG</label>
            }>
            <Tabs aria-label="Timeline views" value={tab} onChange={setTab} tabs={[
              { id: 'timeline', label: 'Live timeline' },
              { id: 'comparison', label: 'Baseline vs candidate' },
              { id: 'decisions', label: `Decisions (${s.clock.decision})` },
            ]} />
            <div className="pt-4">
              {tab === 'timeline' && (
                s.clock.decision === 0
                  ? <div className="relative">
                      <GlucoseTimeline series={series} totalMin={s.clock.total_min} markers={markers} height={340} />
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <span className="badge">No steps yet. Press Run or Step to begin.</span>
                      </div>
                    </div>
                  : <GlucoseTimeline series={series} totalMin={s.clock.total_min} markers={markers} height={340} />
              )}
              {tab === 'comparison' && <ComparisonView s={s} />}
              {tab === 'decisions' && <DecisionsTable s={s} />}
            </div>
          </Card>
        </div>
        {showPanel && <aside className="min-w-0"><SidePanel s={s} /></aside>}
      </div>
    </div>
  )
}
