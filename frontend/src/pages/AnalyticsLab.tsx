import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQueries } from '@tanstack/react-query'
import { api, waitForJob } from '../api/client'
import type { PairedResult, ResultSet, Settings } from '../api/types'
import { useMeta, useResults, useRuns } from '../hooks/useCatalog'
import { useDraft, useSession } from '../hooks/useSession'
import { fmt, policyLabel, seriesColor } from '../lib/format'
import { Card, Empty, ErrorState, Loading, SplitBadge, Tabs } from '../components/ui'
import { GlucoseTimeline, type TimelineSeries } from '../components/charts/GlucoseTimeline'
import { RangeBars } from '../components/charts/RangeBars'
import { LineChart, type XYSeries } from '../components/charts/LineChart'
import { PolicySelect } from '../components/PolicySelect'
import { PatientSelect } from '../components/PatientSelect'

type Tab = 'trajectory' | 'ranges' | 'training' | 'validation'

export function AnalyticsLab() {
  const [tab, setTab] = useState<Tab>('trajectory')
  return (
    <div className="p-4 md:p-6 space-y-4 max-w-[1600px] mx-auto">
      <Tabs aria-label="Analytics sections" value={tab} onChange={setTab} tabs={[
        { id: 'trajectory', label: 'Glucose trajectory' },
        { id: 'ranges', label: 'Time in range' },
        { id: 'training', label: 'Training' },
        { id: 'validation', label: 'Validation progress' },
      ]} />
      <div className="fade-in" key={tab}>
        {tab === 'trajectory' && <TrajectoryTab />}
        {tab === 'ranges' && <RangesTab />}
        {tab === 'training' && <TrainingTab />}
        {tab === 'validation' && <ValidationTab />}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ trajectory: live session or paired rollouts */
function TrajectoryTab() {
  const { state: s } = useSession()
  const [draft] = useDraft()
  const meta = useMeta()
  const [source, setSource] = useState<'live' | 'paired'>(s ? 'live' : 'paired')
  const [policies, setPolicies] = useState<string[]>(['cgm_rule', 'fixed_basal'])
  const [cfg, setCfg] = useState<Partial<Settings>>({})
  const [signal, setSignal] = useState<'cgm' | 'bg'>('bg')
  const settings = { ...(meta.data?.default_settings ?? {}), ...draft.settings, ...cfg } as Settings

  const paired = useMutation({
    mutationFn: async () => {
      const j = await api.pairedRollouts(policies, settings)
      const done = await waitForJob<PairedResult>(j.id)
      if (done.status === 'error') throw new Error(done.error ?? 'failed')
      return done.result!
    },
  })

  const liveSeries = useMemo<TimelineSeries[]>(() => {
    if (!s?.trace) return []
    return [
      { id: 'cgm', label: 'CGM (observed)', color: 'var(--color-accent)', t_min: s.trace.t_min, values: s.trace.cgm, kind: 'live', width: 2 },
      { id: 'bg', label: 'true BG (simulator-only)', color: 'var(--color-fg-3)', t_min: s.trace.t_min, values: s.trace.bg_sim_only, kind: 'live', width: 1.2 },
    ]
  }, [s?.trace])

  const pairedSeries = useMemo<TimelineSeries[]>(() => (paired.data?.rollouts ?? []).flatMap((r, i) => r.trace
    ? [{ id: `p${i}`, label: policyLabel(r.policy), color: seriesColor(i), t_min: r.trace.t_min, values: signal === 'bg' ? r.trace.bg_sim_only : r.trace.cgm, kind: 'historical' as const }] : []), [paired.data, signal])
  const meals = useMemo(() => {
    const tr = paired.data?.rollouts.find((r) => r.trace)?.trace
    return tr ? tr.meal_g.flatMap((g, i) => (g > 0 ? [{ t_min: tr.t_min[i], kind: 'meal' as const, label: `${g.toFixed(0)} g` }] : [])) : []
  }, [paired.data])

  return (
    <div className="space-y-4">
      <Card title="Glucose trajectory and target range" subtitle="True glucose is simulator-only; CGM is what the policy observes. 70–180 mg/dL band, 54 and 250 thresholds."
        actions={
          <div className="flex rounded-md border border-line-2 overflow-hidden" role="radiogroup" aria-label="Data source">
            {(['live', 'paired'] as const).map((x) => (
              <button key={x} role="radio" aria-checked={source === x} onClick={() => setSource(x)} className={`px-3 h-7 text-[12px] font-medium ${source === x ? 'bg-surface-3 text-fg' : 'bg-bg-1 text-fg-3 hover:text-fg-2'}`}>
                {x === 'live' ? 'Live session' : 'Paired rollouts'}
              </button>
            ))}
          </div>
        }>
        {source === 'live' ? (
          !s ? <Empty title="No live session" hint="Start one on the Live Simulation page; its trace appears here as it advances." />
          : s.clock.decision === 0 ? <Empty title="Session has not stepped yet" hint="Run or step the live simulation to populate this chart." />
          : (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2 text-[11.5px]"><span className="badge badge-accent">{policyLabel(s.policy)}</span><span className="badge mono">{s.settings.patient}</span><SplitBadge split={s.patient_split} /><span className="badge mono">seed {s.settings.seed}</span><span className="badge">{s.status}</span></div>
              <GlucoseTimeline series={liveSeries} totalMin={s.clock.total_min} height={420}
                markers={s.events.map((e) => ({ t_min: e.t_min, kind: e.kind, label: e.detail }))} />
              <RangeBars compact rows={[{ label: 'live session so far', m: s.metrics_so_far }]} />
            </div>
          )
        ) : (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[2fr_1fr_1fr_1fr_auto] items-end">
              <div className="space-y-1.5">
                <span className="label">Policies (same episode for all)</span>
                {policies.map((p, i) => (
                  <div key={i} className="flex gap-1.5">
                    <span className="w-2 self-center h-2 rounded-sm shrink-0" style={{ background: seriesColor(i) }} />
                    <PolicySelect value={p} onChange={(v) => setPolicies(policies.map((x, j) => (j === i ? v : x)))} />
                    <button className="btn btn-icon btn-ghost" onClick={() => setPolicies(policies.filter((_, j) => j !== i))} disabled={policies.length <= 1} aria-label="Remove policy">×</button>
                  </div>
                ))}
                {policies.length < 4 && <button className="btn btn-sm btn-ghost" onClick={() => setPolicies([...policies, 'random'])}>+ add policy</button>}
              </div>
              <label><span className="label">Patient</span><PatientSelect value={settings.patient} onChange={(patient) => setCfg({ ...cfg, patient })} /></label>
              <label><span className="label">Seed</span><input className="input mono" type="number" value={settings.seed} onChange={(e) => setCfg({ ...cfg, seed: Number(e.target.value) })} /></label>
              <label><span className="label">Days</span><input className="input mono" type="number" step={0.25} min={0.125} max={7} value={settings.episode_days} onChange={(e) => setCfg({ ...cfg, episode_days: Number(e.target.value) })} /></label>
              <button className="btn btn-primary" onClick={() => paired.mutate()} disabled={paired.isPending}>{paired.isPending ? 'Simulating…' : 'Run rollouts'}</button>
            </div>
            {paired.isPending && <Loading label="SimGlucose is running every policy on the identical episode" rows={4} />}
            {paired.isError && <ErrorState title="Rollouts failed" detail={(paired.error as Error).message} retry={() => paired.mutate()} />}
            {!paired.data && !paired.isPending && !paired.isError && <Empty title="No rollouts yet" hint="Pick policies, a patient and a seed. Every policy sees the same meals, sensor noise and initial state." />}
            {paired.data && !paired.isPending && (
              <div className="space-y-3 fade-in">
                <div className="flex flex-wrap items-center gap-2 text-[11.5px]">
                  <span className="badge badge-warn">counterfactual full-episode rollouts</span>
                  <span className="badge mono">{paired.data.settings.patient}</span><span className="badge mono">seed {paired.data.settings.seed}</span>
                  <span className="ml-auto flex rounded-md border border-line-2 overflow-hidden" role="radiogroup" aria-label="Signal">
                    {(['bg', 'cgm'] as const).map((x) => <button key={x} role="radio" aria-checked={signal === x} onClick={() => setSignal(x)} className={`px-2.5 h-6 text-[11px] ${signal === x ? 'bg-surface-3 text-fg' : 'bg-bg-1 text-fg-3'}`}>{x === 'bg' ? 'true BG (sim-only)' : 'CGM'}</button>)}
                  </span>
                </div>
                <GlucoseTimeline series={pairedSeries} totalMin={paired.data.settings.episode_days * 1440} markers={meals} height={420} />
                <RangeBars compact rows={paired.data.rollouts.map((r, i) => ({ label: policyLabel(r.policy), color: seriesColor(i), m: r.metrics ?? {}, sub: r.error }))} />
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ ranges: the five metrics from evaluation result sets */
function RangesTab() {
  const results = useResults()
  const sets = results.data?.results ?? []
  const [name, setName] = useState<string>('')
  useEffect(() => { if (!name && sets.length) setName(sets[0].name) }, [sets, name])
  const set = sets.find((r) => r.name === name)
  if (results.isLoading) return <Card><Loading label="Loading evaluation results" /></Card>
  if (!sets.length) return <Card><Empty title="No evaluation results on disk" hint={<>Run <code className="mono">python evaluate.py --checkpoints runs/&lt;run&gt;/best.pt --out results_test</code>, or launch one from the Experiments page. Only real evaluate.py output is shown here.</>} /></Card>
  return (
    <Card title="Time in range, below and above range" subtitle="Means over evaluation episodes from evaluate.py summary.csv. Hover a segment for the exact share."
      actions={<ResultPicker sets={sets} value={name} onChange={setName} />}>
      {set && <RangeSetView set={set} />}
    </Card>
  )
}

function ResultPicker({ sets, value, onChange }: { sets: ResultSet[]; value: string; onChange: (v: string) => void }) {
  return (
    <select className="select !w-auto !h-7 text-[12px]" value={value} onChange={(e) => onChange(e.target.value)} aria-label="Result set">
      {sets.map((r) => <option key={r.name} value={r.name}>{r.name} · {r.split} · {r.policies.length} policies</option>)}
    </select>
  )
}

function RangeSetView({ set }: { set: ResultSet }) {
  const rows = set.summary.map((r, i) => ({
    label: r.policy, color: seriesColor(i),
    m: { tir_70_180: num(r.tir_70_180_mean), tbr_lt70: num(r.tbr_lt70_mean), tbr_lt54: num(r.tbr_lt54_mean), tar_gt180: num(r.tar_gt180_mean), tar_gt250: num(r.tar_gt250_mean) },
    sub: `n=${r.n_episodes}`,
  }))
  const metric = ['tir_70_180', 'tbr_lt70', 'tbr_lt54', 'tar_gt180', 'tar_gt250'] as const
  const names: Record<typeof metric[number], string> = { tir_70_180: 'TIR 70–180', tbr_lt70: 'TBR < 70', tbr_lt54: 'TBR < 54', tar_gt180: 'TAR > 180', tar_gt250: 'TAR > 250' }
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
      <div>
        <div className="flex items-center gap-2 mb-3"><SplitBadge split={set.split} /><span className="text-[11.5px] text-fg-3">{set.dir}/summary.csv · {fmt.ago(set.modified)}</span></div>
        <RangeBars rows={rows} />
      </div>
      <div className="overflow-auto">
        <table className="table">
          <thead><tr><th>Policy</th>{metric.map((m) => <th key={m} className="num">{names[m]}</th>)}</tr></thead>
          <tbody>
            {set.summary.map((r, i) => (
              <tr key={r.policy}>
                <td><span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />{r.policy}</span></td>
                {metric.map((m) => (
                  <td key={m} className="num" title={`95% bootstrap CI: ${fmt.n(num(r[`${m}_ci_lo`]), 1)} – ${fmt.n(num(r[`${m}_ci_hi`]), 1)}`}>
                    {fmt.pct(num(r[`${m}_mean`]))}<span className="text-fg-4 text-[10.5px]"> ±{fmt.n(num(r[`${m}_sd`]), 1)}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-[11px] text-fg-4 mt-2">mean ± sd over episodes; hover a cell for the 95% bootstrap CI.</div>
      </div>
    </div>
  )
}

const num = (v: unknown) => (typeof v === 'number' ? v : v == null ? undefined : Number(v))

/* ------------------------------------------------------------------ training curves */
function useRunSelection() {
  const runs = useRuns()
  const names = (runs.data?.runs ?? []).map((r) => r.name)
  const [sel, setSel] = useState<string[]>([])
  useEffect(() => { if (!sel.length && names.length) setSel(names.slice(0, 2)) }, [names.join(','), sel.length]) // eslint-disable-line react-hooks/exhaustive-deps
  const logs = useQueries({ queries: sel.map((n) => ({ queryKey: ['runlogs', n], queryFn: () => api.runLogs(n), staleTime: 30_000 })) })
  return { runs, names, sel, setSel, logs }
}

function RunChips({ names, sel, setSel, demo }: { names: string[]; sel: string[]; setSel: (s: string[]) => void; demo: Set<string> }) {
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Runs to compare">
      {names.map((n) => {
        const i = sel.indexOf(n)
        const on = i >= 0
        return (
          <button key={n} aria-pressed={on} onClick={() => setSel(on ? sel.filter((x) => x !== n) : sel.length < 6 ? [...sel, n] : sel)}
            className={`badge transition-colors ${on ? 'text-fg' : 'opacity-70 hover:opacity-100'}`} style={on ? { borderColor: seriesColor(i), background: `color-mix(in oklab, ${seriesColor(i)} 14%, var(--color-surface-2))` } : undefined}>
            {on && <span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />}{n}{demo.has(n) && <span className="text-fg-4">demo</span>}
          </button>
        )
      })}
    </div>
  )
}

function TrainingTab() {
  const { runs, names, sel, setSel, logs } = useRunSelection()
  const [smooth, setSmooth] = useState(5)
  const demo = new Set((runs.data?.runs ?? []).filter((r) => r.demo).map((r) => r.name))
  if (runs.isLoading) return <Card><Loading label="Loading runs" /></Card>
  if (!names.length) return <Card><Empty title="No training runs in runs/" hint={<>Train with <code className="mono">python train_dqn.py --episodes 200 --run-name dqn_200ep</code>. Curves are read from train_log.csv.</>} /></Card>
  const loading = logs.some((l) => l.isLoading)
  const mk = (key: 'return' | 'mean_loss' | 'epsilon' | 'tir' | 'tbr_lt70', dashed = false): XYSeries[] => sel.flatMap((n, i) => {
    const d = logs[i]?.data
    if (!d) return []
    const ys = d.train.map((r) => (r[key] == null ? null : Number(r[key])))
    return [{ id: n, label: n, color: seriesColor(i), x: d.train.map((r) => r.episode), y: key === 'epsilon' || key === 'mean_loss' ? ys : movingAvg(ys, smooth), dashed }]
  })
  return (
    <div className="space-y-4">
      <Card title="Training runs" subtitle="One point per training episode, read from runs/<name>/train_log.csv. Choose up to 6 runs."
        actions={<label className="text-[12px] text-fg-3 inline-flex items-center gap-2">smoothing <input type="range" min={1} max={25} value={smooth} onChange={(e) => setSmooth(Number(e.target.value))} aria-label="Moving-average window" /> <span className="mono w-6">{smooth}</span></label>}>
        <RunChips names={names} sel={sel} setSel={setSel} demo={demo} />
      </Card>
      {loading ? <Card><Loading label="Reading logs" rows={4} /></Card> : (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Episode return" subtitle={`Training return per episode (moving average, window ${smooth}).`}><LineChart series={mk('return')} xLabel="episode" yLabel="return" yFormat={(v) => v.toFixed(0)} height={280} /></Card>
          <Card title="DQN loss" subtitle="Mean Huber loss per episode, log scale. Empty before learning starts."><LineChart series={mk('mean_loss')} xLabel="episode" yLabel="loss" yFormat={(v) => v < 0.01 ? v.toExponential(1) : v.toFixed(3)} logY height={280} /></Card>
          <Card title="Epsilon decay" subtitle="Exploration rate used in that episode (linear decay to eps-end)."><LineChart series={mk('epsilon')} xLabel="episode" yLabel="ε" yFormat={(v) => v.toFixed(2)} yDomain={[0, 1]} height={280} /></Card>
          <Card title="Training TIR and time < 70" subtitle="Per-episode glycaemic metrics on training patients (solid TIR, dashed < 70)."><LineChart series={[...mk('tir'), ...mk('tbr_lt70', true).map((s) => ({ ...s, id: s.id + '_tbr', label: s.label + ' <70' }))]} xLabel="episode" yLabel="%" yFormat={(v) => `${v.toFixed(0)}%`} yDomain={[0, 100]} height={280} /></Card>
        </div>
      )}
    </div>
  )
}

function movingAvg(ys: (number | null)[], w: number) {
  if (w <= 1) return ys
  const out: (number | null)[] = []
  for (let i = 0; i < ys.length; i++) {
    const win = ys.slice(Math.max(0, i - w + 1), i + 1).filter((v): v is number => v != null)
    out.push(win.length ? win.reduce((a, b) => a + b, 0) / win.length : null)
  }
  return out
}

function ValidationTab() {
  const { runs, names, sel, setSel, logs } = useRunSelection()
  const demo = new Set((runs.data?.runs ?? []).filter((r) => r.demo).map((r) => r.name))
  if (runs.isLoading) return <Card><Loading label="Loading runs" /></Card>
  if (!names.length) return <Card><Empty title="No training runs in runs/" hint="Validation curves come from val_log.csv, written every --eval-every episodes." /></Card>
  const loading = logs.some((l) => l.isLoading)
  const mk = (key: 'val_return' | 'val_tir' | 'val_tbr_lt70' | 'val_tbr_lt54'): XYSeries[] => sel.flatMap((n, i) => {
    const d = logs[i]?.data
    return d ? [{ id: n, label: n, color: seriesColor(i), x: d.val.map((r) => r.episode), y: d.val.map((r) => Number(r[key])) }] : []
  })
  const bestMarks = sel.flatMap((n, i) => {
    const d = logs[i]?.data
    const best = d?.val.filter((r) => r.is_best === 1).slice(-1)[0]
    return best ? [{ x: best.episode, y: best.val_return, label: `${n} best.pt`, color: seriesColor(i) }] : []
  })
  return (
    <div className="space-y-4">
      <Card title="Validation progress" subtitle="Fixed validation episodes on the 6 validation patients, identical at every evaluation. The marked point is the checkpoint saved as best.pt (highest validation return so far).">
        <RunChips names={names} sel={sel} setSel={setSel} demo={demo} />
      </Card>
      {loading ? <Card><Loading label="Reading logs" rows={4} /></Card> : (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Validation return" subtitle="Mean episode return over validation episodes; model selection criterion."><LineChart series={mk('val_return')} marks={bestMarks} xLabel="episode" yLabel="return" yFormat={(v) => v.toFixed(0)} height={300} /></Card>
          <Card title="Validation TIR" subtitle="Mean time in range 70–180 on validation patients."><LineChart series={mk('val_tir')} xLabel="episode" yLabel="%" yFormat={(v) => `${v.toFixed(0)}%`} yDomain={[0, 100]} height={300} /></Card>
          <Card title="Validation time < 70" subtitle="Hypoglycaemia exposure on validation patients."><LineChart series={mk('val_tbr_lt70')} xLabel="episode" yLabel="%" yFormat={(v) => `${v.toFixed(1)}%`} height={300} /></Card>
          <Card title="Validation time < 54" subtitle="Severe hypoglycaemia exposure on validation patients."><LineChart series={mk('val_tbr_lt54')} xLabel="episode" yLabel="%" yFormat={(v) => `${v.toFixed(1)}%`} height={300} /></Card>
        </div>
      )}
    </div>
  )
}
