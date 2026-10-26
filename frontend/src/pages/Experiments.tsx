import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, waitForJob } from '../api/client'
import type { Checkpoint, ResultSet, Run, SummaryRow } from '../api/types'
import { useResults, useRuns } from '../hooks/useCatalog'
import { fmt, seriesColor } from '../lib/format'
import { Card, Collapsible, Drawer, Empty, ErrorState, Loading, SplitBadge, Tabs } from '../components/ui'
import { RangeBars } from '../components/charts/RangeBars'

type Tab = 'runs' | 'results' | 'compare'

const METRICS: { k: string; label: string; d: number; pct?: boolean; lowerBetter?: boolean; noBest?: boolean }[] = [
  { k: 'tir_70_180', label: 'TIR', d: 1, pct: true },
  { k: 'tbr_lt70', label: '< 70', d: 1, pct: true, lowerBetter: true },
  { k: 'tbr_lt54', label: '< 54', d: 1, pct: true, lowerBetter: true },
  { k: 'tar_gt180', label: '> 180', d: 1, pct: true, lowerBetter: true },
  { k: 'tar_gt250', label: '> 250', d: 1, pct: true, lowerBetter: true },
  { k: 'mean_bg', label: 'Mean BG', d: 0, noBest: true },
  { k: 'lbgi', label: 'LBGI', d: 2, lowerBetter: true },
  { k: 'hbgi', label: 'HBGI', d: 2, lowerBetter: true },
  { k: 'terminated_early', label: 'Term.', d: 2, lowerBetter: true },
  { k: 'episode_return', label: 'Return', d: 0 },
]

export function Experiments() {
  const [tab, setTab] = useState<Tab>('runs')
  return (
    <div className="p-4 md:p-6 space-y-4 max-w-[1600px] mx-auto">
      <Tabs aria-label="Experiment views" value={tab} onChange={setTab} tabs={[
        { id: 'runs', label: 'Training runs' }, { id: 'results', label: 'Evaluation results' }, { id: 'compare', label: 'Compare checkpoints' },
      ]} />
      <div className="fade-in" key={tab}>
        {tab === 'runs' && <RunsTab />}
        {tab === 'results' && <ResultsTab />}
        {tab === 'compare' && <CompareTab />}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ runs */
function RunsTab() {
  const runs = useRuns()
  const [open, setOpen] = useState<Run | null>(null)
  if (runs.isLoading) return <Card><Loading label="Scanning runs/" /></Card>
  if (runs.isError) return <Card><ErrorState detail={(runs.error as Error).message} retry={() => runs.refetch()} /></Card>
  const list = runs.data?.runs ?? []
  return (
    <Card title="Training runs" subtitle={`From ${runs.data?.runs_dir ?? 'runs'}/<name>/config.json, train_log.csv, val_log.csv and checkpoints. Click a row for details.`}>
      {!list.length ? (
        <Empty title="No runs found" hint={<>Train one from the repo root: <code className="mono">python train_dqn.py --episodes 200 --run-name dqn_200ep</code> and <code className="mono">--episodes 500 --run-name dqn_500ep</code>. The directory is gitignored, so a fresh clone has none.</>} />
      ) : (
        <div className="overflow-auto">
          <table className="table">
            <thead><tr><th>Run</th><th>Status</th><th className="num">Episodes</th><th className="num">Env steps</th><th>Selected checkpoint</th><th className="num">Val return</th><th className="num">Val TIR</th><th className="num">Val &lt; 70</th><th>Files</th></tr></thead>
            <tbody>
              {list.map((r) => (
                <tr key={r.name} tabIndex={0} className="cursor-pointer" onClick={() => setOpen(r)} onKeyDown={(e) => { if (e.key === 'Enter') setOpen(r) }}>
                  <td className="font-medium">{r.name}{r.demo && <span className="badge ml-2" style={{ height: 17, fontSize: 10 }}>demo</span>}</td>
                  <td>{r.complete ? <span className="badge badge-ok">complete</span> : r.episodes_done ? <span className="badge badge-warn">partial</span> : <span className="badge">no log</span>}</td>
                  <td className="num">{r.episodes_done}{r.episodes_planned ? ` / ${r.episodes_planned}` : ''}</td>
                  <td className="num">{r.last_train?.env_steps ?? '—'}</td>
                  <td>{r.selected_checkpoint ? <span className="mono text-[12px]">best.pt @ ep {r.selected_checkpoint.episode}</span> : <span className="text-fg-4">none yet</span>}</td>
                  <td className="num">{fmt.n(r.selected_checkpoint?.val_return, 1)}</td>
                  <td className="num">{fmt.pct(r.selected_checkpoint?.val_tir)}</td>
                  <td className="num">{fmt.pct(r.selected_checkpoint?.val_tbr_lt70)}</td>
                  <td className="text-fg-3 text-[12px]">{r.checkpoints.map((c) => c.file).join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Drawer open={!!open} onClose={() => setOpen(null)} title={open?.name ?? ''} width={560}>
        {open && <RunDetails run={open} />}
      </Drawer>
    </Card>
  )
}

function RunDetails({ run }: { run: Run }) {
  const logs = useQuery({ queryKey: ['runlogs', run.name], queryFn: () => api.runLogs(run.name) })
  const args = run.config.args ?? {}
  const sel = run.selected_checkpoint
  return (
    <div className="space-y-4 text-[12.5px]">
      {run.demo && <div className="badge badge-warn">demo run: short pipeline check, not a trained model</div>}
      <section>
        <div className="label">Checkpoint selection</div>
        {sel ? (
          <p className="text-fg-2 leading-relaxed">
            <span className="mono text-fg">best.pt</span> was saved at episode <span className="mono text-fg">{sel.episode}</span> because its validation return
            <span className="mono text-fg"> {fmt.n(sel.val_return, 1)}</span> was the highest of the {run.n_val_evals} evaluations on the 6 validation patients
            (TIR {fmt.pct(sel.val_tir)}, &lt;70 {fmt.pct(sel.val_tbr_lt70)}, &lt;54 {fmt.pct(sel.val_tbr_lt54)}). <span className="mono text-fg">final.pt</span> is the network after the last episode regardless of validation.
          </p>
        ) : <p className="text-fg-3">No validation evaluation recorded yet.</p>}
      </section>
      <section>
        <div className="label">Validation log</div>
        {logs.isLoading ? <Loading rows={2} /> : (
          <div className="overflow-auto max-h-56 rounded-md border border-line">
            <table className="table text-[12px]">
              <thead><tr><th className="num">Ep</th><th className="num">Return</th><th className="num">TIR</th><th className="num">&lt;70</th><th className="num">&lt;54</th><th className="num">Term.</th><th>Best</th></tr></thead>
              <tbody>{(logs.data?.val ?? []).map((v) => (
                <tr key={v.episode} aria-selected={v.episode === sel?.episode}><td className="num">{v.episode}</td><td className="num">{fmt.n(v.val_return, 1)}</td><td className="num">{fmt.pct(v.val_tir)}</td><td className="num">{fmt.pct(v.val_tbr_lt70)}</td><td className="num">{fmt.pct(v.val_tbr_lt54)}</td><td className="num">{fmt.n(v.val_terminated, 2)}</td><td>{v.is_best ? '●' : ''}</td></tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </section>
      <Collapsible title="Training arguments" defaultOpen>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 mono text-[11.5px]">
          {Object.entries(args).map(([k, v]) => <><dt key={k + 'k'} className="text-fg-3">{k}</dt><dd key={k + 'v'}>{String(v)}</dd></>)}
        </dl>
      </Collapsible>
      <Collapsible title="Checkpoints" defaultOpen>
        {run.checkpoints.map((c) => <CheckpointMeta key={c.id} c={c} />)}
      </Collapsible>
      <Collapsible title="Patient splits">
        <div className="text-[11.5px]"><span className="text-fg-3">train:</span> <span className="mono">{run.config.train_patients?.join(', ')}</span></div>
        <div className="text-[11.5px] mt-1"><span className="text-fg-3">val:</span> <span className="mono">{run.config.val_patients?.join(', ')}</span></div>
      </Collapsible>
    </div>
  )
}

export function CheckpointMeta({ c }: { c: Checkpoint }) {
  return (
    <div className="rounded-md border border-line bg-bg-1/60 p-3 mb-2 text-[12px]">
      <div className="flex items-center justify-between"><span className="mono text-fg">{c.file}</span><span className="text-fg-4">{fmt.bytes(c.size_bytes)} · {fmt.ago(c.modified)}</span></div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 mt-2 mono text-[11.5px]">
        <dt className="text-fg-3">obs_dim</dt><dd>{c.obs_dim}</dd>
        <dt className="text-fg-3">n_actions</dt><dd>{c.n_actions}</dd>
        <dt className="text-fg-3">hidden</dt><dd>{c.hidden} ({c.layers.length} linear layers, {c.n_params.toLocaleString()} params)</dd>
        <dt className="text-fg-3">env_config</dt><dd className="break-all">{JSON.stringify(c.env_config)}</dd>
      </dl>
    </div>
  )
}

/* ------------------------------------------------------------------ results */
function ResultsTab() {
  const results = useResults()
  const runs = useRuns()
  const qc = useQueryClient()
  const [open, setOpen] = useState<{ set: ResultSet; policy: string } | null>(null)
  const [form, setForm] = useState({ split: 'test', episodes_per_patient: 1, out: 'results_ui', checkpoints: [] as string[] })
  const [jobMsg, setJobMsg] = useState<string>('')
  const launch = useMutation({
    mutationFn: async () => {
      const j = await api.evaluate(form)
      const done = await waitForJob(j.id, (x) => setJobMsg(x.message), 1500)
      if (done.status === 'error') throw new Error(done.error ?? 'evaluation failed')
      return done
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['results'] }),
  })
  const ckpts = (runs.data?.runs ?? []).flatMap((r) => r.checkpoints)
  const sets = results.data?.results ?? []
  return (
    <div className="space-y-4">
      <Card title="Evaluation results" subtitle="Each result set is one evaluate.py run: every policy on identical episodes of one patient split. Click a policy for per-episode rows.">
        {results.isLoading ? <Loading label="Loading results" /> : !sets.length ? (
          <Empty title="No evaluation results yet" hint="Launch one below, or run evaluate.py from the repo root. Only real evaluate.py output is listed here." />
        ) : sets.map((set) => <ResultTable key={set.name} set={set} onOpen={(policy) => setOpen({ set, policy })} />)}
      </Card>

      <Card title="Launch an evaluation" subtitle="Runs evaluate.py as a background job and writes a new results directory. Baselines are always included.">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[1fr_auto_auto_auto_auto] items-end">
          <div>
            <span className="label">Checkpoints</span>
            {!ckpts.length ? <div className="text-[12px] text-fg-3">No checkpoints on disk; baselines only.</div> : (
              <div className="flex flex-wrap gap-1.5">
                {ckpts.map((c) => {
                  const on = form.checkpoints.includes(c.id)
                  return <button key={c.id} aria-pressed={on} className={`badge ${on ? 'badge-accent' : ''}`} onClick={() => setForm({ ...form, checkpoints: on ? form.checkpoints.filter((x) => x !== c.id) : [...form.checkpoints, c.id] })}>{c.id.slice(5)}</button>
                })}
              </div>
            )}
          </div>
          <label><span className="label">Split</span>
            <select className="select" value={form.split} onChange={(e) => setForm({ ...form, split: e.target.value })}>
              <option value="test">held-out test</option><option value="val">validation</option><option value="train">training</option>
            </select>
          </label>
          <label><span className="label">Episodes / patient</span><input className="input mono w-24" type="number" min={1} max={10} value={form.episodes_per_patient} onChange={(e) => setForm({ ...form, episodes_per_patient: Number(e.target.value) })} /></label>
          <label><span className="label">Output dir</span><input className="input mono w-40" value={form.out} onChange={(e) => setForm({ ...form, out: e.target.value })} /></label>
          <button className="btn btn-primary" onClick={() => launch.mutate()} disabled={launch.isPending}>{launch.isPending ? 'Evaluating…' : 'Run evaluate.py'}</button>
        </div>
        {launch.isPending && <div className="mt-3 text-[12px] text-fg-3 flex items-center gap-2"><span className="live-dot" /><span className="mono truncate">{jobMsg || 'starting'}</span><span className="text-fg-4">(6 patients × policies × episodes; roughly 5 s per episode)</span></div>}
        {launch.isError && <div className="mt-3"><ErrorState title="Evaluation failed" detail={(launch.error as Error).message} /></div>}
        {launch.isSuccess && <div className="mt-3 text-[12px] text-ok">Done. Results refreshed above.</div>}
      </Card>

      <Drawer open={!!open} onClose={() => setOpen(null)} title={open ? `${open.policy} · ${open.set.name}` : ''} width={640}>
        {open && <EpisodeRows set={open.set} policy={open.policy} />}
      </Drawer>
    </div>
  )
}

function ResultTable({ set, onOpen }: { set: ResultSet; onOpen: (policy: string) => void }) {
  const best = useMemo(() => {
    const out: Record<string, string> = {}
    for (const m of METRICS) {
      if (m.noBest) continue
      const vals = set.summary.map((r) => Number(r[`${m.k}_mean`]))
      const idx = vals.indexOf(m.lowerBetter ? Math.min(...vals) : Math.max(...vals))
      if (idx >= 0) out[m.k] = set.summary[idx].policy
    }
    return out
  }, [set])
  return (
    <div className="mb-6 last:mb-0">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className="font-medium text-[13px]">{set.name}</span><SplitBadge split={set.split} />{set.name.includes('demo') && <span className="badge" style={{ height: 17, fontSize: 10 }}>demo</span>}
        <span className="text-[11.5px] text-fg-3">{set.policies.length} policies · {set.episodes.length} episodes · {fmt.ago(set.modified)}</span>
        {set.meta.command != null && <span className="text-[11px] text-fg-4 mono truncate max-w-md" title={String((set.meta.command as string[]).join(' '))}>{(set.meta.command as string[]).slice(2).join(' ')}</span>}
      </div>
      <div className="overflow-auto rounded-md border border-line">
        <table className="table">
          <thead><tr><th>Policy</th><th className="num">n</th>{METRICS.map((m) => <th key={m.k} className="num">{m.label}</th>)}</tr></thead>
          <tbody>
            {set.summary.map((r) => (
              <tr key={r.policy} tabIndex={0} className="cursor-pointer" onClick={() => onOpen(r.policy)} onKeyDown={(e) => { if (e.key === 'Enter') onOpen(r.policy) }}>
                <td className="font-medium">{r.policy}</td><td className="num text-fg-3">{r.n_episodes}</td>
                {METRICS.map((m) => <MetricCell key={m.k} r={r} m={m} best={best[m.k] === r.policy} />)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MetricCell({ r, m, best }: { r: SummaryRow; m: typeof METRICS[number]; best: boolean }) {
  const v = Number(r[`${m.k}_mean`]); const lo = Number(r[`${m.k}_ci_lo`]); const hi = Number(r[`${m.k}_ci_hi`])
  return (
    <td className={`num ${best ? 'text-fg font-medium' : 'text-fg-2'}`} title={`95% bootstrap CI [${lo.toFixed(m.d)}, ${hi.toFixed(m.d)}]`}>
      {Number.isFinite(v) ? (m.pct ? fmt.pct(v, m.d) : v.toFixed(m.d)) : '—'}{best && <span className="text-accent"> ●</span>}
    </td>
  )
}

function EpisodeRows({ set, policy }: { set: ResultSet; policy: string }) {
  const rows = set.episodes.filter((e) => e.policy === policy)
  const groups = ['adolescent', 'adult', 'child']
  return (
    <div className="space-y-4 text-[12px]">
      <div className="flex gap-2 items-center"><SplitBadge split={set.split} /><span className="text-fg-3">{rows.length} episodes</span></div>
      <RangeBars compact rows={groups.map((g, i) => {
        const gr = rows.filter((e) => String(e.patient).startsWith(g))
        const mean = (k: string) => gr.length ? gr.reduce((a, e) => a + Number(e[k]), 0) / gr.length : undefined
        return { label: g, color: seriesColor(i), sub: `n=${gr.length}`, m: { tir_70_180: mean('tir_70_180'), tbr_lt70: mean('tbr_lt70'), tbr_lt54: mean('tbr_lt54'), tar_gt180: mean('tar_gt180'), tar_gt250: mean('tar_gt250') } }
      })} />
      <div className="overflow-auto rounded-md border border-line max-h-[60vh]">
        <table className="table text-[11.5px]">
          <thead><tr><th>Patient</th><th className="num">Seed</th><th className="num">TIR</th><th className="num">&lt;70</th><th className="num">&lt;54</th><th className="num">&gt;180</th><th className="num">&gt;250</th><th className="num">Mean</th><th className="num">LBGI</th><th className="num">HBGI</th><th className="num">Ins U</th><th className="num">Ret</th><th>Term</th></tr></thead>
          <tbody>{rows.map((e, i) => (
            <tr key={i}><td className="mono">{e.patient}</td><td className="num">{e.seed}</td><td className="num">{fmt.n(Number(e.tir_70_180))}</td><td className="num">{fmt.n(Number(e.tbr_lt70))}</td><td className="num">{fmt.n(Number(e.tbr_lt54))}</td><td className="num">{fmt.n(Number(e.tar_gt180))}</td><td className="num">{fmt.n(Number(e.tar_gt250))}</td><td className="num">{fmt.n(Number(e.mean_bg), 0)}</td><td className="num">{fmt.n(Number(e.lbgi), 2)}</td><td className="num">{fmt.n(Number(e.hbgi), 2)}</td><td className="num">{fmt.n(Number(e.total_insulin_u), 1)}</td><td className="num">{fmt.n(Number(e.episode_return), 0)}</td><td>{Number(e.terminated_early) ? <span className="badge badge-danger" style={{ height: 16, fontSize: 10 }}>yes</span> : ''}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ compare two checkpoints (e.g. 200- vs 500-episode) */
function CompareTab() {
  const runs = useRuns()
  const results = useResults()
  const list = runs.data?.runs ?? []
  const [a, setA] = useState(''); const [b, setB] = useState('')
  useEffect(() => {
    if (list.length && !a) { setA(list[0].name); setB(list[1]?.name ?? list[0].name) }
  }, [list, a])
  if (runs.isLoading) return <Card><Loading label="Loading runs" /></Card>
  if (list.length < 1) return <Card><Empty title="Nothing to compare" hint="Train at least two runs (for example 200 and 500 episodes) and evaluate their best.pt on the held-out test split." /></Card>
  const ra = list.find((r) => r.name === a); const rb = list.find((r) => r.name === b)
  const sets = results.data?.results ?? []
  const findRow = (run: Run | undefined, split: string): { set: ResultSet; row: SummaryRow } | null => {
    if (!run) return null
    for (const s of sets.filter((s) => s.split === split).sort((x, y) => y.modified - x.modified)) {
      const row = s.summary.find((r) => r.policy === `${run.name}_best` || r.policy.startsWith(run.name))
      if (row) return { set: s, row }
    }
    return null
  }
  const cols = [ra, rb]
  return (
    <div className="space-y-4">
      <Card title="Compare two runs" subtitle="Training, validation and held-out test are kept apart: training numbers come from train_log.csv, validation from val_log.csv (used to pick best.pt), test from an evaluate.py run on the test split.">
        <div className="grid sm:grid-cols-2 gap-3 max-w-xl">
          {[[a, setA, 'Run A'], [b, setB, 'Run B']].map(([v, set, lab], i) => (
            <label key={i}><span className="label flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />{lab as string}</span>
              <select className="select" value={v as string} onChange={(e) => (set as (s: string) => void)(e.target.value)}>{list.map((r) => <option key={r.name} value={r.name}>{r.name}{r.demo ? ' [demo]' : ''}</option>)}</select>
            </label>
          ))}
        </div>
      </Card>
      <Card title="Side by side">
        <div className="overflow-auto">
          <table className="table">
            <thead><tr><th>Quantity</th><th>Stage</th>{cols.map((r, i) => <th key={i} className="num"><span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: seriesColor(i) }} />{r?.name ?? '—'}</span></th>)}</tr></thead>
            <tbody>
              <Row label="Episodes trained" stage="training" vals={cols.map((r) => r ? `${r.episodes_done}${r.episodes_planned ? ` / ${r.episodes_planned}` : ''}` : '—')} />
              <Row label="Environment steps" stage="training" vals={cols.map((r) => r?.last_train?.env_steps?.toLocaleString() ?? '—')} />
              <Row label="Final-episode training return" stage="training" vals={cols.map((r) => fmt.n(r?.last_train?.return, 1))} />
              <Row label="best.pt chosen at episode" stage="validation" vals={cols.map((r) => r?.selected_checkpoint ? String(r.selected_checkpoint.episode) : '—')} />
              <Row label="Validation return (selection criterion)" stage="validation" vals={cols.map((r) => fmt.n(r?.selected_checkpoint?.val_return, 1))} />
              <Row label="Validation TIR" stage="validation" vals={cols.map((r) => fmt.pct(r?.selected_checkpoint?.val_tir))} />
              <Row label="Validation time < 70" stage="validation" vals={cols.map((r) => fmt.pct(r?.selected_checkpoint?.val_tbr_lt70))} />
              {METRICS.map((m) => (
                <Row key={m.k} label={m.label === 'TIR' ? 'Test TIR 70–180' : `Test ${m.label}`} stage="held-out test" vals={cols.map((r) => {
                  const f = findRow(r, 'test'); if (!f) return 'not evaluated'
                  const v = Number(f.row[`${m.k}_mean`]); return m.pct ? fmt.pct(v, m.d) : v.toFixed(m.d)
                })} />
              ))}
              <Row label="Test result set" stage="held-out test" vals={cols.map((r) => findRow(r, 'test')?.set.name ?? '—')} />
            </tbody>
          </table>
        </div>
        <p className="text-[11.5px] text-fg-4 mt-3">Test rows match the policy label <span className="mono">&lt;run&gt;_best</span> in the newest test result set. If a run shows "not evaluated", launch an evaluation on the Results tab with that run's best.pt.</p>
      </Card>
    </div>
  )
}

function Row({ label, stage, vals }: { label: string; stage: string; vals: string[] }) {
  const badge = stage === 'training' ? 'badge' : stage === 'validation' ? 'badge badge-warn' : 'badge badge-accent'
  return (
    <tr><td>{label}</td><td><span className={badge} style={{ height: 18, fontSize: 10.5 }}>{stage}</span></td>{vals.map((v, i) => <td key={i} className="num">{v}</td>)}</tr>
  )
}

