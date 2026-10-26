import type { SessionState } from '../../api/types'
import { fmt } from '../../lib/format'
import { FiveMetrics, RangeBars } from '../charts/RangeBars'
import { Collapsible, SplitBadge } from '../ui'
import { QGrid } from './DecisionsTable'

export function SidePanel({ s }: { s: SessionState }) {
  const m = s.metrics_so_far
  const ev = s.events.slice().reverse()
  const d = s.last_decision
  return (
    <div className="card px-4 py-1">
      <Collapsible title="Glycaemic metrics so far" defaultOpen badge={<span className="text-[11px] text-fg-4 mono">{m.n_samples ?? 0} samples</span>}>
        {m.tir_70_180 == null ? (
          <div className="text-[12px] text-fg-3">No samples yet.</div>
        ) : (
          <div className="space-y-3">
            <FiveMetrics m={m} scopeNote="Computed from true glucose over the simulated part of the episode." />
            <RangeBars compact rows={[{ label: 'episode so far', m }]} />
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
              <dt className="text-fg-3">Mean BG</dt><dd className="mono">{fmt.n(m.mean_bg, 0)} mg/dL</dd>
              <dt className="text-fg-3">CV</dt><dd className="mono">{fmt.pct(m.cv_bg, 0)}</dd>
              <dt className="text-fg-3">LBGI / HBGI</dt><dd className="mono">{fmt.n(m.lbgi, 2)} / {fmt.n(m.hbgi, 2)}</dd>
              <dt className="text-fg-3">Min / max</dt><dd className="mono">{fmt.n(m.min_bg, 0)} / {fmt.n(m.max_bg, 0)}</dd>
              <dt className="text-fg-3">Insulin / carbs</dt><dd className="mono">{fmt.n(m.total_insulin_u, 1)} U / {fmt.n(m.total_carbs_g, 0)} g</dd>
            </dl>
          </div>
        )}
      </Collapsible>

      <Collapsible title="Safety-shield events" badge={<span className="text-[11px] text-fg-4 mono">{s.shield_counts.suspends} suspend · {s.shield_counts.bolus_blocks} blocked</span>}>
        {!s.settings.shield && <div className="text-[12px] text-warn mb-2">Shield disabled for this session. Hard dose caps still apply.</div>}
        {ev.filter((e) => e.kind !== 'meal').length === 0 ? (
          <div className="text-[12px] text-fg-3">No shield interventions yet.</div>
        ) : (
          <ul className="space-y-1.5 max-h-56 overflow-auto pr-1">
            {ev.filter((e) => e.kind !== 'meal').map((e, i) => (
              <li key={i} className="flex items-start gap-2 text-[12px]">
                <span className={`badge shrink-0 ${e.kind === 'suspend' ? 'badge-warn' : 'badge-danger'}`} style={{ height: 18, fontSize: 10 }}>{e.kind.replace('_', ' ')}</span>
                <span className="mono text-fg-4 shrink-0">{fmt.clock(e.t_min)}</span>
                <span className="text-fg-2">{e.detail}</span>
              </li>
            ))}
          </ul>
        )}
      </Collapsible>

      <Collapsible title="Meals (hidden from the policy)" badge={<span className="text-[11px] text-fg-4 mono">{ev.filter((e) => e.kind === 'meal').length}</span>}>
        {ev.filter((e) => e.kind === 'meal').length === 0 ? <div className="text-[12px] text-fg-3">No meals yet.</div> : (
          <ul className="space-y-1 text-[12px] max-h-40 overflow-auto">
            {ev.filter((e) => e.kind === 'meal').map((e, i) => <li key={i} className="flex gap-2"><span className="mono text-fg-4">{fmt.clock(e.t_min)}</span><span className="text-fg-2">{e.detail}</span></li>)}
          </ul>
        )}
        <div className="text-[11px] text-fg-4 mt-2">Meals come from the RandomScenario generator. The policy only sees them if announce_meals is on.</div>
      </Collapsible>

      {d?.q_values && (
        <Collapsible title="Q-values at last decision">
          <QGrid q={d.q_values} basal={s.action_space.basal_mults} bolus={s.action_space.bolus_fracs_of_tdi} chosen={d.action} />
        </Collapsible>
      )}

      <Collapsible title="Virtual patient">
        <div className="flex items-center gap-2 mb-2"><span className="mono text-[13px]">{s.settings.patient}</span><SplitBadge split={s.patient_split} /></div>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
          <dt className="text-fg-3">Age</dt><dd className="mono">{s.patient_info.Age}</dd>
          <dt className="text-fg-3">Body weight</dt><dd className="mono">{s.patient_info.BW.toFixed(1)} kg</dd>
          <dt className="text-fg-3">TDI</dt><dd className="mono">{s.patient_info.TDI.toFixed(1)} U</dd>
          <dt className="text-fg-3">Nominal basal</dt><dd className="mono">{(s.patient_info.basal_U_per_min * 60).toFixed(3)} U/h</dd>
          <dt className="text-fg-3">CR / CF</dt><dd className="mono">{s.patient_info.CR.toFixed(1)} / {s.patient_info.CF.toFixed(1)}</dd>
        </dl>
        <div className="text-[11px] text-fg-4 mt-2">From SimGlucose's vpatient_params.csv and Quest.csv. Identity and parameters are hidden from the policy.</div>
      </Collapsible>
    </div>
  )
}
