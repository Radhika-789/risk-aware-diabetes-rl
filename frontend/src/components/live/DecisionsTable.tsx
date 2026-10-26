import { useState } from 'react'
import type { Decision, SessionState } from '../../api/types'
import { actionLabel, fmt, glucoseZone, ZONE_COLOR } from '../../lib/format'
import { Drawer, Empty } from '../ui'

export function DecisionsTable({ s }: { s: SessionState }) {
  const [open, setOpen] = useState<Decision | null>(null)
  const rows = (s.decisions ?? []).slice().reverse()
  const as = s.action_space
  if (!rows.length) return <Empty title="No decisions yet" hint="Each row is one 15-minute decision interval: what the policy requested, what the shield applied, and the reward computed from true glucose." />
  return (
    <>
      <div className="overflow-auto max-h-[420px] rounded-md border border-line">
        <table className="table">
          <thead>
            <tr><th>#</th><th>Sim time</th><th className="num">CGM</th><th className="num">True BG*</th><th className="num">IOB</th><th>Action</th><th>Requested → applied</th><th>Shield</th><th className="num">Reward</th></tr>
          </thead>
          <tbody>
            {rows.map((d) => (
              <tr key={d.i} tabIndex={0} className="cursor-pointer" onClick={() => setOpen(d)} onKeyDown={(e) => { if (e.key === 'Enter') setOpen(d) }}>
                <td className="num text-fg-3">{d.i}</td>
                <td className="mono">{fmt.simTime(d.sim_time)}</td>
                <td className="num" style={{ color: ZONE_COLOR[glucoseZone(d.cgm)] }}>{fmt.mgdl(d.cgm)}</td>
                <td className="num text-fg-3">{fmt.mgdl(d.bg_sim_only)}</td>
                <td className="num">{d.iob_u.toFixed(2)}</td>
                <td className="mono">{d.action < 0 ? 'cont' : d.action}</td>
                <td className="text-fg-2 whitespace-nowrap">×{d.req_basal_mult} / {d.req_bolus_u.toFixed(2)} U <span className="text-fg-4">→</span> <span className={d.basal_mult !== d.req_basal_mult || d.bolus_u !== d.req_bolus_u ? 'text-warn' : ''}>×{d.basal_mult} / {d.bolus_u.toFixed(2)} U</span></td>
                <td>{d.shield_suspend ? <span className="badge badge-warn">suspend</span> : d.shield_bolus_blocked ? <span className="badge badge-danger">blocked</span> : <span className="text-fg-4">—</span>}</td>
                <td className={`num ${d.reward < 0 ? 'text-danger' : 'text-ok'}`}>{d.reward.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] text-fg-4 mt-1.5">* True BG is simulator-only and hidden from the policy. Click a row for details.</div>
      <Drawer open={!!open} onClose={() => setOpen(null)} title={open ? `Decision ${open.i} · ${fmt.simTime(open.sim_time)}` : ''}>
        {open && (
          <div className="space-y-4 text-[12.5px]">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
              <Row k="Observed CGM" v={`${fmt.mgdl(open.cgm)} mg/dL`} />
              <Row k="True BG (sim-only)" v={`${fmt.mgdl(open.bg_sim_only)} mg/dL`} />
              <Row k="IOB estimate" v={`${open.iob_u.toFixed(3)} U`} />
              <Row k="Meal in interval" v={`${open.meal_g.toFixed(1)} g`} />
              <Row k="Action" v={actionLabel(open.action, as.basal_mults, as.bolus_fracs_of_tdi, as.tdi_u)} />
              <Row k="Reward" v={open.reward.toFixed(4)} />
              <Row k="Requested" v={`basal ×${open.req_basal_mult}, bolus ${open.req_bolus_u.toFixed(3)} U`} />
              <Row k="Applied" v={`basal ×${open.basal_mult}, bolus ${open.bolus_u.toFixed(3)} U`} />
              <Row k="Shield" v={open.shield_suspend ? 'suspended basal + bolus' : open.shield_bolus_blocked ? 'bolus blocked (IOB cap)' : 'pass-through'} />
              <Row k="Compute" v={`${open.wall_ms} ms`} />
            </dl>
            {open.q_values && (
              <div>
                <div className="label">Q-values (online network, this observation)</div>
                <QGrid q={open.q_values} basal={as.basal_mults} bolus={as.bolus_fracs_of_tdi} chosen={open.action} />
              </div>
            )}
          </div>
        )}
      </Drawer>
    </>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return <><dt className="text-fg-3">{k}</dt><dd className="mono text-fg">{v}</dd></>
}

export function QGrid({ q, basal, bolus, chosen }: { q: number[]; basal: number[]; bolus: number[]; chosen: number }) {
  const max = Math.max(...q), min = Math.min(...q)
  return (
    <table className="table text-[11.5px]">
      <thead><tr><th>basal \ bolus</th>{bolus.map((b) => <th key={b} className="num">{(b * 100).toFixed(0)}% TDI</th>)}</tr></thead>
      <tbody>
        {basal.map((bm, i) => (
          <tr key={bm}><td className="text-fg-3">×{bm}</td>
            {bolus.map((_, j) => {
              const a = i * bolus.length + j
              const v = q[a]
              const t = max === min ? 0 : (v - min) / (max - min)
              return <td key={j} className={`num ${a === chosen ? 'font-semibold text-fg' : 'text-fg-2'}`} style={{ background: `color-mix(in oklab, var(--color-accent) ${Math.round(t * 28)}%, transparent)` }}>{v.toFixed(2)}{a === chosen ? ' ●' : ''}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
