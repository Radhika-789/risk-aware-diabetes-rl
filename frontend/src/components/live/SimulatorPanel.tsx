import type { SessionState } from '../../api/types'
import { actionLabel, fmt, glucoseZone, ZONE_COLOR, ZONE_LABEL } from '../../lib/format'
import { Hint, Stat } from '../ui'

export function SimulatorPanel({ s }: { s: SessionState }) {
  const zone = glucoseZone(s.current.cgm)
  const d = s.last_decision
  const as = s.action_space
  const tdi = as.tdi_u
  const shielded = d && (d.shield_suspend || d.shield_bolus_blocked)
  const pct = s.clock.total_min ? (s.clock.elapsed_min / s.clock.total_min) * 100 : 0

  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
      {/* ------- readings ------- */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-5">
        <div className="col-span-2 sm:col-span-1">
          <div className="text-[11px] uppercase tracking-wide text-fg-3">Current CGM <Hint text="Noisy Dexcom sensor reading, sampled every 3 minutes. This is what the policy observes."><span className="normal-case text-fg-4">(observed)</span></Hint></div>
          <div className="flex items-baseline gap-2 mt-0.5">
            <span className="mono text-[44px] leading-none font-medium transition-colors duration-300" style={{ color: ZONE_COLOR[zone] }}>{fmt.mgdl(s.current.cgm)}</span>
            <span className="text-[12px] text-fg-3">mg/dL</span>
          </div>
          <div className="text-[11.5px] mt-1" style={{ color: ZONE_COLOR[zone] }}>{ZONE_LABEL[zone]}</div>
        </div>
        <div className="col-span-2 sm:col-span-1">
          <div className="text-[11px] uppercase tracking-wide text-fg-3">True glucose <span className="badge badge-warn normal-case" style={{ height: 17, fontSize: 10 }}>simulator-only</span></div>
          <div className="flex items-baseline gap-2 mt-0.5">
            <span className="mono text-[30px] leading-none text-fg-2">{fmt.mgdl(s.current.bg_sim_only)}</span>
            <span className="text-[12px] text-fg-3">mg/dL</span>
          </div>
          <div className="text-[11px] text-fg-4 mt-1">Hidden from the agent. Used only for reward and metrics.</div>
        </div>
        <Stat label="IOB estimate" value={fmt.n(s.current.iob_u, 2)} unit="U" sub={`cap ${s.current.iob_cap_u.toFixed(2)} U · linear decay, not the simulator's state`} title="Our own estimate of insulin on board above nominal basal; negative while basal is suspended." />
        <Stat label="Simulation clock" value={fmt.simTime(s.clock.now)} sub={`${fmt.clock(s.clock.elapsed_min)} elapsed of ${fmt.hours(s.clock.total_min)} · decision ${s.clock.decision}/${s.clock.max_decisions}`} />
        <div className="col-span-2">
          <div className="h-1 rounded-full bg-bg-1 overflow-hidden" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label="Episode progress">
            <div className="h-full bg-accent/70 transition-[width] duration-300" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>

      {/* ------- latest action ------- */}
      <div className="rounded-lg border border-line bg-bg-1/60 p-3.5 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <div className="text-[11px] uppercase tracking-wide text-fg-3">Latest {s.policy.startsWith('ckpt:') ? 'DQN' : 'policy'} action</div>
          {d && <span className="mono text-[11px] text-fg-4">t = {fmt.clock(d.t_min)} · {d.wall_ms} ms</span>}
        </div>
        {!d ? (
          <div className="text-[12.5px] text-fg-3 mt-3">No decision yet. Step or run the simulation.</div>
        ) : (
          <div className="mt-2.5 space-y-2.5">
            <div className="flex items-center gap-2 text-[13px]">
              <span className="badge mono">a = {d.action < 0 ? 'cont.' : d.action}</span>
              <span className="text-fg-2 truncate">{actionLabel(d.action, as.basal_mults, as.bolus_fracs_of_tdi, tdi)}</span>
            </div>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-x-3 gap-y-1 text-[12px] items-center">
              <span />
              <span className="text-fg-4 uppercase text-[10.5px] tracking-wide">Requested</span>
              <span className="text-fg-4 uppercase text-[10.5px] tracking-wide">Applied</span>
              <span className="text-fg-3">Basal</span>
              <span className="mono">×{d.req_basal_mult}</span>
              <span className={`mono ${d.basal_mult !== d.req_basal_mult ? 'text-warn' : ''}`}>×{d.basal_mult}</span>
              <span className="text-fg-3">Bolus</span>
              <span className="mono">{d.req_bolus_u.toFixed(2)} U</span>
              <span className={`mono ${d.bolus_u !== d.req_bolus_u ? 'text-warn' : ''}`}>{d.bolus_u.toFixed(2)} U</span>
              <span className="text-fg-3">Reward</span>
              <span className={`mono col-span-2 ${d.reward < 0 ? 'text-danger' : 'text-ok'}`}>{d.reward >= 0 ? '+' : ''}{d.reward.toFixed(3)}</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-0.5">
              {d.shield_suspend && <span className="badge badge-warn">shield: suspended (CGM &lt; {s.settings.suspend_below_cgm})</span>}
              {d.shield_bolus_blocked && <span className="badge badge-danger">shield: bolus blocked (IOB cap)</span>}
              {!shielded && <span className="badge badge-ok">shield: pass-through</span>}
              {d.meal_g > 0 && <span className="badge">meal {d.meal_g.toFixed(0)} g</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
