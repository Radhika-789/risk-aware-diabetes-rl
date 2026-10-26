import { useNavigate } from 'react-router-dom'
import type { Settings } from '../api/types'
import { useMeta, useRuns } from '../hooks/useCatalog'
import { useDraft, useSession } from '../hooks/useSession'
import { Card, Field, Loading, Empty, SplitBadge } from '../components/ui'
import { PolicySelect } from '../components/PolicySelect'
import { PatientSelect } from '../components/PatientSelect'
import { CheckpointMeta } from './Experiments'

export function Configuration() {
  const meta = useMeta()
  const runs = useRuns()
  const [draft, setDraft] = useDraft()
  const { state: live, create } = useSession()
  const nav = useNavigate()
  if (meta.isLoading || !meta.data) return <div className="p-6"><Card><Loading label="Loading configuration" /></Card></div>
  const m = meta.data
  const s: Settings = { ...m.default_settings, ...draft.settings }
  const ckpt = (runs.data?.runs ?? []).flatMap((r) => r.checkpoints).find((c) => c.id === draft.policy)
  const isCkpt = draft.policy.startsWith('ckpt:')
  const locked = (k: keyof Settings) => isCkpt && m.checkpoint_locked_fields.includes(k) && ckpt?.env_config[k] !== undefined
  const set = (k: keyof Settings, v: unknown) => setDraft({ settings: { [k]: v } })
  const dirty = live && (live.policy !== draft.policy || (Object.keys(s) as (keyof Settings)[]).some((k) => live.settings[k] !== (locked(k) ? ckpt?.env_config[k] : s[k])))

  return (
    <div className="p-4 md:p-6 max-w-[1400px] mx-auto grid gap-4 xl:grid-cols-[1fr_1fr]">
      {/* ---------------- editable: simulation settings ---------------- */}
      <div className="space-y-4">
        <Card title="Simulation settings" subtitle="Editable. Applied when you start the next live session; a running session keeps its own settings."
          actions={<span className="badge badge-accent">editable</span>}>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Policy"><PolicySelect value={draft.policy} onChange={(policy) => setDraft({ policy })} /></Field>
            <Field label="Virtual patient" hint="Train / validation / held-out test split is enforced by the environment."><PatientSelect value={s.patient} onChange={(v) => set('patient', v)} /></Field>
            <Field label="Seed" hint="Patient initial state, sensor noise and meal schedule are all derived from it."><input className="input mono" type="number" min={0} value={s.seed} onChange={(e) => set('seed', Number(e.target.value))} /></Field>
            <Field label="Simulation duration (days)" hint="Truncates the episode; the simulator may terminate earlier."><input className="input mono" type="number" min={0.125} max={7} step={0.125} value={s.episode_days} onChange={(e) => set('episode_days', Number(e.target.value))} /></Field>
            <Field label="Reward mode" hint="risk: asymmetric, hypo-averse. tir: in-range indicator only. Affects reward, not the policy's weights.">
              <select className="select" value={s.reward_mode} onChange={(e) => set('reward_mode', e.target.value)}><option value="risk">risk (asymmetric)</option><option value="tir">tir (indicator)</option></select>
            </Field>
            <Field label="Decision interval (sensor samples)" locked={locked('action_repeat')} hint={`${(locked('action_repeat') ? ckpt!.env_config.action_repeat! : s.action_repeat) * m.sample_min} minutes per decision.`}>
              <input className="input mono" type="number" min={1} max={20} value={locked('action_repeat') ? ckpt!.env_config.action_repeat : s.action_repeat} disabled={locked('action_repeat')} onChange={(e) => set('action_repeat', Number(e.target.value))} />
            </Field>
            <Field label="CGM history length" locked={locked('history_len')} hint="Number of past CGM readings in the observation.">
              <input className="input mono" type="number" min={1} max={32} value={locked('history_len') ? ckpt!.env_config.history_len : s.history_len} disabled={locked('history_len')} onChange={(e) => set('history_len', Number(e.target.value))} />
            </Field>
            <Field label="Announce meals" locked={locked('announce_meals')} hint="Adds grams eaten in the last interval to the observation (privileged information).">
              <select className="select" value={String(locked('announce_meals') ? ckpt!.env_config.announce_meals : s.announce_meals)} disabled={locked('announce_meals')} onChange={(e) => set('announce_meals', e.target.value === 'true')}><option value="false">off (default POMDP)</option><option value="true">on</option></select>
            </Field>
          </div>
        </Card>

        <Card title="Safety shield" subtitle="Rule layer between the policy and the pump. Uses CGM only. Hard dose caps apply even when the shield is disabled." actions={<span className="badge badge-accent">editable</span>}>
          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Shield"><select className="select" value={String(s.shield)} onChange={(e) => set('shield', e.target.value === 'true')}><option value="true">enabled</option><option value="false">disabled (research ablation)</option></select></Field>
            <Field label="Suspend below CGM (mg/dL)" hint="Basal and bolus set to zero under this reading."><input className="input mono" type="number" min={40} max={150} value={s.suspend_below_cgm} disabled={!s.shield} onChange={(e) => set('suspend_below_cgm', Number(e.target.value))} /></Field>
            <Field label="IOB cap (fraction of TDI)" hint="A bolus that would push estimated IOB above this is blocked."><input className="input mono" type="number" min={0.05} max={1} step={0.01} value={s.iob_cap_frac_of_tdi} disabled={!s.shield} onChange={(e) => set('iob_cap_frac_of_tdi', Number(e.target.value))} /></Field>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 mt-4 text-[12px]">
            <dt className="text-fg-3">Hard max basal multiplier</dt><dd className="mono">×{String(m.shield_defaults.hard_max_basal_mult)} (always)</dd>
            <dt className="text-fg-3">Hard max bolus</dt><dd className="mono">{Number(m.shield_defaults.hard_max_bolus_frac_of_tdi) * 100}% of TDI (always)</dd>
          </dl>
        </Card>

        <div className="flex flex-wrap items-center gap-3">
          <button className="btn btn-primary" onClick={() => create.mutate(draft, { onSuccess: () => nav('/') })} disabled={create.isPending}>{create.isPending ? 'Creating…' : live ? 'Start new session with these settings' : 'Start session with these settings'}</button>
          <button className="btn btn-ghost" onClick={() => setDraft({ policy: 'cgm_rule', settings: { ...m.default_settings } })}>Reset to defaults</button>
          {dirty && <span className="text-[12px] text-warn">Differs from the current live session.</span>}
          {create.isError && <span className="text-[12px] text-danger">{(create.error as Error).message}</span>}
        </div>
      </div>

      {/* ---------------- immutable: checkpoint & model ---------------- */}
      <div className="space-y-4">
        <Card title="Checkpoint identity" subtitle="Read from the .pt file. Immutable; the fields it fixes are locked on the left." actions={<span className="badge">immutable</span>}>
          {!isCkpt ? (
            <Empty title="A baseline policy is selected" hint={`"${draft.policy}" is a rule-based or random baseline from baselines.py and has no weights. Choose a DQN checkpoint to see its metadata.`} />
          ) : !ckpt ? <Loading rows={3} /> : <CheckpointMeta c={ckpt} />}
        </Card>

        <Card title="Model" subtitle="dqn.py — fixed architecture and training rule; nothing here is editable from the UI." actions={<span className="badge">immutable</span>}>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[12.5px]">
            <dt className="text-fg-3">Algorithm</dt><dd>Double DQN (online net picks the action, target net values it); Huber loss; Adam; gradient clipping; periodic hard target copy</dd>
            <dt className="text-fg-3">Q-network</dt><dd className="mono">Linear(obs_dim → {ckpt?.hidden ?? 128}) → ReLU → Linear({ckpt?.hidden ?? 128} → {ckpt?.hidden ?? 128}) → ReLU → Linear({ckpt?.hidden ?? 128} → 16)</dd>
            <dt className="text-fg-3">Bootstrapping</dt><dd>Terminal only on simulator termination; time-limit truncation is bootstrapped through</dd>
            <dt className="text-fg-3">Inference</dt><dd>Greedy (ε = 0) in evaluation and in the live session</dd>
            <dt className="text-fg-3">Checkpoint format</dt><dd className="mono">{'{state_dict, obs_dim, n_actions, hidden, env_config_json}'}</dd>
          </dl>
        </Card>

        <Card title="Observation space" subtitle="float32 vector seen by the policy. True BG, patient identity, meal schedule and simulator state are hidden.">
          <ObsSpace historyLen={Number(locked('history_len') ? ckpt?.env_config.history_len : s.history_len)} announce={Boolean(locked('announce_meals') ? ckpt?.env_config.announce_meals : s.announce_meals)} />
        </Card>

        <Card title="Action space" subtitle={`Discrete(${m.action_space.basal_mults.length * m.action_space.bolus_fracs_of_tdi.length}): index = basal_idx × ${m.action_space.bolus_fracs_of_tdi.length} + bolus_idx. Bolus is delivered in the first 3-minute sample of the interval.`}>
          <table className="table text-[12px]">
            <thead><tr><th>basal \ bolus</th>{m.action_space.bolus_fracs_of_tdi.map((b) => <th key={b} className="num">{(b * 100).toFixed(0)}% TDI</th>)}</tr></thead>
            <tbody>{m.action_space.basal_mults.map((bm, i) => (
              <tr key={bm}><td className="text-fg-3">×{bm} basal</td>{m.action_space.bolus_fracs_of_tdi.map((_, j) => <td key={j} className="num">a = {i * m.action_space.bolus_fracs_of_tdi.length + j}</td>)}</tr>
            ))}</tbody>
          </table>
        </Card>

        <Card title="Reward (risk mode)" subtitle="Per decision interval, averaged over true-BG samples. From t1d_env.RewardConfig.">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-[12px] mono">
            <dt className="text-fg-3">in range {String(m.reward.low)}–{String(m.reward.high)}</dt><dd>+{String(m.reward.in_range)}</dd>
            <dt className="text-fg-3">hyper</dt><dd>−min({String(m.reward.hyper_base)} + {String(m.reward.hyper_slope)}·(BG − {String(m.reward.high)}), {String(m.reward.hyper_cap)})</dd>
            <dt className="text-fg-3">hypo</dt><dd>−min({String(m.reward.hypo_base)} + {String(m.reward.hypo_slope)}·({String(m.reward.low)} − BG), {String(m.reward.hypo_cap)})</dd>
            <dt className="text-fg-3">below {String(m.reward.severe_low)}</dt><dd>extra −{String(m.reward.severe_extra)}</dd>
            <dt className="text-fg-3">simulator termination</dt><dd>−{String(m.reward.terminal_penalty)}</dd>
          </dl>
        </Card>

        {live && (
          <Card title="Current live session" subtitle="Settings the running session was created with." actions={<span className="badge">read-only</span>}>
            <div className="flex flex-wrap gap-2 text-[12px]"><span className="badge badge-accent">{live.policy.replace('ckpt:', '')}</span><span className="badge mono">{live.settings.patient}</span><SplitBadge split={live.patient_split} /><span className="badge mono">seed {live.settings.seed}</span><span className="badge">{live.settings.episode_days} d</span><span className="badge">{live.settings.shield ? `shield <${live.settings.suspend_below_cgm}, cap ${live.settings.iob_cap_frac_of_tdi}` : 'shield off'}</span></div>
          </Card>
        )}
      </div>
    </div>
  )
}

function ObsSpace({ historyLen, announce }: { historyLen: number; announce: boolean }) {
  const dim = historyLen + 3 + (announce ? 1 : 0)
  return (
    <div className="text-[12.5px]">
      <div className="mb-2">obs_dim = <span className="mono text-fg">{dim}</span></div>
      <ol className="space-y-1 list-decimal pl-5 text-fg-2">
        <li><span className="mono">cgm[0..{historyLen - 1}] / 400</span> — last {historyLen} CGM readings, one per decision interval</li>
        <li><span className="mono">iob / (0.1 × TDI)</span> — our own IOB estimate (linear decay over DIA), clipped to [−3, 5]</li>
        <li><span className="mono">sin(2π·hour/24), cos(2π·hour/24)</span> — time of day</li>
        {announce && <li><span className="mono">meal_g / 100</span> — grams eaten in the last interval (announce_meals on)</li>}
      </ol>
    </div>
  )
}
