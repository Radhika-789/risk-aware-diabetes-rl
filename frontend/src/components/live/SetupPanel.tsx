import { useMeta } from '../../hooks/useCatalog'
import { useDraft } from '../../hooks/useSession'
import { Card, Field, Loading } from '../ui'
import { PolicySelect } from '../PolicySelect'
import { PatientSelect } from '../PatientSelect'

export function SetupPanel({ onStart, busy, error }: { onStart: () => void; busy: boolean; error?: string | null }) {
  const meta = useMeta()
  const [draft, setDraft] = useDraft()
  if (meta.isLoading || !meta.data) return <Card><Loading label="Loading simulator catalogue" /></Card>
  const d = meta.data.default_settings
  const s = { ...d, ...draft.settings }
  const isCkpt = draft.policy.startsWith('ckpt:')
  return (
    <Card title="Start a live simulation" subtitle="One SimGlucose virtual patient, stepped only when the backend completes real simulator steps.">
      <form className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" onSubmit={(e) => { e.preventDefault(); onStart() }}>
        <Field label="Policy" hint={isCkpt ? 'Checkpoint fixes history length, decision interval and meal announcement.' : 'Rule-based or random baseline from baselines.py.'}>
          <PolicySelect value={draft.policy} onChange={(policy) => setDraft({ policy })} />
        </Field>
        <Field label="Virtual patient" hint="Held-out test patients are the fair comparison set.">
          <PatientSelect value={s.patient} onChange={(patient) => setDraft({ settings: { patient } })} />
        </Field>
        <Field label="Seed" hint="Determines meals, sensor noise and initial state.">
          <input className="input mono" type="number" min={0} step={1} value={s.seed} onChange={(e) => setDraft({ settings: { seed: Number(e.target.value) } })} />
        </Field>
        <Field label="Duration (days)" hint="1 day = 96 decisions of 15 min.">
          <input className="input mono" type="number" min={0.125} max={7} step={0.125} value={s.episode_days} onChange={(e) => setDraft({ settings: { episode_days: Number(e.target.value) } })} />
        </Field>
        <div className="sm:col-span-2 lg:col-span-4 flex items-center gap-3 pt-1">
          <button type="submit" className="btn btn-primary" disabled={busy}>{busy ? 'Creating session…' : 'Start session'}</button>
          <span className="text-[11.5px] text-fg-3">More environment and shield settings on the Configuration page.</span>
          {error && <span className="text-[12px] text-danger ml-auto">{error}</span>}
        </div>
      </form>
    </Card>
  )
}
