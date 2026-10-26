import { usePolicyOptions } from '../hooks/useCatalog'

export function PolicySelect({ value, onChange, id, disabled, exclude = [] }:
  { value: string; onChange: (v: string) => void; id?: string; disabled?: boolean; exclude?: string[] }) {
  const { options, isLoading } = usePolicyOptions()
  const groups = ['Baselines', 'DQN checkpoints'] as const
  return (
    <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled || isLoading}>
      {groups.map((g) => {
        const opts = options.filter((o) => o.group === g && !exclude.includes(o.id))
        return (
          <optgroup key={g} label={g}>
            {opts.length === 0 && <option disabled value="">{g === 'DQN checkpoints' ? 'none on disk (train one with train_dqn.py)' : 'none'}</option>}
            {opts.map((o) => <option key={o.id} value={o.id}>{o.label}{o.demo ? '  [demo]' : ''}</option>)}
          </optgroup>
        )
      })}
    </select>
  )
}
