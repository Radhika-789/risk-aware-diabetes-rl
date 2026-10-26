import { useMeta } from '../hooks/useCatalog'

export function PatientSelect({ value, onChange, id, disabled }: { value: string; onChange: (v: string) => void; id?: string; disabled?: boolean }) {
  const meta = useMeta()
  const p = meta.data?.patients
  return (
    <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled || !p}>
      {p && (
        <>
          <optgroup label="Held-out test (never seen in training or selection)">{p.test.map((x) => <option key={x} value={x}>{x}</option>)}</optgroup>
          <optgroup label="Validation (used for checkpoint selection)">{p.val.map((x) => <option key={x} value={x}>{x}</option>)}</optgroup>
          <optgroup label="Training">{p.train.map((x) => <option key={x} value={x}>{x}</option>)}</optgroup>
        </>
      )}
    </select>
  )
}
