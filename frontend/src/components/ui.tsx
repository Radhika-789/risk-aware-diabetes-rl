import { useEffect, useId, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/* ---------------------------------------------------------------- status states */

export function Loading({ label = 'Loading', rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div role="status" aria-live="polite" className="p-5 space-y-3">
      <div className="text-xs text-fg-3">{label}…</div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton h-4" style={{ width: `${85 - i * 17}%` }} />
      ))}
    </div>
  )
}

export function Empty({ title, hint, action }: { title: string; hint?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center px-6 py-14 gap-2">
      <div className="w-10 h-10 rounded-full border border-line-2 flex items-center justify-center text-fg-4 mb-1">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M4 7h16M4 12h10M4 17h7" /></svg>
      </div>
      <div className="text-sm font-medium text-fg">{title}</div>
      {hint && <div className="text-xs text-fg-3 max-w-md leading-relaxed">{hint}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export function ErrorState({ title = 'Something went wrong', detail, retry }: { title?: string; detail?: string; retry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center justify-center text-center px-6 py-12 gap-2">
      <div className="w-10 h-10 rounded-full border border-danger/40 text-danger flex items-center justify-center mb-1">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 8v5M12 16.5v.5" /><circle cx="12" cy="12" r="9" /></svg>
      </div>
      <div className="text-sm font-medium">{title}</div>
      {detail && <pre className="text-xs text-fg-3 whitespace-pre-wrap max-w-lg font-mono leading-relaxed">{detail}</pre>}
      {retry && <button className="btn btn-sm mt-2" onClick={retry}>Retry</button>}
    </div>
  )
}

/* ---------------------------------------------------------------- layout bits */

export function Card({ title, subtitle, actions, children, className = '', bodyClass = '' }:
  { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={`card flex flex-col min-w-0 ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 pt-3.5 pb-2.5">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold tracking-tight text-fg leading-5">{title}</h2>}
            {subtitle && <p className="text-[11.5px] text-fg-3 leading-4 mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-1.5 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={`px-4 pb-4 min-w-0 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, unit, sub, tone, title }:
  { label: ReactNode; value: ReactNode; unit?: string; sub?: ReactNode; tone?: string; title?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <div className="text-[11px] uppercase tracking-wide text-fg-3 leading-4 truncate">{label}</div>
      <div className="flex items-baseline gap-1 mt-0.5">
        <span className="mono text-[22px] leading-7 font-medium" style={tone ? { color: tone } : undefined}>{value}</span>
        {unit && <span className="text-[11px] text-fg-3">{unit}</span>}
      </div>
      {sub && <div className="text-[11px] text-fg-3 leading-4 truncate">{sub}</div>}
    </div>
  )
}

export function Tabs<T extends string>({ tabs, value, onChange, 'aria-label': ariaLabel }:
  { tabs: { id: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; 'aria-label'?: string }) {
  const onKey = (e: React.KeyboardEvent, i: number) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length
      onChange(tabs[j].id)
      ;(e.currentTarget.parentElement?.children[j] as HTMLElement)?.focus()
    }
  }
  return (
    <div className="tabs" role="tablist" aria-label={ariaLabel}>
      {tabs.map((t, i) => (
        <button key={t.id} role="tab" className="tab" aria-selected={value === t.id} tabIndex={value === t.id ? 0 : -1}
          onClick={() => onChange(t.id)} onKeyDown={(e) => onKey(e, i)}>{t.label}</button>
      ))}
    </div>
  )
}

export function Drawer({ open, onClose, title, children, width }:
  { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; width?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const id = useId()
  useEffect(() => {
    if (!open) return
    const prev = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); prev?.focus() }
  }, [open, onClose])
  if (!open) return null
  // Portal to <body>: an animated ancestor would otherwise become the containing block of the fixed drawer.
  return createPortal(
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div ref={ref} className="drawer" role="dialog" aria-modal="true" aria-labelledby={id} tabIndex={-1}
        style={width ? { width: `min(${width}px, 92vw)` } : undefined}>
        <header className="flex items-center justify-between px-4 h-12 border-b border-line shrink-0">
          <h2 id={id} className="text-[13px] font-semibold">{title}</h2>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">Esc</button>
        </header>
        <div className="flex-1 overflow-auto p-4">{children}</div>
      </div>
    </>,
    document.body,
  )
}

export function Collapsible({ title, children, defaultOpen = false, badge }:
  { title: ReactNode; children: ReactNode; defaultOpen?: boolean; badge?: ReactNode }) {
  return (
    <details className="group border-t border-line first:border-t-0" open={defaultOpen}>
      <summary className="flex items-center justify-between gap-2 cursor-pointer select-none py-2.5 text-[13px] font-medium text-fg-2 hover:text-fg list-none [&::-webkit-details-marker]:hidden">
        <span className="flex items-center gap-2">
          <svg className="transition-transform group-open:rotate-90 text-fg-4" width="10" height="10" viewBox="0 0 10 10"><path d="M3 1l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
          {title}
        </span>
        {badge}
      </summary>
      <div className="pb-3 pl-5">{children}</div>
    </details>
  )
}

export function Field({ label, children, hint, locked }: { label: string; children: ReactNode; hint?: ReactNode; locked?: boolean }) {
  return (
    <label className="block min-w-0">
      <span className="label flex items-center gap-1.5">{label}{locked && <span className="badge" style={{ height: 16, fontSize: 10 }} title="Fixed by the selected checkpoint (it defines the observation vector)">locked</span>}</span>
      {children}
      {hint && <span className="block text-[11px] text-fg-4 mt-1">{hint}</span>}
    </label>
  )
}

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div role="note" className={`flex items-start gap-2 rounded-md border border-warn/30 bg-warn/5 text-[11.5px] text-fg-2 ${compact ? 'px-2.5 py-1.5' : 'px-3 py-2'}`}>
      <span className="text-warn shrink-0 mt-px">⚠</span>
      <span>
        <strong className="text-fg">Research simulation only.</strong> SimGlucose virtual patients and an experimental DQN.
        Not a medical device, not clinically validated, and never insulin dosing advice.
      </span>
    </div>
  )
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { cls: string; label: string; dot?: boolean }> = {
    idle: { cls: 'badge', label: 'Idle' },
    running: { cls: 'badge badge-accent', label: 'Running', dot: true },
    paused: { cls: 'badge badge-warn', label: 'Paused' },
    completed: { cls: 'badge badge-ok', label: 'Completed' },
    terminated: { cls: 'badge badge-danger', label: 'Terminated by simulator' },
    error: { cls: 'badge badge-danger', label: 'Error' },
    queued: { cls: 'badge', label: 'Queued' },
    done: { cls: 'badge badge-ok', label: 'Done' },
  }
  const m = map[status] ?? { cls: 'badge', label: status }
  return <span className={m.cls}>{m.dot && <span className="live-dot" />}{m.label}</span>
}

export function SplitBadge({ split }: { split: string }) {
  const cls = split === 'test' ? 'badge badge-accent' : split === 'val' ? 'badge badge-warn' : 'badge'
  const label = split === 'test' ? 'held-out test' : split === 'val' ? 'validation' : split === 'train' ? 'training' : split
  return <span className={cls} title="Patient split: train is used for learning, val for checkpoint selection, test is never seen before final evaluation">{label}</span>
}

export function Hint({ text, children }: { text: string; children: ReactNode }) {
  return <span title={text} className="underline decoration-dotted decoration-fg-4 underline-offset-2 cursor-help">{children}</span>
}
