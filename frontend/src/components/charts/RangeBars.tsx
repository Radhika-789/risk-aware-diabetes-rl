/**
 * Time-in-range distribution for one or more policies as horizontal stacked bars:
 * <54 | 54–70 | 70–180 | 180–250 | >250. The five glucose metrics are shown exactly, with gaps between segments.
 */
import type { EpisodeMetrics } from '../../api/types'
import { fmt } from '../../lib/format'

export interface RangeRow { label: string; color?: string; m: Partial<EpisodeMetrics>; sub?: string }

const SEGS = [
  { key: 'severe_low', label: '< 54', color: 'var(--color-severe-low)' },
  { key: 'low', label: '54–70', color: 'var(--color-low)' },
  { key: 'in_range', label: '70–180', color: 'var(--color-in-range)' },
  { key: 'high', label: '180–250', color: 'var(--color-high)' },
  { key: 'very_high', label: '> 250', color: 'var(--color-very-high)' },
] as const

export function splitRanges(m: Partial<EpisodeMetrics>) {
  const lt54 = m.tbr_lt54 ?? 0, lt70 = m.tbr_lt70 ?? 0, gt180 = m.tar_gt180 ?? 0, gt250 = m.tar_gt250 ?? 0
  return {
    severe_low: lt54, low: Math.max(0, lt70 - lt54), in_range: m.tir_70_180 ?? 0,
    high: Math.max(0, gt180 - gt250), very_high: gt250,
  }
}

export function RangeBars({ rows, compact = false }: { rows: RangeRow[]; compact?: boolean }) {
  return (
    <div className="space-y-2.5">
      {rows.map((r, i) => {
        const v = splitRanges(r.m)
        const has = r.m.tir_70_180 != null
        return (
          <div key={i}>
            <div className="flex items-baseline justify-between gap-2 mb-1">
              <span className="text-[12px] text-fg-2 inline-flex items-center gap-1.5 min-w-0">
                {r.color && <span className="w-2 h-2 rounded-sm shrink-0" style={{ background: r.color }} />}
                <span className="truncate">{r.label}</span>
                {r.sub && <span className="text-fg-4">{r.sub}</span>}
              </span>
              <span className="mono text-[12px] text-fg-3">TIR {fmt.pct(r.m.tir_70_180)}</span>
            </div>
            <div className={`flex w-full ${compact ? 'h-2' : 'h-3'} gap-[2px] rounded-sm overflow-hidden bg-bg-1`} role="img"
              aria-label={has ? SEGS.map((s) => `${s.label}: ${v[s.key].toFixed(1)}%`).join(', ') : 'no data'}>
              {has ? SEGS.map((s) => v[s.key] > 0 && (
                <div key={s.key} style={{ width: `${v[s.key]}%`, background: s.color }} title={`${s.label} mg/dL: ${v[s.key].toFixed(1)}%`} className="rounded-[2px] transition-[width] duration-300" />
              )) : <div className="w-full skeleton h-full" />}
            </div>
          </div>
        )
      })}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-fg-3 pt-1">
        {SEGS.map((s) => <span key={s.key} className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-sm" style={{ background: s.color }} />{s.label}</span>)}
      </div>
    </div>
  )
}

export function FiveMetrics({ m, scopeNote }: { m: Partial<EpisodeMetrics>; scopeNote?: string }) {
  const cells: { k: keyof EpisodeMetrics; label: string; title: string; tone: string }[] = [
    { k: 'tir_70_180', label: 'TIR', title: 'Time in range 70–180 mg/dL', tone: 'var(--color-in-range)' },
    { k: 'tbr_lt70', label: '< 70', title: 'Time below range, under 70 mg/dL', tone: 'var(--color-low)' },
    { k: 'tbr_lt54', label: '< 54', title: 'Time below range, under 54 mg/dL (severe)', tone: 'var(--color-severe-low)' },
    { k: 'tar_gt180', label: '> 180', title: 'Time above range, over 180 mg/dL', tone: 'var(--color-high)' },
    { k: 'tar_gt250', label: '> 250', title: 'Time above range, over 250 mg/dL', tone: 'var(--color-very-high)' },
  ]
  return (
    <div>
      <div className="grid grid-cols-5 gap-2">
        {cells.map((c) => (
          <div key={c.k} className="min-w-0" title={c.title}>
            <div className="text-[10.5px] uppercase tracking-wide text-fg-3 whitespace-nowrap">{c.label}</div>
            <div className="mono text-[15px] font-medium" style={{ color: c.tone }}>{fmt.pct(m[c.k] as number | undefined)}</div>
          </div>
        ))}
      </div>
      {scopeNote && <div className="text-[11px] text-fg-4 mt-1.5">{scopeNote}</div>}
    </div>
  )
}
