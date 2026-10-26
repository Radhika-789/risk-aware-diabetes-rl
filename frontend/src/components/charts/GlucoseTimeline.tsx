import { useMemo } from 'react'
import {
  ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Scatter,
} from 'recharts'
import { fmt, THRESH, glucoseZone, ZONE_COLOR } from '../../lib/format'

export interface TimelineSeries {
  id: string
  label: string
  color: string
  t_min: number[]
  values: number[]
  /** 'live' = actual simulated samples; 'counterfactual' = branch/rollout future; 'historical' = finished episode */
  kind: 'live' | 'counterfactual' | 'historical'
  width?: number
}

export interface Marker { t_min: number; kind: 'meal' | 'suspend' | 'bolus_blocked' | 'terminated'; label: string }

interface Props {
  series: TimelineSeries[]
  /** x-axis extent in minutes (fixed so the chart does not rescale while an episode fills in) */
  totalMin: number
  markers?: Marker[]
  branchAt?: number | null
  height?: number
  yDomain?: [number, number]
  unit?: string
  showLegend?: boolean
}

const MARK_COLOR: Record<Marker['kind'], string> = {
  meal: '#3f86da', suspend: '#b08a2c', bolus_blocked: '#d85c63', terminated: '#d85c63',
}

function xTicks(totalMin: number) {
  const step = totalMin <= 6 * 60 ? 60 : totalMin <= 48 * 60 ? 180 : 720
  const out: number[] = []
  for (let t = 0; t <= totalMin; t += step) out.push(t)
  return out
}

export function GlucoseTimeline({ series, totalMin, markers = [], branchAt, height = 320, yDomain = [40, 400], unit = 'mg/dL', showLegend = true }: Props) {
  void unit // unit is stated in the surrounding card copy; axis stays uncluttered
  const data = useMemo(() => {
    const rows = new Map<number, Record<string, number>>()
    for (const s of series) {
      for (let i = 0; i < s.t_min.length; i++) {
        const t = s.t_min[i]
        const r = rows.get(t) ?? { t }
        r[s.id] = s.values[i]
        rows.set(t, r)
      }
    }
    return [...rows.values()].sort((a, b) => a.t - b.t)
  }, [series])

  const markerData = useMemo(() => markers.map((m) => ({ t: m.t_min, y: yDomain[1] - 12, kind: m.kind, label: m.label })), [markers, yDomain])
  const ticks = useMemo(() => xTicks(totalMin), [totalMin])

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <ReferenceArea y1={THRESH.low} y2={THRESH.high} fill="var(--color-in-range)" fillOpacity={0.06} ifOverflow="hidden" />
          <ReferenceLine y={THRESH.severeLow} stroke="var(--color-severe-low)" strokeDasharray="3 4" strokeOpacity={0.8}
            label={{ value: '54', position: 'insideLeft', fill: 'var(--color-severe-low)', fontSize: 10, dx: 2 }} />
          <ReferenceLine y={THRESH.low} stroke="var(--color-low)" strokeDasharray="2 4" strokeOpacity={0.7}
            label={{ value: '70', position: 'insideLeft', fill: 'var(--color-low)', fontSize: 10, dx: 2 }} />
          <ReferenceLine y={THRESH.high} stroke="var(--color-high)" strokeDasharray="2 4" strokeOpacity={0.7}
            label={{ value: '180', position: 'insideLeft', fill: 'var(--color-high)', fontSize: 10, dx: 2 }} />
          <ReferenceLine y={THRESH.veryHigh} stroke="var(--color-very-high)" strokeDasharray="3 4" strokeOpacity={0.5}
            label={{ value: '250', position: 'insideLeft', fill: 'var(--color-very-high)', fontSize: 10, dx: 2 }} />
          {branchAt != null && (
            <ReferenceLine x={branchAt} stroke="var(--color-fg-3)" strokeDasharray="4 3"
              label={{ value: 'branch', position: 'insideTopRight', fill: 'var(--color-fg-3)', fontSize: 10 }} />
          )}
          <XAxis dataKey="t" type="number" domain={[0, totalMin]} ticks={ticks} tickFormatter={(v) => fmt.clock(v)}
            stroke="var(--color-line-2)" tick={{ fill: 'var(--color-fg-3)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'var(--color-line)' }} />
          <YAxis domain={yDomain} ticks={[50, 100, 150, 200, 250, 300, 350, 400].filter((v) => v >= yDomain[0] && v <= yDomain[1])}
            stroke="var(--color-line-2)" tick={{ fill: 'var(--color-fg-3)', fontSize: 11 }} tickLine={false} axisLine={false} width={44} unit="" />
          <Tooltip content={<TimelineTooltip series={series} />} cursor={{ stroke: 'var(--color-line-2)', strokeWidth: 1 }} isAnimationActive={false} />
          {series.map((s) => (
            <Line key={s.id} type="monotone" dataKey={s.id} stroke={s.color} strokeWidth={s.width ?? (s.kind === 'counterfactual' ? 1.6 : 2)}
              strokeDasharray={s.kind === 'counterfactual' ? '5 4' : undefined} strokeOpacity={s.kind === 'counterfactual' ? 0.9 : 1}
              dot={false} activeDot={{ r: 4, strokeWidth: 0 }} isAnimationActive={false} connectNulls={false} name={s.label} />
          ))}
          {markerData.length > 0 && (
            <Scatter data={markerData} dataKey="y" isAnimationActive={false}
              shape={(p: { cx?: number; cy?: number; payload?: { kind: Marker['kind'] } }) => (
                <MarkerGlyph cx={p.cx ?? 0} cy={p.cy ?? 0} kind={p.payload?.kind ?? 'meal'} />
              )} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
      {showLegend && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-2 -mt-1 text-[11px] text-fg-3">
          {series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5">
              <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.kind === 'counterfactual' ? '4 3' : undefined} /></svg>
              {s.label}
              {s.kind === 'counterfactual' && <span className="text-fg-4">(counterfactual)</span>}
            </span>
          ))}
          {markers.length > 0 && (
            <>
              <span className="inline-flex items-center gap-1.5"><MarkerGlyph cx={5} cy={5} kind="meal" inline /> meal</span>
              <span className="inline-flex items-center gap-1.5"><MarkerGlyph cx={5} cy={5} kind="suspend" inline /> shield suspend</span>
              <span className="inline-flex items-center gap-1.5"><MarkerGlyph cx={5} cy={5} kind="bolus_blocked" inline /> bolus blocked</span>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function MarkerGlyph({ cx, cy, kind, inline = false }: { cx: number; cy: number; kind: Marker['kind']; inline?: boolean }) {
  const c = MARK_COLOR[kind]
  const g = kind === 'meal' ? <circle cx={cx} cy={cy} r={3.2} fill={c} />
    : kind === 'suspend' ? <rect x={cx - 3} y={cy - 3} width={6} height={6} fill={c} />
    : <path d={`M${cx} ${cy - 4} L${cx + 3.6} ${cy + 2.5} L${cx - 3.6} ${cy + 2.5} Z`} fill={c} />
  return inline ? <svg width="10" height="10">{g}</svg> : g
}

function TimelineTooltip({ active, payload, label, series }: { active?: boolean; payload?: { dataKey?: string; value?: number }[]; label?: number; series: TimelineSeries[] }) {
  if (!active || !payload?.length || label == null) return null
  const rows = payload.filter((p) => typeof p.value === 'number' && p.dataKey !== 'y')
  if (!rows.length) return null
  return (
    <div className="chart-tooltip">
      <div className="text-fg-3 mb-1">t = {fmt.clock(label)}</div>
      {rows.map((p) => {
        const s = series.find((x) => x.id === p.dataKey)
        const v = p.value as number
        return (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <span className="inline-flex items-center gap-1.5 text-fg-2">
              <span className="w-2 h-2 rounded-sm" style={{ background: s?.color }} />{s?.label ?? p.dataKey}
              {s?.kind === 'counterfactual' && <span className="text-fg-4">cf</span>}
            </span>
            <span className="mono" style={{ color: ZONE_COLOR[glucoseZone(v)] }}>{Math.round(v)}</span>
          </div>
        )
      })}
    </div>
  )
}
