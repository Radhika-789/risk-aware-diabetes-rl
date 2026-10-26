/** Generic multi-series line chart (one y-axis, shared x) used in the Analytics Lab for training curves. */
import { useMemo } from 'react'
import { CartesianGrid, Line, LineChart as RLineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export interface XYSeries { id: string; label: string; color: string; x: number[]; y: (number | null)[]; dashed?: boolean }
export interface DotMark { x: number; y: number; label: string; color?: string }

interface Props {
  series: XYSeries[]
  height?: number
  xLabel?: string
  yLabel?: string
  yFormat?: (v: number) => string
  xFormat?: (v: number) => string
  yDomain?: [number | 'auto', number | 'auto']
  marks?: DotMark[]
  logY?: boolean
}

export function LineChart({ series, height = 300, xLabel, yLabel, yFormat = (v) => `${v}`, xFormat = (v) => `${v}`, yDomain = ['auto', 'auto'], marks = [], logY = false }: Props) {
  const data = useMemo(() => {
    const rows = new Map<number, Record<string, number | null>>()
    for (const s of series) {
      for (let i = 0; i < s.x.length; i++) {
        const r = rows.get(s.x[i]) ?? { x: s.x[i] }
        r[s.id] = s.y[i]
        rows.set(s.x[i], r)
      }
    }
    return [...rows.values()].sort((a, b) => (a.x as number) - (b.x as number))
  }, [series])

  return (
    <div className="w-full" style={{ height }}>
      {yLabel && <div className="text-[10.5px] text-fg-4 pl-1 -mb-2">{yLabel}</div>}
      <ResponsiveContainer width="100%" height="100%">
        <RLineChart data={data} margin={{ top: 10, right: 16, bottom: xLabel ? 18 : 4, left: 0 }}>
          <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tickFormatter={xFormat}
            tick={{ fill: 'var(--color-fg-3)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'var(--color-line)' }}
            label={xLabel ? { value: xLabel, position: 'insideBottom', dy: 14, fill: 'var(--color-fg-4)', fontSize: 10 } : undefined} />
          <YAxis domain={yDomain} scale={logY ? 'log' : 'auto'} allowDataOverflow={logY} tickFormatter={yFormat} width={52}
            tick={{ fill: 'var(--color-fg-3)', fontSize: 11 }} tickLine={false} axisLine={false} />
          <Tooltip isAnimationActive={false} cursor={{ stroke: 'var(--color-line-2)' }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              return (
                <div className="chart-tooltip">
                  <div className="text-fg-3 mb-1">{xLabel ?? 'x'} {xFormat(label as number)}</div>
                  {payload.filter((p) => p.value != null).map((p) => {
                    const s = series.find((x) => x.id === p.dataKey)
                    return (
                      <div key={p.dataKey as string} className="flex items-center justify-between gap-4">
                        <span className="inline-flex items-center gap-1.5 text-fg-2"><span className="w-2 h-2 rounded-sm" style={{ background: s?.color }} />{s?.label}</span>
                        <span className="mono text-fg">{yFormat(p.value as number)}</span>
                      </div>
                    )
                  })}
                </div>
              )
            }} />
          {series.map((s) => (
            <Line key={s.id} type="monotone" dataKey={s.id} name={s.label} stroke={s.color} strokeWidth={1.8}
              strokeDasharray={s.dashed ? '5 4' : undefined} dot={false} activeDot={{ r: 3.5, strokeWidth: 0 }} isAnimationActive={false} connectNulls />
          ))}
          {marks.map((m, i) => (
            <ReferenceDot key={i} x={m.x} y={m.y} r={5} fill={m.color ?? 'var(--color-accent)'} stroke="var(--color-bg)" strokeWidth={2}
              label={{ value: m.label, position: 'top', fill: 'var(--color-fg-2)', fontSize: 10 }} />
          ))}
        </RLineChart>
      </ResponsiveContainer>
      {series.length > 1 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 text-[11px] text-fg-3 -mt-1">
          {series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5">
              <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '4 3' : undefined} /></svg>{s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
