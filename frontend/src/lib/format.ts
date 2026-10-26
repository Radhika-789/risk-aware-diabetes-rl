export const fmt = {
  n: (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(d)),
  pct: (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? '—' : `${v.toFixed(d)}%`),
  mgdl: (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v)}`),
  units: (v: number | null | undefined, d = 2) => (v == null ? '—' : `${v.toFixed(d)} U`),
  /** minutes since start -> "1d 03:15" / "03:15" */
  clock: (min: number) => {
    const d = Math.floor(min / 1440)
    const h = Math.floor((min % 1440) / 60)
    const m = Math.round(min % 60)
    const hm = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
    return d > 0 ? `${d}d ${hm}` : hm
  },
  hours: (min: number) => `${(min / 60).toFixed(min % 60 === 0 ? 0 : 1)} h`,
  simTime: (iso: string) => {
    const d = new Date(iso)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  },
  bytes: (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`),
  ago: (unixSeconds: number) => {
    const s = Date.now() / 1000 - unixSeconds
    if (s < 60) return 'just now'
    if (s < 3600) return `${Math.floor(s / 60)} min ago`
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`
    return `${Math.floor(s / 86400)} d ago`
  },
}

/** Human label for a policy id. */
export function policyLabel(id: string): string {
  if (id.startsWith('ckpt:')) return id.slice(5).replace('.pt', '')
  return id.replace('_', ' ')
}

export const THRESH = { severeLow: 54, low: 70, high: 180, veryHigh: 250 }

export function glucoseZone(v: number): 'severe_low' | 'low' | 'in_range' | 'high' | 'very_high' {
  if (v < THRESH.severeLow) return 'severe_low'
  if (v < THRESH.low) return 'low'
  if (v <= THRESH.high) return 'in_range'
  if (v <= THRESH.veryHigh) return 'high'
  return 'very_high'
}

export const ZONE_COLOR: Record<ReturnType<typeof glucoseZone>, string> = {
  severe_low: 'var(--color-severe-low)',
  low: 'var(--color-low)',
  in_range: 'var(--color-in-range)',
  high: 'var(--color-high)',
  very_high: 'var(--color-very-high)',
}

export const ZONE_LABEL: Record<ReturnType<typeof glucoseZone>, string> = {
  severe_low: '< 54 severe low',
  low: '< 70 low',
  in_range: '70–180 in range',
  high: '> 180 high',
  very_high: '> 250 very high',
}

/** Decode a discrete action index into (basal multiplier, bolus fraction of TDI). */
export function decodeAction(a: number, basalMults: number[], bolusFracs: number[]) {
  if (a < 0) return { basalMult: null as number | null, bolusFrac: null as number | null }
  const nb = bolusFracs.length
  return { basalMult: basalMults[Math.floor(a / nb)], bolusFrac: bolusFracs[a % nb] }
}

export function actionLabel(a: number, basalMults: number[], bolusFracs: number[], tdi?: number) {
  if (a < 0) return 'continuous dose'
  const { basalMult, bolusFrac } = decodeAction(a, basalMults, bolusFracs)
  const bolus = bolusFrac == null ? '' : tdi ? `${(bolusFrac * tdi).toFixed(2)} U` : `${(bolusFrac * 100).toFixed(0)}% TDI`
  return `basal ×${basalMult} · bolus ${bolusFrac === 0 ? '0' : bolus}`
}

/** Series colours for comparing policies (restrained, distinguishable on dark). */
// Validated with the dataviz palette checker against the dark card surface (#141c25): lightness band,
// chroma floor, CVD separation, normal-vision floor and contrast all pass. Assign in fixed order, never cycle.
export const SERIES = ['#1fa897', '#b08a2c', '#9470e8', '#c77a3c', '#3f86da', '#d85c63']
export const seriesColor = (i: number) => SERIES[i % SERIES.length]
