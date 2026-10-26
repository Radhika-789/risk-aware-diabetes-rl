import { useEffect, useState } from 'react'
import type { SessionState } from '../../api/types'

interface Props {
  s: SessionState
  busy: boolean
  onRun: (speed: number) => void
  onPause: () => void
  onStep: (n: number) => void
  onReset: (seed?: number) => void
  onNew: () => void
}

const SPEEDS = [0.5, 1, 2, 5, 10, 25]

export function Controls({ s, busy, onRun, onPause, onStep, onReset, onNew }: Props) {
  const [speed, setSpeed] = useState(s.speed || 2)
  const over = s.status === 'completed' || s.status === 'terminated' || s.status === 'error'
  const running = s.status === 'running'

  // keyboard: Space run/pause, N step, Shift+N step x4, R reset
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(t.tagName) || t.isContentEditable) return
      if (e.code === 'Space') { e.preventDefault(); if (running) onPause(); else if (!over) onRun(speed) }
      else if (e.key === 'n' && !running && !over) onStep(1)
      else if (e.key === 'N' && !running && !over) onStep(4)
      else if (e.key === 'r' && !running) onReset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [running, over, speed, onPause, onRun, onStep, onReset])

  return (
    <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Simulation controls">
      {running ? (
        <button className="btn btn-primary" onClick={onPause} disabled={busy} title="Pause (Space)">
          <Icon d="M7 5h3v14H7zM14 5h3v14h-3z" /> Pause
        </button>
      ) : (
        <button className="btn btn-primary" onClick={() => onRun(speed)} disabled={busy || over} title="Run (Space)">
          <Icon d="M7 4l12 8-12 8z" /> {s.status === 'paused' ? 'Resume' : 'Run'}
        </button>
      )}
      <button className="btn" onClick={() => onStep(1)} disabled={busy || running || over} title="Advance one 15-minute decision (N)">
        <Icon d="M5 4l9 8-9 8zM16 4h3v16h-3z" /> Step
      </button>
      <button className="btn" onClick={() => onStep(4)} disabled={busy || running || over} title="Advance one hour (Shift+N)">×4</button>
      <label className="inline-flex items-center gap-1.5 text-[12px] text-fg-3 ml-1">
        Speed
        <select className="select !w-auto !h-[34px]" value={speed} onChange={(e) => { const v = Number(e.target.value); setSpeed(v); if (running) onRun(v) }} aria-label="Run speed, decisions per second">
          {SPEEDS.map((v) => <option key={v} value={v}>{v}×/s</option>)}
        </select>
      </label>
      <span className="flex-1" />
      <button className="btn btn-ghost" onClick={() => onReset()} disabled={busy || running} title="Reset this episode with the same seed (R)">
        <Icon d="M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5" outline /> Reset
      </button>
      <button className="btn btn-ghost" onClick={onNew} disabled={busy} title="Discard this session and set up a new one">New session</button>
    </div>
  )
}

function Icon({ d, outline = false }: { d: string; outline?: boolean }) {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill={outline ? 'none' : 'currentColor'} stroke="currentColor" strokeWidth={outline ? 1.8 : 0.5} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
}
