import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Disclaimer } from '../components/ui'

const NAV = [
  { to: '/', label: 'Live Simulation', key: '1', icon: <path d="M3 12h3l3-7 4 14 3-7h5" /> },
  { to: '/analytics', label: 'Analytics Lab', key: '2', icon: <path d="M4 20V10M10 20V4M16 20v-8M22 20H2" /> },
  { to: '/experiments', label: 'Experiments', key: '3', icon: <path d="M9 3v6L4 19h16L15 9V3M8 3h8" /> },
  { to: '/config', label: 'Configuration & Model', key: '4', icon: <><circle cx="12" cy="12" r="3" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" /></> },
]

export function AppShell() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('gg.sidebar') === '1')
  const [mobileOpen, setMobileOpen] = useState(false)
  const loc = useLocation()
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 5000, retry: 0 })

  useEffect(() => { localStorage.setItem('gg.sidebar', collapsed ? '1' : '0') }, [collapsed])
  useEffect(() => { setMobileOpen(false) }, [loc.pathname])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '[' && !e.metaKey && !e.ctrlKey && (e.target as HTMLElement).tagName !== 'INPUT') setCollapsed((c) => !c)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const online = health.isSuccess
  const current = NAV.find((n) => n.to === loc.pathname) ?? NAV[0]

  const nav = (
    <nav aria-label="Primary" className="flex flex-col gap-0.5 px-2">
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} end={n.to === '/'} title={collapsed ? n.label : undefined}
          className={({ isActive }) =>
            `group flex items-center gap-2.5 h-9 rounded-md px-2.5 text-[13px] transition-colors ${
              isActive ? 'bg-surface-2 text-fg' : 'text-fg-2 hover:bg-surface-2/70 hover:text-fg'}`}>
          {({ isActive }) => (
            <>
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
                className={`shrink-0 ${isActive ? 'text-accent' : 'text-fg-3 group-hover:text-fg-2'}`}>{n.icon}</svg>
              {!collapsed && <span className="truncate">{n.label}</span>}
              {!collapsed && <kbd className="kbd ml-auto opacity-0 group-hover:opacity-100 transition-opacity">{n.key}</kbd>}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )

  const sidebar = (
    <aside className={`flex flex-col h-full bg-bg-1 border-r border-line transition-[width] duration-200 ease-out ${collapsed ? 'w-[60px]' : 'w-[232px]'}`}>
      <div className={`flex items-center gap-2.5 h-14 px-3.5 border-b border-line ${collapsed ? 'justify-center' : ''}`}>
        <span className="w-7 h-7 rounded-md bg-gradient-to-br from-accent/80 to-accent-2/60 flex items-center justify-center text-bg font-semibold text-[13px] shrink-0">G</span>
        {!collapsed && (
          <div className="min-w-0 leading-tight">
            <div className="text-[13px] font-semibold tracking-tight">GlucoGuard</div>
            <div className="text-[10.5px] text-fg-3 uppercase tracking-wider">Research workstation</div>
          </div>
        )}
      </div>
      <div className="py-3 flex-1">{nav}</div>
      <div className="px-2 pb-3 space-y-2">
        <div className={`flex items-center gap-2 px-2.5 h-8 text-[11.5px] text-fg-3 ${collapsed ? 'justify-center' : ''}`} title={online ? 'Backend connected' : 'Backend unreachable'}>
          <span className={`w-2 h-2 rounded-full ${online ? 'bg-ok' : 'bg-danger'}`} />
          {!collapsed && <span>{online ? 'API connected' : 'API offline'}</span>}
        </div>
        <button className="btn btn-ghost btn-sm w-full justify-center hidden md:flex" onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-expanded={!collapsed} title="Toggle sidebar  [">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={`transition-transform ${collapsed ? 'rotate-180' : ''}`}><path d="M15 6l-6 6 6 6" /></svg>
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  )

  return (
    <div className="h-full flex">
      <div className="hidden md:block h-full shrink-0">{sidebar}</div>
      {mobileOpen && (
        <div className="md:hidden">
          <div className="drawer-backdrop" onClick={() => setMobileOpen(false)} />
          <div className="fixed inset-y-0 left-0 z-50 w-[232px] shadow-2xl">{sidebar}</div>
        </div>
      )}
      <div className="flex-1 min-w-0 flex flex-col h-full">
        <header className="h-14 shrink-0 border-b border-line flex items-center gap-3 px-4 bg-bg/80 backdrop-blur">
          <button className="btn btn-icon btn-ghost md:hidden" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
          </button>
          <h1 className="text-[14px] font-semibold tracking-tight">{current.label}</h1>
          <div className="ml-auto hidden sm:block max-w-xl"><Disclaimer compact /></div>
        </header>
        <div className="sm:hidden px-3 pt-3"><Disclaimer compact /></div>
        <main className="flex-1 min-h-0 overflow-auto">
          {health.isError ? (
            <div className="p-6 max-w-xl">
              <div className="card p-5">
                <div className="text-sm font-semibold mb-1">Backend unreachable</div>
                <p className="text-xs text-fg-3 leading-relaxed">
                  The UI needs the FastAPI server. From the repository root run:
                </p>
                <pre className="mono text-xs bg-bg-1 border border-line rounded-md p-3 mt-2">uvicorn backend.app:app --reload --port 8000</pre>
                <p className="text-xs text-fg-3 mt-2">Retrying every 5 seconds.</p>
              </div>
            </div>
          ) : (
            <Outlet />
          )}
        </main>
      </div>
    </div>
  )
}
