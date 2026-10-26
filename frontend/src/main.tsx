import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import { AppShell } from './app/AppShell'
import { LiveSimulation } from './pages/LiveSimulation'
import { AnalyticsLab } from './pages/AnalyticsLab'
import { Experiments } from './pages/Experiments'
import { Configuration } from './pages/Configuration'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 10_000 } } })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<LiveSimulation />} />
            <Route path="analytics" element={<AnalyticsLab />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="config" element={<Configuration />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
