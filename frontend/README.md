# GlucoGuard research workstation (frontend)

Vite + React 19 + TypeScript, react-router, Recharts, TanStack Query, Tailwind v4. Research simulation only.

## Run

```bash
# 1. API (repo root; backend/ must sit at the repo root, next to t1d_env.py)
pip install -r requirements.txt fastapi "uvicorn[standard]"
uvicorn backend.app:app --reload --port 8000

# 2. UI
cd frontend && npm install
npm run dev          # http://localhost:5173, /api proxied to :8000
npm run build        # writes dist/, which the FastAPI process serves at http://localhost:8000
```

Pages: `/` Live Simulation · `/analytics` Analytics Lab · `/experiments` Experiments · `/config` Configuration & Model.
Keyboard: `Space` run/pause, `N` step, `Shift+N` step ×4, `R` reset, `[` toggle sidebar, `Esc` close drawers.

## Tests

```bash
python -m pytest backend/tests -q             # API
python frontend/e2e/ui_smoke.py               # headless Chromium against the built UI (needs playwright)
```

Evaluation directories are picked up from results*/ at the root and from runs/eval_*/ (any runs/ folder with summary.csv
and no train_log.csv). Nothing in the UI is synthesized: every glucose value, action, metric and curve comes from a completed SimGlucose
step or a file written by `train_dqn.py` / `evaluate.py`. Runs named `demo*` are short pipeline checks and are
labelled as such.
