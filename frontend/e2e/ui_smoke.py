"""Browser smoke test for the GlucoGuard UI (Playwright, headless Chromium).

Prerequisites: `npm run build` in frontend/, a running API  (uvicorn backend.app:app --port 8000)  from the repo root,
and at least one checkpoint in runs/ (the test falls back to baselines when there is none).

    python frontend/e2e/ui_smoke.py
"""
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "ERR_TUNNEL" not in m.text and "fonts" not in m.text else None)

    # ---- Live Simulation: create, step, run/pause, keyboard, compare, drawer, complete, reset
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.evaluate("localStorage.clear()")
    pg.reload(wait_until="networkidle")
    opts = pg.eval_on_selector_all("select >> nth=0 >> option", "els => els.map(e => e.value)")
    policy = next((o for o in opts if o.startswith("ckpt:")), "cgm_rule")
    pg.select_option("select >> nth=0", policy)
    pg.fill("input[type=number] >> nth=1", "0.25")
    pg.click("text=Start session")
    pg.wait_for_selector("text=SimGlucose virtual patient")
    ok("session created idle", pg.inner_text(".badge >> nth=0") == "Idle")
    pg.click("button:has-text('Step')")
    pg.wait_for_selector("text=Decisions (1)")
    pg.click("button:has-text('×4')")
    pg.wait_for_selector("text=Decisions (5)")
    ok("manual stepping advances exactly N decisions", True)
    pg.click("button:has-text('Resume')")
    pg.wait_for_selector("text=Running")
    time.sleep(1.5)
    pg.click("button:has-text('Pause')")
    pg.wait_for_selector("text=Paused")
    n1 = pg.inner_text("[role=tab]:has-text('Decisions')")
    time.sleep(1.2)
    ok("pause holds the clock", n1 == pg.inner_text("[role=tab]:has-text('Decisions')"))
    pg.keyboard.press("n")
    pg.wait_for_function(f"document.querySelector(\"[role=tab][aria-selected]\") && document.body.innerText.indexOf('Decisions ({int(n1.split('(')[1][:-1]) + 1})') >= 0")
    ok("keyboard N steps once", True)
    pg.click("[role=tab]:has-text('Baseline vs candidate')")
    pg.click("button:has-text('Branch at')")
    pg.wait_for_selector("text=Counterfactual future from", timeout=90000)
    ok("branch comparison renders counterfactual futures", pg.locator("text=(counterfactual)").count() >= 1)
    pg.click("[role=radio]:has-text('Paired full episode')")
    pg.click("button:has-text('Run paired episode')")
    pg.wait_for_selector("text=Range distribution (full episode)", timeout=180000)
    ok("paired full-episode comparison renders", True)
    pg.click("[role=tab]:has-text('Decisions')")
    pg.click("tbody tr >> nth=0")
    pg.wait_for_selector("[role=dialog]")
    pg.keyboard.press("Escape")
    time.sleep(0.3)
    ok("decision drawer opens and closes with Esc", pg.query_selector("[role=dialog]") is None)
    pg.click("[role=tab]:has-text('Live timeline')")
    pg.select_option("select[aria-label*='speed']", "25")
    pg.keyboard.press("Space")
    pg.wait_for_selector("text=Episode complete", timeout=120000)
    ok("runs to completion", True)
    pg.keyboard.press("r")
    pg.wait_for_selector("text=Decisions (0)")
    ok("reset returns to idle", pg.inner_text(".badge >> nth=0") == "Idle")

    # ---- other pages
    pg.goto(BASE + "/analytics", wait_until="networkidle")
    for t in ("Time in range", "Training", "Validation progress", "Glucose trajectory"):
        pg.click(f"[role=tab]:has-text('{t}')")
        time.sleep(0.6)
    ok("analytics tabs render", pg.locator("[role=tab]").count() == 4)
    pg.goto(BASE + "/experiments", wait_until="networkidle")
    for t in ("Evaluation results", "Compare checkpoints", "Training runs"):
        pg.click(f"[role=tab]:has-text('{t}')")
        time.sleep(0.5)
    ok("experiments tabs render", True)
    pg.goto(BASE + "/config", wait_until="networkidle")
    ok("configuration page renders", pg.locator("text=Checkpoint identity").count() == 1)
    ok("research disclaimer visible on every page", pg.locator("text=Research simulation only").count() >= 1)

    # ---- responsive
    pg.set_viewport_size({"width": 390, "height": 844})
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.click("[aria-label='Open navigation']")
    time.sleep(0.4)
    ok("mobile navigation drawer opens", pg.locator("nav[aria-label=Primary]:visible").count() >= 1)
    ok("no horizontal overflow at 390px", pg.evaluate("document.documentElement.scrollWidth <= 390"))
    pg.set_viewport_size({"width": 1440, "height": 1000})
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.keyboard.press("[")
    time.sleep(0.4)
    ok("sidebar collapses with [", pg.evaluate("document.querySelector('aside').offsetWidth") < 80)
    ok("no page errors", not errs)
    if errs:
        print(errs[:5])
    b.close()

failed = [n for n, c in checks if not c]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
