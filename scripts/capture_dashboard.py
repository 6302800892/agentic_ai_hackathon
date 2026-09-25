"""Capture reports/dashboard.png from the live Phoenix UI (optional helper).

Requires Phoenix to be running (e.g. `python -m src.cli run --keep-ui`, or `phoenix serve`) and Playwright:
    pip install playwright && python -m playwright install chromium
    python scripts/capture_dashboard.py --url http://localhost:6006

Without Playwright: open the Phoenix project page, show the latency / token / cost columns, and save a screenshot
as reports/dashboard.png. reports/dashboard_data.csv (scripts/dashboard.py) is the data behind it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dashboard.png"


def capture(url: str) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed - take the screenshot manually (see module docstring).")
        return False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(url.rstrip("/") + "/projects", wait_until="networkidle")
        link = page.get_by_text("patient-intake-copilot").first
        if link.count():
            link.click()
            page.wait_for_load_state("networkidle")
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"saved {OUT}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6006")
    sys.exit(0 if capture(ap.parse_args().url) else 1)
