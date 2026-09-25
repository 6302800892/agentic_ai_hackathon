"""Cost / latency dashboard data (AC-09).

  reports/dashboard_data.csv   span-level data exported from the same spans Phoenix shows
                               (run_id, span_id, name, category, agent, start, latency_ms, tokens, cost_usd, status)
  reports/dashboard_chart.png  chart drawn from that CSV (latency by category, tokens + cost by agent)
  reports/dashboard.png        Phoenix UI screenshot - captured by scripts/capture_dashboard.py (Playwright) or by
                               hand from http://localhost:6006; it is NOT generated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.golden_signals import spans_file  # noqa: E402
from src.config import REPORTS  # noqa: E402
from src.observability.spans import load_spans  # noqa: E402

CSV = REPORTS / "dashboard_data.csv"
CHART = REPORTS / "dashboard_chart.png"
COLS = ["trace_id", "span_id", "parent_id", "name", "category", "agent", "start_time", "latency_ms", "prompt_tokens",
        "completion_tokens", "total_tokens", "model", "cost_usd", "status"]

# Reference palette (dataviz skill): series-1 blue, series-2 orange; text/grid tokens
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def build() -> pd.DataFrame:
    df = load_spans(spans_file())
    out = df[COLS].rename(columns={"trace_id": "run_id"})
    out.to_csv(CSV, index=False)
    chart(out)
    return out


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    # 1. latency p50 / p95 by category (two series -> legend + direct labels)
    cats = ["thinking", "acting", "tool"]
    sub = df[df["category"].isin(cats) & (df["name"] != "copilot.request")]
    p50 = [sub.loc[sub["category"] == c, "latency_ms"].quantile(0.5) if (sub["category"] == c).any() else 0 for c in cats]
    p95 = [sub.loc[sub["category"] == c, "latency_ms"].quantile(0.95) if (sub["category"] == c).any() else 0 for c in cats]
    y = range(len(cats))
    ax = axes[0]
    ax.barh([i - 0.18 for i in y], p50, height=0.34, color=BLUE, label="p50")
    ax.barh([i + 0.18 for i in y], p95, height=0.34, color=ORANGE, label="p95")
    for i, (a, b) in enumerate(zip(p50, p95)):
        ax.text(b, i + 0.18, f" {b:,.0f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), cats)
    ax.invert_yaxis()
    ax.set_xlim(0, max(p95 + [1]) * 1.18)  # room for value labels
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    _style(ax, "Span latency by category (ms)")

    # 2. tokens by agent, 3. cost by agent (single series each; no legend needed)
    llm = df[df["category"] == "thinking"]
    for ax, col, title, fmt in ((axes[1], "total_tokens", "LLM tokens by agent", "{:,.0f}"),
                                (axes[2], "cost_usd", "Estimated cost by agent (USD)", "${:.4f}")):
        g = llm.groupby(llm["agent"].fillna("unknown"))[col].sum().sort_values()
        if g.empty:
            ax.text(0.5, 0.5, "no LLM spans\n(rules-only run)", ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.barh(g.index, g.values, color=BLUE, height=0.6)
            ax.set_xlim(0, float(g.max()) * 1.25)
            for i, v in enumerate(g.values):
                ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=MUTED)
        _style(ax, title)
    runs = df["run_id"].nunique()
    fig.suptitle(f"Patient Intake Copilot - cost / latency ({runs} runs, source: reports/dashboard_data.csv)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(CHART, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    d = build()
    print(f"wrote {CSV.name} ({len(d)} spans) and {CHART.name}")
