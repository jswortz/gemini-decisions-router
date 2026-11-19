#!/usr/bin/env python3
"""Renders the 4-panel dark-theme benchmark dashboard (`1600x1040` PNG).

Uses Pillow (`PIL`) with 2x supersampling + Lanczos downsampling and
DejaVuSans TrueType fonts for crisp typography and zero text clipping.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")

# 2x supersampling factor -> 3200x2080 internal canvas -> 1600x1040 output PNG.
SCALE = 2
OUT_W = 1600
OUT_H = 1040

# Dark theme palette (RGB tuples)
BG_CANVAS = (11, 15, 25)
BG_HEADER = (17, 24, 39)
BG_CARD = (17, 24, 39)
BORDER_CARD = (38, 52, 75)
GRID_LINE = (30, 41, 59)
AXIS_LINE = (51, 65, 85)

TEXT_PRIMARY = (248, 250, 252)
TEXT_SECONDARY = (148, 163, 184)
TEXT_MUTED = (100, 116, 139)

COL_CYAN = (56, 189, 248)
COL_BLUE = (59, 130, 246)
COL_INDIGO = (129, 140, 248)
COL_EMERALD = (16, 185, 129)
COL_MINT = (52, 211, 153)
COL_AMBER = (245, 158, 11)
COL_ROSE = (244, 63, 94)
COL_SLATE = (100, 116, 139)
COL_PRO_SLATE = (71, 85, 105)

CATEGORY_LABELS = {
    "quick_lookup": "Quick Lookup",
    "coding_edit": "Coding Edit",
    "writing_docs": "Writing & Docs",
    "data_analysis": "Data Analysis",
    "deep_debug_architecture": "Deep Debug & Arch",
    "agentic_lifecycle": "Agentic Lifecycle",
}


def _load_font(bold: bool, size_px: int, mono: bool = False) -> ImageFont.FreeTypeFont:
  if mono:
    fname = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
  else:
    fname = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  fpath = FONT_DIR / fname
  if fpath.is_file():
    return ImageFont.truetype(str(fpath), size_px * SCALE)
  return ImageFont.load_default()  # type: ignore[return-value]


def _s(val: float) -> int:
  return int(round(val * SCALE))


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> tuple[int, int]:
  bbox = draw.textbbox((0, 0), text, font=font)
  return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    anchor: str = "la",
) -> None:
  draw.text((_s(x), _s(y)), text, font=font, fill=fill, anchor=anchor)


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x_right: float,
    y_top: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    border: tuple[int, int, int],
    pad_x: float = 11.0,
    pad_y: float = 6.0,
) -> float:
  """Draws a right-aligned rounded badge and returns its left logical X."""
  tw_s, th_s = _text_size(draw, text, font)
  tw = tw_s / SCALE
  th = th_s / SCALE
  w = tw + 2 * pad_x
  h = th + 2 * pad_y
  x0 = x_right - w
  y0 = y_top
  draw.rounded_rectangle(
      (_s(x0), _s(y0), _s(x_right), _s(y0 + h)),
      radius=_s(h / 2.0),
      fill=bg,
      outline=border,
      width=max(1, SCALE),
  )
  draw.text(
      (_s(x0 + w / 2.0), _s(y0 + h / 2.0)),
      text,
      font=font,
      fill=fg,
      anchor="mm",
  )
  return x0


def _draw_card(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
) -> None:
  draw.rounded_rectangle(
      (_s(x), _s(y), _s(x + w), _s(y + h)),
      radius=_s(14),
      fill=BG_CARD,
      outline=BORDER_CARD,
      width=max(1, int(1.5 * SCALE)),
  )
  f_title = _load_font(bold=True, size_px=15)
  f_sub = _load_font(bold=False, size_px=11)
  _draw_text(draw, x + 20, y + 16, title, f_title, TEXT_PRIMARY, anchor="lt")
  _draw_text(draw, x + 20, y + 39, subtitle, f_sub, TEXT_SECONDARY, anchor="lt")


def _draw_legend_row(
    draw: ImageDraw.ImageDraw,
    x_right: float,
    y_center: float,
    items: list[tuple[str, tuple[int, int, int]]],
) -> None:
  f_leg = _load_font(bold=False, size_px=11)
  cur_x = x_right
  for label, color in reversed(items):
    tw_s, _ = _text_size(draw, label, f_leg)
    tw = tw_s / SCALE
    cur_x -= tw
    _draw_text(draw, cur_x, y_center, label, f_leg, TEXT_SECONDARY, anchor="lm")
    cur_x -= 15
    draw.rounded_rectangle(
        (_s(cur_x), _s(y_center - 5), _s(cur_x + 10), _s(y_center + 5)),
        radius=_s(3),
        fill=color,
    )
    cur_x -= 14


def _nice_ceil(val: float, step: float) -> float:
  if val <= 0:
    return step
  return math.ceil(val / step) * step


def _render_panel1_latency(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    results: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "1. Router Latency Overhead (p50 / p90 / p99 ms)",
      "Single-pass prefill (0 output tokens) vs GenerateContent JSON decoding (lower is better)",
  )
  _draw_legend_row(
      draw,
      x + w - 20,
      y + 24,
      [("p50", COL_CYAN), ("p90", COL_INDIGO), ("p99", COL_AMBER)],
  )

  strat = results["strategies"]
  full = strat["decisions_router_full"]
  t1 = strat["decisions_tier1_only"]
  llm = strat["llm_json_router"]

  groups = [
      (
          "Tier 0 + Tier 1",
          f"N={full['sample_size']} | 0 out tok",
          float(full["latency"]["p50_ms"]),
          float(full["latency"]["p90_ms"]),
          float(full["latency"]["p99_ms"]),
      ),
      (
          "Pure gemini-decision-1",
          f"N={t1['sample_size']} | 0 out tok",
          float(t1["latency"]["p50_ms"]),
          float(t1["latency"]["p90_ms"]),
          float(t1["latency"]["p99_ms"]),
      ),
      (
          "GenerateContent JSON",
          f"N={llm['sample_size']} | ~{int(round(llm['tokens']['mean_output_tokens']))} out tok",
          float(llm["latency"]["p50_ms"]),
          float(llm["latency"]["p90_ms"]),
          float(llm["latency"]["p99_ms"]),
      ),
  ]

  max_val = max(g[4] for g in groups)
  step = 250.0 if max_val <= 1500 else (500.0 if max_val <= 3500 else 1000.0)
  y_max = _nice_ceil(max_val * 1.18, step)

  plot_x0 = x + 64
  plot_x1 = x + w - 24
  plot_y0 = y + 92
  plot_y1 = y + h - 62
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0

  # Speedup callout pill above plot area
  speedup = float(llm["latency"]["p50_ms"]) / max(1.0, float(full["latency"]["p50_ms"]))
  f_badge = _load_font(bold=True, size_px=11)
  _draw_pill(
      draw,
      plot_x1,
      y + 58,
      f"{speedup:.1f}x faster p50 vs GenerateContent JSON",
      f_badge,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=10,
      pad_y=4,
  )

  f_axis = _load_font(bold=False, size_px=11)
  num_ticks = int(round(y_max / step))
  for i in range(num_ticks + 1):
    val = i * step
    yy = plot_y1 - (val / y_max) * plot_h
    line_col = AXIS_LINE if i == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 8, yy, f"{int(val)} ms", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(groups)
  bar_w = 36.0
  bar_gap = 8.0
  colors = (COL_CYAN, COL_INDIGO, COL_AMBER)
  f_val = _load_font(bold=True, size_px=11)
  f_lbl = _load_font(bold=True, size_px=12)
  f_sub = _load_font(bold=False, size_px=10)

  for idx, (g_title, g_sub, v50, v90, v99) in enumerate(groups):
    cx = plot_x0 + (idx + 0.5) * group_w
    total_bars_w = 3 * bar_w + 2 * bar_gap
    start_x = cx - total_bars_w / 2.0
    for b_idx, val in enumerate((v50, v90, v99)):
      bx0 = start_x + b_idx * (bar_w + bar_gap)
      bx1 = bx0 + bar_w
      bh = max(3.0, (min(val, y_max) / y_max) * plot_h)
      by0 = plot_y1 - bh
      draw.rounded_rectangle(
          (_s(bx0), _s(by0), _s(bx1), _s(plot_y1)),
          radius=_s(4),
          fill=colors[b_idx],
      )
      lbl_txt = f"{val:.1f}" if val < 10.0 else f"{int(round(val))}"
      _draw_text(
          draw,
          (bx0 + bx1) / 2.0,
          by0 - 5,
          lbl_txt,
          f_val,
          TEXT_PRIMARY,
          anchor="mb",
      )
    _draw_text(draw, cx, plot_y1 + 14, g_title, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(draw, cx, plot_y1 + 33, g_sub, f_sub, TEXT_SECONDARY, anchor="mt")


def _render_panel2_accuracy_safety(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    results: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "2. Tier Accuracy vs Critical Pro-Miss Rate (%)",
      "Exact tier match (higher is better) vs Pro tasks under-routed (lower is better)",
  )
  _draw_legend_row(
      draw,
      x + w - 20,
      y + 24,
      [("Exact Tier Acc %", COL_EMERALD), ("Pro-Miss Rate %", COL_ROSE)],
  )

  strat = results["strategies"]
  items = [
      (
          "Tier 0 + Tier 1",
          f"N={strat['decisions_router_full']['sample_size']}",
          float(strat["decisions_router_full"]["accuracy"]["exact_tier_accuracy_pct"]),
          float(strat["decisions_router_full"]["accuracy"]["pro_miss_rate_pct"]),
      ),
      (
          "Pure Tier 1",
          f"N={strat['decisions_tier1_only']['sample_size']}",
          float(strat["decisions_tier1_only"]["accuracy"]["exact_tier_accuracy_pct"]),
          float(strat["decisions_tier1_only"]["accuracy"]["pro_miss_rate_pct"]),
      ),
      (
          "GenerateContent",
          f"N={strat['llm_json_router']['sample_size']}",
          float(strat["llm_json_router"]["accuracy"]["exact_tier_accuracy_pct"]),
          float(strat["llm_json_router"]["accuracy"]["pro_miss_rate_pct"]),
      ),
      (
          "Lexical Baseline",
          f"N={strat['lexical_only']['sample_size']}",
          float(strat["lexical_only"]["accuracy"]["exact_tier_accuracy_pct"]),
          float(strat["lexical_only"]["accuracy"]["pro_miss_rate_pct"]),
      ),
  ]

  plot_x0 = x + 56
  plot_x1 = x + w - 24
  plot_y0 = y + 86
  plot_y1 = y + h - 62
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0
  y_max = 108.0

  f_axis = _load_font(bold=False, size_px=11)
  for pct in (0, 25, 50, 75, 100):
    yy = plot_y1 - (pct / y_max) * plot_h
    line_col = AXIS_LINE if pct == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 8, yy, f"{pct}%", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(items)
  bar_w = 40.0
  bar_gap = 10.0
  f_val = _load_font(bold=True, size_px=11)
  f_lbl = _load_font(bold=True, size_px=12)
  f_sub = _load_font(bold=False, size_px=10)

  for idx, (label, sub, acc_pct, miss_pct) in enumerate(items):
    cx = plot_x0 + (idx + 0.5) * group_w
    start_x = cx - (2 * bar_w + bar_gap) / 2.0

    # Accuracy bar
    bh_acc = max(3.0, (acc_pct / y_max) * plot_h)
    by_acc = plot_y1 - bh_acc
    draw.rounded_rectangle(
        (_s(start_x), _s(by_acc), _s(start_x + bar_w), _s(plot_y1)),
        radius=_s(4),
        fill=COL_EMERALD,
    )
    _draw_text(
        draw,
        start_x + bar_w / 2.0,
        by_acc - 5,
        f"{acc_pct:.1f}%",
        f_val,
        COL_MINT,
        anchor="mb",
    )

    # Pro-Miss bar
    bx_m = start_x + bar_w + bar_gap
    bh_miss = max(3.0, (miss_pct / y_max) * plot_h)
    by_miss = plot_y1 - bh_miss
    miss_col = COL_SLATE if miss_pct == 0.0 else COL_ROSE
    draw.rounded_rectangle(
        (_s(bx_m), _s(by_miss), _s(bx_m + bar_w), _s(plot_y1)),
        radius=_s(4),
        fill=miss_col,
    )
    _draw_text(
        draw,
        bx_m + bar_w / 2.0,
        by_miss - 5,
        f"{miss_pct:.1f}%",
        f_val,
        TEXT_SECONDARY if miss_pct == 0.0 else COL_ROSE,
        anchor="mb",
    )

    _draw_text(draw, cx, plot_y1 + 14, label, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(draw, cx, plot_y1 + 33, sub, f_sub, TEXT_SECONDARY, anchor="mt")


def _render_panel3_categories(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    results: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "3. Per-Category Routing Accuracy (N=100)",
      "Gemini Decisions Router (Tier 0+1) vs Offline Lexical Baseline across 6 categories",
  )
  _draw_legend_row(
      draw,
      x + w - 20,
      y + 24,
      [("Decisions Router", COL_CYAN), ("Lexical Baseline", COL_SLATE)],
  )

  full_cats = results["strategies"]["decisions_router_full"]["per_category"]
  lex_cats = results["strategies"]["lexical_only"]["per_category"]

  cat_keys = [k for k in CATEGORY_LABELS if k in full_cats]
  plot_x0 = x + 196
  plot_x1 = x + w - 64
  plot_y0 = y + 74
  plot_y1 = y + h - 42
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0

  f_axis = _load_font(bold=False, size_px=11)
  for pct in (0, 25, 50, 75, 100):
    xx = plot_x0 + (pct / 100.0) * plot_w
    line_col = AXIS_LINE if pct == 0 else GRID_LINE
    draw.line((_s(xx), _s(plot_y0), _s(xx), _s(plot_y1)), fill=line_col, width=SCALE)
    _draw_text(draw, xx, plot_y1 + 10, f"{pct}%", f_axis, TEXT_MUTED, anchor="mt")

  row_h = plot_h / max(1, len(cat_keys))
  bar_h = 13.0
  bar_gap = 4.0
  f_cat = _load_font(bold=True, size_px=12)
  f_val = _load_font(bold=True, size_px=11)
  f_val_dim = _load_font(bold=False, size_px=10)

  for idx, ckey in enumerate(cat_keys):
    cy = plot_y0 + (idx + 0.5) * row_h
    cnt = int(full_cats[ckey]["count"])
    acc_dec = float(full_cats[ckey]["tier_accuracy_pct"])
    acc_lex = float(lex_cats.get(ckey, {}).get("tier_accuracy_pct", 0.0))

    cat_title = f"{CATEGORY_LABELS[ckey]} (n={cnt})"
    _draw_text(draw, plot_x0 - 12, cy, cat_title, f_cat, TEXT_PRIMARY, anchor="rm")

    top_y = cy - bar_h - bar_gap / 2.0
    bot_y = cy + bar_gap / 2.0

    w_dec = max(3.0, (acc_dec / 100.0) * plot_w)
    draw.rounded_rectangle(
        (_s(plot_x0), _s(top_y), _s(plot_x0 + w_dec), _s(top_y + bar_h)),
        radius=_s(3),
        fill=COL_CYAN,
    )
    _draw_text(
        draw,
        plot_x0 + w_dec + 7,
        top_y + bar_h / 2.0,
        f"{acc_dec:.0f}%" if acc_dec.is_integer() else f"{acc_dec:.1f}%",
        f_val,
        COL_CYAN,
        anchor="lm",
    )

    w_lex = max(3.0, (acc_lex / 100.0) * plot_w)
    draw.rounded_rectangle(
        (_s(plot_x0), _s(bot_y), _s(plot_x0 + w_lex), _s(bot_y + bar_h)),
        radius=_s(3),
        fill=COL_SLATE,
    )
    _draw_text(
        draw,
        plot_x0 + w_lex + 7,
        bot_y + bar_h / 2.0,
        f"{acc_lex:.0f}%" if acc_lex.is_integer() else f"{acc_lex:.1f}%",
        f_val_dim,
        TEXT_SECONDARY,
        anchor="lm",
    )


def _render_panel4_e2e_workload(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    results: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "4. 100-Prompt End-to-End Time Breakdown",
      "Stacked wall-clock seconds (Generation + Router Overhead) & API cost per 100 prompts",
  )
  _draw_legend_row(
      draw,
      x + w - 20,
      y + 24,
      [("Generation Time", COL_BLUE), ("Router Overhead", COL_AMBER)],
  )

  proj = results["e2e"]["projected_100_prompts"]
  ap = proj["always_pro"]
  lj = proj["llm_json_router_routed"]
  dr = proj["decisions_router_routed"]

  time_saved = float(dr["time_saved_vs_always_pro_pct"])
  cost_saved = float(dr["cost_saved_vs_always_pro_pct"])
  f_badge = _load_font(bold=True, size_px=11)
  _draw_pill(
      draw,
      x + w - 20,
      y + 58,
      f"Decisions Router vs Always-Pro: -{time_saved:.1f}% Time | -{cost_saved:.1f}% Cost",
      f_badge,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=10,
      pad_y=4,
  )

  bars = [
      (
          "Always-Pro",
          "0s router overhead",
          float(ap["generation_s"]),
          float(ap["router_overhead_s"]),
          float(ap["total_cost_usd"]),
          COL_PRO_SLATE,
          "Baseline",
      ),
      (
          "GenerateContent-Routed",
          f"+{float(lj['router_overhead_s']):.1f}s router overhead",
          float(lj["generation_s"]),
          float(lj["router_overhead_s"]),
          float(lj["total_cost_usd"]),
          COL_BLUE,
          f"-{float(lj['time_saved_vs_always_pro_pct']):.1f}% time | -{float(lj['cost_saved_vs_always_pro_pct']):.1f}% cost",
      ),
      (
          "Gemini Decisions Router",
          f"+{float(dr['router_overhead_s']):.1f}s router overhead",
          float(dr["generation_s"]),
          float(dr["router_overhead_s"]),
          float(dr["total_cost_usd"]),
          COL_EMERALD,
          f"-{time_saved:.1f}% time | -{cost_saved:.1f}% cost",
      ),
  ]

  max_total = max(b[2] + b[3] for b in bars)
  step = 50.0 if max_total <= 300 else (100.0 if max_total <= 700 else 200.0)
  y_max = _nice_ceil(max_total * 1.24, step)

  plot_x0 = x + 62
  plot_x1 = x + w - 24
  plot_y0 = y + 94
  plot_y1 = y + h - 62
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0

  f_axis = _load_font(bold=False, size_px=11)
  num_ticks = int(round(y_max / step))
  for i in range(num_ticks + 1):
    val = i * step
    yy = plot_y1 - (val / y_max) * plot_h
    line_col = AXIS_LINE if i == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 8, yy, f"{int(val)}s", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(bars)
  bar_w = 104.0
  f_tot = _load_font(bold=True, size_px=12)
  f_cost = _load_font(bold=True, size_px=11)
  f_in = _load_font(bold=True, size_px=11)
  f_lbl = _load_font(bold=True, size_px=12)
  f_sub = _load_font(bold=False, size_px=10)

  for idx, (title, sub, gen_s, rtr_s, cost_usd, gen_col, delta_lbl) in enumerate(bars):
    cx = plot_x0 + (idx + 0.5) * group_w
    bx0 = cx - bar_w / 2.0
    bx1 = cx + bar_w / 2.0

    tot_s = gen_s + rtr_s
    h_gen = max(4.0, (gen_s / y_max) * plot_h)
    h_rtr = (rtr_s / y_max) * plot_h if rtr_s > 0.2 else 0.0

    y_gen_top = plot_y1 - h_gen
    draw.rounded_rectangle(
        (_s(bx0), _s(y_gen_top), _s(bx1), _s(plot_y1)),
        radius=_s(5),
        fill=gen_col,
    )
    if h_gen >= 28.0:
      _draw_text(
          draw,
          cx,
          (y_gen_top + plot_y1) / 2.0,
          f"Gen {gen_s:.0f}s",
          f_in,
          TEXT_PRIMARY,
          anchor="mm",
      )

    y_top = y_gen_top
    if h_rtr >= 2.0:
      y_top = y_gen_top - h_rtr
      draw.rounded_rectangle(
          (_s(bx0), _s(y_top), _s(bx1), _s(y_gen_top + 2)),
          radius=_s(4),
          fill=COL_AMBER,
      )

    _draw_text(
        draw,
        cx,
        y_top - 21,
        f"{tot_s:.1f}s (${cost_usd:.3f})",
        f_tot,
        TEXT_PRIMARY,
        anchor="mb",
    )
    _draw_text(
        draw,
        cx,
        y_top - 5,
        delta_lbl,
        f_cost,
        COL_MINT if idx == 2 else TEXT_SECONDARY,
        anchor="mb",
    )
    _draw_text(draw, cx, plot_y1 + 14, title, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(draw, cx, plot_y1 + 33, sub, f_sub, TEXT_SECONDARY, anchor="mt")


def render_dashboard(results_path: pathlib.Path, output_path: pathlib.Path) -> None:
  results = json.loads(results_path.read_text(encoding="utf-8"))

  img = Image.new("RGB", (OUT_W * SCALE, OUT_H * SCALE), BG_CANVAS)
  draw = ImageDraw.Draw(img)

  # Top header banner (x=24..1576, y=18..96)
  draw.rounded_rectangle(
      (_s(24), _s(18), _s(1576), _s(96)),
      radius=_s(14),
      fill=BG_HEADER,
      outline=BORDER_CARD,
      width=max(1, int(1.5 * SCALE)),
  )
  f_h1 = _load_font(bold=True, size_px=18)
  f_h2 = _load_font(bold=False, size_px=11)

  meta = results["metadata"]
  full = results["strategies"]["decisions_router_full"]
  proj = results["e2e"]["projected_100_prompts"]["decisions_router_routed"]

  _draw_text(
      draw,
      44,
      32,
      "Gemini Decisions Router: Benchmark Dashboard",
      f_h1,
      TEXT_PRIMARY,
      anchor="lt",
  )
  mode_tag = "LIVE API" if meta.get("mode") == "live" else "MOCK SIMULATION"
  sub_line = (
      f"{mode_tag} | N={meta['dataset_size']} Prompts (6 Categories) + "
      f"N={meta['sample_e2e_size']} Multi-Tier E2E | "
      f"Model: {meta['decisions_model']} (0 Output Tokens)"
  )
  _draw_text(draw, 44, 61, sub_line, f_h2, TEXT_SECONDARY, anchor="lt")

  # KPI summary pills on right side of header
  f_kpi = _load_font(bold=True, size_px=11)
  x_cursor = 1556.0
  x_cursor = _draw_pill(
      draw,
      x_cursor,
      42,
      f"E2E: -{proj['time_saved_vs_always_pro_pct']:.1f}% Time / -{proj['cost_saved_vs_always_pro_pct']:.1f}% Cost",
      f_kpi,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=10,
      pad_y=5,
  )
  x_cursor -= 8
  x_cursor = _draw_pill(
      draw,
      x_cursor,
      42,
      f"Tier Acc: {full['accuracy']['exact_tier_accuracy_pct']:.1f}% (Pro-Miss {full['accuracy']['pro_miss_rate_pct']:.1f}%)",
      f_kpi,
      COL_CYAN,
      (15, 35, 58),
      (30, 78, 120),
      pad_x=10,
      pad_y=5,
  )
  x_cursor -= 8
  _draw_pill(
      draw,
      x_cursor,
      42,
      f"p50 Overhead: {full['latency']['p50_ms']:.0f} ms (0 Out Tok)",
      f_kpi,
      COL_INDIGO,
      (26, 28, 60),
      (67, 72, 138),
      pad_x=10,
      pad_y=5,
  )

  # 2x2 Panel Grid
  _render_panel1_latency(draw, 24, 112, 764, 444, results)
  _render_panel2_accuracy_safety(draw, 812, 112, 764, 444, results)
  _render_panel3_categories(draw, 24, 576, 764, 444, results)
  _render_panel4_e2e_workload(draw, 812, 576, 764, 444, results)

  final_img = img.resize((OUT_W, OUT_H), resample=Image.Resampling.LANCZOS)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  final_img.save(output_path, format="PNG", optimize=True)
  print(f"Rendered {OUT_W}x{OUT_H} dashboard -> {output_path}")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Render the 4-panel dark-theme benchmark dashboard PNG."
  )
  parser.add_argument(
      "--input",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_100.json",
      help="Path to benchmark results JSON.",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=REPO_ROOT / "assets" / "benchmark-dashboard.png",
      help="Path to output PNG image.",
  )
  args = parser.parse_args(argv)
  render_dashboard(args.input, args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
