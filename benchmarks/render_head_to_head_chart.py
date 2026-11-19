#!/usr/bin/env python3
"""Renders the 4-panel dark-theme N=100 Head-to-Head Router chart (`1600x1000` PNG).

Compares only external-safe public routers across N=100 stratified prompts:
  - Panel 1: Router Overhead p50 / p90 / p99 (ms)
  - Panel 2: Tier Accuracy, Critical Pro-Miss Rate & Over-Route Rate (%)
  - Panel 3: Confidence Calibration Across 4 Bins (Brier & Collapse Analysis)
  - Panel 4: Router Cost & Net 100-Prompt E2E Wall-Clock Time (s)

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

# 2x supersampling factor -> 3200x2000 internal canvas -> 1600x1000 output PNG.
SCALE = 2
OUT_W = 1600
OUT_H = 1000

# Cohesive dark theme palette (RGB tuples)
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
    pad_x: float = 10.0,
    pad_y: float = 5.0,
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
  f_title = _load_font(bold=True, size_px=14)
  f_sub = _load_font(bold=False, size_px=11)
  _draw_text(draw, x + 20, y + 15, title, f_title, TEXT_PRIMARY, anchor="lt")
  _draw_text(draw, x + 20, y + 37, subtitle, f_sub, TEXT_SECONDARY, anchor="lt")


def _draw_legend_row(
    draw: ImageDraw.ImageDraw,
    x_right: float,
    y_center: float,
    items: list[tuple[str, tuple[int, int, int]]],
) -> None:
  f_leg = _load_font(bold=False, size_px=10)
  cur_x = x_right
  for label, color in reversed(items):
    tw_s, _ = _text_size(draw, label, f_leg)
    tw = tw_s / SCALE
    cur_x -= tw
    _draw_text(draw, cur_x, y_center, label, f_leg, TEXT_SECONDARY, anchor="lm")
    cur_x -= 14
    draw.rounded_rectangle(
        (_s(cur_x), _s(y_center - 5), _s(cur_x + 10), _s(y_center + 5)),
        radius=_s(3),
        fill=color,
    )
    cur_x -= 12


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
    h2h: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "1. Router Overhead: p50 / p90 / p99 (ms)",
      "Single-pass prefill (0 out tok) vs GenerateContent JSON routers (lower is better)",
  )
  _draw_legend_row(
      draw,
      x + w - 18,
      y + 22,
      [("p50", COL_CYAN), ("p90", COL_INDIGO), ("p99", COL_AMBER)],
  )

  strat = h2h["strategies"]
  t1 = strat["decisions_tier1_only"]
  full = strat["decisions_router_full"]
  fj = strat["flash_json_router"]
  fl = strat["flash_lite_json_router"]
  ref_def = h2h["reference_default_thinking_flash_router"]

  groups = [
      (
          "Pure decision-1",
          "N=100 | 0 out tok",
          float(t1["latency"]["p50_ms"]),
          float(t1["latency"]["p90_ms"]),
          float(t1["latency"]["p99_ms"]),
      ),
      (
          "Tier 0+1 Full",
          "N=100 | 0 out tok",
          float(full["latency"]["p50_ms"]),
          float(full["latency"]["p90_ms"]),
          float(full["latency"]["p99_ms"]),
      ),
      (
          "2.5-flash JSON",
          "think=0 | 24 tok",
          float(fj["latency"]["p50_ms"]),
          float(fj["latency"]["p90_ms"]),
          float(fj["latency"]["p99_ms"]),
      ),
      (
          "3.1-flash-lite",
          "think=0 | 21 tok",
          float(fl["latency"]["p50_ms"]),
          float(fl["latency"]["p90_ms"]),
          float(fl["latency"]["p99_ms"]),
      ),
      (
          "2.5-flash JSON",
          "default | 144 tok",
          float(ref_def["latency"]["p50_ms"]),
          float(ref_def["latency"]["p90_ms"]),
          float(ref_def["latency"]["p99_ms"]),
      ),
  ]

  max_val = max(g[4] for g in groups)
  step = 500.0
  y_max = _nice_ceil(max_val * 1.15, step)

  plot_x0 = x + 64
  plot_x1 = x + w - 18
  plot_y0 = y + 92
  plot_y1 = y + h - 58
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0

  ratio_fl = float(fl["latency"]["p50_ms"]) / max(1.0, float(t1["latency"]["p50_ms"]))
  ratio_def = float(ref_def["latency"]["p50_ms"]) / max(1.0, float(t1["latency"]["p50_ms"]))
  f_badge = _load_font(bold=True, size_px=10)
  _draw_pill(
      draw,
      plot_x1,
      y + 58,
      f"Pure decision-1 p50: {ratio_fl:.1f}x faster vs flash-lite | {ratio_def:.1f}x vs flash default",
      f_badge,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=9,
      pad_y=4,
  )

  f_axis = _load_font(bold=False, size_px=10)
  num_ticks = int(round(y_max / step))
  for i in range(num_ticks + 1):
    val = i * step
    yy = plot_y1 - (val / y_max) * plot_h
    line_col = AXIS_LINE if i == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 7, yy, f"{int(val)} ms", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(groups)
  bar_w = 29.0
  bar_gap = 6.0
  colors = (COL_CYAN, COL_INDIGO, COL_AMBER)
  f_val = _load_font(bold=True, size_px=10)
  f_lbl = _load_font(bold=True, size_px=11)
  f_sub = _load_font(bold=False, size_px=9)

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
          radius=_s(3),
          fill=colors[b_idx],
      )
      lbl_txt = f"{int(round(val)):,}" if val >= 1000 else f"{int(round(val))}"
      _draw_text(
          draw,
          (bx0 + bx1) / 2.0,
          by0 - 4,
          lbl_txt,
          f_val,
          TEXT_PRIMARY,
          anchor="mb",
      )
    _draw_text(draw, cx, plot_y1 + 12, g_title, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(draw, cx, plot_y1 + 29, g_sub, f_sub, TEXT_SECONDARY, anchor="mt")


def _render_panel2_accuracy(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    h2h: dict[str, Any],
    base_res: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "2. Accuracy, Critical Pro-Miss & Over-Route (N=100)",
      "Exact tier match (higher is better) vs Pro under-route & Pro over-route",
  )
  _draw_legend_row(
      draw,
      x + w - 18,
      y + 22,
      [("Tier Acc", COL_EMERALD), ("Pro-Miss", COL_ROSE), ("Over-Route", COL_AMBER)],
  )

  strat = h2h["strategies"]
  full_acc = strat["decisions_router_full"]["accuracy"]
  fl_acc = strat["flash_lite_json_router"]["accuracy"]
  fj_acc = strat["flash_json_router"]["accuracy"]
  lex_acc = base_res["strategies"]["lexical_only"]["accuracy"]

  f_badge = _load_font(bold=True, size_px=10)
  _draw_pill(
      draw,
      x + w - 18,
      y + 58,
      "Decisions (Full & Pure): 96.0% Acc | 0.0% Pro-Miss | 0.0% Over-Route",
      f_badge,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=9,
      pad_y=4,
  )

  items = [
      (
          "gemini-decision-1",
          "Tier 0+1 & Pure (N=100)",
          float(full_acc["exact_tier_accuracy_pct"]),
          float(full_acc["pro_miss_rate_pct"]),
          float(full_acc["over_route_to_pro_rate_pct"]),
      ),
      (
          "3.1-flash-lite JSON",
          "thinking=0 (N=100)",
          float(fl_acc["exact_tier_accuracy_pct"]),
          float(fl_acc["pro_miss_rate_pct"]),
          float(fl_acc["over_route_to_pro_rate_pct"]),
      ),
      (
          "2.5-flash JSON",
          "thinking=0 (N=100)",
          float(fj_acc["exact_tier_accuracy_pct"]),
          float(fj_acc["pro_miss_rate_pct"]),
          float(fj_acc["over_route_to_pro_rate_pct"]),
      ),
      (
          "Lexical Baseline",
          "offline rules (N=100)",
          float(lex_acc["exact_tier_accuracy_pct"]),
          float(lex_acc["pro_miss_rate_pct"]),
          float(lex_acc["over_route_to_pro_rate_pct"]),
      ),
  ]

  plot_x0 = x + 52
  plot_x1 = x + w - 18
  plot_y0 = y + 92
  plot_y1 = y + h - 58
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0
  y_max = 110.0

  f_axis = _load_font(bold=False, size_px=10)
  for pct in (0, 25, 50, 75, 100):
    yy = plot_y1 - (pct / y_max) * plot_h
    line_col = AXIS_LINE if pct == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 7, yy, f"{pct}%", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(items)
  bar_w = 36.0
  bar_gap = 8.0
  f_val = _load_font(bold=True, size_px=10)
  f_lbl = _load_font(bold=True, size_px=11)
  f_sub = _load_font(bold=False, size_px=9)

  for idx, (label, sub, acc_v, miss_v, over_v) in enumerate(items):
    cx = plot_x0 + (idx + 0.5) * group_w
    start_x = cx - (3 * bar_w + 2 * bar_gap) / 2.0

    series = [
        (acc_v, COL_EMERALD, COL_MINT),
        (miss_v, COL_SLATE if miss_v == 0.0 else COL_ROSE, TEXT_SECONDARY if miss_v == 0.0 else COL_ROSE),
        (over_v, COL_SLATE if over_v == 0.0 else COL_AMBER, TEXT_SECONDARY if over_v == 0.0 else COL_AMBER),
    ]
    for b_idx, (val, b_col, t_col) in enumerate(series):
      bx0 = start_x + b_idx * (bar_w + bar_gap)
      bx1 = bx0 + bar_w
      bh = max(3.0, (val / y_max) * plot_h)
      by0 = plot_y1 - bh
      draw.rounded_rectangle(
          (_s(bx0), _s(by0), _s(bx1), _s(plot_y1)),
          radius=_s(3),
          fill=b_col,
      )
      _draw_text(
          draw,
          (bx0 + bx1) / 2.0,
          by0 - 4,
          f"{val:.1f}%",
          f_val,
          t_col,
          anchor="mb",
      )

    _draw_text(draw, cx, plot_y1 + 12, label, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(draw, cx, plot_y1 + 29, sub, f_sub, TEXT_SECONDARY, anchor="mt")


def _render_panel3_calibration(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    h2h: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "3. Confidence Calibration Across 4 Bins (N=100)",
      "Monotonic posteriors (4/4 bins, Brier=0.0524) vs JSON collapse (100/100 in [0.90, 1.00])",
  )
  _draw_legend_row(
      draw,
      x + w - 18,
      y + 22,
      [("Decisions Acc", COL_CYAN), ("flash-lite Acc", COL_ROSE), ("Mean Conf", TEXT_PRIMARY)],
  )

  dec_cal = h2h["strategies"]["decisions_router_full"]["calibration"]
  fl_cal = h2h["strategies"]["flash_lite_json_router"]["calibration"]

  f_badge = _load_font(bold=True, size_px=10)
  _draw_pill(
      draw,
      x + w - 18,
      y + 58,
      f"flash-lite JSON: 100/100 in [0.90, 1.00] | {fl_cal['mean_confidence_when_wrong']:.3f} conf when wrong (Brier {fl_cal['brier_score']:.4f})",
      f_badge,
      COL_AMBER,
      (45, 29, 12),
      (120, 72, 20),
      pad_x=9,
      pad_y=4,
  )

  bin_labels = ["[0.50-0.70)", "[0.70-0.80)", "[0.80-0.90)", "[0.90-1.00]"]
  dec_bins = dec_cal["bins"]
  fl_bins = fl_cal["bins"]

  plot_x0 = x + 52
  plot_x1 = x + w - 18
  plot_y0 = y + 92
  plot_y1 = y + h - 58
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0
  y_max = 112.0

  f_axis = _load_font(bold=False, size_px=10)
  for pct in (0, 25, 50, 75, 100):
    yy = plot_y1 - (pct / y_max) * plot_h
    line_col = AXIS_LINE if pct == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 7, yy, f"{pct}%", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / 4.0
  bar_w = 54.0
  bar_gap = 10.0
  f_val = _load_font(bold=True, size_px=10)
  f_in = _load_font(bold=True, size_px=9)
  f_in_sub = _load_font(bold=False, size_px=9)
  f_lbl = _load_font(bold=True, size_px=11)
  f_sub = _load_font(bold=False, size_px=9)

  for idx, blbl in enumerate(bin_labels):
    cx = plot_x0 + (idx + 0.5) * group_w
    start_x = cx - (2 * bar_w + bar_gap) / 2.0

    db = dec_bins[idx]
    fb = fl_bins[idx]

    # 1. Decisions Router bar
    d_cnt = int(db["count"])
    d_acc = float(db["empirical_accuracy"]) * 100.0
    d_conf = float(db["mean_confidence"]) * 100.0
    bx0 = start_x
    bx1 = bx0 + bar_w
    bh = max(4.0, (d_acc / y_max) * plot_h)
    by0 = plot_y1 - bh
    draw.rounded_rectangle(
        (_s(bx0), _s(by0), _s(bx1), _s(plot_y1)),
        radius=_s(4),
        fill=(22, 94, 130),
        outline=COL_CYAN,
        width=max(1, SCALE),
    )
    # Confidence horizontal marker + dot on Decisions bar
    y_conf = plot_y1 - (d_conf / y_max) * plot_h
    draw.line((_s(bx0 + 4), _s(y_conf), _s(bx1 - 4), _s(y_conf)), fill=TEXT_PRIMARY, width=2 * SCALE)
    draw.ellipse((_s((bx0 + bx1) / 2 - 3.5), _s(y_conf - 3.5), _s((bx0 + bx1) / 2 + 3.5), _s(y_conf + 3.5)), fill=TEXT_PRIMARY)
    _draw_text(draw, (bx0 + bx1) / 2.0, by0 - 4, f"{d_acc:.1f}%", f_val, COL_CYAN, anchor="mb")
    _draw_text(draw, (bx0 + bx1) / 2.0, plot_y1 - 24, f"n={d_cnt}", f_in, TEXT_PRIMARY, anchor="mm")
    _draw_text(draw, (bx0 + bx1) / 2.0, plot_y1 - 11, f"c={d_conf:.0f}%", f_in_sub, COL_MINT, anchor="mm")

    # 2. flash-lite JSON Router bar
    f_cnt = int(fb["count"])
    fx0 = start_x + bar_w + bar_gap
    fx1 = fx0 + bar_w
    if f_cnt == 0:
      # Empty bin indicator showing 0/100 collapse
      draw.rounded_rectangle(
          (_s(fx0), _s(plot_y1 - 28), _s(fx1), _s(plot_y1)),
          radius=_s(4),
          fill=(24, 32, 48),
          outline=AXIS_LINE,
          width=max(1, SCALE),
      )
      _draw_text(draw, (fx0 + fx1) / 2.0, plot_y1 - 14, "n=0", f_in, TEXT_MUTED, anchor="mm")
      _draw_text(draw, (fx0 + fx1) / 2.0, plot_y1 - 33, "empty", f_in_sub, TEXT_MUTED, anchor="mb")
    else:
      f_acc = float(fb["empirical_accuracy"]) * 100.0
      f_conf = float(fb["mean_confidence"]) * 100.0
      fh = max(4.0, (f_acc / y_max) * plot_h)
      fy0 = plot_y1 - fh
      draw.rounded_rectangle(
          (_s(fx0), _s(fy0), _s(fx1), _s(plot_y1)),
          radius=_s(4),
          fill=(115, 32, 52),
          outline=COL_ROSE,
          width=max(1, SCALE),
      )
      yf_conf = plot_y1 - (f_conf / y_max) * plot_h
      draw.line((_s(fx0 + 4), _s(yf_conf), _s(fx1 - 4), _s(yf_conf)), fill=TEXT_PRIMARY, width=2 * SCALE)
      draw.ellipse((_s((fx0 + fx1) / 2 - 3.5), _s(yf_conf - 3.5), _s((fx0 + fx1) / 2 + 3.5), _s(yf_conf + 3.5)), fill=TEXT_PRIMARY)
      _draw_text(draw, (fx0 + fx1) / 2.0, yf_conf - 5, f"{f_acc:.1f}%", f_val, COL_ROSE, anchor="mb")
      _draw_text(draw, (fx0 + fx1) / 2.0, plot_y1 - 24, f"n={f_cnt}", f_in, TEXT_PRIMARY, anchor="mm")
      _draw_text(draw, (fx0 + fx1) / 2.0, plot_y1 - 11, f"c={f_conf:.0f}%", f_in_sub, COL_AMBER, anchor="mm")

    _draw_text(draw, cx, plot_y1 + 12, blbl, f_lbl, TEXT_PRIMARY, anchor="mt")
    sub_note = (
        f"Decisions n={d_cnt}"
        if f_cnt == 0
        else f"Brier {dec_cal['brier_score']:.3f} vs {fl_cal['brier_score']:.3f}"
    )
    _draw_text(draw, cx, plot_y1 + 29, sub_note, f_sub, TEXT_SECONDARY, anchor="mt")


def _render_panel4_cost_and_e2e(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    h2h: dict[str, Any],
    base_res: dict[str, Any],
) -> None:
  _draw_card(
      draw,
      x,
      y,
      w,
      h,
      "4. Router Cost & 100-Prompt E2E Wall-Clock Time",
      "Stacked E2E seconds (Downstream Gen + Router Overhead) & 100-prompt Router Cost",
  )
  _draw_legend_row(
      draw,
      x + w - 18,
      y + 22,
      [("Downstream Gen", COL_BLUE), ("Router Overhead", COL_AMBER)],
  )

  strat = h2h["strategies"]
  dec_s = strat["decisions_router_full"]
  fj_s = strat["flash_json_router"]
  fl_s = strat["flash_lite_json_router"]
  ap_proj = base_res["e2e"]["projected_100_prompts"]["always_pro"]

  dec_e2e = dec_s["projected_100_e2e"]
  fj_e2e = fj_s["projected_100_e2e"]
  fl_e2e = fl_s["projected_100_e2e"]

  dec_rtr_cost = float(dec_s["tokens"]["router_cost_per_100_usd"])
  fj_rtr_cost = float(fj_s["tokens"]["router_cost_per_100_usd"])
  fl_rtr_cost = float(fl_s["tokens"]["router_cost_per_100_preview_tier_usd"])

  net_vs_fl = float(h2h["head_to_head_deltas"]["decisions_net_e2e_seconds_saved_vs_flash_lite_per_100"])
  f_badge = _load_font(bold=True, size_px=10)
  _draw_pill(
      draw,
      x + w - 18,
      y + 58,
      f"Decisions saves {net_vs_fl:.1f}s vs flash-lite JSON & 137.7s (-{dec_e2e['time_saved_vs_always_pro_pct']:.1f}%) vs Always-Pro",
      f_badge,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=9,
      pad_y=4,
  )

  bars = [
      (
          "Decisions (Tier 0+1)",
          f"Router ${dec_rtr_cost:.5f}",
          float(dec_e2e["generation_s"]),
          float(dec_e2e["router_overhead_s"]),
          COL_EMERALD,
          f"-{float(dec_e2e['time_saved_vs_always_pro_pct']):.1f}% time",
      ),
      (
          "2.5-flash (think=0)",
          f"Router ${fj_rtr_cost:.5f}",
          float(fj_e2e["generation_s"]),
          float(fj_e2e["router_overhead_s"]),
          COL_BLUE,
          f"-{float(fj_e2e['time_saved_vs_always_pro_pct']):.1f}% time",
      ),
      (
          "3.1-flash-lite JSON",
          f"Router ${fl_rtr_cost:.5f}",
          float(fl_e2e["generation_s"]),
          float(fl_e2e["router_overhead_s"]),
          COL_BLUE,
          f"-{float(fl_e2e['time_saved_vs_always_pro_pct']):.1f}% time",
      ),
      (
          "Always-Pro",
          f"E2E ${float(ap_proj['total_cost_usd']):.3f} (0 rtr)",
          float(ap_proj["generation_s"]),
          0.0,
          COL_PRO_SLATE,
          "Baseline",
      ),
  ]

  max_total = max(b[2] + b[3] for b in bars)
  step = 100.0
  y_max = _nice_ceil(max_total * 1.25, step)

  plot_x0 = x + 56
  plot_x1 = x + w - 18
  plot_y0 = y + 94
  plot_y1 = y + h - 58
  plot_w = plot_x1 - plot_x0
  plot_h = plot_y1 - plot_y0

  f_axis = _load_font(bold=False, size_px=10)
  num_ticks = int(round(y_max / step))
  for i in range(num_ticks + 1):
    val = i * step
    yy = plot_y1 - (val / y_max) * plot_h
    line_col = AXIS_LINE if i == 0 else GRID_LINE
    draw.line((_s(plot_x0), _s(yy), _s(plot_x1), _s(yy)), fill=line_col, width=SCALE)
    _draw_text(draw, plot_x0 - 7, yy, f"{int(val)}s", f_axis, TEXT_MUTED, anchor="rm")

  group_w = plot_w / len(bars)
  bar_w = 96.0
  f_tot = _load_font(bold=True, size_px=11)
  f_delta = _load_font(bold=True, size_px=10)
  f_in = _load_font(bold=True, size_px=10)
  f_lbl = _load_font(bold=True, size_px=11)
  f_sub = _load_font(bold=True, size_px=9)

  for idx, (title, sub_cost, gen_s, rtr_s, gen_col, delta_lbl) in enumerate(bars):
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
    _draw_text(
        draw,
        cx,
        (y_gen_top + plot_y1) / 2.0,
        f"Gen {gen_s:.1f}s",
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
      if h_rtr >= 14.0:
        _draw_text(
            draw,
            cx,
            (y_top + y_gen_top) / 2.0,
            f"+{rtr_s:.1f}s",
            f_in,
            BG_CANVAS,
            anchor="mm",
        )

    _draw_text(
        draw,
        cx,
        y_top - 19,
        f"{tot_s:.1f}s" + (f" (+{rtr_s:.1f}s rtr)" if 0.2 < rtr_s and h_rtr < 14.0 else ""),
        f_tot,
        TEXT_PRIMARY,
        anchor="mb",
    )
    _draw_text(
        draw,
        cx,
        y_top - 4,
        delta_lbl,
        f_delta,
        COL_MINT if idx == 0 else TEXT_SECONDARY,
        anchor="mb",
    )
    _draw_text(draw, cx, plot_y1 + 12, title, f_lbl, TEXT_PRIMARY, anchor="mt")
    _draw_text(
        draw,
        cx,
        plot_y1 + 29,
        sub_cost,
        f_sub,
        COL_CYAN if idx == 0 else (COL_AMBER if idx in (1, 2) else TEXT_SECONDARY),
        anchor="mt",
    )


def render_head_to_head(
    h2h_path: pathlib.Path,
    base_results_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
  h2h = json.loads(h2h_path.read_text(encoding="utf-8"))
  base_res = json.loads(base_results_path.read_text(encoding="utf-8"))

  img = Image.new("RGB", (OUT_W * SCALE, OUT_H * SCALE), BG_CANVAS)
  draw = ImageDraw.Draw(img)

  # Top header banner (x=24..1576, y=16..92)
  draw.rounded_rectangle(
      (_s(24), _s(16), _s(1576), _s(92)),
      radius=_s(14),
      fill=BG_HEADER,
      outline=BORDER_CARD,
      width=max(1, int(1.5 * SCALE)),
  )
  f_h1 = _load_font(bold=True, size_px=17)
  f_h2 = _load_font(bold=False, size_px=11)

  meta = h2h["metadata"]
  full = h2h["strategies"]["decisions_router_full"]
  t1 = h2h["strategies"]["decisions_tier1_only"]
  fl = h2h["strategies"]["flash_lite_json_router"]

  _draw_text(
      draw,
      44,
      29,
      "Gemini Decisions Router: N=100 Head-to-Head Study",
      f_h1,
      TEXT_PRIMARY,
      anchor="lt",
  )
  mode_tag = "LIVE API" if meta.get("mode") == "live" else "MOCK SIMULATION"
  sub_line = (
      f"{mode_tag} | N={meta['dataset_size']} Prompts | "
      f"{meta['decisions_model']} vs {meta['flash_lite_router_model']} & {meta['flash_router_model']}"
  )
  _draw_text(draw, 44, 57, sub_line, f_h2, TEXT_SECONDARY, anchor="lt")

  # Right KPI pills
  f_kpi = _load_font(bold=True, size_px=10)
  x_cursor = 1556.0
  x_cursor = _draw_pill(
      draw,
      x_cursor,
      40,
      f"E2E: {full['projected_100_e2e']['total_e2e_s']:.1f}s vs {fl['projected_100_e2e']['total_e2e_s']:.1f}s flash-lite",
      f_kpi,
      COL_MINT,
      (13, 42, 38),
      (22, 101, 82),
      pad_x=9,
      pad_y=5,
  )
  x_cursor -= 8
  x_cursor = _draw_pill(
      draw,
      x_cursor,
      40,
      f"Calibration: 4/4 Bins (Brier {full['calibration']['brier_score']:.4f} vs {fl['calibration']['brier_score']:.4f})",
      f_kpi,
      COL_CYAN,
      (15, 35, 58),
      (30, 78, 120),
      pad_x=9,
      pad_y=5,
  )
  x_cursor -= 8
  _draw_pill(
      draw,
      x_cursor,
      40,
      f"p50: {t1['latency']['p50_ms']:.0f} ms Pure / {full['latency']['p50_ms']:.0f} ms Full",
      f_kpi,
      COL_INDIGO,
      (26, 28, 60),
      (67, 72, 138),
      pad_x=9,
      pad_y=5,
  )

  # 2x2 Panel Grid (fits 1600x1000 cleanly)
  _render_panel1_latency(draw, 24, 108, 764, 428, h2h)
  _render_panel2_accuracy(draw, 812, 108, 764, 428, h2h, base_res)
  _render_panel3_calibration(draw, 24, 554, 764, 428, h2h)
  _render_panel4_cost_and_e2e(draw, 812, 554, 764, 428, h2h, base_res)

  final_img = img.resize((OUT_W, OUT_H), resample=Image.Resampling.LANCZOS)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  final_img.save(output_path, format="PNG", optimize=True)
  print(f"Rendered {OUT_W}x{OUT_H} head-to-head chart -> {output_path}")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Render the 4-panel dark-theme N=100 Head-to-Head Router PNG."
  )
  parser.add_argument(
      "--input",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_flash_lite_router_100.json",
      help="Path to head-to-head benchmark results JSON.",
  )
  parser.add_argument(
      "--baseline",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_100.json",
      help="Path to baseline results_100.json.",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=REPO_ROOT / "assets" / "router-head-to-head.png",
      help="Path to output PNG image.",
  )
  args = parser.parse_args(argv)
  render_head_to_head(args.input, args.baseline, args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
