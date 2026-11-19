#!/usr/bin/env python3
"""Head-to-head 100-prompt router benchmark: flash-lite JSON vs gemini-decision-1.

Evaluates all 100 stratified engineering and agentic prompts in
`benchmarks/prompts_100.json` across:
  1. `decisions_router_full`: Tier 0 fast rules + Tier 1 `gemini-decision-1`
     single-pass prefill (0 output tokens).
  2. `decisions_tier1_only`: 100% pure `gemini-decision-1` single-pass prefill
     (0 output tokens).
  3. `flash_lite_json_router`: `gemini-3.1-flash-lite-preview` (fallback
     `gemini-2.5-flash-lite`) via `GenerateContent` with
     `responseMimeType: "application/json"`, `thinkingConfig: {"thinkingBudget": 0}`,
     and `maxOutputTokens: 96`.
  4. `flash_json_router`: `gemini-2.5-flash` via `GenerateContent` across all
     100 prompts with `responseMimeType: "application/json"`,
     `thinkingConfig: {"thinkingBudget": 0}`, and `maxOutputTokens: 96`
     (plus comparison against default-thinking `gemini-2.5-flash` from
     `benchmarks/results_100.json`).

Streams live per-prompt routing lines and a rich `ROUTED WITH INTENT` summary
table to `stdout` (`flush=True`), with `--quiet` for summary-only output and
`--mock` for deterministic offline unit testing (<0.5s, zero network calls).
Pure Python 3.11+ standard library + `decisions_engine`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import concurrent.futures
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import pathlib
import ssl
import statistics
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import decisions_engine  # noqa: E402

TIERS = ("flash_lite", "flash", "pro")
TIER_RANKS = {"flash_lite": 1, "flash": 2, "pro": 3}

FIVE_WAY_TASK_CLASSES = (
    "quick_lookup",
    "coding_edit",
    "writing_docs",
    "data_analysis",
    "deep_debug_architecture",
)

ALL_TASK_CLASSES = FIVE_WAY_TASK_CLASSES + ("explicit_override",)

CATEGORIES_ORDER = (
    "quick_lookup",
    "coding_edit",
    "writing_docs",
    "data_analysis",
    "deep_debug_architecture",
    "agentic_lifecycle",
)

# Public Gemini API per-1M token pricing (USD), aligned with run_benchmark.py.
PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gemini_decision_1": {"input": 0.10, "output": 0.00},
    "flash_lite": {"input": 0.10, "output": 0.40},
    "flash_lite_3_1_preview": {"input": 0.25, "output": 1.50},
    "flash": {"input": 0.30, "output": 2.50},
    "pro": {"input": 1.25, "output": 10.00},
}

DEFAULT_TIER_DISPLAY_MODELS: dict[str, str] = {
    "flash_lite": "gemini-3.1-flash-lite",
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
}

DEFAULT_COMPLETION_STATS: dict[str, dict[str, float]] = {
    "flash_lite": {"mean_latency_ms": 1435.0, "mean_cost_usd": 0.000143},
    "flash": {"mean_latency_ms": 2726.9, "mean_cost_usd": 0.001285},
    "pro": {"mean_latency_ms": 4742.6, "mean_cost_usd": 0.005151},
}

FLASH_LITE_CANDIDATES = (
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
)
FLASH_MODEL = "gemini-2.5-flash"

STRUCTURED_JSON_ROUTER_SYSTEM = (
    "You are an engineering prompt router. Classify the user request into a"
    " model tier, task_class, and calibrated confidence in [0.0, 1.0]. Return"
    " ONLY compact JSON matching"
    ' {"tier":"flash_lite|flash|pro","task_class":"quick_lookup|coding_edit|'
    'writing_docs|data_analysis|deep_debug_architecture|explicit_override",'
    '"confidence":0.85}.'
    " Routing rules: use flash_lite for simple factual/CLI lookups, syntax"
    " checks, lightweight subagent file reads, or short standalone"
    " confirmations; use flash for standard single-service code edits, tests,"
    " documentation, or standard analytical SQL queries; use pro for"
    " multi-region or distributed architecture, subtle concurrency/memory/"
    "security debugging, complex causal/statistical root-cause investigations,"
    " or explicit user requests to use a frontier/pro/opus model."
)


def _percentile(values: list[float], q: float) -> float:
  """Computes linear-interpolated percentile q in [0, 100]."""
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return round(ordered[0], 2)
  pos = (len(ordered) - 1) * (q / 100.0)
  lo = int(math.floor(pos))
  hi = int(math.ceil(pos))
  if lo == hi:
    return round(ordered[lo], 2)
  frac = pos - lo
  return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
  """Returns count, min, p50, p90, p95, p99, max, mean, and total_ms."""
  if not values:
    return {
        "count": 0,
        "min_ms": 0.0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.0,
        "mean_ms": 0.0,
        "total_ms": 0.0,
    }
  return {
      "count": len(values),
      "min_ms": round(min(values), 2),
      "p50_ms": _percentile(values, 50),
      "p90_ms": _percentile(values, 90),
      "p95_ms": _percentile(values, 95),
      "p99_ms": _percentile(values, 99),
      "max_ms": round(max(values), 2),
      "mean_ms": round(statistics.fmean(values), 2),
      "total_ms": round(sum(values), 2),
  }


def token_cost_usd(
    pricing_key: str, input_tokens: float, output_tokens: float
) -> float:
  rates = PRICING_PER_1M[pricing_key]
  return (
      input_tokens * rates["input"] + output_tokens * rates["output"]
  ) / 1_000_000.0


def _short_prompt_snippet(prompt: str, width: int = 38) -> str:
  single = " ".join(prompt.split())
  if len(single) <= width:
    return single
  return single[: width - 3] + "..."


def _classify_backend_bucket(
    rec: Mapping[str, Any], item: Mapping[str, Any]
) -> str:
  """Classifies routing mechanism into fast_rules / thread_lock / gemini_api."""
  if rec.get("thread_floor_applied"):
    return "thread_lock"
  turn_cnt = int(item.get("user_turn_count") or 1)
  t_state = item.get("thread_state")
  if (
      turn_cnt >= 2
      and isinstance(t_state, Mapping)
      and str(rec.get("backend") or "") not in ("", "static_always_pro")
      and not str(rec.get("backend") or "").startswith("generateContent:")
  ):
    return "thread_lock"
  if not rec.get("api_called", True):
    return "fast_rules"
  backend_str = str(rec.get("backend") or "gemini_api")
  if backend_str.startswith("generateContent:"):
    return backend_str.split(":", 1)[1]
  if backend_str in ("mock_decisions_api", "gemini_api", "gemini_adc"):
    return "gemini_api"
  if backend_str.startswith("vertex_adc"):
    return "vertex_adc"
  return backend_str


def _safe_print(text: str = "") -> None:
  try:
    print(text, flush=True)
  except BrokenPipeError:
    try:
      sys.stdout.close()
    except OSError:
      pass
    raise SystemExit(0) from None


def format_live_route_line(
    idx: int,
    total: int,
    rec: Mapping[str, Any],
    item: Mapping[str, Any],
    tier_models: Mapping[str, str],
) -> str:
  """Formats a single live per-prompt routing telemetry line."""
  status = "PASS" if rec.get("tier_correct") else "MISS"
  cat = str(rec.get("category") or item.get("category") or "")[:18]
  tier = str(rec.get("recommended_tier") or "flash")
  raw_model = tier_models.get(
      tier, DEFAULT_TIER_DISPLAY_MODELS.get(tier, tier)
  )
  model_lbl = raw_model.removesuffix("-preview")[:23]
  conf = float(rec.get("confidence") or 0.0)
  lat_ms = float(rec.get("latency_ms") or 0.0)
  b_bucket = _classify_backend_bucket(rec, item).removesuffix("-preview")[:22]
  snippet = _short_prompt_snippet(str(item.get("prompt") or ""), width=36)
  return (
      f"[{idx:03d}/{total:03d}] {status:4s}  {cat:<18s} -> "
      f"{tier:<10s} ({model_lbl:<21s}) | conf={conf:0.2f} | "
      f"{lat_ms:7.1f} ms | {b_bucket:<22s} | \"{snippet}\""
  )


def _build_tls_opener() -> urllib.request.OpenerDirector:
  ctx = ssl.create_default_context()
  ctx.minimum_version = ssl.TLSVersion.TLSv1_2
  return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


@dataclasses.dataclass
class LiveAuthContext:
  """Resolved live HTTP headers and base URL without exposing raw secrets."""

  settings: decisions_engine.Settings
  env: Mapping[str, str]
  headers: dict[str, str]
  opener: urllib.request.OpenerDirector
  retried_calls: int = 0

  def generate_content_url(self, model: str) -> str:
    clean_model = model.removeprefix("models/")
    if self.settings.auth_mode == "vertex_adc":
      project = self.settings.project_id
      loc = self.settings.location
      base = (
          self.settings.vertex_base_override
          if self.settings.vertex_base_override
          else (
              "https://aiplatform.googleapis.com"
              if loc == "global"
              else f"https://{loc}-aiplatform.googleapis.com"
          )
      )
      return (
          f"{base}/{self.settings.vertex_api_version}/projects/{project}"
          f"/locations/{loc}/publishers/google/models/{clean_model}:generateContent"
      )
    base = self.settings.api_base.rstrip("/")
    return f"{base}/models/{clean_model}:generateContent"


def build_live_auth_context() -> LiveAuthContext:
  """Resolves credentials via `decisions_engine.load_settings()`."""
  env = decisions_engine.effective_environ()
  settings = decisions_engine.load_settings(environ=env)
  headers: dict[str, str] = {"Content-Type": "application/json"}

  if settings.auth_mode == "api_key":
    api_key = str(env.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
      raise decisions_engine.ConfigError(
          "GEMINI_API_KEY is empty in api_key mode."
      )
    headers["x-goog-api-key"] = api_key
  else:
    scopes = (
        decisions_engine.ADC_SCOPES_VERTEX
        if settings.auth_mode == "vertex_adc"
        else decisions_engine.ADC_SCOPES_GEMINI
    )
    adc = decisions_engine.resolve_adc_token(
        scopes, project_hint=settings.project_id, environ=env
    )
    headers["Authorization"] = f"Bearer {adc.access_token.strip()}"
    eff_proj = (settings.project_id or adc.project_id).strip()
    if settings.auth_mode == "gemini_adc" and eff_proj:
      headers["x-goog-user-project"] = eff_proj
    if eff_proj and not settings.project_id:
      settings = dataclasses.replace(settings, project_id=eff_proj)

  return LiveAuthContext(
      settings=settings,
      env=env,
      headers=headers,
      opener=_build_tls_opener(),
  )


def call_json_router_live(
    ctx: LiveAuthContext,
    model: str,
    prompt: str,
    *,
    thinking_budget: int | None = 0,
    max_output_tokens: int = 96,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
  """Calls GenerateContent configured as a low-latency JSON classifier."""
  gen_cfg: dict[str, Any] = {
      "responseMimeType": "application/json",
      "maxOutputTokens": max_output_tokens,
      "temperature": 0.0,
  }
  if thinking_budget is not None:
    gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}

  payload: dict[str, Any] = {
      "contents": [
          {"role": "user", "parts": [{"text": f"User request:\n{prompt}"}]}
      ],
      "systemInstruction": {"parts": [{"text": STRUCTURED_JSON_ROUTER_SYSTEM}]},
      "generationConfig": gen_cfg,
  }
  body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
  url = ctx.generate_content_url(model)

  last_err: Exception | None = None
  for attempt in range(3):
    req = urllib.request.Request(
        url, data=body, headers=dict(ctx.headers), method="POST"
    )
    t0 = time.perf_counter()
    try:
      with ctx.opener.open(req, timeout=timeout_s) as resp:
        raw = resp.read(2 * 1024 * 1024)
      latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
      data = json.loads(raw.decode("utf-8"))
      break
    except urllib.error.HTTPError as e:
      try:
        err_bytes = e.read(8192) if e.fp is not None else b""
      finally:
        e.close()
      if e.code in (429, 500, 502, 503, 504) and attempt < 2:
        ctx.retried_calls += 1
        time.sleep(0.8 * (attempt + 1))
        last_err = RuntimeError(f"HTTP {e.code} on {model}")
        continue
      msg = err_bytes.decode("utf-8", "replace")[:200]
      raise RuntimeError(
          f"generateContent {model} failed HTTP {e.code}: {msg}"
      ) from None
    except (TimeoutError, urllib.error.URLError) as e:
      if attempt < 2:
        ctx.retried_calls += 1
        time.sleep(0.8 * (attempt + 1))
        last_err = e
        continue
      raise RuntimeError(
          f"generateContent {model} network error: {type(e).__name__}"
      ) from None
  else:
    raise RuntimeError(f"generateContent {model} exhausted retries: {last_err}")

  usage = data.get("usageMetadata") or {}
  prompt_tok = int(usage.get("promptTokenCount") or 0)
  cand_tok = int(usage.get("candidatesTokenCount") or 0)
  thought_tok = int(usage.get("thoughtsTokenCount") or 0)
  total_tok = int(
      usage.get("totalTokenCount") or (prompt_tok + cand_tok + thought_tok)
  )
  out_tok = cand_tok + thought_tok
  if out_tok == 0 and total_tok > prompt_tok:
    out_tok = total_tok - prompt_tok

  text_parts: list[str] = []
  candidates = data.get("candidates") or []
  finish_reason = ""
  if candidates and isinstance(candidates[0], Mapping):
    finish_reason = str(candidates[0].get("finishReason") or "")
    content = candidates[0].get("content") or {}
    for part in content.get("parts") or []:
      if (
          isinstance(part, Mapping)
          and not part.get("thought")
          and isinstance(part.get("text"), str)
      ):
        text_parts.append(part["text"])

  return {
      "model": model,
      "latency_ms": latency_ms,
      "prompt_tokens": prompt_tok,
      "candidates_tokens": cand_tok,
      "thoughts_tokens": thought_tok,
      "output_tokens": out_tok,
      "total_tokens": total_tok,
      "finish_reason": finish_reason,
      "text": "".join(text_parts).strip(),
  }


def probe_router_models(ctx: LiveAuthContext) -> tuple[str, str]:
  """Selects reachable flash-lite model (3.1 preview -> 2.5) and warms TLS."""
  chosen_lite = ""
  last_exc: Exception | None = None
  for candidate in FLASH_LITE_CANDIDATES:
    try:
      call_json_router_live(
          ctx, candidate, "What is 2+2?", thinking_budget=0, max_output_tokens=96
      )
      chosen_lite = candidate
      break
    except Exception as exc:  # pylint: disable=broad-except
      last_exc = exc
      continue
  if not chosen_lite:
    raise RuntimeError(f"No reachable flash-lite model: {last_exc}")

  call_json_router_live(
      ctx, FLASH_MODEL, "What is 2+2?", thinking_budget=0, max_output_tokens=96
  )
  return chosen_lite, FLASH_MODEL


def parse_json_router_output(raw_text: str) -> dict[str, Any]:
  """Parses JSON router output, tracking strict JSON and schema errors."""
  cleaned = raw_text.strip()
  fenced = False
  if cleaned.startswith("```"):
    fenced = True
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]
    cleaned = "\n".join(lines).strip()

  try:
    obj = json.loads(cleaned)
  except ValueError:
    return {
        "tier": "flash",
        "task_class": "coding_edit",
        "confidence": 0.5,
        "json_parse_error": True,
        "schema_error": True,
        "markdown_fenced": fenced,
    }

  if not isinstance(obj, Mapping):
    return {
        "tier": "flash",
        "task_class": "coding_edit",
        "confidence": 0.5,
        "json_parse_error": False,
        "schema_error": True,
        "markdown_fenced": fenced,
    }

  raw_tier = str(obj.get("tier") or "").strip().lower().replace("-", "_")
  tier_aliases = {
      "lite": "flash_lite",
      "gemini_flash_lite": "flash_lite",
      "gemini_flash": "flash",
      "gemini_pro": "pro",
  }
  norm_tier = tier_aliases.get(raw_tier, raw_tier)
  tier_valid = norm_tier in TIER_RANKS
  tier = norm_tier if tier_valid else "flash"

  raw_tc = str(obj.get("task_class") or "").strip().lower().replace("-", "_")
  tc_valid = raw_tc in ALL_TASK_CLASSES
  tc = raw_tc if tc_valid else "coding_edit"

  raw_conf = obj.get("confidence")
  conf_valid = isinstance(raw_conf, (int, float)) and not isinstance(
      raw_conf, bool
  )
  if conf_valid:
    conf = float(raw_conf)
    if math.isnan(conf) or math.isinf(conf) or conf < 0.0 or conf > 1.0:
      conf_valid = False
      conf = max(0.0, min(1.0, 0.5 if math.isnan(conf) else conf))
  else:
    conf = 0.5

  schema_error = (not tier_valid) or (not tc_valid) or (not conf_valid)
  return {
      "tier": tier,
      "task_class": tc,
      "confidence": round(conf, 4),
      "json_parse_error": False,
      "schema_error": schema_error,
      "markdown_fenced": fenced,
  }


# ---- Deterministic Offline Mock (`--mock`) ----------------------------------


def _det_hash_float(seed: str, salt: str = "") -> float:
  digest = hashlib.sha256(f"{seed}|{salt}".encode("utf-8")).digest()
  return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _mock_json_router_item(
    item: Mapping[str, Any], *, model: str, variant: str
) -> dict[str, Any]:
  """Deterministic simulation of GenerateContent JSON routers for offline CI."""
  prompt = str(item["prompt"])
  cat = str(item["category"])
  exp_tier = str(item["expected_tier"])
  exp_tc = str(item["expected_task_class"])
  turn_count = int(item.get("user_turn_count") or 1)

  u_lat = _det_hash_float(prompt, f"{variant}_lat")
  u_err = _det_hash_float(prompt, f"{variant}_err")
  u_conf = _det_hash_float(prompt, f"{variant}_conf")

  if variant == "flash_lite":
    sim_ms = round(580.0 + 240.0 * u_lat, 2)
    miss_thresh = 0.84
  else:
    sim_ms = round(430.0 + 180.0 * u_lat, 2)
    miss_thresh = 0.86

  if turn_count >= 2 and exp_tier == "pro":
    pred_tier = "flash_lite"
    pred_tc = "quick_lookup"
  elif cat == "data_analysis" and exp_tier == "pro" and u_err > miss_thresh:
    pred_tier = "flash"
    pred_tc = "data_analysis"
  elif exp_tier == "flash" and u_err > 0.90:
    pred_tier = "pro"
    pred_tc = exp_tc
  else:
    pred_tier = exp_tier
    pred_tc = exp_tc

  # Verbalized LLM confidence clusters in [0.90, 0.99] even on errors.
  verbal_conf = round(0.91 + 0.08 * u_conf, 2)
  p_tok = max(145, len(prompt) // 4 + 140)
  c_tok = 22

  return {
      "id": item["id"],
      "category": cat,
      "expected_tier": exp_tier,
      "expected_task_class": exp_tc,
      "recommended_tier": pred_tier,
      "task_class": pred_tc,
      "confidence": verbal_conf,
      "backend": f"generateContent:{model}",
      "api_called": True,
      "fallback_used": False,
      "thread_floor_applied": False,
      "json_parse_error": False,
      "schema_error": False,
      "latency_ms": sim_ms,
      "input_tokens": p_tok,
      "candidates_tokens": c_tok,
      "thoughts_tokens": 0,
      "output_tokens": c_tok,
      "tier_correct": pred_tier == exp_tier,
      "task_class_correct": pred_tc == exp_tc,
  }


def _mock_decisions_item(
    item: Mapping[str, Any], *, enable_fast_rules: bool
) -> dict[str, Any]:
  """Deterministic offline mock for `decisions_engine.route_prompt`."""
  prompt = str(item["prompt"])
  exp_tier = str(item["expected_tier"])
  exp_tc = str(item["expected_task_class"])
  turn_count = int(item.get("user_turn_count") or 1)
  thread_state = item.get("thread_state")
  if not isinstance(thread_state, Mapping):
    thread_state = None

  u_lat = _det_hash_float(prompt, "dec_lat")
  u_acc = _det_hash_float(prompt, "dec_acc")
  sim_ms = round(185.0 + 95.0 * u_lat, 2)
  est_in_tokens = max(96, len(prompt) // 4 + 120)
  captured = {"api_called": False}

  def _decision_fn(
      call_args: Mapping[str, Any], _settings: Any
  ) -> dict[str, Any]:
    del call_args
    captured["api_called"] = True
    base_tc = (
        exp_tc
        if exp_tc in FIVE_WAY_TASK_CLASSES
        else ("deep_debug_architecture" if exp_tier == "pro" else "quick_lookup")
    )
    if u_acc > 0.94 and exp_tier == "flash_lite":
      chosen_tc = "coding_edit"
      conf = 0.72
    else:
      chosen_tc = base_tc
      conf = round(0.76 + 0.21 * _det_hash_float(prompt, "dec_conf"), 4)

    rem = round(1.0 - conf, 4)
    per_other = round(rem / (len(FIVE_WAY_TASK_CLASSES) - 1), 4)
    tc_probs = {k: per_other for k in FIVE_WAY_TASK_CLASSES}
    tc_probs[chosen_tc] = round(
        1.0 - per_other * (len(FIVE_WAY_TASK_CLASSES) - 1), 4
    )

    if exp_tier == "pro":
      p_pro = round(0.80 + 0.16 * u_lat, 4)
      eff_probs = {"0": 0.01, "1": 0.04, "2": 0.25, "3": 0.70}
      eff_score = 2.65
    elif chosen_tc == "quick_lookup" and exp_tier == "flash_lite":
      p_pro = round(0.03 + 0.06 * u_lat, 4)
      eff_probs = {"0": 0.84, "1": 0.13, "2": 0.02, "3": 0.01}
      eff_score = 0.20
    else:
      p_pro = round(0.12 + 0.20 * u_lat, 4)
      eff_probs = {"0": 0.05, "1": 0.56, "2": 0.34, "3": 0.05}
      eff_score = 1.39

    return {
        "backend": "mock_decisions_api",
        "model": "gemini-decision-1",
        "latency_ms": sim_ms,
        "usage": {"input_tokens": est_in_tokens},
        "answers": {
            "task_class": {
                "type": "choice",
                "choice": chosen_tc,
                "confidence": conf,
                "probabilities": tc_probs,
            },
            "needs_pro_reasoning": {
                "type": "noul",
                "noul": p_pro,
                "value": p_pro >= 0.5,
                "confidence": conf,
            },
            "effort_rubric": {
                "type": "score",
                "score": eff_score,
                "level": max(range(4), key=lambda idx: eff_probs[str(idx)]),
                "confidence": conf,
                "probabilities": eff_probs,
            },
        },
    }

  t0 = time.perf_counter()
  res = decisions_engine.route_prompt(
      prompt,
      user_turn_count=turn_count,
      thread_state=thread_state,
      enable_fast_rules=enable_fast_rules,
      ignore_cooldown=True,
      environ={},
      decision_fn=_decision_fn,
  )
  local_ms = round((time.perf_counter() - t0) * 1000.0, 3)
  wall_ms = sim_ms if captured["api_called"] else max(0.01, local_ms)
  in_tok = est_in_tokens if captured["api_called"] else 0

  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": exp_tier,
      "expected_task_class": exp_tc,
      "recommended_tier": res["recommended_tier"],
      "task_class": res["task_class"],
      "confidence": float(res["confidence"]),
      "backend": res["backend"],
      "api_called": captured["api_called"],
      "fallback_used": False,
      "thread_floor_applied": bool(res.get("thread_floor_applied")),
      "json_parse_error": False,
      "schema_error": False,
      "latency_ms": wall_ms,
      "input_tokens": in_tok,
      "candidates_tokens": 0,
      "thoughts_tokens": 0,
      "output_tokens": 0,
      "tier_correct": res["recommended_tier"] == exp_tier,
      "task_class_correct": res["task_class"] == exp_tc,
  }


def evaluate_json_router_item_live(
    item: Mapping[str, Any],
    *,
    ctx: LiveAuthContext,
    model: str,
) -> dict[str, Any]:
  """Evaluates a live GenerateContent JSON router on a single dataset item."""
  prompt = str(item["prompt"])
  resp = call_json_router_live(
      ctx,
      model,
      prompt,
      thinking_budget=0,
      max_output_tokens=96,
      timeout_s=30.0,
  )
  parsed = parse_json_router_output(resp["text"])
  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": item["expected_tier"],
      "expected_task_class": item["expected_task_class"],
      "recommended_tier": parsed["tier"],
      "task_class": parsed["task_class"],
      "confidence": parsed["confidence"],
      "backend": f"generateContent:{model}",
      "api_called": True,
      "fallback_used": False,
      "thread_floor_applied": False,
      "json_parse_error": parsed["json_parse_error"],
      "schema_error": parsed["schema_error"],
      "markdown_fenced": parsed["markdown_fenced"],
      "finish_reason": resp["finish_reason"],
      "latency_ms": resp["latency_ms"],
      "input_tokens": resp["prompt_tokens"],
      "candidates_tokens": resp["candidates_tokens"],
      "thoughts_tokens": resp["thoughts_tokens"],
      "output_tokens": resp["output_tokens"],
      "tier_correct": parsed["tier"] == item["expected_tier"],
      "task_class_correct": parsed["task_class"] == item["expected_task_class"],
  }


def evaluate_decisions_item_live(
    item: Mapping[str, Any],
    *,
    ctx: LiveAuthContext,
    enable_fast_rules: bool,
) -> dict[str, Any]:
  """Evaluates live `gemini-decision-1` on one dataset item."""
  prompt = str(item["prompt"])
  turn_count = int(item.get("user_turn_count") or 1)
  thread_state = item.get("thread_state")
  if not isinstance(thread_state, Mapping):
    thread_state = None

  captured: dict[str, Any] = {
      "api_called": False,
      "input_tokens": 0,
      "api_latency_ms": 0.0,
      "fallback_used": False,
  }

  def _wrapped_live(
      call_args: Mapping[str, Any], _settings: Any
  ) -> dict[str, Any]:
    router_cfg = dataclasses.replace(ctx.settings, timeout_s=12.0)
    last_err: decisions_engine.DecisionError | None = None
    for attempt in range(3):
      t_call = time.perf_counter()
      try:
        dec = decisions_engine.create_decision(
            call_args, router_cfg, environ=ctx.env
        )
        captured["api_called"] = True
        captured["input_tokens"] = int(
            dec.get("usage", {}).get("input_tokens")
            or max(48, len(prompt) // 4 + 95)
        )
        captured["api_latency_ms"] = round(
            (time.perf_counter() - t_call) * 1000.0, 2
        )
        return dec
      except (
          decisions_engine.QuotaError,
          decisions_engine.UnavailableError,
          decisions_engine.DeadlineError,
      ) as exc:
        last_err = exc
        if attempt < 2:
          ctx.retried_calls += 1
          time.sleep(0.6 * (attempt + 1))
          continue
        captured["fallback_used"] = True
        raise
    assert last_err is not None
    raise last_err

  t0 = time.perf_counter()
  res = decisions_engine.route_prompt(
      prompt,
      user_turn_count=turn_count,
      thread_state=thread_state,
      enable_fast_rules=enable_fast_rules,
      ignore_cooldown=True,
      settings=ctx.settings,
      environ=ctx.env,
      decision_fn=_wrapped_live,
  )
  measured_ms = round((time.perf_counter() - t0) * 1000.0, 2)
  wall_ms = (
      captured["api_latency_ms"] if captured["api_called"] else measured_ms
  )

  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": item["expected_tier"],
      "expected_task_class": item["expected_task_class"],
      "recommended_tier": res["recommended_tier"],
      "task_class": res["task_class"],
      "confidence": float(res["confidence"]),
      "backend": res["backend"],
      "api_called": captured["api_called"],
      "fallback_used": captured["fallback_used"],
      "thread_floor_applied": bool(res.get("thread_floor_applied")),
      "json_parse_error": False,
      "schema_error": False,
      "latency_ms": wall_ms,
      "input_tokens": captured["input_tokens"],
      "candidates_tokens": 0,
      "thoughts_tokens": 0,
      "output_tokens": 0,
      "tier_correct": res["recommended_tier"] == item["expected_tier"],
      "task_class_correct": res["task_class"] == item["expected_task_class"],
  }


def _compute_auroc(
    pos_scores: list[float], neg_scores: list[float]
) -> float:
  """Mann-Whitney U AUROC measuring whether confidence separates hits from misses."""
  if not pos_scores or not neg_scores:
    return 0.5
  wins = 0.0
  pairs = len(pos_scores) * len(neg_scores)
  for p in pos_scores:
    for n in neg_scores:
      if p > n:
        wins += 1.0
      elif p == n:
        wins += 0.5
  return round(wins / pairs, 4)


def compute_calibration_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
  """Computes 4-bin and 10-bin ECE, Brier score, AUROC, and bin spread."""
  bins_4 = (
      ("0.00-0.70", 0.0, 0.70, False),
      ("0.70-0.80", 0.70, 0.80, False),
      ("0.80-0.90", 0.80, 0.90, False),
      ("0.90-1.00", 0.90, 1.00, True),
  )
  n_total = len(records)
  ece_4 = 0.0
  populated_bins = 0
  rows_4: list[dict[str, Any]] = []
  for label, lo, hi, inc_hi in bins_4:
    bucket = [
        r
        for r in records
        if (
            (lo <= float(r["confidence"]) <= hi)
            if inc_hi
            else (lo <= float(r["confidence"]) < hi)
        )
    ]
    cnt = len(bucket)
    if cnt > 0:
      populated_bins += 1
      m_conf = statistics.fmean(float(r["confidence"]) for r in bucket)
      acc = sum(1 for r in bucket if r["tier_correct"]) / cnt
      gap = abs(acc - m_conf)
      ece_4 += (cnt / max(1, n_total)) * gap
      rows_4.append({
          "bin": label,
          "count": cnt,
          "mean_confidence": round(m_conf, 4),
          "empirical_accuracy": round(acc, 4),
          "calibration_gap": round(gap, 4),
      })
    else:
      rows_4.append({
          "bin": label,
          "count": 0,
          "mean_confidence": 0.0,
          "empirical_accuracy": 0.0,
          "calibration_gap": 0.0,
      })

  ece_10 = 0.0
  for idx in range(10):
    lo = idx * 0.1
    hi = (idx + 1) * 0.1
    inc_hi = idx == 9
    bucket = [
        r
        for r in records
        if (
            (lo <= float(r["confidence"]) <= hi)
            if inc_hi
            else (lo <= float(r["confidence"]) < hi)
        )
    ]
    if bucket:
      m_conf = statistics.fmean(float(r["confidence"]) for r in bucket)
      acc = sum(1 for r in bucket if r["tier_correct"]) / len(bucket)
      ece_10 += (len(bucket) / max(1, n_total)) * abs(acc - m_conf)

  all_confs = [float(r["confidence"]) for r in records]
  brier = (
      statistics.fmean(
          (float(r["confidence"]) - (1.0 if r["tier_correct"] else 0.0)) ** 2
          for r in records
      )
      if records
      else 0.0
  )
  wrong = [float(r["confidence"]) for r in records if not r["tier_correct"]]
  right = [float(r["confidence"]) for r in records if r["tier_correct"]]
  conf_stdev = (
      round(statistics.pstdev(all_confs), 4) if len(all_confs) > 1 else 0.0
  )

  return {
      "ece_4bin": round(ece_4, 4),
      "ece_10bin": round(ece_10, 4),
      "brier_score": round(brier, 4),
      "confidence_stdev": conf_stdev,
      "populated_bins_4": populated_bins,
      "confidence_auroc": _compute_auroc(right, wrong),
      "mean_confidence_overall": (
          round(statistics.fmean(all_confs), 4) if all_confs else 0.0
      ),
      "mean_confidence_when_correct": (
          round(statistics.fmean(right), 4) if right else 0.0
      ),
      "mean_confidence_when_wrong": (
          round(statistics.fmean(wrong), 4) if wrong else 0.0
      ),
      "wrong_count": len(wrong),
      "high_conf_error_count_ge_0_85": sum(1 for c in wrong if c >= 0.85),
      "bins": rows_4,
  }


def summarize_router_strategy(
    name: str,
    records: list[dict[str, Any]],
    *,
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    pricing_key: str,
    completion_stats: Mapping[str, Mapping[str, float]],
    always_pro_total_s: float,
    always_pro_total_cost: float,
    alt_pricing_key: str | None = None,
) -> dict[str, Any]:
  """Aggregates latency, accuracy, Pro-miss, calibration, backends, E2E, cost."""
  n = len(records)
  latencies = [float(r["latency_ms"]) for r in records]
  api_latencies = [
      float(r["latency_ms"]) for r in records if r.get("api_called", True)
  ]
  tier0_count = sum(1 for r in records if not r.get("api_called", True))

  backend_breakdown: dict[str, int] = {
      "fast_rules": 0,
      "gemini_api": 0,
      "thread_lock": 0,
      "generate_content_json": 0,
  }
  for r in records:
    it = dataset_by_id[str(r["id"])]
    b_bucket = _classify_backend_bucket(r, it)
    if b_bucket == "fast_rules":
      backend_breakdown["fast_rules"] += 1
    elif b_bucket == "thread_lock":
      backend_breakdown["thread_lock"] += 1
    elif str(r.get("backend") or "").startswith("generateContent:"):
      backend_breakdown["generate_content_json"] += 1
    else:
      backend_breakdown["gemini_api"] += 1

  exact_hits = sum(1 for r in records if r["tier_correct"])
  non_lifecycle = [r for r in records if r["category"] != "agentic_lifecycle"]
  tc_5way_hits = sum(1 for r in non_lifecycle if r["task_class_correct"])
  tc_all_hits = sum(1 for r in records if r["task_class_correct"])

  expected_pro = [r for r in records if r["expected_tier"] == "pro"]
  pro_misses = [r for r in expected_pro if r["recommended_tier"] != "pro"]
  single_turn_pro = [
      r for r in non_lifecycle if r["expected_tier"] == "pro"
  ]
  single_turn_pro_misses = [
      r for r in single_turn_pro if r["recommended_tier"] != "pro"
  ]

  expected_non_pro = [r for r in records if r["expected_tier"] != "pro"]
  over_to_pro = [
      r for r in expected_non_pro if r["recommended_tier"] == "pro"
  ]
  under_any = [
      r
      for r in records
      if TIER_RANKS[r["recommended_tier"]] < TIER_RANKS[r["expected_tier"]]
  ]
  over_any = [
      r
      for r in records
      if TIER_RANKS[r["recommended_tier"]] > TIER_RANKS[r["expected_tier"]]
  ]

  thread_floor_hits = 0
  thread_floor_pro_misses = 0
  for r in records:
    it = dataset_by_id[str(r["id"])]
    turn_cnt = int(it.get("user_turn_count") or 1)
    t_state = it.get("thread_state")
    adj_tier = str(r["recommended_tier"])
    if turn_cnt >= 2 and isinstance(t_state, Mapping):
      locked = str(t_state.get("locked_tier") or "")
      if locked in TIER_RANKS and TIER_RANKS[locked] > TIER_RANKS[adj_tier]:
        adj_tier = locked
    if adj_tier == r["expected_tier"]:
      thread_floor_hits += 1
    if r["expected_tier"] == "pro" and adj_tier != "pro":
      thread_floor_pro_misses += 1

  json_parse_errors = sum(1 for r in records if r.get("json_parse_error"))
  schema_errors = sum(1 for r in records if r.get("schema_error"))
  markdown_fenced = sum(1 for r in records if r.get("markdown_fenced"))

  confusion: dict[str, dict[str, int]] = {
      exp_t: {pred_t: 0 for pred_t in TIERS} for exp_t in TIERS
  }
  routed_dist = {t: 0 for t in TIERS}
  for r in records:
    confusion[r["expected_tier"]][r["recommended_tier"]] += 1
    routed_dist[r["recommended_tier"]] += 1

  per_category: dict[str, dict[str, Any]] = {}
  for cat in CATEGORIES_ORDER:
    cat_recs = [r for r in records if r["category"] == cat]
    if not cat_recs:
      continue
    c_hits = sum(1 for r in cat_recs if r["tier_correct"])
    c_tc = sum(1 for r in cat_recs if r["task_class_correct"])
    c_pro = [r for r in cat_recs if r["expected_tier"] == "pro"]
    c_pro_miss = sum(1 for r in c_pro if r["recommended_tier"] != "pro")
    per_category[cat] = {
        "count": len(cat_recs),
        "tier_accuracy_pct": round(100.0 * c_hits / len(cat_recs), 1),
        "tier_correct": c_hits,
        "task_class_accuracy_pct": round(100.0 * c_tc / len(cat_recs), 1),
        "expected_pro_count": len(c_pro),
        "pro_miss_count": c_pro_miss,
        "mean_latency_ms": round(
            statistics.fmean(float(r["latency_ms"]) for r in cat_recs), 2
        ),
    }

  total_in = sum(int(r.get("input_tokens", 0)) for r in records)
  total_out = sum(int(r.get("output_tokens", 0)) for r in records)
  total_cand = sum(int(r.get("candidates_tokens", 0)) for r in records)
  total_thought = sum(int(r.get("thoughts_tokens", 0)) for r in records)

  cost_100 = token_cost_usd(pricing_key, total_in, total_out) * (
      100.0 / max(1, n)
  )
  tokens_block: dict[str, Any] = {
      "total_input_tokens": total_in,
      "total_output_tokens": total_out,
      "total_candidates_tokens": total_cand,
      "total_thoughts_tokens": total_thought,
      "mean_input_tokens": round(total_in / max(1, n), 1),
      "mean_output_tokens": round(total_out / max(1, n), 1),
      "mean_candidates_tokens": round(total_cand / max(1, n), 1),
      "mean_thoughts_tokens": round(total_thought / max(1, n), 1),
      "router_cost_per_100_usd": round(cost_100, 6),
  }
  if alt_pricing_key:
    alt_cost_100 = token_cost_usd(alt_pricing_key, total_in, total_out) * (
        100.0 / max(1, n)
    )
    tokens_block["router_cost_per_100_preview_tier_usd"] = round(
        alt_cost_100, 6
    )

  # Projected 100-prompt E2E wall-clock time & cost using routed tier counts.
  scale_100 = 100.0 / max(1, n)
  router_overhead_100_s = (sum(latencies) / 1000.0) * scale_100
  gen_100_s = (
      sum(
          routed_dist[t] * float(completion_stats[t]["mean_latency_ms"])
          for t in TIERS
      )
      / 1000.0
  ) * scale_100
  gen_100_cost = (
      sum(
          routed_dist[t] * float(completion_stats[t]["mean_cost_usd"])
          for t in TIERS
      )
      * scale_100
  )
  tot_100_s = router_overhead_100_s + gen_100_s
  tot_100_cost = cost_100 + gen_100_cost
  e2e_projection = {
      "router_overhead_s": round(router_overhead_100_s, 2),
      "generation_s": round(gen_100_s, 2),
      "total_e2e_s": round(tot_100_s, 2),
      "time_saved_vs_always_pro_pct": round(
          100.0 * (always_pro_total_s - tot_100_s) / max(0.01, always_pro_total_s),
          1,
      ),
      "router_cost_usd": round(cost_100, 5),
      "generation_cost_usd": round(gen_100_cost, 5),
      "total_cost_usd": round(tot_100_cost, 5),
      "cost_saved_vs_always_pro_pct": round(
          100.0
          * (always_pro_total_cost - tot_100_cost)
          / max(0.00001, always_pro_total_cost),
          1,
      ),
  }

  return {
      "strategy": name,
      "sample_size": n,
      "tier0_fast_rule_count": tier0_count,
      "tier1_api_call_count": n - tier0_count,
      "backend_breakdown": backend_breakdown,
      "latency": summarize_latencies(latencies),
      "api_only_latency": summarize_latencies(api_latencies),
      "tokens": tokens_block,
      "json_reliability": {
          "json_parse_error_count": json_parse_errors,
          "json_parse_error_rate_pct": round(
              100.0 * json_parse_errors / max(1, n), 1
          ),
          "schema_error_count": schema_errors,
          "schema_error_rate_pct": round(100.0 * schema_errors / max(1, n), 1),
          "markdown_fenced_count": markdown_fenced,
      },
      "accuracy": {
          "exact_tier_accuracy_pct": round(100.0 * exact_hits / max(1, n), 1),
          "exact_tier_correct": exact_hits,
          "single_turn_n90_tier_accuracy_pct": round(
              100.0
              * sum(1 for r in non_lifecycle if r["tier_correct"])
              / max(1, len(non_lifecycle)),
              1,
          ),
          "with_external_thread_floor_accuracy_pct": round(
              100.0 * thread_floor_hits / max(1, n), 1
          ),
          "task_class_5way_accuracy_pct": round(
              100.0 * tc_5way_hits / max(1, len(non_lifecycle)), 1
          ),
          "task_class_5way_correct": tc_5way_hits,
          "task_class_5way_total": len(non_lifecycle),
          "task_class_all_accuracy_pct": round(
              100.0 * tc_all_hits / max(1, n), 1
          ),
          "pro_miss_rate_pct": round(
              100.0 * len(pro_misses) / max(1, len(expected_pro)), 1
          ),
          "pro_miss_count": len(pro_misses),
          "expected_pro_count": len(expected_pro),
          "single_turn_pro_miss_rate_pct": round(
              100.0
              * len(single_turn_pro_misses)
              / max(1, len(single_turn_pro)),
              1,
          ),
          "single_turn_pro_miss_count": len(single_turn_pro_misses),
          "single_turn_expected_pro_count": len(single_turn_pro),
          "with_external_thread_floor_pro_miss_rate_pct": round(
              100.0 * thread_floor_pro_misses / max(1, len(expected_pro)), 1
          ),
          "over_route_to_pro_rate_pct": round(
              100.0 * len(over_to_pro) / max(1, len(expected_non_pro)), 1
          ),
          "over_route_to_pro_count": len(over_to_pro),
          "expected_non_pro_count": len(expected_non_pro),
          "under_route_any_rate_pct": round(
              100.0 * len(under_any) / max(1, n), 1
          ),
          "over_route_any_rate_pct": round(100.0 * len(over_any) / max(1, n), 1),
      },
      "routed_tier_distribution": routed_dist,
      "confusion_matrix": confusion,
      "per_category": per_category,
      "calibration": compute_calibration_metrics(records),
      "projected_100_e2e": e2e_projection,
  }


def _run_items_streaming(
    strategy_title: str,
    items: list[dict[str, Any]],
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    concurrency: int,
    quiet: bool,
    tier_models: Mapping[str, str],
) -> list[dict[str, Any]]:
  """Executes items and streams per-prompt routing lines to stdout."""
  total = len(items)
  if not quiet:
    _safe_print(
        f"\n--- Streaming Live Routes: {strategy_title} (N={total}) ---"
    )
  print_lock = threading.Lock()
  counter = {"done": 0}

  def _wrapped(item: dict[str, Any]) -> dict[str, Any]:
    rec = fn(item)
    if not quiet:
      with print_lock:
        counter["done"] += 1
        line = format_live_route_line(
            counter["done"], total, rec, item, tier_models
        )
        _safe_print(line)
    return rec

  if concurrency <= 1:
    return [_wrapped(item) for item in items]
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    return list(pool.map(_wrapped, items))


def _emit_cached_records_stream(
    strategy_title: str,
    records: list[dict[str, Any]],
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    *,
    quiet: bool,
    tier_models: Mapping[str, str],
) -> None:
  if quiet:
    return
  total = len(records)
  _safe_print(
      f"\n--- Baseline Live Routes: {strategy_title} (N={total}) ---"
  )
  for idx, rec in enumerate(records, start=1):
    it = dataset_by_id[str(rec["id"])]
    _safe_print(format_live_route_line(idx, total, rec, it, tier_models))


def run_head_to_head(
    *,
    dataset_path: pathlib.Path,
    baseline_results_path: pathlib.Path,
    output_path: pathlib.Path,
    mock: bool = False,
    concurrency: int = 1,
    rerun_decisions: bool = False,
    reuse_existing_output: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
  """Executes the 100-prompt head-to-head router comparison."""
  t_start = time.perf_counter()
  dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
  if not isinstance(dataset, list) or not dataset:
    raise ValueError("Dataset must be a non-empty JSON list.")
  dataset_by_id = {str(it["id"]): it for it in dataset}

  completion_stats: dict[str, Mapping[str, float]] = dict(
      DEFAULT_COMPLETION_STATS
  )
  always_pro_total_s = 473.21
  always_pro_total_cost = 0.5154
  tier_models = dict(DEFAULT_TIER_DISPLAY_MODELS)
  baseline_thinking_ref: dict[str, Any] | None = None

  if baseline_results_path.is_file():
    base_json = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    if "downstream_tier_models" in base_json.get("metadata", {}):
      tier_models.update(base_json["metadata"]["downstream_tier_models"])
    e2e_block = base_json.get("e2e", {})
    if "tier_completion_stats" in e2e_block:
      completion_stats = e2e_block["tier_completion_stats"]
    proj_pro = e2e_block.get("projected_100_prompts", {}).get("always_pro", {})
    if proj_pro:
      always_pro_total_s = float(proj_pro.get("total_e2e_s", always_pro_total_s))
      always_pro_total_cost = float(
          proj_pro.get("total_cost_usd", always_pro_total_cost)
      )

  if mock:
    ctx: LiveAuthContext | None = None
    flash_lite_model = "gemini-3.1-flash-lite-preview"
    flash_model = FLASH_MODEL
    auth_mode = "mock_offline"
    decisions_source = "mock_offline"

    dec_full_records = _run_items_streaming(
        "decisions_router_full (Tier 0 + gemini-decision-1)",
        dataset,
        lambda it: _mock_decisions_item(it, enable_fast_rules=True),
        concurrency=1,
        quiet=quiet,
        tier_models=tier_models,
    )
    dec_tier1_records = [
        _mock_decisions_item(it, enable_fast_rules=False) for it in dataset
    ]
    flash_lite_records = _run_items_streaming(
        f"flash_lite_json_router ({flash_lite_model})",
        dataset,
        lambda it: _mock_json_router_item(
            it, model=flash_lite_model, variant="flash_lite"
        ),
        concurrency=1,
        quiet=quiet,
        tier_models=tier_models,
    )
    flash_records = _run_items_streaming(
        f"flash_json_router ({flash_model})",
        dataset,
        lambda it: _mock_json_router_item(
            it, model=flash_model, variant="flash"
        ),
        concurrency=1,
        quiet=quiet,
        tier_models=tier_models,
    )
    elapsed_s = round(time.perf_counter() - t_start, 3)
  elif reuse_existing_output and output_path.is_file():
    ctx = None
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    base_json = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    flash_lite_model = str(existing["metadata"]["flash_lite_router_model"])
    flash_model = str(existing["metadata"]["flash_router_model"])
    auth_mode = str(existing["metadata"]["auth_mode"])
    decisions_source = str(existing["metadata"]["decisions_source"])
    dec_full_records = list(base_json["records"]["decisions_router_full"])
    dec_tier1_records = list(base_json["records"]["decisions_tier1_only"])
    flash_lite_records = list(existing["records"]["flash_lite_json_router"])
    flash_records = list(existing["records"]["flash_json_router"])
    baseline_thinking_ref = existing.get("reference_default_thinking_flash_router")
    elapsed_s = float(existing["metadata"].get("suite_wall_time_s", 0.0))
    _emit_cached_records_stream(
        "decisions_router_full (Tier 0 + gemini-decision-1)",
        dec_full_records,
        dataset_by_id,
        quiet=quiet,
        tier_models=tier_models,
    )
    _emit_cached_records_stream(
        f"flash_lite_json_router ({flash_lite_model})",
        flash_lite_records,
        dataset_by_id,
        quiet=quiet,
        tier_models=tier_models,
    )
  else:
    ctx = build_live_auth_context()
    flash_lite_model, flash_model = probe_router_models(ctx)
    auth_mode = ctx.settings.auth_mode

    if not rerun_decisions and baseline_results_path.is_file():
      base_json = json.loads(baseline_results_path.read_text(encoding="utf-8"))
      dec_full_records = list(base_json["records"]["decisions_router_full"])
      dec_tier1_records = list(base_json["records"]["decisions_tier1_only"])
      decisions_source = f"live_records:{baseline_results_path.name}"
      _emit_cached_records_stream(
          "decisions_router_full (Tier 0 + gemini-decision-1)",
          dec_full_records,
          dataset_by_id,
          quiet=quiet,
          tier_models=tier_models,
      )
      if "llm_json_router" in base_json.get("strategies", {}):
        ref_s = base_json["strategies"]["llm_json_router"]
        baseline_thinking_ref = {
            "model": base_json["metadata"].get(
                "llm_json_router_model", "gemini-2.5-flash"
            ),
            "thinking_config": "default_dynamic_thinking",
            "sample_size": ref_s["sample_size"],
            "latency": ref_s["latency"],
            "tokens": ref_s["tokens"],
            "accuracy": ref_s["accuracy"],
        }
    else:
      warmup_cfg = dataclasses.replace(ctx.settings, timeout_s=10.0)
      decisions_engine.create_decision(
          {
              "input": [{"text": "Warmup ping"}],
              "questions": {
                  "ok": {"type": "noul", "instructions": "Is this a short ping?"}
              },
          },
          warmup_cfg,
          environ=ctx.env,
      )
      dec_full_records = _run_items_streaming(
          "decisions_router_full (Tier 0 + gemini-decision-1)",
          dataset,
          lambda it: evaluate_decisions_item_live(
              it, ctx=ctx, enable_fast_rules=True
          ),
          concurrency=concurrency,
          quiet=quiet,
          tier_models=tier_models,
      )
      dec_tier1_records = _run_items_streaming(
          "decisions_tier1_only (Pure gemini-decision-1)",
          dataset,
          lambda it: evaluate_decisions_item_live(
              it, ctx=ctx, enable_fast_rules=False
          ),
          concurrency=concurrency,
          quiet=quiet,
          tier_models=tier_models,
      )
      decisions_source = "live_fresh_run"

    flash_lite_records = _run_items_streaming(
        f"flash_lite_json_router ({flash_lite_model})",
        dataset,
        lambda it: evaluate_json_router_item_live(
            it, ctx=ctx, model=flash_lite_model
        ),
        concurrency=concurrency,
        quiet=quiet,
        tier_models=tier_models,
    )
    flash_records = _run_items_streaming(
        f"flash_json_router ({flash_model})",
        dataset,
        lambda it: evaluate_json_router_item_live(
            it, ctx=ctx, model=flash_model
        ),
        concurrency=concurrency,
        quiet=quiet,
        tier_models=tier_models,
    )
    elapsed_s = round(time.perf_counter() - t_start, 3)

  dec_full_summary = summarize_router_strategy(
      "decisions_router_full",
      dec_full_records,
      dataset_by_id=dataset_by_id,
      pricing_key="gemini_decision_1",
      completion_stats=completion_stats,
      always_pro_total_s=always_pro_total_s,
      always_pro_total_cost=always_pro_total_cost,
  )
  dec_tier1_summary = summarize_router_strategy(
      "decisions_tier1_only",
      dec_tier1_records,
      dataset_by_id=dataset_by_id,
      pricing_key="gemini_decision_1",
      completion_stats=completion_stats,
      always_pro_total_s=always_pro_total_s,
      always_pro_total_cost=always_pro_total_cost,
  )
  flash_lite_summary = summarize_router_strategy(
      "flash_lite_json_router",
      flash_lite_records,
      dataset_by_id=dataset_by_id,
      pricing_key="flash_lite",
      completion_stats=completion_stats,
      always_pro_total_s=always_pro_total_s,
      always_pro_total_cost=always_pro_total_cost,
      alt_pricing_key=(
          "flash_lite_3_1_preview" if "3.1" in flash_lite_model else None
      ),
  )
  flash_summary = summarize_router_strategy(
      "flash_json_router",
      flash_records,
      dataset_by_id=dataset_by_id,
      pricing_key="flash",
      completion_stats=completion_stats,
      always_pro_total_s=always_pro_total_s,
      always_pro_total_cost=always_pro_total_cost,
  )

  d_p50 = float(dec_full_summary["latency"]["p50_ms"])
  t1_p50 = float(dec_tier1_summary["latency"]["p50_ms"])
  fl_p50 = float(flash_lite_summary["latency"]["p50_ms"])
  f_p50 = float(flash_summary["latency"]["p50_ms"])

  head_to_head_deltas = {
      "flash_lite_vs_decisions_full_p50_ratio": round(
          fl_p50 / max(1.0, d_p50), 2
      ),
      "flash_lite_vs_pure_tier1_p50_ratio": round(
          fl_p50 / max(1.0, t1_p50), 2
      ),
      "flash_nothink_vs_pure_tier1_p50_ratio": round(
          f_p50 / max(1.0, t1_p50), 2
      ),
      "decisions_full_tier_acc_advantage_vs_flash_lite_pp": round(
          float(dec_full_summary["accuracy"]["exact_tier_accuracy_pct"])
          - float(flash_lite_summary["accuracy"]["exact_tier_accuracy_pct"]),
          1,
      ),
      "decisions_full_pro_miss_reduction_vs_flash_lite_pp": round(
          float(flash_lite_summary["accuracy"]["pro_miss_rate_pct"])
          - float(dec_full_summary["accuracy"]["pro_miss_rate_pct"]),
          1,
      ),
      "decisions_brier_improvement_vs_flash_lite": round(
          float(flash_lite_summary["calibration"]["brier_score"])
          - float(dec_full_summary["calibration"]["brier_score"]),
          4,
      ),
      "decisions_net_e2e_seconds_saved_vs_flash_lite_per_100": round(
          float(flash_lite_summary["projected_100_e2e"]["total_e2e_s"])
          - float(dec_full_summary["projected_100_e2e"]["total_e2e_s"]),
          2,
      ),
  }

  results: dict[str, Any] = {
      "metadata": {
          "study": "router_head_to_head_100",
          "benchmark_version": "1.1.0",
          "mode": "mock" if mock else "live",
          "timestamp_utc": datetime.now(timezone.utc).strftime(
              "%Y-%m-%dT%H:%M:%SZ"
          ),
          "dataset_file": dataset_path.name,
          "dataset_size": len(dataset),
          "concurrency": concurrency,
          "auth_mode": auth_mode,
          "decisions_model": "gemini-decision-1",
          "decisions_source": decisions_source,
          "flash_lite_router_model": flash_lite_model,
          "flash_router_model": flash_model,
          "always_pro_100_baseline_s": always_pro_total_s,
          "always_pro_100_baseline_cost_usd": always_pro_total_cost,
          "json_router_generation_config": {
              "responseMimeType": "application/json",
              "thinkingConfig": {"thinkingBudget": 0},
              "maxOutputTokens": 96,
              "temperature": 0.0,
          },
          "pricing_per_1m_usd": PRICING_PER_1M,
          "retried_http_calls": 0 if ctx is None else ctx.retried_calls,
          "suite_wall_time_s": elapsed_s,
      },
      "head_to_head_deltas": head_to_head_deltas,
      "strategies": {
          "decisions_router_full": dec_full_summary,
          "decisions_tier1_only": dec_tier1_summary,
          "flash_lite_json_router": flash_lite_summary,
          "flash_json_router": flash_summary,
      },
      "reference_default_thinking_flash_router": baseline_thinking_ref,
      "records": {
          "flash_lite_json_router": flash_lite_records,
          "flash_json_router": flash_records,
      },
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(results, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  return results


def print_routing_stats_table(
    results: Mapping[str, Any], output_path: pathlib.Path
) -> None:
  """Prints the rich `ROUTED WITH INTENT` summary box and comparison table."""
  meta = results["metadata"]
  strat = results["strategies"]
  full = strat["decisions_router_full"]
  fl = strat["flash_lite_json_router"]
  fj = strat["flash_json_router"]
  t1 = strat["decisions_tier1_only"]

  dist = full["routed_tier_distribution"]
  bb = full["backend_breakdown"]
  e2e = full["projected_100_e2e"]
  fl_e2e = fl["projected_100_e2e"]

  border = "=" * 96
  rule = "-" * 96
  print(f"\n{border}", flush=True)
  print(
      f"  ✦ ROUTED WITH INTENT | BENCHMARK ROUTING STATS "
      f"(mode={meta['mode']}, N={meta['dataset_size']})",
      flush=True,
  )
  print(border, flush=True)
  print(
      f"  Tier Distribution (Decisions):  "
      f"flash_lite={dist['flash_lite']:2d} ({dist['flash_lite']}%) | "
      f"flash={dist['flash']:2d} ({dist['flash']}%) | "
      f"pro={dist['pro']:2d} ({dist['pro']}%)",
      flush=True,
  )
  print(
      f"  Backend Breakdown (Decisions):  "
      f"fast_rules={bb['fast_rules']:2d} | "
      f"gemini_api={bb['gemini_api']:2d} | "
      f"thread_lock={bb['thread_lock']:2d}",
      flush=True,
  )
  print(rule, flush=True)
  print(
      "  Category Accuracy (Decisions Tier 0+1 vs flash-lite JSON Router):",
      flush=True,
  )
  for cat in CATEGORIES_ORDER:
    if cat not in full["per_category"]:
      continue
    dc = full["per_category"][cat]
    fc = fl["per_category"][cat]
    print(
        f"    {cat:<25s} N={dc['count']:2d} | "
        f"Decisions={dc['tier_accuracy_pct']:5.1f}% (ProMiss {dc['pro_miss_count']}/{dc['expected_pro_count']}) | "
        f"flash-lite={fc['tier_accuracy_pct']:5.1f}% (ProMiss {fc['pro_miss_count']}/{fc['expected_pro_count']})",
        flush=True,
    )
  print(rule, flush=True)
  print(
      "  Head-to-Head Router Comparison (N=100):",
      flush=True,
  )
  for key, label in (
      ("decisions_router_full", "Decisions (Tier 0+1)"),
      ("decisions_tier1_only", "Pure gemini-decision-1"),
      ("flash_lite_json_router", f"flash-lite JSON ({meta['flash_lite_router_model']})"),
      ("flash_json_router", f"flash JSON ({meta['flash_router_model']})"),
  ):
    s = strat[key]
    lat = s["latency"]
    acc = s["accuracy"]
    cal = s["calibration"]
    tok = s["tokens"]
    pe = s["projected_100_e2e"]
    print(
        f"    {label[:34]:<34s} | p50={lat['p50_ms']:6.1f}ms "
        f"p90={lat['p90_ms']:6.1f}ms p99={lat['p99_ms']:6.1f}ms | "
        f"Acc={acc['exact_tier_accuracy_pct']:5.1f}% "
        f"ProMiss={acc['pro_miss_rate_pct']:4.1f}% | "
        f"OutTok={tok['mean_output_tokens']:4.1f} "
        f"AUROC={cal['confidence_auroc']:.2f} | "
        f"E2E={pe['total_e2e_s']:5.1f}s (-{pe['time_saved_vs_always_pro_pct']:4.1f}%)",
        flush=True,
    )
  print(rule, flush=True)
  print(
      f"  100-Prompt E2E Savings vs Always-Pro ({meta['always_pro_100_baseline_s']:.1f}s / "
      f"${meta['always_pro_100_baseline_cost_usd']:.3f}):\n"
      f"    Decisions Router:   {e2e['total_e2e_s']:6.2f}s (-{e2e['time_saved_vs_always_pro_pct']}% time) | "
      f"${e2e['total_cost_usd']:.4f} (-{e2e['cost_saved_vs_always_pro_pct']}% cost)\n"
      f"    flash-lite Router:  {fl_e2e['total_e2e_s']:6.2f}s (-{fl_e2e['time_saved_vs_always_pro_pct']}% time) | "
      f"${fl_e2e['total_cost_usd']:.4f} (-{fl_e2e['cost_saved_vs_always_pro_pct']}% cost)",
      flush=True,
  )
  print(border, flush=True)
  print(
      f"[head-to-head:{meta['mode']}] N={meta['dataset_size']} "
      f"decisions_p50={full['latency']['p50_ms']}ms "
      f"pure_t1_p50={t1['latency']['p50_ms']}ms "
      f"flash_lite_p50={fl['latency']['p50_ms']}ms "
      f"flash_p50={fj['latency']['p50_ms']}ms -> {output_path}",
      flush=True,
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Run 100-prompt head-to-head benchmark comparing flash-lite and flash"
          " GenerateContent JSON routers against gemini-decision-1."
      )
  )
  parser.add_argument(
      "--dataset",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "prompts_100.json",
      help="Path to 100-prompt dataset JSON.",
  )
  parser.add_argument(
      "--baseline-results",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_100.json",
      help="Path to baseline results_100.json with live gemini-decision-1 runs.",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_flash_lite_router_100.json",
      help="Path to write head-to-head results JSON.",
  )
  parser.add_argument(
      "--mock",
      action="store_true",
      help="Run deterministic offline simulation (<0.5s, zero network calls).",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=1,
      help="Worker concurrency (default: 1 for uncontended latency percentiles).",
  )
  parser.add_argument(
      "--rerun-decisions",
      action="store_true",
      help="Re-run all 200 live gemini-decision-1 calls instead of reusing results_100.json.",
  )
  parser.add_argument(
      "--recompute-only",
      action="store_true",
      help="Recompute summary metrics and streaming table from existing live records.",
  )
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress per-prompt streaming lines and print summary stats table only.",
  )
  args = parser.parse_args(argv)
  if args.concurrency < 1 or args.concurrency > 32:
    parser.error("--concurrency must be between 1 and 32")

  results = run_head_to_head(
      dataset_path=args.dataset,
      baseline_results_path=args.baseline_results,
      output_path=args.output,
      mock=bool(args.mock),
      concurrency=int(args.concurrency),
      rerun_decisions=bool(args.rerun_decisions),
      reuse_existing_output=bool(args.recompute_only),
      quiet=bool(args.quiet),
  )
  print_routing_stats_table(results, args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
