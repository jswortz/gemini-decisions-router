#!/usr/bin/env python3
"""Live and offline benchmark runner for Gemini Decisions Router.

Evaluates 100 stratified developer and agent prompts across:
  1. `decisions_router_full` (Tier 0 fast rules + Tier 1 `gemini-decision-1`)
  2. `decisions_tier1_only` (100% pure `gemini-decision-1` single-pass prefill)
  3. `lexical_only` (offline keyword + thread-floor baseline)
  4. `llm_json_router` (`gemini-2.5-flash` GenerateContent JSON router on N=15)
  5. End-to-end completion latency and API token cost across `flash_lite`,
     `flash`, and `pro` (`Always-Pro` vs `GenerateContent-Routed` vs
     `Gemini Decisions Router`).

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
import urllib.parse
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

CATEGORIES_ORDER = (
    "quick_lookup",
    "coding_edit",
    "writing_docs",
    "data_analysis",
    "deep_debug_architecture",
    "agentic_lifecycle",
)

# Standard public Gemini API per-1M token pricing (USD).
PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gemini_decision_1": {"input": 0.10, "output": 0.00},
    "llm_json_router": {"input": 0.30, "output": 2.50},
    "flash_lite": {"input": 0.10, "output": 0.40},
    "flash": {"input": 0.30, "output": 2.50},
    "pro": {"input": 1.25, "output": 10.00},
}

LLM_JSON_ROUTER_MODEL = "gemini-2.5-flash"
LLM_JSON_ROUTER_SYSTEM = (
    "You are an engineering prompt router. Classify the user request into a"
    " model tier and task_class, and return ONLY compact JSON matching"
    ' {"tier":"flash_lite|flash|pro","task_class":"quick_lookup|coding_edit|'
    'writing_docs|data_analysis|deep_debug_architecture"}. Routing rules: use'
    " flash_lite for simple factual/CLI lookups, syntax checks, or short"
    " confirmations; use flash for standard single-service code edits, tests,"
    " documentation, or analytical SQL queries; use pro for multi-region or"
    " distributed architecture, subtle concurrency/memory/security debugging,"
    " or complex causal/statistical root-cause investigations."
)

TIER_CANDIDATE_MODELS: dict[str, tuple[str, ...]] = {
    "flash_lite": ("gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"),
    "flash": ("gemini-2.5-flash",),
    "pro": ("gemini-2.5-pro",),
}


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


def token_cost_usd(pricing_key: str, input_tokens: float, output_tokens: float) -> float:
  rates = PRICING_PER_1M[pricing_key]
  return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0


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


def call_generate_content_live(
    ctx: LiveAuthContext,
    model: str,
    prompt: str,
    *,
    system_instruction: str = "",
    max_output_tokens: int = 512,
    response_mime_type: str = "",
    temperature: float | None = None,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
  """Executes a timed `generateContent` call and extracts latency + tokens."""
  payload: dict[str, Any] = {
      "contents": [{"role": "user", "parts": [{"text": prompt}]}],
      "generationConfig": {"maxOutputTokens": max_output_tokens},
  }
  if system_instruction:
    payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
  if response_mime_type:
    payload["generationConfig"]["responseMimeType"] = response_mime_type
  if temperature is not None:
    payload["generationConfig"]["temperature"] = temperature

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
        raw = resp.read(4 * 1024 * 1024)
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
      raise RuntimeError(f"generateContent {model} failed HTTP {e.code}: {msg}") from None
    except (TimeoutError, urllib.error.URLError) as e:
      if attempt < 2:
        ctx.retried_calls += 1
        time.sleep(0.8 * (attempt + 1))
        last_err = e
        continue
      raise RuntimeError(f"generateContent {model} network error: {type(e).__name__}") from None
  else:
    raise RuntimeError(f"generateContent {model} exhausted retries: {last_err}")

  usage = data.get("usageMetadata") or {}
  prompt_tok = int(usage.get("promptTokenCount") or 0)
  cand_tok = int(usage.get("candidatesTokenCount") or 0)
  thought_tok = int(usage.get("thoughtsTokenCount") or 0)
  total_tok = int(usage.get("totalTokenCount") or (prompt_tok + cand_tok + thought_tok))
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
      if isinstance(part, Mapping) and not part.get("thought") and isinstance(part.get("text"), str):
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


def probe_tier_models(ctx: LiveAuthContext) -> dict[str, str]:
  """Probes candidate models per tier once and warms TLS connections."""
  # Warm up Decisions API connection once before timed benchmark runs.
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

  selected: dict[str, str] = {}
  for tier, candidates in TIER_CANDIDATE_MODELS.items():
    chosen = ""
    last_exc: Exception | None = None
    for candidate in candidates:
      try:
        call_generate_content_live(
            ctx,
            candidate,
            "Reply with OK.",
            max_output_tokens=32,
            timeout_s=20.0,
        )
        chosen = candidate
        break
      except Exception as exc:  # pylint: disable=broad-except
        last_exc = exc
        continue
    if not chosen:
      raise RuntimeError(
          f"No reachable GenerateContent model for tier {tier}: {last_exc}"
      )
    selected[tier] = chosen
  return selected


# ---- Deterministic Offline Mock Backend (`--mock` < 0.5s) -------------------


def _det_hash_float(seed: str, salt: str = "") -> float:
  digest = hashlib.sha256(f"{seed}|{salt}".encode("utf-8")).digest()
  return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _mock_decision_for_item(item: Mapping[str, Any]) -> Callable[..., dict[str, Any]]:
  """Builds a deterministic offline `decision_fn` for `--mock` testing."""
  prompt = str(item["prompt"])
  exp_tier = str(item["expected_tier"])
  exp_tc = str(item["expected_task_class"])
  u_jitter = _det_hash_float(prompt, "lat")
  u_acc = _det_hash_float(prompt, "acc")
  sim_ms = int(round(165.0 + 75.0 * u_jitter))
  est_in_tokens = max(48, len(prompt) // 4 + 95)

  def _fn(call_args: Mapping[str, Any], _settings: Any) -> dict[str, Any]:
    del call_args
    tc_keys = list(FIVE_WAY_TASK_CLASSES)
    base_tc = exp_tc if exp_tc in tc_keys else (
        "quick_lookup" if exp_tier == "flash_lite" else (
            "deep_debug_architecture" if exp_tier == "pro" else "coding_edit"
        )
    )
    # Introduce small deterministic disagreement (~6%) so calibration bins are realistic.
    if u_acc > 0.94 and exp_tier != "pro":
      chosen_tc = "deep_debug_architecture" if exp_tier == "flash" else "coding_edit"
      conf = 0.66
    else:
      chosen_tc = base_tc
      conf = round(0.76 + 0.21 * _det_hash_float(prompt, "conf"), 4)

    rem = round(1.0 - conf, 4)
    per_other = round(rem / (len(tc_keys) - 1), 4)
    tc_probs = {k: per_other for k in tc_keys}
    tc_probs[chosen_tc] = round(1.0 - per_other * (len(tc_keys) - 1), 4)

    if chosen_tc == "deep_debug_architecture" or exp_tier == "pro":
      p_pro = round(0.78 + 0.18 * u_jitter, 4)
      eff_probs = {"0": 0.01, "1": 0.04, "2": 0.25, "3": 0.70}
      eff_score = 2.64
    elif chosen_tc == "quick_lookup" and exp_tier == "flash_lite":
      p_pro = round(0.03 + 0.08 * u_jitter, 4)
      eff_probs = {"0": 0.82, "1": 0.15, "2": 0.02, "3": 0.01}
      eff_score = 0.22
    else:
      p_pro = round(0.14 + 0.22 * u_jitter, 4)
      eff_probs = {"0": 0.05, "1": 0.55, "2": 0.35, "3": 0.05}
      eff_score = 1.40

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

  return _fn


# ---- Router Strategy Evaluators ---------------------------------------------


def evaluate_decisions_strategy(
    item: Mapping[str, Any],
    *,
    enable_fast_rules: bool,
    mock: bool,
    ctx: LiveAuthContext | None,
) -> dict[str, Any]:
  """Runs `decisions_engine.route_prompt` on one dataset item."""
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

  if mock:
    mock_fn = _mock_decision_for_item(item)

    def _wrapped_mock(call_args: Mapping[str, Any], settings: Any) -> dict[str, Any]:
      dec = mock_fn(call_args, settings)
      captured["api_called"] = True
      captured["input_tokens"] = int(dec["usage"].get("input_tokens", 0))
      captured["api_latency_ms"] = float(dec["latency_ms"])
      return dec

    t0 = time.perf_counter()
    res = decisions_engine.route_prompt(
        prompt,
        user_turn_count=turn_count,
        thread_state=thread_state,
        enable_fast_rules=enable_fast_rules,
        ignore_cooldown=True,
        environ={},
        decision_fn=_wrapped_mock,
    )
    local_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    wall_ms = captured["api_latency_ms"] if captured["api_called"] else max(0.05, local_ms)
  else:
    assert ctx is not None

    def _wrapped_live(call_args: Mapping[str, Any], _settings: Any) -> dict[str, Any]:
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
    wall_ms = captured["api_latency_ms"] if captured["api_called"] else measured_ms

  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": item["expected_tier"],
      "expected_task_class": item["expected_task_class"],
      "recommended_tier": res["recommended_tier"],
      "task_class": res["task_class"],
      "thinking_level": res["thinking_level"],
      "confidence": float(res["confidence"]),
      "backend": res["backend"],
      "api_called": captured["api_called"],
      "fallback_used": captured["fallback_used"],
      "latency_ms": wall_ms,
      "input_tokens": captured["input_tokens"],
      "output_tokens": 0,
      "candidates_tokens": 0,
      "thoughts_tokens": 0,
      "thread_floor_applied": bool(res.get("thread_floor_applied")),
      "tier_correct": res["recommended_tier"] == item["expected_tier"],
      "task_class_correct": res["task_class"] == item["expected_task_class"],
      "probabilities": res.get("probabilities", {}),
  }


def evaluate_lexical_strategy(item: Mapping[str, Any]) -> dict[str, Any]:
  """Runs the offline lexical baseline + asymmetric thread floor."""
  prompt = str(item["prompt"])
  turn_count = int(item.get("user_turn_count") or 1)
  thread_state = item.get("thread_state")
  if not isinstance(thread_state, Mapping):
    thread_state = None

  t0 = time.perf_counter()
  base = decisions_engine._lexical_fallback_route(prompt, tier_models=decisions_engine.DEFAULT_TIER_MODELS)  # pylint: disable=protected-access
  res = decisions_engine._apply_thread_floor(  # pylint: disable=protected-access
      base,
      user_turn_count=turn_count,
      thread_state=thread_state,
      tier_models=decisions_engine.DEFAULT_TIER_MODELS,
  )
  wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": item["expected_tier"],
      "expected_task_class": item["expected_task_class"],
      "recommended_tier": res["recommended_tier"],
      "task_class": res["task_class"],
      "thinking_level": res["thinking_level"],
      "confidence": float(res["confidence"]),
      "backend": res["backend"],
      "api_called": False,
      "fallback_used": False,
      "latency_ms": wall_ms,
      "input_tokens": 0,
      "output_tokens": 0,
      "candidates_tokens": 0,
      "thoughts_tokens": 0,
      "tier_correct": res["recommended_tier"] == item["expected_tier"],
      "task_class_correct": res["task_class"] == item["expected_task_class"],
  }


def _parse_llm_router_json(raw_text: str) -> tuple[str, str, bool]:
  cleaned = raw_text.strip()
  if cleaned.startswith("```"):
    cleaned = cleaned.strip("`")
    if cleaned.lower().startswith("json"):
      cleaned = cleaned[4:].strip()
  try:
    obj = json.loads(cleaned)
  except ValueError:
    return "flash", "coding_edit", True
  if not isinstance(obj, Mapping):
    return "flash", "coding_edit", True

  raw_tier = str(obj.get("tier") or "").strip().lower().replace("-", "_")
  tier_aliases = {
      "lite": "flash_lite",
      "gemini_flash_lite": "flash_lite",
      "gemini_flash": "flash",
      "gemini_pro": "pro",
  }
  raw_tier = tier_aliases.get(raw_tier, raw_tier)
  parse_err = raw_tier not in TIER_RANKS
  tier = raw_tier if raw_tier in TIER_RANKS else "flash"

  raw_tc = str(obj.get("task_class") or "").strip().lower()
  tc = raw_tc if raw_tc in FIVE_WAY_TASK_CLASSES else "coding_edit"
  return tier, tc, parse_err


def evaluate_llm_json_router_item(
    item: Mapping[str, Any],
    *,
    mock: bool,
    ctx: LiveAuthContext | None,
) -> dict[str, Any]:
  """Evaluates Traditional `GenerateContent` JSON Router (`gemini-2.5-flash`)."""
  prompt = str(item["prompt"])
  if mock:
    u_lat = _det_hash_float(prompt, "llm_lat")
    u_acc = _det_hash_float(prompt, "llm_acc")
    sim_ms = round(760.0 + 420.0 * u_lat, 2)
    p_tok = max(115, len(prompt) // 4 + 115)
    c_tok = 20
    t_tok = int(round(65 + 75 * u_lat))
    exp_tier = str(item["expected_tier"])
    exp_tc = str(item["expected_task_class"])
    if u_acc > 0.86:
      pred_tier = "flash" if exp_tier == "flash_lite" else exp_tier
    else:
      pred_tier = exp_tier
    pred_tc = exp_tc if exp_tc in FIVE_WAY_TASK_CLASSES else "coding_edit"
    return {
        "id": item["id"],
        "category": item["category"],
        "expected_tier": exp_tier,
        "expected_task_class": exp_tc,
        "recommended_tier": pred_tier,
        "task_class": pred_tc,
        "confidence": 0.0,
        "backend": f"generateContent:{LLM_JSON_ROUTER_MODEL}",
        "api_called": True,
        "parse_error": False,
        "latency_ms": sim_ms,
        "input_tokens": p_tok,
        "candidates_tokens": c_tok,
        "thoughts_tokens": t_tok,
        "output_tokens": c_tok + t_tok,
        "tier_correct": pred_tier == exp_tier,
        "task_class_correct": pred_tc == exp_tc,
    }

  assert ctx is not None
  resp = call_generate_content_live(
      ctx,
      LLM_JSON_ROUTER_MODEL,
      f"User request:\n{prompt}",
      system_instruction=LLM_JSON_ROUTER_SYSTEM,
      max_output_tokens=1024,
      response_mime_type="application/json",
      temperature=0.0,
  )
  pred_tier, pred_tc, parse_err = _parse_llm_router_json(resp["text"])
  return {
      "id": item["id"],
      "category": item["category"],
      "expected_tier": item["expected_tier"],
      "expected_task_class": item["expected_task_class"],
      "recommended_tier": pred_tier,
      "task_class": pred_tc,
      "confidence": 0.0,
      "backend": f"generateContent:{LLM_JSON_ROUTER_MODEL}",
      "api_called": True,
      "parse_error": parse_err,
      "latency_ms": resp["latency_ms"],
      "input_tokens": resp["prompt_tokens"],
      "candidates_tokens": resp["candidates_tokens"],
      "thoughts_tokens": resp["thoughts_tokens"],
      "output_tokens": resp["output_tokens"],
      "tier_correct": pred_tier == item["expected_tier"],
      "task_class_correct": pred_tc == item["expected_task_class"],
  }


def evaluate_downstream_completion(
    prompt: str,
    tier: str,
    *,
    tier_models: Mapping[str, str],
    mock: bool,
    ctx: LiveAuthContext | None,
) -> dict[str, Any]:
  """Measures downstream generation latency and token usage on a target tier."""
  model = tier_models[tier]
  if mock:
    u = _det_hash_float(prompt, f"gen_{tier}")
    p_tok = max(18, len(prompt) // 4 + 12)
    if tier == "flash_lite":
      lat = round(580.0 + 260.0 * u, 2)
      c_tok = int(round(110 + 90 * u))
      t_tok = 0
    elif tier == "flash":
      lat = round(1750.0 + 750.0 * u, 2)
      c_tok = int(round(190 + 110 * u))
      t_tok = int(round(140 + 90 * u))
    else:
      lat = round(4350.0 + 1450.0 * u, 2)
      c_tok = int(round(150 + 90 * u))
      t_tok = int(round(270 + 80 * u))
    out_tok = min(512, c_tok + t_tok)
    return {
        "tier": tier,
        "model": model,
        "latency_ms": lat,
        "prompt_tokens": p_tok,
        "candidates_tokens": c_tok,
        "thoughts_tokens": t_tok,
        "output_tokens": out_tok,
        "cost_usd": round(token_cost_usd(tier, p_tok, out_tok), 8),
    }

  assert ctx is not None
  resp = call_generate_content_live(
      ctx,
      model,
      prompt,
      max_output_tokens=512,
      timeout_s=60.0,
  )
  cost = token_cost_usd(tier, resp["prompt_tokens"], resp["output_tokens"])
  return {
      "tier": tier,
      "model": model,
      "latency_ms": resp["latency_ms"],
      "prompt_tokens": resp["prompt_tokens"],
      "candidates_tokens": resp["candidates_tokens"],
      "thoughts_tokens": resp["thoughts_tokens"],
      "output_tokens": resp["output_tokens"],
      "finish_reason": resp["finish_reason"],
      "cost_usd": round(cost, 8),
  }


# ---- Metric Aggregation & Calibration ---------------------------------------


def compute_calibration(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
  """Computes 4-bin confidence calibration and Expected Calibration Error."""
  bins_spec = (
      ("0.00-0.70", 0.0, 0.70, False),
      ("0.70-0.80", 0.70, 0.80, False),
      ("0.80-0.90", 0.80, 0.90, False),
      ("0.90-1.00", 0.90, 1.00, True),
  )
  n_total = len(records)
  bin_rows: list[dict[str, Any]] = []
  ece = 0.0

  for label, lo, hi, inclusive_hi in bins_spec:
    bucket = [
        r
        for r in records
        if (
            (lo <= float(r["confidence"]) <= hi)
            if inclusive_hi
            else (lo <= float(r["confidence"]) < hi)
        )
    ]
    cnt = len(bucket)
    if cnt > 0:
      mean_conf = statistics.fmean(float(r["confidence"]) for r in bucket)
      acc = sum(1 for r in bucket if r["tier_correct"]) / cnt
      gap = abs(acc - mean_conf)
      ece += (cnt / max(1, n_total)) * gap
      bin_rows.append({
          "bin": label,
          "count": cnt,
          "mean_confidence": round(mean_conf, 4),
          "empirical_accuracy": round(acc, 4),
          "calibration_gap": round(gap, 4),
      })
    else:
      bin_rows.append({
          "bin": label,
          "count": 0,
          "mean_confidence": 0.0,
          "empirical_accuracy": 0.0,
          "calibration_gap": 0.0,
      })

  return {
      "ece": round(ece, 4),
      "bins": bin_rows,
  }


def summarize_strategy(
    name: str,
    records: list[dict[str, Any]],
    *,
    router_pricing_key: str | None,
) -> dict[str, Any]:
  """Computes latency, token overhead, accuracy, confusion matrix, and safety."""
  n = len(records)
  latencies = [float(r["latency_ms"]) for r in records]
  api_latencies = [float(r["latency_ms"]) for r in records if r.get("api_called")]
  tier0_count = sum(1 for r in records if not r.get("api_called"))

  exact_tier_hits = sum(1 for r in records if r["tier_correct"])
  non_lifecycle = [r for r in records if r["category"] != "agentic_lifecycle"]
  tc_5way_hits = sum(1 for r in non_lifecycle if r["task_class_correct"])
  tc_all_hits = sum(1 for r in records if r["task_class_correct"])

  expected_pro = [r for r in records if r["expected_tier"] == "pro"]
  pro_misses = [r for r in expected_pro if r["recommended_tier"] != "pro"]
  expected_non_pro = [r for r in records if r["expected_tier"] != "pro"]
  over_routed_to_pro = [r for r in expected_non_pro if r["recommended_tier"] == "pro"]

  under_routed_any = [
      r
      for r in records
      if TIER_RANKS[r["recommended_tier"]] < TIER_RANKS[r["expected_tier"]]
  ]
  over_routed_any = [
      r
      for r in records
      if TIER_RANKS[r["recommended_tier"]] > TIER_RANKS[r["expected_tier"]]
  ]

  confusion: dict[str, dict[str, int]] = {
      exp_t: {pred_t: 0 for pred_t in TIERS} for exp_t in TIERS
  }
  routed_tier_counts = {t: 0 for t in TIERS}
  for r in records:
    confusion[r["expected_tier"]][r["recommended_tier"]] += 1
    routed_tier_counts[r["recommended_tier"]] += 1

  per_category: dict[str, dict[str, Any]] = {}
  for cat in CATEGORIES_ORDER:
    cat_recs = [r for r in records if r["category"] == cat]
    if not cat_recs:
      continue
    cat_hits = sum(1 for r in cat_recs if r["tier_correct"])
    cat_tc_hits = sum(1 for r in cat_recs if r["task_class_correct"])
    cat_pro = [r for r in cat_recs if r["expected_tier"] == "pro"]
    cat_pro_miss = sum(1 for r in cat_pro if r["recommended_tier"] != "pro")
    per_category[cat] = {
        "count": len(cat_recs),
        "tier_accuracy_pct": round(100.0 * cat_hits / len(cat_recs), 1),
        "tier_correct": cat_hits,
        "task_class_accuracy_pct": round(100.0 * cat_tc_hits / len(cat_recs), 1),
        "expected_pro_count": len(cat_pro),
        "pro_miss_count": cat_pro_miss,
        "mean_latency_ms": round(
            statistics.fmean(float(r["latency_ms"]) for r in cat_recs), 2
        ),
    }

  in_tokens = [int(r.get("input_tokens", 0)) for r in records]
  out_tokens = [int(r.get("output_tokens", 0)) for r in records]
  cand_tokens = [int(r.get("candidates_tokens", 0)) for r in records]
  thought_tokens = [int(r.get("thoughts_tokens", 0)) for r in records]

  total_in = sum(in_tokens)
  total_out = sum(out_tokens)
  router_cost = (
      token_cost_usd(router_pricing_key, total_in, total_out)
      if router_pricing_key
      else 0.0
  )

  return {
      "strategy": name,
      "sample_size": n,
      "tier0_fast_rule_count": tier0_count,
      "tier1_api_call_count": n - tier0_count,
      "fallback_count": sum(1 for r in records if r.get("fallback_used")),
      "latency": summarize_latencies(latencies),
      "api_only_latency": summarize_latencies(api_latencies),
      "tokens": {
          "total_input_tokens": total_in,
          "total_output_tokens": total_out,
          "total_candidates_tokens": sum(cand_tokens),
          "total_thoughts_tokens": sum(thought_tokens),
          "mean_input_tokens": round(total_in / max(1, n), 1),
          "mean_output_tokens": round(total_out / max(1, n), 1),
          "mean_candidates_tokens": round(sum(cand_tokens) / max(1, n), 1),
          "mean_thoughts_tokens": round(sum(thought_tokens) / max(1, n), 1),
          "router_cost_usd": round(router_cost, 6),
      },
      "accuracy": {
          "exact_tier_accuracy_pct": round(100.0 * exact_tier_hits / max(1, n), 1),
          "exact_tier_correct": exact_tier_hits,
          "task_class_5way_accuracy_pct": round(
              100.0 * tc_5way_hits / max(1, len(non_lifecycle)), 1
          ),
          "task_class_5way_correct": tc_5way_hits,
          "task_class_5way_total": len(non_lifecycle),
          "task_class_all_accuracy_pct": round(100.0 * tc_all_hits / max(1, n), 1),
          "pro_miss_rate_pct": round(
              100.0 * len(pro_misses) / max(1, len(expected_pro)), 1
          ),
          "pro_miss_count": len(pro_misses),
          "expected_pro_count": len(expected_pro),
          "over_route_to_pro_rate_pct": round(
              100.0 * len(over_routed_to_pro) / max(1, len(expected_non_pro)), 1
          ),
          "over_route_to_pro_count": len(over_routed_to_pro),
          "expected_non_pro_count": len(expected_non_pro),
          "under_route_any_rate_pct": round(100.0 * len(under_routed_any) / max(1, n), 1),
          "over_route_any_rate_pct": round(100.0 * len(over_routed_any) / max(1, n), 1),
      },
      "routed_tier_distribution": routed_tier_counts,
      "confusion_matrix": confusion,
      "per_category": per_category,
      "calibration": compute_calibration(records),
  }


def _safe_print(text: str = "") -> None:
  try:
    print(text, flush=True)
  except BrokenPipeError:
    try:
      sys.stdout.close()
    except OSError:
      pass
    raise SystemExit(0) from None


def _classify_backend_bucket(
    rec: Mapping[str, Any], item: Mapping[str, Any]
) -> str:
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
  b_str = str(rec.get("backend") or "gemini_api")
  if b_str in ("mock_decisions_api", "gemini_api", "gemini_adc"):
    return "gemini_api"
  return b_str.removeprefix("generateContent:")


def _format_live_route_line(
    idx: int,
    total: int,
    rec: Mapping[str, Any],
    item: Mapping[str, Any],
    tier_models: Mapping[str, str],
) -> str:
  status = "PASS" if rec.get("tier_correct") else "MISS"
  cat = str(rec.get("category") or item.get("category") or "")[:18]
  tier = str(rec.get("recommended_tier") or "flash")
  model_lbl = tier_models.get(tier, tier).removesuffix("-preview")[:21]
  conf = float(rec.get("confidence") or 0.0)
  lat_ms = float(rec.get("latency_ms") or 0.0)
  b_bucket = _classify_backend_bucket(rec, item)[:16]
  raw_p = " ".join(str(item.get("prompt") or "").split())
  snippet = raw_p if len(raw_p) <= 36 else raw_p[:33] + "..."
  return (
      f"[{idx:03d}/{total:03d}] {status:4s}  {cat:<18s} -> "
      f"{tier:<10s} ({model_lbl:<21s}) | conf={conf:0.2f} | "
      f"{lat_ms:7.1f} ms | {b_bucket:<16s} | \"{snippet}\""
  )


def _run_items(
    items: list[dict[str, Any]],
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    concurrency: int,
    *,
    stream_title: str = "",
    tier_models: Mapping[str, str] | None = None,
    quiet: bool = True,
) -> list[dict[str, Any]]:
  total = len(items)
  if stream_title and not quiet:
    _safe_print(f"\n--- Streaming Live Routes: {stream_title} (N={total}) ---")
  print_lock = threading.Lock()
  counter = {"done": 0}
  t_models = tier_models or {}

  def _wrapped(item: dict[str, Any]) -> dict[str, Any]:
    rec = fn(item)
    if stream_title and not quiet and isinstance(rec, Mapping):
      with print_lock:
        counter["done"] += 1
        _safe_print(
            _format_live_route_line(
                counter["done"], total, rec, item, t_models
            )
        )
    return rec

  if concurrency <= 1:
    return [_wrapped(item) for item in items]
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    return list(pool.map(_wrapped, items))


def compute_e2e_and_projections(
    *,
    dataset: list[dict[str, Any]],
    sample_items: list[dict[str, Any]],
    full_by_id: dict[str, dict[str, Any]],
    tier1_by_id: dict[str, dict[str, Any]],
    llm_by_id: dict[str, dict[str, Any]],
    completions_by_id: dict[str, dict[str, dict[str, Any]]],
    full_summary: dict[str, Any],
    llm_summary: dict[str, Any],
) -> dict[str, Any]:
  """Computes empirical 15-prompt E2E totals and projected 100-prompt savings."""
  per_sample_rows: list[dict[str, Any]] = []

  always_pro_gen_ms = 0.0
  always_pro_cost = 0.0

  llm_router_ms = 0.0
  llm_gen_ms = 0.0
  llm_router_cost = 0.0
  llm_gen_cost = 0.0

  dec_router_ms = 0.0
  dec_gen_ms = 0.0
  dec_router_cost = 0.0
  dec_gen_cost = 0.0

  # Collect empirical completion latency and cost per tier across measured runs.
  tier_completion_samples: dict[str, list[dict[str, Any]]] = {
      t: [] for t in TIERS
  }
  pro_by_expected_stratum: dict[str, list[dict[str, Any]]] = {
      t: [] for t in TIERS
  }

  for item in sample_items:
    pid = str(item["id"])
    exp_tier = str(item["expected_tier"])
    dec_rec = full_by_id[pid]
    llm_rec = llm_by_id[pid]
    comp_map = completions_by_id[pid]

    for t_key, comp in comp_map.items():
      tier_completion_samples[t_key].append(comp)
    pro_comp = comp_map["pro"]
    pro_by_expected_stratum[exp_tier].append(pro_comp)

    dec_comp = comp_map[dec_rec["recommended_tier"]]
    llm_comp = comp_map[llm_rec["recommended_tier"]]

    d_r_cost = token_cost_usd("gemini_decision_1", dec_rec["input_tokens"], 0)
    l_r_cost = token_cost_usd(
        "llm_json_router", llm_rec["input_tokens"], llm_rec["output_tokens"]
    )

    always_pro_gen_ms += float(pro_comp["latency_ms"])
    always_pro_cost += float(pro_comp["cost_usd"])

    llm_router_ms += float(llm_rec["latency_ms"])
    llm_gen_ms += float(llm_comp["latency_ms"])
    llm_router_cost += l_r_cost
    llm_gen_cost += float(llm_comp["cost_usd"])

    dec_router_ms += float(dec_rec["latency_ms"])
    dec_gen_ms += float(dec_comp["latency_ms"])
    dec_router_cost += d_r_cost
    dec_gen_cost += float(dec_comp["cost_usd"])

    per_sample_rows.append({
        "id": pid,
        "category": item["category"],
        "expected_tier": exp_tier,
        "decisions_tier": dec_rec["recommended_tier"],
        "decisions_router_ms": dec_rec["latency_ms"],
        "decisions_generation_ms": dec_comp["latency_ms"],
        "decisions_e2e_ms": round(
            float(dec_rec["latency_ms"]) + float(dec_comp["latency_ms"]), 2
        ),
        "tier1_only_tier": tier1_by_id[pid]["recommended_tier"],
        "tier1_only_ms": tier1_by_id[pid]["latency_ms"],
        "llm_json_tier": llm_rec["recommended_tier"],
        "llm_json_router_ms": llm_rec["latency_ms"],
        "llm_json_output_tokens": llm_rec["output_tokens"],
        "llm_json_generation_ms": llm_comp["latency_ms"],
        "llm_json_e2e_ms": round(
            float(llm_rec["latency_ms"]) + float(llm_comp["latency_ms"]), 2
        ),
        "always_pro_e2e_ms": pro_comp["latency_ms"],
        "always_pro_cost_usd": pro_comp["cost_usd"],
        "decisions_e2e_cost_usd": round(d_r_cost + float(dec_comp["cost_usd"]), 8),
        "llm_json_e2e_cost_usd": round(l_r_cost + float(llm_comp["cost_usd"]), 8),
    })

  always_pro_15_s = always_pro_gen_ms / 1000.0
  llm_15_s = (llm_router_ms + llm_gen_ms) / 1000.0
  dec_15_s = (dec_router_ms + dec_gen_ms) / 1000.0

  llm_15_cost = llm_router_cost + llm_gen_cost
  dec_15_cost = dec_router_cost + dec_gen_cost

  # Empirical per-tier completion statistics across sample prompts.
  tier_completion_stats: dict[str, dict[str, float | int]] = {}
  for t in TIERS:
    samples = tier_completion_samples[t]
    if samples:
      tier_completion_stats[t] = {
          "count": len(samples),
          "mean_latency_ms": round(
              statistics.fmean(float(s["latency_ms"]) for s in samples), 2
          ),
          "p50_latency_ms": _percentile([float(s["latency_ms"]) for s in samples], 50),
          "mean_prompt_tokens": round(
              statistics.fmean(float(s["prompt_tokens"]) for s in samples), 1
          ),
          "mean_output_tokens": round(
              statistics.fmean(float(s["output_tokens"]) for s in samples), 1
          ),
          "mean_cost_usd": round(
              statistics.fmean(float(s["cost_usd"]) for s in samples), 8
          ),
      }
    else:
      # Fallback if no prompt ever selected this tier during the sample run.
      fallback_ms = 650.0 if t == "flash_lite" else (2000.0 if t == "flash" else 4800.0)
      tier_completion_stats[t] = {
          "count": 0,
          "mean_latency_ms": fallback_ms,
          "p50_latency_ms": fallback_ms,
          "mean_prompt_tokens": 60.0,
          "mean_output_tokens": 250.0,
          "mean_cost_usd": round(token_cost_usd(t, 60.0, 250.0), 8),
      }

  # Ground-truth tier counts across all 100 prompts.
  gt_counts = {t: sum(1 for x in dataset if x["expected_tier"] == t) for t in TIERS}
  n_dataset = len(dataset)

  # 1. Projected 100-prompt Always-Pro (stratified by ground-truth complexity).
  proj_pro_gen_ms = 0.0
  proj_pro_cost = 0.0
  overall_pro_mean_ms = float(tier_completion_stats["pro"]["mean_latency_ms"])
  overall_pro_mean_cost = float(tier_completion_stats["pro"]["mean_cost_usd"])
  for t in TIERS:
    stratum = pro_by_expected_stratum[t]
    s_ms = (
        statistics.fmean(float(s["latency_ms"]) for s in stratum)
        if stratum
        else overall_pro_mean_ms
    )
    s_cost = (
        statistics.fmean(float(s["cost_usd"]) for s in stratum)
        if stratum
        else overall_pro_mean_cost
    )
    proj_pro_gen_ms += gt_counts[t] * s_ms
    proj_pro_cost += gt_counts[t] * s_cost

  # 2. Projected 100-prompt Gemini Decisions Router (exact 100-prompt router overhead
  # + exact 100-prompt routed tier distribution * empirical tier completion means).
  dec_dist = full_summary["routed_tier_distribution"]
  proj_dec_router_s = float(full_summary["latency"]["total_ms"]) / 1000.0
  proj_dec_router_cost = float(full_summary["tokens"]["router_cost_usd"])
  proj_dec_gen_s = (
      sum(
          dec_dist[t] * float(tier_completion_stats[t]["mean_latency_ms"])
          for t in TIERS
      )
      / 1000.0
  )
  proj_dec_gen_cost = sum(
      dec_dist[t] * float(tier_completion_stats[t]["mean_cost_usd"])
      for t in TIERS
  )
  proj_dec_total_s = proj_dec_router_s + proj_dec_gen_s
  proj_dec_total_cost = proj_dec_router_cost + proj_dec_gen_cost

  # 3. Projected 100-prompt GenerateContent JSON Router (stratified transition matrix
  # from N=15 applied to ground-truth 100-prompt counts + 100x measured mean overhead).
  llm_trans: dict[str, dict[str, int]] = {
      exp_t: {pred_t: 0 for pred_t in TIERS} for exp_t in TIERS
  }
  for item in sample_items:
    exp_t = str(item["expected_tier"])
    pred_t = llm_by_id[str(item["id"])]["recommended_tier"]
    llm_trans[exp_t][pred_t] += 1

  llm_proj_dist = {t: 0.0 for t in TIERS}
  for exp_t in TIERS:
    row_tot = sum(llm_trans[exp_t].values())
    if row_tot > 0:
      for pred_t in TIERS:
        llm_proj_dist[pred_t] += gt_counts[exp_t] * (llm_trans[exp_t][pred_t] / row_tot)
    else:
      llm_proj_dist[exp_t] += gt_counts[exp_t]

  proj_llm_router_s = (n_dataset * float(llm_summary["latency"]["mean_ms"])) / 1000.0
  proj_llm_router_cost = n_dataset * (
      float(llm_summary["tokens"]["router_cost_usd"])
      / max(1, int(llm_summary["sample_size"]))
  )
  proj_llm_gen_s = (
      sum(
          llm_proj_dist[t] * float(tier_completion_stats[t]["mean_latency_ms"])
          for t in TIERS
      )
      / 1000.0
  )
  proj_llm_gen_cost = sum(
      llm_proj_dist[t] * float(tier_completion_stats[t]["mean_cost_usd"])
      for t in TIERS
  )
  proj_llm_total_s = proj_llm_router_s + proj_llm_gen_s
  proj_llm_total_cost = proj_llm_router_cost + proj_llm_gen_cost

  proj_pro_total_s = proj_pro_gen_ms / 1000.0

  def _pct_saved(base_val: float, new_val: float) -> float:
    if base_val <= 0:
      return 0.0
    return round(100.0 * (base_val - new_val) / base_val, 1)

  # Also compute N=15 apples-to-apples accuracy for Decisions Router on the exact same 15 prompts.
  dec_15_recs = [full_by_id[str(it["id"])] for it in sample_items]
  tier1_15_recs = [tier1_by_id[str(it["id"])] for it in sample_items]
  dec_15_acc = round(
      100.0 * sum(1 for r in dec_15_recs if r["tier_correct"]) / max(1, len(dec_15_recs)),
      1,
  )
  tier1_15_acc = round(
      100.0 * sum(1 for r in tier1_15_recs if r["tier_correct"]) / max(1, len(tier1_15_recs)),
      1,
  )

  return {
      "sample_size": len(sample_items),
      "sample_15_apples_to_apples": {
          "decisions_router_full_tier_accuracy_pct": dec_15_acc,
          "decisions_tier1_only_tier_accuracy_pct": tier1_15_acc,
          "llm_json_router_tier_accuracy_pct": llm_summary["accuracy"]["exact_tier_accuracy_pct"],
          "decisions_router_p50_ms": _percentile(
              [float(r["latency_ms"]) for r in dec_15_recs], 50
          ),
          "tier1_only_p50_ms": _percentile(
              [float(r["latency_ms"]) for r in tier1_15_recs], 50
          ),
          "llm_json_router_p50_ms": llm_summary["latency"]["p50_ms"],
      },
      "tier_completion_stats": tier_completion_stats,
      "empirical_15_prompts": {
          "always_pro": {
              "router_overhead_s": 0.0,
              "generation_s": round(always_pro_15_s, 2),
              "total_e2e_s": round(always_pro_15_s, 2),
              "router_cost_usd": 0.0,
              "generation_cost_usd": round(always_pro_cost, 6),
              "total_cost_usd": round(always_pro_cost, 6),
              "time_saved_vs_always_pro_pct": 0.0,
              "cost_saved_vs_always_pro_pct": 0.0,
          },
          "llm_json_router_routed": {
              "router_overhead_s": round(llm_router_ms / 1000.0, 2),
              "generation_s": round(llm_gen_ms / 1000.0, 2),
              "total_e2e_s": round(llm_15_s, 2),
              "router_cost_usd": round(llm_router_cost, 6),
              "generation_cost_usd": round(llm_gen_cost, 6),
              "total_cost_usd": round(llm_15_cost, 6),
              "time_saved_vs_always_pro_pct": _pct_saved(always_pro_15_s, llm_15_s),
              "cost_saved_vs_always_pro_pct": _pct_saved(always_pro_cost, llm_15_cost),
          },
          "decisions_router_routed": {
              "router_overhead_s": round(dec_router_ms / 1000.0, 2),
              "generation_s": round(dec_gen_ms / 1000.0, 2),
              "total_e2e_s": round(dec_15_s, 2),
              "router_cost_usd": round(dec_router_cost, 6),
              "generation_cost_usd": round(dec_gen_cost, 6),
              "total_cost_usd": round(dec_15_cost, 6),
              "time_saved_vs_always_pro_pct": _pct_saved(always_pro_15_s, dec_15_s),
              "cost_saved_vs_always_pro_pct": _pct_saved(always_pro_cost, dec_15_cost),
          },
      },
      "projected_100_prompts": {
          "ground_truth_tier_counts": gt_counts,
          "always_pro": {
              "router_overhead_s": 0.0,
              "generation_s": round(proj_pro_total_s, 2),
              "total_e2e_s": round(proj_pro_total_s, 2),
              "router_cost_usd": 0.0,
              "generation_cost_usd": round(proj_pro_cost, 5),
              "total_cost_usd": round(proj_pro_cost, 5),
              "time_saved_vs_always_pro_pct": 0.0,
              "cost_saved_vs_always_pro_pct": 0.0,
          },
          "llm_json_router_routed": {
              "router_overhead_s": round(proj_llm_router_s, 2),
              "generation_s": round(proj_llm_gen_s, 2),
              "total_e2e_s": round(proj_llm_total_s, 2),
              "router_cost_usd": round(proj_llm_router_cost, 5),
              "generation_cost_usd": round(proj_llm_gen_cost, 5),
              "total_cost_usd": round(proj_llm_total_cost, 5),
              "time_saved_vs_always_pro_pct": _pct_saved(
                  proj_pro_total_s, proj_llm_total_s
              ),
              "cost_saved_vs_always_pro_pct": _pct_saved(
                  proj_pro_cost, proj_llm_total_cost
              ),
          },
          "decisions_router_routed": {
              "router_overhead_s": round(proj_dec_router_s, 2),
              "generation_s": round(proj_dec_gen_s, 2),
              "total_e2e_s": round(proj_dec_total_s, 2),
              "router_cost_usd": round(proj_dec_router_cost, 5),
              "generation_cost_usd": round(proj_dec_gen_cost, 5),
              "total_cost_usd": round(proj_dec_total_cost, 5),
              "time_saved_vs_always_pro_pct": _pct_saved(
                  proj_pro_total_s, proj_dec_total_s
              ),
              "cost_saved_vs_always_pro_pct": _pct_saved(
                  proj_pro_cost, proj_dec_total_cost
              ),
          },
      },
      "sample_details": per_sample_rows,
  }


def validate_dataset(dataset: Any) -> list[dict[str, Any]]:
  if not isinstance(dataset, list) or not dataset:
    raise ValueError("Benchmark dataset must be a non-empty JSON list.")
  for idx, item in enumerate(dataset):
    if not isinstance(item, dict):
      raise ValueError(f"Dataset item {idx} must be an object.")
    for key in ("id", "category", "prompt", "expected_tier", "expected_task_class"):
      if key not in item:
        raise ValueError(f"Dataset item {idx} missing required key {key!r}.")
    if item["expected_tier"] not in TIER_RANKS:
      raise ValueError(
          f"Dataset item {item['id']} has invalid expected_tier {item['expected_tier']!r}."
      )
  return dataset


def run_benchmark(
    *,
    dataset_path: pathlib.Path,
    output_path: pathlib.Path,
    mock: bool = False,
    concurrency: int = 1,
    quiet: bool = False,
) -> dict[str, Any]:
  """Runs the complete routing and end-to-end benchmark suite."""
  t_suite_start = time.perf_counter()
  dataset = validate_dataset(
      json.loads(dataset_path.read_text(encoding="utf-8"))
  )
  sample_items = [item for item in dataset if bool(item.get("sample_e2e"))]
  if not sample_items:
    sample_items = dataset[: min(15, len(dataset))]

  if mock:
    ctx: LiveAuthContext | None = None
    tier_models = {
        "flash_lite": "gemini-3.1-flash-lite-preview",
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }
    auth_mode_label = "mock_offline"
    decisions_model = "gemini-decision-1"
  else:
    ctx = build_live_auth_context()
    tier_models = probe_tier_models(ctx)
    auth_mode_label = ctx.settings.auth_mode
    decisions_model = ctx.settings.model

  # 1. Evaluate all 100 prompts on `decisions_router_full` (Tier 0 + Tier 1).
  full_records = _run_items(
      dataset,
      lambda it: evaluate_decisions_strategy(
          it, enable_fast_rules=True, mock=mock, ctx=ctx
      ),
      concurrency,
      stream_title="decisions_router_full (Tier 0 + gemini-decision-1)",
      tier_models=tier_models,
      quiet=quiet,
  )

  # 2. Evaluate all 100 prompts on `decisions_tier1_only` (100% pure `gemini-decision-1`).
  tier1_records = _run_items(
      dataset,
      lambda it: evaluate_decisions_strategy(
          it, enable_fast_rules=False, mock=mock, ctx=ctx
      ),
      concurrency,
  )

  # 3. Evaluate all 100 prompts on `lexical_only` (offline lexical + thread floor).
  lexical_records = [evaluate_lexical_strategy(it) for it in dataset]

  # 4. Always-Pro baseline across all 100 prompts.
  always_pro_records = [
      {
          "id": it["id"],
          "category": it["category"],
          "expected_tier": it["expected_tier"],
          "expected_task_class": it["expected_task_class"],
          "recommended_tier": "pro",
          "task_class": "deep_debug_architecture",
          "thinking_level": 3,
          "confidence": 1.0,
          "backend": "static_always_pro",
          "api_called": False,
          "fallback_used": False,
          "latency_ms": 0.0,
          "input_tokens": 0,
          "output_tokens": 0,
          "candidates_tokens": 0,
          "thoughts_tokens": 0,
          "tier_correct": it["expected_tier"] == "pro",
          "task_class_correct": it["expected_task_class"] == "deep_debug_architecture",
      }
      for it in dataset
  ]

  # 5. Evaluate the 15 `sample_e2e` prompts on Traditional `GenerateContent` JSON Router.
  llm_records = _run_items(
      sample_items,
      lambda it: evaluate_llm_json_router_item(it, mock=mock, ctx=ctx),
      concurrency,
      stream_title=f"llm_json_router ({LLM_JSON_ROUTER_MODEL} sample)",
      tier_models=tier_models,
      quiet=quiet,
  )

  full_by_id = {str(r["id"]): r for r in full_records}
  tier1_by_id = {str(r["id"]): r for r in tier1_records}
  llm_by_id = {str(r["id"]): r for r in llm_records}

  # 6. Measure downstream completions on BOTH Routed Tier(s) AND Always-Pro for all 15 sample prompts.
  completions_by_id: dict[str, dict[str, dict[str, Any]]] = {}

  def _eval_sample_completions(item: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    pid = str(item["id"])
    prompt = str(item["prompt"])
    needed_tiers = {
        "pro",
        full_by_id[pid]["recommended_tier"],
        llm_by_id[pid]["recommended_tier"],
    }
    comp_for_prompt: dict[str, dict[str, Any]] = {}
    for tier in sorted(needed_tiers, key=lambda k: TIER_RANKS[k]):
      comp_for_prompt[tier] = evaluate_downstream_completion(
          prompt,
          tier,
          tier_models=tier_models,
          mock=mock,
          ctx=ctx,
      )
    return pid, comp_for_prompt

  for pid, comp_map in _run_items(sample_items, _eval_sample_completions, concurrency):  # type: ignore[arg-type]
    completions_by_id[pid] = comp_map

  full_summary = summarize_strategy(
      "decisions_router_full",
      full_records,
      router_pricing_key="gemini_decision_1",
  )
  tier1_summary = summarize_strategy(
      "decisions_tier1_only",
      tier1_records,
      router_pricing_key="gemini_decision_1",
  )
  lexical_summary = summarize_strategy(
      "lexical_only",
      lexical_records,
      router_pricing_key=None,
  )
  always_pro_summary = summarize_strategy(
      "always_pro",
      always_pro_records,
      router_pricing_key=None,
  )
  llm_summary = summarize_strategy(
      "llm_json_router",
      llm_records,
      router_pricing_key="llm_json_router",
  )

  e2e_results = compute_e2e_and_projections(
      dataset=dataset,
      sample_items=sample_items,
      full_by_id=full_by_id,
      tier1_by_id=tier1_by_id,
      llm_by_id=llm_by_id,
      completions_by_id=completions_by_id,
      full_summary=full_summary,
      llm_summary=llm_summary,
  )

  suite_elapsed_s = round(time.perf_counter() - t_suite_start, 3)

  results: dict[str, Any] = {
      "metadata": {
          "benchmark_version": "1.0.0",
          "mode": "mock" if mock else "live",
          "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "dataset_file": dataset_path.name,
          "dataset_size": len(dataset),
          "sample_e2e_size": len(sample_items),
          "concurrency": concurrency,
          "auth_mode": auth_mode_label,
          "decisions_model": decisions_model,
          "llm_json_router_model": LLM_JSON_ROUTER_MODEL,
          "downstream_tier_models": tier_models,
          "downstream_max_output_tokens": 512,
          "pricing_per_1m_usd": PRICING_PER_1M,
          "retried_http_calls": 0 if ctx is None else ctx.retried_calls,
          "suite_wall_time_s": suite_elapsed_s,
      },
      "strategies": {
          "decisions_router_full": full_summary,
          "decisions_tier1_only": tier1_summary,
          "llm_json_router": llm_summary,
          "lexical_only": lexical_summary,
          "always_pro": always_pro_summary,
      },
      "e2e": e2e_results,
      "records": {
          "decisions_router_full": full_records,
          "decisions_tier1_only": tier1_records,
          "lexical_only": lexical_records,
          "llm_json_router": llm_records,
      },
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(results, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  return results


def _print_summary(results: Mapping[str, Any], output_path: pathlib.Path) -> None:
  meta = results["metadata"]
  strat = results["strategies"]
  full = strat["decisions_router_full"]
  t1 = strat["decisions_tier1_only"]
  llm = strat["llm_json_router"]
  lex = strat["lexical_only"]
  proj = results["e2e"]["projected_100_prompts"]

  full_recs = results.get("records", {}).get("decisions_router_full", [])
  fast_rules_cnt = sum(
      1
      for r in full_recs
      if not r.get("api_called") and not r.get("thread_floor_applied")
  )
  thread_lock_cnt = sum(1 for r in full_recs if r.get("thread_floor_applied"))
  gemini_api_cnt = sum(1 for r in full_recs if r.get("api_called"))
  dist = full["routed_tier_distribution"]

  border = "=" * 96
  rule = "-" * 96
  _safe_print(f"\n{border}")
  _safe_print(
      f"  ✦ ROUTED WITH INTENT | BENCHMARK ROUTING STATS "
      f"(mode={meta['mode']}, N={meta['dataset_size']})"
  )
  _safe_print(border)
  _safe_print(
      f"  Tier Distribution:   "
      f"flash_lite={dist['flash_lite']:2d} ({dist['flash_lite']}%) | "
      f"flash={dist['flash']:2d} ({dist['flash']}%) | "
      f"pro={dist['pro']:2d} ({dist['pro']}%)"
  )
  _safe_print(
      f"  Backend Breakdown:   "
      f"fast_rules={fast_rules_cnt:2d} | "
      f"gemini_api={gemini_api_cnt:2d} | "
      f"thread_lock={thread_lock_cnt:2d}"
  )
  _safe_print(rule)
  _safe_print("  Per-Category Routing Accuracy (Decisions Tier 0+1):")
  for cat in CATEGORIES_ORDER:
    if cat not in full["per_category"]:
      continue
    c = full["per_category"][cat]
    _safe_print(
        f"    {cat:<25s} N={c['count']:2d} | "
        f"TierAcc={c['tier_accuracy_pct']:5.1f}% ({c['tier_correct']:2d}/{c['count']:2d}) | "
        f"TaskClassAcc={c['task_class_accuracy_pct']:5.1f}% | "
        f"ProMiss={c['pro_miss_count']}/{c['expected_pro_count']} | "
        f"mean={c['mean_latency_ms']:6.1f}ms"
    )
  _safe_print(rule)

  speedup_p50 = round(
      float(llm["latency"]["p50_ms"]) / max(1.0, float(full["latency"]["p50_ms"])),
      2,
  )
  _safe_print(
      f"[benchmark:{meta['mode']}] dataset={meta['dataset_size']} "
      f"sample_e2e={meta['sample_e2e_size']} concurrency={meta['concurrency']} "
      f"elapsed={meta['suite_wall_time_s']}s"
  )
  _safe_print(
      f"  Router Overhead p50/p90/p99 (ms): "
      f"Tier0+1={full['latency']['p50_ms']}/{full['latency']['p90_ms']}/{full['latency']['p99_ms']} | "
      f"PureTier1={t1['latency']['p50_ms']}/{t1['latency']['p90_ms']}/{t1['latency']['p99_ms']} (0 out tok) | "
      f"GenerateContent={llm['latency']['p50_ms']}/{llm['latency']['p90_ms']}/{llm['latency']['p99_ms']} "
      f"({llm['tokens']['mean_output_tokens']} out tok, {speedup_p50}x slower p50)"
  )
  _safe_print(
      f"  Exact Tier Accuracy (%):          "
      f"Tier0+1={full['accuracy']['exact_tier_accuracy_pct']}% | "
      f"PureTier1={t1['accuracy']['exact_tier_accuracy_pct']}% | "
      f"GenerateContent(N=15)={llm['accuracy']['exact_tier_accuracy_pct']}% | "
      f"Lexical={lex['accuracy']['exact_tier_accuracy_pct']}%"
  )
  _safe_print(
      f"  Critical Pro-Miss Rate (%):       "
      f"Tier0+1={full['accuracy']['pro_miss_rate_pct']}% | "
      f"PureTier1={t1['accuracy']['pro_miss_rate_pct']}% | "
      f"GenerateContent(N=15)={llm['accuracy']['pro_miss_rate_pct']}% | "
      f"Lexical={lex['accuracy']['pro_miss_rate_pct']}%"
  )
  _safe_print(
      f"  100-Prompt Projected E2E Time:    "
      f"Always-Pro={proj['always_pro']['total_e2e_s']}s -> "
      f"Decisions-Routed={proj['decisions_router_routed']['total_e2e_s']}s "
      f"(-{proj['decisions_router_routed']['time_saved_vs_always_pro_pct']}% time, "
      f"-{proj['decisions_router_routed']['cost_saved_vs_always_pro_pct']}% cost)"
  )
  _safe_print(f"  Saved results -> {output_path}")
  _safe_print(border)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Run Gemini Decisions Router accuracy, latency, and E2E benchmark."
  )
  parser.add_argument(
      "--dataset",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "prompts_100.json",
      help="Path to 100-prompt dataset JSON.",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=REPO_ROOT / "benchmarks" / "results_100.json",
      help="Path to write benchmark results JSON.",
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
      help="Worker concurrency (default: 1 for uncontended wall-clock percentiles).",
  )
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress per-prompt streaming lines and print summary stats table only.",
  )
  args = parser.parse_args(argv)
  if args.concurrency < 1 or args.concurrency > 32:
    parser.error("--concurrency must be between 1 and 32")

  results = run_benchmark(
      dataset_path=args.dataset,
      output_path=args.output,
      mock=bool(args.mock),
      concurrency=int(args.concurrency),
      quiet=bool(args.quiet),
  )
  _print_summary(results, args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
