#!/usr/bin/env python3
"""Antigravity lifecycle hooks and CLI for the Gemini Decisions Router plugin.

Hooks supported:
  - `pre-invocation`: Scores incoming user turns via Tier 0 (<1 ms) fast rules
    and Tier 1 Gemini Decisions API, enforces asymmetric Turn 2+ thread floors,
    deduplicates multi-step tool loops, verifies cascade ownership before
    `auto_switch`, and renders a zero-token inline routing banner
    (`metadata.user_markdown` + `renderInfo`).
  - `pre-tool-use`: Intercepts `invoke_subagent` calls when `Model` is unset or
    `"inherit"` and assigns the calibrated `flash_lite`, `flash`, or `pro` tier.

CLI commands:
  - `auth-setup [--from-env] [--api-key-stdin] [--project ID] [--auth-mode MODE]`
  - `auth-check [--live]`
  - `route --prompt "..."`
  - `status`
  - `list-models`
  - `set-models [--flash-lite ID] [--flash ID] [--pro ID]`
  - `set-mode <advisor|auto_switch|off>`
  - `set-verbosity <silent|compact|normal|verbose>`
  - `watch [--verbosity LEVEL] [--interval SEC] [--once]`
  - `stats [--verbosity LEVEL] [--json]`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import time
from typing import Any
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions_engine  # noqa: E402

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_SAFE_ENUM_RE = re.compile(r"^[A-Z0-9_]{3,96}$")
_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

MODEL_ENUMS: dict[str, str] = dict(decisions_engine.MODEL_ENUMS)

VALID_VERBOSITY_LEVELS = ("silent", "compact", "normal", "verbose")

_RECOGNIZED_DOTENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GEMINI_DECISIONS_AUTH_MODE",
    "GEMINI_DECISIONS_MODEL",
    "GEMINI_DECISIONS_HOOK_TIMEOUT_S",
)

DEFAULT_ROUTER_CONFIG: dict[str, Any] = {
    "mode": "advisor",  # "advisor" | "auto_switch" | "off"
    "verbosity": "normal",  # "silent" | "compact" | "normal" | "verbose"
    "min_confidence": 0.45,
    "hook_timeout_s": 1.5,
    "enable_fast_rules": True,
    "enable_subagent_routing": True,
    "tier_models": dict(decisions_engine.DEFAULT_TIER_MODELS),
    "custom_model_enums": {},
}


def default_state_dir() -> pathlib.Path:
  env_dir = os.environ.get("GEMINI_DECISIONS_STATE_DIR", "").strip()
  if env_dir:
    return pathlib.Path(env_dir)
  return pathlib.Path.home() / ".cache" / "gemini-decisions-router"


def default_router_config_path() -> pathlib.Path:
  env_cfg = os.environ.get("GEMINI_DECISIONS_ROUTER_CONFIG", "").strip()
  if env_cfg:
    return pathlib.Path(env_cfg)
  return (
      pathlib.Path.home()
      / ".config"
      / "gemini-decisions-router"
      / "router_config.json"
  )


def _default_dotenv_path() -> pathlib.Path:
  fn = getattr(decisions_engine, "default_private_dotenv_path", None)
  if callable(fn):
    return pathlib.Path(fn())
  return pathlib.Path.home() / ".config" / "gemini-decisions-router" / ".env"


def _effective_environ() -> Mapping[str, str]:
  fn = getattr(decisions_engine, "effective_environ", None)
  if callable(fn):
    try:
      return fn()
    except decisions_engine.DecisionError:
      return os.environ
  return os.environ


def _ensure_private_dir(path: pathlib.Path) -> None:
  path.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    os.chmod(path, 0o700)
  except OSError:
    pass


def safe_state_path(
    state_dir: pathlib.Path, conversation_id: str, prefix: str
) -> pathlib.Path:
  """Maps an arbitrary conversation_id safely inside state_dir (no traversal)."""
  raw = (conversation_id or "default").strip()
  if _SAFE_ID_RE.fullmatch(raw):
    token = raw
  else:
    token = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:24]
  safe_prefix = re.sub(r"[^A-Za-z0-9_]", "_", prefix)[:24] or "state"
  return state_dir / f"{safe_prefix}_{token}.json"


def load_router_config(
    config_path: pathlib.Path | None = None,
) -> dict[str, Any]:
  cfg = {
      **DEFAULT_ROUTER_CONFIG,
      "tier_models": dict(decisions_engine.DEFAULT_TIER_MODELS),
      "custom_model_enums": {},
  }
  p = config_path or default_router_config_path()
  if p.is_file():
    try:
      loaded = json.loads(p.read_text(encoding="utf-8"))
      if isinstance(loaded, dict):
        for k in DEFAULT_ROUTER_CONFIG:
          if k not in loaded:
            continue
          if k == "tier_models" and isinstance(loaded[k], dict):
            cfg["tier_models"] = decisions_engine.resolve_tier_models(
                loaded[k], environ={}
            )
          elif k == "custom_model_enums" and isinstance(loaded[k], dict):
            cfg["custom_model_enums"] = {
                str(mid).strip(): str(enum_val).strip()
                for mid, enum_val in loaded[k].items()
                if str(mid).strip()
                and _SAFE_ENUM_RE.fullmatch(str(enum_val).strip())
            }
          elif k == "hook_timeout_s":
            try:
              cfg["hook_timeout_s"] = max(
                  0.3, min(8.0, float(loaded["hook_timeout_s"]))
              )
            except (TypeError, ValueError):
              pass
          else:
            cfg[k] = loaded[k]
    except (OSError, ValueError):
      pass
  cfg["tier_models"] = decisions_engine.resolve_tier_models(cfg["tier_models"])
  env_mode = os.environ.get("GEMINI_DECISIONS_ROUTER_MODE", "").strip().lower()
  if env_mode in ("advisor", "auto_switch", "off"):
    cfg["mode"] = env_mode
  env_verb = os.environ.get("GEMINI_DECISIONS_VERBOSITY", "").strip().lower()
  if env_verb in VALID_VERBOSITY_LEVELS:
    cfg["verbosity"] = env_verb
  if str(cfg.get("verbosity") or "").lower() not in VALID_VERBOSITY_LEVELS:
    cfg["verbosity"] = "normal"
  env_timeout = os.environ.get("GEMINI_DECISIONS_HOOK_TIMEOUT_S", "").strip()
  if env_timeout:
    try:
      cfg["hook_timeout_s"] = max(0.3, min(8.0, float(env_timeout)))
    except ValueError:
      pass
  return cfg


def resolve_model_enums(cfg: dict[str, Any] | None = None) -> dict[str, str]:
  enums = dict(MODEL_ENUMS)
  if isinstance(cfg, dict) and isinstance(cfg.get("custom_model_enums"), dict):
    for mid, enum_val in cfg["custom_model_enums"].items():
      if str(mid).strip() and _SAFE_ENUM_RE.fullmatch(str(enum_val).strip()):
        enums[str(mid).strip()] = str(enum_val).strip()
  return enums


def save_router_config(
    updates: dict[str, Any], config_path: pathlib.Path | None = None
) -> dict[str, Any]:
  p = config_path or default_router_config_path()
  _ensure_private_dir(p.parent)
  cfg = load_router_config(p)
  for key, val in updates.items():
    if key == "tier_models" and isinstance(val, Mapping):
      merged = dict(
          cfg.get("tier_models") or decisions_engine.DEFAULT_TIER_MODELS
      )
      for tier_key in ("flash_lite", "flash", "pro"):
        raw_model = val.get(tier_key)
        if isinstance(raw_model, str) and raw_model.strip():
          merged[tier_key] = raw_model.strip()
      cfg["tier_models"] = merged
    else:
      cfg[key] = val
  p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
  return cfg


def _load_json_state(
    path: pathlib.Path, max_age_s: float
) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
      return None
    if time.time() - float(data.get("timestamp", 0)) > max_age_s:
      return None
    return data
  except (OSError, ValueError, TypeError):
    return None


def _save_json_state(path: pathlib.Path, payload: dict[str, Any]) -> None:
  try:
    _ensure_private_dir(path.parent)
    out = dict(payload)
    out["timestamp"] = time.time()
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
  except OSError:
    pass


def _append_telemetry(state_dir: pathlib.Path, event: dict[str, Any]) -> None:
  try:
    _ensure_private_dir(state_dir)
    log_file = state_dir / "telemetry.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
      f.write(json.dumps(event, ensure_ascii=False) + "\n")
  except OSError:
    pass


def _log_redispatch_error(message: str) -> None:
  try:
    sdir = default_state_dir()
    _ensure_private_dir(sdir)
    log_file = sdir / "redispatch.log"
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
      f.write(f"[{ts}] {message}\n")
  except OSError:
    pass


def _discover_ls_endpoint() -> tuple[str, str]:
  addr = os.environ.get("ANTIGRAVITY_LS_ADDRESS", "").strip()
  csrf = os.environ.get("ANTIGRAVITY_CSRF_TOKEN", "").strip()
  if addr and csrf:
    return addr, csrf
  for port in (5387,):
    candidate = f"127.0.0.1:{port}"
    try:
      req = urllib.request.Request(f"http://{candidate}/")
      with urllib.request.urlopen(req, timeout=0.4) as resp:
        html = resp.read(65536).decode("utf-8", "replace")
      m = re.search(r'["\']csrfToken["\']\s*:\s*["\']([^"\']+)["\']', html)
      if m:
        return candidate, m.group(1)
    except OSError:
      continue
  return "", ""


def _ls_rpc(
    method: str, payload: dict[str, Any], timeout: float = 0.8
) -> dict[str, Any] | None:
  addr, csrf = _discover_ls_endpoint()
  if not addr or not csrf:
    return None
  url = f"http://{addr}/exa.language_server_pb.LanguageServerService/{method}"
  req = urllib.request.Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      headers={
          "Content-Type": "application/json",
          "Connect-Protocol-Version": "1",
          "x-codeium-csrf-token": csrf,
      },
      method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return json.loads(resp.read(262144).decode("utf-8"))
  except (OSError, ValueError):
    return None


def _verify_cascade_ownership(conversation_id: str) -> bool:
  """Verifies the discovered Language Server owns `conversation_id`."""
  if not conversation_id:
    return False
  resp = _ls_rpc(
      "GetCascadeTrajectory", {"cascadeId": conversation_id}, timeout=0.4
  )
  if not isinstance(resp, dict):
    return False
  if "error" in resp:
    return False
  return "trajectory" in resp or "status" in resp


def _spawn_redispatch(
    conversation_id: str,
    model_enum: str,
    thinking_level: int,
    *,
    allowed_enums: set[str] | None = None,
) -> bool:
  addr, csrf = _discover_ls_endpoint()
  if not addr or not csrf or not conversation_id or not model_enum:
    return False
  valid_enums = allowed_enums or set(MODEL_ENUMS.values())
  if model_enum not in valid_enums:
    return False
  child_env = dict(os.environ)
  child_env["ANTIGRAVITY_LS_ADDRESS"] = addr
  child_env["ANTIGRAVITY_CSRF_TOKEN"] = csrf
  try:
    subprocess.Popen(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "redispatch",
            "--cascade-id",
            conversation_id,
            "--model-enum",
            model_enum,
            "--thinking-level",
            str(max(1, min(4, int(thinking_level)))),
        ],
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return True
  except OSError:
    return False


def _clean_user_prompt(raw: str) -> str:
  txt = re.sub(
      r"<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>", " ", raw or ""
  )
  txt = re.sub(r"</?USER_REQUEST>", " ", txt)
  return re.sub(r"\s+", " ", txt).strip()


def _prompt_preview(text: str, max_len: int = 160) -> str:
  """Returns a single-line truncated prompt preview for telemetry."""
  collapsed = re.sub(r"\s+", " ", text or "").strip()
  if len(collapsed) <= max_len:
    return collapsed
  return collapsed[: max_len - 3].rstrip() + "..."


def _format_intent_banner(
    route: dict[str, Any],
    *,
    current_model: str,
    switched: bool,
    verbosity: str = "normal",
) -> str:
  """Builds an aesthetic 'Routed with Intent' ASCII/Unicode art header."""
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  model_id = str(route.get("recommended_model_id") or "auto")
  thinking = str(route.get("thinking_label") or "Medium")
  tier = str(route.get("subagent_tier") or "flash")
  latency = int(route.get("latency_ms", 0))
  arrow = ">>" if switched else "->"
  tag = "AUTO-SWITCHED" if switched else "ADVISOR"

  if verbosity == "silent":
    return ""

  if verbosity == "compact":
    return (
        f"✦ Routed with Intent {arrow} {model_id} [{thinking}]"
        f" ({conf_pct}% | {latency} ms)"
    )

  line1 = f"  ✦  ROUTED WITH INTENT  |  {tag}  |  tier={tier}"
  line2 = f"     {current_model}  {arrow}  {model_id} [{thinking}]  ({conf_pct}% | {latency} ms)"
  width = max(len(line1), len(line2), 54) + 2
  top = "╭" + "─" * width + "╮"
  bot = "╰" + "─" * width + "╯"
  return "\n".join([
      top,
      f"│{line1.ljust(width)}│",
      f"│{line2.ljust(width)}│",
      bot,
  ])


def _build_render_info(
    route: dict[str, Any],
    *,
    current_model: str,
    switched: bool,
    verbosity: str = "normal",
) -> dict[str, str]:
  verb = verbosity if verbosity in VALID_VERBOSITY_LEVELS else "normal"
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  action = "Auto-Switched" if switched else "Recommended"
  model_id = str(route["recommended_model_id"])
  thinking = str(route["thinking_label"])
  latency = route["latency_ms"]

  if verb == "silent":
    return {
        "titlePrefix": "✦ Routed with Intent",
        "title": f"{action}: {model_id} [{thinking}] ({conf_pct}%)",
        "markdown": "",
    }

  if verb == "compact":
    title = f"{action}: {model_id} [{thinking}] ({conf_pct}% | {latency} ms)"
    markdown = (
        f"✦ **Routed with Intent** -> `{model_id}`"
        f" (`{route['task_class']}`, {conf_pct}%,"
        f" `{route['backend']}` in `{latency} ms`)"
    )
    return {
        "titlePrefix": "✦ Routed with Intent",
        "title": title,
        "markdown": markdown,
    }

  title = f"{action}: {model_id} / {thinking} effort ({conf_pct}%)"
  banner = _format_intent_banner(
      route, current_model=current_model, switched=switched, verbosity=verb
  )
  lines = [
      f"✦ **Routed with Intent** ({action}: `{model_id}` [{thinking}] · {conf_pct}%)",
      f"```text\n{banner}\n```",
      f"- **Active Dropdown Model**: `{current_model}`",
      (
          f"- **Calibrated Route**: `{model_id}`"
          f" (`subagent_tier={route['subagent_tier']}`,"
          f" thinking=`{thinking}`)"
      ),
      f"- **Task Class**: `{route['task_class']}` ({conf_pct}% confidence)",
      (
          f"- **Decision Gate**: `{route['backend']}` in"
          f" `{latency} ms` (0 output tokens)"
      ),
      f"- **Rationale**: {route['rationale']}",
  ]
  if verb == "verbose":
    probs = route.get("probabilities") or {}
    if probs:
      tc_probs = probs.get("task_class") or {}
      if isinstance(tc_probs, dict) and tc_probs:
        top_tc = ", ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(
                tc_probs.items(), key=lambda kv: kv[1], reverse=True
            )[:4]
        )
        lines.append(f"- **Class Probabilities**: `{top_tc}`")
      if "needs_pro_reasoning" in probs:
        lines.append(
            f"- **P(pro reasoning)**: `{float(probs['needs_pro_reasoning']):.3f}`"
        )
      if "effort_score" in probs:
        lines.append(
            f"- **Effort Rubric**: `{float(probs['effort_score']):.2f} / 3.0`"
        )
    lines.append(
        f"- **Thread Lock**: `{bool(route.get('thread_lock_applied'))}`"
        f" | **Explicit Request**: `{bool(route.get('explicit_model_requested'))}`"
    )
  return {
      "titlePrefix": "✦ Routed with Intent",
      "title": title,
      "markdown": "\n".join(lines),
  }


def _build_user_markdown(
    route: dict[str, Any],
    *,
    current_model: str,
    switched: bool,
    verbosity: str = "normal",
    render_info: dict[str, str] | None = None,
) -> str:
  """Builds universal zero-token inline markdown for App and agy CLI."""
  verb = verbosity if verbosity in VALID_VERBOSITY_LEVELS else "normal"
  if verb == "silent":
    return ""
  if verb == "compact":
    conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
    action = "Auto-Switched" if switched else "Recommended"
    model_id = str(route["recommended_model_id"])
    thinking = str(route["thinking_label"])
    latency = route["latency_ms"]
    return (
        f"✦ **Routed with Intent** ({action}) -> `{model_id}`"
        f" [{thinking}] · {conf_pct}% · `{route['backend']}` ({latency} ms)"
    )
  ri = (
      render_info
      if render_info is not None
      else _build_render_info(
          route,
          current_model=current_model,
          switched=switched,
          verbosity=verb,
      )
  )
  return ri.get("markdown", "")


def _make_inject_steps(
    render_info: dict[str, str],
    user_md: str,
    *,
    verbosity: str = "normal",
) -> dict[str, Any]:
  """Constructs zero-model-token injectSteps payload across App and agy CLI."""
  if verbosity == "silent" or not user_md:
    return {}
  return {
      "injectSteps": [{
          "systemMessage": {
              "systemMessage": "",
              "metadata": {"user_markdown": user_md},
              "renderInfo": render_info,
          }
      }]
  }


def _call_route_fn(
    route_fn: Callable[..., dict[str, Any]],
    prompt: str,
    *,
    cfg: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
  """Calls `route_fn`, passing `hook_timeout_s` when calling engine directly."""
  call_kwargs = dict(kwargs)
  if route_fn is decisions_engine.route_prompt:
    call_kwargs.setdefault("tier_models", cfg.get("tier_models"))
    hook_timeout = float(cfg.get("hook_timeout_s", 1.5))
    try:
      return route_fn(prompt, hook_timeout_s=hook_timeout, **call_kwargs)
    except TypeError:
      return route_fn(prompt, **call_kwargs)
  return route_fn(prompt, **call_kwargs)


def handle_pre_invocation(
    hook_input: dict[str, Any],
    *,
    state_dir: pathlib.Path | None = None,
    config_path: pathlib.Path | None = None,
    route_fn: Callable[..., dict[str, Any]] = decisions_engine.route_prompt,
) -> dict[str, Any]:
  """Executes PreInvocation routing and returns Antigravity hook JSON."""
  sdir = state_dir or default_state_dir()
  cfg = load_router_config(config_path)
  mode = str(cfg.get("mode") or "advisor")
  if mode == "off":
    return {}

  raw_input = str(hook_input.get("lastUserInput") or "")
  if "<SYSTEM_MESSAGE>" in raw_input or raw_input.lstrip().startswith(
      ("/", "[AGENT WAKE-UP]", "[CROSS-PROJECT AGENT MESSAGE]")
  ):
    return {}

  cleaned = _clean_user_prompt(raw_input)
  if not cleaned:
    return {}

  conv_id = str(hook_input.get("conversationId") or "default")
  model_name = str(hook_input.get("modelName") or "auto").strip()
  raw_inv = hook_input.get("invocationNum")
  turn_num = max(1, int(raw_inv)) if isinstance(raw_inv, int) else 1
  verbosity = str(cfg.get("verbosity") or "normal")

  prompt_hash = hashlib.sha256(
      cleaned.encode("utf-8", "ignore")
  ).hexdigest()[:16]

  reroute_file = safe_state_path(sdir, conv_id, "reroute")
  reroute_hit = _load_json_state(reroute_file, max_age_s=600.0)
  if reroute_hit and reroute_hit.get("prompt_hash") == prompt_hash:
    if reroute_hit.get("rendered"):
      return {}
    saved_ri = reroute_hit.get("render_info") or {}
    saved_verb = str(reroute_hit.get("verbosity") or verbosity)
    saved_md = str(
        reroute_hit.get("user_markdown") or saved_ri.get("markdown") or ""
    )
    _save_json_state(
        reroute_file,
        {
            "prompt_hash": prompt_hash,
            "render_info": saved_ri,
            "user_markdown": saved_md,
            "verbosity": saved_verb,
            "rendered": True,
        },
    )
    if saved_ri:
      return _make_inject_steps(saved_ri, saved_md, verbosity=saved_verb)
    return {}

  seen_file = safe_state_path(sdir, conv_id, "seen_turn")
  seen_hit = _load_json_state(seen_file, max_age_s=600.0)
  if seen_hit and seen_hit.get("prompt_hash") == prompt_hash:
    return {}

  thread_file = safe_state_path(sdir, conv_id, "thread")
  thread_state = _load_json_state(thread_file, max_age_s=7 * 86400.0)
  if thread_state and turn_num == 1:
    turn_num = int(thread_state.get("turn_count_seen", 1)) + 1

  route = _call_route_fn(
      route_fn,
      cleaned,
      cfg=cfg,
      current_model=model_name,
      user_turn_count=turn_num,
      thread_state=thread_state,
      enable_fast_rules=bool(cfg.get("enable_fast_rules", True)),
  )

  min_conf = float(cfg.get("min_confidence", 0.45))
  target_id = str(route["recommended_model_id"])
  enums_map = resolve_model_enums(cfg)
  target_enum = enums_map.get(target_id, "")
  should_switch = (
      mode == "auto_switch"
      and bool(target_enum)
      and target_id != model_name
      and float(route["confidence"]) >= min_conf
  )

  did_switch = False
  auto_switch_fallback: str | None = None

  if should_switch and not _verify_cascade_ownership(conv_id):
    should_switch = False
    did_switch = False
    auto_switch_fallback = "cascade_unowned"

  render_info = _build_render_info(
      route,
      current_model=model_name,
      switched=should_switch,
      verbosity=verbosity,
  )
  user_md = _build_user_markdown(
      route,
      current_model=model_name,
      switched=should_switch,
      verbosity=verbosity,
      render_info=render_info,
  )

  if should_switch:
    _save_json_state(
        reroute_file,
        {
            "prompt_hash": prompt_hash,
            "render_info": render_info,
            "user_markdown": user_md,
            "verbosity": verbosity,
            "rendered": False,
        },
    )
    did_switch = _spawn_redispatch(
        conv_id,
        target_enum,
        int(route["thinking_level"]),
        allowed_enums=set(enums_map.values()),
    )
    if did_switch:
      seen_file.unlink(missing_ok=True)
    else:
      reroute_file.unlink(missing_ok=True)
      auto_switch_fallback = "spawn_failed"
      render_info = _build_render_info(
          route, current_model=model_name, switched=False, verbosity=verbosity
      )
      user_md = _build_user_markdown(
          route,
          current_model=model_name,
          switched=False,
          verbosity=verbosity,
          render_info=render_info,
      )

  if not did_switch:
    reroute_file.unlink(missing_ok=True)
    _save_json_state(seen_file, {"prompt_hash": prompt_hash})

  _save_json_state(
      thread_file,
      {
          "locked_tier": route["subagent_tier"],
          "locked_model_id": route["recommended_model_id"],
          "locked_thinking_level": route["thinking_level"],
          "locked_task_class": route["task_class"],
          "turn_count_seen": turn_num,
      },
  )
  _save_json_state(
      safe_state_path(sdir, conv_id, "active_turn"),
      {
          "subagent_tier": route["subagent_tier"],
          "explicit_model_requested": bool(
              route.get("explicit_model_requested")
          ),
      },
  )

  telemetry_event: dict[str, Any] = {
      "ts": datetime.now(timezone.utc).isoformat(),
      "event": "PreInvocation",
      "prompt": _prompt_preview(cleaned),
      "mode": mode,
      "did_auto_switch": did_switch,
      "current_model": model_name,
      "recommended_model_id": route["recommended_model_id"],
      "subagent_tier": route["subagent_tier"],
      "thinking_label": route["thinking_label"],
      "task_class": route["task_class"],
      "confidence": route["confidence"],
      "latency_ms": route["latency_ms"],
      "backend": route["backend"],
  }
  if auto_switch_fallback:
    telemetry_event["auto_switch_fallback"] = auto_switch_fallback
  _append_telemetry(sdir, telemetry_event)

  if did_switch:
    return {"terminationBehavior": "terminate"}

  return _make_inject_steps(render_info, user_md, verbosity=verbosity)


def handle_pre_tool_use(
    hook_input: dict[str, Any],
    *,
    state_dir: pathlib.Path | None = None,
    config_path: pathlib.Path | None = None,
    route_fn: Callable[..., dict[str, Any]] = decisions_engine.route_prompt,
) -> dict[str, Any]:
  """Evaluates `invoke_subagent` calls when `Model` is unset or `inherit`."""
  sdir = state_dir or default_state_dir()
  cfg = load_router_config(config_path)
  if cfg.get("mode") == "off" or not cfg.get("enable_subagent_routing", True):
    return {"decision": "allow"}

  tool_call = hook_input.get("toolCall") or {}
  if str(tool_call.get("name") or "") != "invoke_subagent":
    return {"decision": "allow"}

  args = tool_call.get("args") or {}
  current_model_arg = args.get("Model")
  if current_model_arg and current_model_arg != "inherit":
    return {"decision": "allow"}

  last_user = str(hook_input.get("lastUserInput") or "").lower()
  if re.search(r"\b(?:same\s+model|inherit)\b", last_user):
    return {"decision": "allow"}

  conv_id = str(hook_input.get("conversationId") or "default")
  active_state = _load_json_state(
      safe_state_path(sdir, conv_id, "active_turn"), max_age_s=7200.0
  )
  if active_state and active_state.get("explicit_model_requested"):
    return {"decision": "allow"}

  verbosity = str(cfg.get("verbosity") or "normal")
  sub_prompt = str(args.get("Prompt") or "").strip()
  sub_role = str(args.get("Role") or "").strip()
  if not sub_prompt:
    tier = str((active_state or {}).get("subagent_tier") or "flash")
    if tier not in ("flash_lite", "flash", "pro"):
      tier = "flash"
    empty_out: dict[str, Any] = {
        "decision": "allow",
        "overwrite": {"Model": tier},
    }
    if verbosity != "silent":
      empty_out["reason"] = (
          f"✦ Routed with Intent -> subagent Model={tier} (100% | 0 ms)"
      )
    return empty_out

  eval_text = (
      sub_prompt
      if len(sub_prompt.split()) <= 4
      else f"Subagent Role: {sub_role}\nTask: {sub_prompt}"
  )
  sub_route = _call_route_fn(
      route_fn,
      eval_text,
      cfg=cfg,
      user_turn_count=1,
      enable_fast_rules=bool(cfg.get("enable_fast_rules", True)),
  )
  tier = str(sub_route.get("subagent_tier") or "flash")
  if tier not in ("flash_lite", "flash", "pro"):
    tier = "flash"

  conf_pct = int(round(float(sub_route.get("confidence", 0.0)) * 100))
  lat_ms = int(sub_route.get("latency_ms", 0))

  _append_telemetry(
      sdir,
      {
          "ts": datetime.now(timezone.utc).isoformat(),
          "event": "PreToolUse:invoke_subagent",
          "role": sub_role[:80],
          "prompt": _prompt_preview(sub_prompt),
          "assigned_tier": tier,
          "task_class": sub_route.get("task_class"),
          "confidence": sub_route.get("confidence"),
          "latency_ms": sub_route.get("latency_ms"),
          "backend": sub_route.get("backend"),
      },
  )
  out: dict[str, Any] = {"decision": "allow", "overwrite": {"Model": tier}}
  if verbosity != "silent":
    out["reason"] = (
        f"✦ Routed with Intent -> subagent Model={tier}"
        f" ({conf_pct}% | {lat_ms} ms)"
    )
  return out


# ---- CLI Entrypoints --------------------------------------------------------


def _cmd_pre_invocation() -> int:
  raw = sys.stdin.read()
  if not raw.strip():
    sys.stdout.write("{}\n")
    return 0
  try:
    payload = json.loads(raw)
    out = handle_pre_invocation(payload if isinstance(payload, dict) else {})
  except Exception:  # pylint: disable=broad-except
    out = {}
  sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
  sys.stdout.flush()
  return 0


def _cmd_pre_tool_use() -> int:
  raw = sys.stdin.read()
  if not raw.strip():
    sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")
    return 0
  try:
    payload = json.loads(raw)
    out = handle_pre_tool_use(payload if isinstance(payload, dict) else {})
  except Exception:  # pylint: disable=broad-except
    out = {"decision": "allow"}
  sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
  sys.stdout.flush()
  return 0


def _cmd_redispatch(ns: argparse.Namespace) -> int:
  try:
    deadline = time.monotonic() + 5.0
    time.sleep(0.15)
    saw_idle = False
    last_resp: dict[str, Any] | None = None
    while time.monotonic() < deadline:
      last_resp = _ls_rpc(
          "GetCascadeTrajectory", {"cascadeId": ns.cascade_id}, timeout=0.9
      )
      if last_resp and last_resp.get("status") == "CASCADE_RUN_STATUS_IDLE":
        saw_idle = True
        break
      time.sleep(0.1)
    if not saw_idle:
      _log_redispatch_error(
          f"GetCascadeTrajectory did not reach IDLE for cascade={ns.cascade_id}"
          f" (last_resp={last_resp!r})"
      )
    send_resp = _ls_rpc(
        "SendUserCascadeMessage",
        {
            "cascadeId": ns.cascade_id,
            "items": [],
            "cascadeConfig": {
                "plannerConfig": {
                    "planModel": ns.model_enum,
                    "completionConfigOverride": {
                        "thinkingLevel": int(ns.thinking_level)
                    },
                }
            },
            "blocking": False,
        },
        timeout=4.0,
    )
    if send_resp is None or (
        isinstance(send_resp, dict) and "error" in send_resp
    ):
      _log_redispatch_error(
          f"SendUserCascadeMessage failed for cascade={ns.cascade_id}"
          f" model={ns.model_enum} (resp={send_resp!r})"
      )
      return 1
  except Exception as exc:  # pylint: disable=broad-except
    _log_redispatch_error(
        f"Unhandled exception in redispatch for cascade={ns.cascade_id}: {exc!r}"
    )
    return 1
  return 0


def _cmd_route(ns: argparse.Namespace) -> int:
  rcfg = load_router_config()
  res = decisions_engine.route_prompt(
      ns.prompt,
      current_model=ns.current_model,
      user_turn_count=ns.turns,
      enable_fast_rules=not ns.no_fast_rules,
      ignore_cooldown=ns.no_fast_rules,
      tier_models=rcfg.get("tier_models"),
  )
  verb = ns.verbosity or str(rcfg.get("verbosity") or "normal")
  if ns.pretty or ns.verbosity is not None:
    if ns.verbosity == "silent":
      return 0
    if verb == "silent":
      verb = "normal"
    banner = _format_intent_banner(
        res,
        current_model=ns.current_model,
        switched=False,
        verbosity=verb,
    )
    lines = [banner]
    if verb in ("normal", "verbose"):
      lines.append(
          f"  Task: {res['task_class']}  |  Backend: {res['backend']}"
          f"  |  Latency: {res['latency_ms']} ms"
      )
      lines.append(f"  Rationale: {res['rationale']}")
    if verb == "verbose":
      probs = res.get("probabilities") or {}
      if probs:
        lines.append(f"  Probabilities: {json.dumps(probs)}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0
  sys.stdout.write(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
  return 0


def _parse_existing_dotenv(path: pathlib.Path) -> dict[str, str]:
  entries: dict[str, str] = {}
  try:
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
      return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
  except OSError:
    return {}
  for line in raw.splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
      continue
    if s.startswith("export "):
      s = s[len("export ") :].strip()
    if "=" not in s:
      continue
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
      v = v[1:-1]
    if _ENV_KEY_RE.fullmatch(k) and v:
      entries[k] = v
  return entries


def _cmd_auth_setup(ns: argparse.Namespace) -> int:
  """Creates a 0600 private .env file for GUI App and CLI sessions."""
  dotenv_path = (
      pathlib.Path(ns.dotenv_path).expanduser()
      if ns.dotenv_path
      else _default_dotenv_path()
  )
  _ensure_private_dir(dotenv_path.parent)

  try:
    st = os.lstat(dotenv_path)
    if stat.S_ISLNK(st.st_mode):
      sys.stdout.write(
          json.dumps(
              {
                  "ok": False,
                  "error": "Refusing to write to symlinked .env file.",
              },
              indent=2,
          )
          + "\n"
      )
      return 1
  except FileNotFoundError:
    pass

  entries = _parse_existing_dotenv(dotenv_path)

  if ns.from_env:
    for key in _RECOGNIZED_DOTENV_KEYS:
      val = os.environ.get(key, "").strip()
      if val:
        entries[key] = val

  if ns.api_key_stdin:
    stdin_key = sys.stdin.read().strip()
    if stdin_key:
      entries["GEMINI_API_KEY"] = stdin_key

  if ns.project is not None and ns.project.strip():
    entries["GOOGLE_CLOUD_PROJECT"] = ns.project.strip()

  if ns.auth_mode is not None and ns.auth_mode.strip():
    entries["GEMINI_DECISIONS_AUTH_MODE"] = ns.auth_mode.strip()

  cleaned_entries: dict[str, str] = {}
  for k, v in entries.items():
    clean_v = str(v).replace("\r", "").replace("\n", "").strip()
    if _ENV_KEY_RE.fullmatch(k) and clean_v:
      cleaned_entries[k] = clean_v

  lines = [
      "# Private credentials for Gemini Decisions Router (permissions: 0600)",
  ]
  for k in sorted(cleaned_entries):
    val = cleaned_entries[k]
    if any(ch in val for ch in (" ", "\t", "#", '"', "'")):
      escaped = val.replace("\\", "\\\\").replace('"', '\\"')
      lines.append(f'{k}="{escaped}"')
    else:
      lines.append(f"{k}={val}")
  payload = ("\n".join(lines) + "\n").encode("utf-8")

  fd = os.open(
      dotenv_path,
      os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
      0o600,
  )
  try:
    os.write(fd, payload)
  finally:
    os.close(fd)
  os.chmod(dotenv_path, 0o600)

  report = {
      "ok": True,
      "dotenv_path": str(dotenv_path),
      "permissions": "0600",
      "keys_written": sorted(cleaned_entries.keys()),
  }
  sys.stdout.write(json.dumps(report, indent=2) + "\n")
  return 0


def _cmd_auth_check(ns: argparse.Namespace) -> int:
  """Verifies ADC or API key setup and optionally makes a 1-question probe."""
  try:
    settings = decisions_engine.load_settings()
  except decisions_engine.DecisionError as e:
    sys.stdout.write(
        json.dumps(
            {"ok": False, "stage": "settings", "error": e.to_dict()}, indent=2
        )
        + "\n"
    )
    return 1

  report: dict[str, Any] = {
      "ok": True,
      "auth_mode": settings.auth_mode,
      "model": settings.model,
      "project_id": settings.project_id or "(from ADC)",
      "location": settings.location,
  }

  try:
    eff_env = _effective_environ()
    if settings.auth_mode == "api_key":
      has_env_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
      has_eff_key = bool(str(eff_env.get("GEMINI_API_KEY", "")).strip())
      if not has_eff_key:
        raise decisions_engine.ConfigError(
            "GEMINI_API_KEY is empty. Use keyless ADC"
            " (GEMINI_DECISIONS_AUTH_MODE=gemini_adc), run"
            " `agy-router auth-setup --api-key-stdin`, or export"
            " GEMINI_API_KEY."
        )
      report["credential_source"] = (
          "env:GEMINI_API_KEY" if has_env_key else "dotenv:GEMINI_API_KEY"
      )
      report["endpoint"] = decisions_engine.build_decisions_url(settings)
    else:
      scopes = (
          decisions_engine.ADC_SCOPES_VERTEX
          if settings.auth_mode == "vertex_adc"
          else decisions_engine.ADC_SCOPES_GEMINI
      )
      adc = decisions_engine.resolve_adc_token(
          scopes, project_hint=settings.project_id
      )
      eff_project = settings.project_id or adc.project_id
      report["credential_source"] = adc.source
      report["project_id"] = eff_project or "(unset)"
      report["endpoint"] = decisions_engine.build_decisions_url(
          settings, project_override=eff_project
      )

    if ns.live:
      probe = decisions_engine.create_decision(
          {
              "input": [{"text": "Health check ping."}],
              "questions": {
                  "ready": {
                      "type": "noul",
                      "instructions": "Is this a health check message?",
                  }
              },
          },
          settings,
      )
      report["live_probe"] = {
          "backend": probe["backend"],
          "latency_ms": probe["latency_ms"],
          "answer": probe["answers"]["ready"],
      }
  except decisions_engine.DecisionError as e:
    report["ok"] = False
    report["error"] = e.to_dict()
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 1

  sys.stdout.write(json.dumps(report, indent=2) + "\n")
  return 0


def _cmd_status() -> int:
  rcfg = load_router_config()
  eff_env = _effective_environ()
  dotenv_path = _default_dotenv_path()
  try:
    settings = decisions_engine.load_settings()
    backend_info = {
        "auth_mode": settings.auth_mode,
        "backend": settings.label,
        "model": settings.model,
        "project_id": settings.project_id or "(auto from ADC)",
        "location": settings.location,
        "gemini_api_key_set": bool(
            str(eff_env.get("GEMINI_API_KEY", "")).strip()
        ),
        "private_dotenv_present": dotenv_path.is_file(),
    }
  except decisions_engine.DecisionError as e:
    backend_info = {"error": e.to_dict()}

  ls_addr, _ = _discover_ls_endpoint()
  catalog = decisions_engine.list_available_target_models(
      tier_models=rcfg.get("tier_models"),
      custom_model_enums=rcfg.get("custom_model_enums"),
  )
  out = {
      "router_config_path": str(default_router_config_path()),
      "private_dotenv_path": str(dotenv_path),
      "router_config": rcfg,
      "available_target_models": [
          m["id"] for m in catalog["available_target_models"]
      ],
      "decisions_backend": backend_info,
      "language_server_detected": bool(ls_addr),
      "state_dir": str(default_state_dir()),
  }
  sys.stdout.write(json.dumps(out, indent=2) + "\n")
  return 0


def _cmd_list_models() -> int:
  rcfg = load_router_config()
  catalog = decisions_engine.list_available_target_models(
      tier_models=rcfg.get("tier_models"),
      custom_model_enums=rcfg.get("custom_model_enums"),
  )
  sys.stdout.write(json.dumps(catalog, indent=2) + "\n")
  return 0


def _cmd_set_models(ns: argparse.Namespace) -> int:
  rcfg = load_router_config()
  tier_models = (
      dict(decisions_engine.DEFAULT_TIER_MODELS)
      if ns.reset
      else dict(rcfg.get("tier_models") or decisions_engine.DEFAULT_TIER_MODELS)
  )
  custom_enums = (
      {} if ns.reset else dict(rcfg.get("custom_model_enums") or {})
  )
  if ns.flash_lite:
    tier_models["flash_lite"] = ns.flash_lite.strip()
  if ns.flash:
    tier_models["flash"] = ns.flash.strip()
  if ns.pro:
    tier_models["pro"] = ns.pro.strip()
  for pair in ns.custom_enum or []:
    if "=" not in pair:
      sys.stderr.write(
          f"Error: --custom-enum expects MODEL_ID=MODEL_ENUM, got {pair!r}\n"
      )
      return 2
    mid, enum_val = [p.strip() for p in pair.split("=", 1)]
    if not mid or not _SAFE_ENUM_RE.fullmatch(enum_val):
      sys.stderr.write(
          f"Error: invalid --custom-enum {pair!r} (enum must match"
          " ^[A-Z0-9_]{3,96}$)\n"
      )
      return 2
    custom_enums[mid] = enum_val

  cfg = save_router_config({
      "tier_models": tier_models,
      "custom_model_enums": custom_enums,
  })
  catalog = decisions_engine.list_available_target_models(
      tier_models=cfg.get("tier_models"),
      custom_model_enums=cfg.get("custom_model_enums"),
  )
  sys.stdout.write(
      json.dumps(
          {
              "status": "updated",
              "router_config": cfg,
              "active_tier_models": catalog["active_tier_models"],
          },
          indent=2,
      )
      + "\n"
  )
  return 0


def _cmd_set_mode(ns: argparse.Namespace) -> int:
  updates: dict[str, Any] = {"mode": ns.mode}
  if ns.min_confidence is not None:
    updates["min_confidence"] = float(ns.min_confidence)
  if getattr(ns, "verbosity", None) is not None:
    updates["verbosity"] = ns.verbosity
  cfg = save_router_config(updates)
  sys.stdout.write(
      json.dumps({"status": "updated", "router_config": cfg}, indent=2) + "\n"
  )
  return 0


def _cmd_set_verbosity(ns: argparse.Namespace) -> int:
  cfg = save_router_config({"verbosity": ns.verbosity})
  sys.stdout.write(
      json.dumps({"status": "updated", "router_config": cfg}, indent=2) + "\n"
  )
  return 0


def _format_watch_line(r: dict[str, Any], *, verbosity: str) -> str:
  ts = str(r.get("ts") or "")[11:19] or "--:--:--"
  ev = "SUB" if "PreToolUse" in str(r.get("event") or "") else "TURN"
  target = str(r.get("recommended_model_id") or r.get("assigned_tier") or "?")
  tier = str(r.get("subagent_tier") or r.get("assigned_tier") or "?")
  lat = r.get("latency_ms", 0)
  conf = r.get("confidence")
  conf_str = (
      f"{int(round(float(conf) * 100))}%"
      if isinstance(conf, (int, float))
      else "?%"
  )
  fallback = r.get("auto_switch_fallback")
  sw = (
      "SWITCH"
      if r.get("did_auto_switch")
      else ("FALLBACK" if fallback else "ADVISE")
  )
  prompt = _prompt_preview(str(r.get("prompt") or ""), max_len=72)

  if verbosity == "compact":
    return (
        f"  [{ts}] {ev} {sw} -> {target} [{tier}]"
        f" ({conf_str} | {lat} ms)  {prompt}"
    )

  lines = [
      f"  [{ts}] {ev:<4} {sw:<8} -> {target:<22} [{tier:<10}]"
      f" ({conf_str} | {lat:>3} ms)"
  ]
  if prompt:
    lines.append(f'           prompt: "{prompt}"')
  if verbosity == "verbose":
    detail = (
        f"           from={r.get('current_model', 'auto')}"
        f"  task={r.get('task_class', '?')}"
        f"  backend={r.get('backend', '?')}"
    )
    if fallback:
      detail += f"  fallback={fallback}"
    lines.append(detail)
  return "\n".join(lines)


def _cmd_watch(ns: argparse.Namespace) -> int:
  """Streams formatted routing telemetry events in real time."""
  rcfg = load_router_config()
  verb = getattr(ns, "verbosity", None) or str(rcfg.get("verbosity") or "normal")
  if verb == "silent":
    verb = "compact"
  sdir = (
      pathlib.Path(ns.state_dir).expanduser()
      if getattr(ns, "state_dir", None)
      else default_state_dir()
  )
  log_file = sdir / "telemetry.jsonl"

  header1 = "  ✦ ROUTED WITH INTENT | LIVE MONITOR"
  header2 = f"    telemetry={log_file}  verbosity={verb}"
  w = max(len(header1), len(header2), 62) + 2
  sys.stdout.write(
      "\n".join([
          "╭" + "─" * w + "╮",
          f"│{header1.ljust(w)}│",
          f"│{header2.ljust(w)}│",
          "╰" + "─" * w + "╯",
      ])
      + "\n"
  )

  offset = 0
  if log_file.is_file():
    try:
      content = log_file.read_bytes()
      offset = len(content)
      rows: list[dict[str, Any]] = []
      for line in content.decode("utf-8", errors="replace").splitlines():
        if line.strip():
          try:
            rows.append(json.loads(line))
          except ValueError:
            continue
      for r in rows[-20:]:
        sys.stdout.write(_format_watch_line(r, verbosity=verb) + "\n")
    except OSError:
      pass
  elif ns.once:
    sys.stdout.write("  (no telemetry events recorded yet)\n")

  sys.stdout.flush()
  if ns.once:
    return 0

  interval = max(0.1, float(ns.interval))
  try:
    while True:
      time.sleep(interval)
      if not log_file.is_file():
        continue
      try:
        size = log_file.stat().st_size
        if size < offset:
          offset = 0
        if size > offset:
          with open(log_file, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            offset = f.tell()
          for line in chunk.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
              continue
            try:
              row = json.loads(line)
            except ValueError:
              continue
            sys.stdout.write(_format_watch_line(row, verbosity=verb) + "\n")
          sys.stdout.flush()
      except OSError:
        continue
  except KeyboardInterrupt:
    return 0


def _format_stats_pretty(
    recent: list[dict[str, Any]],
    *,
    by_tier: dict[str, int],
    avg_ms: int,
    verbosity: str,
) -> str:
  """Formats recent telemetry events with 'Routed with Intent' CLI aesthetics."""
  tail = recent[-10:] if verbosity == "verbose" else recent[-5:]
  if verbosity == "compact":
    lines = [
        f"✦ Routed with Intent | events={len(recent)} avg={avg_ms}ms tiers={by_tier}"
    ]
    for r in tail:
      target = r.get("recommended_model_id") or r.get("assigned_tier") or "?"
      prompt = _prompt_preview(str(r.get("prompt") or ""), max_len=60)
      lat = r.get("latency_ms", 0)
      sw = ">>" if r.get("did_auto_switch") else "->"
      lines.append(f"  {sw} {target:<22} ({lat:>3} ms)  {prompt}")
    return "\n".join(lines)

  tier_summary = "  ".join(f"{k}:{v}" for k, v in sorted(by_tier.items()))
  header1 = "  ✦  ROUTED WITH INTENT  |  GEMINI DECISIONS TELEMETRY"
  header2 = f"     events={len(recent)}   avg_latency={avg_ms} ms   {tier_summary}"
  w = max(len(header1), len(header2), 62) + 2
  lines = [
      "╭" + "─" * w + "╮",
      f"│{header1.ljust(w)}│",
      f"│{header2.ljust(w)}│",
      "╰" + "─" * w + "╯",
  ]
  for r in tail:
    ts = str(r.get("ts") or "")[11:19]
    ev = "SUB" if "PreToolUse" in str(r.get("event") or "") else "TURN"
    target = str(r.get("recommended_model_id") or r.get("assigned_tier") or "?")
    tier = str(r.get("subagent_tier") or r.get("assigned_tier") or "?")
    backend = str(r.get("backend") or "?")
    lat = r.get("latency_ms", 0)
    sw = "SWITCH" if r.get("did_auto_switch") else "ADVISE"
    prompt = _prompt_preview(str(r.get("prompt") or ""), max_len=72)
    lines.append(
        f"  [{ts}] {ev:<4} {sw:<6} -> {target:<22} [{tier:<10}] {lat:>3} ms"
    )
    if prompt:
      lines.append(f'           prompt: "{prompt}"')
    if verbosity == "verbose":
      lines.append(
          f"           from={r.get('current_model', 'auto')}"
          f"  task={r.get('task_class', '?')}"
          f"  conf={r.get('confidence', '?')}"
          f"  backend={backend}"
      )
  return "\n".join(lines)


def _cmd_stats(ns: argparse.Namespace) -> int:
  log_file = default_state_dir() / "telemetry.jsonl"
  if not log_file.is_file():
    sys.stdout.write(
        json.dumps({"total_events": 0, "events": []}, indent=2) + "\n"
    )
    return 0
  rows = []
  for line in log_file.read_text(
      encoding="utf-8", errors="replace"
  ).splitlines():
    if line.strip():
      try:
        rows.append(json.loads(line))
      except ValueError:
        continue
  recent = rows[-ns.limit :] if ns.limit > 0 else rows
  by_tier: dict[str, int] = {}
  latencies: list[int] = []
  for r in recent:
    t = str(r.get("subagent_tier") or r.get("assigned_tier") or "unknown")
    by_tier[t] = by_tier.get(t, 0) + 1
    if isinstance(r.get("latency_ms"), int):
      latencies.append(r["latency_ms"])
  avg_ms = int(round(sum(latencies) / len(latencies))) if latencies else 0
  if not getattr(ns, "as_json", False):
    rcfg = load_router_config()
    verb = getattr(ns, "verbosity", None) or str(
        rcfg.get("verbosity") or "normal"
    )
    if getattr(ns, "verbosity", None) == "silent":
      return 0
    if verb == "silent":
      verb = "normal"
    sys.stdout.write(
        _format_stats_pretty(
            recent, by_tier=by_tier, avg_ms=avg_ms, verbosity=verb
        )
        + "\n"
    )
    return 0
  sys.stdout.write(
      json.dumps(
          {
              "events_analyzed": len(recent),
              "avg_latency_ms": avg_ms,
              "by_tier": by_tier,
              "recent": recent[-5:],
          },
          indent=2,
      )
      + "\n"
  )
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Gemini Decisions Router hooks and CLI for Antigravity"
  )
  sub = parser.add_subparsers(dest="cmd")
  sub.add_parser("pre-invocation")
  sub.add_parser("pre-tool-use")

  p_redisp = sub.add_parser("redispatch")
  p_redisp.add_argument("--cascade-id", required=True)
  p_redisp.add_argument("--model-enum", required=True)
  p_redisp.add_argument("--thinking-level", type=int, default=2)

  p_route = sub.add_parser("route")
  p_route.add_argument("--prompt", required=True)
  p_route.add_argument("--current-model", default="auto")
  p_route.add_argument("--turns", type=int, default=1)
  p_route.add_argument("--no-fast-rules", action="store_true")
  p_route.add_argument("--pretty", action="store_true")
  p_route.add_argument(
      "--verbosity", choices=list(VALID_VERBOSITY_LEVELS), default=None
  )

  p_setup = sub.add_parser("auth-setup")
  p_setup.add_argument(
      "--from-env",
      action="store_true",
      help="Copy active GEMINI_API_KEY / GOOGLE_CLOUD_* variables from shell.",
  )
  p_setup.add_argument(
      "--api-key-stdin",
      action="store_true",
      help="Read GEMINI_API_KEY securely from standard input.",
  )
  p_setup.add_argument(
      "--project",
      default=None,
      help="Set GOOGLE_CLOUD_PROJECT in private .env.",
  )
  p_setup.add_argument(
      "--auth-mode",
      choices=["auto", "gemini_adc", "vertex_adc", "api_key"],
      default=None,
      help="Set GEMINI_DECISIONS_AUTH_MODE in private .env.",
  )
  p_setup.add_argument(
      "--dotenv-path",
      default=None,
      help="Override target .env path (default: ~/.config/gemini-decisions-router/.env).",
  )

  p_auth = sub.add_parser("auth-check")
  p_auth.add_argument(
      "--live",
      action="store_true",
      help="Send a 1-question probe to the Gemini Decisions API.",
  )

  sub.add_parser("status")
  sub.add_parser("list-models")

  p_models = sub.add_parser("set-models")
  p_models.add_argument("--flash-lite", default=None)
  p_models.add_argument("--flash", default=None)
  p_models.add_argument("--pro", default=None)
  p_models.add_argument(
      "--custom-enum",
      action="append",
      default=[],
      help="Register custom model ID to enum mapping (MODEL_ID=MODEL_ENUM).",
  )
  p_models.add_argument(
      "--reset",
      action="store_true",
      help="Reset tier_models and custom_model_enums to defaults.",
  )

  p_mode = sub.add_parser("set-mode")
  p_mode.add_argument("mode", choices=["advisor", "auto_switch", "off"])
  p_mode.add_argument("--min-confidence", type=float, default=None)
  p_mode.add_argument(
      "--verbosity", choices=list(VALID_VERBOSITY_LEVELS), default=None
  )

  p_verb = sub.add_parser("set-verbosity")
  p_verb.add_argument("verbosity", choices=list(VALID_VERBOSITY_LEVELS))

  p_watch = sub.add_parser("watch")
  p_watch.add_argument(
      "--verbosity", choices=list(VALID_VERBOSITY_LEVELS), default=None
  )
  p_watch.add_argument("--interval", type=float, default=1.0)
  p_watch.add_argument("--once", action="store_true")
  p_watch.add_argument("--state-dir", default=None)

  p_stats = sub.add_parser("stats")
  p_stats.add_argument("--limit", type=int, default=50)
  p_stats.add_argument(
      "--verbosity", choices=list(VALID_VERBOSITY_LEVELS), default=None
  )
  p_stats.add_argument("--json", dest="as_json", action="store_true")

  ns = parser.parse_args()
  if ns.cmd in (None, "pre-invocation"):
    return _cmd_pre_invocation()
  if ns.cmd == "pre-tool-use":
    return _cmd_pre_tool_use()
  if ns.cmd == "redispatch":
    return _cmd_redispatch(ns)
  if ns.cmd == "route":
    return _cmd_route(ns)
  if ns.cmd == "auth-setup":
    return _cmd_auth_setup(ns)
  if ns.cmd == "auth-check":
    return _cmd_auth_check(ns)
  if ns.cmd == "status":
    return _cmd_status()
  if ns.cmd == "list-models":
    return _cmd_list_models()
  if ns.cmd == "set-models":
    return _cmd_set_models(ns)
  if ns.cmd == "set-mode":
    return _cmd_set_mode(ns)
  if ns.cmd == "set-verbosity":
    return _cmd_set_verbosity(ns)
  if ns.cmd == "watch":
    return _cmd_watch(ns)
  if ns.cmd == "stats":
    return _cmd_stats(ns)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
