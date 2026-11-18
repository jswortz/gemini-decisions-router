#!/usr/bin/env python3
"""Antigravity lifecycle hooks and CLI for the Gemini Decisions Router plugin.

Hooks supported:
  - `pre-invocation`: Scores incoming user turns via Tier 0 (<1 ms) fast rules
    and Tier 1 Gemini Decisions API, enforces asymmetric Turn 2+ thread floors,
    and renders a zero-token collapsible routing badge (`renderInfo`).
  - `pre-tool-use`: Intercepts `invoke_subagent` calls when `Model` is unset or
    `"inherit"` and assigns the calibrated `flash_lite`, `flash`, or `pro` tier.

CLI commands:
  - `auth-check [--live]`
  - `route --prompt "..."`
  - `status`
  - `set-mode <advisor|auto_switch|off>`
  - `stats`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions_engine  # noqa: E402

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_SAFE_ENUM_RE = re.compile(r"^[A-Z0-9_]{3,96}$")

MODEL_ENUMS: dict[str, str] = dict(decisions_engine.MODEL_ENUMS)

DEFAULT_ROUTER_CONFIG: dict[str, Any] = {
    "mode": "advisor",  # "advisor" | "auto_switch" | "off"
    "min_confidence": 0.45,
    "enable_fast_rules": True,
    "enable_subagent_routing": True,
    "tier_models": dict(decisions_engine.DEFAULT_TIER_MODELS),
    "custom_model_enums": {},
}


def default_state_dir() -> pathlib.Path:
  return pathlib.Path.home() / ".cache" / "gemini-decisions-router"


def default_router_config_path() -> pathlib.Path:
  return (
      pathlib.Path.home()
      / ".config"
      / "gemini-decisions-router"
      / "router_config.json"
  )


def _ensure_private_dir(path: pathlib.Path) -> None:
  path.mkdir(parents=True, exist_ok=True, mode=0o700)


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
          else:
            cfg[k] = loaded[k]
    except (OSError, ValueError):
      pass
  cfg["tier_models"] = decisions_engine.resolve_tier_models(cfg["tier_models"])
  env_mode = os.environ.get("GEMINI_DECISIONS_ROUTER_MODE", "").strip().lower()
  if env_mode in ("advisor", "auto_switch", "off"):
    cfg["mode"] = env_mode
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
  cfg.update(updates)
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


def _build_render_info(
    route: dict[str, Any], *, current_model: str, switched: bool
) -> dict[str, str]:
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  action = "Auto-Switched" if switched else "Recommended"
  title = (
      f"{action}: {route['recommended_model_id']} /"
      f" {route['thinking_label']} effort ({conf_pct}%)"
  )
  lines = [
      f"- **Active Dropdown Model**: `{current_model}`",
      (
          f"- **Calibrated Route**: `{route['recommended_model_id']}`"
          f" (`subagent_tier={route['subagent_tier']}`,"
          f" thinking=`{route['thinking_label']}`)"
      ),
      f"- **Task Class**: `{route['task_class']}` ({conf_pct}% confidence)",
      (
          f"- **Decision Gate**: `{route['backend']}` in"
          f" `{route['latency_ms']} ms` (0 output tokens)"
      ),
      f"- **Rationale**: {route['rationale']}",
  ]
  return {
      "titlePrefix": "Gemini Decisions Router",
      "title": title,
      "markdown": "\n".join(lines),
  }


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

  reroute_file = safe_state_path(sdir, conv_id, "reroute")
  reroute_hit = _load_json_state(reroute_file, max_age_s=600.0)
  prompt_hash = hashlib.sha256(
      cleaned.encode("utf-8", "ignore")
  ).hexdigest()[:16]
  if reroute_hit and reroute_hit.get("prompt_hash") == prompt_hash:
    if reroute_hit.get("rendered"):
      return {}
    _save_json_state(
        reroute_file,
        {
            "prompt_hash": prompt_hash,
            "render_info": reroute_hit.get("render_info") or {},
            "rendered": True,
        },
    )
    render_info = reroute_hit.get("render_info") or {}
    if render_info:
      return {
          "injectSteps": [{
              "systemMessage": {
                  "systemMessage": "",
                  "renderInfo": render_info,
              }
          }]
      }
    return {}

  thread_file = safe_state_path(sdir, conv_id, "thread")
  thread_state = _load_json_state(thread_file, max_age_s=7 * 86400.0)
  if thread_state and turn_num == 1:
    turn_num = int(thread_state.get("turn_count_seen", 1)) + 1

  route_kwargs: dict[str, Any] = {
      "current_model": model_name,
      "user_turn_count": turn_num,
      "thread_state": thread_state,
      "enable_fast_rules": bool(cfg.get("enable_fast_rules", True)),
  }
  if route_fn is decisions_engine.route_prompt:
    route_kwargs["tier_models"] = cfg.get("tier_models")
  route = route_fn(cleaned, **route_kwargs)

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
  render_info = _build_render_info(
      route, current_model=model_name, switched=should_switch
  )

  if should_switch:
    _save_json_state(
        reroute_file,
        {
            "prompt_hash": prompt_hash,
            "render_info": render_info,
            "rendered": False,
        },
    )
    did_switch = _spawn_redispatch(
        conv_id,
        target_enum,
        int(route["thinking_level"]),
        allowed_enums=set(enums_map.values()),
    )
    if not did_switch:
      render_info = _build_render_info(
          route, current_model=model_name, switched=False
      )

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
  _append_telemetry(
      sdir,
      {
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
      },
  )

  if did_switch:
    return {"terminationBehavior": "terminate"}

  return {
      "injectSteps": [{
          "systemMessage": {
              "systemMessage": "",
              "renderInfo": render_info,
          }
      }]
  }


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

  sub_prompt = str(args.get("Prompt") or "").strip()
  sub_role = str(args.get("Role") or "").strip()
  if not sub_prompt:
    tier = str((active_state or {}).get("subagent_tier") or "flash")
    if tier not in ("flash_lite", "flash", "pro"):
      tier = "flash"
    return {"decision": "allow", "overwrite": {"Model": tier}}

  eval_text = (
      sub_prompt
      if len(sub_prompt.split()) <= 4
      else f"Subagent Role: {sub_role}\nTask: {sub_prompt}"
  )
  sub_route = route_fn(
      eval_text,
      user_turn_count=1,
      enable_fast_rules=bool(cfg.get("enable_fast_rules", True)),
  )
  tier = str(sub_route.get("subagent_tier") or "flash")
  if tier not in ("flash_lite", "flash", "pro"):
    tier = "flash"

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
  return {"decision": "allow", "overwrite": {"Model": tier}}


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
  deadline = time.monotonic() + 5.0
  time.sleep(0.15)
  while time.monotonic() < deadline:
    resp = _ls_rpc(
        "GetCascadeTrajectory", {"cascadeId": ns.cascade_id}, timeout=0.9
    )
    if resp and resp.get("status") == "CASCADE_RUN_STATUS_IDLE":
      break
    time.sleep(0.1)
  _ls_rpc(
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
  sys.stdout.write(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
  return 0


def _cmd_auth_check(ns: argparse.Namespace) -> int:
  """Verifies ADC or API key setup and optionally makes a 1-question probe."""
  try:
    settings = decisions_engine.load_settings()
  except decisions_engine.DecisionError as e:
    sys.stdout.write(
        json.dumps({"ok": False, "stage": "settings", "error": e.to_dict()}, indent=2)
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
    if settings.auth_mode == "api_key":
      has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
      if not has_key:
        raise decisions_engine.ConfigError(
            "GEMINI_API_KEY is empty. Use keyless ADC"
            " (GEMINI_DECISIONS_AUTH_MODE=gemini_adc) or export GEMINI_API_KEY."
        )
      report["credential_source"] = "env:GEMINI_API_KEY"
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
  try:
    settings = decisions_engine.load_settings()
    backend_info = {
        "auth_mode": settings.auth_mode,
        "backend": settings.label,
        "model": settings.model,
        "project_id": settings.project_id or "(auto from ADC)",
        "location": settings.location,
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
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
      {}
      if ns.reset
      else dict(rcfg.get("custom_model_enums") or {})
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
  cfg = save_router_config(updates)
  sys.stdout.write(
      json.dumps({"status": "updated", "router_config": cfg}, indent=2) + "\n"
  )
  return 0


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

  p_stats = sub.add_parser("stats")
  p_stats.add_argument("--limit", type=int, default=50)

  ns = parser.parse_args()
  if ns.cmd in (None, "pre-invocation"):
    return _cmd_pre_invocation()
  if ns.cmd == "pre-tool-use":
    return _cmd_pre_tool_use()
  if ns.cmd == "redispatch":
    return _cmd_redispatch(ns)
  if ns.cmd == "route":
    return _cmd_route(ns)
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
  if ns.cmd == "stats":
    return _cmd_stats(ns)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
