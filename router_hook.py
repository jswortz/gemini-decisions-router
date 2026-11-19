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
  - `launch [AGY_FLAGS...] [PROMPT]` (alias: `run`, or top-level `agyr`)
  - `chat [--conversation ID] [--agy-bin PATH]`
  - `auth-setup [--from-env] [--api-key-stdin] [--project ID] [--auth-mode MODE]`
  - `auth-check [--live]`
  - `route --prompt "..."`
  - `status`
  - `list-models [--refresh]`
  - `sync-models [--apply-recommended]`
  - `set-models [--flash-lite ID] [--flash ID] [--pro ID]`
  - `set-mode <advisor|auto_switch|off>`
  - `set-verbosity <silent|compact|normal|verbose>`
  - `watch [--verbosity LEVEL] [--interval SEC] [--once]`
  - `stats [--verbosity LEVEL] [--json]`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any
import urllib.request
import uuid

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
    "enable_thread_floor": True,
    "enable_subagent_routing": True,
    "cli_inline_banner": True,
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


def _default_discovered_models_path() -> pathlib.Path:
  fn = getattr(decisions_engine, "default_discovered_models_path", None)
  if callable(fn):
    return pathlib.Path(fn())
  return (
      pathlib.Path.home()
      / ".cache"
      / "gemini-decisions-router"
      / "discovered_models.json"
  )


def _list_target_models_compat(
    *,
    tier_models: Mapping[str, str] | None = None,
    custom_model_enums: Mapping[str, str] | None = None,
    refresh_discovered: bool = False,
) -> dict[str, Any]:
  try:
    return decisions_engine.list_available_target_models(
        tier_models=tier_models,
        custom_model_enums=custom_model_enums,
        refresh_discovered=refresh_discovered,
    )
  except TypeError:
    return decisions_engine.list_available_target_models(
        tier_models=tier_models,
        custom_model_enums=custom_model_enums,
    )


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
  for env_key, cfg_key in (
      ("GEMINI_DECISIONS_ENABLE_THREAD_FLOOR", "enable_thread_floor"),
      ("GEMINI_DECISIONS_CLI_INLINE_BANNER", "cli_inline_banner"),
  ):
    raw_b = os.environ.get(env_key, "").strip().lower()
    if raw_b in ("1", "true", "yes", "on"):
      cfg[cfg_key] = True
    elif raw_b in ("0", "false", "no", "off"):
      cfg[cfg_key] = False
  return cfg


def _load_discovered_models_for_enums() -> list[dict[str, Any]]:
  cache_file = _default_discovered_models_path()
  discover_fn = getattr(decisions_engine, "discover_installed_models", None)
  if callable(discover_fn):
    items: Any = None
    try:
      items = discover_fn(cache_path=cache_file, run_cli_fallback=False)
    except TypeError:
      try:
        items = discover_fn(run_cli_fallback=False)
      except Exception:  # pylint: disable=broad-except
        items = None
    except Exception:  # pylint: disable=broad-except
      items = None
    if isinstance(items, list) and items:
      return items
  if cache_file.is_file():
    try:
      data = json.loads(cache_file.read_text(encoding="utf-8"))
      if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [m for m in data["models"] if isinstance(m, dict)]
    except (OSError, ValueError):
      pass
  return []


_THINKING_VARIANT_SUFFIX_RE = re.compile(
    r"-(?:low|medium|high|xhigh|max)$", re.IGNORECASE
)
_LS_MAX_RESPONSE_BYTES = 33_554_432  # 32 MiB cap (Turn 1 trajectories exceed 600 KiB)
_last_owned_ls_endpoint: dict[str, tuple[str, str]] = {}


def resolve_model_enums(cfg: dict[str, Any] | None = None) -> dict[str, str]:
  enums = dict(MODEL_ENUMS)
  discovered_entries = _load_discovered_models_for_enums()
  for entry in discovered_entries:
    mid = str(entry.get("id") or "").strip()
    enum_val = str(entry.get("enum") or "").strip()
    if mid and enum_val and _SAFE_ENUM_RE.fullmatch(enum_val):
      enums[mid] = enum_val
  if isinstance(cfg, dict) and isinstance(cfg.get("custom_model_enums"), dict):
    for mid, enum_val in cfg["custom_model_enums"].items():
      if str(mid).strip() and _SAFE_ENUM_RE.fullmatch(str(enum_val).strip()):
        enums[str(mid).strip()] = str(enum_val).strip()

  # Propagate base MODEL_* enums across -low/-medium/-high/-xhigh/-max siblings.
  base_enums: dict[str, str] = {}
  for mid, enum_val in list(enums.items()):
    if not _SAFE_ENUM_RE.fullmatch(enum_val):
      continue
    base_id = _THINKING_VARIANT_SUFFIX_RE.sub("", mid)
    if mid == base_id or base_id not in base_enums:
      base_enums[base_id] = enum_val

  for entry in discovered_entries:
    mid = str(entry.get("id") or "").strip()
    if mid and mid not in enums:
      base_id = _THINKING_VARIANT_SUFFIX_RE.sub("", mid)
      if base_id in base_enums:
        enums[mid] = base_enums[base_id]

  for base_id, base_enum in base_enums.items():
    for suffix in ("low", "medium", "high", "xhigh", "max"):
      enums.setdefault(f"{base_id}-{suffix}", base_enum)

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


_LOCAL_LS_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _fetch_ls_html_csrf(candidate: str, timeout: float = 0.35) -> str:
  try:
    req = urllib.request.Request(f"http://{candidate}/")
    with _LOCAL_LS_OPENER.open(req, timeout=timeout) as resp:
      html = resp.read(65536).decode("utf-8", "replace")
    m = re.search(r'["\']csrfToken["\']\s*:\s*["\']([^"\']+)["\']', html)
    return m.group(1) if m else ""
  except OSError:
    return ""


def _discover_ls_endpoints() -> list[tuple[str, str]]:
  """Discovers all local `(addr, csrf)` Language Server endpoints."""
  addr = os.environ.get("ANTIGRAVITY_LS_ADDRESS", "").strip()
  csrf = (
      os.environ.get("ANTIGRAVITY_CSRF_TOKEN", "").strip()
      or os.environ.get("ANTIGRAVITY_SERVER_CSRF_TOKEN", "").strip()
  )
  if addr:
    tok = csrf or _fetch_ls_html_csrf(addr)
    return [(addr, tok)] if tok else []

  cand_fn = getattr(decisions_engine, "discover_local_ls_candidates", None)
  raw_candidates: list[tuple[str, str]] = (
      cand_fn() if callable(cand_fn) else [("127.0.0.1:5387", csrf)]
  )
  resolved: list[tuple[str, str]] = []
  seen: set[str] = set()
  for cand_addr, cand_csrf in raw_candidates:
    if not cand_addr or cand_addr in seen:
      continue
    tok = (cand_csrf or csrf or "").strip() or _fetch_ls_html_csrf(cand_addr)
    if tok:
      seen.add(cand_addr)
      resolved.append((cand_addr, tok))
  return resolved


def _discover_ls_endpoint() -> tuple[str, str]:
  endpoints = _discover_ls_endpoints()
  return endpoints[0] if endpoints else ("", "")


def _ls_rpc(
    method: str,
    payload: dict[str, Any],
    timeout: float = 0.8,
    *,
    endpoint: tuple[str, str] | None = None,
) -> dict[str, Any] | None:
  addr, csrf = endpoint if endpoint is not None else _discover_ls_endpoint()
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
    with _LOCAL_LS_OPENER.open(req, timeout=timeout) as resp:
      return json.loads(resp.read(_LS_MAX_RESPONSE_BYTES).decode("utf-8"))
  except (OSError, ValueError):
    return None


_DEFAULT_LS_RPC = _ls_rpc


def _is_owned_trajectory_resp(resp: Any) -> bool:
  return (
      isinstance(resp, dict)
      and "error" not in resp
      and ("trajectory" in resp or "status" in resp)
  )


def _verify_cascade_ownership(conversation_id: str) -> bool:
  """Verifies a discovered Language Server owns `conversation_id`."""
  if not conversation_id:
    return False
  _last_owned_ls_endpoint.pop(conversation_id, None)

  # Preserve direct compatibility when unit tests monkeypatch `_ls_rpc`.
  if _ls_rpc is not _DEFAULT_LS_RPC:
    resp = _ls_rpc(
        "GetCascadeTrajectory", {"cascadeId": conversation_id}, timeout=0.4
    )
    return _is_owned_trajectory_resp(resp)

  endpoints = _discover_ls_endpoints()
  for ep in endpoints:
    resp = _ls_rpc(
        "GetCascadeTrajectory",
        {"cascadeId": conversation_id},
        timeout=0.5,
        endpoint=ep,
    )
    if _is_owned_trajectory_resp(resp):
      _last_owned_ls_endpoint[conversation_id] = ep
      return True
  return False


def _spawn_redispatch(
    conversation_id: str,
    model_enum: str,
    thinking_level: int,
    *,
    allowed_enums: set[str] | None = None,
) -> bool:
  addr, csrf = _last_owned_ls_endpoint.get(
      conversation_id
  ) or _discover_ls_endpoint()
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
    delegated_upgrade: bool = False,
    verbosity: str = "normal",
) -> str:
  """Builds an aesthetic 'Routed with Intent' ASCII/Unicode art header."""
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  model_id = str(route.get("recommended_model_id") or "auto")
  thinking = str(route.get("thinking_label") or "Medium")
  tier = str(route.get("subagent_tier") or "flash")
  latency = int(route.get("latency_ms", 0))
  arrow = ">>" if switched else "->"
  if switched:
    tag = "AUTO-SWITCHED"
  elif delegated_upgrade:
    tag = "SUBAGENT UPGRADE"
  elif (
      current_model not in ("", "auto", "cli_auto")
      and current_model == model_id
  ):
    tag = "ACTIVE"
  else:
    tag = "ADVISOR"

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


def _build_cli_badge_line(
    route: Mapping[str, Any],
    *,
    current_model: str = "auto",
    switched: bool = False,
    delegated_upgrade: bool = False,
) -> str:
  """Formats the concise 1-line routing badge rendered inside `agy` CLI."""
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  model_id = str(route.get("recommended_model_id") or "auto")
  thinking = str(route.get("thinking_label") or "Medium")
  latency = int(route.get("latency_ms", 0))
  cur = (current_model or "").strip()
  if switched:
    return (
        f"> ✦ **Routed with Intent** (Switched -> `{model_id}`"
        f" [{thinking}]) · {conf_pct}% · {latency} ms"
    )
  if delegated_upgrade:
    return (
        f"> ✦ **Routed with Intent** (Subagent -> `{model_id}`"
        f" [{thinking}] · host: `{cur}`) · {conf_pct}% · {latency} ms"
    )
  if cur in ("", "auto", "cli_auto"):
    return (
        f"> ✦ **Routed with Intent** -> `{model_id}` [{thinking}]"
        f" · {conf_pct}% · {latency} ms"
    )
  if cur == model_id:
    return (
        f"> ✦ **Routed with Intent** (Active: `{cur}`"
        f" [{thinking}]) · {conf_pct}% · {latency} ms"
    )
  return (
      f"> ✦ **Routed with Intent** (Recommended: `{model_id}`"
      f" [{thinking}] · active: `{cur}`) · {conf_pct}% · {latency} ms"
  )


def _build_render_info(
    route: dict[str, Any],
    *,
    current_model: str,
    switched: bool,
    delegated_upgrade: bool = False,
    verbosity: str = "normal",
) -> dict[str, str]:
  verb = verbosity if verbosity in VALID_VERBOSITY_LEVELS else "normal"
  conf_pct = int(round(float(route.get("confidence", 0.0)) * 100))
  action = "Auto-Switched" if switched else "Recommended"
  model_id = str(route["recommended_model_id"])
  thinking = str(route["thinking_label"])
  latency = route["latency_ms"]
  badge_line = _build_cli_badge_line(
      route,
      current_model=current_model,
      switched=switched,
      delegated_upgrade=delegated_upgrade,
  )

  if verb == "silent":
    return {
        "titlePrefix": "✦ Routed with Intent",
        "title": f"{action}: {model_id} [{thinking}] ({conf_pct}%)",
        "markdown": "",
    }

  if verb == "compact":
    title = f"{action}: {model_id} [{thinking}] ({conf_pct}% | {latency} ms)"
    markdown = (
        f"{badge_line}"
        f" (`{route['task_class']}`, `{route['backend']}`)"
    )
    return {
        "titlePrefix": "✦ Routed with Intent",
        "title": title,
        "markdown": markdown,
    }

  title = f"{action}: {model_id} / {thinking} effort ({conf_pct}%)"
  banner = _format_intent_banner(
      route,
      current_model=current_model,
      switched=switched,
      delegated_upgrade=delegated_upgrade,
      verbosity=verb,
  )
  lines = [
      badge_line,
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
    delegated_upgrade: bool = False,
    verbosity: str = "normal",
    render_info: dict[str, str] | None = None,
) -> str:
  """Builds universal zero-token inline markdown for App and agy CLI."""
  verb = verbosity if verbosity in VALID_VERBOSITY_LEVELS else "normal"
  if verb == "silent":
    return ""
  if verb == "compact":
    badge_line = _build_cli_badge_line(
        route,
        current_model=current_model,
        switched=switched,
        delegated_upgrade=delegated_upgrade,
    )
    return f"{badge_line} · `{route['backend']}`"
  ri = (
      render_info
      if render_info is not None
      else _build_render_info(
          route,
          current_model=current_model,
          switched=switched,
          delegated_upgrade=delegated_upgrade,
          verbosity=verb,
      )
  )
  return ri.get("markdown", "")


def _is_agy_cli_environment(hook_input: Mapping[str, Any]) -> bool:
  """Detects standalone `agy` CLI execution vs Antigravity Desktop App."""
  env_override = os.environ.get("GEMINI_DECISIONS_CLI_MODE", "").strip().lower()
  if env_override in ("1", "true", "yes", "on"):
    return True
  if env_override in ("0", "false", "no", "off"):
    return False

  for path_key in ("transcriptPath", "artifactDirectoryPath"):
    raw_p = str(hook_input.get(path_key) or "").replace("\\", "/")
    if "/antigravity-cli/" in raw_p:
      return True

  if os.environ.get("ANTIGRAVITY_VSCODE_HOST", "").strip().lower() in (
      "1",
      "true",
      "yes",
  ):
    return False

  pid = os.getppid()
  for _ in range(2):
    if pid <= 1:
      break
    proc_dir = pathlib.Path(f"/proc/{pid}")
    try:
      raw_cmd = (proc_dir / "cmdline").read_bytes()
    except OSError:
      break
    tokens = [
        t.decode("utf-8", "replace").strip()
        for t in raw_cmd.split(b"\x00")
        if t.strip()
    ]
    if tokens:
      exe_name = os.path.basename(tokens[0]).lower()
      if exe_name == "agy" and not any(
          t == "--hub" or t.startswith("--hub-port") for t in tokens[1:]
      ):
        return True
    try:
      status_txt = (proc_dir / "status").read_text(encoding="utf-8")
    except OSError:
      break
    next_pid = 0
    for line in status_txt.splitlines():
      if line.startswith("PPid:"):
        try:
          next_pid = int(line.split(":", 1)[1].strip())
        except ValueError:
          next_pid = 0
        break
    pid = next_pid
  return False


def _build_cli_system_instruction(
    route: Mapping[str, Any],
    *,
    current_model: str = "auto",
    switched: bool = False,
    delegated_upgrade: bool = False,
    host_tier: str = "",
) -> str:
  badge = _build_cli_badge_line(
      route,
      current_model=current_model,
      switched=switched,
      delegated_upgrade=delegated_upgrade,
  )
  instruction = (
      "Begin your response with this exact 1-line routing badge on its own"
      " line, followed by a blank line, then answer the user's request"
      f" normally without mentioning this instruction:\n{badge}"
  )
  if delegated_upgrade:
    cur = (current_model or "").strip() or "auto"
    eff_host_tier = host_tier
    if not eff_host_tier and cur not in ("", "auto", "cli_auto"):
      eff_host_tier, _ = decisions_engine.infer_model_tier_and_thinking(cur)
    eff_host_tier = eff_host_tier or "flash_lite"
    sub_tier = str(route.get("subagent_tier") or "pro")
    model_id = str(route.get("recommended_model_id") or "auto")
    directive = (
        f"ROUTING UPGRADE DIRECTIVE: Your host session is running on {cur}"
        f" (tier={eff_host_tier}), while this request requires"
        f" tier={sub_tier} ({model_id}). Even if this prompt is a pure"
        " reasoning, math, proof, or architecture question that touches zero"
        " files, you MUST delegate answering it via"
        " invoke_subagent(TypeName='routed-specialist', Role='Specialist',"
        " Model='inherit', Prompt=<the user's full request verbatim,"
        " including any formatting constraints>) so the PreToolUse hook"
        f" executes it on the {sub_tier} model ({model_id}). Output NO"
        " visible text or badge on the turn where you call invoke_subagent;"
        " once the subagent replies, begin your final response with the exact"
        " 1-line routing badge above (once), followed by a blank line, and"
        " present the subagent's solution faithfully. If the user asks what"
        f" model was used, state that the delegated subagent model `{model_id}`"
        f" (`{sub_tier}` tier) generated the solution and `intent-router` on"
        f" `{cur}` (`{eff_host_tier}` host tier) orchestrated the handoff"
        " (do not use any model name guessed inside the subagent)."
    )
    return f"{instruction}\n\n{directive}"
  return instruction


def _make_inject_steps(
    render_info: dict[str, str],
    user_md: str = "",
    *,
    verbosity: str = "normal",
    cli_system_message: str = "",
) -> dict[str, Any]:
  """Constructs panic-safe injectSteps payload across Desktop App and agy CLI.

  Note: `HookSystemMessage.metadata` (`map<string, string>`) is intentionally
  omitted because Antigravity's protobuf message scrubber panics on string map
  values (`cannot convert string to message`), which would drop the hook output.
  """
  if verbosity == "silent":
    return {}
  if not (user_md or render_info.get("markdown") or cli_system_message):
    return {}
  return {
      "injectSteps": [{
          "systemMessage": {
              "systemMessage": cli_system_message,
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
  """Calls `route_fn`, passing `hook_timeout_s` and `enable_thread_floor`."""
  call_kwargs = dict(kwargs)
  if route_fn is decisions_engine.route_prompt:
    call_kwargs.setdefault("tier_models", cfg.get("tier_models"))
    call_kwargs.setdefault(
        "enable_thread_floor", bool(cfg.get("enable_thread_floor", True))
    )
    hook_timeout = float(cfg.get("hook_timeout_s", 1.5))
    try:
      return route_fn(prompt, hook_timeout_s=hook_timeout, **call_kwargs)
    except TypeError:
      call_kwargs.pop("enable_thread_floor", None)
      return route_fn(prompt, **call_kwargs)
  return route_fn(prompt, **call_kwargs)


def _compute_next_thread_state(
    route: Mapping[str, Any],
    prev_thread_state: Mapping[str, Any] | None,
    *,
    cfg: Mapping[str, Any],
    turn_num: int,
) -> dict[str, Any]:
  """Computes updated asymmetric Turn 2+ thread floor state."""
  if route.get("thread_lock_applied") and prev_thread_state:
    persisted_task_class = str(
        prev_thread_state.get("locked_task_class") or route["task_class"]
    )
    persisted_tier = str(route["subagent_tier"])
    persisted_model = str(route["recommended_model_id"])
    persisted_think = int(route["thinking_level"])
  elif str(route["task_class"]) == "quick_lookup":
    persisted_task_class = "quick_lookup"
    persisted_tier = "flash_lite"
    persisted_model = str(
        (cfg.get("tier_models") or {}).get("flash_lite")
        or route["recommended_model_id"]
    )
    persisted_think = 1
  else:
    persisted_task_class = str(route["task_class"])
    persisted_tier = str(route["subagent_tier"])
    persisted_model = str(route["recommended_model_id"])
    persisted_think = int(route["thinking_level"])
  return {
      "locked_tier": persisted_tier,
      "locked_model_id": persisted_model,
      "locked_thinking_level": persisted_think,
      "locked_task_class": persisted_task_class,
      "turn_count_seen": turn_num,
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
  verbosity = str(cfg.get("verbosity") or "normal")
  is_cli = _is_agy_cli_environment(hook_input)

  prompt_hash = hashlib.sha256(
      cleaned.encode("utf-8", "ignore")
  ).hexdigest()[:16]

  seen_file = safe_state_path(sdir, conv_id, "seen_turn")
  child_guard_file = safe_state_path(sdir, prompt_hash, "subagent_child")
  child_guard_hit = _load_json_state(child_guard_file, max_age_s=60.0)
  if child_guard_hit and child_guard_hit.get("prompt_hash") == prompt_hash:
    child_guard_file.unlink(missing_ok=True)
    _save_json_state(seen_file, {"prompt_hash": prompt_hash})
    return {}
  if child_guard_file.is_file():
    child_guard_file.unlink(missing_ok=True)
  if sdir.is_dir():
    for cand_guard in sdir.glob("subagent_child_*.json"):
      cand_hit = _load_json_state(cand_guard, max_age_s=60.0)
      if cand_hit is None:
        cand_guard.unlink(missing_ok=True)
        continue
      saved_sub_prompt = str(cand_hit.get("cleaned_prompt") or "")
      saved_parent_conv = str(cand_hit.get("parent_conversation_id") or "")
      if (
          len(saved_sub_prompt) >= 16
          and saved_parent_conv
          and conv_id != saved_parent_conv
          and saved_sub_prompt in cleaned
      ):
        cand_guard.unlink(missing_ok=True)
        _save_json_state(seen_file, {"prompt_hash": prompt_hash})
        return {}

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

  seen_hit = _load_json_state(seen_file, max_age_s=600.0)
  if seen_hit and seen_hit.get("prompt_hash") == prompt_hash:
    return {}

  thread_file = safe_state_path(sdir, conv_id, "thread")
  thread_state = _load_json_state(thread_file, max_age_s=7 * 86400.0)
  if thread_state and turn_num == 1:
    turn_num = int(thread_state.get("turn_count_seen", 1)) + 1

  # Pre-routed Turn-1 handoff from `agyr` (`launch`) or `agy-router chat`:
  # skips duplicate route_fn scoring and duplicate telemetry while seeding state.
  pre_raw = os.environ.get("GEMINI_DECISIONS_PRE_ROUTED", "").strip()
  if pre_raw:
    try:
      pre = json.loads(pre_raw)
    except ValueError:
      pre = None
    if isinstance(pre, dict) and str(pre.get("prompt_hash") or "") == prompt_hash:
      pre_src = (
          pre.get("route")
          if isinstance(pre.get("route"), Mapping)
          else pre
      )
      pre_task_class = str(
          pre.get("task_class")
          or pre_src.get("task_class")
          or "coding_edit"
      )
      pre_tier = str(
          pre.get("subagent_tier")
          or pre.get("tier")
          or pre_src.get("subagent_tier")
          or pre_src.get("recommended_tier")
          or "flash"
      )
      if pre_tier not in ("flash_lite", "flash", "pro"):
        pre_tier = "flash"
      pre_model_id = str(
          pre.get("recommended_model_id")
          or pre.get("model_id")
          or pre_src.get("recommended_model_id")
          or (cfg.get("tier_models") or {}).get(pre_tier)
          or model_name
      )
      try:
        pre_think = max(
            1,
            min(
                4,
                int(
                    pre.get("thinking_level")
                    or pre_src.get("thinking_level")
                    or 2
                ),
            ),
        )
      except (TypeError, ValueError):
        pre_think = 2
      pre_think_label = str(
          pre.get("thinking_label")
          or pre_src.get("thinking_label")
          or decisions_engine.THINKING_LABELS.get(pre_think, "Medium")
      )
      try:
        pre_conf = float(
            pre.get("confidence")
            if pre.get("confidence") is not None
            else pre_src.get("confidence", 0.95)
        )
      except (TypeError, ValueError):
        pre_conf = 0.95
      try:
        pre_lat = int(
            pre.get("latency_ms")
            if pre.get("latency_ms") is not None
            else pre_src.get("latency_ms", 0)
        )
      except (TypeError, ValueError):
        pre_lat = 0
      pre_backend = str(
          pre.get("backend") or pre_src.get("backend") or "pre_routed"
      )
      pre_rationale = str(
          pre.get("rationale")
          or pre_src.get("rationale")
          or "Pre-routed by CLI launcher."
      )
      pre_explicit = bool(
          pre.get("explicit_model_requested")
          if "explicit_model_requested" in pre
          else pre_src.get("explicit_model_requested", False)
      )

      _save_json_state(seen_file, {"prompt_hash": prompt_hash})
      _save_json_state(
          safe_state_path(sdir, conv_id, "active_turn"),
          {
              "subagent_tier": pre_tier,
              "explicit_model_requested": pre_explicit,
          },
      )
      if bool(cfg.get("enable_thread_floor", True)) and (
          pre_task_class != "quick_lookup"
      ):
        _save_json_state(
            thread_file,
            {
                "locked_tier": pre_tier,
                "locked_model_id": pre_model_id,
                "locked_thinking_level": pre_think,
                "locked_task_class": pre_task_class,
                "turn_count_seen": turn_num,
            },
        )
      handoff_tok = str(pre.get("handoff_token") or "").strip()
      if handoff_tok and _SAFE_ID_RE.fullmatch(handoff_tok):
        _save_json_state(
            safe_state_path(sdir, handoff_tok, "chat_handoff"),
            {"conversation_id": conv_id},
        )

      if (
          verbosity != "silent"
          and bool(pre.get("inject_cli_badge", True))
          and bool(cfg.get("cli_inline_banner", True))
      ):
        synth_route: dict[str, Any] = {
            "recommended_model_id": pre_model_id,
            "subagent_tier": pre_tier,
            "thinking_level": pre_think,
            "thinking_label": pre_think_label,
            "task_class": pre_task_class,
            "confidence": pre_conf,
            "latency_ms": pre_lat,
            "backend": pre_backend,
            "rationale": pre_rationale,
            "probabilities": {},
            "thread_lock_applied": bool(pre.get("thread_lock_applied", False)),
            "explicit_model_requested": pre_explicit,
        }
        pre_ri = _build_render_info(
            synth_route,
            current_model=model_name,
            switched=True,
            verbosity=verbosity,
        )
        pre_md = _build_user_markdown(
            synth_route,
            current_model=model_name,
            switched=True,
            verbosity=verbosity,
            render_info=pre_ri,
        )
        pre_cli_msg = _build_cli_system_instruction(
            synth_route, current_model=model_name, switched=True
        )
        return _make_inject_steps(
            pre_ri,
            pre_md,
            verbosity=verbosity,
            cli_system_message=pre_cli_msg,
        )
      return {}

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
      and not is_cli
      and bool(target_enum)
      and target_id != model_name
      and float(route["confidence"]) >= min_conf
  )

  did_switch = False
  auto_switch_fallback: str | None = None

  if (
      mode == "auto_switch"
      and bool(target_enum)
      and target_id != model_name
      and float(route["confidence"]) >= min_conf
      and (is_cli or not _verify_cascade_ownership(conv_id))
  ):
    should_switch = False
    did_switch = False
    auto_switch_fallback = "cascade_unowned"

  render_info: dict[str, str] = {}
  user_md = ""

  if should_switch:
    render_info = _build_render_info(
        route,
        current_model=model_name,
        switched=True,
        verbosity=verbosity,
    )
    user_md = _build_user_markdown(
        route,
        current_model=model_name,
        switched=True,
        verbosity=verbosity,
        render_info=render_info,
    )
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

  host_tier = ""
  if model_name not in ("", "auto", "cli_auto"):
    host_tier, _ = decisions_engine.infer_model_tier_and_thinking(model_name)
  route_tier_rank = int(
      route.get("tier_rank")
      or decisions_engine.TIER_RANKS.get(
          str(route.get("subagent_tier") or "flash"), 2
      )
  )
  delegated_upgrade = (
      is_cli
      and not did_switch
      and mode != "off"
      and bool(cfg.get("enable_subagent_routing", True))
      and model_name not in ("", "auto", "cli_auto")
      and route_tier_rank > decisions_engine.TIER_RANKS.get(host_tier, 2)
  )

  if not did_switch:
    reroute_file.unlink(missing_ok=True)
    _save_json_state(seen_file, {"prompt_hash": prompt_hash})
    render_info = _build_render_info(
        route,
        current_model=model_name,
        switched=False,
        delegated_upgrade=delegated_upgrade,
        verbosity=verbosity,
    )
    user_md = _build_user_markdown(
        route,
        current_model=model_name,
        switched=False,
        delegated_upgrade=delegated_upgrade,
        verbosity=verbosity,
        render_info=render_info,
    )

  # Only lock substantive non-lookup work onto flash/pro; preserve locked_task_class
  # when thread_lock_applied held an earlier floor, and decay on standalone quick_lookup.
  _save_json_state(
      thread_file,
      _compute_next_thread_state(
          route, thread_state, cfg=cfg, turn_num=turn_num
      ),
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

  cli_sys_msg = (
      _build_cli_system_instruction(
          route,
          current_model=model_name,
          switched=did_switch,
          delegated_upgrade=delegated_upgrade,
          host_tier=host_tier,
      )
      if (
          is_cli
          and bool(cfg.get("cli_inline_banner", True))
          and verbosity != "silent"
      )
      else ""
  )

  telemetry_event: dict[str, Any] = {
      "ts": datetime.now(timezone.utc).isoformat(),
      "event": "PreInvocation",
      "prompt": _prompt_preview(cleaned),
      "mode": mode,
      "did_auto_switch": did_switch,
      "delegated_upgrade": delegated_upgrade,
      "thread_lock_applied": bool(route.get("thread_lock_applied")),
      "cli_banner_injected": bool(cli_sys_msg),
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

  return _make_inject_steps(
      render_info,
      user_md,
      verbosity=verbosity,
      cli_system_message=cli_sys_msg,
  )


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
  cleaned_sub = _clean_user_prompt(sub_prompt)
  if cleaned_sub:
    sub_hash = hashlib.sha256(
        cleaned_sub.encode("utf-8", "ignore")
    ).hexdigest()[:16]
    _save_json_state(
        safe_state_path(sdir, sub_hash, "subagent_child"),
        {
            "prompt_hash": sub_hash,
            "assigned_tier": tier,
            "cleaned_prompt": cleaned_sub,
            "parent_conversation_id": conv_id,
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
  catalog = _list_target_models_compat(
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
      "discovered_models_count": int(catalog.get("discovered_count", 0)),
      "suggested_tier_models": dict(
          catalog.get("suggested_tier_models") or {}
      ),
      "decisions_backend": backend_info,
      "language_server_detected": bool(ls_addr),
      "state_dir": str(default_state_dir()),
  }
  sys.stdout.write(json.dumps(out, indent=2) + "\n")
  return 0


def _cmd_list_models(ns: argparse.Namespace | None = None) -> int:
  rcfg = load_router_config()
  refresh = (
      bool(getattr(ns, "refresh", False))
      or not _default_discovered_models_path().is_file()
  )
  catalog = _list_target_models_compat(
      tier_models=rcfg.get("tier_models"),
      custom_model_enums=rcfg.get("custom_model_enums"),
      refresh_discovered=refresh,
  )
  sys.stdout.write(json.dumps(catalog, indent=2) + "\n")
  return 0


def _cmd_sync_models(ns: argparse.Namespace) -> int:
  rcfg = load_router_config()
  catalog = _list_target_models_compat(
      tier_models=rcfg.get("tier_models"),
      custom_model_enums=rcfg.get("custom_model_enums"),
      refresh_discovered=True,
  )
  suggested = dict(catalog.get("suggested_tier_models") or {})
  active_tier_models = dict(catalog.get("active_tier_models") or {})
  applied = False
  if bool(getattr(ns, "apply_recommended", False)) and suggested:
    updated_cfg = save_router_config({"tier_models": suggested})
    active_tier_models = decisions_engine.resolve_tier_models(
        updated_cfg.get("tier_models")
    )
    applied = True

  if isinstance(catalog.get("discovered_models"), list):
    discovered_models = list(catalog["discovered_models"])
  else:
    discovered_models = []
    discover_fn = getattr(decisions_engine, "discover_installed_models", None)
    if callable(discover_fn):
      try:
        cached = discover_fn(force_refresh=False, run_cli_fallback=False)
        if isinstance(cached, list):
          discovered_models = cached
      except Exception:  # pylint: disable=broad-except
        discovered_models = []
    if not discovered_models:
      discovered_models = [
          m
          for m in catalog.get("available_target_models", [])
          if str(m.get("source") or "").startswith("discovered")
      ]

  summary = {
      "status": "applied" if applied else "synced",
      "applied_recommended": applied,
      "cache_path": str(_default_discovered_models_path()),
      "discovered_count": int(
          catalog.get("discovered_count", len(discovered_models))
      ),
      "active_tier_models": active_tier_models,
      "suggested_tier_models": suggested,
      "discovered_models": discovered_models,
  }
  sys.stdout.write(json.dumps(summary, indent=2) + "\n")
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
  if getattr(ns, "enable_thread_floor", None) is not None:
    updates["enable_thread_floor"] = bool(ns.enable_thread_floor)
  if getattr(ns, "cli_inline_banner", None) is not None:
    updates["cli_inline_banner"] = bool(ns.cli_inline_banner)
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


def _format_action_and_reason(r: Mapping[str, Any]) -> tuple[str, str]:
  """Returns crystal-clear (action_label, reason_label) for watch and stats."""
  fallback = str(r.get("auto_switch_fallback") or "").strip()
  if "PreToolUse" in str(r.get("event") or ""):
    action = "ROUTE"
  elif r.get("did_auto_switch"):
    action = "SWITCH"
  elif fallback == "cascade_unowned":
    action = "CLI-ADVISE"
  elif fallback:
    action = "FALLBACK"
  else:
    action = "ADVISE"

  raw_backend = str(r.get("backend") or "?").strip()
  if raw_backend.startswith("thread_lock"):
    base_reason = "thread_lock"
  elif raw_backend.startswith("fast_rules"):
    base_reason = "fast_rules"
  elif raw_backend.startswith("lexical_fallback"):
    base_reason = "lexical_fallback"
  else:
    base_reason = raw_backend

  if r.get("thread_lock_applied") and base_reason != "thread_lock":
    reason = f"{base_reason}+thread_lock"
  else:
    reason = base_reason
  return action, reason


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
  sw, reason = _format_action_and_reason(r)
  prompt = _prompt_preview(str(r.get("prompt") or ""), max_len=72)

  if verbosity == "compact":
    return (
        f"  [{ts}] {ev} {sw} -> {target} [{tier}]"
        f" ({conf_str} | {lat} ms | reason={reason})  {prompt}"
    )

  lines = [
      f"  [{ts}] {ev:<4} {sw:<10} -> {target:<22} [{tier:<10}]"
      f" ({conf_str} | {lat:>3} ms | reason={reason})"
  ]
  if prompt:
    lines.append(f'           prompt: "{prompt}"')
  if verbosity == "verbose":
    detail = (
        f"           from={r.get('current_model', 'auto')}"
        f"  task={r.get('task_class', '?')}"
        f"  backend={r.get('backend', '?')}"
        f"  thread_lock={bool(r.get('thread_lock_applied'))}"
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
    sw, reason = _format_action_and_reason(r)
    prompt = _prompt_preview(str(r.get("prompt") or ""), max_len=72)
    lines.append(
        f"  [{ts}] {ev:<4} {sw:<10} -> {target:<22} [{tier:<10}]"
        f" {lat:>3} ms (reason={reason})"
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


_THINKING_EFFORT_FALLBACK: dict[int, str] = {
    1: "low",
    2: "medium",
    3: "high",
    4: "max",
}

_AGY_KNOWN_SUBCOMMANDS = frozenset({
    "acp",
    "agent",
    "agents",
    "changelog",
    "help",
    "install",
    "mcp",
    "mic-serve",
    "models",
    "plugin",
    "plugins",
    "remote-control",
    "update",
    "version",
})

_AGY_VALUE_FLAGS = frozenset({
    "--add-dir",
    "--agent",
    "--conversation",
    "--effort",
    "--input-format",
    "--json-schema",
    "--log-file",
    "--mode",
    "--model",
    "--output-format",
    "--print-timeout",
    "--project",
    "--settings",
})

_AGY_PRINT_FLAGS = frozenset({"-p", "--print", "--prompt"})
_AGY_INTERACTIVE_FLAGS = frozenset({"-i", "--prompt-interactive"})


def _resolve_agy_executable(agy_bin_override: str | None = None) -> str:
  if agy_bin_override and agy_bin_override.strip():
    return agy_bin_override.strip()
  env_bin = os.environ.get("AGY_BIN", "").strip()
  if env_bin:
    return env_bin
  found = shutil.which("agy")
  if found:
    return found
  fallback = pathlib.Path.home() / ".local" / "bin" / "agy"
  if fallback.is_file() and os.access(fallback, os.X_OK):
    return str(fallback)
  return "agy"


def _resolve_cli_launch_target_compat(
    route: Mapping[str, Any],
    *,
    tier_models: Mapping[str, str] | None = None,
    discovered_models: Sequence[Mapping[str, Any]] | None = None,
    discovered_cache_path: str | os.PathLike[str] | None = None,
    agy_bin: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Resolves CLI `--model` and `--effort` via `decisions_engine` with fallback."""
  engine_fn = getattr(decisions_engine, "resolve_cli_launch_target", None)
  if callable(engine_fn):
    return engine_fn(
        route,
        tier_models=tier_models,
        discovered_models=discovered_models,
        discovered_cache_path=discovered_cache_path,
        agy_bin=agy_bin,
        environ=environ,
    )

  # Standalone compatibility fallback prior to branch merge
  if discovered_models is not None:
    discovered = [m for m in discovered_models if isinstance(m, Mapping)]
  elif (
      environ is not None
      and discovered_cache_path is None
      and agy_bin is None
  ):
    discovered = []
  else:
    discover_fn = getattr(decisions_engine, "discover_installed_models", None)
    discovered = (
        discover_fn(
            agy_bin=agy_bin,
            cache_path=discovered_cache_path,
            run_cli_fallback=True,
        )
        if callable(discover_fn)
        else []
    )

  cli_only = [
      m for m in discovered if str(m.get("source") or "") == "discovered_cli"
  ]
  cli_models = cli_only if cli_only else list(discovered)
  cli_model_ids = {
      str(m.get("id") or "").strip()
      for m in cli_models
      if str(m.get("id") or "").strip()
  }

  tier = str(
      route.get("subagent_tier") or route.get("recommended_tier") or "flash"
  )
  if tier not in decisions_engine.TIER_RANKS:
    tier = "flash"
  try:
    thinking_level = max(1, min(4, int(route.get("thinking_level") or 2)))
  except (TypeError, ValueError):
    thinking_level = 2
  effort_map = getattr(
      decisions_engine, "THINKING_EFFORT_FLAGS", _THINKING_EFFORT_FALLBACK
  )
  effort = str(effort_map.get(thinking_level, "medium"))

  resolved_tiers = decisions_engine.resolve_tier_models(
      tier_models, environ=environ
  )
  original_id = str(
      route.get("recommended_model_id") or resolved_tiers[tier]
  ).strip()
  chosen_id = original_id

  if cli_models and chosen_id not in cli_model_ids:
    suggested_map = decisions_engine.suggest_tier_models(cli_models)
    suggested = str(suggested_map.get(tier) or "").strip()
    if suggested and suggested in cli_model_ids:
      chosen_id = suggested
    else:
      tier_match = next(
          (
              str(m.get("id")).strip()
              for m in cli_models
              if str(m.get("tier") or "") == tier
              and str(m.get("id") or "").strip()
          ),
          "",
      )
      chosen_id = (
          tier_match
          if tier_match
          else str(cli_models[0].get("id") or original_id).strip()
      )

  base_id = _THINKING_VARIANT_SUFFIX_RE.sub("", chosen_id)
  final_id = chosen_id
  if f"{base_id}-{effort}" in cli_model_ids:
    final_id = f"{base_id}-{effort}"
  elif effort == "max":
    for alt_suffix in ("xhigh", "high"):
      if f"{base_id}-{alt_suffix}" in cli_model_ids:
        final_id = f"{base_id}-{alt_suffix}"
        break

  return {
      "model_id": final_id,
      "effort": effort,
      "tier": tier,
      "thinking_level": thinking_level,
      "fallback_from": original_id if final_id != original_id else "",
  }


def _resolve_cli_target_for_session(
    route: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    agy_bin: str,
    discovered_models: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
  """Resolves CLI launch target while staying hermetic in isolated test dirs."""
  if discovered_models is not None:
    return _resolve_cli_launch_target_compat(
        route,
        tier_models=cfg.get("tier_models"),
        discovered_models=discovered_models,
        agy_bin=agy_bin,
    )

  explicit_state = os.environ.get("GEMINI_DECISIONS_STATE_DIR", "").strip()
  explicit_cfg = os.environ.get("GEMINI_DECISIONS_ROUTER_CONFIG", "").strip()
  explicit_cpath = (
      (pathlib.Path(explicit_state) / "discovered_models.json")
      if explicit_state
      else None
  )
  if (explicit_state or explicit_cfg) and (
      explicit_cpath is None or not explicit_cpath.is_file()
  ) and not os.environ.get("ANTIGRAVITY_LS_ADDRESS", "").strip():
    return _resolve_cli_launch_target_compat(
        route,
        tier_models=cfg.get("tier_models"),
        discovered_models=[],
    )
  return _resolve_cli_launch_target_compat(
      route,
      tier_models=cfg.get("tier_models"),
      discovered_cache_path=explicit_cpath,
      agy_bin=agy_bin,
  )


def _parse_launch_argv(raw_args: Sequence[str]) -> dict[str, Any]:
  """Parses `agyr` / `agy-router launch` arguments while preserving `agy` flags."""
  dry_run = False
  want_help = False
  agy_bin_override: str | None = None
  has_explicit_model = False
  has_explicit_effort = False
  has_explicit_agent = False
  has_continue = False
  has_conversation = False
  output_format = "text"

  prompt_flag_mode: str | None = None  # "print" | "interactive" | None
  prompt_from_flag: str | None = None
  passthrough_all: list[str] = []
  passthrough_without_prompt: list[str] = []
  positionals: list[str] = []

  idx = 0
  n = len(raw_args)
  after_ddash = False
  while idx < n:
    tok = raw_args[idx]
    if after_ddash:
      positionals.append(tok)
      passthrough_all.append(tok)
      idx += 1
      continue
    if tok == "--":
      after_ddash = True
      passthrough_all.append(tok)
      idx += 1
      continue
    if tok in ("-h", "--help"):
      want_help = True
      idx += 1
      continue
    if tok == "--dry-run":
      dry_run = True
      idx += 1
      continue
    if tok == "--agy-bin":
      if idx + 1 < n:
        agy_bin_override = raw_args[idx + 1]
        idx += 2
      else:
        idx += 1
      continue
    if tok.startswith("--agy-bin="):
      agy_bin_override = tok.split("=", 1)[1]
      idx += 1
      continue

    # Print / interactive prompt flags
    if tok in _AGY_PRINT_FLAGS:
      prompt_flag_mode = "print"
      passthrough_all.append(tok)
      if idx + 1 < n:
        prompt_from_flag = raw_args[idx + 1]
        passthrough_all.append(raw_args[idx + 1])
        idx += 2
      else:
        prompt_from_flag = ""
        idx += 1
      continue
    if any(tok.startswith(f"{pf}=") for pf in ("--print", "--prompt")):
      prompt_flag_mode = "print"
      prompt_from_flag = tok.split("=", 1)[1]
      passthrough_all.append(tok)
      idx += 1
      continue
    if tok in _AGY_INTERACTIVE_FLAGS:
      prompt_flag_mode = "interactive"
      passthrough_all.append(tok)
      if idx + 1 < n:
        prompt_from_flag = raw_args[idx + 1]
        passthrough_all.append(raw_args[idx + 1])
        idx += 2
      else:
        prompt_from_flag = ""
        idx += 1
      continue
    if tok.startswith("--prompt-interactive="):
      prompt_flag_mode = "interactive"
      prompt_from_flag = tok.split("=", 1)[1]
      passthrough_all.append(tok)
      idx += 1
      continue

    # Track explicit flags in passthrough
    if tok in ("-c", "--continue"):
      has_continue = True
      passthrough_all.append(tok)
      passthrough_without_prompt.append(tok)
      idx += 1
      continue
    if tok == "--model" or tok.startswith("--model="):
      has_explicit_model = True
    elif tok == "--effort" or tok.startswith("--effort="):
      has_explicit_effort = True
    elif tok == "--agent" or tok.startswith("--agent="):
      has_explicit_agent = True
    elif tok == "--conversation" or tok.startswith("--conversation="):
      has_conversation = True

    if tok.startswith("--output-format="):
      output_format = tok.split("=", 1)[1].strip().lower()
    elif tok == "--output-format" and idx + 1 < n:
      output_format = raw_args[idx + 1].strip().lower()

    if tok in _AGY_VALUE_FLAGS:
      passthrough_all.append(tok)
      passthrough_without_prompt.append(tok)
      if idx + 1 < n:
        passthrough_all.append(raw_args[idx + 1])
        passthrough_without_prompt.append(raw_args[idx + 1])
        idx += 2
      else:
        idx += 1
      continue

    if tok.startswith("-"):
      passthrough_all.append(tok)
      passthrough_without_prompt.append(tok)
      idx += 1
      continue

    positionals.append(tok)
    passthrough_all.append(tok)
    idx += 1

  return {
      "dry_run": dry_run,
      "want_help": want_help,
      "agy_bin_override": agy_bin_override,
      "has_explicit_model": has_explicit_model,
      "has_explicit_effort": has_explicit_effort,
      "has_explicit_agent": has_explicit_agent,
      "has_continue": has_continue,
      "has_conversation": has_conversation,
      "output_format": output_format,
      "prompt_flag_mode": prompt_flag_mode,
      "prompt_from_flag": prompt_from_flag,
      "passthrough_all": passthrough_all,
      "passthrough_without_prompt": passthrough_without_prompt,
      "positionals": positionals,
  }


def _build_pre_routed_payload(
    prompt_text: str,
    display_route: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    inject_cli_badge: bool = False,
    handoff_token: str = "",
) -> dict[str, Any]:
  cleaned = _clean_user_prompt(prompt_text)
  prompt_hash = hashlib.sha256(
      cleaned.encode("utf-8", "ignore")
  ).hexdigest()[:16]
  payload: dict[str, Any] = {
      "prompt_hash": prompt_hash,
      "task_class": display_route["task_class"],
      "subagent_tier": target["tier"],
      "recommended_model_id": target["model_id"],
      "thinking_level": target["thinking_level"],
      "thinking_label": display_route["thinking_label"],
      "confidence": display_route["confidence"],
      "latency_ms": display_route["latency_ms"],
      "backend": display_route["backend"],
      "rationale": display_route["rationale"],
      "thread_lock_applied": bool(display_route.get("thread_lock_applied")),
      "explicit_model_requested": bool(
          display_route.get("explicit_model_requested")
      ),
      "inject_cli_badge": inject_cli_badge,
  }
  if handoff_token:
    payload["handoff_token"] = handoff_token
  return payload


def build_launch_plan(
    raw_args: Sequence[str],
    *,
    config_path: pathlib.Path | None = None,
    route_fn: Callable[..., dict[str, Any]] = decisions_engine.route_prompt,
    discovered_models: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
  """Computes the `agyr` / `agy-router launch` execution plan."""
  parsed = _parse_launch_argv(raw_args)
  agy_bin = _resolve_agy_executable(parsed["agy_bin_override"])
  cfg = load_router_config(config_path)

  # 1. Explicit `--model` passed by user -> passthrough untouched
  if parsed["has_explicit_model"]:
    return {
        "action": "passthrough_explicit_model",
        "agy_cmd": [agy_bin, *parsed["passthrough_all"]],
        "route": None,
        "target": None,
        "dry_run": parsed["dry_run"],
    }

  is_subcommand = (
      parsed["prompt_flag_mode"] is None
      and bool(parsed["positionals"])
      and parsed["positionals"][0] in _AGY_KNOWN_SUBCOMMANDS
  )
  if is_subcommand:
    return {
        "action": "passthrough_subcommand",
        "agy_cmd": [agy_bin, *parsed["passthrough_all"]],
        "route": None,
        "target": None,
        "dry_run": parsed["dry_run"],
    }

  if parsed["prompt_flag_mode"] is not None:
    prompt: str | None = str(parsed["prompt_from_flag"] or "")
    is_print = parsed["prompt_flag_mode"] == "print"
  elif parsed["positionals"]:
    prompt = " ".join(parsed["positionals"])
    is_print = False
  else:
    prompt = None
    is_print = False

  # 2. `-c` / `--continue` or `--conversation` without a prompt -> passthrough
  if (parsed["has_continue"] or parsed["has_conversation"]) and not (
      prompt and prompt.strip()
  ):
    return {
        "action": "passthrough_resume",
        "agy_cmd": [agy_bin, *parsed["passthrough_all"]],
        "route": None,
        "target": None,
        "dry_run": parsed["dry_run"],
    }

  # 3. Case 1: Prompt provided -> score prompt & resolve CLI `--model` + `--effort`
  if prompt is not None and prompt.strip():
    cleaned_prompt = _clean_user_prompt(prompt) or prompt.strip()
    route = _call_route_fn(
        route_fn,
        cleaned_prompt,
        cfg=cfg,
        current_model="cli_auto",
        user_turn_count=1,
        enable_fast_rules=bool(cfg.get("enable_fast_rules", True)),
    )
    target = _resolve_cli_target_for_session(
        route,
        cfg=cfg,
        agy_bin=agy_bin,
        discovered_models=discovered_models,
    )
    think_label = decisions_engine.THINKING_LABELS.get(
        int(target["thinking_level"]),
        str(route.get("thinking_label") or "Medium"),
    )
    display_route: dict[str, Any] = {
        **route,
        "recommended_model_id": target["model_id"],
        "subagent_tier": target["tier"],
        "thinking_level": target["thinking_level"],
        "thinking_label": think_label,
    }
    cmd = [agy_bin, "--model", str(target["model_id"])]
    if not parsed["has_explicit_effort"]:
      cmd.extend(["--effort", str(target["effort"])])
    cmd.extend(parsed["passthrough_without_prompt"])
    cmd.extend(["-p" if is_print else "-i", prompt])

    pre_routed = _build_pre_routed_payload(
        cleaned_prompt,
        display_route,
        target,
        inject_cli_badge=False,
    )
    return {
        "action": "route_print" if is_print else "route_interactive",
        "agy_cmd": cmd,
        "route": display_route,
        "target": target,
        "pre_routed": pre_routed,
        "is_print": is_print,
        "output_format": parsed["output_format"],
        "cleaned_prompt": cleaned_prompt,
        "dry_run": parsed["dry_run"],
    }

  # 4. Case 2: Bare `agyr` (no prompt, no -p, no -c) -> TUI with `intent-router`
  seed_route = {
      "task_class": "quick_lookup",
      "recommended_tier": "flash_lite",
      "subagent_tier": "flash_lite",
      "recommended_model_id": str(
          (cfg.get("tier_models") or {}).get("flash_lite")
          or decisions_engine.DEFAULT_TIER_MODELS["flash_lite"]
      ),
      "thinking_level": 1,
      "thinking_label": "Low",
      "confidence": 1.0,
      "latency_ms": 0,
      "backend": "tui_intent_router",
  }
  target = _resolve_cli_target_for_session(
      seed_route,
      cfg=cfg,
      agy_bin=agy_bin,
      discovered_models=discovered_models,
  )
  cmd = [agy_bin]
  if not parsed["has_explicit_agent"]:
    cmd.extend(["--agent", "intent-router"])
  cmd.extend(["--model", str(target["model_id"])])
  if not parsed["has_explicit_effort"]:
    cmd.extend(["--effort", "low"])
  cmd.extend(parsed["passthrough_without_prompt"])
  return {
      "action": "launch_tui_intent_router",
      "agy_cmd": cmd,
      "route": None,
      "target": target,
      "dry_run": parsed["dry_run"],
  }


def _cmd_launch(raw_args: Sequence[str]) -> int:
  parsed = _parse_launch_argv(raw_args)
  if parsed["want_help"]:
    sys.stdout.write(
        "Usage: agyr [AGY_FLAGS...] [PROMPT]\n"
        "       agy-router launch [--dry-run] [--agy-bin PATH] [AGY_FLAGS...] [PROMPT]\n"
        "       agy-router run [--dry-run] [--agy-bin PATH] [AGY_FLAGS...] [PROMPT]\n\n"
        "Scores prompts via Gemini Decisions API (`gemini-decision-1`) before\n"
        "launching `agy` with the calibrated `--model` and `--effort` flags,\n"
        "or starts the interactive `intent-router` TUI agent when invoked bare.\n"
    )
    return 0

  cfg = load_router_config()
  verbosity = str(cfg.get("verbosity") or "normal")
  sdir = default_state_dir()
  plan = build_launch_plan(raw_args)

  if plan["dry_run"]:
    dry_out = {
        "action": plan["action"],
        "agy_cmd": plan["agy_cmd"],
        "route": plan["route"],
        "target": plan["target"],
    }
    if "pre_routed" in plan:
      dry_out["pre_routed"] = plan["pre_routed"]
    sys.stdout.write(json.dumps(dry_out, indent=2, ensure_ascii=False) + "\n")
    return 0

  action = str(plan["action"])
  agy_cmd = list(plan["agy_cmd"])
  agy_bin = agy_cmd[0]

  if action in ("route_print", "route_interactive"):
    route = plan["route"]
    target = plan["target"]
    cleaned_prompt = str(plan["cleaned_prompt"])
    telemetry_event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "PreInvocation",
        "prompt": _prompt_preview(cleaned_prompt),
        "mode": "cli_launcher",
        "did_auto_switch": True,
        "thread_lock_applied": bool(route.get("thread_lock_applied")),
        "cli_banner_injected": False,
        "current_model": "cli_auto",
        "recommended_model_id": target["model_id"],
        "subagent_tier": target["tier"],
        "thinking_label": route["thinking_label"],
        "task_class": route["task_class"],
        "confidence": route["confidence"],
        "latency_ms": route["latency_ms"],
        "backend": route["backend"],
    }
    if target.get("fallback_from"):
      telemetry_event["cli_model_fallback_from"] = target["fallback_from"]
    _append_telemetry(sdir, telemetry_event)

    suppress_stderr_banner = bool(plan.get("is_print")) and str(
        plan.get("output_format") or "text"
    ) in ("json", "stream-json")
    if verbosity != "silent" and not suppress_stderr_banner:
      banner = _format_intent_banner(
          route,
          current_model="cli_auto",
          switched=True,
          verbosity=verbosity,
      )
      if banner:
        sys.stderr.write(banner + "\n")
        sys.stderr.flush()

    os.environ["GEMINI_DECISIONS_PRE_ROUTED"] = json.dumps(
        plan["pre_routed"], ensure_ascii=False
    )
  elif action == "launch_tui_intent_router":
    target = plan["target"]
    if verbosity != "silent":
      sys.stderr.write(
          "✦ Routed with Intent >> agy --agent intent-router"
          f" --model {target['model_id']} [Low]"
          " (subagents auto-routed via PreToolUse)\n"
      )
      sys.stderr.flush()

  os.execvp(agy_bin, agy_cmd)


def _cmd_chat(raw_args: Sequence[str]) -> int:
  """Interactive multi-turn REPL switching `--model` & `--effort` per turn."""
  chat_parser = argparse.ArgumentParser(
      prog="agy-router chat",
      description="Multi-turn routed REPL wrapping `agy` with per-turn model switching.",
  )
  chat_parser.add_argument(
      "--conversation",
      default=None,
      help="Resume an existing `agy` conversation ID.",
  )
  chat_parser.add_argument(
      "--agy-bin",
      default=None,
      help="Path to the `agy` executable.",
  )
  chat_parser.add_argument(
      "--verbosity",
      choices=list(VALID_VERBOSITY_LEVELS),
      default=None,
      help="Override routing banner verbosity for this session.",
  )
  ns, passthrough_flags = chat_parser.parse_known_args(list(raw_args))

  cfg = load_router_config()
  verbosity = ns.verbosity or str(cfg.get("verbosity") or "normal")
  sdir = default_state_dir()
  agy_bin = _resolve_agy_executable(ns.agy_bin)

  explicit_conv = str(ns.conversation or "").strip()
  conv_id = explicit_conv or str(uuid.uuid4())
  resolved_agy_conv_id = explicit_conv
  session_token = uuid.uuid4().hex[:24]
  thread_state: dict[str, Any] | None = (
      _load_json_state(
          safe_state_path(sdir, explicit_conv, "thread"),
          max_age_s=7 * 86400.0,
      )
      if explicit_conv
      else None
  )
  turn_num = (
      int(thread_state.get("turn_count_seen", 0))
      if isinstance(thread_state, dict)
      else 0
  )
  prev_model = "cli_auto"
  is_tty = sys.stdin.isatty()

  if is_tty and verbosity != "silent":
    sys.stderr.write(
        "✦ Routed with Intent | Multi-Turn CLI Chat"
        " (/new to reset thread, /exit to quit)\n"
    )
    sys.stderr.flush()

  while True:
    if is_tty and verbosity != "silent":
      sys.stderr.write(f"\nagyr[{turn_num + 1}]> ")
      sys.stderr.flush()
    try:
      raw_line = sys.stdin.readline()
    except KeyboardInterrupt:
      if is_tty:
        sys.stderr.write("\n")
      break
    if raw_line == "":
      break
    line = raw_line.strip()
    if not line:
      continue
    low_cmd = line.lower()
    if low_cmd in ("/exit", "/quit"):
      break
    if low_cmd in ("/new", "/reset"):
      conv_id = str(uuid.uuid4())
      resolved_agy_conv_id = ""
      session_token = uuid.uuid4().hex[:24]
      thread_state = None
      turn_num = 0
      prev_model = "cli_auto"
      if verbosity != "silent":
        sys.stderr.write("✦ Started fresh conversation (thread floor reset).\n")
        sys.stderr.flush()
      continue

    turn_num += 1
    cleaned_line = _clean_user_prompt(line) or line
    route = _call_route_fn(
        decisions_engine.route_prompt,
        cleaned_line,
        cfg=cfg,
        current_model=prev_model,
        user_turn_count=turn_num,
        thread_state=thread_state,
        enable_fast_rules=bool(cfg.get("enable_fast_rules", True)),
    )
    target = _resolve_cli_target_for_session(
        route,
        cfg=cfg,
        agy_bin=agy_bin,
    )
    think_label = decisions_engine.THINKING_LABELS.get(
        int(target["thinking_level"]),
        str(route.get("thinking_label") or "Medium"),
    )
    display_route: dict[str, Any] = {
        **route,
        "recommended_model_id": target["model_id"],
        "subagent_tier": target["tier"],
        "thinking_level": target["thinking_level"],
        "thinking_label": think_label,
    }

    telemetry_event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "PreInvocation",
        "conversation_id": resolved_agy_conv_id or conv_id,
        "prompt": _prompt_preview(cleaned_line),
        "mode": "cli_chat",
        "did_auto_switch": True,
        "thread_lock_applied": bool(display_route.get("thread_lock_applied")),
        "cli_banner_injected": False,
        "current_model": prev_model,
        "recommended_model_id": target["model_id"],
        "subagent_tier": target["tier"],
        "thinking_label": think_label,
        "task_class": display_route["task_class"],
        "confidence": display_route["confidence"],
        "latency_ms": display_route["latency_ms"],
        "backend": display_route["backend"],
    }
    if target.get("fallback_from"):
      telemetry_event["cli_model_fallback_from"] = target["fallback_from"]
    _append_telemetry(sdir, telemetry_event)

    if verbosity != "silent":
      banner = _format_intent_banner(
          display_route,
          current_model=prev_model,
          switched=True,
          verbosity=verbosity,
      )
      if banner:
        sys.stderr.write(banner + "\n")
        sys.stderr.flush()

    if bool(cfg.get("enable_thread_floor", True)):
      thread_state = _compute_next_thread_state(
          display_route, thread_state, cfg=cfg, turn_num=turn_num
      )
    else:
      thread_state = None

    handoff_file = safe_state_path(sdir, session_token, "chat_handoff")
    handoff_file.unlink(missing_ok=True)
    pre_routed = _build_pre_routed_payload(
        cleaned_line,
        display_route,
        target,
        inject_cli_badge=False,
        handoff_token=session_token,
    )
    child_env = dict(os.environ)
    child_env["GEMINI_DECISIONS_PRE_ROUTED"] = json.dumps(
        pre_routed, ensure_ascii=False
    )

    # Note on `agy` CLI conversation creation vs resumption:
    # Passing `--conversation <unknown-uuid>` on Turn 1 prints a warning in `agy`
    # ("Conversation <id> not found, ignoring --conversation flag") and creates
    # a server-generated UUID. Therefore:
    # - When an existing conversation ID is known (`--conversation ID` or
    #   captured from Turn 1's PreInvocation hook via `chat_handoff`), pass
    #   `["--conversation", resolved_agy_conv_id]`.
    # - On Turn 1 of a fresh chat without `--conversation`, omit `--conversation`
    #   so `agy` creates the conversation cleanly.
    # - On Turn 2+ if no `chat_handoff` ID was written (e.g. hooks disabled or
    #   standalone mock `agy` binary), pass `["--continue"]`.
    cmd: list[str] = [agy_bin]
    if resolved_agy_conv_id:
      cmd.extend(["--conversation", resolved_agy_conv_id])
    elif turn_num > 1:
      cmd.append("--continue")
    cmd.extend([
        "--model",
        str(target["model_id"]),
        "--effort",
        str(target["effort"]),
        *passthrough_flags,
        "-p",
        line,
    ])
    subprocess.run(
        cmd,
        env=child_env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    prev_model = str(target["model_id"])

    if not resolved_agy_conv_id:
      handoff_data = _load_json_state(handoff_file, max_age_s=600.0)
      captured = str((handoff_data or {}).get("conversation_id") or "").strip()
      if captured and captured != "default":
        resolved_agy_conv_id = captured
        conv_id = captured
    handoff_file.unlink(missing_ok=True)

    if thread_state is not None:
      _save_json_state(
          safe_state_path(sdir, resolved_agy_conv_id or conv_id, "thread"),
          thread_state,
      )

  return 0


def main(argv: Sequence[str] | None = None) -> int:
  raw_argv = list(sys.argv[1:] if argv is None else argv)
  if raw_argv and raw_argv[0] in ("launch", "run"):
    return _cmd_launch(raw_argv[1:])
  if raw_argv and raw_argv[0] == "chat":
    return _cmd_chat(raw_argv[1:])

  parser = argparse.ArgumentParser(
      description="Gemini Decisions Router hooks and CLI for Antigravity"
  )
  sub = parser.add_subparsers(dest="cmd")
  sub.add_parser("pre-invocation")
  sub.add_parser("pre-tool-use")
  sub.add_parser(
      "launch",
      help="Score prompt via Gemini Decisions API and exec `agy` with calibrated --model and --effort.",
  )
  sub.add_parser(
      "run",
      help="Alias for `launch`.",
  )
  sub.add_parser(
      "chat",
      help="Interactive multi-turn REPL switching `--model` and `--effort` across turns.",
  )

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

  p_list_models = sub.add_parser("list-models")
  p_list_models.add_argument(
      "--refresh",
      action="store_true",
      help="Force live re-discovery of models from Antigravity LS and agy CLI.",
  )

  p_sync = sub.add_parser("sync-models")
  p_sync.add_argument(
      "--apply-recommended",
      action="store_true",
      help="Save suggested_tier_models into router_config.json.",
  )

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
  p_mode.add_argument(
      "--enable-thread-floor",
      dest="enable_thread_floor",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Enable or disable asymmetric Turn 2+ thread floor.",
  )
  p_mode.add_argument(
      "--cli-inline-banner",
      dest="cli_inline_banner",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Enable or disable 1-line inline routing badge inside agy CLI.",
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

  ns = parser.parse_args(raw_argv)
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
    return _cmd_list_models(ns)
  if ns.cmd == "sync-models":
    return _cmd_sync_models(ns)
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
