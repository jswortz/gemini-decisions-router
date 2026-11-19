#!/usr/bin/env python3
"""Stdio JSON-RPC 2.0 MCP server for the Gemini Decisions Router plugin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import pathlib
import sys
from typing import Any, BinaryIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions_engine  # noqa: E402
import router_hook  # noqa: E402

MAX_LINE_BYTES = 16 * 1024 * 1024
_JSONRPC_VERSION = "2.0"
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602

_SERVER_NAME = "gemini-decisions-router"
_DEFAULT_PROTOCOL_VERSION = "2025-11-25"
_SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

_CREATE_DECISION_TOOL: dict[str, Any] = {
    "name": "create_decision",
    "description": (
        "Ask the Gemini Decisions API calibrated structured questions in one"
        " prefill pass with 0 output tokens. Supports `noul` (yes/no"
        " probability in [0.01, 0.99]), `choice` (2-255 discrete options), and"
        " `score` (2-255 ordered rubric levels). Batch all questions about the"
        " same input into one call."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "questions"],
        "properties": {
            "input": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "description": (
                    "Shared input evaluated once across all questions. Each"
                    " item has either `text` or `file`."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "file": {"type": "string"},
                    },
                },
            },
            "questions": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 64,
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "instructions"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["noul", "choice", "score"],
                        },
                        "instructions": {"type": "string"},
                        "criteria": {},
                    },
                },
            },
        },
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": True},
}

_ROUTE_PROMPT_TOOL: dict[str, Any] = {
    "name": "route_prompt",
    "description": (
        "Classify a user prompt or subagent brief using Tier 0 (<1 ms) fast"
        " rules and Tier 1 Gemini Decisions API scoring. Returns recommended"
        " Antigravity model tier (`flash_lite`, `flash`, `pro`), model ID,"
        " thinking effort level, confidence, and calibrated probabilities."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The user prompt or subagent task description.",
            },
            "current_model": {
                "type": "string",
                "description": "Active model ID (default: auto).",
            },
            "user_turn_count": {
                "type": "integer",
                "minimum": 1,
                "description": "Explicit user turn index in the thread.",
            },
            "prev_context": {
                "type": "string",
                "description": "Optional brief context from prior turn.",
            },
            "enable_fast_rules": {
                "type": "boolean",
                "description": (
                    "Use <1 ms Tier 0 fast rules before calling Decisions API"
                    " (default: true)."
                ),
            },
        },
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": True},
}

_LIST_MODELS_TOOL: dict[str, Any] = {
    "name": "list_models",
    "description": (
        "List all available built-in, dynamically auto-discovered, and custom"
        " target models supported by the router, their Antigravity model enums,"
        " default thinking levels, suggested_tier_models, and the currently"
        " configured tier_models mapping (`flash_lite`, `flash`, `pro`)."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "refresh": {
                "type": "boolean",
                "description": (
                    "Force live re-discovery of newly available models from"
                    " Antigravity LS and agy CLI (default: false)."
                ),
            },
        },
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": False},
}

_CONFIGURE_ROUTER_TOOL: dict[str, Any] = {
    "name": "configure_router",
    "description": (
        "Update or inspect Gemini Decisions Router settings (`mode`,"
        " `verbosity`, `min_confidence`, `enable_subagent_routing`, and"
        " `tier_models` for `flash_lite`, `flash`, `pro`) in"
        " `~/.config/gemini-decisions-router/router_config.json`."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["advisor", "auto_switch", "off"],
                "description": "Routing mode (`advisor`, `auto_switch`, `off`).",
            },
            "verbosity": {
                "type": "string",
                "enum": ["silent", "compact", "normal", "verbose"],
                "description": (
                    "Banner and subagent notice verbosity (`silent`, `compact`,"
                    " `normal`, `verbose`)."
                ),
            },
            "min_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Minimum confidence in [0.0, 1.0] required for auto_switch."
                ),
            },
            "enable_subagent_routing": {
                "type": "boolean",
                "description": (
                    "Enable PreToolUse tier routing on invoke_subagent calls."
                ),
            },
            "enable_thread_floor": {
                "type": "boolean",
                "description": (
                    "Enable asymmetric Turn 2+ thread floor for coding/pro"
                    " threads (default: true)."
                ),
            },
            "cli_inline_banner": {
                "type": "boolean",
                "description": (
                    "Render a concise 1-line routing badge inside agy CLI"
                    " responses when verbosity != silent (default: true)."
                ),
            },
            "tier_models": {
                "type": "object",
                "additionalProperties": False,
                "description": (
                    "Optional target model IDs per tier (`flash_lite`,"
                    " `flash`, `pro`)."
                ),
                "properties": {
                    "flash_lite": {"type": "string"},
                    "flash": {"type": "string"},
                    "pro": {"type": "string"},
                },
            },
        },
    },
    "annotations": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}

_ALLOWED_CONFIGURE_KEYS = frozenset({
    "mode",
    "verbosity",
    "min_confidence",
    "enable_subagent_routing",
    "enable_thread_floor",
    "cli_inline_banner",
    "tier_models",
})


def _load_router_model_config(
    config_path: pathlib.Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
  cfg = router_hook.load_router_config(config_path)
  tier_models = {
      str(k): str(v)
      for k, v in (cfg.get("tier_models") or {}).items()
      if v
  }
  custom_enums = {
      str(k): str(v)
      for k, v in (cfg.get("custom_model_enums") or {}).items()
      if v
  }
  return tier_models, custom_enums


def _validate_configure_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
  unknown = set(arguments.keys()) - _ALLOWED_CONFIGURE_KEYS
  if unknown:
    raise decisions_engine.InvalidInputError(
        f"Unsupported configure_router keys: {sorted(unknown)}"
    )
  updates: dict[str, Any] = {}
  if "mode" in arguments:
    mode = arguments["mode"]
    if mode not in ("advisor", "auto_switch", "off"):
      raise decisions_engine.InvalidInputError(
          "mode must be one of: advisor, auto_switch, off"
      )
    updates["mode"] = mode

  if "verbosity" in arguments:
    verbosity = arguments["verbosity"]
    if verbosity not in router_hook.VALID_VERBOSITY_LEVELS:
      raise decisions_engine.InvalidInputError(
          "verbosity must be one of: silent, compact, normal, verbose"
      )
    updates["verbosity"] = verbosity

  if "min_confidence" in arguments:
    mc = arguments["min_confidence"]
    if isinstance(mc, bool) or not isinstance(mc, (int, float)):
      raise decisions_engine.InvalidInputError(
          "min_confidence must be a number in [0.0, 1.0]"
      )
    mc_f = float(mc)
    if mc_f < 0.0 or mc_f > 1.0:
      raise decisions_engine.InvalidInputError(
          "min_confidence must be between 0.0 and 1.0"
      )
    updates["min_confidence"] = mc_f

  if "enable_subagent_routing" in arguments:
    esr = arguments["enable_subagent_routing"]
    if not isinstance(esr, bool):
      raise decisions_engine.InvalidInputError(
          "enable_subagent_routing must be a boolean"
      )
    updates["enable_subagent_routing"] = esr

  for bool_key in ("enable_thread_floor", "cli_inline_banner"):
    if bool_key in arguments:
      bval = arguments[bool_key]
      if not isinstance(bval, bool):
        raise decisions_engine.InvalidInputError(
            f"{bool_key} must be a boolean"
        )
      updates[bool_key] = bval

  if "tier_models" in arguments:
    tm = arguments["tier_models"]
    if not isinstance(tm, Mapping):
      raise decisions_engine.InvalidInputError(
          "tier_models must be an object mapping flash_lite/flash/pro to model IDs"
      )
    bad_tiers = set(tm.keys()) - {"flash_lite", "flash", "pro"}
    if bad_tiers:
      raise decisions_engine.InvalidInputError(
          f"Unsupported tier_models keys: {sorted(bad_tiers)}"
      )
    cleaned_tm: dict[str, str] = {}
    for tk, tv in tm.items():
      if not isinstance(tv, str) or not tv.strip():
        raise decisions_engine.InvalidInputError(
            f"tier_models[{tk!r}] must be a non-empty string"
        )
      cleaned_tm[str(tk)] = tv.strip()
    if cleaned_tm:
      updates["tier_models"] = cleaned_tm

  return updates


def _reject_json_constant(token: str) -> Any:
  raise ValueError(f"Non-standard JSON constant rejected: {token}")


def _log_stderr(msg: str) -> None:
  for env_name in ("GEMINI_API_KEY", "GOOGLE_OAUTH_ACCESS_TOKEN"):
    secret = os.environ.get(env_name, "")
    if secret and secret in msg:
      msg = msg.replace(secret, "[REDACTED]")
  try:
    sys.stderr.write(f"gemini-decisions-router: {msg}\n")
    sys.stderr.flush()
  except OSError:
    pass


def _write_all(stdout: BinaryIO, data: bytes) -> None:
  view = memoryview(data)
  sent = 0
  while sent < len(view):
    n = stdout.write(view[sent:])
    if n is None:
      raise BlockingIOError("stdout write returned None")
    sent += n
  stdout.flush()


class McpServer:
  """Newline-delimited JSON-RPC 2.0 MCP stdio server."""

  def __init__(
      self,
      settings: decisions_engine.Settings | None = None,
      create_fn: Callable[..., dict[str, Any]] = decisions_engine.create_decision,
      route_fn: Callable[..., dict[str, Any]] = decisions_engine.route_prompt,
      config_path: os.PathLike[str] | str | None = None,
  ) -> None:
    self._settings = settings
    self._create_fn = create_fn
    self._route_fn = route_fn
    self._config_path = pathlib.Path(config_path) if config_path else None

  def _encode(self, payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8", "replace"
        )
        + b"\n"
    )

  def _ok(self, req_id: str | int, result: dict[str, Any]) -> bytes:
    return self._encode(
        {"jsonrpc": _JSONRPC_VERSION, "id": req_id, "result": result}
    )

  def _err(self, req_id: str | int, code: int, message: str) -> bytes:
    return self._encode({
        "jsonrpc": _JSONRPC_VERSION,
        "id": req_id,
        "error": {"code": code, "message": message},
    })

  def handle_line(self, line: bytes) -> bytes | None:
    if len(line) > MAX_LINE_BYTES:
      _log_stderr("ignored oversized JSON-RPC input line")
      return None

    try:
      msg = json.loads(
          line.decode("utf-8"), parse_constant=_reject_json_constant
      )
    except (UnicodeDecodeError, ValueError, RecursionError):
      _log_stderr("ignored invalid JSON input line")
      return None

    if not isinstance(msg, dict) or "id" not in msg:
      return None

    req_id = msg["id"]
    if isinstance(req_id, bool) or not isinstance(req_id, (str, int)):
      return None

    if msg.get("jsonrpc") != _JSONRPC_VERSION:
      return self._err(req_id, _INVALID_REQUEST, "jsonrpc must be '2.0'")

    method = msg.get("method")
    params = msg.get("params") or {}
    if not isinstance(method, str) or not isinstance(params, Mapping):
      return self._err(req_id, _INVALID_REQUEST, "invalid method or params")

    if method == "initialize":
      requested_v = str(params.get("protocolVersion") or "")
      proto_v = (
          requested_v
          if requested_v in _SUPPORTED_PROTOCOL_VERSIONS
          else _DEFAULT_PROTOCOL_VERSION
      )
      return self._ok(
          req_id,
          {
              "protocolVersion": proto_v,
              "capabilities": {"tools": {"listChanged": False}},
              "serverInfo": {
                  "name": _SERVER_NAME,
                  "version": decisions_engine.__version__,
              },
          },
      )

    if method == "ping":
      return self._ok(req_id, {})

    if method == "tools/list":
      return self._ok(
          req_id,
          {
              "tools": [
                  _CREATE_DECISION_TOOL,
                  _ROUTE_PROMPT_TOOL,
                  _LIST_MODELS_TOOL,
                  _CONFIGURE_ROUTER_TOOL,
              ]
          },
      )

    if method == "tools/call":
      tname = params.get("name")
      targs = params.get("arguments") or {}
      if not isinstance(tname, str) or not isinstance(targs, Mapping):
        return self._err(req_id, _INVALID_PARAMS, "invalid tool call params")
      return self._handle_tool_call(req_id, tname, targs)

    return self._err(req_id, _METHOD_NOT_FOUND, f"unknown method: {method}")

  def _handle_tool_call(
      self, req_id: str | int, name: str, arguments: Mapping[str, Any]
  ) -> bytes:
    try:
      if name == "create_decision":
        cfg = self._settings or decisions_engine.load_settings()
        res = self._create_fn(arguments, cfg)
      elif name == "route_prompt":
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
          raise decisions_engine.InvalidInputError(
              "prompt must be a non-empty string"
          )
        tier_models, _ = _load_router_model_config(self._config_path)
        route_kwargs: dict[str, Any] = {
            "current_model": str(arguments.get("current_model") or "auto"),
            "user_turn_count": max(
                1, int(arguments.get("user_turn_count") or 1)
            ),
            "prev_context": str(arguments.get("prev_context") or ""),
            "enable_fast_rules": bool(arguments.get("enable_fast_rules", True)),
            "settings": self._settings,
        }
        if self._route_fn is decisions_engine.route_prompt:
          route_kwargs["tier_models"] = tier_models
        res = self._route_fn(prompt, **route_kwargs)
      elif name == "list_models":
        unknown_lm = set(arguments.keys()) - {"refresh"}
        if unknown_lm:
          raise decisions_engine.InvalidInputError(
              f"Unsupported list_models keys: {sorted(unknown_lm)}"
          )
        if "refresh" in arguments and not isinstance(
            arguments["refresh"], bool
        ):
          raise decisions_engine.InvalidInputError(
              "refresh must be a boolean"
          )
        refresh_flag = bool(arguments.get("refresh", False))
        tier_models, custom_enums = _load_router_model_config(self._config_path)
        try:
          res = decisions_engine.list_available_target_models(
              tier_models=tier_models,
              custom_model_enums=custom_enums,
              refresh_discovered=refresh_flag,
          )
        except TypeError:
          res = decisions_engine.list_available_target_models(
              tier_models=tier_models,
              custom_model_enums=custom_enums,
          )
      elif name == "configure_router":
        updates = _validate_configure_args(arguments)
        updated_cfg = (
            router_hook.save_router_config(updates, config_path=self._config_path)
            if updates
            else router_hook.load_router_config(self._config_path)
        )
        catalog = decisions_engine.list_available_target_models(
            tier_models=updated_cfg.get("tier_models"),
            custom_model_enums=updated_cfg.get("custom_model_enums"),
        )
        res = {
            "status": "updated" if updates else "unchanged",
            "router_config": updated_cfg,
            "active_tier_models": catalog["active_tier_models"],
        }
      else:
        return self._err(req_id, _INVALID_PARAMS, f"unknown tool: {name}")

      return self._ok(
          req_id,
          {
              "content": [{
                  "type": "text",
                  "text": json.dumps(res, indent=2, ensure_ascii=False),
              }],
              "structuredContent": res,
              "isError": False,
          },
      )
    except decisions_engine.DecisionError as e:
      err_payload = {"error": e.to_dict()}
      return self._ok(
          req_id,
          {
              "content": [{
                  "type": "text",
                  "text": json.dumps(err_payload, ensure_ascii=False),
              }],
              "structuredContent": err_payload,
              "isError": True,
          },
      )
    except Exception:  # pylint: disable=broad-except
      _log_stderr("internal error while handling MCP tool call")
      err_payload = {
          "error": {
              "code": "INTERNAL",
              "message": (
                  "Internal error in gemini-decisions-router MCP server."
              ),
              "retryable": False,
          }
      }
      return self._ok(
          req_id,
          {
              "content": [{"type": "text", "text": json.dumps(err_payload)}],
              "structuredContent": err_payload,
              "isError": True,
          },
      )


def main() -> int:
  server = McpServer()
  stdin_b = sys.stdin.buffer
  stdout_b = sys.stdout.buffer
  while True:
    line = stdin_b.readline(MAX_LINE_BYTES + 1)
    if not line:
      break
    resp = server.handle_line(line.rstrip(b"\r\n"))
    if resp is not None:
      _write_all(stdout_b, resp)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
