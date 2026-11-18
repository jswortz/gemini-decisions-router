#!/usr/bin/env python3
"""Stdio JSON-RPC 2.0 MCP server for the Gemini Decisions Router plugin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import sys
import traceback
from typing import Any, BinaryIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions_engine  # noqa: E402

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
        "List all available target models supported by the router, their"
        " Antigravity model enums, default thinking levels, and the currently"
        " configured tier_models mapping (`flash_lite`, `flash`, `pro`)."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": False},
}


def _load_router_model_config() -> tuple[dict[str, str], dict[str, str]]:
  cfg_path = (
      os.path.expanduser("~/.config/gemini-decisions-router/router_config.json")
  )
  tier_models: dict[str, str] = {}
  custom_enums: dict[str, str] = {}
  try:
    with open(cfg_path, encoding="utf-8") as f:
      loaded = json.load(f)
    if isinstance(loaded, dict):
      if isinstance(loaded.get("tier_models"), dict):
        tier_models = {
            str(k): str(v) for k, v in loaded["tier_models"].items() if v
        }
      if isinstance(loaded.get("custom_model_enums"), dict):
        custom_enums = {
            str(k): str(v)
            for k, v in loaded["custom_model_enums"].items()
            if v
        }
  except (OSError, ValueError):
    pass
  return tier_models, custom_enums


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
  ) -> None:
    self._settings = settings
    self._create_fn = create_fn
    self._route_fn = route_fn

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
          {"tools": [_CREATE_DECISION_TOOL, _ROUTE_PROMPT_TOOL, _LIST_MODELS_TOOL]},
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
        tier_models, _ = _load_router_model_config()
        route_kwargs: dict[str, Any] = {
            "current_model": str(arguments.get("current_model") or "auto"),
            "user_turn_count": max(1, int(arguments.get("user_turn_count") or 1)),
            "prev_context": str(arguments.get("prev_context") or ""),
            "enable_fast_rules": bool(arguments.get("enable_fast_rules", True)),
            "settings": self._settings,
        }
        if self._route_fn is decisions_engine.route_prompt:
          route_kwargs["tier_models"] = tier_models
        res = self._route_fn(prompt, **route_kwargs)
      elif name == "list_models":
        tier_models, custom_enums = _load_router_model_config()
        res = decisions_engine.list_available_target_models(
            tier_models=tier_models,
            custom_model_enums=custom_enums,
        )
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
              "message": "Internal error in gemini-decisions-router MCP server.",
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
