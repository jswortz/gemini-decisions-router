"""TDD unit, ADC authentication, security, and external-hygiene tests."""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import threading
from typing import Any
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import decisions_engine  # noqa: E402
import mcp_server  # noqa: E402
import router_hook  # noqa: E402


# ---- 1. External Hygiene Scan (Zero Internal Google APIs or References) -----


def test_zero_internal_google_references() -> None:
  """Ensures no internal APIs, endpoints, CLIs, or shortlinks appear in repo."""
  forbidden_tokens = [
      "gemini-4-" + "carbon",
      "gemini-4-" + "hydrogen",
      "gemini-4-" + "argon",
      "gemini-4-" + "neon",
      "gemini-" + "3.8",
      "gemini-" + "3.7",
      "gemini-" + "3.6",
      "hydrogen-" + "l-tf",
      "//" + "depot/",
      "be" + "yond",
      "stu" + "bby",
      "bl" + "ade:",
      "v4" + "xs",
      "goo" + "gle3",
      "jet" + "ski",
      "gc" + "ert",
      "lo" + "as2",
      "geel" + "inux",
      ".corp." + "google",
      "/usr/local/" + "google",
      "wortz-" + "project",
      "down" + "shifter",
      "fa" + "ble",
      "ba" + "ki",
      "ken" + "limmj",
      "AIza" + "Sy",
      "ya" + "29.",
      "http://" + "go/",
      "http://" + "cl/",
      "http://" + "b/",
  ]
  pattern = re.compile(
      "|".join(re.escape(tok) for tok in forbidden_tokens), re.IGNORECASE
  )

  checked_files = 0
  for path in REPO_ROOT.rglob("*"):
    if not path.is_file():
      continue
    parts = set(path.relative_to(REPO_ROOT).parts)
    if parts & {".git", "__pycache__", ".pytest_cache"}:
      continue
    if path.suffix in {".pyc", ".png", ".jpg", ".ico"}:
      continue
    text = path.read_text(encoding="utf-8", errors="replace")
    checked_files += 1
    match = pattern.search(text)
    assert match is None, (
        f"Forbidden internal reference {match.group(0)!r} found in"
        f" {path.relative_to(REPO_ROOT)}"
    )
  assert checked_files >= 8


# ---- 2. Settings, URL Construction, and Secret Hygiene ----------------------


def test_config_rejects_embedded_secrets(tmp_path: pathlib.Path) -> None:
  cfg_file = tmp_path / "config.json"
  cfg_file.write_text(
      json.dumps({"auth_mode": "api_key", "api_key": "mock_embedded_key_val"}),
      encoding="utf-8",
  )
  with pytest.raises(decisions_engine.ConfigError, match="api_key"):
    decisions_engine.load_settings(config_path=str(cfg_file), environ={})


def test_vertex_global_and_regional_urls() -> None:
  s_global = decisions_engine.Settings(
      auth_mode="vertex_adc",
      model="gemini-decision-1",
      project_id="acme-prod-123",
      location="global",
      api_base="https://generativelanguage.googleapis.com/v1beta",
      vertex_api_version="v1beta1",
      vertex_base_override="",
      timeout_s=10.0,
      allowed_dirs=(),
  )
  assert decisions_engine.build_decisions_url(s_global) == (
      "https://aiplatform.googleapis.com/v1beta1/projects/acme-prod-123"
      "/locations/global/decisions"
  )

  s_regional = decisions_engine.Settings(
      auth_mode="vertex_adc",
      model="gemini-decision-1",
      project_id="acme-prod-123",
      location="us-central1",
      api_base="https://generativelanguage.googleapis.com/v1beta",
      vertex_api_version="v1beta1",
      vertex_base_override="",
      timeout_s=10.0,
      allowed_dirs=(),
  )
  assert decisions_engine.build_decisions_url(s_regional) == (
      "https://us-central1-aiplatform.googleapis.com/v1beta1/projects"
      "/acme-prod-123/locations/us-central1/decisions"
  )


def test_auto_auth_mode_prefers_adc_when_no_api_key(
    tmp_path: pathlib.Path,
) -> None:
  s_adc = decisions_engine.load_settings(
      config_path=str(tmp_path / "none.json"),
      environ={"GOOGLE_CLOUD_PROJECT": "customer-proj-01"},
  )
  assert s_adc.auth_mode == "gemini_adc"
  assert s_adc.project_id == "customer-proj-01"

  s_vertex = decisions_engine.load_settings(
      config_path=str(tmp_path / "none.json"),
      environ={
          "GOOGLE_GENAI_USE_VERTEXAI": "true",
          "GOOGLE_CLOUD_PROJECT": "customer-proj-01",
          "GOOGLE_CLOUD_LOCATION": "us-central1",
      },
  )
  assert s_vertex.auth_mode == "vertex_adc"
  assert s_vertex.location == "us-central1"

  s_key = decisions_engine.load_settings(
      config_path=str(tmp_path / "none.json"),
      environ={"GEMINI_API_KEY": "mock_api_key_12345"},
  )
  assert s_key.auth_mode == "api_key"


# ---- 3. Mock HTTP Server for Gemini ADC, Vertex ADC, and API Key ------------


class _MockCloudDecisionsHandler(http.server.BaseHTTPRequestHandler):
  scenario = "ok"
  last_path = ""
  last_headers: dict[str, str] = {}
  last_body: dict[str, Any] = {}

  def log_message(self, fmt: str, *args: Any) -> None:
    del fmt, args

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(length)
    type(self).last_path = self.path
    type(self).last_headers = {
        k.lower(): v for k, v in self.headers.items()
    }
    type(self).last_body = json.loads(raw.decode("utf-8"))

    if self.scenario == "redirect":
      self.send_response(302)
      self.send_header("Location", "http://127.0.0.1:1/exfil")
      self.end_headers()
      return

    if self.scenario == "scope_insufficient":
      err = {
          "error": {
              "code": 403,
              "message": "Request had insufficient authentication scopes.",
              "status": "PERMISSION_DENIED",
              "details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}],
          }
      }
      data = json.dumps(err).encode("utf-8")
      self.send_response(403)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
      return

    if self.scenario == "vertex_not_allowlisted":
      err = {
          "error": {
              "code": 404,
              "message": "Method CreateDecisionHttp not found for service aiplatform.googleapis.com.",
              "status": "NOT_FOUND",
          }
      }
      data = json.dumps(err).encode("utf-8")
      self.send_response(404)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
      return

    resp = {
        "answers": {
            "is_arch": {"type": "noul", "noul": 0.91, "confidence": 0.93},
        },
        "usage": {"total_input_tokens": 28},
    }
    data = json.dumps(resp).encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def test_gemini_adc_vertex_adc_and_scope_diagnostics() -> None:
  server = http.server.HTTPServer(("127.0.0.1", 0), _MockCloudDecisionsHandler)
  port = server.server_port
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()

  fake_token = decisions_engine.AdcToken(
      access_token="mock_oauth_bearer_98765",
      project_id="customer-gcp-proj",
      source="test_mock",
  )
  call_args = {
      "input": [{"text": "Design a multi-region Cloud Spanner architecture."}],
      "questions": {
          "is_arch": {
              "type": "noul",
              "instructions": "Does this require distributed architecture?",
          }
      },
  }

  try:
    # 1. Keyless Gemini API via ADC (`Authorization: Bearer` + `x-goog-user-project`)
    _MockCloudDecisionsHandler.scenario = "ok"
    s_gem_adc = decisions_engine.Settings(
        auth_mode="gemini_adc",
        model="gemini-decision-1",
        project_id="customer-gcp-proj",
        location="global",
        api_base=f"http://127.0.0.1:{port}/v1beta",
        vertex_api_version="v1beta1",
        vertex_base_override="",
        timeout_s=5.0,
        allowed_dirs=(),
    )
    res_adc = decisions_engine.create_decision(
        call_args, s_gem_adc, token_provider=lambda _scopes: fake_token
    )
    assert res_adc["backend"] == "gemini_adc"
    assert res_adc["answers"]["is_arch"]["value"] is True
    assert (
        _MockCloudDecisionsHandler.last_headers["authorization"]
        == "Bearer mock_oauth_bearer_98765"
    )
    assert (
        _MockCloudDecisionsHandler.last_headers["x-goog-user-project"]
        == "customer-gcp-proj"
    )
    assert "x-goog-api-key" not in _MockCloudDecisionsHandler.last_headers

    # 2. Keyless Vertex AI via ADC (`projects/.../locations/.../decisions`)
    s_vtx_adc = decisions_engine.Settings(
        auth_mode="vertex_adc",
        model="gemini-decision-1",
        project_id="customer-gcp-proj",
        location="global",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        vertex_api_version="v1beta1",
        vertex_base_override=f"http://127.0.0.1:{port}",
        timeout_s=5.0,
        allowed_dirs=(),
    )
    res_vtx = decisions_engine.create_decision(
        call_args, s_vtx_adc, token_provider=lambda _scopes: fake_token
    )
    assert res_vtx["backend"] == "vertex_adc:global"
    assert _MockCloudDecisionsHandler.last_path == (
        "/v1beta1/projects/customer-gcp-proj/locations/global/decisions"
    )
    assert (
        _MockCloudDecisionsHandler.last_headers["authorization"]
        == "Bearer mock_oauth_bearer_98765"
    )

    # 3. Actionable remediation when gcloud user ADC lacks generative-language scope
    _MockCloudDecisionsHandler.scenario = "scope_insufficient"
    with pytest.raises(
        decisions_engine.PermissionDeniedError,
        match="generative-language.retriever",
    ) as exc_info:
      decisions_engine.create_decision(
          call_args, s_gem_adc, token_provider=lambda _scopes: fake_token
      )
    assert "mock_oauth_bearer_98765" not in str(exc_info.value)

    # 4. Actionable remediation when Vertex AI project is not yet allowlisted
    _MockCloudDecisionsHandler.scenario = "vertex_not_allowlisted"
    with pytest.raises(decisions_engine.ConfigError, match="gemini_adc"):
      decisions_engine.create_decision(
          call_args, s_vtx_adc, token_provider=lambda _scopes: fake_token
      )

    # 5. HTTP redirect refused so Bearer token never crosses origins
    _MockCloudDecisionsHandler.scenario = "redirect"
    with pytest.raises(decisions_engine.BackendError, match="redirect"):
      decisions_engine.create_decision(
          call_args, s_gem_adc, token_provider=lambda _scopes: fake_token
      )
  finally:
    server.shutdown()
    server.server_close()


# ---- 4. File Security, Router, MCP, and Antigravity Hooks -------------------


def test_file_security_blocks_symlinks_dotfiles_and_secrets(
    tmp_path: pathlib.Path,
) -> None:
  allowed = tmp_path / "workspace"
  allowed.mkdir()
  outside = tmp_path / "outside.txt"
  outside.write_text("secret", encoding="utf-8")
  (allowed / "sym.txt").symlink_to(outside)
  (allowed / ".env").write_text("KEY=1", encoding="utf-8")
  (allowed / "adc_credentials.json").write_text("{}", encoding="utf-8")

  settings = decisions_engine.Settings(
      auth_mode="api_key",
      model="gemini-decision-1",
      project_id="",
      location="global",
      api_base="http://127.0.0.1:9",
      vertex_api_version="v1beta1",
      vertex_base_override="",
      timeout_s=3.0,
      allowed_dirs=(str(allowed),),
  )
  q = {"q": {"type": "noul", "instructions": "Valid?"}}
  for bad_file in (
      outside,
      allowed / "sym.txt",
      allowed / ".env",
      allowed / "adc_credentials.json",
  ):
    with pytest.raises(decisions_engine.InvalidInputError):
      decisions_engine.create_decision(
          {"input": [{"file": str(bad_file)}], "questions": q},
          settings,
          environ={"GEMINI_API_KEY": "mock_key_file_test"},
      )


def test_two_tier_router_and_thread_floor() -> None:
  r_quick = decisions_engine.route_prompt("lgtm", user_turn_count=1)
  assert r_quick["subagent_tier"] == "flash_lite"

  locked = {
      "locked_tier": "pro",
      "locked_model_id": "gemini-3.1-pro-high",
      "locked_thinking_level": 3,
      "locked_task_class": "deep_debug_architecture",
  }
  r_cont = decisions_engine.route_prompt(
      "yes go ahead", user_turn_count=2, thread_state=locked
  )
  assert r_cont["subagent_tier"] == "pro"
  assert r_cont["thread_lock_applied"] is True


def test_mcp_and_hooks_end_to_end(tmp_path: pathlib.Path) -> None:
  srv = mcp_server.McpServer(
      create_fn=lambda _args, _cfg: {
          "backend": "gemini_adc",
          "model": "gemini-decision-1",
          "answers": {
              "q": {
                  "type": "noul",
                  "noul": 0.85,
                  "value": True,
                  "confidence": 0.9,
              }
          },
          "usage": {"input_tokens": 10},
          "latency_ms": 135,
      }
  )
  tools_resp = json.loads(
      srv.handle_line(
          json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(
              "utf-8"
          )
      ).decode("utf-8")
  )
  assert {t["name"] for t in tools_resp["result"]["tools"]} == {
      "create_decision",
      "route_prompt",
      "list_models",
      "configure_router",
  }

  cfg_path = tmp_path / "router_config.json"
  cfg_path.write_text(json.dumps({"mode": "advisor"}), encoding="utf-8")
  out_pre = router_hook.handle_pre_invocation(
      {
          "conversationId": "../escape-attempt",
          "invocationNum": 1,
          "modelName": "auto",
          "lastUserInput": "ship it",
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  assert out_pre["injectSteps"][0]["systemMessage"]["systemMessage"] == ""

  out_sub = router_hook.handle_pre_tool_use(
      {
          "conversationId": "conv-1",
          "lastUserInput": "check status",
          "toolCall": {
              "name": "invoke_subagent",
              "args": {
                  "TypeName": "research",
                  "Role": "Checker",
                  "Prompt": "ship it",
                  "Model": "inherit",
              },
          },
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  assert out_sub["decision"] == "allow"
  assert out_sub["overwrite"] == {"Model": "flash_lite"}
  assert "Routed with Intent" in out_sub["reason"]

  telemetry_lines = [
      json.loads(line)
      for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  assert telemetry_lines[0]["event"] == "PreInvocation"
  assert telemetry_lines[0]["prompt"] == "ship it"
  assert telemetry_lines[1]["event"] == "PreToolUse:invoke_subagent"
  assert telemetry_lines[1]["prompt"] == "ship it"


def test_wif_env_token_and_vertex_model_prefix_normalization() -> None:
  tok = decisions_engine.resolve_adc_token(
      decisions_engine.ADC_SCOPES_GEMINI,
      project_hint="wif-proj-99",
      environ={"GOOGLE_OAUTH_ACCESS_TOKEN": "mock_wif_short_lived_token"},
  )
  assert tok.access_token == "mock_wif_short_lived_token"
  assert tok.project_id == "wif-proj-99"
  assert tok.source == "env:GOOGLE_OAUTH_ACCESS_TOKEN"

  with pytest.raises(decisions_engine.ConfigError):
    decisions_engine.resolve_adc_token(
        decisions_engine.ADC_SCOPES_GEMINI,
        environ={"GOOGLE_OAUTH_ACCESS_TOKEN": "bad token with spaces"},
    )

  server = http.server.HTTPServer(("127.0.0.1", 0), _MockCloudDecisionsHandler)
  port = server.server_port
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    _MockCloudDecisionsHandler.scenario = "ok"
    s_vtx = decisions_engine.Settings(
        auth_mode="vertex_adc",
        model="models/gemini-decision-1",
        project_id="wif-proj-99",
        location="us-central1",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        vertex_api_version="v1beta1",
        vertex_base_override=f"http://127.0.0.1:{port}",
        timeout_s=5.0,
        allowed_dirs=(),
    )
    res = decisions_engine.create_decision(
        {
            "input": [{"text": "Ping"}],
            "questions": {
                "is_arch": {"type": "noul", "instructions": "Arch?"}
            },
        },
        s_vtx,
        token_provider=lambda _s: tok,
    )
    assert res["model"] == "gemini-decision-1"
    assert _MockCloudDecisionsHandler.last_body["model"] == "gemini-decision-1"
  finally:
    server.shutdown()
    server.server_close()


def test_cli_status_and_auth_check_never_expose_tokens() -> None:
  env = dict(os.environ)
  env["GOOGLE_CLOUD_PROJECT"] = "customer-cli-proj"
  env["GOOGLE_OAUTH_ACCESS_TOKEN"] = "mock_cli_bearer_do_not_print"
  env["GEMINI_DECISIONS_AUTH_MODE"] = "gemini_adc"

  proc = subprocess.run(
      [
          sys.executable,
          "-B",
          str(REPO_ROOT / "router_hook.py"),
          "auth-check",
      ],
      env=env,
      capture_output=True,
      text=True,
      timeout=8.0,
      check=True,
  )
  assert "mock_cli_bearer_do_not_print" not in proc.stdout
  assert "mock_cli_bearer_do_not_print" not in proc.stderr
  parsed = json.loads(proc.stdout)
  assert parsed["ok"] is True
  assert parsed["auth_mode"] == "gemini_adc"
  assert parsed["project_id"] == "customer-cli-proj"
  assert parsed["credential_source"] == "env:GOOGLE_OAUTH_ACCESS_TOKEN"


def test_choice_and_score_validation_and_safety_blocks() -> None:
  q_choice = decisions_engine.Question(
      key="intent",
      type="choice",
      instructions="Pick intent",
      criteria=(("code", "Code"), ("docs", "Docs")),
  )
  q_score = decisions_engine.Question(
      key="effort",
      type="score",
      instructions="Rate effort",
      criteria=("Low", "Med", "High"),
  )
  parsed = decisions_engine.parse_rest_response(
      {
          "answers": {
              "intent": {
                  "type": "choice",
                  "choice": "code",
                  "confidence": 0.88,
                  "probabilities": {"code": 0.8, "docs": 0.2},
              },
              "effort": {
                  "type": "score",
                  "score": 1.7,
                  "confidence": 0.82,
                  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
              },
          },
          "usageMetadata": {"promptTokenCount": 42},
      },
      [q_choice, q_score],
  )
  assert parsed["answers"]["intent"]["choice"] == "code"
  assert parsed["answers"]["effort"]["level"] == 2
  assert parsed["usage"] == {"input_tokens": 42}

  with pytest.raises(decisions_engine.SafetyBlockedError):
    decisions_engine.parse_rest_response(
        {"promptFeedback": {"blockReason": "SAFETY"}},
        [q_choice],
    )


def test_list_and_configure_target_models(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies target model catalog listing, config overrides, and env overrides."""
  catalog = decisions_engine.list_available_target_models(environ={})
  model_ids = {m["id"] for m in catalog["available_target_models"]}
  assert {
      "gemini-3.1-flash-lite",
      "gemini-3-flash",
      "gemini-3.1-pro-low",
      "gemini-3.1-pro-high",
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash",
      "gemini-2.5-pro",
      "claude-sonnet-4-6",
      "claude-opus-4-6",
      "gpt-oss-120b-medium",
  }.issubset(model_ids)
  assert catalog["active_tier_models"] == {
      "flash_lite": "gemini-3.1-flash-lite",
      "flash": "gemini-3-flash",
      "pro": "gemini-3.1-pro-high",
  }

  # Config file overrides + custom model enum
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {
          "mode": "advisor",
          "tier_models": {
              "flash_lite": "gemini-2.5-flash-lite",
              "flash": "gemini-2.5-flash",
              "pro": "claude-sonnet-4-6",
          },
          "custom_model_enums": {
              "my-custom-pro": "MODEL_CUSTOM_PRO_V1",
          },
      },
      config_path=cfg_path,
  )
  loaded = router_hook.load_router_config(config_path=cfg_path)
  assert loaded["tier_models"]["pro"] == "claude-sonnet-4-6"
  assert router_hook.resolve_model_enums(loaded)["my-custom-pro"] == (
      "MODEL_CUSTOM_PRO_V1"
  )

  # PreInvocation routes to configured custom target model
  out_pre = router_hook.handle_pre_invocation(
      {
          "conversationId": "cfg-model-conv",
          "invocationNum": 1,
          "modelName": "auto",
          "lastUserInput": "use gemini pro to audit this auth flow",
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  title = out_pre["injectSteps"][0]["systemMessage"]["renderInfo"]["title"]
  assert "claude-sonnet-4-6" in title

  # Environment variable override takes precedence over config
  monkeypatch.setenv("GEMINI_DECISIONS_MODEL_PRO", "gemini-2.5-pro")
  loaded_env = router_hook.load_router_config(config_path=cfg_path)
  assert loaded_env["tier_models"]["pro"] == "gemini-2.5-pro"

  # MCP list_models tool returns the full catalog
  srv = mcp_server.McpServer(config_path=cfg_path)
  resp = json.loads(
      srv.handle_line(
          json.dumps({
              "jsonrpc": "2.0",
              "id": 9,
              "method": "tools/call",
              "params": {"name": "list_models", "arguments": {}},
          }).encode("utf-8")
      ).decode("utf-8")
  )
  assert resp["result"]["isError"] is False
  sc = resp["result"]["structuredContent"]
  assert len(sc["available_target_models"]) >= 10
  assert sc["active_tier_models"]["pro"] == "gemini-2.5-pro"


def test_auto_routing_prompt_matrix_and_telemetry_logs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies auto_switch routing paths across a matrix of test prompts + logs."""
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {
          "mode": "auto_switch",
          "min_confidence": 0.45,
          "enable_fast_rules": True,
          "enable_subagent_routing": True,
      },
      config_path=cfg_path,
  )

  dispatched: list[tuple[str, str, int]] = []

  def fake_spawn(
      conv_id: str,
      model_enum: str,
      thinking_level: int,
      *,
      allowed_enums: set[str] | None = None,
  ) -> bool:
    assert allowed_enums and model_enum in allowed_enums
    dispatched.append((conv_id, model_enum, thinking_level))
    return True

  monkeypatch.setattr(router_hook, "_verify_cascade_ownership", lambda _cid: True)
  monkeypatch.setattr(router_hook, "_spawn_redispatch", fake_spawn)

  def fake_decision(call_args: dict[str, Any], _cfg: Any) -> dict[str, Any]:
    txt = str(call_args["input"][-1]["text"]).lower()
    if "capital of france" in txt or "markdown files" in txt:
      return {
          "backend": "gemini_api",
          "model": "gemini-decision-1",
          "latency_ms": 180,
          "answers": {
              "task_class": {
                  "choice": "quick_lookup",
                  "confidence": 0.96,
                  "probabilities": {"quick_lookup": 0.99},
              },
              "needs_pro_reasoning": {"noul": 0.02, "confidence": 0.96},
              "effort_rubric": {"score": 0.25, "confidence": 0.94},
          },
      }
    if "fastapi" in txt or "refactor" in txt:
      return {
          "backend": "gemini_api",
          "model": "gemini-decision-1",
          "latency_ms": 205,
          "answers": {
              "task_class": {
                  "choice": "coding_edit",
                  "confidence": 0.92,
                  "probabilities": {"coding_edit": 0.98},
              },
              "needs_pro_reasoning": {"noul": 0.12, "confidence": 0.90},
              "effort_rubric": {"score": 1.75, "confidence": 0.88},
          },
      }
    return {
        "backend": "gemini_api",
        "model": "gemini-decision-1",
        "latency_ms": 290,
        "answers": {
            "task_class": {
                "choice": "deep_debug_architecture",
                "confidence": 0.93,
                "probabilities": {"deep_debug_architecture": 0.94},
            },
            "needs_pro_reasoning": {"noul": 0.95, "confidence": 0.92},
            "effort_rubric": {"score": 2.85, "confidence": 0.90},
        },
    }

  def route_with_mock(prompt: str, **kwargs: Any) -> dict[str, Any]:
    return decisions_engine.route_prompt(
        prompt, decision_fn=fake_decision, **kwargs
    )

  cases = [
      (
          "conv-t0-confirm",
          1,
          "ship it",
          "flash_lite",
          "gemini-3.1-flash-lite",
          "fast_rules (<1 ms)",
      ),
      (
          "conv-t0-override",
          1,
          "use gemini pro to review this design",
          "pro",
          "gemini-3.1-pro-high",
          "fast_rules (<1 ms)",
      ),
      (
          "conv-t1-lite",
          1,
          "What is the capital of France?",
          "flash_lite",
          "gemini-3.1-flash-lite",
          "gemini_api",
      ),
      (
          "conv-t1-flash",
          1,
          "Refactor this FastAPI endpoint to validate pagination parameters",
          "flash",
          "gemini-3-flash",
          "gemini_api",
      ),
      (
          "conv-t1-pro",
          1,
          "Design a multi-region Cloud Spanner schema with conflict-free failover",
          "pro",
          "gemini-3.1-pro-high",
          "gemini_api",
      ),
      (
          "conv-t1-pro",
          2,
          "yes go ahead",
          "pro",
          "gemini-3.1-pro-high",
          "thread_lock (<1 ms)",
      ),
  ]

  for conv_id, inv_num, prompt, _exp_tier, _exp_model, _exp_backend in cases:
    out = router_hook.handle_pre_invocation(
        {
            "conversationId": conv_id,
            "invocationNum": inv_num,
            "modelName": "auto",
            "lastUserInput": prompt,
        },
        state_dir=tmp_path,
        config_path=cfg_path,
        route_fn=route_with_mock,
    )
    assert out == {"terminationBehavior": "terminate"}

  sub_out = router_hook.handle_pre_tool_use(
      {
          "conversationId": "conv-t1-pro",
          "lastUserInput": "list docs",
          "toolCall": {
              "name": "invoke_subagent",
              "args": {
                  "TypeName": "research",
                  "Role": "Codebase Researcher",
                  "Prompt": "List the markdown files in docs/",
                  "Model": "inherit",
              },
          },
      },
      state_dir=tmp_path,
      config_path=cfg_path,
      route_fn=route_with_mock,
  )
  assert sub_out["decision"] == "allow"
  assert sub_out["overwrite"] == {"Model": "flash_lite"}
  assert "Routed with Intent" in sub_out["reason"]

  rows = [
      json.loads(line)
      for line in (tmp_path / "telemetry.jsonl")
      .read_text(encoding="utf-8")
      .splitlines()
      if line.strip()
  ]
  assert len(rows) == len(cases) + 1
  for idx, (_cid, _inv, prompt, exp_tier, exp_model, exp_backend) in enumerate(
      cases
  ):
    row = rows[idx]
    assert row["event"] == "PreInvocation"
    assert row["mode"] == "auto_switch"
    assert row["did_auto_switch"] is True
    assert row["prompt"] == prompt
    assert row["subagent_tier"] == exp_tier
    assert row["recommended_model_id"] == exp_model
    assert row["backend"] == exp_backend

  sub_row = rows[-1]
  assert sub_row["event"] == "PreToolUse:invoke_subagent"
  assert sub_row["assigned_tier"] == "flash_lite"
  assert sub_row["prompt"] == "List the markdown files in docs/"
  assert len(dispatched) == len(cases)


def test_auto_switch_prevents_multi_step_loop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Ensures auto_switch fires once per prompt even if current_model never changes."""
  cfg_path = tmp_path / "router_config.json"
  cfg_path.write_text(
      json.dumps({"mode": "auto_switch", "min_confidence": 0.45}),
      encoding="utf-8",
  )
  redispatch_calls: list[tuple[str, str, int]] = []

  def fake_redispatch(
      conv_id: str,
      model_enum: str,
      thinking_level: int,
      *,
      allowed_enums: set[str] | None = None,
  ) -> bool:
    del allowed_enums
    redispatch_calls.append((conv_id, model_enum, thinking_level))
    return True

  monkeypatch.setattr(router_hook, "_verify_cascade_ownership", lambda _cid: True)
  monkeypatch.setattr(router_hook, "_spawn_redispatch", fake_redispatch)

  hook_payload = {
      "conversationId": "loop-test-conv",
      "invocationNum": 1,
      "modelName": "custom-unmapped-model",
      "lastUserInput": "ship it",
  }

  step1 = router_hook.handle_pre_invocation(
      hook_payload, state_dir=tmp_path, config_path=cfg_path
  )
  assert step1 == {"terminationBehavior": "terminate"}
  assert len(redispatch_calls) == 1

  step2 = router_hook.handle_pre_invocation(
      hook_payload, state_dir=tmp_path, config_path=cfg_path
  )
  assert "Auto-Switched:" in (
      step2["injectSteps"][0]["systemMessage"]["renderInfo"]["title"]
  )
  assert len(redispatch_calls) == 1

  step3 = router_hook.handle_pre_invocation(
      hook_payload, state_dir=tmp_path, config_path=cfg_path
  )
  assert step3 == {}
  assert len(redispatch_calls) == 1


def test_verbosity_levels_and_routed_with_intent_aesthetics(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies compact, normal, and verbose 'Routed with Intent' rendering."""
  route = {
      "task_class": "deep_debug_architecture",
      "recommended_tier": "pro",
      "subagent_tier": "pro",
      "recommended_model_id": "gemini-3.1-pro-high",
      "tier_rank": 3,
      "thinking_level": 4,
      "thinking_label": "Max",
      "confidence": 0.9133,
      "backend": "gemini_api",
      "latency_ms": 320,
      "rationale": "task=deep_debug_architecture, P(pro)=0.95",
      "thread_lock_applied": False,
      "explicit_model_requested": False,
      "probabilities": {
          "task_class": {"deep_debug_architecture": 0.92, "coding_edit": 0.04},
          "needs_pro_reasoning": 0.95,
          "effort_score": 2.83,
      },
  }
  compact = router_hook._build_render_info(
      route, current_model="auto", switched=True, verbosity="compact"
  )
  assert compact["titlePrefix"] == "✦ Routed with Intent"
  assert "Routed with Intent" in compact["markdown"]
  assert "╭" not in compact["markdown"]

  normal = router_hook._build_render_info(
      route, current_model="auto", switched=False, verbosity="normal"
  )
  assert normal["titlePrefix"] == "✦ Routed with Intent"
  assert "ROUTED WITH INTENT" in normal["markdown"]
  assert "Class Probabilities" not in normal["markdown"]

  verbose = router_hook._build_render_info(
      route, current_model="auto", switched=True, verbosity="verbose"
  )
  assert "ROUTED WITH INTENT" in verbose["markdown"]
  assert "Class Probabilities" in verbose["markdown"]
  assert "P(pro reasoning)" in verbose["markdown"]

  cfg_path = tmp_path / "router_config.json"
  saved = router_hook.save_router_config(
      {"verbosity": "verbose"}, config_path=cfg_path
  )
  assert saved["verbosity"] == "verbose"


def test_private_dotenv_permissions_and_symlink_rejection(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies 0600 private .env loading, env precedence, 0644, and symlinks."""
  # Missing .env returns empty dict
  assert decisions_engine.load_private_dotenv(tmp_path / "missing.env") == {}

  # Valid 0600 .env loads GEMINI_API_KEY and GOOGLE_CLOUD_PROJECT
  dotenv_file = tmp_path / "private.env"
  dotenv_file.write_text(
      "# Private router credentials\n"
      "GEMINI_API_KEY=\"mock_dotenv_key_0600\"\n"
      "export GOOGLE_CLOUD_PROJECT='dotenv-project-01'\n"
      "GEMINI_DECISIONS_HOOK_TIMEOUT_S=1.2\n",
      encoding="utf-8",
  )
  dotenv_file.chmod(0o600)

  loaded = decisions_engine.load_private_dotenv(dotenv_file)
  assert loaded == {
      "GEMINI_API_KEY": "mock_dotenv_key_0600",
      "GOOGLE_CLOUD_PROJECT": "dotenv-project-01",
      "GEMINI_DECISIONS_HOOK_TIMEOUT_S": "1.2",
  }

  # Process environ overrides .env while keeping fallback keys from .env
  eff = decisions_engine.effective_environ(
      {"GOOGLE_CLOUD_PROJECT": "process-override-proj"},
      dotenv_path=dotenv_file,
  )
  assert eff["GEMINI_API_KEY"] == "mock_dotenv_key_0600"
  assert eff["GOOGLE_CLOUD_PROJECT"] == "process-override-proj"

  # Hermeticity: explicit environ with dotenv_path=None ignores user home .env
  hermetic = decisions_engine.effective_environ({"GOOGLE_CLOUD_PROJECT": "p1"})
  assert dict(hermetic) == {"GOOGLE_CLOUD_PROJECT": "p1"}

  # load_settings wires effective_environ with dotenv_path
  settings = decisions_engine.load_settings(
      config_path=str(tmp_path / "none.json"),
      environ={},
      dotenv_path=dotenv_file,
  )
  assert settings.auth_mode == "api_key"
  assert settings.project_id == "dotenv-project-01"

  # Tightened hook_timeout_s kwarg accepted by route_prompt
  routed = decisions_engine.route_prompt("ship it", hook_timeout_s=1.2)
  assert routed["subagent_tier"] == "flash_lite"

  # 0644 (group/world-readable) .env raises ConfigError
  world_env = tmp_path / "world_readable.env"
  world_env.write_text("GEMINI_API_KEY=exposed_val\n", encoding="utf-8")
  world_env.chmod(0o644)
  with pytest.raises(decisions_engine.ConfigError):
    decisions_engine.load_private_dotenv(world_env)

  # 0640 (group-readable) .env also raises ConfigError
  group_env = tmp_path / "group_readable.env"
  group_env.write_text("GEMINI_API_KEY=exposed_val\n", encoding="utf-8")
  group_env.chmod(0o640)
  with pytest.raises(decisions_engine.ConfigError):
    decisions_engine.load_private_dotenv(group_env)

  # Symlinked .env pointing to a valid 0600 file raises ConfigError
  symlink_env = tmp_path / "symlink.env"
  symlink_env.symlink_to(dotenv_file)
  with pytest.raises(decisions_engine.ConfigError, match="symlink"):
    decisions_engine.load_private_dotenv(symlink_env)


def test_user_markdown_inline_banners_and_four_verbosity_levels(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies silent/compact/normal/verbose user_markdown banners + PreToolUse."""
  cfg_path = tmp_path / "router_config.json"

  # 1. silent: returns {} (0 steps injected) while still writing telemetry
  router_hook.save_router_config(
      {"mode": "advisor", "verbosity": "silent"}, config_path=cfg_path
  )
  out_silent = router_hook.handle_pre_invocation(
      {
          "conversationId": "verb-silent-conv",
          "invocationNum": 1,
          "modelName": "auto",
          "lastUserInput": "ship it",
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  assert out_silent == {}

  sub_silent = router_hook.handle_pre_tool_use(
      {
          "conversationId": "verb-silent-conv",
          "lastUserInput": "run check",
          "toolCall": {
              "name": "invoke_subagent",
              "args": {
                  "TypeName": "research",
                  "Role": "Checker",
                  "Prompt": "ship it",
                  "Model": "inherit",
              },
          },
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  assert sub_silent == {"decision": "allow", "overwrite": {"Model": "flash_lite"}}
  assert "reason" not in sub_silent

  # 2. compact, normal, verbose: 0 model tokens in Desktop App + panic-safe renderInfo
  for level in ("compact", "normal", "verbose"):
    router_hook.save_router_config(
        {"mode": "advisor", "verbosity": level}, config_path=cfg_path
    )
    out = router_hook.handle_pre_invocation(
        {
            "conversationId": f"verb-{level}-conv",
            "invocationNum": 1,
            "modelName": "auto",
            "lastUserInput": "use gemini pro to review distributed consensus",
        },
        state_dir=tmp_path,
        config_path=cfg_path,
    )
    sys_msg = out["injectSteps"][0]["systemMessage"]
    assert sys_msg["systemMessage"] == ""
    assert "metadata" not in sys_msg
    user_md = sys_msg["renderInfo"]["markdown"]
    assert "routed with intent" in user_md.lower()
    if level == "compact":
      assert "Routed with Intent" in user_md
      assert "╭" not in user_md
    elif level == "normal":
      assert "Rationale" in user_md
      assert "Thread Lock" not in user_md
    elif level == "verbose":
      assert "Thread Lock" in user_md

    sub_out = router_hook.handle_pre_tool_use(
        {
            "conversationId": f"verb-{level}-sub",
            "lastUserInput": "delegate check",
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "research",
                    "Role": "Checker",
                    "Prompt": "ship it",
                    "Model": "inherit",
                },
            },
        },
        state_dir=tmp_path,
        config_path=cfg_path,
    )
    assert sub_out["decision"] == "allow"
    assert sub_out["overwrite"] == {"Model": "flash_lite"}
    assert "Routed with Intent" in sub_out["reason"]

  # Verify silent PreInvocation still wrote telemetry alongside the other levels
  rows = [
      json.loads(line)
      for line in (tmp_path / "telemetry.jsonl")
      .read_text(encoding="utf-8")
      .splitlines()
      if line.strip()
  ]
  pre_events = [r for r in rows if r["event"] == "PreInvocation"]
  assert len(pre_events) == 4
  assert pre_events[0]["prompt"] == "ship it"


def test_advisor_deduplicates_multi_step_tool_loops(
    tmp_path: pathlib.Path,
) -> None:
  """Ensures advisor mode scores once per user turn and skips tool-loop steps."""
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {"mode": "advisor", "verbosity": "normal"}, config_path=cfg_path
  )

  route_calls: list[str] = []

  def counting_route(prompt: str, **kwargs: Any) -> dict[str, Any]:
    route_calls.append(prompt)
    return decisions_engine.route_prompt(prompt, **kwargs)

  turn1_payload = {
      "conversationId": "advisor-dedup-conv",
      "invocationNum": 1,
      "modelName": "auto",
      "lastUserInput": "ship it",
  }

  step1 = router_hook.handle_pre_invocation(
      turn1_payload,
      state_dir=tmp_path,
      config_path=cfg_path,
      route_fn=counting_route,
  )
  assert "injectSteps" in step1
  assert "metadata" not in step1["injectSteps"][0]["systemMessage"]
  assert (
      "routed with intent"
      in step1["injectSteps"][0]["systemMessage"]["renderInfo"][
          "markdown"
      ].lower()
  )
  assert len(route_calls) == 1

  step2 = router_hook.handle_pre_invocation(
      turn1_payload,
      state_dir=tmp_path,
      config_path=cfg_path,
      route_fn=counting_route,
  )
  assert step2 == {}
  assert len(route_calls) == 1

  step3 = router_hook.handle_pre_invocation(
      turn1_payload,
      state_dir=tmp_path,
      config_path=cfg_path,
      route_fn=counting_route,
  )
  assert step3 == {}
  assert len(route_calls) == 1

  # Sending a new lastUserInput on Turn 2 scores again and injects a fresh banner
  turn2_payload = {
      "conversationId": "advisor-dedup-conv",
      "invocationNum": 2,
      "modelName": "auto",
      "lastUserInput": "use gemini pro to audit this consensus protocol",
  }
  turn2_step1 = router_hook.handle_pre_invocation(
      turn2_payload,
      state_dir=tmp_path,
      config_path=cfg_path,
      route_fn=counting_route,
  )
  assert "injectSteps" in turn2_step1
  assert len(route_calls) == 2


def test_auto_switch_preflight_falls_back_to_advisor_when_cli_unowned(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies auto_switch never terminates when GetCascadeTrajectory is unowned."""
  # Unit check on _verify_cascade_ownership RPC responses
  monkeypatch.setattr(router_hook, "_ls_rpc", lambda *_a, **_k: None)
  assert router_hook._verify_cascade_ownership("conv-none") is False

  monkeypatch.setattr(
      router_hook, "_ls_rpc", lambda *_a, **_k: {"error": "not_found"}
  )
  assert router_hook._verify_cascade_ownership("conv-err") is False

  monkeypatch.setattr(
      router_hook,
      "_ls_rpc",
      lambda *_a, **_k: {"status": "CASCADE_RUN_STATUS_RUNNING"},
  )
  assert router_hook._verify_cascade_ownership("conv-owned") is True

  # Integration check: unowned CLI conversation falls back to inline advisor banner
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {"mode": "auto_switch", "verbosity": "normal", "min_confidence": 0.45},
      config_path=cfg_path,
  )
  monkeypatch.setattr(
      router_hook, "_verify_cascade_ownership", lambda _cid: False
  )
  spawned: list[str] = []
  monkeypatch.setattr(
      router_hook,
      "_spawn_redispatch",
      lambda conv_id, *_a, **_k: spawned.append(conv_id) or True,
  )

  out = router_hook.handle_pre_invocation(
      {
          "conversationId": "cli-ephemeral-ls-conv",
          "invocationNum": 1,
          "modelName": "auto",
          "lastUserInput": "use gemini pro to design a distributed lock manager",
      },
      state_dir=tmp_path,
      config_path=cfg_path,
  )
  assert "terminationBehavior" not in out
  assert spawned == []
  sys_msg = out["injectSteps"][0]["systemMessage"]
  assert sys_msg["systemMessage"] == ""
  assert "metadata" not in sys_msg
  assert "routed with intent" in sys_msg["renderInfo"]["markdown"].lower()
  assert sys_msg["renderInfo"]["title"].startswith("Recommended:")

  rows = [
      json.loads(line)
      for line in (tmp_path / "telemetry.jsonl")
      .read_text(encoding="utf-8")
      .splitlines()
      if line.strip()
  ]
  assert len(rows) == 1
  assert rows[0]["mode"] == "auto_switch"
  assert rows[0]["did_auto_switch"] is False
  assert rows[0]["auto_switch_fallback"] == "cascade_unowned"


def test_configure_router_mcp_tool_and_cli_commands(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies configure_router MCP tool plus auth-setup and watch --once CLI."""
  cfg_path = tmp_path / ".config" / "gemini-decisions-router" / "router_config.json"
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setattr(router_hook, "default_router_config_path", lambda: cfg_path)

  srv = mcp_server.McpServer()
  list_resp = json.loads(
      srv.handle_line(
          json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(
              "utf-8"
          )
      ).decode("utf-8")
  )
  assert {t["name"] for t in list_resp["result"]["tools"]} == {
      "create_decision",
      "route_prompt",
      "list_models",
      "configure_router",
  }

  cfg_resp = json.loads(
      srv.handle_line(
          json.dumps({
              "jsonrpc": "2.0",
              "id": 2,
              "method": "tools/call",
              "params": {
                  "name": "configure_router",
                  "arguments": {
                      "verbosity": "compact",
                      "tier_models": {"pro": "claude-opus-4-6"},
                  },
              },
          }).encode("utf-8")
      ).decode("utf-8")
  )
  assert cfg_resp["result"]["isError"] is False
  saved_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
  assert saved_disk["verbosity"] == "compact"
  assert saved_disk["tier_models"]["pro"] == "claude-opus-4-6"

  # Test auth-setup CLI writes 0600 .env without leaking secrets to stdout/stderr
  dotenv_out = tmp_path / ".config" / "gemini-decisions-router" / ".env"
  env = dict(os.environ)
  env["HOME"] = str(tmp_path)
  env["GEMINI_API_KEY"] = "mock_auth_setup_secret_999"
  env["GOOGLE_CLOUD_PROJECT"] = "setup-cli-proj-42"

  setup_proc = subprocess.run(
      [
          sys.executable,
          "-B",
          str(REPO_ROOT / "router_hook.py"),
          "auth-setup",
          "--from-env",
          "--dotenv-path",
          str(dotenv_out),
      ],
      env=env,
      capture_output=True,
      text=True,
      timeout=8.0,
      check=True,
  )
  assert "mock_auth_setup_secret_999" not in setup_proc.stdout
  assert "mock_auth_setup_secret_999" not in setup_proc.stderr
  setup_json = json.loads(setup_proc.stdout)
  assert setup_json["ok"] is True
  assert stat.S_IMODE(dotenv_out.stat().st_mode) == 0o600
  parsed_dotenv = decisions_engine.load_private_dotenv(dotenv_out)
  assert parsed_dotenv["GEMINI_API_KEY"] == "mock_auth_setup_secret_999"
  assert parsed_dotenv["GOOGLE_CLOUD_PROJECT"] == "setup-cli-proj-42"

  # Test watch --once CLI prints live monitor header and recent telemetry entries
  cache_dir = tmp_path / ".cache" / "gemini-decisions-router"
  cache_dir.mkdir(parents=True, exist_ok=True)
  (cache_dir / "telemetry.jsonl").write_text(
      json.dumps({
          "ts": "2026-01-01T00:00:00+00:00",
          "event": "PreInvocation",
          "prompt": "Refactor pagination handler",
          "mode": "advisor",
          "did_auto_switch": False,
          "current_model": "auto",
          "recommended_model_id": "gemini-3-flash",
          "subagent_tier": "flash",
          "thinking_label": "Medium",
          "task_class": "coding_edit",
          "confidence": 0.91,
          "latency_ms": 185,
          "backend": "gemini_api",
      })
      + "\n",
      encoding="utf-8",
  )
  watch_proc = subprocess.run(
      [
          sys.executable,
          "-B",
          str(REPO_ROOT / "router_hook.py"),
          "watch",
          "--once",
      ],
      env=env,
      capture_output=True,
      text=True,
      timeout=8.0,
      check=True,
  )
  assert "ROUTED WITH INTENT | LIVE MONITOR" in watch_proc.stdout
  assert "gemini-3-flash" in watch_proc.stdout

  # Test install.sh into isolated directories + generated agy-router wrapper
  plugins_dir = tmp_path / "plugins"
  bin_dir = tmp_path / "bin"
  install_env = dict(env)
  install_env["ANTIGRAVITY_PLUGINS_DIR"] = str(plugins_dir)
  install_env["BIN_DIR"] = str(bin_dir)
  subprocess.run(
      ["bash", str(REPO_ROOT / "install.sh")],
      env=install_env,
      capture_output=True,
      text=True,
      timeout=15.0,
      check=True,
  )
  wrapper = bin_dir / "agy-router"
  assert wrapper.is_file()
  assert os.access(wrapper, os.X_OK)
  status_proc = subprocess.run(
      [str(wrapper), "status"],
      env=install_env,
      capture_output=True,
      text=True,
      timeout=8.0,
      check=True,
  )
  status_json = json.loads(status_proc.stdout)
  assert status_json["router_config"]["verbosity"] == "compact"


def test_dynamic_model_discovery_cli_and_ls_parsing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies dynamic CLI/LS model auto-discovery, 0600 cache, CLI, and MCP."""
  # 1. parse_agy_models_output with realistic ANSI spinner prefixes and log lines
  raw_cli_output = (
      "\x1b[36m\r⠋ Fetching available models...\x1b[0m"
      "\rgemini-5.0-flash-low    Gemini 5.0 Flash (Low)\n"
      "I1119 00:00:00.000000 12345 client.go:42] transient info log\n"
      "gemini-5.0-flash-medium  Gemini 5.0 Flash (Medium)\n"
      "\r⠙ Refreshing catalog...\rgemini-5.0-pro-high     Gemini 5.0 Pro (High)\n"
      "gemini-pro-orchestrator\tGemini Pro Orchestrator\n"
      "ERROR: ignored diagnostic line\n"
  )
  parsed_cli = decisions_engine.parse_agy_models_output(raw_cli_output)
  assert [m["id"] for m in parsed_cli] == [
      "gemini-5.0-flash-low",
      "gemini-5.0-flash-medium",
      "gemini-5.0-pro-high",
      "gemini-pro-orchestrator",
  ]
  by_cli_id = {m["id"]: m for m in parsed_cli}
  assert by_cli_id["gemini-5.0-flash-low"]["tier"] == "flash_lite"
  assert by_cli_id["gemini-5.0-flash-low"]["default_thinking_level"] == 1
  assert by_cli_id["gemini-5.0-flash-medium"]["tier"] == "flash"
  assert by_cli_id["gemini-5.0-flash-medium"]["default_thinking_level"] == 2
  assert by_cli_id["gemini-5.0-pro-high"]["tier"] == "pro"
  assert by_cli_id["gemini-5.0-pro-high"]["default_thinking_level"] == 4
  assert by_cli_id["gemini-pro-orchestrator"]["source"] == "discovered_cli"

  # 2. infer_model_tier_and_thinking across -low/-medium/-high/lite/flash/pro/orchestrator
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-flash-lite", "Gemini 5.0 Flash Lite"
  ) == ("flash_lite", 1)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-flash-low", "Gemini 5.0 Flash (Low)"
  ) == ("flash_lite", 1)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-flash-medium", "Gemini 5.0 Flash (Medium)"
  ) == ("flash", 2)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-flash-high", "Gemini 5.0 Flash (High)"
  ) == ("flash", 3)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-pro-low", "Gemini 5.0 Pro (Low)"
  ) == ("pro", 1)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-pro-medium", "Gemini 5.0 Pro (Medium)"
  ) == ("pro", 2)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-5.0-pro-high", "Gemini 5.0 Pro (High)"
  ) == ("pro", 4)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-pro-orchestrator", "Gemini Pro Orchestrator"
  ) == ("pro", 2)
  assert decisions_engine.infer_model_tier_and_thinking(
      "gemini-pro-orchestrator-high", "Gemini Pro Orchestrator (High)"
  ) == ("pro", 4)

  # 3. Mock Antigravity Language Server GetAvailableModels + CSRF discovery
  class _MockLsDiscoveryHandler(http.server.BaseHTTPRequestHandler):
    seen_csrf = ""

    def log_message(self, fmt: str, *args: Any) -> None:
      del fmt, args

    def do_GET(self) -> None:  # pylint: disable=invalid-name
      body = b'<html><script>window.cfg = {"csrfToken": "mock-ls-csrf-500"};</script></html>'
      self.send_response(200)
      self.send_header("Content-Type", "text/html")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get("Content-Length", "0"))
      if length:
        _ = self.rfile.read(length)
      type(self).seen_csrf = self.headers.get("x-codeium-csrf-token", "")
      if (
          self.path
          != "/exa.language_server_pb.LanguageServerService/GetAvailableModels"
          or type(self).seen_csrf != "mock-ls-csrf-500"
      ):
        self.send_response(403)
        self.end_headers()
        return
      payload = {
          "response": {
              "tabModelIds": ["tab-autocomplete-v1"],
              "agentModelSorts": [{
                  "groups": [{
                      "modelIds": [
                          "gemini-5.0-flash-low",
                          "gemini-5.0-flash-medium",
                          "gemini-5.0-pro-high",
                          "gemini-pro-orchestrator",
                      ]
                  }]
              }],
              "models": {
                  "gemini-5.0-flash-low": {
                      "label": "Gemini 5.0 Flash (Low)",
                      "model": "MODEL_GEMINI_5_0_FLASH_LOW",
                  },
                  "gemini-5.0-flash-medium": {
                      "label": "Gemini 5.0 Flash (Medium)",
                      "model": "MODEL_GEMINI_5_0_FLASH_MEDIUM",
                  },
                  "gemini-5.0-pro-high": {
                      "label": "Gemini 5.0 Pro (High)",
                      "model": "MODEL_GEMINI_5_0_PRO_HIGH",
                  },
                  "gemini-pro-orchestrator": {
                      "label": "Gemini Pro Orchestrator",
                      "model": "MODEL_GEMINI_PRO_ORCHESTRATOR",
                  },
                  "tab-autocomplete-v1": {
                      "label": "Tab Autocomplete",
                      "model": "MODEL_TAB_V1",
                  },
                  "hidden-experiment-model": {
                      "label": "Hidden Experiment",
                      "isInternal": True,
                      "model": "MODEL_HIDDEN_EXP",
                  },
              },
          }
      }
      raw = json.dumps(payload).encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(raw)))
      self.end_headers()
      self.wfile.write(raw)

  ls_server = http.server.HTTPServer(("127.0.0.1", 0), _MockLsDiscoveryHandler)
  ls_port = ls_server.server_port
  ls_thread = threading.Thread(target=ls_server.serve_forever, daemon=True)
  ls_thread.start()

  try:
    ls_addr = f"127.0.0.1:{ls_port}"
    ls_models = decisions_engine.discover_models_from_ls(ls_addr=ls_addr)
    assert [m["id"] for m in ls_models] == [
        "gemini-5.0-flash-low",
        "gemini-5.0-flash-medium",
        "gemini-5.0-pro-high",
        "gemini-pro-orchestrator",
    ]
    assert {m["id"]: m["enum"] for m in ls_models}["gemini-5.0-pro-high"] == (
        "MODEL_GEMINI_5_0_PRO_HIGH"
    )
    assert decisions_engine.discover_models_from_ls(ls_addr="127.0.0.1:9") == []

    # 4. discover_installed_models 0600 cache + suggest_tier_models + merged catalog
    fake_home = tmp_path / "isolated_home"
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(parents=True)
    fake_agy = fake_bin / "agy"
    fake_agy.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:] == ['models']:\n"
        f"  sys.stdout.write({raw_cli_output!r})\n",
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("ANTIGRAVITY_LS_ADDRESS", ls_addr)
    monkeypatch.setenv("ANTIGRAVITY_CSRF_TOKEN", "mock-ls-csrf-500")
    monkeypatch.delenv("GEMINI_DECISIONS_MODEL_FLASH_LITE", raising=False)
    monkeypatch.delenv("GEMINI_DECISIONS_MODEL_FLASH", raising=False)
    monkeypatch.delenv("GEMINI_DECISIONS_MODEL_PRO", raising=False)

    cache_file = (
        fake_home
        / ".cache"
        / "gemini-decisions-router"
        / "discovered_models.json"
    )
    discovered = decisions_engine.discover_installed_models(
        agy_bin=str(fake_agy),
        cache_path=cache_file,
        force_refresh=True,
    )
    assert len(discovered) == 4
    assert cache_file.is_file()
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600
    by_merged_id = {m["id"]: m for m in discovered}
    assert (
        by_merged_id["gemini-5.0-pro-high"]["enum"]
        == "MODEL_GEMINI_5_0_PRO_HIGH"
    )
    assert (
        by_merged_id["gemini-5.0-pro-high"]["label"] == "Gemini 5.0 Pro (High)"
    )

    # Fresh cache hit avoids network/CLI calls when force_refresh=False
    cached_again = decisions_engine.discover_installed_models(
        agy_bin=str(tmp_path / "nonexistent-agy"),
        cache_path=cache_file,
        force_refresh=False,
    )
    assert [m["id"] for m in cached_again] == [m["id"] for m in discovered]

    suggested = decisions_engine.suggest_tier_models(discovered)
    assert suggested == {
        "flash_lite": "gemini-5.0-flash-low",
        "flash": "gemini-5.0-flash-medium",
        "pro": "gemini-5.0-pro-high",
    }

    merged_catalog = decisions_engine.list_available_target_models(
        discovered_cache_path=cache_file,
    )
    assert merged_catalog["discovered_count"] == 4
    assert merged_catalog["suggested_tier_models"] == suggested
    merged_ids = {m["id"] for m in merged_catalog["available_target_models"]}
    assert {
        "gemini-3.1-pro-high",
        "gemini-5.0-flash-low",
        "gemini-5.0-flash-medium",
        "gemini-5.0-pro-high",
        "gemini-pro-orchestrator",
    }.issubset(merged_ids)

    # Hermeticity check: explicit environ without cache/refresh ignores disk cache
    hermetic_catalog = decisions_engine.list_available_target_models(environ={})
    assert hermetic_catalog["discovered_count"] == 0

    # router_hook.resolve_model_enums picks up discovered LS enums automatically
    resolved_enums = router_hook.resolve_model_enums({})
    assert (
        resolved_enums["gemini-5.0-pro-high"] == "MODEL_GEMINI_5_0_PRO_HIGH"
    )

    # 5. CLI sync-models --apply-recommended and list-models --refresh + MCP list_models
    cli_env = dict(os.environ)
    cli_env["HOME"] = str(fake_home)
    cli_env["PATH"] = f"{fake_bin}{os.pathsep}{cli_env.get('PATH', '')}"
    cli_env["ANTIGRAVITY_LS_ADDRESS"] = ls_addr
    cli_env["ANTIGRAVITY_CSRF_TOKEN"] = "mock-ls-csrf-500"
    for k in (
        "GEMINI_DECISIONS_MODEL_FLASH_LITE",
        "GEMINI_DECISIONS_MODEL_FLASH",
        "GEMINI_DECISIONS_MODEL_PRO",
    ):
      cli_env.pop(k, None)

    sync_proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPO_ROOT / "router_hook.py"),
            "sync-models",
            "--apply-recommended",
        ],
        env=cli_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=True,
    )
    sync_json = json.loads(sync_proc.stdout)
    assert sync_json["applied_recommended"] is True
    assert sync_json["discovered_count"] == 4
    assert sync_json["active_tier_models"] == suggested
    assert sync_json["suggested_tier_models"] == suggested

    list_proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPO_ROOT / "router_hook.py"),
            "list-models",
            "--refresh",
        ],
        env=cli_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=True,
    )
    list_json = json.loads(list_proc.stdout)
    assert list_json["discovered_count"] == 4
    assert list_json["active_tier_models"] == suggested

    isolated_cfg = (
        fake_home
        / ".config"
        / "gemini-decisions-router"
        / "router_config.json"
    )
    mcp_srv = mcp_server.McpServer(config_path=isolated_cfg)
    mcp_resp = json.loads(
        mcp_srv.handle_line(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {
                    "name": "list_models",
                    "arguments": {"refresh": False},
                },
            }).encode("utf-8")
        ).decode("utf-8")
    )
    assert mcp_resp["result"]["isError"] is False
    mcp_sc = mcp_resp["result"]["structuredContent"]
    assert mcp_sc["discovered_count"] == 4
    assert mcp_sc["active_tier_models"] == suggested

    mcp_bad = json.loads(
        mcp_srv.handle_line(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "list_models",
                    "arguments": {"refresh": "invalid"},
                },
            }).encode("utf-8")
        ).decode("utf-8")
    )
    assert mcp_bad["result"]["isError"] is True
  finally:
    ls_server.shutdown()
    ls_server.server_close()


def test_benchmark_suite_dataset_mock_runner_and_live_results(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies 100-prompt dataset, offline --mock CI runner, and live results."""
  dataset_path = REPO_ROOT / "benchmarks" / "prompts_100.json"
  results_path = REPO_ROOT / "benchmarks" / "results_100.json"
  runner_path = REPO_ROOT / "benchmarks" / "run_benchmark.py"
  dashboard_path = REPO_ROOT / "assets" / "benchmark-dashboard.png"
  bench_doc_path = REPO_ROOT / "docs" / "BENCHMARKS.md"

  assert dataset_path.is_file()
  assert results_path.is_file()
  assert runner_path.is_file()
  assert dashboard_path.is_file()
  assert bench_doc_path.is_file()

  # 1. Validate prompts_100.json stratification and schema
  dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
  assert isinstance(dataset, list) and len(dataset) == 100
  assert [item["id"] for item in dataset] == [
      f"P{idx:03d}" for idx in range(1, 101)
  ]
  cat_counts: dict[str, int] = {}
  sample_tiers: dict[str, int] = {}
  for item in dataset:
    cat = item["category"]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    assert item["expected_tier"] in ("flash_lite", "flash", "pro")
    if item.get("sample_e2e"):
      t = item["expected_tier"]
      sample_tiers[t] = sample_tiers.get(t, 0) + 1
  assert cat_counts == {
      "quick_lookup": 20,
      "coding_edit": 20,
      "writing_docs": 15,
      "data_analysis": 15,
      "deep_debug_architecture": 20,
      "agentic_lifecycle": 10,
  }
  assert sample_tiers == {"flash_lite": 5, "flash": 5, "pro": 5}

  # 2. Verify deterministic offline --mock benchmark mode (<5s, zero network)
  mock_out = tmp_path / "mock_results.json"
  proc = subprocess.run(
      [
          sys.executable,
          "-B",
          str(runner_path),
          "--dataset",
          str(dataset_path),
          "--output",
          str(mock_out),
          "--mock",
      ],
      capture_output=True,
      text=True,
      timeout=10.0,
      check=True,
  )
  assert "[benchmark:mock]" in proc.stdout
  mock_res = json.loads(mock_out.read_text(encoding="utf-8"))
  assert mock_res["metadata"]["mode"] == "mock"
  assert mock_res["metadata"]["dataset_size"] == 100
  assert (
      mock_res["strategies"]["decisions_router_full"]["tokens"][
          "total_output_tokens"
      ]
      == 0
  )

  # 3. Validate live results_100.json invariants, secret hygiene, and PNG header
  raw_live = results_path.read_text(encoding="utf-8")
  assert ("AIza" + "Sy") not in raw_live
  assert ("ya" + "29.") not in raw_live
  live_res = json.loads(raw_live)
  assert live_res["metadata"]["mode"] == "live"
  assert live_res["metadata"]["dataset_size"] == 100
  assert live_res["metadata"]["sample_e2e_size"] == 15

  full_strat = live_res["strategies"]["decisions_router_full"]
  tier1_strat = live_res["strategies"]["decisions_tier1_only"]
  assert full_strat["tokens"]["total_output_tokens"] == 0
  assert tier1_strat["tokens"]["total_output_tokens"] == 0
  assert full_strat["accuracy"]["exact_tier_accuracy_pct"] >= 90.0
  assert full_strat["accuracy"]["pro_miss_rate_pct"] == 0.0
  assert (
      live_res["e2e"]["projected_100_prompts"]["decisions_router_routed"][
          "time_saved_vs_always_pro_pct"
      ]
      > 15.0
  )

  png_bytes = dashboard_path.read_bytes()
  assert len(png_bytes) > 50_000
  assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_docs_contain_no_em_or_en_dashes() -> None:
  doc_paths = [
      REPO_ROOT / "README.md",
      REPO_ROOT / "agents" / "intent-router.md",
      REPO_ROOT / "rules" / "AGENTS.md",
      REPO_ROOT / "docs" / "BENCHMARKS.md",
      REPO_ROOT / "docs" / "CUSTOMER_ONBOARDING.md",
      REPO_ROOT / "docs" / "AUTHENTICATION_ADC.md",
      REPO_ROOT / "docs" / "API_KEY_QUICKSTART.md",
      REPO_ROOT / "docs" / "TESTING_STRATEGY.md",
      REPO_ROOT / "skills" / "gemini-decisions-router" / "SKILL.md",
      REPO_ROOT / "rules" / "AGENTS.md",
      REPO_ROOT / "agents" / "intent-router.md",
      REPO_ROOT / "agents" / "routed-specialist.md",
  ]
  for doc in doc_paths:
    text = doc.read_text(encoding="utf-8")
    assert "\u2014" not in text, f"Em-dash found in {doc}"
    assert "\u2013" not in text, f"En-dash found in {doc}"


def test_head_to_head_flash_lite_benchmark_and_streaming_stdout(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies N=100 flash-lite head-to-head runner, live JSON schema, and stdout streaming."""
  dataset_path = REPO_ROOT / "benchmarks" / "prompts_100.json"
  h2h_runner = REPO_ROOT / "benchmarks" / "run_router_head_to_head.py"
  h2h_results = REPO_ROOT / "benchmarks" / "results_flash_lite_router_100.json"
  bench_runner = REPO_ROOT / "benchmarks" / "run_benchmark.py"

  assert h2h_runner.is_file()
  assert h2h_results.is_file()

  # 1. Default --mock streams per-prompt [001/100] lines + ROUTED WITH INTENT summary box
  mock_h2h_out = tmp_path / "mock_h2h.json"
  proc_h2h = subprocess.run(
      [
          sys.executable,
          "-B",
          str(h2h_runner),
          "--dataset",
          str(dataset_path),
          "--output",
          str(mock_h2h_out),
          "--mock",
      ],
      capture_output=True,
      text=True,
      timeout=12.0,
      check=True,
  )
  assert "[001/100]" in proc_h2h.stdout
  assert "[100/100]" in proc_h2h.stdout
  assert "ROUTED WITH INTENT | BENCHMARK ROUTING STATS" in proc_h2h.stdout
  assert "[head-to-head:mock]" in proc_h2h.stdout

  mock_h2h_json = json.loads(mock_h2h_out.read_text(encoding="utf-8"))
  assert mock_h2h_json["metadata"]["mode"] == "mock"
  assert mock_h2h_json["metadata"]["dataset_size"] == 100
  assert (
      mock_h2h_json["strategies"]["decisions_router_full"]["tokens"][
          "total_output_tokens"
      ]
      == 0
  )
  assert (
      mock_h2h_json["strategies"]["flash_lite_json_router"]["tokens"][
          "total_output_tokens"
      ]
      > 0
  )

  # 2. --quiet suppresses per-prompt lines while preserving summary stats box on both runners
  mock_h2h_quiet = tmp_path / "mock_h2h_quiet.json"
  proc_h2h_q = subprocess.run(
      [
          sys.executable,
          "-B",
          str(h2h_runner),
          "--dataset",
          str(dataset_path),
          "--output",
          str(mock_h2h_quiet),
          "--mock",
          "--quiet",
      ],
      capture_output=True,
      text=True,
      timeout=12.0,
      check=True,
  )
  assert "[001/100]" not in proc_h2h_q.stdout
  assert "ROUTED WITH INTENT | BENCHMARK ROUTING STATS" in proc_h2h_q.stdout

  mock_bench_quiet = tmp_path / "mock_bench_quiet.json"
  proc_bench_q = subprocess.run(
      [
          sys.executable,
          "-B",
          str(bench_runner),
          "--dataset",
          str(dataset_path),
          "--output",
          str(mock_bench_quiet),
          "--mock",
          "--quiet",
      ],
      capture_output=True,
      text=True,
      timeout=12.0,
      check=True,
  )
  assert "[001/100]" not in proc_bench_q.stdout
  assert "ROUTED WITH INTENT | BENCHMARK ROUTING STATS" in proc_bench_q.stdout

  # 3. Validate live results_flash_lite_router_100.json schema, invariants, and hygiene
  raw_live_h2h = h2h_results.read_text(encoding="utf-8")
  assert ("AIza" + "Sy") not in raw_live_h2h
  assert ("ya" + "29.") not in raw_live_h2h
  live_h2h = json.loads(raw_live_h2h)
  assert live_h2h["metadata"]["mode"] == "live"
  assert live_h2h["metadata"]["dataset_size"] == 100
  assert live_h2h["metadata"]["flash_lite_router_model"] == (
      "gemini-3.1-flash-lite-preview"
  )
  assert live_h2h["metadata"]["flash_router_model"] == "gemini-2.5-flash"

  strats = live_h2h["strategies"]
  for key in (
      "decisions_router_full",
      "decisions_tier1_only",
      "flash_lite_json_router",
      "flash_json_router",
  ):
    assert key in strats
    assert strats[key]["sample_size"] == 100
    assert strats[key]["latency"]["count"] == 100

  assert strats["decisions_router_full"]["tokens"]["total_output_tokens"] == 0
  assert strats["decisions_tier1_only"]["tokens"]["total_output_tokens"] == 0
  assert strats["flash_lite_json_router"]["tokens"]["total_output_tokens"] > 0
  assert strats["flash_json_router"]["tokens"]["total_output_tokens"] > 0
  assert (
      strats["decisions_router_full"]["accuracy"]["pro_miss_rate_pct"] == 0.0
  )
  assert (
      strats["flash_lite_json_router"]["accuracy"]["pro_miss_rate_pct"] == 10.0
  )
  assert (
      strats["decisions_tier1_only"]["latency"]["p50_ms"]
      < strats["flash_lite_json_router"]["latency"]["p50_ms"]
  )
  assert (
      strats["decisions_router_full"]["calibration"]["populated_bins_4"] == 4
  )
  assert (
      strats["flash_lite_json_router"]["calibration"]["populated_bins_4"] == 1
  )

  # 4. Verify head-to-head 1600x1000 PNG chart asset and renderer script exist
  h2h_chart_script = REPO_ROOT / "benchmarks" / "render_head_to_head_chart.py"
  h2h_chart_png = REPO_ROOT / "assets" / "router-head-to-head.png"
  assert h2h_chart_script.is_file()
  assert h2h_chart_png.is_file()
  h2h_png_bytes = h2h_chart_png.read_bytes()
  assert len(h2h_png_bytes) > 50_000
  assert h2h_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
  # PNG IHDR stores big-endian width (bytes 16..20) and height (bytes 20..24)
  assert int.from_bytes(h2h_png_bytes[16:20], "big") == 1600
  assert int.from_bytes(h2h_png_bytes[20:24], "big") == 1000


def test_smart_thread_floor_quick_lookup_unlock_and_config(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies conceptual quick_lookup threshold, smart thread_lock, and config."""
  def fake_decision(call_args: dict[str, Any], _cfg: Any) -> dict[str, Any]:
    txt = str(call_args["input"][-1]["text"]).lower()
    if "navier stokes" in txt:
      return {
          "backend": "gemini_api",
          "model": "gemini-decision-1",
          "latency_ms": 190,
          "answers": {
              "task_class": {
                  "choice": "quick_lookup",
                  "confidence": 0.8516,
                  "probabilities": {"quick_lookup": 0.8238, "writing_docs": 0.14},
              },
              "needs_pro_reasoning": {"noul": 0.049, "confidence": 0.95},
              "effort_rubric": {"score": 1.037, "confidence": 0.82},
          },
      }
    if (
        "weather in chicago" in txt
        or "lexical baseline" in txt
        or "go ahead" in txt
    ):
      return {
          "backend": "gemini_api",
          "model": "gemini-decision-1",
          "latency_ms": 180,
          "answers": {
              "task_class": {
                  "choice": "quick_lookup",
                  "confidence": 0.93,
                  "probabilities": {"quick_lookup": 0.99},
              },
              "needs_pro_reasoning": {"noul": 0.01, "confidence": 0.98},
              "effort_rubric": {"score": 0.35, "confidence": 0.94},
          },
      }
    return {
        "backend": "gemini_api",
        "model": "gemini-decision-1",
        "latency_ms": 210,
        "answers": {
            "task_class": {
                "choice": "coding_edit",
                "confidence": 0.91,
                "probabilities": {"coding_edit": 0.94},
            },
            "needs_pro_reasoning": {"noul": 0.12, "confidence": 0.90},
            "effort_rubric": {"score": 1.65, "confidence": 0.88},
        },
    }

  # 1. Conceptual quick_lookup (eff_score ~1.04, p_pro < 0.25) routes to flash_lite
  r_navier = decisions_engine.route_prompt(
      "tell me about the navier stokes theorem",
      decision_fn=fake_decision,
      environ={},
  )
  assert r_navier["task_class"] == "quick_lookup"
  assert r_navier["subagent_tier"] == "flash_lite"

  # 2. A stale thread_state with locked_task_class == "quick_lookup" NEVER locks flash
  stale_quick_thread = {
      "locked_tier": "flash",
      "locked_model_id": "gemini-3-flash",
      "locked_thinking_level": 2,
      "locked_task_class": "quick_lookup",
  }
  r_weather = decisions_engine.route_prompt(
      "what is the current weather in chicago",
      user_turn_count=2,
      thread_state=stale_quick_thread,
      decision_fn=fake_decision,
      environ={},
  )
  assert r_weather["subagent_tier"] == "flash_lite"
  assert r_weather["thread_lock_applied"] is False

  # 3. Short standalone question (<=5 words with '?') is NOT hijacked by Tier 0 continuation
  pro_thread = {
      "locked_tier": "pro",
      "locked_model_id": "gemini-3.1-pro-high",
      "locked_thinking_level": 3,
      "locked_task_class": "deep_debug_architecture",
  }
  r_short_q = decisions_engine.route_prompt(
      "what is lexical baseline?",
      user_turn_count=2,
      thread_state=pro_thread,
      decision_fn=fake_decision,
      environ={},
  )
  assert r_short_q["subagent_tier"] == "flash_lite"
  assert r_short_q["thread_lock_applied"] is False
  assert r_short_q["backend"] == "gemini_api"

  # 4. Genuine short continuations ("yes go ahead", "do step 2 next") still hold pro floor
  for cont_prompt in ("yes go ahead", "do step 2 next"):
    r_cont = decisions_engine.route_prompt(
        cont_prompt,
        user_turn_count=2,
        thread_state=pro_thread,
        decision_fn=fake_decision,
        environ={},
    )
    assert r_cont["subagent_tier"] == "pro"
    assert r_cont["thread_lock_applied"] is True

  # 5. enable_thread_floor=False disables asymmetric floor via router_config.json & MCP
  cfg_path = tmp_path / "router_config.json"
  srv = mcp_server.McpServer(config_path=cfg_path)
  cfg_resp = json.loads(
      srv.handle_line(
          json.dumps({
              "jsonrpc": "2.0",
              "id": 77,
              "method": "tools/call",
              "params": {
                  "name": "configure_router",
                  "arguments": {"enable_thread_floor": False},
              },
          }).encode("utf-8")
      ).decode("utf-8")
  )
  assert cfg_resp["result"]["isError"] is False
  assert router_hook.load_router_config(cfg_path)["enable_thread_floor"] is False

  r_disabled_floor = decisions_engine.route_prompt(
      "yes go ahead",
      user_turn_count=2,
      thread_state=pro_thread,
      enable_thread_floor=False,
      decision_fn=fake_decision,
      environ={},
  )
  assert r_disabled_floor["subagent_tier"] == "flash_lite"
  assert r_disabled_floor["thread_lock_applied"] is False


def test_agy_cli_inline_banner_detection_and_toggle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies agy CLI detection, 1-line inline routing badge, and toggle."""
  monkeypatch.delenv("ANTIGRAVITY_VSCODE_HOST", raising=False)
  monkeypatch.delenv("GEMINI_DECISIONS_CLI_MODE", raising=False)

  cli_hook_input = {
      "conversationId": "cli-banner-conv-1",
      "invocationNum": 0,
      "modelName": "auto",
      "lastUserInput": "ship it",
      "transcriptPath": str(
          tmp_path / ".gemini" / "antigravity-cli" / "brain" / "c1" / "transcript.jsonl"
      ),
  }
  desktop_hook_input = {
      "conversationId": "desktop-banner-conv-1",
      "invocationNum": 1,
      "modelName": "auto",
      "lastUserInput": "ship it",
      "transcriptPath": str(
          tmp_path / ".gemini" / "antigravity" / "brain" / "d1" / "transcript.jsonl"
      ),
  }

  assert router_hook._is_agy_cli_environment(cli_hook_input) is True
  assert router_hook._is_agy_cli_environment(desktop_hook_input) is False

  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {"mode": "advisor", "verbosity": "normal", "cli_inline_banner": True},
      config_path=cfg_path,
  )

  # Step 1 in agy CLI injects the 1-line routing badge instruction in systemMessage
  out_cli = router_hook.handle_pre_invocation(
      cli_hook_input, state_dir=tmp_path, config_path=cfg_path
  )
  sys_msg = out_cli["injectSteps"][0]["systemMessage"]
  assert "metadata" not in sys_msg
  assert "> ✦ **Routed with Intent** -> `" in sys_msg["systemMessage"]
  assert "[Low]" in sys_msg["systemMessage"]

  # Step 2 (tool loop deduplication) returns {} so badge is never duplicated
  out_cli_step2 = router_hook.handle_pre_invocation(
      cli_hook_input, state_dir=tmp_path, config_path=cfg_path
  )
  assert out_cli_step2 == {}

  # Desktop App keeps systemMessage == "" (0 model tokens)
  out_desktop = router_hook.handle_pre_invocation(
      desktop_hook_input, state_dir=tmp_path, config_path=cfg_path
  )
  assert out_desktop["injectSteps"][0]["systemMessage"]["systemMessage"] == ""

  # Toggling cli_inline_banner=False via MCP configure_router suppresses CLI prompt tokens
  srv = mcp_server.McpServer(config_path=cfg_path)
  resp = json.loads(
      srv.handle_line(
          json.dumps({
              "jsonrpc": "2.0",
              "id": 88,
              "method": "tools/call",
              "params": {
                  "name": "configure_router",
                  "arguments": {"cli_inline_banner": False},
              },
          }).encode("utf-8")
      ).decode("utf-8")
  )
  assert resp["result"]["isError"] is False
  cli_hook_turn2 = dict(cli_hook_input, lastUserInput="lgtm")
  out_cli_off = router_hook.handle_pre_invocation(
      cli_hook_turn2, state_dir=tmp_path, config_path=cfg_path
  )
  assert out_cli_off["injectSteps"][0]["systemMessage"]["systemMessage"] == ""


def test_watch_and_stats_action_and_reason_labels() -> None:
  """Verifies ADVISE, CLI-ADVISE, SWITCH, and reason= labels in watch & stats."""
  events = [
      {
          "ts": "2026-11-19T02:39:22+00:00",
          "event": "PreInvocation",
          "prompt": "What is 2+2?",
          "mode": "advisor",
          "did_auto_switch": False,
          "recommended_model_id": "gemini-3.1-flash-lite",
          "subagent_tier": "flash_lite",
          "confidence": 0.96,
          "latency_ms": 180,
          "backend": "gemini_api",
          "thread_lock_applied": False,
      },
      {
          "ts": "2026-11-19T02:40:01+00:00",
          "event": "PreInvocation",
          "prompt": "Design consensus protocol",
          "mode": "auto_switch",
          "did_auto_switch": False,
          "auto_switch_fallback": "cascade_unowned",
          "recommended_model_id": "gemini-3.1-pro-high",
          "subagent_tier": "pro",
          "confidence": 0.94,
          "latency_ms": 220,
          "backend": "gemini_api",
          "thread_lock_applied": False,
      },
      {
          "ts": "2026-11-19T02:40:15+00:00",
          "event": "PreInvocation",
          "prompt": "yes go ahead",
          "mode": "auto_switch",
          "did_auto_switch": True,
          "recommended_model_id": "gemini-3.1-pro-high",
          "subagent_tier": "pro",
          "confidence": 0.92,
          "latency_ms": 1,
          "backend": "thread_lock (<1 ms)",
          "thread_lock_applied": True,
      },
  ]
  w0 = router_hook._format_watch_line(events[0], verbosity="normal")
  w1 = router_hook._format_watch_line(events[1], verbosity="normal")
  w2 = router_hook._format_watch_line(events[2], verbosity="normal")
  assert "ADVISE" in w0 and "reason=gemini_api" in w0
  assert "CLI-ADVISE" in w1 and "reason=gemini_api" in w1
  assert "SWITCH" in w2 and "reason=thread_lock" in w2

  stats_txt = router_hook._format_stats_pretty(
      events,
      by_tier={"flash_lite": 1, "pro": 2},
      avg_ms=134,
      verbosity="normal",
  )
  assert "CLI-ADVISE" in stats_txt
  assert "SWITCH" in stats_txt
  assert "reason=thread_lock" in stats_txt
  assert "reason=gemini_api" in stats_txt


def test_large_trajectory_ls_rpc_sibling_enum_propagation_and_proc_discovery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies >600 KiB _ls_rpc reads, sibling enum backfill, and /proc LS discovery."""
  # 1. >600 KiB GetCascadeTrajectory payload parses cleanly (no 256 KiB truncation)
  big_padding = "x" * 750_000  # ~750 KiB > old 262,144-byte limit

  class _LargeTrajectoryLsHandler(http.server.BaseHTTPRequestHandler):
    owned_cascade = "owned-big-cascade-1"

    def log_message(self, fmt: str, *args: Any) -> None:
      del fmt, args

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get("Content-Length", "0"))
      req_body = json.loads(self.rfile.read(length).decode("utf-8"))
      if self.headers.get("x-codeium-csrf-token") != "mock-big-csrf":
        self.send_response(403)
        self.end_headers()
        return
      if req_body.get("cascadeId") != self.owned_cascade:
        payload: dict[str, Any] = {"error": "cascade_not_found"}
      else:
        payload = {
            "status": "CASCADE_RUN_STATUS_IDLE",
            "trajectory": {"skillsBlob": big_padding},
        }
      raw = json.dumps(payload).encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(raw)))
      self.end_headers()
      self.wfile.write(raw)

  server1 = http.server.HTTPServer(("127.0.0.1", 0), _LargeTrajectoryLsHandler)
  server1_port = server1.server_port
  t1 = threading.Thread(target=server1.serve_forever, daemon=True)
  t1.start()

  class _SecondHubLsHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any) -> None:
      del fmt, args

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get("Content-Length", "0"))
      req_body = json.loads(self.rfile.read(length).decode("utf-8"))
      if (
          self.headers.get("x-codeium-csrf-token") == "mock-hub2-csrf"
          and req_body.get("cascadeId") == "owned-by-second-hub"
      ):
        payload: dict[str, Any] = {
            "status": "CASCADE_RUN_STATUS_RUNNING",
            "trajectory": {"skillsBlob": big_padding},
        }
      else:
        payload = {"error": "cascade_not_found"}
      raw = json.dumps(payload).encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(raw)))
      self.end_headers()
      self.wfile.write(raw)

  server2 = http.server.HTTPServer(("127.0.0.1", 0), _SecondHubLsHandler)
  server2_port = server2.server_port
  t2 = threading.Thread(target=server2.serve_forever, daemon=True)
  t2.start()

  try:
    monkeypatch.setenv("ANTIGRAVITY_LS_ADDRESS", f"127.0.0.1:{server1_port}")
    monkeypatch.setenv("ANTIGRAVITY_CSRF_TOKEN", "mock-big-csrf")

    rpc_res = router_hook._ls_rpc(
        "GetCascadeTrajectory",
        {"cascadeId": "owned-big-cascade-1"},
        timeout=2.0,
    )
    assert isinstance(rpc_res, dict)
    assert rpc_res["status"] == "CASCADE_RUN_STATUS_IDLE"
    assert len(rpc_res["trajectory"]["skillsBlob"]) == 750_000
    assert router_hook._verify_cascade_ownership("owned-big-cascade-1") is True
    assert router_hook._verify_cascade_ownership("unowned-cascade-99") is False

    # 2. Multi-endpoint /proc ephemeral LS discovery selects the second Hub port
    monkeypatch.delenv("ANTIGRAVITY_LS_ADDRESS", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_CSRF_TOKEN", raising=False)
    fake_proc = tmp_path / "fake_proc"
    (fake_proc / "net").mkdir(parents=True)
    port1_hex = f"{server1_port:04X}"
    port2_hex = f"{server2_port:04X}"
    (fake_proc / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when"
        " retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:{port1_hex} 00000000:0000 0A 00000000:00000000 00:00000000"
        f" 00000000  {os.getuid()}        0 11111 1 0000000000000000 100 0 0 10 0\n"
        f"   1: 0100007F:{port2_hex} 00000000:0000 0A 00000000:00000000 00:00000000"
        f" 00000000  {os.getuid()}        0 22222 1 0000000000000000 100 0 0 10 0\n",
        encoding="utf-8",
    )
    for pid_str, hub_port, csrf_val, sock_ino in (
        ("900001", server1_port, "mock-big-csrf", "11111"),
        ("900002", server2_port, "mock-hub2-csrf", "22222"),
    ):
      pdir = fake_proc / pid_str
      (pdir / "fd").mkdir(parents=True)
      (pdir / "cmdline").write_bytes(
          f"/opt/bin/agy\x00--hub\x00--hub-port={hub_port}\x00--csrf_token={csrf_val}\x00".encode(
              "utf-8"
          )
      )
      (pdir / "environ").write_bytes(b"PATH=/usr/bin\x00")
      (pdir / "status").write_text("PPid:\t1\n", encoding="utf-8")
      (pdir / "fd" / "3").symlink_to(f"socket:[{sock_ino}]")

    cands = decisions_engine.discover_local_ls_candidates(proc_root=fake_proc)
    assert (f"127.0.0.1:{server1_port}", "mock-big-csrf") in cands
    assert (f"127.0.0.1:{server2_port}", "mock-hub2-csrf") in cands

    monkeypatch.setattr(
        decisions_engine,
        "discover_local_ls_candidates",
        lambda: cands[:2],
    )
    assert router_hook._verify_cascade_ownership("owned-by-second-hub") is True
    assert router_hook._last_owned_ls_endpoint["owned-by-second-hub"] == (
        f"127.0.0.1:{server2_port}",
        "mock-hub2-csrf",
    )
  finally:
    server1.shutdown()
    server1.server_close()
    server2.shutdown()
    server2.server_close()

  # 3. Sibling -low/-medium/-high/-xhigh/-max MODEL_* enum propagation
  raw_siblings = [
      {
          "id": "gemini-5.0-flash-low",
          "label": "Gemini 5.0 Flash (Low)",
          "tier": "flash_lite",
          "enum": "MODEL_PLACEHOLDER_M999",
          "default_thinking_level": 1,
          "source": "discovered_ls",
      },
      {
          "id": "gemini-5.0-flash-medium",
          "label": "Gemini 5.0 Flash (Medium)",
          "tier": "flash",
          "enum": "",
          "default_thinking_level": 2,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-flash-high",
          "label": "Gemini 5.0 Flash (High)",
          "tier": "flash",
          "enum": "",
          "default_thinking_level": 3,
          "source": "discovered_cli",
      },
  ]
  propagated = decisions_engine.propagate_sibling_model_enums(raw_siblings)
  by_id = {m["id"]: m["enum"] for m in propagated}
  assert by_id["gemini-5.0-flash-low"] == "MODEL_PLACEHOLDER_M999"
  assert by_id["gemini-5.0-flash-medium"] == "MODEL_PLACEHOLDER_M999"
  assert by_id["gemini-5.0-flash-high"] == "MODEL_PLACEHOLDER_M999"

  # Built-in explicit distinct enums (gemini-3.1-pro-low vs gemini-3.1-pro-high) are never overwritten
  catalog = decisions_engine.list_available_target_models(
      discovered_models=raw_siblings, environ={}
  )
  cat_enums = {m["id"]: m["enum"] for m in catalog["available_target_models"]}
  assert cat_enums["gemini-5.0-flash-medium"] == "MODEL_PLACEHOLDER_M999"
  assert cat_enums["gemini-5.0-flash-high"] == "MODEL_PLACEHOLDER_M999"
  assert cat_enums["gemini-3.1-pro-low"] == "MODEL_GOOGLE_GEMINI_3_1_PRO_LOW"
  assert cat_enums["gemini-3.1-pro-high"] == "MODEL_GOOGLE_GEMINI_3_1_PRO"

  monkeypatch.setattr(
      router_hook, "_load_discovered_models_for_enums", lambda: raw_siblings
  )
  resolved = router_hook.resolve_model_enums({})
  assert resolved["gemini-5.0-flash-low"] == "MODEL_PLACEHOLDER_M999"
  assert resolved["gemini-5.0-flash-medium"] == "MODEL_PLACEHOLDER_M999"
  assert resolved["gemini-5.0-flash-high"] == "MODEL_PLACEHOLDER_M999"
  assert resolved["gemini-5.0-flash-xhigh"] == "MODEL_PLACEHOLDER_M999"
  assert resolved["gemini-5.0-flash-max"] == "MODEL_PLACEHOLDER_M999"


def test_resolve_cli_launch_target_and_sibling_effort_matching() -> None:
  """Verifies CLI model validation, fallback to discovered_cli, and sibling effort swapping."""
  assert decisions_engine.THINKING_EFFORT_FLAGS == {
      1: "low",
      2: "medium",
      3: "high",
      4: "max",
  }

  # 1. Hermetic mode (`environ={}` with no cache/agy_bin) avoids disk/subprocess I/O
  hermetic = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "flash_lite",
          "recommended_model_id": "gemini-3.1-flash-lite",
          "thinking_level": 1,
      },
      environ={},
  )
  assert hermetic == {
      "model_id": "gemini-3.1-flash-lite",
      "effort": "low",
      "tier": "flash_lite",
      "thinking_level": 1,
      "fallback_from": "",
  }

  # 2. Mixed LS-only + CLI models: excludes LS-only IDs and maps static IDs to discovered_cli
  discovered = [
      {
          "id": "gemini-pro-orchestrator-ls-only",
          "label": "Gemini Pro Orchestrator (LS Only)",
          "tier": "pro",
          "enum": "MODEL_LS_ONLY_ORCH",
          "default_thinking_level": 4,
          "source": "discovered_ls",
      },
      {
          "id": "gemini-5.0-flash-low",
          "label": "Gemini 5.0 Flash (Low)",
          "tier": "flash_lite",
          "default_thinking_level": 1,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-flash-medium",
          "label": "Gemini 5.0 Flash (Medium)",
          "tier": "flash",
          "default_thinking_level": 2,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-flash-high",
          "label": "Gemini 5.0 Flash (High)",
          "tier": "flash",
          "default_thinking_level": 3,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-pro-low",
          "label": "Gemini 5.0 Pro (Low)",
          "tier": "pro",
          "default_thinking_level": 2,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-pro-high",
          "label": "Gemini 5.0 Pro (High)",
          "tier": "pro",
          "default_thinking_level": 3,
          "source": "discovered_cli",
      },
      {
          "id": "gemini-5.0-pro-xhigh",
          "label": "Gemini 5.0 Pro (XHigh)",
          "tier": "pro",
          "default_thinking_level": 4,
          "source": "discovered_cli",
      },
  ]

  # Non-CLI static model ID (`gemini-3.1-flash-lite`) falls back to `gemini-5.0-flash-low`
  t_lite = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "flash_lite",
          "recommended_model_id": "gemini-3.1-flash-lite",
          "thinking_level": 1,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_lite == {
      "model_id": "gemini-5.0-flash-low",
      "effort": "low",
      "tier": "flash_lite",
      "thinking_level": 1,
      "fallback_from": "gemini-3.1-flash-lite",
  }

  # LS-only model ID (`gemini-pro-orchestrator-ls-only`) is rejected in favor of CLI model
  t_ls_rejected = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "pro",
          "recommended_model_id": "gemini-pro-orchestrator-ls-only",
          "thinking_level": 3,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_ls_rejected["model_id"] == "gemini-5.0-pro-high"
  assert t_ls_rejected["effort"] == "high"
  assert t_ls_rejected["fallback_from"] == "gemini-pro-orchestrator-ls-only"

  # 3. Sibling suffix swapping across thinking_level 1..4
  t_lvl1 = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "flash",
          "recommended_model_id": "gemini-5.0-flash-medium",
          "thinking_level": 1,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_lvl1["model_id"] == "gemini-5.0-flash-low"
  assert t_lvl1["effort"] == "low"
  assert t_lvl1["fallback_from"] == "gemini-5.0-flash-medium"

  t_lvl2 = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "flash",
          "recommended_model_id": "gemini-3-flash",
          "thinking_level": 2,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_lvl2["model_id"] == "gemini-5.0-flash-medium"
  assert t_lvl2["effort"] == "medium"
  assert t_lvl2["fallback_from"] == "gemini-3-flash"

  t_lvl3 = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "flash",
          "recommended_model_id": "gemini-5.0-flash-medium",
          "thinking_level": 3,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_lvl3["model_id"] == "gemini-5.0-flash-high"
  assert t_lvl3["effort"] == "high"
  assert t_lvl3["fallback_from"] == "gemini-5.0-flash-medium"

  # Level 4 (`max`) selects `-xhigh` when `-max` is absent
  t_lvl4_xhigh = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "pro",
          "recommended_model_id": "gemini-5.0-pro-high",
          "thinking_level": 4,
      },
      discovered_models=discovered,
      environ={},
  )
  assert t_lvl4_xhigh["model_id"] == "gemini-5.0-pro-xhigh"
  assert t_lvl4_xhigh["effort"] == "max"
  assert t_lvl4_xhigh["fallback_from"] == "gemini-5.0-pro-high"

  # Level 4 (`max`) falls back to `-high` when neither `-max` nor `-xhigh` exists
  discovered_no_xhigh = [
      m for m in discovered if m["id"] != "gemini-5.0-pro-xhigh"
  ]
  t_lvl4_high = decisions_engine.resolve_cli_launch_target(
      {
          "subagent_tier": "pro",
          "recommended_model_id": "gemini-3.1-pro-high",
          "thinking_level": 4,
      },
      discovered_models=discovered_no_xhigh,
      environ={},
  )
  assert t_lvl4_high["model_id"] == "gemini-5.0-pro-high"
  assert t_lvl4_high["effort"] == "max"
  assert t_lvl4_high["fallback_from"] == "gemini-3.1-pro-high"


def test_agyr_launcher_modes_and_pre_routed_handoff(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies agyr launch modes, install.sh artifacts, and 0-call pre-routed handoff."""
  state_dir = tmp_path / "state"
  state_dir.mkdir(parents=True)
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {"mode": "auto_switch", "verbosity": "normal", "cli_inline_banner": True},
      config_path=cfg_path,
  )

  # Seed synthetic CLI models in isolated state_dir cache
  cache_payload = {
      "updated_at": 9999999999.0,
      "cli_checked": True,
      "models": [
          {
              "id": "gemini-5.0-flash-low",
              "label": "Gemini 5.0 Flash (Low)",
              "tier": "flash_lite",
              "default_thinking_level": 1,
              "source": "discovered_cli",
          },
          {
              "id": "gemini-5.0-flash-medium",
              "label": "Gemini 5.0 Flash (Medium)",
              "tier": "flash",
              "default_thinking_level": 2,
              "source": "discovered_cli",
          },
          {
              "id": "gemini-5.0-pro-high",
              "label": "Gemini 5.0 Pro (High)",
              "tier": "pro",
              "default_thinking_level": 3,
              "source": "discovered_cli",
          },
      ],
  }
  (state_dir / "discovered_models.json").write_text(
      json.dumps(cache_payload), encoding="utf-8"
  )

  fake_agy = tmp_path / "fake-agy"
  fake_agy.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
  fake_agy.chmod(0o755)

  env = dict(os.environ)
  env["HOME"] = str(tmp_path)
  env["GEMINI_DECISIONS_STATE_DIR"] = str(state_dir)
  env["GEMINI_DECISIONS_ROUTER_CONFIG"] = str(cfg_path)
  env.pop("ANTIGRAVITY_LS_ADDRESS", None)

  def run_launch_dry(*extra_args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPO_ROOT / "router_hook.py"),
            "launch",
            "--dry-run",
            "--agy-bin",
            str(fake_agy),
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=8.0,
        check=True,
    )
    return json.loads(proc.stdout)

  # 1. `agyr "prompt"` -> interactive `-i` with calibrated model + effort
  plan_interactive = run_launch_dry("ship it")
  assert plan_interactive["action"] == "route_interactive"
  assert plan_interactive["agy_cmd"] == [
      str(fake_agy),
      "--model",
      "gemini-5.0-flash-low",
      "--effort",
      "low",
      "-i",
      "ship it",
  ]
  assert plan_interactive["pre_routed"]["inject_cli_badge"] is False

  # 2. `agyr -p "prompt"` -> print `-p` routing to pro + high effort
  plan_print = run_launch_dry(
      "-p", "Investigate a deadlock and race condition in the worker pool"
  )
  assert plan_print["action"] == "route_print"
  assert plan_print["agy_cmd"] == [
      str(fake_agy),
      "--model",
      "gemini-5.0-pro-high",
      "--effort",
      "high",
      "-p",
      "Investigate a deadlock and race condition in the worker pool",
  ]
  assert plan_print["target"]["tier"] == "pro"

  # 3. Bare `agyr` -> launches `--agent intent-router` on `flash_lite` (`--effort low`)
  plan_bare = run_launch_dry()
  assert plan_bare["action"] == "launch_tui_intent_router"
  assert plan_bare["agy_cmd"] == [
      str(fake_agy),
      "--agent",
      "intent-router",
      "--model",
      "gemini-5.0-flash-low",
      "--effort",
      "low",
  ]

  # 4. Explicit `--model` passthrough -> untouched
  plan_explicit = run_launch_dry(
      "--model", "claude-sonnet-4-6", "-p", "Explain this snippet"
  )
  assert plan_explicit["action"] == "passthrough_explicit_model"
  assert plan_explicit["route"] is None
  assert plan_explicit["agy_cmd"] == [
      str(fake_agy),
      "--model",
      "claude-sonnet-4-6",
      "-p",
      "Explain this snippet",
  ]

  # 5. `./install.sh` creates executable `agyr` + `agy-router` + `agents/intent-router.md` + `rules/AGENTS.md`
  plugins_dir = tmp_path / "plugins"
  bin_dir = tmp_path / "bin"
  install_env = dict(env)
  install_env["ANTIGRAVITY_PLUGINS_DIR"] = str(plugins_dir)
  install_env["BIN_DIR"] = str(bin_dir)
  install_env["PATH"] = "/usr/bin:/bin"
  install_proc = subprocess.run(
      ["bash", str(REPO_ROOT / "install.sh")],
      env=install_env,
      capture_output=True,
      text=True,
      timeout=10.0,
      check=True,
  )
  assert "Installed routed agy launcher to:" in install_proc.stdout
  installed_plugin = plugins_dir / "gemini-decisions-router"
  assert (installed_plugin / "agents" / "intent-router.md").is_file()
  assert (installed_plugin / "agents" / "routed-specialist.md").is_file()
  assert (installed_plugin / "rules" / "AGENTS.md").is_file()
  agyr_wrapper = bin_dir / "agyr"
  agy_router_wrapper = bin_dir / "agy-router"
  assert agyr_wrapper.is_file() and os.access(agyr_wrapper, os.X_OK)
  assert agy_router_wrapper.is_file() and os.access(agy_router_wrapper, os.X_OK)

  wrapper_dry = subprocess.run(
      [str(agyr_wrapper), "--dry-run", "--agy-bin", str(fake_agy), "ship it"],
      env=env,
      capture_output=True,
      text=True,
      timeout=8.0,
      check=True,
  )
  assert json.loads(wrapper_dry.stdout)["action"] == "route_interactive"

  # 6. `handle_pre_invocation` honors `GEMINI_DECISIONS_PRE_ROUTED` with 0 calls to `route_fn`
  route_calls = 0

  def forbidden_route_fn(*_a: Any, **_kw: Any) -> dict[str, Any]:
    nonlocal route_calls
    route_calls += 1
    raise AssertionError("route_fn should not be called when pre-routed!")

  prompt_text = "Investigate a deadlock and race condition in the worker pool"
  monkeypatch.setenv(
      "GEMINI_DECISIONS_PRE_ROUTED",
      json.dumps(plan_print["pre_routed"]),
  )
  hook_res = router_hook.handle_pre_invocation(
      {
          "conversationId": "pre-routed-conv-1",
          "invocationNum": 0,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": prompt_text,
          "transcriptPath": str(
              tmp_path
              / ".gemini"
              / "antigravity-cli"
              / "brain"
              / "c1"
              / "transcript.jsonl"
          ),
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=forbidden_route_fn,
  )
  assert route_calls == 0
  assert hook_res == {}
  assert not (state_dir / "telemetry.jsonl").exists()
  seeded_thread = json.loads(
      router_hook.safe_state_path(
          state_dir, "pre-routed-conv-1", "thread"
      ).read_text(encoding="utf-8")
  )
  assert seeded_thread["locked_tier"] == "pro"
  assert seeded_thread["locked_model_id"] == "gemini-5.0-pro-high"


def test_agy_router_chat_repl_multi_turn_switching(
    tmp_path: pathlib.Path,
) -> None:
  """Verifies multi-turn REPL hops flash_lite -> pro while keeping conversation continuity."""

  class _ChatDecisionsHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any) -> None:
      del fmt, args

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get("Content-Length", "0"))
      body = json.loads(self.rfile.read(length).decode("utf-8"))
      text = " ".join(
          str(part.get("text") or "") for part in body.get("input", [])
      ).lower()
      if "2+2" in text:
        resp = {
            "answers": {
                "task_class": {
                    "type": "choice",
                    "choice": "quick_lookup",
                    "confidence": 0.96,
                    "probabilities": {
                        "quick_lookup": 0.96,
                        "coding_edit": 0.01,
                        "writing_docs": 0.01,
                        "data_analysis": 0.01,
                        "deep_debug_architecture": 0.01,
                    },
                },
                "needs_pro_reasoning": {
                    "type": "noul",
                    "noul": 0.02,
                    "confidence": 0.97,
                },
                "effort_rubric": {
                    "type": "score",
                    "score": 0.2,
                    "confidence": 0.95,
                    "probabilities": {
                        "0": 0.85,
                        "1": 0.10,
                        "2": 0.03,
                        "3": 0.02,
                    },
                },
            },
            "usage": {"total_input_tokens": 24},
        }
      else:
        resp = {
            "answers": {
                "task_class": {
                    "type": "choice",
                    "choice": "deep_debug_architecture",
                    "confidence": 0.95,
                    "probabilities": {
                        "quick_lookup": 0.01,
                        "coding_edit": 0.02,
                        "writing_docs": 0.01,
                        "data_analysis": 0.01,
                        "deep_debug_architecture": 0.95,
                    },
                },
                "needs_pro_reasoning": {
                    "type": "noul",
                    "noul": 0.96,
                    "confidence": 0.96,
                },
                "effort_rubric": {
                    "type": "score",
                    "score": 2.9,
                    "confidence": 0.94,
                    "probabilities": {
                        "0": 0.01,
                        "1": 0.02,
                        "2": 0.07,
                        "3": 0.90,
                    },
                },
            },
            "usage": {"total_input_tokens": 36},
        }
      raw = json.dumps(resp).encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(raw)))
      self.end_headers()
      self.wfile.write(raw)

  server = http.server.HTTPServer(("127.0.0.1", 0), _ChatDecisionsHandler)
  port = server.server_port
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()

  try:
    state_dir = tmp_path / "chat_state"
    state_dir.mkdir(parents=True)
    cfg_path = tmp_path / "router_config.json"
    router_hook.save_router_config(
        {"mode": "auto_switch", "verbosity": "compact"},
        config_path=cfg_path,
    )
    (state_dir / "discovered_models.json").write_text(
        json.dumps({
            "updated_at": 9999999999.0,
            "cli_checked": True,
            "models": [
                {
                    "id": "gemini-5.0-flash-low",
                    "label": "Gemini 5.0 Flash (Low)",
                    "tier": "flash_lite",
                    "default_thinking_level": 1,
                    "source": "discovered_cli",
                },
                {
                    "id": "gemini-5.0-flash-medium",
                    "label": "Gemini 5.0 Flash (Medium)",
                    "tier": "flash",
                    "default_thinking_level": 2,
                    "source": "discovered_cli",
                },
                {
                    "id": "gemini-5.0-pro-high",
                    "label": "Gemini 5.0 Pro (High)",
                    "tier": "pro",
                    "default_thinking_level": 4,
                    "source": "discovered_cli",
                },
            ],
        }),
        encoding="utf-8",
    )

    invocations_log = tmp_path / "agy_invocations.jsonl"
    fake_agy = tmp_path / "fake_agy.py"
    fake_agy.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "argv = sys.argv[1:]\n"
        "conv_id = 'chat-conv-uuid-42'\n"
        "if '--conversation' in argv:\n"
        "  conv_id = argv[argv.index('--conversation') + 1]\n"
        "prompt = argv[argv.index('-p') + 1] if '-p' in argv else ''\n"
        "model = argv[argv.index('--model') + 1] if '--model' in argv else 'auto'\n"
        f"hook_res = subprocess.run([{sys.executable!r}, '-B', {str(REPO_ROOT / 'router_hook.py')!r}, 'pre-invocation'],\n"
        "  input=json.dumps({'conversationId': conv_id, 'invocationNum': 0, 'modelName': model, 'lastUserInput': prompt}),\n"
        "  text=True, capture_output=True, check=True)\n"
        f"with open({str(invocations_log)!r}, 'a', encoding='utf-8') as f:\n"
        "  f.write(json.dumps({'argv': argv, 'conv_id': conv_id, 'hook_out': json.loads(hook_res.stdout or '{}')}) + '\\n')\n"
        "sys.stdout.write(f'[{model}] ok\\n')\n",
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["GEMINI_DECISIONS_STATE_DIR"] = str(state_dir)
    env["GEMINI_DECISIONS_ROUTER_CONFIG"] = str(cfg_path)
    env["GEMINI_DECISIONS_AUTH_MODE"] = "api_key"
    env["GEMINI_API_KEY"] = "mock_chat_repl_key"
    env["GEMINI_DECISIONS_API_BASE"] = f"http://127.0.0.1:{port}/v1beta"
    env.pop("ANTIGRAVITY_LS_ADDRESS", None)

    repl_input = (
        "What is 2+2?\n"
        "Design a distributed Cloud Spanner failover consensus protocol\n"
        "/exit\n"
    )
    chat_proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPO_ROOT / "router_hook.py"),
            "chat",
            "--agy-bin",
            str(fake_agy),
        ],
        input=repl_input,
        env=env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=True,
    )

    calls = [
        json.loads(line)
        for line in invocations_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(calls) == 2

    # Turn 1: fresh launch on flash_lite (`gemini-5.0-flash-low`, `--effort low`)
    assert calls[0]["argv"] == [
        "--model",
        "gemini-5.0-flash-low",
        "--effort",
        "low",
        "-p",
        "What is 2+2?",
    ]
    assert calls[0]["hook_out"] == {}

    # Turn 2: hops to pro (`gemini-5.0-pro-high`, `--effort max`) while resuming `chat-conv-uuid-42`
    assert calls[1]["argv"] == [
        "--conversation",
        "chat-conv-uuid-42",
        "--model",
        "gemini-5.0-pro-high",
        "--effort",
        "max",
        "-p",
        "Design a distributed Cloud Spanner failover consensus protocol",
    ]
    assert calls[1]["hook_out"] == {}

    # Telemetry records exactly 2 `cli_chat` SWITCH events (0 duplicate hook scorings)
    telemetry_rows = [
        json.loads(line)
        for line in (state_dir / "telemetry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(telemetry_rows) == 2
    assert telemetry_rows[0]["mode"] == "cli_chat"
    assert telemetry_rows[0]["did_auto_switch"] is True
    assert telemetry_rows[0]["subagent_tier"] == "flash_lite"
    assert telemetry_rows[0]["recommended_model_id"] == "gemini-5.0-flash-low"
    assert telemetry_rows[1]["mode"] == "cli_chat"
    assert telemetry_rows[1]["did_auto_switch"] is True
    assert telemetry_rows[1]["subagent_tier"] == "pro"
    assert telemetry_rows[1]["recommended_model_id"] == "gemini-5.0-pro-high"
    assert "gemini-5.0-flash-low" in chat_proc.stderr
    assert "gemini-5.0-pro-high" in chat_proc.stderr
  finally:
    server.shutdown()
    server.server_close()


def test_honest_cli_badges_upgrade_directive_and_subagent_child_guard(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies all 4 honest CLI badges, upgrade directive, and 0-call child guard."""
  monkeypatch.setenv("GEMINI_DECISIONS_CLI_MODE", "1")
  monkeypatch.delenv("GEMINI_DECISIONS_PRE_ROUTED", raising=False)

  state_dir = tmp_path / "honest_cli_state"
  state_dir.mkdir(parents=True)
  cfg_path = tmp_path / "router_config.json"
  router_hook.save_router_config(
      {
          "mode": "auto_switch",
          "verbosity": "normal",
          "cli_inline_banner": True,
          "enable_subagent_routing": True,
          "tier_models": {
              "flash_lite": "gemini-5.0-flash-low",
              "flash": "gemini-5.0-flash-medium",
              "pro": "gemini-5.0-pro-high",
          },
      },
      config_path=cfg_path,
  )

  synth_pro_route: dict[str, Any] = {
      "recommended_model_id": "gemini-5.0-pro-high",
      "subagent_tier": "pro",
      "tier_rank": 3,
      "thinking_level": 3,
      "thinking_label": "High",
      "task_class": "deep_debug_architecture",
      "confidence": 0.94,
      "latency_ms": 120,
      "backend": "gemini_adc",
      "rationale": "Formal concurrency proof requires pro reasoning.",
      "probabilities": {},
      "thread_lock_applied": False,
      "explicit_model_requested": False,
  }
  synth_lite_route: dict[str, Any] = {
      "recommended_model_id": "gemini-5.0-flash-low",
      "subagent_tier": "flash_lite",
      "tier_rank": 1,
      "thinking_level": 1,
      "thinking_label": "Low",
      "task_class": "quick_lookup",
      "confidence": 0.97,
      "latency_ms": 4,
      "backend": "fast_rules (<1 ms)",
      "rationale": "Quick lookup.",
      "probabilities": {},
      "thread_lock_applied": False,
      "explicit_model_requested": False,
  }

  # 1. All 4 CLI badge states with synthetic external-safe model IDs
  badge_switched = router_hook._build_cli_badge_line(
      synth_pro_route,
      current_model="gemini-5.0-flash-low",
      switched=True,
  )
  assert badge_switched == (
      "> ✦ **Routed with Intent** (Switched -> `gemini-5.0-pro-high` [High])"
      " · 94% · 120 ms"
  )

  badge_subagent = router_hook._build_cli_badge_line(
      synth_pro_route,
      current_model="gemini-5.0-flash-low",
      switched=False,
      delegated_upgrade=True,
  )
  assert badge_subagent == (
      "> ✦ **Routed with Intent** (Subagent -> `gemini-5.0-pro-high` [High]"
      " · host: `gemini-5.0-flash-low`) · 94% · 120 ms"
  )

  badge_active = router_hook._build_cli_badge_line(
      synth_pro_route,
      current_model="gemini-5.0-pro-high",
      switched=False,
      delegated_upgrade=False,
  )
  assert badge_active == (
      "> ✦ **Routed with Intent** (Active: `gemini-5.0-pro-high` [High])"
      " · 94% · 120 ms"
  )

  badge_recommended = router_hook._build_cli_badge_line(
      synth_lite_route,
      current_model="gemini-5.0-pro-high",
      switched=False,
      delegated_upgrade=False,
  )
  assert badge_recommended == (
      "> ✦ **Routed with Intent** (Recommended: `gemini-5.0-flash-low` [Low]"
      " · active: `gemini-5.0-pro-high`) · 97% · 4 ms"
  )

  # Pre-routed Turn 1 (`inject_cli_badge=True`) emits State 1 (`Switched ->`)
  switched_prompt = "Design a multi-region consensus quorum protocol"
  switched_hash = router_hook.hashlib.sha256(
      switched_prompt.encode("utf-8")
  ).hexdigest()[:16]
  monkeypatch.setenv(
      "GEMINI_DECISIONS_PRE_ROUTED",
      json.dumps({
          "prompt_hash": switched_hash,
          "inject_cli_badge": True,
          "route": synth_pro_route,
      }),
  )
  out_switched = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-switched-1",
          "invocationNum": 1,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": switched_prompt,
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: (_ for _ in ()).throw(
          AssertionError("pre-routed turn must not call route_fn")
      ),
  )
  monkeypatch.delenv("GEMINI_DECISIONS_PRE_ROUTED", raising=False)
  switched_sys = out_switched["injectSteps"][0]["systemMessage"][
      "systemMessage"
  ]
  assert "(Switched -> `gemini-5.0-pro-high` [High])" in switched_sys
  assert "ROUTING UPGRADE DIRECTIVE" not in switched_sys

  # 2. File-less proof prompt on `flash_lite` host -> `Subagent -> ... · host:` + upgrade directive
  proof_prompt = (
      "Prove why a naive lock-free Treiber stack suffers from the ABA problem"
      " under concurrent pop/push in 5 concise lines, and tell me at the end"
      " what model you used"
  )
  out_upgrade = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-host-lite",
          "invocationNum": 1,
          "modelName": "gemini-5.0-flash-low",
          "lastUserInput": proof_prompt,
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: dict(synth_pro_route),
  )
  upgrade_sys = out_upgrade["injectSteps"][0]["systemMessage"]["systemMessage"]
  upgrade_ri = out_upgrade["injectSteps"][0]["systemMessage"]["renderInfo"]
  assert (
      "(Subagent -> `gemini-5.0-pro-high` [High] · host:"
      " `gemini-5.0-flash-low`)"
      in upgrade_sys
  )
  assert "ROUTING UPGRADE DIRECTIVE:" in upgrade_sys
  assert (
      "invoke_subagent(TypeName='routed-specialist', Role='Specialist',"
      " Model='inherit'"
      in upgrade_sys
  )
  assert "zero files" in upgrade_sys
  assert "`gemini-5.0-pro-high`" in upgrade_sys
  assert upgrade_ri["title"].startswith("Recommended: gemini-5.0-pro-high")

  # Verify bundled `routed-specialist.md` omits `invoke_subagent` / `define_subagent` tools
  specialist_md = (
      REPO_ROOT / "agents" / "routed-specialist.md"
  ).read_text(encoding="utf-8")
  specialist_frontmatter = specialist_md.split("---", 2)[1]
  assert "subagent: true" in specialist_frontmatter
  assert "mainAgent: false" in specialist_frontmatter
  assert "invoke_subagent" not in specialist_frontmatter
  assert "define_subagent" not in specialist_frontmatter

  # Host already on `pro` (`gemini-5.0-pro-high`) -> `Active:`, NO upgrade directive
  out_already_pro = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-host-pro",
          "invocationNum": 1,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": proof_prompt,
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: dict(synth_pro_route),
  )
  already_pro_sys = out_already_pro["injectSteps"][0]["systemMessage"][
      "systemMessage"
  ]
  assert "(Active: `gemini-5.0-pro-high` [High])" in already_pro_sys
  assert "ROUTING UPGRADE DIRECTIVE" not in already_pro_sys

  # Host on `pro` (`gemini-5.0-pro-high`) with `flash_lite` prompt -> `Recommended: ... · active:`, NO upgrade directive
  out_advisor_down = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-host-pro-lite-prompt",
          "invocationNum": 1,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": "What is 2+2?",
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: dict(synth_lite_route),
  )
  advisor_down_sys = out_advisor_down["injectSteps"][0]["systemMessage"][
      "systemMessage"
  ]
  assert (
      "(Recommended: `gemini-5.0-flash-low` [Low] · active:"
      " `gemini-5.0-pro-high`)"
      in advisor_down_sys
  )
  assert "ROUTING UPGRADE DIRECTIVE" not in advisor_down_sys

  # 3. `handle_pre_tool_use` writes `subagent_child` guard and child `handle_pre_invocation`
  #    short-circuits with `{}` (0 `route_fn` calls, 0 inner badges, 0 recursive directives)
  out_pre_tool = router_hook.handle_pre_tool_use(
      {
          "conversationId": "conv-host-lite",
          "lastUserInput": proof_prompt,
          "toolCall": {
              "name": "invoke_subagent",
              "args": {
                  "TypeName": "self",
                  "Role": "Specialist",
                  "Model": "inherit",
                  "Prompt": proof_prompt,
              },
          },
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: dict(synth_pro_route),
  )
  assert out_pre_tool["decision"] == "allow"
  assert out_pre_tool["overwrite"] == {"Model": "pro"}

  sub_hash = router_hook.hashlib.sha256(
      proof_prompt.encode("utf-8")
  ).hexdigest()[:16]
  guard_file = router_hook.safe_state_path(state_dir, sub_hash, "subagent_child")
  assert guard_file.is_file()
  guard_data = json.loads(guard_file.read_text(encoding="utf-8"))
  assert guard_data["prompt_hash"] == sub_hash
  assert guard_data["assigned_tier"] == "pro"

  child_route_calls = 0

  def forbidden_child_route(*_a: Any, **_kw: Any) -> dict[str, Any]:
    nonlocal child_route_calls
    child_route_calls += 1
    raise AssertionError("Child subagent must not call route_fn!")

  child_out = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-spawned-child-1",
          "invocationNum": 1,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": proof_prompt,
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=forbidden_child_route,
  )
  assert child_out == {}
  assert child_route_calls == 0
  assert not guard_file.exists()

  # Role-wrapped child prompt fallback also short-circuits with 0 calls to `route_fn`
  router_hook.handle_pre_tool_use(
      {
          "conversationId": "conv-host-lite",
          "lastUserInput": proof_prompt,
          "toolCall": {
              "name": "invoke_subagent",
              "args": {
                  "TypeName": "self",
                  "Role": "Concurrency Theorist",
                  "Model": "inherit",
                  "Prompt": proof_prompt,
              },
          },
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=lambda *_a, **_kw: dict(synth_pro_route),
  )
  wrapped_child_out = router_hook.handle_pre_invocation(
      {
          "conversationId": "conv-spawned-child-2",
          "invocationNum": 1,
          "modelName": "gemini-5.0-pro-high",
          "lastUserInput": f"[Role: Concurrency Theorist]\n\n{proof_prompt}",
      },
      state_dir=state_dir,
      config_path=cfg_path,
      route_fn=forbidden_child_route,
  )
  assert wrapped_child_out == {}
  assert child_route_calls == 0

