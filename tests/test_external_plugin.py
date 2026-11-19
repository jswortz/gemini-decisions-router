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
  srv = mcp_server.McpServer()
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

  # 2. compact, normal, verbose: 0 model tokens + metadata.user_markdown populated
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
    user_md = sys_msg["metadata"]["user_markdown"]
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
  assert (
      "routed with intent"
      in step1["injectSteps"][0]["systemMessage"]["metadata"][
          "user_markdown"
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
  assert "routed with intent" in sys_msg["metadata"]["user_markdown"].lower()
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


def test_docs_contain_no_em_or_en_dashes() -> None:
  doc_paths = [
      REPO_ROOT / "README.md",
      REPO_ROOT / "docs" / "CUSTOMER_ONBOARDING.md",
      REPO_ROOT / "docs" / "AUTHENTICATION_ADC.md",
      REPO_ROOT / "docs" / "API_KEY_QUICKSTART.md",
      REPO_ROOT / "docs" / "TESTING_STRATEGY.md",
      REPO_ROOT / "skills" / "gemini-decisions-router" / "SKILL.md",
  ]
  for doc in doc_paths:
    text = doc.read_text(encoding="utf-8")
    assert "\u2014" not in text, f"Em-dash found in {doc}"
    assert "\u2013" not in text, f"En-dash found in {doc}"
