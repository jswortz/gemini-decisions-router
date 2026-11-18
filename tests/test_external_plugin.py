"""TDD unit, ADC authentication, security, and external-hygiene tests."""

from __future__ import annotations

import http.server
import json
import pathlib
import re
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
  assert out_sub == {"decision": "allow", "overwrite": {"Model": "flash_lite"}}


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
  import os
  import subprocess

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


def test_docs_contain_no_em_or_en_dashes() -> None:
  doc_paths = [
      REPO_ROOT / "README.md",
      REPO_ROOT / "docs" / "CUSTOMER_ONBOARDING.md",
      REPO_ROOT / "docs" / "AUTHENTICATION_ADC.md",
      REPO_ROOT / "skills" / "gemini-decisions-router" / "SKILL.md",
  ]
  for doc in doc_paths:
    text = doc.read_text(encoding="utf-8")
    assert "\u2014" not in text, f"Em-dash found in {doc}"
    assert "\u2013" not in text, f"En-dash found in {doc}"
