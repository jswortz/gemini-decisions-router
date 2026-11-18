"""Gemini Decisions API client and 2-tier calibrated model router for Antigravity.

Supports three external authentication modes using only the Python 3.11+
standard library (with optional `google-auth` acceleration when installed):
  1. `gemini_adc` (default keyless ADC): OAuth2 Bearer token + `x-goog-user-project`
     against `https://generativelanguage.googleapis.com/v1beta/decisions`.
  2. `vertex_adc` (keyless Vertex AI ADC): OAuth2 Bearer token against
     `https://{location-}aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/decisions`.
  3. `api_key` (optional sandbox fallback): `GEMINI_API_KEY` header against
     `https://generativelanguage.googleapis.com/v1beta/decisions`.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import dataclasses
import json
import math
import os
import random
import re
import shutil
import ssl
import stat
import subprocess
import threading
import time
from typing import Any, ClassVar, TypeGuard
import urllib.error
import urllib.parse
import urllib.request

__version__ = "1.0.0"


# ---- Errors -----------------------------------------------------------------


class DecisionError(Exception):
  """Base error for all Decisions API validation, config, and RPC failures."""

  code: ClassVar[str] = "BACKEND_ERROR"
  retryable: ClassVar[bool] = False

  def __init__(self, message: str, **details: str | int | float | bool) -> None:
    super().__init__(message)
    self.message = message
    self.details = details

  def to_dict(self) -> dict[str, Any]:
    out: dict[str, Any] = {
        "code": self.code,
        "message": self.message,
        "retryable": self.retryable,
    }
    for k, v in self.details.items():
      if k not in out:
        out[k] = v
    return out


class InvalidInputError(DecisionError):
  code = "INVALID_INPUT"


class ConfigError(DecisionError):
  code = "CONFIG_ERROR"


class PermissionDeniedError(DecisionError):
  code = "PERMISSION_DENIED"


class QuotaError(DecisionError):
  code = "QUOTA_EXHAUSTED"
  retryable = True


class DeadlineError(DecisionError):
  code = "TIMEOUT"
  retryable = True


class UnavailableError(DecisionError):
  code = "UNAVAILABLE"
  retryable = True


class SafetyBlockedError(DecisionError):
  code = "SAFETY_BLOCKED"


class IncompleteResponseError(DecisionError):
  code = "INCOMPLETE_RESPONSE"


class BackendError(DecisionError):
  code = "BACKEND_ERROR"


# ---- Constants & Scopes -----------------------------------------------------

AUTH_MODES = ("gemini_adc", "vertex_adc", "api_key")
ADC_SCOPES_GEMINI: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language.retriever",
)
ADC_SCOPES_VERTEX: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
)

_DEFAULT_MODEL = "gemini-decision-1"
_DEFAULT_LOCATION = "global"
_DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_VERTEX_API_VERSION = "v1beta1"
_DEFAULT_TIMEOUT_S = 15.0
_API_KEY_ENV = "GEMINI_API_KEY"
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_METADATA_BASE = "http://169.254.169.254/computeMetadata/v1"

_FORBIDDEN_CONFIG_KEYS = frozenset({
    "api_key",
    "gemini_api_key",
    "google_api_key",
    "access_token",
    "oauth_token",
    "refresh_token",
    "client_secret",
    "secret",
    "token",
    "password",
})

_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9.:-]{3,128}$")
_LOCATION_RE = re.compile(r"^[a-z0-9-]{2,64}$")
_API_VERSION_RE = re.compile(r"^v1(?:alpha1?|beta1?)?$")
_SECRET_NAME_RE = re.compile(
    r"(?:^|[._-])(env|secret|secrets|token|tokens|credential|credentials|"
    r"password|passwd|private|apikey|api_key|id_rsa|id_ed25519|kubeconfig|"
    r"netrc|pgpass)(?:$|[._-])|\.(?:pem|key|p12|pfx|jks|keystore)$",
    re.IGNORECASE,
)

_MAX_PARTS = 32
_MAX_FILES = 8
_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_FILE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_CHARS = 250_000
_MAX_QUESTIONS = 64
_MAX_OPTIONS = 255
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_READ_CHUNK = 16 * 1024
_SUM_TOLERANCE = 0.03
_QUESTION_TYPES = ("noul", "choice", "score")

_BLOCK_REASONS = {
    1: "SAFETY",
    2: "OTHER",
    3: "BLOCKLIST",
    4: "PROHIBITED_CONTENT",
    5: "IMAGE_SAFETY",
    6: "MODEL_ARMOR",
    7: "JAILBREAK",
}
_REST_SAFETY_CODES = frozenset(_BLOCK_REASONS.values())


# ---- Credentials & Settings -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdcToken:
  """Resolved OAuth2 Application Default Credentials access token."""

  access_token: str
  project_id: str = ""
  source: str = "adc"
  expires_at: float = 0.0


@dataclasses.dataclass(frozen=True)
class Settings:
  """Immutable configuration for the Gemini Decisions Router engine."""

  auth_mode: str
  model: str = _DEFAULT_MODEL
  project_id: str = ""
  location: str = _DEFAULT_LOCATION
  api_base: str = _DEFAULT_API_BASE
  vertex_api_version: str = _DEFAULT_VERTEX_API_VERSION
  vertex_base_override: str = ""
  timeout_s: float = _DEFAULT_TIMEOUT_S
  allowed_dirs: tuple[str, ...] = ()

  @property
  def label(self) -> str:
    if self.auth_mode == "vertex_adc":
      return f"vertex_adc:{self.location}"
    if self.auth_mode == "gemini_adc":
      return "gemini_adc"
    return "gemini_api"


def _default_config_path() -> str:
  home = os.path.expanduser("~")
  return os.path.join(home, ".config", "gemini-decisions-router", "config.json")


def _create_tls_context() -> ssl.SSLContext:
  ctx = ssl.create_default_context()
  ctx.minimum_version = ssl.TLSVersion.TLSv1_2
  return ctx


def _validate_api_base(api_base: str, field_name: str = "api_base") -> str:
  base = api_base.strip().rstrip("/")
  parsed = urllib.parse.urlsplit(base)
  host = (parsed.hostname or "").lower()
  if parsed.scheme == "https" and host:
    return base
  if parsed.scheme == "http" and host in ("127.0.0.1", "localhost"):
    return base
  raise ConfigError(
      f"{field_name} must use https:// (or http://127.0.0.1 for local testing)"
  )


def load_settings(
    *,
    config_path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
  """Loads and validates settings from optional config.json + env overrides."""
  env = os.environ if environ is None else environ
  cfg_path = (
      config_path
      or env.get("GEMINI_DECISIONS_CONFIG")
      or _default_config_path()
  )

  raw_cfg: dict[str, Any] = {}
  if cfg_path and os.path.isfile(cfg_path):
    try:
      with open(cfg_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    except (OSError, ValueError) as e:
      raise ConfigError(f"invalid config file {cfg_path}: {e}") from None
    if not isinstance(loaded, dict):
      raise ConfigError("config.json must be a JSON object")
    for k in loaded:
      if str(k).lower() in _FORBIDDEN_CONFIG_KEYS:
        raise ConfigError(
            f"Do not store {k!r} in config.json; use Application Default"
            " Credentials (ADC) or set GEMINI_API_KEY in the environment."
        )
    raw_cfg = loaded

  project_id = str(
      env.get("GOOGLE_CLOUD_PROJECT")
      or env.get("GCLOUD_PROJECT")
      or env.get("GOOGLE_CLOUD_QUOTA_PROJECT")
      or raw_cfg.get("project_id")
      or ""
  ).strip()
  if project_id and not _PROJECT_ID_RE.fullmatch(project_id):
    raise ConfigError(f"invalid GCP project_id format: {project_id!r}")

  location = (
      str(
          env.get("GOOGLE_CLOUD_LOCATION")
          or env.get("CLOUD_ML_REGION")
          or env.get("VERTEX_LOCATION")
          or raw_cfg.get("location")
          or _DEFAULT_LOCATION
      )
      .strip()
      .lower()
  )
  if not _LOCATION_RE.fullmatch(location):
    raise ConfigError(f"invalid Google Cloud location: {location!r}")

  raw_mode = (
      str(
          env.get("GEMINI_DECISIONS_AUTH_MODE")
          or raw_cfg.get("auth_mode")
          or "auto"
      )
      .strip()
      .lower()
  )
  if raw_mode == "gemini_api":
    raw_mode = "api_key"

  if raw_mode == "auto":
    use_vertex = (
        str(env.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
        in ("true", "1", "yes")
    )
    if use_vertex:
      auth_mode = "vertex_adc"
    elif str(env.get(_API_KEY_ENV) or "").strip():
      auth_mode = "api_key"
    else:
      auth_mode = "gemini_adc"
  elif raw_mode in AUTH_MODES:
    auth_mode = raw_mode
  else:
    raise ConfigError(
        f"auth_mode must be 'auto' or one of {AUTH_MODES}, got {raw_mode!r}"
    )

  model = str(
      env.get("GEMINI_DECISIONS_MODEL")
      or raw_cfg.get("model")
      or _DEFAULT_MODEL
  ).strip()
  if not model:
    model = _DEFAULT_MODEL

  raw_timeout = env.get("GEMINI_DECISIONS_TIMEOUT_S") or raw_cfg.get(
      "timeout_s", _DEFAULT_TIMEOUT_S
  )
  try:
    timeout_s = float(raw_timeout)
  except (TypeError, ValueError):
    raise ConfigError("timeout_s must be a positive number") from None
  if not math.isfinite(timeout_s) or not 0.5 <= timeout_s <= 120.0:
    raise ConfigError("timeout_s must be between 0.5 and 120.0 seconds")

  api_base = _validate_api_base(
      str(
          env.get("GEMINI_DECISIONS_API_BASE")
          or raw_cfg.get("api_base")
          or _DEFAULT_API_BASE
      ),
      "api_base",
  )

  vertex_api_version = (
      str(
          env.get("GEMINI_DECISIONS_VERTEX_API_VERSION")
          or raw_cfg.get("vertex_api_version")
          or _DEFAULT_VERTEX_API_VERSION
      )
      .strip()
      .lower()
  )
  if not _API_VERSION_RE.fullmatch(vertex_api_version):
    raise ConfigError(f"invalid vertex_api_version: {vertex_api_version!r}")

  raw_vtx_override = str(
      env.get("GEMINI_DECISIONS_VERTEX_BASE_OVERRIDE")
      or raw_cfg.get("vertex_base_override")
      or ""
  ).strip()
  vertex_base_override = (
      _validate_api_base(raw_vtx_override, "vertex_base_override")
      if raw_vtx_override
      else ""
  )

  raw_dirs = raw_cfg.get("allowed_dirs")
  if raw_dirs is None:
    env_dirs = env.get("GEMINI_DECISIONS_ALLOWED_DIRS", "").strip()
    allowed_dirs = tuple(
        d for d in (p.strip() for p in env_dirs.split(":")) if d
    )
  elif isinstance(raw_dirs, list) and all(isinstance(x, str) for x in raw_dirs):
    allowed_dirs = tuple(x.strip() for x in raw_dirs if x.strip())
  else:
    raise ConfigError("allowed_dirs must be a list of directory paths")

  for d in allowed_dirs:
    if not os.path.isabs(d):
      raise ConfigError(f"allowed_dirs entry must be absolute: {d!r}")

  return Settings(
      auth_mode=auth_mode,
      model=model,
      project_id=project_id,
      location=location,
      api_base=api_base,
      vertex_api_version=vertex_api_version,
      vertex_base_override=vertex_base_override,
      timeout_s=timeout_s,
      allowed_dirs=allowed_dirs,
  )


def build_decisions_url(
    settings: Settings, *, project_override: str = ""
) -> str:
  """Builds the verified HTTPS endpoint URL for the configured auth mode."""
  if settings.auth_mode == "vertex_adc":
    project = (project_override or settings.project_id).strip()
    if not project or not _PROJECT_ID_RE.fullmatch(project):
      raise ConfigError(
          "vertex_adc requires a valid GCP project ID. Export"
          " GOOGLE_CLOUD_PROJECT=your-project-id or configure ADC with a"
          " default project."
      )
    if not _LOCATION_RE.fullmatch(settings.location):
      raise ConfigError(f"invalid Vertex AI location: {settings.location!r}")
    if settings.vertex_base_override:
      base = _validate_api_base(
          settings.vertex_base_override, "vertex_base_override"
      )
    elif settings.location == "global":
      base = "https://aiplatform.googleapis.com"
    else:
      base = f"https://{settings.location}-aiplatform.googleapis.com"
    return (
        f"{base}/{settings.vertex_api_version}/projects/{project}"
        f"/locations/{settings.location}/decisions"
    )

  base = _validate_api_base(settings.api_base, "api_base")
  return f"{base}/decisions"


# ---- Keyless Application Default Credentials (ADC) Resolver -----------------


class _RedirectRefusedError(Exception):
  pass


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
  """Refuses all HTTP redirects so Bearer tokens and API keys stay on-origin."""

  def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
    del req, fp, msg, headers, newurl
    raise _RedirectRefusedError(code)


_ADC_LOCK = threading.Lock()
_ADC_CACHE: dict[tuple[tuple[str, ...], str], AdcToken] = {}


def _adc_file_candidates(env: Mapping[str, str]) -> list[str]:
  explicit = str(env.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
  if explicit:
    return [explicit]
  candidates = [
      os.path.join(
          os.path.expanduser("~"),
          ".config",
          "gcloud",
          "application_default_credentials.json",
      )
  ]
  appdata = env.get("APPDATA", "").strip()
  if appdata:
    candidates.append(
        os.path.join(appdata, "gcloud", "application_default_credentials.json")
    )
  return candidates


def _try_google_auth_library(scopes: tuple[str, ...]) -> AdcToken | None:
  """Uses `google-auth` if installed, via a stdlib urllib transport."""
  try:
    import google.auth  # type: ignore[import-not-found]
    import google.auth.transport  # type: ignore[import-not-found]
  except ImportError:
    return None

  class _StdlibAuthResponse(google.auth.transport.Response):  # type: ignore[misc]

    def __init__(
        self, status: int, headers: Mapping[str, str], data: bytes
    ) -> None:
      self._status = status
      self._headers = dict(headers)
      self._data = data

    @property
    def status(self) -> int:
      return self._status

    @property
    def headers(self) -> Mapping[str, str]:
      return self._headers

    @property
    def data(self) -> bytes:
      return self._data

  class _StdlibAuthRequest(google.auth.transport.Request):  # type: ignore[misc]

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> _StdlibAuthResponse:
      del kwargs
      parsed = urllib.parse.urlsplit(url)
      handlers: list[Any] = [_NoRedirectHandler()]
      if parsed.scheme == "https":
        handlers.append(
            urllib.request.HTTPSHandler(context=_create_tls_context())
        )
      elif (parsed.hostname or "") == "169.254.169.254":
        handlers.append(urllib.request.ProxyHandler({}))
      opener = urllib.request.build_opener(*handlers)
      req = urllib.request.Request(
          url,
          data=body,
          headers={str(k): str(v) for k, v in (headers or {}).items()},
          method=method,
      )
      try:
        with opener.open(req, timeout=timeout or 8.0) as resp:
          return _StdlibAuthResponse(
              resp.status, dict(resp.headers.items()), resp.read(262144)
          )
      except urllib.error.HTTPError as e:
        try:
          err_body = e.read(65536) if e.fp is not None else b""
        finally:
          e.close()
        return _StdlibAuthResponse(e.code, dict(e.headers.items()), err_body)

  try:
    creds, default_proj = google.auth.default(scopes=list(scopes))
    if not getattr(creds, "valid", False) or not getattr(creds, "token", None):
      creds.refresh(_StdlibAuthRequest())
    token = str(getattr(creds, "token", "") or "").strip()
    if not token:
      return None
    quota_proj = str(
        getattr(creds, "quota_project_id", "") or default_proj or ""
    ).strip()
    exp = getattr(creds, "expiry", None)
    expires_at = (
        exp.timestamp() if exp is not None else (time.time() + 1800.0)
    )
    return AdcToken(
        access_token=token,
        project_id=quota_proj,
        source="google.auth.default",
        expires_at=expires_at,
    )
  except Exception:  # pylint: disable=broad-except
    return None


def _try_authorized_user_adc_file(env: Mapping[str, str]) -> AdcToken | None:
  """Exchanges a user ADC refresh_token directly via oauth2.googleapis.com."""
  for path in _adc_file_candidates(env):
    if not os.path.isfile(path):
      continue
    try:
      with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    except (OSError, ValueError):
      continue
    if not isinstance(info, dict) or info.get("type") != "authorized_user":
      continue
    cid = str(info.get("client_id") or "").strip()
    csec = str(info.get("client_secret") or "").strip()
    rtok = str(info.get("refresh_token") or "").strip()
    qproj = str(info.get("quota_project_id") or "").strip()
    if not (cid and csec and rtok):
      continue
    form = urllib.parse.urlencode({
        "client_id": cid,
        "client_secret": csec,
        "refresh_token": rtok,
        "grant_type": "refresh_token",
    }).encode("utf-8")
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=_create_tls_context()),
    )
    req = urllib.request.Request(
        _OAUTH_TOKEN_URL,
        data=form,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
      with opener.open(req, timeout=8.0) as resp:
        data = json.loads(resp.read(65536).decode("utf-8"))
      tok = str(data.get("access_token") or "").strip()
      ttl = float(data.get("expires_in") or 3000.0)
      if tok:
        return AdcToken(
            access_token=tok,
            project_id=qproj,
            source="adc_authorized_user_file",
            expires_at=time.time() + ttl,
        )
    except (OSError, ValueError, _RedirectRefusedError):
      continue
  return None


def _try_compute_metadata_server(scopes: tuple[str, ...]) -> AdcToken | None:
  """Queries GCE / Cloud Run / GKE Workload Identity metadata server."""
  opener = urllib.request.build_opener(
      _NoRedirectHandler(), urllib.request.ProxyHandler({})
  )
  scope_param = urllib.parse.quote(",".join(scopes), safe=":,/")
  tok_url = f"{_METADATA_BASE}/instance/service-accounts/default/token?scopes={scope_param}"
  req = urllib.request.Request(
      tok_url, headers={"Metadata-Flavor": "Google"}, method="GET"
  )
  try:
    with opener.open(req, timeout=1.5) as resp:
      data = json.loads(resp.read(65536).decode("utf-8"))
  except (OSError, ValueError, _RedirectRefusedError):
    return None

  tok = str(data.get("access_token") or "").strip()
  if not tok:
    return None
  ttl = float(data.get("expires_in") or 1800.0)

  proj = ""
  proj_req = urllib.request.Request(
      f"{_METADATA_BASE}/project/project-id",
      headers={"Metadata-Flavor": "Google"},
      method="GET",
  )
  try:
    with opener.open(proj_req, timeout=1.0) as presp:
      proj = presp.read(256).decode("utf-8", "replace").strip()
  except (OSError, _RedirectRefusedError):
    pass

  return AdcToken(
      access_token=tok,
      project_id=proj,
      source="gce_metadata_server",
      expires_at=time.time() + ttl,
  )


def _try_gcloud_adc_cli(env: Mapping[str, str]) -> AdcToken | None:
  """Falls back to `gcloud auth application-default print-access-token`."""
  gcloud_bin = shutil.which("gcloud")
  if not gcloud_bin or not os.path.isabs(gcloud_bin):
    return None
  try:
    proc = subprocess.run(
        [gcloud_bin, "auth", "application-default", "print-access-token"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=8.0,
        check=False,
        env={k: v for k, v in env.items() if k != _API_KEY_ENV},
    )
  except (OSError, subprocess.SubprocessError):
    return None
  if proc.returncode != 0:
    return None
  tok = (proc.stdout or "").strip()
  if not tok or " " in tok:
    return None
  return AdcToken(
      access_token=tok,
      project_id="",
      source="gcloud_adc_cli",
      expires_at=time.time() + 1500.0,
  )


def resolve_adc_token(
    scopes: tuple[str, ...],
    *,
    project_hint: str = "",
    environ: Mapping[str, str] | None = None,
) -> AdcToken:
  """Resolves an OAuth2 ADC Bearer token without requiring any API key."""
  env = os.environ if environ is None else environ

  env_tok = str(env.get("GOOGLE_OAUTH_ACCESS_TOKEN") or "").strip()
  if env_tok:
    if any(ord(c) < 0x21 or ord(c) > 0x7E for c in env_tok):
      raise ConfigError("GOOGLE_OAUTH_ACCESS_TOKEN contains invalid characters")
    return AdcToken(
        access_token=env_tok,
        project_id=project_hint,
        source="env:GOOGLE_OAUTH_ACCESS_TOKEN",
        expires_at=time.time() + 600.0,
    )

  adc_hint = str(env.get("GOOGLE_APPLICATION_CREDENTIALS") or "")
  cache_key = (tuple(scopes), adc_hint)
  now = time.time()
  with _ADC_LOCK:
    cached = _ADC_CACHE.get(cache_key)
    if cached is not None and cached.expires_at - 60.0 > now:
      if project_hint and not cached.project_id:
        return dataclasses.replace(cached, project_id=project_hint)
      return cached

  resolved = (
      _try_google_auth_library(scopes)
      or _try_authorized_user_adc_file(env)
      or _try_compute_metadata_server(scopes)
      or _try_gcloud_adc_cli(env)
  )
  if resolved is None:
    raise ConfigError(
        "Could not locate Google Cloud Application Default Credentials (ADC)."
        " Run: gcloud auth application-default login"
        " --project=YOUR_PROJECT_ID"
        " --scopes=https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/generative-language.retriever"
    )

  if project_hint and not resolved.project_id:
    resolved = dataclasses.replace(resolved, project_id=project_hint)

  with _ADC_LOCK:
    _ADC_CACHE[cache_key] = resolved
  return resolved


# ---- Input & Question Validation --------------------------------------------


@dataclasses.dataclass(frozen=True)
class Part:
  kind: str  # "text" | "image" | "pdf"
  text: str = ""
  mime_type: str = ""
  data: bytes = b""


@dataclasses.dataclass(frozen=True)
class Question:
  key: str
  type: str  # "noul" | "choice" | "score"
  instructions: str
  criteria: Any  # tuple of (k, v) or tuple of str


def _is_num(x: Any) -> TypeGuard[int | float]:
  return (
      isinstance(x, (int, float))
      and not isinstance(x, bool)
      and math.isfinite(x)
  )


def _unit(x: Any) -> TypeGuard[int | float]:
  return _is_num(x) and 0.0 <= x <= 1.0


def validate_decision_args(
    arguments: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], list[Question], int]:
  """Validates `input` and `questions` arguments before file loading or HTTP."""
  if not isinstance(arguments, Mapping):
    raise InvalidInputError("arguments must be a JSON object")
  extra = set(arguments.keys()) - {"input", "questions"}
  if extra:
    raise InvalidInputError(f"unexpected top-level field(s): {sorted(extra)}")

  raw_input = arguments.get("input")
  if not isinstance(raw_input, list) or not 1 <= len(raw_input) <= _MAX_PARTS:
    raise InvalidInputError(f"input must be a list of 1..{_MAX_PARTS} items")

  raw_parts: list[tuple[str, str]] = []
  total_chars = 0
  for idx, item in enumerate(raw_input):
    if not isinstance(item, Mapping):
      raise InvalidInputError(f"input[{idx}] must be an object")
    has_text = "text" in item
    has_file = "file" in item
    if has_text == has_file or len(item) != 1:
      raise InvalidInputError(
          f"input[{idx}] must contain exactly one of 'text' or 'file'"
      )
    if has_text:
      val = item["text"]
      if not isinstance(val, str):
        raise InvalidInputError(f"input[{idx}].text must be a string")
      total_chars += len(val)
      if total_chars > _MAX_TOTAL_CHARS:
        raise InvalidInputError("input text exceeds maximum character budget")
      raw_parts.append(("text", val))
    else:
      val = item["file"]
      if not isinstance(val, str) or not val.strip():
        raise InvalidInputError(f"input[{idx}].file must be a non-empty path")
      raw_parts.append(("file", val))

  raw_questions = arguments.get("questions")
  if (
      not isinstance(raw_questions, Mapping)
      or not 1 <= len(raw_questions) <= _MAX_QUESTIONS
  ):
    raise InvalidInputError(
        f"questions must be an object with 1..{_MAX_QUESTIONS} entries"
    )

  questions: list[Question] = []
  for qkey, qspec in raw_questions.items():
    if not isinstance(qkey, str) or not _KEY_RE.fullmatch(qkey):
      raise InvalidInputError(f"invalid question key: {qkey!r}")
    if not isinstance(qspec, Mapping):
      raise InvalidInputError(f"questions.{qkey} must be an object")
    qtype = qspec.get("type")
    if qtype not in _QUESTION_TYPES:
      raise InvalidInputError(
          f"questions.{qkey}.type must be one of {_QUESTION_TYPES}"
      )
    instr = qspec.get("instructions")
    if not isinstance(instr, str) or not instr.strip():
      raise InvalidInputError(
          f"questions.{qkey}.instructions must be a non-empty string"
      )
    total_chars += len(instr)
    if total_chars > _MAX_TOTAL_CHARS:
      raise InvalidInputError("total text + instructions exceeds budget")

    crit = qspec.get("criteria")
    if qtype == "noul":
      if crit is None:
        parsed_crit: Any = ()
      elif isinstance(crit, Mapping):
        allowed_c = {"true_description", "false_description"}
        if set(crit.keys()) - allowed_c:
          raise InvalidInputError(
              f"questions.{qkey}.criteria only allows {sorted(allowed_c)}"
          )
        items = []
        for ck in ("true_description", "false_description"):
          if ck in crit:
            if not isinstance(crit[ck], str):
              raise InvalidInputError(
                  f"questions.{qkey}.criteria.{ck} must be str"
              )
            items.append((ck, crit[ck]))
        parsed_crit = tuple(items)
      else:
        raise InvalidInputError(f"questions.{qkey}.criteria must be an object")
    elif qtype == "choice":
      if not isinstance(crit, Mapping) or not 2 <= len(crit) <= _MAX_OPTIONS:
        raise InvalidInputError(
            f"questions.{qkey}.criteria must map 2..{_MAX_OPTIONS} options"
        )
      opts = []
      for ok, ov in crit.items():
        if not isinstance(ok, str) or not _KEY_RE.fullmatch(ok):
          raise InvalidInputError(
              f"questions.{qkey} option key {ok!r} must match [A-Za-z0-9_-]+"
          )
        if not isinstance(ov, str):
          raise InvalidInputError(
              f"questions.{qkey}.criteria.{ok} must be a string"
          )
        opts.append((ok, ov))
      parsed_crit = tuple(opts)
    else:  # score
      if (
          not isinstance(crit, list)
          or not 2 <= len(crit) <= _MAX_OPTIONS
          or not all(isinstance(x, str) and x.strip() for x in crit)
      ):
        raise InvalidInputError(
            f"questions.{qkey}.criteria must be 2..{_MAX_OPTIONS} non-empty"
            " level strings"
        )
      parsed_crit = tuple(crit)

    questions.append(
        Question(key=qkey, type=qtype, instructions=instr, criteria=parsed_crit)
    )

  return raw_parts, questions, total_chars


def _sniff_file_bytes(data: bytes, path: str, field: str) -> Part:
  if data.startswith(b"\x89PNG\r\n\x1a\n"):
    return Part(kind="image", mime_type="image/png", data=data)
  if data.startswith(b"\xff\xd8\xff"):
    return Part(kind="image", mime_type="image/jpeg", data=data)
  if data.startswith((b"GIF87a", b"GIF89a")):
    return Part(kind="image", mime_type="image/gif", data=data)
  if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
    return Part(kind="image", mime_type="image/webp", data=data)
  if data.startswith(b"%PDF-"):
    return Part(kind="pdf", mime_type="application/pdf", data=data)
  if b"\x00" in data:
    raise InvalidInputError(f"{field}: unsupported binary file {path!r}")
  try:
    text = data.decode("utf-8")
  except UnicodeDecodeError:
    raise InvalidInputError(f"{field}: text file is not valid UTF-8") from None
  return Part(kind="text", text=text)


def _safe_read_part_file(
    path: str, field: str, roots: list[str], budget: int
) -> Part:
  if not os.path.isabs(path):
    raise InvalidInputError(f"{field}: path must be absolute: {path!r}")
  if os.path.islink(path):
    raise InvalidInputError(f"{field}: symlink files are not allowed: {path!r}")

  real = os.path.realpath(path)
  matched_root: str | None = None
  for root in roots:
    try:
      if os.path.commonpath([root, real]) == root:
        if matched_root is None or len(root) > len(matched_root):
          matched_root = root
    except ValueError:
      continue

  if matched_root is None:
    raise InvalidInputError(
        f"{field}: {path!r} is not under an allowed directory"
    )

  rel = os.path.relpath(real, matched_root)
  if rel != ".":
    for comp in rel.split(os.sep):
      if comp.startswith("."):
        raise InvalidInputError(
            f"{field}: {path!r} is a hidden file or inside a hidden directory"
        )

  for candidate_name in (os.path.basename(path), os.path.basename(real)):
    if _SECRET_NAME_RE.search(candidate_name):
      raise InvalidInputError(
          f"{field}: {path!r} looks like a secret, credential, or key file"
      )

  flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
  try:
    fd = os.open(real, flags)
  except OSError as e:
    raise InvalidInputError(
        f"{field}: cannot open {path!r}: {e.strerror or 'error'}"
    ) from None

  try:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
      raise InvalidInputError(f"{field}: {path!r} is not a regular file")
    limit = min(_MAX_FILE_BYTES, budget)
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
      chunk = os.read(fd, min(_READ_CHUNK * 8, limit + 1 - total))
      if not chunk:
        break
      chunks.append(chunk)
      total += len(chunk)
  finally:
    os.close(fd)

  if total == 0:
    raise InvalidInputError(f"{field}: {path!r} is empty")
  if total > _MAX_FILE_BYTES or total > budget:
    raise InvalidInputError(f"{field}: {path!r} exceeds file size limit")
  return _sniff_file_bytes(b"".join(chunks), path, field)


def _materialize_parts(
    raw_parts: list[tuple[str, str]], settings: Settings, chars: int
) -> list[Part]:
  n_files = sum(1 for kind, _ in raw_parts if kind == "file")
  if n_files and not settings.allowed_dirs:
    raise InvalidInputError(
        "file input is disabled unless allowed_dirs is configured"
    )
  if n_files > _MAX_FILES:
    raise InvalidInputError(f"at most {_MAX_FILES} files per decision call")

  roots = (
      [os.path.realpath(d) for d in settings.allowed_dirs] if n_files else []
  )
  budget = _MAX_TOTAL_FILE_BYTES
  out: list[Part] = []
  for i, (kind, val) in enumerate(raw_parts):
    if kind == "text":
      out.append(Part(kind="text", text=val))
    else:
      part = _safe_read_part_file(val, f"input[{i}].file", roots, budget)
      budget -= len(part.data) if part.data else len(part.text.encode("utf-8"))
      if part.kind == "text":
        chars += len(part.text)
        if chars > _MAX_TOTAL_CHARS:
          raise InvalidInputError("total text exceeds maximum character budget")
      out.append(part)

  if not any(p.kind != "text" or p.text.strip() for p in out):
    raise InvalidInputError("input must not be blank whitespace")
  return out


# ---- REST Serialization & Response Validation -------------------------------


def _block_reason_name(reason: Any) -> str | None:
  if reason is None or (reason == 0 and not isinstance(reason, bool)):
    return None
  if isinstance(reason, bool):
    return "UNKNOWN" if reason else None
  if isinstance(reason, int):
    return _BLOCK_REASONS.get(reason, f"UNKNOWN_{reason}")
  if isinstance(reason, str):
    up = reason.strip().upper()
    if not up or up == "BLOCK_REASON_UNSPECIFIED":
      return None
    return up if re.fullmatch(r"[A-Z0-9_]{1,64}", up) else "UNKNOWN"
  return "UNKNOWN"


def _finish_answer(
    q: Question, qtype: Any, confidence: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
  def fail(why: str) -> IncompleteResponseError:
    return IncompleteResponseError(
        f"Invalid answer for question {q.key!r} ({why}).", question=q.key
    )

  if qtype != q.type:
    raise fail(f"expected type {q.type!r}, got {qtype!r}")
  if not _unit(confidence):
    raise fail("confidence missing or outside [0, 1]")
  conf = round(float(confidence), 4)

  if q.type == "noul":
    p = payload.get("noul")
    if not _unit(p):
      raise fail("noul probability missing or outside [0, 1]")
    p_round = round(float(p), 4)
    return {
        "type": "noul",
        "noul": p_round,
        "value": p_round >= 0.5,
        "confidence": conf,
    }

  probs = payload.get("probabilities")
  if not isinstance(probs, Mapping):
    raise fail("probabilities missing")

  if q.type == "choice":
    keys = [k for k, _ in q.criteria]
    choice = payload.get("choice")
    if not isinstance(choice, str) or choice not in keys:
      raise fail("choice is not one of the declared option keys")
    if set(probs.keys()) != set(keys):
      raise fail("probabilities keys do not match declared options")
    if not all(_unit(probs[k]) for k in keys):
      raise fail("probability outside [0, 1]")
    if abs(sum(float(probs[k]) for k in keys) - 1.0) > _SUM_TOLERANCE:
      raise fail("probabilities do not sum to 1.0")
    return {
        "type": "choice",
        "choice": choice,
        "confidence": conf,
        "probabilities": {k: round(float(probs[k]), 4) for k in keys},
    }

  n = len(q.criteria)
  score = payload.get("score")
  if not _is_num(score) or not 0.0 <= float(score) <= n - 1:
    raise fail(f"score outside [0, {n - 1}]")
  if set(probs.keys()) != set(range(n)):
    raise fail("score probabilities do not match levels 0..n-1")
  if not all(_unit(probs[i]) for i in range(n)):
    raise fail("score level probability outside [0, 1]")
  if abs(sum(float(probs[i]) for i in range(n)) - 1.0) > _SUM_TOLERANCE:
    raise fail("score probabilities do not sum to 1.0")
  rounded = [round(float(probs[i]), 4) for i in range(n)]
  best_level = max(range(n), key=lambda i: (rounded[i], -i))
  return {
      "type": "score",
      "score": round(float(score), 4),
      "level": best_level,
      "confidence": conf,
      "probabilities": {str(i): rounded[i] for i in range(n)},
  }


def _extract_usage(resp: Mapping[str, Any]) -> dict[str, int]:
  usage = resp.get("usage") or resp.get("usageMetadata")
  if isinstance(usage, Mapping):
    for k in ("total_input_tokens", "totalInputTokens", "promptTokenCount"):
      raw = usage.get(k)
      if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return {"input_tokens": raw}
      if isinstance(raw, str) and raw.isdigit():
        return {"input_tokens": int(raw)}
  return {}


def build_rest_request(
    model: str, parts: list[Part], questions: list[Question]
) -> bytes:
  inputs: list[dict[str, Any]] = []
  for p in parts:
    if p.kind == "text":
      inputs.append({"type": "text", "text": p.text})
    else:
      inputs.append({
          "type": p.kind,
          "mime_type": p.mime_type,
          "data": base64.b64encode(p.data).decode("ascii"),
      })
  qs: dict[str, Any] = {}
  for q in questions:
    item: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
    if q.type == "noul":
      if q.criteria:
        item["criteria"] = dict(q.criteria)
    elif q.type == "choice":
      item["criteria"] = dict(q.criteria)
    else:
      item["criteria"] = list(q.criteria)
    qs[q.key] = item
  return json.dumps(
      {"model": model, "input": inputs, "questions": qs}, ensure_ascii=False
  ).encode("utf-8")


def parse_rest_response(resp: Any, questions: list[Question]) -> dict[str, Any]:
  if not isinstance(resp, dict):
    raise BackendError("Decisions API returned non-object JSON")
  for fb_key in ("prompt_feedback", "promptFeedback"):
    fb = resp.get(fb_key)
    if isinstance(fb, Mapping):
      for br_key in ("block_reason", "blockReason"):
        bname = _block_reason_name(fb.get(br_key))
        if bname:
          raise SafetyBlockedError(
              f"Input blocked by safety filter ({bname}).", block_reason=bname
          )

  answers = resp.get("answers")
  if not isinstance(answers, Mapping):
    raise BackendError("Malformed answers in Decisions API response")

  out: dict[str, Any] = {}
  for q in questions:
    ans = answers.get(q.key)
    if not isinstance(ans, Mapping):
      raise IncompleteResponseError(
          f"Missing answer for question {q.key!r}.", question=q.key
      )
    payload = dict(ans)
    qtype = ans.get("type")
    if q.type == "score" and isinstance(ans.get("probabilities"), Mapping):
      int_probs: dict[int, Any] = {}
      for k, v in ans["probabilities"].items():
        if isinstance(k, str) and k.isdigit():
          int_probs[int(k)] = v
        elif isinstance(k, int) and not isinstance(k, bool):
          int_probs[k] = v
        else:
          qtype = "malformed"
      payload["probabilities"] = int_probs
    out[q.key] = _finish_answer(q, qtype, ans.get("confidence"), payload)
  return {"answers": out, "usage": _extract_usage(resp)}


def _call_decisions_http(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    *,
    auth_mode: str,
    project_id: str,
    backend_label: str,
    questions: list[Question],
    timeout_s: float,
) -> dict[str, Any]:
  parsed = urllib.parse.urlsplit(url)
  is_local = parsed.scheme == "http" and (parsed.hostname or "") in (
      "127.0.0.1",
      "localhost",
  )

  handlers: list[Any] = [_NoRedirectHandler()]
  if is_local:
    handlers.append(urllib.request.ProxyHandler({}))
  else:
    handlers.append(urllib.request.HTTPSHandler(context=_create_tls_context()))
  opener = urllib.request.build_opener(*handlers)
  req = urllib.request.Request(
      url,
      data=body,
      method="POST",
      headers=dict(headers),
  )

  try:
    with opener.open(req, timeout=timeout_s) as resp:
      status = resp.status
      raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    if status != 200:
      raise BackendError(f"Decisions API returned unexpected HTTP {status}")
    if len(raw) > _MAX_RESPONSE_BYTES:
      raise BackendError("Decisions API response exceeded size cap")
  except _RedirectRefusedError as e:
    raise BackendError(
        f"Decisions API returned HTTP redirect ({e.args[0]}); refused for"
        " credential safety"
    ) from None
  except urllib.error.HTTPError as e:
    try:
      err_bytes = e.read(16384) if e.fp is not None else b""
    finally:
      e.close()
    raw_err_text = err_bytes.decode("utf-8", "replace") if err_bytes else ""
    detail = ""
    if raw_err_text:
      try:
        err_obj = json.loads(raw_err_text)
        err_inner = err_obj.get("error") or {}
        msg = err_inner.get("message")
        if isinstance(msg, str) and msg.strip():
          detail = f": {msg.strip()[:240]}"
        for fk in ("code", "status"):
          fv = err_inner.get(fk)
          if isinstance(fv, str) and fv.upper() in _REST_SAFETY_CODES:
            raise SafetyBlockedError(
                f"Blocked by safety filter ({fv.upper()}).",
                block_reason=fv.upper(),
            )
      except SafetyBlockedError:
        raise
      except ValueError:
        pass

    if e.code == 400:
      raise InvalidInputError(f"Decisions API rejected request{detail}") from None

    if e.code in (401, 403):
      proj_hint = project_id or "YOUR_PROJECT_ID"
      if auth_mode == "gemini_adc" and (
          "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in raw_err_text
          or "insufficient authentication scopes" in raw_err_text.lower()
      ):
        raise PermissionDeniedError(
            "ADC token lacks the required Generative Language scope. Run:"
            f" gcloud auth application-default login --project={proj_hint}"
            " --scopes=https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/generative-language.retriever"
        ) from None
      if auth_mode == "gemini_adc" and (
          "USER_PROJECT_DENIED" in raw_err_text
          or "SERVICE_DISABLED" in raw_err_text
      ):
        raise PermissionDeniedError(
            "GCP project denied or Generative Language API disabled. Enable"
            " generativelanguage.googleapis.com and set GOOGLE_CLOUD_PROJECT"
            f"{detail}"
        ) from None
      hint = (
          "check GEMINI_API_KEY"
          if auth_mode == "api_key"
          else "check ADC credentials and IAM permissions"
      )
      raise PermissionDeniedError(
          f"Decisions API denied access (HTTP {e.code}; {hint}){detail}"
      ) from None

    if e.code == 404 and auth_mode == "vertex_adc" and (
        "CreateDecisionHttp" in raw_err_text or "not found" in raw_err_text.lower()
    ):
      raise ConfigError(
          "Vertex AI Decisions endpoint returned HTTP 404 (Method"
          " CreateDecisionHttp not found). If your GCP project is not yet"
          " enrolled in the Vertex AI Decisions preview allowlist, set"
          " GEMINI_DECISIONS_AUTH_MODE=gemini_adc to use keyless ADC on"
          " generativelanguage.googleapis.com immediately."
      ) from None

    if e.code == 429:
      raise QuotaError(
          f"Decisions API quota exhausted (HTTP 429){detail}"
      ) from None
    if e.code in (500, 502, 503, 504):
      raise UnavailableError(
          f"Decisions API unavailable (HTTP {e.code}){detail}"
      ) from None
    raise BackendError(f"Decisions API HTTP {e.code}{detail}") from None
  except TimeoutError:
    raise DeadlineError(
        f"Decisions API timed out after {timeout_s:.1f}s"
    ) from None
  except urllib.error.URLError as e:
    if isinstance(e.reason, TimeoutError):
      raise DeadlineError(
          f"Decisions API timed out after {timeout_s:.1f}s"
      ) from None
    raise UnavailableError(
        f"Cannot reach Decisions API: {type(e.reason).__name__}"
    ) from None

  try:
    resp_json = json.loads(raw.decode("utf-8"))
  except ValueError:
    raise BackendError("Decisions API returned invalid JSON") from None
  out = parse_rest_response(resp_json, questions)
  out["backend_label"] = backend_label
  return out


def _redact_secret(err: DecisionError, *secrets: str) -> None:
  for secret in secrets:
    if secret and secret in err.message:
      err.message = err.message.replace(secret, "[REDACTED]")
      err.args = (err.message,)


def create_decision(
    arguments: Mapping[str, Any],
    settings: Settings | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    token_provider: Callable[[tuple[str, ...]], AdcToken] | None = None,
) -> dict[str, Any]:
  """Evaluates calibrated `noul`, `choice`, and `score` questions in one pass."""
  env = os.environ if environ is None else environ
  cfg = settings or load_settings(environ=env)
  api_key = ""
  bearer_token = ""

  t0 = time.monotonic()
  try:
    raw_parts, questions, chars = validate_decision_args(arguments)
    parts = _materialize_parts(raw_parts, cfg, chars)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    wire_model = cfg.model

    if cfg.auth_mode == "api_key":
      effective_project = cfg.project_id
      api_key = str(env.get(_API_KEY_ENV) or "").strip()
      if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set in the environment. Prefer keyless ADC"
            " (GEMINI_DECISIONS_AUTH_MODE=gemini_adc) or export GEMINI_API_KEY."
        )
      if any(ord(c) < 0x21 or ord(c) > 0x7E for c in api_key):
        raise ConfigError(
            "GEMINI_API_KEY contains invalid whitespace/control characters"
        )
      headers["x-goog-api-key"] = api_key
      url = build_decisions_url(cfg)
      backend_label = "gemini_api"
    else:
      scopes = (
          ADC_SCOPES_VERTEX
          if cfg.auth_mode == "vertex_adc"
          else ADC_SCOPES_GEMINI
      )
      adc = (
          token_provider(scopes)
          if token_provider is not None
          else resolve_adc_token(
              scopes, project_hint=cfg.project_id, environ=env
          )
      )
      bearer_token = adc.access_token.strip()
      if not bearer_token or any(
          ord(c) < 0x21 or ord(c) > 0x7E for c in bearer_token
      ):
        raise ConfigError("Resolved ADC access token is empty or malformed")
      effective_project = (cfg.project_id or adc.project_id).strip()
      headers["Authorization"] = f"Bearer {bearer_token}"

      if cfg.auth_mode == "gemini_adc":
        if effective_project:
          headers["x-goog-user-project"] = effective_project
        url = build_decisions_url(cfg)
        backend_label = "gemini_adc"
      else:
        wire_model = cfg.model.removeprefix("models/")
        url = build_decisions_url(cfg, project_override=effective_project)
        backend_label = f"vertex_adc:{cfg.location}"

    payload = build_rest_request(wire_model, parts, questions)

    res: dict[str, Any] | None = None
    last_err: DecisionError | None = None
    for attempt in range(2):
      rem = cfg.timeout_s - (time.monotonic() - t0)
      if rem < 0.5:
        break
      try:
        res = _call_decisions_http(
            url,
            payload,
            headers,
            auth_mode=cfg.auth_mode,
            project_id=effective_project,
            backend_label=backend_label,
            questions=questions,
            timeout_s=rem,
        )
        break
      except (UnavailableError, QuotaError, DeadlineError) as e:
        last_err = e
        if attempt == 0:
          pause = 0.25 + random.uniform(0.0, 0.15)
          if cfg.timeout_s - (time.monotonic() - t0) - pause >= 1.0:
            time.sleep(pause)
            continue
        raise

    if res is None:
      raise last_err or DeadlineError(
          "Decisions API request exhausted timeout budget"
      )

    latency_ms = max(1, int(round((time.monotonic() - t0) * 1000)))
    return {
        "backend": res["backend_label"],
        "model": wire_model,
        "answers": res["answers"],
        "usage": res["usage"],
        "latency_ms": latency_ms,
    }
  except DecisionError as e:
    _redact_secret(e, api_key, bearer_token)
    raise


# ---- 2-Tier Calibrated Model Router -----------------------------------------

TIER_RANKS = {"flash_lite": 1, "flash": 2, "pro": 3}

AVAILABLE_TARGET_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "gemini-3.1-flash-lite",
        "label": "Gemini 3.1 Flash-Lite",
        "provider": "google",
        "tier": "flash_lite",
        "enum": "MODEL_GOOGLE_GEMINI_3_1_FLASH_LITE",
        "default_thinking_level": 1,
        "description": "Fastest, lowest-cost model for lookups, confirmations, and lightweight subagents.",
    },
    {
        "id": "gemini-3-flash",
        "label": "Gemini 3 Flash",
        "provider": "google",
        "tier": "flash",
        "enum": "MODEL_GOOGLE_GEMINI_3_0_FLASH",
        "default_thinking_level": 2,
        "description": "Balanced speed and coding quality for edits, refactors, tests, and docs.",
    },
    {
        "id": "gemini-3.1-pro-low",
        "label": "Gemini 3.1 Pro (Low)",
        "provider": "google",
        "tier": "pro",
        "enum": "MODEL_GOOGLE_GEMINI_3_1_PRO_LOW",
        "default_thinking_level": 2,
        "description": "Pro reasoning with lower thinking budget for faster architectural tasks.",
    },
    {
        "id": "gemini-3.1-pro-high",
        "label": "Gemini 3.1 Pro (High)",
        "provider": "google",
        "tier": "pro",
        "enum": "MODEL_GOOGLE_GEMINI_3_1_PRO",
        "default_thinking_level": 4,
        "description": "Frontier multi-step reasoning for system architecture, concurrency, and security.",
    },
    {
        "id": "gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash-Lite",
        "provider": "google",
        "tier": "flash_lite",
        "enum": "MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE",
        "default_thinking_level": 1,
        "description": "Gemini 2.5 ultra-fast lightweight target model.",
    },
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "provider": "google",
        "tier": "flash",
        "enum": "MODEL_GOOGLE_GEMINI_2_5_FLASH",
        "default_thinking_level": 2,
        "description": "Gemini 2.5 balanced speed and reasoning target model.",
    },
    {
        "id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "provider": "google",
        "tier": "pro",
        "enum": "MODEL_GOOGLE_GEMINI_2_5_PRO",
        "default_thinking_level": 3,
        "description": "Gemini 2.5 high-capability reasoning target model.",
    },
    {
        "id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6 (Thinking)",
        "provider": "anthropic",
        "tier": "pro",
        "enum": "MODEL_CLAUDE_4_6_SONNET_THINKING",
        "default_thinking_level": 3,
        "description": "Anthropic Claude Sonnet 4.6 target model in Antigravity.",
    },
    {
        "id": "claude-opus-4-6",
        "label": "Claude Opus 4.6 (Thinking)",
        "provider": "anthropic",
        "tier": "pro",
        "enum": "MODEL_CLAUDE_4_6_OPUS_THINKING",
        "default_thinking_level": 4,
        "description": "Anthropic Claude Opus 4.6 frontier reasoning target model in Antigravity.",
    },
    {
        "id": "gpt-oss-120b-medium",
        "label": "GPT-OSS 120B (Medium)",
        "provider": "openai",
        "tier": "flash",
        "enum": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
        "default_thinking_level": 2,
        "description": "Open-weight 120B target model in Antigravity.",
    },
)

MODEL_ENUMS: dict[str, str] = {
    m["id"]: str(m["enum"]) for m in AVAILABLE_TARGET_MODELS
}

DEFAULT_TIER_MODELS: dict[str, str] = {
    "flash_lite": "gemini-3.1-flash-lite",
    "flash": "gemini-3-flash",
    "pro": "gemini-3.1-pro-high",
}
TIER_MODELS = dict(DEFAULT_TIER_MODELS)

TIER_ENV_VARS: dict[str, str] = {
    "flash_lite": "GEMINI_DECISIONS_MODEL_FLASH_LITE",
    "flash": "GEMINI_DECISIONS_MODEL_FLASH",
    "pro": "GEMINI_DECISIONS_MODEL_PRO",
}


def resolve_tier_models(
    tier_models: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
  """Resolves active tier -> target model ID mapping (defaults < config < env)."""
  resolved = dict(DEFAULT_TIER_MODELS)
  if isinstance(tier_models, Mapping):
    for tier in TIER_RANKS:
      val = str(tier_models.get(tier) or "").strip()
      if val:
        resolved[tier] = val
  env = os.environ if environ is None else environ
  for tier, env_key in TIER_ENV_VARS.items():
    val = str(env.get(env_key) or "").strip()
    if val:
      resolved[tier] = val
  return resolved


def list_available_target_models(
    *,
    tier_models: Mapping[str, str] | None = None,
    custom_model_enums: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Returns all catalog and custom target models plus active tier assignments."""
  active = resolve_tier_models(tier_models, environ=environ)
  assigned_by_model: dict[str, list[str]] = {}
  for tier, model_id in active.items():
    assigned_by_model.setdefault(model_id, []).append(tier)

  models: list[dict[str, Any]] = []
  seen_ids: set[str] = set()
  for entry in AVAILABLE_TARGET_MODELS:
    mid = str(entry["id"])
    seen_ids.add(mid)
    item = dict(entry)
    item["assigned_tiers"] = assigned_by_model.get(mid, [])
    item["custom"] = False
    models.append(item)

  if isinstance(custom_model_enums, Mapping):
    for mid_raw, enum_raw in custom_model_enums.items():
      mid = str(mid_raw or "").strip()
      enum_val = str(enum_raw or "").strip()
      if not mid or mid in seen_ids:
        continue
      seen_ids.add(mid)
      models.append({
          "id": mid,
          "label": mid,
          "provider": "custom",
          "tier": (assigned_by_model.get(mid) or ["custom"])[0],
          "enum": enum_val,
          "default_thinking_level": 2,
          "description": "User-configured custom target model.",
          "assigned_tiers": assigned_by_model.get(mid, []),
          "custom": True,
      })

  for tier, mid in active.items():
    if mid not in seen_ids:
      seen_ids.add(mid)
      models.append({
          "id": mid,
          "label": mid,
          "provider": "custom",
          "tier": tier,
          "enum": "",
          "default_thinking_level": 2,
          "description": "User-configured target model override.",
          "assigned_tiers": assigned_by_model.get(mid, []),
          "custom": True,
      })

  env = os.environ if environ is None else environ
  env_overrides = {
      tier: str(env.get(env_key)).strip()
      for tier, env_key in TIER_ENV_VARS.items()
      if str(env.get(env_key) or "").strip()
  }
  return {
      "active_tier_models": active,
      "default_tier_models": dict(DEFAULT_TIER_MODELS),
      "env_overrides": env_overrides,
      "available_target_models": models,
  }
THINKING_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Max"}

_SHORT_CONFIRMATIONS = frozenset({
    "yes",
    "yes please",
    "ship it",
    "send it",
    "lgtm",
    "looks good",
    "status",
    "status?",
    "thanks",
    "thank you",
})

_THREAD_CONTINUATIONS = _SHORT_CONFIRMATIONS | frozenset({
    "ok",
    "okay",
    "sure",
    "go ahead",
    "do it",
    "proceed",
    "sounds good",
    "try that",
    "try it",
    "run it",
    "test it",
    "fix it",
    "please fix",
    "continue",
    "keep going",
})

_EXPLICIT_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\buse\s+(?:gemini[-\s]*)?flash[-\s]*lite\b", re.I),
        "flash_lite",
    ),
    (re.compile(r"\buse\s+(?:gemini[-\s]*)?flash\b", re.I), "flash"),
    (re.compile(r"\buse\s+(?:gemini[-\s]*)?pro\b", re.I), "pro"),
    (re.compile(r"\buse\s+claude\b", re.I), "pro"),
)

_ROUTER_QUESTIONS: dict[str, Any] = {
    "task_class": {
        "type": "choice",
        "instructions": (
            "Classify the primary engineering or workspace intent of this user"
            " request."
        ),
        "criteria": {
            "quick_lookup": (
                "Short status check, calendar/email lookup, simple factual"
                " question, or one-step confirmation."
            ),
            "coding_edit": (
                "Writing, editing, refactoring, scaffolding, or unit-testing"
                " code, scripts, configs, or plugins."
            ),
            "writing_docs": (
                "Drafting or editing prose documents, RFCs, emails, summaries,"
                " or slides."
            ),
            "data_analysis": (
                "SQL queries, BigQuery analysis, log inspection, metrics, or"
                " spreadsheet modeling."
            ),
            "deep_debug_architecture": (
                "Multi-component system architecture, subtle concurrency or"
                " security root-cause debugging, or complex trade-off design."
            ),
        },
    },
    "needs_pro_reasoning": {
        "type": "noul",
        "instructions": (
            "Does solving this request reliably require frontier multi-step"
            " reasoning, cross-file architecture design, security/concurrency"
            " verification, or subtle root-cause synthesis?"
        ),
        "criteria": {
            "true_description": (
                "Requires deep multi-step architecture, subtle bug analysis,"
                " security scrutiny, or high-stakes technical synthesis."
            ),
            "false_description": (
                "Can be handled cleanly by a fast model following clear rules"
                " or standard single-task execution."
            ),
        },
    },
    "effort_rubric": {
        "type": "score",
        "instructions": "Rate the thinking effort required for this request.",
        "criteria": [
            "Level 0: Trivial status ping, confirmation, or single lookup.",
            "Level 1: Standard single-file edit, short summary, or basic query.",
            "Level 2: Multi-step implementation, test suite, or structured doc.",
            "Level 3: Deep architectural design, security audit, or hard bug.",
        ],
    },
}


def _build_route_dict(
    *,
    task_class: str,
    tier: str,
    thinking_level: int,
    confidence: float,
    backend: str,
    latency_ms: int,
    rationale: str,
    thread_lock_applied: bool = False,
    explicit_model_requested: bool = False,
    probabilities: Mapping[str, Any] | None = None,
    tier_models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  t_lvl = max(1, min(4, int(thinking_level)))
  models = tier_models or TIER_MODELS
  return {
      "task_class": task_class,
      "recommended_tier": tier,
      "subagent_tier": tier,
      "recommended_model_id": str(models.get(tier) or TIER_MODELS[tier]),
      "tier_rank": TIER_RANKS[tier],
      "thinking_level": t_lvl,
      "thinking_label": THINKING_LABELS[t_lvl],
      "confidence": round(float(confidence), 4),
      "backend": backend,
      "latency_ms": latency_ms,
      "rationale": rationale,
      "thread_lock_applied": thread_lock_applied,
      "explicit_model_requested": explicit_model_requested,
      "probabilities": dict(probabilities or {}),
  }


def _apply_thread_floor(
    candidate: dict[str, Any],
    *,
    user_turn_count: int,
    thread_state: Mapping[str, Any] | None,
    tier_models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Prevents mid-thread model downgrades on Turn 2+ unless user overrode."""
  if (
      user_turn_count < 2
      or not thread_state
      or candidate.get("explicit_model_requested")
  ):
    return candidate

  locked_tier = str(thread_state.get("locked_tier") or "")
  if locked_tier not in TIER_RANKS:
    return candidate

  models = tier_models or TIER_MODELS
  if TIER_RANKS[locked_tier] > TIER_RANKS[candidate["subagent_tier"]]:
    locked_think = max(
        int(thread_state.get("locked_thinking_level", 2)),
        int(candidate["thinking_level"]),
    )
    out = dict(candidate)
    out["recommended_tier"] = locked_tier
    out["subagent_tier"] = locked_tier
    out["recommended_model_id"] = str(
        thread_state.get("locked_model_id")
        or models.get(locked_tier)
        or TIER_MODELS[locked_tier]
    )
    out["tier_rank"] = TIER_RANKS[locked_tier]
    out["thinking_level"] = locked_think
    out["thinking_label"] = THINKING_LABELS.get(locked_think, "High")
    out["thread_lock_applied"] = True
    out["rationale"] = (
        f"Preserved Turn 1 thread floor ({locked_tier}) to avoid mid-thread"
        " downgrade and KV-cache invalidation."
    )
    return out
  return candidate


def _fast_rule_route(
    prompt: str,
    *,
    user_turn_count: int,
    thread_state: Mapping[str, Any] | None,
    tier_models: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
  cleaned = re.sub(r"\s+", " ", prompt).strip()
  low = cleaned.lower().rstrip(".!")
  words = low.split()

  for rx, explicit_tier in _EXPLICIT_MODEL_PATTERNS:
    if rx.search(cleaned):
      think = (
          3
          if explicit_tier == "pro"
          else (2 if explicit_tier == "flash" else 1)
      )
      return _build_route_dict(
          task_class="explicit_override",
          tier=explicit_tier,
          thinking_level=think,
          confidence=0.99,
          backend="fast_rules (<1 ms)",
          latency_ms=1,
          rationale=f"User explicitly requested {explicit_tier} tier.",
          explicit_model_requested=True,
          tier_models=tier_models,
      )

  if (
      user_turn_count >= 2
      and thread_state
      and (low in _THREAD_CONTINUATIONS or len(words) <= 5)
  ):
    locked_tier = str(thread_state.get("locked_tier") or "flash")
    if locked_tier not in TIER_RANKS:
      locked_tier = "flash"
    locked_think = int(thread_state.get("locked_thinking_level", 2))
    out = _build_route_dict(
        task_class=str(thread_state.get("locked_task_class") or "coding_edit"),
        tier=locked_tier,
        thinking_level=locked_think,
        confidence=0.92,
        backend="thread_lock (<1 ms)",
        latency_ms=1,
        rationale=(
            f"Short Turn {user_turn_count} continuation inherits"
            f" {locked_tier} floor."
        ),
        thread_lock_applied=True,
        tier_models=tier_models,
    )
    if thread_state.get("locked_model_id"):
      out["recommended_model_id"] = str(thread_state["locked_model_id"])
    return out

  if low in _SHORT_CONFIRMATIONS:
    return _build_route_dict(
        task_class="quick_lookup",
        tier="flash_lite",
        thinking_level=1,
        confidence=0.93,
        backend="fast_rules (<1 ms)",
        latency_ms=1,
        rationale="Short operational confirmation or status ping.",
        tier_models=tier_models,
    )

  if any(
      marker in low
      for marker in ("race condition", "deadlock", "segfault", "stack trace")
  ):
    return _build_route_dict(
        task_class="deep_debug_architecture",
        tier="pro",
        thinking_level=3,
        confidence=0.91,
        backend="fast_rules (<1 ms)",
        latency_ms=1,
        rationale="Concurrency/crash diagnostic marker detected.",
        tier_models=tier_models,
    )

  return None


def _lexical_fallback_route(
    prompt: str, *, tier_models: Mapping[str, str] | None = None
) -> dict[str, Any]:
  low = prompt.lower()
  words = low.split()
  if any(
      k in low
      for k in (
          "architecture",
          "design",
          "root cause",
          "security",
          "distributed",
          "multi-agent",
          "trade-off",
          "tradeoff",
      )
  ):
    return _build_route_dict(
        task_class="deep_debug_architecture",
        tier="pro",
        thinking_level=3,
        confidence=0.70,
        backend="lexical_fallback (<1 ms)",
        latency_ms=1,
        rationale="Offline lexical fallback matched architecture/security terms.",
        tier_models=tier_models,
    )
  if any(
      k in low
      for k in (
          "implement",
          "refactor",
          "pytest",
          "plugin",
          "python",
          "function",
          "sql",
          "bigquery",
          "draft",
          "doc",
      )
  ):
    tier = "pro" if len(words) > 65 else "flash"
    think = 3 if tier == "pro" else 2
    return _build_route_dict(
        task_class="coding_edit",
        tier=tier,
        thinking_level=think,
        confidence=0.70,
        backend="lexical_fallback (<1 ms)",
        latency_ms=1,
        rationale="Offline lexical fallback matched implementation/doc terms.",
        tier_models=tier_models,
    )
  return _build_route_dict(
      task_class="quick_lookup",
      tier="flash_lite" if len(words) < 20 else "flash",
      thinking_level=1 if len(words) < 20 else 2,
      confidence=0.68,
      backend="lexical_fallback (<1 ms)",
      latency_ms=1,
      rationale="Offline lexical fallback default route.",
      tier_models=tier_models,
  )


def _cooldown_path() -> str:
  return os.path.join(
      os.path.expanduser("~"),
      ".cache",
      "gemini-decisions-router",
      "api_cooldown.json",
  )


def is_api_on_cooldown(
    cooldown_file: str | None = None, ttl_s: float = 45.0
) -> bool:
  path = cooldown_file or _cooldown_path()
  if not os.path.isfile(path):
    return False
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return (time.time() - float(data.get("ts", 0.0))) < ttl_s
  except (OSError, ValueError, TypeError):
    return False


def trip_api_cooldown(cooldown_file: str | None = None) -> None:
  path = cooldown_file or _cooldown_path()
  try:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump({"ts": time.time()}, f)
  except OSError:
    pass


def route_prompt(
    prompt: str,
    *,
    current_model: str = "auto",
    user_turn_count: int = 1,
    prev_context: str = "",
    thread_state: Mapping[str, Any] | None = None,
    enable_fast_rules: bool = True,
    ignore_cooldown: bool = False,
    cooldown_file: str | None = None,
    settings: Settings | None = None,
    environ: Mapping[str, str] | None = None,
    decision_fn: Callable[..., dict[str, Any]] | None = None,
    tier_models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Scores a prompt via Tier 0 fast rules + Tier 1 Gemini Decisions API."""
  del current_model  # Reserved for future catalog-aware cost comparison
  env = os.environ if environ is None else environ
  resolved_models = resolve_tier_models(tier_models, environ=env)
  text = (prompt or "").strip()
  if not text:
    return _build_route_dict(
        task_class="quick_lookup",
        tier="flash_lite",
        thinking_level=1,
        confidence=1.0,
        backend="fast_rules (<1 ms)",
        latency_ms=0,
        rationale="Empty prompt.",
        tier_models=resolved_models,
    )

  if enable_fast_rules:
    fast = _fast_rule_route(
        text,
        user_turn_count=user_turn_count,
        thread_state=thread_state,
        tier_models=resolved_models,
    )
    if fast is not None:
      return _apply_thread_floor(
          fast,
          user_turn_count=user_turn_count,
          thread_state=thread_state,
          tier_models=resolved_models,
      )

  context_parts = []
  if prev_context.strip():
    context_parts.append(
        {"text": f"Prior context:\n{prev_context.strip()[:600]}\n\n"}
    )
  context_parts.append({"text": f"User request:\n{text[:8000]}"})

  call_args = {"input": context_parts, "questions": _ROUTER_QUESTIONS}
  try:
    if decision_fn is not None:
      dec = decision_fn(call_args, settings)
    else:
      cfg = settings or load_settings(environ=env)
      if not ignore_cooldown and is_api_on_cooldown(cooldown_file):
        raise UnavailableError("Decisions API on temporary 45s cooldown")
      budget = 10.0 if ignore_cooldown else min(cfg.timeout_s, 3.8)
      router_cfg = dataclasses.replace(cfg, timeout_s=budget)
      try:
        dec = create_decision(call_args, router_cfg, environ=env)
      except (UnavailableError, QuotaError, DeadlineError):
        trip_api_cooldown(cooldown_file)
        raise
  except DecisionError:
    fallback = _lexical_fallback_route(text, tier_models=resolved_models)
    return _apply_thread_floor(
        fallback,
        user_turn_count=user_turn_count,
        thread_state=thread_state,
        tier_models=resolved_models,
    )

  answers = dec["answers"]
  tc_ans = answers["task_class"]
  pro_ans = answers["needs_pro_reasoning"]
  eff_ans = answers["effort_rubric"]

  task_class = str(tc_ans["choice"])
  p_pro = float(pro_ans["noul"])
  eff_score = float(eff_ans["score"])

  if (
      task_class == "deep_debug_architecture"
      or p_pro >= 0.55
      or eff_score >= 2.25
  ):
    tier = "pro"
    thinking_level = 4 if (p_pro >= 0.88 and eff_score >= 2.75) else 3
  elif task_class == "quick_lookup" and p_pro < 0.25 and eff_score < 0.85:
    tier = "flash_lite"
    thinking_level = 1
  else:
    tier = "flash"
    thinking_level = 2 if eff_score >= 0.9 else 1

  combined_conf = (
      0.45 * float(tc_ans["confidence"])
      + 0.35 * float(pro_ans["confidence"])
      + 0.20 * float(eff_ans["confidence"])
  )
  rationale = (
      f"task={task_class}, P(pro)={p_pro:.2f}, effort={eff_score:.2f}/3.0"
      f" via {dec['model']}"
  )
  candidate = _build_route_dict(
      task_class=task_class,
      tier=tier,
      thinking_level=thinking_level,
      confidence=combined_conf,
      backend=str(dec["backend"]),
      latency_ms=int(dec.get("latency_ms", 1)),
      rationale=rationale,
      probabilities={
          "task_class": tc_ans.get("probabilities", {}),
          "needs_pro_reasoning": p_pro,
          "effort_score": eff_score,
      },
      tier_models=resolved_models,
  )
  return _apply_thread_floor(
      candidate,
      user_turn_count=user_turn_count,
      thread_state=thread_state,
      tier_models=resolved_models,
  )
