---
trigger: model_decision
description: "Contributor architecture map, security and hermeticity invariants, hook/MCP contracts, and TDD/benchmark commands when editing the gemini-decisions-router repository."
---

# AGENTS.md - Engineering & Architecture Guide

This document defines the architecture, non-negotiable security invariants, runtime hook/MCP contracts, and verification commands for contributors and coding agents working in `gemini-decisions-router`.

## 1. Repository Purpose & Architecture Map

**Gemini Decisions Router** is a zero-dependency Antigravity plugin, CLI (`agy-router`), and stdio MCP server that routes user prompts and `invoke_subagent` tasks using the **Gemini Decisions API** (`gemini-decision-1`). Instead of decoding JSON routing text token-by-token (`400-1,500 ms`), `gemini-decision-1` scores structured yes/no (`noul`), multiple-choice (`choice`), and rubric (`score`) questions in **one prefill pass with `0` output tokens** (`~120-250 ms`) across keyless Google Cloud Application Default Credentials (`gemini_adc`, `vertex_adc`) or `GEMINI_API_KEY` (`api_key`).

### Architecture Map

| Path | Responsibility |
| :--- | :--- |
| `decisions_engine.py` | Core runtime engine (pure Python 3.11+ stdlib). Implements `gemini_adc` / `vertex_adc` / `api_key` credential resolution, hardened `0600` `.env` loading (`load_private_dotenv`, `effective_environ`), the `CreateDecision` HTTP client (`create_decision`), input/file security validation, dynamic Antigravity Language Server (`GetAvailableModels`) and `agy models` auto-discovery (`discover_installed_models`, `suggest_tier_models`, `list_available_target_models`, `resolve_cli_launch_target`), and the 2-tier calibrated router (`route_prompt`). |
| `router_hook.py` | Antigravity lifecycle hook dispatcher (`pre-invocation`, `pre-tool-use`), `GEMINI_DECISIONS_PRE_ROUTED` zero-cost Turn-1 handoff, background Language Server turn redispatcher (`redispatch`), conversation state & JSONL telemetry manager, top-level routed CLI launcher (`launch` / `run` / `agyr`) and multi-turn REPL (`chat`), and the `agy-router` management CLI (`auth-setup`, `auth-check`, `status`, `route`, `watch`, `stats`, `set-mode`, `set-verbosity`, `list-models`, `sync-models`, `set-models`). |
| `agents/intent-router.md` | Custom Markdown main agent (`name: intent-router`, `mainAgent: true`, `subagent: false`) for `agy --agent intent-router` and bare `agyr`: answers lightweight `quick_lookup` turns directly in-turn on `flash_lite` and delegates substantive engineering tasks via `invoke_subagent(TypeName="self", Model="inherit")` for dynamic `PreToolUse` tier assignment. |
| `mcp_server.py` | Newline-delimited JSON-RPC 2.0 stdio MCP server (`McpServer`) exposing `create_decision`, `route_prompt`, `list_models`, and `configure_router` under the `decisions` server (`gemini-decisions-router_decisions` namespace). |
| `plugin.json` | Antigravity plugin manifest (`name: "gemini-decisions-router"`, `category: "agents"`, version, license, keywords, and `suggestedPrompts`). |
| `hooks.json` | Lifecycle hook registration wiring `PreInvocation` (`8s` timeout) and `PreToolUse` on matcher `invoke_subagent` (`6s` timeout) to `router_hook.py`. |
| `mcp_config.json` | MCP stdio server manifest invoking `python3 -E -s -B -u ${PLUGIN_ROOT}/mcp_server.py` (isolated environment, no user site-packages, no `.pyc` generation, unbuffered binary stdio). |
| `install.sh` | Verifies Python 3.11+, installs plugin files (including `agents/intent-router.md` and `rules/AGENTS.md`) into `${ANTIGRAVITY_PLUGINS_DIR:-~/.gemini/config/plugins}/gemini-decisions-router`, writes executable `${BIN_DIR:-~/.local/bin}/agy-router` and `${BIN_DIR:-~/.local/bin}/agyr` wrappers, and runs `agy plugin validate ... --strict` when `agy` is on `PATH`. |
| `skills/gemini-decisions-router/SKILL.md` | Agent skill instructions loaded by Antigravity covering MCP tool schemas, hook behavior, `agy-router` CLI subcommands, and safety rules. |
| `benchmarks/` | Evaluation harness: `prompts_100.json` (`N=100` stratified prompts across 6 categories + `N=15` E2E subset), `run_benchmark.py` (live API + deterministic offline `--mock` CI runner with live `stdout` routing stats), `run_router_head_to_head.py` (`N=100` `flash-lite` & `flash` JSON routers vs `gemini-decision-1` runner), `render_charts.py` (`1600x1040` Pillow dashboard generator outputting `assets/benchmark-dashboard.png`), `results_100.json`, and `results_flash_lite_router_100.json`. |
| `docs/` | Operator and architecture guides: `AUTHENTICATION_ADC.md`, `API_KEY_QUICKSTART.md`, `CUSTOMER_ONBOARDING.md`, `BENCHMARKS.md`, and `TESTING_STRATEGY.md`. |
| `tests/test_external_plugin.py` | Offline, hermetic TDD test suite covering unit logic, mock HTTP servers (Gemini ADC, Vertex ADC, API key, Language Server CSRF/RPCs), `.env` and cache permissions, file security, multi-step hook deduplication, anti-hang pre-flight checks, MCP protocol hardening, `--mock` benchmarks, and public-repo hygiene. |

---

## 2. Non-Negotiable Engineering & Security Invariants

Every change to this repository must preserve the following invariants:

### Pure Python 3.11+ Standard Library at Runtime
- **Zero mandatory `pip` packages:** `decisions_engine.py`, `router_hook.py`, `mcp_server.py`, and `benchmarks/run_benchmark.py` must import only from the Python 3.11+ standard library. Note that `mcp_config.json` runs `python3 -E -s -B -u`, which disables user site-packages at runtime.
- **Optional `google-auth` acceleration with stdlib fallback:** When `google-auth` is importable in a non-isolated environment, ADC token resolution uses it opportunistically; otherwise it falls back seamlessly to pure-stdlib OAuth2 refresh token exchange, `gcloud` CLI token fallback, and GCE / Cloud Run / GKE Metadata Server requests.
- **Dev/chart tooling separation:** `pytest` is used only for running tests, and `Pillow` (`PIL`) is imported only by `benchmarks/render_charts.py` when regenerating static benchmark charts.

### Secret & Filesystem Hardening
- **Never store secrets in JSON config files:** `decisions_engine.load_settings` rejects embedded secret keys (`api_key`, `gemini_api_key`, `google_api_key`, `access_token`, `oauth_token`, `refresh_token`, `client_secret`, `secret`, `token`, `password`) inside `config.json` by raising `ConfigError`. Keep `~/.config/gemini-decisions-router/router_config.json` strictly non-secret.
- **Private `.env` and cache files (`0700` dirs, `0600` files, symlink rejection):**
  - `~/.config/gemini-decisions-router/.env` (created via `agy-router auth-setup`) and `~/.cache/gemini-decisions-router/discovered_models.json` reside inside `0700` directories (`_ensure_private_dir`) and are written with strict `0600` permissions.
  - `load_private_dotenv` and model discovery cache helpers inspect `os.lstat` / `os.fstat` and open files with `O_NOFOLLOW`, raising `ConfigError` (or refusing the cache read/write) if a target is a symlink, is not a regular file, or has group/world permission bits (`(st_mode & 0o077) != 0`).
  - Active process environment variables always override `.env` entries.
- **Path-traversal-safe state paths:** Always construct per-conversation state file paths via `router_hook.safe_state_path(state_dir, conversation_id, prefix)`, which allows `^[A-Za-z0-9_-]{1,80}$` tokens directly and SHA-256 hashes any other string so untrusted `conversationId` values cannot traverse outside `state_dir`.
- **Local file input guards in `create_decision`:** File inputs must resolve inside `Settings.allowed_dirs`, open with `O_NOFOLLOW`, and pass regular-file, UTF-8, and size limits (`4 MiB` per file, `8 MiB` total). Symlinks, dotfiles (any path segment starting with `.`), and secret filenames or key extensions (`_SECRET_NAME_RE`, such as `.env`, `*secret*`, `*token*`, `*credential*`, `.pem`, `.key`, `id_rsa`, `kubeconfig`) must raise `InvalidInputError`.
- **Outbound HTTP & log hygiene:** Outbound `CreateDecision` calls disable HTTP redirects so `Authorization: Bearer` tokens and `GEMINI_API_KEY` headers never cross origins, enforce bounded response reads (`4 MiB`), and never leak raw credentials into exception messages, `stderr`, `telemetry.jsonl`, or `benchmarks/results_100.json`.

### Unit-Test Hermeticity Invariant
- Functions that inspect environment/config state (`effective_environ`, `load_settings`, `route_prompt`, `list_available_target_models`) must remain **100% hermetic** when unit tests pass an explicit `environ` mapping (such as `environ={}`).
- Specifically: when `environ is not None` and `dotenv_path` is `None`, `effective_environ` must return `environ` unchanged without reading `~/.config/gemini-decisions-router/.env`; similarly, when `environ is not None`, `discovered_cache_path is None`, and `refresh_discovered is False`, `list_available_target_models` must not read `~/.cache/gemini-decisions-router/discovered_models.json`.

### Public-Repo Hygiene (Enforced by CI)
- **Zero internal references (`test_zero_internal_google_references`):** Every tracked file in the repository is scanned by CI to ensure zero internal hostnames, internal RPC/build/auth tool names, workstation names, codenames, or internal shortlinks appear anywhere in code, tests, benchmarks, or docs. Use only public `*.googleapis.com` endpoints and public `gcloud` / `agy` CLI workflows.
- **Zero em-dashes or en-dashes in documentation (`test_docs_contain_no_em_or_en_dashes`):** Documentation and skill markdown files must contain **zero em-dashes (`\u2014`) and zero en-dashes (`\u2013`)**. Always use ASCII hyphens (`-`) or colons (`:`). When adding a new `.md` file, register its path in `test_docs_contain_no_em_or_en_dashes` in `tests/test_external_plugin.py`.
- **Interactive dropdown safety:** Never configure `gemini-decision-1` as an interactive chat model in Antigravity `settings.json`; it evaluates `CreateDecision` requests and intentionally produces `0` text tokens on `GenerateContent` chat calls.

---

## 3. Hook & MCP Runtime Contracts

### `PreInvocation` (`router_hook.py pre-invocation`)
- **System & slash-command fast exit:** Immediately returns `{}` when `mode == "off"`, when `lastUserInput` is empty after cleanup, or when `lastUserInput` contains `<SYSTEM_MESSAGE>` or starts with `/`, `[AGENT WAKE-UP]`, or `[CROSS-PROJECT AGENT MESSAGE]`.
- **2-Tier Calibrated Routing:**
  - **Tier 0 (`<1 ms`):** `_fast_rule_route` handles explicit user tier overrides (`"use gemini pro"`, `"use flash lite"`), short confirmations (`"lgtm"`, `"ship it"`), concurrency/crash markers (`"race condition"`, `"deadlock"`, `"segfault"`, `"stack trace"`), and short Turn 2+ continuations (`<= 5` words or `_THREAD_CONTINUATIONS`).
  - **Tier 1 (`gemini-decision-1`, `1.5s` hook budget, `0` output tokens):** Evaluates `task_class` (`choice`), `needs_pro_reasoning` (`noul`), and `effort_rubric` (`score`) in a single prefill pass under `hook_timeout_s` (default `1.5s`, configurable via `router_config.json` or `GEMINI_DECISIONS_HOOK_TIMEOUT_S`, clamped to `[0.3, 8.0]`). On transient timeout/quota/unavailable errors, trips a `45s` circuit-breaker (`api_cooldown.json`) and falls back to `<1 ms` `_lexical_fallback_route` so user turns never stall.
- **Per-message tool-loop deduplication & `GEMINI_DECISIONS_PRE_ROUTED` handoff (`<1 ms`):** Records `(conversationId, prompt_hash)` in `seen_turn_<conv>.json` (`600s` TTL). Within a multi-step agent tool loop where `lastUserInput` stays constant across Steps 2..N, `PreInvocation` short-circuits in `<1 ms` and returns `{}` with zero extra API calls and zero duplicate banners. When launched via `agyr` (`launch`/`run`) or `agy-router chat`, `GEMINI_DECISIONS_PRE_ROUTED` passes the Turn-1 route digest so `PreInvocation` seeds `thread_<conv>.json`, `active_turn_<conv>.json`, and `seen_turn_<conv>.json` without calling `route_fn` (`0` duplicate API calls and `0` duplicate telemetry entries). Switched Desktop App turns use `reroute_<conv>.json` (`600s` TTL) to render their switched banner once on Step 2 and return `{}` on Steps 3..N.
- **Asymmetric Turn 2+ thread floor (`enable_thread_floor`):** Persists `thread_<conv>.json` (`7d` TTL) for coding, data, docs, and architecture turns (`locked_task_class != "quick_lookup"`) and runs `_apply_thread_floor` on Turn 2+ so brief follow-ups (`"yes go ahead"`, `"do step 2 next"`) never downgrade an active higher-tier session (`pro` -> `flash_lite`), while standalone high-confidence (`>0.85`) `quick_lookup` questions cleanly unlock back to `flash_lite`.
- **Inline `"✦ Routed with Intent"` banners across Desktop App & `agy` CLI (`silent`, `compact`, `normal`, `verbose`):**
  - On the Antigravity Desktop App, injects `injectSteps` containing `"systemMessage": ""` (`0` model context tokens) and `renderInfo` (never attach a string `"metadata"` value on `HookSystemMessage`, which triggers a protobuf scrubber type-mismatch panic in Antigravity).
  - In `agy` CLI (`_is_agy_cli_environment` is `True` and `cli_inline_banner` is `True`), injects a concise Step-1 `systemMessage` badge instruction (`> ✦ **Routed with Intent** -> ...`) when `verbosity != "silent"`.
  - When `verbosity == "silent"`, returns `{}` (no `injectSteps`) while still recording the event in `~/.cache/gemini-decisions-router/telemetry.jsonl`.
- **Mandatory `GetCascadeTrajectory` ownership pre-flight (`auto_switch`):**
  - Before `auto_switch` can spawn background `redispatch` and return `{"terminationBehavior": "terminate"}`, `_verify_cascade_ownership(conv_id)` calls `GetCascadeTrajectory` (`400 ms` timeout) against the discovered Language Server to prove it owns `conversationId`.
  - In `agy` CLI (which uses an ephemeral Language Server) or whenever cascade ownership cannot be confirmed, `auto_switch` **must never return `terminate`**: it falls back cleanly to `advisor` mode, displays the inline banner, and logs `"auto_switch_fallback": "cascade_unowned"` in `telemetry.jsonl`.

### `PreToolUse` (`router_hook.py pre-tool-use`, matcher `invoke_subagent`)
- Fires on `invoke_subagent` calls. Returns `{"decision": "allow"}` unchanged if `mode == "off"`, `enable_subagent_routing` is `False`, `args.Model` is already explicitly set to a value other than `"inherit"`, the user asked for `"same model"` / `"inherit"`, or the active turn had an explicit model override.
- Otherwise scores the subagent `Role` + `Prompt` (or reuses the active turn's `subagent_tier` when `Prompt` is empty) and returns `{"decision": "allow", "overwrite": {"Model": "<flash_lite|flash|pro>"}}`, adding an inline `reason` notice (`✦ Routed with Intent -> subagent Model=...`) whenever `verbosity != "silent"`.
- Fails open (`{"decision": "allow"}`) on any malformed hook payload or unexpected exception.

### MCP Stdio Server (`mcp_server.py`, server key `decisions`)
- Implements strict newline-delimited JSON-RPC 2.0 over binary `stdin`/`stdout` (`MAX_LINE_BYTES = 16 MiB`, supporting MCP protocol versions `2024-11-05` through `2025-11-25`).
- Exposes 4 tools with `additionalProperties: False` input schemas and structured outputs (`content`, `structuredContent`, `isError`):
  1. `create_decision`: Evaluates 1 to 32 shared `text` or allowlisted `file` inputs across 1 to 64 `noul`, `choice`, and `score` questions (`0` output tokens).
  2. `route_prompt`: Classifies a prompt or subagent task into `flash_lite`, `flash`, or `pro` with `recommended_model_id`, `thinking_level` (`1..4`), `confidence`, and calibrated probabilities.
  3. `list_models`: Returns built-in, auto-discovered (`refresh: bool`), and custom target models, Antigravity model enums, `suggested_tier_models`, and `active_tier_models`.
  4. `configure_router`: Validates and persists `mode`, `verbosity`, `min_confidence`, `enable_subagent_routing`, `enable_thread_floor`, `cli_inline_banner`, and `tier_models` in `router_config.json`.
- Rejects non-standard JSON constants (`NaN`, `Infinity`, `-Infinity`) via `_reject_json_constant` and scrubs `GEMINI_API_KEY` and `GOOGLE_OAUTH_ACCESS_TOKEN` from `stderr` logs via `_log_stderr`.

---

## 4. Development, Verification & Benchmark Commands

Follow Test-Driven Development (TDD): add or update tests in `tests/test_external_plugin.py` first, implement the minimal pure-stdlib change, and verify all checks pass before committing.

### 1. Offline Hermetic Test Suite (Required Before Every Commit)
```bash
python3 -m pytest -v tests/test_external_plugin.py
```
- Must pass 100% offline with zero live network calls and remain hermetic even when `~/.config/gemini-decisions-router/.env` or `~/.cache/gemini-decisions-router/` is populated on disk.

### 2. Strict Plugin Manifest Validation
```bash
agy plugin validate . --strict
```

### 3. Local Install & `agyr` / `agy-router` CLI Workflows
```bash
# Install plugin (including agents/intent-router.md and rules/AGENTS.md) and ~/.local/bin/{agy-router,agyr}
./install.sh

# Launch agy with automatic pre-exec model + effort routing (or --dry-run JSON plan)
agyr "Refactor this FastAPI endpoint to validate pagination parameters"
agyr -p "What is 2+2?"
agyr                                     # Bare TUI mode: agy --agent intent-router --effort low
agy-router run --dry-run -p "What is 2+2?"

# Multi-turn terminal REPL switching --model and --effort across turns (/new, /exit)
agy-router chat
agy-router chat --conversation <conversation-id>
agy --agent intent-router

# Inspect router config, auth mode, Language Server status, and active tier models
agy-router status

# Persist workstation credentials into a private 0600 ~/.config/gemini-decisions-router/.env
agy-router auth-setup --from-env
printf '%s\n' "${GEMINI_API_KEY}" | agy-router auth-setup --api-key-stdin
agy-router auth-check --live

# Discover newly available Antigravity / agy CLI models and optionally adopt suggested_tier_models
agy-router sync-models
agy-router sync-models --apply-recommended
agy-router list-models --refresh
agy-router set-models --flash-lite gemini-3.1-flash-lite --flash gemini-3-flash --pro gemini-3.1-pro-high

# Configure routing mode, verbosity, and test live prompt routing
agy-router set-mode advisor
agy-router set-verbosity normal
agy-router route --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover" --no-fast-rules --pretty

# Monitor live routing telemetry or summarize historical distribution and latency
agy-router watch
agy-router stats
```

### 4. Benchmarks & Dashboard Chart Generation
```bash
# Deterministic offline CI benchmark verification (<0.5s, zero network calls, live stdout stats)
python3 benchmarks/run_benchmark.py \
  --dataset benchmarks/prompts_100.json \
  --output /tmp/mock_results.json \
  --mock

# Deterministic offline N=100 flash-lite & flash JSON head-to-head verification
python3 benchmarks/run_router_head_to_head.py \
  --dataset benchmarks/prompts_100.json \
  --output /tmp/mock_head_to_head.json \
  --mock

# Live API benchmark run (N=100 routing + N=15 E2E completions; requires ADC or GEMINI_API_KEY)
python3 benchmarks/run_benchmark.py \
  --dataset benchmarks/prompts_100.json \
  --output benchmarks/results_100.json

# Regenerate the 1600x1040 4-panel PNG dashboard (assets/benchmark-dashboard.png; requires Pillow)
python3 benchmarks/render_charts.py
```
