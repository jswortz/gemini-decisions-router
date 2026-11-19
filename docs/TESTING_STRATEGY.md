# Testing Strategy & Auto-Routing Verification

This document describes how **Gemini Decisions Router** lists and configures target models, how security boundaries, UI visibility, multi-step deduplication, and `auto_switch` anti-hang guards are verified in automated tests, and the live verification results generated from `telemetry.jsonl`.

## 1. Available Target Models & Configuration

List all built-in and custom target models along with active tier assignments:

```bash
agy-router list-models
```

Built-in target model catalog (10 models):

| Model ID | Provider | Default Tier | Antigravity Model Enum | Default Thinking |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.1-flash-lite` | google | `flash_lite` | `MODEL_GOOGLE_GEMINI_3_1_FLASH_LITE` | 1 (Low) |
| `gemini-3-flash` | google | `flash` | `MODEL_GOOGLE_GEMINI_3_0_FLASH` | 2 (Medium) |
| `gemini-3.1-pro-low` | google | `pro` | `MODEL_GOOGLE_GEMINI_3_1_PRO_LOW` | 2 (Medium) |
| `gemini-3.1-pro-high` | google | `pro` | `MODEL_GOOGLE_GEMINI_3_1_PRO` | 4 (Max) |
| `gemini-2.5-flash-lite` | google | `flash_lite` | `MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE` | 1 (Low) |
| `gemini-2.5-flash` | google | `flash` | `MODEL_GOOGLE_GEMINI_2_5_FLASH` | 2 (Medium) |
| `gemini-2.5-pro` | google | `pro` | `MODEL_GOOGLE_GEMINI_2_5_PRO` | 3 (High) |
| `claude-sonnet-4-6` | anthropic | `pro` | `MODEL_CLAUDE_4_6_SONNET_THINKING` | 3 (High) |
| `claude-opus-4-6` | anthropic | `pro` | `MODEL_CLAUDE_4_6_OPUS_THINKING` | 4 (Max) |
| `gpt-oss-120b-medium` | openai | `flash` | `MODEL_OPENAI_GPT_OSS_120B_MEDIUM` | 2 (Medium) |

### Configuring Target Models

Override tier mappings persistently via `agy-router` (stored in `~/.config/gemini-decisions-router/router_config.json`):

```bash
agy-router set-models \
  --flash-lite gemini-2.5-flash-lite \
  --flash gemini-3-flash \
  --pro claude-opus-4-6

# Register a custom model enum if needed:
agy-router set-models --custom-enum my-custom-model=MODEL_CUSTOM_FAST_V1

# Reset back to defaults:
agy-router set-models --reset
```

Or override temporarily via environment variables (highest precedence):

- `GEMINI_DECISIONS_MODEL_FLASH_LITE`
- `GEMINI_DECISIONS_MODEL_FLASH`
- `GEMINI_DECISIONS_MODEL_PRO`

Agents can also inspect and update target models at runtime via the `list_models` and `configure_router` MCP tools.

## 2. Testing Strategy & Hardening Verification

Run the full automated test suite and strict plugin validator:

```bash
python3 -m pytest -v tests/test_external_plugin.py
agy plugin validate . --strict
```

### A. Private `0600` `.env` Permission & Symlink Security (`test_private_dotenv_permissions_and_symlink_rejection`)

GUI-launched Antigravity Desktop App sessions rely on `~/.config/gemini-decisions-router/.env` (created by `agy-router auth-setup`) when shell profiles are not inherited. The test suite verifies strict filesystem hygiene on `.env`:
- **Valid `0600` `.env` Loading & Precedence:** Confirms `decisions_engine.load_private_dotenv` and `effective_environ` parse `KEY=VALUE`, `export KEY=VALUE`, comments, and quoted values when `.env` has `0600` permissions, while active process environment variables (`environ` / `os.environ`) override `.env` entries. Also verifies unit-test hermeticity: passing an explicit `environ` dict without `dotenv_path` never reads `~/.config/gemini-decisions-router/.env`.
- **Group/World Permission Rejection:** Confirms a `0644` (or any `(st_mode & 0o077) != 0`) `.env` file raises `decisions_engine.ConfigError` instructing `chmod 600`.
- **Symlink Rejection:** Confirms `os.lstat` detects symlinked `.env` targets (`stat.S_ISLNK`) and raises `decisions_engine.ConfigError` before opening the file.
- **Redacted `auth-setup` CLI:** Verifies `agy-router auth-setup` creates a `0700` parent directory and `0600` `.env` file without printing secret values to `stdout` or `stderr`.

### B. Universal `metadata.user_markdown` Banners & 4-Level Verbosity (`test_user_markdown_inline_banners_and_four_verbosity_levels`)

Because `renderInfo` only renders on certain builds whereas `systemMessage.metadata.user_markdown` renders inline in **both** the Antigravity Desktop App and `agy` CLI, tests exercise all four verbosity levels (`silent`, `compact`, `normal`, `verbose`):
- **`silent`:** Verifies `handle_pre_invocation` returns `{}` (zero injected UI steps) while still appending the full routing record to `telemetry.jsonl`, and verifies `handle_pre_tool_use` (`invoke_subagent`) omits `reason` while still applying `"overwrite": {"Model": "<tier>"}`.
- **`compact`, `normal`, and `verbose`:** Verifies `injectSteps[0]["systemMessage"]["systemMessage"] == ""` (guaranteeing **0 model context tokens** consumed) and confirms `injectSteps[0]["systemMessage"]["metadata"]["user_markdown"]` contains `"Routed with Intent"` at the requested detail level (single-line summary for `compact`, `ROUTED WITH INTENT` box + rationale for `normal`, and class probabilities + `P(pro reasoning)` + `Effort Rubric` + `Thread Lock` for `verbose`).
- **Subagent Right-Sizing Notice:** Confirms `PreToolUse` (`invoke_subagent`) includes `"Routed with Intent"` in `reason` whenever `verbosity != "silent"`.

### C. Multi-Step `PreInvocation` Deduplication (`<1 ms` on Steps 2..N)

Antigravity invokes `PreInvocation` before **every** model step inside a multi-step tool loop with the same `(conversationId, lastUserInput)`. Two complementary guards ensure zero wasted latency or loops:
- **Advisor Deduplication (`test_advisor_deduplicates_multi_step_tool_loops`):** Calling `handle_pre_invocation` three times with the same `(conversationId, lastUserInput)` invokes `route_fn` **only once** on Step 1 (persisting `seen_turn_<conv>.json` with `prompt_hash`, `600s` TTL). Step 1 returns the `injectSteps` banner, while Steps 2 and 3 short-circuit before `route_fn` in `<1 ms` and return `{}` (0 API calls, 0 duplicate banners). Submitting a new `lastUserInput` on Turn 2 scores fresh and injects a new banner.
- **Auto-Switch Loop Guard (`test_auto_switch_prevents_multi_step_loop`):** When `auto_switch` terminates Step 1 to redispatch on a new model, `reroute_<conv>.json` transitions `rendered: false` -> `rendered: true` (rendering the switched banner on Step 2) -> `{}` on Steps 3..N (`<1 ms`), preventing re-termination loops even if `modelName` remains unchanged.

### D. `GetCascadeTrajectory` Pre-Flight Anti-Hang Guard (`test_auto_switch_preflight_falls_back_to_advisor_when_cli_unowned`)

`agy` CLI launches an ephemeral Language Server port per session rather than sharing a fixed Desktop App Language Server port. Terminating a CLI turn and posting `SendUserCascadeMessage` to an unowned desktop Language Server would strand the CLI turn.
- Tests verify that in `auto_switch` mode, `handle_pre_invocation` calls `_verify_cascade_ownership(conversation_id)` (`GetCascadeTrajectory` with a `400 ms` timeout) **before** terminating.
- When `_verify_cascade_ownership` returns `False` (unowned cascade or unreachable LS), `handle_pre_invocation` **never** returns `{"terminationBehavior": "terminate"}` and **never** spawns `redispatch` - instead, it gracefully falls back to `advisor`, returns the inline `injectSteps` banner, and logs `"auto_switch_fallback": "cascade_unowned"` in `telemetry.jsonl`.

### E. MCP `configure_router`, CLI Subcommands, & Hook Timeout (`test_configure_router_mcp_tool_and_cli_commands`)

- Verifies `tools/list` exposes all 4 MCP tools (`create_decision`, `route_prompt`, `list_models`, `configure_router`).
- Verifies `configure_router` updates `verbosity`, `mode`, `min_confidence`, `enable_subagent_routing`, and `tier_models` in `router_config.json`.
- Verifies `auth-setup`, `watch --once`, `stats`, `route --pretty`, and the tightened `1.5s` default hook HTTP timeout budget (`GEMINI_DECISIONS_HOOK_TIMEOUT_S`, clamped to `[0.3, 8.0]`).

### F. Strict Plugin Validation & Repository Hygiene (`agy plugin validate --strict`)

- **`agy plugin validate . --strict`:** Validates `plugin.json` (including `"category": "agents"`), `mcp_config.json`, `hooks.json`, `assets/logo.svg`, and `skills/gemini-decisions-router/SKILL.md` with zero warnings or errors (also executed automatically during `./install.sh`).
- **Repository Hygiene Tests:** `test_zero_internal_google_references`, `test_config_rejects_embedded_secrets`, `test_file_security_blocks_symlinks_dotfiles_and_secrets`, and `test_docs_contain_no_em_or_en_dashes` enforce zero internal endpoints/identifiers, zero embedded secrets, strict path-traversal/symlink blocks, and ASCII-clean markdown across every documentation and skill file.

## 3. Live Auto-Routing Results (Generated from Telemetry Logs)

The table below was generated directly from live `telemetry.jsonl` entries produced against `gemini-decision-1` (`auth_mode=api_key`, `mode=auto_switch`, 3 simulated steps per prompt confirming `TOTAL_DISPATCHES: 6/6` with zero loops):

| Hook Event | Turn | Prompt | Expected Tier | Routed Model | Thinking | Backend | Confidence | Latency | Auto-Switched |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PreInvocation` | 1 | `ship it` | `flash_lite` | `gemini-3.1-flash-lite` | Low | `fast_rules (<1 ms)` | 0.93 | 1 ms | `true` |
| `PreInvocation` | 1 | `What is the capital of France?` | `flash_lite` | `gemini-3.1-flash-lite` | Low | `gemini_api` | 0.952 | 236 ms | `true` |
| `PreInvocation` | 1 | `Refactor this FastAPI endpoint to validate pagination parameters and add unit tests` | `flash` | `gemini-3-flash` | Medium | `gemini_api` | 0.9024 | 243 ms | `true` |
| `PreInvocation` | 1 | `Design a multi-region Cloud Spanner schema with conflict-free failover and zero-downtime migration` | `pro` | `gemini-3.1-pro-high` | Max | `gemini_api` | 0.9133 | 320 ms | `true` |
| `PreInvocation` | 2 (on `pro` thread) | `yes go ahead` | `pro` | `gemini-3.1-pro-high` | Max | `thread_lock (<1 ms)` | 0.92 | 1 ms | `true` |
| `PreInvocation` | 1 | `use gemini pro to review this terraform module` | `pro` | `gemini-3.1-pro-high` | High | `fast_rules (<1 ms)` | 0.99 | 1 ms | `true` |
| `PreToolUse:invoke_subagent` | N/A | `List the markdown files in docs/` (`Role: Codebase Researcher`) | `flash_lite` | `Model -> flash_lite` | N/A | `gemini_api` | 0.8182 | 208 ms | overwrite |

Inspect live or historical telemetry at any time with:

```bash
agy-router watch
agy-router stats
```
