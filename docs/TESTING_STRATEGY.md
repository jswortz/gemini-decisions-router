# Testing Strategy & Auto-Routing Verification

This document describes how **Gemini Decisions Router** lists and configures target models, how auto-routing is tested across Tier 0 and Tier 1 paths, and the live verification results generated from `telemetry.jsonl`.

## 1. Available Target Models & Configuration

List all built-in and custom target models along with active tier assignments:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py list-models
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

Override tier mappings persistently via CLI (stored in `~/.config/gemini-decisions-router/router_config.json`):

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-models \
  --flash-lite gemini-2.5-flash-lite \
  --flash gemini-3-flash \
  --pro claude-opus-4-6

# Register a custom model enum if needed:
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-models \
  --custom-enum my-custom-model=MODEL_CUSTOM_FAST_V1

# Reset back to defaults:
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-models --reset
```

Or override temporarily via environment variables (highest precedence):

- `GEMINI_DECISIONS_MODEL_FLASH_LITE`
- `GEMINI_DECISIONS_MODEL_FLASH`
- `GEMINI_DECISIONS_MODEL_PRO`

Agents can also inspect available models and active tier bindings via the MCP tool `list_models`.

## 2. Testing Strategy

1. **Automated Unit & Contract Suite (`pytest`):**
   - `python3 -m pytest -v tests/test_external_plugin.py`
   - Covers external hygiene, secret rejection, ADC/Vertex/API key URLs, file security boundaries, Tier 0/Tier 1 routing, Turn 2+ asymmetric thread floor, configurable target models (`list_models`, `set-models`, env overrides), MCP end-to-end tool schemas, prompt telemetry logging, and **multi-step `auto_switch` anti-loop protection** (`test_auto_switch_prevents_multi_step_loop`).
2. **Multi-Step Auto-Switch Loop Guard:**
   - `PreInvocation` fires before every step in a multi-step turn with the same `lastUserInput`.
   - When `auto_switch` terminates and redispatches a turn, `reroute_<conv>.json` is persisted with `rendered: false` -> updated to `rendered: true` on the redispatched step -> and returns `{}` on all subsequent steps for the same `prompt_hash` (TTL 600s), preventing re-termination loops even if the active model string differs from `recommended_model_id`.
3. **Live Auto-Routing Verification (`GEMINI_API_KEY` -> `gemini-decision-1`):**
   - Exercises all 7 expected routing paths (Tier 0 fast rules, Tier 1 `flash_lite`/`flash`/`pro`, Turn 2+ thread lock, explicit model override, and `PreToolUse:invoke_subagent` right-sizing) and validates the resulting `telemetry.jsonl` records.

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

Inspect live telemetry at any time with:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py stats
```
