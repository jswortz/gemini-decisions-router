---
name: gemini-decisions-router
description: Route prompts and subagents or ask calibrated yes/no (`noul`), multiple-choice (`choice`), and rubric (`score`) questions using the Gemini Decisions API (`gemini-decision-1`) via keyless Google Cloud Application Default Credentials (ADC) or `GEMINI_API_KEY`. Use when classifying tasks, gating tool actions on confidence, scoring documents against rubrics, or configuring the Gemini Decisions Router mode (`advisor`, `auto_switch`, `off`) in Antigravity.
---

# Gemini Decisions Router

This plugin connects Antigravity to the Gemini Decisions API (`gemini-decision-1`) for calibrated, zero-output-token classification and automatic model routing.

## MCP Tools

The bundled MCP server registers three tools under the `gemini-decisions-router_decisions` namespace:

1. `create_decision`: Evaluates shared text or local files once across up to 64 independent questions of three types:
   - `noul`: Binary yes/no question returning calibrated `P(yes)` in `[0.01, 0.99]`, boolean `value`, and `confidence` in `[0, 1]`.
   - `choice`: Discrete multiple-choice map (2 to 255 options) returning `choice`, `confidence`, and normalized `probabilities`.
   - `score`: Ordered rubric array (2 to 255 levels, index `0..N-1`) returning expected `score`, modal `level`, `confidence`, and per-level `probabilities`.
2. `route_prompt`: Classifies a prompt or subagent task using Tier 0 (`<1 ms`) fast rules plus a single-pass `create_decision` call, returning `recommended_tier` (`flash_lite`, `flash`, `pro`), `recommended_model_id`, `thinking_level`, and `confidence`.
3. `list_models`: Lists all available target models, their Antigravity model enums, default thinking levels, and the active `tier_models` configuration.

## Lifecycle Hooks

- `PreInvocation`: Scores each explicit user turn, preserves Turn 2+ thread floors so short follow-ups (`"yes"`, `"ship it"`) never downgrade an active `pro` session, and renders a zero-token routing summary card in the UI.
- `PreToolUse` (`invoke_subagent`): When a subagent is invoked with `Model` unset or `"inherit"`, scores the subagent task and sets `Model` to `flash_lite`, `flash`, or `pro`.

## CLI Commands

Run these commands to verify credentials, inspect telemetry, or change router behavior:

```bash
# Verify ADC or API key setup (pass --live to send a live probe)
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live

# Check active authentication mode, GCP project, and router status
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py status

# Test routing on any prompt
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py route \
  --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover" \
  --no-fast-rules

# Switch mode: advisor (default), auto_switch, or off
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode advisor
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode auto_switch

# Inspect all available target models and active tier assignments
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py list-models

# Customize target models per tier (or pass --reset for defaults)
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-models --pro claude-sonnet-4-6

# Inspect recent routing telemetry and average latency
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py stats
```

## Authentication Rules

- **Keyless Gemini API via ADC (`gemini_adc`, default when no API key is set):** Uses OAuth2 Bearer tokens from Application Default Credentials plus `x-goog-user-project: $GOOGLE_CLOUD_PROJECT`. On developer workstations, include `https://www.googleapis.com/auth/generative-language.retriever` when running `gcloud auth application-default login`.
- **Keyless Vertex AI via ADC (`vertex_adc`, enabled when `GOOGLE_GENAI_USE_VERTEXAI=true`):** Uses standard `https://www.googleapis.com/auth/cloud-platform` ADC against `aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/decisions`.
- **API Key Fallback (`api_key`):** Export `GEMINI_API_KEY` in the shell environment. Never place secrets or tokens inside `config.json`.
- **Model Dropdown Rule:** Never select `gemini-decision-1` as an interactive chat model in `settings.json`. Decision models emit zero output tokens on `GenerateContent` and only answer `CreateDecision` calls.
