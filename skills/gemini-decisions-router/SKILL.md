---
name: gemini-decisions-router
description: Route prompts and subagents or ask calibrated yes/no (`noul`), multiple-choice (`choice`), and rubric (`score`) questions using the Gemini Decisions API (`gemini-decision-1`) via keyless Google Cloud Application Default Credentials (ADC) or `GEMINI_API_KEY`. Use when classifying tasks, gating tool actions on confidence, scoring documents against rubrics, listing or customizing target models, or configuring router mode (`advisor`, `auto_switch`, `off`) and verbosity (`silent`, `compact`, `normal`, `verbose`) in Antigravity.
---

# Gemini Decisions Router

This plugin connects the Antigravity Desktop App and `agy` CLI to the Gemini Decisions API (`gemini-decision-1`) for calibrated, zero-output-token classification and automatic model routing.

## MCP Tools (`gemini-decisions-router_decisions`)

1. `create_decision`: Evaluates shared `input` (`text` or allowlisted `file` items) once across 1 to 64 independent questions:
   - `noul`: Binary yes/no question returning calibrated `P(yes)` in `[0.01, 0.99]`, boolean `value`, and `confidence` in `[0, 1]`.
   - `choice`: Discrete multiple-choice map (2 to 255 options) returning `choice`, `confidence`, and normalized `probabilities`.
   - `score`: Ordered rubric array (2 to 255 levels, index `0..N-1`) returning expected `score`, modal `level`, `confidence`, and per-level `probabilities`.
2. `route_prompt`: Classifies a prompt or subagent task using Tier 0 (`<1 ms`) fast rules plus a single-pass `create_decision` call, returning `recommended_tier` (`flash_lite`, `flash`, `pro`), `recommended_model_id`, `thinking_level` (`1..4`), `confidence`, and `probabilities`.
3. `list_models`: Lists all built-in, dynamically auto-discovered (`~/.cache/gemini-decisions-router/discovered_models.json`), and custom target models, their Antigravity model enums, default thinking levels, `suggested_tier_models`, and active `tier_models` (`flash_lite`, `flash`, `pro`). Pass optional `{"refresh": true}` to force live re-discovery from the local Antigravity Language Server and `agy` CLI.
4. `configure_router`: Updates persistent settings in `~/.config/gemini-decisions-router/router_config.json` and returns the updated config plus `active_tier_models`. Supported fields:
   - `mode`: `"advisor"` | `"auto_switch"` | `"off"`
   - `verbosity`: `"silent"` | `"compact"` | `"normal"` | `"verbose"`
   - `min_confidence`: number in `[0.0, 1.0]`
   - `enable_subagent_routing`: boolean
   - `enable_thread_floor`: boolean (asymmetric Turn 2+ floor for `coding_edit` / `pro` threads; default `true`)
   - `cli_inline_banner`: boolean (1-line `"✦ Routed with Intent"` badge inside `agy` CLI replies when `verbosity != "silent"`; default `true`)
   - `tier_models`: optional object mapping `flash_lite`, `flash`, and/or `pro` to target model IDs

## Lifecycle Hooks & Runtime Behavior

- `PreInvocation`:
  - Scores Step 1 of each explicit user turn (`1.5s` default hook HTTP deadline via `GEMINI_DECISIONS_HOOK_TIMEOUT_S`), enforces smart Turn 2+ asymmetric thread floors on `coding_edit`/`pro` threads while letting standalone high-confidence (`>0.85`) `quick_lookup` questions route to `flash_lite`, and deduplicates multi-step tool loops (`<1 ms` on Steps 2..N).
  - Emits zero-token inline `"✦ Routed with Intent"` cards via `renderInfo` (`"systemMessage": ""`) in the Antigravity Desktop App, and a toggleable 1-line inline routing badge (`cli_inline_banner`) in `agy` CLI according to `verbosity` (`silent`, `compact`, `normal`, `verbose`).
  - In `auto_switch` mode, merges built-in, cached auto-discovered, and custom Antigravity model enums and runs a `GetCascadeTrajectory` ownership pre-flight check before terminating a turn; if the Language Server does not own the cascade (such as in `agy` CLI), it automatically falls back to `advisor` mode so turns never hang.
- `PreToolUse` (`invoke_subagent`): When `Model` is unset or `"inherit"`, assigns `flash_lite`, `flash`, or `pro` and includes a `"✦ Routed with Intent"` tool `reason` notice (omitted when `verbosity` is `"silent"`).

## Top-Level `agy` CLI Auto-Switching (`agyr`, `chat`, `intent-router`)

Because standalone `agy` CLI runs an ephemeral Language Server per invocation, use the plugin's top-level CLI entrypoints for automatic model and effort switching in the terminal:

1. **`agyr` (alias for `agy-router launch` / `agy-router run`):**
   - `agyr "Refactor auth.py"` (or `-i`): Scores the prompt via `route_prompt`, resolves the installed CLI `--model` and sibling `--effort` (`low`, `medium`, `high`, `max`), prints the `AUTO-SWITCHED` banner to `stderr`, sets `GEMINI_DECISIONS_PRE_ROUTED` (so `PreInvocation` skips duplicate scoring in `0 ms`), and `exec`s `agy --model <id> --effort <effort> -i "<prompt>"`.
   - `agyr -p "What is 2+2?"`: Headless print mode (`-p`), omitting the `stderr` banner automatically when `--output-format json` is passed.
   - `agyr` (no prompt, no `-p`, no `-c`): Starts the interactive `agy` TUI on `flash_lite` with `--agent intent-router --model <lite_id> --effort low` so lightweight questions stay instant in-turn and heavier work delegates via `invoke_subagent(TypeName="self", Model="inherit")` into `PreToolUse` (`flash` / `pro`).
   - Explicit `--model` or bare `-c` / `--conversation` without a prompt passes straight through to `agy` untouched. Pass `--dry-run` to inspect the JSON execution plan without launching `agy`.
2. **`agy-router chat [--conversation ID] [--agy-bin PATH]`:**
   - Interactive multi-turn terminal REPL that scores each turn with `route_prompt` (evolving `thread_state` under `enable_thread_floor`) and switches `--model` and `--effort` across turns while preserving conversation continuity (`--continue` / `--conversation`). Supports `/new` or `/reset` to start a fresh conversation and `/exit` or `/quit` to leave.
3. **`agy --agent intent-router`:**
   - Custom Markdown main agent (`agents/intent-router.md`, `mainAgent: true`, `subagent: false`) that handles `quick_lookup` turns directly and delegates substantive `coding_edit`, `writing_docs`, `data_analysis`, and `deep_debug_architecture` tasks to `invoke_subagent(TypeName="self", Model="inherit")`.

## `agy-router` & `agyr` CLI Reference

`./install.sh` installs `~/.local/bin/agy-router` and `~/.local/bin/agyr` (fallback: `python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py <subcommand>`):

```bash
# Top-level routed agy launcher (interactive -i, headless -p, bare TUI intent-router, or --dry-run)
agyr "Refactor this FastAPI endpoint to validate pagination parameters"
agyr -p "What is 2+2?"
agyr                                     # Launches agy --agent intent-router on flash_lite
agy-router run --dry-run -p "What is 2+2?"

# Multi-turn REPL with automatic per-turn model & effort switching (/new, /exit)
agy-router chat
agy-router chat --conversation <conversation-id>

# Native agy TUI with the Intent-Routed Main Agent
agy --agent intent-router

# Write a 0600 ~/.config/gemini-decisions-router/.env file for Desktop App + agy CLI
agy-router auth-setup --from-env
printf '%s\n' 'your-ai-studio-api-key' | agy-router auth-setup --api-key-stdin
agy-router auth-setup --project your-gcp-project-id --auth-mode gemini_adc

# Verify ADC or API key resolution and run a live 1-question probe
agy-router auth-check --live

# Inspect router config, active backend, LS detection, and target models
agy-router status

# Test prompt routing with formatted "Routed with Intent" output
agy-router route \
  --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover" \
  --no-fast-rules \
  --pretty

# Stream live routing events or print current tail once
agy-router watch
agy-router watch --verbosity verbose --once

# Inspect historical routing telemetry and average latency
agy-router stats
agy-router stats --verbosity compact
agy-router stats --verbosity verbose --limit 100
agy-router stats --json

# Configure operating mode and inline verbosity preset
agy-router set-mode advisor
agy-router set-mode auto_switch --min-confidence 0.55
agy-router set-verbosity compact   # silent | compact | normal | verbose

# Discover newly released Antigravity / agy CLI models and cache in discovered_models.json (0600)
agy-router sync-models
agy-router sync-models --apply-recommended

# List or customize target models mapped to flash_lite, flash, and pro
agy-router list-models
agy-router list-models --refresh
agy-router set-models --flash-lite gemini-3.1-flash-lite --flash gemini-next-flash-medium --pro gemini-next-pro-high
agy-router set-models --reset
```

## Authentication & Safety Rules

- **Private `.env` (`0600`):** `~/.config/gemini-decisions-router/.env` must be a regular file with `0600` permissions; symlinks and group/world-readable files raise `ConfigError`. Process environment variables override `.env`. Never put secrets in `config.json` or `router_config.json`.
- **Keyless ADC (`gemini_adc` / `vertex_adc`):** Default when no API key is set. Workstation `gemini_adc` requires `https://www.googleapis.com/auth/generative-language.retriever` in `gcloud auth application-default login`; `vertex_adc` (`GOOGLE_GENAI_USE_VERTEXAI=true`) uses `cloud-platform`.
- **Model Dropdown Rule:** Never select `gemini-decision-1` as an interactive chat model in `settings.json`; it emits zero text tokens on `GenerateContent` and only answers `CreateDecision` requests.
