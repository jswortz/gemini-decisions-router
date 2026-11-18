# Gemini API Key Quickstart & Router Usage Guide

This guide walks through installing **Gemini Decisions Router** into Antigravity (`agy`), testing it with `GEMINI_API_KEY`, and using it as an automatic prompt and subagent router.

## 1. Prerequisites & Environment Setup

Export your Google AI Studio API key in your shell profile (`~/.bashrc` or `~/.zshrc`) **before** launching Antigravity so both the Antigravity MCP server (`decisions`) and lifecycle hooks (`PreInvocation`, `PreToolUse`) inherit it:

```bash
export GEMINI_API_KEY="your-ai-studio-api-key"
```

> **Important:** Never place `GEMINI_API_KEY` inside `config.json` or commit it to source control. The plugin rejects configuration files containing embedded secrets.

> **Why shell inheritance matters for Antigravity:** If Antigravity is launched from a desktop shortcut or daemon that does not source `~/.bashrc`, the `decisions` MCP server and hooks will not see `GEMINI_API_KEY` and will fall back to ADC (or `lexical_fallback` for `route_prompt`). Launch Antigravity from a shell where `GEMINI_API_KEY` is exported, or verify with `router_hook.py status` and `router_hook.py stats`.

## 2. Install the Plugin into Antigravity

Clone the repository and run `./install.sh` (requires Python 3.11+):

```bash
git clone https://github.com/YOUR_ORG/gemini-decisions-router.git
cd gemini-decisions-router
chmod +x install.sh
./install.sh
```

`install.sh` copies the plugin manifests, engine, MCP server, hooks, and skill into `~/.gemini/config/plugins/gemini-decisions-router` (override with `ANTIGRAVITY_PLUGINS_DIR`).

Verify the installation and run a live 1-question `noul` probe against `gemini-decision-1`:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py status
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live
```

Expected output from `auth-check --live`:

```json
{
  "ok": true,
  "auth_mode": "api_key",
  "model": "gemini-decision-1",
  "project_id": "(from ADC)",
  "location": "global",
  "credential_source": "env:GEMINI_API_KEY",
  "endpoint": "https://generativelanguage.googleapis.com/v1beta/decisions",
  "live_probe": {
    "backend": "gemini_api",
    "latency_ms": 218,
    "answer": {
      "type": "noul",
      "noul": 0.9859,
      "value": true,
      "confidence": 0.9859
    }
  }
}
```

Reload Antigravity after installation so it picks up `plugin.json`, `mcp_config.json`, `hooks.json`, and `skills/gemini-decisions-router/SKILL.md`.

## 3. Using It as a Router in Antigravity

Once installed and reloaded, the router operates in three complementary ways:

### A. Automatic Turn Routing (`PreInvocation` Hook)

Every time you submit a prompt in Antigravity, `hooks.json` invokes `router_hook.py pre-invocation`:

1. **Tier 0 (`<1 ms`):** Fast local rules handle confirmations (`"yes"`, `"ship it"`), explicit model requests (`"use gemini pro"`), crash markers, and Turn 2+ asymmetric thread floors (so short follow-ups never downgrade an active `pro` session).
2. **Tier 1 (`~200-370 ms`, 0 output tokens):** Calls `gemini-decision-1` once to score `task_class` (`choice`), `needs_pro_reasoning` (`noul`), and `effort_score` (`score`).

In **`advisor` mode (default)**, Antigravity renders a collapsible **Gemini Decisions Router** card above the response without interrupting your selected model:

- **Active Dropdown Model**: `auto`
- **Calibrated Route**: `gemini-3.1-pro-high` (`subagent_tier=pro`, thinking=`High`)
- **Task Class**: `deep_debug_architecture` (87% confidence)
- **Decision Gate**: `gemini_api` in `366 ms` (0 output tokens)

To let the router **automatically switch** the active Antigravity conversation model when confidence exceeds `min_confidence`:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode auto_switch --min-confidence 0.55
```

Switch back to `advisor` or `off` at any time:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode advisor
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode off
```

### B. Automatic Subagent Right-Sizing (`PreToolUse` Hook)

When an Antigravity agent calls `invoke_subagent` with `Model` unset or `"inherit"`, `router_hook.py pre-tool-use` scores the subagent's `Role` and `Prompt` and overwrites `Model` to `flash_lite`, `flash`, or `pro`.

For example, a lightweight file-listing subagent brief (`"List the markdown files in docs/"`) is automatically right-sized to:

```json
{"decision": "allow", "overwrite": {"Model": "flash_lite"}}
```

### C. CLI Testing & Telemetry

Test Tier 1 routing directly from the terminal with `--no-fast-rules`:

```bash
# Simple lookup -> flash_lite
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py route \
  --prompt "What is the capital of France?" \
  --no-fast-rules

# Medium code edit -> flash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py route \
  --prompt "Refactor this FastAPI endpoint to validate pagination parameters and add unit tests" \
  --no-fast-rules

# Deep architecture -> pro
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py route \
  --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover and zero-downtime migration" \
  --no-fast-rules
```

Inspect recent routing events, tier distribution, and average latency:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py stats
```

### D. On-Demand MCP Tools (`decisions`)

Agents in Antigravity can also call the two bundled MCP tools directly:

- `route_prompt`: Returns `recommended_tier`, `recommended_model_id`, `thinking_level`, `confidence`, and raw probabilities for any prompt or subagent brief.
- `create_decision`: Evaluates shared text or allowlisted local files once across up to 64 `noul`, `choice`, and `score` questions in a single zero-output-token pass.

## 4. Demo

![Gemini Decisions Router in action in Antigravity](../assets/agy-router-demo.gif)
