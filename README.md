# Gemini Decisions Router for Antigravity

A zero-dependency Antigravity plugin, CLI (`agy-router`), and MCP server that routes prompts and subagent tasks using the **Gemini Decisions API** (`gemini-decision-1`).

Traditional LLM routers generate JSON text token-by-token, adding 400 to 1,500 ms of decoding latency and brittle string parsing before an agent even starts work. The Gemini Decisions API evaluates your prompt in **one prefill pass with zero output tokens**, returning calibrated probabilities and confidence scores across yes/no (`noul`), multiple-choice (`choice`), and rubric (`score`) questions in roughly 120 to 250 ms.

## Key Capabilities

- **Frictionless Install & `agy-router` CLI:** `./install.sh` installs the plugin into `~/.gemini/config/plugins/gemini-decisions-router`, creates the `~/.local/bin/agy-router` CLI command, and validates the package with `agy plugin validate --strict`.
- **Reliable Auth Across Desktop App & `agy` CLI:** Supports keyless OAuth2 Application Default Credentials (`gemini_adc` or `vertex_adc`) and `GEMINI_API_KEY`. Run `agy-router auth-setup --from-env` (or `--api-key-stdin`) once to write a `0600`-permissioned `~/.config/gemini-decisions-router/.env` file so both the GUI-launched Antigravity Desktop App and `agy` CLI resolve credentials without shell inheritance pitfalls.
- **2-Tier Calibrated Prompt Routing (`PreInvocation`):**
  - **Tier 0 (`<1 ms`):** Instant local fast rules handle confirmations (`"ship it"`, `"lgtm"`), explicit model requests (`"use gemini pro"`), crash markers, Turn 2+ thread continuations, and multi-step tool-loop deduplication.
  - **Tier 1 (`~150-300 ms`, 0 output tokens, `1.5s` hook deadline):** Calls `gemini-decision-1` once to score task intent (`choice`), frontier reasoning need (`noul`), and thinking effort (`score`), routing across `flash_lite`, `flash`, and `pro` target models.
- **Universal `"✦ Routed with Intent"` Visibility (0 Model Tokens):** Renders inline `metadata.user_markdown` routing banners in **both** the Antigravity Desktop App and `agy` CLI with `"systemMessage": ""` (zero prompt token overhead), alongside `PreToolUse` subagent notices, 4 verbosity presets (`silent`, `compact`, `normal`, `verbose`), live terminal streaming (`agy-router watch`), and historical summaries (`agy-router stats`).
- **Asymmetric Thread Floor & Multi-Step Deduplication:** Prevents mid-conversation downgrades on Turn 2+ so short replies like `"yes go ahead"` never drop an active `pro` session down to `flash_lite`, and deduplicates multi-step tool loops so Steps 2..N execute in `<1 ms` with zero extra API calls or duplicate banners.
- **Automatic Subagent Right-Sizing (`PreToolUse`):** Intercepts `invoke_subagent` calls when `Model` is unset or `"inherit"` and assigns `flash_lite`, `flash`, or `pro` based on each subagent's task.
- **Dynamic Model Auto-Discovery (`sync-models` & `list-models --refresh`):** Automatically discovers newly released Antigravity Desktop App and `agy` CLI models at runtime (for example `gemini-next-flash-medium` or `gemini-next-pro-high`), classifies each model into `flash_lite`, `flash`, or `pro` with an inferred thinking level, caches results in `~/.cache/gemini-decisions-router/discovered_models.json` (`0600`), and supports one-step adoption via `agy-router sync-models --apply-recommended`.
- **4 MCP Tools (`create_decision`, `route_prompt`, `list_models`, `configure_router`):** Exposes calibrated decision gates, prompt classification, live target model discovery (`list_models(refresh=true)`), and live router configuration directly to your agents.
- **Pure Python 3.11+ Standard Library:** Runs out of the box with zero mandatory `pip` packages. Automatically uses `google-auth` if present, and falls back to native stdlib OAuth2 refresh and GCE/Cloud Run/GKE Metadata Server queries.

## Live Benchmarks (`N=100` Prompts + `N=15` End-to-End)

![Benchmark Dashboard](assets/benchmark-dashboard.png)

Evaluated live against `generativelanguage.googleapis.com/v1beta` across 100 stratified engineering and agentic prompts (6 categories) and 15 multi-tier downstream completions (`maxOutputTokens: 512`):

| Strategy | Router `p50` / `p90` | Output Tokens | Exact Tier Accuracy | Critical Pro-Miss Rate | 100-Prompt E2E Time | 100-Prompt E2E Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini Decisions Router (Tier 0+1)** | **`278 ms` / `411 ms`** *(Tier 0: `<0.1 ms`)* | **`0`** | **`96.0%`** (`96/100`) | **`0.0%`** (`0/30`) | **`335.5 s` (`-29.1%`)** | **`$0.225` (`-56.3%`)** |
| **Pure `gemini-decision-1` (Tier 1)** | **`235 ms` / `374 ms`** | **`0`** | **`96.0%`** (`96/100`) | **`0.0%`** (`0/30`) | `331.3 s` (`-30.0%`) | `$0.226` (`-56.2%`) |
| **`GenerateContent` JSON (`gemini-2.5-flash`)** | `1,124 ms` / `1,644 ms` (`4.0x-4.8x` slower) | `144.3` | `100.0%` (`15/15`) | `0.0%` (`0/5`) | `416.0 s` (`-12.1%`) | `$0.258` (`-50.0%`) |
| **Offline Lexical Baseline** | `0.01 ms` / `0.01 ms` | `0` | `79.0%` (`79/100`) | `36.7%` (`11/30` missed) | N/A | N/A |
| **Always-Pro Baseline** | `0 ms` | `0` | `30.0%` (`70%` over-routed) | `0.0%` (`0/30`) | `473.2 s` *(Baseline)* | `$0.515` *(Baseline)* |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology, confusion matrices, confidence calibration (`ECE = 0.103`), and reproducibility commands (`python3 benchmarks/run_benchmark.py --mock` runs offline in `<0.1s`).

## Authentication Modes

| Mode | Endpoint | Credentials Used | Best For |
| :--- | :--- | :--- | :--- |
| **`gemini_adc`** *(Default)* | `generativelanguage.googleapis.com/v1beta/decisions` | Keyless OAuth2 ADC + `x-goog-user-project` | Immediate keyless rollout on any GCP project with Generative Language API enabled |
| **`vertex_adc`** | `{region-}aiplatform.googleapis.com/v1beta1/.../decisions` | Keyless OAuth2 ADC (`cloud-platform` scope) | Enterprise GCP projects enrolled in Vertex AI Decisions Preview (VPC-SC, regional endpoints) |
| **`api_key`** | `generativelanguage.googleapis.com/v1beta/decisions` | `GEMINI_API_KEY` (env or `0600` `.env`) | Solo prototyping in Google AI Studio without a GCP project |

See [docs/AUTHENTICATION_ADC.md](docs/AUTHENTICATION_ADC.md) for credential chain details and troubleshooting, and [docs/CUSTOMER_ONBOARDING.md](docs/CUSTOMER_ONBOARDING.md) for enterprise rollout guidance.

## Quick Start (Keyless ADC)

### 1. Install the Plugin and `agy-router` CLI

```bash
git clone https://github.com/YOUR_ORG/gemini-decisions-router.git
cd gemini-decisions-router
chmod +x install.sh
./install.sh
```

`./install.sh` performs three steps:
1. Verifies Python 3.11+ and copies all plugin manifests, engine, MCP server, hooks, logo, and skill into `~/.gemini/config/plugins/gemini-decisions-router` (override with `ANTIGRAVITY_PLUGINS_DIR`).
2. Installs the executable `agy-router` CLI wrapper into `~/.local/bin/agy-router` (override with `BIN_DIR`; ensure `~/.local/bin` is on your `PATH`).
3. Automatically runs `agy plugin validate ~/.gemini/config/plugins/gemini-decisions-router --strict` when `agy` is installed.

### 2. Enable the API and Configure Workstation ADC

On a developer laptop or workstation with the Google Cloud SDK installed:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Enable the Generative Language API on your project (one-time per project)
gcloud services enable generativelanguage.googleapis.com \
  --project="${GOOGLE_CLOUD_PROJECT}"

# Log in with ADC including the Generative Language retriever scope
gcloud auth application-default login \
  --project="${GOOGLE_CLOUD_PROJECT}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever"

# Persist GOOGLE_CLOUD_PROJECT into a private 0600 .env file for both Desktop App and agy CLI
agy-router auth-setup --from-env
```

> **Running on Cloud Workstations, GCE, Cloud Run, or GKE?** Skip `gcloud auth application-default login`. The plugin requests the required OAuth scopes directly from the Compute Metadata Server using the attached service account.

### 3. Verify Authentication and Live Routing

```bash
# Verify ADC token resolution and run a live 1-question probe
agy-router auth-check --live

# Inspect active router configuration, detected backend, and target models
agy-router status

# Test Tier 1 calibrated prompt routing with formatted banner output
agy-router route \
  --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover" \
  --no-fast-rules \
  --pretty
```

Reload Antigravity after installation to activate the MCP server, lifecycle hooks, and skill.

## Quick Start (Optional API Key Fallback)

Desktop GUI launchers on macOS and Linux often start the Antigravity Desktop App without sourcing `~/.bashrc` or `~/.zshrc`. Use `agy-router auth-setup` to save credentials into a `0600`-permissioned `~/.config/gemini-decisions-router/.env` file so both the **Antigravity Desktop App** and **`agy` CLI** resolve `GEMINI_API_KEY` reliably:

```bash
./install.sh

# Option A: Capture from your current shell environment
export GEMINI_API_KEY="your-ai-studio-api-key"
agy-router auth-setup --from-env

# Option B: Pipe via stdin so the key never touches shell history
printf '%s\n' 'your-ai-studio-api-key' | agy-router auth-setup --api-key-stdin

# Confirm live connectivity
agy-router auth-check --live
```

Security guarantees for `~/.config/gemini-decisions-router/.env`:
- Created inside a `0700` directory with strict `0600` owner-only file permissions (`chmod 600`).
- The loader (`load_private_dotenv`) inspects `os.lstat` and raises `ConfigError` if `.env` is a symlink or has group/world permissions (`(st_mode & 0o077) != 0`).
- Active process environment variables always override `.env` entries. Never store secrets in `config.json` or `router_config.json`.

See [docs/API_KEY_QUICKSTART.md](docs/API_KEY_QUICKSTART.md) for a complete walkthrough of `GEMINI_API_KEY` setup, hook/MCP routing, verbosity controls, and live telemetry.

## Seeing Active Routes in Antigravity App & `agy` CLI

You can observe every routing decision in real time across both surfaces:

### 1. Inline Zero-Token `"✦ Routed with Intent"` Banners

- **User Prompts (`PreInvocation`):** On Step 1 of every user turn, the hook injects `systemMessage.metadata.user_markdown` alongside `renderInfo` with `"systemMessage": ""`. Because `metadata.user_markdown` is rendered directly by the UI and terminal renderer while `"systemMessage"` is empty, you see the route inline in **both** the Antigravity Desktop App and `agy` CLI at **0 model context tokens**.
- **Subagent Dispatches (`PreToolUse`):** Whenever `invoke_subagent` is right-sized from `"inherit"`, the hook attaches an inline tool notice (`✦ Routed with Intent -> subagent Model=flash_lite (95% | 180 ms)`) whenever verbosity is not `silent`.

### 2. Configurable Verbosity Presets (`silent`, `compact`, `normal`, `verbose`)

| Level | Inline Banner (`PreInvocation`) | Subagent Notice (`PreToolUse`) | Telemetry Logged |
| :--- | :--- | :--- | :--- |
| **`silent`** | Hidden (`{}` returned) | Hidden (`reason` omitted) | Yes (`telemetry.jsonl`) |
| **`compact`** | Single line: `✦ Routed with Intent (Recommended) -> gemini-3.1-pro-high [Max] · 91% · gemini_api (240 ms)` | Shown | Yes |
| **`normal`** *(Default)* | `ROUTED WITH INTENT` box header + active model, calibrated route, task class, backend latency, and rationale | Shown | Yes |
| **`verbose`** | `normal` card + top class probabilities, `P(pro reasoning)`, `Effort Rubric`, `Thread Lock`, and explicit override flags | Shown | Yes |

Change verbosity via CLI, environment variable, or directly in chat via the `configure_router` MCP tool:

```bash
# Persistent CLI setting (saved to ~/.config/gemini-decisions-router/router_config.json)
agy-router set-verbosity compact
agy-router set-verbosity normal
agy-router set-verbosity verbose
agy-router set-verbosity silent

# Per-session environment override (takes precedence over router_config.json)
export GEMINI_DECISIONS_VERBOSITY="compact"
```

### 3. Live Terminal Stream (`watch`) & Historical Summary (`stats`)

Open a side terminal while working in the Antigravity App or `agy` CLI to watch routing events stream live or inspect aggregate performance:

```bash
# Stream live routing decisions under the "✦ ROUTED WITH INTENT | LIVE MONITOR" header
agy-router watch
agy-router watch --verbosity verbose
agy-router watch --once

# View historical tier distribution, average latency, and recent prompts
agy-router stats
agy-router stats --verbosity compact
agy-router stats --verbosity verbose --limit 100
agy-router stats --json
```

## Router Modes: `advisor` vs Hardened `auto_switch`

- **`advisor` *(Default)*:** Scores Step 1 of each user turn, renders the zero-token `"✦ Routed with Intent"` inline banner, and right-sizes `invoke_subagent` calls without interrupting your active chat model. Multi-step tool loops are deduplicated per `(conversationId, prompt_hash)` (`seen_turn` state) so Steps 2..N complete in `<1 ms` with 0 API calls and 0 duplicate banners.
- **`auto_switch`:** Automatically switches the active Antigravity Desktop App conversation model when confidence meets `min_confidence` (default `0.45`):
  - **`GetCascadeTrajectory` Anti-Hang Pre-Flight:** Before terminating a turn to redispatch on a new model, `router_hook.py` probes the Language Server with `GetCascadeTrajectory` (`400 ms` timeout) to verify that the reachable server actually owns `conversationId`. In `agy` CLI (which runs an ephemeral Language Server port) or if ownership cannot be verified, `auto_switch` **never terminates the turn** - it automatically falls back to `advisor` mode, displays the inline banner, and records `"auto_switch_fallback": "cascade_unowned"` in `telemetry.jsonl` so CLI turns never hang.
  - **Multi-Step Loop Protection:** Tracks `reroute_<conv>.json` (`600s` TTL) so redispatched turns render their switched banner once on Step 2 and short-circuit in `<1 ms` on Steps 3..N. Any background RPC errors are logged to `~/.cache/gemini-decisions-router/redispatch.log`.
- **`off`:** Disables `PreInvocation` and `PreToolUse` hooks while keeping all 4 MCP tools available on demand.

```bash
agy-router set-mode advisor
agy-router set-mode auto_switch --min-confidence 0.55
agy-router set-mode off
```

## Dynamic Model Auto-Discovery & Configurable Target Models

In addition to the 10 built-in target models (across Gemini 3.1/3/2.5, Claude Sonnet/Opus 4.6, and GPT-OSS), the router dynamically discovers newly released models directly from your local Antigravity Language Server (`GetAvailableModels`) and `agy models` CLI at runtime:

- **Automatic Tier & Thinking Classification:** Newly discovered models (such as `gemini-next-flash-medium`, `gemini-next-pro-high`, or `gemini-pro-orchestrator`) are automatically classified into `flash_lite`, `flash`, or `pro` with calibrated default thinking levels (`1..4`), and any Antigravity model enums returned by the Language Server are merged automatically for `auto_switch` redispatch.
- **Private `0600` Discovery Cache:** Discovered models are cached in `~/.cache/gemini-decisions-router/discovered_models.json` (`0700` directory, `0600` file, `3600s` TTL) so `PreInvocation` hooks and status checks read them in `<1 ms` without spawning subprocesses.
- **Immediate Adoption:** Discovered models are immediately selectable via `agy-router set-models`, `configure_router`, or one-command `agy-router sync-models --apply-recommended`.

```bash
# Discover newly released Antigravity / agy CLI models and view suggested_tier_models
agy-router sync-models

# Discover models and persist suggested_tier_models into router_config.json
agy-router sync-models --apply-recommended

# List all built-in, discovered, and custom target models (auto-populates cache on first run)
agy-router list-models

# Force a live re-discovery refresh from Antigravity LS and agy CLI
agy-router list-models --refresh

# Customize tier targets persistently (supports built-in or newly discovered model IDs)
agy-router set-models \
  --flash-lite gemini-3.1-flash-lite \
  --flash gemini-next-flash-medium \
  --pro gemini-next-pro-high

# Reset back to built-in defaults
agy-router set-models --reset
```

Environment variables (`GEMINI_DECISIONS_MODEL_FLASH_LITE`, `GEMINI_DECISIONS_MODEL_FLASH`, `GEMINI_DECISIONS_MODEL_PRO`) override `router_config.json` when set.

## MCP Tools Reference

Once installed, agents in Antigravity can call four MCP tools under the `decisions` server (`gemini-decisions-router_decisions` namespace):

### 1. `create_decision`

Evaluates shared text or allowlisted local files once across 1 to 64 independent structured questions (`noul`, `choice`, `score`) with zero output tokens:

```json
{
  "input": [
    {"text": "Customer requests deleting production BigQuery dataset analytics_prod without a backup snapshot."}
  ],
  "questions": {
    "destructive_change": {
      "type": "noul",
      "instructions": "Could executing this action cause irreversible production data loss?"
    },
    "approval_tier": {
      "type": "choice",
      "instructions": "Which approval workflow applies to this request?",
      "criteria": {
        "auto_approve": "Read-only query or reversible sandbox action.",
        "peer_review": "Standard staging or non-destructive production change.",
        "vp_signoff": "Irreversible production data deletion or schema drop."
      }
    },
    "risk_level": {
      "type": "score",
      "instructions": "Score the operational blast radius.",
      "criteria": [
        "0: Read-only operation.",
        "1: Single developer sandbox impact.",
        "2: Shared staging service impact.",
        "3: Production data loss or customer outage risk."
      ]
    }
  }
}
```

### 2. `route_prompt`

Scores any prompt or subagent task description using Tier 0 fast rules plus Tier 1 `gemini-decision-1` evaluation, returning `recommended_tier` (`flash_lite`, `flash`, `pro`), `recommended_model_id`, `thinking_level` (`1..4`), `confidence`, and raw probabilities.

### 3. `list_models`

Lists all built-in, dynamically auto-discovered (`~/.cache/gemini-decisions-router/discovered_models.json`), and custom target models, their Antigravity model enums, default thinking effort levels, `suggested_tier_models`, and the active `tier_models` mapping. Pass `{"refresh": true}` to force live re-discovery from the local Antigravity Language Server and `agy` CLI:

```json
{
  "refresh": true
}
```

### 4. `configure_router`

Updates persistent router settings in `~/.config/gemini-decisions-router/router_config.json` directly from an agent turn and returns the updated configuration plus `active_tier_models`:

```json
{
  "mode": "advisor",
  "verbosity": "compact",
  "min_confidence": 0.5,
  "enable_subagent_routing": true,
  "tier_models": {
    "pro": "claude-opus-4-6"
  }
}
```

## Important Configuration Note

Do **not** add `gemini-decision-1` to Antigravity's interactive chat model dropdown in `settings.json`. Decision models answer `CreateDecision` requests with calibrated probabilities and intentionally return zero text tokens on standard `GenerateContent` chat calls.

## Running Tests & Plugin Validation

```bash
python3 -m pytest -v tests/test_external_plugin.py
agy plugin validate . --strict
```

## Repository Layout

- `plugin.json`, `mcp_config.json`, `hooks.json`: Antigravity plugin (`category: "agents"`), MCP, and hook manifests
- `install.sh`: Installer for `~/.gemini/config/plugins/gemini-decisions-router`, `~/.local/bin/agy-router`, and `agy plugin validate --strict`
- `decisions_engine.py`: Keyless ADC resolver, `0600` `.env` loader, Decisions HTTP client, input/file security validator, dynamic Antigravity / `agy` model auto-discovery, and 2-tier router
- `mcp_server.py`: Stdio JSON-RPC 2.0 MCP server (`create_decision`, `route_prompt`, `list_models`, `configure_router`)
- `router_hook.py`: `PreInvocation` / `PreToolUse` hooks and `agy-router` CLI (`auth-setup`, `auth-check`, `status`, `route`, `watch`, `stats`, `set-mode`, `set-verbosity`, `list-models`, `sync-models`, `set-models`)
- `skills/gemini-decisions-router/SKILL.md`: Agent skill reference loaded by Antigravity
- `docs/CUSTOMER_ONBOARDING.md`: Enterprise Google Cloud customer rollout guide
- `docs/AUTHENTICATION_ADC.md`: Keyless ADC architecture and troubleshooting guide
- `docs/API_KEY_QUICKSTART.md`: Frictionless install, `0600` `.env` setup, inline visibility, and CLI/MCP usage guide
- `docs/BENCHMARKS.md`: Live 100-prompt routing accuracy, latency percentiles, calibration, and end-to-end cost/time report
- `docs/TESTING_STRATEGY.md`: Target model catalog, security/deduplication/anti-hang test strategy, and live telemetry results
- `benchmarks/`: `prompts_100.json` dataset, `run_benchmark.py` (live + `--mock` runner), `render_charts.py`, and `results_100.json`
- `tests/test_external_plugin.py`: Unit, mock HTTP, `.env` permission/symlink, verbosity, deduplication, pre-flight, and hygiene test suite

## License

Apache License 2.0. See [LICENSE](LICENSE).
