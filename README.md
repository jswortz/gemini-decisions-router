# Gemini Decisions Router for Antigravity

A zero-dependency Antigravity plugin and MCP server that routes prompts and subagent tasks using the **Gemini Decisions API** (`gemini-decision-1`).

Traditional LLM routers generate JSON text token-by-token, adding 400 to 1,500 ms of decoding latency and brittle string parsing before an agent even starts work. The Gemini Decisions API evaluates your prompt in **one prefill pass with zero output tokens**, returning calibrated probabilities and confidence scores across yes/no (`noul`), multiple-choice (`choice`), and rubric (`score`) questions in roughly 120 to 250 ms.

## Key Capabilities

- **Keyless Google Cloud Authentication (ADC First):** Authenticates with short-lived OAuth2 Application Default Credentials (`gemini_adc` or `vertex_adc`) so enterprise teams never have to distribute or rotate static API keys. Also supports `GEMINI_API_KEY` for quick local sandboxes.
- **2-Tier Calibrated Prompt Routing (`PreInvocation`):**
  - **Tier 0 (`<1 ms`):** Instant local fast rules handle confirmations (`"ship it"`, `"lgtm"`), explicit model requests (`"use gemini pro"`), crash markers, and Turn 2+ thread continuations.
  - **Tier 1 (`~150 ms`, 0 output tokens):** Calls `gemini-decision-1` once to score task intent (`choice`), frontier reasoning need (`noul`), and thinking effort (`score`), recommending `gemini-3.1-flash-lite`, `gemini-3-flash`, or `gemini-3.1-pro-high`.
- **Asymmetric Thread Floor:** Prevents mid-conversation downgrades on Turn 2+ so short replies like `"yes go ahead"` never drop an active `pro` architecture session down to `flash_lite` or invalidate your prompt cache.
- **Automatic Subagent Right-Sizing (`PreToolUse`):** Intercepts `invoke_subagent` calls when `Model` is `"inherit"` and assigns `flash_lite`, `flash`, or `pro` based on each subagent's specific task.
- **MCP Server (`create_decision` & `route_prompt`):** Exposes calibrated decision gates directly to your agents for confidence-gated tool execution, rubric grading, and triage.
- **Pure Python 3.11+ Standard Library:** Runs out of the box with zero mandatory `pip` packages. Automatically uses `google-auth` if present, and falls back to native stdlib OAuth2 refresh and GCE/Cloud Run/GKE Metadata Server queries.

## Authentication Modes

| Mode | Endpoint | Credentials Used | Best For |
| :--- | :--- | :--- | :--- |
| **`gemini_adc`** *(Default)* | `generativelanguage.googleapis.com/v1beta/decisions` | Keyless OAuth2 ADC + `x-goog-user-project` | Immediate keyless rollout on any GCP project with Generative Language API enabled |
| **`vertex_adc`** | `{region-}aiplatform.googleapis.com/v1beta1/.../decisions` | Keyless OAuth2 ADC (`cloud-platform` scope) | Enterprise GCP projects enrolled in Vertex AI Decisions Preview (VPC-SC, regional endpoints) |
| **`api_key`** | `generativelanguage.googleapis.com/v1beta/decisions` | `GEMINI_API_KEY` env var | Solo prototyping in Google AI Studio without a GCP project |

See [docs/AUTHENTICATION_ADC.md](docs/AUTHENTICATION_ADC.md) for credential chain details and troubleshooting, and [docs/CUSTOMER_ONBOARDING.md](docs/CUSTOMER_ONBOARDING.md) for enterprise rollout guidance.

## Quick Start (Keyless ADC)

### 1. Enable the API and Configure Workstation ADC

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
```

> **Running on Cloud Workstations, GCE, Cloud Run, or GKE?** Skip `gcloud auth application-default login`. The plugin requests the required OAuth scopes directly from the Compute Metadata Server using the attached service account.

### 2. Install the Plugin into Antigravity

```bash
git clone https://github.com/YOUR_ORG/gemini-decisions-router.git
cd gemini-decisions-router
chmod +x install.sh
./install.sh
```

By default, `./install.sh` installs the plugin into `~/.gemini/config/plugins/gemini-decisions-router` (override with `ANTIGRAVITY_PLUGINS_DIR`).

### 3. Verify Authentication and Live Routing

```bash
# Persist GOOGLE_CLOUD_PROJECT in your shell profile (~/.zshrc or ~/.bashrc)
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Verify ADC token resolution and run a live 1-question probe
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live

# Test Tier 1 calibrated prompt routing
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py route \
  --prompt "Design a multi-region Cloud Spanner schema with conflict-free failover" \
  --no-fast-rules
```

Reload Antigravity after installation to activate the MCP server, lifecycle hooks, and skill.

## Quick Start (Optional API Key Fallback)

If you are prototyping without a Google Cloud project, export a Google AI Studio key before starting Antigravity:

```bash
export GEMINI_API_KEY="your-ai-studio-api-key"
./install.sh
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live
```

Never store `GEMINI_API_KEY` or OAuth tokens in `config.json`; the plugin rejects config files containing secret keys.

See [docs/API_KEY_QUICKSTART.md](docs/API_KEY_QUICKSTART.md) for a full walkthrough of `GEMINI_API_KEY` installation, hook/MCP routing, and live telemetry.

## Router Modes

The plugin supports three operating modes:

- **`advisor` (Default):** Evaluates every turn, displays a zero-token collapsible routing badge in the UI with the recommended model and confidence score, and right-sizes `invoke_subagent` calls without interrupting your active chat model.
- **`auto_switch`:** Automatically switches the active Antigravity conversation model when confidence exceeds `min_confidence` (default `0.45`) and the local Antigravity Language Server endpoint is reachable.
- **`off`:** Disables `PreInvocation` and `PreToolUse` hooks while keeping the `create_decision` and `route_prompt` MCP tools available on demand.

Switch modes at any time from the terminal:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode advisor
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py set-mode auto_switch --min-confidence 0.55
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py stats
```

## MCP Tools Reference

Once installed, agents in Antigravity can call two MCP tools under the `decisions` server:

### `create_decision`

Evaluates shared text or allowlisted local files once across 1 to 64 independent structured questions:

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

### `route_prompt`

Scores any prompt or subagent task description and returns `recommended_tier` (`flash_lite`, `flash`, `pro`), `recommended_model_id`, `thinking_level` (`1..4`), `confidence`, and raw probabilities.

## Important Configuration Note

Do **not** add `gemini-decision-1` to Antigravity's interactive chat model dropdown in `settings.json`. Decision models answer `CreateDecision` requests with calibrated probabilities and intentionally return zero text tokens on standard `GenerateContent` chat calls.

## Running Tests

```bash
python3 -m pytest -v tests/test_external_plugin.py
```

## Repository Layout

- `plugin.json`, `mcp_config.json`, `hooks.json`: Antigravity plugin, MCP, and hook manifests
- `decisions_engine.py`: Keyless ADC resolver, Decisions HTTP client, input/file security validator, and 2-tier router
- `mcp_server.py`: Stdio JSON-RPC 2.0 MCP server (`create_decision`, `route_prompt`)
- `router_hook.py`: `PreInvocation` / `PreToolUse` hooks and operator CLI (`auth-check`, `status`, `route`, `set-mode`, `stats`)
- `skills/gemini-decisions-router/SKILL.md`: Agent skill instructions loaded by Antigravity
- `docs/CUSTOMER_ONBOARDING.md`: Enterprise Google Cloud customer rollout guide
- `docs/AUTHENTICATION_ADC.md`: Keyless ADC architecture and troubleshooting guide
- `docs/API_KEY_QUICKSTART.md`: `GEMINI_API_KEY` install, hook/MCP routing, and telemetry walkthrough
- `docs/AUTO_ROUTING_TEST_STRATEGY.md`: Configurable target model catalog, auto-routing test strategy, and live `telemetry.jsonl` evaluation results
- `tests/test_external_plugin.py`: Unit, mock HTTP, security boundary, and repository hygiene test suite

## License

Apache License 2.0. See [LICENSE](LICENSE).
