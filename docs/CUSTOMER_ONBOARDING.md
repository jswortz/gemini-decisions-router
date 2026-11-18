# Google Cloud Customer Onboarding Guide

This guide walks Google Cloud platform administrators, security teams, and engineering leads through rolling out the **Gemini Decisions Router** plugin across Antigravity developer environments without distributing API keys.

---

## 1. Choose Your Authentication Architecture

Enterprise Google Cloud environments normally enforce Organization Policy `constraints/iam.disableServiceAccountKeyCreation` and prohibit unmanaged developer API keys. This plugin supports keyless **Application Default Credentials (ADC)** out of the box across two Google Cloud endpoints:

| Criterion | Path A: `gemini_adc` (Recommended Start) | Path B: `vertex_adc` (Vertex AI Preview) | Path C: `api_key` (Sandbox Only) |
| :--- | :--- | :--- | :--- |
| **Endpoint** | `generativelanguage.googleapis.com/v1beta/decisions` | `{region-}aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/decisions` | `generativelanguage.googleapis.com/v1beta/decisions` |
| **Static Keys Needed?** | **No** (Short-lived OAuth2 ADC) | **No** (Short-lived OAuth2 ADC) | Yes (`GEMINI_API_KEY`) |
| **Org Policy Compliant?** | Yes (`disableServiceAccountKeyCreation` safe) | Yes (`disableServiceAccountKeyCreation` safe) | Often restricted in enterprise orgs |
| **GCP Service to Enable** | `generativelanguage.googleapis.com` | `aiplatform.googleapis.com` | None (AI Studio project) |
| **Preview Allowlist?** | **Available immediately** | Requires Vertex AI Decisions Preview project allowlisting | Available immediately |
| **Regional Endpoints / VPC-SC** | Global endpoint | Supports `global` and regional endpoints (for example `us-central1`) | Global endpoint |

### Recommended Customer Strategy

1. **Start immediately on `gemini_adc`:** Enable `generativelanguage.googleapis.com` on a designated Google Cloud billing/quota project so developers get keyless OAuth2 authentication, per-project quota governance, and Cloud Audit Logs on day one.
2. **Switch to `vertex_adc` when Vertex AI Preview enrollment completes:** If your organization requires regional endpoints (`us-central1`) or Vertex AI VPC Service Controls, ask your Google Cloud account team to enroll your project ID in the **Vertex AI Decisions API Preview**. Once enrolled, set `GOOGLE_GENAI_USE_VERTEXAI=true` with zero code changes.

---

## 2. Cloud Platform Admin Prerequisites (One-Time Setup)

### Option A: Setup for `gemini_adc` (Immediate Keyless Rollout)

1. Select or create a Google Cloud project for Antigravity developer routing (for example `acme-ai-devtools-prod`).
2. Enable the Generative Language API:
   ```bash
   gcloud services enable generativelanguage.googleapis.com \
     --project="acme-ai-devtools-prod"
   ```
3. Grant your developer Google Group the **Service Usage Consumer** role so their ADC tokens can attribute quota to the project via `x-goog-user-project`:
   ```bash
   gcloud projects add-iam-policy-binding acme-ai-devtools-prod \
     --member="group:antigravity-devs@acme.com" \
     --role="roles/serviceusage.serviceUsageConsumer"
   ```

### Option B: Setup for `vertex_adc` (After Vertex Preview Enrollment)

1. Confirm with your Google Cloud account team that your project ID is allowlisted for `CreateDecisionHttp` on `aiplatform.googleapis.com`.
2. Enable the Vertex AI API:
   ```bash
   gcloud services enable aiplatform.googleapis.com \
     --project="acme-ai-devtools-prod"
   ```
3. Grant your developer Google Group (or workstation service account) **Vertex AI User** (`roles/aiplatform.user`):
   ```bash
   gcloud projects add-iam-policy-binding acme-ai-devtools-prod \
     --member="group:antigravity-devs@acme.com" \
     --role="roles/aiplatform.user"
   ```

---

## 3. Rollout by Environment Type

### A. Developer Laptops (macOS / Linux)

Each developer runs a one-time ADC login that includes the Generative Language retriever OAuth scope, installs the plugin, and sets their project ID:

```bash
# 1. Authenticate keyless ADC with Cloud Platform + Generative Language scopes
gcloud auth application-default login \
  --project="acme-ai-devtools-prod" \
  --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever"

# 2. Add environment variables to ~/.zshrc or ~/.bashrc
cat <<'EOF' >> ~/.zshrc
export GOOGLE_CLOUD_PROJECT="acme-ai-devtools-prod"
export GEMINI_DECISIONS_AUTH_MODE="gemini_adc"
EOF
source ~/.zshrc

# 3. Install plugin and run live verification
./install.sh
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live
```

*(If using `vertex_adc`, set `export GOOGLE_GENAI_USE_VERTEXAI=true` and `export GOOGLE_CLOUD_LOCATION=us-central1`; standard `gcloud auth application-default login` without custom `--scopes` is sufficient for `vertex_adc`.)*

### B. Google Cloud Workstations, Compute Engine, Cloud Run, and GKE

Managed Google Cloud compute environments attach a service account directly via the local Metadata Server (`169.254.169.254`). No interactive `gcloud` login or JSON key file is needed:

1. Ensure the VM or Cloud Workstation configuration grants the `cloud-platform` OAuth scope (or custom service account scopes) and that the attached service account holds `roles/serviceusage.serviceUsageConsumer` (for `gemini_adc`) or `roles/aiplatform.user` (for `vertex_adc`).
2. Set environment variables in the container image or `/etc/profile.d/gemini-decisions.sh`:
   ```bash
   export GOOGLE_CLOUD_PROJECT="acme-ai-devtools-prod"
   export GEMINI_DECISIONS_AUTH_MODE="gemini_adc"
   ```
3. Run `./install.sh` during workstation startup script execution. The plugin queries the Metadata Server directly with `?scopes=...` and caches short-lived tokens in memory.

### C. CI/CD Pipelines (GitHub Actions / GitLab CI via Workload Identity Federation)

For automated evaluation pipelines calling `create_decision` or `route_prompt`, use **Workload Identity Federation** (`google-github-actions/auth@v2`) instead of long-lived secrets:

```yaml
permissions:
  contents: read
  id-token: write

steps:
  - uses: actions/checkout@v4
  - id: auth
    uses: google-github-actions/auth@v2
    with:
      workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
      service_account: ${{ vars.GCP_DECISIONS_SA }}
      token_format: access_token
      access_token_scopes: https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever
  - name: Verify keyless Decisions call
    env:
      GOOGLE_CLOUD_PROJECT: ${{ vars.GCP_PROJECT_ID }}
      GOOGLE_OAUTH_ACCESS_TOKEN: ${{ steps.auth.outputs.access_token }}
      GEMINI_DECISIONS_AUTH_MODE: gemini_adc
    run: python3 router_hook.py auth-check --live
```

---

## 4. Phased Enterprise Adoption Plan

We recommend onboarding engineering teams in two phases:

### Phase 1: Week 1 Shadow & Subagent Right-Sizing (`advisor` Mode)

`advisor` is the default mode after `./install.sh`.

- **Zero disruption to developer muscle memory:** Main chat turns continue using whatever model the developer picked in the Antigravity dropdown.
- **Immediate subagent cost savings:** Whenever an agent spawns helper subagents with `Model="inherit"`, `PreToolUse` automatically routes simple lookup/verification subagents to `flash_lite` or `flash` while reserving `pro` for deep architectural subagents.
- **Transparent UI feedback:** Every turn displays a collapsible `Gemini Decisions Router` badge showing the calibrated task class, confidence percentage, and latency.

At the end of Week 1, review local routing distribution and latency on pilot workstations:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py stats --limit 200
```

### Phase 2: Week 2+ Full Calibrated Routing (`auto_switch` Mode)

Once pilot developers validate the recommendations in `advisor` mode, enable `auto_switch`:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py \
  set-mode auto_switch --min-confidence 0.50
```

- Simple status pings route to `gemini-3.1-flash-lite` (`Low` effort).
- Standard implementation, SQL, and documentation tasks route to `gemini-3-flash` (`Medium` effort).
- Multi-file architecture, concurrency debugging, and security reviews route to `gemini-3.1-pro-high` (`High` or `Max` effort), and **stay locked at `pro`** for subsequent follow-up turns in that conversation.
- Developers can always override the router on any prompt by writing `"use gemini pro"` or `"use flash"`.

---

## 5. Cost, Quota, and Security Profile

### Latency and Token Economics

- **Tier 0 Fast Rules (`<1 ms`, $0):** Handles ~25 to 35% of turns (confirmations, short Turn 2+ continuations, explicit model overrides) locally with zero network calls.
- **Tier 1 `gemini-decision-1` (`~120 to 250 ms`):** Charges **input tokens only** (`$0.10 per 1M input tokens` on published Gemini API pricing) and **zero output tokens**.
- **Monthly Router Cost Estimate:** At ~450 input tokens per routed prompt (rubric + prompt excerpt), **100,000 routed developer prompts cost roughly $4.50 total**, while typically shifting 50 to 70% of routine turns and subagents off frontier `pro` pricing.

### Security Controls Built into the Plugin

1. **No Secret Persistence:** `load_settings()` fails closed if anyone places `api_key`, `access_token`, `refresh_token`, or `client_secret` inside `config.json`.
2. **Token Redaction & Origin Pinning:** HTTPS with TLS 1.2+ is enforced for all non-loopback endpoints, HTTP redirects (`301/302/307/308`) are refused so `Authorization: Bearer` headers never cross origins, and tokens are scrubbed from all exception and stderr strings.
3. **Fail-Open Offline Fallback:** If the network is unreachable or project quota is temporarily exhausted, the router trips a 45-second cooldown latch and falls back to `<1 ms` local lexical classification so developer workflows never stall.
4. **Local File Guardrails:** `create_decision` rejects file inputs unless `allowed_dirs` is explicitly configured, and blocks symlinks (`O_NOFOLLOW`), path traversal, hidden dotfiles, and credential filenames (`.env`, `*.pem`, `*credentials*`).

---

## 6. Customer Go-Live Checklist

- [ ] Designated GCP project created and `GOOGLE_CLOUD_PROJECT` documented for the team
- [ ] `generativelanguage.googleapis.com` (for `gemini_adc`) or `aiplatform.googleapis.com` (for `vertex_adc`) enabled
- [ ] Developer group granted `roles/serviceusage.serviceUsageConsumer` (or `roles/aiplatform.user`)
- [ ] Pilot developer ran `./install.sh` and `router_hook.py auth-check --live` returned `"ok": true`
- [ ] Verified `gemini-decision-1` is **not** added as a chat model in Antigravity `settings.json`
- [ ] Pilot group running in `advisor` mode with `router_hook.py stats` confirming `<250 ms` decision latency
