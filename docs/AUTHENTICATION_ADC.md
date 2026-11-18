# Keyless Authentication with Application Default Credentials (ADC)

Yes, you can run the **Gemini Decisions Router** plugin without creating, storing, or rotating any `GEMINI_API_KEY`. This document explains how keyless ADC works in the plugin, how credentials are discovered, and how to resolve common Google Cloud IAM and OAuth scope errors.

---

## 1. How Keyless ADC Works Across Endpoints

The plugin supports two keyless OAuth2 paths plus an optional API key fallback:

### Mode 1: `gemini_adc` (Default Keyless Mode)

- **Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/decisions`
- **Headers Sent:**
  - `Authorization: Bearer <oauth2_access_token>`
  - `x-goog-user-project: <GOOGLE_CLOUD_PROJECT>`
- **OAuth Scopes Requested:**
  - `https://www.googleapis.com/auth/cloud-platform`
  - `https://www.googleapis.com/auth/generative-language.retriever`
- **Why Both Scopes Matter:** A default `gcloud auth application-default login` on a laptop only requests `cloud-platform`. The `generativelanguage.googleapis.com` service requires an explicit Generative Language scope (`generative-language.retriever` or `generative-language`) on user OAuth tokens, plus a billing/quota project header (`x-goog-user-project`). Service accounts and GCE/GKE/Cloud Run Metadata Servers mint both scopes automatically without interactive login.

### Mode 2: `vertex_adc` (Vertex AI Endpoint)

- **Endpoint:**
  - Global (`location=global`): `POST https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/decisions`
  - Regional (for example `location=us-central1`): `POST https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/us-central1/decisions`
- **Headers Sent:**
  - `Authorization: Bearer <oauth2_access_token>`
- **OAuth Scopes Requested:**
  - `https://www.googleapis.com/auth/cloud-platform`
- **Availability Note:** Vertex AI's `CreateDecisionHttp` method is currently gated by project allowlisting during Preview. Enable `GOOGLE_GENAI_USE_VERTEXAI=true` (or `GEMINI_DECISIONS_AUTH_MODE=vertex_adc`) once your GCP project is enrolled.

### Automatic Mode Selection (`auth_mode = "auto"`)

When `GEMINI_DECISIONS_AUTH_MODE` is unset or `"auto"`, `load_settings()` selects:

1. `vertex_adc` if `GOOGLE_GENAI_USE_VERTEXAI` is `"true"`, `"1"`, or `"yes"`.
2. `api_key` if `GEMINI_API_KEY` is non-empty in the environment.
3. `gemini_adc` otherwise (keyless ADC by default).

---

## 2. Built-In ADC Discovery Chain

`decisions_engine.resolve_adc_token()` resolves short-lived tokens using an in-memory TTL cache (refreshing 60 seconds before expiry) in this order:

1. **`GOOGLE_OAUTH_ACCESS_TOKEN` Environment Variable:** Used by CI/CD steps such as `google-github-actions/auth` with Workload Identity Federation.
2. **`google-auth` Python Package (If Installed):** Calls `google.auth.default(scopes=...)` and refreshes credentials through a built-in Python stdlib `urllib.request` TLS 1.2+ transport (no `requests` package needed). Supports user ADC, service accounts, impersonated credentials, and external Workload Identity pools.
3. **Pure-Stdlib User ADC File (`~/.config/gcloud/application_default_credentials.json` or `$GOOGLE_APPLICATION_CREDENTIALS`):** Exchanges `authorized_user` refresh tokens directly with `https://oauth2.googleapis.com/token` when `google-auth` is not installed.
4. **Pure-Stdlib Compute Metadata Server (`http://169.254.169.254`):** Requests scoped tokens (`?scopes=...`) and project ID directly from the local Metadata Server on Cloud Workstations, GCE, Cloud Run, and GKE Workload Identity, even in minimal containers without `gcloud` or `pip` packages.
5. **`gcloud auth application-default print-access-token` Fallback:** Invokes the `gcloud` binary directly (`shell=False`) as a final fallback.

---

## 3. Verifying Your Setup

Run the built-in diagnostic command to verify credential resolution, effective project ID, target URL, and live API connectivity:

```bash
python3 ~/.gemini/config/plugins/gemini-decisions-router/router_hook.py auth-check --live
```

Expected healthy output on `gemini_adc`:

```json
{
  "ok": true,
  "auth_mode": "gemini_adc",
  "model": "gemini-decision-1",
  "project_id": "acme-ai-devtools-prod",
  "location": "global",
  "credential_source": "google.auth.default",
  "endpoint": "https://generativelanguage.googleapis.com/v1beta/decisions",
  "live_probe": {
    "backend": "gemini_adc",
    "latency_ms": 168,
    "answer": {
      "type": "noul",
      "noul": 0.96,
      "value": true,
      "confidence": 0.94
    }
  }
}
```

---

## 4. Troubleshooting Common ADC Errors

### Error 1: `ACCESS_TOKEN_SCOPE_INSUFFICIENT` (HTTP 403 on `gemini_adc`)

**Cause:** Your workstation ADC file (`~/.config/gcloud/application_default_credentials.json`) was created with default `gcloud` scopes (`cloud-platform` only) and lacks `generative-language.retriever`.

**Fix:** Re-run ADC login once with both scopes:

```bash
gcloud auth application-default login \
  --project="${GOOGLE_CLOUD_PROJECT}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever"
```

### Error 2: `USER_PROJECT_DENIED` or `SERVICE_DISABLED` (HTTP 403 on `gemini_adc`)

**Cause:** Either `generativelanguage.googleapis.com` is not enabled on `GOOGLE_CLOUD_PROJECT`, or your user/service account lacks `serviceusage.services.use` on that project.

**Fix:**

```bash
gcloud services enable generativelanguage.googleapis.com --project="${GOOGLE_CLOUD_PROJECT}"

gcloud projects add-iam-policy-binding "${GOOGLE_CLOUD_PROJECT}" \
  --member="user:you@yourdomain.com" \
  --role="roles/serviceusage.serviceUsageConsumer"
```

### Error 3: `Method CreateDecisionHttp not found for service aiplatform.googleapis.com` (HTTP 404 on `vertex_adc`)

**Cause:** `GOOGLE_GENAI_USE_VERTEXAI=true` or `GEMINI_DECISIONS_AUTH_MODE=vertex_adc` is set, and your GCP project ID has not yet been added to the Vertex AI Decisions Preview allowlist.

**Fix:** Switch immediately to keyless `gemini_adc` while your Google Cloud account team enrolls the project on Vertex AI:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=false
export GEMINI_DECISIONS_AUTH_MODE=gemini_adc
```

### Error 4: `Could not locate Google Cloud Application Default Credentials (ADC)`

**Cause:** Neither `~/.config/gcloud/application_default_credentials.json`, `$GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_OAUTH_ACCESS_TOKEN`, nor a Compute Metadata Server was found.

**Fix:** Install the Google Cloud SDK and run the `gcloud auth application-default login` command above, or attach a service account when running on Google Cloud compute infrastructure.
