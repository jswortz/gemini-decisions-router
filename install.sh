#!/usr/bin/env bash
# Installs the Gemini Decisions Router plugin into Antigravity's plugin directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_BASE="${ANTIGRAVITY_PLUGINS_DIR:-${HOME}/.gemini/config/plugins}"
TARGET_DIR="${TARGET_BASE}/gemini-decisions-router"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 (3.11+) is required to run gemini-decisions-router." >&2
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
  raise SystemExit("Error: Python 3.11 or newer is required.")
PY

mkdir -p "${TARGET_DIR}/assets" "${TARGET_DIR}/skills/gemini-decisions-router"

cp "${SCRIPT_DIR}/plugin.json" "${TARGET_DIR}/plugin.json"
cp "${SCRIPT_DIR}/mcp_config.json" "${TARGET_DIR}/mcp_config.json"
cp "${SCRIPT_DIR}/hooks.json" "${TARGET_DIR}/hooks.json"
cp "${SCRIPT_DIR}/decisions_engine.py" "${TARGET_DIR}/decisions_engine.py"
cp "${SCRIPT_DIR}/mcp_server.py" "${TARGET_DIR}/mcp_server.py"
cp "${SCRIPT_DIR}/router_hook.py" "${TARGET_DIR}/router_hook.py"
cp "${SCRIPT_DIR}/assets/logo.svg" "${TARGET_DIR}/assets/logo.svg"
cp "${SCRIPT_DIR}/skills/gemini-decisions-router/SKILL.md" \
   "${TARGET_DIR}/skills/gemini-decisions-router/SKILL.md"

chmod +x "${TARGET_DIR}/mcp_server.py" "${TARGET_DIR}/router_hook.py"

echo "Installed gemini-decisions-router to: ${TARGET_DIR}"
echo ""
echo "Current router status:"
python3 "${TARGET_DIR}/router_hook.py" status
echo ""
echo "Next step: run 'python3 ${TARGET_DIR}/router_hook.py auth-check --live' to verify credentials, then reload Antigravity."
