#!/usr/bin/env bash
# Installs the Gemini Decisions Router plugin into Antigravity's plugin directory
# and installs the `agy-router` and `agyr` CLI wrappers into ~/.local/bin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_BASE="${ANTIGRAVITY_PLUGINS_DIR:-${HOME}/.gemini/config/plugins}"
TARGET_DIR="${TARGET_BASE}/gemini-decisions-router"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"
WRAPPER_PATH="${BIN_DIR}/agy-router"
AGYR_PATH="${BIN_DIR}/agyr"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 (3.11+) is required to run gemini-decisions-router." >&2
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
  raise SystemExit("Error: Python 3.11 or newer is required.")
PY

mkdir -p \
  "${TARGET_DIR}/assets" \
  "${TARGET_DIR}/agents" \
  "${TARGET_DIR}/rules" \
  "${TARGET_DIR}/skills/gemini-decisions-router" \
  "${BIN_DIR}"

cp "${SCRIPT_DIR}/plugin.json" "${TARGET_DIR}/plugin.json"
cp "${SCRIPT_DIR}/mcp_config.json" "${TARGET_DIR}/mcp_config.json"
cp "${SCRIPT_DIR}/hooks.json" "${TARGET_DIR}/hooks.json"
cp "${SCRIPT_DIR}/decisions_engine.py" "${TARGET_DIR}/decisions_engine.py"
cp "${SCRIPT_DIR}/mcp_server.py" "${TARGET_DIR}/mcp_server.py"
cp "${SCRIPT_DIR}/router_hook.py" "${TARGET_DIR}/router_hook.py"
cp "${SCRIPT_DIR}/assets/logo.svg" "${TARGET_DIR}/assets/logo.svg"
cp "${SCRIPT_DIR}/agents/intent-router.md" "${TARGET_DIR}/agents/intent-router.md"
cp "${SCRIPT_DIR}/agents/routed-specialist.md" "${TARGET_DIR}/agents/routed-specialist.md"
cp "${SCRIPT_DIR}/rules/AGENTS.md" "${TARGET_DIR}/rules/AGENTS.md"
cp "${SCRIPT_DIR}/skills/gemini-decisions-router/SKILL.md" \
   "${TARGET_DIR}/skills/gemini-decisions-router/SKILL.md"

chmod +x "${TARGET_DIR}/mcp_server.py" "${TARGET_DIR}/router_hook.py"

cat > "${WRAPPER_PATH}" <<EOF
#!/usr/bin/env bash
exec python3 "${TARGET_DIR}/router_hook.py" "\$@"
EOF
chmod +x "${WRAPPER_PATH}"

cat > "${AGYR_PATH}" <<EOF
#!/usr/bin/env bash
exec python3 "${TARGET_DIR}/router_hook.py" launch "\$@"
EOF
chmod +x "${AGYR_PATH}"

echo "Installed gemini-decisions-router to: ${TARGET_DIR}"
echo "Installed CLI wrapper to: ${WRAPPER_PATH}"
echo "Installed routed agy launcher to: ${AGYR_PATH}"

if command -v agy >/dev/null 2>&1; then
  echo ""
  echo "Validating plugin manifest with agy:"
  agy plugin validate "${TARGET_DIR}" --strict
fi

echo ""
echo "Current router status:"
python3 "${TARGET_DIR}/router_hook.py" status
echo ""
echo "Next step: run 'agy-router auth-setup --from-env' (or 'agy-router auth-check --live') to verify credentials, then launch 'agyr \"your prompt\"' or reload Antigravity."
