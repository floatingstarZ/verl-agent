#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QGENIE_CLI_HOME="${QGENIE_CLI_HOME:-$ROOT_DIR/.qgenie-agent}"
export PATH="$QGENIE_CLI_HOME/bin:$PATH"

if [[ ! -x "$QGENIE_CLI_HOME/bin/qgenie" ]]; then
  echo "qgenie not found at $QGENIE_CLI_HOME/bin/qgenie" >&2
  echo "Please run installer first." >&2
  exit 1
fi

exec "$QGENIE_CLI_HOME/bin/qgenie" agent "$@"
