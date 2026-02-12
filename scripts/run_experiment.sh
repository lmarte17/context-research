#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXPERIMENT="${1:-e0}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${EXPERIMENT}" in
  e0)
    ALLOW_SIMULATED="${CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND:-0}"
    EXTRA_ARGS=()
    if [[ $# -gt 0 ]]; then
      EXTRA_ARGS=("$@")
    fi
    if [[ "${ALLOW_SIMULATED}" == "1" || "${ALLOW_SIMULATED}" == "true" ]]; then
      EXTRA_ARGS+=("--allow-simulated-backend")
    fi
    if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
      PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
        python -m context_research.cli run-e0
    else
      PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
        python -m context_research.cli run-e0 "${EXTRA_ARGS[@]}"
    fi
    ;;
  *)
    echo "Unsupported experiment '${EXPERIMENT}'. Currently supported: e0" >&2
    exit 2
    ;;
esac
