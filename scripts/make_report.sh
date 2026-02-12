#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TARGET="${1:-latest}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
  python -m context_research.profiling.reporter "${RUN_TARGET}"
