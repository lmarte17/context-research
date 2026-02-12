#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNTIME_REQUIREMENTS_FILE="${CONTEXT_RESEARCH_REQUIREMENTS_FILE:-requirements.txt}"
USE_EXISTING_ENV="${CONTEXT_RESEARCH_USE_EXISTING_ENV:-0}"
ACTIVE_ENV_LABEL="system"

if [[ "${USE_EXISTING_ENV}" == "1" || "${USE_EXISTING_ENV}" == "true" ]]; then
  echo "Using existing Python environment (CONTEXT_RESEARCH_USE_EXISTING_ENV=${USE_EXISTING_ENV})."
elif [[ -f ".venv/bin/activate" ]]; then
  # Reuse previously created virtual environment when available.
  source .venv/bin/activate
  ACTIVE_ENV_LABEL=".venv"
else
  VENV_ERR_FILE="$(mktemp)"
  if python3 -m venv .venv 2>"${VENV_ERR_FILE}"; then
    source .venv/bin/activate
    ACTIVE_ENV_LABEL=".venv"
  else
    if grep -q "Venv creation is not allowed" "${VENV_ERR_FILE}"; then
      echo "Venv creation is blocked in this environment; continuing with existing Python environment."
      ACTIVE_ENV_LABEL="system"
    else
      cat "${VENV_ERR_FILE}" >&2
      rm -f "${VENV_ERR_FILE}"
      exit 1
    fi
  fi
  rm -f "${VENV_ERR_FILE}"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --force-reinstall --no-cache-dir --upgrade-strategy eager \
  -r "${RUNTIME_REQUIREMENTS_FILE}"

if [[ -n "${CONTEXT_RESEARCH_EXTRA_REQUIREMENTS:-}" ]]; then
  read -r -a EXTRA_REQUIREMENTS <<< "${CONTEXT_RESEARCH_EXTRA_REQUIREMENTS}"
  python -m pip install --upgrade --force-reinstall --no-cache-dir "${EXTRA_REQUIREMENTS[@]}"
fi

python - <<'PY'
from pathlib import Path
import site

project_src = Path.cwd() / "src"
site_packages = Path(site.getsitepackages()[0])
pth_file = site_packages / "context_research_local.pth"
pth_file.write_text(f"{project_src}\n", encoding="utf-8")
print(f"Wrote {pth_file}")
PY

if [[ "${1:-}" == "--dev" ]]; then
  python -m pip install --upgrade --force-reinstall --no-cache-dir -r requirements-dev.txt
fi

python -m pip check

echo "Bootstrap complete."
if [[ "${ACTIVE_ENV_LABEL}" == ".venv" ]]; then
  echo "Activate with: source .venv/bin/activate"
else
  echo "Using existing environment; no .venv activation required."
fi
