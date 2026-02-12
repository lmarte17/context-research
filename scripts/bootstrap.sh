#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNTIME_REQUIREMENTS_FILE="${CONTEXT_RESEARCH_REQUIREMENTS_FILE:-requirements.txt}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${RUNTIME_REQUIREMENTS_FILE}"

if [[ -n "${CONTEXT_RESEARCH_EXTRA_REQUIREMENTS:-}" ]]; then
  read -r -a EXTRA_REQUIREMENTS <<< "${CONTEXT_RESEARCH_EXTRA_REQUIREMENTS}"
  python -m pip install "${EXTRA_REQUIREMENTS[@]}"
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
  python -m pip install -r requirements-dev.txt
fi

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"
