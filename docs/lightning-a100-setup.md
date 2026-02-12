# Lightning A100 Setup (Qwen3-8B)

This setup targets Lightning AI hosted NVIDIA A100 40GB machines.

## 1. Required Environment Variables

```bash
export HF_TOKEN=""
export HF_HOME=""

export CONTEXT_RESEARCH_REQUIREMENTS_FILE="requirements.txt"
export CONTEXT_RESEARCH_EXTRA_REQUIREMENTS=""
export CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND="0"

export LIGHTNING_STUDIO_URL="https://lightning.ai/${LIGHTNING_ORG}/studios/${LIGHTNING_STUDIO_ID}"
export LIGHTNING_RUN_URL="https://lightning.ai/${LIGHTNING_ORG}/jobs/${LIGHTNING_JOB_ID}"
export LIGHTNING_WORKSPACE_URL="https://lightning.ai/${LIGHTNING_ORG}/home"
```

## 2. Where Each Variable Is Used

| Variable | Used In | Effect |
| :--- | :--- | :--- |
| `HF_TOKEN` | runtime environment | authorizes model pull from Hugging Face when required |
| `CONTEXT_RESEARCH_REQUIREMENTS_FILE` | `scripts/bootstrap.sh` | selects requirements file for runtime install |
| `CONTEXT_RESEARCH_EXTRA_REQUIREMENTS` | `scripts/bootstrap.sh` | appends extra pinned pip packages |
| `CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND` | `scripts/run_experiment.sh` | appends `--allow-simulated-backend` to E0 command |
| `HF_HOME` | `src/context_research/experiments/runner.py` | stored in E0 `environment_manifest.hf_home` |
| `LIGHTNING_STUDIO_URL` | `src/context_research/experiments/runner.py` | stored in E0 `environment_manifest.lightning_env` |
| `LIGHTNING_RUN_URL` | `src/context_research/experiments/runner.py` | stored in E0 `environment_manifest.lightning_env` |
| `LIGHTNING_WORKSPACE_URL` | `src/context_research/experiments/runner.py` | stored in E0 `environment_manifest.lightning_env` |

## 3. Run Sequence

```bash
./scripts/bootstrap.sh --dev
source .venv/bin/activate

./scripts/run_experiment.sh e0
./scripts/run_experiment.sh e0 --serving-config configs/serving/disaggregated.yaml

./scripts/make_report.sh latest
```

## 4. Strictness and Reproducibility Rules

- `configs/experiments/e0_smoke.yaml` now requires `seed` and defaults to `42`.
- `scripts/run_experiment.sh e0` is strict by default and fails if vLLM is unavailable.
- Simulated backend execution is opt-in via `CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND=1`.
