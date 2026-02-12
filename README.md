# context-research

Research codebase for long-context LLM systems experiments.

This repository is now wired for strict real-backend execution in E0:

- `vLLM` is required by default.
- simulated backend fallback is opt-in only.
- E0 requires a fixed `seed`.
- aggregated and disaggregated paths both execute concrete prefill/decode phases.

## Lightning A100 Quickstart

Use a Lightning AI Linux studio with NVIDIA GPU (A100 40GB).

Detailed setup notes: `docs/lightning-a100-setup.md`

```bash
cp .env.example .env
set -a
source .env
set +a

./scripts/bootstrap.sh --dev
# If bootstrap created .venv, activate it:
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

./scripts/run_experiment.sh e0
./scripts/run_experiment.sh e0 --serving-config configs/serving/disaggregated.yaml

./scripts/make_report.sh latest
```

## Strict Backend Behavior

Default behavior is strict:

```bash
./scripts/run_experiment.sh e0
```

If you explicitly need simulated fallback (not for production metrics):

```bash
export CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND=1
./scripts/run_experiment.sh e0
```

## Artifact Outputs

- Run metadata: `outputs/runs/<run_id>/run_metadata.json`
- E0 summary: `outputs/runs/<run_id>/e0_summary.json`
- Markdown report: `outputs/reports/<run_id>.md`

## Fillers (Env Vars / Lightning URLs Only)

All fillers are env vars. No non-env placeholders are required.

| Filler | Purpose | Where Used |
| :--- | :--- | :--- |
| `HF_TOKEN` | Hugging Face auth for gated or rate-limited pulls | runtime environment consumed by `huggingface-hub`/vLLM model download path |
| `HF_HOME` | Hugging Face cache location | `src/context_research/experiments/runner.py` (`environment_manifest.hf_home`) |
| `CONTEXT_RESEARCH_REQUIREMENTS_FILE` | alternate requirements file for bootstrap | `scripts/bootstrap.sh` |
| `CONTEXT_RESEARCH_EXTRA_REQUIREMENTS` | extra pip requirements for host-specific overrides | `scripts/bootstrap.sh` |
| `CONTEXT_RESEARCH_USE_EXISTING_ENV` | force bootstrap to skip `.venv` creation | `scripts/bootstrap.sh` |
| `CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND` | opt-in simulated fallback for E0 CLI wrapper | `scripts/run_experiment.sh` |
| `LIGHTNING_STUDIO_URL` | Lightning studio URL (for run traceability) | `src/context_research/experiments/runner.py` (`environment_manifest.lightning_env`) |
| `LIGHTNING_RUN_URL` | Lightning run/job URL | `src/context_research/experiments/runner.py` (`environment_manifest.lightning_env`) |
| `LIGHTNING_WORKSPACE_URL` | Lightning workspace URL | `src/context_research/experiments/runner.py` (`environment_manifest.lightning_env`) |

### Lightning URL Fill Examples

All examples below are env-var-based URL fillers:

```bash
export LIGHTNING_STUDIO_URL="https://lightning.ai/${LIGHTNING_ORG}/studios/${LIGHTNING_STUDIO_ID}"
export LIGHTNING_RUN_URL="https://lightning.ai/${LIGHTNING_ORG}/jobs/${LIGHTNING_JOB_ID}"
export LIGHTNING_WORKSPACE_URL="https://lightning.ai/${LIGHTNING_ORG}/home"
```

## Commands

Create run metadata only:

```bash
python -m context_research.cli init-run \
  --model-name "Qwen/Qwen3-8B" \
  --config-path configs/experiments/e0_smoke.yaml
```

List CLI commands:

```bash
python -m context_research.cli --help
```
