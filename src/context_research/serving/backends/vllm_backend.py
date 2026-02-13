from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
import os
from time import perf_counter
from typing import Any

from context_research.serving.backends.base import GenerationRequest, GenerationResult, ServingBackend


def _safe_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


@dataclass(frozen=True)
class VLLMBackendConfig:
    model: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = True
    dtype: str | None = None
    enforce_eager: bool = True
    enable_thinking: bool = False
    simulate_if_unavailable: bool = False
    simulated_ttft_ms: float = 20.0
    simulated_tpot_ms: float = 1.5
    visible_devices: str | None = None
    device: str | None = None


class VLLMBackend(ServingBackend):
    def __init__(self, config: VLLMBackendConfig) -> None:
        self._config = config
        self._started = False
        self._mode = "not_started"
        self._llm: Any | None = None
        self._sampling_params_cls: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def start(self) -> None:
        if self._started:
            return

        with _temporary_cuda_visible_devices(self._config.visible_devices):
            try:
                from vllm import LLM, SamplingParams  # type: ignore
            except ImportError:
                if not self._config.simulate_if_unavailable:
                    raise RuntimeError(
                        "vLLM is not installed and simulate_if_unavailable=False."
                    ) from None
                self._mode = "simulated"
                self._started = True
                return

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "tensor_parallel_size": self._config.tensor_parallel_size,
                "gpu_memory_utilization": self._config.gpu_memory_utilization,
                "trust_remote_code": self._config.trust_remote_code,
                "enforce_eager": self._config.enforce_eager,
                "enable_thinking": self._config.enable_thinking,
            }
            if self._config.max_model_len is not None:
                kwargs["max_model_len"] = self._config.max_model_len
            if self._config.dtype is not None:
                kwargs["dtype"] = self._config.dtype
            if self._config.device is not None:
                kwargs["device"] = self._config.device

            init_params = set(inspect.signature(LLM.__init__).parameters.keys())
            init_params.discard("self")
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}

            self._llm = LLM(**filtered_kwargs)
            self._sampling_params_cls = SamplingParams
            self._tokenizer = self._maybe_get_tokenizer()
            self._mode = "vllm"
            self._started = True

    def stop(self) -> None:
        self._llm = None
        self._sampling_params_cls = None
        self._tokenizer = None
        self._started = False
        self._mode = "stopped"

    def warmup(self) -> None:
        self._require_started()
        request = GenerationRequest(
            prompt="Warmup request.",
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        self.generate(request)

    def prefill(self, request: GenerationRequest) -> dict[str, Any]:
        self._require_started()
        if self._mode != "vllm":
            prompt_tokens = _safe_token_count(request.prompt)
            return {
                "mode": self._mode,
                "prompt_tokens": prompt_tokens,
                "prefill_ms": self._config.simulated_ttft_ms,
            }

        started = perf_counter()
        token_ids = self._encode_prompt(request.prompt)
        prefill_ms = (perf_counter() - started) * 1000.0
        return {
            "mode": self._mode,
            "prompt_tokens": len(token_ids),
            "prefill_ms": prefill_ms,
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._require_started()
        if self._mode == "vllm":
            return self._generate_vllm(request)
        return self._generate_simulated(request)

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        self._require_started()
        if not requests:
            return []
        if self._mode == "vllm":
            return self._generate_batch_vllm(requests)
        return [self._generate_simulated(request) for request in requests]

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Backend is not started. Call start() first.")

    def _generate_simulated(self, request: GenerationRequest) -> GenerationResult:
        prompt_tokens = _safe_token_count(request.prompt)
        completion_tokens = max(1, min(request.max_new_tokens, 8))
        text = " ".join(["simulated"] * completion_tokens)
        ttft_ms = self._config.simulated_ttft_ms + (0.02 * prompt_tokens)
        tpot_ms = self._config.simulated_tpot_ms if completion_tokens > 1 else 0.0

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            metadata={
                "backend": "vllm",
                "mode": self._mode,
                "requested_max_new_tokens": request.max_new_tokens,
                "latency_source": "simulated",
                "seed": request.seed,
                "visible_devices": self._config.visible_devices,
            },
        )

    def _generate_vllm(self, request: GenerationRequest) -> GenerationResult:
        assert self._llm is not None
        assert self._sampling_params_cls is not None

        sampling_params = self._build_sampling_params(request)
        started = perf_counter()
        outputs = self._llm.generate([request.prompt], sampling_params=sampling_params)
        elapsed_ms = (perf_counter() - started) * 1000.0

        request_output = outputs[0]
        first_output = request_output.outputs[0] if request_output.outputs else None
        text = first_output.text if first_output is not None else ""

        prompt_tokens = self._extract_prompt_tokens(request_output, request.prompt)
        completion_tokens = self._extract_completion_tokens(text, first_output)
        ttft_ms, tpot_ms, latency_source = self._extract_latency_metrics(
            request_output=request_output,
            elapsed_ms=elapsed_ms,
            completion_tokens=completion_tokens,
        )

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            metadata={
                "backend": "vllm",
                "mode": self._mode,
                "elapsed_ms": elapsed_ms,
                "latency_source": latency_source,
                "seed": request.seed,
                "visible_devices": self._config.visible_devices,
            },
        )

    def _generate_batch_vllm(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        assert self._llm is not None
        assert self._sampling_params_cls is not None

        first = requests[0]
        if not all(
            request.max_new_tokens == first.max_new_tokens
            and request.temperature == first.temperature
            and request.top_p == first.top_p
            and request.seed == first.seed
            for request in requests
        ):
            return [self._generate_vllm(request) for request in requests]

        sampling_params = self._build_sampling_params(first)
        prompts = [request.prompt for request in requests]
        started = perf_counter()
        outputs = self._llm.generate(prompts, sampling_params=sampling_params)
        elapsed_ms = (perf_counter() - started) * 1000.0
        if len(outputs) != len(requests):
            return [self._generate_vllm(request) for request in requests]

        fallback_elapsed_ms = elapsed_ms / max(len(requests), 1)
        results: list[GenerationResult] = []
        for request, request_output in zip(requests, outputs):
            first_output = request_output.outputs[0] if request_output.outputs else None
            text = first_output.text if first_output is not None else ""

            prompt_tokens = self._extract_prompt_tokens(request_output, request.prompt)
            completion_tokens = self._extract_completion_tokens(text, first_output)
            ttft_ms, tpot_ms, latency_source = self._extract_latency_metrics(
                request_output=request_output,
                elapsed_ms=fallback_elapsed_ms,
                completion_tokens=completion_tokens,
            )

            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    metadata={
                        "backend": "vllm",
                        "mode": self._mode,
                        "elapsed_ms": fallback_elapsed_ms,
                        "latency_source": latency_source,
                        "seed": request.seed,
                        "visible_devices": self._config.visible_devices,
                        "batch_size": len(requests),
                    },
                )
            )
        return results

    def _maybe_get_tokenizer(self) -> Any | None:
        if self._llm is None:
            return None
        getter = getattr(self._llm, "get_tokenizer", None)
        if callable(getter):
            return getter()
        return None

    def _encode_prompt(self, prompt: str) -> list[int]:
        tokenizer = self._tokenizer or self._maybe_get_tokenizer()
        if tokenizer is None:
            return []

        self._tokenizer = tokenizer
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(prompt)

        if isinstance(token_ids, list):
            return [int(x) for x in token_ids]

        to_list = getattr(token_ids, "tolist", None)
        if callable(to_list):
            values = to_list()
            if isinstance(values, list):
                return [int(x) for x in values]
        return []

    def _build_sampling_params(self, request: GenerationRequest) -> Any:
        assert self._sampling_params_cls is not None
        kwargs: dict[str, Any] = {
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            init_params = set(inspect.signature(self._sampling_params_cls.__init__).parameters)
            if "seed" in init_params:
                kwargs["seed"] = request.seed
        return self._sampling_params_cls(**kwargs)

    def _extract_prompt_tokens(self, request_output: Any, prompt: str) -> int:
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        if isinstance(prompt_token_ids, list):
            return len(prompt_token_ids)
        return _safe_token_count(prompt)

    def _extract_completion_tokens(self, text: str, first_output: Any | None) -> int:
        if first_output is not None:
            token_ids = getattr(first_output, "token_ids", None)
            if isinstance(token_ids, list):
                return len(token_ids)
        return _safe_token_count(text)

    def _extract_latency_metrics(
        self,
        *,
        request_output: Any,
        elapsed_ms: float,
        completion_tokens: int,
    ) -> tuple[float, float, str]:
        metrics = getattr(request_output, "metrics", None)

        arrival_time = _to_float(getattr(metrics, "arrival_time", None))
        first_token_time = _to_float(getattr(metrics, "first_token_time", None))
        last_token_time = _to_float(getattr(metrics, "last_token_time", None))
        finished_time = _to_float(getattr(metrics, "finished_time", None))
        total_metrics_ms: float | None = None
        if arrival_time is not None and finished_time is not None and finished_time >= arrival_time:
            total_metrics_ms = (finished_time - arrival_time) * 1000.0

        ttft_ms: float | None = None
        tpot_ms: float | None = None
        latency_source = "elapsed_wall_fallback"

        if arrival_time is not None and first_token_time is not None and first_token_time >= arrival_time:
            ttft_ms = (first_token_time - arrival_time) * 1000.0
            latency_source = "vllm_metrics"

        if completion_tokens <= 1:
            tpot_ms = 0.0
        elif first_token_time is not None and last_token_time is not None and last_token_time >= first_token_time:
            tpot_ms = ((last_token_time - first_token_time) * 1000.0) / float(
                completion_tokens - 1
            )
            latency_source = "vllm_metrics"
        elif first_token_time is not None and finished_time is not None and finished_time >= first_token_time:
            tpot_ms = ((finished_time - first_token_time) * 1000.0) / float(
                completion_tokens - 1
            )
            latency_source = "vllm_metrics"

        if ttft_ms is None:
            ttft_ms = elapsed_ms

        if completion_tokens <= 1:
            tpot_ms = 0.0
        elif tpot_ms is None or tpot_ms <= 0.0:
            residual_ms = max(elapsed_ms - ttft_ms, 0.0)
            if residual_ms > 0.0:
                tpot_ms = residual_ms / float(completion_tokens - 1)
                if latency_source == "vllm_metrics":
                    latency_source = "vllm_metrics+elapsed_residual_fallback"
                else:
                    latency_source = "elapsed_residual_fallback"
            else:
                reference_ms = (
                    total_metrics_ms
                    if total_metrics_ms is not None and total_metrics_ms > 0.0
                    else max(elapsed_ms, 0.0)
                )
                tpot_ms = reference_ms / float(completion_tokens) if reference_ms > 0.0 else 0.0
                if latency_source == "vllm_metrics":
                    latency_source = "vllm_metrics+elapsed_average_fallback"
                else:
                    latency_source = "elapsed_average_fallback"

        return (ttft_ms, tpot_ms, latency_source)


@contextmanager
def _temporary_cuda_visible_devices(visible_devices: str | None):
    if not visible_devices:
        yield
        return

    key = "CUDA_VISIBLE_DEVICES"
    previous = os.environ.get(key)
    os.environ[key] = visible_devices
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
