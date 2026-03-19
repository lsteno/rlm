import threading
import time
from unittest.mock import patch

import pytest

from rlm.clients.base_lm import BaseLM
from rlm.core.comms_utils import LMRequest, send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.core.rlm import RLM


class RecordingBatchClient(BaseLM):
    def __init__(self):
        super().__init__(model_name="recording")
        self.batch_sizes: list[int] = []
        self.last_prompt_tokens = 1
        self.last_completion_tokens = 1

    def completion(self, prompt, model=None) -> str:
        return f"single:{prompt}"

    async def acompletion(self, prompt, model=None) -> str:
        return f"async:{prompt}"

    async def acompletion_batched(self, prompts, model=None) -> list[str]:
        self.batch_sizes.append(len(prompts))
        await __import__("asyncio").sleep(0.02)
        return [f"batched:{prompt}" for prompt in prompts]

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=1,
                    total_output_tokens=1,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(total_calls=1, total_input_tokens=1, total_output_tokens=1)


def test_lm_handler_coalesces_cross_request_batches():
    client = RecordingBatchClient()
    handler = LMHandler(
        client,
        batch_wait_ms=50,
        batch_max_size=64,
        max_inflight_batches=2,
        max_pending_prompts=1024,
    )
    handler.start()

    results: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def call_one(prompt: str) -> None:
        responses = send_lm_request_batched(handler.address, [prompt])
        response = responses[0]
        with lock:
            if response.success:
                results.append(response.chat_completion.response)
            else:
                errors.append(response.error)

    threads = [threading.Thread(target=call_one, args=(f"p{i}",)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    handler.stop()

    assert not errors
    assert len(results) == 12
    assert max(client.batch_sizes) > 1, f"Expected coalesced batch, got sizes={client.batch_sizes}"


def test_lm_handler_preserves_prompt_order_in_batch_response():
    client = RecordingBatchClient()
    handler = LMHandler(client, batch_wait_ms=10, batch_max_size=16)
    handler.start()

    responses = send_lm_request_batched(handler.address, ["a", "b", "c", "d"])
    handler.stop()

    outputs = [resp.chat_completion.response for resp in responses if resp.success]
    assert outputs == ["batched:a", "batched:b", "batched:c", "batched:d"]


def test_rlm_forwards_lm_handler_kwargs():
    class FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def cleanup(self):
            return None

    with patch("rlm.core.rlm.get_client") as mock_get_client, patch(
        "rlm.core.rlm.get_environment"
    ) as mock_get_environment:
        mock_get_client.return_value = RecordingBatchClient()
        mock_get_environment.return_value = FakeEnv()

        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "mock-model"},
            lm_handler_kwargs={"batch_wait_ms": 77, "batch_max_size": 33},
        )

        with rlm._spawn_completion_context("hello") as (handler, _):
            assert handler._batch_executor.batch_max_size == 33
            assert handler._batch_executor.batch_wait_s == pytest.approx(0.077, abs=0.01)
