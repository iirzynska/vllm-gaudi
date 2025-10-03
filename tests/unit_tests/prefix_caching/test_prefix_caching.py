import pytest
import torch
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData, CachedRequestData
from vllm.sampling_params import SamplingParams
from vllm.config import (
    VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig
)
from vllm.attention import Attention
from vllm.platforms import current_platform

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE = current_platform.device_type

def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config

@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    model_config = vllm_config.model_config
    num_heads = 2
    head_size = 64
    # Patch Attention for static context if needed
    vllm_config.compilation_config.static_forward_context = {"layer.0": Attention(num_heads, head_size, 0.1)}
    runner = HPUModelRunner(vllm_config, DEVICE)
    # Optionally initialize KV cache here if needed for your backend
    return runner

def make_new_request(req_id, prompt_token_ids, num_computed_tokens=0):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_ids=[[0]],
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )

@pytest.mark.parametrize("prompt1, prompt2, num_common_prefix, expected_tokens", [
    ([1, 2, 3, 4], [1, 2, 3, 4], 4, 0),      # full prefix cache hit
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 3, 2) # partial prefix cache hit (3 cached, 2 new)
])
def test_prefix_cache_hits(model_runner, prompt1, prompt2, num_common_prefix, expected_tokens):
    req_id1 = "req1"
    req_id2 = "req2"

    # First request: all tokens need compute
    new_req1 = make_new_request(req_id1, prompt1)
    sched_out1 = SchedulerOutput(
        scheduled_new_reqs=[new_req1],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id1: len(prompt1)},
        total_num_scheduled_tokens=len(prompt1),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )
    model_runner._update_states(sched_out1)
    assert req_id1 in model_runner.requests

    # Second request: simulate prefix cache hit/partial hit
    new_req2 = make_new_request(req_id2, prompt2)
    sched_out2 = SchedulerOutput(
        scheduled_new_reqs=[new_req2],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id2: expected_tokens},
        total_num_scheduled_tokens=expected_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=num_common_prefix,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )
    model_runner._update_states(sched_out2)
    assert req_id2 in model_runner.requests

# @pytest.mark.parametrize("prompt, cache_first, cache_second", [
#     ([10, 11, 12], 3, 0), # first: all tokens cached, second: cache reset, all tokens need compute
# ])
# def test_prefix_cache_reset(model_runner, prompt, cache_first, cache_second):
#     req_id = "req_reset"
#     new_req = make_new_request(req_id, prompt)
#     # First: all tokens cached (simulate by setting num_scheduled_tokens=0)
#     sched_out1 = SchedulerOutput(
#         scheduled_new_reqs=[new_req],
#         scheduled_cached_reqs=CachedRequestData.make_empty(),
#         num_scheduled_tokens={req_id: 0},
#         total_num_scheduled_tokens=0,
#         scheduled_spec_decode_tokens={},
#         scheduled_encoder_inputs={},
#         num_common_prefix_blocks=cache_first,
#         finished_req_ids=set(),
#         free_encoder_mm_hashes=[],
#         structured_output_request_ids={},
#         grammar_bitmask=None,
#     )
#     model_runner._update_states(sched_out1)
#     assert req_id in model_runner.requests

#     # Second: cache reset, all tokens need compute
#     sched_out2 = SchedulerOutput(
#         scheduled_new_reqs=[new_req],
#         scheduled_cached_reqs=CachedRequestData.make_empty(),
#         num_scheduled_tokens={req_id: len(prompt)},
#         total_num_scheduled_tokens=len(prompt),
#         scheduled_spec_decode_tokens={},
#         scheduled_encoder_inputs={},
#         num_common_prefix_blocks=cache_second,
#         finished_req_ids=set(),
#         free_encoder_mm_hashes=[],
#         structured_output_request_ids={},
#         grammar_bitmask=None,
#     )
#     model_runner._update_states(sched_out2)
#     assert req_id in model_runner.requests