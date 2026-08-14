# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    ExpertLayerPlacement,
    ExpertRoutingCaptureBackend,
    ExpertRoutingStatsRecorder,
    ExpertRoutingStatsTensors,
    ExpertRoutingStep,
    RoutedExpertsCapturer,
    summarize_expert_routing,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    c.flashinfer_device_buffer = None
    c._stats_capture_backends = {}
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        global_num_experts=16,
        quant_method=types.SimpleNamespace(is_monolithic=False),
        expert_map_manager=types.SimpleNamespace(
            get_local_expert_ids=lambda: list(range(16)),
            placement_strategy="linear",
        ),
    )


def _make_monolithic_routed_experts():
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        TrtLlmBf16ExpertsMonolithic,
    )

    fused_experts = TrtLlmBf16ExpertsMonolithic.__new__(TrtLlmBf16ExpertsMonolithic)
    fused_experts.topk = 2
    fused_experts._routing_replay_out = None
    return types.SimpleNamespace(
        global_num_experts=16,
        moe_config=types.SimpleNamespace(is_sequence_parallel=False),
        quant_method=types.SimpleNamespace(
            is_monolithic=True,
            moe_kernel=types.SimpleNamespace(fused_experts=fused_experts),
        ),
        expert_map_manager=types.SimpleNamespace(
            get_local_expert_ids=lambda: list(range(16)),
            placement_strategy="linear",
        ),
    )


def test_summarize_expert_routing_separates_useful_and_physical_assignments():
    modular_routing_data = torch.tensor(
        [
            [[0, 1], [-1, -1]],
            [[2, 3], [-1, -1]],
            [[2, 1], [-1, -1]],
        ],
        dtype=torch.int32,
    )
    flashinfer_routing_data = torch.tensor(
        [
            [[-1, -1], [-1, -1], [-1, -1]],
            [[3, 2], [1, 0], [3, 2]],
        ],
        dtype=torch.int16,
    )
    stats = ExpertRoutingStatsTensors(
        modular_routing_data=modular_routing_data,
        flashinfer_routing_data=flashinfer_routing_data,
        num_reqs=2,
        num_scheduled_tokens=2,
        num_physical_tokens=3,
    )
    step = ExpertRoutingStep(decode_step=7, stats=stats)
    placements = (
        ExpertLayerPlacement(0, 4, (0, 2), "round_robin"),
        ExpertLayerPlacement(
            1,
            4,
            (1, 3),
            "round_robin",
            ExpertRoutingCaptureBackend.FLASHINFER_TRTLLM,
        ),
    )

    summary = summarize_expert_routing(step, placements)

    assert summary == {
        "record_type": "decode_step",
        "decode_step": 7,
        "num_reqs": 2,
        "num_scheduled_tokens": 2,
        "num_physical_tokens": 3,
        "num_padding_tokens": 1,
        "useful_assignments": 4,
        "physical_assignments": 6,
        "padding_assignments": 2,
        "layers": [
            {
                "layer_id": 0,
                "capture_backend": "vllm_router",
                "useful_route_sha256": (
                    "a1b3f5c4a2c43bb0546498837867ecbe02dce16caa8e7b7c0ccd3850f7f73df0"
                ),
                "physical_route_sha256": (
                    "03815bae958156ebf478b19fa21d8715629510990286d68ede0805a04ceafbd5"
                ),
                "useful_local_assignments_per_expert": [1, 1],
                "physical_local_assignments_per_expert": [1, 2],
                "padding_local_assignments_per_expert": [0, 1],
            },
            {
                "layer_id": 1,
                "capture_backend": "flashinfer_trtllm",
                "useful_route_sha256": (
                    "4739aba795ad92973ca4152371f6d721bd37a1b2c3f9c08c8e77bafee6111516"
                ),
                "physical_route_sha256": (
                    "725a3a7d64f1a9c722a09a45e533ca2b4de22ab0b3cb62599212f7791237af6f"
                ),
                "useful_local_assignments_per_expert": [1, 1],
                "physical_local_assignments_per_expert": [1, 2],
                "padding_local_assignments_per_expert": [0, 1],
            },
        ],
    }


@pytest.mark.parametrize(
    ("routing_data", "match"),
    (
        (torch.tensor([[0, -1], [1, 2]], dtype=torch.int32), "invalid expert ID"),
        (torch.tensor([[0, 0], [1, 2]], dtype=torch.int32), "duplicate expert IDs"),
        (torch.tensor([[0, 4], [1, 2]], dtype=torch.int32), "invalid expert ID"),
    ),
)
def test_summarize_expert_routing_fails_closed(routing_data, match):
    stats = ExpertRoutingStatsTensors(
        modular_routing_data=routing_data[:, None, :],
        flashinfer_routing_data=None,
        num_reqs=1,
        num_scheduled_tokens=1,
        num_physical_tokens=2,
    )

    with pytest.raises(ValueError, match=match):
        summarize_expert_routing(
            ExpertRoutingStep(decode_step=0, stats=stats),
            (ExpertLayerPlacement(0, 4, (0, 1), "linear"),),
        )


def test_expert_routing_stats_recorder_writes_complete_window(tmp_path):
    recorder = ExpertRoutingStatsRecorder(
        global_rank=3,
        layer_placements=(ExpertLayerPlacement(0, 4, (0, 1), "linear"),),
        top_k=2,
    )

    path = recorder.begin(str(tmp_path), iteration=5)
    recorder.submit(
        stats=ExpertRoutingStatsTensors(
            modular_routing_data=torch.tensor([[[0, 1]], [[1, 3]]], dtype=torch.int32),
            flashinfer_routing_data=None,
            num_reqs=2,
            num_scheduled_tokens=2,
            num_physical_tokens=2,
        ),
    )
    result = recorder.finish()

    assert result == {"status": "complete", "decode_steps": 1, "path": path}
    with open(path, encoding="utf-8") as stats_file:
        records = [json.loads(line) for line in stats_file]
    assert [record["record_type"] for record in records] == [
        "metadata",
        "decode_step",
        "summary",
    ]
    assert records[0]["global_rank"] == 3
    assert records[0]["schema_version"] == 2
    assert records[0]["capture_backend"] == "vllm_router"
    assert records[0]["layers"][0]["capture_backend"] == "vllm_router"
    assert records[1]["layers"][0]["physical_local_assignments_per_expert"] == [1, 2]
    assert records[2] == {
        "record_type": "summary",
        "status": "complete",
        "decode_steps": 1,
    }


def test_expert_routing_stats_recorder_rejects_mixed_capture_backends():
    placements = (
        ExpertLayerPlacement(0, 4, (0, 1), "linear"),
        ExpertLayerPlacement(
            1,
            4,
            (0, 1),
            "linear",
            ExpertRoutingCaptureBackend.FLASHINFER_TRTLLM,
        ),
    )

    with pytest.raises(ValueError, match="one capture backend"):
        ExpertRoutingStatsRecorder(
            global_rank=0,
            layer_placements=placements,
            top_k=2,
        )


def test_route_hashes_detect_divergence_hidden_by_local_histograms():
    placement = ExpertLayerPlacement(0, 4, (0, 1), "linear")

    def summarize(routing_data):
        stats = ExpertRoutingStatsTensors(
            modular_routing_data=routing_data[:, None, :],
            flashinfer_routing_data=None,
            num_reqs=2,
            num_scheduled_tokens=2,
            num_physical_tokens=2,
        )
        return summarize_expert_routing(
            ExpertRoutingStep(decode_step=0, stats=stats),
            (placement,),
        )["layers"][0]

    first = summarize(torch.tensor([[0, 2], [1, 3]], dtype=torch.int16))
    second = summarize(torch.tensor([[0, 3], [1, 2]], dtype=torch.int64))

    assert first["physical_local_assignments_per_expert"] == [1, 1]
    assert second["physical_local_assignments_per_expert"] == [1, 1]
    assert first["useful_route_sha256"] == (
        "fe75a7d0fef37822777df982bd8ef5e299d29e1c0c4a996c51b29be014d968e7"
    )
    assert first["physical_route_sha256"] == (
        "017a0f41174bbf1a3c8b085bfb87d2f1ab3aa4cf6b17e388baa230ba2c1e88b8"
    )
    assert second["useful_route_sha256"] == (
        "f5c2fd25de4c481464ac7017c4a7c76df579fc223530ef71cbb1658cbacd3357"
    )
    assert second["physical_route_sha256"] == (
        "7fe332f94aa80e1a5201073d6b5aabc8c973a49c70b973f7e7da4ca0494dea6f"
    )


def test_expert_routing_stats_recorder_waits_for_async_copy(tmp_path):
    recorder = ExpertRoutingStatsRecorder(
        global_rank=1,
        layer_placements=(ExpertLayerPlacement(0, 4, (0, 1), "linear"),),
        top_k=2,
    )
    routing_data = torch.full((1, 1, 2), -1, dtype=torch.int32)
    ready_event = SimpleNamespace(
        synchronize=lambda: routing_data.copy_(torch.tensor([[[1, 2]]]))
    )

    path = recorder.begin(str(tmp_path), iteration=2)
    recorder.submit(
        stats=ExpertRoutingStatsTensors(
            modular_routing_data=routing_data,
            flashinfer_routing_data=None,
            num_reqs=1,
            num_scheduled_tokens=1,
            num_physical_tokens=1,
        ),
        ready_event=ready_event,
        source_tensors=(torch.empty(0),),
    )
    recorder.finish()

    with open(path, encoding="utf-8") as stats_file:
        records = [json.loads(line) for line in stats_file]
    assert records[1]["layers"][0]["physical_local_assignments_per_expert"] == [0, 1]


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_buffer_is_updated_by_compiled_execution():
    class InputRouter(DummyRouter):
        def _compute_routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            return torch.ones_like(router_logits), router_logits.to(torch.int64)

        def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
            return topk_ids

    router = InputRouter(top_k=2, global_num_experts=16)
    buffer = torch.full((4, 2, 2), -1, dtype=torch.int32)
    router.set_capture_buffer(buffer, layer_id=1)
    compiled_select = torch.compile(
        router.select_experts, backend="eager", fullgraph=True
    )

    _, topk_ids = compiled_select(
        hidden_states=torch.empty((2, 1)),
        router_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert torch.equal(buffer[:2, 1], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[1, 2], [3, 4]]))

    compiled_select(
        hidden_states=torch.empty((2, 1)),
        router_logits=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    assert torch.equal(buffer[:2, 1], torch.tensor([[5, 6], [7, 8]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_flashinfer_stats_buffer_is_layer_major_int16_and_sentinel_filled():
    capturer = _capturer_with_buffer(max_tokens=8, num_layers=4)

    layer_buffer = capturer.bind_flashinfer_stats_buffer(layer_id=2)

    assert capturer.flashinfer_device_buffer.shape == (4, 8, 2)
    assert layer_buffer.shape == (8, 2)
    assert layer_buffer.dtype == torch.int16
    assert layer_buffer.is_contiguous()
    assert torch.all(layer_buffer == -1)

    layer_buffer[0] = torch.tensor([1, 2], dtype=torch.int16)
    capturer.clear_stats_buffers()
    assert torch.all(layer_buffer == -1)


def test_stats_snapshot_preserves_m_and_slices_b_for_both_backends():
    capturer = _capturer_with_buffer(max_tokens=8, num_layers=4)
    capturer.bind_modular_stats_buffer(layer_id=0)
    capturer.bind_flashinfer_stats_buffer(layer_id=2)

    stats = capturer.get_stats_tensors(
        num_reqs=3,
        num_scheduled_tokens=3,
        num_physical_tokens=4,
    )

    assert stats.num_scheduled_tokens == 3
    assert stats.num_physical_tokens == 4
    assert stats.modular_routing_data is not None
    assert stats.flashinfer_routing_data is not None
    assert stats.modular_routing_data.shape == (4, 4, 2)
    assert stats.flashinfer_routing_data.shape == (4, 4, 2)


def test_trtllm_monolithic_forwards_exact_routing_replay_buffer(monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import trtllm_bf16_moe

    calls = []
    fake_flashinfer = SimpleNamespace(
        fused_moe=SimpleNamespace(
            trtllm_bf16_moe=lambda **kwargs: calls.append(kwargs) or torch.empty(0)
        )
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setattr(trtllm_bf16_moe, "fi_moe_largest_bucket", lambda _: 256)
    monkeypatch.setattr(trtllm_bf16_moe, "activation_to_flashinfer_int", lambda _: 3)

    experts = trtllm_bf16_moe.TrtLlmBf16ExpertsMonolithic.__new__(
        trtllm_bf16_moe.TrtLlmBf16ExpertsMonolithic
    )
    experts.topk = 2
    experts.intermediate_size_per_partition = 16
    experts.ep_rank = 0
    experts.local_num_experts = 4
    experts.routing_method_type = RoutingMethodType.Default
    experts.moe_config = object()
    experts._routing_replay_out = None
    replay_buffer = torch.full((8, 2), -1, dtype=torch.int16)
    experts.set_routing_replay_out(replay_buffer)

    experts.apply(
        hidden_states=torch.empty((3, 4)),
        w1=torch.empty(0),
        w2=torch.empty(0),
        router_logits=torch.empty((3, 8)),
        activation=MoEActivation.SILU,
        global_num_experts=8,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 1
    assert calls[0]["routing_replay_out"] is replay_buffer


def test_trtllm_monolithic_rejects_noncontiguous_routing_replay_buffer():
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        TrtLlmBf16ExpertsMonolithic,
    )

    experts = TrtLlmBf16ExpertsMonolithic.__new__(TrtLlmBf16ExpertsMonolithic)
    experts.topk = 2
    buffer = torch.empty((2, 8), dtype=torch.int16).transpose(0, 1)

    with pytest.raises(ValueError, match="must be contiguous"):
        experts.set_routing_replay_out(buffer)


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self.is_monolithic = False

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    static_marks = []
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_tensor_static",
        lambda tensor: static_marks.append(("cudagraph", tensor)),
        raising=False,
    )
    monkeypatch.setattr(
        torch._dynamo,
        "mark_static_address",
        lambda tensor: static_marks.append(("address", tensor)),
    )

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module]),
    )

    capturer = _capturer_with_buffer(max_tokens=8, num_layers=12)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is None
    layer_buffer = dummy_module.router._routing_replay_out
    assert layer_buffer is not None
    assert layer_buffer.data_ptr() == (capturer.device_buffer[:, 7, :].data_ptr())
    assert static_marks == [
        ("cudagraph", capturer.device_buffer),
        ("address", capturer.device_buffer),
        ("cudagraph", layer_buffer),
        ("address", layer_buffer),
    ]


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self.is_monolithic = False

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_tensor_static",
        lambda tensor: None,
        raising=False,
    )
    monkeypatch.setattr(torch._dynamo, "mark_static_address", lambda tensor: None)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module]),
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = _capturer_with_buffer(max_tokens=8, num_layers=12)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # DP1 binds the compiler-visible device buffer instead of a Python hook.
    assert dummy_module.router.capture_fn is None
    assert dummy_module.router._routing_replay_out.data_ptr() == (
        capturer.device_buffer[:, 11, :].data_ptr()
    )


def test_gpu_model_runner_keeps_callback_for_data_parallel_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self.is_monolithic = False

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    module = DummyFusedMoE()
    capturer = types.SimpleNamespace(capture=lambda *_: None)
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module]),
    )

    gmr.GPUModelRunner._bind_routed_experts_capturer(runner, capturer)

    assert callable(module.router.capture_fn)
    assert module.router._routing_replay_out is None


def test_gpu_model_runner_does_not_bind_draft_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self.is_monolithic = False

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_tensor_static",
        lambda tensor: None,
        raising=False,
    )
    monkeypatch.setattr(torch._dynamo, "mark_static_address", lambda tensor: None)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [target_module]),
        compilation_config=types.SimpleNamespace(
            static_forward_context={
                "model.layers.7.mlp.experts": target_module,
                "mtp.layers.0.mlp.experts": draft_module,
            }
        ),
    )

    capturer = _capturer_with_buffer(max_tokens=8, num_layers=12)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert target_module.router._routing_replay_out.data_ptr() == (
        capturer.device_buffer[:, 7, :].data_ptr()
    )
    assert draft_module.router._routing_replay_out is None


def test_gpu_model_runner_binds_flashinfer_monolithic_stats_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 1
            self.router = _make_router()
            self.routed_experts = _make_monolithic_routed_experts()
            self.is_monolithic = True

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_tensor_static",
        lambda tensor: None,
        raising=False,
    )
    monkeypatch.setattr(torch._dynamo, "mark_static_address", lambda tensor: None)
    module = DummyFusedMoE()
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module]),
    )
    capturer = _capturer_with_buffer(max_tokens=8, num_layers=12)

    placements = gmr.GPUModelRunner._bind_routed_experts_capturer(runner, capturer)

    replay_buffer = (
        module.routed_experts.quant_method.moe_kernel.fused_experts._routing_replay_out
    )
    assert replay_buffer is not None
    assert replay_buffer.shape == (8, 2)
    assert replay_buffer.dtype == torch.int16
    assert replay_buffer.is_contiguous()
    assert replay_buffer.data_ptr() == capturer.flashinfer_device_buffer[1].data_ptr()
    assert module.router._routing_replay_out is None
    assert placements[0].capture_backend == (
        ExpertRoutingCaptureBackend.FLASHINFER_TRTLLM
    )


def test_gpu_model_runner_rejects_other_monolithic_stats_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 1
            self.router = _make_router()
            self.routed_experts = types.SimpleNamespace(
                global_num_experts=16,
                moe_config=types.SimpleNamespace(is_sequence_parallel=False),
                quant_method=types.SimpleNamespace(
                    moe_kernel=types.SimpleNamespace(fused_experts=object())
                ),
                expert_map_manager=types.SimpleNamespace(
                    get_local_expert_ids=lambda: list(range(16)),
                    placement_strategy="linear",
                ),
            )
            self.is_monolithic = True

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_tensor_static",
        lambda tensor: None,
        raising=False,
    )
    monkeypatch.setattr(torch._dynamo, "mark_static_address", lambda tensor: None)
    module = DummyFusedMoE()
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module]),
    )
    capturer = _capturer_with_buffer(max_tokens=8, num_layers=12)

    with pytest.raises(ValueError, match="only for FlashInfer TRT-LLM BF16"):
        gmr.GPUModelRunner._bind_routed_experts_capturer(runner, capturer)


def test_gpu_model_runner_rejects_data_parallel_stats_capture(monkeypatch):
    from vllm.config.compilation import CompilationMode
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    runner = types.SimpleNamespace(
        routed_experts_initialized=False,
        model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        parallel_config=types.SimpleNamespace(
            enable_eplb=False,
            use_ubatching=False,
            data_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
        ),
        compilation_config=types.SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    with pytest.raises(ValueError, match="requires DP1"):
        gmr.GPUModelRunner.init_routed_experts_capturer(runner)


def test_gpu_model_runner_rejects_microbatched_stats_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    runner = types.SimpleNamespace(
        routed_experts_initialized=False,
        model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        parallel_config=types.SimpleNamespace(
            enable_eplb=False,
            use_ubatching=True,
        ),
    )

    with pytest.raises(ValueError, match="does not support microbatching"):
        gmr.GPUModelRunner.init_routed_experts_capturer(runner)


def test_gpu_worker_binds_stats_after_weights_pool_before_first_profile(monkeypatch):
    from vllm.v1.worker import gpu_worker

    events = []

    @contextmanager
    def weights_pool():
        events.append("weights_pool_enter")
        yield
        events.append("weights_pool_exit")

    model_runner = types.SimpleNamespace(
        load_model=lambda *, load_dummy_weights: events.append("load_model"),
        init_routed_experts_capturer=lambda: events.append("bind_capture"),
    )
    worker = types.SimpleNamespace(
        model_runner=model_runner,
        vllm_config=types.SimpleNamespace(weight_transfer_config=None),
        _maybe_get_memory_pool_context=lambda *, tag: weights_pool(),
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(gpu_worker.envs, "VLLM_EXPERT_ROUTING_STATS", True)
    monkeypatch.setattr(
        gpu_worker, "set_current_vllm_config", lambda config: nullcontext()
    )

    gpu_worker.Worker.load_model(worker)

    assert events == [
        "weights_pool_enter",
        "load_model",
        "weights_pool_exit",
        "bind_capture",
    ]


@pytest.mark.parametrize(
    ("return_routed_experts", "capture_stats", "expected_clear"),
    (
        (True, False, "return"),
        (False, True, "stats"),
        (False, False, None),
    ),
)
def test_gpu_model_runner_only_clears_routing_needed_by_this_step(
    return_routed_experts,
    capture_stats,
    expected_clear,
):
    from vllm.v1.worker import gpu_model_runner as gmr

    class StopAfterClear(RuntimeError):
        pass

    class Runner:
        execute_model_state = None
        model_config = types.SimpleNamespace(
            enable_return_routed_experts=return_routed_experts
        )

        def __init__(self):
            self.clear_calls = []
            self.routed_experts_capturer = types.SimpleNamespace(
                clear_buffer=lambda: self.clear_calls.append("return"),
                clear_stats_buffers=lambda: self.clear_calls.append("stats"),
            )

        def _should_capture_expert_routing_stats(self, scheduler_output):
            return capture_stats

        @property
        def speculative_config(self):
            raise StopAfterClear

    runner = Runner()

    with pytest.raises(StopAfterClear):
        gmr.GPUModelRunner.execute_model(runner, types.SimpleNamespace())

    assert runner.clear_calls == ([] if expected_clear is None else [expected_clear])


def test_stats_d2h_does_not_require_return_routed_experts(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class CopyComplete(RuntimeError):
        pass

    def stop_after_copy(_):
        raise CopyComplete

    routing_data = torch.tensor(
        [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]],
        dtype=torch.int32,
    )
    capturer = _capturer_with_buffer(max_tokens=4, num_layers=1)
    capturer.bind_modular_stats_buffer(layer_id=0)
    capturer.device_buffer.copy_(routing_data)
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            num_reqs=2,
            generators={},
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
        ),
        discard_request_mask=types.SimpleNamespace(
            np=np.array([False, False]),
        ),
        use_async_scheduling=False,
        model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        routed_experts_capturer=capturer,
        _pending_expert_routing_stats=None,
        _should_capture_expert_routing_stats=lambda _: True,
        _to_list=stop_after_copy,
    )
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"a": 1, "b": 1},
    )
    sampler_output = types.SimpleNamespace(
        sampled_token_ids=torch.zeros((2, 1), dtype=torch.int32),
        logprobs_tensors=None,
    )
    monkeypatch.setattr(gmr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False)

    with pytest.raises(CopyComplete):
        gmr.GPUModelRunner._bookkeeping_sync(
            runner,
            scheduler_output,
            sampler_output,
            logits=None,
            hidden_states=torch.empty((2, 1)),
            num_scheduled_tokens=2,
            num_physical_tokens=4,
        )

    snapshot = runner._pending_expert_routing_stats
    assert snapshot is not None
    assert snapshot.num_reqs == 2
    assert snapshot.num_scheduled_tokens == 2
    assert snapshot.num_physical_tokens == 4
    assert torch.equal(snapshot.modular_routing_data, routing_data)


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1
