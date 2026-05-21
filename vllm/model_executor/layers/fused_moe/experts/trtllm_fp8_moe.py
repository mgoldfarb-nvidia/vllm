# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.trace_utils import (
    enum_metadata,
    next_moe_call_id,
    tensor_metadata,
    trace_enabled,
    trace_event,
)
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

logger = init_logger(__name__)


def _trace_trtllm_fp8_call(
    *,
    path: str,
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor | None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor | None,
    routing_logits: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    global_num_experts: int | None = None,
    n_group: int | None = None,
    topk_group: int | None = None,
    routed_scaling_factor: float | None = None,
    routing_method_type: RoutingMethodType | None = None,
    activation: MoEActivation | None = None,
    fp8_quant_type: object | None = None,
    weight_layout: object | None = None,
    use_shuffled_weight: bool | None = None,
) -> None:
    if not trace_enabled():
        return

    block_shape = quant_config.block_shape
    is_mxfp8 = block_shape == [1, 32]
    local_num_experts = moe_config.num_local_experts
    ep_rank = moe_config.moe_parallel_config.ep_rank
    local_expert_offset = ep_rank * local_num_experts
    payload = {
        "moe_call_id": next_moe_call_id(),
        "path": path,
        "hidden_states": tensor_metadata(hidden_states),
        "hidden_states_scale": tensor_metadata(hidden_states_scale),
        "routing_logits": tensor_metadata(routing_logits),
        "topk_ids": tensor_metadata(topk_ids),
        "topk_weights": tensor_metadata(topk_weights),
        "output": tensor_metadata(output),
        "gemm1_weights": tensor_metadata(gemm1_weights),
        "gemm1_weights_scale": tensor_metadata(gemm1_weights_scale),
        "gemm2_weights": tensor_metadata(gemm2_weights),
        "gemm2_weights_scale": tensor_metadata(gemm2_weights_scale),
        "num_tokens": int(hidden_states.shape[0]),
        "hidden_size": int(hidden_states.shape[-1]),
        "intermediate_size": moe_config.intermediate_size_per_partition,
        "num_experts": global_num_experts or moe_config.num_experts,
        "num_local_experts": local_num_experts,
        "local_expert_offset": local_expert_offset,
        "top_k": moe_config.experts_per_token,
        "n_group": n_group,
        "topk_group": topk_group,
        "routed_scaling_factor": routed_scaling_factor,
        "routing_method_type": enum_metadata(routing_method_type),
        "activation": enum_metadata(activation),
        "fp8_quantization_type": enum_metadata(fp8_quant_type),
        "weight_layout": enum_metadata(weight_layout),
        "use_shuffled_weight": use_shuffled_weight,
        "quant_block_shape": block_shape,
        "is_mxfp8": is_mxfp8,
        "moe_tp_size": moe_config.moe_parallel_config.tp_size,
        "moe_tp_rank": moe_config.moe_parallel_config.tp_rank,
        "moe_ep_size": moe_config.moe_parallel_config.ep_size,
        "moe_ep_rank": ep_rank,
    }
    dedupe_key = {
        "path": path,
        "num_tokens": int(hidden_states.shape[0]),
        "hidden_size": int(hidden_states.shape[-1]),
        "intermediate_size": moe_config.intermediate_size_per_partition,
        "num_local_experts": local_num_experts,
        "top_k": moe_config.experts_per_token,
        "block_shape": block_shape,
    }
    trace_event(
        "vllm.flashinfer.trtllm_fp8_moe.call",
        payload,
        dedupe_key=dedupe_key,
    )


class TrtLlmFp8ExpertsBase:
    """
    Fp8 TRTLLM-Gen MoE kernels. Shared base for modular and monolithic
    interfaces.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        self.moe_config = moe_config
        self.quant_config = quant_config

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Does not support non-gated MoE (i.e. Nanotron-3-Nano)."""
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports only SiLU and RELU^2 non-gated activation."""
        return activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """Monolithic kernel so only use with naive DP/EP and TP."""
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False


class TrtLlmFp8ExpertsModular(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsModular):
    """
    Fp8 TRTLLM-Gen MoE kernels. Supports modular interface.
    """

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports Fp8 block and MXFP8."""
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """Override to handle 4D BlockMajorK weights (E, K/bk, Mn, bk)."""
        if w1.dim() == 4:
            # BlockMajorK: (E, K/bk, Mn, bk)
            E = w1.shape[0]
            N = w1.shape[2]
            K = a1.size(-1)
            M = a1.size(0) if a1.dim() == 2 else a1.size(1)
            topk = topk_ids.size(1)
            return E, M, N, K, topk
        return super().moe_problem_size(a1, w1, w2, topk_ids)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The workspaces for this implementation are managed by flashinfer.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)

        return (workspace1, workspace2, output)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        import flashinfer
        from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

        # Pack topk ids and weights into format expected by the kernel.
        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        assert a1q_scale is not None

        is_mxfp8 = self.quant_config.block_shape == [1, 32]
        if is_mxfp8:
            fp8_quant_type = Fp8QuantizationType.MxFp8
            use_shuffled_weight = True
            weight_layout = WeightLayout.MajorK
            hidden_states_scale = a1q_scale
        else:
            fp8_quant_type = Fp8QuantizationType.DeepSeekFp8
            use_shuffled_weight = True
            weight_layout = WeightLayout.BlockMajorK
            hidden_states_scale = a1q_scale.t().contiguous()

        _trace_trtllm_fp8_call(
            path="routed_precomputed",
            moe_config=self.moe_config,
            quant_config=self.quant_config,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            topk_ids=packed_topk_ids,
            topk_weights=topk_weights,
            output=output,
            global_num_experts=global_num_experts,
            routed_scaling_factor=None,
            routing_method_type=self.routing_method_type,
            activation=activation,
            fp8_quant_type=fp8_quant_type,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
        )

        flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # not used
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            fp8_quantization_type=fp8_quant_type,
            output=output,
        )


class TrtLlmFp8ExpertsMonolithic(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsMonolithic):
    """
    Fp8 TRTLLM-Gen MoE kernels. Supports monolithic interface.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        # Make additional scales for per-tensor interface.
        if self.quant_config.is_per_tensor:
            w1_scale = self.quant_config.w1_scale
            assert w1_scale is not None
            a1_scale = self.quant_config.a1_scale
            assert a1_scale is not None
            w2_scale = self.quant_config.w2_scale
            assert w2_scale is not None
            a2_scale = self.quant_config.a2_scale
            assert a2_scale is not None

            self._g1_alphas = (w1_scale * a1_scale).squeeze()
            self._g2_alphas = (w2_scale * a2_scale).squeeze()
            self._g1_scale_c = (
                self._g1_alphas / self.quant_config.a2_scale
                if moe_config.is_act_and_mul
                else torch.ones_like(self._g1_alphas) / self.quant_config.a2_scale
            )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports Fp8 per-tensor, Fp8 block, and MXFP8."""
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Monolithic kernels need to express router support."""
        # NOTE(dbari): TopK routing could also be enabled, but need to validate models
        # NOTE(dbari): Default is not implemented and should not be enabled until it is

        if (weight_key, activation_key) in [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]:
            # NOTE(rob): potentially allow others here. This is a conservative list.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
                RoutingMethodType.SigmoidRenorm,
                RoutingMethodType.MiniMax2,
                RoutingMethodType.Simulated,
            ]
        elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
            # NOTE(dbari): as above, potentially allow others here.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Llama4,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
                RoutingMethodType.SigmoidRenorm,
                RoutingMethodType.MiniMax2,
                RoutingMethodType.Simulated,
            ]
        else:
            raise ValueError("Unsupported quantization scheme.")

    def _apply_block_scale(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        import flashinfer
        from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        assert self.topk <= global_num_experts
        assert self.topk <= 10
        assert global_num_experts % 4 == 0
        assert self.quant_config.block_shape in [[128, 128], [1, 32]]
        # Kernel expects #experts <= #threads 512
        assert global_num_experts <= 512
        # TODO: fuse into the quant kernel.
        assert a1q_scale is not None

        is_mxfp8 = self.quant_config.block_shape == [1, 32]
        if is_mxfp8:
            fp8_quant_type = Fp8QuantizationType.MxFp8
            use_shuffled_weight = True
            weight_layout = WeightLayout.MajorK
            hidden_states_scale = a1q_scale
        else:
            fp8_quant_type = Fp8QuantizationType.DeepSeekFp8
            use_shuffled_weight = True
            weight_layout = WeightLayout.BlockMajorK
            hidden_states_scale = a1q_scale.t().contiguous()

        _trace_trtllm_fp8_call(
            path="block_scale_routing_logits",
            moe_config=self.moe_config,
            quant_config=self.quant_config,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            routing_logits=router_logits,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            global_num_experts=global_num_experts,
            n_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            activation=activation,
            fp8_quant_type=fp8_quant_type,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
        )

        return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=(num_expert_group or 0),
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            fp8_quantization_type=fp8_quant_type,
        )

    def _apply_per_tensor(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        # Delay import for non-CUDA.
        import flashinfer

        # Confirm supported activation function.
        assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

        activation_type = activation_to_flashinfer_int(activation)

        # Confirm Llama-4 routing is proper.
        if self.routing_method_type == RoutingMethodType.Llama4:
            assert apply_router_weight_on_input
        else:
            assert not apply_router_weight_on_input

        # Currently FI requires bfloat16 routing bias.
        # https://github.com/flashinfer-ai/flashinfer/issues/2909
        if e_score_correction_bias is not None:
            e_score_correction_bias = e_score_correction_bias.to(torch.bfloat16)

        _trace_trtllm_fp8_call(
            path="per_tensor_routing_logits",
            moe_config=self.moe_config,
            quant_config=self.quant_config,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            routing_logits=router_logits,
            gemm1_weights=w1,
            gemm1_weights_scale=None,
            gemm2_weights=w2,
            gemm2_weights_scale=None,
            global_num_experts=global_num_experts,
            n_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            activation=activation,
            fp8_quant_type="per_tensor",
            weight_layout=None,
            use_shuffled_weight=None,
        )

        out = flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            output1_scales_scalar=self._g1_scale_c,
            output1_scales_gate_scalar=self._g1_alphas,
            gemm2_weights=w2,
            output2_scales_scalar=self._g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group or 0,
            topk_group=topk_group or 0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=apply_router_weight_on_input,
            routing_method_type=self.routing_method_type,
            activation_type=activation_type,
        )
        return out

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        if self.quant_config.block_shape is not None:
            return self._apply_block_scale(
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                topk_group=topk_group,
            )
        elif self.quant_config.is_per_tensor:
            return self._apply_per_tensor(
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            raise NotImplementedError(
                "Only per-block, per-tensor, and MXFP8 quantization are "
                f"supported in {self.__class__.__name__}."
            )
