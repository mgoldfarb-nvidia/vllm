# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import hashlib
import json
import logging
import queue
import struct
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)

_EXPERT_ROUTING_STATS_SCHEMA_VERSION = 2
_ROUTING_SENTINEL = -1
_ROUTING_HASH_DOMAIN = b"vllm.expert-routing.v2\0"


class ExpertRoutingCaptureBackend(str, Enum):
    """Source of the exact top-k expert IDs for a layer."""

    VLLM_ROUTER = "vllm_router"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"


@dataclass(frozen=True)
class ExpertLayerPlacement:
    """Static logical-to-local expert placement for one MoE layer."""

    layer_id: int
    global_num_experts: int
    local_expert_ids: tuple[int, ...]
    placement_strategy: str
    capture_backend: ExpertRoutingCaptureBackend = (
        ExpertRoutingCaptureBackend.VLLM_ROUTER
    )


@dataclass(frozen=True)
class ExpertRoutingStep:
    """One pure-decode routing snapshot waiting for CPU aggregation."""

    decode_step: int
    stats: ExpertRoutingStatsTensors
    ready_event: torch.cuda.Event | None = None
    # Retain CUDA allocations until their asynchronous D2H copies complete.
    source_tensors: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class ExpertRoutingStatsTensors:
    """Routing snapshots and scheduler metadata for one decode step."""

    modular_routing_data: torch.Tensor | None
    flashinfer_routing_data: torch.Tensor | None
    num_reqs: int
    num_scheduled_tokens: int
    num_physical_tokens: int

    def __post_init__(self) -> None:
        if self.modular_routing_data is None and self.flashinfer_routing_data is None:
            raise ValueError("Expert routing statistics require a capture buffer")
        if not 0 <= self.num_scheduled_tokens <= self.num_physical_tokens:
            raise ValueError(
                "Expert routing token counts must satisfy "
                "0 <= num_scheduled_tokens <= num_physical_tokens"
            )

    @property
    def source_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for tensor in (
                self.modular_routing_data,
                self.flashinfer_routing_data,
            )
            if tensor is not None
        )

    def clone(self) -> ExpertRoutingStatsTensors:
        return ExpertRoutingStatsTensors(
            modular_routing_data=(
                self.modular_routing_data.clone()
                if self.modular_routing_data is not None
                else None
            ),
            flashinfer_routing_data=(
                self.flashinfer_routing_data.clone()
                if self.flashinfer_routing_data is not None
                else None
            ),
            num_reqs=self.num_reqs,
            num_scheduled_tokens=self.num_scheduled_tokens,
            num_physical_tokens=self.num_physical_tokens,
        )

    def to_cpu_nonblocking(
        self, *, pin_memory: bool = False
    ) -> ExpertRoutingStatsTensors:
        def _copy(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or tensor.device.type == "cpu":
                return tensor
            if not pin_memory:
                return tensor.to("cpu", non_blocking=True)
            output = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=True,
                memory_format=torch.contiguous_format,
            )
            output.copy_(tensor, non_blocking=True)
            return output

        return ExpertRoutingStatsTensors(
            modular_routing_data=_copy(self.modular_routing_data),
            flashinfer_routing_data=_copy(self.flashinfer_routing_data),
            num_reqs=self.num_reqs,
            num_scheduled_tokens=self.num_scheduled_tokens,
            num_physical_tokens=self.num_physical_tokens,
        )


def _routing_data_for_layer(
    stats: ExpertRoutingStatsTensors,
    placement: ExpertLayerPlacement,
) -> np.ndarray:
    if placement.capture_backend == ExpertRoutingCaptureBackend.VLLM_ROUTER:
        if stats.modular_routing_data is None:
            raise ValueError(
                f"MoE layer {placement.layer_id} has no vLLM router capture"
            )
        return stats.modular_routing_data.numpy()[:, placement.layer_id, :]
    if placement.capture_backend == ExpertRoutingCaptureBackend.FLASHINFER_TRTLLM:
        if stats.flashinfer_routing_data is None:
            raise ValueError(
                f"MoE layer {placement.layer_id} has no FlashInfer capture"
            )
        return stats.flashinfer_routing_data.numpy()[placement.layer_id, :, :]
    raise ValueError(
        f"MoE layer {placement.layer_id} uses unknown capture backend "
        f"{placement.capture_backend!r}"
    )


def _validate_routing_data(
    routing_data: np.ndarray,
    placement: ExpertLayerPlacement,
    num_physical_tokens: int,
) -> None:
    if routing_data.ndim != 2 or routing_data.shape[0] != num_physical_tokens:
        raise ValueError(
            f"MoE layer {placement.layer_id} routing shape {routing_data.shape} does "
            f"not match physical token count {num_physical_tokens}"
        )
    invalid = (routing_data < 0) | (routing_data >= placement.global_num_experts)
    if invalid.any():
        first_row, first_col = np.argwhere(invalid)[0]
        raise ValueError(
            f"MoE layer {placement.layer_id} has invalid expert ID "
            f"{routing_data[first_row, first_col]} at physical token {first_row}, "
            f"top-k slot {first_col}"
        )
    if routing_data.shape[1] > 1:
        sorted_ids = np.sort(routing_data, axis=1)
        duplicate_rows = np.flatnonzero((np.diff(sorted_ids, axis=1) == 0).any(axis=1))
        if duplicate_rows.size:
            raise ValueError(
                f"MoE layer {placement.layer_id} has duplicate expert IDs at "
                f"physical token {duplicate_rows[0]}"
            )


def _local_expert_counts(
    routing_data: np.ndarray,
    placement: ExpertLayerPlacement,
) -> np.ndarray:
    global_counts = np.bincount(
        routing_data.reshape(-1).astype(np.int64),
        minlength=placement.global_num_experts,
    )
    return global_counts[np.asarray(placement.local_expert_ids, dtype=np.int64)]


def _routing_sha256(
    routing_data: np.ndarray,
    *,
    scope: str,
    layer_id: int,
    num_scheduled_tokens: int,
    num_physical_tokens: int,
) -> str:
    """Hash a canonical expert-ID matrix for cross-rank comparison.

    The input is normalized to row-major little-endian int32. The hash binds the
    domain, scope, layer, useful and physical token counts, and matrix shape.
    """
    normalized = np.ascontiguousarray(routing_data, dtype="<i4")
    rows, top_k = normalized.shape
    header = struct.pack(
        "<5q",
        layer_id,
        num_scheduled_tokens,
        num_physical_tokens,
        rows,
        top_k,
    )
    digest = hashlib.sha256()
    digest.update(_ROUTING_HASH_DOMAIN)
    digest.update(scope.encode("ascii"))
    digest.update(b"\0")
    digest.update(header)
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def summarize_expert_routing(
    step: ExpertRoutingStep,
    layer_placements: tuple[ExpertLayerPlacement, ...],
) -> dict[str, Any]:
    """Summarize exact local expert occupancy for one decode step."""
    if step.ready_event is not None:
        step.ready_event.synchronize()
    stats = step.stats
    layers = []
    top_k = None
    for placement in layer_placements:
        routing_data = _routing_data_for_layer(stats, placement)
        _validate_routing_data(routing_data, placement, stats.num_physical_tokens)
        if top_k is None:
            top_k = routing_data.shape[1]
        elif top_k != routing_data.shape[1]:
            raise ValueError("MoE layers have inconsistent top-k dimensions")
        useful_counts = _local_expert_counts(
            routing_data[: stats.num_scheduled_tokens], placement
        )
        physical_counts = _local_expert_counts(routing_data, placement)
        padding_counts = physical_counts - useful_counts
        layers.append(
            {
                "layer_id": placement.layer_id,
                "capture_backend": placement.capture_backend.value,
                "useful_route_sha256": _routing_sha256(
                    routing_data[: stats.num_scheduled_tokens],
                    scope="useful",
                    layer_id=placement.layer_id,
                    num_scheduled_tokens=stats.num_scheduled_tokens,
                    num_physical_tokens=stats.num_physical_tokens,
                ),
                "physical_route_sha256": _routing_sha256(
                    routing_data,
                    scope="physical",
                    layer_id=placement.layer_id,
                    num_scheduled_tokens=stats.num_scheduled_tokens,
                    num_physical_tokens=stats.num_physical_tokens,
                ),
                "useful_local_assignments_per_expert": useful_counts.tolist(),
                "physical_local_assignments_per_expert": physical_counts.tolist(),
                "padding_local_assignments_per_expert": padding_counts.tolist(),
            }
        )

    assert top_k is not None
    num_padding_tokens = stats.num_physical_tokens - stats.num_scheduled_tokens
    return {
        "record_type": "decode_step",
        "decode_step": step.decode_step,
        "num_reqs": stats.num_reqs,
        "num_scheduled_tokens": stats.num_scheduled_tokens,
        "num_physical_tokens": stats.num_physical_tokens,
        "num_padding_tokens": num_padding_tokens,
        "useful_assignments": stats.num_scheduled_tokens * top_k,
        "physical_assignments": stats.num_physical_tokens * top_k,
        "padding_assignments": num_padding_tokens * top_k,
        "layers": layers,
    }


class ExpertRoutingStatsRecorder:
    """Writes per-decode expert occupancy without blocking model execution."""

    def __init__(
        self,
        *,
        global_rank: int,
        layer_placements: tuple[ExpertLayerPlacement, ...],
        top_k: int,
    ) -> None:
        if not layer_placements:
            raise ValueError("Expert routing statistics require at least one MoE layer")
        if top_k <= 0:
            raise ValueError(
                f"Expert routing statistics require positive top_k, got {top_k}"
            )
        for placement in layer_placements:
            local_expert_ids = np.asarray(placement.local_expert_ids, dtype=np.int64)
            if (
                placement.global_num_experts <= 0
                or (local_expert_ids < 0).any()
                or (local_expert_ids >= placement.global_num_experts).any()
                or np.unique(local_expert_ids).size != local_expert_ids.size
            ):
                raise ValueError(
                    f"MoE layer {placement.layer_id} has invalid local expert placement"
                )
        capture_backends = {placement.capture_backend for placement in layer_placements}
        if len(capture_backends) != 1:
            raise ValueError(
                "Expert routing statistics require one capture backend across all "
                "MoE layers"
            )
        self._global_rank = global_rank
        self._layer_placements = layer_placements
        self._capture_backend = capture_backends.pop()
        self._top_k = top_k
        self._queue: queue.SimpleQueue[ExpertRoutingStep | str] | None = None
        self._thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._path: Path | None = None
        self._decode_steps = 0

    @property
    def active(self) -> bool:
        return self._queue is not None

    def begin(self, output_dir: str, iteration: int) -> str:
        if self.active:
            raise RuntimeError("Expert routing statistics window is already active")
        self._raise_writer_error()

        stats_dir = Path(output_dir) / "expert-routing-stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        path = stats_dir / (
            f"expert-routing-rank-{self._global_rank}-iteration-{iteration}.jsonl"
        )
        metadata = {
            "record_type": "metadata",
            "schema_version": _EXPERT_ROUTING_STATS_SCHEMA_VERSION,
            "iteration": iteration,
            "global_rank": self._global_rank,
            "top_k": self._top_k,
            "capture_backend": self._capture_backend.value,
            "layers": [asdict(layer) for layer in self._layer_placements],
        }
        with path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(metadata, separators=(",", ":")) + "\n")

        self._path = path
        self._decode_steps = 0
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._write_steps,
            name=f"expert-routing-stats-rank-{self._global_rank}",
            daemon=True,
        )
        self._thread.start()
        return str(path)

    def submit(
        self,
        stats: ExpertRoutingStatsTensors,
        *,
        ready_event: torch.cuda.Event | None = None,
        source_tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        if self._queue is None:
            return
        self._raise_writer_error()
        step = ExpertRoutingStep(
            decode_step=self._decode_steps,
            stats=stats,
            ready_event=ready_event,
            source_tensors=source_tensors,
        )
        self._decode_steps += 1
        self._queue.put(step)

    def finish(self) -> dict[str, Any]:
        return self._close("complete")

    def abort(self) -> dict[str, Any]:
        return self._close("aborted")

    def _close(self, status: str) -> dict[str, Any]:
        if self._queue is None or self._thread is None or self._path is None:
            return {"status": "inactive", "decode_steps": 0, "path": None}
        self._queue.put(status)
        self._thread.join()
        path = self._path
        decode_steps = self._decode_steps
        self._queue = None
        self._thread = None
        self._path = None
        self._raise_writer_error()
        return {
            "status": status,
            "decode_steps": decode_steps,
            "path": str(path),
        }

    def _write_steps(self) -> None:
        assert self._queue is not None
        assert self._path is not None
        try:
            with self._path.open("a", encoding="utf-8") as output:
                while True:
                    item = self._queue.get()
                    if isinstance(item, str):
                        output.write(
                            json.dumps(
                                {
                                    "record_type": "summary",
                                    "status": item,
                                    "decode_steps": self._decode_steps,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        return
                    output.write(
                        json.dumps(
                            summarize_expert_routing(item, self._layer_placements),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        except BaseException as error:
            self._writer_error = error

    def _raise_writer_error(self) -> None:
        if self._writer_error is None:
            return
        error = self._writer_error
        self._writer_error = None
        raise RuntimeError("Failed to write expert routing statistics") from error


def _get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config.

    Different model families store this under different attribute names
    (e.g. ``num_experts_per_tok`` for DeepSeek, ``top_k_experts`` for Gemma 4).
    """
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve ``num_experts`` across HuggingFace config naming conventions.

    Different MoE model families expose this under different keys:
      - ``num_experts``: Mixtral, Qwen2-MoE, Qwen3-MoE
      - ``n_routed_experts``: DeepSeek-V2/V3
      - ``num_local_experts``: Mixtral (older exports)
    """
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` from inside the forward pass
    with the per-layer ``topk_ids`` tensor. The tensor is sliced to the
    tokens owned by this DP rank and written into a preallocated device
    buffer. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`RoutedExpertsLists`.

    The device / pinned-CPU transit buffers use ``torch.int32`` (not a
    narrow ``uint8``/``uint16`` sized by ``num_experts``). This keeps the
    SP all-gather path free of dtype casts, matches the router's native
    ``topk_ids`` indices dtype more closely, and costs only a few MB per
    worker (``max_num_batched_tokens * num_layers * top_k * 4`` bytes).
    The scheduler-side slot buffer
    (``RoutedExpertsManager.routed_experts_by_slot``) still uses the
    narrow dtype -- numpy fancy-index assignment in ``store_batch``
    narrows the data on the way in.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - :meth:`clear_buffer` preserves zero-fill semantics for routed-expert
          return; statistics use :meth:`clear_stats_buffers` and ``-1`` sentinels.
        - ``device_buffer.dtype`` is ``torch.int32``.
        - FlashInfer TRT-LLM capture uses a separate layer-major ``int16``
          buffer so each per-layer view is contiguous as required by its API.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            # Use int32 for the device / host transit buffers: it
            # matches the router's native topk_ids dtype, is universally
            # supported by NCCL (uint8/uint16 are version-dependent),
            # and the extra bytes are small (few MB per worker). The
            # big scheduler-side slot buffer stays narrow.
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.flashinfer_device_buffer: torch.Tensor | None = None
        self._stats_capture_backends: dict[int, ExpertRoutingCaptureBackend] = {}
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def bind_modular_stats_buffer(self, layer_id: int) -> torch.Tensor:
        """Bind one modular layer to its compiler-visible capture view."""
        self._register_stats_backend(layer_id, ExpertRoutingCaptureBackend.VLLM_ROUTER)
        return self.device_buffer[:, layer_id, :]

    def bind_flashinfer_stats_buffer(self, layer_id: int) -> torch.Tensor:
        """Bind one monolithic layer to a contiguous FlashInfer replay view."""
        self._register_stats_backend(
            layer_id, ExpertRoutingCaptureBackend.FLASHINFER_TRTLLM
        )
        if self.flashinfer_device_buffer is None:
            max_tokens, num_layers, top_k = self.device_buffer.shape
            self.flashinfer_device_buffer = torch.full(
                (num_layers, max_tokens, top_k),
                _ROUTING_SENTINEL,
                dtype=torch.int16,
                device=self.device_buffer.device,
            )
        layer_buffer = self.flashinfer_device_buffer[layer_id]
        assert layer_buffer.is_contiguous()
        return layer_buffer

    def _register_stats_backend(
        self,
        layer_id: int,
        backend: ExpertRoutingCaptureBackend,
    ) -> None:
        if not 0 <= layer_id < self.device_buffer.shape[1]:
            raise ValueError(
                f"MoE layer {layer_id} exceeds routing capture capacity "
                f"{self.device_buffer.shape[1]}"
            )
        previous = self._stats_capture_backends.setdefault(layer_id, backend)
        if previous != backend:
            raise ValueError(
                f"MoE layer {layer_id} cannot use both {previous.value} and "
                f"{backend.value} routing capture"
            )

    def clear_stats_buffers(self) -> None:
        """Fill active statistics buffers with an invalid-ID sentinel."""
        active_backends = set(self._stats_capture_backends.values())
        if ExpertRoutingCaptureBackend.VLLM_ROUTER in active_backends:
            self.device_buffer.fill_(_ROUTING_SENTINEL)
        if self.flashinfer_device_buffer is not None:
            self.flashinfer_device_buffer.fill_(_ROUTING_SENTINEL)

    def get_stats_tensors(
        self,
        *,
        num_reqs: int,
        num_scheduled_tokens: int,
        num_physical_tokens: int,
    ) -> ExpertRoutingStatsTensors:
        """Return B-row views for all active routing capture backends."""
        if num_physical_tokens > self.device_buffer.shape[0]:
            raise ValueError(
                f"Physical token count {num_physical_tokens} exceeds routing "
                f"capture capacity {self.device_buffer.shape[0]}"
            )
        active_backends = set(self._stats_capture_backends.values())
        modular_data = (
            self.device_buffer[:num_physical_tokens]
            if ExpertRoutingCaptureBackend.VLLM_ROUTER in active_backends
            else None
        )
        flashinfer_data = (
            self.flashinfer_device_buffer[:, :num_physical_tokens, :]
            if self.flashinfer_device_buffer is not None
            else None
        )
        return ExpertRoutingStatsTensors(
            modular_routing_data=modular_data,
            flashinfer_routing_data=flashinfer_data,
            num_reqs=num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            num_physical_tokens=num_physical_tokens,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have three different batch
        layouts depending on where the DP combine happens and whether
        Sequence Parallelism (SP) is active for the MoE layer:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.
          - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
            path): tokens were split along dim=0 across the TP group by
            ``_sequence_parallel_context``
            (``moe_runner_base.py:_sequence_parallel_context``), so each
            TP rank only sees its shard. We all-gather along dim=0 to
            reconstruct this DP rank's full routing tensor. SP pads with
            ceil-div (see ``_compute_sp_num_tokens`` in
            ``forward_context.py``), so the gathered tensor may contain a
            few trailing padding rows which are trimmed by the downstream
            ``[:token_num_per_dp]`` slice.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated
                # before routing. This rank owns tokens
                # [end_loc - token_num_per_dp, end_loc).
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens, take the whole tensor.
                start_loc = 0
                end_loc = token_num_per_dp
            elif (
                self.tp_size > 1
                and n != token_num_per_dp
                and n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
            ):
                # SP + modular-kernel path. All-gather across the TP
                # group along dim=0 to reconstruct the full per-DP-rank
                # tensor; keep only the first ``token_num_per_dp`` rows
                # (trailing rows are SP ceil-div padding). The TP group
                # is always initialized on real rollout workers, and
                # every rank in the group reaches this branch in
                # lockstep (bind is per-FusedMoE layer, SP is a global
                # condition), so a bare all_gather here will not
                # deadlock -- let it raise if the precondition is
                # violated rather than skip silently.
                #
                # ``topk_ids`` is already whatever the router produced
                # (typically int32/int64, both supported by NCCL); the
                # downstream ``device_buffer[...] = topk_ids[...]``
                # setitem narrows into int32 automatically.
                topk_ids = get_tp_group().all_gather(topk_ids, dim=0)
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                sp_expected = (
                    (token_num_per_dp + self.tp_size - 1) // self.tp_size
                    if self.tp_size > 0
                    else -1
                )
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {n} (expected {total}, {token_num_per_dp}, "
                    f"or {sp_expected} for dp_rank={self.dp_rank}, "
                    f"tp_size={self.tp_size})"
                )

        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step so
        slots belonging to finished / preempted tokens don't leak into
        the next step.
        """
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass runs
        :meth:`clear_buffer`.
        """
        return self.device_buffer


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`RoutedExpertsLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(routing_data, slot_mapping)`` — a single CPU->CPU
         fancy-index assign, ~few MB per step.
      3. On request completion / abort / preemption, the scheduler
         calls :meth:`get` with the request's block IDs to recover
         the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block
    pool (``num_blocks * block_size`` slots). For large block pools
    this can reach multiple GB; see the init log for the exact size.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Pick the attention group for block/slot mapping. We require
        # a FullAttentionSpec group rather than any AttentionSpec to
        # stay consistent with the worker-side lookup in
        # ``GPUModelRunner._get_attention_kv_cache_gid``; hybrid models
        # (Mamba / linear attention) also have other AttentionSpec
        # groups whose slot layout differs.
        self.attn_gid = next(
            gid
            for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # Expert IDs are 0..num_experts-1; uint8 fits 256 distinct
        # values so the boundary is ``<= 256`` (NOT ``< 256``). Keeping
        # this narrow matters because the slot buffer is sized for the
        # whole block pool and can reach multiple GB.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=expert_id_dtype,
        )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. Called once
        per scheduler step in ``update_from_output``.
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed experts data for a completed / preempted request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Because numpy fancy indexing
        returns a **copy** (not a view), the returned ndarray is safe
        to hold across subsequent :meth:`store_batch` calls — do not
        replace the fancy index with a slice without re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.
            token_start: Skip the first ``token_start`` tokens from the
                result. The slot_mapping is sliced before the fancy-index
                read, so only the requested slots are fetched — no large
                intermediate array is allocated. Clamped to
                ``[0, num_tokens]`` automatically.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        # slot = block_id * block_size + offset_in_block; flatten the
        # (num_blocks, block_size) grid and trim to num_tokens, then
        # skip the first token_start entries so only the requested
        # range is fetched in a single fancy-index read.
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        slot_mapping = slot_mapping[token_start:]
        return self.routed_experts_by_slot[slot_mapping]
