# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import torch

_TRACE_ENV = "VLLM_FLASHINFER_MOE_TRACE"
_TRACE_FILE_ENV = "VLLM_FLASHINFER_MOE_TRACE_FILE"
_TRACE_MODE_ENV = "VLLM_FLASHINFER_MOE_TRACE_MODE"
_TRACE_FIRST_N_ENV = "VLLM_FLASHINFER_MOE_TRACE_FIRST_N"

_DEFAULT_TRACE_FILE = "/tmp/vllm_flashinfer_moe_trace.rank%r.local%l.pid%p.jsonl"

_LOCK = threading.Lock()
_SEEN_KEYS: set[str] = set()
_EVENT_COUNTS: dict[str, int] = {}
_CALL_COUNTER = itertools.count()


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    return _truthy(os.getenv(_TRACE_ENV))


def _rank() -> str:
    return os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))


def _local_rank() -> str:
    return os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0"))


def _trace_path() -> Path:
    path = os.getenv(_TRACE_FILE_ENV, _DEFAULT_TRACE_FILE)
    substitutions = {
        "%p": str(os.getpid()),
        "%r": _rank(),
        "%l": _local_rank(),
        "%h": socket.gethostname(),
    }
    for key, value in substitutions.items():
        path = path.replace(key, value)
    return Path(path)


def _json_default(value: Any) -> str:
    return str(value)


def _dedupe_enabled() -> bool:
    return os.getenv(_TRACE_MODE_ENV, "shape_once").lower() == "shape_once"


def _first_n_limit() -> int | None:
    mode = os.getenv(_TRACE_MODE_ENV, "shape_once").lower()
    if mode != "first_n":
        return None
    try:
        return max(0, int(os.getenv(_TRACE_FIRST_N_ENV, "10")))
    except ValueError:
        return 10


def trace_event(
    event: str,
    payload: dict[str, Any],
    *,
    dedupe_key: Any | None = None,
) -> None:
    if not trace_enabled():
        return

    if _dedupe_enabled() and dedupe_key is not None:
        key = json.dumps([event, dedupe_key], sort_keys=True, default=_json_default)
        with _LOCK:
            if key in _SEEN_KEYS:
                return
            _SEEN_KEYS.add(key)

    first_n = _first_n_limit()
    if first_n is not None:
        with _LOCK:
            count = _EVENT_COUNTS.get(event, 0)
            if count >= first_n:
                return
            _EVENT_COUNTS[event] = count + 1

    record = {
        "schema_version": 1,
        "source": "vllm",
        "event": event,
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "rank": _rank(),
        "local_rank": _local_rank(),
    }
    record.update(payload)

    path = _trace_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True, default=_json_default))
            out.write("\n")


def next_moe_call_id() -> str:
    return f"{_rank()}:{os.getpid()}:{next(_CALL_COUNTER)}"


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def enum_metadata(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"name": value.name, "value": value.value}
    return value
