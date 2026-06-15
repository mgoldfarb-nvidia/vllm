# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import torch

import vllm.envs as envs
from vllm.multimodal.audio import resample_audio_scipy
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import check_sparse_tensor_invariants_threadsafe

from .base import MediaIO

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


# Public libsndfile error codes exposed via `soundfile.LibsndfileError.code`,
# soundfile being the main audio loading backend. Used to validate if an audio
# loading error is due to a server error vs a client error (invalid audio file).
# 0 = sf_error(NULL) race condition: when multiple threads fail sf_open_virtual
#     concurrently, one thread may clear the global error before another reads it,
#     producing code=0 ("Garbled error message from libsndfile" in soundfile).
#     See: https://github.com/bastibe/python-soundfile/issues/479
# 1 = unrecognised format      (file is not a supported audio container)
# 3 = malformed file           (corrupt or structurally invalid audio)
# 4 = unsupported encoding     (codec not supported by this libsndfile build)
_BAD_SF_CODES = {0, 1, 3, 4}


def load_audio_soundfile(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio via soundfile"""
    with soundfile.SoundFile(path) as f:
        native_sr = f.samplerate
        if max_duration_s is not None:
            file_duration_s = f.frames / native_sr
            if file_duration_s > max_duration_s:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (file contains "
                    f"{file_duration_s:.1f}s at {native_sr}Hz). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                    f"increase this limit."
                )
        y = f.read(dtype="float32", always_2d=False).T

    # After `.T`, any 2D input is already (channels, time), so the channels
    # axis is always axis=0. The previous `tuple(range(y.ndim - 1))` form
    # was equivalent but harder to read.
    if mono and y.ndim > 1:
        y = np.mean(y, axis=0)

    if sr is not None and sr != native_sr:
        y = resample_audio_scipy(y, orig_sr=native_sr, target_sr=sr)
        # Round to match the rounding used inside `resample_audio_scipy`
        # itself, so half-integer rates don't silently truncate.
        return y, int(round(sr))
    return y, native_sr


def load_audio(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
):
    """Load audio from a file or buffer via soundfile (FFmpeg-free).

    Reliably supports WAV, FLAC, OGG/Vorbis, and other formats native to
    libsndfile. MP3 works on libsndfile >= 1.1.0 (soundfile >= 0.13) but
    not all packaged wheels are built with MP3 support. AAC/MP4/M4A and
    WebM/Opus container formats are not supported following the
    royalty-bearing codec removal; re-encode to FLAC, OGG, or WAV.
    """
    try:
        return load_audio_soundfile(
            path, sr=sr, mono=mono, max_duration_s=max_duration_s
        )
    except ImportError:
        raise  # Let PlaceholderModule's message ("install vllm[audio]") propagate.
    except soundfile.LibsndfileError as exc:
        if exc.code not in _BAD_SF_CODES:
            raise
        raise ValueError(
            "Invalid or unsupported audio format. "
            "Reliably supported by this build: WAV, FLAC, OGG/Vorbis "
            "(MP3 is libsndfile-build dependent — present in soundfile >= "
            "0.13 / libsndfile >= 1.1.0 but not all packaged wheels). "
            "AAC/MP4/M4A and WebM/Opus container formats were previously "
            "accepted via PyAV/FFmpeg but are no longer supported following "
            "the royalty-bearing codec removal — re-encode to WAV, FLAC, "
            "or OGG/Vorbis before submitting."
        ) from exc


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return load_audio(
            BytesIO(data),
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
        )

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return load_audio(
            filepath,
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
        )

    def encode_base64(
        self,
        media: tuple[npt.NDArray, int],
        *,
        audio_format: str = "WAV",
    ) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format=audio_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class AudioEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
