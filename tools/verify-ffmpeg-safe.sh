#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Verify that the installed FFmpeg does NOT support the royalty-bearing
# codecs (H.264, H.265/HEVC, AAC).
#
# Exits non-zero if any forbidden codec is reported as a decoder (D), encoder
# (E), or parser. Intended to gate the Docker build.

set -euo pipefail

FFMPEG="${FFMPEG:-/opt/ffmpeg-safe/bin/ffmpeg}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-/opt/ffmpeg-safe}"

# Ensure the shared libs are findable even if ldconfig was not run.
# Use the `${VAR:+:$VAR}` idiom so an unset upstream value does not leave
# a trailing colon — a trailing `:` in LD_LIBRARY_PATH means "CWD".
export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "${FFMPEG}" ]]; then
    echo "[verify-ffmpeg-safe] ERROR: ${FFMPEG} not found or not executable" >&2
    exit 1
fi

echo "[verify-ffmpeg-safe] Verifying ${FFMPEG}..."
"${FFMPEG}" -version | head -1

forbidden_codecs=(h264 hevc aac)
fail=0

# `ffmpeg -codecs` lists every codec with capability flags: DEV.LS h264 ...
# Columns: D=decode, E=encode, V/A/S=video/audio/subtitle, I/L/S=intra/lossy/lossless.
# We fail if any forbidden codec line shows D or E in the first 6 chars.
codec_table="$("${FFMPEG}" -hide_banner -codecs 2>/dev/null)"

for codec in "${forbidden_codecs[@]}"; do
    # Match a codec name appearing as the 2nd field of -codecs output,
    # with at least one of D/E in the capability flags.
    line="$(echo "${codec_table}" | awk -v c="${codec}" '
        $2 == c {
            flags = $1
            if (flags ~ /D/ || flags ~ /E/) print
        }
    ')"
    if [[ -n "${line}" ]]; then
        echo "[verify-ffmpeg-safe] FAIL: ${codec} present with D/E capability: ${line}" >&2
        fail=1
    else
        echo "[verify-ffmpeg-safe] OK: ${codec} absent or non-functional"
    fi
done

# Also confirm at least one royalty-free codec is present so the build is useful.
# `av1` is included because the build wires `--enable-libdav1d`; without this
# check a broken dav1d link would silently produce an AV1-less binary that
# still passes the forbidden-codec checks above.
required_codecs=(vp9 opus vorbis av1)
for codec in "${required_codecs[@]}"; do
    if ! echo "${codec_table}" | awk -v c="${codec}" '$2 == c { print; found=1; exit } END { exit !found }' >/dev/null; then
        echo "[verify-ffmpeg-safe] FAIL: required royalty-free codec ${codec} missing" >&2
        fail=1
    else
        echo "[verify-ffmpeg-safe] OK: ${codec} present"
    fi
done

# Verify forbidden-codec parser absence by inspecting libavcodec's exported
# symbols. FFmpeg does NOT expose a `-parsers` CLI option (verified against
# ffmpeg 7.1 — only -codecs / -decoders / -encoders / -formats / -muxers /
# -demuxers / -bsfs / -filters / -pix_fmts are listable). Parser registration
# symbols are named `ff_<codec>_parser` in libavcodec; if the configure-time
# `--disable-parser=<codec>` was honored, those symbols will be absent.
libavcodec="$(ls -1 "${FFMPEG_PREFIX}/lib/libavcodec.so."* 2>/dev/null | head -1 || true)"
if [[ -z "${libavcodec}" || ! -f "${libavcodec}" ]]; then
    echo "[verify-ffmpeg-safe] FAIL: cannot locate libavcodec.so under ${FFMPEG_PREFIX}/lib/" >&2
    fail=1
else
    if ! command -v nm >/dev/null 2>&1; then
        echo "[verify-ffmpeg-safe] WARNING: nm not available; skipping parser symbol check" >&2
    else
        parser_symbols="$(nm -D --defined-only "${libavcodec}" 2>/dev/null | awk '{print $NF}' || true)"
        forbidden_parsers=(h264 hevc aac aac_latm)
        for parser in "${forbidden_parsers[@]}"; do
            if echo "${parser_symbols}" | grep -q "^ff_${parser}_parser$"; then
                echo "[verify-ffmpeg-safe] FAIL: ff_${parser}_parser symbol present in $(basename "${libavcodec}")" >&2
                fail=1
            else
                echo "[verify-ffmpeg-safe] OK: ff_${parser}_parser absent from $(basename "${libavcodec}")"
            fi
        done
    fi
fi

# Verify that opencv-python-headless, if installed, has had its bundled
# FFmpeg shared libraries stripped (see tools/strip-opencv-bundled-codecs.sh).
# opencv-python-headless wheels ship a private FFmpeg under
# site-packages/opencv_python_headless.libs/ that includes H.264/HEVC/AAC
# decoders compiled into libavcodec.so. cv2 is RPATH-linked to those bundles,
# so /opt/ffmpeg-safe does not close that path; the only sound mitigation is
# physical removal of the bundled FFmpeg libraries.
opencv_libs_dir="$(python3 - <<'PY' 2>/dev/null || true
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("cv2")
if spec is None or spec.origin is None:
    sys.exit(0)
parent = pathlib.Path(spec.origin).parent
for cand in (parent.parent / "opencv_python_headless.libs",
             parent / "opencv_python_headless.libs"):
    if cand.is_dir():
        print(cand)
        sys.exit(0)
PY
)"
if [[ -n "${opencv_libs_dir}" && -d "${opencv_libs_dir}" ]]; then
    # Glob explicitly lists each FFmpeg family — `libavif` (AVIF image codec,
    # royalty-free, separate library) matches `libav*` but is not FFmpeg
    # and must be left alone.
    remaining="$(find "${opencv_libs_dir}" -maxdepth 1 -type f \
        \( -name 'libavcodec-*.so.*'      -o -name 'libavdevice-*.so.*' \
           -o -name 'libavfilter-*.so.*'  -o -name 'libavformat-*.so.*' \
           -o -name 'libavutil-*.so.*'    -o -name 'libswresample-*.so.*' \
           -o -name 'libswscale-*.so.*'   -o -name 'libpostproc-*.so.*' \) \
        -printf '%f\n' 2>/dev/null || true)"
    if [[ -n "${remaining}" ]]; then
        echo "[verify-ffmpeg-safe] FAIL: opencv-python-headless bundles FFmpeg under" >&2
        echo "  ${opencv_libs_dir}:" >&2
        echo "${remaining}" | sed 's/^/    /' >&2
        echo "  Run tools/strip-opencv-bundled-codecs.sh after pip install." >&2
        fail=1
    else
        echo "[verify-ffmpeg-safe] OK: no bundled FFmpeg in ${opencv_libs_dir}"
    fi
else
    echo "[verify-ffmpeg-safe] OK: opencv-python-headless not installed (no bundled FFmpeg to check)"
fi

if [[ ${fail} -ne 0 ]]; then
    echo "[verify-ffmpeg-safe] Verification FAILED" >&2
    exit 1
fi

echo "[verify-ffmpeg-safe] Verification PASSED — no royalty-bearing codecs detected"
