#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Replace the bundled FFmpeg shared libraries that opencv-python-headless
# wheels ship under site-packages/opencv_python_headless.libs/ with symlinks
# to a codec-safe FFmpeg shim of matching SONAME major version.
#
# Why this matters: those bundled libraries include FFmpeg's internal native
# H.264, H.265/HEVC, and AAC decoders compiled into libavcodec.so.
# /opt/ffmpeg-safe (the system codec-safe FFmpeg at the latest major) does
# NOT close this path because cv2 is RPATH-linked to its own bundled libs
# and the dynamic linker resolves them via the cv2 module's NEEDED entries,
# not the system soname registry. Distributing a binary that contains the
# codec implementations triggers MPEG-LA / Access Advance royalty
# obligations even if no code path ever invokes them at runtime.
#
# Why symlinks and not deletion: cv2.cpython-XXX-XXX-linux-gnu.so has a hard
# NEEDED entry on libavcodec-<hash>.so.<major> (from when opencv was
# wheel-built). Deleting the bundled lib causes the dynamic linker to fail
# when `import cv2` runs, breaking ALL cv2 use including image operations
# (imread, imwrite, resize, cvtColor). Symlinking to a same-SONAME-major
# codec-safe library keeps cv2 importable, keeps image operations working,
# and keeps cv2.VideoCapture working for non-royalty codec containers, while
# removing the H.264/HEVC/AAC code from distribution.
#
# Why multiple shims: opencv-python-headless 4.13 bundles different FFmpeg
# major versions per platform: avcodec.so.59 (FFmpeg 5.1.x) on arm64,
# avcodec.so.62 (FFmpeg 8.x) on amd64. We install both shims and pick the
# matching one per bundled-lib SONAME major. Shim directories are
# /opt/ffmpeg-safe-cv2-shim-v<N> where N is the major. Override via
# FFMPEG_CV2_SHIM_GLOB if you install them elsewhere.

set -euo pipefail

FFMPEG_CV2_SHIM_GLOB="${FFMPEG_CV2_SHIM_GLOB:-/opt/ffmpeg-safe-cv2-shim-v*}"

# Build the soname → shim-path map by enumerating every shim's lib dir.
declare -A shim_for_soname
shopt -s nullglob
for shim_dir in ${FFMPEG_CV2_SHIM_GLOB}; do
    lib_dir="${shim_dir}/lib"
    [[ -d "${lib_dir}" ]] || continue
    for lib in "${lib_dir}"/lib{av,sw,postproc}*.so.[0-9]*; do
        [[ -f "${lib}" ]] || continue
        soname="$(basename "${lib}")"
        # Keep only sonames of the form libfoo.so.MAJOR (no .MINOR.PATCH suffix).
        if [[ "${soname}" =~ ^lib[a-z]+\.so\.[0-9]+$ ]]; then
            shim_for_soname["${soname}"]="${lib}"
        fi
    done
done

if [[ ${#shim_for_soname[@]} -eq 0 ]]; then
    echo "[strip-opencv-bundled-codecs] ERROR: no codec-safe shim found under ${FFMPEG_CV2_SHIM_GLOB}" >&2
    echo "  Build at least one cv2 shim before running this script, e.g.:" >&2
    echo "    FFMPEG_VERSION=5.1.6 FFMPEG_PREFIX=/opt/ffmpeg-safe-cv2-shim-v5 bash build-ffmpeg-safe.sh" >&2
    echo "    FFMPEG_VERSION=8.1.1 FFMPEG_PREFIX=/opt/ffmpeg-safe-cv2-shim-v8 bash build-ffmpeg-safe.sh" >&2
    exit 1
fi

echo "[strip-opencv-bundled-codecs] discovered shim libraries:"
for soname in $(printf '%s\n' "${!shim_for_soname[@]}" | sort); do
    echo "  ${soname} -> ${shim_for_soname[${soname}]}"
done

# Locate the bundled-libs directory next to the cv2 package.
opencv_libs_dir="$(python3 - <<'PY'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("cv2")
if spec is None or spec.origin is None:
    sys.exit(0)  # cv2 not installed; nothing to do
parent = pathlib.Path(spec.origin).parent
for cand in (parent.parent / "opencv_python_headless.libs",
             parent / "opencv_python_headless.libs"):
    if cand.is_dir():
        print(cand)
        sys.exit(0)
PY
)"

if [[ -z "${opencv_libs_dir}" ]]; then
    echo "[strip-opencv-bundled-codecs] cv2 not installed; nothing to do"
    exit 0
fi

if [[ ! -d "${opencv_libs_dir}" ]]; then
    echo "[strip-opencv-bundled-codecs] no bundled-libs directory at ${opencv_libs_dir}; nothing to do"
    exit 0
fi

echo "[strip-opencv-bundled-codecs] bundled-libs dir: ${opencv_libs_dir}"

# For each bundled FFmpeg shared library, extract its family + major SONAME
# and replace the bundled file with a symlink to the matching shim lib.
#
# Bundled filename format produced by pip's auditwheel:
#   libavcodec-5696b3bf.so.59.37.100
#       ^         ^         ^  ^   ^
#       family    hash      major minor patch
replaced_any=0
# libav* family that ships in FFmpeg. Note: `libavif` is the AVIF image codec
# library (royalty-free, separate from FFmpeg) and is intentionally NOT in
# this list — it has its own SONAME (libavif.so.16) that no FFmpeg shim
# provides, and it carries no royalty obligation, so we leave it alone.
for prefix in libavcodec libavdevice libavfilter libavformat libavutil libswresample libswscale libpostproc; do
    for bundled in "${opencv_libs_dir}/${prefix}"-*.so.*; do
        name="$(basename "${bundled}")"
        # Extract major version: libavcodec-5696b3bf.so.59.37.100 -> 59
        major="$(echo "${name}" | sed -E 's/^lib[a-z]+-[0-9a-f]+\.so\.([0-9]+).*$/\1/')"
        if ! [[ "${major}" =~ ^[0-9]+$ ]]; then
            echo "[strip-opencv-bundled-codecs] WARNING: could not parse major from ${name}; skipping" >&2
            continue
        fi
        soname="${prefix}.so.${major}"
        shim_path="${shim_for_soname[${soname}]:-}"
        if [[ -z "${shim_path}" ]]; then
            # No installed shim provides this soname. The configure-time
            # disable flags in build-ffmpeg-safe.sh may have removed this
            # library entirely (e.g. libavfilter if --disable-avfilter was
            # passed). Remove the bundled copy so it does not ship; cv2
            # paths that NEEDED it will fail at runtime, which is acceptable
            # since those paths were either the royalty-bearing ones we are
            # removing or unrelated paths that an upstream change broke.
            echo "[strip-opencv-bundled-codecs] WARN: no shim provides ${soname}; removing ${name}" >&2
            rm -f -- "${bundled}"
            replaced_any=1
            continue
        fi
        rm -f -- "${bundled}"
        ln -s "${shim_path}" "${bundled}"
        echo "[strip-opencv-bundled-codecs] ${name} -> ${shim_path}"
        replaced_any=1
    done
done

if [[ ${replaced_any} -eq 0 ]]; then
    echo "[strip-opencv-bundled-codecs] no bundled FFmpeg libraries found; idempotent re-run"
fi

# Confirm cv2 imports. Image ops (imread/imwrite/resize/cvtColor) do not
# call into FFmpeg, so they should keep working through the symlinked
# safe libavcodec. cv2.VideoCapture for safe codec containers (WebM/Opus,
# etc.) should also keep working; H.264/HEVC/AAC video opens will fail
# because the shim has those codecs explicitly disabled.
if ! python3 -c 'import cv2; print(f"[strip-opencv-bundled-codecs] cv2 imports OK ({cv2.__version__})")'; then
    echo "[strip-opencv-bundled-codecs] ERROR: cv2 fails to import after strip" >&2
    exit 1
fi

# Final assertion: every FFmpeg lib file (libavcodec/libavdevice/libavfilter/
# libavformat/libavutil/libswresample/libswscale/libpostproc) under the
# bundled-libs dir is now either absent (no shim provided it) or a symlink
# that resolves outside that directory. We must not ship a regular file
# there, since that would mean we left the bundled royalty-bearing binary
# in place.
#
# Glob explicitly lists each FFmpeg family rather than using `libav*` —
# `libavif` (AVIF image codec, royalty-free, separate library) matches
# `libav*` but is not FFmpeg and is intentionally left untouched.
bad="$(find "${opencv_libs_dir}" -maxdepth 1 -type f \
    \( -name 'libavcodec-*.so.*'      -o -name 'libavdevice-*.so.*' \
       -o -name 'libavfilter-*.so.*'  -o -name 'libavformat-*.so.*' \
       -o -name 'libavutil-*.so.*'    -o -name 'libswresample-*.so.*' \
       -o -name 'libswscale-*.so.*'   -o -name 'libpostproc-*.so.*' \) \
    -printf '%f\n' 2>/dev/null || true)"
if [[ -n "${bad}" ]]; then
    echo "[strip-opencv-bundled-codecs] FAIL: bundled FFmpeg regular files still present:" >&2
    echo "${bad}" | sed 's/^/    /' >&2
    exit 1
fi

# Confirm any symlinks we created resolve to one of the shim prefixes.
echo "[strip-opencv-bundled-codecs] Symlink resolution check:"
for link in "${opencv_libs_dir}"/libavcodec-*.so.* \
            "${opencv_libs_dir}"/libavdevice-*.so.* \
            "${opencv_libs_dir}"/libavfilter-*.so.* \
            "${opencv_libs_dir}"/libavformat-*.so.* \
            "${opencv_libs_dir}"/libavutil-*.so.* \
            "${opencv_libs_dir}"/libswresample-*.so.* \
            "${opencv_libs_dir}"/libswscale-*.so.* \
            "${opencv_libs_dir}"/libpostproc-*.so.*; do
    [[ -L "${link}" ]] || continue
    target="$(readlink -f "${link}")"
    matched=0
    for shim_dir in ${FFMPEG_CV2_SHIM_GLOB}; do
        case "${target}" in
            "${shim_dir}/"*) matched=1; status="OK (${shim_dir})"; break ;;
        esac
    done
    if [[ ${matched} -eq 0 ]]; then
        status="UNEXPECTED → ${target}"
    fi
    echo "  ${status}: $(basename "${link}")"
done

echo "[strip-opencv-bundled-codecs] Done. Bundled royalty-bearing FFmpeg replaced with shim symlinks."
