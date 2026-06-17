#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build a codec-safe FFmpeg from source.
#
# LOOSE variant: disables the H.264, H.265/HEVC, and AAC codec
# encoders/decoders/parsers, but KEEPS the MP4/M4A/MOV container muxers
# and demuxers. This means VP9-in-MP4 still works (MP4 container is
# demuxable; only its AAC audio stream is rejected by codec). MP4 files
# that contain AAC audio will fail at decode time, not at container open.
#
# Installs to /opt/ffmpeg-safe.
#
# Requires apt-based system. Installs and removes build deps (build-only
# packages are purged at the end to keep Docker layer sizes small).

set -euo pipefail

FFMPEG_VERSION="${FFMPEG_VERSION:-8.1.1}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-/opt/ffmpeg-safe}"
FFMPEG_TARBALL_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
# Tarball hash pinned in-script for tamper-evident builds. Update when
# bumping FFMPEG_VERSION. The upstream .sha256sum URL is not reliably
# served (404 on some releases), so we inline the canonical hash here
# instead of fetching it from the same host as the tarball.
#
# Default 8.1.1 pin is shared with the amd64 cv2 shim. See the
# FFMPEG_SHA256_8_1_1 declaration below.
# FFmpeg 5.1 LTS pin — used by the cv2-shim variant whose SONAME majors
# (libavcodec.so.59, libavformat.so.59, libavutil.so.57, libswscale.so.6,
# libswresample.so.4) match opencv-python-headless 4.13's bundled FFmpeg
# on arm64 / aarch64.
FFMPEG_SHA256_5_1_6="f4fa066278f7a47feab316fef905f4db0d5e9b589451949740f83972b30901bd"
# FFmpeg 8.1.1 pin — used by the cv2-shim variant whose SONAME majors
# (libavcodec.so.62, libavformat.so.62, libavutil.so.60, libswscale.so.9,
# libswresample.so.6) match opencv-python-headless 4.13's bundled FFmpeg
# on amd64 / x86_64. The wheel builders apparently bundle different
# FFmpeg majors per platform, so both shims are required.
#
# 8.1.x is the first release line that contains the CENC-subsample
# integer-overflow fix (CVE-2026-40962) in libavformat/mov.c. SONAME
# stays at 62 across 8.x, so this is ABI-compatible with the opencv
# amd64 wheel's bundled libavcodec.so.62.
FFMPEG_SHA256_8_1_1="b6863adde98898f42602017462871b5f6333e65aec803fdd7a6308639c52edf3"
BUILD_DIR="$(mktemp -d)"

echo "[build-ffmpeg-safe] Installing build dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    xz-utils \
    patch \
    pkg-config \
    build-essential \
    nasm \
    yasm \
    libvpx-dev \
    libvorbis-dev \
    libopus-dev \
    libdav1d-dev \
    libsoxr-dev

echo "[build-ffmpeg-safe] Downloading FFmpeg ${FFMPEG_VERSION}..."
cd "${BUILD_DIR}"
tarball="ffmpeg-${FFMPEG_VERSION}.tar.xz"
wget -q "${FFMPEG_TARBALL_URL}" -O "${tarball}"
# Look up the pinned hash for this version. Fail loudly if the version
# was bumped without updating the pin above.
sha256_var="FFMPEG_SHA256_${FFMPEG_VERSION//./_}"
expected_sha256="${!sha256_var:-}"
if [[ -z "${expected_sha256}" ]]; then
    echo "[build-ffmpeg-safe] ERROR: no pinned SHA-256 for FFmpeg ${FFMPEG_VERSION}." >&2
    echo "  Add ${sha256_var} near the top of this script." >&2
    exit 1
fi
echo "${expected_sha256}  ${tarball}" | sha256sum -c -
tar -xf "${tarball}"
cd "ffmpeg-${FFMPEG_VERSION}"

# Apply any version-specific backport patches. Patches live next to this
# script under ffmpeg-patches/ and are named "${FFMPEG_VERSION}-<slug>.patch"
# so the script can pick the right set for whichever version is being built.
# This is how downstream CVE backports land on FFmpeg versions whose SONAME
# is pinned by ABI compatibility constraints (e.g. the 5.1.6 shim that has
# to match opencv-python-headless's bundled libavcodec.so.59 on arm64).
FFMPEG_PATCHES_DIR="${FFMPEG_PATCHES_DIR:-/tmp/ffmpeg-patches}"
if [[ -d "${FFMPEG_PATCHES_DIR}" ]]; then
    shopt -s nullglob
    patches=("${FFMPEG_PATCHES_DIR}"/"${FFMPEG_VERSION}"-*.patch)
    shopt -u nullglob
    if (( ${#patches[@]} > 0 )); then
        echo "[build-ffmpeg-safe] Applying ${#patches[@]} backport patch(es) for FFmpeg ${FFMPEG_VERSION}:"
        for p in "${patches[@]}"; do
            echo "  - $(basename "${p}")"
            patch -p1 < "${p}"
        done
    else
        echo "[build-ffmpeg-safe] No backport patches found for FFmpeg ${FFMPEG_VERSION} under ${FFMPEG_PATCHES_DIR}"
    fi
else
    echo "[build-ffmpeg-safe] Patches dir ${FFMPEG_PATCHES_DIR} not present; skipping patch application."
fi

echo "[build-ffmpeg-safe] Configuring (LOOSE: codecs disabled, MP4/MOV demuxers preserved)..."
./configure \
    --prefix="${FFMPEG_PREFIX}" \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-debug \
    --disable-gpl \
    --disable-nonfree \
    \
    --enable-libvpx \
    --enable-libvorbis \
    --enable-libopus \
    --enable-libdav1d \
    --enable-libsoxr \
    \
    --disable-libopenh264 \
    --disable-libx264 \
    --disable-libx265 \
    --disable-libfdk-aac \
    \
    --disable-decoder=h264 \
    --disable-decoder=hevc \
    --disable-decoder=h264_v4l2m2m \
    --disable-decoder=h264_vaapi \
    --disable-decoder=h264_nvdec \
    --disable-decoder=h264_cuvid \
    --disable-decoder=h264_qsv \
    --disable-decoder=hevc_v4l2m2m \
    --disable-decoder=hevc_vaapi \
    --disable-decoder=hevc_nvdec \
    --disable-decoder=hevc_cuvid \
    --disable-decoder=hevc_qsv \
    --disable-decoder=aac \
    --disable-decoder=aac_latm \
    --disable-decoder=aac_fixed \
    \
    --disable-encoder=h264_v4l2m2m \
    --disable-encoder=h264_vaapi \
    --disable-encoder=h264_nvenc \
    --disable-encoder=h264_amf \
    --disable-encoder=h264_mf \
    --disable-encoder=h264_qsv \
    --disable-encoder=hevc_v4l2m2m \
    --disable-encoder=hevc_vaapi \
    --disable-encoder=hevc_nvenc \
    --disable-encoder=hevc_amf \
    --disable-encoder=hevc_mf \
    --disable-encoder=hevc_qsv \
    --disable-encoder=aac \
    --disable-encoder=aac_mf \
    --disable-encoder=aac_at \
    \
    --disable-parser=h264 \
    --disable-parser=hevc \
    --disable-parser=aac \
    --disable-parser=aac_latm \
    \
    --disable-bsf=aac_adtstoasc \
    --disable-bsf=h264_metadata \
    --disable-bsf=h264_mp4toannexb \
    --disable-bsf=h264_redundant_pps \
    --disable-bsf=hevc_metadata \
    --disable-bsf=hevc_mp4toannexb

echo "[build-ffmpeg-safe] Building (this may take 5-10 minutes)..."
make -j"$(nproc)"

echo "[build-ffmpeg-safe] Installing to ${FFMPEG_PREFIX}..."
make install

echo "[build-ffmpeg-safe] Registering shared libraries with ldconfig..."
# Use a per-prefix config filename so multiple invocations (system FFmpeg
# at /opt/ffmpeg-safe + cv2-shim variants at /opt/ffmpeg-safe-cv2-shim-vN)
# do not stomp on each other. A single shared filename caused
# /etc/ld.so.conf.d/ffmpeg-safe.conf to be overwritten on each call,
# leaving only the last-built prefix in the linker search path.
ldconf_basename="$(basename "${FFMPEG_PREFIX}")"
echo "${FFMPEG_PREFIX}/lib" > "/etc/ld.so.conf.d/${ldconf_basename}.conf"
ldconfig

echo "[build-ffmpeg-safe] Cleaning up build tree and build-only packages..."
cd /
rm -rf "${BUILD_DIR}"
# Protect the runtime shared libraries from autoremove. FFmpeg dynamically
# links against these at runtime, but apt does not know that — `apt` only
# tracks dpkg-level dependencies, and /opt/ffmpeg-safe is not a dpkg package.
# Without this step the runtime libs (libvpx9, libvorbis0a, libopus0,
# libdav1d7, libsoxr0, libogg0, libvorbisfile3, libvorbisenc2, libsoxr-lsr0)
# get swept by `apt-get autoremove` after the -dev packages are purged below.
#
# Use `dpkg-query` (lists actually-installed packages, independent of
# whether the apt cache is generated) rather than `apt-cache pkgnames`,
# which returns nothing when the cache has not been pre-generated.
#
# Regexes are broadened to cover every runtime variant (e.g. libvorbisfile3
# and libvorbisenc2 in addition to libvorbis0a), and libogg* is added
# because libvorbis pulls it in transitively.
runtime_libs="$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
    | grep -E '^(libvpx[0-9]+|libvorbis[0-9a-z]+|libogg[0-9]+|libopus[0-9]+|libdav1d[0-9]+|libsoxr[0-9]+|libsoxr-lsr[0-9]+)$' \
    || true)"
if [[ -n "${runtime_libs}" ]]; then
    echo "[build-ffmpeg-safe] Marking runtime libs as manually-installed:"
    echo "${runtime_libs}" | sed 's/^/  /'
    # shellcheck disable=SC2086
    apt-mark manual ${runtime_libs}
else
    echo "[build-ffmpeg-safe] WARNING: no runtime libs matched; autoremove may sweep them" >&2
fi
# NOTE: `build-essential` is intentionally NOT purged. It transitively
# pulls in dpkg-dev / libc6-dev / header packages that downstream
# Dockerfile RUN layers may rely on. Purging it here can silently
# break later stages.
apt-get purge -y \
    nasm \
    yasm \
    libvpx-dev \
    libvorbis-dev \
    libopus-dev \
    libdav1d-dev \
    libsoxr-dev
apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*

# Final sanity check: verify FFmpeg can actually load its dependencies.
# This catches the autoremove-stripped-runtime-libs failure mode before
# verify-ffmpeg-safe.sh runs (which would otherwise produce a confusing
# error about "ffmpeg: error while loading shared libraries").
if ! "${FFMPEG_PREFIX}/bin/ffmpeg" -version >/dev/null 2>&1; then
    echo "[build-ffmpeg-safe] FAIL: ${FFMPEG_PREFIX}/bin/ffmpeg cannot load its dependencies." >&2
    echo "[build-ffmpeg-safe] Missing libraries:" >&2
    ldd "${FFMPEG_PREFIX}/bin/ffmpeg" 2>&1 | grep "not found" | sed 's/^/  /' >&2
    exit 1
fi

echo "[build-ffmpeg-safe] Done. FFmpeg installed at ${FFMPEG_PREFIX}"
"${FFMPEG_PREFIX}/bin/ffmpeg" -version | head -1
