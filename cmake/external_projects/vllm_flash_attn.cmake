# vLLM flash attention requires VLLM_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_GPU_ARCHES here.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component _vllm_fa2_C (for FA2), --component _vllm_fa3_C (for FA3),
# or --component _vllm_fa4_cutedsl_C (for FA4 CuteDSL Python files).
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          vllm-flash-attn SOURCE_DIR 
          ${VLLM_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG bce29425653ec0fbc579d329883030e832d15ada
          GIT_PROGRESS TRUE
          # Don't share the vllm-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
endif()

# Make sure vllm-flash-attn install rules are nested under vllm/
# ALL_COMPONENTS ensures the save/modify/restore runs exactly once regardless
# of how many components are being installed, avoiding double-append of /vllm/.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" ALL_COMPONENTS)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm/\")" ALL_COMPONENTS)

# Fetch the vllm-flash-attn library. Use explicit populate so we can patch the
# vendored CMake before add_subdirectory configures it.
FetchContent_GetProperties(vllm-flash-attn)
if(NOT vllm-flash-attn_POPULATED)
  FetchContent_Populate(vllm-flash-attn)
endif()

set(_VLLM_CUDA_MIN_ARCH)
if(DEFINED VLLM_CUDA_MIN_ARCH)
  set(_VLLM_CUDA_MIN_ARCH "${VLLM_CUDA_MIN_ARCH}")
elseif(DEFINED ENV{VLLM_CUDA_MIN_ARCH})
  set(_VLLM_CUDA_MIN_ARCH "$ENV{VLLM_CUDA_MIN_ARCH}")
endif()

if(_VLLM_CUDA_MIN_ARCH AND "${_VLLM_CUDA_MIN_ARCH}" VERSION_GREATER_EQUAL "10.0")
  set(_VLLM_FLASH_ATTN_CMAKELISTS "${vllm-flash-attn_SOURCE_DIR}/CMakeLists.txt")
  file(READ "${_VLLM_FLASH_ATTN_CMAKELISTS}" _VLLM_FLASH_ATTN_CMAKE)
  string(REPLACE
    [[cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" "${CUDA_ARCHS}")]]
    [[cuda_archs_loose_intersection(FA2_ARCHS "10.0+PTX" "${CUDA_ARCHS}")]]
    _VLLM_FLASH_ATTN_CMAKE
    "${_VLLM_FLASH_ATTN_CMAKE}")
  file(WRITE "${_VLLM_FLASH_ATTN_CMAKELISTS}" "${_VLLM_FLASH_ATTN_CMAKE}")
  message(STATUS "Patched vllm-flash-attn FA2 minimum architecture to "
                 "${_VLLM_CUDA_MIN_ARCH}.")
endif()

add_subdirectory("${vllm-flash-attn_SOURCE_DIR}" "${vllm-flash-attn_BINARY_DIR}")
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Restore the install prefix after FA's install rules
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Install shared Python files for both FA2 and FA3 components
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  # Ensure the vllm/vllm_flash_attn directory exists before installation
  install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")"
    COMPONENT ${_FA_COMPONENT})

  # Copy vllm_flash_attn python files (except __init__.py and flash_attn_interface.py
  # which are source-controlled in vllm)
  install(
    DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT ${_FA_COMPONENT}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__init__.py" EXCLUDE
    PATTERN "flash_attn_interface.py" EXCLUDE
  )

endforeach()

#
# FA4 CuteDSL component
# This is a Python-only component that copies the flash_attn/cute directory
# and transforms imports to match our package structure.
#
add_custom_target(_vllm_fa4_cutedsl_C)

# Install flash_attn/cute directory (needed for FA4).
# When using a local source dir (VLLM_FLASH_ATTN_SRC_DIR), create a symlink
# so edits to cute-dsl Python files take effect immediately without rebuilding.
# Otherwise, copy files and transform flash_attn.cute imports to
# vllm.vllm_flash_attn.cute to match our package structure.
if(VLLM_FLASH_ATTN_SRC_DIR)
  install(CODE "
    set(LINK_TARGET \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\")
    set(LINK_NAME \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute\")
    file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")
    file(REMOVE_RECURSE \"\${LINK_NAME}\")
    file(CREATE_LINK \"\${LINK_TARGET}\" \"\${LINK_NAME}\" SYMBOLIC)
  " COMPONENT _vllm_fa4_cutedsl_C)
else()
  install(CODE "
    file(GLOB_RECURSE CUTE_PY_FILES \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute/*.py\")
    foreach(SRC_FILE \${CUTE_PY_FILES})
      file(RELATIVE_PATH REL_PATH \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
      set(DST_FILE \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute/\${REL_PATH}\")
      get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
      file(MAKE_DIRECTORY \${DST_DIR})
      file(READ \${SRC_FILE} FILE_CONTENTS)
      string(REPLACE \"flash_attn.cute\" \"vllm.vllm_flash_attn.cute\" FILE_CONTENTS \"\${FILE_CONTENTS}\")
      file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
    endforeach()
  " COMPONENT _vllm_fa4_cutedsl_C)
endif()
