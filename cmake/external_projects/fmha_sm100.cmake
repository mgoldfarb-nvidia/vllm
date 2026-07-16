include(FetchContent)

# fmha_sm100 source, in preference order:
#   1. FMHA_SM100_SRC_DIR env var  — explicit local MSA checkout (dev override)
#   2. third_party/MSA submodule   — vendored fork of vllm-project/MSA @ fee7831
#      plus the sm_100f family gencode target, so the JIT'd fmha_sm100 kernels
#      have an image for SM107 (VR200). Upstream only ships sm_100a/sm_103a,
#      which fails on cc 10.7 with "no kernel image is available for execution
#      on the device". The submodule is fetched at (authenticated) CI checkout
#      time, so the credential-less wheel build never has to clone it.
#   3. upstream github  — anonymous clone fallback for envs without the
#      submodule (e.g. non-SM107 dev); lacks the sm_100f fix.
if(DEFINED ENV{FMHA_SM100_SRC_DIR})
  set(FMHA_SM100_SRC_DIR $ENV{FMHA_SM100_SRC_DIR})
elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/MSA/python/fmha_sm100/jit.py")
  set(FMHA_SM100_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/MSA")
endif()

if(FMHA_SM100_SRC_DIR)
  message(STATUS "fmha_sm100 using local source: ${FMHA_SM100_SRC_DIR}")
  FetchContent_Declare(
    fmha_sm100
    SOURCE_DIR ${FMHA_SM100_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
    fmha_sm100
    # Fork of vllm-project/MSA @ fee7831 + the sm_100f family gencode target so
    # the JIT'd fmha_sm100 kernels have an image for SM107 (VR200); upstream only
    # ships sm_100a/sm_103a, which fails on cc 10.7 with "no kernel image is
    # available for execution on the device". Revert to upstream once merged.
    GIT_REPOSITORY https://gitlab-master.nvidia.com/zaristei/MSA.git
    GIT_TAG 12d9ca91be380b444e384b3b101ea5ba44d01396.
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

FetchContent_GetProperties(fmha_sm100)
if(NOT fmha_sm100_POPULATED)
  FetchContent_Populate(fmha_sm100)
endif()
message(STATUS "fmha_sm100 is available at ${fmha_sm100_SOURCE_DIR}")

add_custom_target(fmha_sm100)

set(FMHA_SM100_PY_ROOT "${fmha_sm100_SOURCE_DIR}/python/fmha_sm100")

install(FILES
  "${FMHA_SM100_PY_ROOT}/__init__.py"
  "${FMHA_SM100_PY_ROOT}/api.py"
  "${FMHA_SM100_PY_ROOT}/bench_utils.py"
  "${FMHA_SM100_PY_ROOT}/jit.py"
  "${FMHA_SM100_PY_ROOT}/sparse.py"
  "${FMHA_SM100_PY_ROOT}/sparse_fmha_adapter.py"
  DESTINATION vllm/third_party/fmha_sm100
  COMPONENT fmha_sm100)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/csrc/"
  DESTINATION vllm/third_party/fmha_sm100/csrc
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cute/"
  DESTINATION vllm/third_party/fmha_sm100/cute
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cutlass/include/"
  DESTINATION vllm/third_party/fmha_sm100/cutlass/include
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(DIRECTORY "${FMHA_SM100_PY_ROOT}/cutlass/tools/util/include/"
  DESTINATION vllm/third_party/fmha_sm100/cutlass/tools/util/include
  COMPONENT fmha_sm100
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)
