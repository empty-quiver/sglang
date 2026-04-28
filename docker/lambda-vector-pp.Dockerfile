# syntax=docker/dockerfile:1.7
# sglang-kt + kt-kernel image for serving MoE models with CPU expert offload.
# Targets RTX 3060 (sm_86) + RTX 4090 (sm_89) heterogeneous PP, both Ada/Ampere
# class GPUs without FP4 support. Built from THIS fork (empty-quiver/sglang)
# at the lambda-vector-pp branch — see LAMBDA-VECTOR-PP.md for context.
#
# We build sgl-kernel from the in-tree source (sgl-kernel/) for sm_86+sm_89
# because the published wheels (0.3.21+) only ship sm_90 (Hopper) + sm_100
# (Blackwell) binaries.
# Base is CUDA 13 + Ubuntu 24.04 so the default torch wheel (cu130) works
# without needing a custom PyTorch index URL.

# ---- Stage 1: build sgl-kernel for sm_86 + sm_89 ----
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        git build-essential cmake ninja-build ccache \
        curl ca-certificates pkg-config \
        libnuma-dev libibverbs-dev libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_BUILD_PARALLEL_LEVEL=12 \
    MAX_JOBS=12 \
    CCACHE_DIR=/root/.ccache \
    CMAKE_POLICY_VERSION_MINIMUM=3.5

# Python 3.12 venv (Ubuntu 24.04 default Python).
RUN uv venv --python 3.12 /opt/venv

# Install torch first (sgl-kernel build links against torch C++ ABI).
# torch>=2.10 from default PyPI pulls the cu130 wheel which matches our base.
RUN uv pip install --no-cache --link-mode=copy \
    "torch>=2.10,<2.12" "torchvision<0.27,>=0.25" "torchaudio<2.12,>=2.10" \
    ninja wheel setuptools packaging \
    "scikit-build-core>=0.10" pybind11 cmake cython numpy

# Copy the entire fork into /src/sglang. The build context is the repo root
# so this gets us sgl-kernel/, python/, etc.
COPY . /src/sglang

# Inject sm_86 gencode right after the sm_89 line so the build emits cubins
# for both 4090 (sm_89) and 3060 (sm_86) MoE/Mamba/RMSNorm kernels. NOTE:
# the source already has an sm_86 entry in the SGL_FLASH_KERNEL_CUDA_FLAGS
# block lower in the file; we only target the SGL_KERNEL_CUDA_FLAGS block,
# which is right after the sm_89 line that this sed-append-after matches.
RUN sed -i '/"-gencode=arch=compute_89,code=sm_89"/a\        "-gencode=arch=compute_86,code=sm_86"' /src/sglang/sgl-kernel/CMakeLists.txt \
    && grep -nE "gencode=arch=compute_8[69]" /src/sglang/sgl-kernel/CMakeLists.txt

# Strip sgl-kernel down to just our targets (sm_86 + sm_89). Without this
# the build emits SASS for sm_80, sm_89, sm_90, sm_90a, sm_100a, sm_120a,
# sm_103a (and a separate common_ops_sm90_build target) — taking ~5x as long
# as needed and burning ~50 GB extra RAM. The loader picks the sm100/
# directory for any non-Hopper GPU, so we keep that target and drop sm90.
RUN sed -i \
        -e '/"-gencode=arch=compute_90,code=sm_90"/d' \
        -e '/"-gencode=arch=compute_80,code=sm_80"/d' \
        -e '/"-gencode=arch=compute_87,code=sm_87"/d' \
        -e '/"-gencode=arch=compute_90a,code=sm_90a"/d' \
        -e '/"-gencode=arch=compute_100a,code=sm_100a"/d' \
        -e '/"-gencode=arch=compute_103a,code=sm_103a"/d' \
        -e '/"-gencode=arch=compute_110a,code=sm_110a"/d' \
        -e '/"-gencode=arch=compute_120a,code=sm_120a"/d' \
        -e '/"-gencode=arch=compute_121a,code=sm_121a"/d' \
        -e '/"-gencode=arch=compute_101a,code=sm_101a"/d' \
        /src/sglang/sgl-kernel/CMakeLists.txt
RUN grep -n -E "gencode|SGL_KERNEL_CUDA_FLAGS\s*$" /src/sglang/sgl-kernel/CMakeLists.txt | head -10

# Drop the common_ops_sm90_build target entirely. We have no Hopper, and the
# load_utils.py loader picks the sm100 directory for everything not sm_90.
# Need to strip whole multi-line CMake calls (e.g. target_compile_options(name
# PRIVATE ... )) — naive line deletion leaves orphan args + close parens that
# break parsing. Do a balanced-paren scan to delete the full call.
RUN python3 - <<'PY'
import pathlib, re
p = pathlib.Path("/src/sglang/sgl-kernel/CMakeLists.txt")
src = p.read_text()
TARGET = "common_ops_sm90_build"
out = []
i = 0
lines = src.splitlines(keepends=True)
removed = 0
while i < len(lines):
    line = lines[i]
    if TARGET in line:
        depth = line.count("(") - line.count(")")
        if depth <= 0:
            removed += 1
            i += 1
            continue
        j = i + 1
        while j < len(lines) and depth > 0:
            depth += lines[j].count("(") - lines[j].count(")")
            j += 1
        removed += (j - i)
        i = j
    else:
        out.append(line)
        i += 1
pathlib.Path("/src/sglang/sgl-kernel/CMakeLists.txt").write_text("".join(out))
print(f"dropped sm90_build references: {removed} lines removed")
PY
RUN if grep -q "common_ops_sm90_build" /src/sglang/sgl-kernel/CMakeLists.txt; then \
        echo "STILL THERE — fail"; \
        grep -n "common_ops_sm90_build" /src/sglang/sgl-kernel/CMakeLists.txt; \
        exit 1; \
    else \
        echo "(0 references — clean)"; \
    fi
RUN python3 -c "src = open('/src/sglang/sgl-kernel/CMakeLists.txt').read(); o=src.count('('); c=src.count(')'); print(f'paren balance: {o} open, {c} close, diff={o-c}'); assert o==c, 'CMakeLists.txt has unbalanced parens after sm90 strip'"

# Build sgl-kernel wheel for sm_86+sm_89 from the in-tree source.
WORKDIR /src/sglang/sgl-kernel
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --no-deps --no-build-isolation -v .

# Clone kt-kernel source from kvcache-ai/ktransformers, apply the
# single-socket NUMA fix from issue #1754 (community patch the maintainer
# never merged upstream), then build kt-kernel for sm_86+sm_89.
# --recurse-submodules pulls in third_party/pybind11 and third_party/llama.cpp
# which kt-kernel/CMakeLists.txt needs as add_subdirectory targets.
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/kvcache-ai/ktransformers.git /src/ktransformers
COPY docker/kt-kernel-numa-single-socket.patch /tmp/
RUN cd /src/ktransformers && git apply -v /tmp/kt-kernel-numa-single-socket.patch \
 && grep -A2 "numa_num_configured_nodes" kt-kernel/cpu_backend/worker_pool.h | head -10
RUN test -f /src/ktransformers/third_party/pybind11/CMakeLists.txt && \
    test -f /src/ktransformers/third_party/llama.cpp/CMakeLists.txt && \
    echo "submodules present"
WORKDIR /src/ktransformers/kt-kernel
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --no-deps --no-build-isolation -v .

# ---- Stage 2: runtime ----
# Using -devel (not -runtime) because flashinfer JITs CUDA kernels on first
# import (e.g., flashinfer/data/csrc/norm.cu) and needs nvcc + CUDA headers
# at runtime. The image is ~3-4 GB larger but disk is comfortable.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
# gcc + python3-dev are required at RUNTIME because Triton JIT-compiles a
# small CUDA driver helper module on first import (CudaUtils) — without
# a C compiler in PATH the boot fails during CUDA graph capture init.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        gcc g++ \
        curl ca-certificates \
        libhwloc15 libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    KT_GPTQ_INT4_BACKEND=avxvnni \
    CC=gcc

# Copy the entire venv from the builder (includes our sm_86+sm_89 sgl-kernel
# and the patched kt-kernel).
COPY --from=builder /opt/venv /opt/venv

# Copy the fork's python/ directory so we can install our patched sglang-kt
# in editable mode from local source. The python/ subdirectory has its own
# pyproject.toml (sglang-kt's) and is the install target. We do NOT need
# to copy the repo-root files (no root pyproject.toml exists; sgl-kernel
# is already built and installed in /opt/venv from stage 1).
COPY --from=builder /src/sglang/python /opt/sglang/python

# Install sglang-kt from the local fork checkout WITHOUT deps so we don't
# overwrite the sm_86+sm_89 sgl-kernel or our patched kt-kernel that we
# built in the builder stage. Then install runtime deps explicitly.
# Match the upstream sglang-kt 0.6.1 version string. The python/setup.py reads
# SGLANG_KT_VERSION at build time; without it the version falls back to
# 0.0.0.dev0 which can confuse package-version probing downstream.
ENV SGLANG_KT_VERSION=0.6.1
WORKDIR /opt/sglang
RUN uv pip install --no-cache --no-deps -e ./python \
 && uv pip install --no-cache \
        IPython aiohttp "apache-tvm-ffi<0.2,>=0.1.5" "anthropic>=0.20.0" \
        blobfile==3.0.0 build compressed-tensors "cuda-python<13.3,>=13.1" \
        decord2 datasets einops fastapi \
        flashinfer_python==0.6.3 flashinfer_cubin==0.6.3 \
        gguf hf_transfer huggingface_hub interegular \
        "llguidance<0.8.0,>=0.7.11" modelscope msgspec ninja numpy \
        "nvidia-cutlass-dsl>=4.3.4" nvidia-ml-py \
        openai-harmony==0.0.4 openai==2.6.1 orjson outlines==0.1.11 \
        packaging partial_json_parser pillow "prometheus-client>=0.20.0" \
        psutil py-spy pybase64 pydantic python-multipart "pyzmq>=25.1.2" \
        requests scipy sentencepiece setproctitle soundfile==0.13.1 \
        tiktoken timm==1.0.16 torchcodec==0.8.0 tqdm \
        transformers==4.57.1 uvicorn uvloop xgrammar==0.1.27 \
        "smg-grpc-proto>=0.3.3" "grpcio>=1.78.0" \
        "grpcio-reflection>=1.78.0" "grpcio-health-checking>=1.78.0"

EXPOSE 8001
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
