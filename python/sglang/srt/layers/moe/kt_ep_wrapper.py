# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import copy
import ctypes
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.utils import get_compiler_backend, is_cuda

if is_cuda():
    from sgl_kernel import gptq_marlin_repack

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global cache for GPU experts masks (initialized once per session)
_KT_GPU_EXPERTS_MASKS: Optional[torch.Tensor] = None


def _kt_timing_enabled() -> bool:
    return os.getenv("SGLANG_KT_TIMING") in ("1", "true", "TRUE")


def _kt_timing_sync_enabled() -> bool:
    return os.getenv("SGLANG_KT_TIMING_SYNC") in ("1", "true", "TRUE")


def _kt_timing_sync_allowed() -> bool:
    if not _kt_timing_sync_enabled():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return not torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _kt_hitmiss_timing_enabled() -> bool:
    return os.getenv("SGLANG_KT_HITMISS_TIMING") in ("1", "true", "TRUE")


def _kt_timing_layer_enabled(layer_idx: int) -> bool:
    layers = os.getenv("SGLANG_KT_TIMING_LAYERS")
    if not layers:
        return True
    wanted = set()
    for raw in layers.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw:
            start, end = raw.split("-", 1)
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(raw))
    return int(layer_idx) in wanted


def _kt_full_gpu_fallback_min_free_bytes() -> int:
    raw = os.getenv("SGLANG_KT_FULL_GPU_FALLBACK_MIN_FREE_GB", "2.5")
    try:
        value_gb = float(raw)
    except ValueError:
        logger.warning(
            "Invalid SGLANG_KT_FULL_GPU_FALLBACK_MIN_FREE_GB=%r; using 2.5",
            raw,
        )
        value_gb = 2.5
    return int(max(value_gb, 0.0) * 1024**3)


def _looks_like_cuda_oom(err: Exception) -> bool:
    if isinstance(err, torch.OutOfMemoryError):
        return True
    text = str(err).lower()
    return (
        "out of memory" in text
        or "cuda calloc" in text
        or "ncclunhandledcudaerror" in text
        or "failed to cuda" in text
    )


def _kt_log_timing(method, phase: str, start_time: Optional[float] = None, **fields):
    if not _kt_timing_enabled() or not _kt_timing_layer_enabled(method.kt_config.layer_idx):
        return
    elapsed_ms = None
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    parts = [
        f"phase={phase}",
        f"layer={method.kt_config.layer_idx}",
        f"pp={getattr(getattr(method, 'pp_group', None), 'rank', None)}",
        f"tp={getattr(method, 'tp_rank', None)}",
        f"active={getattr(method, '_is_kt_active_rank', None)}",
        f"gpu_experts={getattr(method, 'num_gpu_experts', None)}",
        f"cpu_experts={getattr(method, 'global_num_experts', 0) - getattr(method, 'num_gpu_experts', 0)}",
    ]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.3f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info("KT timing %s", " ".join(parts))


def _kt_log_hitmiss(
    method,
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    num_tokens: int,
) -> None:
    if not _kt_hitmiss_timing_enabled():
        return
    if not _kt_timing_enabled() or not _kt_timing_layer_enabled(method.kt_config.layer_idx):
        return
    if not getattr(method, "_is_kt_active_rank", False):
        return
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return

    t_hitmiss = time.perf_counter()
    with torch.no_grad():
        is_gpu_expert = gpu_experts_mask[topk_ids]
        total_choices = int(topk_ids.numel())
        gpu_choices = int(is_gpu_expert.sum().item())
        cpu_choices = total_choices - gpu_choices
        unique_gpu = (
            int(topk_ids[is_gpu_expert].unique().numel())
            if gpu_choices > 0
            else 0
        )
        unique_cpu = (
            int(topk_ids[~is_gpu_expert].unique().numel())
            if cpu_choices > 0
            else 0
        )

    cpu_pct = (100.0 * cpu_choices / total_choices) if total_choices else 0.0
    _kt_log_timing(
        method,
        "kt.hitmiss",
        t_hitmiss,
        tokens=num_tokens,
        topk_shape=tuple(topk_ids.shape),
        total_choices=total_choices,
        gpu_choices=gpu_choices,
        cpu_choices=cpu_choices,
        cpu_pct=f"{cpu_pct:.1f}",
        unique_gpu=unique_gpu,
        unique_cpu=unique_cpu,
    )


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        gpu_experts_mask: Boolean tensor of shape [num_experts] indicating which experts are on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        numa_nodes: Optional explicit NUMA node ids for each KT threadpool
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        gpu_prefill_token_threshold: token threshold for enabling full GPU fallback
        kt_enable_dynamic_expert_update: Enable dynamic GPU expert updates based on runtime statistics
    """

    layer_idx: int
    gpu_experts_mask: torch.Tensor  # bool tensor of shape [num_experts]
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    numa_nodes: Optional[List[int]] = None
    num_layers: Optional[int] = None
    pp_end_layer: Optional[int] = None
    gpu_prefill_token_threshold: Optional[int] = None
    kt_enable_dynamic_expert_update: bool = False


_SHARED_FULL_CONTEXT = None
_SHARED_STAGING_BUFFER = None  # Global shared staging buffer for all MoE layers
_KT_FULL_GPU_FALLBACK_DISABLED = False
_KT_STAGING_PROBE_METHODS = []
_KT_STAGING_PROBE_CURSOR = 0
_KT_STAGING_PROBE_WEIGHT_NAMES_WNA16 = [
    "w13_qweight",
    "w13_scales",
    "w2_qweight",
    "w2_scales",
]


def _kt_staging_probe_enabled() -> bool:
    return os.getenv("SGLANG_KT_STAGING_PROBE") in ("1", "true", "TRUE")


def _kt_staging_swap_enabled() -> bool:
    return os.getenv("SGLANG_KT_STAGING_SWAP") in ("1", "true", "TRUE")


def _kt_staging_swap_limit() -> int:
    raw = os.getenv("SGLANG_KT_STAGING_SWAP_LIMIT", "1")
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid SGLANG_KT_STAGING_SWAP_LIMIT=%r; using 1", raw)
        return 1


def _parse_int_set(raw: Optional[str]) -> Optional[set]:
    if raw is None or not raw.strip():
        return None
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def _kt_staging_probe_layers() -> Optional[set]:
    return _parse_int_set(os.getenv("SGLANG_KT_STAGING_PROBE_LAYERS"))


def _kt_staging_probe_log(phase: str, **fields) -> None:
    if not _kt_staging_probe_enabled():
        return
    parts = [f"phase={phase}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info("KT staging probe %s", " ".join(parts))


def _register_kt_staging_probe_method(method) -> None:
    if method not in _KT_STAGING_PROBE_METHODS:
        _KT_STAGING_PROBE_METHODS.append(method)


def run_kt_staging_probe_once(
    device: Optional[torch.device] = None,
    batch_size: int = 0,
    forward_mode: str = "",
) -> Optional[dict]:
    """Run one non-mutating KT expert staging probe step.

    This is called outside CUDA graph replay by the scheduler. It pipelines one
    CPU expert write from the previous step into a scratch H2D copy for this
    step, then submits the next CPU expert write. It never mutates resident
    expert weights or routing maps.
    """
    global _KT_STAGING_PROBE_CURSOR

    if not _kt_staging_probe_enabled() or not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.is_current_stream_capturing():
            return None
    except Exception:
        return None

    wanted_layers = _kt_staging_probe_layers()
    eligible = []
    for method in _KT_STAGING_PROBE_METHODS:
        if wanted_layers is not None and method.kt_config.layer_idx not in wanted_layers:
            continue
        if method._kt_staging_probe_is_eligible(device):
            eligible.append(method)

    if not eligible:
        _kt_staging_probe_log("no_eligible_method", forward_mode=forward_mode)
        return None

    method = eligible[_KT_STAGING_PROBE_CURSOR % len(eligible)]
    _KT_STAGING_PROBE_CURSOR += 1
    return method._kt_staging_probe_step(
        batch_size=batch_size,
        forward_mode=forward_mode,
    )


def finish_kt_staging_probe(probe: Optional[dict]) -> None:
    if probe is None:
        return
    wait_t = time.perf_counter()
    probe["done_event"].synchronize()
    wait_ms = (time.perf_counter() - wait_t) * 1000.0
    try:
        copy_ms = probe["start_event"].elapsed_time(probe["end_event"])
    except Exception:
        copy_ms = None
    total_ms = (time.perf_counter() - probe["start_t"]) * 1000.0
    hidden_ms = max(0.0, total_ms - wait_ms)
    _kt_staging_probe_log(
        "h2d_finish",
        layer=probe["layer"],
        expert=probe["expert"],
        slot=probe["slot"],
        mb=f"{probe['bytes'] / 1024**2:.3f}",
        wait_ms=f"{wait_ms:.3f}",
        hidden_ms=f"{hidden_ms:.3f}",
        total_ms=f"{total_ms:.3f}",
        copy_ms=f"{copy_ms:.3f}" if copy_ms is not None else None,
    )
    method = probe.get("method")
    if method is not None:
        method._kt_staging_probe_maybe_swap(probe)


class SharedStagingBuffer:
    """Global shared staging buffer for CPU expert input across all MoE layers.

    This avoids allocating a separate staging buffer per layer, which would
    consume significant GPU memory (chunked_prefill_size * hidden_size * N_layers).
    Instead, all layers share a single buffer since MoE layers are processed
    sequentially, not in parallel.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        buffer_size_mb = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            f"[KT] Created shared staging buffer: {buffer_size_mb:.1f} MiB "
            f"(shape={self.buffer.shape}, dtype={dtype})"
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        """Get a slice of the buffer for the given number of tokens."""
        assert num_tokens <= self.max_tokens, (
            f"Batch size {num_tokens} exceeds staging buffer max size {self.max_tokens}"
        )
        return self.buffer[:num_tokens]


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the global shared staging buffer."""
    global _SHARED_STAGING_BUFFER
    if _SHARED_STAGING_BUFFER is None:
        _SHARED_STAGING_BUFFER = SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
    return _SHARED_STAGING_BUFFER


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        init_args: tuple,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
    ):
        self._build_layers(layer, init_args, global_num_experts, moe_runner_config)

        # Capture original tensors to support restoration before loading
        self.original_params = {
            name: param for name, param in self.gpu_layer.named_parameters()
        }
        self.original_buffers = {
            name: buf for name, buf in self.gpu_layer.named_buffers()
        }

        # Create CPU buffers once for weight loading (shared across layers)
        self._create_cpu_buffers()

    def _build_layers(self, layer, init_args, global_num_experts, moe_runner_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        hidden_size, intermediate_size_per_partition, params_dtype = init_args
        target_device = next(layer.parameters()).device

        # Create gpu_layer as a shallow copy, then override specific attributes
        self.gpu_layer = copy.copy(layer)
        # Clear module state that shouldn't be shared
        self.gpu_layer._parameters = {}
        self.gpu_layer._buffers = {}
        self.gpu_layer._modules = {}

        # Override expert counts for full GPU execution
        self.gpu_layer.num_experts = global_num_experts
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.num_gpu_experts = global_num_experts

        # Create quant_method for gpu_layer
        if self.gpu_layer.quant_config is not None:
            self.gpu_method = self.gpu_layer.quant_config.get_quant_method(
                self.gpu_layer, prefix=""
            )
        else:
            self.gpu_method = UnquantizedFusedMoEMethod(
                self.gpu_layer.use_triton_kernels
            )
        self.gpu_layer.quant_method = self.gpu_method

        self.gpu_method.create_weights(
            layer=self.gpu_layer,
            num_experts=global_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.gpu_layer.weight_loader,
        )

        # Detect quantization type for weight loading based on actually created weights.
        # This is more robust than class-based detection when quant methods are wrapped
        # (e.g., KT wrapper -> compressed-tensors scheme), especially in layerwise prefill.
        self._detect_quant_type_from_created_weights()

        # Move all parameters to target device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)

        # Create runner config - update both num_experts and num_local_experts for full GPU fallback
        # Set routed_scaling_factor=None to avoid double scaling:
        # - moe_sum_reduce would apply routed_scaling_factor internally
        # - deepseek_v2.py forward_normal also applies routed_scaling_factor for KTEPWrapperMethod
        # By setting it to None here, we ensure it's only applied once in forward_normal
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
            routed_scaling_factor=None,
        )
        self.gpu_layer.moe_runner_config = runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, runner_config)

    def _get_base_quant_method(self):
        """Unwrap nested quant methods to get the underlying base method.

        Some paths may wrap the real quant method with KT wrappers/schemes.
        """
        method = self.gpu_method
        visited = set()

        while method is not None and id(method) not in visited:
            visited.add(id(method))

            # KT wrapper pattern: method.gpu_method
            nested = getattr(method, "gpu_method", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            # Compressed-tensors scheme pattern: method.scheme
            nested = getattr(method, "scheme", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            break

        return method

    def _detect_quant_type_from_created_weights(self) -> None:
        """Detect quant type from weight attributes created on gpu_layer."""
        layer = self.gpu_layer
        self.is_wna16_quant = False

        # AutoRound/GPTQ/AWQ WNA16 MoE path. This layout uses qweight/scales
        # directly instead of the compressed-tensors Marlin packed names.
        if hasattr(layer, "w13_qweight") and hasattr(layer, "w2_qweight"):
            self.is_wna16_quant = True
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # INT4 Marlin
        if hasattr(layer, "w13_weight_packed") and hasattr(layer, "w2_weight_packed"):
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 block
        if hasattr(layer, "w13_weight_scale_inv") and hasattr(layer, "w2_weight_scale_inv"):
            self.is_fp8_quant = True
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 per-channel
        if hasattr(layer, "w13_weight_scale") and hasattr(layer, "w2_weight_scale"):
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = True
            self.is_bf16_quant = False
            return

        # BF16 / unquantized
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = True
            return

        # Fallback to class-based detection for unknown layouts.
        self.is_fp8_quant = self._detect_fp8_quant()
        self.is_fp8_channel_quant = self._detect_fp8_channel_quant()
        self.is_bf16_quant = self._detect_bf16_quant()

    def _detect_fp8_quant(self) -> bool:
        """Detect if the quantization method is FP8 block quant.

        Returns:
            True if FP8 block quant, False otherwise (INT4 Marlin, BF16, etc.)
        """
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = self._get_base_quant_method()
        # Check for Fp8MoEMethod with block_quant
        if isinstance(method, Fp8MoEMethod) and getattr(method, "block_quant", False):
            return True

        # Check for CompressedTensorsW8A8Fp8MoEMethod with block_quant
        method_name = method.__class__.__name__
        if "W8A8Fp8" in method_name and getattr(method, "block_quant", False):
            return True

        return False

    def _detect_fp8_channel_quant(self) -> bool:
        """Detect if the quantization method is FP8 per-channel quant.

        Per-channel FP8 differs from block FP8:
        - Per-channel: scale shape is (num_experts, output_dim, 1), weight_scale name
        - Block FP8: scale shape is (num_experts, blocks_n, blocks_k), weight_scale_inv name

        Returns:
            True if FP8 per-channel quant, False otherwise
        """
        try:
            from compressed_tensors.quantization import QuantizationStrategy
        except ImportError:
            return False

        method = self._get_base_quant_method()
        method_name = method.__class__.__name__

        # Check for CompressedTensorsW8A8Fp8MoEMethod with channel strategy
        if "W8A8Fp8" in method_name:
            weight_quant = getattr(method, "weight_quant", None)
            if weight_quant is not None:
                if weight_quant.strategy == QuantizationStrategy.CHANNEL:
                    return True

        return False

    def _detect_bf16_quant(self) -> bool:
        """Detect if the quantization method is BF16/unquantized.

        Returns:
            True if BF16/unquantized, False otherwise (INT4 Marlin, FP8, etc.)
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        method = self._get_base_quant_method()
        # Check for UnquantizedFusedMoEMethod
        if isinstance(method, UnquantizedFusedMoEMethod):
            return True

        return False

    def _resolve_int4_quant_params(self):
        """Resolve INT4 quant params from potentially wrapped quant methods.

        Some quantization paths (e.g., compressed-tensors) expose INT4 metadata on
        the underlying scheme instead of the outer fused method wrapper.
        """
        candidates = []
        seen = set()

        def add_candidate(obj):
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            candidates.append(obj)

        base_method = self._get_base_quant_method()
        add_candidate(self.gpu_method)
        add_candidate(getattr(self.gpu_method, "gpu_method", None))
        add_candidate(getattr(self.gpu_method, "scheme", None))
        add_candidate(base_method)
        add_candidate(getattr(base_method, "scheme", None))
        add_candidate(getattr(self.gpu_layer, "scheme", None))

        required = ("num_bits", "packed_factor", "group_size")
        for candidate in candidates:
            if all(hasattr(candidate, attr) for attr in required):
                return (
                    getattr(candidate, "num_bits"),
                    getattr(candidate, "packed_factor"),
                    getattr(candidate, "group_size"),
                    getattr(candidate, "actorder", None),
                )

        raise AttributeError(
            "Unable to resolve INT4 quantization params: expected attributes "
            "num_bits/packed_factor/group_size on quant method or scheme"
        )

    @property
    def weight_names(self) -> list:
        """Get weight names based on quantization type."""
        if getattr(self, "is_wna16_quant", False):
            return self.WEIGHT_NAMES_WNA16
        elif self.is_fp8_quant:
            return self.WEIGHT_NAMES_FP8
        elif self.is_fp8_channel_quant:
            return self.WEIGHT_NAMES_FP8_CHANNEL
        elif self.is_bf16_quant:
            return self.WEIGHT_NAMES_BF16
        else:
            return self.WEIGHT_NAMES_INT4

    # Weight names for shared memory buffers (AutoRound/GPTQ/AWQ WNA16 format)
    WEIGHT_NAMES_WNA16 = [
        "w13_qweight",
        "w13_scales",
        "w2_qweight",
        "w2_scales",
    ]

    # Weight names for shared memory buffers (INT4 Marlin format)
    WEIGHT_NAMES_INT4 = [
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    ]

    # Weight names for FP8 block quant format
    WEIGHT_NAMES_FP8 = [
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ]

    # Weight names for FP8 per-channel quant format
    # Per-channel differs from block quant:
    # - Scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
    # - Weight name: w13_weight_scale vs w13_weight_scale_inv
    WEIGHT_NAMES_FP8_CHANNEL = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for BF16/unquantized format (no scales)
    WEIGHT_NAMES_BF16 = [
        "w13_weight",
        "w2_weight",
    ]

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses double buffering (2 experts) to reduce memory usage while maintaining
        pipeline efficiency: write(e+1) || copy(e) only needs 2 buffers.
        """
        # Set NUMA local allocation policy to allocate on local NUMA node
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            raise RuntimeError("NUMA is not available on this system")
        libnuma.numa_set_localalloc()

        self.cpu_buffers = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}
        tp_rank = get_tensor_model_parallel_rank()
        num_experts = self.gpu_layer.num_experts

        # Generate unique ID on rank 0 and broadcast to all ranks
        if tp_rank == 0:
            self.shm_unique_id = uuid.uuid4().hex[:8]
        else:
            self.shm_unique_id = None
        if dist.is_initialized():
            unique_id_list = [self.shm_unique_id]
            _group = get_tp_group().cpu_group
            # Under PP, each TP group may not contain global rank 0.
            # PyTorch expects the source as a global rank in the group.
            _src = dist.get_global_rank(_group, 0)
            dist.broadcast_object_list(unique_id_list, src=_src, group=_group)
            self.shm_unique_id = unique_id_list[0]

        for name in self.weight_names:
            gpu_tensor = getattr(self.gpu_layer, name)
            # Only allocate 2 experts worth of buffer (double buffering)
            expert_shape = gpu_tensor.shape[1:]  # Shape per expert
            expert_nbytes = (
                gpu_tensor.numel() // num_experts * gpu_tensor.element_size()
            )
            double_buf_nbytes = expert_nbytes * 2

            shm_name = f"kt_buf_{name}_r{tp_rank}_{self.shm_unique_id}"
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=double_buf_nbytes
            )
            self.shm_handles[name] = shm

            # Shape: [2, ...expert_shape...]
            cpu_buffer = torch.frombuffer(shm.buf, dtype=gpu_tensor.dtype).reshape(
                (2,) + expert_shape
            )

            # Register as pinned memory for fast DMA
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostRegister(
                    cpu_buffer.data_ptr(), double_buf_nbytes, 0
                )

            self.cpu_buffers[name] = cpu_buffer

        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()

        # Unlink shared memory after all ranks have collected pointers.
        # The memory remains accessible as long as we hold references via mmap.
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)
        for shm in self.shm_handles.values():
            shm.unlink()

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    ptr = self.cpu_buffers[name].data_ptr()
                elif tp_rank == 0:
                    shm_name = f"kt_buf_{name}_r{rank}_{self.shm_unique_id}"
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self._opened_shm_refs[f"{name}_r{rank}"] = shm
                        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    except FileNotFoundError:
                        logger.error(
                            "Rank %d: Failed to open shared memory '%s'",
                            tp_rank,
                            shm_name,
                        )
                        ptr = 0
                else:
                    ptr = 0
                all_rank_ptrs[name].append(ptr)

        return all_rank_ptrs

    def _prepare_weight_int4(self, wrapper):
        """Prepare INT4 Marlin weights by writing from KT, copying to GPU, and postprocessing.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        Postprocessing extracted from CompressedTensorsWNA16MoEMethod.process_weights_after_loading
        in python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_bits, packed_factor, group_size, actorder = (
            self._resolve_int4_quant_params()
        )
        num_experts = layer.num_experts
        device = layer.w13_weight_packed.device

        # Create empty g_idx tensors for non-grouped actorder
        if actorder != "group":
            for name in [
                "w13_weight_g_idx",
                "w2_weight_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
            ]:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                        requires_grad=False,
                    ),
                )

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_INT4:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            # Reshape gpu_t to match expert shape for per-expert copy
            expert_shape = cpu_buf.shape[1:]
            gpu_t.set_(gpu_t.view((num_experts,) + expert_shape))
            weight_infos.append((cpu_buf, gpu_t))

        w13_p, w13_s = layer.w13_weight_packed, layer.w13_weight_scale
        w2_p, w2_s = layer.w2_weight_packed, layer.w2_weight_scale
        w13_k, w13_n = w13_p.shape[1] * packed_factor, w13_p.shape[2]
        w2_k, w2_n = w2_p.shape[1] * packed_factor, w2_p.shape[2]
        w2_sk = w2_s.shape[1] * (group_size if group_size != -1 else packed_factor)
        perm = torch.empty(0, dtype=torch.int32, device=device)

        # Tmp buffers for transpose
        tmp_bufs = [
            torch.empty(t.size(1), t.size(2), dtype=t.dtype, device=device)
            for _, t in weight_infos
        ]

        def postprocess_expert(e):
            # Transpose
            for (_, gpu_t), tmp in zip(weight_infos, tmp_bufs):
                d1, d2 = gpu_t.size(1), gpu_t.size(2)
                tmp.copy_(gpu_t[e].reshape(d2, d1).T, non_blocking=True)
                gpu_t[e].copy_(tmp, non_blocking=True)
            # Repack weights
            w13_p[e].copy_(
                gptq_marlin_repack(w13_p[e], perm, w13_k, w13_n, num_bits).view(
                    w13_p[e].shape
                )
            )
            w2_p[e].copy_(
                gptq_marlin_repack(w2_p[e], perm, w2_k, w2_n, num_bits).view(
                    w2_p[e].shape
                )
            )
            # Permute scales
            w13_s[e].copy_(
                marlin_permute_scales(w13_s[e], w13_n, w13_s.shape[2], group_size).view(
                    w13_s[e].shape
                )
            )
            w2_s[e].copy_(
                marlin_permute_scales(w2_s[e], w2_sk, w2_s.shape[2], group_size).view(
                    w2_s[e].shape
                )
            )

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_packed_buf = self.cpu_buffers["w13_weight_packed"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_packed_buf = self.cpu_buffers["w2_weight_packed"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_packed_expert_nbytes = (
                w13_packed_buf.numel() // 2 * w13_packed_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_packed_expert_nbytes = (
                w2_packed_buf.numel() // 2 * w2_packed_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_packed_ptrs = [
                    ptr + slot * w13_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_packed"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_packed_ptrs = [
                    ptr + slot * w2_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_packed"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_packed_ptrs,
                    w13_scale_ptrs,
                    w2_packed_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, make sure the previous
                    # copy from that slot has completed to avoid overwriting
                    # pinned host memory while DMA is in-flight.
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

        # Reshape to final shape
        w13_p.set_(w13_p.view(num_experts, w13_k // 16, w13_n * (num_bits // 2)))
        w2_p.set_(w2_p.view(num_experts, w2_k // 16, w2_n * (num_bits // 2)))

    def _prepare_weight_fp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                            logical_to_gpu_index=None):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 block quant is simpler than INT4 Marlin:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optional DeepGemm ue8m0 conversion is handled after all experts are loaded.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_fp8_channel(self, wrapper, original_layer=None, gpu_experts_mask=None,
                                     logical_to_gpu_index=None):
        """Prepare FP8 per-channel quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 per-channel quant differs from FP8 block quant:
        - Per-channel scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
        - Weight name: w13_weight_scale vs w13_weight_scale_inv
        - Both use float8_e4m3fn weights

        Similar to block FP8:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8_CHANNEL:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 per-channel doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_wna16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Prepare AutoRound/GPTQ/AWQ WNA16 weights for full-GPU fallback.

        WNA16 MoE weights are already in the qweight/scales layout consumed by
        MoeWNA16Method.apply(), so unlike Marlin INT4 they do not need transpose
        or repack postprocessing after staging from KT.
        """
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_qweight.device

        for optional_name in ("w13_qzeros", "w2_qzeros"):
            optional = getattr(layer, optional_name, None)
            if optional is not None and optional.numel() != 0:
                raise NotImplementedError(
                    "KT full-GPU WNA16 fallback only supports symmetric quantization "
                    f"without zero-points; found non-empty {optional_name}"
                )

        weight_infos = []
        for name in self.WEIGHT_NAMES_WNA16:
            cpu_buf = self.cpu_buffers[name]
            gpu_t = getattr(layer, name)
            weight_infos.append((name, cpu_buf, gpu_t))

        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            cpu_expert_ids = list(range(num_experts))

        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        if not cpu_expert_ids:
            return

        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            w13_weight_buf = self.cpu_buffers["w13_qweight"]
            w13_scale_buf = self.cpu_buffers["w13_scales"]
            w2_weight_buf = self.cpu_buffers["w2_qweight"]
            w2_scale_buf = self.cpu_buffers["w2_scales"]

            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_qweight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_scales"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_qweight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_scales"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2

            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])

        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                             logical_to_gpu_index=None):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_BF16:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # BF16 doesn't need actual postprocessing (no repack/permute/transpose).
            # This function provides a pipeline synchronization point and
            # can be extended for future BF16-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                # For BF16, we pass empty scale pointer lists (no scales)
                w13_scale_ptrs = [0] * tp_world_size
                w2_scale_ptrs = [0] * tp_world_size
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def load(self, layer_idx, wrapper, original_layer=None, gpu_experts_mask=None,
             logical_to_gpu_index=None):
        """Load weights from disk to GPU via shared memory.

        Args:
            layer_idx: Layer index in the model
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts (optional)
            gpu_experts_mask: bool tensor [num_experts], True = on GPU (optional)
            logical_to_gpu_index: int tensor [num_experts], maps logical ID to GPU index (optional)
        """
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        t0 = time.perf_counter()

        # Select appropriate prepare_weight method based on quantization type
        # FP8/BF16/WNA16 methods support GPU expert optimization; INT4 uses full CPU pipeline
        if getattr(self, "is_wna16_quant", False):
            self._prepare_weight_wna16(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif self.is_fp8_quant:
            self._prepare_weight_fp8(wrapper, original_layer, gpu_experts_mask,
                                     logical_to_gpu_index)
        elif self.is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper, original_layer, gpu_experts_mask,
                                             logical_to_gpu_index)
        elif self.is_bf16_quant:
            self._prepare_weight_bf16(wrapper, original_layer, gpu_experts_mask,
                                      logical_to_gpu_index)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT layerwise prefill: layer %d prepare weight = %.2f ms",
                layer_idx,
                total_time,
            )


def generate_front_loading_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks by filling layers from first MoE layer onwards.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers (e.g., 1 = every layer, 2 = every other layer)

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    remaining = num_gpu_experts

    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if not is_moe:
            # Dense layer - set all True (bypass KT wrapper)
            masks[layer_idx, :] = True
        elif remaining > 0:
            # MoE layer - allocate GPU experts
            num_for_this_layer = min(remaining, num_experts)
            masks[layer_idx, :num_for_this_layer] = True
            remaining -= num_for_this_layer

    return masks


def generate_uniform_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks with equal GPU experts per MoE layer.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Identify MoE layers
    moe_layers = [
        i for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    num_moe_layers = len(moe_layers)

    if num_moe_layers == 0:
        return masks

    # Distribute GPU experts evenly
    experts_per_layer = num_gpu_experts // num_moe_layers
    remainder = num_gpu_experts % num_moe_layers

    for idx, layer_idx in enumerate(moe_layers):
        # First 'remainder' layers get one extra expert
        num_for_this_layer = experts_per_layer + (1 if idx < remainder else 0)
        num_for_this_layer = min(num_for_this_layer, num_experts)
        masks[layer_idx, :num_for_this_layer] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def generate_frequency_uniform_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate frequency masks while preserving uniform per-layer counts."""
    num_layers, num_experts = activation_freq.shape
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    moe_layers = [
        i
        for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    num_moe_layers = len(moe_layers)
    if num_moe_layers == 0:
        return masks

    experts_per_layer = num_gpu_experts // num_moe_layers
    remainder = num_gpu_experts % num_moe_layers

    for idx, layer_idx in enumerate(moe_layers):
        num_for_this_layer = experts_per_layer + (1 if idx < remainder else 0)
        num_for_this_layer = min(num_for_this_layer, num_experts)
        if num_for_this_layer <= 0:
            continue
        selected = torch.argsort(
            activation_freq[layer_idx], descending=True, stable=True
        )[:num_for_this_layer]
        masks[layer_idx, selected] = True

    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def _moe_layer_indices(
    num_layers: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> List[int]:
    return [
        i
        for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]


def _set_non_moe_layers_to_gpu(
    masks: torch.Tensor,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> None:
    for layer_idx in range(masks.shape[0]):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True


def _parse_int_env_list(name: str) -> Optional[List[int]]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None

    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as err:
            raise ValueError(f"Invalid integer in {name}={raw!r}: {item!r}") from err
        if value < 0:
            raise ValueError(f"{name} must not contain negative counts: {raw!r}")
        values.append(value)

    if not values:
        raise ValueError(f"{name} was set but did not contain any counts")
    return values


def _get_pp_layer_ranges(num_layers: int, pp_size: int) -> List[Tuple[int, int]]:
    from sglang.srt.distributed import get_pp_indices

    return [get_pp_indices(num_layers, pp_rank, pp_size) for pp_rank in range(pp_size)]


def _infer_pp_size_for_kt_layer_budgets() -> int:
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION")
    if partition_list_str:
        return len([p for p in partition_list_str.split(",") if p.strip()])

    try:
        from sglang.srt.distributed import get_pp_group

        return get_pp_group().world_size
    except Exception:
        return 1


def _resolve_kt_gpu_expert_layer_counts(
    num_layers: int,
    num_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> Optional[List[int]]:
    """Return explicit per-layer GPU expert counts from env, if configured.

    These envs are intentionally outside ServerArgs so we can tune asymmetric
    local deployments without adding another public serving flag. They only
    change the initial GPU expert mask; dynamic expert update still keeps each
    layer's current slot count by reading mask.sum() from the layer wrapper.
    """

    by_layer = _parse_int_env_list("SGLANG_KT_NUM_GPU_EXPERTS_BY_LAYER")
    by_pp = _parse_int_env_list("SGLANG_KT_NUM_GPU_EXPERTS_BY_PP")

    if by_layer is not None and by_pp is not None:
        raise ValueError(
            "Set only one of SGLANG_KT_NUM_GPU_EXPERTS_BY_LAYER or "
            "SGLANG_KT_NUM_GPU_EXPERTS_BY_PP"
        )
    if by_layer is None and by_pp is None:
        return None

    moe_layers = _moe_layer_indices(
        num_layers, first_k_dense_replace, moe_layer_freq
    )
    counts = [num_experts] * num_layers

    if by_layer is not None:
        if len(by_layer) == num_layers:
            for layer_idx, count in enumerate(by_layer):
                counts[layer_idx] = min(count, num_experts)
        elif len(by_layer) == len(moe_layers):
            for layer_idx, count in zip(moe_layers, by_layer):
                counts[layer_idx] = min(count, num_experts)
        else:
            raise ValueError(
                "SGLANG_KT_NUM_GPU_EXPERTS_BY_LAYER must have either "
                f"{num_layers} entries (one per layer) or {len(moe_layers)} "
                f"entries (one per MoE layer); got {len(by_layer)}"
            )
    else:
        pp_size = _infer_pp_size_for_kt_layer_budgets()
        if len(by_pp) != pp_size:
            raise ValueError(
                "SGLANG_KT_NUM_GPU_EXPERTS_BY_PP entry count must match PP "
                f"world size ({pp_size}); got {len(by_pp)}"
            )
        for pp_rank, (start_layer, end_layer) in enumerate(
            _get_pp_layer_ranges(num_layers, pp_size)
        ):
            pp_count = min(by_pp[pp_rank], num_experts)
            for layer_idx in range(start_layer, end_layer):
                counts[layer_idx] = pp_count

    return counts


def generate_layer_count_masks(
    num_layers: int,
    num_experts: int,
    per_layer_gpu_experts: Sequence[int],
    first_k_dense_replace: int,
    moe_layer_freq: int,
    *,
    activation_freq: Optional[torch.Tensor] = None,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate masks from explicit per-layer GPU expert counts."""

    if len(per_layer_gpu_experts) != num_layers:
        raise ValueError(
            f"Expected {num_layers} per-layer expert counts, "
            f"got {len(per_layer_gpu_experts)}"
        )

    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    rng = None
    if random_seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(random_seed)

    for layer_idx in _moe_layer_indices(
        num_layers, first_k_dense_replace, moe_layer_freq
    ):
        num_for_this_layer = min(
            max(int(per_layer_gpu_experts[layer_idx]), 0), num_experts
        )
        if num_for_this_layer <= 0:
            continue

        if activation_freq is not None:
            selected = torch.argsort(
                activation_freq[layer_idx], descending=True, stable=True
            )[:num_for_this_layer]
        elif rng is not None:
            selected = torch.randperm(num_experts, generator=rng, device="cpu")[
                :num_for_this_layer
            ]
        else:
            selected = torch.arange(num_for_this_layer, device="cpu")
        masks[layer_idx, selected] = True

    _set_non_moe_layers_to_gpu(masks, first_k_dense_replace, moe_layer_freq)
    return masks


def generate_random_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate masks by randomly selecting GPU experts (fixed seed).

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers
        seed: Random seed for reproducibility

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Collect all MoE (layer, expert) positions
    moe_positions = []
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if is_moe:
            for expert_idx in range(num_experts):
                moe_positions.append((layer_idx, expert_idx))

    # Randomly select positions
    if len(moe_positions) > 0:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        num_to_select = min(num_gpu_experts, len(moe_positions))
        selected_indices = torch.randperm(len(moe_positions), generator=rng, device='cpu')[:num_to_select]

        for idx in selected_indices:
            layer_idx, expert_idx = moe_positions[idx]
            masks[layer_idx, expert_idx] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def _init_kt_gpu_experts_masks(server_args: "ServerArgs") -> Optional[torch.Tensor]:
    """Initialize GPU experts masks from activation frequency data.

    Args:
        server_args: Global server arguments

    Returns:
        Masks tensor of shape [num_layers, num_experts], or None if KT not configured
    """
    global _KT_GPU_EXPERTS_MASKS

    if _KT_GPU_EXPERTS_MASKS is not None:
        return _KT_GPU_EXPERTS_MASKS

    # Get model config (unwrap VL configs that nest the text model config)
    hf_config = server_args.get_hf_config()

    # fix for kimi-k2.5 models where text_config holds the actual config
    if getattr(hf_config, "text_config", None) is not None:
        hf_config = hf_config.text_config

    num_layers = getattr(hf_config, "num_hidden_layers", None)
    # Try different attribute names for num_experts
    num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", None)

    if num_layers is None or num_experts is None:
        logger.warning(
            "Could not determine num_layers or num_experts from model config."
        )
        return None

    # Get first_k_dense_replace to identify which layers are MoE layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)

    # Normalize list-form moe_layer_freq (e.g., MiMo-V2-Flash: [0, 1, 1, ...])
    # to standard (first_k_dense_replace, moe_layer_freq=1) form
    if isinstance(moe_layer_freq, list):
        # Find first MoE layer index from the mask
        first_moe = next((i for i, v in enumerate(moe_layer_freq) if v), 0)
        first_k_dense_replace = max(first_k_dense_replace or 0, first_moe)
        moe_layer_freq = 1

    # Count actual MoE layers
    moe_layers = _moe_layer_indices(
        num_layers, first_k_dense_replace, moe_layer_freq
    )
    num_moe_layers = len(moe_layers)
    total_experts = num_moe_layers * num_experts
    explicit_layer_gpu_experts = _resolve_kt_gpu_expert_layer_counts(
        num_layers, num_experts, first_k_dense_replace, moe_layer_freq
    )

    # Determine num_gpu_experts (total across all layers)
    if explicit_layer_gpu_experts is not None:
        num_gpu_experts = sum(explicit_layer_gpu_experts[i] for i in moe_layers)
        logger.info(
            "Using explicit KT per-layer GPU expert budgets from env, "
            "total GPU experts: %d (= %d MoE layers with counts %s)",
            num_gpu_experts,
            num_moe_layers,
            sorted({explicit_layer_gpu_experts[i] for i in moe_layers}),
        )
        if (
            server_args.kt_gpu_experts_ratio is not None
            or server_args.kt_num_gpu_experts is not None
        ):
            logger.warning(
                "Explicit KT per-layer GPU expert budgets override "
                "--kt-gpu-experts-ratio=%s and --kt-num-gpu-experts=%s",
                server_args.kt_gpu_experts_ratio,
                server_args.kt_num_gpu_experts,
            )
    elif server_args.kt_gpu_experts_ratio is not None:
        # Use ratio to calculate total GPU experts
        num_gpu_experts = int(total_experts * server_args.kt_gpu_experts_ratio)
        if server_args.kt_num_gpu_experts is not None:
            logger.warning(
                f"--kt-gpu-experts-ratio={server_args.kt_gpu_experts_ratio} is set, "
                f"ignoring --kt-num-gpu-experts={server_args.kt_num_gpu_experts}. "
                f"Actual total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
        else:
            logger.info(
                f"Using kt_gpu_experts_ratio={server_args.kt_gpu_experts_ratio}, "
                f"total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
    elif server_args.kt_num_gpu_experts is not None:
        # kt_num_gpu_experts is per-layer, multiply by num_moe_layers
        num_gpu_experts = server_args.kt_num_gpu_experts * num_moe_layers
        logger.info(
            f"Using kt_num_gpu_experts={server_args.kt_num_gpu_experts} per layer, "
            f"total GPU experts: {num_gpu_experts} "
            f"(= {server_args.kt_num_gpu_experts} × {num_moe_layers} MoE layers)"
        )
    else:
        logger.warning("Either kt_num_gpu_experts or kt_gpu_experts_ratio is required but not set.")
        return None

    # Get GPU expert placement strategy
    strategy = server_args.kt_expert_placement_strategy

    # Generate masks based on strategy
    tp_rank = get_tensor_model_parallel_rank()

    if strategy in ("frequency", "frequency-uniform"):
        # Load activation frequency from init_expert_location if it's a .pt file
        init_loc = server_args.init_expert_location
        has_activation_freq = init_loc and init_loc.endswith(".pt")

        if has_activation_freq:
            logger.info("Loading activation frequency from %s", init_loc)
            loaded_data = torch.load(init_loc, map_location="cpu", weights_only=True)
            # Handle both dict format (from ExpertDistributionRecorder) and raw tensor
            if isinstance(loaded_data, dict):
                if "logical_count" in loaded_data:
                    activation_counts = loaded_data["logical_count"]
                else:
                    raise ValueError(
                        f"Loaded dict does not contain 'logical_count' key. "
                        f"Available keys: {list(loaded_data.keys())}"
                    )
            else:
                activation_counts = loaded_data
            # Expected shape: [buffer_size, num_layers, num_experts]
            if activation_counts.dim() != 3:
                raise ValueError(
                    f"Expected activation counts tensor with 3 dims [buffer_size, num_layers, num_experts], "
                    f"got {activation_counts.dim()} dims with shape {activation_counts.shape}"
                )
            _, file_num_layers, file_num_experts = activation_counts.shape
            if file_num_layers != num_layers:
                raise ValueError(
                    f"Activation counts num_layers ({file_num_layers}) doesn't match "
                    f"model num_layers ({num_layers})"
                )
            if file_num_experts != num_experts:
                raise ValueError(
                    f"Activation counts num_experts ({file_num_experts}) doesn't match "
                    f"model num_experts ({num_experts})"
                )
            # Sum across buffer_size (dim0) to get total activation counts per expert
            activation_freq = activation_counts.sum(dim=0).float()  # [num_layers, num_experts]
            logger.info("Using %s strategy with activation frequency data", strategy)
        else:
            # No activation frequency file, use zeros (uniform distribution)
            logger.warning(
                "Using %s strategy WITHOUT activation frequency data "
                "(uniform distribution fallback)"
                % strategy
            )
            activation_freq = torch.zeros(num_layers, num_experts, dtype=torch.float32)
            # For layers that are actually MoE layers, set uniform distribution
            for layer_idx in range(num_layers):
                if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
                    activation_freq[layer_idx, :] = 1.0

        # Generate masks on rank 0
        if tp_rank == 0:
            if explicit_layer_gpu_experts is not None:
                logger.info(
                    "Using %s strategy with explicit per-layer GPU expert budgets",
                    strategy,
                )
                masks = generate_layer_count_masks(
                    num_layers,
                    num_experts,
                    explicit_layer_gpu_experts,
                    first_k_dense_replace,
                    moe_layer_freq,
                    activation_freq=activation_freq,
                )
            elif strategy == "frequency":
                masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts)
            else:
                masks = generate_frequency_uniform_masks(
                    activation_freq,
                    num_gpu_experts,
                    first_k_dense_replace,
                    moe_layer_freq,
                )
            # For non-MoE layers, set all experts to GPU
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
                    masks[layer_idx, :] = True
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "front-loading":
        if tp_rank == 0:
            logger.info("Using front-loading strategy for GPU expert placement")
            if explicit_layer_gpu_experts is not None:
                logger.info(
                    "Explicit per-layer KT budgets override front-loading "
                    "layer fill order"
                )
                masks = generate_layer_count_masks(
                    num_layers,
                    num_experts,
                    explicit_layer_gpu_experts,
                    first_k_dense_replace,
                    moe_layer_freq,
                )
            else:
                masks = generate_front_loading_masks(
                    num_layers, num_experts, num_gpu_experts,
                    first_k_dense_replace, moe_layer_freq
                )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "uniform":
        if tp_rank == 0:
            logger.info("Using uniform strategy for GPU expert placement")
            if explicit_layer_gpu_experts is not None:
                masks = generate_layer_count_masks(
                    num_layers,
                    num_experts,
                    explicit_layer_gpu_experts,
                    first_k_dense_replace,
                    moe_layer_freq,
                )
            else:
                masks = generate_uniform_masks(
                    num_layers, num_experts, num_gpu_experts,
                    first_k_dense_replace, moe_layer_freq
                )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "random":
        if tp_rank == 0:
            logger.info("Using random strategy for GPU expert placement (seed=42)")
            if explicit_layer_gpu_experts is not None:
                masks = generate_layer_count_masks(
                    num_layers,
                    num_experts,
                    explicit_layer_gpu_experts,
                    first_k_dense_replace,
                    moe_layer_freq,
                    random_seed=42,
                )
            else:
                masks = generate_random_masks(
                    num_layers, num_experts, num_gpu_experts,
                    first_k_dense_replace, moe_layer_freq, seed=42
                )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    else:
        raise ValueError(f"Unknown kt_expert_placement_strategy: {strategy}")

    if dist.is_initialized():
        _group = get_tp_group().cpu_group
        # PP-aware: translate group-local rank 0 to its global rank so the
        # broadcast source is in the group (under PP, rank 1's TP group does
        # not contain global rank 0).
        _src = dist.get_global_rank(_group, 0)
        dist.broadcast(masks, src=_src, group=_group)

    _KT_GPU_EXPERTS_MASKS = masks

    # Log per-layer GPU expert counts (rank 0 only, MoE layers only)
    if tp_rank == 0:
        per_layer_gpu_experts = masks.sum(dim=1).cpu().tolist()
        for layer_idx, num_gpu in enumerate(per_layer_gpu_experts):
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            # Only log for actual MoE layers
            if is_moe_layer:
                logger.info(
                    "KT GPU experts: layer %d (MoE) has %d GPU experts",
                    layer_idx,
                    int(num_gpu),
                )

        # Count total GPU experts only for actual MoE layers
        total_moe_gpu_experts = sum(
            masks[i].sum().item()
            for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        num_moe_layers = sum(
            1 for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        logger.info(
            "Generated KT GPU experts masks using '%s' strategy: %d MoE layers (out of %d total layers) x %d experts, "
            "total GPU experts in MoE layers = %d",
            strategy, num_moe_layers, num_layers, num_experts, total_moe_gpu_experts
        )

    return _KT_GPU_EXPERTS_MASKS


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured and not disabled, None otherwise
    """
    # Check if KT EP wrapper is disabled (e.g., for draft models in speculative decoding)
    from sglang.srt.layers.moe.utils import is_kt_ep_wrapper_disabled

    if is_kt_ep_wrapper_disabled():
        return None

    if server_args.kt_weight_path is None:
        return None

    # Get GPU experts masks (initializes if needed)
    masks = _init_kt_gpu_experts_masks(server_args)
    if masks is None:
        return None

    # Get mask for this specific layer
    gpu_experts_mask = masks[layer_idx]

    # Get num_layers from model config (unwrap VL configs)
    hf_config = server_args.get_hf_config()
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    num_layers = getattr(hf_config, "num_hidden_layers", None)

    # If PP is active, compute this rank's local end_layer so KTEPWrapperMethod
    # can correctly disable deferred experts on the last *local* MoE layer.
    pp_end_layer = None
    if num_layers is not None:
        try:
            from sglang.srt.distributed import get_pp_group, get_pp_indices

            pp_group = get_pp_group()
            if pp_group.world_size > 1:
                _, pp_end_layer = get_pp_indices(
                    num_layers,
                    pp_group.rank_in_group,
                    pp_group.world_size,
                )
        except Exception:
            pp_end_layer = None

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=gpu_experts_mask,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        numa_nodes=server_args.kt_numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        pp_end_layer=pp_end_layer,
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
        kt_enable_dynamic_expert_update=server_args.kt_enable_dynamic_expert_update,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU expert IDs and remap GPU expert IDs to weight indices.

    This function:
    1. Sets CPU expert IDs (gpu_experts_mask=False) to -1 so GPU kernel skips them
    2. Remaps GPU expert IDs to GPU weight indices (0 to num_gpu_experts-1)

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        gpu_experts_mask: Boolean tensor of shape [num_experts] where True indicates GPU expert
        logical_to_gpu_index: Int tensor of shape [num_experts] mapping logical ID to GPU index

    Returns:
        Remapped topk_ids tensor with GPU indices for GPU experts, -1 for CPU experts
    """
    is_gpu_expert = gpu_experts_mask[topk_ids]
    # For GPU experts: remap to GPU weight index; for CPU experts: set to -1
    remapped_ids = torch.where(is_gpu_expert, logical_to_gpu_index[topk_ids], -1)
    return remapped_ids


def select_top_experts_from_batch(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_gpu_experts: int,
) -> torch.Tensor:
    """Select top N most frequently activated experts from batch routing results.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        num_experts: Total number of experts in the layer
        num_gpu_experts: Number of experts to select for GPU

    Returns:
        Tensor of shape [num_gpu_experts] containing selected expert IDs (sorted)

    Edge cases:
        - If batch has fewer unique experts than num_gpu_experts, fills remaining
          slots with least-activated experts (maintaining determinism)
        - Handles ties by preferring lower expert IDs (deterministic)
    """
    # Count activation frequency for each expert in this batch
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)

    # Flatten topk_ids and count occurrences
    flat_ids = topk_ids.flatten()
    # Filter out invalid IDs (< 0 or >= num_experts)
    valid_mask = (flat_ids >= 0) & (flat_ids < num_experts)
    valid_ids = flat_ids[valid_mask]

    if valid_ids.numel() > 0:
        expert_counts.index_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64))

    # Select top num_gpu_experts by frequency
    # For ties, torch.topk with sorted=True will prefer earlier indices (deterministic)
    _, selected_indices = torch.topk(
        expert_counts,
        k=min(num_gpu_experts, num_experts),
        largest=True,
        sorted=True  # Ensures deterministic tie-breaking
    )

    # Sort selected indices for easier debugging and consistent ordering
    selected_experts = selected_indices.sort()[0]

    return selected_experts


def copy_experts_weights_wna16(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy AutoRound/GPTQ/AWQ WNA16 expert weights into resident GPU slots."""
    weight_names = ["w13_qweight", "w13_scales", "w2_qweight", "w2_scales"]
    for optional_name in ("w13_qzeros", "w2_qzeros"):
        src_optional = getattr(src_layer, optional_name, None)
        dst_optional = getattr(dst_layer, optional_name, None)
        if (
            src_optional is not None
            and dst_optional is not None
            and src_optional.numel() != 0
            and dst_optional.numel() != 0
        ):
            weight_names.append(optional_name)

    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)
        dst_weight = getattr(dst_layer, weight_name)
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_int4(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy INT4 Marlin expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight_packed: Packed INT4 weights for gate+up projection
        - w13_weight_scale: FP16 scales for w13
        - w2_weight_packed: Packed INT4 weights for down projection
        - w2_weight_scale: FP16 scales for w2
    """
    weight_names = ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            # In src_layer, expert at logical_id is at index logical_id
            # In dst_layer, we write to gpu_index dst_idx
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 block quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale_inv: FP32 inverse scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale_inv: FP32 inverse scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8_channel(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 per-channel quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale: FP32 per-channel scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale: FP32 per-channel scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_bf16(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy BF16/unquantized expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: BF16 weights for gate+up projection
        - w2_weight: BF16 weights for down projection
    """
    weight_names = ["w13_weight", "w2_weight"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def update_gpu_expert_mappings(
    selected_experts: torch.Tensor,
    num_experts: int,
    device: torch.device,
):
    """Update GPU expert mapping tables based on newly selected experts.

    Args:
        selected_experts: Tensor of logical expert IDs now on GPU (shape: [num_gpu_experts])
        num_experts: Total number of experts in layer
        device: Target CUDA device for mapping tensors

    Returns:
        Tuple of (gpu_experts_mask, logical_to_gpu_index, gpu_index_to_logical):
            - gpu_experts_mask: CPU bool tensor [num_experts], True = on GPU
            - logical_to_gpu_index: CUDA int32 tensor [num_experts], maps logical -> GPU index
            - gpu_index_to_logical: CPU int32 tensor [num_gpu_experts], reverse mapping
    """
    num_gpu_experts = len(selected_experts)

    # Create new mask (CPU tensor)
    gpu_experts_mask_cpu = torch.zeros(num_experts, dtype=torch.bool, device='cpu')
    gpu_experts_mask_cpu[selected_experts.cpu()] = True

    # Create logical_to_gpu_index (CUDA tensor)
    logical_to_gpu_index = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=device
    )
    for gpu_idx, logical_id in enumerate(selected_experts):
        logical_to_gpu_index[logical_id] = gpu_idx

    # Create gpu_index_to_logical (CPU tensor for weight loading)
    gpu_index_to_logical_cpu = selected_experts.cpu().to(torch.int32)

    return gpu_experts_mask_cpu, logical_to_gpu_index, gpu_index_to_logical_cpu


def update_kt_wrapper_masks(
    wrapper: Optional["KTMoEWrapper"],
    gpu_experts_mask_cpu: torch.Tensor,
) -> None:
    """Update KT wrapper's internal GPU experts mask (rank 0 only).

    Args:
        wrapper: KTMoEWrapper instance (None if not rank 0)
        gpu_experts_mask_cpu: New GPU experts mask to apply

    The wrapper needs updated masks to correctly route tokens to CPU vs GPU experts.
    This is called on rank 0 only since only rank 0 has the wrapper instance.

    CRITICAL: wrapper.gpu_experts_mask is a pinned memory tensor whose pointer is shared
    with C++ code. We MUST use .copy_() to update in-place, not replace the reference.
    """
    if wrapper is None:
        return

    # Update wrapper's internal mask IN-PLACE
    # CRITICAL: The C++ code holds a pointer to this tensor's memory.
    # Replacing the reference would leave C++ pointing to old/freed memory.
    wrapper.gpu_experts_mask.copy_(gpu_experts_mask_cpu)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (identified by gpu_experts_mask=True) using any quantization method
    - CPU experts (identified by gpu_experts_mask=False) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoEMethod(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, gpu_experts_mask=mask, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask  # bool tensor [num_experts], on CPU
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()
        # KT CPU offload must run on every PP rank that owns MoE layers.  The
        # local-layer gating in FusedMoE only constructs this wrapper for
        # layers resident on the current PP rank; if we disabled KT on
        # non-last PP ranks, routed cold experts on those layers would be
        # masked out of the GPU path and never computed.
        from sglang.srt.distributed import get_pp_group
        self.pp_group = get_pp_group()
        self._is_kt_active_rank = self.tp_rank == 0

        # Mapping tables for non-contiguous GPU expert allocation (CPU tensors)
        # Used by weight_loader to remap expert_id when loading weights
        gpu_expert_indices = torch.where(self.gpu_experts_mask)[0]
        self.logical_to_gpu_index = torch.full(
            (len(self.gpu_experts_mask),), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )
        self.gpu_index_to_logical = gpu_expert_indices.to(torch.int32)

        # CUDA tensors for inference (will be set in create_weights)
        self.gpu_experts_mask_cuda = None
        self.logical_to_gpu_index_cuda = None

        self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
        if (
            self.gpu_prefill_token_threshold > 0
            and (kt_config.method or "").upper() == "GPTQ_INT4"
        ):
            logger.warning(
                "Disabling KT layerwise full-GPU prefill and dynamic expert update "
                "for layer %d: kt-kernel GPTQ_INT4 does not implement "
                "write_weight_scale_to_buffer.",
                kt_config.layer_idx,
            )
            self.gpu_prefill_token_threshold = 0
            kt_config.kt_enable_dynamic_expert_update = False
        self._full_init_args = None
        self.wrapper: Optional[KTMoEWrapper] = None

        # Dual-stream parallelism: cpu_stream for CPU expert operations,
        # main stream for GPU computation (initialized in create_weights)
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None  # CPU computation done

        # Shared staging buffer reference (initialized in create_weights, shared across all layers)
        self._shared_staging_buffer: Optional[SharedStagingBuffer] = None
        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192
        self._kt_staging_probe_layer: Optional[torch.nn.Module] = None
        self._kt_staging_probe_device: Optional[torch.device] = None
        self._kt_staging_probe_state: Optional[dict] = None
        self._kt_staging_probe_pending: Optional[dict] = None
        self._kt_staging_probe_cursor: int = 0
        self._kt_staging_probe_warned: bool = False
        self._kt_staging_probe_swaps: int = 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        # Prefer the PP-local end_layer when available so the last *local*
        # MoE layer disables deferred experts; otherwise fall back to the
        # global num_layers boundary (single-PP behavior).
        last_local_layer = (
            (self.kt_config.pp_end_layer - 1)
            if self.kt_config.pp_end_layer is not None
            else (
                self.kt_config.num_layers - 1
                if self.kt_config.num_layers is not None
                else None
            )
        )
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and last_local_layer is not None
            and self.kt_config.layer_idx == last_local_layer
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU weights are indexed by gpu_index (0 to num_gpu_experts-1), not logical expert ID
        # The mapping logical_to_gpu_index is used to remap IDs during weight loading and inference
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # Move mask and mapping tables to GPU for inference
        target_device = next(layer.parameters()).device
        self._kt_staging_probe_layer = layer
        self._kt_staging_probe_device = target_device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(device=target_device)

        # Initialize dual-stream for CPU-GPU parallelism (active rank only)
        if self._is_kt_active_rank:
            self._cpu_stream = torch.cuda.Stream(device=target_device)
            self._sync_done_event = torch.cuda.Event()

            # Get or create shared staging buffer (shared across all MoE layers to save GPU memory)
            self._shared_staging_buffer = get_or_create_shared_staging_buffer(
                max_tokens=self._staging_buffer_max_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts are identified by gpu_experts_mask=False
        if self._is_kt_active_rank:
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=self.gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=self.kt_config.numa_nodes,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )
            _register_kt_staging_probe_method(self)

    def _kt_staging_probe_weight_names(self) -> Optional[List[str]]:
        layer = self._kt_staging_probe_layer
        if layer is None:
            return None
        if hasattr(layer, "w13_qweight") and hasattr(layer, "w2_qweight"):
            for optional_name in ("w13_qzeros", "w2_qzeros"):
                optional = getattr(layer, optional_name, None)
                if optional is not None and optional.numel() != 0:
                    if not self._kt_staging_probe_warned:
                        _kt_staging_probe_log(
                            "unsupported_wna16_qzeros",
                            layer=self.kt_config.layer_idx,
                            optional=optional_name,
                        )
                        self._kt_staging_probe_warned = True
                    return None
            return _KT_STAGING_PROBE_WEIGHT_NAMES_WNA16
        if not self._kt_staging_probe_warned:
            _kt_staging_probe_log(
                "unsupported_layout",
                layer=self.kt_config.layer_idx,
                has_w13_qweight=hasattr(layer, "w13_qweight"),
                has_w13_weight_packed=hasattr(layer, "w13_weight_packed"),
            )
            self._kt_staging_probe_warned = True
        return None

    def _kt_staging_probe_is_eligible(
        self, device: Optional[torch.device] = None
    ) -> bool:
        if (
            not self._is_kt_active_rank
            or self.wrapper is None
            or self._kt_staging_probe_layer is None
            or self._kt_staging_probe_device is None
        ):
            return False
        if device is not None and torch.device(device) != self._kt_staging_probe_device:
            return False
        if get_tensor_model_parallel_world_size() != 1:
            if not self._kt_staging_probe_warned:
                _kt_staging_probe_log(
                    "unsupported_tp_world_size",
                    layer=self.kt_config.layer_idx,
                    tp_world_size=get_tensor_model_parallel_world_size(),
                )
                self._kt_staging_probe_warned = True
            return False
        return self._kt_staging_probe_weight_names() is not None

    def _kt_staging_probe_configured_experts(self) -> Optional[List[int]]:
        raw = os.getenv("SGLANG_KT_STAGING_PROBE_EXPERTS")
        if not raw:
            return None
        layer_prefix = f"{self.kt_config.layer_idx}:"
        for spec in raw.split(";"):
            spec = spec.strip()
            if not spec or not spec.startswith(layer_prefix):
                continue
            ids = spec[len(layer_prefix) :]
            out = []
            for item in ids.split(","):
                item = item.strip()
                if not item:
                    continue
                out.append(int(item))
            return out or None
        return None

    def _kt_staging_probe_next_expert(self) -> Optional[int]:
        configured = self._kt_staging_probe_configured_experts()
        if configured is not None:
            candidates = [
                expert
                for expert in configured
                if 0 <= expert < int(getattr(self, "global_num_experts", 0))
            ]
        else:
            with torch.no_grad():
                cold = torch.where(~self.gpu_experts_mask.cpu())[0]
                candidates = [int(x) for x in cold.tolist()]
        if not candidates:
            return None
        expert = candidates[self._kt_staging_probe_cursor % len(candidates)]
        self._kt_staging_probe_cursor += 1
        return int(expert)

    def _kt_staging_probe_ensure_state(self) -> Optional[dict]:
        if self._kt_staging_probe_state is not None:
            return self._kt_staging_probe_state

        layer = self._kt_staging_probe_layer
        names = self._kt_staging_probe_weight_names()
        if layer is None or names is None:
            return None

        cpu_buffers = {}
        gpu_scratch = {}
        per_expert_nbytes = {}
        total_bytes = 0
        pinned = True
        alloc_t = time.perf_counter()
        for name in names:
            gpu_tensor = getattr(layer, name)
            shape = (2,) + tuple(gpu_tensor.shape[1:])
            try:
                cpu_buf = torch.empty(shape, dtype=gpu_tensor.dtype, pin_memory=True)
            except Exception:
                cpu_buf = torch.empty(shape, dtype=gpu_tensor.dtype)
                pinned = False
            gpu_buf = torch.empty(shape, dtype=gpu_tensor.dtype, device=gpu_tensor.device)
            cpu_buffers[name] = cpu_buf
            gpu_scratch[name] = gpu_buf
            nbytes = int(cpu_buf[0].numel()) * int(cpu_buf.element_size())
            per_expert_nbytes[name] = nbytes
            total_bytes += nbytes

        state = {
            "cpu_buffers": cpu_buffers,
            "gpu_scratch": gpu_scratch,
            "per_expert_nbytes": per_expert_nbytes,
            "stream": torch.cuda.Stream(device=self._kt_staging_probe_device),
            "slot_done": [None, None],
            "pinned": pinned,
            "total_bytes": total_bytes,
            "next_slot": 0,
            "swap_cursor": 0,
        }
        self._kt_staging_probe_state = state
        _kt_staging_probe_log(
            "alloc",
            layer=self.kt_config.layer_idx,
            names=",".join(names),
            mb=f"{total_bytes / 1024**2:.3f}",
            pinned=pinned,
            alloc_ms=f"{(time.perf_counter() - alloc_t) * 1000.0:.3f}",
        )
        return state

    def _kt_staging_probe_submit(self, expert: int, slot: int, state: dict) -> bool:
        names = self._kt_staging_probe_weight_names()
        if names != _KT_STAGING_PROBE_WEIGHT_NAMES_WNA16:
            return False

        prior_event = state["slot_done"][slot]
        if prior_event is not None:
            prior_event.synchronize()

        cpu_buffers = state["cpu_buffers"]
        per_expert_nbytes = state["per_expert_nbytes"]
        ptrs = {
            name: int(cpu_buffers[name][slot].data_ptr())
            for name in _KT_STAGING_PROBE_WEIGHT_NAMES_WNA16
        }
        t_submit = time.perf_counter()
        try:
            self.wrapper.submit_write_weight_scale_to_buffer(
                1,
                int(expert),
                [ptrs["w13_qweight"]],
                [ptrs["w13_scales"]],
                [ptrs["w2_qweight"]],
                [ptrs["w2_scales"]],
            )
        except Exception as err:
            _kt_staging_probe_log(
                "submit_error",
                layer=self.kt_config.layer_idx,
                expert=expert,
                slot=slot,
                error=str(err).splitlines()[0],
            )
            return False

        self._kt_staging_probe_pending = {
            "expert": int(expert),
            "slot": int(slot),
            "submit_t": time.perf_counter(),
            "bytes": sum(per_expert_nbytes.values()),
        }
        _kt_staging_probe_log(
            "cpu_submit",
            layer=self.kt_config.layer_idx,
            expert=expert,
            slot=slot,
            mb=f"{sum(per_expert_nbytes.values()) / 1024**2:.3f}",
            elapsed_ms=f"{(time.perf_counter() - t_submit) * 1000.0:.3f}",
        )
        return True

    def _kt_staging_probe_step(
        self, batch_size: int = 0, forward_mode: str = ""
    ) -> Optional[dict]:
        state = self._kt_staging_probe_ensure_state()
        if state is None:
            return None

        probe = None
        pending = self._kt_staging_probe_pending
        if pending is not None:
            t_sync = time.perf_counter()
            try:
                self.wrapper.sync_write_weight_scale_to_buffer()
            except Exception as err:
                _kt_staging_probe_log(
                    "cpu_sync_error",
                    layer=self.kt_config.layer_idx,
                    expert=pending["expert"],
                    slot=pending["slot"],
                    error=str(err).splitlines()[0],
                )
                self._kt_staging_probe_pending = None
                return None
            cpu_wait_ms = (time.perf_counter() - t_sync) * 1000.0
            cpu_total_ms = (time.perf_counter() - pending["submit_t"]) * 1000.0
            slot = pending["slot"]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            done_event = torch.cuda.Event()
            start_t = time.perf_counter()
            with torch.cuda.stream(state["stream"]):
                start_event.record(state["stream"])
                for name, cpu_buf in state["cpu_buffers"].items():
                    state["gpu_scratch"][name][slot].copy_(
                        cpu_buf[slot], non_blocking=True
                    )
                end_event.record(state["stream"])
                done_event.record(state["stream"])
            state["slot_done"][slot] = done_event
            probe = {
                "start_t": start_t,
                "start_event": start_event,
                "end_event": end_event,
                "done_event": done_event,
                "layer": self.kt_config.layer_idx,
                "expert": pending["expert"],
                "slot": slot,
                "bytes": pending["bytes"],
                "method": self,
            }
            _kt_staging_probe_log(
                "h2d_start",
                layer=self.kt_config.layer_idx,
                expert=pending["expert"],
                slot=slot,
                batch_size=batch_size,
                forward_mode=forward_mode,
                mb=f"{pending['bytes'] / 1024**2:.3f}",
                cpu_wait_ms=f"{cpu_wait_ms:.3f}",
                cpu_total_ms=f"{cpu_total_ms:.3f}",
                pinned=state["pinned"],
            )
            self._kt_staging_probe_pending = None

        next_expert = self._kt_staging_probe_next_expert()
        if next_expert is not None:
            slot = state["next_slot"]
            state["next_slot"] = (slot + 1) % 2
            self._kt_staging_probe_submit(next_expert, slot, state)
        else:
            _kt_staging_probe_log(
                "no_candidate",
                layer=self.kt_config.layer_idx,
            )

        return probe

    def _kt_staging_probe_configured_evicts(self) -> Optional[List[int]]:
        raw = os.getenv("SGLANG_KT_STAGING_SWAP_EVICTS")
        if not raw:
            return None
        layer_prefix = f"{self.kt_config.layer_idx}:"
        for spec in raw.split(";"):
            spec = spec.strip()
            if not spec or not spec.startswith(layer_prefix):
                continue
            ids = spec[len(layer_prefix) :]
            out = []
            for item in ids.split(","):
                item = item.strip()
                if not item:
                    continue
                out.append(int(item))
            return out or None
        return None

    def _kt_staging_probe_choose_evict(
        self, staged_expert: int, state: dict
    ) -> Optional[Tuple[int, int]]:
        configured = self._kt_staging_probe_configured_evicts()
        if configured is not None:
            for logical_id in configured:
                if logical_id == staged_expert:
                    continue
                if 0 <= logical_id < int(self.gpu_experts_mask.numel()):
                    gpu_idx = int(self.logical_to_gpu_index[int(logical_id)].item())
                    if gpu_idx >= 0:
                        return int(logical_id), gpu_idx
            return None

        num_slots = int(self.num_gpu_experts)
        if num_slots <= 0:
            return None
        for _ in range(num_slots):
            gpu_idx = int(state["swap_cursor"] % num_slots)
            state["swap_cursor"] += 1
            logical_id = int(self.gpu_index_to_logical[gpu_idx].item())
            if logical_id >= 0 and logical_id != staged_expert:
                return logical_id, gpu_idx
        return None

    def _kt_staging_probe_maybe_swap(self, probe: dict) -> None:
        if not _kt_staging_swap_enabled():
            return
        limit = _kt_staging_swap_limit()
        if limit > 0 and self._kt_staging_probe_swaps >= limit:
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                _kt_staging_probe_log(
                    "swap_skip_capture",
                    layer=self.kt_config.layer_idx,
                    expert=probe["expert"],
                )
                return
        except Exception:
            return

        state = self._kt_staging_probe_state
        layer = self._kt_staging_probe_layer
        names = self._kt_staging_probe_weight_names()
        if state is None or layer is None or names is None:
            return

        staged_expert = int(probe["expert"])
        if bool(self.gpu_experts_mask[staged_expert].item()):
            _kt_staging_probe_log(
                "swap_skip_resident",
                layer=self.kt_config.layer_idx,
                expert=staged_expert,
            )
            return

        evict = self._kt_staging_probe_choose_evict(staged_expert, state)
        if evict is None:
            _kt_staging_probe_log(
                "swap_skip_no_evict",
                layer=self.kt_config.layer_idx,
                expert=staged_expert,
            )
            return

        evicted_expert, gpu_idx = evict
        t_swap = time.perf_counter()
        current_stream = torch.cuda.current_stream(self._kt_staging_probe_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(current_stream)
        for name in names:
            dst_weight = getattr(layer, name)
            dst_weight[gpu_idx].copy_(
                state["gpu_scratch"][name][probe["slot"]],
                non_blocking=True,
            )
        end_event.record(current_stream)
        end_event.synchronize()
        try:
            weight_copy_ms = start_event.elapsed_time(end_event)
        except Exception:
            weight_copy_ms = None

        self.gpu_experts_mask[evicted_expert] = False
        self.gpu_experts_mask[staged_expert] = True
        self.logical_to_gpu_index[evicted_expert] = -1
        self.logical_to_gpu_index[staged_expert] = int(gpu_idx)
        self.gpu_index_to_logical[gpu_idx] = int(staged_expert)
        self.kt_config.gpu_experts_mask = self.gpu_experts_mask
        self.gpu_experts_mask_cuda.copy_(
            self.gpu_experts_mask.to(device=self._kt_staging_probe_device)
        )
        self.logical_to_gpu_index_cuda.copy_(
            self.logical_to_gpu_index.to(device=self._kt_staging_probe_device)
        )
        if self._is_kt_active_rank:
            update_kt_wrapper_masks(self.wrapper, self.gpu_experts_mask)

        self._kt_staging_probe_swaps += 1
        _kt_staging_probe_log(
            "swap",
            layer=self.kt_config.layer_idx,
            expert=staged_expert,
            evicted=evicted_expert,
            gpu_idx=gpu_idx,
            swaps=self._kt_staging_probe_swaps,
            weight_copy_ms=(
                f"{weight_copy_ms:.3f}" if weight_copy_ms is not None else None
            ),
            elapsed_ms=f"{(time.perf_counter() - t_swap) * 1000.0:.3f}",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self._is_kt_active_rank and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if (
                metadata is not None
                and getattr(metadata, "physical_to_logical_map_cpu", None) is not None
            ):
                physical_to_logical_map_cpu = (
                    metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx]
                    .contiguous()
                )
            else:
                # Fallback for setups without EPLB metadata: identity mapping.
                physical_to_logical_map_cpu = torch.arange(
                    layer.num_experts, dtype=torch.int64, device="cpu"
                )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config

        # Create a separate config for GPU method without routed_scaling_factor.
        # This is because:
        # 1. GPU method's moe_sum_reduce would apply routed_scaling_factor internally
        # 2. KT CPU kernel does NOT apply routed_scaling_factor
        # 3. The combined output (GPU + CPU) would have inconsistent scaling
        # 4. routed_scaling_factor is applied uniformly in deepseek_v2.py forward_normal
        # So we disable it in GPU method to avoid double scaling on GPU part.
        gpu_runner_config = replace(moe_runner_config, routed_scaling_factor=None)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )

        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if not self._is_kt_active_rank or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        t_submit = time.perf_counter()
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        _kt_log_timing(
            self,
            "kt.cpu_submit_enqueue",
            t_submit,
            tokens=int(x.shape[0]) if x.dim() > 0 else 0,
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if not self._is_kt_active_rank or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        t_sync = time.perf_counter()
        result = self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )
        _kt_log_timing(
            self,
            "kt.cpu_sync_enqueue",
            t_sync,
            tokens=int(x.shape[0]) if x.dim() > 0 else 0,
        )
        return result

    def _submit_with_staged_input(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        staged_hidden_states: torch.Tensor,
    ) -> None:
        """Submit CPU expert computation using staged hidden states.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
            staged_hidden_states: Pre-copied hidden states in staging buffer
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if not self._is_kt_active_rank or self.wrapper is None:
            return

        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task using staged buffer
        t_submit = time.perf_counter()
        self.wrapper.submit_forward(
            staged_hidden_states,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(staged_hidden_states.device).cuda_stream,
        )
        _kt_log_timing(
            self,
            "kt.cpu_submit_enqueue",
            t_submit,
            tokens=int(staged_hidden_states.shape[0])
            if staged_hidden_states.dim() > 0
            else 0,
            staged=True,
        )

    def _sync_with_staged_input(
        self, staged_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Synchronize CPU computation using staged hidden states reference.

        Args:
            staged_hidden_states: Staged buffer used in submit

        Returns:
            CPU expert computation results
        """
        if not self._is_kt_active_rank or self.wrapper is None:
            return torch.zeros_like(staged_hidden_states)

        t_sync = time.perf_counter()
        result = self.wrapper.sync_forward(
            staged_hidden_states,
            torch.cuda.current_stream(staged_hidden_states.device).cuda_stream,
        )
        _kt_log_timing(
            self,
            "kt.cpu_sync_enqueue",
            t_sync,
            tokens=int(staged_hidden_states.shape[0])
            if staged_hidden_states.dim() > 0
            else 0,
            staged=True,
        )
        return result

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        global _KT_FULL_GPU_FALLBACK_DISABLED, _SHARED_FULL_CONTEXT

        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # Record GPU expert mask for distribution tracking (active rank only)
        # Use gpu_experts_mask_cuda which is already on GPU for CUDA graph compatibility
        if self._is_kt_active_rank:
            recorder = get_global_expert_distribution_recorder()
            recorder.on_gpu_expert_mask(
                self.kt_config.layer_idx, self.gpu_experts_mask_cuda
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0
        _kt_log_timing(
            self,
            "kt.apply.begin",
            tokens=num_tokens,
            threshold=self.gpu_prefill_token_threshold,
            full_gpu=(
                not _KT_FULL_GPU_FALLBACK_DISABLED
                and self.gpu_prefill_token_threshold > 0
                and num_tokens >= self.gpu_prefill_token_threshold
            ),
        )

        # Check for full GPU fallback
        if (
            not _KT_FULL_GPU_FALLBACK_DISABLED
            and self.gpu_prefill_token_threshold > 0
            and num_tokens >= self.gpu_prefill_token_threshold
        ):
            min_free_bytes = _kt_full_gpu_fallback_min_free_bytes()
            free_bytes = None
            if min_free_bytes > 0 and torch.cuda.is_available():
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(x.device)
                except Exception:
                    free_bytes = None
            if free_bytes is not None and free_bytes < min_free_bytes:
                _KT_FULL_GPU_FALLBACK_DISABLED = True
                self.gpu_prefill_token_threshold = 0
                _kt_log_timing(
                    self,
                    "kt.full_gpu.low_mem_disable",
                    tokens=num_tokens,
                    free_mb=f"{free_bytes / 1024**2:.1f}",
                    min_free_mb=f"{min_free_bytes / 1024**2:.1f}",
                )
                if self._is_kt_active_rank:
                    logger.warning(
                        "KT full-GPU prefill fallback disabled before layer %d: "
                        "free GPU memory %.1f MiB is below %.1f MiB. "
                        "Continuing with hybrid CPU/GPU experts.",
                        self.kt_config.layer_idx,
                        free_bytes / 1024**2,
                        min_free_bytes / 1024**2,
                    )
            else:
                try:
                    ctx = self._build_full_context(layer)

                    t_compute = time.perf_counter()
                    result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    compute_time = (time.perf_counter() - t_compute) * 1000.0
                    _kt_log_timing(
                        self,
                        "kt.full_gpu.compute",
                        tokens=num_tokens,
                        elapsed_ms=f"{compute_time:.3f}",
                    )

                    # Dynamic expert update: analyze batch and update GPU experts
                    if self.kt_config.kt_enable_dynamic_expert_update:
                        t_update = time.perf_counter()
                        self._update_gpu_experts_from_batch(
                            layer=layer,
                            ctx=ctx,
                            dispatch_output=dispatch_output,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        update_time = (time.perf_counter() - t_update) * 1000.0
                        _kt_log_timing(
                            self,
                            "kt.dynamic_update",
                            tokens=num_tokens,
                            elapsed_ms=f"{update_time:.3f}",
                        )

                        if self._is_kt_active_rank:
                            logger.info(
                                "KT layerwise prefill: layer %d compute = %.2f ms, expert update = %.2f ms",
                                self.kt_config.layer_idx,
                                compute_time,
                                update_time,
                            )
                    else:
                        if self._is_kt_active_rank:
                            logger.info(
                                "KT layerwise prefill: layer %d compute = %.2f ms",
                                self.kt_config.layer_idx,
                                compute_time,
                            )

                    return result
                except Exception as err:
                    if not _looks_like_cuda_oom(err):
                        raise
                    _KT_FULL_GPU_FALLBACK_DISABLED = True
                    _SHARED_FULL_CONTEXT = None
                    self.gpu_prefill_token_threshold = 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _kt_log_timing(
                        self,
                        "kt.full_gpu.oom_disable",
                        tokens=num_tokens,
                        error=str(err).splitlines()[0],
                    )
                    if self._is_kt_active_rank:
                        logger.warning(
                            "KT full-GPU prefill fallback disabled after OOM at "
                            "layer %d with %d tokens and %d resident GPU experts; "
                            "continuing with hybrid CPU/GPU experts. Error: %s",
                            self.kt_config.layer_idx,
                            num_tokens,
                            self.num_gpu_experts,
                            str(err).splitlines()[0],
                        )
        # Step 1: Copy hidden_states to staging buffer and submit CPU computation
        # Staging buffer allows GPU computation to proceed without waiting for D2H copy
        staging_buffer = None
        cpu_timing_start_event = None
        cpu_timing_end_event = None
        if self._is_kt_active_rank and self._cpu_stream is not None:
            # Use shared staging buffer (shared across all MoE layers to save GPU memory)
            assert self._shared_staging_buffer is not None, "Shared staging buffer not initialized"
            staging_buffer = self._shared_staging_buffer.get_slice(x.shape[0])

            # Copy to staging buffer on main stream
            t_stage = time.perf_counter()
            staging_buffer.copy_(x, non_blocking=True)
            _kt_log_timing(
                self,
                "kt.staging_copy_enqueue",
                t_stage,
                tokens=num_tokens,
            )

            # Fork to cpu_stream (waits for staging copy to complete)
            t_wait_stream = time.perf_counter()
            self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            _kt_log_timing(
                self,
                "kt.cpu_stream_wait_enqueue",
                t_wait_stream,
                tokens=num_tokens,
            )
            with torch.cuda.stream(self._cpu_stream):
                if _kt_timing_sync_allowed() and _kt_timing_layer_enabled(
                    self.kt_config.layer_idx
                ):
                    cpu_timing_start_event = torch.cuda.Event(enable_timing=True)
                    cpu_timing_end_event = torch.cuda.Event(enable_timing=True)
                    cpu_timing_start_event.record(self._cpu_stream)
                # Submit uses staging_buffer, so GPU can modify original x freely
                self._submit_with_staged_input(
                    layer, dispatch_output, staging_buffer
                )

        # Step 2: Prepare GPU computation by masking and remapping expert IDs
        # CPU expert IDs are set to -1; GPU expert IDs are remapped to GPU weight indices
        topk_ids = topk_output.topk_ids
        _kt_log_hitmiss(
            self,
            topk_ids,
            self.gpu_experts_mask_cuda,
            num_tokens,
        )
        t_mask = time.perf_counter()
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids, self.gpu_experts_mask_cuda, self.logical_to_gpu_index_cuda
        )
        _kt_log_timing(
            self,
            "kt.mask_remap_enqueue",
            t_mask,
            tokens=num_tokens,
            topk=tuple(topk_ids.shape),
        )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation on main stream
        # No wait needed - staging buffer decouples CPU and GPU data access
        current_stream = torch.cuda.current_stream(x.device)
        gpu_timing_start_event = None
        gpu_timing_end_event = None
        if _kt_timing_sync_allowed() and _kt_timing_layer_enabled(
            self.kt_config.layer_idx
        ):
            gpu_timing_start_event = torch.cuda.Event(enable_timing=True)
            gpu_timing_end_event = torch.cuda.Event(enable_timing=True)
            gpu_timing_start_event.record(current_stream)
        t_gpu = time.perf_counter()
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
        if gpu_timing_end_event is not None:
            gpu_timing_end_event.record(current_stream)
        _kt_log_timing(
            self,
            "kt.gpu_apply_enqueue",
            t_gpu,
            tokens=num_tokens,
        )

        # Step 4: Sync CPU results on cpu_stream, then synchronize streams
        output = gpu_combine_input.hidden_states
        if self._is_kt_active_rank and self._cpu_stream is not None:
            with torch.cuda.stream(self._cpu_stream):
                # Use staging_buffer for sync to get correct buffer reference
                cpu_output = self._sync_with_staged_input(staging_buffer)
                if cpu_timing_end_event is not None:
                    cpu_timing_end_event.record(self._cpu_stream)
                self._sync_done_event.record(self._cpu_stream)

            # Main stream waits for cpu_stream to complete before merging results
            t_wait_event = time.perf_counter()
            main_wait_start_event = None
            main_wait_end_event = None
            if _kt_timing_sync_allowed() and _kt_timing_layer_enabled(
                self.kt_config.layer_idx
            ):
                main_wait_start_event = torch.cuda.Event(enable_timing=True)
                main_wait_end_event = torch.cuda.Event(enable_timing=True)
                main_wait_start_event.record(current_stream)
            current_stream.wait_event(self._sync_done_event)
            if main_wait_end_event is not None:
                main_wait_end_event.record(current_stream)
            _kt_log_timing(
                self,
                "kt.main_wait_cpu_event_enqueue",
                t_wait_event,
                tokens=num_tokens,
            )
            if gpu_timing_start_event is not None and gpu_timing_end_event is not None:
                gpu_timing_end_event.synchronize()
                _kt_log_timing(
                    self,
                    "kt.gpu_apply_elapsed",
                    tokens=num_tokens,
                    elapsed_ms=(
                        f"{gpu_timing_start_event.elapsed_time(gpu_timing_end_event):.3f}"
                    ),
                )
            if main_wait_start_event is not None and main_wait_end_event is not None:
                main_wait_end_event.synchronize()
                _kt_log_timing(
                    self,
                    "kt.main_wait_cpu_event_elapsed",
                    tokens=num_tokens,
                    elapsed_ms=(
                        f"{main_wait_start_event.elapsed_time(main_wait_end_event):.3f}"
                    ),
                )
            if cpu_timing_start_event is not None and cpu_timing_end_event is not None:
                cpu_timing_end_event.synchronize()
                _kt_log_timing(
                    self,
                    "kt.cpu_stream_elapsed",
                    tokens=num_tokens,
                    elapsed_ms=(
                        f"{cpu_timing_start_event.elapsed_time(cpu_timing_end_event):.3f}"
                    ),
                )
            t_merge = time.perf_counter()
            output = output + cpu_output
            _kt_log_timing(
                self,
                "kt.merge_enqueue",
                t_merge,
                tokens=num_tokens,
            )
        elif gpu_timing_start_event is not None and gpu_timing_end_event is not None:
            gpu_timing_end_event.synchronize()
            _kt_log_timing(
                self,
                "kt.gpu_apply_elapsed",
                tokens=num_tokens,
                elapsed_ms=(
                    f"{gpu_timing_start_event.elapsed_time(gpu_timing_end_event):.3f}"
                ),
            )

        return StandardCombineInput(hidden_states=output)

    def _update_gpu_experts_from_batch(
        self,
        layer: torch.nn.Module,
        ctx: "SharedFullContext",
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Update original layer's GPU experts based on current batch statistics.

        This method:
        1. Analyzes topk_ids to find most frequently activated experts
        2. Copies selected expert weights from ctx.gpu_layer to layer
        3. Updates all mapping tables (gpu_experts_mask, logical_to_gpu_index, etc.)
        4. Broadcasts changes across TP ranks for consistency

        Args:
            layer: Original MoE layer with subset of GPU experts
            ctx: SharedFullContext containing temporary full GPU layer
            dispatch_output: Current batch dispatch output with routing information
        """
        # Step 1: Select top experts (rank 0 computes, broadcasts to all ranks)
        topk_ids = dispatch_output.topk_output.topk_ids
        device = topk_ids.device

        if self._is_kt_active_rank:
            selected_experts = select_top_experts_from_batch(
                topk_ids=topk_ids,
                num_experts=self.global_num_experts,
                num_gpu_experts=self.num_gpu_experts,
            )
        else:
            # Create placeholder on other ranks
            selected_experts = torch.zeros(
                self.num_gpu_experts, dtype=torch.int64, device=device
            )

        # Broadcast selected experts to all ranks for consistent weight updates
        if dist.is_initialized():
            _group = get_tp_group().device_group
            _src = dist.get_global_rank(_group, 0)
            dist.broadcast(selected_experts, src=_src, group=_group)

        # Step 2: Copy weights from temporary layer to original layer
        if getattr(ctx, "is_wna16_quant", False):
            copy_experts_weights_wna16(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx.is_fp8_quant:
            copy_experts_weights_fp8(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx.is_fp8_channel_quant:
            copy_experts_weights_fp8_channel(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx.is_bf16_quant:
            copy_experts_weights_bf16(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        else:
            copy_experts_weights_int4(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )

        # Step 3: Update mapping tables
        gpu_experts_mask_cpu, logical_to_gpu_index_cuda, gpu_index_to_logical_cpu = (
            update_gpu_expert_mappings(
                selected_experts=selected_experts,
                num_experts=self.global_num_experts,
                device=device,
            )
        )

        # Update instance variables (both CPU and CUDA versions)
        # CRITICAL: Use .copy_() for CUDA tensors to maintain same buffer for CUDA graph compatibility
        # CUDA graph captures tensor memory addresses during decode phase, so we must update
        # in-place rather than replacing the tensor reference
        self.gpu_experts_mask = gpu_experts_mask_cpu  # CPU tensor, safe to replace
        self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)  # In-place update for CUDA graph
        self.logical_to_gpu_index = logical_to_gpu_index_cuda.cpu()  # CPU version for weight loading
        self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cuda)  # In-place update for CUDA graph
        self.gpu_index_to_logical = gpu_index_to_logical_cpu  # CPU tensor, safe to replace

        # Step 4: Update KT wrapper (active rank only)
        if self._is_kt_active_rank:
            update_kt_wrapper_masks(self.wrapper, gpu_experts_mask_cpu)

        # Log expert changes (active rank only)
        if self._is_kt_active_rank:
            logger.debug(
                "KT dynamic update: layer %d updated GPU experts to: %s",
                self.kt_config.layer_idx,
                selected_experts.cpu().tolist(),
            )

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)

    def _build_full_context(self, layer: torch.nn.Module) -> "SharedFullContext":
        global _SHARED_FULL_CONTEXT

        if _SHARED_FULL_CONTEXT is None:
            _SHARED_FULL_CONTEXT = SharedFullContext(
                layer=layer,
                init_args=self._full_init_args,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )

        _SHARED_FULL_CONTEXT.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
            original_layer=layer,
            gpu_experts_mask=self.gpu_experts_mask,
            logical_to_gpu_index=self.logical_to_gpu_index,
        )
        return _SHARED_FULL_CONTEXT
