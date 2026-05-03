"""DFLASH spec-v2 overlap scheduling data structures."""

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils.common import is_pin_memory_available

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}
logger = logging.getLogger(__name__)


def _dflash_timeline_enabled() -> bool:
    return os.getenv("SGLANG_DFLASH_PP_TIMELINE") in ("1", "true", "TRUE")


def _dflash_log_timeline(
    phase: str, start_time: Optional[float] = None, **fields
) -> None:
    if not _dflash_timeline_enabled():
        return
    elapsed_ms = None
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    parts = [f"phase={phase}"]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.3f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info("DFLASH plan timeline %s", " ".join(parts))


def _dflash_block_size() -> int:
    block_size = int(get_global_server_args().speculative_num_draft_tokens)
    if block_size <= 0:
        raise RuntimeError(
            f"DFLASH invalid speculative_num_draft_tokens={block_size}."
        )
    return block_size


def _validate_flat_block_tensor(
    name: str,
    tensor: Optional[torch.Tensor],
    bs: int,
    block_size: int,
    *,
    required: bool,
) -> None:
    if tensor is None:
        if required:
            raise RuntimeError(f"DFLASH spec-v2 missing {name} for bs={bs}.")
        return
    expected = bs * block_size
    got = int(tensor.numel())
    if got != expected:
        raise RuntimeError(
            f"DFLASH spec-v2 {name} shape mismatch: "
            f"bs={bs}, block_size={block_size}, expected={expected}, got={got}."
        )


def _validate_next_block_tensors(
    owner: str,
    next_candidates: Optional[torch.Tensor],
    next_positions: Optional[torch.Tensor],
    bs: int,
    block_size: int,
) -> None:
    prefix = f"{owner}." if owner else ""
    if next_positions is not None and next_candidates is None:
        raise RuntimeError(
            f"DFLASH spec-v2 {prefix}next_positions present without "
            "next_candidates."
        )
    _validate_flat_block_tensor(
        f"{prefix}next_candidates",
        next_candidates,
        bs,
        block_size,
        required=next_candidates is not None,
    )
    _validate_flat_block_tensor(
        f"{prefix}next_positions",
        next_positions,
        bs,
        block_size,
        required=next_positions is not None,
    )


def _request_tensor_rows(tensor: Optional[torch.Tensor]) -> Optional[int]:
    if tensor is None or tensor.dim() == 0:
        return None
    return int(tensor.shape[0])


def _request_index_for_tensor(
    new_indices: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    return new_indices.to(device=tensor.device, dtype=torch.long)


def _invalidate_request_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1)[:0]
    return tensor[:0].contiguous()


def _filter_request_tensor(
    name: str,
    tensor: Optional[torch.Tensor],
    new_indices: torch.Tensor,
    old_bs: int,
    *,
    allow_invalidate: bool,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    rows = _request_tensor_rows(tensor)
    new_bs = int(new_indices.numel())
    if rows == old_bs:
        return tensor[_request_index_for_tensor(new_indices, tensor)]
    if rows == new_bs:
        return tensor
    if allow_invalidate:
        return _invalidate_request_tensor(tensor)
    raise RuntimeError(
        f"DFLASH spec-v2 cannot filter {name}: rows={rows}, "
        f"old_bs={old_bs}, new_bs={new_bs}."
    )


def _merge_request_tensor(
    name: str,
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
    left_bs: int,
    right_bs: int,
    *,
    allow_invalidate: bool,
) -> Optional[torch.Tensor]:
    if left is None and right is None:
        return None
    if left is None:
        if left_bs == 0:
            return right
        if allow_invalidate and right is not None:
            return _invalidate_request_tensor(right)
        raise RuntimeError(
            f"DFLASH spec-v2 cannot merge {name}: left missing for "
            f"left_bs={left_bs}, right_bs={right_bs}."
        )
    if right is None:
        if right_bs == 0:
            return left
        if allow_invalidate:
            return _invalidate_request_tensor(left)
        raise RuntimeError(
            f"DFLASH spec-v2 cannot merge {name}: right missing for "
            f"left_bs={left_bs}, right_bs={right_bs}."
        )

    left_rows = _request_tensor_rows(left)
    right_rows = _request_tensor_rows(right)
    if left_rows == left_bs and right_rows == right_bs:
        return torch.cat([left, right], dim=0)
    if left_rows == 0 and left_bs == 0 and right_rows == right_bs:
        return right
    if right_rows == 0 and right_bs == 0 and left_rows == left_bs:
        return left
    if allow_invalidate:
        return _invalidate_request_tensor(left)
    raise RuntimeError(
        f"DFLASH spec-v2 cannot merge {name}: left_rows={left_rows}, "
        f"right_rows={right_rows}, left_bs={left_bs}, right_bs={right_bs}."
    )


def _merge_future_recomputed_tensor(
    name: str,
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
    left_bs: int,
    right_bs: int,
) -> Optional[torch.Tensor]:
    """Merge future-backed CPU metadata that can be recomputed before decode."""
    merged = _merge_request_tensor(
        name,
        left,
        right,
        left_bs,
        right_bs,
        allow_invalidate=True,
    )
    if _request_tensor_rows(merged) == left_bs + right_bs:
        return merged
    return None


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Return an optional plan stream/context for overlap scheduling prep kernels."""
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    # Legacy Eagle-shaped fields kept only for dataclass compatibility. DFLASH
    # overlap only relays verified_id/new_seq_lens through FutureMap.
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    verified_id: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_sum: Optional[int] = None
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None

    # PP DFLASH carries the drafter block one iteration ahead through the
    # standard PP output ring. Only the last PP rank produces these tensors;
    # earlier ranks consume them to build the symmetric target-verify forward.
    next_candidates: Optional[torch.Tensor] = None
    next_positions: Optional[torch.Tensor] = None

    _prepare_committed_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_planning_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    # Filled by scheduler after dispatch.
    future_indices: Optional[FutureIndices] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Spec v2 draft state itself does not change token accounting.
        return (1, 1)

    def row_count(self) -> int:
        if self.future_indices is not None:
            return int(self.future_indices.indices.numel())

        request_fields = (
            "new_seq_lens",
            "verified_id",
            "cur_allocated_seq_lens_cpu",
            "planning_seq_lens_cpu",
            "reserved_seq_lens_cpu",
            "topk_p",
            "topk_index",
            "hidden_states",
        )
        for name in request_fields:
            rows = _request_tensor_rows(getattr(self, name, None))
            if rows is None:
                continue
            if rows > 0:
                return rows
        return 0

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str
    ) -> None:
        pin_memory = is_pin_memory_available()

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or buf.is_pinned() != pin_memory

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        if needs_cpu_alloc(self._prepare_committed_kv_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_committed_kv_lens_cpu_buf)
            self._prepare_committed_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_planning_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_batch_seq_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int64, device="cpu"
            )
            self._prepare_cur_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_nxt_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )

        if needs_gpu_alloc(self._prepare_cur_kv_lens_gpu_buf):
            capacity = grown_capacity(self._prepare_cur_kv_lens_gpu_buf)
            self._prepare_cur_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )
            self._prepare_nxt_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
            verify_done=None,
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate headroom in the shared req_to_token pool for the next DFLASH step.

        DFLASH spec-v2 uses overlap scheduling's "over-allocation" approach: we reserve
        future KV slots ahead of time so the worker can gather `out_cache_loc` directly
        from `req_to_token` without allocator backup/restore. CPU metadata intentionally
        lags by one iteration; keep it separate from the reserved upper bound that backs
        the overallocated mapping.
        """
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)
        if plan_stream is None:
            # Ensure previous forward is completed before mutating shared buffers.
            phase_t = time.perf_counter() if _dflash_timeline_enabled() else None
            batch.maybe_wait_verify_done()
            _dflash_log_timeline(
                "dflash.plan.wait_verify",
                phase_t,
                bs=batch.batch_size(),
                plan_stream=False,
            )

        bs = batch.batch_size()
        if bs == 0:
            return
        prepare_start_t = time.perf_counter() if _dflash_timeline_enabled() else None
        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_committed_kv_lens_cpu_buf is not None
        assert self._prepare_planning_kv_lens_cpu_buf is not None
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        committed_kv_lens_cpu_t = self._prepare_committed_kv_lens_cpu_buf[:bs]
        planning_kv_lens_cpu_t = self._prepare_planning_kv_lens_cpu_buf[:bs]
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]
        cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = _dflash_block_size()

        page_size = batch.token_to_kv_pool_allocator.page_size
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]
        committed_seq_lens_sum = 0
        planning_seq_lens_sum = 0
        reserved_seq_lens_sum = 0
        num_needed_tokens = 0
        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        for i, req in enumerate(batch.reqs):
            committed_len = int(req.kv_committed_len)
            if cur_allocated_seq_lens_cpu is not None and i < len(
                cur_allocated_seq_lens_cpu
            ):
                cur_alloc_len = int(cur_allocated_seq_lens_cpu[i])
            else:
                cur_alloc_len = int(req.kv_allocated_len)
            planning_len = committed_len + block_size
            reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
            top_k = int(req.sampling_params.top_k)

            committed_kv_lens_cpu_t[i] = committed_len
            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur_alloc_len
            planning_kv_lens_cpu_t[i] = planning_len
            nxt_kv_lens_cpu_t[i] = reserved_len

            committed_seq_lens_sum += committed_len
            planning_seq_lens_sum += planning_len
            reserved_seq_lens_sum += reserved_len
            num_needed_tokens += reserved_len - cur_alloc_len

            if top_k > max_top_k:
                max_top_k = top_k
            if i == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max(max_top_k, 1)
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        with plan_stream_ctx:
            if plan_stream is not None and caller_stream is not None:
                # `batch.seq_lens`, `batch.req_pool_indices`, and related tensors may
                # have just been rebuilt on the scheduler stream by filter/merge ops.
                # The plan stream must wait for those writes before reading them.
                plan_stream.wait_stream(caller_stream)

            if plan_stream is not None and self.verify_done is not None:
                phase_t = time.perf_counter() if _dflash_timeline_enabled() else None
                plan_stream.wait_event(self.verify_done)
                _dflash_log_timeline(
                    "dflash.plan.wait_verify",
                    phase_t,
                    bs=bs,
                    plan_stream=True,
                )

            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)

            if num_needed_tokens > 0:
                phase_t = time.perf_counter() if _dflash_timeline_enabled() else None
                if page_size == 1:
                    out_cache_loc = alloc_token_slots(
                        batch.tree_cache, num_needed_tokens
                    )
                else:
                    last_loc = get_last_loc(
                        batch.req_to_token_pool.req_to_token,
                        batch.req_pool_indices,
                        cur_kv_lens,
                    )
                    out_cache_loc = alloc_paged_token_slots_extend(
                        batch.tree_cache,
                        cur_kv_lens,
                        cur_kv_lens_cpu_t,
                        nxt_kv_lens,
                        nxt_kv_lens_cpu_t,
                        last_loc,
                        num_needed_tokens,
                    )
                _dflash_log_timeline(
                    "dflash.plan.alloc_kv",
                    phase_t,
                    bs=bs,
                    needed=num_needed_tokens,
                    page_size=page_size,
                )

                # Updating req_to_token is a write to a shared tensor: it must not overlap
                # with the previous batch's forward, which also reads req_to_token.
                phase_t = time.perf_counter() if _dflash_timeline_enabled() else None
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    cur_kv_lens,
                    nxt_kv_lens,
                    out_cache_loc,
                    bs,
                )
                _dflash_log_timeline(
                    "dflash.plan.assign_req_to_token",
                    phase_t,
                    bs=bs,
                    needed=num_needed_tokens,
                )
        if caller_stream is not None:
            # Enqueue the dependency on the caller's stream, not inside the
            # plan-stream context, so forward work cannot observe partially
            # prepared req_to_token / KV allocation state.
            caller_stream.wait_stream(plan_stream)

        for i, req in enumerate(batch.reqs):
            req.kv_allocated_len = int(nxt_kv_lens_cpu_t[i])

        # Preserve the lagging committed CPU view on the batch and carry the
        # tighter host-side planning bound separately from the full reserved
        # allocator upper bound. Overlap scheduling only drifts by at most one
        # DFlash block on the committed prefix lengths.
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.planning_seq_lens_cpu = planning_kv_lens_cpu_t
        self.planning_seq_lens_sum = planning_seq_lens_sum
        self.reserved_seq_lens_cpu = nxt_kv_lens_cpu_t
        self.reserved_seq_lens_sum = reserved_seq_lens_sum
        _dflash_log_timeline(
            "dflash.plan.prepare_for_decode",
            prepare_start_t,
            bs=bs,
            block_size=block_size,
            needed=num_needed_tokens,
            committed_sum=committed_seq_lens_sum,
            planning_sum=planning_seq_lens_sum,
            reserved_sum=reserved_seq_lens_sum,
            plan_stream=plan_stream is not None,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_bs = self.row_count()
        future_backed = self.future_indices is not None

        self.cur_allocated_seq_lens_cpu = _filter_request_tensor(
            "cur_allocated_seq_lens_cpu",
            self.cur_allocated_seq_lens_cpu,
            new_indices,
            old_bs,
            allow_invalidate=False,
        )
        self.planning_seq_lens_cpu = _filter_request_tensor(
            "planning_seq_lens_cpu",
            self.planning_seq_lens_cpu,
            new_indices,
            old_bs,
            allow_invalidate=False,
        )
        if self.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        else:
            self.planning_seq_lens_sum = None
        self.reserved_seq_lens_cpu = _filter_request_tensor(
            "reserved_seq_lens_cpu",
            self.reserved_seq_lens_cpu,
            new_indices,
            old_bs,
            allow_invalidate=False,
        )
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        else:
            self.reserved_seq_lens_sum = None

        if self.next_candidates is not None or self.next_positions is not None:
            block_size = _dflash_block_size()
            _validate_next_block_tensors(
                "",
                self.next_candidates,
                self.next_positions,
                old_bs,
                block_size,
            )
            if self.next_candidates is not None:
                candidate_indices = _request_index_for_tensor(
                    new_indices, self.next_candidates
                )
                self.next_candidates = (
                    self.next_candidates.view(old_bs, block_size)[candidate_indices]
                    .reshape(-1)
                    .contiguous()
                )
            if self.next_positions is not None:
                position_indices = _request_index_for_tensor(
                    new_indices, self.next_positions
                )
                self.next_positions = (
                    self.next_positions.view(old_bs, block_size)[position_indices]
                    .reshape(-1)
                    .contiguous()
                )

        if self.future_indices is not None:
            self.future_indices = FutureIndices(
                indices=self.future_indices.indices[
                    _request_index_for_tensor(new_indices, self.future_indices.indices)
                ]
            )

        self.topk_p = _filter_request_tensor(
            "topk_p",
            self.topk_p,
            new_indices,
            old_bs,
            allow_invalidate=future_backed,
        )
        self.topk_index = _filter_request_tensor(
            "topk_index",
            self.topk_index,
            new_indices,
            old_bs,
            allow_invalidate=future_backed,
        )
        self.verified_id = _filter_request_tensor(
            "verified_id",
            self.verified_id,
            new_indices,
            old_bs,
            allow_invalidate=future_backed,
        )
        self.new_seq_lens = _filter_request_tensor(
            "new_seq_lens",
            self.new_seq_lens,
            new_indices,
            old_bs,
            allow_invalidate=future_backed,
        )
        self.hidden_states = _filter_request_tensor(
            "hidden_states",
            self.hidden_states,
            new_indices,
            old_bs,
            allow_invalidate=future_backed,
        )

        if self.next_candidates is not None or self.next_positions is not None:
            _validate_next_block_tensors(
                "",
                self.next_candidates,
                self.next_positions,
                self.row_count(),
                block_size,
            )

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        left_bs = self.row_count()
        right_bs = spec_info.row_count()

        if left_bs == 0:
            for name in self.__dataclass_fields__:
                setattr(self, name, getattr(spec_info, name))
            return
        if right_bs == 0:
            return

        future_backed = self.future_indices is not None
        other_future_backed = spec_info.future_indices is not None
        if future_backed != other_future_backed:
            raise RuntimeError(
                "DFLASH spec-v2 cannot merge future-backed and concrete batches."
            )

        self.cur_allocated_seq_lens_cpu = _merge_request_tensor(
            "cur_allocated_seq_lens_cpu",
            self.cur_allocated_seq_lens_cpu,
            spec_info.cur_allocated_seq_lens_cpu,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )

        if future_backed:
            # `planning_*` and `reserved_*` are host-side planning mirrors. They
            # may be absent on a newly-prefilled future-backed batch and are
            # recomputed for the merged request set by `prepare_for_decode`.
            self.planning_seq_lens_cpu = _merge_future_recomputed_tensor(
                "planning_seq_lens_cpu",
                self.planning_seq_lens_cpu,
                spec_info.planning_seq_lens_cpu,
                left_bs,
                right_bs,
            )
        else:
            self.planning_seq_lens_cpu = _merge_request_tensor(
                "planning_seq_lens_cpu",
                self.planning_seq_lens_cpu,
                spec_info.planning_seq_lens_cpu,
                left_bs,
                right_bs,
                allow_invalidate=False,
            )
        if self.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        else:
            self.planning_seq_lens_sum = None

        if future_backed:
            self.reserved_seq_lens_cpu = _merge_future_recomputed_tensor(
                "reserved_seq_lens_cpu",
                self.reserved_seq_lens_cpu,
                spec_info.reserved_seq_lens_cpu,
                left_bs,
                right_bs,
            )
        else:
            self.reserved_seq_lens_cpu = _merge_request_tensor(
                "reserved_seq_lens_cpu",
                self.reserved_seq_lens_cpu,
                spec_info.reserved_seq_lens_cpu,
                left_bs,
                right_bs,
                allow_invalidate=False,
            )
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        else:
            self.reserved_seq_lens_sum = None

        # DFLASH PP carries the next verify block as concrete tensors outside
        # the FutureMap. Merge them even when the rest of the spec state is
        # represented by future indices.
        self_has_candidates = self.next_candidates is not None
        other_has_candidates = spec_info.next_candidates is not None
        self_has_positions = self.next_positions is not None
        other_has_positions = spec_info.next_positions is not None
        if (
            self_has_candidates
            or other_has_candidates
            or self_has_positions
            or other_has_positions
        ):
            block_size = _dflash_block_size()
            _validate_next_block_tensors(
                "left",
                self.next_candidates,
                self.next_positions,
                left_bs,
                block_size,
            )
            _validate_next_block_tensors(
                "right",
                spec_info.next_candidates,
                spec_info.next_positions,
                right_bs,
                block_size,
            )
        if self_has_candidates != other_has_candidates:
            raise RuntimeError(
                "DFLASH spec-v2 cannot merge batches with mismatched "
                f"next_candidates presence: left={self_has_candidates}, "
                f"right={other_has_candidates}."
            )
        if self_has_positions != other_has_positions:
            raise RuntimeError(
                "DFLASH spec-v2 cannot merge batches with mismatched "
                f"next_positions presence: left={self_has_positions}, "
                f"right={other_has_positions}."
            )
        if self_has_positions and not self_has_candidates:
            raise RuntimeError(
                "DFLASH spec-v2 cannot merge next_positions without candidates."
            )
        if self_has_candidates:
            self.next_candidates = torch.cat(
                [self.next_candidates, spec_info.next_candidates], dim=0
            )
            if self_has_positions:
                self.next_positions = torch.cat(
                    [self.next_positions, spec_info.next_positions], dim=0
                )
            else:
                self.next_positions = None
        else:
            self.next_candidates = None
            self.next_positions = None

        if future_backed:
            self.topk_p = _merge_request_tensor(
                "topk_p",
                self.topk_p,
                spec_info.topk_p,
                left_bs,
                right_bs,
                allow_invalidate=True,
            )
            self.topk_index = _merge_request_tensor(
                "topk_index",
                self.topk_index,
                spec_info.topk_index,
                left_bs,
                right_bs,
                allow_invalidate=True,
            )
            self.verified_id = _merge_request_tensor(
                "verified_id",
                self.verified_id,
                spec_info.verified_id,
                left_bs,
                right_bs,
                allow_invalidate=True,
            )
            self.new_seq_lens = _merge_request_tensor(
                "new_seq_lens",
                self.new_seq_lens,
                spec_info.new_seq_lens,
                left_bs,
                right_bs,
                allow_invalidate=True,
            )
            self.hidden_states = _merge_request_tensor(
                "hidden_states",
                self.hidden_states,
                spec_info.hidden_states,
                left_bs,
                right_bs,
                allow_invalidate=True,
            )
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        self.topk_p = _merge_request_tensor(
            "topk_p",
            self.topk_p,
            spec_info.topk_p,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )
        self.topk_index = _merge_request_tensor(
            "topk_index",
            self.topk_index,
            spec_info.topk_index,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )
        self.verified_id = _merge_request_tensor(
            "verified_id",
            self.verified_id,
            spec_info.verified_id,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )
        self.new_seq_lens = _merge_request_tensor(
            "new_seq_lens",
            self.new_seq_lens,
            spec_info.new_seq_lens,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )
        self.hidden_states = _merge_request_tensor(
            "hidden_states",
            self.hidden_states,
            spec_info.hidden_states,
            left_bs,
            right_bs,
            allow_invalidate=False,
        )
