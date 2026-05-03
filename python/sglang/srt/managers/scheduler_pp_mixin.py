from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.io_struct import ExpertDistributionReq
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _dflash_stable_rid_hash(rid: str) -> int:
    digest = hashlib.blake2b(str(rid).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def _dflash_env_enabled(name: str) -> bool:
    return os.getenv(name) in ("1", "true", "TRUE")


def _dflash_timeline_enabled() -> bool:
    return _dflash_env_enabled("SGLANG_DFLASH_PP_TIMELINE")


def _dflash_route_timing_enabled() -> bool:
    return _dflash_env_enabled("SGLANG_DFLASH_PP_ROUTE_TIMING")


_DFLASH_ROUTE_TIMING_PHASES = {
    "pp.select_batch",
    "pp.proxy.recv",
    "pp.proxy.send",
    "pp.d2h.sync",
    "pp.process_result",
    "pp.output_ring.wait_ready",
    "pp.output_ring.send",
    "pp.output_ring.recv",
    "pp.output_ring.prep_result",
    "pp.run_batch",
    "pp.output_ring.result_pending",
    "pp.output_ring.needs_forward",
    "pp.output_ring.forwarded",
    "pp.output_ring.result_drain",
    "pp.output_ring.forward_drain",
    "pp.output_intent.arbitrate",
    "pp.coalesce.skip",
    "pp.coalesce.owner",
    "pp.proxy.recv.defer",
    "pp.idle.defer",
}

_DFLASH_ROUTE_TIMING_EMPTY_PHASES = {
    "pp.output_ring.result_pending",
    "pp.output_ring.needs_forward",
    "pp.output_ring.forwarded",
    "pp.output_ring.result_drain",
    "pp.output_ring.forward_drain",
    "pp.output_intent.arbitrate",
    "pp.coalesce.skip",
    "pp.coalesce.owner",
    "pp.idle.defer",
}


def _pp_needs_prehandle_forward(req) -> bool:
    return isinstance(req, ExpertDistributionReq)


def _dflash_tensor_shape(tensor: Optional[torch.Tensor]):
    return None if tensor is None else tuple(tensor.shape)


def _dflash_tensor_scalar_int(
    name: str, tensor: Optional[torch.Tensor]
) -> Optional[int]:
    if tensor is None:
        return None
    if tensor.ndim != 0:
        raise RuntimeError(
            f"DFLASH PP {name} metadata must be scalar, "
            f"got shape={_dflash_tensor_shape(tensor)}."
        )
    return int(tensor.detach().to("cpu").item())


def _dflash_forward_mode_name(mode: int) -> str:
    try:
        return ForwardMode(mode).name
    except ValueError:
        return str(mode)


def _dflash_batch_rids(batch: Optional[ScheduleBatch]) -> List[str]:
    if batch is None or batch.is_empty():
        return []
    return [str(req.rid) for req in batch.reqs]


def _dflash_batch_rid_hashes(batch: Optional[ScheduleBatch]) -> List[int]:
    if batch is None or batch.is_empty():
        return []
    return [_dflash_stable_rid_hash(req.rid) for req in batch.reqs]


def _dflash_batch_token_count(batch: Optional[ScheduleBatch]) -> Optional[int]:
    if batch is None or batch.is_empty():
        return 0
    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is not None and forward_mode.is_decode_or_idle():
        return batch.batch_size()
    input_ids = getattr(batch, "input_ids", None)
    if input_ids is not None:
        return int(input_ids.numel())
    extend_num_tokens = getattr(batch, "extend_num_tokens", None)
    if extend_num_tokens is not None:
        return int(extend_num_tokens)
    return None


def _dflash_log_timeline(
    scheduler,
    phase: str,
    *,
    mb_id: Optional[int] = None,
    batch: Optional[ScheduleBatch] = None,
    metadata: Optional["PPBatchMetadata"] = None,
    start_time: Optional[float] = None,
    **fields,
) -> None:
    timeline_enabled = _dflash_timeline_enabled()
    route_timing_only = _dflash_route_timing_enabled() and not timeline_enabled
    if not timeline_enabled and not route_timing_only:
        return
    if route_timing_only and phase not in _DFLASH_ROUTE_TIMING_PHASES:
        return
    if (
        route_timing_only
        and (batch is None or batch.is_empty())
        and metadata is None
        and phase not in _DFLASH_ROUTE_TIMING_EMPTY_PHASES
    ):
        return
    if (
        timeline_enabled
        and not _dflash_env_enabled("SGLANG_DFLASH_PP_TIMELINE_IDLE")
        and (batch is None or batch.is_empty())
        and int(fields.get("recv", 0) or 0) == 0
        and int(fields.get("waiting", 0) or 0) == 0
        and phase not in ("pp.coalesce.skip",)
    ):
        return

    elapsed_ms = None
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    dispatch_seq = None
    token_count = _dflash_batch_token_count(batch)
    if metadata is not None:
        dispatch_seq = metadata.dispatch_seq
        if metadata.token_count >= 0:
            token_count = metadata.token_count

    parts = [
        f"phase={phase}",
        f"pp={getattr(scheduler, 'pp_rank', None)}",
        f"mb={mb_id}",
        f"dispatch={dispatch_seq}",
        f"mode={batch.forward_mode.name if batch is not None else None}",
        f"bs={batch.batch_size() if batch is not None else 0}",
        f"tokens={token_count}",
        f"rid_hashes={_dflash_batch_rid_hashes(batch)}",
    ]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.3f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    log_kind = "route_timing" if route_timing_only else "timeline"
    logger.info("DFLASH PP %s %s", log_kind, " ".join(parts))


def _dflash_running_mbs_summary(running_mbs: List[ScheduleBatch]):
    return [
        {
            "mb": i,
            "bs": batch.batch_size(),
            "rids": _dflash_batch_rids(batch),
        }
        for i, batch in enumerate(running_mbs)
        if batch is not None and not batch.is_empty()
    ]


def _dflash_batch_has_unfinished_reqs(batch: Optional[ScheduleBatch]) -> bool:
    if batch is None or batch.is_empty():
        return False
    return any(not req.finished() and not req.is_retracted for req in batch.reqs)


def _dflash_slot_live_summary(batches: List[Optional[ScheduleBatch]]):
    return [
        {
            "mb": i,
            "bs": batch.batch_size(),
            "rids": _dflash_batch_rids(batch),
        }
        for i, batch in enumerate(batches)
        if _dflash_batch_has_unfinished_reqs(batch)
    ]


def _dflash_validate_pp_payload(
    *,
    bs: int,
    block_size: int,
    dflash_next_candidates: Optional[torch.Tensor],
    dflash_next_positions: Optional[torch.Tensor],
    dflash_commit_lens: Optional[torch.Tensor],
    dflash_committed_tokens: Optional[torch.Tensor],
    dflash_rid_hashes: Optional[torch.Tensor],
) -> None:
    expected_tokens = bs * block_size
    if dflash_rid_hashes is not None and int(dflash_rid_hashes.numel()) != bs:
        raise RuntimeError(
            "DFLASH PP payload request-id hash count mismatch: "
            f"expected={bs}, got={int(dflash_rid_hashes.numel())}."
        )
    if dflash_next_candidates is not None:
        got = int(dflash_next_candidates.numel())
        if got != expected_tokens:
            raise RuntimeError(
                "DFLASH PP next-candidates shape mismatch: "
                f"bs={bs}, block_size={block_size}, expected={expected_tokens}, got={got}."
            )
    if dflash_next_positions is not None:
        if dflash_next_candidates is None:
            raise RuntimeError(
                "DFLASH PP received next_positions without next_candidates."
            )
        got = int(dflash_next_positions.numel())
        if got != expected_tokens:
            raise RuntimeError(
                "DFLASH PP next-positions shape mismatch: "
                f"bs={bs}, block_size={block_size}, expected={expected_tokens}, got={got}."
            )
    if (dflash_commit_lens is None) != (dflash_committed_tokens is None):
        raise RuntimeError(
            "DFLASH PP commit_lens and committed_tokens must be present together."
        )
    if dflash_commit_lens is not None:
        got = int(dflash_commit_lens.numel())
        if got != bs:
            raise RuntimeError(
                "DFLASH PP commit_lens shape mismatch: "
                f"expected={bs}, got={got}."
            )
        got = int(dflash_committed_tokens.numel())
        if got != expected_tokens:
            raise RuntimeError(
                "DFLASH PP committed_tokens shape mismatch: "
                f"bs={bs}, block_size={block_size}, expected={expected_tokens}, got={got}."
            )


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool
    dispatch_seq: int = -1
    mb_id: int = -1
    batch_size: int = 0
    forward_mode: int = -1
    rid_hashes: Tuple[int, ...] = ()
    token_count: int = -1
    mamba_cache_indices: Optional[torch.Tensor] = None


def _dflash_validate_local_pp_metadata(
    batch: ScheduleBatch, metadata: PPBatchMetadata
) -> None:
    route_mismatch = []
    local_bs = batch.batch_size()
    local_rid_hashes = tuple(_dflash_batch_rid_hashes(batch))
    local_token_count = _dflash_batch_token_count(batch)

    if metadata.batch_size != local_bs:
        route_mismatch.append(
            f"bs local={local_bs} metadata={metadata.batch_size}"
        )
    if metadata.rid_hashes != local_rid_hashes:
        route_mismatch.append(
            f"rid_hashes local={list(local_rid_hashes)} "
            f"metadata={list(metadata.rid_hashes)}"
        )
    if local_token_count is not None:
        if metadata.token_count < 0:
            route_mismatch.append(
                f"tokens local={local_token_count} metadata=missing"
            )
        elif metadata.token_count != local_token_count:
            route_mismatch.append(
                f"tokens local={local_token_count} metadata={metadata.token_count}"
            )

    if route_mismatch:
        raise RuntimeError(
            "DFLASH PP local route metadata is stale before follower commit: "
            + ", ".join(route_mismatch)
        )


def _dflash_clone_current_mamba_cache_indices(
    scheduler,
) -> Optional[torch.Tensor]:
    """Snapshot the route-local Mamba request indices for a later PP commit."""
    worker = (
        getattr(scheduler, "draft_worker", None)
        or getattr(scheduler, "model_worker", None)
        or getattr(scheduler, "tp_worker", None)
    )
    target_worker = getattr(worker, "target_worker", None)
    model_runner = getattr(target_worker, "model_runner", None)
    attn_backend = getattr(model_runner, "attn_backend", None)
    linear_attn_backend = getattr(attn_backend, "linear_attn_backend", None)
    forward_metadata = getattr(linear_attn_backend, "forward_metadata", None)
    mamba_cache_indices = getattr(forward_metadata, "mamba_cache_indices", None)
    if mamba_cache_indices is None:
        return None
    return mamba_cache_indices.clone()


class SchedulerPPMixin:
    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                phase_t = time.perf_counter()
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.recv_requests()
                    pp_reqs_pre_sent = False
                    if not self.pp_group.is_last_rank and recv_reqs:
                        if any(_pp_needs_prehandle_forward(req) for req in recv_reqs):
                            self._pp_commit_comm_work(self.send_req_work)
                            self.send_req_work = self._pp_send_pyobj_to_next_stage(
                                recv_reqs,
                                async_send=True,
                            )
                            self._pp_commit_comm_work(self.send_req_work)
                            pp_reqs_pre_sent = True
                    self.process_input_requests(recv_reqs)
                _dflash_log_timeline(
                    self,
                    "pp.recv_requests",
                    mb_id=mb_id,
                    start_time=phase_t,
                    recv=len(recv_reqs),
                    waiting=len(self.waiting_queue),
                )
                if not self.pp_group.is_last_rank:
                    phase_t = time.perf_counter()
                    self._pp_commit_comm_work(self.send_req_work)
                    _dflash_log_timeline(
                        self,
                        "pp.req.prev_send_wait",
                        mb_id=mb_id,
                        start_time=phase_t,
                    )
                    if not pp_reqs_pre_sent:
                        with torch.profiler.record_function("send_reqs_to_next_stage"):
                            self.send_req_work = self._pp_send_pyobj_to_next_stage(
                                recv_reqs,
                                async_send=True,
                            )
                    _dflash_log_timeline(
                        self,
                        "pp.req.send",
                        mb_id=mb_id,
                        recv=len(recv_reqs),
                    )
                phase_t = time.perf_counter()
                with torch.profiler.record_function("get_next_batch_to_run"):
                    dflash_run_control = None
                    selected_batch = None
                    if self._pp_dflash_run_control_enabled():
                        if self.pp_group.is_first_rank:
                            selected_batch = (
                                self._pp_dflash_select_authoritative_batch(mb_id)
                            )
                            dflash_run_control = self._pp_dflash_make_run_control(
                                mb_id, selected_batch
                            )
                        else:
                            dflash_run_control = self._pp_dflash_recv_run_control(
                                mb_id
                            )
                            selected_batch = self._pp_dflash_select_follower_batch(
                                mb_id, dflash_run_control
                            )
                        self._pp_dflash_validate_run_control(
                            mb_id, selected_batch, dflash_run_control
                        )
                        self._pp_dflash_forward_run_control(dflash_run_control)
                        if bool(dflash_run_control["run"]):
                            self.mbs[mb_id] = selected_batch
                    else:
                        selected_batch = self._pp_dflash_select_authoritative_batch(mb_id)
                        self.mbs[mb_id] = selected_batch
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = selected_batch
                _dflash_log_timeline(
                    self,
                    "pp.select_batch",
                    mb_id=mb_id,
                    batch=self.cur_batch,
                    start_time=phase_t,
                    waiting=len(self.waiting_queue),
                )
                if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
                    logger.info(
                        "DFLASH BS trace PP%d mb=%d selected mode=%s bs=%d "
                        "rids=%s running_mbs=%s waiting=%d",
                        self.pp_rank,
                        mb_id,
                        self.cur_batch.forward_mode.name if self.cur_batch else None,
                        self.cur_batch.batch_size() if self.cur_batch else 0,
                        _dflash_batch_rids(self.cur_batch),
                        _dflash_running_mbs_summary(self.running_mbs),
                        len(self.waiting_queue),
                    )
                dflash_async_output_prelaunch = (
                    self._pp_dflash_async_output_prelaunch_enabled()
                )
                dflash_defer_current_proxy_recv = (
                    dflash_async_output_prelaunch
                    and self.cur_batch is not None
                    and not self.pp_group.is_first_rank
                )
                pp_proxy_tensors = None
                if self.cur_batch:
                    dflash_debug = os.getenv("SGLANG_DFLASH_DEBUG") in (
                        "1",
                        "true",
                        "TRUE",
                    )
                    if dflash_debug:
                        print(
                            f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                            f"mb_id={mb_id} cur_batch_mode="
                            f"{self.cur_batch.forward_mode.name}",
                            flush=True,
                        )
                    server_is_idle = False
                    if dflash_debug:
                        print(
                            f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                            f"mb_id={mb_id} about to _pp_recv_proxy_tensors()",
                            flush=True,
                        )
                    if not dflash_defer_current_proxy_recv:
                        phase_t = time.perf_counter()
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()
                        _dflash_log_timeline(
                            self,
                            "pp.proxy.recv",
                            mb_id=mb_id,
                            batch=self.cur_batch,
                            start_time=phase_t,
                            proxy_keys=(
                                list(pp_proxy_tensors.tensors.keys())
                                if pp_proxy_tensors is not None
                                else None
                            ),
                        )
                        if dflash_debug:
                            print(
                                f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                                f"mb_id={mb_id} _pp_recv_proxy_tensors() DONE keys="
                                f"{list(pp_proxy_tensors.tensors.keys()) if pp_proxy_tensors else None}",
                                flush=True,
                            )
                        if (
                            os.getenv("SGLANG_DFLASH_PP_PROXY_PROBE")
                            in ("1", "true", "TRUE")
                            and pp_proxy_tensors is not None
                            and self.cur_batch.spec_algorithm.is_dflash()
                            and self.cur_batch.forward_mode.is_decode()
                        ):
                            hidden = pp_proxy_tensors.tensors.get("hidden_states", None)
                            if hidden is not None and hidden.numel() > 0:
                                rows = hidden[: min(int(hidden.shape[0]), 4)].float()
                                logger.info(
                                    "DFLASH PP%d recv proxy hidden shape=%s row_sums=%s row_norms=%s",
                                    self.pp_rank,
                                    tuple(hidden.shape),
                                    rows.sum(dim=1).detach().cpu().tolist(),
                                    rows.norm(dim=1).detach().cpu().tolist(),
                                )
                    else:
                        _dflash_log_timeline(
                            self,
                            "pp.proxy.recv.defer",
                            mb_id=mb_id,
                            batch=self.cur_batch,
                            reason="dflash_async_output_prelaunch",
                        )
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                dflash_defer_prelaunch_output = (
                    self.server_args.pp_async_batch_depth > 0
                    and self._pp_dflash_run_control_enabled()
                    and not dflash_async_output_prelaunch
                )
                if (
                    self.server_args.pp_async_batch_depth > 0
                    and not dflash_defer_prelaunch_output
                ):
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                phase_t = time.perf_counter()
                self._pp_commit_comm_work(self.send_proxy_work)
                _dflash_log_timeline(
                    self,
                    "pp.proxy.prev_send_wait",
                    mb_id=mb_id,
                    start_time=phase_t,
                )
                if dflash_defer_current_proxy_recv:
                    phase_t = time.perf_counter()
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                    _dflash_log_timeline(
                        self,
                        "pp.proxy.recv",
                        mb_id=mb_id,
                        batch=self.cur_batch,
                        start_time=phase_t,
                        deferred=True,
                        proxy_keys=(
                            list(pp_proxy_tensors.tensors.keys())
                            if pp_proxy_tensors is not None
                            else None
                        ),
                    )
                    if (
                        os.getenv("SGLANG_DFLASH_PP_PROXY_PROBE")
                        in ("1", "true", "TRUE")
                        and pp_proxy_tensors is not None
                        and self.cur_batch.spec_algorithm.is_dflash()
                        and self.cur_batch.forward_mode.is_decode()
                    ):
                        hidden = pp_proxy_tensors.tensors.get("hidden_states", None)
                        if hidden is not None and hidden.numel() > 0:
                            rows = hidden[: min(int(hidden.shape[0]), 4)].float()
                            logger.info(
                                "DFLASH PP%d recv proxy hidden shape=%s row_sums=%s row_norms=%s",
                                self.pp_rank,
                                tuple(hidden.shape),
                                rows.sum(dim=1).detach().cpu().tolist(),
                                rows.norm(dim=1).detach().cpu().tolist(),
                            )
                if self.cur_batch:
                    if dflash_debug:
                        print(
                            f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                            f"mb_id={mb_id} about to _pp_launch_batch (run_batch)",
                            flush=True,
                        )
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                    if dflash_debug:
                        print(
                            f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                            f"mb_id={mb_id} _pp_launch_batch DONE",
                            flush=True,
                        )
                    if (
                        os.getenv("SGLANG_DFLASH_PP_PROXY_PROBE") in ("1", "true", "TRUE")
                        and self.cur_batch.spec_algorithm.is_dflash()
                        and self.cur_batch.forward_mode.is_decode()
                        and result.pp_hidden_states_proxy_tensors is not None
                    ):
                        hidden = result.pp_hidden_states_proxy_tensors.tensors.get(
                            "hidden_states", None
                        )
                        if hidden is not None and hidden.numel() > 0:
                            rows = hidden[: min(int(hidden.shape[0]), 4)].float()
                            logger.info(
                                "DFLASH PP%d send proxy hidden shape=%s row_sums=%s row_norms=%s",
                                self.pp_rank,
                                tuple(hidden.shape),
                                rows.sum(dim=1).detach().cpu().tolist(),
                                rows.norm(dim=1).detach().cpu().tolist(),
                            )
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                            )
                        _dflash_log_timeline(
                            self,
                            "pp.proxy.send",
                            mb_id=mb_id,
                            batch=self.cur_batch,
                            metadata=self.mb_metadata[mb_id],
                        )
                        if dflash_debug:
                            print(
                                f"[DFLASH-DEBUG PP{self.pp_rank}] sched_loop "
                                f"mb_id={mb_id} send_proxy queued (async)",
                                flush=True,
                            )

                if (
                    self.server_args.pp_async_batch_depth == 0
                    or dflash_defer_prelaunch_output
                ):
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                result_mb_id = getattr(self, "pp_output_result_mb_id", next_mb_id)
                if self.mbs[result_mb_id] is not None and d2h_event is not None:
                    server_is_idle = False
                    phase_t = time.perf_counter()
                    d2h_event.synchronize()
                    _dflash_log_timeline(
                        self,
                        "pp.d2h.sync",
                        mb_id=result_mb_id,
                        batch=self.mbs[result_mb_id],
                        metadata=self.mb_metadata[result_mb_id],
                        start_time=phase_t,
                    )
                    phase_t = time.perf_counter()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[result_mb_id],
                            next_batch_result,
                        )
                    _dflash_log_timeline(
                        self,
                        "pp.process_result",
                        mb_id=result_mb_id,
                        batch=self.mbs[result_mb_id],
                        metadata=self.mb_metadata[result_mb_id],
                        start_time=phase_t,
                    )
                    self.last_mbs[result_mb_id] = self.mbs[result_mb_id]

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states.
            # DFlash PP can have drain-only iterations where no new batch is
            # selected, but output-ring tensors are still queued or a prior
            # microbatch result has not made it back to rank 0 yet. Those
            # allocations are live, not leaks, so defer the idle checker until
            # the ring is empty.
            if server_is_idle and self._pp_dflash_has_pending_ring_work():
                _dflash_log_timeline(
                    self,
                    "pp.idle.defer",
                    pending_results=sorted(
                        getattr(self, "dflash_pp_pending_result_mb_ids", set())
                    ),
                    pending_forward=sorted(
                        getattr(self, "dflash_pp_pending_forward_mb_ids", set())
                    ),
                    last_rank_queue=len(getattr(self, "last_rank_comm_queue", [])),
                    has_pp_outputs=getattr(self, "pp_outputs", None) is not None,
                    send_output_work=len(getattr(self, "send_output_work", [])),
                    send_proxy_work=len(getattr(self, "send_proxy_work", [])),
                    send_run_control_work=len(
                        getattr(self, "send_dflash_run_control_work", [])
                    ),
                    send_output_intent_work=len(
                        getattr(self, "send_dflash_output_intent_work", [])
                    ),
                    live_running=_dflash_slot_live_summary(
                        getattr(self, "running_mbs", [])
                    ),
                    live_mbs=_dflash_slot_live_summary(getattr(self, "mbs", [])),
                    live_last=_dflash_slot_live_summary(
                        getattr(self, "last_mbs", [])
                    ),
                )
                server_is_idle = False

            if server_is_idle:
                self.self_check_during_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk()
                batch = self.get_new_batch_prefill()
                batch = self.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.self_check_during_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                batch = self.get_next_disagg_decode_batch_to_run()
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not self.cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch and not self.cur_batch.forward_mode.is_prebuilt():
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.self_check_during_idle()

    def init_pp_loop_state(self: Scheduler):
        self.pp_loop_size: int = self.pp_size + self.server_args.pp_async_batch_depth
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        self.require_attn_tp_allgather = (
            not self.server_args.enable_nsa_prefill_context_parallel
        )
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]] = (
            deque()
        )

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.send_dflash_run_control_work = []
        self.send_dflash_output_intent_work = []
        self.launch_event = None
        self.dflash_pp_prefill_hold_until = 0.0
        self.dflash_pp_dispatch_seq = 0
        self.dflash_pp_unsafe_coalesce_warned = False
        self.dflash_pp_pending_result_mb_ids = set()
        self.dflash_pp_pending_forward_mb_ids = set()
        self.dflash_pp_sticky_chunked_req_mb_id = None
        self.dflash_pp_sticky_chunked_req_hash = None

    def _pp_dflash_has_pending_ring_work(self: Scheduler) -> bool:
        if not self._pp_dflash_run_control_enabled():
            return False
        if getattr(self, "dflash_pp_pending_result_mb_ids", None):
            return True
        if getattr(self, "dflash_pp_pending_forward_mb_ids", None):
            return True
        if getattr(self, "pp_outputs", None) is not None:
            return True
        if len(getattr(self, "last_rank_comm_queue", [])) > 0:
            return True
        if len(getattr(self, "send_output_work", [])) > 0:
            return True
        if len(getattr(self, "send_proxy_work", [])) > 0:
            return True
        for attr in ("running_mbs", "mbs", "last_mbs"):
            batches = getattr(self, attr, None)
            if batches is not None and any(
                _dflash_batch_has_unfinished_reqs(batch) for batch in batches
            ):
                return True
        return False

    def _pp_dflash_pipeline_slots_enabled(self: Scheduler) -> bool:
        return self._pp_dflash_run_control_enabled() and _dflash_env_enabled(
            "SGLANG_DFLASH_PP_PIPELINE_SLOTS"
        )

    def _pp_dflash_async_output_prelaunch_enabled(self: Scheduler) -> bool:
        """Allow async-depth output drain before current DFlash PP proxy recv.

        In PP=2, the previous DFlash async-depth ordering deadlocked because
        rank 0 waited for rank 1's output intent while rank 1 was already
        waiting for rank 0's current-route proxy tensors. This opt-in keeps the
        output-intent prelaunch ordering but delays the follower's current proxy
        receive until after that intent exchange. Default behavior stays on the
        conservative post-launch drain path. Keep this disabled for separate
        pipeline slots until their request/KV lifecycle is fixed.
        """
        return (
            self.server_args.pp_async_batch_depth > 0
            and self.pp_size == 2
            and self._pp_dflash_run_control_enabled()
            and not self._pp_dflash_pipeline_slots_enabled()
            and _dflash_env_enabled("SGLANG_DFLASH_PP_ASYNC_OUTPUT_PRELAUNCH")
        )

    def _pp_dflash_sticky_chunked_prefill_enabled(self: Scheduler) -> bool:
        return (
            self._pp_dflash_run_control_enabled()
            and _dflash_env_enabled("SGLANG_DFLASH_PP_STICKY_CHUNKED_PREFILL")
            and _dflash_env_enabled("SGLANG_DFLASH_PP_COALESCE")
            and not self._pp_dflash_pipeline_slots_enabled()
        )

    def _pp_dflash_note_selected_batch(
        self: Scheduler, mb_id: int, batch: Optional[ScheduleBatch]
    ) -> None:
        if not self._pp_dflash_sticky_chunked_prefill_enabled():
            return

        chunked_req = getattr(self, "chunked_req", None)
        if chunked_req is None:
            self.dflash_pp_sticky_chunked_req_mb_id = None
            self.dflash_pp_sticky_chunked_req_hash = None
            return

        batch_chunked_req = getattr(batch, "chunked_req", None)
        forward_mode = getattr(batch, "forward_mode", None)
        if (
            batch is None
            or forward_mode is None
            or not forward_mode.is_extend()
            or batch_chunked_req is None
        ):
            return

        chunked_req_hash = _dflash_stable_rid_hash(batch_chunked_req.rid)
        self.dflash_pp_sticky_chunked_req_mb_id = mb_id
        self.dflash_pp_sticky_chunked_req_hash = chunked_req_hash
        _dflash_log_timeline(
            self,
            "pp.coalesce.owner",
            mb_id=mb_id,
            batch=batch,
            reason="sticky_chunked_prefill_owner",
            owner_mb=mb_id,
            chunked_rid_hash=chunked_req_hash,
        )

    def _pp_should_skip_slot_for_dflash_sticky_chunked_prefill(
        self: Scheduler, mb_id: int
    ) -> bool:
        if not self._pp_dflash_sticky_chunked_prefill_enabled():
            return False
        if not self.pp_group.is_first_rank:
            return False

        chunked_req = getattr(self, "chunked_req", None)
        if chunked_req is None:
            self.dflash_pp_sticky_chunked_req_mb_id = None
            self.dflash_pp_sticky_chunked_req_hash = None
            return False

        owner_mb = getattr(self, "dflash_pp_sticky_chunked_req_mb_id", None)
        if owner_mb is None:
            return False
        if owner_mb < 0 or owner_mb >= getattr(self, "pp_loop_size", 0):
            self.dflash_pp_sticky_chunked_req_mb_id = None
            self.dflash_pp_sticky_chunked_req_hash = None
            return False

        chunked_req_hash = _dflash_stable_rid_hash(chunked_req.rid)
        owner_hash = getattr(self, "dflash_pp_sticky_chunked_req_hash", None)
        if owner_hash is not None and owner_hash != chunked_req_hash:
            # A stale owner from an earlier request should never hold the loop.
            self.dflash_pp_sticky_chunked_req_mb_id = None
            self.dflash_pp_sticky_chunked_req_hash = None
            return False

        if owner_mb == mb_id:
            return False

        if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
            logger.info(
                "DFLASH PP sticky chunked prefill skip PP%d empty_mb=%d "
                "owner_mb=%d chunked_rid=%s",
                self.pp_rank,
                mb_id,
                owner_mb,
                chunked_req.rid,
            )
        _dflash_log_timeline(
            self,
            "pp.coalesce.skip",
            mb_id=mb_id,
            waiting=len(self.waiting_queue),
            reason="sticky_chunked_prefill",
            owner_mb=owner_mb,
            chunked_rid_hash=chunked_req_hash,
        )
        return True

    def _pp_should_skip_slot_for_dflash_coalesce(self: Scheduler, mb_id: int) -> bool:
        """Leave an empty PP slot idle so DFlash can form larger PP batches.

        This is safe only when one rank owns the RUN/IDLE cadence. For DFlash
        PP the first rank sends authoritative run-control metadata before the
        follower selects a batch, so PP0 may intentionally leave an empty slot
        idle while followers simply obey the RUN=false control. Without that
        controller, local coalescing remains behind an explicit unsafe opt-in.
        """
        if not _dflash_env_enabled("SGLANG_DFLASH_PP_COALESCE"):
            return False
        authoritative_pp_control = self._pp_dflash_run_control_enabled()
        if authoritative_pp_control and not self.pp_group.is_first_rank:
            return False
        if not authoritative_pp_control:
            if not _dflash_env_enabled("SGLANG_DFLASH_PP_ALLOW_UNSAFE_LOCAL_COALESCE"):
                if not getattr(self, "dflash_pp_unsafe_coalesce_warned", False):
                    logger.warning(
                        "Ignoring SGLANG_DFLASH_PP_COALESCE because local DFlash PP "
                        "RUN/IDLE coalescing can desynchronize PP route cadence. "
                        "Set SGLANG_DFLASH_PP_ALLOW_UNSAFE_LOCAL_COALESCE=1 to "
                        "run this experimental path."
                    )
                    self.dflash_pp_unsafe_coalesce_warned = True
                return False
        if self.pp_size <= 1 or not self.spec_algorithm.is_dflash():
            return False
        if self._pp_dflash_pipeline_slots_enabled():
            return False
        if len(self.waiting_queue) == 0:
            return False

        current = self.running_mbs[mb_id]
        if current is not None and not current.is_empty():
            return False

        max_bs = self.server_args.pp_max_micro_batch_size
        if max_bs is None or max_bs <= 1:
            return False

        if any(str(req.rid).startswith("HEALTH_CHECK_") for req in self.waiting_queue):
            self.dflash_pp_prefill_hold_until = 0.0
            return False

        active_batches = [
            batch
            for batches in (self.running_mbs, self.mbs, self.last_mbs)
            for batch in batches
            if batch is not None and not batch.is_empty()
        ]
        waiting_bs = len(self.waiting_queue)
        if not active_batches and waiting_bs < max_bs:
            now = time.monotonic()
            hold_ms = float(os.getenv("SGLANG_DFLASH_PP_PREFILL_HOLD_MS", "50"))
            if self.dflash_pp_prefill_hold_until <= 0.0:
                self.dflash_pp_prefill_hold_until = now + max(hold_ms, 0.0) / 1000.0
            if now < self.dflash_pp_prefill_hold_until:
                if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
                    logger.info(
                        "DFLASH PP coalesce hold PP%d empty_mb=%d waiting=%d "
                        "target_bs=%d hold_ms=%.1f",
                        self.pp_rank,
                        mb_id,
                        waiting_bs,
                        max_bs,
                        hold_ms,
                    )
                _dflash_log_timeline(
                    self,
                    "pp.coalesce.skip",
                    mb_id=mb_id,
                    waiting=waiting_bs,
                    reason="prefill_hold",
                    target_bs=max_bs,
                )
                return True
            self.dflash_pp_prefill_hold_until = 0.0
        else:
            self.dflash_pp_prefill_hold_until = 0.0

        if not _dflash_env_enabled("SGLANG_DFLASH_PP_COALESCE_ACTIVE"):
            return False

        # Skipping an empty slot while another PP slot is active can break the
        # output-ring send/recv cadence. Keep that experiment behind a separate
        # gate until the ring carries explicit per-slot phase markers.
        for other_id, other in enumerate(self.running_mbs):
            if other_id == mb_id or other is None or other.is_empty():
                continue
            if other.batch_size() >= max_bs or other.batch_is_full:
                continue
            if (
                other.spec_algorithm is not None
                and other.spec_algorithm.is_dflash()
            ):
                if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
                    logger.info(
                        "DFLASH PP coalesce skip PP%d empty_mb=%d target_mb=%d "
                        "target_bs=%d target_rids=%s waiting=%d",
                        self.pp_rank,
                        mb_id,
                        other_id,
                        other.batch_size(),
                        _dflash_batch_rids(other),
                        len(self.waiting_queue),
                    )
                _dflash_log_timeline(
                    self,
                    "pp.coalesce.skip",
                    mb_id=mb_id,
                    batch=other,
                    waiting=len(self.waiting_queue),
                    reason="active_batch_has_capacity",
                    active_mb=other_id,
                    target_bs=max_bs,
                )
                return True
        return False

    def _pp_dflash_run_control_enabled(self: Scheduler) -> bool:
        return (
            self.pp_size > 1
            and self.spec_algorithm is not None
            and self.spec_algorithm.is_dflash()
        )

    def _pp_dflash_select_authoritative_batch(
        self: Scheduler, mb_id: int
    ) -> Optional[ScheduleBatch]:
        if mb_id in getattr(self, "dflash_pp_pending_result_mb_ids", set()):
            _dflash_log_timeline(
                self,
                "pp.output_ring.result_drain",
                mb_id=mb_id,
                pending=sorted(self.dflash_pp_pending_result_mb_ids),
            )
            return None
        if mb_id in getattr(self, "dflash_pp_pending_forward_mb_ids", set()):
            _dflash_log_timeline(
                self,
                "pp.output_ring.forward_drain",
                mb_id=mb_id,
                pending=sorted(self.dflash_pp_pending_forward_mb_ids),
            )
            return None
        if self._pp_should_skip_slot_for_dflash_sticky_chunked_prefill(mb_id):
            return None
        if self._pp_should_skip_slot_for_dflash_coalesce(mb_id):
            return None
        selected_batch = self.get_next_batch_to_run()
        self._pp_dflash_note_selected_batch(mb_id, selected_batch)
        return selected_batch

    def _pp_dflash_select_follower_batch(
        self: Scheduler, mb_id: int, control: Dict[str, object]
    ) -> Optional[ScheduleBatch]:
        if not bool(control["run"]):
            self._pp_dflash_cleanup_follower_idle_slot(mb_id)
            return None
        if self._pp_should_skip_slot_for_dflash_coalesce(mb_id):
            raise RuntimeError(
                "DFLASH PP follower attempted a local coalesce skip for an "
                f"authoritative RUN slot: mb={mb_id}, control={control}."
            )
        return self.get_next_batch_to_run()

    def _pp_dflash_cleanup_follower_idle_slot(self: Scheduler, mb_id: int) -> None:
        current = self.running_batch
        if current is None or current.is_empty():
            return
        before_bs = current.batch_size()
        current.filter_batch(v1_spec_info_filtered=True)
        after_bs = current.batch_size()
        if current.is_empty():
            current.batch_is_full = False
        if after_bs < before_bs:
            _dflash_log_timeline(
                self,
                "pp.follower_idle.cleanup",
                mb_id=mb_id,
                before_bs=before_bs,
                after_bs=after_bs,
                remaining_rids=_dflash_batch_rids(current),
            )

    def _pp_dflash_make_run_control(
        self: Scheduler, mb_id: int, batch: Optional[ScheduleBatch]
    ) -> Dict[str, object]:
        run = batch is not None and not batch.is_empty()
        token_count = _dflash_batch_token_count(batch)
        control: Dict[str, object] = {
            "kind": "dflash_pp_run_control",
            "mb_id": int(mb_id),
            "run": bool(run),
            "batch_size": int(batch.batch_size()) if run else 0,
            "forward_mode": int(batch.forward_mode) if run else -1,
            "rid_hashes": tuple(_dflash_batch_rid_hashes(batch)) if run else (),
            "token_count": int(token_count) if token_count is not None else -1,
        }
        _dflash_log_timeline(
            self,
            "pp.run_control.make",
            mb_id=mb_id,
            batch=batch,
            control=control,
        )
        return control

    def _pp_dflash_recv_run_control(
        self: Scheduler, mb_id: int
    ) -> Dict[str, object]:
        data = self._pp_recv_pyobj_from_prev_stage()
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError(
                "DFLASH PP expected exactly one run-control object from the "
                f"previous stage for mb={mb_id}, got {data!r}."
            )
        control = data[0]
        if not isinstance(control, dict) or control.get("kind") != "dflash_pp_run_control":
            raise RuntimeError(
                "DFLASH PP received invalid run-control object from the "
                f"previous stage for mb={mb_id}: {control!r}."
            )
        if int(control.get("mb_id", -1)) != int(mb_id):
            raise RuntimeError(
                "DFLASH PP run-control mb mismatch: "
                f"local={mb_id}, control={control}."
            )
        _dflash_log_timeline(
            self,
            "pp.run_control.recv",
            mb_id=mb_id,
            control=control,
        )
        return control

    def _pp_dflash_forward_run_control(
        self: Scheduler, control: Dict[str, object]
    ) -> None:
        if self.pp_group.is_last_rank:
            return
        self._pp_commit_comm_work(self.send_dflash_run_control_work)
        self.send_dflash_run_control_work = self._pp_send_pyobj_to_next_stage(
            [control], async_send=True
        )
        _dflash_log_timeline(
            self,
            "pp.run_control.send",
            mb_id=int(control["mb_id"]),
            control=control,
        )

    def _pp_dflash_validate_run_control(
        self: Scheduler,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        control: Dict[str, object],
    ) -> None:
        local_run = batch is not None and not batch.is_empty()
        control_run = bool(control["run"])
        mismatches = []
        if local_run != control_run:
            mismatches.append(f"run local={local_run} control={control_run}")
        if local_run:
            local_token_count = _dflash_batch_token_count(batch)
            local = {
                "batch_size": int(batch.batch_size()),
                "forward_mode": int(batch.forward_mode),
                "rid_hashes": tuple(_dflash_batch_rid_hashes(batch)),
                "token_count": int(local_token_count)
                if local_token_count is not None
                else -1,
            }
            for key, value in local.items():
                if control.get(key) != value:
                    mismatches.append(
                        f"{key} local={value!r} control={control.get(key)!r}"
                    )
        if mismatches:
            raise RuntimeError(
                "DFLASH PP authoritative RUN/IDLE control mismatch: "
                f"pp={self.pp_rank}, mb={mb_id}, "
                + ", ".join(mismatches)
            )

    def _pp_dflash_make_output_intent(
        self: Scheduler, mb_id: int, will_send: bool
    ) -> Dict[str, object]:
        return {
            "kind": "dflash_pp_output_intent",
            "mb_id": int(mb_id),
            "send": bool(will_send),
        }

    def _pp_dflash_will_send_output(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> bool:
        if self.pp_group.is_last_rank:
            batch = mbs[next_first_rank_mb_id]
            return (
                batch is not None
                and not batch.forward_mode.is_prebuilt()
                and len(last_rank_comm_queue) > 0
            )
        return bool(pp_outputs)

    def _pp_dflash_forward_output_intent(
        self: Scheduler, intent: Dict[str, object]
    ) -> None:
        self._pp_commit_comm_work(self.send_dflash_output_intent_work)
        self.send_dflash_output_intent_work = self._pp_send_pyobj_to_next_stage(
            [intent], async_send=True
        )
        _dflash_log_timeline(
            self,
            "pp.output_intent.send",
            mb_id=int(intent["mb_id"]),
            send=bool(intent["send"]),
        )

    def _pp_dflash_recv_output_intent(
        self: Scheduler, mb_id: int
    ) -> Dict[str, object]:
        data = self._pp_recv_pyobj_from_prev_stage()
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError(
                "DFLASH PP expected exactly one output-intent object from the "
                f"previous stage for mb={mb_id}, got {data!r}."
            )
        intent = data[0]
        if not isinstance(intent, dict) or intent.get("kind") != "dflash_pp_output_intent":
            raise RuntimeError(
                "DFLASH PP received invalid output-intent object from the "
                f"previous stage for mb={mb_id}: {intent!r}."
            )
        _dflash_log_timeline(
            self,
            "pp.output_intent.recv",
            mb_id=mb_id,
            send=bool(intent["send"]),
            source_mb=int(intent.get("mb_id", -1)),
        )
        return intent

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list = []
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = np.random.randint(
                    0, 10000, size=chunk_size, dtype=np.int64
                ).tolist()
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.logprob_start_len = -1
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = len(req.fill_ids)

                if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                }

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with CUDA synchronization for accurate timing
                # Synchronize before starting timing to ensure clean measurement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                batch.prepare_for_extend()
                model_worker_batch = batch.get_model_worker_batch()

                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(req.fill_ids)
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    # Patch: release Mamba state before req_pool to fix leak in
                    # dynamic-chunking profile on hybrid Mamba/DeltaNet models.
                    # Without this each profile iteration leaks a Mamba slot;
                    # default pool of 17-19 slots fills around iteration 17 and
                    # the next alloc raises Not enough space for mamba cache.
                    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
                    if (
                        isinstance(self.req_to_token_pool, HybridReqToTokenPool)
                        and getattr(req, "mamba_pool_idx", None) is not None
                    ):
                        self.req_to_token_pool.free_mamba_cache(req)
                    self.req_to_token_pool.free(req)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = getattr(self, "max_prefill_tokens", None)
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    rids_to_check=good_consensus_bootstrapped_rids
                    + bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[PPProxyTensors, GenerationBatchResult, torch.cuda.Event]:
        phase_t = time.perf_counter()
        self._pp_commit_comm_work(work=self.send_output_work)
        _dflash_log_timeline(
            self,
            "pp.output_ring.prev_send_wait",
            mb_id=next_mb_id,
            batch=self.mbs[next_mb_id] if self.mbs[next_mb_id] is not None else None,
            metadata=self.mb_metadata[next_mb_id],
            start_time=phase_t,
        )
        self.pp_output_result_mb_id = next_mb_id
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.pp_rank * self.tp_size + dp_offset,
                ((self.pp_rank + 1) % self.pp_size) * self.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.pp_rank - 1) % self.pp_size) * self.tp_size + dp_offset,
                self.pp_rank * self.tp_size + dp_offset,
            )
        else:
            data = None

        if self.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler,
        result: GenerationBatchResult,
        batch: ScheduleBatch,
        metadata: Optional[PPBatchMetadata] = None,
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }

        # DFLASH spec-v1 PP=2: piggyback the next iter's drafter candidates and
        # this iter's verify-commit info onto the standard PP output ring.
        #   - next_candidates / next_positions: PP1's drafter output for the
        #     NEXT decode iter. PP0 attaches to running_batch.spec_info before
        #     the next decode forward.
        #   - commit_lens / committed_tokens: PP1's verify() result for THIS
        #     iter. PP0 mirrors verify()'s side effects (output_ids extension,
        #     KV slot free, seq_lens advance) via pp_apply_follower_commit.
        if result.dflash_next_candidates is not None:
            tensor_dict["dflash_next_candidates"] = result.dflash_next_candidates
        if result.dflash_next_positions is not None:
            tensor_dict["dflash_next_positions"] = result.dflash_next_positions
        if result.dflash_commit_lens is not None:
            tensor_dict["dflash_commit_lens"] = result.dflash_commit_lens
        if result.dflash_committed_tokens is not None:
            tensor_dict["dflash_committed_tokens"] = result.dflash_committed_tokens
            # Pack accepted-length scalar so PP0's spec metrics stay accurate.
            if result.dflash_commit_lens is not None:
                tensor_dict["dflash_num_accepted_tokens"] = torch.clamp(
                    result.dflash_commit_lens.to(torch.int32) - 1, min=0
                ).sum()
            else:
                tensor_dict["dflash_num_accepted_tokens"] = torch.tensor(
                    int(result.num_accepted_tokens),
                    dtype=torch.int32,
                    device=result.dflash_committed_tokens.device,
                )
        dflash_payload_tensor = result.dflash_next_candidates
        if dflash_payload_tensor is None:
            dflash_payload_tensor = result.dflash_commit_lens
        if dflash_payload_tensor is None:
            dflash_payload_tensor = result.dflash_committed_tokens
        if dflash_payload_tensor is not None:
            tensor_dict["dflash_rid_hashes"] = torch.tensor(
                [_dflash_stable_rid_hash(req.rid) for req in batch.reqs],
                dtype=torch.int64,
                device=dflash_payload_tensor.device,
            )
        if batch.spec_algorithm.is_dflash():
            meta_tensor = dflash_payload_tensor
            if meta_tensor is None:
                meta_tensor = result.next_token_ids
            if meta_tensor is not None and "dflash_rid_hashes" not in tensor_dict:
                tensor_dict["dflash_rid_hashes"] = torch.tensor(
                    [_dflash_stable_rid_hash(req.rid) for req in batch.reqs],
                    dtype=torch.int64,
                    device=meta_tensor.device,
                )
            if meta_tensor is not None and metadata is not None:
                tensor_dict["dflash_pp_mb_id"] = torch.tensor(
                    int(metadata.mb_id), dtype=torch.int32, device=meta_tensor.device
                )
                tensor_dict["dflash_pp_dispatch_seq"] = torch.tensor(
                    int(metadata.dispatch_seq),
                    dtype=torch.int64,
                    device=meta_tensor.device,
                )
                tensor_dict["dflash_pp_batch_size"] = torch.tensor(
                    int(metadata.batch_size), dtype=torch.int32, device=meta_tensor.device
                )
                tensor_dict["dflash_pp_token_count"] = torch.tensor(
                    int(metadata.token_count), dtype=torch.int32, device=meta_tensor.device
                )
                tensor_dict["dflash_pp_forward_mode"] = torch.tensor(
                    int(metadata.forward_mode),
                    dtype=torch.int32,
                    device=meta_tensor.device,
                )

        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
    ):
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=(
                        self.attn_tp_group if self.require_attn_tp_allgather else None
                    )
                )
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        res = self.pp_group.recv_tensor_dict(
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )
        return res

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        next_token_ids = pp_outputs["next_token_ids"]

        # DFLASH spec-v1 PP=2: pluck the next iter's drafter candidates and this
        # iter's verify commit info off the output ring. PPProxyTensors's
        # __getitem__ raises on missing keys, so look at the underlying dict.
        # Two side effects on PP0:
        #   - apply pp_apply_follower_commit so output_ids / seq_lens / KV free
        #     mirror PP1's verify() this iter (uses batch.spec_info, the
        #     prior-forward verify_input still attached).
        #   - replace batch.spec_info with the persistent DFlashDraftInput
        #     (verify-time spec_info is one-shot) and stash the next-iter
        #     candidates on it for the upcoming decode forward.
        proxy_dict = pp_outputs.tensors
        dflash_next_candidates = proxy_dict.get("dflash_next_candidates", None)
        dflash_next_positions = proxy_dict.get("dflash_next_positions", None)
        dflash_commit_lens = proxy_dict.get("dflash_commit_lens", None)
        dflash_committed_tokens = proxy_dict.get("dflash_committed_tokens", None)
        dflash_num_accepted = proxy_dict.get("dflash_num_accepted_tokens", None)
        dflash_rid_hashes = proxy_dict.get("dflash_rid_hashes", None)
        dflash_pp_mb_id = proxy_dict.get("dflash_pp_mb_id", None)
        dflash_pp_dispatch_seq = proxy_dict.get("dflash_pp_dispatch_seq", None)
        dflash_pp_batch_size = proxy_dict.get("dflash_pp_batch_size", None)
        dflash_pp_token_count = proxy_dict.get("dflash_pp_token_count", None)
        dflash_pp_forward_mode = proxy_dict.get("dflash_pp_forward_mode", None)
        dflash_accept_length_per_req_cpu = None
        is_dflash_pp_follower = (
            not batch.spec_algorithm.is_none()
            and batch.spec_algorithm.is_dflash()
            and not self.pp_group.is_last_rank
        )
        if is_dflash_pp_follower:
            payload_mb_id = _dflash_tensor_scalar_int(
                "dflash_pp_mb_id", dflash_pp_mb_id
            )
            payload_dispatch_seq = _dflash_tensor_scalar_int(
                "dflash_pp_dispatch_seq", dflash_pp_dispatch_seq
            )
            payload_bs = _dflash_tensor_scalar_int(
                "dflash_pp_batch_size", dflash_pp_batch_size
            )
            payload_token_count = _dflash_tensor_scalar_int(
                "dflash_pp_token_count", dflash_pp_token_count
            )
            payload_mode = _dflash_tensor_scalar_int(
                "dflash_pp_forward_mode", dflash_pp_forward_mode
            )
            _dflash_log_timeline(
                self,
                "pp.output_ring.payload",
                mb_id=mb_metadata.mb_id if mb_metadata is not None else None,
                batch=batch,
                metadata=mb_metadata,
                payload_mb=payload_mb_id,
                payload_dispatch=payload_dispatch_seq,
                payload_bs=payload_bs,
                payload_tokens=payload_token_count,
                payload_mode=payload_mode,
            )
            if mb_metadata is None:
                raise RuntimeError(
                    "DFLASH PP missing local route metadata before follower commit."
                )
            _dflash_validate_local_pp_metadata(batch, mb_metadata)

            required_route_metadata = {
                "dflash_pp_mb_id": payload_mb_id,
                "dflash_pp_dispatch_seq": payload_dispatch_seq,
                "dflash_pp_batch_size": payload_bs,
                "dflash_pp_token_count": payload_token_count,
                "dflash_pp_forward_mode": payload_mode,
                "dflash_rid_hashes": dflash_rid_hashes,
            }
            missing_route_metadata = [
                name
                for name, value in required_route_metadata.items()
                if value is None
            ]
            if missing_route_metadata:
                raise RuntimeError(
                    "DFLASH PP payload missing route metadata before follower "
                    f"commit: {missing_route_metadata}."
                )

            route_mismatch = []
            local_token_count = _dflash_batch_token_count(batch)
            if payload_mb_id != mb_metadata.mb_id:
                route_mismatch.append(
                    f"mb local={mb_metadata.mb_id} payload={payload_mb_id}"
                )
            if payload_dispatch_seq != mb_metadata.dispatch_seq:
                route_mismatch.append(
                    "dispatch "
                    f"local={mb_metadata.dispatch_seq} "
                    f"payload={payload_dispatch_seq}"
                )
            if payload_bs != batch.batch_size() or payload_bs != mb_metadata.batch_size:
                route_mismatch.append(
                    f"bs local={batch.batch_size()} metadata={mb_metadata.batch_size} "
                    f"payload={payload_bs}"
                )
            if payload_mode != mb_metadata.forward_mode:
                route_mismatch.append(
                    "mode "
                    f"local={_dflash_forward_mode_name(mb_metadata.forward_mode)}"
                    f"({mb_metadata.forward_mode}) "
                    f"payload={_dflash_forward_mode_name(payload_mode)}"
                    f"({payload_mode})"
                )
            if local_token_count is not None:
                if payload_token_count != local_token_count:
                    route_mismatch.append(
                        f"tokens local={local_token_count} "
                        f"payload={payload_token_count}"
                    )
            if (
                mb_metadata.token_count >= 0
                and payload_token_count != mb_metadata.token_count
            ):
                route_mismatch.append(
                    f"tokens metadata={mb_metadata.token_count} "
                    f"payload={payload_token_count}"
                )
            if route_mismatch:
                raise RuntimeError(
                    "DFLASH PP route metadata mismatch before follower commit: "
                    + ", ".join(route_mismatch)
                )
            if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
                logger.info(
                    "DFLASH BS trace PP%d payload local_bs=%d local_rids=%s "
                    "candidate_shape=%s position_shape=%s commit_shape=%s "
                    "committed_shape=%s rid_hash_shape=%s payload_hashes=%s",
                    self.pp_rank,
                    batch.batch_size(),
                    _dflash_batch_rids(batch),
                    _dflash_tensor_shape(dflash_next_candidates),
                    _dflash_tensor_shape(dflash_next_positions),
                    _dflash_tensor_shape(dflash_commit_lens),
                    _dflash_tensor_shape(dflash_committed_tokens),
                    _dflash_tensor_shape(dflash_rid_hashes),
                    dflash_rid_hashes.detach().cpu().tolist()
                    if dflash_rid_hashes is not None
                    else None,
                )
            _dflash_validate_pp_payload(
                bs=batch.batch_size(),
                block_size=int(self.server_args.speculative_num_draft_tokens),
                dflash_next_candidates=dflash_next_candidates,
                dflash_next_positions=dflash_next_positions,
                dflash_commit_lens=dflash_commit_lens,
                dflash_committed_tokens=dflash_committed_tokens,
                dflash_rid_hashes=dflash_rid_hashes,
            )
            dflash_accept_length_per_req_cpu = (
                [max(0, int(x) - 1) for x in dflash_commit_lens.to("cpu").tolist()]
                if dflash_commit_lens is not None
                else None
            )
            if dflash_rid_hashes is not None:
                local_rid_hashes = torch.tensor(
                    [_dflash_stable_rid_hash(req.rid) for req in batch.reqs],
                    dtype=torch.int64,
                    device=dflash_rid_hashes.device,
                )
                payload_hashes = dflash_rid_hashes.to(
                    device=local_rid_hashes.device, dtype=torch.int64
                )
                if local_rid_hashes.shape != payload_hashes.shape or not torch.equal(
                    local_rid_hashes, payload_hashes
                ):
                    logger.error(
                        "DFLASH PP rid mismatch: local_rids=%s local_hashes=%s payload_hashes=%s",
                        [req.rid for req in batch.reqs],
                        local_rid_hashes.detach().cpu().tolist(),
                        payload_hashes.detach().cpu().tolist(),
                    )
                    raise RuntimeError(
                        "DFLASH PP payload request ids do not match local batch "
                        f"order (dispatch={mb_metadata.dispatch_seq}, "
                        f"mb={mb_metadata.mb_id})"
                    )
                if _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE"):
                    logger.info(
                        "DFLASH PP rid match: rids=%s hashes=%s",
                        [req.rid for req in batch.reqs],
                        local_rid_hashes.detach().cpu().tolist(),
                    )
            if not batch.is_spec_v2:
                batch.output_ids = next_token_ids
            self._pp_dflash_apply_follower_commit(
                batch=batch,
                mb_metadata=mb_metadata,
                commit_lens=dflash_commit_lens,
                committed_tokens=dflash_committed_tokens,
                next_candidates=dflash_next_candidates,
                next_positions=dflash_next_positions,
            )

        if not is_dflash_pp_follower and not batch.is_spec_v2:
            batch.output_ids = next_token_ids
        accept_lens = None
        if batch.is_spec_v2 and dflash_commit_lens is not None:
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            accept_lens = dflash_commit_lens.to("cpu", non_blocking=True)

        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
            num_accepted_tokens=(
                int(dflash_num_accepted.item())
                if dflash_num_accepted is not None
                else 0
            ),
            accept_length_per_req_cpu=dflash_accept_length_per_req_cpu,
            dflash_next_candidates=dflash_next_candidates,
            dflash_next_positions=dflash_next_positions,
            dflash_commit_lens=dflash_commit_lens,
            dflash_committed_tokens=dflash_committed_tokens,
        )
        return output_result

    def _pp_dflash_apply_follower_commit(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        commit_lens: Optional[torch.Tensor],
        committed_tokens: Optional[torch.Tensor],
        next_candidates: Optional[torch.Tensor],
        next_positions: Optional[torch.Tensor],
    ) -> None:
        """PP0-side hook for DFLASH spec-v1 PP=2.

        Called from _pp_prep_batch_result on the non-last rank. For decode
        iters, applies PP1's verify-commit (output_ids extension, KV free,
        seq_lens advance) using the prior forward's verify_input still attached
        to batch.spec_info. Always installs a fresh DFlashDraftInput carrying
        the next iter's drafter candidates so the upcoming decode forward can
        read them.

        PP0 does not need verified_id / target_hidden / ctx_lens for local
        computation (they're drafter-only state on PP1); we keep dummy zero
        tensors so filter_batch / merge_batch don't crash if reqs finish or
        new prefills arrive.
        """
        from sglang.srt.speculative.dflash_info import (
            DFlashDraftInput,
            DFlashVerifyInput,
        )
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        if next_candidates is None and commit_lens is None and committed_tokens is None:
            # Nothing to do; PP1 didn't ship DFlash state this iter (idle batch?).
            return
        if (commit_lens is None) != (committed_tokens is None):
            raise RuntimeError(
                "DFLASH PP follower received an incomplete commit payload: "
                f"commit_lens={commit_lens is not None}, "
                f"committed_tokens={committed_tokens is not None}."
            )
        trace_bs = _dflash_env_enabled("SGLANG_DFLASH_BS_TRACE")
        mode_before = batch.forward_mode.name
        if trace_bs:
            logger.info(
                "DFLASH BS trace PP%d follower input bs=%d rids=%s mode=%s "
                "commit_lens=%s committed_shape=%s next_candidates_shape=%s "
                "next_positions_shape=%s spec_v2=%s",
                self.pp_rank,
                batch.batch_size(),
                _dflash_batch_rids(batch),
                mode_before,
                commit_lens.detach().cpu().tolist() if commit_lens is not None else None,
                _dflash_tensor_shape(committed_tokens),
                _dflash_tensor_shape(next_candidates),
                _dflash_tensor_shape(next_positions),
                batch.is_spec_v2,
            )

        if batch.is_spec_v2:
            old_draft_input = (
                batch.spec_info
                if isinstance(batch.spec_info, DFlashDraftInputV2)
                else None
            )
            if commit_lens is not None and committed_tokens is not None:
                worker = getattr(self, "draft_worker", None) or getattr(
                    self, "model_worker", None
                ) or self.tp_worker
                if not hasattr(worker, "pp_apply_follower_commit_v2"):
                    raise RuntimeError(
                        "DFLASH PP spec-v2 follower could not locate "
                        "pp_apply_follower_commit_v2 on the spec worker "
                        f"(got {type(worker).__name__})."
                    )
                attn_backend = getattr(
                    worker.target_worker.model_runner, "attn_backend", None
                )
                need_mamba_verify_commit = hasattr(
                    attn_backend, "update_mamba_state_after_mtp_verify"
                )
                seq_lens_pre_verify = (
                    batch.seq_lens.clone() if need_mamba_verify_commit else None
                )
                worker.pp_apply_follower_commit_v2(
                    batch=batch,
                    commit_lens=commit_lens,
                )
                if need_mamba_verify_commit:
                    worker._update_target_mamba_state_after_verify(
                        batch=batch,
                        seq_lens_pre_verify=seq_lens_pre_verify,
                        commit_lens=commit_lens,
                        mamba_cache_indices=mb_metadata.mamba_cache_indices,
                    )
                batch.forward_mode = ForwardMode.DECODE

            bs = batch.batch_size()
            zero32 = torch.zeros((bs,), dtype=torch.int32, device=batch.device)
            new_draft_input = DFlashDraftInputV2(
                topk_p=torch.empty((bs, 0), device=batch.device, dtype=torch.float32),
                topk_index=torch.empty((bs, 0), device=batch.device, dtype=torch.int64),
                verified_id=zero32.clone(),
                new_seq_lens=batch.seq_lens.to(dtype=torch.int32),
                hidden_states=torch.empty(
                    (bs, 0), device=batch.device, dtype=torch.float16
                ),
                cur_allocated_seq_lens_cpu=(
                    old_draft_input.reserved_seq_lens_cpu
                    if old_draft_input is not None
                    and old_draft_input.reserved_seq_lens_cpu is not None
                    else batch.seq_lens_cpu
                ),
                next_candidates=next_candidates,
                next_positions=next_positions,
            )
            batch.spec_info = new_draft_input
            if trace_bs:
                logger.info(
                    "DFLASH BS trace PP%d follower output bs=%d rids=%s "
                    "mode_before=%s mode_after=%s spec_info_candidates=%s",
                    self.pp_rank,
                    batch.batch_size(),
                    _dflash_batch_rids(batch),
                    mode_before,
                    batch.forward_mode.name,
                    _dflash_tensor_shape(next_candidates),
                )
            return

        # Decode iters carry verify-commit; prefill iters (drafter-prime only)
        # ship just the candidates.
        if commit_lens is not None and committed_tokens is not None:
            verify_input = batch.spec_info
            if not isinstance(verify_input, DFlashVerifyInput):
                raise RuntimeError(
                    "DFLASH PP follower expected DFlashVerifyInput on batch.spec_info "
                    f"after decode forward, got {type(verify_input).__name__}."
                )
            # DFlash spec worker is self.draft_worker (a DFlashWorker), aliased
            # to self.model_worker by init_model_worker. Fall back to tp_worker
            # only as a defensive last resort.
            worker = getattr(self, "draft_worker", None) or getattr(
                self, "model_worker", None
            ) or self.tp_worker
            if not hasattr(worker, "pp_apply_follower_commit"):
                raise RuntimeError(
                    "DFLASH PP follower could not locate pp_apply_follower_commit on "
                    f"the spec worker (got {type(worker).__name__})."
                )
            attn_backend = getattr(worker.target_worker.model_runner, "attn_backend", None)
            need_mamba_verify_commit = hasattr(
                attn_backend, "update_mamba_state_after_mtp_verify"
            )
            seq_lens_pre_verify = (
                batch.seq_lens.clone() if need_mamba_verify_commit else None
            )
            worker.pp_apply_follower_commit(
                batch=batch,
                verify_input=verify_input,
                commit_lens=commit_lens,
                committed_tokens=committed_tokens,
            )
            if need_mamba_verify_commit:
                worker._update_target_mamba_state_after_verify(
                    batch=batch,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                    commit_lens=commit_lens,
                    mamba_cache_indices=mb_metadata.mamba_cache_indices,
                )

            # Mirror PP1's `batch.forward_mode = ForwardMode.DECODE` reset
            # (dflash_worker.py:1855) so the next iter's get_next_batch_to_run
            # does not see this batch as still-extend. With TARGET_VERIFY left
            # in place, last_batch.forward_mode.is_extend() is True; combined
            # with running_mbs[mb_id] === last_mbs[mb_id] (the same decode
            # ScheduleBatch object after the prior iter's TARGET_VERIFY hop)
            # the scheduler runs `running.merge_batch(last)` against itself,
            # which torch.cats every batch field on top of itself and silently
            # doubles bs. PP1's identical reset means PP1 stays at the real
            # bs while PP0 grows; the proxy chain then ships PP0's bs=2 hidden
            # states into PP1's bs=1 verify, blowing up at set_kv_buffer with
            # the kt-kernel TVM "expected 2 got 1" mismatch the user sees.
            batch.forward_mode = ForwardMode.DECODE

        bs = batch.batch_size()
        zero32 = torch.zeros((bs,), dtype=torch.int32, device=batch.device)
        zero64 = torch.zeros((bs,), dtype=torch.int64, device=batch.device)
        draft_input = DFlashDraftInput(
            verified_id=zero64,
            target_hidden=torch.empty(
                (0,), dtype=torch.float32, device=batch.device
            ),
            ctx_lens=zero32,
            draft_seq_lens=zero32.clone(),
            next_candidates=next_candidates,
            next_positions=next_positions,
        )
        batch.spec_info = draft_input
        if trace_bs:
            logger.info(
                "DFLASH BS trace PP%d follower output bs=%d rids=%s "
                "mode_before=%s mode_after=%s spec_info_candidates=%s",
                self.pp_rank,
                batch.batch_size(),
                _dflash_batch_rids(batch),
                mode_before,
                batch.forward_mode.name,
                _dflash_tensor_shape(next_candidates),
            )

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            if mbs[next_first_rank_mb_id] is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if not mbs[next_first_rank_mb_id].forward_mode.is_prebuilt():
                    phase_t = time.perf_counter()
                    torch.cuda.current_stream().wait_event(q_event)
                    _dflash_log_timeline(
                        self,
                        "pp.output_ring.wait_ready",
                        mb_id=next_first_rank_mb_id,
                        batch=mbs[next_first_rank_mb_id],
                        start_time=phase_t,
                    )
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        phase_t = time.perf_counter()
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                        )
                    _dflash_log_timeline(
                        self,
                        "pp.output_ring.send",
                        mb_id=next_first_rank_mb_id,
                        batch=mbs[next_first_rank_mb_id],
                        start_time=phase_t,
                        keys=list(pp_outputs_to_send.tensors.keys()),
                    )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    phase_t = time.perf_counter()
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                    )
                _dflash_log_timeline(
                    self,
                    "pp.output_ring.send",
                    mb_id=next_first_rank_mb_id,
                    batch=mbs[next_first_rank_mb_id]
                    if mbs[next_first_rank_mb_id] is not None
                    else None,
                    start_time=phase_t,
                    keys=list(pp_outputs.tensors.keys()),
                )
                if self._pp_dflash_run_control_enabled():
                    payload_mb_tensor = pp_outputs.tensors.get("dflash_pp_mb_id", None)
                    if payload_mb_tensor is not None:
                        forwarded_mb_id = _dflash_tensor_scalar_int(
                            "dflash_pp_mb_id", payload_mb_tensor
                        )
                        if forwarded_mb_id is not None:
                            self.dflash_pp_pending_forward_mb_ids.discard(
                                forwarded_mb_id
                            )
                            _dflash_log_timeline(
                                self,
                                "pp.output_ring.forwarded",
                                mb_id=forwarded_mb_id,
                                pending=sorted(
                                    self.dflash_pp_pending_forward_mb_ids
                                ),
                            )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[PPProxyTensors, List[P2PWork], torch.cuda.Event]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        result_mb_id = next_mb_id
        if self._pp_dflash_run_control_enabled():
            will_send_output = self._pp_dflash_will_send_output(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )
            output_intent = self._pp_dflash_make_output_intent(
                next_first_rank_mb_id, will_send_output
            )
            self._pp_dflash_forward_output_intent(output_intent)
            prev_output_intent = self._pp_dflash_recv_output_intent(next_mb_id)
            prev_has_output = bool(prev_output_intent["send"])
            send_output_now = will_send_output
            should_recv_output = prev_has_output
            if self.pp_size == 2 and will_send_output and prev_has_output:
                # NCCL point-to-point ops are order-sensitive. In PP=2, both
                # ranks can have output at the same time, and posting both
                # sends before both receives can deadlock. Drain the pending
                # ring-forward from the non-last rank first; the last rank's
                # local output remains queued for the next scheduler tick.
                send_output_now = not self.pp_group.is_last_rank
                should_recv_output = self.pp_group.is_last_rank
                _dflash_log_timeline(
                    self,
                    "pp.output_intent.arbitrate",
                    mb_id=next_first_rank_mb_id,
                    local_has=will_send_output,
                    prev_has=prev_has_output,
                    send_now=send_output_now,
                    recv_now=should_recv_output,
                )
            send_output_work = []
            if send_output_now:
                send_output_work = self._pp_send_output_to_next_stage(
                    next_first_rank_mb_id,
                    mbs,
                    last_rank_comm_queue,
                    pp_outputs,
                )
            if send_output_now and not send_output_work:
                raise RuntimeError(
                    "DFLASH PP output intent predicted a tensor send, but no "
                    f"send work was posted on pp={self.pp_rank}, "
                    f"mb={next_first_rank_mb_id}."
                )
        else:
            send_output_work = self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )
            should_recv_output = (
                mbs[next_mb_id] is not None
                and not mbs[next_mb_id].forward_mode.is_prebuilt()
            )

        if should_recv_output:
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                phase_t = time.perf_counter()
                next_pp_outputs = PPProxyTensors(
                    self._pp_recv_dict_from_prev_stage()
                )
                if self._pp_dflash_run_control_enabled():
                    payload_mb_tensor = next_pp_outputs.tensors.get(
                        "dflash_pp_mb_id", None
                    )
                    if payload_mb_tensor is not None:
                        payload_mb_id = _dflash_tensor_scalar_int(
                            "dflash_pp_mb_id", payload_mb_tensor
                        )
                        if payload_mb_id is None or not (0 <= payload_mb_id < len(mbs)):
                            raise RuntimeError(
                                "DFLASH PP received output payload with invalid "
                                f"mb id {payload_mb_id!r}; expected range "
                                f"[0, {len(mbs)})."
                            )
                        result_mb_id = payload_mb_id
                if (
                    self._pp_dflash_run_control_enabled()
                    and not self.pp_group.is_last_rank
                ):
                    self.dflash_pp_pending_result_mb_ids.discard(result_mb_id)
                    self.dflash_pp_pending_forward_mb_ids.add(result_mb_id)
                    _dflash_log_timeline(
                        self,
                        "pp.output_ring.needs_forward",
                        mb_id=result_mb_id,
                        pending_results=sorted(
                            self.dflash_pp_pending_result_mb_ids
                        ),
                        pending=sorted(self.dflash_pp_pending_forward_mb_ids),
                    )
                _dflash_log_timeline(
                    self,
                    "pp.output_ring.recv",
                    mb_id=result_mb_id,
                    batch=mbs[result_mb_id],
                    metadata=mb_metadata[result_mb_id],
                    start_time=phase_t,
                    expected_mb=next_mb_id,
                    keys=list(next_pp_outputs.tensors.keys()),
                )

        if (
            mbs[result_mb_id] is not None
            and next_pp_outputs is not None
            and not mbs[result_mb_id].forward_mode.is_prebuilt()
        ):
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.default_stream)
                phase_t = time.perf_counter()
                batch_result = self._pp_prep_batch_result(
                    mbs[result_mb_id], mb_metadata[result_mb_id], next_pp_outputs
                )
                _dflash_log_timeline(
                    self,
                    "pp.output_ring.prep_result",
                    mb_id=result_mb_id,
                    batch=mbs[result_mb_id],
                    metadata=mb_metadata[result_mb_id],
                    start_time=phase_t,
                )
                d2h_event = torch.cuda.Event()
                d2h_event.record(torch.cuda.current_stream())

        self.pp_output_result_mb_id = result_mb_id
        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.default_stream)
                dispatch_seq = self.dflash_pp_dispatch_seq
                self.dflash_pp_dispatch_seq += 1
                phase_t = time.perf_counter()
                route_batch = self.cur_batch
                route_token_count = _dflash_batch_token_count(route_batch)
                route_metadata = {
                    "batch_size": route_batch.batch_size(),
                    "forward_mode": int(route_batch.forward_mode),
                    "rid_hashes": tuple(_dflash_batch_rid_hashes(route_batch)),
                    "token_count": route_token_count if route_token_count is not None else -1,
                }
                result = self.run_batch(route_batch, pp_proxy_tensors)
                metadata = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                    dispatch_seq=dispatch_seq,
                    mb_id=mb_id,
                    batch_size=route_metadata["batch_size"],
                    forward_mode=route_metadata["forward_mode"],
                    rid_hashes=route_metadata["rid_hashes"],
                    token_count=route_metadata["token_count"],
                    mamba_cache_indices=(
                        _dflash_clone_current_mamba_cache_indices(self)
                        if self._pp_dflash_run_control_enabled()
                        else None
                    ),
                )
                mb_metadata[mb_id] = metadata
                _dflash_log_timeline(
                    self,
                    "pp.run_batch",
                    mb_id=mb_id,
                    batch=self.cur_batch,
                    metadata=metadata,
                    start_time=phase_t,
                    cuda_graph=result.can_run_cuda_graph,
                )
                if (
                    self._pp_dflash_run_control_enabled()
                    and self.pp_group.is_first_rank
                ):
                    self.dflash_pp_pending_result_mb_ids.add(mb_id)
                    _dflash_log_timeline(
                        self,
                        "pp.output_ring.result_pending",
                        mb_id=mb_id,
                        metadata=metadata,
                        pending=sorted(self.dflash_pp_pending_result_mb_ids),
                    )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(
                                    result, self.cur_batch, metadata
                                )
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                rids_to_check=good_consensus_prealloc_rids
                + bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
