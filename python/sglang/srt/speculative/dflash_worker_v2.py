import contextlib
import logging
import os
import time
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    apply_dflash_verify_logits_adjustments,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
    _compute_dflash_accept_bonus_triton_unchecked,
)
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)


def _dflash_timeline_enabled() -> bool:
    return os.getenv("SGLANG_DFLASH_PP_TIMELINE") in ("1", "true", "TRUE")


def _dflash_worker_timing_enabled() -> bool:
    return _dflash_timeline_enabled() or os.getenv(
        "SGLANG_DFLASH_WORKER_TIMING"
    ) in ("1", "true", "TRUE")


def _dflash_timeline_values_enabled() -> bool:
    return os.getenv("SGLANG_DFLASH_PP_TIMELINE_VALUES") in ("1", "true", "TRUE")


def _maybe_cpu_list(tensor: Optional[torch.Tensor]):
    if tensor is None or not _dflash_timeline_values_enabled():
        return None
    return tensor.detach().to("cpu").tolist()


def _dflash_log_timeline(worker, phase: str, start_time: Optional[float] = None, **fields):
    if not _dflash_worker_timing_enabled():
        return
    elapsed_ms = None
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    parts = [
        f"phase={phase}",
        f"pp={getattr(worker.pp_group, 'rank', None)}",
        f"tp={getattr(worker, 'tp_rank', None)}",
        f"drafter={getattr(worker, 'is_drafter_rank', None)}",
    ]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.3f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    log_kind = "timeline" if _dflash_timeline_enabled() else "timing"
    logger.info("DFLASH worker %s %s", log_kind, " ".join(parts))


def _get_plan_stream(device: str):
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    return None, contextlib.nullcontext()


class DFlashWorkerV2(DFlashWorker):
    """DFLASH speculative decoding worker (spec-v2 overlap scheduling).

    This is intentionally implemented as a *separate* worker from the existing
    spec-v1 `DFlashWorker` (non-overlap), to keep the v1 path stable and to
    minimize risk while bringing up overlap scheduling.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def _validate_phase1_sampling_support(
        self, model_worker_batch: ModelWorkerBatch
    ) -> None:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=seq_lens.to(dtype=torch.int32),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        verified_id: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=new_seq_lens.to(dtype=torch.int32),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _build_target_verify_custom_mask(
        self, model_worker_batch: ModelWorkerBatch, block_size: int
    ) -> torch.Tensor:
        mask_chunks = []
        q_idx = torch.arange(
            block_size,
            device=self.device,
            dtype=torch.int32,
        ).unsqueeze(1)
        seq_lens_cpu = model_worker_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = model_worker_batch.seq_lens.to("cpu", dtype=torch.int32)

        for prefix_len in seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + block_size
            k_idx = torch.arange(
                kv_len,
                device=self.device,
                dtype=torch.int32,
            ).unsqueeze(0)
            mask_chunks.append((k_idx <= (prefix_len_i + q_idx)).flatten())

        return (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=self.device)
        )

    def _attach_target_verify_chain_metadata(
        self, verify_input: DFlashVerifyInput, bs: int, block_size: int
    ) -> None:
        (
            _retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            _predicts,
            _accept_index,
            _accept_token_num,
        ) = _get_or_create_chain_verify_buffers(
            bs=bs,
            draft_token_num=block_size,
            device=torch.device(self.device),
        )
        verify_input.retrive_next_token = retrieve_next_token
        verify_input.retrive_next_sibling = retrieve_next_sibling

    def _prepare_v2_verify_from_candidates(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInputV2,
        candidates_flat: torch.Tensor,
        positions_flat: Optional[torch.Tensor],
    ):
        bs = len(model_worker_batch.seq_lens)
        block_size = int(self.block_size)
        expected = bs * block_size
        if int(candidates_flat.numel()) != expected:
            raise RuntimeError(
                f"DFLASH PP spec-v2 candidates mismatch: bs={bs} block_size={block_size} "
                f"expected={expected} got={int(candidates_flat.numel())}"
            )

        device = self.device
        prefix_lens = model_worker_batch.seq_lens
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_positions_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None

        positions_2d = self._draft_block_positions_buf[:bs]
        if positions_flat is not None:
            if int(positions_flat.numel()) != expected:
                raise RuntimeError(
                    f"DFLASH PP spec-v2 positions mismatch: bs={bs} block_size={block_size} "
                    f"expected={expected} got={int(positions_flat.numel())}"
                )
            positions_2d.copy_(positions_flat.view(bs, block_size))
        else:
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )

        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        end_offset = prefix_lens + block_size
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=model_worker_batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=block_size,
            device=device,
        )
        verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        verify_input = DFlashVerifyInput(
            draft_token=candidates_flat.to(device=device, dtype=torch.long),
            positions=positions_2d.reshape(-1),
            draft_token_num=block_size,
            custom_mask=self._build_target_verify_custom_mask(
                model_worker_batch, block_size
            ),
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        self._attach_target_verify_chain_metadata(verify_input, bs, block_size)
        model_worker_batch.out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        caller_stream = torch.get_device_module(self.device).current_stream()
        with self.plan_stream_ctx:
            if self.plan_stream is not None:
                self.plan_stream.wait_stream(caller_stream)
            verify_forward_batch, _ = verify_input.prepare_for_v2_verify(
                model_worker_batch, self.target_worker
            )
        if self.plan_stream:
            caller_stream.wait_stream(self.plan_stream)

        return verify_input, verify_forward_batch, verify_out_cache_loc_2d

    def _draft_v2_candidates_for_next_iter(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInputV2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the DFlash drafter only and return a flat candidate block.

        This mirrors the first phase of the normal spec-v2 worker. PP uses it
        after prefill and after verify so the next decode iteration can be
        driven by candidates shipped through the standard PP output ring.
        """
        bs = len(model_worker_batch.seq_lens)
        device = self.device
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    verified_id=draft_input.verified_id.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.verified_id)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=model_worker_batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(
                    verify_out_cache_loc.view(bs, block_size)
                )
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.verified_id)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = self._dflash_embed_block_ids(embed_module, block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
                self._validate_compact_draft_seq_lens_cpu(
                    seq_lens_cpu, draft_prefix_lens
                )
            else:
                seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(seq_lens_cpu.sum().item())

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
        else:
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = int(model_worker_batch.seq_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_model_runner.attn_backend,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, block_size, -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, block_size - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        return draft_tokens.reshape(-1).clone(), positions_2d.reshape(-1).clone()

    def pp_apply_follower_commit_v2(
        self,
        batch,
        commit_lens: torch.Tensor,
    ) -> None:
        """Mirror PP1's target-verify KV/seq side effects on a non-drafter PP rank.

        Spec-v2 output processing owns req.output_ids and req.kv_committed_len.
        This hook only advances the scheduler batch tensors that the next PP0
        forward will read. Spec-v2's overallocated reservation owns the
        uncommitted verify slots and releases them during normal result cleanup.
        """
        bs = batch.batch_size()
        if batch.out_cache_loc is None:
            raise RuntimeError("DFLASH PP follower commit expected out_cache_loc.")
        device = batch.out_cache_loc.device
        block_size = int(self.block_size)
        commit_lens = commit_lens.to(device=device, dtype=torch.int32)
        if int(commit_lens.numel()) != bs:
            raise RuntimeError(
                "DFLASH PP follower commit_lens shape mismatch: "
                f"expected={bs}, got={int(commit_lens.numel())}."
            )
        if int(batch.out_cache_loc.numel()) < bs * block_size:
            raise RuntimeError(
                "DFLASH PP follower out_cache_loc shape mismatch: "
                f"expected_at_least={bs * block_size}, "
                f"got={int(batch.out_cache_loc.numel())}."
            )
        bad_commit_lens = commit_lens[
            torch.logical_or(commit_lens < 1, commit_lens > block_size)
        ]
        if bad_commit_lens.numel() > 0:
            raise RuntimeError(
                "DFLASH PP follower commit_lens out of bounds: "
                f"values={bad_commit_lens.detach().to('cpu').tolist()}, "
                f"block_size={block_size}."
            )
        commit_lens_cpu = commit_lens.to("cpu").tolist()

        out_cache_loc = batch.out_cache_loc[: bs * block_size].view(bs, block_size)
        row_offsets = torch.arange(block_size, device=device)[None, :]
        keep_mask = row_offsets < commit_lens[:, None]
        batch.out_cache_loc = out_cache_loc[keep_mask]

        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu.add_(
                torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
            )
        batch.seq_lens_sum += sum(commit_lens_cpu)

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            _dflash_log_timeline(
                self,
                "dflash.prefill.target",
                phase_t,
                bs=len(model_worker_batch.seq_lens),
                mode=model_worker_batch.forward_mode.name,
            )

            if not self.is_drafter_rank:
                # In PP mode only the last rank owns the drafter and final logits.
                # Non-drafter ranks have already forwarded target activations to
                # the next pipeline stage; there is no local hidden-state buffer
                # to materialize into the draft KV cache.
                return batch_output

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, "
                    "but got None."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            ctx_lens = torch.tensor(
                model_worker_batch.extend_seq_lens, dtype=torch.int32, device=device
            )
            draft_seq_lens = torch.tensor(
                model_worker_batch.extend_prefix_lens, dtype=torch.int32, device=device
            )

            if model_worker_batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                draft_seq_lens,
                ctx_lens,
                int(sum(model_worker_batch.extend_seq_lens)),
            )
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=model_worker_batch.out_cache_loc,
                positions=positions,
            )
            _dflash_log_timeline(
                self,
                "dflash.prefill.materialize_draft_kv",
                phase_t,
                bs=len(model_worker_batch.seq_lens),
                tokens=int(sum(model_worker_batch.extend_seq_lens)),
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            next_draft_input = self._make_next_draft_input_prefill(
                verified_id=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
                cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            if self.pp_group.world_size > 1:
                phase_t = None
                if _dflash_worker_timing_enabled():
                    phase_t = time.perf_counter()
                next_candidates, next_positions = self._draft_v2_candidates_for_next_iter(
                    model_worker_batch, next_draft_input
                )
                _dflash_log_timeline(
                    self,
                    "dflash.prefill.draft_prime",
                    phase_t,
                    bs=len(model_worker_batch.seq_lens),
                    candidate_shape=tuple(next_candidates.shape),
                )
                next_draft_input.next_candidates = next_candidates
                next_draft_input.next_positions = next_positions
                batch_output.dflash_next_candidates = next_candidates
                batch_output.dflash_next_positions = next_positions

            batch_output.next_draft_input = next_draft_input
            return batch_output

        # Decode / target-verify stage.
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=torch.empty((0,), device=self.device, dtype=torch.int32),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int32),
            )
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(model_worker_batch.seq_lens)
        device = self.device

        if self.pp_group.world_size > 1:
            if (
                draft_input.next_candidates is None
                or draft_input.next_positions is None
            ):
                raise RuntimeError(
                    "DFLASH PP spec-v2 decode missing pre-built next_candidates/next_positions. "
                    "The drafter rank must prime them during prefill or the prior decode iter."
                )

            prefix_lens = model_worker_batch.seq_lens
            need_mamba_verify_commit = hasattr(
                self.target_worker.model_runner.attn_backend,
                "update_mamba_state_after_mtp_verify",
            )
            seq_lens_pre_verify = (
                prefix_lens.clone() if need_mamba_verify_commit else None
            )
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            verify_input, verify_forward_batch, verify_out_cache_loc_2d = (
                self._prepare_v2_verify_from_candidates(
                    model_worker_batch=model_worker_batch,
                    draft_input=draft_input,
                    candidates_flat=draft_input.next_candidates,
                    positions_flat=draft_input.next_positions,
                )
            )
            _dflash_log_timeline(
                self,
                "dflash.decode.prepare_verify",
                phase_t,
                bs=bs,
                block_size=int(self.block_size),
                candidates=tuple(draft_input.next_candidates.shape),
            )
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            target_out = self.target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
                **kwargs,
            )
            _dflash_log_timeline(
                self,
                "dflash.decode.target_verify",
                phase_t,
                bs=bs,
                block_size=int(self.block_size),
            )
            if (
                not getattr(self, "_logged_target_verify_parent_probe", False)
                and os.environ.get("SGLANG_DFLASH_PARENT_PROBE") == "1"
                and self.tp_rank == 0
            ):
                linear_backend = getattr(
                    self.target_worker.model_runner.attn_backend,
                    "linear_attn_backend",
                    None,
                )
                metadata = getattr(linear_backend, "forward_metadata", None)
                retrieve_next = getattr(metadata, "retrieve_next_token", None)
                retrieve_sibling = getattr(metadata, "retrieve_next_sibling", None)
                retrieve_parent = getattr(metadata, "retrieve_parent_token", None)
                query_start_loc = getattr(metadata, "query_start_loc", None)

                def _sample_tensor(tensor):
                    if tensor is None:
                        return None
                    if tensor.ndim == 0:
                        return tensor.detach().to("cpu").item()
                    return (
                        tensor[: min(int(tensor.shape[0]), 2), : min(int(tensor.shape[-1]), 16)]
                        .detach()
                        .to("cpu")
                        .tolist()
                    )

                logger.info(
                    "DFLASH target verify parent probe pp_rank=%s next=%s sibling=%s parent=%s qloc=%s",
                    getattr(self.pp_group, "rank", None),
                    _sample_tensor(retrieve_next),
                    _sample_tensor(retrieve_sibling),
                    _sample_tensor(retrieve_parent),
                    query_start_loc.detach().to("cpu").tolist()
                    if query_start_loc is not None
                    else None,
                )
                self._logged_target_verify_parent_probe = True
            if not self.is_drafter_rank:
                return target_out

            logits_output = target_out.logits_output
            can_run_cuda_graph = target_out.can_run_cuda_graph
            sampling_info = model_worker_batch.sampling_info
            if sampling_info is not None:
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    draft_token_num=int(self.block_size),
                )

            candidates = draft_input.next_candidates.view(bs, int(self.block_size))
            target_predict_for_log = None
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            if (
                sampling_info is not None
                and not sampling_info.is_all_greedy
                and is_dflash_sampling_verify_available()
            ):
                accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    max_top_k=draft_input.max_top_k,
                    uniform_top_k_value=draft_input.uniform_top_k_value,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (bs, int(self.block_size)), dtype=torch.int64, device=device
                )
                if int(self.block_size) > 1:
                    out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
                out_tokens[:, int(self.block_size) - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
            else:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, int(self.block_size))
                target_predict_for_log = target_predict
                if self._use_triton_accept_bonus:
                    try:
                        accept_len = torch.empty(
                            (bs,), dtype=torch.int32, device=device
                        )
                        commit_lens = torch.empty(
                            (bs,), dtype=torch.int32, device=device
                        )
                        bonus = torch.empty(
                            (bs,), dtype=candidates.dtype, device=device
                        )
                        out_tokens = torch.empty(
                            (bs, int(self.block_size)),
                            dtype=candidates.dtype,
                            device=device,
                        )
                        _compute_dflash_accept_bonus_triton_unchecked(
                            candidates=candidates,
                            target_top1=target_predict,
                            accept_lens_out=accept_len,
                            commit_lens_out=commit_lens,
                            bonus_ids_out=bonus,
                            out_tokens_out=out_tokens,
                        )
                    except Exception as e:
                        self._use_triton_accept_bonus = False
                        logger.warning(
                            "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                            e,
                        )
                        accept_len, bonus = compute_dflash_accept_len_and_bonus(
                            candidates=candidates,
                            target_predict=target_predict,
                        )
                        commit_lens = accept_len.to(torch.int32) + 1
                        out_tokens = torch.empty(
                            (bs, int(self.block_size)),
                            dtype=torch.int64,
                            device=device,
                        )
                        if int(self.block_size) > 1:
                            out_tokens[:, : int(self.block_size) - 1].copy_(
                                candidates[:, 1:]
                            )
                        out_tokens[:, int(self.block_size) - 1].fill_(0)
                        out_tokens.scatter_(
                            1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                        )
                else:
                    accept_len, bonus = compute_dflash_accept_len_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1
                    out_tokens = torch.empty(
                        (bs, int(self.block_size)), dtype=torch.int64, device=device
                    )
                    if int(self.block_size) > 1:
                        out_tokens[:, : int(self.block_size) - 1].copy_(
                            candidates[:, 1:]
                        )
                    out_tokens[:, int(self.block_size) - 1].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )

            _dflash_log_timeline(
                self,
                "dflash.decode.accept_bonus",
                phase_t,
                bs=bs,
                block_size=int(self.block_size),
                commit_lens=_maybe_cpu_list(commit_lens),
            )

            if (
                not self._logged_first_verify
                and self.tp_rank == 0
                and target_predict_for_log is not None
            ):
                logger.info(
                    "DFLASH spec-v2 PP verify sample: candidates0=%s target0=%s "
                    "commit_lens=%s out0=%s bonus=%s",
                    candidates[0].detach().to("cpu").tolist() if bs > 0 else [],
                    target_predict_for_log[0].detach().to("cpu").tolist()
                    if bs > 0
                    else [],
                    commit_lens.detach().to("cpu").tolist(),
                    out_tokens[0].detach().to("cpu").tolist() if bs > 0 else [],
                    bonus.detach().to("cpu").tolist(),
                )
                self._logged_first_verify = True

            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=bonus,
                new_seq_lens=new_seq_lens,
                cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
            )
            compact_seq_lens_cpu_mirror = (
                self._start_dflash_compact_seq_lens_cpu_mirror(
                    model_worker_batch.seq_lens_cpu, commit_lens
                )
            )

            if need_mamba_verify_commit:
                assert seq_lens_pre_verify is not None
                phase_t = None
                if _dflash_worker_timing_enabled():
                    phase_t = time.perf_counter()
                self._update_target_mamba_state_after_verify(
                    batch=model_worker_batch,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                    commit_lens=commit_lens,
                )
                _dflash_log_timeline(
                    self,
                    "dflash.decode.mamba_commit",
                    phase_t,
                    bs=bs,
                    block_size=int(self.block_size),
                )

            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "DFLASH verify requires target hidden states, but got None."
                )
            hidden = hidden.view(bs, int(self.block_size), -1)
            verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=hidden.reshape(-1, hidden.shape[-1]),
                cache_loc=verify_out_cache_loc,
                cache_loc_2d=verify_out_cache_loc_2d,
                positions=verify_input.positions,
                commit_lens=commit_lens,
            )
            _dflash_log_timeline(
                self,
                "dflash.decode.materialize_draft_kv",
                phase_t,
                bs=bs,
                block_size=int(self.block_size),
            )
            logits_output.hidden_states = None

            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done

            old_seq_lens = model_worker_batch.seq_lens
            old_seq_lens_cpu = model_worker_batch.seq_lens_cpu
            old_seq_lens_sum = model_worker_batch.seq_lens_sum
            model_worker_batch.seq_lens = new_seq_lens
            if (
                not self.use_compact_draft_cache
                and model_worker_batch.seq_lens_cpu is not None
            ):
                model_worker_batch.seq_lens_cpu = (
                    model_worker_batch.seq_lens_cpu
                    + commit_lens.to("cpu", dtype=model_worker_batch.seq_lens_cpu.dtype)
                )
                model_worker_batch.seq_lens_sum = int(
                    model_worker_batch.seq_lens_cpu.sum().item()
                )
            phase_t = None
            if _dflash_worker_timing_enabled():
                phase_t = time.perf_counter()
            planning_seq_lens_cpu, planning_seq_lens_sum = (
                self._finish_dflash_compact_seq_lens_cpu_mirror(
                    compact_seq_lens_cpu_mirror
                )
            )
            if planning_seq_lens_cpu is not None:
                next_draft_input.planning_seq_lens_cpu = planning_seq_lens_cpu
                next_draft_input.planning_seq_lens_sum = planning_seq_lens_sum
            next_candidates, next_positions = self._draft_v2_candidates_for_next_iter(
                model_worker_batch, next_draft_input
            )
            _dflash_log_timeline(
                self,
                "dflash.decode.draft_next",
                phase_t,
                bs=bs,
                block_size=int(self.block_size),
                candidate_shape=tuple(next_candidates.shape),
            )
            model_worker_batch.seq_lens = old_seq_lens
            model_worker_batch.seq_lens_cpu = old_seq_lens_cpu
            model_worker_batch.seq_lens_sum = old_seq_lens_sum
            next_draft_input.next_candidates = next_candidates
            next_draft_input.next_positions = next_positions

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=out_tokens.reshape(-1),
                accept_lens=commit_lens,
                can_run_cuda_graph=can_run_cuda_graph,
                next_draft_input=next_draft_input,
                prepared_kv_allocated_lens_cpu=draft_input.reserved_seq_lens_cpu,
                dflash_next_candidates=next_candidates,
                dflash_next_positions=next_positions,
                dflash_commit_lens=commit_lens.to(torch.int32),
                dflash_committed_tokens=out_tokens.to(torch.int64),
            )

        # --- 1) Draft a fixed block with the draft model.
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    verified_id=draft_input.verified_id.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.verified_id)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=model_worker_batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.verified_id)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = self._dflash_embed_block_ids(embed_module, block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
                self._validate_compact_draft_seq_lens_cpu(
                    seq_lens_cpu, draft_prefix_lens
                )
            else:
                seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(seq_lens_cpu.sum().item())

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = int(model_worker_batch.seq_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_model_runner.attn_backend,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=self._build_target_verify_custom_mask(
                model_worker_batch, int(self.block_size)
            ),
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        self._attach_target_verify_chain_metadata(
            verify_input, bs, int(self.block_size)
        )

        model_worker_batch.out_cache_loc = verify_out_cache_loc
        sampling_info = model_worker_batch.sampling_info

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        caller_stream = torch.get_device_module(self.device).current_stream()
        with self.plan_stream_ctx:
            if self.plan_stream is not None:
                self.plan_stream.wait_stream(caller_stream)
            verify_forward_batch, _ = verify_input.prepare_for_v2_verify(
                model_worker_batch, self.target_worker
            )
        if self.plan_stream:
            caller_stream.wait_stream(self.plan_stream)

        target_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
            **kwargs,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )

        candidates = draft_tokens
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            commit_lens = accept_len.to(torch.int32) + 1  # [bs]
            out_tokens = torch.empty(
                (bs, int(self.block_size)), dtype=torch.int64, device=device
            )
            if int(self.block_size) > 1:
                out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
            out_tokens[:, int(self.block_size) - 1].fill_(0)
            out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
            if self._use_triton_accept_bonus:
                try:
                    accept_len = torch.empty((bs,), dtype=torch.int32, device=device)
                    commit_lens = torch.empty((bs,), dtype=torch.int32, device=device)
                    bonus = torch.empty((bs,), dtype=candidates.dtype, device=device)
                    out_tokens = torch.empty(
                        (bs, int(self.block_size)),
                        dtype=candidates.dtype,
                        device=device,
                    )
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                    )
                except Exception as e:
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                        e,
                    )
                    accept_len, bonus = compute_dflash_accept_len_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                    out_tokens = torch.empty(
                        (bs, int(self.block_size)), dtype=torch.int64, device=device
                    )
                    if int(self.block_size) > 1:
                        out_tokens[:, : int(self.block_size) - 1].copy_(
                            candidates[:, 1:]
                        )
                    out_tokens[:, int(self.block_size) - 1].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )
            else:
                accept_len, bonus = compute_dflash_accept_len_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                out_tokens = torch.empty(
                    (bs, int(self.block_size)), dtype=torch.int64, device=device
                )
                if int(self.block_size) > 1:
                    out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
                out_tokens[:, int(self.block_size) - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(self.block_size), -1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        next_draft_input = self._make_next_draft_input_decode(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            prepared_kv_allocated_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
