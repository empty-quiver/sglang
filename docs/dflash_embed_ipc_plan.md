# DFlash PP=2 Embed-IPC Replacement: Implementation Plan

## Background recap

- Today: PP1 holds a duplicated `VocabParallelEmbedding` (~2.5 GB BF16) created by commit `87a53b9e7` in `qwen3_5.py` at lines 698-712.
- Goal: drop the duplicate, do a synchronous in-iter embed lookup IPC PP1→PP0→PP1 around the single call site in `dflash_worker.py:817-845`.
- `verified_id` is produced by PP1's `verify_input.verify(...)` at line 1829. It cannot be predicted on PP0, so the IPC must be sync.
- `mask_token_id` is constant for the run; its embedding row can be cached on PP1 forever after a one-time fetch.

## 1. Insertion points

### PP1 (drafter rank) — the producer of the request

The single call is at `/home/eve/sglang-dflash/python/sglang/srt/speculative/dflash_worker.py:817` inside `_run_drafter_for_next_iter`. Replace the `embed_module(block_ids)` call at line 844 with an IPC-based replacement. Concretely, the change centers on these lines:

```
# dflash_worker.py:840-845 (today)
block_ids = self._draft_block_ids_buf[:bs]
block_ids.fill_(int(self._mask_token_id))
block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))
noise_embedding = embed_module(block_ids)
input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
```

After the patch this becomes:
- Build `verified_id_int64` of shape `[bs]` (column 0 of `block_ids`).
- Send `[bs]` int64 to PP0; recv `[bs, hidden_size]` bf16 from PP0.
- Build `noise_embedding[bs, block_size, hidden_size]` from the cached mask-row tile + the per-iter received row.
  - `noise_embedding[:, 0, :] = recv_buf`
  - `noise_embedding[:, 1:, :] = self._cached_mask_embed.broadcast_to(bs, block_size-1, hidden_size)` (or pre-shape a buffer)
- Continue with `.view(-1, hidden_size)` as before.

The two callers of `_run_drafter_for_next_iter` are on PP1 only:
- prefill drafter-prime, `dflash_worker.py:1730`
- decode post-verify, `dflash_worker.py:1849`

Both flow through the same site. No call-site change needed; the internal lookup at line 844 is the only edit.

### PP0 — the responder

PP0 must service the embed query during the time slice between sending the proxy-tensor chain to PP1 (queued async at `scheduler_pp_mixin.py:166`) and blocking on the output-ring recv inside `_pp_send_recv_and_preprocess_output_tensors` (`scheduler_pp_mixin.py:1235`). The cleanest hook is a new helper `_pp_dflash_serve_embed_query` invoked from `event_loop_pp` immediately after `_pp_send_dict_to_next_stage` at line 169 and before `self.pp_outputs = next_pp_outputs` at line 176. Specifically PP0-only and DFlash-only and decode-only.

There is also a prefill drafter-prime case (see section 3). PP0 in prefill returns from `_pp_launch_batch` with proxy queued, then the same hook fires.

A concrete shape for the helper: each invocation does a single `.send/.recv` pair (PP0 receives `[bs]` int64 verified_ids, replies with `[bs, hidden_size]` bf16). That matches PP1 exactly once per iter.

## 2. NCCL primitives

Use `self.pp_group.send(...)` and `self.pp_group.recv(...)` (defined at `parallel_state.py:1344-1370`). Justification:

- `dist.broadcast` is broken for two reasons: (a) we need a request from PP1 first then a reply from PP0, broadcast is one-direction; (b) it engages all PP ranks, which is fine at PP=2 but unnecessary.
- `send_tensor_dict` / `recv_tensor_dict` add metadata negotiation overhead per call; we already know shapes and dtypes statically.
- `dist.send/recv` with explicit src/dst gives us 2 NCCL ops per iter (one each direction), uses the same `pp_group.device_group` already in place, and runs on pynccl when available (auto-fallback to torch.dist via the group helper).

Concrete primitives, all on `self.device` (cuda):

| Tensor | Direction | Shape | Dtype | Notes |
|---|---|---|---|---|
| `verified_ids_req` | PP1 send → PP0 recv | `[bs]` | `torch.int64` | Reuse `_draft_block_ids_buf[:bs, 0]` view; alloc bus-pinned buffer on first call |
| `embed_resp` | PP0 send → PP1 recv | `[bs, hidden_size]` | `torch.bfloat16` | Pre-allocated reply buffer on PP1 as `self._embed_recv_buf[:bs]` |
| `mask_embed_handshake` (boot only) | PP0 send → PP1 recv | `[1, hidden_size]` | `torch.bfloat16` | Stored as `self._cached_mask_embed` |

Use `pp_group.send(verified_ids_req, dst=0)` on PP1 and `pp_group.recv(size=(bs,), dtype=torch.int64, src=0)` on PP0 (peer rank notation in `pp_group` is rank-in-group). Pre-allocate static buffers on both sides sized to `cap_bs` (mirrors `_draft_block_ids_buf` capacity at line 518) to avoid per-iter alloc.

Note: PP1 must send before recv, and PP0 must recv before send, otherwise pynccl serializes both ops and deadlocks.

## 3. Drafter-prime (end of prefill) handling

At end-of-prefill the drafter runs from `dflash_worker.py:1730`. At this point `verified_id` is set at line 1715: `verified_id=next_token_ids.to(torch.int64)`, where `next_token_ids` is what `tp_worker.forward_batch_generation` returned from `model_runner.sample(logits_output, forward_batch)`. Since prefill terminates on PP1 (the lm_head holder), this is a real PP1-local int tensor.

Therefore: yes, the IPC is needed at prefill drafter-prime too. PP0 must serve the embed query at the prefill iter as well. The good news is the loop structure is symmetric: PP0's `event_loop_pp` runs the same body for prefill and decode iters, just the `cur_batch.forward_mode` differs. The hook from section 1 fires for both — we don't need to special-case forward mode on PP0, just gate on `spec_algorithm.is_dflash() and not is_first_rank` and always serve.

The shape for the prime-iter is `bs=1` always at our current operating point (it's per-prompt prime). Code path is identical.

## 4. Caching `mask_token` embedding

`self._mask_token_id` is resolved at `dflash_worker.py:310-329` inside `__init__` (drafter rank only). The id itself is also broadcast to non-drafter ranks via the existing init dance, so PP0 also knows it after `_pp_drafter_init_collective_stub` returns... actually no, PP0 stubs out `_mask_token_id = -1` at line 184. So we need a small handshake.

**Recommended approach**: lazy-on-first-call from `_run_drafter_for_next_iter`, gated by `self._cached_mask_embed is None`. On the first decode iter (or first prefill drafter-prime, whichever fires first):
- PP1 sends `[1]` int64 = `[mask_token_id]`, then `[1]` int64 = `verified_id` (or sends a single `[1+bs]` packed tensor — slightly cleaner).
- PP0 looks up both, sends back `[1+bs, hidden_size]`.
- PP1 splits: row 0 stashed permanently into `self._cached_mask_embed`, rest is per-iter.

Alternative: bake into a one-time boot handshake during `__init__`. This is cleaner architecturally but harder to sequence — the embed table on PP0 isn't loaded until weight-load, and `__init__` runs before that on the current call order. Lazy-on-first-call is simpler and idempotent.

The PP0 side needs to know the mask_token_id without being told it. Two options:
1. Send `[1+bs]` packed int64 every iter (1-token overhead, trivial). Cleanest.
2. Boot handshake of just the mask id PP1→PP0 at the end of `__init__`. More plumbing.

**Pick option 1**: PP1 always sends `[1+bs]`; PP0 always replies with `[1+bs, hidden_size]`; PP1 caches the first row after the first call. Total cost: 1 extra row per iter, negligible vs the `[bs, hidden_size]` payload.

Wait — actually we can do better. After the first call we know mask is cached. So:
- First call: send `[1+bs]`, recv `[1+bs, H]`, cache row 0.
- Subsequent calls: send `[bs]`, recv `[bs, H]`.

PP0 can disambiguate via the recv-tensor size header... but `pp_group.recv` requires a known size. So use a state machine: PP0 also tracks "have I shipped mask?" booleanly. After the first call, both sides switch to `[bs]`. State must stay consistent — wrap in a single bool `self._mask_embed_synced` on both ranks.

## 5. Edge cases

**Abort mid-iter**: If a request aborts during PP1's verify (between `verify_input.verify(...)` returning and `_run_drafter_for_next_iter`), the IPC has not yet started — no risk. If PP1 dies after sending the request before recv, PP0 is already past send-side, but PP0's `pp_group.send(reply)` blocks if PP1 is gone — same hang as the existing output-ring. We inherit, not worsen, this behavior. Document: PP1 must always complete the IPC pair if it starts it.

**PP=1 (no PP)**: Current code path is `is_pp = self.pp_group.world_size > 1` at line 1820. The new IPC must also gate on `is_pp` AND `not self.pp_group.is_first_rank` for the lookup. When `is_pp` is False, fall back to the original local `embed_module(block_ids)` call. This is just an `if/else` around the lookup at line 844.

**Multi-request batches (bs > 1)**: All buffers in plan use `[:bs]` slices of pre-allocated `cap_bs`-sized tensors, mirroring the existing `_draft_block_ids_buf` pattern. PP0 must learn `bs` per call: PP0 reads it implicitly from the request tensor's first dim, but `pp_group.recv` needs a size argument. We'll send `[bs]` as a tiny `[1]` int32 header tensor first — OR use a fixed `cap_bs` request size with a sentinel padding — OR thread `bs` via the existing scheduler state on PP0, which already knows `cur_batch.batch_size()` post `_pp_launch_batch`. Simplest: PP0 reads `bs = self.cur_batch.batch_size()` and uses that for both recv and send sizes. Both ranks use the same scheduler clock, so this is consistent.

**Deadlock if PP0 is busy**: PP0's `_pp_launch_batch` returns (its 8-layer forward is done and proxy is async-queued) before PP1's lm_head + sample + verify completes. The natural ordering is:

1. PP0: queues proxy chain async (line 166), control returns, hits new IPC hook (recv).
2. PP1: receives proxy, runs its 56 layers + lm_head + sample + verify, then hits drafter prime IPC (send + recv).
3. PP1: send unblocks PP0's recv. PP0: send. PP1: recv unblocks. Done.
4. PP1: builds output-ring tensors and sends.
5. PP0: blocks on output-ring recv inside `_pp_send_recv_and_preprocess_output_tensors`.

So PP0 idles inside the IPC recv during PP1's compute. This adds zero compute to PP0 and replaces PP0's earlier "do-nothing" wait. **No deadlock risk** as long as both ranks gate the IPC on the same condition. Critical: gate on `cur_batch.spec_algorithm.is_dflash() and not is_first_rank` — same gate both ranks see, derived from scheduler state. Pre-flush `send_proxy_work` before entering the IPC (the existing `_pp_commit_comm_work(self.send_proxy_work)` at line 127 covers the previous iter's send, but we must also flush the just-queued one at line 166 before calling recv — see risk register).

## 6. Reverting commit `87a53b9e7`

Single file: `/home/eve/sglang-dflash/python/sglang/srt/models/qwen3_5.py`.

Lines to revert (current numbering):
- Line 69: `from sglang.srt.server_args import get_global_server_args` — keep if used elsewhere; check with `grep`. If only used in the replication block, drop.
- Lines 688-712: replace the full DFlash-replication block back to the simple `if self.pp_group.is_first_rank:` form.

Specifically delete lines 698-703 (the `srv_args` and `replicate_embed_for_dflash` logic) and the `or replicate_embed_for_dflash` clause in line 704. Net delete ~16-18 lines. The trailing `else: PPMissingLayer()` at line 712 stays.

Also remove the comment block at `dflash_worker.py:120-124` ("The target's embed_tokens is replicated on the last rank...") and replace with a one-liner pointing to the IPC path.

`qwen3_vl.py` is **not** touched by `87a53b9e7` (verified — see lines 315-316 use `pos_embed`, the `embed_tokens` pattern at 939-941 is unchanged from upstream). The original task brief was wrong about that. No revert needed there.

## 7. Estimated LoC and file list

| File | Lines added | Lines removed | Notes |
|---|---|---|---|
| `python/sglang/srt/models/qwen3_5.py` | ~3 | ~18 | Revert 87a53b9e7's hunk |
| `python/sglang/srt/speculative/dflash_worker.py` | ~80 | ~5 | New `_pp_embed_lookup` helper + buffer init in `__init__` (drafter side) and `_init_pp_embed_listener` stub on follower side; replace lines 840-845 |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | ~40 | ~0 | New `_pp_dflash_serve_embed_query` helper, hook in `event_loop_pp` after line 169 |
| **Total** | **~123** | **~23** | Net +100 LoC |

Files only read, not modified: `parallel_state.py`, `utils.py`, `qwen3_vl.py`, `qwen3.py`, `tp_worker.py`, `scheduler.py`.

## 8. Test plan

All from the host (not inside the container) over SSH; the bind-mount picks up the patched files automatically on container restart.

**(a) Boot regression smoke**
- Restart container: `ssh -t eve@eve-lambda-vector.local "cd /home/eve/vllm-qwen36 && docker compose -f docker-compose.27b-clean.yml restart"`.
- Tail logs for "DFLASH PP rank" init lines on both ranks; confirm no `embed_tokens` weight on PP1 (memory log should drop ~2.5 GB at PP1 init).
- Send one warm-up prompt; expect normal first-token latency.

**(b) Accept-length non-zero**
- Run the 6-prompt battery: `ssh eve@eve-lambda-vector.local "/tmp/dflash_battery.sh"`.
- Grep `accept_length_per_req` in container logs. Expect non-zero values matching pre-patch baseline (within 5%).
- Specifically: prefill drafter-prime should log `(prefill drafter) primed_candidates_shape=(N,)` on PP1 with N>0.

**(c) Memory headroom: raise `max_total_tokens`**
- After confirming (a) and (b), edit docker-compose to set `max_total_tokens=4096` and `max_running_requests=4`. Restart. Should boot without OOM. PP1 memory should now be `target (~16 GB) + drafter (~3.4 GB) + spec buffers (~1 GB) ≈ 20.4 GB` vs 22.9 GB baseline.
- Re-run battery; confirm bs>1 batches accept correctly (the bs>1 path is currently dead from KV pool limits).

**(d) Throughput regression**
- Re-run battery 3x at `bs=1` and compare median tok/s to working baseline (25-121 tok/s). The IPC adds 2 NCCL ops per iter on a single 100GB/s NVLink-equivalent (4090↔3060 over PCIe Gen4 here, ~16 GB/s effective). Per-iter payload: `bs * hidden_size * 2 bytes` = `1 * 5120 * 2 = 10 KB` request reply = 20 KB total. Expected latency: ~50 µs round-trip. Compared to ~30 ms decode iter, this is <0.2% overhead. **Acceptance criterion: median tok/s within 3% of baseline.**

If (d) fails: profile with `nsys` and check whether `pp_group.send`/`recv` is forcing a `cudaStreamSynchronize`. May need to issue on a dedicated CUDA stream.

## 9. Risk register

**R1 — Stream sync deadlock (highest risk)**: PP1 has multiple async streams in play (forward stream, copy stream, default stream from `_pp_launch_batch`). If `_run_drafter_for_next_iter` runs on the forward stream but the IPC `pp_group.send` runs on the default stream, PP0's recv may complete before PP1's `block_ids[:,0].copy_(verified_id)` is visible to NCCL. **Mitigation**: explicit `torch.cuda.current_stream().synchronize()` before send, OR run send on the same stream as the verify_input.verify call. The existing output-ring writes in `_pp_prepare_tensor_dict` also face this and it works — we should mirror their pattern (no explicit sync, because `_pp_launch_batch` ends with a recorded event and the output-ring send waits on it). Read `_pp_launch_batch` carefully and replicate.

**R2 — `bs` mismatch between PP0 and PP1**: If PP0's `cur_batch.batch_size()` differs from PP1's at the moment of IPC (e.g. due to retraction or filter_batch racing), recv shape mismatch hangs both ranks silently. **First-try-fail signature**: scheduler stops emitting new tokens, `nvidia-smi` shows GPU utilization at 0% on both ranks, no Python traceback. **Mitigation**: send a 2-element `[bs, mask_synced_flag]` int32 header before the verified-id payload; PP0 validates against its own `cur_batch.batch_size()` and raises if mismatched.

**R3 — Mask-cache desync**: If the boot handshake is not strictly in lockstep (e.g., PP0 thinks "mask synced" but PP1 doesn't because the first `_run_drafter_for_next_iter` was skipped due to early abort), all subsequent IPCs use mismatched sizes. **Mitigation**: keep the `mask_synced` flag in the per-iter header (R2's same int32). Cheap and self-healing.

**R4 — PP0 hits the IPC hook but PP1 didn't run drafter this iter**: Possible if PP1's forward path errored out and skipped `_run_drafter_for_next_iter`. PP0 blocks on recv forever. **Mitigation**: wrap PP1's send in try/except; on any exception in verify/drafter, send a sentinel `[bs] = -1` payload so PP0 can recover. Mirror the existing sentinel-marker IPC pattern (commit `27228fd3f`).

**R5 — Async batch depth > 0**: At `pp_async_batch_depth > 0`, the loop reorders ops (line 120-126). The IPC hook must move accordingly or stay disabled. **Mitigation**: assert `pp_async_batch_depth == 0` (which is current production setting) at scheduler init when DFlash + PP is on; document. We're not touching `_pp_commit_send_output_work_and_preprocess_output_tensors`.

**R6 — Bind-mount staleness**: Patches go into `/home/eve/sglang-dflash/python/sglang/srt/...` and are bind-mounted into `/opt/sglang/python/sglang/srt/`. If `__pycache__` is also bind-mounted, stale `.pyc` shadows new `.py`. **Mitigation**: `find /home/eve/sglang-dflash -name __pycache__ -exec rm -rf {} +` between deploys (this is a read-only env so the user runs that, not me).

**R7 — Subtle bug: PP0's `cur_batch` is None on idle iter**: When the server is idle and `cur_batch` is None on line 97, the IPC hook must skip — otherwise PP0 hangs on a recv that will never come (PP1 also has `cur_batch is None` and skips its drafter). **Mitigation**: gate the entire IPC block on `if self.cur_batch and self.cur_batch.spec_algorithm.is_dflash()`.

---

### Critical Files for Implementation

- `/home/eve/sglang-dflash/python/sglang/srt/speculative/dflash_worker.py`
- `/home/eve/sglang-dflash/python/sglang/srt/managers/scheduler_pp_mixin.py`
- `/home/eve/sglang-dflash/python/sglang/srt/models/qwen3_5.py`
- `/home/eve/sglang-dflash/python/sglang/srt/distributed/parallel_state.py` (read-only — confirms send/recv API)
- `/home/eve/sglang-dflash/python/sglang/srt/managers/utils.py` (read-only — `GenerationBatchResult` shape unchanged)
