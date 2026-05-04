# DFlash KT Correctness Harness

`dflash_kt_correctness_harness.py` drives concurrent `/generate` requests against an already-running DFlash server and parses server logs into KT/DFlash telemetry summaries.

Recommended server-side environment for PP=2 + KT dynamic staging diagnostics:

```bash
export SGLANG_KT_TIMING=1
export SGLANG_KT_HITMISS_TIMING=1
export SGLANG_KT_TIMING_SYNC=1
export SGLANG_DFLASH_RUN_BATCH_TIMING=1
export SGLANG_DFLASH_H2D_OVERLAP_PROBE=1
export SGLANG_DFLASH_H2D_OVERLAP_PROBE_RANK=all
export SGLANG_KT_STAGING_PROBE=1
export SGLANG_KT_STAGING_SWAP=1
```

Run the harness against the default local endpoint:

```bash
python3 scripts/playground/dflash_kt_correctness_harness.py \
  --base-url http://127.0.0.1:8001 \
  --log-path /tmp/sglang-dflash.log \
  --requests 64 \
  --concurrency 16 \
  --output-json /tmp/dflash-kt-harness.json
```

The JSON output includes:

- CPU expert elapsed time per layer from `KT timing phase=kt.cpu_stream_elapsed`.
- GPU expert hit rate and CPU miss rate from `KT timing phase=kt.hitmiss`.
- KT staging H2D wait/hidden/total/copy time from `KT staging probe phase=h2d_finish`.
- DFlash synthetic H2D overlap wait/hidden/total/copy time from `DFLASH H2D overlap probe phase=finish`.
- Swap count and eviction churn from `KT staging probe phase=swap`.
- PP decode worker throughput from `DFLASH run_batch timing phase=scheduler.overlap.worker_forward mode=...DECODE`.
- DFlash accept rate/length from per-response `meta_info` and `/server_info`.
- Request identity checks from explicit `rid` values under concurrent `/generate` load.

`--strict-text-marker` additionally fails the run if the model does not echo each request's prompt marker. This is useful for catching response-content swaps, but it is model-behavior dependent; the `rid` check is deterministic.
