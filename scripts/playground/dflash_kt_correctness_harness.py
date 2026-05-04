#!/usr/bin/env python3
"""DFlash PP=2 + KT staging correctness and telemetry harness.

This script is intentionally standalone: it can run against an already-started
server and parse server logs without importing SGLang runtime modules.

Typical use:
    python3 scripts/playground/dflash_kt_correctness_harness.py \
        --base-url http://127.0.0.1:8001 \
        --log-path /tmp/sglang-dflash.log \
        --requests 64 \
        --concurrency 16 \
        --output-json /tmp/dflash-kt-harness.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_RID_PREFIX = "dflash-kt"
EVENT_PREFIXES = (
    ("DFLASH H2D overlap probe", "dflash_h2d_overlap_probe"),
    ("DFLASH run_batch timing", "dflash_run_batch_timing"),
    ("KT staging probe", "kt_staging_probe"),
    ("KT timing", "kt_timing"),
)
KV_RE = re.compile(r"(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=")


@dataclass(frozen=True)
class TelemetryEvent:
    kind: str
    fields: dict[str, str]
    line: str


@dataclass
class RequestProbeResult:
    index: int
    rid: str
    marker: str
    elapsed_s: float
    response: Any = None
    error: str | None = None
    meta_id: str | None = None
    text: str = ""
    rid_ok: bool = False
    marker_found: bool = False
    cross_marker: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.rid_ok and self.cross_marker is None


def parse_kv_fields(text: str) -> dict[str, str]:
    """Parse key=value telemetry fields while preserving tuple/list spaces."""

    matches = list(KV_RE.finditer(text))
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[key] = text[value_start:value_end].strip()
    return fields


def parse_log_line(line: str) -> TelemetryEvent | None:
    for prefix, kind in EVENT_PREFIXES:
        if prefix in line:
            fields_text = line.split(prefix, 1)[1].strip()
            return TelemetryEvent(kind=kind, fields=parse_kv_fields(fields_text), line=line)
    return None


def parse_log_paths(paths: Iterable[Path]) -> list[TelemetryEvent]:
    events: list[TelemetryEvent] = []
    for path in paths:
        with path.open(errors="replace") as f:
            for line in f:
                event = parse_log_line(line.rstrip("\n"))
                if event is not None:
                    events.append(event)
    return events


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "max": max(vals),
    }


def summarize_telemetry(events: Iterable[TelemetryEvent]) -> dict[str, Any]:
    event_list = list(events)
    cpu_ms_by_layer: dict[str, list[float]] = defaultdict(list)
    cpu_ms_by_layer_pp: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    hitmiss_by_layer: dict[str, Counter] = defaultdict(Counter)
    hitmiss_by_layer_pp: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    kt_h2d_by_layer: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dflash_h2d_by_pp: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    swaps_by_layer: dict[str, list[TelemetryEvent]] = defaultdict(list)
    decode_tps_by_pp: dict[str, list[float]] = defaultdict(list)
    decode_elapsed_by_pp: dict[str, list[float]] = defaultdict(list)
    counts = Counter(event.kind for event in event_list)

    for event in event_list:
        fields = event.fields
        phase = fields.get("phase")
        layer = fields.get("layer", "unknown")
        pp = fields.get("pp", "unknown")

        if event.kind == "kt_timing":
            if phase == "kt.cpu_stream_elapsed":
                elapsed = _float(fields.get("elapsed_ms"))
                if elapsed is not None:
                    cpu_ms_by_layer[layer].append(elapsed)
                    cpu_ms_by_layer_pp[layer][pp].append(elapsed)
            if phase == "kt.hitmiss":
                total = _int(fields.get("total_choices"), 0) or 0
                gpu = _int(fields.get("gpu_choices"), 0) or 0
                cpu = _int(fields.get("cpu_choices"), 0) or 0
                unique_gpu = _int(fields.get("unique_gpu"), 0) or 0
                unique_cpu = _int(fields.get("unique_cpu"), 0) or 0
                for bucket in (hitmiss_by_layer[layer], hitmiss_by_layer_pp[layer][pp]):
                    bucket["samples"] += 1
                    bucket["total_choices"] += total
                    bucket["gpu_choices"] += gpu
                    bucket["cpu_choices"] += cpu
                    bucket["unique_gpu_max"] = max(bucket["unique_gpu_max"], unique_gpu)
                    bucket["unique_cpu_max"] = max(bucket["unique_cpu_max"], unique_cpu)

        elif event.kind == "kt_staging_probe":
            if phase == "h2d_finish":
                for metric in ("wait_ms", "hidden_ms", "total_ms", "copy_ms"):
                    value = _float(fields.get(metric))
                    if value is not None:
                        kt_h2d_by_layer[layer][metric].append(value)
            elif phase == "swap":
                swaps_by_layer[layer].append(event)

        elif event.kind == "dflash_h2d_overlap_probe" and phase == "finish":
            for metric in ("wait_ms", "hidden_ms", "total_ms", "copy_ms"):
                value = _float(fields.get(metric))
                if value is not None:
                    dflash_h2d_by_pp[pp][metric].append(value)

        elif event.kind == "dflash_run_batch_timing":
            elapsed = _float(fields.get("elapsed_ms"))
            bs = _int(fields.get("bs"), 0) or 0
            mode = fields.get("mode", "")
            if phase == "scheduler.overlap.worker_forward" and mode.endswith("DECODE"):
                if elapsed and elapsed > 0 and bs > 0:
                    decode_tps_by_pp[pp].append(bs * 1000.0 / elapsed)
                    decode_elapsed_by_pp[pp].append(elapsed)

    def hitmiss_summary(counter: Counter) -> dict[str, Any]:
        total = int(counter["total_choices"])
        gpu = int(counter["gpu_choices"])
        cpu = int(counter["cpu_choices"])
        return {
            "samples": int(counter["samples"]),
            "total_choices": total,
            "gpu_choices": gpu,
            "cpu_choices": cpu,
            "gpu_hit_rate": (gpu / total) if total else 0.0,
            "cpu_miss_rate": (cpu / total) if total else 0.0,
            "unique_gpu_max": int(counter["unique_gpu_max"]),
            "unique_cpu_max": int(counter["unique_cpu_max"]),
        }

    kt_cpu = {
        layer: {
            **summarize_values(values),
            "by_pp": {
                pp: summarize_values(pp_values)
                for pp, pp_values in sorted(cpu_ms_by_layer_pp[layer].items())
            },
        }
        for layer, values in sorted(cpu_ms_by_layer.items(), key=lambda item: item[0])
    }
    kt_hitmiss = {
        layer: {
            **hitmiss_summary(counter),
            "by_pp": {
                pp: hitmiss_summary(pp_counter)
                for pp, pp_counter in sorted(hitmiss_by_layer_pp[layer].items())
            },
        }
        for layer, counter in sorted(hitmiss_by_layer.items(), key=lambda item: item[0])
    }
    kt_h2d = {
        layer: {
            metric: summarize_values(values)
            for metric, values in sorted(metrics.items())
        }
        for layer, metrics in sorted(kt_h2d_by_layer.items(), key=lambda item: item[0])
    }
    dflash_h2d = {
        pp: {
            metric: summarize_values(values)
            for metric, values in sorted(metrics.items())
        }
        for pp, metrics in sorted(dflash_h2d_by_pp.items(), key=lambda item: item[0])
    }
    swaps = {
        layer: summarize_swaps(layer_events)
        for layer, layer_events in sorted(swaps_by_layer.items(), key=lambda item: item[0])
    }
    decode_tps = {
        pp: {
            "tokens_per_s": summarize_values(values),
            "worker_forward_elapsed_ms": summarize_values(decode_elapsed_by_pp[pp]),
        }
        for pp, values in sorted(decode_tps_by_pp.items(), key=lambda item: item[0])
    }

    return {
        "event_counts": dict(counts),
        "kt_cpu_expert_ms_by_layer": kt_cpu,
        "kt_hitmiss_by_layer": kt_hitmiss,
        "kt_staging_h2d_by_layer": kt_h2d,
        "kt_staging_swaps_by_layer": swaps,
        "dflash_h2d_overlap_by_pp": dflash_h2d,
        "dflash_decode_throughput_by_pp": decode_tps,
    }


def summarize_swaps(events: list[TelemetryEvent]) -> dict[str, Any]:
    staged = Counter()
    evicted = Counter()
    pairs = Counter()
    global_swaps_max = 0
    epoch_swaps_max = 0
    elapsed_ms: list[float] = []
    weight_copy_ms: list[float] = []
    for event in events:
        fields = event.fields
        expert = fields.get("expert", "unknown")
        evicted_expert = fields.get("evicted", "unknown")
        staged[expert] += 1
        evicted[evicted_expert] += 1
        pairs[f"{expert}<-{evicted_expert}"] += 1
        global_swaps_max = max(global_swaps_max, _int(fields.get("global_swaps"), 0) or 0)
        epoch_swaps_max = max(epoch_swaps_max, _int(fields.get("epoch_swaps"), 0) or 0)
        elapsed = _float(fields.get("elapsed_ms"))
        if elapsed is not None:
            elapsed_ms.append(elapsed)
        weight_copy = _float(fields.get("weight_copy_ms"))
        if weight_copy is not None:
            weight_copy_ms.append(weight_copy)

    return {
        "count": len(events),
        "staged_unique": len(staged),
        "evicted_unique": len(evicted),
        "repeat_evictions": max(0, len(events) - len(evicted)),
        "top_staged": staged.most_common(8),
        "top_evicted": evicted.most_common(8),
        "top_pairs": pairs.most_common(8),
        "epoch_swaps_max": epoch_swaps_max,
        "global_swaps_max": global_swaps_max,
        "elapsed_ms": summarize_values(elapsed_ms),
        "weight_copy_ms": summarize_values(weight_copy_ms),
    }


def http_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        if not raw:
            return None
        return json.loads(raw)


def fetch_server_info(base_url: str, timeout: float) -> dict[str, Any] | None:
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            result = http_json("GET", f"{base_url}{endpoint}", timeout=timeout)
            return result if isinstance(result, dict) else None
        except Exception:
            continue
    return None


def summarize_server_info(server_info: Mapping[str, Any] | None) -> dict[str, Any]:
    if not server_info:
        return {}
    states = server_info.get("internal_states") or []
    if not isinstance(states, list):
        return {}
    out: dict[str, Any] = {"state_count": len(states), "states": []}
    throughputs: list[float] = []
    accept_lengths: list[float] = []
    for idx, state in enumerate(states):
        if not isinstance(state, Mapping):
            continue
        throughput = _float(state.get("last_gen_throughput"))
        accept_length = _float(state.get("avg_spec_accept_length"))
        item = {
            "index": idx,
            "last_gen_throughput": throughput,
            "avg_spec_accept_length": accept_length,
        }
        out["states"].append(item)
        if throughput is not None:
            throughputs.append(throughput)
        if accept_length is not None:
            accept_lengths.append(accept_length)
    out["last_gen_throughput"] = summarize_values(throughputs)
    out["avg_spec_accept_length"] = summarize_values(accept_lengths)
    return out


def make_marker(index: int) -> str:
    return f"DFLASHKTID{index:06d}"


def make_rid(prefix: str, index: int) -> str:
    return f"{prefix}-{index:06d}"


def make_identity_prompt(index: int, repeat: int) -> tuple[str, str]:
    marker = make_marker(index)
    stem = (
        "Request identity validation.\n"
        f"The request marker is {marker}.\n"
        f"Reply with exactly this marker and no other text: {marker}\n"
    )
    if repeat <= 0:
        return stem, marker
    body = "\n".join(
        f"{i}. Keep this request isolated from concurrent DFlash decode work: {marker}."
        for i in range(repeat)
    )
    return f"{stem}{body}\nFinal answer:", marker


def _extract_meta_id(response: Any) -> str | None:
    if isinstance(response, Mapping):
        meta = response.get("meta_info")
        if isinstance(meta, Mapping):
            value = meta.get("id")
            return str(value) if value is not None else None
    return None


def _extract_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("text")
        return text if isinstance(text, str) else ""
    return ""


def send_identity_request(
    base_url: str,
    rid_prefix: str,
    index: int,
    prompt_repeat: int,
    max_new_tokens: int,
    timeout: float,
    start_event: threading.Event,
) -> RequestProbeResult:
    rid = make_rid(rid_prefix, index)
    prompt, marker = make_identity_prompt(index, prompt_repeat)
    payload = {
        "rid": rid,
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    start_event.wait()
    start = time.perf_counter()
    result = RequestProbeResult(index=index, rid=rid, marker=marker, elapsed_s=0.0)
    try:
        response = http_json("POST", f"{base_url}/generate", payload, timeout=timeout)
        result.elapsed_s = time.perf_counter() - start
        result.response = response
        result.meta_id = _extract_meta_id(response)
        result.text = _extract_text(response)
        result.rid_ok = result.meta_id == rid
        result.marker_found = marker in result.text
    except Exception as exc:  # noqa: BLE001 - diagnostic harness should report errors
        result.elapsed_s = time.perf_counter() - start
        result.error = repr(exc)
    return result


def run_identity_probe(
    base_url: str,
    rid_prefix: str,
    request_count: int,
    concurrency: int,
    prompt_repeat: int,
    max_new_tokens: int,
    timeout: float,
) -> list[RequestProbeResult]:
    start_event = threading.Event()
    results: list[RequestProbeResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_identity_request,
                base_url,
                rid_prefix,
                i,
                prompt_repeat,
                max_new_tokens,
                timeout,
                start_event,
            )
            for i in range(request_count)
        ]
        start_event.set()
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    markers = {result.marker: result.rid for result in results}
    for result in results:
        for marker, owner_rid in markers.items():
            if marker != result.marker and marker in result.text:
                result.cross_marker = owner_rid
                break
    return sorted(results, key=lambda item: item.index)


def flatten_response_items(response: Any) -> list[Mapping[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, Mapping)]
    if isinstance(response, Mapping):
        return [response]
    return []


def summarize_response_metrics(results: Iterable[RequestProbeResult]) -> dict[str, Any]:
    accept_rates: list[float] = []
    accept_lengths: list[float] = []
    accepted_tokens = 0
    draft_tokens = 0
    verify_ct = 0
    completion_tokens: list[float] = []
    latencies: list[float] = []
    for result in results:
        if result.error is None:
            latencies.append(result.elapsed_s)
        for item in flatten_response_items(result.response):
            meta = item.get("meta_info")
            if not isinstance(meta, Mapping):
                continue
            rate = _float(meta.get("spec_accept_rate"))
            length = _float(meta.get("spec_accept_length"))
            accepted = _int(meta.get("spec_accept_token_num"), 0) or 0
            draft = _int(meta.get("spec_draft_token_num"), 0) or 0
            verifies = _int(meta.get("spec_verify_ct"), 0) or 0
            completion = _float(meta.get("completion_tokens"))
            if rate is not None:
                accept_rates.append(rate)
            if length is not None:
                accept_lengths.append(length)
            if completion is not None:
                completion_tokens.append(completion)
            accepted_tokens += accepted
            draft_tokens += draft
            verify_ct += verifies
    return {
        "request_latency_s": summarize_values(latencies),
        "completion_tokens": summarize_values(completion_tokens),
        "spec_accept_rate": summarize_values(accept_rates),
        "spec_accept_length": summarize_values(accept_lengths),
        "spec_accept_token_num": accepted_tokens,
        "spec_draft_token_num": draft_tokens,
        "spec_verify_ct": verify_ct,
        "aggregate_spec_accept_rate": (
            accepted_tokens / draft_tokens if draft_tokens else 0.0
        ),
    }


def summarize_identity(results: Iterable[RequestProbeResult], strict_marker: bool) -> dict[str, Any]:
    result_list = list(results)
    failures = []
    for result in result_list:
        marker_failure = strict_marker and not result.marker_found
        if result.error or not result.rid_ok or result.cross_marker is not None or marker_failure:
            failures.append(
                {
                    "rid": result.rid,
                    "error": result.error,
                    "meta_id": result.meta_id,
                    "rid_ok": result.rid_ok,
                    "marker_found": result.marker_found,
                    "cross_marker_from_rid": result.cross_marker,
                }
            )
    return {
        "requests": len(result_list),
        "ok": not failures,
        "failures": failures,
        "rid_mismatches": sum(1 for result in result_list if not result.rid_ok),
        "cross_marker_hits": sum(1 for result in result_list if result.cross_marker is not None),
        "marker_found": sum(1 for result in result_list if result.marker_found),
        "strict_marker": strict_marker,
    }


def metric_coverage(
    telemetry: Mapping[str, Any],
    response_metrics: Mapping[str, Any],
    identity: Mapping[str, Any],
    server_info: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "cpu_expert_time_per_layer": bool(telemetry.get("kt_cpu_expert_ms_by_layer")),
        "gpu_expert_hit_rate": bool(telemetry.get("kt_hitmiss_by_layer")),
        "cpu_miss_rate": bool(telemetry.get("kt_hitmiss_by_layer")),
        "kt_h2d_wait_hidden": bool(telemetry.get("kt_staging_h2d_by_layer")),
        "dflash_h2d_wait_hidden": bool(telemetry.get("dflash_h2d_overlap_by_pp")),
        "swap_count_and_eviction_churn": bool(
            telemetry.get("kt_staging_swaps_by_layer")
        ),
        "dflash_accept_rate_len": bool(
            response_metrics.get("spec_accept_rate", {}).get("count", 0)
            or response_metrics.get("spec_accept_length", {}).get("count", 0)
            or server_info.get("avg_spec_accept_length", {}).get("count", 0)
        ),
        "pp_decode_throughput": bool(
            telemetry.get("dflash_decode_throughput_by_pp")
            or server_info.get("last_gen_throughput", {}).get("count", 0)
        ),
        "request_identity": bool(identity.get("requests")),
    }


def jsonable_request_results(results: Iterable[RequestProbeResult]) -> list[dict[str, Any]]:
    out = []
    for result in results:
        out.append(
            {
                "index": result.index,
                "rid": result.rid,
                "marker": result.marker,
                "elapsed_s": result.elapsed_s,
                "error": result.error,
                "meta_id": result.meta_id,
                "rid_ok": result.rid_ok,
                "marker_found": result.marker_found,
                "cross_marker_from_rid": result.cross_marker,
                "text_prefix": result.text[:160],
            }
        )
    return out


def print_human_summary(summary: Mapping[str, Any]) -> None:
    print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
    identity = summary.get("identity", {})
    print(
        "identity: "
        f"ok={identity.get('ok')} "
        f"requests={identity.get('requests')} "
        f"rid_mismatches={identity.get('rid_mismatches')} "
        f"cross_marker_hits={identity.get('cross_marker_hits')} "
        f"marker_found={identity.get('marker_found')}"
    )
    response_metrics = summary.get("response_metrics", {})
    print(
        "dflash accept: "
        f"aggregate_rate={response_metrics.get('aggregate_spec_accept_rate')} "
        f"rate={response_metrics.get('spec_accept_rate')} "
        f"len={response_metrics.get('spec_accept_length')}"
    )
    telemetry = summary.get("telemetry", {})
    print(f"telemetry events: {telemetry.get('event_counts', {})}")
    print(f"kt cpu layers: {list(telemetry.get('kt_cpu_expert_ms_by_layer', {}).keys())}")
    print(f"kt hitmiss layers: {list(telemetry.get('kt_hitmiss_by_layer', {}).keys())}")
    print(f"decode throughput pp: {telemetry.get('dflash_decode_throughput_by_pp', {})}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-repeat", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--rid-prefix", default=DEFAULT_RID_PREFIX)
    parser.add_argument("--strict-text-marker", action="store_true")
    parser.add_argument("--skip-requests", action="store_true")
    parser.add_argument("--skip-server-info", action="store_true")
    parser.add_argument("--log-path", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.requests < 0:
        raise ValueError("--requests must be >= 0")

    missing_logs = [str(path) for path in args.log_path if not path.exists()]
    if missing_logs:
        raise FileNotFoundError(f"log path(s) do not exist: {missing_logs}")

    server_info_before = None
    server_info_after = None
    if not args.skip_server_info:
        server_info_before = fetch_server_info(args.base_url, timeout=min(args.timeout, 30))

    results: list[RequestProbeResult] = []
    if not args.skip_requests and args.requests:
        results = run_identity_probe(
            base_url=args.base_url,
            rid_prefix=args.rid_prefix,
            request_count=args.requests,
            concurrency=args.concurrency,
            prompt_repeat=args.prompt_repeat,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )

    if not args.skip_server_info:
        server_info_after = fetch_server_info(args.base_url, timeout=min(args.timeout, 30))

    events = parse_log_paths(args.log_path) if args.log_path else []
    telemetry = summarize_telemetry(events)
    response_metrics = summarize_response_metrics(results)
    identity = summarize_identity(results, strict_marker=args.strict_text_marker)
    server_summary = summarize_server_info(server_info_after)

    summary = {
        "config": {
            "base_url": args.base_url,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "prompt_repeat": args.prompt_repeat,
            "max_new_tokens": args.max_new_tokens,
            "log_paths": [str(path) for path in args.log_path],
        },
        "coverage": metric_coverage(
            telemetry=telemetry,
            response_metrics=response_metrics,
            identity=identity,
            server_info=server_summary,
        ),
        "identity": identity,
        "response_metrics": response_metrics,
        "server_info_before": summarize_server_info(server_info_before),
        "server_info_after": server_summary,
        "telemetry": telemetry,
        "request_results": jsonable_request_results(results),
    }
    print_human_summary(summary)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")

    return 0 if identity.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
