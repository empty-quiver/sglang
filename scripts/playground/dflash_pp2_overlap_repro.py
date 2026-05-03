#!/usr/bin/env python3
"""Manual DFLASH spec-v2 PP=2 overlap repro.

This script starts a PP=2 DFLASH server, sends two concurrent long prompts,
and scans the server log for the PP/spec-v2 overlap patterns that usually
explain candidate/future-index alignment failures.

Example:
    PYTHONPATH=python python3 scripts/playground/dflash_pp2_overlap_repro.py

To reuse an already running server:
    python3 scripts/playground/dflash_pp2_overlap_repro.py \
        --no-launch --base-url http://127.0.0.1:30000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"

PROGRESS_PATTERNS = (
    "DFLASH speculative decoding with pp_size=2",
    "DFLASH plan timeline",
    "DFLASH spec-v2 PP verify sample",
    "dflash_next_candidates",
    "next_candidates_shape",
    "candidate_shape",
    "future_indices",
)

FAILURE_PATTERNS = (
    "shape mismatch",
    "next_positions without next_candidates",
    "cannot merge batches with mismatched",
    "IndexError",
    "RuntimeError",
    "CUDA error",
    "illegal memory access",
    "out of bounds",
    "leak",
    "Abort",
)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 30,
) -> dict | list | str | None:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def wait_for_health(
    base_url: str,
    process: subprocess.Popen | None,
    timeout_s: float,
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_s
    last_error = "server did not respond"
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {process.returncode}\n"
                f"log tail:\n{tail(log_path)}"
            )
        try:
            http_json("GET", f"{base_url}/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            last_error = repr(exc)
            time.sleep(2)
    raise TimeoutError(f"timed out waiting for {base_url}/health: {last_error}")


def long_prompt(request_id: int, repeat: int) -> str:
    prefix = (
        f"Request {request_id}: summarize the following numbered deployment "
        "notes and preserve ordering.\n"
    )
    body = "\n".join(
        f"{i}. Pipeline parallel DFLASH overlap note {request_id}-{i}: "
        "keep candidate blocks aligned with request-major future indices."
        for i in range(repeat)
    )
    return f"{prefix}{body}\nSummary:"


def post_generate(
    base_url: str,
    model: str,
    request_id: int,
    prompt_repeat: int,
    max_new_tokens: int,
    timeout_s: float,
    barrier: threading.Barrier,
) -> dict | list | str | None:
    payload = {
        "text": long_prompt(request_id, prompt_repeat),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    barrier.wait()
    start = time.time()
    result = http_json("POST", f"{base_url}/generate", payload, timeout=timeout_s)
    elapsed = time.time() - start
    print(f"request {request_id} completed in {elapsed:.2f}s")
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            print(f"request {request_id} text prefix: {text[:160]!r}")
    return result


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<log file does not exist>"
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def scan_log(path: Path, patterns: Iterable[str]) -> list[str]:
    if not path.exists():
        return []
    matches: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if any(pattern in line for pattern in patterns):
            matches.append(line)
    return matches


def print_patterns() -> None:
    print("Progress log patterns to watch:")
    for pattern in PROGRESS_PATTERNS:
        print(f"  - {pattern}")
    print("Failure log patterns to watch:")
    for pattern in FAILURE_PATTERNS:
        print(f"  - {pattern}")


def launch_server(args: argparse.Namespace, base_url: str, log_path: Path):
    host, port = base_url.replace("http://", "").split(":", maxsplit=1)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--host",
        host,
        "--port",
        port,
        "--pp-size",
        "2",
        "--attention-backend",
        args.attention_backend,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        args.draft_model,
        "--speculative-num-draft-tokens",
        str(args.block_size),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        "2",
        "--cuda-graph-bs",
        "1",
        "2",
        "--log-level",
        args.log_level,
        *args.extra_server_arg,
    ]
    env = os.environ.copy()
    env.update(
        {
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_DFLASH_PP_TIMELINE": "1",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1",
            "SGLANG_SPEC_NAN_DETECTION": "1",
            "SGLANG_SPEC_OOB_DETECTION": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }
    )
    if args.overlap_plan_stream:
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"

    print("Launching server:")
    print(" ".join(cmd))
    log_file = log_path.open("w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return process, log_file


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=os.getenv("SGLANG_DFLASH_TARGET", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--draft-model",
        default=os.getenv("SGLANG_DFLASH_DRAFT", DEFAULT_DRAFT_MODEL),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--chunked-prefill-size", type=int, default=512)
    parser.add_argument("--prompt-repeat", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--server-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--log-level", default="debug")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--overlap-plan-stream", action="store_true")
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        help="Additional launch_server argument. Repeat for multiple args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    port = args.port or find_free_port()
    base_url = args.base_url or f"http://{args.host}:{port}"
    log_path = args.log_path or Path(
        tempfile.gettempdir(), f"dflash_pp2_overlap_{port}.log"
    )

    print(f"base_url={base_url}")
    print(f"log_path={log_path}")
    print_patterns()

    process = None
    log_file = None
    try:
        if not args.no_launch:
            process, log_file = launch_server(args, base_url, log_path)
        wait_for_health(base_url, process, args.server_timeout, log_path)

        barrier = threading.Barrier(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    post_generate,
                    base_url,
                    args.model,
                    request_id,
                    args.prompt_repeat,
                    args.max_new_tokens,
                    args.request_timeout,
                    barrier,
                )
                for request_id in (1, 2)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        progress_matches = scan_log(log_path, PROGRESS_PATTERNS)
        failure_matches = scan_log(log_path, FAILURE_PATTERNS)
        print(f"progress_matches={len(progress_matches)}")
        for line in progress_matches[-20:]:
            print(line)
        print(f"failure_matches={len(failure_matches)}")
        for line in failure_matches[-40:]:
            print(line)
        return 0 if not failure_matches else 2
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"diagnostic failed: {exc}", file=sys.stderr)
        print(f"log tail:\n{tail(log_path)}", file=sys.stderr)
        return 1
    finally:
        if log_file is not None:
            log_file.close()
        if process is not None and not args.keep_server:
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
