# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "kt_staging_controller.py"
)
_SPEC = importlib.util.spec_from_file_location("kt_staging_controller", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

KTStagingController = _MODULE.KTStagingController


def test_runtime_controller_selects_cpu_miss_candidate():
    controller = KTStagingController(num_experts=4)
    controller.apply_counts(
        route_counts=[10, 1, 8, 2],
        cpu_counts=[0, 0, 8, 2],
        gpu_counts=[10, 1, 0, 0],
        decay=1.0,
    )

    selected = controller.choose_candidate(
        cold_experts=[2, 3],
        inflight_experts=set(),
        evict_for_expert=lambda _: (1, 1),
        min_score=1.0,
        min_effective_score=0.0,
        predicted_total_cost_ms=1.0,
        commit_margin=0.0,
        observation_count=1,
    )

    assert selected is not None
    assert selected[0] == 2


def test_runtime_controller_cooldown_blocks_immediate_swap_back():
    controller = KTStagingController(num_experts=4)
    controller.apply_counts(
        route_counts=[10, 1, 8, 2],
        cpu_counts=[0, 0, 8, 2],
        gpu_counts=[10, 1, 0, 0],
        decay=1.0,
    )
    controller.record_commit(staged_expert=2, evicted_expert=1, observation_count=1)

    selected = controller.choose_candidate(
        cold_experts=[1],
        inflight_experts=set(),
        evict_for_expert=lambda _: (2, 0),
        min_score=0.0,
        min_effective_score=0.0,
        predicted_total_cost_ms=1.0,
        commit_margin=0.0,
        observation_count=1,
    )

    assert selected is None
