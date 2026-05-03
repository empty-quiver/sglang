from __future__ import annotations

import enum
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

_MISSING = object()
_STUB_NAMES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.managers",
    "sglang.srt.mem_cache",
    "sglang.srt.speculative",
    "sglang.srt.utils",
    "sglang.srt.environ",
    "sglang.srt.managers.overlap_utils",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.mem_cache.common",
    "sglang.srt.server_args",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.spec_utils",
    "sglang.srt.utils.common",
)


def _remember_module(previous: dict[str, object], name: str):
    if name not in previous:
        previous[name] = sys.modules.get(name, _MISSING)


def _set_stub(previous: dict[str, object], name: str, module: types.ModuleType):
    _remember_module(previous, name)
    sys.modules[name] = module


def _stub_package(previous: dict[str, object], name: str):
    module = types.ModuleType(name)
    module.__path__ = []
    _set_stub(previous, name, module)
    return module


def _stub_module(previous: dict[str, object], name: str):
    module = types.ModuleType(name)
    _set_stub(previous, name, module)
    return module


def _unused(*args, **kwargs):
    raise AssertionError("stubbed dependency should not be called by this test")


def _install_import_stubs():
    previous = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.mem_cache",
        "sglang.srt.speculative",
        "sglang.srt.utils",
    ):
        _stub_package(previous, name)

    environ = _stub_module(previous, "sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_ENABLE_OVERLAP_PLAN_STREAM=SimpleNamespace(get=lambda: False)
    )

    overlap_utils = _stub_module(previous, "sglang.srt.managers.overlap_utils")

    @dataclass
    class FutureIndices:
        indices: torch.Tensor
        interval: slice | None = None

    overlap_utils.FutureIndices = FutureIndices

    schedule_batch = _stub_module(previous, "sglang.srt.managers.schedule_batch")
    schedule_batch.ScheduleBatch = type("ScheduleBatch", (), {})

    mem_cache_common = _stub_module(previous, "sglang.srt.mem_cache.common")
    mem_cache_common.alloc_paged_token_slots_extend = _unused
    mem_cache_common.alloc_token_slots = _unused
    mem_cache_common.get_last_loc = _unused

    server_args = _stub_module(previous, "sglang.srt.server_args")
    server_args.get_global_server_args = lambda: SimpleNamespace(
        speculative_num_draft_tokens=1
    )

    spec_info = _stub_module(previous, "sglang.srt.speculative.spec_info")

    class SpecInput:
        def __init__(self, spec_input_type):
            self.spec_input_type = spec_input_type

    class SpecInputType(enum.Enum):
        DFLASH_DRAFT = "dflash_draft"

    spec_info.SpecInput = SpecInput
    spec_info.SpecInputType = SpecInputType

    spec_utils = _stub_module(previous, "sglang.srt.speculative.spec_utils")
    spec_utils.assign_req_to_token_pool_func = _unused

    utils_common = _stub_module(previous, "sglang.srt.utils.common")
    utils_common.is_pin_memory_available = lambda: False

    return previous


def _restore_import_stubs(previous: dict[str, object]):
    for name in reversed(_STUB_NAMES):
        old_module = previous.get(name, _MISSING)
        if old_module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module


def _load_dflash_info_v2():
    previous = _install_import_stubs()
    module_name = "_dflash_info_v2_under_test"
    source_path = (
        Path(__file__).resolve().parents[2]
        / "srt"
        / "speculative"
        / "dflash_info_v2.py"
    )
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_import_stubs(previous)


dflash_info_v2 = _load_dflash_info_v2()
DFlashDraftInputV2 = dflash_info_v2.DFlashDraftInputV2
FutureIndices = dflash_info_v2.FutureIndices


def _rows_tensor(start: int, rows: int, width: int, dtype: torch.dtype):
    return torch.arange(start, start + rows * width, dtype=dtype).reshape(rows, width)


def _draft_input(
    *,
    rows: int,
    future_rows: int | None = None,
    block_size: int = 3,
    offset: int = 0,
    include_lengths: bool = False,
) -> DFlashDraftInputV2:
    block_rows = future_rows if future_rows is not None else rows
    draft = DFlashDraftInputV2(
        topk_p=_rows_tensor(offset, rows, 2, torch.float32),
        topk_index=_rows_tensor(offset + 100, rows, 2, torch.int64),
        verified_id=torch.arange(
            offset + 200, offset + 200 + rows, dtype=torch.int32
        ),
        new_seq_lens=torch.arange(
            offset + 300, offset + 300 + rows, dtype=torch.int32
        ),
        hidden_states=_rows_tensor(offset + 400, rows, 2, torch.float32),
        next_candidates=torch.arange(
            offset + 500,
            offset + 500 + block_rows * block_size,
            dtype=torch.int32,
        ),
        next_positions=torch.arange(
            offset + 700,
            offset + 700 + block_rows * block_size,
            dtype=torch.int32,
        ),
    )
    if include_lengths:
        draft.cur_allocated_seq_lens_cpu = torch.arange(
            offset + 10, offset + 10 + rows, dtype=torch.int32
        )
        draft.planning_seq_lens_cpu = torch.arange(
            offset + 20, offset + 20 + rows, dtype=torch.int32
        )
        draft.planning_seq_lens_sum = int(draft.planning_seq_lens_cpu.sum().item())
        draft.reserved_seq_lens_cpu = torch.arange(
            offset + 30, offset + 30 + rows, dtype=torch.int32
        )
        draft.reserved_seq_lens_sum = int(draft.reserved_seq_lens_cpu.sum().item())
    if future_rows is not None:
        draft.future_indices = FutureIndices(
            indices=torch.arange(offset + 900, offset + 900 + future_rows)
        )
    return draft


class TestDFlashDraftInputV2(unittest.TestCase):
    def _block_size(self, block_size: int):
        return patch.object(
            dflash_info_v2,
            "get_global_server_args",
            return_value=SimpleNamespace(speculative_num_draft_tokens=block_size),
        )

    def test_row_count_prefers_future_indices(self):
        draft = _draft_input(rows=5, future_rows=3)

        self.assertEqual(draft.row_count(), 3)

    def test_filter_future_backed_fields_and_candidate_blocks(self):
        block_size = 3
        draft = _draft_input(
            rows=4, future_rows=4, block_size=block_size, include_lengths=True
        )
        keep = torch.tensor([3, 1], dtype=torch.long)
        expected_candidates = draft.next_candidates.view(4, block_size)[keep].reshape(
            -1
        )
        expected_positions = draft.next_positions.view(4, block_size)[keep].reshape(-1)

        with self._block_size(block_size):
            draft.filter_batch(keep)

        self.assertEqual(draft.row_count(), 2)
        self.assertTrue(
            torch.equal(draft.future_indices.indices, torch.tensor([903, 901]))
        )
        self.assertTrue(
            torch.equal(draft.verified_id, torch.tensor([203, 201], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                draft.new_seq_lens, torch.tensor([303, 301], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.equal(draft.next_candidates, expected_candidates))
        self.assertTrue(torch.equal(draft.next_positions, expected_positions))
        self.assertTrue(
            torch.equal(
                draft.cur_allocated_seq_lens_cpu,
                torch.tensor([13, 11], dtype=torch.int32),
            )
        )
        self.assertEqual(draft.planning_seq_lens_sum, 44)
        self.assertEqual(draft.reserved_seq_lens_sum, 64)

    def test_filter_invalidates_stale_future_backed_fields(self):
        block_size = 3
        draft = _draft_input(rows=3, future_rows=4, block_size=block_size)

        with self._block_size(block_size):
            draft.filter_batch(torch.tensor([2, 0], dtype=torch.long))

        self.assertEqual(draft.row_count(), 2)
        self.assertEqual(draft.verified_id.shape[0], 0)
        self.assertEqual(draft.new_seq_lens.shape[0], 0)
        self.assertEqual(draft.topk_p.shape[0], 0)

    def test_merge_uses_future_row_count_for_candidate_validation(self):
        block_size = 3
        left = _draft_input(rows=7, future_rows=2, block_size=block_size)
        right = _draft_input(
            rows=5, future_rows=1, block_size=block_size, offset=1000
        )
        expected_candidates = torch.cat([left.next_candidates, right.next_candidates])
        expected_positions = torch.cat([left.next_positions, right.next_positions])

        with self._block_size(block_size):
            left.merge_batch(right)

        self.assertEqual(left.row_count(), 3)
        self.assertTrue(
            torch.equal(left.future_indices.indices, torch.tensor([900, 901, 1900]))
        )
        self.assertTrue(torch.equal(left.next_candidates, expected_candidates))
        self.assertTrue(torch.equal(left.next_positions, expected_positions))
        self.assertEqual(left.verified_id.shape[0], 0)
        self.assertEqual(left.new_seq_lens.shape[0], 0)

    def test_merge_empty_idle_input_adopts_non_empty_future_backed_state(self):
        block_size = 3
        left = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        right = _draft_input(
            rows=5, future_rows=2, block_size=block_size, offset=1000
        )

        with self._block_size(block_size):
            left.merge_batch(right)

        self.assertEqual(left.row_count(), 2)
        self.assertIsNotNone(left.future_indices)
        self.assertTrue(
            torch.equal(left.future_indices.indices, torch.tensor([1900, 1901]))
        )
        self.assertTrue(torch.equal(left.next_candidates, right.next_candidates))
        self.assertTrue(torch.equal(left.next_positions, right.next_positions))

    def test_merge_non_empty_ignores_empty_idle_input(self):
        block_size = 3
        left = _draft_input(rows=2, future_rows=2, block_size=block_size)
        expected_candidates = left.next_candidates.clone()
        right = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))

        with self._block_size(block_size):
            left.merge_batch(right)

        self.assertEqual(left.row_count(), 2)
        self.assertTrue(torch.equal(left.next_candidates, expected_candidates))

    def test_filter_rejects_malformed_candidate_block(self):
        block_size = 3
        draft = _draft_input(rows=2, future_rows=2, block_size=block_size)
        draft.next_positions = draft.next_positions[:-1]

        with self._block_size(block_size):
            with self.assertRaisesRegex(RuntimeError, "next_positions shape mismatch"):
                draft.filter_batch(torch.tensor([0], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
