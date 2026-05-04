from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_harness_module():
    module_name = "_dflash_kt_correctness_harness_under_test"
    source_path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "playground"
        / "dflash_kt_correctness_harness.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness_module()


class TestDFlashKTCorrectnessHarness(unittest.TestCase):
    def test_parse_key_values_preserves_tuple_with_space(self):
        event = harness.parse_log_line(
            "2026-05-03 00:00:00 INFO KT timing "
            "phase=kt.hitmiss layer=7 pp=1 active=True "
            "topk_shape=(8, 6) total_choices=48 gpu_choices=36 "
            "cpu_choices=12 cpu_pct=25.0 unique_gpu=8 unique_cpu=2"
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "kt_timing")
        self.assertEqual(event.fields["phase"], "kt.hitmiss")
        self.assertEqual(event.fields["topk_shape"], "(8, 6)")
        self.assertEqual(event.fields["cpu_pct"], "25.0")

    def test_summarize_telemetry_covers_required_metrics(self):
        lines = [
            "INFO KT timing phase=kt.cpu_stream_elapsed layer=2 pp=0 "
            "tokens=4 elapsed_ms=6.500",
            "INFO KT timing phase=kt.hitmiss layer=2 pp=0 tokens=4 "
            "total_choices=10 gpu_choices=7 cpu_choices=3 cpu_pct=30.0 "
            "unique_gpu=4 unique_cpu=2",
            "INFO KT staging probe phase=h2d_finish layer=2 expert=5 "
            "slot=0 mb=32.000 wait_ms=1.250 hidden_ms=7.750 "
            "total_ms=9.000 copy_ms=6.500",
            "INFO KT staging probe phase=swap layer=2 expert=5 evicted=1 "
            "gpu_idx=0 swaps=1 epoch_swaps=1 global_swaps=1 "
            "elapsed_ms=0.750 weight_copy_ms=0.500",
            "INFO DFLASH H2D overlap probe phase=finish pp=1 "
            "rank_is_last=True mb=64 bs=4 wait_ms=0.500 "
            "hidden_ms=3.500 total_ms=4.000 copy_ms=4.000",
            "INFO DFLASH run_batch timing "
            "phase=scheduler.overlap.worker_forward pp=0 mode=DECODE "
            "bs=4 rank_is_last=False elapsed_ms=20.000",
            "INFO DFLASH run_batch timing "
            "phase=scheduler.overlap.worker_forward pp=1 mode=DECODE "
            "bs=4 rank_is_last=True elapsed_ms=25.000",
        ]
        events = [harness.parse_log_line(line) for line in lines]
        summary = harness.summarize_telemetry(events)

        self.assertEqual(
            summary["kt_cpu_expert_ms_by_layer"]["2"]["count"],
            1,
        )
        self.assertEqual(
            summary["kt_cpu_expert_ms_by_layer"]["2"]["mean"],
            6.5,
        )
        self.assertAlmostEqual(
            summary["kt_hitmiss_by_layer"]["2"]["gpu_hit_rate"],
            0.7,
        )
        self.assertAlmostEqual(
            summary["kt_hitmiss_by_layer"]["2"]["cpu_miss_rate"],
            0.3,
        )
        self.assertEqual(
            summary["kt_staging_h2d_by_layer"]["2"]["wait_ms"]["mean"],
            1.25,
        )
        self.assertEqual(
            summary["kt_staging_h2d_by_layer"]["2"]["hidden_ms"]["mean"],
            7.75,
        )
        self.assertEqual(summary["kt_staging_swaps_by_layer"]["2"]["count"], 1)
        self.assertEqual(
            summary["kt_staging_swaps_by_layer"]["2"]["evicted_unique"],
            1,
        )
        self.assertEqual(
            summary["dflash_h2d_overlap_by_pp"]["1"]["hidden_ms"]["mean"],
            3.5,
        )
        self.assertEqual(
            summary["dflash_decode_throughput_by_pp"]["0"]["tokens_per_s"][
                "mean"
            ],
            200.0,
        )
        self.assertEqual(
            summary["dflash_decode_throughput_by_pp"]["1"]["tokens_per_s"][
                "mean"
            ],
            160.0,
        )

    def test_identity_and_acceptance_summary(self):
        good = harness.RequestProbeResult(
            index=0,
            rid="dflash-kt-000000",
            marker="DFLASHKTID000000",
            elapsed_s=0.5,
            response={
                "text": "DFLASHKTID000000",
                "meta_info": {
                    "id": "dflash-kt-000000",
                    "completion_tokens": 4,
                    "spec_accept_rate": 0.75,
                    "spec_accept_length": 2.0,
                    "spec_accept_token_num": 6,
                    "spec_draft_token_num": 8,
                    "spec_verify_ct": 3,
                },
            },
            meta_id="dflash-kt-000000",
            text="DFLASHKTID000000",
            rid_ok=True,
            marker_found=True,
        )
        bad = harness.RequestProbeResult(
            index=1,
            rid="dflash-kt-000001",
            marker="DFLASHKTID000001",
            elapsed_s=0.4,
            response={"text": "", "meta_info": {"id": "other"}},
            meta_id="other",
            rid_ok=False,
        )

        identity = harness.summarize_identity([good, bad], strict_marker=False)
        response_metrics = harness.summarize_response_metrics([good, bad])

        self.assertFalse(identity["ok"])
        self.assertEqual(identity["rid_mismatches"], 1)
        self.assertEqual(response_metrics["spec_accept_token_num"], 6)
        self.assertEqual(response_metrics["spec_draft_token_num"], 8)
        self.assertEqual(response_metrics["aggregate_spec_accept_rate"], 0.75)
        self.assertEqual(response_metrics["spec_accept_length"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
