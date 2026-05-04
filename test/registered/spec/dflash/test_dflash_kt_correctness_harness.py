"""Registered entrypoint for DFlash KT diagnostic harness tests."""

import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "python"))
_SPEC_TEST_DIR = _REPO_ROOT / "python" / "sglang" / "test" / "speculative"
sys.path.insert(0, str(_SPEC_TEST_DIR))

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from test_dflash_kt_correctness_harness import (  # noqa: E402,F401
    TestDFlashKTCorrectnessHarness,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


if __name__ == "__main__":
    unittest.main()
