"""The group runner must select deterministically and reject empty matches."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests import __main__ as runner


class TestRunnerTest(unittest.TestCase):
    def test_default_and_all_cover_each_group_once(self):
        self.assertEqual(runner.select_suites([]), list(runner.SUITES))
        self.assertEqual(runner.select_suites(["common", "all"]), list(runner.SUITES))
        self.assertEqual(runner.select_suites(["release", "common", "release"]),
                         ["release", "common"])

    def test_invalid_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "misspelled"):
            runner.select_suites(["misspelled"])

    def test_module_listing_does_not_import_test_modules(self):
        with patch.object(unittest.TestLoader, "discover", side_effect=AssertionError("unexpected import")), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(["common", "--list"]), 0)
        self.assertIn("tests.common.test_runner", stream.getvalue())

    def test_zero_matching_tests_is_an_error(self):
        with patch.object(unittest.TestLoader, "discover", return_value=unittest.TestSuite()), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            runner.main(["common", "-k", "no_such_test"])
        self.assertEqual(caught.exception.code, 2)

    def test_subprocess_uses_the_same_source_checkout(self):
        result = subprocess.run(
            [sys.executable, "-c", "from aimet_common import run_context; print(run_context.__file__)"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(Path(result.stdout.strip()).resolve(),
                         (runner.REPO_ROOT / "src/aimet_common/run_context.py").resolve())
