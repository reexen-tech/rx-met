"""Run selected unittest groups: python3 -m tests common release."""

import argparse
from pathlib import Path
import unittest

from tests.support.paths import REPO_ROOT


SUITES = ("common", "torch", "onnx", "recurrent", "integration", "examples", "release")


def select_suites(names):
    names = names or ["all"]
    unknown = set(names) - {*SUITES, "all"}
    if unknown:
        raise ValueError("Unknown test group(s): " + ", ".join(sorted(unknown)))
    return list(SUITES) if "all" in names else list(dict.fromkeys(names))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", metavar="GROUP",
                        help="all (default), or: " + ", ".join(SUITES))
    parser.add_argument("--list", action="store_true", help="List test modules without importing them")
    parser.add_argument("-k", action="append", default=[], metavar="PATTERN",
                        help="Match test names, as in unittest -k; may be repeated")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--failfast", action="store_true")
    args = parser.parse_args(argv)
    try:
        groups = select_suites(args.suites)
    except ValueError as error:
        parser.error(str(error))
    directory = Path(__file__).resolve().parent
    if args.list:
        for group in groups:
            for path in sorted((directory / group).glob("test_*.py")):
                print(".".join(path.relative_to(REPO_ROOT).with_suffix("").parts))
        return 0
    loader = unittest.TestLoader()
    if args.k:
        loader.testNamePatterns = [pattern if "*" in pattern else f"*{pattern}*" for pattern in args.k]
    suite = unittest.TestSuite()
    for group in groups:
        suite.addTests(loader.discover(str(directory / group), pattern="test_*.py",
                                      top_level_dir=str(REPO_ROOT)))
    if suite.countTestCases() == 0:
        parser.error("No tests matched the selected groups and patterns")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1,
                                    failfast=args.failfast).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
