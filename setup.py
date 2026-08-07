"""Setuptools hooks for the platform-specific rx-met wheel."""

import platform
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py


_NATIVE_FILES = (
    "_libpymo.cpython-310-x86_64-linux-gnu.so",
    "libquant_info.cpython-310-x86_64-linux-gnu.so",
    "libaimet_onnxrt_ops.so",
)


class _BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class _BuildPyWithAimetNative(_build_py):
    def run(self):
        if (
            sys.implementation.name != "cpython"
            or sys.version_info[:2] != (3, 10)
            or platform.system() != "Linux"
            or platform.machine() != "x86_64"
        ):
            raise RuntimeError(
                "rx-met native wheels require CPython 3.10 on Linux x86_64"
            )

        super().run()
        native_dir = Path(self.build_lib) / "aimet_common"
        missing = [name for name in _NATIVE_FILES if not (native_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                "AIMET ONNX native runtime is incomplete: "
                f"{', '.join(missing)}. Run scripts/build_aimet_native.sh."
            )


setup(
    distclass=_BinaryDistribution,
    cmdclass={"build_py": _BuildPyWithAimetNative},
)
