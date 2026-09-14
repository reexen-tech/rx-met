"""Setuptools hooks for the platform-specific rx-met wheel."""

import platform
import sys
import sysconfig
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py


_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
_NATIVE_FILES = (
    f"_libpymo{_EXT_SUFFIX}",
    f"libquant_info{_EXT_SUFFIX}",
    "libaimet_onnxrt_ops.so",
)


class _BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class _BuildPyWithAimetNative(_build_py):
    def run(self):
        if (
            sys.implementation.name != "cpython"
            or sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}
            or platform.system() != "Linux"
            or platform.machine() != "x86_64"
        ):
            raise RuntimeError(
                "rx-met native wheels require CPython 3.10-3.12 on Linux x86_64"
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
