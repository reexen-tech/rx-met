"""Explicit GRU CMake activation choices take precedence over environment defaults."""

import itertools
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.support.paths import REPO_ROOT


@unittest.skipUnless(shutil.which('cmake') and shutil.which('c++'),
                     'CMake and a C++ compiler are required')
class GruBuildConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-gru-cmake-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def configure(self, build, environment=None, explicit=None):
        env = os.environ.copy()
        env.pop('RX_MET_USE_REAL_ACTIVATION', None)
        env.pop('RX_MET_ENABLE_CUDA', None)
        if environment is not None:
            env['RX_MET_USE_REAL_ACTIVATION'] = environment
        command = ['cmake', '-S', str(REPO_ROOT / 'operators/quant-gru'),
                   '-B', str(build), '-DRX_MET_ENABLE_CUDA=OFF']
        if explicit is not None:
            command.append(f'-DUSE_REAL_ACTIVATION={explicit}')
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_mode(self, build, expected):
        cache = (build / 'CMakeCache.txt').read_text()
        self.assertRegex(cache, rf'(?m)^USE_REAL_ACTIVATION:BOOL={expected}$')

    def test_explicit_option_overrides_environment_defaults(self):
        for environment, explicit in itertools.product((None, '0', '1'), (None, 'OFF', 'ON')):
            with self.subTest(environment=environment, explicit=explicit):
                build = self.directory / f'{environment}-{explicit}'
                self.configure(build, environment, explicit)
                expected = explicit if explicit is not None else ('ON' if environment == '1' else 'OFF')
                self.assert_mode(build, expected)

    def test_reconfigure_preserves_cache_until_explicitly_changed(self):
        build = self.directory / 'cached'
        self.configure(build, environment='1')
        self.assert_mode(build, 'ON')
        self.configure(build, environment='0')
        self.assert_mode(build, 'ON')
        self.configure(build, environment='1', explicit='OFF')
        self.assert_mode(build, 'OFF')
        self.configure(build, environment='1')
        self.assert_mode(build, 'OFF')


if __name__ == '__main__':
    unittest.main()
