"""Exercise installer failure boundaries without touching the host Docker store."""
import hashlib
import gzip
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallerFailureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rx-met-install-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / 'bundle with spaces'
        self.bundle.mkdir()
        self.prefix = self.root / 'install with spaces'
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.env = os.environ.copy()
        self.env.pop('DOCKER_HOST', None)
        self.env.pop('DOCKER_CONTEXT', None)
        self.env['PATH'] = str(self.tools) + os.pathsep + self.env['PATH']
        self.env['FAKE_DOCKER_LOG'] = str(self.root / 'docker.log')
        self.env['FAKE_IMAGE_PRESENT'] = '0'
        self.env['FAKE_IMAGE_STATE'] = str(self.root / 'image-loaded')
        self.env['FAKE_RUNTIME_TAG'] = 'rx-quant-env-runtime:v1-cu126'
        self.env['FAKE_TAG_STATE'] = str(self.root / 'runtime-tag')
        self.env['FAKE_ARCHIVE'] = base64.b64encode(gzip.compress(b'example docker archive')).decode()
        docker = self.tools / 'docker'
        docker.write_text(f'#!{sys.executable}\n' + '''import os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_DOCKER_LOG'], 'a') as output:
    output.write(repr(args) + '\\n')
if args[:2] == ['context', 'inspect']:
    print('unix:///test.sock')
elif args == ['info']:
    pass
elif args[:2] == ['image', 'inspect']:
    if args[-1] == os.environ['FAKE_RUNTIME_TAG']:
        tag_state = Path(os.environ['FAKE_TAG_STATE'])
        image_id = os.environ.get('FAKE_TAG_ID') or (tag_state.read_text() if tag_state.exists() else '')
        if not image_id:
            sys.exit(1)
        print(image_id)
    else:
        if os.environ.get('FAKE_IMAGE_PRESENT') != '1' and not Path(os.environ['FAKE_IMAGE_STATE']).exists():
            sys.exit(1)
        print('sha256:' + '1' * 64)
elif args and args[0] == 'tag':
    assert args[1] == 'sha256:' + '1' * 64
    assert args[2] == os.environ['FAKE_RUNTIME_TAG']
    Path(os.environ['FAKE_TAG_STATE']).write_text(args[1])
elif args == ['load']:
    assert sys.stdin.buffer.read() == b'example docker archive'
    Path(os.environ['FAKE_IMAGE_STATE']).touch()
elif args and args[0] == 'run':
    sys.stdin.read()
    print('injected container failure', file=sys.stderr)
    sys.exit(42)
else:
    raise SystemExit('unexpected docker operation: ' + repr(args))
''')
        docker.chmod(0o755)
        self.write_bundle()

    def write_bundle(self, url=''):
        manifest = self.bundle / 'manifest.json'
        manifest.write_text(json.dumps({'version': '1.1.0'}))
        bootstrap = {'BUNDLE_NAME': 'rx-met-test', 'MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     'RUNTIME_IMAGE_ID': 'sha256:' + '1' * 64, 'RUNTIME_IMAGE_TAG': self.env['FAKE_RUNTIME_TAG'],
                     'RUNTIME_ARCHIVE': 'runtime.tar.gz',
                     'RUNTIME_ARCHIVE_SHA256': hashlib.sha256(base64.b64decode(self.env['FAKE_ARCHIVE'])).hexdigest(), 'RUNTIME_URL': url, 'APP_INSTALL_BYTES': '1'}
        script = (ROOT / 'packaging/release-bundle/install.sh.in').read_text()
        script = script.replace('@BOOTSTRAP@', '\n'.join(f'readonly {key}={shlex.quote(value)}' for key, value in bootstrap.items()))
        script = script.replace('@VERIFY_INSTALL@', "print('verification fixture')")
        self.installer = self.bundle / 'install.sh'
        self.installer.write_text(script)
        self.installer.chmod(0o755)
        (self.bundle / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in (manifest, self.installer)))

    def invoke(self, *args):
        return subprocess.run([str(self.installer), '--prefix', str(self.prefix), *args], env=self.env, text=True, capture_output=True)

    def assert_failed_with(self, text, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(text, result.stdout + result.stderr)

    def test_offline_missing_runtime_does_not_download_or_run(self):
        result = self.invoke('--offline')
        self.assert_failed_with('离线模式缺少 runtime', result)
        self.assertNotIn("['run'", (self.root / 'docker.log').read_text())

    def test_unconfigured_download_has_actionable_failure(self):
        self.assert_failed_with('尚未配置下载地址', self.invoke())

    def test_corrupt_explicit_archive_fails_even_if_image_is_present(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        archive = self.root / 'runtime.tar.gz'
        archive.write_bytes(b'corrupt archive')
        self.assert_failed_with('镜像归档校验失败', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_missing_runtime_tag_is_created_from_verified_image(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.assert_failed_with('injected container failure', self.invoke('--offline'))
        self.assertEqual((self.root / 'runtime-tag').read_text(), 'sha256:' + '1' * 64)

    def test_matching_runtime_tag_is_reused(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.env['FAKE_TAG_ID'] = 'sha256:' + '1' * 64
        self.assert_failed_with('injected container failure', self.invoke('--offline'))
        self.assertNotIn("['tag'", (self.root / 'docker.log').read_text())

    def test_conflicting_runtime_tag_is_not_overwritten_or_used(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.env['FAKE_TAG_ID'] = 'sha256:' + '2' * 64
        self.assert_failed_with('runtime 标签指向其他镜像', self.invoke('--offline'))
        log = (self.root / 'docker.log').read_text()
        self.assertNotIn("['tag'", log)
        self.assertNotIn("['run'", log)

    def test_tampered_bundle_never_installs(self):
        (self.bundle / 'manifest.json').write_text('{}')
        self.assert_failed_with('发布包校验失败', self.invoke())
        self.assertNotIn("['run'", (self.root / 'docker.log').read_text())

    def test_failure_cleans_staging_and_preserves_previous_version(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        previous = self.prefix / 'previous-version'
        previous.mkdir(parents=True)
        (previous / 'keep').write_text('previous application')
        self.assert_failed_with('injected container failure', self.invoke('--offline'))
        self.assertEqual((previous / 'keep').read_text(), 'previous application')
        self.assertFalse((self.prefix / 'rx-met-test').exists())
        self.assertFalse(list(self.prefix.glob('.rx-met-test.*')))

    def test_existing_different_installation_is_not_overwritten(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        installed = self.prefix / 'rx-met-test'
        installed.mkdir(parents=True)
        (installed / 'manifest.json').write_text('old manifest')
        self.assert_failed_with('已存在其他内容', self.invoke('--offline'))
        self.assertEqual((installed / 'manifest.json').read_text(), 'old manifest')

    def test_remote_daemon_rejected_before_bind_mounts(self):
        self.env['DOCKER_HOST'] = 'ssh://example.invalid'
        self.assert_failed_with('本机 Docker daemon', self.invoke())

    def test_local_attachment_is_verified_and_imported(self):
        images = self.bundle / 'images'
        images.mkdir()
        (images / 'runtime.tar.gz').write_bytes(base64.b64decode(self.env['FAKE_ARCHIVE']))
        self.assert_failed_with('injected container failure', self.invoke('--offline'))
        self.assertTrue((self.root / 'image-loaded').is_file())
        self.assertFalse((self.prefix / 'rx-met-test').exists())

    def test_configured_download_is_verified_cached_and_imported(self):
        self.write_bundle(url='https://example.invalid/runtime.tar.gz')
        curl = self.tools / 'curl'
        curl.write_text(f'#!{sys.executable}\n' +
                        'import base64, os, pathlib, sys\n'
                        'pathlib.Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(base64.b64decode(os.environ["FAKE_ARCHIVE"]))\n')
        curl.chmod(0o755)
        cache = self.root / 'download-cache'
        self.assert_failed_with('injected container failure', self.invoke('--cache-dir', str(cache)))
        self.assertTrue((self.root / 'image-loaded').is_file())
        self.assertTrue((cache / 'runtime.tar.gz').is_file())
        self.assertFalse((cache / 'runtime.tar.gz.partial').exists())

    def test_non_https_download_rejected(self):
        self.write_bundle(url='http://example.invalid/runtime.tar.gz')
        self.assert_failed_with('必须使用 HTTPS', self.invoke())


if __name__ == '__main__':
    unittest.main()
