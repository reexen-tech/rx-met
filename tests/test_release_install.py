"""Exercise installer failure boundaries without touching the host Docker store."""
import hashlib
import gzip
import base64
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
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
        self.config = b'{"architecture":"amd64","os":"linux","rootfs":{"diff_ids":[]}}'
        self.runtime_id = 'sha256:' + hashlib.sha256(self.config).hexdigest()
        self.env['FAKE_EXPECTED_IMAGE_ID'] = self.runtime_id
        self.env['FAKE_ARCHIVE'] = base64.b64encode(self.archive_bytes()).decode()
        docker = self.tools / 'docker'
        docker.write_text(f'#!{sys.executable}\n' + r'''import hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path
args = sys.argv[1:]
expected = os.environ['FAKE_EXPECTED_IMAGE_ID']
state = Path(os.environ['FAKE_IMAGE_STATE'])
images = json.loads(state.read_text()) if state.exists() else {}
with open(os.environ['FAKE_DOCKER_LOG'], 'a') as output:
    output.write(repr(args) + '\n')
if args[:2] == ['context', 'inspect']:
    print('unix:///test.sock')
elif args == ['info']:
    pass
elif args[:2] == ['image', 'inspect']:
    reference = args[-1]
    if reference == os.environ['FAKE_RUNTIME_TAG']:
        tag_state = Path(os.environ['FAKE_TAG_STATE'])
        image_id = os.environ.get('FAKE_TAG_ID') or (tag_state.read_text() if tag_state.exists() else '')
    else:
        image_id = images.get(reference)
        if reference == expected and os.environ.get('FAKE_IMAGE_PRESENT') == '1':
            image_id = expected
    if not image_id:
        sys.exit(1)
    print(image_id)
elif args and args[0] == 'tag':
    assert args[1] == expected
    assert args[2] == os.environ['FAKE_RUNTIME_TAG']
    Path(os.environ['FAKE_TAG_STATE']).write_text(args[1])
elif args[:2] == ['load', '--input']:
    if os.environ.get('FAKE_LOAD_FAIL') == '1':
        sys.exit(1)
    with tarfile.open(args[2], 'r:gz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        for item in manifest:
            image_id = 'sha256:' + hashlib.sha256(archive.extractfile(item['Config']).read()).hexdigest()
            images[image_id] = image_id
            for tag in item.get('RepoTags') or []:
                images[tag] = image_id
                if tag == os.environ['FAKE_RUNTIME_TAG']:
                    Path(os.environ['FAKE_TAG_STATE']).write_text(image_id)
            if os.environ.get('FAKE_LOAD_REPORT', 'normal') != 'none':
                if item.get('RepoTags'):
                    for tag in item['RepoTags']:
                        print('Loaded image: ' + tag)
                else:
                    print('Loaded image ID: ' + image_id)
    state.write_text(json.dumps(images))
elif args and args[0] == 'run':
    code = sys.stdin.read()
    mount = next((arg for arg in args if arg.startswith('type=bind,src=') and ',dst=/runtime-image.tar.gz,' in arg), None)
    if mount:
        path = mount.split('src=', 1)[1].split(',dst=', 1)[0]
        result = subprocess.run([sys.executable, '-B', '-', path, expected], input=code, text=True)
        sys.exit(result.returncode)
    print('injected container failure', file=sys.stderr)
    sys.exit(42)
else:
    raise SystemExit('unexpected docker operation: ' + repr(args))
''')
        docker.chmod(0o755)
        self.write_bundle()

    def archive_bytes(self, config=None, tags=None, extra_image=False, layers=None):
        config = self.config if config is None else config
        config_name = hashlib.sha256(config).hexdigest() + '.json'
        manifest = [{'Config': config_name, 'RepoTags': tags, 'Layers': list(layers or {})}]
        files = {config_name: config, **(layers or {})}
        if extra_image:
            other = b'{"other":"image"}'
            other_name = hashlib.sha256(other).hexdigest() + '.json'
            files[other_name] = other
            manifest.append({'Config': other_name, 'Layers': []})
        files['manifest.json'] = json.dumps(manifest).encode()
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as tar:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        return gzip.compress(output.getvalue(), mtime=0)

    def local_archive(self, data=None):
        path = self.root / 'runtime.tar.gz'
        path.write_bytes(self.archive_bytes() if data is None else data)
        return path

    def fake_curl(self):
        curl = self.tools / 'curl'
        curl.write_text(f'#!{sys.executable}\n' + r'''import base64, hashlib, os, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
payload = base64.b64decode(os.environ['FAKE_ARCHIVE'])
if args[-1].endswith('.sha256'):
    status = os.environ.get('FAKE_CHECKSUM_STATUS', '404')
    digest = os.environ.get('FAKE_CHECKSUM', hashlib.sha256(payload).hexdigest())
    output.write_text(digest + '\n' if status == '200' else 'not available')
    print(status, end='')
else:
    output.write_bytes(payload)
''')
        curl.chmod(0o755)

    def write_bundle(self, url=''):
        manifest = self.bundle / 'manifest.json'
        manifest.write_text(json.dumps({'version': '1.1.0'}))
        bootstrap = {'BUNDLE_NAME': 'rx-met-test', 'MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     'RUNTIME_IMAGE_ID': self.runtime_id, 'RUNTIME_IMAGE_TAG': self.env['FAKE_RUNTIME_TAG'],
                     'RUNTIME_ARCHIVE': 'runtime.tar.gz',
                     'RUNTIME_URL': url, 'RUNTIME_CHECKSUM_URL': url + '.sha256' if url else '', 'APP_INSTALL_BYTES': '1'}
        script = (ROOT / 'packaging/release-bundle/install.sh.in').read_text()
        script = script.replace('@BOOTSTRAP@', '\n'.join(f'readonly {key}={shlex.quote(value)}' for key, value in bootstrap.items()))
        script = script.replace('@VERIFY_RUNTIME_ARCHIVE@', (ROOT / 'scripts/release/verify_runtime_archive.py').read_text())
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
        self.assert_failed_with('镜像归档损坏', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_missing_runtime_tag_is_created_from_verified_image(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.assert_failed_with('injected container failure', self.invoke('--offline'))
        self.assertEqual((self.root / 'runtime-tag').read_text(), self.runtime_id)

    def test_matching_runtime_tag_is_reused(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.env['FAKE_TAG_ID'] = self.runtime_id
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

    def test_adjacent_images_directory_is_used_by_default_install(self):
        images = self.bundle.parent / 'images'
        images.mkdir()
        (images / 'runtime.tar.gz').write_bytes(base64.b64decode(self.env['FAKE_ARCHIVE']))
        self.assert_failed_with('injected container failure', self.invoke())
        self.assertTrue((self.root / 'image-loaded').is_file())

    def test_configured_download_is_verified_cached_and_imported(self):
        self.write_bundle(url='https://example.invalid/runtime.tar.gz')
        self.fake_curl()
        cache = self.root / 'download-cache'
        self.assert_failed_with('injected container failure', self.invoke('--cache-dir', str(cache)))
        self.assertTrue((self.root / 'image-loaded').is_file())
        self.assertTrue((cache / 'runtime.tar.gz').is_file())
        self.assertFalse((cache / 'runtime.tar.gz.partial').exists())

    def test_recompressed_archive_is_accepted_without_product_checksum_binding(self):
        original = self.archive_bytes()
        repacked = gzip.compress(gzip.decompress(original), compresslevel=1, mtime=1)
        self.assertNotEqual(hashlib.sha256(original).hexdigest(), hashlib.sha256(repacked).hexdigest())
        self.assertNotIn('RUNTIME_ARCHIVE_SHA256', self.installer.read_text())
        archive = self.local_archive(repacked)
        self.assert_failed_with('injected container failure', self.invoke('--runtime-image', str(archive)))
        self.assertTrue((self.root / 'image-loaded').exists())

    def test_standalone_checksum_is_checked_before_loading(self):
        archive = self.local_archive()
        archive.with_name(archive.name + '.sha256').write_text('0' * 64 + '  ' + archive.name + '\n')
        self.assert_failed_with('镜像归档校验失败', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_matching_sha256sums_entry_is_used(self):
        archive = self.local_archive()
        (self.root / 'SHA256SUMS').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  runtime.tar.gz\n')
        result = self.invoke('--runtime-image', str(archive))
        self.assert_failed_with('injected container failure', result)
        self.assertIn('独立文件校验通过', result.stdout)

    def test_invalid_matching_sha256sums_record_is_not_ignored(self):
        archive = self.local_archive()
        (self.root / 'SHA256SUMS').write_text('invalid  runtime.tar.gz\n')
        self.assert_failed_with('无效的镜像校验清单', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_invalid_or_duplicate_checksum_records_are_rejected(self):
        archive = self.local_archive()
        sidecar = archive.with_name(archive.name + '.sha256')
        for value in ('not a checksum\n', 'a' * 64 + '\n' + 'a' * 64 + '\n', 'a' * 64 + '  wrong.tar.gz\n'):
            with self.subTest(value=value):
                sidecar.write_text(value)
                self.assertNotEqual(self.invoke('--runtime-image', str(archive)).returncode, 0)
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_existing_expected_image_cannot_mask_wrong_explicit_archive(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        archive = self.local_archive(self.archive_bytes(config=b'{"wrong":"image"}'))
        self.assert_failed_with('runtime 归档身份或内容校验失败', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_fresh_install_checks_actual_load_result(self):
        archive = self.local_archive(self.archive_bytes(config=b'{"wrong":"image"}'))
        self.assert_failed_with('本次归档导入了不匹配的镜像', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['run'", (self.root / 'docker.log').read_text())

    def test_archive_with_extra_image_is_rejected(self):
        archive = self.local_archive(self.archive_bytes(extra_image=True))
        self.assert_failed_with('本次归档导入了不匹配的镜像', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['run'", (self.root / 'docker.log').read_text())

    def test_conflicting_tag_is_rejected_before_loading_an_explicit_archive(self):
        self.env['FAKE_TAG_ID'] = 'sha256:' + '2' * 64
        archive = self.local_archive(self.archive_bytes(tags=[self.env['FAKE_RUNTIME_TAG']]))
        self.assert_failed_with('runtime 标签指向其他镜像', self.invoke('--runtime-image', str(archive)))
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_load_without_identity_report_is_not_accepted(self):
        self.env['FAKE_IMAGE_PRESENT'] = '1'
        self.env['FAKE_LOAD_REPORT'] = 'none'
        archive = self.local_archive()
        self.assert_failed_with('docker load 未返回', self.invoke('--runtime-image', str(archive)))

    def test_tagged_archive_is_resolved_to_exact_image_id(self):
        archive = self.local_archive(self.archive_bytes(tags=['runtime-fixture:test']))
        self.assert_failed_with('injected container failure', self.invoke('--runtime-image', str(archive)))

    def test_downloaded_checksum_is_optional_but_enforced_when_available(self):
        self.write_bundle(url='https://example.invalid/runtime.tar.gz')
        self.fake_curl()
        self.env['FAKE_CHECKSUM_STATUS'] = '200'
        cache = self.root / 'download-cache'
        self.assert_failed_with('injected container failure', self.invoke('--cache-dir', str(cache)))
        self.assertTrue((cache / 'runtime.tar.gz.sha256').is_file())
        self.assertFalse(list(cache.glob('.download.*')))

    def test_wrong_downloaded_checksum_or_image_is_not_cached(self):
        self.write_bundle(url='https://example.invalid/runtime.tar.gz')
        self.fake_curl()
        cache = self.root / 'download-cache'
        self.env['FAKE_CHECKSUM_STATUS'] = '200'
        self.env['FAKE_CHECKSUM'] = '0' * 64
        self.assert_failed_with('镜像归档校验失败', self.invoke('--cache-dir', str(cache)))
        self.assertFalse((cache / 'runtime.tar.gz').exists())
        self.assertFalse(list(cache.glob('.download.*')))
        self.env['FAKE_CHECKSUM_STATUS'] = '404'
        self.env['FAKE_ARCHIVE'] = base64.b64encode(self.archive_bytes(config=b'{"wrong":"image"}')).decode()
        self.assert_failed_with('本次归档导入了不匹配的镜像', self.invoke('--cache-dir', str(cache)))
        self.assertFalse((cache / 'runtime.tar.gz').exists())
        self.assertFalse(list(cache.glob('.download.*')))

    def test_checksum_server_error_is_not_treated_as_missing(self):
        self.write_bundle(url='https://example.invalid/runtime.tar.gz')
        self.fake_curl()
        self.env['FAKE_CHECKSUM_STATUS'] = '500'
        cache = self.root / 'download-cache'
        self.assert_failed_with('HTTP 500', self.invoke('--cache-dir', str(cache)))
        self.assertFalse((cache / 'runtime.tar.gz').exists())
        self.assertNotIn("['load'", (self.root / 'docker.log').read_text())

    def test_non_https_download_rejected(self):
        self.write_bundle(url='http://example.invalid/runtime.tar.gz')
        self.assert_failed_with('必须使用 HTTPS', self.invoke())


if __name__ == '__main__':
    unittest.main()
