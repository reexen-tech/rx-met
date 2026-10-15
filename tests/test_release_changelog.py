"""Validate release-note selection before creating user-facing artifacts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_artifacts', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)

CURRENT = '''## [1.1.0] - 2026-10-12

### Added

- 本版新增功能。

### Known limitations

- 本版已知限制。

### Upgrade notes

- 使用 deps-v2。
'''
DOCUMENT = '''# Changelog

完整历史。

## [Unreleased]

- 尚未发布的功能。

## [1.1.0-rc.1] - 2026-10-11

- 候选版本。

''' + CURRENT + '''
## [1.0.0] - 2026-09-02

- 历史版本。
'''


class ReleaseChangelogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-changelog-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(artifacts, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        (self.root / 'VERSION').write_text('1.1.0\n')
        self.changelog = self.root / 'CHANGELOG.md'
        self.changelog.write_text(DOCUMENT, encoding='utf-8')

    def test_selects_exact_version_and_preserves_all_release_sections(self):
        self.assertEqual(artifacts.release_changelog('1.1.0'), '# Changelog\n\n' + CURRENT)
        self.assertEqual(artifacts.release_changelog('1.0.0'),
                         '# Changelog\n\n## [1.0.0] - 2026-09-02\n\n- 历史版本。\n')

    def test_rejects_missing_duplicate_and_empty_release(self):
        for document in [
            DOCUMENT.replace('## [1.1.0]', '## [1.1.1]'),
            DOCUMENT + '\n' + CURRENT,
            '# Changelog\n\n## [1.1.0] - 2026-10-12\n\n',
            '## [1.1.0]\n\n### Added\n\n### Fixed\n\n## [1.0.0]\n\n- Old.\n',
        ]:
            with self.subTest(document=document):
                self.changelog.write_text(document, encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, r'CHANGELOG\.md.*\[1\.1\.0\]'):
                    artifacts.release_changelog('1.1.0')
        self.changelog.write_text(DOCUMENT, encoding='utf-8')
        with self.assertRaises(RuntimeError):
            artifacts.release_changelog('Unreleased')

    def test_invalid_notes_stop_before_docker_or_output_creation(self):
        self.changelog.write_text('# Changelog\n')
        matrix = {'environment_version': 'deps-v2', 'variants': {'cu118': {}}}
        with patch.object(artifacts.dependencies, 'load_matrix', return_value=matrix), \
                patch.object(artifacts.dependencies, 'check_generated_artifacts'), \
                patch.object(artifacts, 'image_info') as inspect_image, \
                patch('sys.argv', ['artifacts.py', 'build', 'cu118']):
            with self.assertRaisesRegex(RuntimeError, 'CHANGELOG'):
                artifacts.main()
        inspect_image.assert_not_called()
        self.assertFalse((self.root / '.release').exists())

    def test_delivery_identity_tracks_current_notes_only(self):
        for name in ['packaging/release-bundle/install.sh.in', 'packaging/release-bundle/README.md.in',
                     'scripts/release/verify_install.py', 'scripts/release/verify_runtime_archive.py', 'examples/README.md']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        before = artifacts.delivery_digest(artifacts.release_changelog('1.1.0'))
        self.changelog.write_text(DOCUMENT.replace('尚未发布的功能', '未来功能更新').replace('历史版本。', '历史说明修订。'), encoding='utf-8')
        self.assertEqual(before, artifacts.delivery_digest(artifacts.release_changelog('1.1.0')))
        self.changelog.write_text(DOCUMENT.replace('本版新增功能。', '本版新增功能与修复。'), encoding='utf-8')
        self.assertNotEqual(before, artifacts.delivery_digest(artifacts.release_changelog('1.1.0')))


if __name__ == '__main__':
    unittest.main()
