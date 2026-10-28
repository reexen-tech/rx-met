"""Check CPU/GPU bundle verification commands without a Docker daemon."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from tests.support.paths import REPO_ROOT as ROOT


class BundleVerifierTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rx-met-bundle-verifier-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.log = self.root / "docker.jsonl"
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.tools) + os.pathsep + self.env["PATH"]
        self.env["RX_MET_INSTALL_PREFIX"] = str(self.root / "installed")
        self.env["FAKE_DOCKER_LOG"] = str(self.log)
        self.env["FAKE_VERIFY_CODE"] = str(self.root / "verify.py")
        docker = self.tools / "docker"
        docker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["image", "inspect"]:
    print("sha256:runtime")
elif args[0] == "run":
    code = sys.stdin.read()
    if code:
        compile(code, "bundle-verification", "exec")
        Path(os.environ["FAKE_VERIFY_CODE"]).write_text(code)
else:
    raise SystemExit("unexpected docker arguments")
''')
        docker.chmod(0o755)

    def installed(self, variant):
        version = (ROOT / "VERSION").read_text().strip()
        path = Path(self.env["RX_MET_INSTALL_PREFIX"]) / f"rx-met-v{version}-{variant}-linux-amd64"
        path.mkdir(parents=True)
        cpu = variant == "cpu"
        manifest = {
            "variant": variant,
            "runtime": {"image_tag": "runtime:" + variant},
            "compatibility": {
                "cuda_version": None if cpu else "12.6",
                "python_version": "3.10",
                "packages": {
                    "torch": "2.8.0+cpu" if cpu else "2.8.0+cu126",
                    "torchvision": "0.23.0+cpu" if cpu else "0.23.0+cu126",
                    "torchaudio": "2.8.0+cpu" if cpu else "2.8.0+cu126",
                    "onnxruntime" if cpu else "onnxruntime-gpu": "1.23.2",
                },
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest))

    def verify(self, *arguments):
        result = subprocess.run([str(ROOT / "scripts/release/verify_bundle.sh"), *arguments],
                                env=self.env, input="", text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_cpu_manifest_uses_cpu_ort_and_never_requests_gpu(self):
        self.installed("cpu")
        commands = self.verify("--gpu", "cpu")
        runs = [command for command in commands if command[0] == "run"]
        self.assertEqual(len(runs), 2)
        for command in runs:
            self.assertNotIn("--gpus", command)
            self.assertIn("VERIFY_GPU=0", command)
            self.assertIn("EXPECTED_CUDA=none", command)
            self.assertIn("EXPECTED_ONNXRUNTIME=1.23.2", command)
            self.assertIn("EXPECTED_PYTHON=3.10", command)
        self.assertIn("CPU calibration/PTQ OK", Path(self.env["FAKE_VERIFY_CODE"]).read_text())

    def test_cuda_bundle_preserves_gpu_device_selection(self):
        self.installed("cu126")
        commands = self.verify("--gpu", "--gpu-device", "2", "cu126")
        runs = [command for command in commands if command[0] == "run"]
        for command in runs:
            self.assertEqual(command[command.index("--gpus") + 1], "device=2")
            self.assertIn("VERIFY_GPU=1", command)
            self.assertIn("EXPECTED_CUDA=12.6", command)


if __name__ == "__main__":
    unittest.main()
