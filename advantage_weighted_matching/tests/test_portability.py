"""Portable config paths, dependency declarations and non-destructive setup helpers."""
import os
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from config import dgx_awm

ROOT = Path(__file__).resolve().parents[2]


class PortabilityTests(unittest.TestCase):
    def test_paths_independent_of_cwd_and_overridable(self):
        factories = [getattr(dgx_awm, n) for n in vars(dgx_awm)
                     if n.startswith(("geneval_", "ocr_", "pickscore_"))]
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("AWM_DATA_DIR", None)
            os.environ.pop("AWM_OUTPUT_DIR", None)
            try:
                os.chdir(directory)
                for factory in factories:
                    config = factory()
                    task = factory.__name__.split("_")[0]
                    nodes = factory.__name__.split("_")[-1]
                    self.assertEqual(Path(config.dataset), ROOT / "advantage_weighted_matching/dataset" / task)
                    self.assertEqual(Path(config.save_dir).parent, ROOT / "outputs" / task / nodes)
                    self.assertEqual(Path(config.logdir), ROOT / "outputs/logs")
                os.environ["AWM_DATA_DIR"] = str(Path(directory) / "data")
                os.environ["AWM_OUTPUT_DIR"] = str(Path(directory) / "runs")
                for factory in factories:
                    config = factory()
                    self.assertTrue(Path(config.save_dir).is_relative_to(Path(directory) / "runs"))
                    self.assertTrue(Path(config.dataset).is_relative_to(Path(directory) / "data"))
                    config.save_dir = "/explicit/override"
                    self.assertEqual(config.save_dir, "/explicit/override")
            finally:
                os.chdir(original)

    def test_setup_reads_single_dependency_source(self):
        received = {}
        setuptools = types.ModuleType("setuptools")
        setuptools.setup = lambda **kw: received.update(kw)
        setuptools.find_packages = lambda: []
        path = ROOT / "advantage_weighted_matching/setup.py"
        with patch.dict(sys.modules, {"setuptools": setuptools}):
            exec(compile(path.read_text(), str(path), "exec"), {"__file__": str(path)})
        required = received["install_requires"]
        for pin in ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                    "accelerate==1.9.0", "transformers==4.54.0", "tokenizers==0.21.4",
                    "huggingface-hub==0.36.2", "wandb==0.29.0", "xformers==0.0.29.post3",
                    "albumentations==1.4.10", "albucore==0.0.13"):
            self.assertIn(pin, required)
        main = (ROOT / "setup/01_setup_env_awm.sh").read_text()
        self.assertIn('export PIP_CONSTRAINT="$PROJECT_ROOT/advantage_weighted_matching/requirements.txt"', main)
        self.assertNotIn("pip install accelerate", main)
        self.assertIn('python -c "import albumentations"', main)
        reward = (ROOT / "reward-server/requirements-geneval.txt").read_text()
        self.assertIn("torch==2.1.2", reward)
        self.assertIn("transformers==4.38.2", reward)
        self.assertIn("ninja==1.13.2", reward)

    def test_cuda_build_layouts_and_fail_fast(self):
        script = (ROOT / "setup/02_setup_env_reward_server.sh").read_text()
        helper = "configure_cuda_build() {" + script.split("configure_cuda_build() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        for layout in ("standard", "conda", "symlink", "auto", "invalid", "missing_crt", "bad_jobs", "bad_header", "no_ninja"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "toolkit with spaces"
                runtime = root / ("include" if layout == "standard" else "targets/x86_64-linux/include")
                crt = root / "include/crt"
                runtime.mkdir(parents=True)
                crt.mkdir(parents=True, exist_ok=True)
                (runtime / "cuda_runtime_api.h").write_text('#include <crt/host_defines.h>\n')
                if layout != "missing_crt":
                    (crt / "host_defines.h").write_text("#error invalid header\n" if layout == "bad_header" else "#pragma once\n")
                library = root / "targets/x86_64-linux/lib"
                library.mkdir(parents=True)
                binaries = root / "bin"
                binaries.mkdir()
                for name, output in (("nvcc", "Cuda compilation tools, release 12.1"), ("ninja", "1.13.2")):
                    executable = binaries / name
                    executable.write_text("#!/bin/sh\n" + ("exit 1\n" if layout == "no_ninja" and name == "ninja" else f"echo '{output}'\n"))
                    executable.chmod(0o755)
                home = root
                if layout == "symlink":
                    home = Path(temp) / "cuda-shim"
                    home.mkdir()
                    (home / "bin").symlink_to(binaries, target_is_directory=True)
                    (home / "include").symlink_to(runtime, target_is_directory=True)
                env = dict(os.environ)
                env.update(CUDA_HOME=str(home), PATH=f"{binaries}:/usr/bin:/bin",
                           CPATH="/existing/includes", LIBRARY_PATH="/existing/libs",
                           LD_LIBRARY_PATH="/unchanged/runtime", CPLUS_INCLUDE_PATH="/unchanged/cxx",
                           CXX="/usr/bin/c++")
                env.pop("MAX_JOBS", None)
                if layout == "auto":
                    env.pop("CUDA_HOME")
                if layout == "invalid":
                    env["CUDA_HOME"] = str(Path(temp) / "missing")
                if layout == "conda":
                    env["MAX_JOBS"] = "24"
                if layout == "bad_jobs":
                    env["MAX_JOBS"] = "0"
                command = ('set -euo pipefail; log() { echo "$*"; }; ' + helper +
                           'configure_cuda_build; printf "RESULT=%s\\n" "$CUDA_HOME" "$CPATH" '
                           '"$LIBRARY_PATH" "$LD_LIBRARY_PATH" "$CPLUS_INCLUDE_PATH" "$MAX_JOBS"')
                result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
                if layout in ("invalid", "missing_crt", "bad_jobs", "bad_header", "no_ninja"):
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertNotIn("RESULT=", result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    values = [line.removeprefix("RESULT=") for line in result.stdout.splitlines() if line.startswith("RESULT=")]
                    self.assertEqual(values[0], str(home))
                    self.assertIn(str(root / "include"), values[1].split(":"))
                    self.assertTrue(values[1].endswith("/existing/includes"))
                    self.assertIn(str(library), values[2].split(":"))
                    self.assertTrue(values[2].endswith("/existing/libs"))
                    self.assertEqual(values[3:5], ["/unchanged/runtime", "/unchanged/cxx"])
                    if layout == "conda":
                        self.assertEqual(values[5], "24")
                    else:
                        self.assertTrue(1 <= int(values[5]) <= 16)

    def test_cuda_version_check(self):
        script = (ROOT / "setup/02_setup_env_reward_server.sh").read_text()
        check = script.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for runtime, compiler, fails in (("12.1", "12.1", False), ("12.1", "12.4", False),
                                         ("12.1", "11.8", True), (None, "12.1", True),
                                         ("12.1", "unknown", True)):
            with self.subTest(runtime=runtime, compiler=compiler), patch.dict(os.environ, CUDA_HOME="/selected/cuda"), \
                 patch.dict(sys.modules, {"torch": types.SimpleNamespace(version=types.SimpleNamespace(cuda=runtime))}), \
                 patch("subprocess.check_output", return_value=f"release {compiler}"), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                if fails:
                    with self.assertRaises(SystemExit):
                        exec(check, {})
                else:
                    exec(check, {})
                    self.assertEqual("Warning:" in output.getvalue(), runtime != compiler)

    def shell(self, code, directory, **overrides):
        env = dict(os.environ)
        for key in ("BASE_DIR", "PROJECT_ROOT", "CONDA_EXE", "CONDA_PATH"):
            env.pop(key, None)
        env.update(TMPDIR=str(Path(directory) / "tmp"),
                   PIP_CACHE_DIR=str(Path(directory) / "pip"),
                   CONDA_PATH=str(Path(directory) / "conda"), **overrides)
        return subprocess.run(["bash", "-c", 'source "$1/setup/common_vars.sh"; ' + code,
                               "test", str(ROOT)], cwd=directory, env=env,
                              text=True, capture_output=True)

    def test_common_paths_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell('test "$PROJECT_ROOT" = "$1" && test "$HUGGINGFACE_TOKEN" = test-sentinel',
                                directory, HUGGINGFACE_TOKEN="test-sentinel")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.shell('test "$PROJECT_ROOT" = /explicit/project', directory,
                                PROJECT_ROOT="/explicit/project")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkout_refuses_dirty_or_wrong_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "dependency"
            subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
            path = repo / "file.txt"
            path.write_text("original")
            subprocess.run(["git", "-C", str(repo), "add", "file.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                            "commit", "-qm", "fixture"], check=True)
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            code = f'ensure_checkout unused-url test-tag {revision} "{repo}"'
            self.assertEqual(self.shell(code, directory).returncode, 0)
            path.write_text("user modification")
            self.assertNotEqual(self.shell(code, directory).returncode, 0)
            self.assertEqual(path.read_text(), "user modification")
            path.write_text("original")
            self.assertNotEqual(self.shell(code.replace(revision, "0" * 40), directory).returncode, 0)
            self.assertEqual(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                    text=True).strip(), revision)


if __name__ == "__main__":
    unittest.main()
