"""Bootstrap safety contracts: no network or large computation in these tests."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lpsd_benchmark_bootstrap", ROOT / "scripts/benchmark_bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def budget(root, size=16 * 1024 ** 2):
    return bootstrap.Budget(root, 0.1, size)


def test_powershell_watchdog_keeps_budgets_when_installer_children_disappear():
    engine = shutil.which("pwsh") or shutil.which("powershell")
    if not engine:
        pytest.skip("PowerShell runtime is exercised by the Windows bootstrap job")
    subprocess.run([
        engine, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(ROOT / "test_fast/check_bootstrap_budget.ps1"),
    ], check=True, timeout=30, capture_output=True, text=True)


@pytest.mark.parametrize("value", ["main", "v1", "a" * 39, "g" * 40, "a" * 40 + ";whoami"])
def test_remote_source_requires_full_commit(value):
    with pytest.raises(bootstrap.argparse.ArgumentTypeError):
        bootstrap.revision_value(value)


@pytest.mark.parametrize("value", [0, -1, 24.00001, float("nan"), float("inf"), "not-a-time"])
def test_rejects_unbounded_time(value):
    with pytest.raises(bootstrap.argparse.ArgumentTypeError):
        bootstrap.positive_hours(value)


def test_inherited_setup_deadline_cannot_be_extended(tmp_path):
    inherited = time.time() + 0.1
    instance = bootstrap.Budget(tmp_path, 24, 1024, inherited_deadline=inherited)
    assert instance.deadline_epoch == inherited
    assert 0 < instance.remaining() <= 0.1


def test_setup_disk_cap_is_checked(tmp_path):
    (tmp_path / "already-written").write_bytes(b"abcdef")
    with pytest.raises(bootstrap.BootstrapError, match="disk budget"):
        budget(tmp_path, 5).check()


def test_setup_memory_guard_stops_work_and_keeps_high_water_mark(tmp_path, monkeypatch):
    instance = budget(tmp_path)
    instance.setup_ram_bytes = 100
    monkeypatch.setattr(bootstrap, "setup_process_memory", lambda: 101)
    with pytest.raises(bootstrap.BootstrapError, match="RAM budget"):
        instance.check()
    assert instance.peak_setup_rss_bytes == 101


def test_setup_memory_monitor_failure_is_not_unbounded_permission(tmp_path, monkeypatch):
    instance = budget(tmp_path)
    instance.setup_ram_bytes = 1024
    monkeypatch.setattr(bootstrap, "setup_process_memory", lambda: None)
    with pytest.raises(bootstrap.BootstrapError, match="Cannot monitor"):
        instance.check()


def test_real_setup_process_memory_includes_current_process():
    measured = bootstrap.setup_process_memory()
    assert isinstance(measured, int) and measured > 0


def test_setup_memory_limit_terminates_oversized_child(tmp_path):
    instance = budget(tmp_path)
    instance.setup_ram_bytes = bootstrap.setup_process_memory() + 32 * 1024 ** 2
    started = time.monotonic()
    code = "import time; payload = bytearray(80 * 1024 ** 2); time.sleep(30)"
    with pytest.raises(bootstrap.BootstrapError, match="RAM budget"):
        bootstrap.run_checked([sys.executable, "-c", code], instance,
                              cwd=tmp_path, environment=os.environ.copy())
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("url", [
    "http://github.com/test", "https://example.org/file", "https://github.com.attacker.invalid/file",
    "https://user:password@github.com/file", "https://github.com:8443/file", "file:///etc/passwd",
])
def test_download_url_rejects_off_host_or_insecure_sources(url):
    with pytest.raises(bootstrap.BootstrapError, match="allowlist"):
        bootstrap.checked_url(url)


class FakeResponse(io.BytesIO):
    headers = {}


class FakeOpener:
    def __init__(self, content):
        self.content = content

    def open(self, *args, **kwargs):
        return FakeResponse(self.content)


def test_bad_archive_digest_never_survives_download(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: FakeOpener(b"tampered"))
    destination = tmp_path / "download"
    with pytest.raises(bootstrap.BootstrapError, match="SHA256"):
        bootstrap.download("https://ziglang.org/file", destination, budget(tmp_path),
                           expected_sha256=hashlib.sha256(b"original").hexdigest())
    assert not destination.exists()


def test_download_size_limit_is_enforced_without_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: FakeOpener(b"a" * 65))
    destination = tmp_path / "download"
    with pytest.raises(bootstrap.BootstrapError, match="file size"):
        bootstrap.download("https://ziglang.org/file", destination, budget(tmp_path), max_bytes=64)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "root/../../escape", "C:/drive", "root\\escape", "root/file\x00hidden"])
def test_zip_traversal_never_writes_outside_destination(tmp_path, name):
    archive = tmp_path / "source.zip"
    # Preserve the deliberately invalid bytes when constructing the fixture:
    # ZipInfo(name) itself normalizes backslashes on Windows and truncates NUL.
    entry = zipfile.ZipInfo("placeholder")
    entry.filename = name
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(entry, "should-not-extract")
    with zipfile.ZipFile(archive) as source:
        assert source.infolist()[0].orig_filename == name
    with pytest.raises(bootstrap.BootstrapError, match="unsafe path"):
        bootstrap.extract_archive(archive, tmp_path / "extracted", budget(tmp_path), max_bytes=1024)
    assert not (tmp_path.parent / "escape").exists()


def test_zip_symlink_is_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    entry = zipfile.ZipInfo("root/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(entry, "../../outside")
    with pytest.raises(bootstrap.BootstrapError, match="links"):
        bootstrap.extract_archive(archive, tmp_path / "extracted", budget(tmp_path), max_bytes=1024)


def test_archive_expansion_respects_declared_budget(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("root/large", b"a" * 2048)
    with pytest.raises(bootstrap.BootstrapError, match="Expanded archive"):
        bootstrap.extract_archive(archive, tmp_path / "extracted", budget(tmp_path), max_bytes=1024)


def test_source_revision_prefix_must_match(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("lpsd-other/file.py", "pass")
    with pytest.raises(bootstrap.BootstrapError, match="pinned revision"):
        bootstrap.extract_archive(archive, tmp_path / "extracted", budget(tmp_path), max_bytes=1024,
                                  expected_root="lpsd-" + "a" * 40)


def test_regular_tar_extracts_and_retains_executable_mode(tmp_path):
    archive = tmp_path / "tool.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        entry = tarfile.TarInfo("tool/program")
        entry.mode = 0o755
        entry.size = 4
        output.addfile(entry, io.BytesIO(b"test"))
    root = bootstrap.extract_archive(archive, tmp_path / "extracted", budget(tmp_path), max_bytes=1024)
    assert (root / "program").read_bytes() == b"test"
    if os.name != "nt":
        assert (root / "program").stat().st_mode & stat.S_IXUSR


def test_isolated_environment_drops_foreign_indexes_and_python_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_INDEX", "https://credentials@example.invalid/simple")
    monkeypatch.setenv("UV_INSECURE_HOST", "github.com")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://example.invalid/simple")
    monkeypatch.setenv("PYTHONPATH", "/some/other/packages")
    monkeypatch.setenv("GH_TELEMETRY", "log")
    monkeypatch.setenv("GH_DEBUG", "api")
    result = bootstrap.child_environment(tmp_path, 12345)
    assert "UV_INDEX" not in result
    assert "UV_INSECURE_HOST" not in result
    assert "PIP_EXTRA_INDEX_URL" not in result
    assert "PYTHONPATH" not in result
    assert "GH_DEBUG" not in result
    assert result["GH_TELEMETRY"] == "0"
    assert result["DO_NOT_TRACK"] == "1"
    assert result["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "python")
    assert result["LPSD_BENCHMARK_DEADLINE_EPOCH"] == "12345"


def test_numerical_temp_uses_original_os_parent_without_changing_private_setup(tmp_path):
    setup = tmp_path / "synchronized-checkout"
    setup.mkdir()
    original_temp = tmp_path / "local-system-temp"
    original_temp.mkdir()
    environment = bootstrap.child_environment(setup, 12345)
    numerical = bootstrap.benchmark_environment(environment, original_temp)
    for key in ("TMPDIR", "TMP", "TEMP"):
        assert numerical[key] == str(original_temp)
        assert environment[key] == str(setup / "temporary")
    assert numerical["UV_CACHE_DIR"] == environment["UV_CACHE_DIR"] == str(setup / "cache")
    assert numerical["LPSD_BENCHMARK_DEADLINE_EPOCH"] == "12345"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process group contract; Windows uses taskkill tree termination")
def test_deadline_stops_subprocess_group(tmp_path, monkeypatch):
    inherited = time.time() + 10
    instance = bootstrap.Budget(tmp_path, 0.1, 1024 ** 2, inherited_deadline=inherited)
    pid_file = tmp_path / "pid"
    code = "import os,time,pathlib; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(pid_file)
    original = bootstrap.subprocess.Popen
    observed = []

    def started_child(*args, **kwargs):
        child = original(*args, **kwargs)
        observed.append(child)
        try:
            # Slow ARM startup must not be mistaken for failure to terminate.
            # Once the real child is ready, shorten the existing deadline.
            while (not pid_file.exists() or
                   pid_file.read_text(encoding="ascii") != str(child.pid)):
                instance.check()
                assert child.poll() is None
                time.sleep(.01)
            instance.deadline_monotonic = min(instance.deadline_monotonic, time.monotonic() + .15)
            return child
        except BaseException:
            bootstrap.stop_process(child)
            raise

    monkeypatch.setattr(bootstrap.subprocess, "Popen", started_child)
    with pytest.raises(bootstrap.BootstrapError, match="deadline"):
        bootstrap.run_checked([sys.executable, "-c", code], instance, cwd=tmp_path,
                              environment=os.environ.copy())
    assert time.monotonic() - instance.deadline_monotonic < 3
    assert observed
    observed[0].wait(timeout=3)
    pid = int(pid_file.read_text())
    assert pid == observed[0].pid
    # waitpid reaps the child if the termination raced with Popen's poll.
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_wrapper_pins_match_machine_readable_manifest():
    pins = json.loads((ROOT / "scripts/benchmark-bootstrap-pins.json").read_text())
    shell = (ROOT / "scripts/run_benchmark.sh").read_text()
    powershell = (ROOT / "scripts/run_benchmark.ps1").read_text()
    for target, values in pins["uv"].items():
        if target == "aarch64-pc-windows-msvc":
            continue  # Windows ARM64 deliberately uses x64 emulation.
        script = powershell if "windows" in target else shell
        assert values["sha256"] in script
        assert target in script
    assert "--require-hashes" in (ROOT / "scripts/benchmark_bootstrap.py").read_text()
    # A private Windows Python must neither add a bin link nor register the
    # interpreter for unrelated launchers. These are independent uv options.
    windows_install = next(line for line in powershell.splitlines()
                           if line.startswith("Start-Bounded $uv ") and "'install'" in line)
    assert "'--no-bin'" in windows_install
    assert "'--no-registry'" in windows_install
    # On PowerShell 7.5+, the .NET setter turns $null into an empty variable.
    # Unset uv preferences and restore absent session variables through Env:.
    assert "Remove-Item -LiteralPath Env:UV_PYTHON_PREFERENCE -ErrorAction SilentlyContinue" in powershell
    assert "SetEnvironmentVariable('UV_PYTHON_PREFERENCE', $null" not in powershell
    restore = powershell.split("} finally {")[-1]
    assert "if ($null -eq $savedEnvironment[$key])" in restore
    assert 'Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue' in restore
    assert "SetEnvironmentVariable($key, $savedEnvironment[$key], 'Process')" in restore


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell entry point")
def test_shell_help_and_invalid_revision_are_offline():
    result = subprocess.run(["sh", str(ROOT / "scripts/run_benchmark.sh"), "--help"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "Results stay local" in result.stdout
    bad = subprocess.run(["sh", str(ROOT / "scripts/run_benchmark.sh"), "--revision", "main"],
                         capture_output=True, text=True, timeout=5)
    assert bad.returncode != 0
    assert "full Git commit" in bad.stderr


def test_optional_github_does_nothing_without_public_opt_in(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No GitHub executable lookup/download/auth is authorized")
    monkeypatch.setattr(bootstrap.shutil, "which", forbidden)
    assert bootstrap.prepare_github({}, tmp_path, budget(tmp_path), {},
                                    upload=True, accepted=False, login=False) == "not-requested"
    with pytest.raises(bootstrap.BootstrapError, match="requires"):
        bootstrap.prepare_github({}, tmp_path, budget(tmp_path), {},
                                 upload=True, accepted=False, login=True)


def test_existing_gh_reuses_login_without_authentication_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda *args, **kwargs: "/existing/gh")
    def forbidden(*args, **kwargs):
        pytest.fail("Existing CLI does not need a download or implicit login")
    monkeypatch.setattr(bootstrap, "download", forbidden)
    monkeypatch.setattr(bootstrap, "run_checked", forbidden)
    environment = {"PATH": "/existing", "GH_CONFIG_DIR": "/existing-config"}
    assert bootstrap.prepare_github({}, tmp_path, budget(tmp_path), environment,
                                    upload=True, accepted=True, login=False) == "existing"
    assert environment["GH_CONFIG_DIR"] == "/existing-config"


def test_private_gh_archive_is_hash_checked_and_added_only_to_child_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Windows")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "AMD64")
    name = "gh.exe" if os.name == "nt" else "gh"
    expected = "c" * 64
    def fake_download(url, destination, *args, **kwargs):
        assert kwargs["expected_sha256"] == expected
        assert kwargs["max_bytes"] == 4096
        with zipfile.ZipFile(destination, "w") as output:
            output.writestr("bin/" + name, b"test-cli")
            output.writestr("LICENSE", "fixture")
    monkeypatch.setattr(bootstrap, "download", fake_download)
    pins = {"gh_version": "test", "gh": {"x86_64-windows": {
        "url": "https://github.com/cli/cli/example.zip", "sha256": expected, "size": 4096}}}
    environment = {"PATH": "/existing"}
    original_path = os.environ.get("PATH")
    assert bootstrap.prepare_github(pins, tmp_path, budget(tmp_path), environment,
                                    upload=True, accepted=True, login=False) == "private-test"
    assert Path(environment["PATH"].split(os.pathsep)[0], name).read_bytes() == b"test-cli"
    assert environment["GH_CONFIG_DIR"] == str(tmp_path / "github-config")
    assert os.environ.get("PATH") == original_path
    assert not (tmp_path / "gh.zip").exists()


def test_explicit_gh_login_uses_private_git_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda *args, **kwargs: "/existing/gh")
    observed = []
    def fake_run(command, instance, *, cwd, environment):
        observed.append((command, environment))
    monkeypatch.setattr(bootstrap, "run_checked", fake_run)
    environment = {"PATH": "/existing", "GH_PROMPT_DISABLED": "1"}
    bootstrap.prepare_github({}, tmp_path, budget(tmp_path), environment,
                             upload=True, accepted=True, login=True)
    command, login_environment = observed[0]
    assert command == ["/existing/gh", "auth", "login", "--hostname", "github.com",
                       "--git-protocol", "https", "--skip-ssh-key", "--web"]
    assert "GH_PROMPT_DISABLED" not in login_environment
    assert login_environment["GIT_CONFIG_GLOBAL"] == str(tmp_path / "github-git-config")
    assert login_environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert "GIT_CONFIG_GLOBAL" not in environment
    assert environment["GH_CONFIG_DIR"] == str(tmp_path / "github-config")


def test_optional_cli_download_failure_keeps_local_benchmark_usable(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "aarch64")
    def failed(*args, **kwargs):
        raise bootstrap.BootstrapError("Network unavailable")
    monkeypatch.setattr(bootstrap, "download", failed)
    pins = {"gh_version": "test", "gh": {"aarch64-linux": {
        "url": "https://github.com/cli/cli/test.tar.gz", "sha256": "c" * 64, "size": 4096}}}
    assert bootstrap.prepare_github(pins, tmp_path, budget(tmp_path), {"PATH": ""},
                                    upload=True, accepted=True, login=False) == "unavailable"
