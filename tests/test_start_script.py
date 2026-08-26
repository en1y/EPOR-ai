from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = PROJECT_ROOT / "start-epor.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_launcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    fake_root = tmp_path / "project"
    fake_bin = tmp_path / "bin"
    fake_root.mkdir()
    fake_bin.mkdir()

    launcher = fake_root / START_SCRIPT.name
    shutil.copy2(START_SCRIPT, launcher)
    launcher.chmod(0o755)

    vite = fake_root / "ui" / "node_modules" / ".bin" / "vite"
    vite.parent.mkdir(parents=True)
    _write_executable(vite, "#!/usr/bin/env bash\nexit 0\n")

    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -eu

service=""
for argument in "$@"; do
  case "${argument}" in
    version | auth | api | worker | ui)
      service="${argument}"
      break
      ;;
  esac
done

printf '%s\n' "${service}" >> "${EPOR_LAUNCH_TEST_LOG:?}"
if [[ "${service}" == version ]]; then
  exit 0
fi
if [[ "${service}" == auth ]]; then
  # No owner credential yet: the launcher must hint and carry on, because
  # reading the console never requires one.
  exit 1
fi
if [[ "${service}" == ui && -n "${EPOR_TEST_UI_EXIT:-}" ]]; then
  sleep 0.2
  exit "${EPOR_TEST_UI_EXIT}"
fi

trap 'printf "%s:term\\n" "${service}" >> "${EPOR_LAUNCH_TEST_LOG}"; exit 0' TERM INT HUP
while :; do
  sleep 0.05
done
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
printf 'health\n' >> "${EPOR_LAUNCH_TEST_LOG:?}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "npm",
        """#!/usr/bin/env bash
printf 'npm:%s\n' "$*" >> "${EPOR_LAUNCH_TEST_LOG:?}"
exit 99
""",
    )

    log_path = tmp_path / "launcher.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["EPOR_LAUNCH_TEST_LOG"] = str(log_path)
    return launcher, environment, log_path


def _wait_for_log_line(log_path: Path, expected: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log_path.is_file() and expected in log_path.read_text(encoding="utf-8").splitlines():
            return
        time.sleep(0.05)
    raise AssertionError(f"launcher never logged {expected!r}")


def test_start_script_is_executable_and_valid_bash() -> None:
    assert os.access(START_SCRIPT, os.X_OK)
    completed = subprocess.run(
        ["bash", "-n", str(START_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_start_script_help_works_outside_the_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(START_SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Start the EPOR API, worker, and browser console together." in completed.stdout
    assert "Press Ctrl+C to stop every service." in completed.stdout


def test_start_script_rejects_unknown_options() -> None:
    completed = subprocess.run(
        [str(START_SCRIPT), "--unknown"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "unknown option: --unknown" in completed.stderr


def test_child_failure_stops_the_other_services_without_installing(tmp_path: Path) -> None:
    launcher, environment, log_path = _fake_launcher(tmp_path)
    environment["EPOR_TEST_UI_EXIT"] = "7"

    completed = subprocess.run(
        [str(launcher)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 7
    lines = log_path.read_text(encoding="utf-8").splitlines()
    # The owner-credential hint runs between the readiness check and the
    # services, and never blocks them.
    assert lines[:6] == ["version", "auth", "api", "health", "worker", "ui"]
    assert "No owner credential yet." in completed.stdout
    assert {"api:term", "worker:term"} <= set(lines)
    assert not any(line.startswith("npm:") for line in lines)
    assert "A service exited unexpectedly (status 7)." in completed.stderr


def test_interrupt_stops_every_service_group(tmp_path: Path) -> None:
    launcher, environment, log_path = _fake_launcher(tmp_path)
    process = subprocess.Popen(
        [str(launcher)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_log_line(log_path, "ui")
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)

    assert process.returncode == 130, (stdout, stderr)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert {"api:term", "worker:term", "ui:term"} <= set(lines)
    assert "EPOR stopped." in stdout
