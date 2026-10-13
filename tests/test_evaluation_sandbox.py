"""The code sandbox: limits, cleanup, environment stripping, and recorded isolation."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from epor.evaluation.sandbox import (
    CodeExecutionRefusedError,
    CodeSandbox,
    SandboxIsolation,
    SandboxLimits,
    require_isolation,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the sandbox is Linux-only")


@pytest.fixture(scope="module")
def isolation() -> SandboxIsolation:
    return CodeSandbox().probe()


def _sandbox(tmp_path: Path, **limits: Any) -> CodeSandbox:
    return CodeSandbox(
        limits=SandboxLimits.model_validate(limits), scratch_root=tmp_path / "scratch"
    )


def test_passing_and_failing_candidates_and_cleanup(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    passed = sandbox.run("print('hello')\n")
    assert passed.status == "passed" and passed.stdout == "hello\n"
    failed = sandbox.run("raise SystemExit(3)\n")
    assert failed.status == "failed" and failed.returncode == 3
    error = sandbox.run("assert 1 == 2\n")
    assert error.status == "failed" and "AssertionError" in error.stderr
    assert "<workdir>" in error.stderr  # the random private path is hidden
    # Every per-run directory was deleted.
    assert list((tmp_path / "scratch").iterdir()) == []


def test_wall_clock_timeout_kills_the_process_group(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path, wall_seconds=1.0, cpu_seconds=30)
    started = time.monotonic()
    result = sandbox.run(
        "import os, time\nif os.fork() == 0:\n    time.sleep(60)\ntime.sleep(60)\n"
    )
    assert result.status == "timeout"
    assert time.monotonic() - started < 10


def test_cpu_limit_stops_busy_loops(tmp_path: Path) -> None:
    result = _sandbox(tmp_path, cpu_seconds=1, wall_seconds=20.0).run("while True:\n    pass\n")
    assert result.status == "resource-limit"
    assert result.signal_name in {"SIGXCPU", "SIGKILL"}


def test_memory_limit(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path, address_space_bytes=256 * 1024 * 1024)
    result = sandbox.run("blob = bytearray(1024 * 1024 * 1024)\n")
    assert result.status == "resource-limit"
    assert "MemoryError" in result.stderr


def test_fork_bomb_is_bounded(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path, processes=8, wall_seconds=10.0)
    started = time.monotonic()
    result = sandbox.run(
        "import os, time\n"
        "made = 0\n"
        "try:\n"
        "    while made < 10000:\n"
        "        if os.fork() == 0:\n"
        "            time.sleep(30)\n"
        "            os._exit(0)\n"
        "        made += 1\n"
        "except OSError:\n"
        "    pass\n"
        "print(made)\n"
    )
    assert result.status == "passed"
    made = int(result.stdout)
    if result.isolation.process_limit_scope == "user-namespace":
        assert made < 8
    else:
        assert made < 10000
    assert time.monotonic() - started < 10


def test_output_is_bounded(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path, max_output_bytes=1024, file_size_bytes=64 * 1024)
    result = sandbox.run("import sys\nsys.stdout.write('x' * 50000)\n")
    assert len(result.stdout) == 1024 and result.stdout_truncated


def test_environment_secrets_are_not_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EPOR_TEST_SECRET", "do-not-leak")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = _sandbox(tmp_path).run("import os\nprint(sorted(os.environ))\n")
    assert result.stdout.strip() == "['HOME', 'LANG', 'PATH', 'TMPDIR']"
    assert "do-not-leak" not in result.stdout


def test_stdin_is_empty(tmp_path: Path) -> None:
    result = _sandbox(tmp_path).run("import sys\nprint(repr(sys.stdin.read()))\n")
    assert result.stdout.strip() == "''"


def test_writes_stay_in_the_working_directory(tmp_path: Path, isolation: SandboxIsolation) -> None:
    if not isolation.filesystem_writes_confined:
        pytest.skip(f"Landlock is unavailable on this kernel: {isolation.notes}")
    outside = tmp_path / "outside.txt"
    result = _sandbox(tmp_path).run(
        "open('inside.txt', 'w').write('ok')\n"
        "try:\n"
        f"    open({str(outside)!r}, 'w').write('escaped')\n"
        "    print('escaped')\n"
        "except OSError:\n"
        "    print('blocked')\n"
    )
    assert result.stdout.strip() == "blocked"
    assert not outside.exists()


def test_network_isolation_is_recorded_and_effective(isolation: SandboxIsolation) -> None:
    assert isolation.sandbox_version == "epor-sandbox-v1"
    if not isolation.network_isolated:
        pytest.skip(f"unprivileged network namespaces are unavailable: {isolation.notes}")
    assert isolation.network_method == "user-network-namespace"
    assert isolation.process_limit_scope == "user-namespace"
    result = CodeSandbox().run(
        "import socket\n"
        "print(socket.if_nameindex())\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
        "except OSError as exc:\n"
        "    print('refused', type(exc).__name__)\n"
    )
    assert result.stdout.splitlines()[0] == "[(1, 'lo')]"
    assert result.stdout.splitlines()[1].startswith("refused")


def test_degraded_sandbox_reports_missing_isolation_and_requires_opt_in() -> None:
    degraded = CodeSandbox(use_namespaces=False, use_landlock=False).probe()
    assert degraded.network_isolated is False
    assert degraded.network_method == "none"
    assert degraded.filesystem_writes_confined is False
    assert degraded.process_limit_scope == "global-user"
    with pytest.raises(CodeExecutionRefusedError, match="allow_unisolated_network"):
        require_isolation(degraded, allow_unisolated_network=False)
    require_isolation(degraded, allow_unisolated_network=True)


def test_a_failed_namespace_setup_falls_back_and_still_needs_the_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a kernel where unshare() succeeds but the ID mapping is denied
    # (Ubuntu's AppArmor restriction): the launcher exits before exec.
    import epor.evaluation.sandbox as sandbox_module

    failing = (
        "import json, os, sys\nif json.loads(sys.argv[1])['try_namespaces']:\n    os._exit(125)\n"
    )
    monkeypatch.setattr(sandbox_module, "_LAUNCHER", failing + sandbox_module._LAUNCHER)
    result = _sandbox(tmp_path).run("print('ran')\n")
    # The candidate ran only in the degraded sandbox; the failed attempt never exec'ed it.
    assert result.status == "passed" and result.stdout == "ran\n"
    assert result.isolation.network_isolated is False
    assert result.isolation.process_limit_scope == "global-user"
    assert result.isolation.notes[0].startswith("user namespace setup failed")
    with pytest.raises(CodeExecutionRefusedError, match="allow_unisolated_network"):
        require_isolation(result.isolation, allow_unisolated_network=False)


def test_degraded_process_limit_counts_the_callers_threads(tmp_path: Path) -> None:
    # Without a user namespace RLIMIT_NPROC counts every task (thread) of the
    # user, so a caller with many threads (PyTorch pools, pytest) must not
    # leave the candidate unable to start even one child process.
    import threading

    stop = threading.Event()
    threads = [threading.Thread(target=stop.wait, daemon=True) for _ in range(48)]
    for thread in threads:
        thread.start()
    try:
        sandbox = CodeSandbox(use_namespaces=False, scratch_root=tmp_path / "scratch")
        result = sandbox.run(
            "import os\npid = os.fork()\nif pid == 0:\n    os._exit(0)\nos.waitpid(pid, 0)\n"
            "print('forked')\n"
        )
    finally:
        stop.set()
    assert result.isolation.process_limit_scope == "global-user"
    assert result.stdout.strip() == "forked", result.stderr
