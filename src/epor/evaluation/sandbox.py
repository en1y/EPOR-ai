"""Defense-in-depth subprocess sandbox for scoring generated code.

**This is not a security boundary against a determined attacker.**  It is a
set of layered, best-effort limits that make an *accidentally* harmful or
runaway candidate (an infinite loop, a fork bomb, a memory hog, a stray file
write, an unexpected network call) fail safely on a developer machine.
Model-generated code must still be treated as untrusted: run code-execution
evaluations only on a machine and account you are prepared to lose, and prefer
a disposable VM or container for anything beyond the project's own fixtures.
Code execution is off by default everywhere in :mod:`epor.evaluation`; a
caller must pass ``allow_code_execution=True`` explicitly.

Each candidate runs in a fresh process:

* ``python -I -S -B`` (isolated mode: no user site, no ``PYTHON*``
  environment, no script-directory import path; no ``site``; no bytecode).
* A fresh private working directory (mode ``0700``) that is deleted after the
  run, plus a separate parent-only directory for captured output.
* A stripped environment: only ``PATH``, ``LANG``, ``HOME`` (the working
  directory), and ``TMPDIR`` (likewise).  Nothing from the caller's
  environment, including secrets, is inherited.
* ``stdin`` is ``/dev/null``; stdout and stderr go to files capped by
  ``RLIMIT_FSIZE`` and are read back truncated to ``max_output_bytes``.
* ``resource`` limits: CPU seconds, address space, file size, open files,
  core dumps disabled, and a process count.
* A wall-clock timeout, after which the whole process group (the child runs in
  its own session) is killed with ``SIGKILL``; the group is also killed after a
  normal exit so leftover children cannot linger.

Before the candidate is ``exec``-ed, a small trusted launcher (running in the
fresh child, single-threaded) applies, *where the kernel permits*:

* an unprivileged **user + network namespace** (``unshare(CLONE_NEWUSER |
  CLONE_NEWNET)``): the candidate sees only a down loopback interface, so it
  has no network at all.  The process limit is then set inside the new user
  namespace, where Linux (5.14+) counts only that namespace's processes.
* **Landlock** (``no_new_privs`` + a ruleset): file *writes, creations, and
  removals* are denied everywhere except beneath the working directory; with
  Landlock ABI 4+, TCP bind/connect are denied too; with ABI 6+, signals to
  processes outside the sandbox and abstract UNIX sockets are scoped away.

What was actually achieved is reported in :class:`SandboxIsolation` and
recorded in evaluation reports.  If the network namespace cannot be created,
``network_isolated`` is ``False`` and :func:`require_isolation` refuses to run
unless the caller additionally passes ``allow_unisolated_network=True``.

Known gaps (honest list): reads are not restricted (a candidate can read any
file the user can read and print it into its captured output); a candidate
can ``setsid`` its children out of the process group (they still inherit the
rlimits, namespaces, and Landlock domain, but the post-run kill misses them
until their CPU limit expires); without a user namespace the process limit
counts *all* of the user's processes and is therefore raised to the current
count plus the budget; there is no seccomp filter, no cgroup, no PID or mount
namespace, and kernel vulnerabilities are out of scope.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SANDBOX_VERSION = "epor-sandbox-v1"


class SandboxError(RuntimeError):
    """The sandbox could not be prepared, or running a candidate was refused."""


class LauncherSetupError(SandboxError):
    """The launcher exited before reporting isolation, so no candidate code ran."""


class CodeExecutionRefusedError(SandboxError):
    """Code execution was not explicitly enabled, or isolation was insufficient."""


class SandboxLimits(BaseModel):
    """Resource limits applied to every candidate process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_seconds: int = Field(default=5, ge=1, le=600)
    wall_seconds: float = Field(default=10.0, gt=0, le=1_800)
    address_space_bytes: int = Field(default=1024 * 1024 * 1024, ge=64 * 1024 * 1024)
    file_size_bytes: int = Field(default=1024 * 1024, ge=1024)
    open_files: int = Field(default=64, ge=16, le=4_096)
    processes: int = Field(default=16, ge=1, le=1_024)
    max_output_bytes: int = Field(default=16 * 1024, ge=256, le=1024 * 1024)


class SandboxIsolation(BaseModel):
    """Which isolation layers the launcher actually achieved for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sandbox_version: Literal["epor-sandbox-v1"] = SANDBOX_VERSION
    network_isolated: bool
    network_method: Literal["user-network-namespace", "landlock-tcp-only", "none"]
    filesystem_writes_confined: bool
    landlock_abi: int = Field(ge=0)
    signals_scoped: bool
    process_limit_scope: Literal["user-namespace", "global-user"]
    notes: list[str] = Field(default_factory=list)


SandboxStatus = Literal["passed", "failed", "timeout", "resource-limit", "error"]


class SandboxResult(BaseModel):
    """Outcome of one candidate run (no timings, so reports stay deterministic)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SandboxStatus
    returncode: int | None
    signal_name: str | None = None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    isolation: SandboxIsolation


# The launcher runs in the fresh child before the candidate.  It is trusted,
# project-authored code executed with ``python -I -S -c`` and receives its
# configuration as one JSON argument.  It reports what it achieved on a pipe
# that it closes before ``exec``, so the candidate cannot forge the status.
_LAUNCHER = r"""
import ctypes, json, os, resource, sys

cfg = json.loads(sys.argv[1])
status_fd = cfg["status_fd"]
notes = []
libc = ctypes.CDLL(None, use_errno=True)
syscall = libc.syscall
syscall.restype = ctypes.c_long

def limit(kind, value):
    try:
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError) as exc:
        notes.append(f"rlimit {kind} not applied: {exc}")

limit(resource.RLIMIT_CORE, 0)
limit(resource.RLIMIT_CPU, cfg["cpu_seconds"])
limit(resource.RLIMIT_AS, cfg["address_space_bytes"])
limit(resource.RLIMIT_FSIZE, cfg["file_size_bytes"])
limit(resource.RLIMIT_NOFILE, cfg["open_files"])

CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
netns = False
uid, gid = os.getuid(), os.getgid()
if cfg["try_namespaces"]:
    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) == 0:
        try:
            with open("/proc/self/setgroups", "w") as handle:
                handle.write("deny")
            with open("/proc/self/uid_map", "w") as handle:
                handle.write(f"{uid} {uid} 1")
            with open("/proc/self/gid_map", "w") as handle:
                handle.write(f"{gid} {gid} 1")
            netns = True
        except OSError as exc:
            notes.append(f"user namespace mapping failed: {exc}")
            os._exit(125)
    else:
        notes.append(f"unshare failed: errno {ctypes.get_errno()}")

if netns:
    limit(resource.RLIMIT_NPROC, cfg["processes"])
    process_scope = "user-namespace"
else:
    limit(resource.RLIMIT_NPROC, cfg["processes"] + cfg["current_user_processes"])
    process_scope = "global-user"

# Landlock: x86_64 and the generic syscall table share these numbers.
SYS_CREATE, SYS_ADD, SYS_RESTRICT = 444, 445, 446
abi = 0
confined = False
scoped = False
tcp_denied = False
if cfg["try_landlock"] and libc.prctl(38, 1, 0, 0, 0) == 0:  # PR_SET_NO_NEW_PRIVS
    abi = syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        abi = 0
        notes.append("landlock unavailable")
    else:
        write = 1 << 1
        for bit in (4, 5, 6, 7, 8, 9, 10, 11, 12):  # remove/make variants
            write |= 1 << bit
        if abi >= 2:
            write |= 1 << 13  # REFER
        if abi >= 3:
            write |= 1 << 14  # TRUNCATE
        net = 0b11 if abi >= 4 else 0  # BIND_TCP | CONNECT_TCP
        scope = 0b11 if abi >= 6 else 0  # ABSTRACT_UNIX_SOCKET | SIGNAL

        class Ruleset(ctypes.Structure):
            _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64),
                        ("scoped", ctypes.c_uint64)]

        class PathBeneath(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("allowed", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

        size = 8 if abi < 4 else (16 if abi < 6 else 24)
        ruleset = Ruleset(write, net, scope)
        ruleset_fd = syscall(SYS_CREATE, ctypes.byref(ruleset), ctypes.c_size_t(size),
                             ctypes.c_uint32(0))
        if ruleset_fd < 0:
            notes.append(f"landlock ruleset failed: errno {ctypes.get_errno()}")
        else:
            work_fd = os.open(cfg["workdir"], os.O_PATH | os.O_DIRECTORY)
            rule = PathBeneath(write, work_fd)
            added = syscall(SYS_ADD, ctypes.c_int(ruleset_fd), ctypes.c_int(1),
                            ctypes.byref(rule), ctypes.c_uint32(0))
            os.close(work_fd)
            if added == 0 and syscall(SYS_RESTRICT, ctypes.c_int(ruleset_fd),
                                      ctypes.c_uint32(0)) == 0:
                confined = True
                scoped = scope != 0
                tcp_denied = net != 0
            else:
                notes.append(f"landlock restrict failed: errno {ctypes.get_errno()}")
            os.close(ruleset_fd)
elif cfg["try_landlock"]:
    notes.append("no_new_privs could not be set")

if netns:
    method = "user-network-namespace"
elif tcp_denied:
    method = "landlock-tcp-only"
else:
    method = "none"
report = {
    "network_isolated": netns,
    "network_method": method,
    "filesystem_writes_confined": confined,
    "landlock_abi": abi,
    "signals_scoped": scoped,
    "process_limit_scope": process_scope,
    "notes": notes,
}
os.write(status_fd, json.dumps(report).encode())
os.close(status_fd)
os.execve(sys.executable, [sys.executable, "-I", "-S", "-B", cfg["script"]], cfg["env"])
"""


def _current_user_processes() -> int:
    """Processes owned by this user (for the no-namespace NPROC fallback)."""

    uid = os.getuid()
    count = 0
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if entry.stat().st_uid == uid:
                    count += 1
            except OSError:
                continue
    return count


def _read_bounded(path: Path, limit: int) -> tuple[str, bool]:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except FileNotFoundError:
        return "", False
    truncated = len(data) > limit
    return data[:limit].decode("utf-8", errors="replace"), truncated


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


@dataclass(frozen=True, slots=True)
class CodeSandbox:
    """Runs Python source in a fresh, limited subprocess; see the module docstring.

    ``scratch_root`` is where the per-run private directories are created
    (default: the system temporary directory).  ``use_namespaces`` and
    ``use_landlock`` exist so tests can exercise the degraded paths; normal
    callers leave them on and read what was achieved from the result.
    """

    limits: SandboxLimits = field(default_factory=SandboxLimits)
    scratch_root: Path | None = None
    use_namespaces: bool = True
    use_landlock: bool = True

    def probe(self) -> SandboxIsolation:
        """Isolation this machine achieves, measured by running an empty candidate."""

        return self.run("pass\n").isolation

    def run(self, source: str) -> SandboxResult:
        if sys.platform != "linux":
            raise SandboxError("the code sandbox is implemented for Linux only")
        root = self.scratch_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        container = Path(tempfile.mkdtemp(prefix="epor-sandbox-", dir=root))
        try:
            container.chmod(0o700)
            workdir = container / "work"
            outputs = container / "outputs"
            workdir.mkdir(mode=0o700)
            outputs.mkdir(mode=0o700)
            script = workdir / "candidate.py"
            script.write_text(source, encoding="utf-8")
            try:
                return self._execute(workdir, outputs, script, self.use_namespaces)
            except LauncherSetupError:
                if not self.use_namespaces:
                    raise
            # Some kernels let unshare() succeed but then deny the ID mapping
            # (Ubuntu's AppArmor restriction on unprivileged user namespaces).
            # The launcher stops before exec, so the candidate never ran; run
            # it once more without namespaces and record what was lost.  The
            # degraded result still needs the separate network opt-in.
            result = self._execute(workdir, outputs, script, False)
            note = "user namespace setup failed before the candidate ran; ran without namespaces"
            isolation = result.isolation.model_copy(
                update={"notes": [note, *result.isolation.notes]}
            )
            return result.model_copy(update={"isolation": isolation})
        finally:
            shutil.rmtree(container, ignore_errors=True)

    def _execute(
        self, workdir: Path, outputs: Path, script: Path, use_namespaces: bool
    ) -> SandboxResult:
        limits = self.limits
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
        }
        status_read, status_write = os.pipe()
        config = {
            "status_fd": status_write,
            "cpu_seconds": limits.cpu_seconds,
            "address_space_bytes": limits.address_space_bytes,
            "file_size_bytes": limits.file_size_bytes,
            "open_files": limits.open_files,
            "processes": limits.processes,
            "current_user_processes": _current_user_processes(),
            "try_namespaces": use_namespaces,
            "try_landlock": self.use_landlock,
            "workdir": str(workdir),
            "script": str(script),
            "env": env,
        }
        stdout_path = outputs / "stdout"
        stderr_path = outputs / "stderr"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-B", "-c", _LAUNCHER, json.dumps(config)],
                    cwd=workdir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    pass_fds=(status_write,),
                    start_new_session=True,
                    close_fds=True,
                )
        finally:
            os.close(status_write)
        timed_out = False
        try:
            process.wait(timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        _kill_group(process.pid)
        process.wait()
        with os.fdopen(status_read, "rb") as status_stream:
            status_bytes = status_stream.read(64 * 1024)
        if not status_bytes:
            raise LauncherSetupError("sandbox launcher failed before reporting its isolation")
        isolation = SandboxIsolation.model_validate(json.loads(status_bytes))
        stdout_text, stdout_truncated = _read_bounded(stdout_path, limits.max_output_bytes)
        stderr_text, stderr_truncated = _read_bounded(stderr_path, limits.max_output_bytes)
        # The private directory name is random; hide it so results are reproducible.
        stdout_text = stdout_text.replace(str(workdir), "<workdir>")
        stderr_text = stderr_text.replace(str(workdir), "<workdir>")
        returncode = process.returncode
        signal_name: str | None = None
        if returncode is not None and returncode < 0:
            signal_name = signal.Signals(-returncode).name
        status: SandboxStatus
        if timed_out:
            status = "timeout"
        elif returncode == 0:
            status = "passed"
        elif signal_name in {"SIGXCPU", "SIGXFSZ", "SIGKILL", "SIGSEGV"} or (
            "MemoryError" in stderr_text
        ):
            status = "resource-limit"
        else:
            status = "failed"
        return SandboxResult(
            status=status,
            returncode=returncode,
            signal_name=signal_name,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            isolation=isolation,
        )


def require_isolation(
    isolation: SandboxIsolation,
    *,
    allow_unisolated_network: bool,
) -> None:
    """Refuse to proceed without network isolation unless separately opted in."""

    if not isolation.network_isolated and not allow_unisolated_network:
        raise CodeExecutionRefusedError(
            "the sandbox could not isolate the network on this machine "
            f"(method: {isolation.network_method}); pass allow_unisolated_network=True "
            "in addition to allow_code_execution=True to run candidates anyway"
        )


__all__ = [
    "SANDBOX_VERSION",
    "CodeExecutionRefusedError",
    "CodeSandbox",
    "LauncherSetupError",
    "SandboxError",
    "SandboxIsolation",
    "SandboxLimits",
    "SandboxResult",
    "SandboxStatus",
    "require_isolation",
]
