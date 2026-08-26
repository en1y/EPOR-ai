from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from epor import __version__
from epor.research.service import USER_AGENT

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_version() -> str:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    assert isinstance(version, str)
    return version


def test_project_version_matches_runtime_and_lock_metadata() -> None:
    version = _canonical_version()
    assert __version__ == version
    assert USER_AGENT.startswith(f"epor-research/{version} ")

    ui_package = json.loads((PROJECT_ROOT / "ui" / "package.json").read_text(encoding="utf-8"))
    ui_lock = json.loads((PROJECT_ROOT / "ui" / "package-lock.json").read_text(encoding="utf-8"))
    assert ui_package["version"] == version
    assert ui_lock["version"] == version
    assert ui_lock["packages"][""]["version"] == version

    uv_lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    local_packages = [
        package
        for package in uv_lock["package"]
        if package["name"] == "epor-ai" and package.get("source") == {"editable": "."}
    ]
    assert len(local_packages) == 1
    assert local_packages[0]["version"] == version


def test_generated_project_version_surfaces_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_project_version.py", "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_synchronizer_updates_every_generated_version_surface(tmp_path: Path) -> None:
    current = _canonical_version()
    core_version = current.split("+", 1)[0].split("-", 1)[0]
    major, minor, patch = (int(part) for part in core_version.split("."))
    updated = f"{major}.{minor}.{patch + 1}-rc.1+sync.test"
    copied_paths = (
        "README.md",
        "pyproject.toml",
        "scripts/sync_project_version.py",
        "src/epor/__init__.py",
        "ui/package.json",
        "ui/package-lock.json",
        "uv.lock",
    )
    for relative_path in copied_paths:
        source = PROJECT_ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    pyproject_path = tmp_path / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    pyproject_path.write_text(
        pyproject.replace(f'version = "{current}"', f'version = "{updated}"', 1),
        encoding="utf-8",
    )

    stale_result = subprocess.run(
        [sys.executable, "scripts/sync_project_version.py", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert stale_result.returncode == 1
    assert "README.md" in stale_result.stderr

    write_result = subprocess.run(
        [sys.executable, "scripts/sync_project_version.py"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert write_result.returncode == 0, write_result.stderr

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    python_init = (tmp_path / "src" / "epor" / "__init__.py").read_text(encoding="utf-8")
    ui_package = json.loads((tmp_path / "ui" / "package.json").read_text(encoding="utf-8"))
    ui_lock = json.loads((tmp_path / "ui" / "package-lock.json").read_text(encoding="utf-8"))
    uv_lock = tomllib.loads((tmp_path / "uv.lock").read_text(encoding="utf-8"))
    local_package = next(package for package in uv_lock["package"] if package["name"] == "epor-ai")

    assert f"**Current project version: `v{updated}`**" in readme
    assert f'__version__ = "{updated}"' in python_init
    assert ui_package["version"] == updated
    assert ui_lock["version"] == updated
    assert ui_lock["packages"][""]["version"] == updated
    assert local_package["version"] == updated

    check_result = subprocess.run(
        [sys.executable, "scripts/sync_project_version.py", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert check_result.returncode == 0, check_result.stderr

    (tmp_path / "README.md").write_text(
        "# EPOR AI\n\n<!-- project-version:end -->\n<!-- project-version:start -->\n",
        encoding="utf-8",
    )
    malformed_result = subprocess.run(
        [sys.executable, "scripts/sync_project_version.py", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert malformed_result.returncode == 2
    assert "one ordered version block" in malformed_result.stderr
