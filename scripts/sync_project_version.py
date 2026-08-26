"""Synchronize generated project-version surfaces with ``pyproject.toml``.

The project version identifies the source tree and is intentionally independent
of tags, published packages, and model releases. Run this script after changing
``[project].version``; CI uses ``--check`` to reject drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_START = "<!-- project-version:start -->"
README_END = "<!-- project-version:end -->"
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class VersionSyncError(RuntimeError):
    """Raised when a generated version surface cannot be updated safely."""


def project_version(root: Path = PROJECT_ROOT) -> str:
    """Return the canonical SemVer project version."""

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not SEMVER_PATTERN.fullmatch(version):
        raise VersionSyncError("[project].version must be a complete SemVer value")
    return version


def readme_version_block(version: str) -> str:
    """Render the highlighted README version notice."""

    return "\n".join(
        (
            README_START,
            "> [!IMPORTANT]",
            f"> **Current project version: `v{version}`**",
            ">",
            "> This identifies the code in this checkout; it does not imply a formal release.",
            "> Follow implementation status in [`docs/ROADMAP.md`](docs/ROADMAP.md).",
            README_END,
        )
    )


def _updated_readme(text: str, version: str) -> str:
    block = readme_version_block(version)
    start_count = text.count(README_START)
    end_count = text.count(README_END)
    if (start_count, end_count) == (0, 0):
        heading = "# EPOR AI\n"
        if not text.startswith(heading):
            raise VersionSyncError("README.md must start with '# EPOR AI'")
        return text.replace(heading, f"{heading}\n{block}\n", 1)
    if (start_count, end_count) != (1, 1):
        raise VersionSyncError("README.md must contain exactly one complete version block")
    pattern = re.compile(f"{re.escape(README_START)}.*?{re.escape(README_END)}", re.DOTALL)
    updated, count = pattern.subn(block, text, count=1)
    if count != 1:
        raise VersionSyncError("README.md must contain one ordered version block")
    return updated


def _updated_python_version(text: str, version: str) -> str:
    pattern = re.compile(r'^__version__ = "[^"\n]+"$', re.MULTILINE)
    updated, count = pattern.subn(f'__version__ = "{version}"', text)
    if count != 1:
        raise VersionSyncError("src/epor/__init__.py must contain one __version__ assignment")
    return updated


def _updated_json_version(text: str, version: str, *, lockfile: bool) -> str:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        raise VersionSyncError("UI package metadata must contain a string version")
    data["version"] = version
    if lockfile:
        packages = data.get("packages")
        root_package = packages.get("") if isinstance(packages, dict) else None
        if not isinstance(root_package, dict) or not isinstance(root_package.get("version"), str):
            raise VersionSyncError("UI lockfile must contain the root package version")
        root_package["version"] = version
    return json.dumps(data, indent=2) + "\n"


def _updated_uv_lock(text: str, version: str) -> str:
    parts = re.split(r"(?=^\[\[package\]\]$)", text, flags=re.MULTILINE)
    matching = [index for index, part in enumerate(parts) if '\nname = "epor-ai"\n' in part]
    if len(matching) != 1:
        raise VersionSyncError("uv.lock must contain exactly one epor-ai package entry")
    index = matching[0]
    updated, count = re.subn(
        r'^version = "[^"\n]+"$',
        f'version = "{version}"',
        parts[index],
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise VersionSyncError("the epor-ai uv.lock entry must contain one version")
    parts[index] = updated
    return "".join(parts)


def expected_files(root: Path, version: str) -> dict[Path, str]:
    """Return each generated file and its synchronized contents."""

    readme = root / "README.md"
    python_init = root / "src" / "epor" / "__init__.py"
    ui_package = root / "ui" / "package.json"
    ui_lock = root / "ui" / "package-lock.json"
    uv_lock = root / "uv.lock"
    return {
        readme: _updated_readme(readme.read_text(encoding="utf-8"), version),
        python_init: _updated_python_version(python_init.read_text(encoding="utf-8"), version),
        ui_package: _updated_json_version(
            ui_package.read_text(encoding="utf-8"), version, lockfile=False
        ),
        ui_lock: _updated_json_version(ui_lock.read_text(encoding="utf-8"), version, lockfile=True),
        uv_lock: _updated_uv_lock(uv_lock.read_text(encoding="utf-8"), version),
    }


def synchronize(root: Path = PROJECT_ROOT, *, check: bool = False) -> int:
    """Write generated version surfaces, or report whether they are current."""

    version = project_version(root)
    expected = expected_files(root, version)
    drifted = [
        path for path, content in expected.items() if path.read_text(encoding="utf-8") != content
    ]

    if check:
        if not drifted:
            print(f"project version v{version} is synchronized")
            return 0
        print("project version metadata is out of sync:", file=sys.stderr)
        for path in drifted:
            print(f"  - {path.relative_to(root)}", file=sys.stderr)
        print(
            "run 'uv run --frozen --no-sync python scripts/sync_project_version.py'",
            file=sys.stderr,
        )
        return 1

    for path in drifted:
        path.write_text(expected[path], encoding="utf-8")
        print(f"updated {path.relative_to(root)}")
    if not drifted:
        print(f"project version v{version} is already synchronized")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without modifying files.",
    )
    args = parser.parse_args()
    try:
        return synchronize(check=args.check)
    except (OSError, ValueError, VersionSyncError) as exc:
        print(f"project version synchronization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
