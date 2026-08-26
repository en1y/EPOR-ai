"""Validate the v0.0.1 source-archive resource contract.

The v0.0.1 wheel intentionally contains only the Python package and is not a
standalone console distribution.  This check applies only to the source
distribution, which must retain the repository resources used by the launcher.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = frozenset(
    {
        ".env.example",
        ".nvmrc",
        ".python-version",
        "README.md",
        "configs/covenant-v1.yaml",
        "configs/covenant-v1-ratification.yaml",
        "configs/models/README.md",
        "configs/models/epor-alpha.yaml",
        "configs/models/epor-beta.yaml",
        "configs/models/epor-gamma.yaml",
        "configs/models/epor-reference.yaml",
        "configs/models/epor-tiny.yaml",
        "fixtures/tiny_corpus.txt",
        "migrations/alembic.ini",
        "migrations/env.py",
        "migrations/script.py.mako",
        "migrations/versions/0001_control_plane.py",
        "migrations/versions/0002_safety_identity.py",
        "pyproject.toml",
        "research/catalog.yaml",
        "src/epor/__init__.py",
        "src/epor/cli.py",
        "src/epor/safety.py",
        "start-epor.sh",
        "ui/index.html",
        "ui/package-lock.json",
        "ui/package.json",
        "ui/src/App.test.tsx",
        "ui/src/App.tsx",
        "ui/src/api.ts",
        "ui/src/main.tsx",
        "ui/src/styles.css",
        "ui/src/test/setup.ts",
        "ui/src/types.ts",
        "ui/tsconfig.json",
        "ui/vite.config.ts",
        "uv.lock",
    }
)


class DistributionContractError(RuntimeError):
    """Raised when a source archive cannot run the source-checkout workflow."""


def validate_sdist(archive_path: Path) -> tuple[str, int]:
    """Return the archive root and member count after validating the contract."""

    if not archive_path.name.endswith(".tar.gz"):
        raise DistributionContractError(
            "expected a .tar.gz source distribution; wheels are not standalone in v0.0.1"
        )

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise DistributionContractError(f"cannot read source distribution: {exc}") from exc

    if not members:
        raise DistributionContractError("source distribution is empty")

    parsed_names = [PurePosixPath(member.name) for member in members]
    unsafe_names = [str(name) for name in parsed_names if name.is_absolute() or ".." in name.parts]
    if unsafe_names:
        raise DistributionContractError(
            "source distribution has unsafe member paths: " + ", ".join(sorted(unsafe_names))
        )

    roots = {name.parts[0] for name in parsed_names if name.parts}
    if len(roots) != 1:
        raise DistributionContractError("source distribution must have exactly one root directory")
    archive_root = roots.pop()

    by_relative_path = {
        str(PurePosixPath(*name.parts[1:])): member
        for name, member in zip(parsed_names, members, strict=True)
        if len(name.parts) > 1
    }
    missing = sorted(REQUIRED_PATHS - by_relative_path.keys())
    if missing:
        raise DistributionContractError("missing required resources: " + ", ".join(missing))

    launcher = by_relative_path["start-epor.sh"]
    if not launcher.isfile():
        raise DistributionContractError("start-epor.sh is not a regular file")
    if launcher.mode & 0o111 == 0:
        raise DistributionContractError("start-epor.sh lost its executable mode")

    return archive_root, len(members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to epor_ai-*.tar.gz")
    args = parser.parse_args()
    try:
        archive_root, member_count = validate_sdist(args.archive)
    except DistributionContractError as exc:
        print(f"source-distribution contract failed: {exc}", file=sys.stderr)
        return 1
    print(f"source-distribution contract passed: {archive_root} ({member_count} archive members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
