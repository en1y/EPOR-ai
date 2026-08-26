"""Check or apply the covenant-v1 ratification record.

The manifest's ``covenant_sha256`` pins the covenant *in its ratified form*.
This script is the only supported way to move between draft and ratified,
because doing it by hand invites a status flip whose hash nobody recomputed —
which the loader would then reject at startup, in production, at the worst
possible moment.

``--check`` verifies the tracked hash still matches the current covenant text
and is what CI runs.  ``--apply`` performs the owner's ratification: it sets
``status: ratified`` and refreshes the hash in one step.  Neither mode writes
the rationale or the evidence list; those are human text and are preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epor.safety import (
    DEFAULT_COVENANT_PATH,
    DEFAULT_RATIFICATION_PATH,
    Covenant,
    load_ratification,
)

_HASH_LINE = re.compile(r"^covenant_sha256: [0-9a-f]{64}$", re.MULTILINE)
_STATUS_LINE = re.compile(r"^status: (draft|ratified)$", re.MULTILINE)


def ratified_hash(covenant_path: Path) -> str:
    """Return the hash the covenant would have once its status is ratified."""

    data = yaml.safe_load(covenant_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("covenant must contain a mapping")
    return Covenant.model_validate({**data, "status": "ratified"}).sha256


def _rewrite(path: Path, pattern: re.Pattern[str], replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"{path.name} must contain exactly one {pattern.pattern!r} line")
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ratify: set the covenant status and refresh the attested hash.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the tracked manifest attests to the current covenant text.",
    )
    arguments = parser.parse_args()
    if arguments.apply == arguments.check:
        parser.error("choose exactly one of --check or --apply")

    covenant_path = Path(DEFAULT_COVENANT_PATH)
    manifest_path = Path(DEFAULT_RATIFICATION_PATH)
    manifest = load_ratification(manifest_path)
    if manifest is None:
        print(f"no ratification manifest at {manifest_path}", file=sys.stderr)
        return 2

    expected = ratified_hash(covenant_path)
    if arguments.check:
        if manifest.covenant_sha256 == expected:
            print(f"ratification manifest attests to covenant v{manifest.covenant_version}")
            return 0
        print(
            "the ratification manifest does not attest to the current covenant text:\n"
            f"  manifest: {manifest.covenant_sha256}\n"
            f"  covenant: {expected}\n"
            "run 'uv run --frozen --no-sync python scripts/ratify_covenant.py --apply' "
            "after reviewing the change",
            file=sys.stderr,
        )
        return 1

    _rewrite(manifest_path, _HASH_LINE, f"covenant_sha256: {expected}")
    _rewrite(covenant_path, _STATUS_LINE, "status: ratified")
    print(f"covenant v{manifest.covenant_version} ratified at {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
