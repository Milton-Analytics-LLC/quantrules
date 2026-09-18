# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Licensing guards.

Apache-2.0 asks that the licence travel with the source. These tests make that
mechanical rather than a thing to remember at review time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COPYRIGHT_LINE = "# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC"
LICENCE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_HOLDER = "Milton Analytics, LLC"


def has_spdx_header(path: Path) -> bool:
    """Return whether ``path`` opens with both SPDX lines, in order."""
    lines = path.read_text(encoding="utf-8").splitlines()[:2]
    return lines == [COPYRIGHT_LINE, LICENCE_LINE]


def python_sources() -> list[Path]:
    """Return every Python source file that must carry a header."""
    return sorted(
        path for directory in ("src", "tests") for path in (REPO_ROOT / directory).rglob("*.py")
    )


class TestHeaderCheck:
    def test_accepts_a_file_with_both_lines(self, tmp_path: Path) -> None:
        good = tmp_path / "good.py"
        good.write_text(f"{COPYRIGHT_LINE}\n{LICENCE_LINE}\n", encoding="utf-8")

        assert has_spdx_header(good)

    def test_rejects_a_file_with_no_header(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.py"
        bare.write_text("x = 1\n", encoding="utf-8")

        assert not has_spdx_header(bare)

    def test_rejects_a_file_missing_the_licence_line(self, tmp_path: Path) -> None:
        partial = tmp_path / "partial.py"
        partial.write_text(f"{COPYRIGHT_LINE}\nx = 1\n", encoding="utf-8")

        assert not has_spdx_header(partial)

    def test_rejects_the_lines_in_the_wrong_order(self, tmp_path: Path) -> None:
        swapped = tmp_path / "swapped.py"
        swapped.write_text(f"{LICENCE_LINE}\n{COPYRIGHT_LINE}\n", encoding="utf-8")

        assert not has_spdx_header(swapped)


class TestRepositoryHeaders:
    def test_at_least_one_source_file_was_found(self) -> None:
        """Guards against the walk silently matching nothing."""
        assert len(python_sources()) > 5

    @pytest.mark.parametrize(
        "source", python_sources(), ids=lambda p: str(p.relative_to(REPO_ROOT))
    )
    def test_every_source_file_declares_its_licence(self, source: Path) -> None:
        assert has_spdx_header(source), (
            f"{source.relative_to(REPO_ROOT)} is missing its SPDX header. "
            f"Every .py file must begin with:\n{COPYRIGHT_LINE}\n{LICENCE_LINE}"
        )


class TestLicenceFiles:
    def test_the_licence_is_apache_2(self) -> None:
        licence = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")

        assert "Apache License" in licence
        assert "Version 2.0, January 2004" in licence

    def test_the_notice_names_the_copyright_holder(self) -> None:
        notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")

        assert f"Copyright 2026 {COPYRIGHT_HOLDER}" in notice

    def test_the_notice_records_the_clean_room_provenance(self) -> None:
        """The GPL-avoidance statement is a licensing fact, not decoration."""
        notice = " ".join((REPO_ROOT / "NOTICE").read_text(encoding="utf-8").split())

        assert "pysystemtrade" in notice
        assert "written from first principles" in notice

    def test_pyproject_declares_the_same_licence(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        assert 'license = "Apache-2.0"' in pyproject

    def test_the_disclaimer_is_present_and_linked_from_the_readme(self) -> None:
        assert (REPO_ROOT / "DISCLAIMER.md").exists()
        assert "DISCLAIMER.md" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
