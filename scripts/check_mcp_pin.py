"""Fail if project.dependencies lets mcp 2.x or newer resolve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def mcp_requirement_from_deps(deps: list[str]) -> Requirement | None:
    for dep in deps:
        parsed = Requirement(dep)
        if parsed.name == "mcp":
            return parsed
    return None


def _get_candidate_versions(spec) -> list[Version]:
    candidates = {
        Version("2.0.0"),
        Version("2.0.1"),
        Version("2.1.0"),
        Version("2.99.0"),
        Version("3.0.0"),
        Version("10.0.0"),
    }
    for s in spec:
        raw_v = s.version.rstrip(".*")
        if not raw_v:
            continue
        try:
            v = Version(raw_v)
        except InvalidVersion:
            pass
        else:
            base_versions = [v]
            if v.major < 2:
                base_versions.append(Version(f"2.{v.minor}.{v.micro}"))

            for base_v in base_versions:
                candidates.add(base_v)
                if base_v.major >= 2:
                    candidates.add(Version(f"{base_v.major}.{base_v.minor}.{base_v.micro + 1}"))
                    if base_v.micro > 0:
                        candidates.add(Version(f"{base_v.major}.{base_v.minor}.{base_v.micro - 1}"))
                    candidates.add(Version(f"{base_v.major}.{base_v.minor + 1}.0"))
                    if base_v.minor > 0:
                        candidates.add(Version(f"{base_v.major}.{base_v.minor - 1}.0"))
                    candidates.add(Version(f"{base_v.major + 1}.0.0"))
                    if base_v.major > 2:
                        candidates.add(Version(f"{base_v.major - 1}.0.0"))

    return [c for c in candidates if c >= Version("2.0.0")]


def allows_mcp_two_or_newer(spec) -> bool:
    """True if any released mcp 2+ version satisfies the specifier."""
    if not spec:
        return True
    return any(spec.contains(candidate, prereleases=False) for candidate in _get_candidate_versions(spec))


def pin_excludes_two_plus(dep: str) -> bool:
    req = Requirement(dep)
    if req.name != "mcp":
        return False
    return not allows_mcp_two_or_newer(req.specifier)


def verify_pyproject(path: Path) -> str:
    data = tomllib.load(path.open("rb"))
    deps = data.get("project", {}).get("dependencies", [])
    req = mcp_requirement_from_deps(deps)
    if req is None:
        raise SystemExit("mcp dependency is missing from project.dependencies")
    if not req.specifier:
        raise SystemExit("mcp dependency must be pinned below 2 (unpinned specifier)")
    if allows_mcp_two_or_newer(req.specifier):
        raise SystemExit(f"mcp dependency must not allow 2.x or newer (e.g. 'mcp>=1,<2'). Got: {req}")
    return str(req)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pyproject",
        nargs="?",
        type=Path,
        default=Path("pyproject.toml"),
    )
    args = parser.parse_args(argv)
    req = verify_pyproject(args.pyproject)
    print(f"MCP constraint: {req}")
    print("mcp pin excludes 2.x and newer")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"::error::{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
