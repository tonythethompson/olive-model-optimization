"""Specifier cases for the PyPI publish mcp<2 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_mcp_pin.py"
_SPEC = importlib.util.spec_from_file_location("check_mcp_pin", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    "dep",
    [
        "mcp<2",
        "mcp>=1,<2",
        "mcp==1.*",
        "mcp>=1.0,<2.0",
        "mcp~=1.0",
        "mcp==1.2.0",
        "mcp<2.0.0",
        "mcp<2.0",
    ],
)
def test_accepted_pins(dep: str) -> None:
    assert _MODULE.pin_excludes_two_plus(dep)


@pytest.mark.parametrize(
    "dep",
    [
        "mcp==3.0",
        "mcp~=3.0",
        "mcp>=2",
        "mcp==2",
        "mcp>=2,<3",
        "mcp>1.9",
        "mcp>=1",
        "mcp",
        "mcp!=1",
        "mcp>=3,<4",
        "mcp<2.0.1",
        "mcp<=2",
        "mcp<=2.0.0",
        "mcp==2.5.5",
        "mcp>=2.5.5,<2.6",
        "mcp~=2.5.0",
        "mcp!=2.0.0,!=2.0.1,!=2.1.0",
    ],
)
def test_rejected_pins(dep: str) -> None:
    assert not _MODULE.pin_excludes_two_plus(dep)
