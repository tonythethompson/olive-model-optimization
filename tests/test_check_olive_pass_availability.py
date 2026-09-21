"""Regression coverage for Olive 0.13 pass-availability aliases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_olive_pass_availability.py"
_SPEC = importlib.util.spec_from_file_location("check_olive_pass_availability", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_olive_013_catalog_claims_require_registered_capabilities():
    available = {"qairtpipelinepass", "graphsurgeries"}

    assert _MODULE._claim_in_registry("QairtPipeline", available)
    assert _MODULE._claim_in_registry("QuantizeEmbeddingInt8", available)
    assert _MODULE._claim_in_registry("ShareEmbeddingLmHead", available)
    assert _MODULE._claim_in_registry("SimplifiedLayerNormToRMSNorm", available)
    assert not _MODULE._claim_in_registry("NotARealOlivePass", available)
