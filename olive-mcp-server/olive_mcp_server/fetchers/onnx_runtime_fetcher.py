"""Fetcher for ONNX Runtime execution provider docs."""

from typing import Any

from ._http import fetch_html, markdown_from_html

ONNX_RUNTIME_EP_URL = "https://onnxruntime.ai/docs/execution-providers/"
EP_SLUGS = {
    "CUDAExecutionProvider": "CUDA",
    "TensorRTExecutionProvider": "TensorRT",
    "CoreMLExecutionProvider": "CoreML",
    "QNNExecutionProvider": "QNN",
    "OpenVINOExecutionProvider": "OpenVINO",
    "DirectMLExecutionProvider": "DirectML",
}
_EP_SLUGS_NORMALIZED = {key.lower(): value for key, value in EP_SLUGS.items()}


def fetch_onnx_runtime_docs(execution_providers: list[str] | None = None) -> dict[str, Any]:
    """Fetch EP-specific operator support and gotchas.

    Args:
        execution_providers: EPs to query. If None, queries all common EPs.

    Returns:
        Dict mapping EP name to Markdown docs content or error info.
    """
    eps = (
        execution_providers
        if execution_providers is not None
        else [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CoreMLExecutionProvider",
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "DirectMLExecutionProvider",
        ]
    )
    result: dict[str, Any] = {
        "status": "ok",
        "source": ONNX_RUNTIME_EP_URL,
        "pages": {},
        "sources": {},
    }

    for ep in eps:
        if ep.lower() == "cpuexecutionprovider":
            result.setdefault("informational", {})[ep] = (
                "CPUExecutionProvider is documented by the ONNX Runtime execution-provider overview."
            )
            result["sources"][ep] = ONNX_RUNTIME_EP_URL
            continue
        slug = _EP_SLUGS_NORMALIZED.get(ep.lower(), ep.removesuffix("ExecutionProvider"))
        candidates = [f"{ONNX_RUNTIME_EP_URL}{slug}-ExecutionProvider.html"]
        last_error = "No URL candidates tried"
        for url in candidates:
            try:
                result["pages"][ep] = markdown_from_html(fetch_html(url))
                result["sources"][ep] = url
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        else:
            result["pages"][ep] = {"error": last_error}
            result["status"] = "partial"

    # Always fetch the main landing page as a fallback/aggregate source.
    try:
        result["overview"] = markdown_from_html(fetch_html(ONNX_RUNTIME_EP_URL))
        result["sources"]["overview"] = ONNX_RUNTIME_EP_URL
    except Exception as exc:  # noqa: BLE001
        result["overview_error"] = str(exc)
        result["status"] = "partial"

    if all(isinstance(v, dict) and "error" in v for v in result["pages"].values()) and "overview_error" in result:
        result["status"] = "error"

    return result
