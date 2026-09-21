"""Fetcher for https://microsoft.github.io/Olive/."""

from typing import Any

from ._http import fetch_html, markdown_from_html

# SOURCE: https://microsoft.github.io/Olive/
OFFICIAL_DOCS_URL = "https://microsoft.github.io/Olive/"
PAGE_PATHS = {
    "index": "index.html",
    "why-olive": "why-olive.html",
    "getting-started": "getting-started/getting-started.html",
    "how-to": "how-to/index.html",
    "passes": "reference/pass.html",
    "tutorials": "how-to/index.html",
    "pass-configuration": "how-to/configure-workflows/pass-configuration.html",
    "options": "reference/options.html",
    "cli": "reference/cli.html",
    "quantization": "features/quantization.html",
    "ihv-integration": "features/ihv-integration/index.html",
    "model-compression": "features/model-compression.html",
}


def _try_fetch(url: str) -> str:
    """Fetch a URL and return its Markdown content, raising on failure."""
    return markdown_from_html(fetch_html(url))


def _candidates(page: str) -> list[str]:
    if page in PAGE_PATHS:
        return [f"{OFFICIAL_DOCS_URL}{PAGE_PATHS[page]}"]
    paths = [page, f"{page}.html", f"{page}/", f"{page}/index.html"]
    return list(dict.fromkeys(f"{OFFICIAL_DOCS_URL}{path}" for path in paths))


def fetch_official_docs(pages: list[str] | None = None) -> dict[str, Any]:
    """Fetch official Olive docs pages.

    Args:
        pages: List of doc paths to fetch. If None, fetches core pages.

    Returns:
        Dict mapping page path to Markdown content or error info.
    """
    target_pages = (
        pages
        if pages is not None
        else [
            "index",
            "why-olive",
            "getting-started",
            "how-to",
            "passes",
            "pass-configuration",
            "options",
            "quantization",
            "ihv-integration",
        ]
    )
    result: dict[str, Any] = {
        "status": "ok",
        "source": OFFICIAL_DOCS_URL,
        "pages": {},
        "sources": {},
    }

    for page in target_pages:
        candidates = _candidates(page)
        last_error = "No URL candidates tried"
        for url in candidates:
            try:
                result["pages"][page] = _try_fetch(url)
                result["sources"][page] = url
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        else:
            result["pages"][page] = {"error": last_error}
            result["status"] = "partial"

    if all(isinstance(v, dict) and "error" in v for v in result["pages"].values()):
        result["status"] = "error"

    return result
