"""
LlamaIndex integration for Scrapit.

Provides a BaseReader-compatible reader that can be used in
LlamaIndex ingestion pipelines and RAG applications.

Usage:

    from scraper.integrations.llamaindex import ScrapitReader

    # Load from a directive
    reader = ScrapitReader()
    docs = reader.load_data(directive="wikipedia")

    # Load from a plain URL
    docs = reader.load_data(url="https://example.com")

    # Use with LlamaIndex VectorStoreIndex
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main topic?")
"""

from __future__ import annotations

from scraper.integrations import scrape_url, scrape_directive
from scraper.integrations.langchain import _dict_to_text


class ScrapitReader:
    """
    LlamaIndex-compatible reader for Scrapit.

    Implements the BaseReader interface — works with LlamaIndex's
    ingestion pipeline, SimpleDirectoryReader pattern, and
    VectorStoreIndex.from_documents().
    """

    def load_data(
        self,
        url: str | None = None,
        directive: str | None = None,
        urls: list[str] | None = None,
        directives: list[str] | None = None,
    ) -> list:
        """
        Load scraped content as LlamaIndex Document objects.

        Args:
            url: Single URL to scrape (returns plain text).
            directive: Directive name or path (returns structured data).
            urls: List of URLs to scrape.
            directives: List of directive names/paths.
        """
        try:
            from llama_index.core import Document  # type: ignore
        except ImportError:
            try:
                from llama_index import Document  # type: ignore
            except ImportError:
                raise ImportError(
                    "llama-index is required for ScrapitReader. "
                    "Install with: pip install llama-index-core"
                )

        docs = []

        # Single URL
        if url:
            text = scrape_url(url)
            docs.append(Document(text=text, metadata={"source": url}))

        # Multiple URLs
        for u in urls or []:
            try:
                text = scrape_url(u)
                docs.append(Document(text=text, metadata={"source": u}))
            except Exception as e:
                from scraper.logger import log
                log(f"ScrapitReader: error scraping {u}: {e}", "warning")

        # Single directive
        if directive:
            docs.extend(self._load_directive(directive, Document))

        # Multiple directives
        for d in directives or []:
            docs.extend(self._load_directive(d, Document))

        return docs

    def _load_directive(self, directive: str, Document) -> list:
        result = scrape_directive(directive)
        results = result if isinstance(result, list) else [result]
        docs = []
        for item in results:
            text = _dict_to_text(item)
            metadata = {
                "source": item.get("url", directive),
                "timestamp": str(item.get("timestamp", "")),
                "directive": directive,
            }
            docs.append(Document(text=text, metadata=metadata))
        return docs
