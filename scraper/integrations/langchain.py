"""
LangChain integration for Scrapit.

Works with LangChain agents, CrewAI, LangGraph, and any framework
that accepts LangChain-compatible tools.

Tools:
    ScrapitTool           — scrape any URL, returns clean text
    ScrapitDirectiveTool  — run a named directive, returns structured JSON

Document Loader:
    ScrapitLoader         — load directive output as LangChain Documents

Usage:

    from scraper.integrations.langchain import ScrapitTool, ScrapitDirectiveTool

    # General browsing tool (give to agent, it decides what to scrape)
    tools = [ScrapitTool()]

    # Directive-specific tool
    tools = [ScrapitDirectiveTool(directive="wikipedia")]

    # As a Document Loader (for RAG pipelines)
    from scraper.integrations.langchain import ScrapitLoader
    loader = ScrapitLoader("wikipedia")
    docs = loader.load()
"""

from __future__ import annotations

import json
from typing import Any, Type

from scraper.integrations import scrape_url, scrape_directive


# ── ScrapitTool ───────────────────────────────────────────────────────────────

class ScrapitTool:
    """
    LangChain-compatible tool that scrapes any URL and returns clean text.

    Drop-in for any agent that accepts tools with a `.run(input)` interface.
    Also implements the full `BaseTool` interface when langchain is installed.
    """

    name: str = "web_scraper"
    description: str = (
        "Scrape the text content of a web page. "
        "Use this when you need to read or extract information from a URL. "
        "Input must be a valid URL (starting with http:// or https://). "
        "Returns the readable text of the page."
    )

    def run(self, url: str, **kwargs) -> str:
        """Scrape the URL and return clean text."""
        try:
            return scrape_url(url.strip())
        except Exception as e:
            return f"Error scraping {url}: {e}"

    # LangChain BaseTool compatibility
    def _run(self, url: str, **kwargs) -> str:
        return self.run(url)

    async def _arun(self, url: str, **kwargs) -> str:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.run, url)

    # Allow use as a LangChain StructuredTool / Tool dict
    def as_langchain(self):
        """Return a native langchain Tool object."""
        from langchain.tools import Tool  # type: ignore
        return Tool(name=self.name, func=self.run, description=self.description)


# ── ScrapitDirectiveTool ──────────────────────────────────────────────────────

class ScrapitDirectiveTool:
    """
    LangChain-compatible tool that runs a Scrapit directive by name.

    Returns structured JSON — useful when the agent needs specific fields
    rather than raw page text.

    Usage:
        tool = ScrapitDirectiveTool(directive="wikipedia")
        result = tool.run("wikipedia")   # directive name or path
    """

    name: str = "scrapit_directive"
    description: str = (
        "Run a Scrapit directive to scrape a website with a predefined configuration. "
        "Returns structured JSON with the scraped fields. "
        "Input: directive name (e.g. 'wikipedia') or path to a YAML file."
    )

    def __init__(self, directive: str | None = None):
        self.default_directive = directive
        if directive:
            self.name = f"scrapit_{directive}"
            self.description = (
                f"Scrape data using the '{directive}' directive. "
                "Returns structured JSON with the scraped fields."
            )

    def run(self, directive: str | None = None, **kwargs) -> str:
        target = directive or self.default_directive
        if not target:
            return "Error: no directive specified."
        try:
            result = scrape_directive(target.strip())
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return f"Error running directive '{target}': {e}"

    def _run(self, directive: str | None = None, **kwargs) -> str:
        return self.run(directive)

    async def _arun(self, directive: str | None = None, **kwargs) -> str:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, directive
        )

    def as_langchain(self):
        from langchain.tools import Tool  # type: ignore
        return Tool(name=self.name, func=self.run, description=self.description)


# ── ScrapitLoader ─────────────────────────────────────────────────────────────

class ScrapitLoader:
    """
    LangChain Document Loader — loads scraped content as Document objects.

    Can load from:
    - A directive name/path → structured fields become page_content
    - A plain URL → full page text becomes page_content

    Usage (RAG pipeline):

        from scraper.integrations.langchain import ScrapitLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = ScrapitLoader("wikipedia")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        chunks = splitter.split_documents(docs)
    """

    def __init__(self, source: str, mode: str = "auto"):
        """
        Args:
            source: Directive name/path OR a URL.
            mode: 'directive', 'url', or 'auto' (default: detect from source).
        """
        self.source = source
        self.mode = mode

    def load(self) -> list:
        """Return list of LangChain Document objects."""
        try:
            from langchain_core.documents import Document  # type: ignore
        except ImportError:
            try:
                from langchain.schema import Document  # type: ignore
            except ImportError:
                raise ImportError(
                    "langchain or langchain-core is required for ScrapitLoader. "
                    "Install with: pip install langchain-core"
                )

        mode = self._detect_mode()

        if mode == "url":
            text = scrape_url(self.source)
            return [Document(page_content=text, metadata={"source": self.source})]

        # directive mode
        result = scrape_directive(self.source)
        results = result if isinstance(result, list) else [result]
        docs = []
        for item in results:
            content = _dict_to_text(item)
            metadata = {
                "source": item.get("url", self.source),
                "timestamp": str(item.get("timestamp", "")),
                "directive": self.source,
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def lazy_load(self):
        yield from self.load()

    def _detect_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        if self.source.startswith(("http://", "https://")):
            return "url"
        return "directive"


def _dict_to_text(data: dict) -> str:
    """Convert a scraped dict to readable text for LLM consumption."""
    skip = {"url", "timestamp", "_id", "_page", "_source", "_valid", "_errors"}
    lines = []
    for key, value in data.items():
        if key in skip or value is None:
            continue
        if isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
