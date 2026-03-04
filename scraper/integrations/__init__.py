"""
Scrapit integrations — use Scrapit as a tool in AI agent frameworks.

Quick API (no YAML needed):

    from scraper.integrations import scrape_url, scrape_directive

    # Get clean text from any URL (ready for LLMs)
    text = scrape_url("https://example.com")

    # Run a directive and get structured data
    data = scrape_directive("wikipedia")

LangChain / CrewAI:

    from scraper.integrations.langchain import ScrapitTool, ScrapitDirectiveTool

    tools = [ScrapitTool()]   # plug into any LangChain agent

LlamaIndex:

    from scraper.integrations.llamaindex import ScrapitReader

    reader = ScrapitReader()
    docs = reader.load_data("https://example.com")
"""

import asyncio
import requests
from bs4 import BeautifulSoup

from scraper.scrapers.bs4_scraper import _HEADERS

# ── scrape_url ────────────────────────────────────────────────────────────────

def scrape_url(
    url: str,
    *,
    remove_elements: list[str] | None = None,
    timeout: int = 15,
) -> str:
    """
    Fetch a URL and return clean readable text — no YAML needed.

    Strips scripts, styles, nav, footer automatically.
    Returns a plain string ready to feed to an LLM.

    Args:
        url: The URL to scrape.
        remove_elements: Extra HTML tags to strip (default: script, style, nav, footer, aside).
        timeout: Request timeout in seconds.
    """
    strip_tags = remove_elements or ["script", "style", "nav", "footer", "aside", "header"]

    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in strip_tags:
        for el in soup.find_all(tag):
            el.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── scrape_directive ──────────────────────────────────────────────────────────

def scrape_directive(directive: str) -> dict | list[dict]:
    """
    Run a Scrapit directive and return structured data.

    Args:
        directive: Directive name (e.g. 'wikipedia') or path to YAML file.

    Returns:
        A dict (single result) or list of dicts (paginated/spider/multi-site).
    """
    from scraper.main import _resolve

    path = _resolve(directive)
    return asyncio.run(_grab(str(path)))


async def _grab(path: str):
    from scraper.scrapers import grab_elements_by_directive
    return await grab_elements_by_directive(path)


# ── as_langchain_tool ─────────────────────────────────────────────────────────

def as_langchain_tool(directive: str | None = None):
    """
    Return a ready-to-use LangChain tool.

    If directive is given, returns a ScrapitDirectiveTool.
    Otherwise returns a general ScrapitTool (scrapes any URL).

    Usage:
        tools = [scrapit.integrations.as_langchain_tool()]
    """
    if directive:
        from scraper.integrations.langchain import ScrapitDirectiveTool
        return ScrapitDirectiveTool(directive=directive)
    from scraper.integrations.langchain import ScrapitTool
    return ScrapitTool()


# ── as_llamaindex_reader ──────────────────────────────────────────────────────

def as_llamaindex_reader():
    """Return a ScrapitReader instance for LlamaIndex pipelines."""
    from scraper.integrations.llamaindex import ScrapitReader
    return ScrapitReader()
