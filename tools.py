"""Agent tools: web search, page fetching and GIF lookup.

These are plain async functions executed by the bot when the model requests
them via tool/function calling. The pure parsing helpers (``parse_*``,
``truncate``) are kept separate from the HTTP calls so they can be
unit-tested offline.

Providers (all keyless except GIF search):
  * web_search  — DuckDuckGo HTML endpoint (free, no key; slightly fragile)
  * fetch_page  — Jina Reader (https://r.jina.ai) turns any URL into markdown
  * search_gifs — Klipy (default) / Tenor / Giphy — needs a free API key
"""

from __future__ import annotations

import re
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup

SEARCH_TIMEOUT = 8.0
FETCH_TIMEOUT = 15.0
GIF_TIMEOUT = 8.0

TENOR_SEARCH_URL = "https://tenor.com/v2/search"
JINA_PREFIX = "https://r.jina.ai/"
# Klipy's API is Tenor-compatible (same params/response shape); Giphy differs.
GIF_SEARCH_URLS = {
    "klipy": "https://api.klipy.com/v2/search",
    "tenor": TENOR_SEARCH_URL,
    "giphy": "https://api.giphy.com/v1/gifs/search",
}
# DuckDuckGo's HTML endpoint blocks the default httpx user agent.
DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------- #
# Pure parsing helpers (unit-testable without network)                   #
# ---------------------------------------------------------------------- #

def truncate(text: str, max_chars: int) -> str:
    """Trim ``text`` to at most ``max_chars`` characters, marking the cut.

    Cuts at the last line boundary (falling back to the last word) inside the
    budget so whole lines — e.g. lyrics — survive intact instead of being
    sliced mid-word.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    line_break = cut.rfind("\n")
    if line_break > 0:
        cut = cut[:line_break]
    else:
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
    return cut.rstrip() + "\n…[truncado — conteúdo incompleto]"


JINA_HEADER_PREFIXES = ("Title:", "URL Source:", "Published Time:", "Warning:", "Markdown Content:")


def strip_jina_header(text: str) -> str:
    """Remove Jina Reader's front-matter (Title / URL Source / … / Warning).

    Jina prepends a few metadata lines before the real page content; stripping
    them frees the token budget for the content itself (lyrics, articles…).
    """
    lines = text.splitlines()
    start = 0
    while start < len(lines):
        line = lines[start].strip()
        if not line or line.startswith(JINA_HEADER_PREFIXES):
            start += 1
        else:
            break
    return "\n".join(lines[start:]).strip()


def parse_duckduckgo_results(html: str, limit: int = 5) -> list[dict[str, str]]:
    """Extract ``{title, url, snippet}`` from DuckDuckGo HTML search results."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    for a in soup.select("a.result__a"):
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")
        # DDG result links are redirects: //duckduckgo.com/l/?uddg=<encoded>&rut=…
        match = re.search(r"uddg=([^&]+)", href)
        url = unquote(match.group(1)) if match else ""
        if not url.startswith("http"):
            continue
        snippet_el = a.find_next("a", class_="result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def parse_tenor_results(payload: dict, limit: int = 3) -> list[dict[str, str]]:
    """Extract ``{title, url, preview}`` from a Tenor/Klipy /v2/search response."""
    results: list[dict[str, str]] = []
    for item in payload.get("results", [])[:limit]:
        title = item.get("content_description") or item.get("title") or "GIF"
        media = item.get("media_formats", {})
        if not isinstance(media, dict):
            continue  # defensive: some providers return a (possibly empty) list
        gif = media.get("mediumgif") or media.get("gif") or {}
        url = gif.get("url") or ""
        if not url:
            continue
        preview = media.get("tinygif", {}).get("url", "")
        results.append({"title": title, "url": url, "preview": preview})
    return results


def parse_giphy_results(payload: dict, limit: int = 3) -> list[dict[str, str]]:
    """Extract ``{title, url, preview}`` from a Giphy /v1/gifs/search response."""
    results: list[dict[str, str]] = []
    for item in payload.get("data", [])[:limit]:
        title = item.get("title") or "GIF"
        images = item.get("images", {})
        url = (images.get("fixed_height") or images.get("downsized") or {}).get("url") or ""
        if not url:
            continue
        preview = (images.get("fixed_height_still") or {}).get("url", "")
        results.append({"title": title, "url": url, "preview": preview})
    return results


# ---------------------------------------------------------------------- #
# HTTP-backed tools                                                      #
# ---------------------------------------------------------------------- #

async def web_search(query: str, limit: int = 5) -> str:
    """Search DuckDuckGo and return a compact text summary for the model."""
    query = query.strip()
    if not query:
        return "ERRO: pesquisa sem termos."
    async with httpx.AsyncClient(
        timeout=SEARCH_TIMEOUT, follow_redirects=True, headers=DDG_HEADERS
    ) as client:
        response = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()
        results = parse_duckduckgo_results(response.text, limit=limit)

    if not results:
        return "Nenhum resultado encontrado."
    lines = [f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results)]
    return "\n\n".join(lines)


async def fetch_page(url: str, max_chars: int = 2000) -> str:
    """Fetch a URL via Jina Reader and return readable text/markdown."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return f"ERRO: URL inválida: {url!r}"
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(JINA_PREFIX + url)
        response.raise_for_status()
        return truncate(strip_jina_header(response.text), max_chars)


async def search_gifs(query: str, api_key: str, *, provider: str = "klipy", limit: int = 1) -> str:
    """Search for a GIF and return its URL for the model to share.

    ``provider`` is one of ``klipy`` (default, Tenor-compatible), ``tenor`` or
    ``giphy``. All need a free API key.
    """
    query = query.strip()
    if not query:
        return "ERRO: pesquisa de GIF sem termos."
    if not api_key:
        return (
            "ERRO: GIF_API_KEY não está configurada no .env "
            "(obtém uma chave grátis — Klipy: partner.klipy.com · Giphy: developers.giphy.com)."
        )
    if provider not in GIF_SEARCH_URLS:
        return f"ERRO: provider de GIF desconhecido: {provider!r} (usa klipy, tenor ou giphy)."

    if provider == "giphy":
        params = {"api_key": api_key, "q": query, "limit": limit, "rating": "g"}
        parse = parse_giphy_results
    elif provider == "klipy":
        # Klipy does not support media_filter=minimal — it returns an empty
        # list for media_formats. Without it, the response is Tenor-shaped.
        params = {"key": api_key, "q": query, "limit": limit}
        parse = parse_tenor_results
    else:  # tenor
        params = {"key": api_key, "q": query, "limit": limit, "media_filter": "minimal"}
        parse = parse_tenor_results

    async with httpx.AsyncClient(timeout=GIF_TIMEOUT) as client:
        response = await client.get(GIF_SEARCH_URLS[provider], params=params)
        response.raise_for_status()
        results = parse(response.json(), limit=limit)

    if not results:
        return "Nenhum GIF encontrado."
    return "\n".join(f"- {r['title']}: {r['url']}" for r in results)


# ---------------------------------------------------------------------- #
# Schema + dispatch                                                      #
# ---------------------------------------------------------------------- #

def tool_schemas(gif_api_key: str) -> list[dict]:
    """JSON schemas advertised to the model (GIF tool only if a key exists)."""
    schemas: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Pesquisa na web (DuckDuckGo). Usa para factos atuais, notícias, "
                    "preços, ou qualquer coisa que não saibas de cor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termos da pesquisa"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Lê o conteúdo de uma página web (via Jina Reader) e devolve texto. "
                    "Usa com um URL obtido de web_search para obteres detalhes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL completa (https://…)"},
                    },
                    "required": ["url"],
                },
            },
        },
    ]
    if gif_api_key:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "search_gifs",
                    "description": (
                        "Procura um GIF (Klipy/Tenor/Giphy). Usa quando o pedido for humor, "
                        "reação, celebração ou um GIF explícito."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Descrição do GIF, ex.: 'palmas'"},
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    return schemas


async def run_tool(name: str, arguments: dict, *, gif_api_key: str = "", gif_provider: str = "klipy", max_chars: int = 2000) -> str:
    """Dispatch a tool call by name. Never raises — errors become tool output."""
    try:
        if name == "web_search":
            return await web_search(str(arguments.get("query", "")))
        if name == "fetch_page":
            return await fetch_page(str(arguments.get("url", "")), max_chars=max_chars)
        if name == "search_gifs":
            return await search_gifs(
                str(arguments.get("query", "")), gif_api_key, provider=gif_provider
            )
        return f"ERRO: ferramenta desconhecida: {name}"
    except Exception as exc:  # noqa: BLE001 — surface so the model can recover
        return f"ERRO ao executar {name}: {type(exc).__name__}: {exc}"
