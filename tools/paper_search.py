#!/usr/bin/env python3
"""paper_search.py —— Tool 2：论文检索（本地语料优先，联网为可选增强）。

两类用途，正好对应 SKILL.md 阅读流程第 7 步"补缺"：

1. **本地（默认，离线）**：在 `paper/` 语料里按关键词定位"这句话/这个概念在哪篇哪页"，
   用于跨论文比较，也用于确认某个说法是否真的出自某篇论文。
2. **联网（可选）**：arXiv / Crossref，用于论文之外的背景概念、方法名、
   benchmark 名称（Agent 发现自己不懂某个术语时才该用它）。

命令行::

    python3 tools/paper_search.py list
    python3 tools/paper_search.py search "袖舞 风格" --source local --top 5
    python3 tools/paper_search.py search "agent memory" --source web --max-results 5
    python3 tools/paper_search.py search "tool use" --source all

失败方式与 `pdf_reader.py` 一致：`ToolError` → stderr 一行 + exit 2。
联网失败（无网络/超时/接口限流）不会影响本地结果，只会写进 `warnings`。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # 允许 `python3 tools/paper_search.py` 直接跑
    sys.path.insert(0, str(REPO_ROOT))

from tools.pdf_reader import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    ToolError,
    normalize_with_map,
    open_document,
    snippet,
)

DEFAULT_PAPERS_DIR = REPO_ROOT / "paper"
USER_AGENT = "Agentic-Paper-Reading-Skill/0.1 (research tool; contact: local user)"
ATOM = "{http://www.w3.org/2005/Atom}"


# --------------------------------------------------------------------------- #
# 查询解析
# --------------------------------------------------------------------------- #

_TERM_SPLIT = re.compile(r"[\s,，、;；+/|]+")


def parse_terms(query: str, explicit: str | None = None) -> list[str]:
    """把查询拆成词项。中文长句不加空格时视为单一词项，避免乱切。"""

    raw = explicit if explicit else query
    terms = [t for t in _TERM_SPLIT.split(raw.strip()) if t]
    if not terms:
        raise ToolError("检索词为空")
    ordered: list[str] = []
    for term in terms:
        if term not in ordered:
            ordered.append(term)
    return ordered


# --------------------------------------------------------------------------- #
# 本地语料
# --------------------------------------------------------------------------- #


def list_corpus(papers_dir: Path) -> list[dict[str, Any]]:
    if not papers_dir.exists():
        raise ToolError(f"语料目录不存在：{papers_dir}")
    items: list[dict[str, Any]] = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        try:
            document = open_document(pdf, cache_dir=DEFAULT_CACHE_DIR)
            pages_total = document["pages_total"]
            title = guess_title(document["page_texts"])
            warnings = document.get("warnings", [])
        except ToolError as exc:
            items.append({"paper_id": pdf.stem, "path": str(pdf), "error": str(exc)})
            continue
        items.append(
            {
                "paper_id": pdf.stem,
                "path": str(pdf),
                "title": title,
                "pages_total": pages_total,
                "warnings": warnings,
            }
        )
    return items


def guess_title(page_texts: Sequence[str]) -> str:
    """首页最靠前的、像标题的那一行（标题跨行时继续拼接，遇到句读即停）。"""

    raw_lines = page_texts[0].split("\n") if page_texts else []
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_lines]
    lines = [line for line in lines if line]
    title = ""
    for line in lines[:6]:
        if len(line) < 4 or line.isdigit():
            continue
        if re.match(r"^(摘\s*要|内容提要|关键词|【|中图分类号|Abstract|Keywords)", line):
            continue
        title = f"{title}{line}" if title else line
        if len(title) >= 8 and not title.endswith(("，", ",", "、", "：", ":", "（", "(", "《", "-", "—")):
            break
    return title[:80]


def find_in_document(document: dict[str, Any], terms: Sequence[str], mode: str, context: int) -> dict[str, Any]:
    """在一篇文档里逐词检索，返回命中页码、片段与词项覆盖情况。"""

    per_term: dict[str, list[dict[str, Any]]] = {term: [] for term in terms}
    for page_number, text in enumerate(document["page_texts"], start=1):
        normalized, index_map = normalize_with_map(text)
        for term in terms:
            needle = re.sub(r"\s+", " ", normalize_with_map(term)[0])
            if not needle:
                continue
            start = normalized.find(needle)
            while start != -1 and len(per_term[term]) < 50:
                orig_start = index_map[start]
                orig_end = index_map[min(start + len(needle) - 1, len(index_map) - 1)] + 1
                per_term[term].append(
                    {"page": page_number, "snippet": snippet(text, orig_start, orig_end, context)}
                )
                start = normalized.find(needle, start + len(needle))

    matched = [t for t in terms if per_term[t]]
    missing = [t for t in terms if not per_term[t]]
    if mode == "all" and missing:
        return {"matched_terms": matched, "missing_terms": missing, "hits": []}

    # 每页最多留一条片段，按页排序；词项多的页优先
    by_page: dict[int, dict[str, Any]] = {}
    for term in terms:
        for hit in per_term[term]:
            entry = by_page.setdefault(hit["page"], {"page": hit["page"], "terms": [], "snippets": []})
            if term not in entry["terms"]:
                entry["terms"].append(term)
            if len(entry["snippets"]) < 2:
                entry["snippets"].append(hit["snippet"])
    hits = sorted(by_page.values(), key=lambda e: (-len(e["terms"]), e["page"]))
    return {"matched_terms": matched, "missing_terms": missing, "hits": hits}


def search_local(
    query: str,
    *,
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    top: int = 10,
    mode: str = "all",
    context: int = 60,
    terms: str | None = None,
) -> dict[str, Any]:
    term_list = parse_terms(query, terms)
    if not papers_dir.exists():
        raise ToolError(f"语料目录不存在：{papers_dir}")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        try:
            document = open_document(pdf, cache_dir=DEFAULT_CACHE_DIR)
        except ToolError as exc:
            errors.append({"paper_id": pdf.stem, "error": str(exc)})
            continue
        found = find_in_document(document, term_list, mode, context)
        if not found["hits"]:
            continue
        results.append(
            {
                "paper_id": pdf.stem,
                "title": guess_title(document["page_texts"]),
                "pages_total": document["pages_total"],
                "matched_terms": found["matched_terms"],
                "missing_terms": found["missing_terms"],
                "best_page": found["hits"][0]["page"],
                "hits": found["hits"][:5],
            }
        )

    results.sort(key=lambda r: (-len(r["matched_terms"]), -len(r["hits"]), r["paper_id"]))
    return {
        "source": "local",
        "query": query,
        "terms": term_list,
        "mode": mode,
        "papers_dir": str(papers_dir),
        "corpus_size": len(list(papers_dir.glob("*.pdf"))),
        "result_count": len(results),
        "results": results[:top],
        "truncated": len(results) > top,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# 联网检索
# --------------------------------------------------------------------------- #


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 固定 https 接口
        return response.read()


def search_arxiv(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    payload = _http_get(url, timeout)
    root = ET.fromstring(payload)
    out: list[dict[str, Any]] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        out.append(
            {
                "source": "arxiv",
                "title": title,
                "authors": [
                    (author.findtext(f"{ATOM}name") or "").strip()
                    for author in entry.findall(f"{ATOM}author")
                ][:8],
                "year": (entry.findtext(f"{ATOM}published") or "")[:4] or None,
                "url": (entry.findtext(f"{ATOM}id") or "").strip(),
                "abstract": abstract[:600],
            }
        )
    return out


def search_crossref(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query": query, "rows": max_results, "select": "title,author,issued,container-title,DOI,URL,type"}
    )
    payload = json.loads(_http_get(url, timeout).decode("utf-8", "replace"))
    items = (payload.get("message") or {}).get("items") or []
    out: list[dict[str, Any]] = []
    for item in items:
        titles = item.get("title") or []
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")])) for a in (item.get("author") or [])
        ]
        issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        out.append(
            {
                "source": "crossref",
                "title": " ".join((titles[0] if titles else "").split()),
                "authors": authors[:8],
                "year": issued,
                "venue": (item.get("container-title") or [None])[0],
                "type": item.get("type"),
                "doi": item.get("DOI"),
                "url": item.get("URL"),
            }
        )
    return out


WEB_BACKENDS = {"arxiv": search_arxiv, "crossref": search_crossref}


def search_web(
    query: str, *, sources: Sequence[str] = ("arxiv", "crossref"), max_results: int = 5, timeout: float = 15.0
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source in sources:
        backend = WEB_BACKENDS.get(source)
        if backend is None:
            warnings.append(f"未知联网来源 {source!r}，已跳过")
            continue
        try:
            results.extend(backend(query, max_results, timeout))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            warnings.append(f"{source} 请求失败（{type(exc).__name__}: {exc}）；可能是无网络或被限流")
        except (ET.ParseError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"{source} 返回内容无法解析：{exc}")
    return {
        "source": "web",
        "query": query,
        "sources": list(sources),
        "result_count": len(results),
        "results": results,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_search.py",
        description="Tool 2：论文检索（本地语料 / arXiv / Crossref）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="列出本地语料")
    list_cmd.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    list_cmd.add_argument("--json", action="store_true")

    search = sub.add_parser("search", help="检索")
    search.add_argument("query", help="检索词；空格分隔为多词（默认需全部命中，见 --mode）")
    search.add_argument("--source", default="local", choices=["local", "arxiv", "crossref", "web", "all"])
    search.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    search.add_argument("--terms", help="显式指定词项（逗号分隔），覆盖自动切分")
    search.add_argument("--mode", default="all", choices=["all", "any"], help="多词项的与/或关系")
    search.add_argument("--context", type=int, default=60)
    search.add_argument("--top", type=int, default=10, help="本地结果条数上限")
    search.add_argument("--max-results", type=int, default=5, help="每个联网来源的条数上限")
    search.add_argument("--timeout", type=float, default=15.0)
    search.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            items = list_corpus(Path(args.papers_dir).expanduser())
            payload = {"papers_dir": args.papers_dir, "count": len(items), "papers": items}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"语料目录：{args.papers_dir}（{len(items)} 篇）")
                for item in items:
                    if "error" in item:
                        print(f"  - {item['paper_id']}  [读取失败] {item['error']}")
                    else:
                        print(f"  - {item['paper_id']}  ({item['pages_total']} 页)  {item['title']}")
            return 0

        if args.source == "local":
            payload = search_local(
                args.query,
                papers_dir=Path(args.papers_dir).expanduser(),
                top=args.top,
                mode=args.mode,
                context=args.context,
                terms=args.terms,
            )
            payload["warnings"] = []
        elif args.source in {"arxiv", "crossref", "web"}:
            sources = ["arxiv", "crossref"] if args.source == "web" else [args.source]
            payload = search_web(
                args.query, sources=sources, max_results=args.max_results, timeout=args.timeout
            )
        else:  # all
            local = search_local(
                args.query,
                papers_dir=Path(args.papers_dir).expanduser(),
                top=args.top,
                mode=args.mode,
                context=args.context,
                terms=args.terms,
            )
            web = search_web(args.query, max_results=args.max_results, timeout=args.timeout)
            payload = {"source": "all", "local": local, "web": web}

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_text(payload))
    except ToolError as exc:
        print(f"paper_search: {exc}", file=sys.stderr)
        return 2
    return 0


def render_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    if payload.get("source") == "all":
        lines.append(render_text(payload["local"]).rstrip())
        lines.append("")
        lines.append(render_text(payload["web"]).rstrip())
        return "\n".join(lines).rstrip() + "\n"

    if payload.get("source") == "local":
        lines.append(
            f"本地检索 “{payload['query']}”（词项：{payload['terms']}，模式 {payload['mode']}，"
            f"语料 {payload['corpus_size']} 篇，命中 {payload['result_count']} 篇）"
        )
        for result in payload["results"]:
            lines.append("")
            lines.append(f"• {result['title'] or result['paper_id']}")
            lines.append(f"  paper_id={result['paper_id']}  最佳页=p{result['best_page']}  "
                         f"命中词项={result['matched_terms']}  缺={result['missing_terms']}")
            for hit in result["hits"][:3]:
                joined = " / ".join(hit["snippets"])
                lines.append(f"  [p.{hit['page']}] {joined[:220]}")
        if payload.get("truncated"):
            lines.append(f"\n（结果已截断，仅显示前 {len(payload['results'])} 篇）")
        for err in payload.get("errors") or []:
            lines.append(f"  ! {err['paper_id']} 读取失败：{err['error']}")
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"联网检索 “{payload['query']}”（来源：{payload.get('sources')}，{payload['result_count']} 条）")
    for item in payload["results"]:
        lines.append("")
        lines.append(f"• [{item.get('source')}] {item.get('title')}  ({item.get('year')})")
        if item.get("authors"):
            lines.append(f"  作者：{', '.join(item['authors'])}")
        if item.get("venue"):
            lines.append(f"  出处：{item['venue']}")
        if item.get("doi"):
            lines.append(f"  DOI：{item['doi']}")
        if item.get("url"):
            lines.append(f"  {item['url']}")
        if item.get("abstract"):
            lines.append(f"  摘要：{item['abstract'][:200]}…")
    for warning in payload.get("warnings") or []:
        lines.append(f"  ⚠️ {warning}")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
