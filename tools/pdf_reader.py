#!/usr/bin/env python3
"""pdf_reader.py —— Tool 1：PDF 文本 + 页码提取。

为什么需要它（对应 `skills/paper-reading/SKILL.md`）：
  * 报告里的每条事实都要能写成 `[p.N]`，所以文本必须**按页保留页码**；
  * 同一份 PDF 反复读取要给出稳定结果，所以带可关闭的抽取缓存；
  * 不绑定单一 PDF 库：优先 PyMuPDF，缺失时回退 poppler 的 `pdftotext`，
    两者产出同一份 JSON 结构，Agent 侧无感。

输出结构（`read --json`）::

    {
      "paper_id": "汉代袖舞的风格类型与文化寓意_梁宇",
      "path": ".../汉代袖舞的风格类型与文化寓意_梁宇.pdf",
      "backend": "pdftotext",
      "pages_total": 9,
      "pages": [{"page": 1, "chars": 3120, "truncated": false, "text": "..."}],
      "warnings": []
    }

命令行::

    python3 tools/pdf_reader.py info   <pdf>
    python3 tools/pdf_reader.py read   <pdf> [--pages 1-3,8] [--clean] [--json]
    python3 tools/pdf_reader.py search <pdf> "关键词" [--context 60]
    python3 tools/pdf_reader.py outline <pdf>

约定的失败方式：参数/文件/后端问题一律 `ToolError` → stderr 一行 + exit 2，
不抛 traceback；这样上层 Agent Loop 能把 stderr 当作可读的 observation。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

PAGE_SEP = "\f"  # poppler 的页分隔符；缓存文件同样用它
MAX_PAGES = 400  # 防御性上限：超过就拒绝整本读，避免把上下文撑爆


class ToolError(RuntimeError):
    """工具级错误：可读、可上报，不需要 traceback。"""


# --------------------------------------------------------------------------- #
# 文本规范化
# --------------------------------------------------------------------------- #

_FULLWIDTH = {ord(c): ord(c) - 0xFEE0 for c in map(chr, range(0xFF01, 0xFF5F))}
_FULLWIDTH[0x3000] = 0x20  # 全角空格


def fold_width(text: str) -> str:
    """全角 ASCII → 半角。中文期刊 PDF 常把英文摘要渲染成全角，检索前要折半。"""
    return text.translate(_FULLWIDTH)


def clean_text(text: str) -> str:
    """轻度清理：统一换行、去行尾空白、压缩 3 个以上空行。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """折叠宽度 + 把任意空白压成一个空格，并记录归一化字符 → 原文下标。

    返回 (normalized, index_map)，index_map[i] 是 normalized[i] 在原文中的下标。
    用于检索：既能跨换行匹配中文短语，又能回到原文切 snippet。
    """

    source = fold_width(text)
    out: list[str] = []
    index_map: list[int] = []
    pending_space = False
    for i, ch in enumerate(source):
        if ch.isspace():
            if out:
                pending_space = True
            continue
        if pending_space:
            out.append(" ")
            index_map.append(max(i - 1, 0))
            pending_space = False
        out.append(ch)
        index_map.append(i)
    return "".join(out), index_map


def snippet(text: str, start: int, end: int, context: int) -> str:
    """切出原文片段并把空白压平，便于单行展示。"""

    lo = max(0, start - context)
    hi = min(len(text), end + context)
    piece = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return ("…" if lo > 0 else "") + piece + ("…" if hi < len(text) else "")


# --------------------------------------------------------------------------- #
# 后端：pymupdf（优先）/ pdftotext（回退）
# --------------------------------------------------------------------------- #


def _have_pymupdf() -> bool:
    try:  # pragma: no cover - 取决于环境
        import pymupdf  # noqa: F401  (PyMuPDF >= 1.24 的正式导入名)

        return True
    except Exception:
        try:
            import fitz  # noqa: F401  (旧名)

            return True
        except Exception:
            return False


def _import_pymupdf():
    try:
        import pymupdf as module  # type: ignore

        return module
    except Exception:
        import fitz as module  # type: ignore

        return module


def _have_pdftotext() -> bool:
    return shutil.which("pdftotext") is not None


def select_backend(preferred: str = "auto") -> str:
    if preferred not in {"auto", "pymupdf", "pdftotext"}:
        raise ToolError(f"未知后端 {preferred!r}，可选 auto/pymupdf/pdftotext")
    if preferred == "pymupdf":
        if not _have_pymupdf():
            raise ToolError("指定了 pymupdf 后端但未安装，请 `pip install pymupdf` 或改用 --backend pdftotext")
        return "pymupdf"
    if preferred == "pdftotext":
        if not _have_pdftotext():
            raise ToolError("系统缺少 pdftotext（poppler-utils），请安装或改用 --backend pymupdf")
        return "pdftotext"
    if _have_pymupdf():
        return "pymupdf"
    if _have_pdftotext():
        return "pdftotext"
    raise ToolError("既没有 PyMuPDF 也没有 pdftotext，无法读取 PDF")


def _extract_pdftotext(path: Path, layout: bool) -> tuple[list[str], dict[str, Any]]:
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd += [str(path), "-"]  # stdout
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except FileNotFoundError as exc:  # pragma: no cover - 已由 select_backend 挡住
        raise ToolError("找不到 pdftotext，请安装 poppler-utils") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"pdftotext 超时（>180s）：{path.name}") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ToolError(f"pdftotext 读取失败：{message[-1] if message else '未知错误'}")
    raw = proc.stdout.decode("utf-8", "replace")
    pages = raw.split(PAGE_SEP)
    if pages and not pages[-1].strip():
        pages.pop()  # 末页后的空片段
    return [clean_text(p) for p in pages], {"layout": layout}


def _extract_pymupdf(path: Path, layout: bool) -> tuple[list[str], dict[str, Any]]:
    pymupdf = _import_pymupdf()
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        raise ToolError(f"PyMuPDF 无法打开该 PDF：{exc}") from exc
    try:
        if document.needs_pass:
            raise ToolError("PDF 已加密，无法提取文本")
        mode = "text"
        pages = [clean_text(page.get_text(mode) or "") for page in document]
        meta = {
            "pdf_metadata": {k: v for k, v in (document.metadata or {}).items() if v},
            "pdf_version": getattr(document, "pdf_version", None) or document.metadata.get("format"),
        }
    finally:
        document.close()
    return pages, meta


def extract_pages(path: str | Path, backend: str = "auto", layout: bool = True) -> dict[str, Any]:
    """抽取整本的逐页文本（带缓存）。返回的 pages 为 `({"page": int, "text": str})`。"""

    pdf_path = Path(path).expanduser()
    if not pdf_path.exists():
        raise ToolError(f"文件不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ToolError(f"不是 PDF 文件：{pdf_path.name}")

    chosen = select_backend(backend)
    warnings: list[str] = []
    if chosen == "pymupdf":
        page_texts, meta = _extract_pymupdf(pdf_path, layout)
    else:
        page_texts, meta = _extract_pdftotext(pdf_path, layout)

    if len(page_texts) > MAX_PAGES:
        raise ToolError(
            f"文档共 {len(page_texts)} 页，超过单次读取上限 {MAX_PAGES} 页；请用 --pages 指定范围"
        )

    empty_pages = [i + 1 for i, t in enumerate(page_texts) if len(t) < 20]
    if len(empty_pages) == len(page_texts):
        warnings.append("所有页面都几乎没有文本：可能是扫描件或图片型 PDF，需要 OCR 才能阅读")
    elif empty_pages:
        warnings.append(f"以下页面几乎没有可提取文本（可能是图片）：{empty_pages}")

    return {
        "paper_id": pdf_path.stem,
        "path": str(pdf_path.resolve()),
        "filename": pdf_path.name,
        "backend": chosen,
        "layout": layout,
        "pages_total": len(page_texts),
        "page_texts": page_texts,
        "metadata": meta,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# 缓存（data/cache/<paper_id>.txt + .json）
# --------------------------------------------------------------------------- #


def _cache_paths(cache_dir: Path, paper_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{paper_id}.txt", cache_dir / f"{paper_id}.meta.json"


def load_cached(pdf_path: Path, cache_dir: Path | None, backend: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    text_file, meta_file = _cache_paths(cache_dir, pdf_path.stem)
    if not (text_file.exists() and meta_file.exists()):
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stat = pdf_path.stat()
    if meta.get("source_mtime") != stat.st_mtime or meta.get("source_size") != stat.st_size:
        return None  # 源文件变了，缓存作废
    if backend != "auto" and meta.get("backend") != backend:
        return None
    raw = text_file.read_text(encoding="utf-8")
    pages = raw.split(PAGE_SEP)
    if pages and not pages[-1].strip():
        pages.pop()
    return {
        "paper_id": pdf_path.stem,
        "path": str(pdf_path.resolve()),
        "filename": pdf_path.name,
        "backend": meta.get("backend"),
        "layout": meta.get("layout", True),
        "pages_total": len(pages),
        "page_texts": pages,
        "metadata": meta.get("metadata", {}),
        "warnings": [],
        "cached": True,
    }


def store_cache(document: dict[str, Any], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(document["path"])
    stat = pdf_path.stat()
    text_file, meta_file = _cache_paths(cache_dir, document["paper_id"])
    text_file.write_text(PAGE_SEP.join(document["page_texts"]), encoding="utf-8")
    meta_file.write_text(
        json.dumps(
            {
                "backend": document["backend"],
                "layout": document["layout"],
                "source_mtime": stat.st_mtime,
                "source_size": stat.st_size,
                "metadata": document.get("metadata", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def open_document(
    path: str | Path,
    *,
    backend: str = "auto",
    layout: bool = True,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> dict[str, Any]:
    """读取文档：先查缓存，未命中则抽取并写回缓存。"""

    pdf_path = Path(path).expanduser()
    if not pdf_path.exists():
        raise ToolError(f"文件不存在：{pdf_path}")
    if use_cache and cache_dir is not None:
        cached = load_cached(pdf_path, cache_dir, backend)
        if cached is not None:
            return cached
    document = extract_pages(pdf_path, backend=backend, layout=layout)
    if use_cache and cache_dir is not None:
        try:
            store_cache(document, cache_dir)
        except OSError:
            document.setdefault("warnings", []).append("缓存写入失败，已忽略（不影响本次结果）")
    return document


# --------------------------------------------------------------------------- #
# 页范围
# --------------------------------------------------------------------------- #


def parse_pages(spec: str | None, pages_total: int) -> list[int]:
    """解析 `1-3,8,10-12` 为 1-based 页码列表（去重、升序、越界报错）。"""

    if not spec or not spec.strip():
        return list(range(1, pages_total + 1))
    picked: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+))?", chunk)
        if not match:
            raise ToolError(f"无法解析页范围片段 {chunk!r}，正确写法如 `1-3,8`")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            raise ToolError(f"页范围非法：{chunk!r}")
        if end > pages_total:
            raise ToolError(f"页范围 {chunk!r} 超出总页数 {pages_total}")
        picked.update(range(start, end + 1))
    if not picked:
        raise ToolError(f"页范围为空：{spec!r}")
    return sorted(picked)


# --------------------------------------------------------------------------- #
# 期刊版式噪声（页眉页脚）
# --------------------------------------------------------------------------- #


def strip_boilerplate(page_texts: Sequence[str], min_ratio: float = 0.5) -> tuple[list[str], list[str]]:
    """删掉在多页重复出现的短行（期刊页眉页码等）。返回 (新页面, 被删行样本)。

    只对"短行且出现在 ≥50% 页面"的行动手，避免误删正文里的重复术语。
    """

    if len(page_texts) < 3:
        return list(page_texts), []
    counts: dict[str, int] = {}
    for text in page_texts:
        seen = {
            line.strip()
            for line in text.split("\n")
            if 0 < len(line.strip()) <= 40 and not re.search(r"\d{3,}", line)
        }
        for line in seen:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(2, int(len(page_texts) * min_ratio))
    noisy = {line for line, n in counts.items() if n >= threshold}
    if not noisy:
        return list(page_texts), []
    cleaned = []
    for text in page_texts:
        kept = [line for line in text.split("\n") if line.strip() not in noisy]
        cleaned.append(clean_text("\n".join(kept)))
    return cleaned, sorted(noisy)[:10]


# --------------------------------------------------------------------------- #
# 大纲
# --------------------------------------------------------------------------- #

_HEADING_PATTERNS = (
    re.compile(r"^\s*([一二三四五六七八九十]+)\s*[、.．]\s*(\S.{0,30})$"),
    re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\s*[、.．]?\s+?(\S.{0,30})$"),
    re.compile(r"^\s*(摘\s*要|关键词|引\s*言|绪\s*论|结\s*语|结\s*论|参考文献|Abstract|Introduction|Conclusion|References)\s*$"),
)


def heuristic_outline(page_texts: Sequence[str], max_items: int = 40) -> list[dict[str, Any]]:
    """没有 PDF 书签时，用版式猜测候选标题，供 Agent 决定先读哪几页。"""

    items: list[dict[str, Any]] = []
    for number, text in enumerate(page_texts, start=1):
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) > 40:
                continue
            for pattern in _HEADING_PATTERNS:
                match = pattern.match(line)
                if match:
                    items.append({"page": number, "heading": re.sub(r"\s+", "", line)})
                    break
            if len(items) >= max_items:
                return items
    # 去重（同一标题在页眉重复出现）
    unique: list[dict[str, Any]] = []
    for item in items:
        if item["heading"] not in {u["heading"] for u in unique}:
            unique.append(item)
    return unique


def read_outline(path: str | Path, use_cache: bool = True) -> dict[str, Any]:
    document = open_document(path, use_cache=use_cache)
    bookmarks: list[dict[str, Any]] = []
    if document["backend"] == "pymupdf":
        pymupdf = _import_pymupdf()
        try:
            handle = pymupdf.open(document["path"])
            try:
                bookmarks = [
                    {"level": level, "heading": title.strip(), "page": page}
                    for level, title, page in handle.get_toc()
                ]
            finally:
                handle.close()
        except Exception:
            bookmarks = []
    return {
        "paper_id": document["paper_id"],
        "pages_total": document["pages_total"],
        "source": "pdf-bookmarks" if bookmarks else "heuristic",
        "outline": bookmarks or heuristic_outline(document["page_texts"]),
    }


# --------------------------------------------------------------------------- #
# 文档内检索
# --------------------------------------------------------------------------- #


def search_in_document(
    path: str | Path,
    query: str,
    *,
    context: int = 60,
    max_hits: int = 20,
    use_cache: bool = True,
) -> dict[str, Any]:
    """在文档里找关键词，返回命中的页码与原文片段（页码可直接用于 [p.N]）。"""

    if not query.strip():
        raise ToolError("检索词不能为空")
    document = open_document(path, use_cache=use_cache)
    needle = re.sub(r"\s+", " ", normalize_with_map(query)[0])
    if not needle:
        raise ToolError("检索词规范化后为空")

    hits: list[dict[str, Any]] = []
    pages_scanned = 0
    for page_number, text in enumerate(document["page_texts"], start=1):
        pages_scanned += 1
        normalized, index_map = normalize_with_map(text)
        start = normalized.find(needle)
        while start != -1:
            orig_start = index_map[start]
            orig_end = index_map[min(start + len(needle) - 1, len(index_map) - 1)] + 1
            hits.append(
                {
                    "page": page_number,
                    "snippet": snippet(text, orig_start, orig_end, context),
                    "offset": orig_start,
                }
            )
            if len(hits) >= max_hits:
                break
            start = normalized.find(needle, start + len(needle))
        if len(hits) >= max_hits:
            break

    return {
        "paper_id": document["paper_id"],
        "query": query,
        "pages_scanned": pages_scanned,
        "pages_total": document["pages_total"],
        "hit_count": len(hits),
        "hits": hits,
        "truncated": len(hits) >= max_hits,
    }


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


def render_pages(document: dict[str, Any], pages: Iterable[int], max_chars: int = 0) -> str:
    """渲染成人/Agent 可读的 Markdown，页码保留为 `[Page N]`。"""

    lines = [
        f"# {document['paper_id']}",
        f"<!-- backend={document['backend']} pages_total={document['pages_total']} -->",
        "",
    ]
    for page_number in pages:
        text = document["page_texts"][page_number - 1]
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n…（本页已截断，原长 {len(text)} 字符，可换更小页范围重读）"
        lines += [f"[Page {page_number}]", text, ""]
    for warning in document.get("warnings") or []:
        lines.append(f"> ⚠️ {warning}")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(payload: Any, as_json: bool, renderer=None) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif renderer is not None:
        print(renderer(payload), end="")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_reader.py",
        description="Tool 1：PDF 文本 + 页码提取（保留 [p.N] 引用所需的页码）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("pdf", help="PDF 路径")
        p.add_argument("--backend", default="auto", choices=["auto", "pymupdf", "pdftotext"])
        p.add_argument("--no-cache", action="store_true", help="不用缓存，强制重新抽取")
        p.add_argument("--no-layout", action="store_true", help="不用 pdftotext 的 -layout 版式模式")

    info = sub.add_parser("info", help="元信息：页数、是否有文本层、抽取后端")
    add_common(info)
    info.add_argument("--json", action="store_true")

    read = sub.add_parser("read", help="按页读取文本（默认全文）")
    add_common(read)
    read.add_argument("--pages", help="页范围，如 `1-3,8`；默认全文")
    read.add_argument("--max-chars", type=int, default=0, help="每页最多输出多少字符（0=不限）")
    read.add_argument("--clean", action="store_true", help="删除多页重复的页眉页脚短行")
    read.add_argument("--json", action="store_true")

    search = sub.add_parser("search", help="文档内检索，返回页码 + 片段")
    add_common(search)
    search.add_argument("query", help="检索词（支持中文短语，跨换行匹配）")
    search.add_argument("--context", type=int, default=60, help="片段前后保留的字符数")
    search.add_argument("--max-hits", type=int, default=20)
    search.add_argument("--json", action="store_true")

    outline = sub.add_parser("outline", help="章节大纲：优先 PDF 书签，否则按版式猜测")
    add_common(outline)
    outline.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = None if args.no_cache else DEFAULT_CACHE_DIR
    layout = not args.no_layout
    try:
        if args.command == "info":
            document = open_document(args.pdf, backend=args.backend, layout=layout, cache_dir=cache_dir)
            payload = {
                "paper_id": document["paper_id"],
                "path": document["path"],
                "backend": document["backend"],
                "pages_total": document["pages_total"],
                "chars_per_page": [len(t) for t in document["page_texts"]],
                "has_text_layer": sum(len(t) for t in document["page_texts"]) > 200,
                "metadata": document.get("metadata", {}),
                "warnings": document.get("warnings", []),
                "cached": document.get("cached", False),
            }
            _emit(payload, args.json)
        elif args.command == "read":
            document = open_document(args.pdf, backend=args.backend, layout=layout, cache_dir=cache_dir)
            if args.clean:
                page_texts, removed = strip_boilerplate(document["page_texts"])
                document = {**document, "page_texts": page_texts, "removed_lines": removed}
            pages = parse_pages(args.pages, document["pages_total"])
            payload = {
                "paper_id": document["paper_id"],
                "backend": document["backend"],
                "pages_total": document["pages_total"],
                "pages_read": pages,
                "truncated": any(args.max_chars and len(document["page_texts"][p - 1]) > args.max_chars for p in pages),
                "text": "\n\n".join(
                    f"[Page {p}]\n{document['page_texts'][p - 1]}" for p in pages
                ),
                "warnings": document.get("warnings", []),
            }
            if args.clean:
                payload["removed_lines"] = document.get("removed_lines", [])
            _emit(
                payload,
                args.json,
                renderer=lambda p: render_pages(document, pages, args.max_chars),
            )
        elif args.command == "search":
            payload = search_in_document(
                args.pdf, args.query, context=args.context, max_hits=args.max_hits,
                use_cache=not args.no_cache,
            )
            _emit(payload, args.json)
        elif args.command == "outline":
            payload = read_outline(args.pdf, use_cache=not args.no_cache)
            _emit(payload, args.json)
        else:  # pragma: no cover - argparse 已限制
            raise ToolError(f"未知命令 {args.command}")
    except ToolError as exc:
        print(f"pdf_reader: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
