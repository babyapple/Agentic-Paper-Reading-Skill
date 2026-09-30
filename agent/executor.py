#!/usr/bin/env python3
"""Executor —— 把 Agent 选出的 action 落到第三阶段那三个真实 Tool 上。

三条硬规则：

1. **Action 与 Tool 一一对应**：`READ → read_pdf`、`SEARCH → search_paper`、
   `ANALYZE → analyze_table`；Executor 不发明新工具，Agent 也不能绕过它直接读文件。
2. **失败也是 observation**：`ToolError` 被翻译成 `ok=False` + 可读理由交回 Agent，
   不抛给主循环——这正是第三阶段"exit 2 + 单行 stderr"契约在 Agent 侧的接法。
3. **每次调用都记账**：耗时、页数、命中数写进 `Observation`，最终进 trace，供第八阶段算指标。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import paper_search, pdf_reader, table_analyzer  # noqa: E402
from tools.pdf_reader import ToolError  # noqa: E402

from agent.state import Observation, ReadingState, compress_pages  # noqa: E402

ANALYZE_OPERATIONS = {"extract", "describe", "rank", "diff", "compare", "group", "plot"}
PROMPT_TEXT_LIMIT = 3000


def _short(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"…（截断，原长 {len(text)}）"


class Executor:
    def __init__(
        self,
        paper_path: str | Path,
        *,
        papers_dir: str | Path | None = None,
        top_hits: int = 3,
    ) -> None:
        self.paper_path = Path(paper_path).expanduser()
        self.papers_dir = Path(papers_dir).expanduser() if papers_dir else REPO_ROOT / "paper"
        self.top_hits = top_hits
        self._document: dict[str, Any] | None = None

    # --------------------------------------------------------------- 基础 --
    @property
    def document(self) -> dict[str, Any]:
        if self._document is None:
            self._document = pdf_reader.open_document(self.paper_path)
        return self._document

    def bootstrap(self, state: ReadingState) -> Observation:
        """开局一次性拿到元信息与大纲——这两个调用也计入 tool_calls，指标不注水。"""

        started = time.time()
        try:
            document = self.document
            outline = pdf_reader.read_outline(self.paper_path)
            state.outline = outline.get("outline", [])
            state.outline_source = outline.get("source")
            pages_total = document["pages_total"]
            state.pages_total = pages_total
            headings = "；".join(
                f"{item['heading']}(p{item['page']})" for item in state.outline[:6] if item.get("heading")
            )
            summary = (
                f"read_pdf(info/outline) → {state.paper_id}：{pages_total} 页，"
                f"大纲来源 {state.outline_source}"
                + (f"；{'；'.join(headings.split('；')[:3])}" if headings else "")
            )
            prompt_text = (
                f"论文：{state.paper_id}\n总页数：{pages_total}\n"
                f"大纲（{state.outline_source}）：\n"
                + "\n".join(f"- p{item['page']}: {item['heading']}" for item in state.outline[:12])
            )
            warnings = document.get("warnings") or []
            if warnings:
                summary += f"；⚠️ {'；'.join(warnings)}"
                prompt_text += "\n⚠️ " + "；".join(warnings)
            return Observation(
                step=state.steps,
                action="BOOTSTRAP",
                tool="read_pdf",
                args={"action": "info+outline", "pdf": str(self.paper_path)},
                ok=True,
                summary=summary,
                duration_ms=int((time.time() - started) * 1000),
                detail={"prompt_text": prompt_text},
            )
        except ToolError as exc:
            return self._failure(state, "BOOTSTRAP", "read_pdf", {"pdf": str(self.paper_path)}, str(exc), started)

    @staticmethod
    def _failure(
        state: ReadingState,
        action: str,
        tool: str,
        args: dict[str, Any],
        message: str,
        started: float,
    ) -> Observation:
        return Observation(
            step=state.steps,
            action=action,
            tool=tool,
            args=args,
            ok=False,
            summary=f"{tool} 失败：{message}",
            duration_ms=int((time.time() - started) * 1000),
            error=message,
            detail={"prompt_text": f"工具返回错误：{message}"},
        )

    # ------------------------------------------------------------- 分发 --
    def execute(self, action: dict[str, Any], state: ReadingState) -> Observation:
        name = str(action.get("action", "")).upper()
        if name == "READ":
            return self.do_read(action, state)
        if name == "SEARCH":
            return self.do_search(action, state)
        if name == "ANALYZE":
            return self.do_analyze(action, state)
        return Observation(
            step=state.steps,
            action=name or "?",
            tool="-",
            args=action,
            ok=False,
            summary=f"未知 action：{name!r}；只允许 READ / SEARCH / ANALYZE / FINISH",
            error=f"unknown action {name!r}",
            detail={"prompt_text": "未知 action，请从 READ/SEARCH/ANALYZE/FINISH 里选一个。"},
        )

    # --------------------------------------------------------------- READ --
    def do_read(self, action: dict[str, Any], state: ReadingState) -> Observation:
        args = {"pages": action.get("pages", "")}
        started = time.time()
        try:
            pages = self.resolve_read_pages(action, state)
            if not pages:
                return self._failure(
                    state, "READ", "read_pdf", args, "没有可读的页（可能都已读过或页预算用尽）", started
                )
            document = self.document
            texts = {page: document["page_texts"][page - 1] for page in pages}
            state.mark_read(pages, texts)
            limit = state.budget.max_chars_per_page
            chars = sum(len(text) for text in texts.values())
            truncated = any(len(text) > limit for text in texts.values())
            body = "\n\n".join(
                f"[Page {page}]\n{_short(texts[page], limit)}" for page in pages
            )
            summary = (
                f"read_pdf(pages={compress_pages(pages)}) → {len(pages)} 页 / {chars} 字符；"
                f"已读 {state.pages_read_count}/{state.pages_total} 页"
                + ("（有页被截断）" if truncated else "")
            )
            return Observation(
                step=state.steps,
                action="READ",
                tool="read_pdf",
                args={"pages": compress_pages(pages)},
                ok=True,
                summary=summary,
                duration_ms=int((time.time() - started) * 1000),
                detail={
                    "pages": pages,
                    "chars": chars,
                    "truncated": truncated,
                    "prompt_text": _short(body, PROMPT_TEXT_LIMIT),
                },
            )
        except ToolError as exc:
            return self._failure(state, "READ", "read_pdf", args, str(exc), started)

    def resolve_read_pages(self, action: dict[str, Any], state: ReadingState) -> list[int]:
        """pages 省略时读"下一段没读过的页"——默认行为对模型友好，也避免空转。"""

        raw = str(action.get("pages", "") or "").strip()
        if not raw:
            return state.next_unread_pages(count=2)
        return pdf_reader.parse_pages(raw, state.pages_total)

    # ------------------------------------------------------------- SEARCH --
    def do_search(self, action: dict[str, Any], state: ReadingState) -> Observation:
        query = str(action.get("query", "")).strip()
        scope = str(action.get("scope", "local") or "local").lower()
        args = {"query": query, "scope": scope}
        started = time.time()
        if not query:
            return self._failure(state, "SEARCH", "search_paper", args, "query 不能为空", started)
        try:
            if scope in {"local", "corpus"}:
                payload = paper_search.search_local(
                    query, papers_dir=self.papers_dir, top=self.top_hits, mode="any"
                )
                lines = []
                for result in payload["results"][: self.top_hits]:
                    lines.append(f"- {result['title'] or result['paper_id']}（{result['paper_id']}）")
                    for hit in result["hits"][:2]:
                        lines.append(f"  [p.{hit['page']}] {' / '.join(hit['snippets'])[:180]}")
                summary = (
                    f"search_paper(local, {query!r}) → 命中 {payload['result_count']} 篇"
                    + (f"；最佳 {payload['results'][0]['paper_id']}" if payload["results"] else "")
                )
                prompt_text = f"本地语料检索 {query!r}：\n" + ("\n".join(lines) if lines else "（无命中）")
            else:
                sources = ["arxiv", "crossref"] if scope in {"web", "all"} else [scope]
                payload = paper_search.search_web(query, sources=sources, max_results=self.top_hits)
                lines = [f"- [{item.get('source')}] {item.get('title')} ({item.get('year')})" for item in payload["results"]]
                for warning in payload.get("warnings") or []:
                    lines.append(f"  ⚠️ {warning}")
                summary = f"search_paper({scope}, {query!r}) → {payload['result_count']} 条"
                prompt_text = f"联网检索 {query!r}：\n" + ("\n".join(lines) if lines else "（无结果）")
            return Observation(
                step=state.steps,
                action="SEARCH",
                tool="search_paper",
                args=args,
                ok=True,
                summary=summary,
                duration_ms=int((time.time() - started) * 1000),
                detail={"prompt_text": _short(prompt_text, PROMPT_TEXT_LIMIT), "result_count": payload["result_count"]},
            )
        except ToolError as exc:
            return self._failure(state, "SEARCH", "search_paper", args, str(exc), started)

    # ------------------------------------------------------------ ANALYZE --
    def do_analyze(self, action: dict[str, Any], state: ReadingState) -> Observation:
        operation = str(action.get("operation", "") or "").lower()
        args = {k: v for k, v in action.items() if k != "action"}
        started = time.time()
        if operation not in ANALYZE_OPERATIONS:
            return self._failure(
                state,
                "ANALYZE",
                "analyze_table",
                args,
                f"operation 必须是 {sorted(ANALYZE_OPERATIONS)} 之一，收到 {operation!r}",
                started,
            )
        try:
            if operation == "extract":
                return self._analyze_extract(state, started)
            table = self.resolve_table(action, state)
            payload = self.run_table_op(operation, table, action)
            rendered = table_analyzer.render_result(payload)
            summary = self._analyze_summary(operation, payload)
            return Observation(
                step=state.steps,
                action="ANALYZE",
                tool="analyze_table",
                args=args,
                ok=True,
                summary=summary,
                duration_ms=int((time.time() - started) * 1000),
                detail={"prompt_text": _short(rendered, PROMPT_TEXT_LIMIT), "payload": payload, "table": table},
            )
        except ToolError as exc:
            return self._failure(state, "ANALYZE", "analyze_table", args, str(exc), started)

    def _analyze_extract(self, state: ReadingState, started: float) -> Observation:
        text = "\n".join(state.page_texts[page] for page in sorted(state.page_texts))
        if not text.strip():
            raise ToolError("还没有读过任何正文，先用 READ 读几页再抽表")
        blocks = table_analyzer.extract_tables(text)
        lines = []
        for index, block in enumerate(blocks, start=1):
            ref = f"table:{index}"
            markdown = table_analyzer.render_table(block["headers"], block["rows"])
            state.tables[ref] = markdown
            lines.append(f"{ref}（{block['kind']}，{len(block['rows'])} 行 × {len(block['headers'])} 列）："
                         f"{', '.join(str(h) for h in block['headers'][:6])}")
        summary = f"analyze_table(extract) → 从已读正文里抽到 {len(blocks)} 个候选表"
        prompt_text = "候选表格：\n" + ("\n".join(lines) if lines else "（已读正文里没有版式对齐的表格）")
        return Observation(
            step=state.steps,
            action="ANALYZE",
            tool="analyze_table",
            args={"operation": "extract"},
            ok=True,
            summary=summary,
            duration_ms=int((time.time() - started) * 1000),
            detail={"prompt_text": prompt_text, "table_count": len(blocks), "tables": sorted(state.tables)},
        )

    def resolve_table(self, action: dict[str, Any], state: ReadingState) -> tuple[list[str], list[list[str]]]:
        ref = str(action.get("table", "") or "").strip()
        if not ref:
            raise ToolError("table 不能为空：可以是 `table:N`（extract 的候选）或表格文件路径")
        if ref in state.tables:
            headers, rows = table_analyzer.parse_markdown(state.tables[ref])
            if not headers:
                raise ToolError(f"{ref} 不是可解析的表格")
            return headers, rows
        if ref.startswith("table:"):
            raise ToolError(f"{ref} 不存在；已抽取的候选表：{sorted(state.tables) or '（还没有，先 ANALYZE extract）'}")
        return table_analyzer.load_table(ref)

    @staticmethod
    def run_table_op(operation: str, table: tuple[list[str], list[list[str]]], action: dict[str, Any]) -> dict[str, Any]:
        headers, rows = table
        columns = action.get("columns")
        column_list = [c.strip() for c in str(columns).split(",")] if columns else None
        lower = [c.strip() for c in str(action.get("lower_is_better", "")).split(",") if c.strip()]
        if operation == "describe":
            return table_analyzer.op_describe(headers, rows, column_list)
        if operation == "rank":
            return table_analyzer.op_rank(
                headers, rows, str(action.get("by", "")), not action.get("asc", False),
                int(action.get("top") or 0) or None, action.get("label"),
            )
        if operation == "diff":
            return table_analyzer.op_diff(headers, rows, str(action.get("a", "")), str(action.get("b", "")), action.get("label"))
        if operation == "compare":
            return table_analyzer.resolve_compare(
                headers, rows, str(action.get("baseline", "")), column_list, action.get("label"), lower
            )
        if operation == "group":
            return table_analyzer.op_group(
                headers, rows, str(action.get("by", "")), str(action.get("value", "")), str(action.get("agg", "mean"))
            )
        if operation == "plot":
            out = str(action.get("out", "") or (REPO_ROOT / "tools" / "figures" / "agent_plot.png"))
            return table_analyzer.op_plot(headers, rows, str(action.get("x", "")), str(action.get("y", "")), str(action.get("kind", "bar")), out)
        raise ToolError(f"operation 不支持：{operation}")

    @staticmethod
    def _analyze_summary(operation: str, payload: dict[str, Any]) -> str:
        if operation == "describe":
            return f"analyze_table(describe) → {len(payload.get('stats', []))} 列的均值/极值/中位数"
        if operation == "rank":
            top = (payload.get("ranking") or [{}])[0]
            return f"analyze_table(rank, {payload.get('column')}) → 第 1 名 {top.get('label')} = {top.get('value')}"
        if operation == "diff":
            return (
                f"analyze_table(diff, {payload.get('a')}−{payload.get('b')}) → 平均差 {payload.get('mean_diff')}，"
                f"{payload.get('a')} 更高 {payload.get('a_higher_count')} 行"
            )
        if operation in {"compare", "compare_rows"}:
            key = payload.get("baseline") or payload.get("baseline_row")
            return f"analyze_table(compare, baseline={key}) → {len(payload.get('comparisons') or payload.get('rows') or [])} 项对比"
        if operation == "group":
            return f"analyze_table(group, by {payload.get('by')}) → {len(payload.get('groups', []))} 组"
        return f"analyze_table({operation}) → 完成"
