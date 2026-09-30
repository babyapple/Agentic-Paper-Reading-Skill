#!/usr/bin/env python3
"""`ReadingState` —— Agent 的"当前信息状态"，也是它决定下一步的唯一依据。

设计要点（第四阶段）：

- **只有行动与结果，不留隐藏推理**：状态里存的是"读了哪几页、搜了什么、算了哪张表"，
  以及工具的返回摘要；模型内部的思考过程不进入状态、也不进 trace。
- **预算是状态的一部分**：步数 / 搜索次数 / 阅读页数 / 工具调用数都记在这里，
  超预算时不抛异常，而是把一条可读的拒绝理由作为 observation 交回给 Agent。
- **指标从状态里取**：`Tool Calls`、`Token Cost`、`Latency`、`pages_read` 这些
  第五/八阶段要用的指标，全部由本对象记账，不另外埋点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def compress_pages(pages: Iterable[int]) -> str:
    """`[1,2,3,5,8,9]` → `1-3,5,8-9`；页码列表在人/模型看来都太长。"""

    ordered = sorted(set(int(p) for p in pages))
    if not ordered:
        return ""
    parts: list[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = page
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


@dataclass
class Budget:
    """Agent 预算 —— 第五阶段：不让 Agent"无限乱跑"。

    三类约束叠在一起：

    - **动作预算**：步数 / 工具调用 / 搜索次数 / 阅读页数（默认值直接来自路线图；
      `--max-pages` 按页数算，且只算"新页"）；
    - **成本预算**：token 总量（`max_total_tokens`，0 = 不限）——上下文与输出一起算；
    - **进展预算**：连续若干步没有拿到新信息就判定空转（`max_no_progress_steps`）。
    """

    max_steps: int = 8
    max_searches: int = 3
    max_pages: int = 20
    max_tool_calls: int = 12
    max_chars_per_page: int = 6000
    max_repeat_blocked: int = 2
    max_total_tokens: int = 60000
    max_no_progress_steps: int = 3

    @property
    def tokens_total(self) -> int:
        return self.max_total_tokens

    def violation(self, action: dict[str, Any], state: "ReadingState", pages: Sequence[int] = ()) -> str | None:
        """返回拒绝理由；返回 None 表示这个 action 允许执行。

        注意：`READ` 的页数要在 `pages` 里传进来（先解析页范围再判预算），
        以免出现"已读 19 页，还要一次读 30 页"这种绕过。
        """

        name = str(action.get("action", "")).upper()
        tokens = self.token_exhausted(state)
        if tokens:
            return tokens
        if state.steps >= self.max_steps:
            return f"步数已到上限 {self.max_steps}，必须 FINISH"
        if state.tool_calls >= self.max_tool_calls:
            return f"工具调用已到上限 {self.max_tool_calls}，必须 FINISH"
        if name == "SEARCH" and state.searches >= self.max_searches:
            return f"检索次数已到上限 {self.max_searches}，改用 READ 或直接 FINISH"
        if name == "READ":
            new_pages = [p for p in pages if p not in state.read_pages]
            if state.pages_read_count + len(new_pages) > self.max_pages:
                return (
                    f"阅读页数预算只剩 {self.max_pages - state.pages_read_count} 页，"
                    f"本次请求 {len(pages)} 页（其中新页 {len(new_pages)} 页）超预算"
                )
        return None

    def token_exhausted(self, state: "ReadingState") -> str | None:
        """token 预算检查：Planner 调用**之前**就要查，否则会先花掉再发现超了。"""

        if not self.max_total_tokens:
            return None
        spent = state.input_tokens + state.output_tokens
        if spent >= self.max_total_tokens:
            return f"token 预算耗尽（{spent}/{self.max_total_tokens}），必须 FINISH"
        return None



@dataclass
class Observation:
    """一次工具调用的结果（给 Agent 看的那一份，不含隐藏推理）。"""

    step: int
    action: str
    tool: str
    args: dict[str, Any]
    ok: bool
    summary: str
    duration_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class ReadingState:
    paper_id: str
    paper_path: str
    pages_total: int
    task: str
    budget: Budget = field(default_factory=Budget)

    # 阅读记账
    read_pages: set[int] = field(default_factory=set)
    page_texts: dict[int, str] = field(default_factory=dict)
    read_chars: int = 0
    outline: list[dict[str, Any]] = field(default_factory=list)
    outline_source: str | None = None

    # 工具与行动记账
    steps: int = 0
    tool_calls: int = 0
    searches: int = 0
    observations: list[Observation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)  # "table:1" → Markdown 文本
    last_observation: str = "（还没有任何工具调用）"

    # LLM 记账（Token Cost / Latency 指标的来源）
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    llm_latency_ms: int = 0

    # 收尾
    finish_reason: str | None = None
    stop_cause: str | None = None          # 结构化停止原因，供第八/九阶段做统计
    coverage: dict[str, Any] | None = None  # 收尾时的覆盖度快照
    critic_rounds: list[dict[str, Any]] = field(default_factory=list)
    no_progress_streak: int = 0
    answer: str | None = None
    report: str | None = None

    # ---------------------------------------------------------------- 记账 --
    @property
    def pages_read_count(self) -> int:
        return len(self.read_pages)

    @property
    def pages_read_spec(self) -> str:
        return compress_pages(self.read_pages)

    def unread_pages(self) -> list[int]:
        return [p for p in range(1, self.pages_total + 1) if p not in self.read_pages]

    def next_unread_pages(self, count: int = 2, cap: int | None = None) -> list[int]:
        remaining = self.budget.max_pages - self.pages_read_count if cap is None else cap
        if remaining <= 0:
            return []
        return self.unread_pages()[: max(0, min(count, remaining))]

    def record_tool(self, observation: Observation) -> None:
        """记账。被系统拒绝的 action（`tool == "-"`）不算工具调用，也不吃预算。

        失败的**真实调用**（例如页范围越界）仍然记一次工具调用——它确实发生了、
        也确实花了时间；但失败的检索不计入检索预算，因为什么都没取回来。
        """

        if observation.tool not in (None, "", "-"):
            self.tool_calls += 1
            if observation.action == "SEARCH" and observation.ok:
                self.searches += 1
        self.observations.append(observation)
        self.last_observation = observation.summary
        if not observation.ok and observation.error:
            self.errors.append(observation.error)

    def record_llm(self, input_tokens: int | None, output_tokens: int | None, latency_ms: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        self.llm_latency_ms += latency_ms

    def mark_read(self, pages: Sequence[int], texts: dict[int, str]) -> None:
        for page in pages:
            if page not in self.read_pages:
                self.read_pages.add(page)
                self.read_chars += len(texts.get(page, ""))
            self.page_texts[page] = texts.get(page, self.page_texts.get(page, ""))

    # ------------------------------------------------------------- 给模型看 --
    def progress_line(self) -> str:
        """一行进度摘要：模型据此判断"还差什么"，也是停止条件的判据来源。"""

        outline = ""
        if self.outline:
            readable = [
                f"{item.get('heading', '')}(p{item.get('page')})"
                for item in self.outline[:6]
                if item.get("heading")
            ]
            if readable:
                outline = "；大纲=" + " > ".join(readable)
        budget = self.budget
        token_note = ""
        if budget.max_total_tokens:
            spent = self.input_tokens + self.output_tokens
            token_note = f"tokens={spent}/{budget.max_total_tokens}；"
        return (
            f"已读 {self.pages_read_count}/{self.pages_total} 页 (p{self.pages_read_spec or '—'})，"
            f"共 {self.read_chars} 字符；搜索 {self.searches} 次；工具调用 {self.tool_calls} 次；"
            f"步数 {self.steps}/{budget.max_steps}；{token_note}"
            f"预算剩余 pages={budget.max_pages - self.pages_read_count}, "
            f"search={budget.max_searches - self.searches}{outline}"
        )

    def context_excerpt(self, max_chars: int = 12000) -> str:
        """把已读正文按页码拼成带 `[p.N]` 的引用块，供写报告时引用。"""

        chunks: list[str] = []
        used = 0
        for page in sorted(self.page_texts):
            text = self.page_texts[page]
            if not text:
                continue
            piece = text if used + len(text) <= max_chars else text[: max(0, max_chars - used)]
            if not piece:
                break
            chunks.append(f"[p.{page}]\n{piece}")
            used += len(piece)
            if used >= max_chars:
                break
        return "\n\n".join(chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_path": self.paper_path,
            "pages_total": self.pages_total,
            "task": self.task,
            "budget": {
                "max_steps": self.budget.max_steps,
                "max_searches": self.budget.max_searches,
                "max_pages": self.budget.max_pages,
                "max_tool_calls": self.budget.max_tool_calls,
                "max_total_tokens": self.budget.max_total_tokens,
                "max_no_progress_steps": self.budget.max_no_progress_steps,
            },
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "searches": self.searches,
            "pages_read": self.pages_read_spec,
            "pages_read_list": sorted(self.read_pages),
            "read_chars": self.read_chars,
            "tables_found": sorted(self.tables),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "llm_latency_ms": self.llm_latency_ms,
            "finish_reason": self.finish_reason,
            "stop_cause": self.stop_cause,
            "no_progress_streak": self.no_progress_streak,
            "coverage": self.coverage,
            "critic_rounds": self.critic_rounds,
            "errors": self.errors,
            "trace": [obs.to_dict() for obs in self.observations],
            "answer": self.answer,
            "report": self.report,
        }
