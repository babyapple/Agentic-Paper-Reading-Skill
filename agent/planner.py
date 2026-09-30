#!/usr/bin/env python3
"""Planner —— 决定"下一步做什么"，这是第四阶段真正加入的自主性。

两种 Planner 共用同一个 action 契约与同一份状态：

- `LlmPlanner`：把 Skill 的工作流 + action 契约 + 当前进度交给模型，换回**一个** action JSON。
  模型只被要求输出 action，不要求暴露思维过程（系统只需要行动、工具调用与返回结果）。
- `HeuristicPlanner`：不联网的确定性策略（读开头 → 按大纲补章节 → 收尾），
  用于没 API key 时跑通整条循环、以及后续第七阶段当"固定工作流"参照的雏形。

解析这一层刻意写得宽：真实模型会加 ```json 围栏、会写解释、会返回 tool_calls 形态。
解析失败不是崩溃，而是一条可读的反馈，交给主循环重试。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agent.llm import ChatResult, ScriptedClient  # noqa: F401  (ScriptedClient 供类型注释与测试)
from agent.state import ReadingState

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "paper-reading" / "SKILL.md"
TEMPLATE_PATH = REPO_ROOT / "skills" / "paper-reading" / "references" / "review_template.md"

ALLOWED_ACTIONS = ("READ", "SEARCH", "ANALYZE", "FINISH")
SEARCH_SCOPES = ("local", "web", "arxiv", "crossref", "all")
ANALYZE_OPERATIONS = ("extract", "describe", "rank", "diff", "compare", "group", "plot")
TOOL_NAME_TO_ACTION = {"read_pdf": "READ", "search_paper": "SEARCH", "analyze_table": "ANALYZE"}

ACTION_CONTRACT = """\
## 动作空间（每步只选一个）

```json
{"action": "READ",    "pages": "1-3",  "reason": "可选，一句话"}
{"action": "SEARCH",  "query": "关键词", "scope": "local"}
{"action": "ANALYZE", "operation": "extract"}
{"action": "ANALYZE", "operation": "rank|describe|diff|compare|group",
                      "table": "table:1", "by": "列名", "a": "列", "b": "列",
                      "baseline": "baseline 行名或列名", "label": "行名列",
                      "lower_is_better": "Cost,Latency"}
{"action": "FINISH",  "reason": "为什么可以收尾"}
```

- `READ`：读论文本身，`pages` 省略时读下一段没读过的页。页码一律用 PDF 页码（1 起）。
- `SEARCH`：本地语料默认 `local`（离线）；论文外的背景概念才用 `web`。
- `ANALYZE`：`extract` 先从已读正文里抽候选表格（得到 `table:1`、`table:2`），
  其余 operation 必须给 `table`（`table:N` 或表格文件路径）。
- `FINISH`：认为信息已足够写报告时收尾。预算耗尽会被系统强制收尾。"""

HARD_RULES = """\
## 硬规则

1. 每步**只输出一个 JSON 对象**，不要输出解释文字、不要输出推理过程（系统只记录 action 与工具结果）。
2. `README`/`SKILL.md` 要求的一切事实都要有页码；没读到就再 `READ`，不要猜。
3. 不要重复已经做过的 action：看"已做过的动作"，换页、换词或直接 `FINISH`。
4. 论文里没有实验/消融时，不要为了补齐条目去虚构或去搜不相关的东西。"""


# --------------------------------------------------------------------------- #
# Prompt 组装
# --------------------------------------------------------------------------- #


def load_skill_text(path: Path | None = None) -> str:
    """Skill 的渐进式披露：循环启动时只加载 SKILL.md 主体，references 按需再读。"""

    target = path or SKILL_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return ""
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S).strip()


def build_system_prompt(skill_text: str | None = None, extra_rules: str = "") -> str:
    skill = load_skill_text() if skill_text is None else skill_text
    return "\n\n".join(
        part
        for part in [
            "你是一个论文阅读 Agent。你的工作方式由下面的 Skill 规定，"
            "你的动作空间由 action 契约规定；每一步你只能选一个动作。",
            ACTION_CONTRACT,
            HARD_RULES,
            f"# Skill：paper-reading\n\n{skill}" if skill else "",
            extra_rules,
        ]
        if part
    )


def _history_block(state: ReadingState, limit: int = 6) -> str:
    recent = state.observations[-limit:]
    if not recent:
        return "（还没有）"
    return "\n".join(
        f"{obs.step}. {obs.action}{'✓' if obs.ok else '✗'} {obs.summary}" for obs in recent
    )


def build_user_message(
    state: ReadingState,
    feedback: str | None = None,
    coverage: Any | None = None,
) -> str:
    last = state.observations[-1].detail.get("prompt_text", state.last_observation) if state.observations else ""
    parts = [
        f"任务：{state.task}",
        f"论文：{state.paper_id}（共 {state.pages_total} 页）",
        f"进度：{state.progress_line()}",
        f"已做过的动作：\n{_history_block(state)}",
    ]
    if coverage is not None:
        parts.append(
            f"覆盖度线索（关键词启发式，只说明原文里有没有相关线索）：{coverage.status_line}\n"
            f"缺口：{coverage.missing_line()}"
        )
        if coverage.all_covered():
            parts.append("四个维度都已有原文线索：如果报告所需信息已足够，可以 FINISH。")
    if state.tables:
        parts.append(f"已抽取的表格：{', '.join(sorted(state.tables))}")
    if last:
        parts.append(f"上一步的工具返回：\n{last[:2400]}")
    if feedback:
        parts.append(f"上一次输出无法执行/被驳回：{feedback}\n请据此调整，只输出一个合法 JSON action。")
    parts.append("请输出下一步 action（只输出 JSON）：")
    return "\n\n".join(parts)


def build_report_prompt(
    state: ReadingState,
    template: str | None = None,
    revision_notes: Sequence[str] = (),
    context_chars: int = 40000,
) -> list[dict[str, str]]:
    """收尾后的一次性写作调用：把已读正文按 `[p.N]` 拼好，按报告契约写 10 节报告。

    `revision_notes` 是 Critic 上一版指出的缺口——重写时必须逐条补齐，否则
    "Critic 驳回 → 再去 READ" 就会变成空转（缺口在"写"而不在"读"）。

    `context_chars` 是给正文留的字符预算。**别设太小**：实跑里 12000 会在第 5 页
    中间截断，模型于是把"没看到"写成"论文未涉及"，Critic 会一直以为它在偷懒。
    """

    if template is None:
        try:
            template = TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError:
            template = "报告固定包含 SKILL.md 里的 10 个部分。"
    system = (
        "你是论文阅读专家。只依据下面提供的已读原文写报告，不要使用未提供的知识、不要编造页码。\n"
        "具体事实必须写成 [p.N]；信息缺失就写「论文未涉及」或「未找到依据」。\n"
        "报告必须依次写满模板里的 10 个部分，每一节都要有内容。\n\n"
        f"# 报告模板\n\n{template}"
    )
    parts = [
        f"任务：{state.task}",
        f"论文：{state.paper_id}（共 {state.pages_total} 页，已读 p{state.pages_read_spec or '—'}）",
        f"阅读轨迹：{_history_block(state, limit=20)}",
    ]
    if revision_notes:
        notes = "\n".join(f"- {note}" for note in revision_notes[:12])
        parts.append(
            "## 上一版被 Critic 指出的缺口（这一版必须逐条补齐）\n"
            f"{notes}\n"
            "补齐方式：优先用已读原文里对应的页码内容展开；原文确实没有的，写明「论文未涉及」，不要重复上一版的省略。"
        )
    parts.append("## 已读原文（带页码）\n" + (state.context_excerpt(max_chars=context_chars) or "（没有读到正文）"))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


# --------------------------------------------------------------------------- #
# Action 解析
# --------------------------------------------------------------------------- #


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, flags=re.S)
    return fence.group(1) if fence else text


def extract_json_object(text: str) -> str | None:
    """从可能夹带散文的输出里抠出第一个完整的 JSON 对象（括号配平，跳过字符串内括号）。"""

    cleaned = _strip_fences(text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


def _normalize_tool_call_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容 OpenAI tool_calls / function-call 形态：{"name": "read_pdf", "arguments": {...}}。"""

    if "action" in payload:
        return payload
    name = payload.get("name") or payload.get("tool") or payload.get("function")
    if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
        name, args = name.get("name"), name.get("arguments")
    else:
        args = payload.get("arguments") or payload.get("parameters") or payload.get("args")
    if not isinstance(name, str):
        return payload
    action = TOOL_NAME_TO_ACTION.get(name.lower())
    if action is None:
        return payload
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    merged: dict[str, Any] = {"action": action}
    if isinstance(args, dict):
        merged.update(args)
    if action == "READ" and "pdf" not in merged:
        merged.pop("pdf", None)
    return merged


def validate_action(action: dict[str, Any]) -> str | None:
    """返回错误说明；None 表示合法。校验的是"能不能执行"，不是"该不该做"。"""

    if not isinstance(action, dict):
        return "action 必须是 JSON 对象"
    raw = action.get("action")
    if not isinstance(raw, str) or not raw.strip():
        return "缺少 action 字段"
    name = raw.strip().upper()
    if name not in ALLOWED_ACTIONS:
        return f"未知 action {raw!r}；只允许 {'/'.join(ALLOWED_ACTIONS)}"
    if name == "READ":
        pages = action.get("pages", "")
        if pages not in ("", None) and not re.fullmatch(r"[\d,\-\s]+", str(pages)):
            return f"pages 格式不对：{pages!r}，正确写法如 \"1-3,8\""
    elif name == "SEARCH":
        if not str(action.get("query", "")).strip():
            return "SEARCH 缺少 query"
        scope = str(action.get("scope", "local")).lower()
        if scope not in SEARCH_SCOPES:
            return f"scope 只能是 {list(SEARCH_SCOPES)}，收到 {scope!r}"
    elif name == "ANALYZE":
        operation = str(action.get("operation", "")).lower()
        if operation not in ANALYZE_OPERATIONS:
            return f"operation 只能是 {list(ANALYZE_OPERATIONS)}，收到 {operation!r}"
        if operation != "extract" and not str(action.get("table", "")).strip():
            return f"ANALYZE {operation} 需要 table（table:N 或表格文件路径）"
    return None


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """把模型输出变成 action。返回 (action, error)，两者必有其一为 None。"""

    if not isinstance(text, str) or not text.strip():
        return None, "模型输出为空"
    snippet = extract_json_object(text)
    if snippet is None:
        return None, f"输出里找不到合法 JSON 对象：{text.strip()[:120]!r}"
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:  # pragma: no cover - extract 已保证可解析
        return None, f"JSON 解析失败：{exc}"
    if not isinstance(payload, dict):
        return None, "JSON 顶层必须是对象"
    payload = _normalize_tool_call_shape(payload)
    payload["action"] = str(payload.get("action", "")).strip().upper()
    error = validate_action(payload)
    if error:
        return None, error
    return payload, None


@dataclass
class Plan:
    action: dict[str, Any] | None
    error: str | None = None
    result: ChatResult | None = None


# --------------------------------------------------------------------------- #
# Planner 实现
# --------------------------------------------------------------------------- #


class LlmPlanner:
    """在线 Planner：一次调用换一个 action。"""

    name = "llm"

    def __init__(
        self,
        client: Any,
        *,
        max_tokens: int = 800,
        system_prompt: str | None = None,
        disabled_actions: Sequence[str] = (),
        coverage_hook: Any = None,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.disabled_actions = tuple(str(a).upper() for a in disabled_actions)
        # coverage_hook：由主循环注入"当前覆盖度怎么算"，避免 Planner 自己乱猜
        self.coverage_hook = coverage_hook
        if system_prompt is None:
            extra = ""
            if self.disabled_actions:
                extra = (
                    "## 本次运行的实验配置\n\n"
                    f"这些动作已被禁用（消融实验的一部分）：{', '.join(self.disabled_actions)}。"
                    "不要提出它们，被拒绝只会浪费步数。"
                )
            system_prompt = build_system_prompt(extra_rules=extra)
        self.system_prompt = system_prompt
        self.messages_used: list[list[dict[str, Any]]] = []

    def plan(self, state: ReadingState, feedback: str | None = None) -> Plan:
        coverage = self.coverage_hook(state) if callable(self.coverage_hook) else None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_message(state, feedback, coverage)},
        ]
        self.messages_used.append(messages)
        try:
            result = self.client.complete(messages, max_tokens=self.max_tokens)
        except Exception as exc:  # LlmError 及任何网络层异常都不该让进程崩掉
            return Plan(action=None, error=f"模型调用失败：{exc}")
        action, error = parse_action(result.text)
        return Plan(action=action, error=error, result=result)


class HeuristicPlanner:
    """离线 Planner：确定性策略，不需要 API key。

    轨迹形如：`READ 1-2 → READ 下一个未读章节所在页 → FINISH`。
    它有三个用途：(1) 没有 key 时也能把整条循环跑通、跑测试；
    (2) 第七阶段做 Fixed vs Agentic 对照时，"固定工作流"这一侧可以直接从它派生；
    (3) Critic 说"还不完整"时，它会照提示再补读一轮（第六阶段的离线演示靠这个）。
    """

    name = "heuristic"

    def __init__(self, pages_per_section: int = 2, max_critic_supplements: int = 2) -> None:
        self.pages_per_section = pages_per_section
        self.max_critic_supplements = max_critic_supplements
        self._finished_reading = False
        self._supplements = 0

    def plan(self, state: ReadingState, feedback: str | None = None) -> Plan:
        if not state.read_pages:
            return Plan(action={"action": "READ", "pages": "1-2", "reason": "先看题目与摘要，确定研究对象与问题"})

        # Critic 驳回后：按提示再补读一轮未读页（离线策略也遵守 Critic 的裁决）
        if feedback and "Critic" in feedback and self._supplements < self.max_critic_supplements:
            span = state.next_unread_pages(count=self.pages_per_section)
            if span:
                self._supplements += 1
                return Plan(
                    action={
                        "action": "READ",
                        "pages": ",".join(str(p) for p in span),
                        "reason": "按 Critic 的缺口提示补读未读页",
                    }
                )

        if self._finished_reading:
            return Plan(action={"action": "FINISH", "reason": "离线策略：已读完摘要与一个论证章节，交回主循环收尾"})

        remaining = state.budget.max_pages - state.pages_read_count
        for item in state.outline:
            page = int(item.get("page") or 0)
            if page <= 0 or page in state.read_pages:
                continue
            span = [p for p in range(page, min(page + self.pages_per_section, state.pages_total + 1))
                    if p not in state.read_pages][: max(0, remaining)]
            if span:
                pages = ",".join(str(p) for p in span)
                self._finished_reading = True
                return Plan(
                    action={
                        "action": "READ",
                        "pages": pages,
                        "reason": f"按大纲读「{item.get('heading', '')}」所在页（离线策略固定顺序）",
                    }
                )
        self._finished_reading = True
        return Plan(action={"action": "FINISH", "reason": "离线策略：没有更多可读章节"})
