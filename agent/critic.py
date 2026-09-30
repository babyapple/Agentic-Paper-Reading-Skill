#!/usr/bin/env python3
"""Critic —— 第六阶段：在"Agent 说写完了"之后，再有人（或规则）检查一遍。

路线图第六阶段的四问：

```text
1. 是否遗漏核心方法？
2. 是否遗漏实验？
3. 是否存在无依据结论？
4. 是否需要继续搜索？
```

返回结构照抄路线图：

```json
{"complete": false, "missing": ["ablation results"], "next_action": "READ"}
```

两种实现共用同一个判词结构，因此可以互相替换（第十阶段做 `Full vs -Critic` 消融时，
换的就是这一层）：

- `LlmCritic`：把"当前草稿 + 已读页码 + 覆盖度线索"交给模型，要一个 JSON 判词。
- `RuleCritic`：不需要联网，用覆盖度（`agent/coverage.py`）＋ 页码引用 ＋ 工具错误
  做规则判定。它让第六阶段在**没有 API key 的情况下也能跑通、能测、能演示**，
  同时是"规则版 Critic"这一对照组的天然实现。

Critic 本身不执行动作，也不改写草稿：它只回一句"还缺什么、建议先做什么"，
由 Planner 决定是否采纳——决策权始终只有一个。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent.coverage import CoverageReport
from agent.planner import extract_json_object
from agent.state import ReadingState, compress_pages

ALLOWED_NEXT_ACTIONS = ("READ", "SEARCH", "ANALYZE", "FINISH")
CITATION_RE = re.compile(r"\[p\.\s*\d+")

CRITIC_CONTRACT = """\
## 你的输出（只输出一个 JSON 对象）

```json
{"complete": false,
 "missing": ["还缺什么，具体到能执行", "..."],
 "next_action": "READ",
 "reason": "一句话说明判断依据"}
```

- `complete`：当前草稿是否已经**足够**支撑报告契约的 10 个部分（不是"完美"，是不缺关键项）。
- `missing`：逐条写缺什么；描述要能直接变成下一步动作（例如"缺少对比对象的评价标准"）。
- `next_action`：`complete` 为 false 时建议动作，只能是 READ / SEARCH / ANALYZE / FINISH。
  **正文已全部读完时不要建议 READ**——改成 `FINISH`，表示"按你列出的缺口重写答案"。
- `reason`：一句话，不要长篇评论。

## 检查清单（路线图第六阶段）

1. 是否遗漏核心方法 / 论证路径？
2. 是否遗漏材料与证据（实证类：数据集、baseline、指标、消融）？
3. 是否存在**无依据的结论**（没有 `[p.N]` 页码支撑、或超出原文的说法）？
4. 是否还需要继续搜索（论文外概念、术语、对照研究）？"""


@dataclass
class CriticVerdict:
    complete: bool
    missing: list[str] = field(default_factory=list)
    next_action: str | None = None
    reason: str = ""
    source: str = "rules"          # rules | llm
    round: int = 1
    raw: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "complete": self.complete,
            "missing": self.missing,
            "next_action": self.next_action,
            "reason": self.reason,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }

    @property
    def feedback(self) -> str:
        """回灌给 Planner 的那句话（也是 trace 里能看懂的那句）。"""

        if self.complete:
            return "Critic 认为已经足够。"
        gaps = "；".join(self.missing) if self.missing else "（Critic 未给出具体缺口）"
        if (self.next_action or "").upper() == "FINISH":
            # 缺口在"写"而不在"读"：FINISH 会触发带着缺口清单的重写，别让它去重复读旧页
            return (
                f"Critic 认为还不完整：{gaps}。"
                "请直接 FINISH —— 系统会带着这份缺口清单重写答案（不要重复读已经读过的页）。"
            )
        hint = f"建议下一步：{self.next_action}。" if self.next_action else ""
        return f"Critic 认为还不完整：{gaps}。{hint}不要重复 FINISH。"


# --------------------------------------------------------------------------- #
# 判词解析
# --------------------------------------------------------------------------- #


def parse_verdict(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """把 Critic 的输出解析成判词。宽容程度与 action 解析一致。"""

    if not isinstance(text, str) or not text.strip():
        return None, "Critic 输出为空"
    snippet = extract_json_object(text)
    if snippet is None:
        return None, f"Critic 输出里找不到 JSON：{text.strip()[:120]!r}"
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:  # pragma: no cover - extract 已保证可解析
        return None, f"Critic JSON 解析失败：{exc}"
    if not isinstance(payload, dict):
        return None, "Critic 输出顶层必须是对象"

    complete = payload.get("complete")
    if isinstance(complete, str):
        complete = complete.strip().lower() in {"true", "yes", "1"}
    if not isinstance(complete, bool):
        return None, f"critic 判词缺少布尔型 complete：{payload!r}"

    missing_raw = payload.get("missing") or []
    if isinstance(missing_raw, str):
        missing = [missing_raw] if missing_raw.strip() else []
    elif isinstance(missing_raw, list):
        missing = [str(item).strip() for item in missing_raw if str(item).strip()]
    else:
        return None, "missing 必须是字符串或字符串数组"

    next_action = payload.get("next_action")
    if next_action is not None:
        next_action = str(next_action).strip().upper() or None
        if next_action not in ALLOWED_NEXT_ACTIONS:
            return None, f"next_action 只能是 {list(ALLOWED_NEXT_ACTIONS)}，收到 {next_action!r}"
    if complete and not next_action:
        next_action = "FINISH"
    return {
        "complete": complete,
        "missing": missing,
        "next_action": next_action,
        "reason": str(payload.get("reason", "")).strip(),
    }, None


# --------------------------------------------------------------------------- #
# 规则版 Critic（离线）
# --------------------------------------------------------------------------- #


class RuleCritic:
    """不联网的 Critic：覆盖度 + 页码引用 + 工具错误。"""

    name = "rules"
    source = "rules"

    def __init__(self, max_errors: int = 3, require_citation: bool = True) -> None:
        self.max_errors = max_errors
        self.require_citation = require_citation

    def review(
        self,
        state: ReadingState,
        draft: str | None,
        coverage: CoverageReport,
        round_number: int = 1,
    ) -> CriticVerdict:
        missing: list[str] = []
        if not state.read_pages:
            missing.append("还没有读到任何正文")
        for dim in coverage.dimensions:
            if dim.status == "missing":
                missing.append(f"{dim.label}：没有任何原文线索（建议再读相关章节）")
            elif dim.status == "weak":
                missing.append(f"{dim.label}：只找到 {dim.hits} 处线索，可能还没读到关键章节")

        if self.require_citation and state.page_texts:
            if not draft or not CITATION_RE.search(draft):
                missing.append("当前答案没有任何 [p.N] 页码引用，结论无法核对")

        errors = [err for err in state.errors if err]
        if len(errors) >= self.max_errors:
            missing.append(f"已有 {len(errors)} 条工具错误，证据可能没取全")

        complete = not missing
        if complete:
            next_action = "FINISH"
        elif state.unread_pages() and state.pages_read_count < state.budget.max_pages:
            next_action = "READ"
        elif draft:
            # 正文读完 + 已有草稿：缺口在"写"，下一版重写即可（再读只会空转）
            next_action = "FINISH"
        elif state.searches < state.budget.max_searches:
            next_action = "SEARCH"
        else:
            next_action = "FINISH"
        return CriticVerdict(
            complete=complete,
            missing=missing,
            next_action=next_action,
            reason=f"规则判定：{len(missing)} 项待补" if missing else "规则判定：四维度覆盖且答案带页码",
            source=self.source,
            round=round_number,
            raw=json.dumps(
                {"complete": complete, "missing": missing, "next_action": next_action},
                ensure_ascii=False,
            ),
        )


# --------------------------------------------------------------------------- #
# 模型版 Critic
# --------------------------------------------------------------------------- #


def build_critic_prompt(
    state: ReadingState,
    draft: str | None,
    coverage: CoverageReport,
    *,
    skill_contract: str = "",
) -> list[dict[str, str]]:
    system = "\n\n".join(
        part
        for part in [
            "你是论文阅读结果的 Critic。你的任务不是重写报告，而是判断它**是否足够**，"
            "并指出最关键、可直接执行的缺口。",
            CRITIC_CONTRACT,
            skill_contract,
        ]
        if part
    )
    observations = "\n".join(
        f"{obs.step}. {obs.action}{'✓' if obs.ok else '✗'} {obs.summary}" for obs in state.observations[-12:]
    )
    unread = state.unread_pages()
    reads_left = max(0, state.budget.max_pages - state.pages_read_count)
    if unread and reads_left:
        reading_note = f"未读页：{compress_pages(unread[:12])}（页预算还剩 {reads_left} 页）"
    else:
        reading_note = "未读页：无（正文已读完或页预算用尽）—— 这种情况下 next_action 不要给 READ，用 FINISH 表示按缺口重写。"
    user = "\n\n".join(
        [
            f"任务：{state.task}",
            f"论文：{state.paper_id}（共 {state.pages_total} 页，已读 p{state.pages_read_spec or '—'}）",
            reading_note,
            f"覆盖度线索（关键词启发式，仅供参考）：{coverage.status_line}",
            f"缺口提示：{coverage.missing_line()}",
            f"工具轨迹：\n{observations or '（无）'}",
            "## 待审草稿\n" + ((draft or "（还没有草稿，只能用已读页码与工具轨迹判断）")[:4000]),
            "请输出你的 JSON 判词：",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class LlmCritic:
    """让模型当 Critic。判词解析失败时按"不完整"处理，但绝不因此崩掉主循环。"""

    name = "llm"

    def __init__(self, client: Any, *, max_tokens: int = 700, skill_contract: str = "") -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.skill_contract = skill_contract

    def review(
        self,
        state: ReadingState,
        draft: str | None,
        coverage: CoverageReport,
        round_number: int = 1,
    ) -> CriticVerdict:
        messages = build_critic_prompt(state, draft, coverage, skill_contract=self.skill_contract)
        started = time.time()
        try:
            result = self.client.complete(messages, max_tokens=self.max_tokens)
        except Exception as exc:
            return CriticVerdict(
                complete=True,  # Critic 自己坏了不该把 Agent 拖死：放行并在 reason 里说明
                missing=[],
                next_action="FINISH",
                reason=f"Critic 调用失败，按放行处理：{exc}",
                source="llm-error",
                round=round_number,
                raw=str(exc),
                latency_ms=int((time.time() - started) * 1000),
            )

        payload, error = parse_verdict(result.text)
        if payload is None:
            return CriticVerdict(
                complete=True,
                missing=[],
                next_action="FINISH",
                reason=f"Critic 判词无法解析，按放行处理：{error}",
                source="llm-unparsed",
                round=round_number,
                raw=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms or int((time.time() - started) * 1000),
            )
        return CriticVerdict(
            complete=payload["complete"],
            missing=payload["missing"],
            next_action=payload["next_action"],
            reason=payload["reason"],
            source="llm",
            round=round_number,
            raw=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms or int((time.time() - started) * 1000),
        )


def make_critic(client: Any | None, *, enabled: bool, skill_contract: str = "") -> Any | None:
    """没有客户端就走规则版；这样"系统带 Critic"这件事在离线环境也成立。"""

    if not enabled:
        return None
    if client is None:
        return RuleCritic()
    return LlmCritic(client, skill_contract=skill_contract)
