#!/usr/bin/env python3
"""Agent Loop —— 第四阶段的主循环，第五阶段加预算/停止条件，第六阶段加 Critic。

```text
User Request → Planner → {READ | SEARCH | ANALYZE | FINISH} → Executor → Observation ─┐
                    ▲                                                                  │
                    └──────────────────────────────────────────────────────────────────┘
                     FINISH → Answer（草稿） → Critic → 足够？ ─是→ Final Report
                                                      └─否（还缺 X，建议先做 Y）→ 回到 Planner
```

三条设计原则：

1. **下一步做什么由当前信息状态决定**，不写死顺序（第四阶段）。
2. **跑不完/跑飞了要有可解释的停法**（第五阶段）：动作预算、token 预算、无进展检测、
   覆盖度停止条件，每一种停止都落到结构化的 `stop_cause`，不吞。
3. **"Agent 说写完了"不等于写完了**（第六阶段）：FINISH 之后由 Critic 审一遍草稿，
   驳回时把缺口回灌给 Planner 继续做。Critic 只提意见、不执行动作——决策权始终只有一个。

轨迹 trace 里记的是行动、工具、结果、token 与停止原因，不含模型的隐藏推理；
第八阶段的 `Tool Calls` / `Token Cost` / `Latency`、第九阶段的难度分析都从这里取数。

命令行：

    python3 -m agent.agent --paper "paper/某篇.pdf"                 # 读工作区的 .llm.env；没填 key 就降级为离线策略
    python3 -m agent.agent --paper paper/x.pdf --planner heuristic --json
    python3 -m agent.agent --paper paper/x.pdf --trace traces/run.jsonl
    python3 -m agent.agent --paper paper/x.pdf --no-critic          # 消融：去掉 Critic
    python3 -m agent.agent --paper paper/x.pdf --stop-when-covered  # 覆盖度停止条件
    python3 -m agent.agent --paper paper/x.pdf --disable SEARCH,ANALYZE   # 消融：去掉某个工具

配置来源：**命令行 > 工作区根目录的 `.llm.env`（已 gitignore）> 环境变量 > 默认值**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pdf_reader import ToolError  # noqa: E402

from agent.coverage import CoverageReport, summarize as summarize_coverage  # noqa: E402
from agent.critic import CriticVerdict, LlmCritic, make_critic  # noqa: E402
from agent.executor import Executor  # noqa: E402
from agent.llm import LlmError, resolve_config  # noqa: E402
from agent.planner import (  # noqa: E402
    HeuristicPlanner,
    LlmPlanner,
    build_report_prompt,
    load_skill_text,
)
from agent.state import Budget, Observation, ReadingState  # noqa: E402

DEFAULT_TASK = "分析这篇论文：它的研究问题、核心主张、方法与证据是什么，结论是否被证据支撑。"
ALL_ACTIONS = ("READ", "SEARCH", "ANALYZE")


@dataclass
class AgentResult:
    state: ReadingState
    trace_path: str | None = None
    transcript: list[str] = field(default_factory=list)

    @property
    def finish_reason(self) -> str:
        return self.state.finish_reason or "unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = self.state.to_dict()
        payload["trace_path"] = self.trace_path
        return payload


class PaperAgent:
    def __init__(
        self,
        paper_path: str | Path,
        *,
        planner: Any,
        task: str = DEFAULT_TASK,
        budget: Budget | None = None,
        executor: Executor | None = None,
        critic: Any | None = None,
        stop_when_covered: bool = False,
        disabled_actions: Sequence[str] = (),
        max_critic_rounds: int = 2,
        report_context_chars: int = 40000,
        write_report: bool = False,
        max_parse_retries: int = 2,
        echo: bool = True,
        stream: Any = None,
    ) -> None:
        self.paper_path = Path(paper_path).expanduser()
        self.planner = planner
        self.task = task
        self.budget = budget or Budget()
        self.executor = executor or Executor(self.paper_path)
        self.critic = critic
        self.stop_when_covered = stop_when_covered
        self.disabled_actions = tuple(str(a).upper() for a in disabled_actions)
        self.max_critic_rounds = max_critic_rounds
        self.report_context_chars = report_context_chars
        self.write_report = write_report
        self.max_parse_retries = max_parse_retries
        self.echo = echo
        self.stream = stream or sys.stderr
        self.transcript: list[str] = []
        self._last_draft_from_llm = False
        self._last_draft: str | None = None
        self._revision_notes: list[str] = []
        # 把"覆盖度怎么算"注入 Planner，保证 prompt 里的覆盖度与主循环判据是同一个
        if isinstance(self.planner, LlmPlanner) and self.planner.coverage_hook is None:
            self.planner.coverage_hook = self.coverage

    # ------------------------------------------------------------- 输出 --
    def _say(self, line: str) -> None:
        self.transcript.append(line)
        if self.echo:
            print(line, file=self.stream, flush=True)

    @staticmethod
    def coverage(state: ReadingState) -> CoverageReport:
        """当前已读正文的覆盖度（第五阶段停止条件的判据来源）。"""

        return summarize_coverage(state)

    # ------------------------------------------------------------- 主循环 --
    def run(self) -> AgentResult:
        if not self.paper_path.exists():
            raise ToolError(f"文件不存在：{self.paper_path}")
        budget = self.budget
        state = ReadingState(
            paper_id=self.paper_path.stem,
            paper_path=str(self.paper_path.resolve()),
            pages_total=0,
            task=self.task,
            budget=budget,
        )
        self._say(f"[Agent] 开始阅读 · {state.paper_id}")
        self._say(f"[Agent] 任务：{self.task}")
        self._say(
            f"[Agent] 预算：steps≤{budget.max_steps} · search≤{budget.max_searches} "
            f"· pages≤{budget.max_pages} · tool_calls≤{budget.max_tool_calls} "
            f"· tokens≤{budget.max_total_tokens or '∞'} · 无进展≤{budget.max_no_progress_steps}"
        )
        self._say(
            f"[Agent] 停止条件：Planner FINISH"
            + (" · 覆盖度达标即停" if self.stop_when_covered else "")
            + (f" · Critic 复核（最多 {self.max_critic_rounds} 轮）" if self.critic else " · 无 Critic")
            + (" · 禁用动作：" + ",".join(self.disabled_actions) if self.disabled_actions else "")
        )

        bootstrap = self.executor.bootstrap(state)
        state.record_tool(bootstrap)
        self._log_tool(bootstrap)

        feedback: str | None = None
        parse_retries = 0
        blocked_streak = 0
        success_counts: dict[str, int] = {}
        started = time.time()

        while state.finish_reason is None:
            coverage = self.coverage(state)

            # ---- 第五阶段：成本/步数/进展三类硬约束，先查再花 ----
            exhausted = budget.token_exhausted(state)
            if exhausted:
                self._finish(state, "budget_tokens", f"budget:tokens", exhausted)
                break
            if state.steps >= budget.max_steps:
                self._finish(state, "budget_steps", "budget:max_steps", f"步数耗尽（{budget.max_steps} 步）")
                break
            if budget.max_no_progress_steps and state.no_progress_streak >= budget.max_no_progress_steps:
                self._finish(
                    state, "no_progress", "budget:no_progress",
                    f"连续 {state.no_progress_streak} 步没有拿到新信息（无进展停止条件）",
                )
                break

            # ---- 第五阶段：覆盖度停止条件（可选，默认关）----
            forced_by_coverage = False
            if self.stop_when_covered and coverage.all_covered():
                action = {"action": "FINISH", "reason": "覆盖度停止条件满足"}
                forced_by_coverage = True
                self._say(f"[Stop ] {coverage.status_line} → 覆盖度达标，直接收尾")
            else:
                plan = self.planner.plan(state, feedback)
                feedback = None
                if plan.result is not None:
                    state.record_llm(plan.result.input_tokens, plan.result.output_tokens, plan.result.latency_ms)
                if plan.action is None:
                    parse_retries += 1
                    state.errors.append(plan.error or "未知解析错误")
                    if parse_retries > self.max_parse_retries:
                        self._finish(state, "planner_error", f"planner_error:{plan.error}",
                                     f"Planner 连续 {parse_retries} 次输出无法执行")
                        break
                    feedback = plan.error
                    self._say(f"[Guard] 输出无法执行（第 {parse_retries} 次）：{plan.error} —— 要求 Planner 重出")
                    continue
                parse_retries = 0
                action = plan.action

            state.steps += 1
            name = action["action"]

            # ---- 第六阶段：FINISH 之前先过 Critic ----
            if name == "FINISH":
                reason = str(action.get("reason", "") or "信息已足够")
                if self.critic is None:
                    cause = "coverage" if forced_by_coverage else "agent_finish"
                    prefix = "coverage" if forced_by_coverage else "agent"
                    self._finish(state, cause, f"{prefix}:{reason}", f"FINISH —— {reason}")
                    break
                verdict = self._review(state, coverage, action)
                if verdict.complete:
                    self._finish(state, "critic_accept", f"critic:accept:{reason}",
                                 f"FINISH 通过 Critic 复核 —— {verdict.reason or reason}")
                    break
                if len(state.critic_rounds) >= self.max_critic_rounds:
                    state.errors.append(f"Critic 未通过（已复核 {len(state.critic_rounds)} 轮）：{'；'.join(verdict.missing)}")
                    if getattr(self, "_last_draft_from_llm", False) and state.answer:
                        # 第七/八阶段要拿这份稿子评 Accuracy/Completeness，所以留下但标明未完成
                        state.report = state.answer
                    self._finish(
                        state, "critic_exhausted", f"critic:exhausted({len(state.critic_rounds)})",
                        f"Critic 复核 {len(state.critic_rounds)} 轮仍未通过，收尾（报告标为未完成）",
                    )
                    break
                feedback = verdict.feedback
                if (verdict.next_action or "").upper() == "READ" and not state.unread_pages():
                    feedback += "（补充：正文已经没有未读页，再 READ 只会空转。）"
                self._say(f"[Critic] round {len(state.critic_rounds)} 驳回：{'；'.join(verdict.missing)}")
                self._say(f"[Agent] step {state.steps} · FINISH 被驳回，回到 Planner 继续补缺")
                continue

            # ---- 消融实验：被禁用的动作直接拒（第十阶段用）----
            if name in self.disabled_actions:
                blocked_streak += 1
                self._refuse(state, action, f"本次实验配置禁用了 {name}（消融），请换别的动作或 FINISH")
                if blocked_streak > budget.max_repeat_blocked:
                    self._finish(state, "blocked_loop", f"budget:blocked_loop({blocked_streak})",
                                 f"连续 {blocked_streak} 次提案被拒（含禁用动作），判定空转")
                continue

            pages: list[int] = []
            if name == "READ":
                try:
                    pages = self.executor.resolve_read_pages(action, state)
                except ToolError:
                    pages = []  # 页范围不合法时让 Executor 产出可读错误

            signature = self._signature(name, action)
            violation = budget.violation(action, state, pages)
            repeat_note = self._repeat_note(name, signature, success_counts)

            if violation or repeat_note:
                blocked_streak += 1
                self._refuse(state, action, violation or repeat_note or "")
                if blocked_streak > budget.max_repeat_blocked:
                    self._finish(state, "blocked_loop", f"budget:blocked_loop({blocked_streak})",
                                 f"连续 {blocked_streak} 次提案被系统拒绝，判定空转")
                continue

            blocked_streak = 0
            before_pages = set(state.read_pages)
            observation = self.executor.execute(action, state)
            state.record_tool(observation)
            success_counts[signature] = success_counts.get(signature, 0) + 1
            if self._information_gain(observation, before_pages):
                state.no_progress_streak = 0
            else:
                state.no_progress_streak += 1
            self._log_tool(observation)

        self._finalize(state)
        elapsed_ms = int((time.time() - started) * 1000)
        self._say(
            f"[Agent] 结束 · stop_cause={state.stop_cause} · finish_reason={state.finish_reason} "
            f"· steps={state.steps} · tool_calls={state.tool_calls} · pages=p{state.pages_read_spec or '—'} "
            f"· tokens={state.input_tokens}/{state.output_tokens} · wall={elapsed_ms}ms"
        )
        if state.coverage:
            marks = " ".join(
                f"{dim['name']}{'✓' if dim['status'] == 'covered' else ('~' if dim['status'] == 'weak' else '✗')}"
                for dim in state.coverage["dimensions"]
            )
            self._say(f"[Agent] 覆盖度：{marks}")
        return AgentResult(state=state, transcript=self.transcript)

    # ------------------------------------------------------------- 收尾 --
    def _finish(self, state: ReadingState, cause: str, reason: str, message: str) -> None:
        state.stop_cause = cause
        state.finish_reason = reason
        self._say(f"[Stop ] {cause} · {message}")

    def _finalize(self, state: ReadingState) -> None:
        state.finish_reason = state.finish_reason or "unknown"
        state.stop_cause = state.stop_cause or "unknown"
        state.coverage = self.coverage(state).to_dict()
        if state.answer is None:
            if self.write_report:
                self._write_report(state)
            else:
                state.answer = self._evidence_bundle(state)
        # 不管为什么停：模型写过的最后一版草稿都留下（标未完成），
        # 否则第七/八阶段没法对 Completeness / Accuracy 打分。
        if state.report is None and self._last_draft is not None:
            state.report = self._last_draft

    def _review(self, state: ReadingState, coverage: CoverageReport, action: dict[str, Any]) -> CriticVerdict:
        """FINISH 之后的复核：先备一份草稿（有 LLM 就是报告草稿），再交给 Critic。"""

        draft, from_llm = self._draft(state)
        verdict = self.critic.review(state, draft, coverage, round_number=len(state.critic_rounds) + 1)
        state.record_llm(verdict.input_tokens, verdict.output_tokens, verdict.latency_ms)
        self._last_draft_from_llm = from_llm
        record = {**verdict.to_dict(), "after_step": state.steps, "finish_claimed": action.get("reason", "")}
        state.critic_rounds.append(record)
        self._say(
            f"[Critic] round {verdict.round} · complete={verdict.complete} · {verdict.reason or '（无理由）'}"
            f" · 覆盖={coverage.status_line}"
        )
        if verdict.complete:
            if from_llm:
                state.report = draft
            state.answer = draft
        else:
            # 缺口记下来，下一次写草稿时要逐条补齐（否则"驳回→再读"会变成空转）
            for item in verdict.missing:
                if item not in self._revision_notes:
                    self._revision_notes.append(item)
            if from_llm:
                state.answer = draft
                state.errors.append("Critic 未通过：当前报告为未完成稿")
        return verdict

    def _draft(self, state: ReadingState) -> tuple[str, bool]:
        """返回 (草稿, 是否由模型写成)。没有客户端或没开报告时，草稿就是证据包。"""

        client = getattr(self.planner, "client", None)
        if client is None or not self.write_report:
            return self._evidence_bundle(state), False
        messages = build_report_prompt(
            state, revision_notes=tuple(self._revision_notes), context_chars=self.report_context_chars
        )
        try:
            result = client.complete(messages, max_tokens=2200)
        except LlmError as exc:
            state.errors.append(f"草稿生成失败：{exc}")
            self._say(f"[Guard] 草稿生成失败：{exc}")
            return self._evidence_bundle(state), False
        state.record_llm(result.input_tokens, result.output_tokens, result.latency_ms)
        draft = result.text.strip()
        self._last_draft = draft
        self._say(f"[Agent] 草稿已生成（{len(draft)} 字符），交给 Critic 复核")
        return draft, True

    # ------------------------------------------------------------- 辅助 --
    @staticmethod
    def _signature(name: str, action: dict[str, Any]) -> str:
        payload = {k: v for k, v in sorted(action.items()) if k not in {"reason", "action"}}
        return f"{name}:{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"

    @staticmethod
    def _repeat_note(name: str, signature: str, success_counts: dict[str, int]) -> str | None:
        """同一个动作已经成功做过两次，再来一次就是在烧预算。"""

        if success_counts.get(signature, 0) >= 2:
            return f"相同动作已成功执行 {success_counts[signature]} 次（{name}），请换参数或 FINISH"
        return None

    @staticmethod
    def _information_gain(observation: Observation, before_pages: set[int]) -> bool:
        """这一步有没有拿到新信息——第五阶段"无进展"停止条件的判据。"""

        if not observation.ok:
            return False
        detail = observation.detail or {}
        if observation.action == "READ":
            return any(page not in before_pages for page in detail.get("pages", []) or [])
        if observation.action == "SEARCH":
            return bool(detail.get("result_count"))
        if observation.action == "ANALYZE":
            return bool(detail.get("table_count")) or bool(detail.get("payload"))
        return False

    def _refuse(self, state: ReadingState, action: dict[str, Any], reason: str) -> None:
        name = str(action.get("action", "")).upper()
        refused = Observation(
            step=state.steps,
            action=name,
            tool="-",
            args={k: v for k, v in action.items() if k != "action"},
            ok=False,
            summary=f"拒绝执行：{reason}",
            error=reason,
            detail={"prompt_text": f"该 action 被系统拒绝：{reason}\n请换成别的页/别的词，或直接 FINISH。"},
        )
        state.record_tool(refused)
        state.no_progress_streak += 1
        self._say(f"[Guard] step {state.steps} · {name} 被拒：{reason}")

    def _log_tool(self, observation: Observation) -> None:
        mark = "✓" if observation.ok else "✗"
        self._say(f"[Tool ] step {observation.step} · {observation.tool} → {observation.summary} "
                  f"({observation.duration_ms}ms) {mark}")

    def _write_report(self, state: ReadingState) -> None:
        client = getattr(self.planner, "client", None)
        if client is None:
            # 离线策略写不了报告，交回证据包（这是配置事实，不算错误）
            state.answer = self._evidence_bundle(state)
            return
        messages = build_report_prompt(
            state, revision_notes=tuple(self._revision_notes), context_chars=self.report_context_chars
        )
        try:
            result = client.complete(messages, max_tokens=2200)
        except LlmError as exc:
            state.errors.append(f"报告生成失败：{exc}")
            state.answer = self._evidence_bundle(state)
            self._say(f"[Guard] 报告生成失败：{exc}")
            return
        state.record_llm(result.input_tokens, result.output_tokens, result.latency_ms)
        state.report = result.text.strip()
        state.answer = state.report
        self._say(f"[Agent] 报告已生成（{len(state.report)} 字符）")

    @staticmethod
    def _evidence_bundle(state: ReadingState) -> str:
        """离线模式没有 LLM 可写报告，就交回可核查的证据包（不是伪造的报告）。"""

        lines = [
            f"# 证据包（离线模式）：{state.paper_id}",
            f"- 已读页：p{state.pages_read_spec or '—'}（共 {state.pages_total} 页，{state.read_chars} 字符）",
            f"- 工具调用：{state.tool_calls} 次（搜索 {state.searches} 次，表格 {len(state.tables)} 个）",
            f"- 停止原因：{state.stop_cause} · {state.finish_reason}",
            "",
            "## 已读正文（带页码）",
            state.context_excerpt(max_chars=6000) or "（未读到正文）",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 便捷入口
# --------------------------------------------------------------------------- #


def run_agent(
    paper: str | Path,
    *,
    planner: Any = None,
    task: str = DEFAULT_TASK,
    budget: Budget | None = None,
    critic: Any | None = None,
    stop_when_covered: bool = False,
    disabled_actions: Sequence[str] = (),
    max_critic_rounds: int = 2,
    write_report: bool = False,
    trace_path: str | Path | None = None,
    echo: bool = True,
    stream: Any = None,
) -> AgentResult:
    agent = PaperAgent(
        paper,
        planner=planner or HeuristicPlanner(),
        task=task,
        budget=budget,
        critic=critic,
        stop_when_covered=stop_when_covered,
        disabled_actions=disabled_actions,
        max_critic_rounds=max_critic_rounds,
        write_report=write_report,
        echo=echo,
        stream=stream,
    )
    result = agent.run()
    if trace_path:
        write_trace(result, trace_path)
    return result


def write_trace(result: AgentResult, path: str | Path) -> str:
    """把轨迹写成 JSONL：工具调用行 + Critic 判词行 + 最后一行汇总。

    Critic 行带 `after_step`，所以即使写在文件末尾也能还原"第几步时被复核过"。
    第八阶段的 Tool Calls / Token Cost / Latency、第九阶段的难度对比都读这个文件。
    """

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for observation in result.state.observations:
            handle.write(json.dumps({"event": "tool_call", **observation.to_dict()}, ensure_ascii=False) + "\n")
        for verdict in result.state.critic_rounds:
            handle.write(json.dumps({"event": "critic", **verdict}, ensure_ascii=False) + "\n")
        summary = result.state.to_dict()
        for key in ("trace", "answer", "report"):
            summary.pop(key, None)
        handle.write(json.dumps({"event": "summary", **summary}, ensure_ascii=False) + "\n")
    return str(target.resolve())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.agent",
        description="Agent Loop：READ / SEARCH / ANALYZE / FINISH + 预算/停止条件 + Critic",
    )
    parser.add_argument("--paper", required=True, help="论文 PDF 路径")
    parser.add_argument("--task", default=DEFAULT_TASK, help="用户任务原话")
    parser.add_argument("--planner", default="auto", choices=["auto", "llm", "heuristic"],
                        help="auto：有 API key 用模型，否则离线策略")
    parser.add_argument("--papers-dir", default=str(REPO_ROOT / "paper"), help="SEARCH 的本地语料目录")
    parser.add_argument("--model", help="模型名；默认取配置文件 / LLM_MODEL，或按 base_url 猜")
    parser.add_argument("--base-url", help="OpenAI 兼容 base url；默认取配置文件 / LLM_BASE_URL")
    parser.add_argument("--api-key", help="API key；一般不用写，填在工作区根的 .llm.env 里即可")
    parser.add_argument("--config", help="配置文件路径（默认 ./.llm.env，已 gitignore）")
    parser.add_argument("--no-config", action="store_true", help="忽略配置文件，只用命令行与环境变量")
    parser.add_argument("--timeout", type=float, default=None, help="请求超时秒数（默认 60，可写在配置里）")
    parser.add_argument("--max-retries", type=int, default=None, help="失败重试次数（默认 2，可写在配置里）")

    handler = parser.add_argument_group("预算 / 停止条件（第五阶段）")
    handler.add_argument("--max-steps", type=int, default=8)
    handler.add_argument("--max-searches", type=int, default=3)
    handler.add_argument("--max-pages", type=int, default=20)
    handler.add_argument("--max-tool-calls", type=int, default=12)
    handler.add_argument("--max-chars-per-page", type=int, default=6000)
    handler.add_argument("--max-total-tokens", type=int, default=60000, help="输入+输出 token 上限，0=不限")
    handler.add_argument("--max-no-progress", type=int, default=3, help="连续多少步没有新信息就停")
    handler.add_argument("--max-parse-retries", type=int, default=2)
    handler.add_argument("--stop-when-covered", action="store_true",
                         help="四个维度都覆盖到原文线索即停（第五阶段的覆盖度停止条件）")

    critic_group = parser.add_argument_group("Critic（第六阶段）")
    critic_group.add_argument("--critic", dest="critic", action="store_true", help="FINISH 之后让 Critic 复核")
    critic_group.add_argument("--no-critic", dest="critic", action="store_false", help="关掉 Critic（消融用）")
    parser.set_defaults(critic=None)
    critic_group.add_argument("--max-critic-rounds", type=int, default=2, help="Critic 最多驳回几次")

    ablation = parser.add_argument_group("消融（第十阶段用）")
    ablation.add_argument("--disable", default="", help="禁用动作，逗号分隔，如 SEARCH,ANALYZE")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--report", dest="report", action="store_true", help="收尾后让模型写 10 节报告")
    group.add_argument("--no-report", dest="report", action="store_false", help="只输出轨迹与证据包")
    parser.set_defaults(report=None)
    parser.add_argument("--report-context-chars", type=int, default=40000,
                        help="写报告时给正文的字符预算（太小会把后几页截掉，表现为'论文未涉及'）")

    parser.add_argument("--trace", help="把轨迹写成 JSONL（第八阶段指标取数）")
    parser.add_argument("--json", action="store_true", help="把最终状态打成 JSON 输出到 stdout")
    parser.add_argument("--quiet", action="store_true", help="不打印过程（给程序调用时用）")
    return parser


def build_runtime(args: argparse.Namespace, stream: Any) -> tuple[Any, Any, str]:
    """按参数装配 Planner 与 Critic。返回 (planner, critic, 人话说明)。"""

    if args.planner == "heuristic":
        planner: Any = HeuristicPlanner()
        client = None
        note = "离线启发式策略（不调用模型）"
    else:
        config, note = resolve_config(
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            timeout=args.timeout, max_retries=args.max_retries,
            config_path=args.config, use_config=not args.no_config,
        )
        if config is None:
            if args.planner == "llm":
                raise LlmError(f"{note}；如需离线跑通循环请用 --planner heuristic")
            print(f"[Agent] 提示：{note}，本次自动改用离线启发式策略（--planner heuristic）", file=stream)
            planner, client = HeuristicPlanner(), None
        else:
            from agent.llm import OpenAICompatibleClient

            client = OpenAICompatibleClient(config)
            planner = LlmPlanner(client)

    critic_enabled = args.critic if args.critic is not None else True  # 默认系统是带 Critic 的
    critic = make_critic(client, enabled=critic_enabled, skill_contract=load_skill_text())
    if critic is not None and getattr(critic, "source", "") == "rules":
        note += "；Critic=规则版（离线）"
    elif critic is not None:
        note += "；Critic=模型版"
    else:
        note += "；Critic=关闭"
    return planner, critic, note


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stream = sys.stderr
    budget = Budget(
        max_steps=args.max_steps,
        max_searches=args.max_searches,
        max_pages=args.max_pages,
        max_tool_calls=args.max_tool_calls,
        max_chars_per_page=args.max_chars_per_page,
        max_total_tokens=args.max_total_tokens,
        max_no_progress_steps=args.max_no_progress,
    )
    disabled = [name.strip().upper() for name in re.split(r"[,\s]+", args.disable) if name.strip()]
    unknown = [name for name in disabled if name not in ALL_ACTIONS]
    if unknown:
        print(f"agent: --disable 只支持 {list(ALL_ACTIONS)}，不支持 {unknown}", file=stream)
        return 2

    try:
        planner, critic, note = build_runtime(args, stream)
    except LlmError as exc:
        print(f"agent: {exc}", file=stream)
        return 2

    report = args.report if args.report is not None else isinstance(planner, LlmPlanner)
    if report and not isinstance(planner, LlmPlanner):
        print("[Agent] 提示：离线策略无法生成报告，改为输出证据包。", file=stream)
        report = False

    agent = PaperAgent(
        args.paper,
        planner=planner,
        task=args.task,
        budget=budget,
        executor=Executor(args.paper, papers_dir=args.papers_dir),
        critic=critic,
        stop_when_covered=args.stop_when_covered,
        disabled_actions=disabled,
        max_critic_rounds=args.max_critic_rounds,
        report_context_chars=args.report_context_chars,
        write_report=report,
        max_parse_retries=args.max_parse_retries,
        echo=not args.quiet,
        stream=stream,
    )
    if not args.quiet:
        print(f"[Agent] Planner：{note}", file=stream)
    try:
        result = agent.run()
    except ToolError as exc:
        print(f"agent: {exc}", file=stream)
        return 2

    if args.trace:
        path = write_trace(result, args.trace)
        if not args.quiet:
            print(f"[Agent] 轨迹已写入 {path}", file=stream)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif result.state.report:
        print(result.state.report)
    elif result.state.answer and not args.quiet:
        print(result.state.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
