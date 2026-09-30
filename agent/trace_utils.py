#!/usr/bin/env python3
"""轨迹工具 —— 把一次运行的 trace 变成"路径"和"指标"。

第四阶段产出的 trace JSONL 是后续实验的原始数据：路线图第七节说的
"路径 A：READ→READ→READ→FINISH / 路径 B：READ→SEARCH→READ→ANALYZE→FINISH"
在这里第一次成为**可打印、可比较**的东西；第十三节要的
`Accuracy / Completeness / Tool Calls / Token Cost / Latency`
里后三个也直接从这里取数（前两个要人工或评测集打分）。

命令行：

    python3 -m agent.trace_utils traces/run.jsonl            # 单次：路径 + 指标
    python3 -m agent.trace_utils traces/*.jsonl --table      # 多次：对比表（阶段 8 的雏形）
    python3 -m agent.trace_utils traces/run.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pdf_reader import ToolError  # noqa: E402
from tools.table_analyzer import render_table  # noqa: E402


@dataclass
class RunTrace:
    path: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    critic_rounds: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def actions(self) -> list[str]:
        return [str(call.get("action", "?")) for call in self.tool_calls if call.get("action") not in {"BOOTSTRAP", None}]

    @property
    def action_path(self) -> str:
        names = self.actions
        return " → ".join(names) if names else "（无动作）"

    @property
    def coverage_line(self) -> str:
        coverage = self.summary.get("coverage") or {}
        dimensions = coverage.get("dimensions") or []
        if not dimensions:
            return "—"
        marks = {"covered": "✓", "weak": "~", "missing": "✗"}
        covered = sum(1 for dim in dimensions if dim.get("status") == "covered")
        detail = "".join(marks.get(dim.get("status"), "?") for dim in dimensions)
        return f"{covered}/{len(dimensions)} {detail}"

    def metrics(self) -> dict[str, Any]:
        summary = self.summary
        return {
            "run": Path(self.path).stem,
            "path": self.action_path,
            "steps": summary.get("steps"),
            "tool_calls": summary.get("tool_calls"),
            "searches": summary.get("searches"),
            "pages_read": summary.get("pages_read"),
            "input_tokens": summary.get("input_tokens"),
            "output_tokens": summary.get("output_tokens"),
            "llm_latency_ms": summary.get("llm_latency_ms"),
            "stop_cause": summary.get("stop_cause"),
            "finish_reason": summary.get("finish_reason"),
            "critic_rounds": len(self.critic_rounds) or len(summary.get("critic_rounds") or []),
            "coverage": self.coverage_line,
            "no_progress_streak": summary.get("no_progress_streak"),
            "errors": len(summary.get("errors") or []),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action_path": self.action_path,
            **self.metrics(),
            "tool_calls": self.tool_calls,
            "critic_rounds": self.critic_rounds,
        }


def load_trace(path: str | Path) -> RunTrace:
    target = Path(path).expanduser()
    if not target.exists():
        raise ToolError(f"轨迹文件不存在：{target}")
    run = RunTrace(path=str(target))
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ToolError(f"{target.name} 第 {line_number} 行不是合法 JSON：{exc}") from exc
        event = record.get("event")
        if event == "summary":
            run.summary = record
        elif event == "critic":
            run.critic_rounds.append(record)
        else:
            run.tool_calls.append(record)
    if not run.summary:
        raise ToolError(f"{target.name} 里没有 summary 行，可能不是完整的 trace")
    return run


def load_traces(paths: Iterable[str | Path]) -> list[RunTrace]:
    runs = [load_trace(path) for path in paths]
    if not runs:
        raise ToolError("没有可读的轨迹文件")
    return runs


def render_run(run: RunTrace) -> str:
    metrics = run.metrics()
    lines = [
        f"# {metrics['run']}",
        "",
        f"- 动作路径：{run.action_path}",
        f"- 停止：stop_cause={metrics['stop_cause']} · {metrics['finish_reason']}（错误 {metrics['errors']} 条）",
        f"- 指标：steps={metrics['steps']} · tool_calls={metrics['tool_calls']} "
        f"· searches={metrics['searches']} · pages=p{metrics['pages_read'] or '—'} "
        f"· 覆盖度={metrics['coverage']}",
        f"- 成本：tokens={metrics['input_tokens']}/{metrics['output_tokens']} "
        f"· llm_latency={metrics['llm_latency_ms']}ms",
        "",
    ]
    rows = [
        [
            call.get("step"),
            call.get("action"),
            call.get("tool"),
            "✓" if call.get("ok") else "✗",
            f"{call.get('duration_ms')}ms",
            str(call.get("summary", ""))[:80],
        ]
        for call in run.tool_calls
    ]
    lines.append(render_table(["step", "action", "tool", "ok", "耗时", "摘要"], rows,
                              ["right", "left", "left", "left", "right", "left"]))
    if run.critic_rounds:
        lines.append("")
        lines.append("## Critic 判词")
        critic_rows = [
            [
                verdict.get("after_step"),
                verdict.get("round"),
                "通过" if verdict.get("complete") else "驳回",
                verdict.get("next_action") or "—",
                str(verdict.get("reason", ""))[:60],
                "；".join(str(item) for item in (verdict.get("missing") or []))[:80] or "—",
            ]
            for verdict in run.critic_rounds
        ]
        lines.append(
            render_table(["发生在第几步", "轮次", "结论", "建议动作", "理由", "缺口"], critic_rows,
                         ["right", "right", "left", "left", "left", "left"])
        )
    return "\n".join(lines).rstrip() + "\n"


def render_comparison(runs: Sequence[RunTrace]) -> str:
    headers = ["run", "path", "steps", "tool_calls", "searches", "pages",
               "tokens(in/out)", "llm_ms", "critic", "coverage", "stop_cause"]
    rows = []
    for run in runs:
        metrics = run.metrics()
        rows.append(
            [
                metrics["run"],
                metrics["path"],
                metrics["steps"],
                metrics["tool_calls"],
                metrics["searches"],
                metrics["pages_read"] or "—",
                f"{metrics['input_tokens']}/{metrics['output_tokens']}",
                metrics["llm_latency_ms"],
                metrics["critic_rounds"],
                metrics["coverage"],
                str(metrics["stop_cause"])[:24],
            ]
        )
    return render_table(headers, rows,
                        ["left", "left", "right", "right", "right", "left", "right", "right", "right", "left", "left"]) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent.trace_utils", description="把 trace 变成路径与指标")
    parser.add_argument("traces", nargs="+", help="trace JSONL 文件（可多个）")
    parser.add_argument("--table", action="store_true", help="多个 trace 时输出对比表（阶段 8 的雏形）")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs = load_traces(args.traces)
    except ToolError as exc:
        print(f"trace_utils: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([run.to_dict() for run in runs], ensure_ascii=False, indent=2))
        return 0
    if args.table or len(runs) > 1:
        print(render_comparison(runs), end="")
    else:
        print(render_run(runs[0]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
