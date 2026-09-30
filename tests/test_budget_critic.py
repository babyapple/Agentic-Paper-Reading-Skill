#!/usr/bin/env python3
"""第五阶段（预算 / 停止条件）与第六阶段（Critic）的回归测试。

跑法：`python3 -m unittest discover -s tests -v`

这些测试**全部离线**：Planner 用 `ScriptedClient`，Critic 用规则版或测试内的小假件，
不读工作区里那个可能带真 key 的 `.llm.env`（CLI 用例统一加 `--no-config`）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agent import PaperAgent, write_trace  # noqa: E402
from agent.coverage import CoverageReport, summarize  # noqa: E402
from agent.critic import (  # noqa: E402
    CriticVerdict,
    LlmCritic,
    RuleCritic,
    build_critic_prompt,
    make_critic,
    parse_verdict,
)
from agent.llm import ScriptedClient  # noqa: E402
from agent.planner import LlmPlanner, build_system_prompt  # noqa: E402
from agent.state import Budget, ReadingState  # noqa: E402
from agent import trace_utils  # noqa: E402

PAPERS_DIR = REPO_ROOT / "paper"


def dance_paper() -> Path:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    return pdfs[0]


def fake_state(pages: dict[int, str], **kwargs) -> ReadingState:
    """只带 page_texts 的状态，用来测覆盖度与规则版 Critic。"""

    state = ReadingState(paper_id="fake", paper_path="/tmp/fake.pdf", pages_total=10, task="t")
    state.page_texts = pages
    state.read_pages = set(pages)
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


class FakeCritic:
    """测试用小 Critic：按剧本给判词，并记录它看到的草稿。"""

    source = "llm"

    def __init__(self, verdicts: list[CriticVerdict] | None = None) -> None:
        self.verdicts = list(verdicts or [])
        self.calls: list[dict] = []

    def review(self, state, draft, coverage, round_number: int = 1) -> CriticVerdict:
        self.calls.append({"draft": draft or "", "coverage": coverage.status_line, "round": round_number})
        verdict = self.verdicts.pop(0) if self.verdicts else CriticVerdict(complete=True, reason="默认通过")
        verdict.round = round_number
        return verdict


class TestCoverage(unittest.TestCase):
    def test_detects_all_four_dimensions(self):
        state = fake_state({
            1: "本文研究汉代袖舞的研究问题与目的，文章旨在考察图像材料。",
            2: "研究方法上采用类型归纳与图像解读的方法，分析路径分三层。",
            3: "史料与文献材料包括汉画像砖石、舞俑壁画等文物遗存。",
            4: "结果表明袖舞分为七种风格，结论是多元共生。",
        })
        report = summarize(state)
        self.assertTrue({dim.name for dim in report.dimensions} == {"problem", "method", "evidence", "results"})
        self.assertTrue(report.all_covered())
        self.assertIn("覆盖：", report.status_line)
        self.assertIn("problem✓", report.status_line)

    def test_single_keyword_spam_is_not_covered(self):
        state = fake_state({1: "本文本文本文本文本文本文"})
        report = summarize(state)
        problem = report.get("problem")
        self.assertEqual(problem.status, "weak")  # 同一个词刷 6 次不算覆盖
        self.assertFalse(report.all_covered())

    def test_missing_dimension_reported(self):
        state = fake_state({1: "本文研究问题与方法：分析路径。"})
        report = summarize(state)
        self.assertIn("evidence", report.missing)
        self.assertIn("results", report.missing)
        self.assertIn("线索", report.missing_line())

    def test_all_covered_needs_enough_pages(self):
        state = fake_state({
            1: "本文研究问题，方法分析，史料材料，结果表明结论。",
        })
        report = summarize(state)
        self.assertFalse(report.all_covered())  # 只读了 1 页，不够触发停止条件

    def test_report_serializes_for_trace(self):
        payload = summarize(fake_state({1: "本文研究问题与方法分析。"})).to_dict()
        self.assertIn("dimensions", payload)
        self.assertIn("all_covered", payload)
        self.assertEqual(len(payload["dimensions"]), 4)


class TestBudgetStage5(unittest.TestCase):
    def make_state(self, **kwargs) -> ReadingState:
        return ReadingState(paper_id="p", paper_path="/x.pdf", pages_total=50, task="t",
                            budget=Budget(**kwargs))

    def test_token_budget_exhausts(self):
        state = self.make_state(max_total_tokens=1000)
        state.input_tokens = 900
        state.output_tokens = 200
        reason = state.budget.token_exhausted(state)
        self.assertIn("token 预算耗尽", reason)
        self.assertIn("1100/1000", reason)
        self.assertEqual(state.budget.violation({"action": "READ"}, state), reason)

    def test_token_budget_disabled_with_zero(self):
        state = self.make_state(max_total_tokens=0)
        state.input_tokens = 10**9
        self.assertIsNone(state.budget.token_exhausted(state))

    def test_progress_line_mentions_tokens(self):
        state = self.make_state(max_total_tokens=5000)
        state.input_tokens = 100
        state.output_tokens = 50
        self.assertIn("tokens=150/5000", state.progress_line())


class TestStopConditions(unittest.TestCase):
    """第五阶段：每一种停法都要有可解释的原因，不许"跑着跑着没了"。"""

    def setUp(self):
        self.pdf = dance_paper()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_agent(self, responses, critic=None, **kwargs):
        client = ScriptedClient(list(responses))
        planner = LlmPlanner(client, system_prompt=build_system_prompt())
        budget = kwargs.pop("budget", Budget())
        agent = PaperAgent(self.pdf, planner=planner, budget=budget, critic=critic, echo=False, **kwargs)
        return agent, client

    def test_token_budget_stops_before_calling_planner(self):
        agent, client = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"READ","pages":"2"}', '{"action":"READ","pages":"3"}'],
            budget=Budget(max_total_tokens=100, max_steps=8),  # 每次脚本调用记 100+20
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "budget_tokens")
        self.assertTrue(result.state.finish_reason.startswith("budget:tokens"))
        self.assertEqual(len(client.calls), 1)  # 花光之后不再问 Planner
        self.assertGreaterEqual(result.state.input_tokens, 100)

    def test_no_progress_stop(self):
        agent, _ = self.make_agent(
            ['{"action":"READ","pages":"1"}'] * 6,
            budget=Budget(max_steps=10, max_no_progress_steps=2),
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "no_progress")
        self.assertEqual(result.state.finish_reason, "budget:no_progress")
        self.assertEqual(result.state.pages_read_count, 1)

    def test_coverage_stop_fires_without_planner_finish(self):
        """覆盖度达标即停：不需要 Planner 自己说 FINISH。"""

        from tools.pdf_reader import open_document

        document = open_document(self.pdf)
        preview = fake_state({i + 1: document["page_texts"][i] for i in range(3)})
        if not summarize(preview).all_covered():
            self.skipTest("这篇论文读 1-3 页尚未达标，换个语料再测")

        agent, client = self.make_agent(
            ['{"action":"READ","pages":"1-3"}', '{"action":"READ","pages":"4-6"}',
             '{"action":"READ","pages":"7"}'],  # 后面的剧本不该被用到
            stop_when_covered=True,
            budget=Budget(max_steps=8),
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "coverage")
        self.assertTrue(result.state.finish_reason.startswith("coverage"))
        self.assertLess(len(client.calls), 3)        # 达标之后不再问 Planner
        self.assertGreaterEqual(len(client.responses), 1)  # 剧本没被用光

    def test_disabled_action_is_refused_for_ablation(self):
        agent, _ = self.make_agent(
            ['{"action":"SEARCH","query":"袖舞","scope":"local"}', '{"action":"FINISH","reason":"够"}'],
            disabled_actions=("SEARCH",),
        )
        result = agent.run()
        self.assertEqual(result.state.searches, 0)          # 检索被禁用 → 没执行
        self.assertEqual(result.state.stop_cause, "agent_finish")
        refused = [obs for obs in result.state.observations if obs.tool == "-"]
        self.assertTrue(refused)
        self.assertIn("禁用", refused[0].summary)

    def test_disable_hint_reaches_prompt(self):
        client = ScriptedClient(['{"action":"FINISH"}'])
        planner = LlmPlanner(client, disabled_actions=("ANALYZE",))
        self.assertIn("禁用", planner.system_prompt)
        self.assertIn("ANALYZE", planner.system_prompt)

    def test_stop_cause_and_coverage_are_recorded(self):
        agent, _ = self.make_agent(['{"action":"READ","pages":"1-2"}', '{"action":"FINISH"}'])
        result = agent.run()
        payload = result.to_dict()
        self.assertEqual(payload["stop_cause"], "agent_finish")
        self.assertIn("dimensions", payload["coverage"])
        self.assertIn("max_total_tokens", payload["budget"])


class TestVerdictParsing(unittest.TestCase):
    def test_parses_roadmap_shape(self):
        payload, error = parse_verdict('{"complete": false, "missing": ["ablation results"], "next_action": "READ"}')
        self.assertIsNone(error)
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["missing"], ["ablation results"])
        self.assertEqual(payload["next_action"], "READ")

    def test_tolerates_fences_and_string_bool(self):
        payload, error = parse_verdict('```json\n{"complete": "true", "missing": [], "reason": "ok"}\n```')
        self.assertIsNone(error)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["next_action"], "FINISH")

    def test_rejects_bad_shapes(self):
        for bad in ["", "没有 JSON", '{"missing": []}', '{"complete": true, "next_action": "SLEEP"}']:
            payload, error = parse_verdict(bad)
            self.assertIsNone(payload, msg=bad)
            self.assertTrue(error)

    def test_missing_can_be_string(self):
        payload, _ = parse_verdict('{"complete": false, "missing": "消融", "next_action": "READ"}')
        self.assertEqual(payload["missing"], ["消融"])


class TestRuleCritic(unittest.TestCase):
    def test_flags_missing_dimensions(self):
        state = fake_state({1: "本文研究问题。"})
        verdict = RuleCritic().review(state, draft="# 草稿\n结论…… [p.1]", coverage=summarize(state))
        self.assertFalse(verdict.complete)
        self.assertTrue(verdict.missing)
        self.assertEqual(verdict.next_action, "READ")
        self.assertEqual(verdict.source, "rules")
        self.assertIn("Critic 认为还不完整", verdict.feedback)

    def test_accepts_when_covered_and_cited(self):
        state = fake_state({
            1: "本文研究问题，方法分析，史料材料，结果表明结论。",
            2: "文章进一步说明研究目的与分析方法，材料包括文献与图像，结论清晰。",
        })
        verdict = RuleCritic().review(state, draft="# 报告\n作者认为…… [p.1]", coverage=summarize(state))
        self.assertTrue(verdict.complete)
        self.assertEqual(verdict.next_action, "FINISH")
        self.assertEqual(verdict.missing, [])

    def test_flags_uncited_draft(self):
        state = fake_state({
            1: "本文研究问题，方法分析，史料材料，结果表明结论。",
            2: "文章进一步说明研究目的与分析方法，材料包括文献与图像，结论清晰。",
        })
        verdict = RuleCritic().review(state, draft="没有任何页码的结论", coverage=summarize(state))
        self.assertFalse(verdict.complete)
        self.assertTrue(any("页码引用" in item for item in verdict.missing))

    def test_make_critic_falls_back_to_rules_without_client(self):
        self.assertIsInstance(make_critic(None, enabled=True), RuleCritic)
        self.assertIsNone(make_critic(None, enabled=False))


class TestLlmCritic(unittest.TestCase):
    def setUp(self):
        self.state = fake_state({1: "本文研究问题。"})

    def test_parses_model_verdict_and_counts_tokens(self):
        client = ScriptedClient(['{"complete": false, "missing": ["缺少消融"], "next_action": "ANALYZE", "reason": "r"}'])
        verdict = LlmCritic(client).review(self.state, draft="草稿 [p.1]", coverage=summarize(self.state))
        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.next_action, "ANALYZE")
        self.assertEqual(verdict.source, "llm")
        self.assertEqual(verdict.input_tokens, 100)

    def test_unparsed_verdict_fails_open(self):
        client = ScriptedClient(["这不是 JSON"])
        verdict = LlmCritic(client).review(self.state, draft="草稿", coverage=summarize(self.state))
        self.assertTrue(verdict.complete)          # Critic 坏了不该把 Agent 拖死
        self.assertEqual(verdict.source, "llm-unparsed")
        self.assertIn("放行", verdict.reason)

    def test_client_error_fails_open(self):
        client = ScriptedClient([])  # 剧本为空 → 直接抛
        verdict = LlmCritic(client).review(self.state, draft="草稿", coverage=summarize(self.state))
        self.assertTrue(verdict.complete)
        self.assertEqual(verdict.source, "llm-error")

    def test_prompt_carries_draft_pages_and_coverage(self):
        client = ScriptedClient(['{"complete": true}'])
        state = fake_state({
            1: "本文研究问题，方法分析，史料材料，结果表明结论。",
            2: "文章进一步说明研究目的与分析方法。",
        })
        state.observations = []
        LlmCritic(client).review(state, draft="# 草稿\n结论 [p.1]", coverage=summarize(state))
        prompt = client.prompts[0]
        self.assertIn("[p.1]", prompt)
        self.assertIn("覆盖度线索", prompt)
        self.assertIn("complete", prompt)


class TestCriticLoop(unittest.TestCase):
    """第六阶段：FINISH → Critic → 通过就收，不通过就带着缺口回到 Planner。"""

    def setUp(self):
        self.pdf = dance_paper()

    def make_agent(self, responses, critic, **kwargs):
        client = ScriptedClient(list(responses))
        planner = LlmPlanner(client, system_prompt=build_system_prompt())
        agent = PaperAgent(self.pdf, planner=planner, critic=critic, echo=False,
                           budget=kwargs.pop("budget", Budget(max_steps=8)), **kwargs)
        return agent, client

    def test_reject_then_accept(self):
        critic = FakeCritic([
            CriticVerdict(complete=False, missing=["缺少结果类内容"], next_action="READ", reason="先补结果"),
            CriticVerdict(complete=True, reason="已经足够"),
        ])
        agent, client = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"FINISH","reason":"够了"}',
             '{"action":"READ","pages":"2"}', '{"action":"FINISH","reason":"这回真的够了"}'],
            critic,
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "critic_accept")
        self.assertTrue(result.state.finish_reason.startswith("critic:accept:"))
        self.assertEqual(len(result.state.critic_rounds), 2)
        self.assertEqual([v["complete"] for v in result.state.critic_rounds], [False, True])
        # 第二次 FINISH 之前的 Planner 调用应当带着 Critic 的缺口
        self.assertIn("Critic", client.prompts[2])
        self.assertIn("缺少结果类内容", client.prompts[2])
        self.assertGreaterEqual(result.state.pages_read_count, 2)  # 确实回去补读了

    def test_exhausted_after_max_rounds(self):
        critic = FakeCritic([
            CriticVerdict(complete=False, missing=["缺 A"], next_action="READ"),
            CriticVerdict(complete=False, missing=["缺 B"], next_action="READ"),
            CriticVerdict(complete=False, missing=["缺 C"], next_action="READ"),
        ])
        agent, _ = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"FINISH"}', '{"action":"FINISH"}', '{"action":"FINISH"}'],
            critic, max_critic_rounds=2,
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "critic_exhausted")
        self.assertEqual(result.state.finish_reason, "critic:exhausted(2)")
        self.assertEqual(len(result.state.critic_rounds), 2)
        self.assertTrue(any("Critic" in error for error in result.state.errors))

    def test_redraft_carries_critic_gaps(self):
        """Critic 的缺口必须进下一版草稿的 prompt，否则第二版会原样重犯（实跑里踩到过）。"""

        critic = FakeCritic([
            CriticVerdict(complete=False, missing=["第6节只覆盖了前两类范型"], next_action="READ"),
            CriticVerdict(complete=True),
        ])
        agent, client = self.make_agent(
            [
                '{"action":"READ","pages":"1"}',
                '{"action":"FINISH","reason":"第一版"}',
                "# 草稿 v1\n结论…… [p.1]",
                '{"action":"READ","pages":"2"}',
                '{"action":"FINISH","reason":"第二版"}',
                "# 草稿 v2\n结论…… [p.1][p.2]",
            ],
            critic, write_report=True,
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "critic_accept")
        self.assertEqual(result.state.report, "# 草稿 v2\n结论…… [p.1][p.2]")
        redraft_prompt = client.prompts[5]
        self.assertIn("上一版被 Critic 指出的缺口", redraft_prompt)
        self.assertIn("第6节只覆盖了前两类范型", redraft_prompt)

    def test_feedback_asks_rewrite_when_nothing_left_to_read(self):
        """缺口在"写"而不在"读"时，Critic 的反馈要让 Agent 去重写，而不是去重复读旧页。"""

        self.assertIn("重写", CriticVerdict(complete=False, missing=["缺"], next_action="FINISH").feedback)
        state = fake_state({1: "本文研究问题。"})
        state.read_pages = set(range(1, 11))          # 10 页全读完
        verdict = RuleCritic().review(state, draft="# 草稿 [p.1]", coverage=summarize(state))
        self.assertEqual(verdict.next_action, "FINISH")   # 有草稿 + 没未读页 → 让它重写
        self.assertIn("重写", verdict.feedback)

        # 还没写过草稿时，退回"去语料里找找"是合理的
        no_draft = RuleCritic().review(state, draft=None, coverage=summarize(state))
        self.assertEqual(no_draft.next_action, "SEARCH")

    def test_critic_prompt_says_when_nothing_is_left(self):
        state = fake_state({1: "本文研究问题。"})
        state.read_pages = set(range(1, 11))
        prompt = build_critic_prompt(state, draft="草稿 [p.1]", coverage=summarize(state))[1]["content"]
        self.assertIn("未读页：无", prompt)

    def test_best_draft_is_kept_when_stopped_by_no_progress(self):
        """预算/无进展停掉时也要留下最后一版草稿，否则第七/八阶段无从打分。"""

        critic = FakeCritic([CriticVerdict(complete=False, missing=["缺结果"], next_action="READ")])
        agent, _ = self.make_agent(
            [
                '{"action":"READ","pages":"1-3"}',
                '{"action":"FINISH","reason":"第一版"}',
                "# 草稿 v1\n主要结果…… [p.1]",
                '{"action":"READ","pages":"1"}',   # 没有新页 → 无进展
            ],
            critic,
            write_report=True,
            budget=Budget(max_steps=8, max_no_progress_steps=1),
        )
        result = agent.run()
        self.assertEqual(result.state.stop_cause, "no_progress")
        self.assertEqual(result.state.report, "# 草稿 v1\n主要结果…… [p.1]")

    def test_critic_sees_page_cited_draft(self):
        critic = FakeCritic([CriticVerdict(complete=True)])
        agent, _ = self.make_agent(['{"action":"READ","pages":"1"}', '{"action":"FINISH"}'], critic)
        agent.run()
        self.assertIn("[p.1]", critic.calls[0]["draft"])   # 离线草稿=证据包，带页码
        self.assertIn("覆盖：", critic.calls[0]["coverage"])

    def test_no_critic_keeps_stage4_behavior(self):
        agent, _ = self.make_agent(['{"action":"READ","pages":"1"}', '{"action":"FINISH","reason":"ok"}'], critic=None)
        result = agent.run()
        self.assertIsNone(agent.critic)
        self.assertEqual(result.state.stop_cause, "agent_finish")
        self.assertEqual(result.state.critic_rounds, [])

    def test_critic_events_land_in_trace(self):
        critic = FakeCritic([CriticVerdict(complete=False, missing=["缺 A"], next_action="READ"),
                             CriticVerdict(complete=True)])
        agent, _ = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"FINISH"}', '{"action":"FINISH"}'], critic,
        )
        result = agent.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_trace(result, path)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            critic_events = [record for record in records if record["event"] == "critic"]
            self.assertEqual(len(critic_events), 2)
            self.assertFalse(critic_events[0]["complete"])
            self.assertIn("after_step", critic_events[0])

            run = trace_utils.load_trace(path)
            self.assertEqual(run.metrics()["critic_rounds"], 2)
            self.assertEqual(run.metrics()["stop_cause"], "critic_accept")
            self.assertIn("Critic 判词", trace_utils.render_run(run))


class TestStage56Cli(unittest.TestCase):
    """CLI 开关：一律 `--no-config`，别碰工作区里的真 key。"""

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        argv = list(args)
        if "--config" not in argv:
            argv.append("--no-config")
        return subprocess.run([sys.executable, "-m", "agent.agent", *argv], cwd=str(REPO_ROOT),
                              capture_output=True, text=True)

    def test_no_critic_flag(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--json", "--quiet",
                            "--no-critic", "--max-steps", "3")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertNotIn("critic_", payload["stop_cause"])
        self.assertEqual(payload["critic_rounds"], [])

    def test_default_has_rules_critic(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--json", "--quiet",
                            "--max-steps", "5")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertGreaterEqual(len(payload["critic_rounds"]), 1)   # 默认系统带 Critic
        self.assertIn(payload["stop_cause"], {"critic_accept", "critic_exhausted"})

    def test_stop_when_covered_flag(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--json", "--quiet",
                            "--stop-when-covered", "--no-critic", "--max-steps", "6")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn(payload["stop_cause"], {"coverage", "agent_finish", "budget_steps"})

    def test_disable_flag_blocks_action(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--json", "--quiet",
                            "--disable", "SEARCH", "--max-steps", "3")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["searches"], 0)

    def test_disable_rejects_unknown_action(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--quiet",
                            "--disable", "SLEEP")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--disable", proc.stderr)

    def test_token_budget_flag_is_wired(self):
        proc = self.run_cli("--paper", str(dance_paper()), "--planner", "heuristic", "--json", "--quiet",
                            "--no-critic", "--max-total-tokens", "12345", "--max-steps", "2")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["budget"]["max_total_tokens"], 12345)
        self.assertEqual(payload["budget"]["max_no_progress_steps"], 3)  # 默认无进展阈值


if __name__ == "__main__":
    unittest.main(verbosity=2)
