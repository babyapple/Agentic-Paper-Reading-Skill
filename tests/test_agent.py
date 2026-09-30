#!/usr/bin/env python3
"""Agent Loop 回归测试（默认不联网；只有 mock 服务那条会在 127.0.0.1 起一个假 LLM）。

跑法：

    python3 -m unittest discover -s tests -v

覆盖：
1. action 解析与校验的"脏输入"容错（围栏、散文、tool_calls 形态、错字段）；
2. prompt 组装：Skill 是否真的被注入、进度与页码是否进上下文；
3. 预算：步数 / 搜索 / 页数 / 工具调用，以及"拒绝而不是崩掉"；
4. Executor：READ / SEARCH / ANALYZE 正常路径与失败路径都变成 observation；
5. 主循环：完整轨迹、解析失败恢复与放弃、预算耗尽、重复动作空转保护、报告生成；
6. LLM 客户端：用本地 mock 服务验证请求头/请求体/用量解析/重试与鉴权失败。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agent import PaperAgent, run_agent, write_trace  # noqa: E402
from agent.executor import Executor  # noqa: E402
from agent.llm import (  # noqa: E402
    ChatResult,
    ClientConfig,
    LlmError,
    OpenAICompatibleClient,
    ScriptedClient,
    parse_env_file,
    resolve_config,
)
from agent.planner import (  # noqa: E402
    HeuristicPlanner,
    LlmPlanner,
    build_report_prompt,
    build_system_prompt,
    build_user_message,
    parse_action,
    validate_action,
)
from agent.state import Budget, Observation, ReadingState, compress_pages  # noqa: E402
from agent import trace_utils  # noqa: E402

PAPERS_DIR = REPO_ROOT / "paper"
FIXTURE_TABLE = REPO_ROOT / "tests" / "fixtures" / "ablation.md"


def any_paper() -> Path | None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf")) if PAPERS_DIR.exists() else []
    return pdfs[0] if pdfs else None


def needs_paper(test) -> Path:
    pdf = any_paper()
    if pdf is None:
        test.skipTest("没有可用论文语料")
    return pdf


class TestActionParsing(unittest.TestCase):
    def test_plain_and_fenced_and_prose(self):
        self.assertEqual(parse_action('{"action":"READ","pages":"1-2"}')[0]["action"], "READ")
        self.assertEqual(parse_action('```json\n{"action":"read","pages":"1-2"}\n```')[0]["action"], "READ")
        action, error = parse_action('我先读开头：{"action":"READ","pages":"1-3"} 就这样')
        self.assertIsNone(error)
        self.assertEqual(action["pages"], "1-3")

    def test_tool_call_shapes_are_accepted(self):
        cases = [
            {"name": "read_pdf", "arguments": {"pages": "4"}},
            {"name": "search_paper", "arguments": json.dumps({"query": "袖舞"})},
            {"tool": "analyze_table", "args": {"operation": "extract"}},
            {"function": {"name": "read_pdf", "arguments": json.dumps({"pages": "5-6"})}},
        ]
        expected = ["READ", "SEARCH", "ANALYZE", "READ"]
        for payload, want in zip(cases, expected):
            action, error = parse_action(json.dumps(payload, ensure_ascii=False))
            self.assertIsNone(error, msg=f"{payload} → {error}")
            self.assertEqual(action["action"], want)

    def test_nested_braces_inside_strings(self):
        text = 'note {"action":"SEARCH","query":"{weird} term"} end'
        action, error = parse_action(text)
        self.assertIsNone(error)
        self.assertEqual(action["query"], "{weird} term")

    def test_invalid_inputs_return_readable_errors(self):
        for bad in ["", "   ", "no json at all", '{"action":"DANCE"}']:
            action, error = parse_action(bad)
            self.assertIsNone(action)
            self.assertTrue(error)

    def test_validation_rules(self):
        self.assertIn("query", validate_action({"action": "SEARCH"}))
        self.assertIn("pages", validate_action({"action": "READ", "pages": "第一页"}))
        self.assertIn("table", validate_action({"action": "ANALYZE", "operation": "rank"}))
        self.assertIn("operation", validate_action({"action": "ANALYZE", "table": "table:1"}))
        self.assertIn("scope", validate_action({"action": "SEARCH", "query": "x", "scope": "pubmed"}))
        self.assertIsNone(validate_action({"action": "ANALYZE", "operation": "extract"}))
        self.assertIsNone(validate_action({"action": "FINISH"}))
        self.assertIsNone(validate_action({"action": "READ"}))


class TestPrompts(unittest.TestCase):
    def test_system_prompt_injects_skill_and_contract(self):
        prompt = build_system_prompt()
        self.assertIn("动作空间", prompt)
        self.assertIn("READ", prompt)
        self.assertIn("报告契约", prompt)      # 来自 SKILL.md
        self.assertIn("不编造", prompt)        # 来自 SKILL.md 的硬规则
        self.assertIn("只输出一个 JSON 对象", prompt)

    def test_user_message_carries_progress_and_history(self):
        state = ReadingState(paper_id="p", paper_path="/x.pdf", pages_total=9, task="分析它")
        state.mark_read([1, 2], {1: "题目与摘要", 2: "正文"})
        state.record_tool(
            Observation(step=1, action="READ", tool="read_pdf", args={}, ok=True,
                        summary="read_pdf(pages=1-2)", detail={"prompt_text": "[Page 1]\n题目与摘要"})
        )
        message = build_user_message(state)
        self.assertIn("分析它", message)
        self.assertIn("已读 2/9 页", message)
        self.assertIn("[Page 1]", message)

    def test_report_prompt_is_page_cited(self):
        state = ReadingState(paper_id="p", paper_path="/x.pdf", pages_total=3, task="分析它")
        state.mark_read([2], {2: "第二页的关键内容"})
        messages = build_report_prompt(state)
        self.assertIn("[p.2]", messages[1]["content"])
        self.assertIn("报告模板", messages[0]["content"])


class TestBudget(unittest.TestCase):
    def make_state(self, **kwargs) -> ReadingState:
        return ReadingState(paper_id="p", paper_path="/x.pdf", pages_total=50, task="t",
                            budget=Budget(**kwargs))

    def test_steps_limit(self):
        state = self.make_state(max_steps=2)
        state.steps = 2
        self.assertIn("步数", state.budget.violation({"action": "READ"}, state))

    def test_search_limit(self):
        state = self.make_state(max_searches=1)
        state.searches = 1
        self.assertIn("检索次数", state.budget.violation({"action": "SEARCH", "query": "x"}, state))

    def test_pages_limit_counts_only_new_pages(self):
        state = self.make_state(max_pages=4)
        state.mark_read([1, 2, 3], {})
        self.assertIsNone(state.budget.violation({"action": "READ"}, state, pages=[1, 2, 3]))
        self.assertIn("阅读页数预算", state.budget.violation({"action": "READ"}, state, pages=[4, 5]))

    def test_tool_call_limit(self):
        state = self.make_state(max_tool_calls=3)
        state.tool_calls = 3
        self.assertIn("工具调用", state.budget.violation({"action": "SEARCH", "query": "x"}, state))

    def test_compress_pages(self):
        self.assertEqual(compress_pages([3, 1, 2, 5, 8, 9]), "1-3,5,8-9")
        self.assertEqual(compress_pages([]), "")


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.pdf = needs_paper(self)
        self.executor = Executor(self.pdf)
        self.state = ReadingState(paper_id=self.pdf.stem, paper_path=str(self.pdf), pages_total=0, task="t")
        self.state.record_tool(self.executor.bootstrap(self.state))

    def test_bootstrap_fills_outline_and_pages(self):
        self.assertGreater(self.state.pages_total, 1)
        self.assertTrue(self.state.outline)

    def test_read_pages_and_prompt_text(self):
        observation = self.executor.do_read({"action": "READ", "pages": "1"}, self.state)
        self.assertTrue(observation.ok)
        self.assertEqual(self.state.read_pages, {1})
        self.assertIn("[Page 1]", observation.detail["prompt_text"])
        self.assertEqual(self.state.pages_read_spec, "1")

    def test_read_defaults_to_next_unread_pages(self):
        observation = self.executor.do_read({"action": "READ"}, self.state)
        self.assertTrue(observation.ok)
        self.assertEqual(sorted(self.state.read_pages), [1, 2])

    def test_read_bad_range_is_observation_not_exception(self):
        observation = self.executor.do_read({"action": "READ", "pages": "999"}, self.state)
        self.assertFalse(observation.ok)
        self.assertIn("超出总页数", observation.error)

    def test_search_local(self):
        observation = self.executor.execute({"action": "SEARCH", "query": "舞蹈", "scope": "local"}, self.state)
        self.state.record_tool(observation)
        self.assertTrue(observation.ok)
        self.assertIn("命中", observation.summary)
        self.assertEqual(self.state.searches, 1)
        self.assertEqual(self.state.tool_calls, 2)  # bootstrap + 这次检索

    def test_search_without_query_fails_cleanly(self):
        observation = self.executor.do_search({"action": "SEARCH", "query": "  "}, self.state)
        self.assertFalse(observation.ok)
        self.assertIn("query", observation.error)

    def test_refused_action_does_not_consume_budget(self):
        refused = Observation(step=1, action="SEARCH", tool="-", args={}, ok=False, summary="拒绝执行", error="超预算")
        before_calls, before_searches = self.state.tool_calls, self.state.searches
        self.state.record_tool(refused)
        self.assertEqual(self.state.tool_calls, before_calls)
        self.assertEqual(self.state.searches, before_searches)
        self.assertTrue(self.state.errors)

    def test_analyze_extract_then_rank(self):
        # 造一段"已读正文"，里面放一个 Markdown 结果表
        self.state.page_texts = {1: FIXTURE_TABLE.read_text(encoding="utf-8")}
        self.state.read_pages = {1}
        extract = self.executor.do_analyze({"action": "ANALYZE", "operation": "extract"}, self.state)
        self.assertTrue(extract.ok)
        self.assertEqual(extract.detail["table_count"], 1)
        self.assertIn("table:1", self.state.tables)

        rank = self.executor.do_analyze(
            {"action": "ANALYZE", "operation": "rank", "table": "table:1", "by": "Accuracy", "label": "Method"},
            self.state,
        )
        self.assertTrue(rank.ok)
        self.assertIn("Full Agent", rank.detail["prompt_text"])
        self.assertIn("第 1 名", rank.summary)

    def test_analyze_compress_and_compare_paths(self):
        self.state.page_texts = {1: FIXTURE_TABLE.read_text(encoding="utf-8")}
        self.state.read_pages = {1}
        self.executor.do_analyze({"action": "ANALYZE", "operation": "extract"}, self.state)
        compare = self.executor.do_analyze(
            {"action": "ANALYZE", "operation": "compare", "table": "table:1",
             "baseline": "Baseline", "label": "Method", "lower_is_better": "Cost"},
            self.state,
        )
        self.assertTrue(compare.ok)
        self.assertEqual(compare.detail["payload"]["operation"], "compare_rows")

    def test_analyze_unknown_table_lists_candidates(self):
        observation = self.executor.do_analyze(
            {"action": "ANALYZE", "operation": "describe", "table": "table:7"}, self.state
        )
        self.assertFalse(observation.ok)
        self.assertIn("table:7", observation.error)

    def test_analyze_bad_operation(self):
        observation = self.executor.do_analyze({"action": "ANALYZE", "operation": "vibe"}, self.state)
        self.assertFalse(observation.ok)
        self.assertIn("operation", observation.error)

    def test_unknown_action_is_observation(self):
        observation = self.executor.execute({"action": "SLEEP"}, self.state)
        self.assertFalse(observation.ok)
        self.assertIn("READ / SEARCH / ANALYZE / FINISH", observation.summary)


class TestLoopWithScriptedPlanner(unittest.TestCase):
    def setUp(self):
        self.pdf = needs_paper(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = Path(self.tmp.name) / "run.jsonl"

    def make_agent(self, responses, **kwargs):
        client = ScriptedClient(list(responses))
        planner = LlmPlanner(client, system_prompt=build_system_prompt())
        budget = kwargs.pop("budget", Budget())
        agent = PaperAgent(self.pdf, planner=planner, budget=budget, echo=False, **kwargs)
        return agent, client

    def test_full_trajectory_read_search_analyze_finish(self):
        agent, client = self.make_agent(
            [
                '{"action":"READ","pages":"1-2"}',
                '{"action":"SEARCH","query":"舞蹈","scope":"local"}',
                '{"action":"ANALYZE","operation":"extract"}',
                '{"action":"FINISH","reason":"信息已足够"}',
            ]
        )
        result = agent.run()
        state = result.state
        self.assertEqual(state.steps, 4)
        self.assertEqual(state.tool_calls, 4)  # bootstrap + 三个动作
        self.assertEqual(state.searches, 1)
        self.assertTrue(state.finish_reason.startswith("agent:"))
        self.assertEqual(sorted(state.read_pages), [1, 2])
        self.assertEqual(state.input_tokens, 400)  # 4 次脚本调用 × 100
        self.assertIn("报告契约", client.prompts[0])

        write_trace(result, self.trace)
        lines = [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), state.tool_calls + 1)
        self.assertEqual(lines[-1]["event"], "summary")
        self.assertEqual(lines[-1]["tool_calls"], state.tool_calls)

    def test_report_generation_after_finish(self):
        agent, client = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"FINISH"}', "## 1. 研究问题\n作者讨论…… [p.1]"],
            write_report=True,
        )
        result = agent.run()
        self.assertIn("[p.1]", result.state.report)
        self.assertEqual(result.state.answer, result.state.report)
        self.assertEqual(len(client.responses), 0)  # 剧本用尽且恰好收尾

    def test_offline_answer_is_evidence_bundle(self):
        agent, _ = self.make_agent(['{"action":"READ","pages":"1"}', '{"action":"FINISH"}'])
        result = agent.run()
        self.assertIsNone(result.state.report)
        self.assertIn("证据包", result.state.answer)

    def test_parse_error_recovery(self):
        agent, client = self.make_agent(["这不是 JSON", '{"action":"FINISH","reason":"ok"}'])
        result = agent.run()
        self.assertTrue(result.state.finish_reason.startswith("agent:"))
        self.assertTrue(any("找不到合法 JSON" in err for err in result.state.errors))
        self.assertEqual(len(client.calls), 2)

    def test_parse_error_gives_up(self):
        agent, _ = self.make_agent(["nope", "nope", "nope", "nope"], max_parse_retries=2)
        result = agent.run()
        self.assertTrue(result.state.finish_reason.startswith("planner_error:"))

    def test_budget_stops_the_loop(self):
        agent, _ = self.make_agent(
            ['{"action":"READ","pages":"1"}', '{"action":"READ","pages":"2"}',
             '{"action":"READ","pages":"3"}', '{"action":"READ","pages":"4"}'],
            budget=Budget(max_steps=2, max_pages=20),
        )
        result = agent.run()
        self.assertEqual(result.state.finish_reason, "budget:max_steps")
        self.assertEqual(result.state.steps, 2)

    def test_search_budget_is_enforced(self):
        agent, _ = self.make_agent(
            ['{"action":"SEARCH","query":"a"}', '{"action":"SEARCH","query":"b","scope":"local"}',
             '{"action":"SEARCH","query":"c","scope":"local"}', '{"action":"SEARCH","query":"d","scope":"local"}'],
            budget=Budget(max_searches=1, max_steps=6),
        )
        result = agent.run()
        self.assertEqual(result.state.searches, 1)
        self.assertIn("blocked_loop", result.state.finish_reason)

    def test_repeated_successful_action_is_guarded(self):
        responses = ['{"action":"READ","pages":"1"}'] * 5
        agent, _ = self.make_agent(responses, budget=Budget(max_steps=8))
        result = agent.run()
        self.assertEqual(result.state.pages_read_count, 1)  # 没有真的反复读同一页
        # 重复读同一页既是"重复动作"也是"没有新信息"，两条空转保护谁先触发都算对
        self.assertIn(result.state.stop_cause, {"blocked_loop", "no_progress"})
        self.assertIn(result.state.finish_reason, {"budget:no_progress"})
        self.assertTrue(result.state.finish_reason.startswith("budget:"))

    def test_llm_failure_is_reported_not_crashed(self):
        agent, _ = self.make_agent([])  # 剧本为空 → 第一次调用就抛
        result = agent.run()
        self.assertTrue(result.state.finish_reason.startswith("planner_error:"))
        self.assertTrue(result.state.errors)

    def test_missing_paper_raises(self):
        agent = PaperAgent(REPO_ROOT / "paper" / "nope.pdf", planner=HeuristicPlanner(), echo=False)
        with self.assertRaises(Exception):
            agent.run()


class TestHeuristicLoop(unittest.TestCase):
    def test_offline_loop_runs_end_to_end(self):
        pdf = needs_paper(self)
        result = run_agent(pdf, echo=False)
        state = result.state
        self.assertTrue(state.finish_reason.startswith("agent:"))
        self.assertGreater(state.pages_read_count, 0)
        self.assertGreater(state.tool_calls, 1)
        self.assertTrue(state.answer)
        self.assertEqual(state.input_tokens, 0)  # 离线不产生 token 记账

    def test_trace_file_written(self):
        pdf = needs_paper(self)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "traces" / "run.jsonl"
            result = run_agent(pdf, echo=False, trace_path=target)
            lines = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), result.state.tool_calls + 1)


class _MockChatHandler(BaseHTTPRequestHandler):
    """一个最小的 OpenAI 兼容假服务：记录收到的请求，按脚本返回。"""

    plan: list[tuple[int, dict]] = []
    received: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server 约定
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).received.append({"body": body, "auth": self.headers.get("Authorization"), "path": self.path})
        status, payload = type(self).plan.pop(0) if type(self).plan else (200, {"choices": [{"message": {"content": "{}"}}]})
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 静音
        return


class TestOpenAiCompatibleClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockChatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _MockChatHandler.received.clear()
        _MockChatHandler.plan = []

    def client(self, **kwargs) -> OpenAICompatibleClient:
        config = ClientConfig(base_url=self.base_url, model="test-model", api_key="test-key", **kwargs)
        return OpenAICompatibleClient(config, sleep=lambda _seconds: None)

    def test_request_shape_and_usage_parsing(self):
        _MockChatHandler.plan = [
            (200, {"model": "test-model", "choices": [{"message": {"content": '{"action":"FINISH"}'}}],
                   "usage": {"prompt_tokens": 111, "completion_tokens": 22}})
        ]
        result = self.client().complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result.text, '{"action":"FINISH"}')
        self.assertEqual(result.input_tokens, 111)
        self.assertEqual(result.output_tokens, 22)
        sent = _MockChatHandler.received[0]
        self.assertEqual(sent["auth"], "Bearer test-key")
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["body"]["model"], "test-model")
        self.assertEqual(sent["body"]["messages"][0]["content"], "hi")

    def test_retry_on_429_then_success(self):
        _MockChatHandler.plan = [
            (429, {"error": "rate limited"}),
            (200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
        result = self.client(max_retries=1).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result.text, "ok")
        self.assertEqual(len(_MockChatHandler.received), 2)

    def test_auth_error_raises_llm_error(self):
        _MockChatHandler.plan = [(401, {"error": "bad key"})]
        with self.assertRaises(LlmError):
            self.client(max_retries=2).complete([{"role": "user", "content": "hi"}])

    def test_empty_choices_raises(self):
        _MockChatHandler.plan = [(200, {"choices": []})]
        with self.assertRaises(LlmError):
            self.client().complete([{"role": "user", "content": "hi"}])

    def test_resolve_config_reads_env(self):
        config, note = resolve_config(env={"DEEPSEEK_API_KEY": "k"}, use_config=False)
        self.assertIsNotNone(config)
        self.assertIn("deepseek", config.base_url)
        self.assertIn("环境变量", note)
        missing, reason = resolve_config(env={}, use_config=False)
        self.assertIsNone(missing)
        self.assertIn("API key", reason)


class TestConfigFile(unittest.TestCase):
    """工作区配置文件（.llm.env）：不设全局变量也能把 key/base_url 配好。"""

    def write_config(self, tmp: str, text: str) -> Path:
        path = Path(tmp) / ".llm.env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_parse_env_file_tolerates_real_world_edits(self):
        values = parse_env_file(
            "# 注释行\n"
            "\n"
            "export LLM_API_KEY = \"sk-abc\"   # 行尾注释\n"
            "LLM_BASE_URL=https://api.deepseek.com/v1\n"
            "LLM_MODEL='deepseek-chat'\r\n"
            "这行没有等号\n"
            "LLM_TIMEOUT=30\n"
        )
        self.assertEqual(values["LLM_API_KEY"], "sk-abc")
        self.assertEqual(values["LLM_MODEL"], "deepseek-chat")
        self.assertEqual(values["LLM_TIMEOUT"], "30")
        self.assertNotIn("这行没有等号", values)

    def test_quoted_value_keeps_hash_inside_url(self):
        values = parse_env_file("LLM_BASE_URL=https://x.example/v1#frag\n")
        self.assertEqual(values["LLM_BASE_URL"], "https://x.example/v1#frag")

    def test_missing_explicit_config_raises(self):
        with self.assertRaises(LlmError):
            resolve_config(config_path="/tmp/definitely-not-here.env", env={})

    def test_file_provides_key_and_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(
                tmp,
                "LLM_API_KEY=sk-from-file\n"
                "LLM_BASE_URL=https://example.test/v1\n"
                "LLM_MODEL=my-model\n"
                "LLM_TIMEOUT=12\n"
                "LLM_MAX_RETRIES=5\n"
                "LLM_MAX_TOKENS=777\n",
            )
            config, note = resolve_config(config_path=path, env={})
        self.assertEqual(config.api_key, "sk-from-file")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "my-model")
        self.assertEqual(config.timeout, 12.0)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.max_tokens, 777)
        self.assertIn(".llm.env", note)
        self.assertNotIn("sk-from-file", note)  # 说明里不回显 key

    def test_precedence_cli_over_file_over_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(
                tmp, "LLM_API_KEY=sk-file\nLLM_BASE_URL=https://file.test/v1\nLLM_MODEL=file-model\n"
            )
            from_file, _ = resolve_config(config_path=path, env={"LLM_API_KEY": "sk-env", "LLM_MODEL": "env-model"})
            self.assertEqual(from_file.api_key, "sk-file")
            self.assertEqual(from_file.model, "file-model")

            from_cli, _ = resolve_config(config_path=path, api_key="sk-cli", model="cli-model", env={})
            self.assertEqual(from_cli.api_key, "sk-cli")
            self.assertEqual(from_cli.model, "cli-model")
            self.assertEqual(from_cli.base_url, "https://file.test/v1")  # 没给的仍回落到文件

    def test_env_fallback_when_no_file(self):
        config, note = resolve_config(env={"LLM_API_KEY": "sk-env"}, use_config=False)
        self.assertEqual(config.api_key, "sk-env")
        self.assertIn("环境变量", note)

    def test_empty_key_in_file_counts_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, "LLM_API_KEY=\nLLM_BASE_URL=https://x.test/v1\n")
            config, reason = resolve_config(config_path=path, env={})
        self.assertIsNone(config)
        self.assertIn("没有 LLM_API_KEY", reason)

    def test_workspace_template_and_gitignore_are_consistent(self):
        example = REPO_ROOT / ".llm.env.example"
        self.assertTrue(example.exists())
        self.assertIn("LLM_API_KEY", example.read_text(encoding="utf-8"))
        proc = subprocess.run(["git", "check-ignore", "-q", ".llm.env"], cwd=str(REPO_ROOT))
        self.assertEqual(proc.returncode, 0, msg=".llm.env 必须在 .gitignore 里")


class TestCli(unittest.TestCase):
    """CLI 测试必须**完全离线**：工作区里可能有真 key，所以默认一律加 `--no-config`。"""

    def run_cli(self, *args: str, env: dict | None = None, allow_config: bool = False) -> subprocess.CompletedProcess:
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith(("LLM_", "OPENAI_", "DEEPSEEK_", "ARK_", "MOONSHOT_"))}
        clean_env.update(env or {})
        argv = list(args)
        if not allow_config and "--config" not in argv and "--no-config" not in argv:
            argv.append("--no-config")
        return subprocess.run([sys.executable, "-m", "agent.agent", *argv], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, env=clean_env)

    def test_heuristic_json_run(self):
        pdf = needs_paper(self)
        proc = self.run_cli("--paper", str(pdf), "--planner", "heuristic", "--json", "--quiet",
                            "--max-steps", "3")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["finish_reason"])
        self.assertGreater(payload["tool_calls"], 0)

    def test_llm_without_key_fails_closed(self):
        pdf = needs_paper(self)
        proc = self.run_cli("--paper", str(pdf), "--planner", "llm", "--quiet")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1)
        self.assertIn("API key", proc.stderr)

    def test_config_file_supplies_key(self):
        """把 key 写在 --config 指向的文件里：应当真的用上它去调模型（这里指向死端口）。"""
        pdf = needs_paper(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".llm.env"
            path.write_text(
                "LLM_API_KEY=sk-from-config\n"
                "LLM_BASE_URL=http://127.0.0.1:9/v1\n"
                "LLM_MODEL=config-model\n"
                "LLM_TIMEOUT=2\n",
                encoding="utf-8",
            )
            proc = self.run_cli(
                "--paper", str(pdf), "--json", "--quiet", "--max-steps", "2",
                "--max-retries", "0", "--config", str(path),
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("planner_error", payload["finish_reason"])  # 说明确实尝试调用了模型

    def test_missing_config_exit_2(self):
        pdf = needs_paper(self)
        proc = self.run_cli("--paper", str(pdf), "--quiet", "--config", "/tmp/definitely-missing.env")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("配置文件不存在", proc.stderr)

    def test_config_discovered_via_env_var(self):
        """LLM_CONFIG_FILE 指向的配置文件也会被读到（不碰工作区里那个真的 .llm.env）。"""
        pdf = needs_paper(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "somewhere.env"
            path.write_text(
                "LLM_API_KEY=sk-discovered\n"
                "LLM_BASE_URL=http://127.0.0.1:9/v1\n"
                "LLM_TIMEOUT=2\n",
                encoding="utf-8",
            )
            proc = self.run_cli(
                "--paper", str(pdf), "--json", "--quiet", "--max-steps", "2", "--max-retries", "0",
                env={"LLM_CONFIG_FILE": str(path)}, allow_config=True,
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("planner_error", payload["finish_reason"])

    def test_auto_falls_back_to_heuristic(self):
        pdf = needs_paper(self)
        proc = self.run_cli("--paper", str(pdf), "--json", "--quiet", "--max-steps", "3")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("离线启发式", proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["finish_reason"].startswith(("agent:", "critic:")))

    def test_missing_paper_exit_2(self):
        proc = self.run_cli("--paper", "paper/nope.pdf", "--planner", "heuristic", "--quiet")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("agent:", proc.stderr)

    def test_defensive_against_real_network_in_tests(self):
        """回归保护：测试一旦忘记 --no-config，就会真的去打工作区配置里的 API。"""

        pdf = needs_paper(self)
        proc = self.run_cli("--paper", str(pdf), "--planner", "llm", "--quiet", "--no-config")
        self.assertEqual(proc.returncode, 2, msg="工作区 .llm.env 不应影响测试")

    def test_env_key_is_detected_as_llm_planner(self):
        pdf = needs_paper(self)
        # 指向不存在的服务：应当以"模型调用失败"收尾，而不是崩溃
        proc = self.run_cli(
            "--paper", str(pdf), "--json", "--quiet", "--max-steps", "2",
            "--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--timeout", "2",
            "--max-retries", "0",
            env={"LLM_API_KEY": "dummy"},
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("planner_error", payload["finish_reason"])


class TestTraceUtils(unittest.TestCase):
    """trace 要能变成"动作路径 + 指标"，否则第四阶段的产物没法进后续实验。"""

    def make_trace(self, tmp: Path) -> Path:
        path = tmp / "run.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"event": "tool_call", "step": 0, "action": "BOOTSTRAP", "tool": "read_pdf",
                                "ok": True, "summary": "info+outline", "duration_ms": 1, "args": {}, "error": None}),
                    json.dumps({"event": "tool_call", "step": 1, "action": "READ", "tool": "read_pdf",
                                "ok": True, "summary": "读了 1-2 页", "duration_ms": 2, "args": {}, "error": None}),
                    json.dumps({"event": "tool_call", "step": 2, "action": "SEARCH", "tool": "search_paper",
                                "ok": True, "summary": "命中 1 篇", "duration_ms": 3, "args": {}, "error": None}),
                    json.dumps({"event": "tool_call", "step": 3, "action": "ANALYZE", "tool": "analyze_table",
                                "ok": False, "summary": "没有表", "duration_ms": 4, "args": {}, "error": "no table"}),
                    json.dumps({"event": "summary", "steps": 4, "tool_calls": 4, "searches": 1, "pages_read": "1-2",
                                "input_tokens": 900, "output_tokens": 120, "llm_latency_ms": 42,
                                "stop_cause": "agent_finish", "finish_reason": "agent:ok", "errors": ["no table"]}),
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_action_path_skips_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = trace_utils.load_trace(self.make_trace(Path(tmp)))
        self.assertEqual(run.action_path, "READ → SEARCH → ANALYZE")
        metrics = run.metrics()
        self.assertEqual(metrics["tool_calls"], 4)
        self.assertEqual(metrics["input_tokens"], 900)

    def test_render_run_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_trace(Path(tmp))
            run = trace_utils.load_trace(path)
            rendered = trace_utils.render_run(run)
            self.assertIn("动作路径", rendered)
            self.assertIn("stop_cause=agent_finish", rendered)
            table = trace_utils.render_comparison([run, run])
            self.assertIn("tool_calls", table)
            self.assertIn("stop_cause", table)
            self.assertIn("agent_finish", table)

    def test_missing_trace_raises(self):
        with self.assertRaises(Exception):
            trace_utils.load_trace("/tmp/definitely-missing-trace.jsonl")

    def test_cli_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_trace(Path(tmp))
            proc = subprocess.run(
                [sys.executable, "-m", "agent.trace_utils", str(path), "--json"],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload[0]["action_path"], "READ → SEARCH → ANALYZE")

    def test_cli_bad_file_exit_2(self):
        proc = subprocess.run(
            [sys.executable, "-m", "agent.trace_utils", "/tmp/nope.jsonl"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("trace_utils:", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
