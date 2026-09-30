#!/usr/bin/env python3
"""工具层回归测试（只用标准库，不联网）。

跑法：

    python3 -m unittest discover -s tests -v

覆盖三类东西：
1. 纯函数：宽度折叠、页范围解析、数字解析（含 `—`/`N%`/`12.5±0.3`/`0.81*`）；
2. 表格操作：describe / rank / diff / compare(行&列) / group；
3. 真实论文上的端到端：读页、文档内检索、大纲、本地语料检索；
   以及 CLI 的正常路径与错误路径（exit code 2 + 单行 stderr）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import paper_search, pdf_reader, table_analyzer  # noqa: E402
from tools.pdf_reader import ToolError  # noqa: E402

PAPERS_DIR = REPO_ROOT / "paper"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ablation.md"


def any_paper() -> Path | None:
    if not PAPERS_DIR.exists():
        return None
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def has_pdf_backend() -> bool:
    try:
        pdf_reader.select_backend("auto")
        return True
    except ToolError:
        return False


class TestTextHelpers(unittest.TestCase):
    def test_fold_width_converts_fullwidth_digits(self):
        self.assertEqual(pdf_reader.fold_width("４３００００"), "430000")

    def test_normalize_with_map_maps_back_to_source(self):
        text = "汉代  袖舞\n风格"
        normalized, index_map = pdf_reader.normalize_with_map(text)
        self.assertEqual(normalized, "汉代 袖舞 风格")
        start = normalized.index("袖舞")
        offset = index_map[start]
        self.assertEqual(text[offset : offset + 2], "袖舞")

    def test_snippet_collapses_whitespace(self):
        text = "前文\n\n  关键词   后面"
        piece = pdf_reader.snippet(text, text.index("关"), text.index("词") + 1, 4)
        self.assertIn("关键词", piece)
        self.assertNotIn("\n", piece)


class TestPageRange(unittest.TestCase):
    def test_single_and_range(self):
        self.assertEqual(pdf_reader.parse_pages("1-3,8", 10), [1, 2, 3, 8])

    def test_empty_means_all(self):
        self.assertEqual(pdf_reader.parse_pages(None, 3), [1, 2, 3])

    def test_out_of_range_raises(self):
        with self.assertRaises(ToolError):
            pdf_reader.parse_pages("9-11", 10)

    def test_garbage_raises(self):
        with self.assertRaises(ToolError):
            pdf_reader.parse_pages("abc", 10)


class TestNumberParsing(unittest.TestCase):
    def test_plain_and_percent(self):
        self.assertEqual(table_analyzer.parse_number("78.3"), 78.3)
        self.assertEqual(table_analyzer.parse_number("78.3%"), 78.3)

    def test_thousands_and_std_and_star(self):
        self.assertEqual(table_analyzer.parse_number("1,234"), 1234.0)
        self.assertEqual(table_analyzer.parse_number("12.5±0.3"), 12.5)
        self.assertEqual(table_analyzer.parse_number("0.81*"), 0.81)

    def test_missing_values_are_none_not_zero(self):
        for cell in ["—", "-", "N/A", "未报告", "", "abc"]:
            self.assertIsNone(table_analyzer.parse_number(cell), msg=cell)

    def test_fullwidth_number(self):
        self.assertEqual(table_analyzer.parse_number("９９"), 99.0)


class TestTableLoad(unittest.TestCase):
    def test_markdown(self):
        headers, rows = table_analyzer.parse_markdown("| a | b |\n| --- | --- |\n| 1 | 2 |\n")
        self.assertEqual(headers, ["a", "b"])
        self.assertEqual(rows, [["1", "2"]])

    def test_tsv_detected(self):
        headers, rows = table_analyzer.parse_delimited("a\tb\n1\t2\n")
        self.assertEqual(headers, ["a", "b"])
        self.assertEqual(rows, [["1", "2"]])

    def test_json_objects(self):
        headers, rows = table_analyzer.parse_json('[{"m": "A", "v": 1}, {"m": "B", "v": 2}]')
        self.assertEqual(headers, ["m", "v"])
        self.assertEqual(rows, [["A", "1"], ["B", "2"]])

    def test_load_fixture_and_pad_short_rows(self):
        headers, rows = table_analyzer.load_table(str(FIXTURE))
        self.assertIn("Accuracy", headers)
        self.assertTrue(all(len(row) == len(headers) for row in rows))

    def test_missing_file_raises(self):
        with self.assertRaises(ToolError):
            table_analyzer.load_table(str(REPO_ROOT / "tests" / "fixtures" / "nope.csv"))


class TestTableOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers, cls.rows = table_analyzer.load_table(str(FIXTURE))

    def test_describe_skips_missing_cells(self):
        payload = table_analyzer.op_describe(self.headers, self.rows, None)
        accuracy = next(s for s in payload["stats"] if s["column"] == "Accuracy")
        self.assertEqual(accuracy["count"], 5)  # 6 行里有 1 行是 "—"
        self.assertAlmostEqual(accuracy["max"], 78.3)
        self.assertEqual(accuracy["max_row"], "Full Agent")

    def test_rank_descending(self):
        payload = table_analyzer.op_rank(self.headers, self.rows, "Accuracy", True, 3, "Method")
        self.assertEqual(payload["ranking"][0]["value"], 78.3)
        self.assertEqual(payload["ranking"][0]["rank"], 1)
        self.assertEqual(len(payload["ranking"]), 3)

    def test_diff_direction(self):
        payload = table_analyzer.op_diff(self.headers, self.rows, "Accuracy", "F1", "Method")
        self.assertEqual(payload["count"], 5)
        self.assertEqual(payload["a_higher_count"], 5)
        self.assertAlmostEqual(payload["rows"][0]["diff"], 3.2)

    def test_compare_by_row_with_lower_is_better(self):
        payload = table_analyzer.resolve_compare(
            self.headers, self.rows, "Baseline", None, "Method", ["Cost"]
        )
        self.assertEqual(payload["operation"], "compare_rows")
        full = payload["rows"][0]
        self.assertAlmostEqual(full["diffs"]["Accuracy"], 7.1)
        self.assertAlmostEqual(full["diffs"]["Cost"], -1.4)  # 成本更高 → 负增益
        self.assertEqual(full["win_count"], 2)

    def test_compare_by_column(self):
        payload = table_analyzer.resolve_compare(self.headers, self.rows, "Accuracy", None, "Method")
        self.assertEqual(payload["operation"], "compare")
        self.assertTrue(payload["comparisons"])

    def test_group_mean(self):
        payload = table_analyzer.op_group(self.headers, self.rows, "Dataset", "Accuracy", "mean")
        easy = next(g for g in payload["groups"] if g["group"] == "Easy")
        self.assertEqual(easy["count"], 4)

    def test_unknown_column_raises(self):
        with self.assertRaises(ToolError):
            table_analyzer.op_rank(self.headers, self.rows, "NoSuchColumn", True, None, None)

    def test_render_contains_markdown_pipe_table(self):
        payload = table_analyzer.op_group(self.headers, self.rows, "Dataset", "F1", "mean")
        rendered = table_analyzer.render_result(payload)
        self.assertIn("|", rendered)
        self.assertIn("Dataset", rendered)


class TestTableExtraction(unittest.TestCase):
    """把 read_pdf 的输出直接变成可分析的表格——这是 Tool 3 接真实论文的那一环。"""

    def test_finds_whitespace_aligned_table(self):
        text = (REPO_ROOT / "tests" / "fixtures" / "layout_page.txt").read_text(encoding="utf-8")
        blocks = table_analyzer.extract_tables(text)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["kind"], "whitespace")
        self.assertEqual(block["headers"][:2], ["Method", "Memory"])
        self.assertEqual(len(block["rows"]), 4)

    def test_does_not_fire_on_prose(self):
        prose = (
            "本文以汉画像砖石与文献为基础讨论汉代袖舞的风格类型。\n"
            "作者认为，袖舞的风格受纺织技术与礼乐制度影响。\n"
            "这一判断在第二、三节的图像分析中逐步展开。\n"
        )
        self.assertEqual(table_analyzer.extract_tables(prose), [])

    def test_does_not_fire_on_single_numeric_line(self):
        text = "Accuracy 78.3\nThis is not a table at all.\nAnother prose line here.\n"
        self.assertEqual(table_analyzer.extract_tables(text), [])

    def test_finds_markdown_table_in_mixed_text(self):
        text = (REPO_ROOT / "tests" / "fixtures" / "ablation.md").read_text(encoding="utf-8")
        blocks = table_analyzer.extract_tables(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["kind"], "markdown")

    def test_op_extract_pick_and_out(self):
        text = (REPO_ROOT / "tests" / "fixtures" / "ablation.md").read_text(encoding="utf-8")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "picked.md"
            payload = table_analyzer.op_extract(text, pick=1, out=str(target))
            self.assertEqual(payload["picked"]["index"], 1)
            self.assertTrue(target.exists())
            headers, rows = table_analyzer.load_table(str(target))
            self.assertIn("Accuracy", headers)
            self.assertEqual(len(rows), 6)

    def test_op_extract_pick_out_of_range(self):
        text = (REPO_ROOT / "tests" / "fixtures" / "ablation.md").read_text(encoding="utf-8")
        with self.assertRaises(ToolError):
            table_analyzer.op_extract(text, pick=9)

    def test_extracted_table_can_be_analyzed(self):
        text = (REPO_ROOT / "tests" / "fixtures" / "layout_page.txt").read_text(encoding="utf-8")
        block = table_analyzer.extract_tables(text)[0]
        payload = table_analyzer.op_rank(block["headers"], block["rows"], "Memory", True, 1, "Method")
        self.assertEqual(payload["ranking"][0]["label"], "Ours (full)")


class TestPaperSearchLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has_pdf_backend() or any_paper() is None:
            raise unittest.SkipTest("没有可用 PDF 后端或语料")
        cls.result = paper_search.search_local("袖舞 风格", papers_dir=PAPERS_DIR, top=5, mode="all")

    def test_finds_expected_paper(self):
        ids = [r["paper_id"] for r in self.result["results"]]
        self.assertTrue(any("袖舞" in pid for pid in ids), msg=ids)

    def test_hits_carry_page_numbers(self):
        first = self.result["results"][0]
        self.assertGreaterEqual(first["best_page"], 1)
        self.assertTrue(all(isinstance(h["page"], int) for h in first["hits"]))

    def test_all_mode_drops_papers_missing_a_term(self):
        result = paper_search.search_local("袖舞 傩舞", papers_dir=PAPERS_DIR, mode="all")
        # "all" 模式下没有一篇同时命中两个词 → 允许 0 篇，但绝不能出现缺词项的结果
        for item in result["results"]:
            self.assertEqual(item["missing_terms"], [])

    def test_any_mode_keeps_partial_matches(self):
        result = paper_search.search_local("袖舞 傩舞", papers_dir=PAPERS_DIR, mode="any")
        self.assertGreaterEqual(len(result["results"]), 2)
        self.assertTrue(all(item["missing_terms"] for item in result["results"]))

    def test_list_corpus_matches_pdf_count(self):
        items = paper_search.list_corpus(PAPERS_DIR)
        self.assertEqual(len(items), len(list(PAPERS_DIR.glob("*.pdf"))))
        self.assertTrue(all(item.get("pages_total", 0) > 0 for item in items))

    def test_terms_are_deduped(self):
        self.assertEqual(paper_search.parse_terms("a a b"), ["a", "b"])


class TestPdfReaderOnRealPaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = any_paper()
        if cls.pdf is None or not has_pdf_backend():
            raise unittest.SkipTest("没有可用 PDF 后端或语料")
        cls.document = pdf_reader.open_document(cls.pdf)

    def test_pages_and_numbering(self):
        self.assertGreater(self.document["pages_total"], 1)
        self.assertEqual(len(self.document["page_texts"]), self.document["pages_total"])
        self.assertTrue(self.document["page_texts"][0].strip())

    def test_render_keeps_page_markers(self):
        rendered = pdf_reader.render_pages(self.document, [1, 2])
        self.assertIn("[Page 1]", rendered)
        self.assertIn("[Page 2]", rendered)

    def test_document_search_returns_page(self):
        needle = self.document["page_texts"][0].strip().split("\n")[0][:4]
        result = pdf_reader.search_in_document(self.pdf, needle, max_hits=3)
        self.assertGreater(result["hit_count"], 0)
        self.assertGreaterEqual(result["hits"][0]["page"], 1)

    def test_search_empty_query_raises(self):
        with self.assertRaises(ToolError):
            pdf_reader.search_in_document(self.pdf, "   ")

    def test_outline_returns_items(self):
        outline = pdf_reader.read_outline(self.pdf)
        self.assertIn(outline["source"], {"pdf-bookmarks", "heuristic"})
        self.assertIsInstance(outline["outline"], list)

    def test_boilerplate_strip_keeps_content(self):
        cleaned, _removed = pdf_reader.strip_boilerplate(self.document["page_texts"])
        self.assertEqual(len(cleaned), len(self.document["page_texts"]))

    def test_missing_file_raises(self):
        with self.assertRaises(ToolError):
            pdf_reader.open_document(REPO_ROOT / "paper" / "definitely-missing.pdf")

    def test_non_pdf_raises(self):
        with self.assertRaises(ToolError):
            pdf_reader.extract_pages(FIXTURE)


class TestCli(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

    def test_table_cli_json(self):
        proc = self.run_cli(
            "tools/table_analyzer.py", "describe", str(FIXTURE), "--json"
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["operation"], "describe")

    def test_table_cli_error_is_single_line_exit_2(self):
        proc = self.run_cli("tools/table_analyzer.py", "describe", "no/such/table.csv")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("table_analyzer:", proc.stderr)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1)

    def test_search_cli_error_exit_2(self):
        proc = self.run_cli("tools/paper_search.py", "search", "x", "--source", "nope")
        self.assertNotEqual(proc.returncode, 0)  # argparse 也会挡住非法枚举值

    def test_skill_script_wrapper(self):
        proc = self.run_cli(
            "skills/paper-reading/scripts/extract_table.py", "describe", str(FIXTURE)
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("数值列", proc.stdout)

    def test_pdf_cli_read_pages(self):
        pdf = any_paper()
        if pdf is None or not has_pdf_backend():
            self.skipTest("没有可用 PDF 后端或语料")
        proc = self.run_cli(
            "tools/pdf_reader.py", "read", str(pdf), "--pages", "1", "--max-chars", "80"
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("[Page 1]", proc.stdout)

    def test_pdf_cli_bad_pages_exit_2(self):
        pdf = any_paper()
        if pdf is None or not has_pdf_backend():
            self.skipTest("没有可用 PDF 后端或语料")
        proc = self.run_cli("tools/pdf_reader.py", "read", str(pdf), "--pages", "999")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("pdf_reader:", proc.stderr)

    def test_table_cli_extract(self):
        proc = self.run_cli(
            "tools/table_analyzer.py", "extract", str(REPO_ROOT / "tests" / "fixtures" / "layout_page.txt"),
            "--json",
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["operation"], "extract")
        self.assertEqual(payload["table_count"], 1)

    def test_manifest_is_valid_json_and_covers_tools(self):
        manifest = json.loads((REPO_ROOT / "tools" / "manifest.json").read_text(encoding="utf-8"))
        names = {tool["name"] for tool in manifest["tools"]}
        self.assertEqual(names, {"read_pdf", "search_paper", "analyze_table"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
