#!/usr/bin/env python3
"""table_analyzer.py —— Tool 3：表格统计（均值 / 极值 / 差值 / 排名 / 分组 / 对比）。

用途（对应 SKILL.md 第 6、7 步"结论 ↔ 证据"核对）：
论文里的结果表往往只是"人眼能看懂"，Agent 需要能真的算一遍——
"这个方法比 baseline 高多少"、"哪一列最好"、"消融删掉某组件掉了几个点"
都不该由模型心算，而应该由工具算出来再写进报告。

输入不止 PDF 表格，也吃现成的表格文本：CSV / TSV / Markdown 表格 / JSON。
输出默认是 Markdown 表格（Agent 可直接读），`--json` 给机器用。

命令行::

    python3 tools/table_analyzer.py describe table.csv
    python3 tools/table_analyzer.py rank    table.csv --by Accuracy --desc --top 5
    python3 tools/table_analyzer.py diff    table.csv --a Ours --b Baseline --label Method
    python3 tools/table_analyzer.py compare table.csv --baseline Baseline --label Method
    python3 tools/table_analyzer.py group   table.csv --by Dataset --value Accuracy
    cat table.md | python3 tools/table_analyzer.py describe - --format markdown

失败方式与其它工具一致：`ToolError` → stderr 一行 + exit 2。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pdf_reader import ToolError, fold_width  # noqa: E402

_MISSING = {"", "-", "--", "—", "–", "n/a", "na", "null", "none", "无", "未报告", "未报告。"}
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")


def parse_number(cell: str) -> float | None:
    """把单元格变成数字。

    能吃：`1,234`、`78.3%`、`12.5±0.3`、`78.3(1.2)`、`0.81*`（显著性星号）；
    认不出的（`—`、`N/A`、`未报告`、纯文字）返回 None——**不当作 0**，
    否则"未报告"会被算进均值，这是这类工具最常见的错误。
    """

    if cell is None:
        return None
    text = fold_width(str(cell)).strip()
    if text.lower() in _MISSING:
        return None
    text = text.replace("≈", "").replace("~", "").strip()
    if "±" in text:
        text = text.split("±")[0]  # 只取均值，方差单独看没有意义
    elif "(" in text and not text.startswith("("):
        text = text.split("(")[0]
    text = text.replace("%", "").replace(",", "").replace(" ", "")
    text = text.rstrip("*†‡#§")
    if not _PLAIN_NUMBER.fullmatch(text):
        return None
    return float(text)


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in text.split("\n") if "|" in line]
    if not lines:
        raise ToolError("没有找到 Markdown 表格（需要包含 `|` 的行）")
    headers = _split_markdown_row(lines[0])
    body = lines[1:]
    if body and re.fullmatch(r"[\s|:\-–—+]+", body[0]):
        body = body[1:]  # 分隔行
    rows = [_split_markdown_row(line) for line in body]
    return headers, rows


def parse_delimited(text: str, delimiter: str | None = None) -> tuple[list[str], list[list[str]]]:
    sample = "\n".join(text.split("\n")[:5])
    if delimiter is None:
        counts = {d: sample.count(d) for d in ["\t", ",", ";", "，"]}
        delimiter = max(counts, key=lambda d: counts[d]) if max(counts.values()) > 0 else ","
    reader = csv.reader(io.StringIO(text), delimiter="," if delimiter == "，" else delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ToolError("表格为空")
    return [cell.strip() for cell in rows[0]], [[cell.strip() for cell in row] for row in rows[1:]]


def parse_json(text: str) -> tuple[list[str], list[list[str]]]:
    data = json.loads(text)
    if isinstance(data, dict) and "headers" in data and "rows" in data:
        return list(data["headers"]), [[str(c) for c in row] for row in data["rows"]]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        headers: list[str] = []
        for item in data:
            for key in item:
                if key not in headers:
                    headers.append(key)
        rows = [[str(item.get(h, "")) for h in headers] for item in data]
        return headers, rows
    raise ToolError("JSON 表格需要是对象数组，或 {\"headers\": [...], \"rows\": [[...]]}")


def read_text_source(source: str) -> tuple[str, str]:
    """读文件或标准输入，返回 (文本, 名字)。"""

    if source == "-":
        return sys.stdin.read(), "<stdin>"
    path = Path(source).expanduser()
    if not path.exists():
        raise ToolError(f"文件不存在：{path}")
    return path.read_text(encoding="utf-8", errors="replace"), path.name


def load_table(source: str, fmt: str = "auto") -> tuple[list[str], list[list[str]]]:
    text, name = read_text_source(source)
    if not text.strip():
        raise ToolError(f"输入为空：{name}")

    if fmt == "auto":
        suffix = Path(name).suffix.lower()
        if suffix == ".json" or text.lstrip().startswith(("[", "{")):
            fmt = "json"
        elif "|" in text and re.search(r"^\s*\|", text, re.M):
            fmt = "markdown"
        else:
            fmt = "delimited"

    try:
        if fmt == "json":
            headers, rows = parse_json(text)
        elif fmt == "markdown":
            headers, rows = parse_markdown(text)
        else:
            headers, rows = parse_delimited(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"JSON 解析失败：{exc}") from exc

    if not headers:
        raise ToolError("没有解析到表头")
    width = len(headers)
    normalized: list[list[str]] = []
    for row in rows:
        if not any(cell for cell in row):
            continue
        if len(row) < width:
            row = row + [""] * (width - len(row))
        normalized.append(row[:width])
    if not normalized:
        raise ToolError("表格没有数据行")
    return headers, normalized


# --------------------------------------------------------------------------- #
# 从论文文本里抽候选表格（把 read_pdf 的输出直接变成可分析的表格）
# --------------------------------------------------------------------------- #

_CELL_SPLIT = re.compile(r"\t|\s{2,}")


def split_cells(line: str) -> list[str]:
    """按制表符或 2 个以上空格切列——PDF 版式化文本里表格就是这么对齐的。"""

    return [cell.strip() for cell in _CELL_SPLIT.split(line.strip()) if cell.strip()]


def looks_tabular(rows: Sequence[Sequence[str]], min_cols: int) -> bool:
    """判定"这堆等宽分列的行像不像表格"：至少一列过半是数字，且没有超长散文行。"""

    if len(rows) < 2:
        return False
    width = max(len(row) for row in rows)
    if width < min_cols:
        return False
    if any(len(cell) > 40 for row in rows for cell in row):
        return False  # 单元格里塞满句子 → 是正文排版，不是表
    numeric_columns_found = 0
    for i in range(width):
        values = [to_number(row[i]) for row in rows[1:]]
        present = [v for v in values if v is not None]
        non_empty = [row[i] for row in rows[1:] if row[i].strip()]
        if present and len(present) >= max(1, len(non_empty) * 0.5):
            numeric_columns_found += 1
    return numeric_columns_found >= 1


def extract_tables(text: str, min_rows: int = 3, min_cols: int = 2) -> list[dict[str, Any]]:
    """在文本里找两类表格：Markdown 管道表、按多空格对齐的分栏块。"""

    lines = text.split("\n")
    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].count("|") >= 2:
            j = i
            while j < len(lines) and lines[j].count("|") >= 2:
                j += 1
            raw = lines[i:j]
            if len(raw) >= min_rows:
                try:
                    headers, rows = parse_markdown("\n".join(raw))
                except ToolError:
                    headers, rows = [], []
                if headers and len(headers) >= min_cols and rows:
                    blocks.append(
                        {"kind": "markdown", "start_line": i + 1, "end_line": j, "headers": headers, "rows": rows}
                    )
            i = j
            continue

        cells = split_cells(lines[i]) if lines[i].strip() else []
        if len(cells) >= min_cols:
            j = i
            group: list[list[str]] = []
            while j < len(lines) and lines[j].strip():
                row_cells = split_cells(lines[j])
                if len(row_cells) < min_cols:
                    break
                group.append(row_cells)
                j += 1
            width = max(len(g) for g in group)
            padded = [g + [""] * (width - len(g)) for g in group]
            if len(padded) >= min_rows and looks_tabular(padded, min_cols):
                blocks.append(
                    {
                        "kind": "whitespace",
                        "start_line": i + 1,
                        "end_line": j,
                        "headers": padded[0],
                        "rows": padded[1:],
                    }
                )
            i = j if j > i else i + 1
            continue
        i += 1
    return blocks


def op_extract(
    text: str,
    min_rows: int = 3,
    min_cols: int = 2,
    pick: int | None = None,
    out: str | None = None,
) -> dict[str, Any]:
    blocks = extract_tables(text, min_rows, min_cols)
    tables = [
        {
            "index": position,
            "kind": block["kind"],
            "start_line": block["start_line"],
            "end_line": block["end_line"],
            "rows": len(block["rows"]),
            "columns": len(block["headers"]),
            "headers": block["headers"],
            "markdown": render_table(block["headers"], block["rows"]),
        }
        for position, block in enumerate(blocks, start=1)
    ]
    payload: dict[str, Any] = {"operation": "extract", "table_count": len(tables), "tables": tables}
    if out and pick is None:
        raise ToolError("--out 需要同时指定 --pick N，说明用第几个候选表")
    if pick is not None:
        if not 1 <= pick <= len(blocks):
            raise ToolError(f"--pick {pick} 越界，本次只找到 {len(blocks)} 个候选表")
        chosen = tables[pick - 1]
        payload["picked"] = chosen
        if out:
            Path(out).write_text(chosen["markdown"] + "\n", encoding="utf-8")
            payload["picked"] = {**chosen, "path": str(Path(out).resolve())}
    return payload


# --------------------------------------------------------------------------- #
# 列语义
# --------------------------------------------------------------------------- #


def index_of(headers: Sequence[str], name: str) -> int:
    if name in headers:
        return headers.index(name)
    for i, header in enumerate(headers):
        if header and name.lower() in fold_width(header).lower():
            return i
    raise ToolError(f"找不到列 {name!r}；可用列：{list(headers)}")


def numeric_columns(headers: Sequence[str], rows: Sequence[Sequence[str]], threshold: float = 0.7) -> list[int]:
    result = []
    for i in range(len(headers)):
        values = [to_number(row[i]) for row in rows]
        present = [v for v in values if v is not None]
        non_empty = [row[i] for row in rows if str(row[i]).strip()]
        if present and len(present) >= max(1, int(len(non_empty) * threshold)):
            result.append(i)
    return result


def label_column(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[int]) -> int | None:
    for i in range(len(headers)):
        if i in numeric:
            continue
        if any(str(row[i]).strip() for row in rows):
            return i
    return None


def to_number(cell: Any) -> float | None:
    return parse_number(str(cell)) if cell is not None else None


def column_values(rows: Sequence[Sequence[str]], index: int, labels: Sequence[str] | None = None) -> dict[str, float]:
    """返回 {行标签: 数值}，跳过非数字单元格（不填 0，避免把"未报告"算成 0）。"""

    out: dict[str, float] = {}
    for i, row in enumerate(rows):
        value = to_number(row[index])
        if value is None:
            continue
        key = labels[i] if labels and i < len(labels) else str(row[index])
        out[key] = value
    return out


# --------------------------------------------------------------------------- #
# 渲染（按东亚字宽对齐）
# --------------------------------------------------------------------------- #


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    space = max(0, width - display_width(text))
    if align == "right":
        return " " * space + text
    return text + " " * space


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], aligns: Sequence[str] | None = None) -> str:
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    aligns = list(aligns or ["left"] * len(headers))
    widths = [display_width(str(h)) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))
    lines = ["| " + " | ".join(pad(str(h), widths[i]) for i, h in enumerate(headers)) + " |"]
    lines.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(headers))) + "|")
    for row in cells:
        lines.append(
            "| " + " | ".join(pad(cell, widths[i], aligns[i]) for i, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# 操作
# --------------------------------------------------------------------------- #


def op_describe(headers, rows, columns: Sequence[str] | None) -> dict[str, Any]:
    numeric = numeric_columns(headers, rows)
    chosen = [index_of(headers, c) for c in columns] if columns else numeric
    stats: list[dict[str, Any]] = []
    label_index = label_column(headers, rows, numeric)
    for i in chosen:
        values = [v for v in (to_number(row[i]) for row in rows) if v is not None]
        if not values:
            stats.append({"column": headers[i], "count": 0, "note": "该列没有可解析的数字"})
            continue
        best_row = ""
        best_value = max(values)
        for row in rows:
            if to_number(row[i]) == best_value:
                best_row = str(row[label_index]) if label_index is not None else ""
                break
        stats.append(
            {
                "column": headers[i],
                "count": len(values),
                "mean": round(statistics.fmean(values), 6),
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": best_value,
                "median": statistics.median(values),
                "sum": round(sum(values), 6),
                "max_row": best_row,
            }
        )
    non_numeric = [headers[i] for i in range(len(headers)) if i not in numeric]
    return {
        "operation": "describe",
        "rows": len(rows),
        "columns": len(headers),
        "numeric_columns": [headers[i] for i in numeric],
        "label_columns": non_numeric,
        "stats": stats,
    }


def op_rank(headers, rows, by: str, desc: bool, top: int | None, label: str | None) -> dict[str, Any]:
    value_index = index_of(headers, by)
    label_index = index_of(headers, label) if label else label_column(headers, rows, [value_index])
    entries = []
    for row in rows:
        value = to_number(row[value_index])
        if value is None:
            continue
        entries.append({"label": str(row[label_index]) if label_index is not None else "", "value": value})
    entries.sort(key=lambda e: e["value"], reverse=desc)
    for position, entry in enumerate(entries, start=1):
        entry["rank"] = position
    shown = entries[:top] if top else entries
    return {
        "operation": "rank",
        "column": headers[value_index],
        "order": "desc" if desc else "asc",
        "count": len(entries),
        "ranking": shown,
    }


def op_diff(headers, rows, a: str, b: str, label: str | None) -> dict[str, Any]:
    ia, ib = index_of(headers, a), index_of(headers, b)
    label_index = index_of(headers, label) if label else label_column(headers, rows, [ia, ib])
    pairs: list[dict[str, Any]] = []
    for row in rows:
        va, vb = to_number(row[ia]), to_number(row[ib])
        if va is None or vb is None:
            continue
        pairs.append(
            {
                "label": str(row[label_index]) if label_index is not None else "",
                "a": va,
                "b": vb,
                "diff": round(va - vb, 6),
            }
        )
    if not pairs:
        raise ToolError(f"列 {a!r} 与 {b!r} 没有可比较的数值对")
    diffs = [p["diff"] for p in pairs]
    return {
        "operation": "diff",
        "a": headers[ia],
        "b": headers[ib],
        "count": len(pairs),
        "mean_diff": round(statistics.fmean(diffs), 6),
        "min_diff": min(diffs),
        "max_diff": max(diffs),
        "a_higher_count": sum(1 for d in diffs if d > 0),
        "b_higher_count": sum(1 for d in diffs if d < 0),
        "rows": pairs,
    }


def op_group(headers, rows, by: str, value: str, agg: str) -> dict[str, Any]:
    group_index, value_index = index_of(headers, by), index_of(headers, value)
    buckets: dict[str, list[float]] = {}
    for row in rows:
        key = str(row[group_index]).strip()
        number = to_number(row[value_index])
        if not key or number is None:
            continue
        buckets.setdefault(key, []).append(number)
    if not buckets:
        raise ToolError(f"按 {by!r} 分组、对 {value!r} 聚合后没有数据")

    def aggregate(values: list[float]) -> float:
        if agg == "mean":
            return round(statistics.fmean(values), 6)
        if agg == "sum":
            return round(sum(values), 6)
        if agg == "max":
            return max(values)
        if agg == "min":
            return min(values)
        if agg == "count":
            return float(len(values))
        raise ToolError(f"未知聚合方式 {agg!r}")

    groups = [
        {"group": key, "count": len(values), "value": aggregate(values), "values": values}
        for key, values in sorted(buckets.items())
    ]
    groups.sort(key=lambda g: g["value"], reverse=True)
    return {"operation": "group", "by": headers[group_index], "value": headers[value_index], "agg": agg, "groups": groups}


def op_compare(
    headers, rows, baseline: str, columns: Sequence[str] | None, label: str | None,
    lower_is_better: Sequence[str] = (),
) -> dict[str, Any]:
    """把某一列当作 baseline，逐列给出"相对 baseline 的增益"——消融表的标准读法。"""

    baseline_index = index_of(headers, baseline)
    baseline_values = [to_number(row[baseline_index]) for row in rows]
    if all(v is None for v in baseline_values):
        raise ToolError(f"baseline 列 {baseline!r} 没有可解析的数字")
    numeric = numeric_columns(headers, rows)
    chosen = [index_of(headers, c) for c in columns] if columns else [i for i in numeric if i != baseline_index]
    label_index = index_of(headers, label) if label else label_column(headers, rows, numeric)
    lower = {name.strip().lower() for name in lower_is_better}

    comparisons: list[dict[str, Any]] = []
    for i in chosen:
        sign = -1 if headers[i].strip().lower() in lower else 1
        diffs: list[float] = []
        for r, row in enumerate(rows):
            base = baseline_values[r] if r < len(baseline_values) else None
            other = to_number(row[i])
            if base is None or other is None:
                continue
            diffs.append(sign * (other - base))
        if not diffs:
            continue
        comparisons.append(
            {
                "column": headers[i],
                "direction": "lower-is-better" if sign < 0 else "higher-is-better",
                "mean_diff_vs_baseline": round(statistics.fmean(diffs), 6),
                "max_diff": round(max(diffs), 6),
                "min_diff": round(min(diffs), 6),
                "win_count": sum(1 for d in diffs if d > 0),
                "loss_count": sum(1 for d in diffs if d < 0),
                "compared_rows": len(diffs),
            }
        )
    return {
        "operation": "compare",
        "baseline": headers[baseline_index],
        "label_column": headers[label_index] if label_index is not None else None,
        "row_labels": [str(row[label_index]) if label_index is not None else f"row{i+1}" for i, row in enumerate(rows)],
        "comparisons": comparisons,
    }


def op_compare_rows(
    headers, rows, baseline_row: str, columns: Sequence[str] | None, label: str | None,
    lower_is_better: Sequence[str] = (),
) -> dict[str, Any]:
    """把某一**行**（如 `Baseline`）当作参照，逐行给出各指标的增益——消融表就是这样读的。"""

    numeric = numeric_columns(headers, rows)
    label_index = index_of(headers, label) if label else label_column(headers, rows, numeric)
    if label_index is None:
        raise ToolError("找不到可用作行名的列，请用 --label 指定")
    metrics = [index_of(headers, c) for c in columns] if columns else numeric
    if not metrics:
        raise ToolError("没有可比较的数值列")
    lower = {name.strip().lower() for name in lower_is_better}

    baseline = next((row for row in rows if str(row[label_index]).strip() == baseline_row.strip()), None)
    if baseline is None:
        available = [str(row[label_index]) for row in rows]
        raise ToolError(f"找不到 baseline 行 {baseline_row!r}；可选行：{available}")

    entries: list[dict[str, Any]] = []
    for row in rows:
        if row is baseline:
            continue
        diffs: dict[str, float] = {}
        for i in metrics:
            base_value, value = to_number(baseline[i]), to_number(row[i])
            if base_value is None or value is None:
                continue
            sign = -1 if headers[i].strip().lower() in lower else 1
            diffs[headers[i]] = round(sign * (value - base_value), 6)
        if not diffs:
            continue
        values = list(diffs.values())
        entries.append(
            {
                "label": str(row[label_index]),
                "diffs": diffs,
                "mean_diff": round(statistics.fmean(values), 6),
                "win_count": sum(1 for d in values if d > 0),
                "loss_count": sum(1 for d in values if d < 0),
                "metric_count": len(values),
            }
        )
    return {
        "operation": "compare_rows",
        "baseline_row": baseline_row,
        "label_column": headers[label_index],
        "metrics": [headers[i] for i in metrics],
        "lower_is_better": sorted(lower),
        "rows": entries,
    }


def resolve_compare(headers, rows, baseline: str, columns, label, lower_is_better=()) -> dict[str, Any]:
    """`--baseline` 既可能是列名（列=方法），也可能是行名（行=方法），自动判断。"""

    if baseline in headers or any(baseline.lower() in fold_width(h).lower() for h in headers):
        return op_compare(headers, rows, baseline, columns, label, lower_is_better)
    numeric = numeric_columns(headers, rows)
    label_index = index_of(headers, label) if label else label_column(headers, rows, numeric)
    if label_index is not None and any(str(row[label_index]).strip() == baseline.strip() for row in rows):
        return op_compare_rows(headers, rows, baseline, columns, label, lower_is_better)
    raise ToolError(
        f"{baseline!r} 既不是列名也不是行名。列：{list(headers)}；"
        f"行：{[str(row[label_index]) for row in rows] if label_index is not None else '（无法识别行名列，请用 --label）'}"
    )


def op_plot(headers, rows, x: str, y: str, kind: str, out_path: str) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 取决于环境
        raise ToolError(f"绘图需要 matplotlib（pip install matplotlib）：{exc}") from exc

    xi, yi = index_of(headers, x), index_of(headers, y)
    labels, values = [], []
    for row in rows:
        value = to_number(row[yi])
        if value is None:
            continue
        labels.append(str(row[xi]))
        values.append(value)
    if not values:
        raise ToolError(f"列 {y!r} 没有可绘制的数字")
    figure, axes = plt.subplots(figsize=(max(4, len(labels) * 0.6), 3.6))
    if kind == "bar":
        axes.bar(range(len(labels)), values)
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(labels, rotation=45, ha="right")
    else:
        axes.plot(range(len(labels)), values, marker="o")
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_ylabel(headers[yi])
    axes.set_title(f"{headers[yi]} vs {headers[xi]}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return {"operation": "plot", "kind": kind, "path": str(Path(out_path).resolve()), "points": len(values)}


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def render_result(payload: dict[str, Any]) -> str:
    op = payload["operation"]
    lines: list[str] = []
    if op == "describe":
        lines.append(f"表格规模：{payload['rows']} 行 × {payload['columns']} 列")
        lines.append(f"数值列：{payload['numeric_columns']}")
        lines.append(f"文本列：{payload['label_columns']}")
        lines.append("")
        rows = []
        for stat in payload["stats"]:
            if stat.get("count", 0) == 0:
                rows.append([stat["column"], 0, "—", "—", "—", "—", "—", stat.get("note", "")])
                continue
            rows.append(
                [
                    stat["column"],
                    stat["count"],
                    fmt(stat["mean"]),
                    fmt(stat["std"]),
                    fmt(stat["min"]),
                    fmt(stat["max"]),
                    fmt(stat["median"]),
                    stat.get("max_row", ""),
                ]
            )
        lines.append(render_table(["列", "n", "均值", "标准差", "最小", "最大", "中位数", "最大值所在行"], rows,
                                  ["left", "right", "right", "right", "right", "right", "right", "left"]))
    elif op == "rank":
        lines.append(f"按 `{payload['column']}` 排序（{payload['order']}，共 {payload['count']} 行）")
        lines.append("")
        rows = [[e["rank"], e["label"], fmt(e["value"])] for e in payload["ranking"]]
        lines.append(render_table(["#", "行", payload["column"]], rows, ["right", "left", "right"]))
    elif op == "diff":
        lines.append(f"`{payload['a']}` − `{payload['b']}`（{payload['count']} 行可比）")
        lines.append(
            f"平均差 {fmt(payload['mean_diff'])}；"
            f"{payload['a']} 更高 {payload['a_higher_count']} 行，{payload['b']} 更高 {payload['b_higher_count']} 行"
        )
        lines.append("")
        rows = [[e["label"], fmt(e["a"]), fmt(e["b"]), fmt(e["diff"])] for e in payload["rows"]]
        lines.append(render_table(["行", payload["a"], payload["b"], "差值"], rows, ["left", "right", "right", "right"]))
    elif op == "group":
        lines.append(f"按 `{payload['by']}` 分组，对 `{payload['value']}` 取 {payload['agg']}")
        lines.append("")
        rows = [[g["group"], g["count"], fmt(g["value"])] for g in payload["groups"]]
        lines.append(render_table([payload["by"], "n", f"{payload['agg']}({payload['value']})"], rows,
                                  ["left", "right", "right"]))
    elif op == "compare":
        lines.append(f"以 `{payload['baseline']}` 为 baseline 的相对增益（正数=优于 baseline）")
        lines.append("")
        rows = [
            [
                c["column"],
                fmt(c["mean_diff_vs_baseline"]),
                fmt(c["min_diff"]),
                fmt(c["max_diff"]),
                c["win_count"],
                c["loss_count"],
                c["compared_rows"],
            ]
            for c in payload["comparisons"]
        ]
        lines.append(
            render_table(
                ["列", "平均增益", "最小增益", "最大增益", "胜", "负", "可比行"],
                rows,
                ["left", "right", "right", "right", "right", "right", "right"],
            )
        )
    elif op == "compare_rows":
        lines.append(f"以 `{payload['baseline_row']}` 行为 baseline 的相对增益（正数=优于 baseline）")
        if payload.get("lower_is_better"):
            lines.append("（这些列按「越小越好」处理：" + "、".join(payload["lower_is_better"]) + "）")
        lines.append("")
        metrics = payload["metrics"]
        header = ["行"] + metrics + ["平均增益", "胜/负"]
        rows_out = []
        for entry in payload["rows"]:
            row = [entry["label"]]
            row += [fmt(entry["diffs"].get(metric)) if metric in entry["diffs"] else "—" for metric in metrics]
            row += [fmt(entry["mean_diff"]), f"{entry['win_count']}/{entry['loss_count']}"]
            rows_out.append(row)
        aligns = ["left"] + ["right"] * (len(metrics) + 2)
        lines.append(render_table(header, rows_out, aligns))
    elif op == "extract":
        lines.append(f"共找到 {payload['table_count']} 个候选表格")
        for table in payload["tables"]:
            lines.append("")
            lines.append(
                f"### 候选 {table['index']}（{table['kind']}，第 {table['start_line']}-{table['end_line']} 行，"
                f"{table['rows']} 行 × {table['columns']} 列）"
            )
            lines.append(table["markdown"])
        if payload.get("picked"):
            lines.append("")
            lines.append(
                f"已选用候选 {payload['picked']['index']}"
                + (f" → {payload['picked']['path']}" if payload["picked"].get("path") else "")
            )
    elif op == "plot":
        lines.append(f"已生成图：{payload['path']}（{payload['points']} 个点，{payload['kind']}）")
    else:  # pragma: no cover
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="table_analyzer.py",
        description="Tool 3：表格统计（均值/极值/差值/排名/分组/baseline 对比）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("table", help="表格文件（CSV/TSV/Markdown/JSON），或 `-` 读标准输入")
        p.add_argument("--format", default="auto", choices=["auto", "delimited", "markdown", "json"])
        p.add_argument("--columns", help="只分析这些列，逗号分隔")
        p.add_argument("--json", action="store_true")

    describe = sub.add_parser("describe", help="每列均值/标准差/极值/中位数")
    add_common(describe)

    rank = sub.add_parser("rank", help="按某列排名")
    add_common(rank)
    rank.add_argument("--by", required=True)
    rank.add_argument("--asc", action="store_true", help="默认降序（大者优先）")
    rank.add_argument("--top", type=int, default=0, help="只显示前 N 名（0=全部）")
    rank.add_argument("--label", help="用作行名的列")

    diff = sub.add_parser("diff", help="两列逐行求差")
    add_common(diff)
    diff.add_argument("--a", required=True)
    diff.add_argument("--b", required=True)
    diff.add_argument("--label")

    compare = sub.add_parser("compare", help="以某列（列=方法）或某行（行=方法）为 baseline 算增益/胜负")
    add_common(compare)
    compare.add_argument("--baseline", required=True, help="列名或行名，两者都支持，自动判断")
    compare.add_argument("--label", help="行名列（baseline 是行名时通常需要）")
    compare.add_argument(
        "--lower-is-better",
        default="",
        help="这些列越小越好（如 Cost、Latency、参数量），逗号分隔；否则默认越大越好",
    )

    group = sub.add_parser("group", help="分组聚合")
    add_common(group)
    group.add_argument("--by", required=True)
    group.add_argument("--value", required=True)
    group.add_argument("--agg", default="mean", choices=["mean", "sum", "max", "min", "count"])

    extract = sub.add_parser("extract", help="从文本（如 read_pdf 的输出）里抽候选表格")
    extract.add_argument("table", help="文本文件（Markdown/纯文本/PDF 抽取结果），或 `-` 读标准输入")
    extract.add_argument("--format", default="auto", choices=["auto", "delimited", "markdown", "json"], help=argparse.SUPPRESS)
    extract.add_argument("--columns", help=argparse.SUPPRESS)
    extract.add_argument("--min-rows", type=int, default=3, help="至少几行才算表格")
    extract.add_argument("--min-cols", type=int, default=2, help="至少几列才算表格")
    extract.add_argument("--pick", type=int, help="选定第 N 个候选表")
    extract.add_argument("--out", help="把选定的表格写成 Markdown 文件（需配合 --pick）")
    extract.add_argument("--json", action="store_true")

    plot = sub.add_parser("plot", help="画图（需要 matplotlib）")
    add_common(plot)
    plot.add_argument("--x", required=True)
    plot.add_argument("--y", required=True)
    plot.add_argument("--kind", default="bar", choices=["bar", "line"])
    plot.add_argument("--out", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            text, _name = read_text_source(args.table)
            if not text.strip():
                raise ToolError("输入为空")
            payload = op_extract(text, args.min_rows, args.min_cols, args.pick, args.out)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(render_result(payload), end="")
            return 0

        headers, rows = load_table(args.table, args.format)
        columns = [c.strip() for c in args.columns.split(",")] if getattr(args, "columns", None) else None
        if args.command == "describe":
            payload = op_describe(headers, rows, columns)
        elif args.command == "rank":
            payload = op_rank(headers, rows, args.by, not args.asc, args.top or None, args.label)
        elif args.command == "diff":
            payload = op_diff(headers, rows, args.a, args.b, args.label)
        elif args.command == "compare":
            lower = [c.strip() for c in args.lower_is_better.split(",") if c.strip()]
            payload = resolve_compare(headers, rows, args.baseline, columns, args.label, lower)
        elif args.command == "group":
            payload = op_group(headers, rows, args.by, args.value, args.agg)
        elif args.command == "plot":
            payload = op_plot(headers, rows, args.x, args.y, args.kind, args.out)
        else:  # pragma: no cover
            raise ToolError(f"未知命令 {args.command}")

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_result(payload), end="")
    except ToolError as exc:
        print(f"table_analyzer: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
