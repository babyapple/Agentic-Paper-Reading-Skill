#!/usr/bin/env python3
"""Coverage —— 第五阶段的"覆盖度"：Agent 到底把哪几件事读到了。

路线图第五阶段的停止条件是：

```text
已经找到：
✓ Problem
✓ Method
✓ Experiment
✓ Results
→ FINISH
```

所以需要一个能把"读到的原文"映射成这四个维度的东西。这里**刻意用可解释的关键词线索**
而不是又一个 LLM 调用：它便宜、确定、可复现，而且能给出页码——既能当 Planner 的输入
（"你还没读到结果类内容"），也能当停止条件的判据。

诚实边界：这是**启发式覆盖度**，只说明"原文里出现过相关线索"，不等于"结论正确"或
"内容完备"。所以 `status` 有三档（covered / weak / missing）而不是布尔值，
并且它在 prompt 里永远标注为线索来源。LLM 逐步把这一层替换成 claim 抽取，是后续阶段的事。

四个维度与 `skills/paper-reading/SKILL.md` 的报告契约对应：

| 维度 | 对应报告部分 |
| --- | --- |
| `problem` | 1 研究问题、2 研究动机 |
| `method` | 4 方法或论证路径 |
| `evidence` | 5 材料与证据、7 支撑性分析 |
| `results` | 3 核心观点、6 主要结果 |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# 维度顺序 = 停止条件的检查顺序
DIMENSIONS = ("problem", "method", "evidence", "results")

DIMENSION_LABELS = {
    "problem": "研究问题",
    "method": "方法/论证",
    "evidence": "材料/证据",
    "results": "结果/结论",
}

DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "problem": (
        "研究问题", "研究对象", "问题意识", "旨在", "试图", "本文", "文章", "目的", "出发点", "追问",
        "关注", "聚焦", "议题", "论题", "考察", "question", "problem", "this paper", "this study",
        "we study", "aim", "focus",
    ),
    "method": (
        "研究方法", "方法论", "方法", "模型", "算法", "框架", "路径", "论证", "进路", "思路", "范式",
        "分析", "阐释", "解读", "归纳", "梳理", "比较研究", "类型学", "文本细读",
        "method", "approach", "framework", "algorithm", "methodology", "analysis", "typology",
    ),
    "evidence": (
        "材料", "史料", "文献", "文本", "案例", "作品", "图像", "画像", "舞俑", "壁画", "遗存", "文物",
        "考古", "数据", "样本", "语料", "田野", "调查", "访谈", "例证", "依据", "实证",
        "dataset", "corpus", "material", "experiment", "survey", "sample", "evidence", "archive",
    ),
    "results": (
        "结果", "结论", "结语", "表明", "显示", "发现", "揭示", "说明", "可见", "归纳出", "优于", "提升",
        "下降", "准确性", "accuracy", "results", "findings", "we find", "outperform", "improvement",
    ),
}

COVERED_MIN_HITS = 3        # 命中至少 3 处
COVERED_MIN_KEYWORDS = 2    # 且至少来自 2 个不同关键词，避免同一个词刷出来的假覆盖
WEAK_MIN_HITS = 1


@dataclass
class DimensionCoverage:
    name: str
    status: str  # covered | weak | missing
    hits: int = 0
    pages: list[int] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return DIMENSION_LABELS.get(self.name, self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "hits": self.hits,
            "pages": self.pages,
            "keywords": self.keywords,
        }


@dataclass
class CoverageReport:
    dimensions: list[DimensionCoverage] = field(default_factory=list)
    pages_scanned: list[int] = field(default_factory=list)
    source: str = "keyword-heuristic"

    def get(self, name: str) -> DimensionCoverage | None:
        return next((dim for dim in self.dimensions if dim.name == name), None)

    @property
    def missing(self) -> list[str]:
        return [dim.name for dim in self.dimensions if dim.status == "missing"]

    @property
    def weak(self) -> list[str]:
        return [dim.name for dim in self.dimensions if dim.status == "weak"]

    def all_covered(self, require_pages: int = 2) -> bool:
        """停止条件的判据：四个维度都 covered，且至少读了 require_pages 页。"""

        if len(self.pages_scanned) < require_pages:
            return False
        return all(dim.status == "covered" for dim in self.dimensions)

    @property
    def status_line(self) -> str:
        marks = {"covered": "✓", "weak": "~", "missing": "✗"}
        parts = []
        for dim in self.dimensions:
            pages = ",".join(f"p{p}" for p in dim.pages[:3])
            parts.append(f"{dim.name}{marks[dim.status]}{f'({pages})' if pages else ''}")
        return "覆盖：" + " ".join(parts)

    def missing_line(self) -> str:
        gaps = []
        for dim in self.dimensions:
            if dim.status == "missing":
                gaps.append(f"{dim.label}（没有任何线索）")
            elif dim.status == "weak":
                gaps.append(f"{dim.label}（线索只有 {dim.hits} 处）")
        return "；".join(gaps) if gaps else "（四个维度都有原文线索）"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pages_scanned": self.pages_scanned,
            "dimensions": [dim.to_dict() for dim in self.dimensions],
            "missing": self.missing,
            "weak": self.weak,
            "all_covered": self.all_covered(),
        }


def _scan_page(text: str, keywords: Sequence[str]) -> tuple[int, list[str]]:
    folded = text.lower()
    hits = 0
    matched: list[str] = []
    for keyword in keywords:
        count = folded.count(keyword.lower())
        if count:
            hits += count
            matched.append(keyword)
    return hits, matched


def summarize(state: Any, dimensions: Iterable[str] = DIMENSIONS) -> CoverageReport:
    """扫描"已读页"的正文，给出四个维度的覆盖度。

    `state` 只需要有 `page_texts`（{页码: 文本}）——所以本函数可以在任何时点调用，
    不需要额外的模型调用或缓存。
    """

    page_texts: dict[int, str] = getattr(state, "page_texts", {}) or {}
    report = CoverageReport(pages_scanned=sorted(page_texts))
    for name in dimensions:
        keywords = DIMENSION_KEYWORDS.get(name, ())
        total_hits = 0
        pages: list[int] = []
        matched_keywords: list[str] = []
        for page in sorted(page_texts):
            hits, matched = _scan_page(page_texts[page], keywords)
            if hits:
                total_hits += hits
                pages.append(page)
                for keyword in matched:
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
        if total_hits >= COVERED_MIN_HITS and len(matched_keywords) >= COVERED_MIN_KEYWORDS:
            status = "covered"
        elif total_hits >= WEAK_MIN_HITS:
            status = "weak"
        else:
            status = "missing"
        report.dimensions.append(
            DimensionCoverage(
                name=name,
                status=status,
                hits=total_hits,
                pages=pages,
                keywords=matched_keywords[:8],
            )
        )
    return report
