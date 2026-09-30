---
name: paper-reading
description: Read and analyze research papers into a structured, page-cited report covering the research problem, method or argument, evidence, results, strengths, and limitations. Use for close reading, critique, or comparison of a specific paper; not for open-ended literature surveys.
metadata:
  short-description: Read and analyze research papers
---

# Paper Reading Skill

## 目的

把一篇论文读成一份**结构化、可核查、带页码出处**的研究报告：它解决什么问题、用了什么方法或论证路径、
证据是什么、结论是否被证据支撑、价值与局限在哪。

本 Skill 只规定**怎么读、怎么写报告**。读取 PDF、检索、表格统计由 `tools/` 里的工具提供。

## Skill 与 Tool 的分工

- **Skill（本文件）**：遇到论文阅读任务时该走什么流程、报告要满足什么契约。
- **Tool（`tools/`）**：实际能执行的动作，共三个——`read_pdf`、`search_paper`、`analyze_table`；
  契约（参数、返回结构、命令行）见 `tools/manifest.json`。

需要原文或页码 → `read_pdf`；论文内部找不到答案 → `search_paper`（先本地语料，必要时联网）；
遇到结果表、消融表 → `analyze_table`（PDF 正文里没有现成表格文件时，先 `analyze_table extract`
从 `read_pdf` 的输出里抽候选表格）。**数字要算，不要心算**；引用要带页码，不要凭印象。

## 第一步总是判断论文类型

类型决定看哪些章节、报告怎么写。**不要默认成实验类论文。**

| 类型 | 特征 | 重点章节 |
| --- | --- | --- |
| `empirical` 实证 / 计算类 | 有方法、实验、baseline、指标、消融 | Method、Experiments、Ablation |
| `humanities` 人文 / 理论类（舞蹈学、史学、美学等） | 有论点、材料 / 文本 / 案例、论证结构；通常**没有**实验与消融 | 绪论、论证章节、材料与史料 |
| `survey` 综述类 | 分类框架、覆盖范围、争议与空白 | 分类体系、纳入标准 |

判断依据：标题、摘要、章节结构、有无图表。
判断不确定时，按 `empirical` 与 `humanities` 两套问题各问一遍，并在报告中标明类型置信度低。

## 阅读流程

默认按此顺序推进；论文结构特殊时可以调整，但每一步的产物都要落到报告里。

1. **摘要与题目** → 研究对象 + 它要解决的问题。一句话说不清就是还没读懂。
2. **引言 / 绪论** → 动机、作者自称的贡献、作者划定的边界。
3. **与核心贡献直接相关的方法 / 论证章节**（其余章节略读，不平均用力）：
   - `empirical`：模型或算法、实验设计、自变量与因变量。
   - `humanities`：论证路径、关键概念的定义、材料的选择与处理方式。
4. **证据部分**：
   - `empirical`：数据集、baseline、指标、主结果表，再找消融与稳健性。
   - `humanities`：材料是否足以支撑论点，反例有没有被处理。
   - `survey`：纳入标准、覆盖范围、比较维度。
5. **对比对象与评价标准**：它和谁比、用什么标准判断"更好 / 更成立"。
6. **核对"结论 ↔ 证据"**：逐条区分作者写了什么（陈述）、你推出什么（解释）、证据不足处（缺口）。
7. **补缺**：缺口先回论文内部补齐；补不齐就在报告里标注"论文未说明 / 未找到依据"。

## 报告契约

报告固定包含以下 10 个部分——后续评测的 Completeness 指标直接按这 10 项计数，不要自行增删标题。
缺信息就写"论文未涉及"或"未找到依据"，不要留空，更不要编造。

1. 研究问题 Research Problem
2. 研究动机 Motivation
3. 核心观点 Main Idea / Core Claim
4. 方法或论证路径 Method / Argument
5. 材料与证据 Materials & Evidence
6. 主要结果 Main Results / Findings
7. 支撑性分析 Supporting Analysis
8. 优势与贡献 Strengths
9. 局限 Limitations
10. 可能的延伸 Possible Extensions

各部分的写法、篇幅与示例见 `references/review_template.md`。

## 必须守住的几条

- **不编造**：不虚构数字、页码、引文或作者没说过的结论。
- **区分陈述与解释**：哪句是作者的、哪句是你的判断，文字上要分开。
- **给出处**：具体事实标页码，格式 `[p.8]`；读取 PDF 时保留页码。
- **语言跟随论文**：中文论文用中文写报告；专有名词与方法名保留原文。
- **只回答被问到的**，并允许在结论不稳时标注"不确定"及理由。

## 资源索引

按需读取，不要一次全加载：

- `tools/manifest.json` —— 三个工具的参数与返回结构；要调用工具时先看这里。
- `tools/pdf_reader.py` / `tools/paper_search.py` / `tools/table_analyzer.py` —— 工具实现与更完整的用法示例。
- `references/review_template.md` —— 写报告时读：逐节写法、篇幅、反例、交付前自检。
- `references/paper_schema.md` —— 需要跨论文对比或批量处理时读：结构化记录的字段定义。
- `scripts/` —— 工具在 Skill 目录下的薄入口（`extract_pdf.py`、`extract_table.py`），
  单独拷走本 Skill 时用得到；实现仍在 `tools/`。
