"""Tool 层：Agent 实际能"做什么"。

- `pdf_reader.py`    读 PDF，保留页码（`[p.N]` 引用的事实来源）
- `paper_search.py`  检索：本地语料离线检索 + arXiv/Crossref 联网检索
- `table_analyzer.py` 表格统计：均值/极值/差值/排名/分组

工具契约（名称、参数、返回结构）见 `manifest.json`；
"应该怎么用、什么时候用"由 `skills/paper-reading/SKILL.md` 规定。
"""
