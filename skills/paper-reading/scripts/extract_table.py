#!/usr/bin/env python3
"""Skill 内嵌入口：表格统计（均值 / 极值 / 差值 / 排名 / baseline 对比）。

实现只有一份，在仓库根的 `tools/table_analyzer.py`。参数完全一致：

    python3 skills/paper-reading/scripts/extract_table.py describe table.md
    python3 skills/paper-reading/scripts/extract_table.py compare table.md \\
        --baseline Baseline --label Method --lower-is-better Cost
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.table_analyzer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
